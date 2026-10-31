"""Build CORE tables from a streamed triple parquet."""

from __future__ import annotations

import csv
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from lib.contracts import PAIR, PAIR_PMID, SENTENCE
from lib.id_rules import parse_core_disease, parse_core_gene, parse_core_pmid, usable_id_status
from lib.schema import (
    BUILDER_EXTRA_HASTYPE,
    CORE_PREDICATES,
    OWL_LABEL_ENUM,
    RDF_TYPE,
    TYPE_ARTICLE,
    TYPE_DATABASE,
    TYPE_DISEASE,
    TYPE_GCS,
    TYPE_GENE,
    TYPE_SENTENCE,
    TYPE_SENTENCE_EQUIV,
)
from lib.stats import head_share, summarize_numeric
from lib.years import parse_core_year

TRIPLE_SCHEMA = pa.schema(
    [
        ("subject", pa.string()),
        ("predicate", pa.string()),
        ("object", pa.string()),
        ("object_kind", pa.string()),
        ("datatype", pa.string()),
        ("lang", pa.string()),
    ]
)


def open_triple_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return pq.ParquetWriter(path, TRIPLE_SCHEMA, compression="zstd")


def write_triple_batch(writer, batch: list[tuple]) -> None:
    columns = list(zip(*batch)) if batch else [[] for _ in range(6)]
    table = pa.Table.from_arrays([pa.array(col, type=pa.string()) for col in columns], schema=TRIPLE_SCHEMA)
    writer.write_table(table)


def _one(con, sql: str):
    row = con.execute(sql).fetchone()
    return row[0] if row else None


def _exec(con, sql: str) -> None:
    con.execute(sql)


def build_core_tables(con, triple_parquet: Path, out_dir: Path, logger) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    P = CORE_PREDICATES
    _exec(con, f"CREATE OR REPLACE VIEW triple AS SELECT * FROM read_parquet('{triple_parquet.as_posix()}')")
    n_triples = _one(con, "SELECT COUNT(*) FROM triple")
    logger.info("phase=core_triples records=%s", n_triples)

    _exec(
        con,
        f"""
        CREATE OR REPLACE TABLE pred_counts AS
        SELECT predicate, object_kind, COUNT(*) AS n
        FROM triple GROUP BY 1, 2
        """,
    )
    _exec(
        con,
        f"""
        CREATE OR REPLACE TABLE type_counts AS
        SELECT object AS type_uri, COUNT(*) AS n
        FROM triple WHERE predicate = '{RDF_TYPE}'
        GROUP BY 1
        """,
    )

    def edges(table: str, predicate: str) -> None:
        _exec(
            con,
            f"""
            CREATE OR REPLACE TABLE {table} AS
            SELECT subject, object, COUNT(*) AS n_triples
            FROM triple WHERE predicate = '{predicate}'
            GROUP BY 1, 2
            """,
        )

    edges("gcs_gene", P["expressedBy"])
    edges("gcs_disease", P["involves"])
    edges("gcs_sentence", P["supportedBy"])
    edges("gcs_type", P["hasType"])
    edges("sent_article", P["extractedFrom"])
    edges("sent_gene_mention", P["mentionsGene"])
    edges("sent_disease_mention", P["mentionsDisease"])
    edges("sent_db", P["includedIn"])

    label_preds = [P[k] for k in ("GCCLabel", "CGELabel", "CCSLabel", "GCILabel", "curatedAnnotation", "hasContent", "GSAMention", "DSAMention", "GSAStartPosition", "GSAEndPosition", "DSAStartPosition", "DSAEndPosition")]
    pred_list = ", ".join(f"'{p}'" for p in label_preds)
    _exec(
        con,
        f"""
        CREATE OR REPLACE TABLE sent_lit AS
        SELECT subject AS sentence_uri, predicate, object AS value, COUNT(*) AS n
        FROM triple
        WHERE predicate IN ({pred_list})
        GROUP BY 1, 2, 3
        """,
    )
    article_preds = ", ".join(f"'{P[k]}'" for k in ("title", "abstract", "hasVenue", "publicationYear", "cites", "hasCitation"))
    _exec(
        con,
        f"""
        CREATE OR REPLACE TABLE article_lit AS
        SELECT subject AS article_uri, predicate, object AS value
        FROM triple
        WHERE predicate IN ({article_preds})
        """,
    )

    _register_maps(con, logger)
    _build_sentence_table(con)
    _assert_columns(con, "sentence", SENTENCE)
    sentence_path = out_dir / "sentence.parquet"
    _exec(con, f"COPY sentence TO '{sentence_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    _exec(con, f"COPY gcs_status TO '{(out_dir / 'gcs_status.parquet').as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    _exec(con, f"COPY article_lit TO '{(out_dir / 'article_literal.parquet').as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    _exec(con, f"COPY gcs_sentence TO '{(out_dir / 'gcs_sentence.parquet').as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")

    summary = _summarize(con, out_dir, n_triples, logger)
    pair_path = out_dir / "pair.parquet"
    pair_pmid_path = out_dir / "pair_pmid.parquet"
    _exec(con, f"COPY pair TO '{pair_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    _exec(con, f"COPY pair_pmid TO '{pair_pmid_path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    _assert_columns(con, "pair", PAIR)
    _assert_columns(con, "pair_pmid", PAIR_PMID)
    summary["outputs"] = {
        "sentence": str(sentence_path),
        "pair": str(pair_path),
        "pair_pmid": str(pair_pmid_path),
        "article_literal": str(out_dir / "article_literal.parquet"),
        "gcs_status": str(out_dir / "gcs_status.parquet"),
        "triple_parquet": str(triple_parquet),
    }
    return summary


def _register_maps(con, logger) -> None:
    gene_rows = []
    for (uri,) in con.execute("SELECT DISTINCT object FROM gcs_gene").fetchall():
        parsed = parse_core_gene(uri)
        gene_rows.append((uri, parsed["normalized_id"], parsed["status"]))
    disease_rows = []
    for (uri,) in con.execute("SELECT DISTINCT object FROM gcs_disease").fetchall():
        parsed = parse_core_disease(uri)
        disease_rows.append((uri, parsed["normalized_id"], parsed["status"]))
    article_rows = []
    for (uri,) in con.execute("SELECT DISTINCT object FROM sent_article").fetchall():
        parsed = parse_core_pmid(uri)
        article_rows.append((uri, parsed["normalized_id"], parsed["status"]))
    logger.info(
        "phase=id_parse distinct_gene_uri=%s distinct_disease_uri=%s distinct_article_uri=%s",
        len(gene_rows),
        len(disease_rows),
        len(article_rows),
    )
    con.execute("CREATE OR REPLACE TABLE gene_map(uri VARCHAR, normalized_id VARCHAR, status VARCHAR)")
    con.execute("CREATE OR REPLACE TABLE disease_map(uri VARCHAR, normalized_id VARCHAR, status VARCHAR)")
    con.execute("CREATE OR REPLACE TABLE article_map(uri VARCHAR, normalized_id VARCHAR, status VARCHAR)")
    if gene_rows:
        con.executemany("INSERT INTO gene_map VALUES (?,?,?)", gene_rows)
    if disease_rows:
        con.executemany("INSERT INTO disease_map VALUES (?,?,?)", disease_rows)
    if article_rows:
        con.executemany("INSERT INTO article_map VALUES (?,?,?)", article_rows)


def _part(alias: str) -> str:
    return f"CASE WHEN COALESCE({alias}.n_values,0) > 1 THEN 'MULTI' ELSE COALESCE({alias}.value, 'FIELD_ABSENT') END"


def _build_sentence_table(con) -> None:
    P = CORE_PREDICATES
    usable = ", ".join(f"'{s}'" for s in ("explicit_namespace", "scheme_variant_explicit", "explicit_namespace_pattern_unusual"))
    con.execute(
        f"""
        CREATE OR REPLACE TABLE gcs_status AS
        WITH uris AS (
          SELECT subject AS gcs_uri FROM gcs_gene
          UNION
          SELECT subject FROM gcs_disease
          UNION
          SELECT subject FROM gcs_sentence
          UNION
          SELECT subject FROM gcs_type
          UNION
          SELECT subject FROM triple WHERE predicate = '{RDF_TYPE}' AND object = '{TYPE_GCS}'
        ), gg AS (
          SELECT subject AS gcs_uri, COUNT(DISTINCT object) AS n_gene_uri FROM gcs_gene GROUP BY 1
        ), gd AS (
          SELECT subject AS gcs_uri, COUNT(DISTINCT object) AS n_disease_uri FROM gcs_disease GROUP BY 1
        ), ge AS (
          SELECT g.subject AS gcs_uri,
                 COUNT(DISTINCT gm.normalized_id) FILTER (WHERE gm.status IN ({usable}) AND gm.normalized_id IS NOT NULL) AS n_gene_id,
                 MIN(gm.normalized_id) FILTER (WHERE gm.status IN ({usable})) AS gene_id,
                 MIN(gm.status) AS gene_status
          FROM gcs_gene g LEFT JOIN gene_map gm ON g.object = gm.uri
          GROUP BY 1
        ), de AS (
          SELECT d.subject AS gcs_uri,
                 COUNT(DISTINCT dm.normalized_id) FILTER (WHERE dm.status IN ({usable}) AND dm.normalized_id IS NOT NULL) AS n_disease,
                 MIN(dm.normalized_id) FILTER (WHERE dm.status IN ({usable})) AS disease_cui,
                 MIN(dm.status) AS disease_status
          FROM gcs_disease d LEFT JOIN disease_map dm ON d.object = dm.uri
          GROUP BY 1
        ), ht AS (
          SELECT subject AS gcs_uri, string_agg(DISTINCT object, '|' ORDER BY object) AS has_type
          FROM gcs_type GROUP BY 1
        )
        SELECT
          u.gcs_uri,
          ge.gene_id,
          ge.gene_status,
          de.disease_cui,
          de.disease_status,
          COALESCE(gg.n_gene_uri, 0) AS n_gene_uri,
          COALESCE(gd.n_disease_uri, 0) AS n_disease_uri,
          COALESCE(ge.n_gene_id, 0) AS n_gene_id,
          COALESCE(de.n_disease, 0) AS n_disease,
          ht.has_type,
          CASE
            WHEN COALESCE(ge.n_gene_id,0)=1 AND COALESCE(de.n_disease,0)=1
                 AND COALESCE(gg.n_gene_uri,0)=1 AND COALESCE(gd.n_disease_uri,0)=1 THEN 'explicit'
            WHEN COALESCE(ge.n_gene_id,0)>1 OR COALESCE(de.n_disease,0)>1
                 OR COALESCE(gg.n_gene_uri,0)>1 OR COALESCE(gd.n_disease_uri,0)>1 THEN 'ambiguous_multiple'
            ELSE 'id_not_explicit'
          END AS pair_status
        FROM uris u
        LEFT JOIN gg USING (gcs_uri)
        LEFT JOIN gd USING (gcs_uri)
        LEFT JOIN ge USING (gcs_uri)
        LEFT JOIN de USING (gcs_uri)
        LEFT JOIN ht USING (gcs_uri)
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE sent_base AS
        WITH uris AS (
          SELECT object AS sentence_uri FROM gcs_sentence
          UNION
          SELECT subject FROM sent_article
          UNION
          SELECT sentence_uri FROM sent_lit
          UNION
          SELECT subject FROM triple WHERE predicate = '{RDF_TYPE}' AND object IN ('{TYPE_SENTENCE}', '{TYPE_SENTENCE_EQUIV}')
        )
        SELECT sentence_uri FROM uris
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE sentence AS
        WITH one_label AS (
          SELECT sentence_uri, predicate,
                 CASE WHEN COUNT(DISTINCT value)=1 THEN MIN(value) ELSE NULL END AS value,
                 COUNT(DISTINCT value) AS n_values
          FROM sent_lit
          GROUP BY 1, 2
        )
        SELECT
          b.sentence_uri,
          gs.subject AS gcs_uri,
          st.gene_id,
          st.disease_cui,
          st.gene_status,
          st.disease_status,
          COALESCE(st.pair_status, 'no_gcs') AS pair_status,
          am.normalized_id AS pmid,
          am.status AS pmid_status,
          sa.object AS article_uri,
          gcc.value AS gcc,
          cge.value AS cge,
          ccs.value AS ccs,
          gci.value AS gci,
          CONCAT_WS('|', {_part('gcc')}, {_part('cge')}, {_part('ccs')}, {_part('gci')}) AS label_tuple,
          COALESCE(gcc.n_values,0)+COALESCE(cge.n_values,0)+COALESCE(ccs.n_values,0)+COALESCE(gci.n_values,0) AS n_label_values,
          cur.value AS curated,
          txt.value AS content,
          gm.value AS gene_mention,
          dm.value AS disease_mention,
          gs1.value AS gene_start,
          ge1.value AS gene_end,
          ds1.value AS disease_start,
          de1.value AS disease_end,
          db.object AS db_uri,
          st.has_type
        FROM sent_base b
        LEFT JOIN gcs_sentence gs ON b.sentence_uri = gs.object
        LEFT JOIN gcs_status st ON gs.subject = st.gcs_uri
        LEFT JOIN sent_article sa ON b.sentence_uri = sa.subject
        LEFT JOIN article_map am ON sa.object = am.uri
        LEFT JOIN one_label gcc ON b.sentence_uri = gcc.sentence_uri AND gcc.predicate = '{P["GCCLabel"]}'
        LEFT JOIN one_label cge ON b.sentence_uri = cge.sentence_uri AND cge.predicate = '{P["CGELabel"]}'
        LEFT JOIN one_label ccs ON b.sentence_uri = ccs.sentence_uri AND ccs.predicate = '{P["CCSLabel"]}'
        LEFT JOIN one_label gci ON b.sentence_uri = gci.sentence_uri AND gci.predicate = '{P["GCILabel"]}'
        LEFT JOIN one_label cur ON b.sentence_uri = cur.sentence_uri AND cur.predicate = '{P["curatedAnnotation"]}'
        LEFT JOIN one_label txt ON b.sentence_uri = txt.sentence_uri AND txt.predicate = '{P["hasContent"]}'
        LEFT JOIN one_label gm ON b.sentence_uri = gm.sentence_uri AND gm.predicate = '{P["GSAMention"]}'
        LEFT JOIN one_label dm ON b.sentence_uri = dm.sentence_uri AND dm.predicate = '{P["DSAMention"]}'
        LEFT JOIN one_label gs1 ON b.sentence_uri = gs1.sentence_uri AND gs1.predicate = '{P["GSAStartPosition"]}'
        LEFT JOIN one_label ge1 ON b.sentence_uri = ge1.sentence_uri AND ge1.predicate = '{P["GSAEndPosition"]}'
        LEFT JOIN one_label ds1 ON b.sentence_uri = ds1.sentence_uri AND ds1.predicate = '{P["DSAStartPosition"]}'
        LEFT JOIN one_label de1 ON b.sentence_uri = de1.sentence_uri AND de1.predicate = '{P["DSAEndPosition"]}'
        LEFT JOIN (
          SELECT subject, string_agg(DISTINCT object, '|' ORDER BY object) AS object
          FROM sent_db GROUP BY 1
        ) db ON b.sentence_uri = db.subject
        """
    )


def _assert_columns(con, table: str, expected: list[str]) -> None:
    got = [row[0] for row in con.execute(f"DESCRIBE {table}").fetchall()]
    if got != expected:
        raise SystemExit(f"column contract mismatch for {table}: got {got} expected {expected}")


def _summarize(con, out_dir: Path, n_triples: int, logger) -> dict:
    P = CORE_PREDICATES
    usable_pmid = "pmid_status IN ('explicit_namespace', 'scheme_variant_explicit')"
    con.execute(
        f"""
        CREATE OR REPLACE TABLE pair_pmid AS
        SELECT
          gene_id,
          disease_cui,
          pmid,
          COUNT(DISTINCT sentence_uri) AS n_sentence,
          COUNT(DISTINCT label_tuple) AS n_distinct_label_tuples,
          COUNT(DISTINCT label_tuple) > 1 AS heterogeneous_within_pmid,
          string_agg(DISTINCT label_tuple, ' || ' ORDER BY label_tuple) AS label_set,
          COUNT(DISTINCT CASE
            WHEN gcc IS DISTINCT FROM 'NOTINF' AND cge IS DISTINCT FROM 'NOTINF'
             AND ccs IS DISTINCT FROM 'NOTINF' AND gci IS DISTINCT FROM 'NOTINF'
             AND (gcc IS NOT NULL OR cge IS NOT NULL OR ccs IS NOT NULL OR gci IS NOT NULL)
            THEN label_tuple END) AS n_distinct_usable_tuples,
          CASE
            WHEN COUNT(*) FILTER (
              WHERE gcc IS NOT NULL AND gcc <> 'NOTINF'
                 OR cge IS NOT NULL AND cge <> 'NOTINF'
                 OR ccs IS NOT NULL AND ccs <> 'NOTINF'
                 OR gci IS NOT NULL AND gci <> 'NOTINF'
            ) = 0 THEN 'no_usable_label'
            ELSE 'has_usable_label'
          END AS usable_status
        FROM sentence
        WHERE pair_status = 'explicit' AND {usable_pmid} AND pmid IS NOT NULL
        GROUP BY 1, 2, 3
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE pair AS
        WITH base AS (
          SELECT gene_id, disease_cui, gcs_uri, sentence_uri, pmid, pmid_status, label_tuple
          FROM sentence
          WHERE pair_status = 'explicit'
        ), pmid_sets AS (
          SELECT gene_id, disease_cui, pmid, string_agg(DISTINCT label_tuple, ' || ' ORDER BY label_tuple) AS sig
          FROM base
          WHERE pmid_status IN ('explicit_namespace', 'scheme_variant_explicit') AND pmid IS NOT NULL
          GROUP BY 1, 2, 3
        ), across AS (
          SELECT gene_id, disease_cui,
                 COUNT(*) AS n_pmid_sig,
                 COUNT(DISTINCT sig) AS n_distinct_pmid_label_sets
          FROM pmid_sets GROUP BY 1, 2
        )
        SELECT
          b.gene_id,
          b.disease_cui,
          COUNT(DISTINCT b.gcs_uri) AS n_gcs,
          COUNT(DISTINCT b.sentence_uri) AS n_sentence,
          COUNT(DISTINCT b.pmid) FILTER (
            WHERE b.pmid_status IN ('explicit_namespace', 'scheme_variant_explicit') AND b.pmid IS NOT NULL
          ) AS n_pmid,
          COUNT(DISTINCT b.sentence_uri) FILTER (
            WHERE b.pmid_status IS NULL OR b.pmid_status NOT IN ('explicit_namespace', 'scheme_variant_explicit') OR b.pmid IS NULL
          ) AS n_sentence_without_pmid,
          CASE WHEN COALESCE(a.n_pmid_sig,0) > 1 AND COALESCE(a.n_distinct_pmid_label_sets,0) > 1 THEN TRUE
               WHEN COALESCE(a.n_pmid_sig,0) <= 1 THEN NULL
               ELSE FALSE END AS heterogeneous_across_pmid,
          a.n_distinct_pmid_label_sets,
          NULL::INTEGER AS first_observable_year
        FROM base b
        LEFT JOIN across a USING (gene_id, disease_cui)
        GROUP BY b.gene_id, b.disease_cui, a.n_pmid_sig, a.n_distinct_pmid_label_sets
        """
    )
    # Publication year is attached from article literals, one row per value.
    years = con.execute(
        f"""
        SELECT article_uri, value
        FROM article_lit
        WHERE predicate = '{P['publicationYear']}'
        """
    ).fetchall()
    parsed_rows = []
    for uri, value in years:
        parsed = parse_core_year(value)
        parsed_rows.append((uri, value, parsed["status"], parsed["year"]))
    con.execute("CREATE OR REPLACE TABLE article_year(article_uri VARCHAR, raw VARCHAR, status VARCHAR, year INTEGER)")
    if parsed_rows:
        con.executemany("INSERT INTO article_year VALUES (?,?,?,?)", parsed_rows)
    con.execute(
        """
        CREATE OR REPLACE TABLE pair_year AS
        SELECT s.gene_id, s.disease_cui, MIN(y.year) FILTER (WHERE y.status = 'parsed_single') AS first_observable_year
        FROM sentence s
        JOIN article_year y ON s.article_uri = y.article_uri
        WHERE s.pair_status = 'explicit'
        GROUP BY 1, 2
        """
    )
    con.execute(
        """
        UPDATE pair
        SET first_observable_year = pair_year.first_observable_year
        FROM pair_year
        WHERE pair.gene_id = pair_year.gene_id AND pair.disease_cui = pair_year.disease_cui
        """
    )
    logger.info("phase=core_aggregate_done")
    summary = {
        "scope": "A_core_local_release",
        "n_triples": n_triples,
        "n_subjects": _one(con, "SELECT COUNT(DISTINCT subject) FROM triple"),
        "type_counts": _rows(con, "SELECT type_uri, n FROM type_counts ORDER BY n DESC"),
        "predicate_counts": _rows(con, "SELECT predicate, object_kind, n FROM pred_counts ORDER BY n DESC"),
        "n_gcs_nodes": _one(con, "SELECT COUNT(*) FROM gcs_status"),
        "gcs_pair_status": _rows(con, "SELECT pair_status, COUNT(*) AS n FROM gcs_status GROUP BY 1 ORDER BY 1"),
        "n_explicit_gcs": _one(con, "SELECT COUNT(*) FROM gcs_status WHERE pair_status='explicit'"),
        "n_explicit_pairs": _one(con, "SELECT COUNT(*) FROM pair"),
        "n_pair_pmid": _one(con, "SELECT COUNT(*) FROM pair_pmid"),
        "n_sentence_rows": _one(con, "SELECT COUNT(*) FROM sentence"),
        "n_distinct_sentence": _one(con, "SELECT COUNT(DISTINCT sentence_uri) FROM sentence"),
        "n_distinct_sentence_text": _one(con, "SELECT COUNT(DISTINCT content) FROM sentence WHERE content IS NOT NULL AND content <> ''"),
        "n_explicit_pmid": _one(
            con,
            """
            SELECT COUNT(DISTINCT pmid) FROM sentence
            WHERE pmid_status IN ('explicit_namespace', 'scheme_variant_explicit') AND pmid IS NOT NULL
            """,
        ),
        "funnel": _funnel(con),
        "integrity": _integrity(con),
        "labels": _label_summary(con),
        "text": _text_summary(con),
        "time": _time_summary(con),
        "distributions": _distributions(con),
        "schema_notes": {
            "owl_label_enum": OWL_LABEL_ENUM,
            "builder_hasType_outside_owl": BUILDER_EXTRA_HASTYPE,
            "scores_in_builder_not_serialized": ["CGE_score", "CCS_score", "GCI_score", "GCC_score"],
            "builder_pre_rdf_filter_code": "utils.combine_data drops rows with CGE_label == 'NA' or GCC_label == 'OTHER'. The comment mentions NOTINF; the code compares CGE to NA.",
            "local_files": ["data.ttl", "schema.owl"],
            "intermediate_pipeline_files_present": False,
            "hasType_recompute": "not_computable_from_local_graph",
            "hasType_recompute_reason": "Builder scores used by tag_gcs2type are not written into the RDF graph.",
        },
        "id_namespaces_expected": {
            "gene": "https://www.ncbi.nlm.nih.gov/gene/{entrez}",
            "disease": "http://linkedlifedata.com/resource/umls/id/{cui}",
            "article": "https://pubmed.ncbi.nlm.nih.gov/{pmid}",
            "source": "GDAMining/core src/core/cecore.py",
        },
    }
    _write_support_tables(con, out_dir, summary)
    return summary


def _rows(con, sql: str) -> list[dict]:
    cur = con.execute(sql)
    names = [d[0] for d in cur.description]
    return [dict(zip(names, row)) for row in cur.fetchall()]


def _funnel(con) -> dict:
    stages = {
        "gcs_nodes": "SELECT COUNT(*) FROM gcs_status",
        "gcs_with_one_gene_uri_and_one_disease_uri": "SELECT COUNT(*) FROM gcs_status WHERE n_gene_uri=1 AND n_disease_uri=1",
        "gcs_explicit_ids": "SELECT COUNT(*) FROM gcs_status WHERE pair_status='explicit'",
        "gcs_ambiguous": "SELECT COUNT(*) FROM gcs_status WHERE pair_status='ambiguous_multiple'",
        "gcs_id_not_explicit": "SELECT COUNT(*) FROM gcs_status WHERE pair_status='id_not_explicit'",
        "explicit_pairs": "SELECT COUNT(*) FROM pair",
        "explicit_pairs_with_pmid": "SELECT COUNT(*) FROM pair WHERE n_pmid > 0",
        "explicit_pairs_without_pmid": "SELECT COUNT(*) FROM pair WHERE n_pmid = 0",
        "explicit_pair_pmid": "SELECT COUNT(*) FROM pair_pmid",
        "explicit_pmid": """
            SELECT COUNT(DISTINCT pmid) FROM pair_pmid
        """,
    }
    return {name: {"numerator": _one(con, sql), "unit": "see_name", "status": "computed"} for name, sql in stages.items()}


def _integrity(con) -> dict:
    checks = {
        "sentence_with_multiple_gcs": """
            SELECT COUNT(*) FROM (
              SELECT object FROM gcs_sentence GROUP BY 1 HAVING COUNT(DISTINCT subject) > 1
            )
        """,
        "gcs_with_multiple_genes": "SELECT COUNT(*) FROM gcs_status WHERE n_gene_uri > 1",
        "gcs_with_multiple_diseases": "SELECT COUNT(*) FROM gcs_status WHERE n_disease_uri > 1",
        "sentence_with_multiple_articles": """
            SELECT COUNT(*) FROM (
              SELECT subject FROM sent_article GROUP BY 1 HAVING COUNT(DISTINCT object) > 1
            )
        """,
        "mention_gene_disagrees_with_gcs": f"""
            SELECT COUNT(*) FROM sent_gene_mention m
            JOIN gcs_sentence gs ON m.subject = gs.object
            JOIN gcs_gene gg ON gs.subject = gg.subject
            WHERE m.object <> gg.object
        """,
        "mention_disease_disagrees_with_gcs": """
            SELECT COUNT(*) FROM sent_disease_mention m
            JOIN gcs_sentence gs ON m.subject = gs.object
            JOIN gcs_disease gd ON gs.subject = gd.subject
            WHERE m.object <> gd.object
        """,
        "explicit_sentence_missing_content": """
            SELECT COUNT(*) FROM sentence
            WHERE pair_status='explicit' AND (content IS NULL OR content='')
        """,
        "subjects_without_type": f"""
            SELECT COUNT(*) FROM (
              SELECT subject FROM triple
              WHERE subject NOT IN (SELECT subject FROM triple WHERE predicate = '{RDF_TYPE}')
            )
        """,
    }
    out = {}
    for name, sql in checks.items():
        out[name] = {"n": _one(con, sql), "status": "computed", "dropped": False}
    return out


def _label_summary(con) -> dict:
    summary = {"sentence_level": {}, "gcs_has_type": _rows(con, "SELECT COALESCE(has_type, 'FIELD_ABSENT') AS has_type, COUNT(*) AS n FROM gcs_status GROUP BY 1 ORDER BY n DESC")}
    for col in ("gcc", "cge", "ccs", "gci", "curated"):
        summary["sentence_level"][col] = _rows(
            con,
            f"SELECT COALESCE({col}, 'FIELD_ABSENT') AS value, COUNT(*) AS n FROM sentence GROUP BY 1 ORDER BY n DESC",
        )
    summary["heterogeneous_pair_pmid"] = _one(con, "SELECT COUNT(*) FROM pair_pmid WHERE heterogeneous_within_pmid")
    summary["pair_pmid_no_usable_label"] = _one(con, "SELECT COUNT(*) FROM pair_pmid WHERE usable_status='no_usable_label'")
    summary["pairs_heterogeneous_across_pmid"] = _one(con, "SELECT COUNT(*) FROM pair WHERE heterogeneous_across_pmid")
    summary["pairs_homogeneous_across_pmid"] = _one(con, "SELECT COUNT(*) FROM pair WHERE heterogeneous_across_pmid = FALSE")
    summary["pairs_single_pmid_across_not_applicable"] = _one(con, "SELECT COUNT(*) FROM pair WHERE heterogeneous_across_pmid IS NULL")
    summary["definition"] = "Within a pair-PMID, heterogeneous means more than one distinct four-label tuple. Across PMIDs, heterogeneous means those per-PMID tuple sets differ. NOTINF is a value in the primary tuple. usable_status drops NOTINF and FIELD_ABSENT; if nothing remains the row is no_usable_label and is not counted as agreement."
    return summary


def _text_summary(con) -> dict:
    lengths = [row[0] for row in con.execute("SELECT length(content) FROM sentence WHERE content IS NOT NULL").fetchall()]
    return {
        "content_length": summarize_numeric(lengths, population="sentence_rows_with_content_literal", unit="characters"),
        "n_missing_content": _one(con, "SELECT COUNT(*) FROM sentence WHERE content IS NULL"),
        "n_empty_content": _one(con, "SELECT COUNT(*) FROM sentence WHERE content = ''"),
        "n_duplicate_content_values": _one(
            con,
            """
            SELECT COUNT(*) FROM (
              SELECT content FROM sentence WHERE content IS NOT NULL AND content <> ''
              GROUP BY 1 HAVING COUNT(*) > 1
            )
            """,
        ),
        "article_literal_counts": _rows(con, "SELECT predicate, COUNT(*) AS n, COUNT(DISTINCT article_uri) AS n_article FROM article_lit GROUP BY 1 ORDER BY 1"),
        "pmid_with_multiple_titles": _one(
            con,
            f"""
            SELECT COUNT(*) FROM (
              SELECT s.pmid
              FROM sentence s JOIN article_lit a ON s.article_uri=a.article_uri
              WHERE a.predicate = '{CORE_PREDICATES['title']}' AND s.pmid IS NOT NULL
              GROUP BY s.pmid
              HAVING COUNT(DISTINCT a.value) > 1
            )
            """,
        ),
        "pmid_with_multiple_abstracts": _one(
            con,
            f"""
            SELECT COUNT(*) FROM (
              SELECT s.pmid
              FROM sentence s JOIN article_lit a ON s.article_uri=a.article_uri
              WHERE a.predicate = '{CORE_PREDICATES['abstract']}' AND s.pmid IS NOT NULL
              GROUP BY s.pmid
              HAVING COUNT(DISTINCT a.value) > 1
            )
            """,
        ),
    }


def _time_summary(con) -> dict:
    status_counts = _rows(con, "SELECT status, COUNT(*) AS n FROM article_year GROUP BY 1 ORDER BY 1")
    multi = _one(
        con,
        """
        SELECT COUNT(*) FROM (
          SELECT article_uri FROM article_year WHERE status='parsed_single' GROUP BY 1 HAVING COUNT(DISTINCT year) > 1
        )
        """,
    )
    by_year = _rows(
        con,
        """
        SELECT y.year,
               COUNT(DISTINCT s.pmid) AS n_pmid,
               COUNT(DISTINCT s.gene_id || '|' || s.disease_cui) FILTER (WHERE s.pair_status='explicit') AS n_active_pair,
               COUNT(*) FILTER (WHERE s.pair_status='explicit' AND s.pmid IS NOT NULL) AS n_sentence_on_explicit_pair
        FROM article_year y
        JOIN sentence s ON y.article_uri = s.article_uri
        WHERE y.status = 'parsed_single'
        GROUP BY 1
        ORDER BY 1
        """,
    )
    return {
        "field": "publicationYear",
        "field_meaning": "Ontology comment says publication year. Builder writes PMIDYear into this predicate and skips values <= 0, so a missing triple is not stored as zero.",
        "status_counts": status_counts,
        "articles_with_multiple_parsed_years": multi,
        "by_year": by_year,
        "first_observable_year_meaning": "Minimum parsed publicationYear among articles linked to the pair in this file. Not the year the association was established.",
    }


def _distributions(con) -> dict:
    pmid_per_pair = [row[0] for row in con.execute("SELECT n_pmid FROM pair").fetchall()]
    sent_per_pair = [row[0] for row in con.execute("SELECT n_sentence FROM pair").fetchall()]
    gcs_per_pair = [row[0] for row in con.execute("SELECT n_gcs FROM pair").fetchall()]
    sent_per_pair_pmid = [row[0] for row in con.execute("SELECT n_sentence FROM pair_pmid").fetchall()]
    labels_per_pair_pmid = [row[0] for row in con.execute("SELECT n_distinct_label_tuples FROM pair_pmid").fetchall()]
    diseases_per_gene = [row[0] for row in con.execute("SELECT COUNT(DISTINCT disease_cui) FROM pair GROUP BY gene_id").fetchall()]
    genes_per_disease = [row[0] for row in con.execute("SELECT COUNT(DISTINCT gene_id) FROM pair GROUP BY disease_cui").fetchall()]
    pairs_per_pmid = [
        row[0]
        for row in con.execute(
            """
            SELECT COUNT(DISTINCT gene_id || '|' || disease_cui) FROM pair_pmid GROUP BY pmid
            """
        ).fetchall()
    ]
    return {
        "pmid_per_explicit_pair": summarize_numeric(pmid_per_pair, "explicit_pairs", "pmid"),
        "sentence_per_explicit_pair": summarize_numeric(sent_per_pair, "explicit_pairs", "sentence"),
        "gcs_per_explicit_pair": summarize_numeric(gcs_per_pair, "explicit_pairs", "gcs"),
        "sentence_per_pair_pmid": summarize_numeric(sent_per_pair_pmid, "explicit_pair_pmid", "sentence"),
        "label_tuples_per_pair_pmid": summarize_numeric(labels_per_pair_pmid, "explicit_pair_pmid", "label_tuple"),
        "diseases_per_gene": summarize_numeric(diseases_per_gene, "explicit_genes", "disease"),
        "genes_per_disease": summarize_numeric(genes_per_disease, "explicit_diseases", "gene"),
        "pairs_per_pmid": summarize_numeric(pairs_per_pmid, "pmid_with_explicit_pair", "pair"),
        "head_1pct_pairs_by_pmid_weight_pmid": head_share(pmid_per_pair, 0.01),
        "head_5pct_pairs_by_pmid_weight_pmid": head_share(pmid_per_pair, 0.05),
        "head_1pct_pairs_by_sentence_weight_sentence": head_share(sent_per_pair, 0.01),
        "head_5pct_pairs_by_sentence_weight_sentence": head_share(sent_per_pair, 0.05),
        "n_pairs_single_pmid": sum(1 for n in pmid_per_pair if n == 1),
        "n_pairs_multi_pmid": sum(1 for n in pmid_per_pair if n > 1),
        "n_pairs_zero_pmid": sum(1 for n in pmid_per_pair if n == 0),
    }


def _write_support_tables(con, out_dir: Path, summary: dict) -> None:
    _write_tsv(out_dir / "predicate_counts.tsv", summary["predicate_counts"])
    _write_tsv(out_dir / "type_counts.tsv", summary["type_counts"])
    _write_tsv(out_dir / "yearly.tsv", summary["time"]["by_year"])
    _write_tsv(out_dir / "gcs_pair_status.tsv", summary["gcs_pair_status"])
    field_rows = []
    for name, uri in CORE_PREDICATES.items():
        field_rows.append(
            {
                "source": "CORE",
                "predicate": name,
                "uri": uri,
                "owl_enum": "|".join(OWL_LABEL_ENUM.get(name, [])),
                "builder_writes": "yes" if name not in {"cites", "hasCitation", "CGE", "CCS", "GCI", "GCC"} else "not_in_convert_functions",
                "local_confirmation": "measured_in_predicate_counts",
            }
        )
    field_rows.append(
        {
            "source": "CORE",
            "predicate": "hasType",
            "uri": CORE_PREDICATES["hasType"],
            "owl_enum": "BIOMARKER|ONCOGENE|TSG",
            "builder_writes": "yes_including_UNRELIABLE_outside_owl",
            "local_confirmation": "measured_in_gcs_has_type",
        }
    )
    _write_tsv(out_dir / "field_dictionary.tsv", field_rows)
    known_types = {
        TYPE_GENE: "gene",
        TYPE_DISEASE: "disease",
        TYPE_GCS: "gcs",
        TYPE_SENTENCE: "sentence",
        TYPE_SENTENCE_EQUIV: "sentence_equivalent",
        TYPE_ARTICLE: "article",
        TYPE_DATABASE: "database",
    }
    summary["known_type_uris"] = known_types


def _write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def usable_status_set() -> set[str]:
    return {s for s in ("explicit_namespace", "scheme_variant_explicit", "explicit_namespace_pattern_unusual") if usable_id_status(s) or s == "explicit_namespace_pattern_unusual"}

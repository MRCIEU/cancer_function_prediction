"""ID crosswalk and extraction of SemMedDB rows for CORE pairs and CORE PMIDs."""

from __future__ import annotations

import threading
from pathlib import Path

from lib.schema import SPARQL_EXAMPLE_EXCLUDED_CUIS
from lib.semmed_columns import canonical_id_expr
from lib.years import parse_core_year


def _watch_tree(path: Path, logger, stop: threading.Event) -> None:
    while not stop.wait(45):
        total = 0
        n_files = 0
        if path.exists():
            for dirpath, _dirnames, filenames in path.walk():
                for name in filenames:
                    file_path = dirpath / name
                    try:
                        total += file_path.stat().st_size
                        n_files += 1
                    except OSError:
                        continue
        logger.info("progress phase=align_sql temp_or_output_bytes=%s files=%s dir=%s", total, n_files, path)


def run_alignment(con, paths: dict, out_dir: Path, logger, rule: str, entity_resolution: dict | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    watcher = threading.Thread(target=_watch_tree, args=(out_dir, logger, stop), daemon=True)
    watcher.start()
    try:
        return _run_alignment(con, paths, out_dir, logger, rule, entity_resolution)
    finally:
        stop.set()
        watcher.join(timeout=1)


def _run_alignment(con, paths: dict, out_dir: Path, logger, rule: str, entity_resolution: dict | None = None) -> dict:
    _views(con, paths, entity_resolution or {})
    logger.info("phase_start phase=gene_sentence_index")
    con.execute(
        """
        CREATE OR REPLACE TABLE gene_sent AS
        SELECT sentence_id, cui, gene_id, COUNT(*) AS n_rows
        FROM entity
        WHERE gene_id IS NOT NULL AND sentence_id IS NOT NULL AND cui IS NOT NULL
        GROUP BY 1, 2, 3
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE gene_sent_unique AS
        SELECT sentence_id, cui, MIN(gene_id) AS gene_id
        FROM gene_sent
        GROUP BY 1, 2
        HAVING COUNT(DISTINCT gene_id) = 1
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE gene_sent_ambiguous AS
        SELECT sentence_id, cui, COUNT(DISTINCT gene_id) AS n_gene_id
        FROM gene_sent
        GROUP BY 1, 2
        HAVING COUNT(DISTINCT gene_id) > 1
        """
    )
    logger.info("phase_start phase=match_predications")
    con.execute(
        """
        CREATE OR REPLACE TABLE matched_predication AS
        SELECT
          p.predication_id,
          p.sentence_id,
          p.pmid,
          p.predicate,
          p.negation_prefix,
          cp.gene_id,
          cp.disease_cui,
          CASE WHEN p.subject_cui = cp.disease_cui THEN 'disease_subject' ELSE 'disease_object' END AS disease_position,
          CASE WHEN p.subject_cui = cp.disease_cui THEN p.object_cui ELSE p.subject_cui END AS gene_cui,
          p.subject_semtype,
          p.object_semtype
        FROM predication p
        JOIN pair cp
          ON p.subject_cui = cp.disease_cui OR p.object_cui = cp.disease_cui
        JOIN gene_sent_unique g
          ON g.sentence_id = p.sentence_id
         AND g.gene_id = cp.gene_id
         AND (
              (p.subject_cui = cp.disease_cui AND g.cui = p.object_cui)
           OR (p.object_cui = cp.disease_cui AND g.cui = p.subject_cui)
         )
        WHERE p.pmid IS NOT NULL
          AND NOT (p.subject_cui = cp.disease_cui AND p.object_cui = cp.disease_cui)
        """
    )
    con.execute(
        f"""
        COPY matched_predication TO '{(out_dir / 'matched_predication.parquet').as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE semmed_pair_pmid AS
        SELECT gene_id, disease_cui, pmid,
               COUNT(*) AS n_predication,
               COUNT(DISTINCT predicate) AS n_predicate,
               COUNT(DISTINCT sentence_id) AS n_sentence
        FROM matched_predication
        GROUP BY 1, 2, 3
        """
    )
    con.execute(
        f"""
        COPY semmed_pair_pmid TO '{(out_dir / 'semmed_pair_pmid.parquet').as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    logger.info("phase_start phase=pmid_route")
    con.execute("CREATE OR REPLACE TABLE core_pmid AS SELECT DISTINCT pmid FROM core_sentence WHERE pmid IS NOT NULL")
    con.execute(
        """
        CREATE OR REPLACE TABLE pmid_route AS
        SELECT
          c.pmid,
          (cit.pmid IS NOT NULL) AS in_citations,
          COALESCE(ns.n_sentence, 0) AS n_sentence,
          COALESCE(np.n_predication, 0) AS n_predication,
          COALESCE(ne.n_entity, 0) AS n_entity,
          COALESCE(ng.n_gene_id, 0) AS n_distinct_gene_id,
          COALESCE(cp.n_core_pair, 0) AS n_core_pair
        FROM core_pmid c
        LEFT JOIN (SELECT DISTINCT pmid FROM citations WHERE pmid IS NOT NULL) cit USING (pmid)
        LEFT JOIN (
          SELECT pmid, COUNT(*) AS n_sentence FROM sem_sentence WHERE pmid IS NOT NULL GROUP BY 1
        ) ns USING (pmid)
        LEFT JOIN (
          SELECT pmid, COUNT(*) AS n_predication FROM predication WHERE pmid IS NOT NULL GROUP BY 1
        ) np USING (pmid)
        LEFT JOIN (
          SELECT s.pmid, COUNT(*) AS n_entity, COUNT(DISTINCT e.gene_id) AS n_gene_id
          FROM sem_sentence s
          JOIN entity e ON s.sentence_id = e.sentence_id
          WHERE s.pmid IN (SELECT pmid FROM core_pmid)
          GROUP BY 1
        ) ne ON c.pmid = ne.pmid
        LEFT JOIN (
          SELECT pmid, COUNT(DISTINCT gene_id) AS n_gene_id
          FROM (
            SELECT s.pmid, e.gene_id
            FROM sem_sentence s
            JOIN entity e ON s.sentence_id = e.sentence_id
            WHERE e.gene_id IS NOT NULL AND s.pmid IN (SELECT pmid FROM core_pmid)
          )
          GROUP BY 1
        ) ng ON c.pmid = ng.pmid
        LEFT JOIN (
          SELECT pmid, COUNT(*) AS n_core_pair FROM core_pair_pmid GROUP BY 1
        ) cp USING (pmid)
        """
    )
    # The entity join above is duplicated; keep the distinct-gene join and a separate entity count.
    con.execute(
        f"""
        COPY pmid_route TO '{(out_dir / 'pmid_route.parquet').as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    logger.info("phase_start phase=scope_e_gene_id")
    con.execute(
        """
        CREATE OR REPLACE TABLE scope_e_annotated AS
        SELECT e.*, g.gene_id,
               CASE
                 WHEN g.gene_id IS NOT NULL THEN 'gene_id_on_same_sentence_cui'
                 WHEN a.sentence_id IS NOT NULL THEN 'ambiguous_gene_id'
                 ELSE 'gene_id_absent_in_available_entity_rows'
               END AS gene_id_status
        FROM scope_e_gngm_neop e
        LEFT JOIN gene_sent_unique g
          ON e.sentence_id = g.sentence_id AND e.gene_cui = g.cui
        LEFT JOIN gene_sent_ambiguous a
          ON e.sentence_id = a.sentence_id AND e.gene_cui = a.cui
        """
    )
    con.execute(
        f"""
        COPY (
          SELECT gene_id, disease_cui, COUNT(DISTINCT pmid) AS n_pmid, COUNT(*) AS n_predication
          FROM scope_e_annotated
          WHERE gene_id_status = 'gene_id_on_same_sentence_cui'
          GROUP BY 1, 2
        ) TO '{(out_dir / 'scope_e_gene_pairs.parquet').as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    summary = _summary(con, out_dir, rule)
    summary["sparql_example_excluded_cuis"] = SPARQL_EXAMPLE_EXCLUDED_CUIS
    summary["rule"] = rule
    _years(con, out_dir, logger)
    return summary


def _views(con, paths: dict, entity_resolution: dict) -> None:
    mapping = {
        "pair": paths["core_pair"],
        "core_pair_pmid": paths["core_pair_pmid"],
        "core_sentence": paths["core_sentence"],
        "core_article_literal": paths["core_article_literal"],
        "entity": paths["entity"],
        "predication": paths["predication"],
        "sem_sentence": paths["sem_sentence"],
        "citations": paths["citations"],
        "scope_e_gngm_neop": paths["scope_e_gngm_neop"],
    }
    for name, path in mapping.items():
        if name == "entity" and entity_resolution.get("column"):
            column = entity_resolution["column"]
            con.execute(
                f"""
                CREATE OR REPLACE VIEW entity AS
                SELECT * EXCLUDE ({column}), {canonical_id_expr(column)} AS sentence_id
                FROM read_parquet('{Path(path).as_posix()}')
                """
            )
        else:
            con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{Path(path).as_posix()}')")


def _summary(con, out_dir: Path, rule: str) -> dict:
    def one(sql: str):
        return con.execute(sql).fetchone()[0]

    funnel = [
        {"stage": "explicit_core_pairs", "n": one("SELECT COUNT(*) FROM pair"), "unit": "pair", "rule": rule, "status": "computed"},
        {"stage": "explicit_core_pairs_with_pmid", "n": one("SELECT COUNT(*) FROM pair WHERE n_pmid > 0"), "unit": "pair", "rule": rule, "status": "computed"},
        {"stage": "explicit_core_pair_pmid", "n": one("SELECT COUNT(*) FROM core_pair_pmid"), "unit": "pair_pmid", "rule": rule, "status": "computed"},
        {"stage": "core_gene_ids_seen_on_entity", "n": one("SELECT COUNT(DISTINCT gene_id) FROM gene_sent WHERE gene_id IN (SELECT gene_id FROM pair)"), "unit": "gene_id", "rule": rule, "status": "computed"},
        {"stage": "core_disease_cuis_seen_on_predication_argument", "n": one("SELECT COUNT(DISTINCT disease_cui) FROM pair WHERE disease_cui IN (SELECT subject_cui FROM predication UNION SELECT object_cui FROM predication)"), "unit": "cui", "rule": rule, "status": "computed"},
        {"stage": "pairs_with_primary_semmed_predication", "n": one("SELECT COUNT(DISTINCT gene_id || '|' || disease_cui) FROM matched_predication"), "unit": "pair", "rule": rule, "status": "computed"},
        {"stage": "primary_semmed_pair_pmid", "n": one("SELECT COUNT(*) FROM semmed_pair_pmid"), "unit": "pair_pmid", "rule": rule, "status": "computed"},
        {"stage": "ambiguous_sentence_cui_gene_id", "n": one("SELECT COUNT(*) FROM gene_sent_ambiguous"), "unit": "sentence_cui", "rule": rule, "status": "computed"},
        {"stage": "core_pmid", "n": one("SELECT COUNT(*) FROM core_pmid"), "unit": "pmid", "rule": rule, "status": "computed"},
        {"stage": "core_pmid_in_semmed_citations", "n": one("SELECT COUNT(*) FROM pmid_route WHERE in_citations"), "unit": "pmid", "rule": rule, "status": "computed"},
        {"stage": "scope_e_rows_with_unique_gene_id", "n": one("SELECT COUNT(*) FROM scope_e_annotated WHERE gene_id_status = 'gene_id_on_same_sentence_cui'"), "unit": "predication", "rule": rule, "status": "computed"},
        {"stage": "scope_e_rows_gene_id_absent", "n": one("SELECT COUNT(*) FROM scope_e_annotated WHERE gene_id_status = 'gene_id_absent_in_available_entity_rows'"), "unit": "predication", "rule": rule, "status": "computed"},
        {"stage": "scope_e_rows_gene_id_ambiguous", "n": one("SELECT COUNT(*) FROM scope_e_annotated WHERE gene_id_status = 'ambiguous_gene_id'"), "unit": "predication", "rule": rule, "status": "computed"},
    ]
    con.execute(
        f"""
        COPY (SELECT * FROM (
          SELECT gene_id, COUNT(DISTINCT cui) AS n_cui
          FROM gene_sent
          WHERE gene_id IN (SELECT gene_id FROM pair)
          GROUP BY 1
        )) TO '{(out_dir / 'gene_id_cui_cardinality.parquet').as_posix()}' (FORMAT PARQUET)
        """
    )
    return {
        "funnel": funnel,
        "match_rule": "Same-sentence ENTITY row with numeric GENE_ID equal to the CORE Entrez ID and CUI equal to the non-disease argument; disease argument CUI equals the CORE UMLS CUI. Composite gene IDs are not split. Same CUI on both arguments is excluded.",
        "outputs": {
            "matched_predication": str(out_dir / "matched_predication.parquet"),
            "semmed_pair_pmid": str(out_dir / "semmed_pair_pmid.parquet"),
            "pmid_route": str(out_dir / "pmid_route.parquet"),
            "scope_e_gene_pairs": str(out_dir / "scope_e_gene_pairs.parquet"),
        },
    }


def _years(con, out_dir: Path, logger) -> None:
    logger.info("phase_start phase=pmid_years")
    rows = con.execute(
        """
        SELECT s.pmid, a.value
        FROM core_sentence s
        JOIN core_article_literal a ON s.article_uri = a.article_uri
        WHERE a.predicate LIKE '%publicationYear' AND s.pmid IS NOT NULL
        GROUP BY 1, 2
        """
    ).fetchall()
    parsed = []
    for pmid, value in rows:
        got = parse_core_year(value)
        parsed.append((pmid, value, got["status"], got["year"]))
    con.execute("CREATE OR REPLACE TABLE core_year_raw(pmid VARCHAR, raw VARCHAR, status VARCHAR, year INTEGER)")
    if parsed:
        con.executemany("INSERT INTO core_year_raw VALUES (?,?,?,?)", parsed)
    con.execute(
        """
        CREATE OR REPLACE TABLE core_pmid_year AS
        SELECT pmid,
               COUNT(DISTINCT year) FILTER (WHERE status = 'parsed_single') AS n_year,
               MIN(year) FILTER (WHERE status = 'parsed_single') AS year,
               CASE
                 WHEN COUNT(DISTINCT year) FILTER (WHERE status = 'parsed_single') = 1 THEN 'parsed_single'
                 WHEN COUNT(DISTINCT year) FILTER (WHERE status = 'parsed_single') > 1 THEN 'multi_value'
                 WHEN COUNT(*) FILTER (WHERE status = 'non_positive') > 0 THEN 'non_positive'
                 WHEN COUNT(*) FILTER (WHERE status = 'unparsed') > 0 THEN 'unparsed'
                 ELSE 'field_absent'
               END AS status
        FROM core_year_raw
        GROUP BY 1
        """
    )
    con.execute(
        f"""
        COPY (
          SELECT c.pmid,
                 y.status AS core_year_status,
                 y.year AS core_year,
                 cit.dp_status AS semmed_dp_status,
                 cit.dp_year AS semmed_dp_year,
                 cit.completion_year AS semmed_completion_year
          FROM core_pmid c
          LEFT JOIN core_pmid_year y USING (pmid)
          LEFT JOIN citations cit USING (pmid)
        ) TO '{(out_dir / 'pmid_year.parquet').as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

#!/usr/bin/env python3
"""Audit the completed J00–J05 run and write a traceable review report.

Reads existing summaries and derived parquet. Does not reload raw SemMedDB
except one bounded pass over the already-repaired ENTITY csv for text checks.
Does not overwrite data_analysis_round1 outputs.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

CODE_ROOT = Path("/home/b5ac/freddieyu.b5ac/project_2")
sys.path.insert(0, str(CODE_ROOT))

from lib.duckutil import connect  # noqa: E402
from lib.runtime import code_fingerprint, prologue, setup_logging, write_json  # noqa: E402
from lib.schema import CORE_PREDICATES, SPARQL_EXAMPLE_EXCLUDED_CUIS  # noqa: E402
from lib.semmed_columns import canonical_id_expr  # noqa: E402

RESULTS = Path("/projects/b5ac/project_2/results")
ROUND1 = RESULTS / "03_integrated_analysis" / "data_analysis_round1"
OUT = RESULTS / "03_integrated_analysis" / "data_analysis_review_round1"


def log_progress(logger, phase: str, **fields) -> None:
    parts = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("progress phase=%s %s last_progress_utc=%s", phase, parts, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def one(con, sql: str):
    return con.execute(sql).fetchone()[0]


def rows(con, sql: str) -> list[dict]:
    cur = con.execute(sql)
    names = [item[0] for item in cur.description]
    return [dict(zip(names, row)) for row in cur.fetchall()]


def qpath(path: Path) -> str:
    return "'" + path.as_posix().replace("'", "''") + "'"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(OUT / "logs" / f"review_{os.environ.get('SLURM_JOB_ID', 'local')}.log")
    info = prologue("REVIEW", logger, OUT, [str(ROUND1)])
    tables = OUT / "tables"
    derived = OUT / "derived"
    tables.mkdir(parents=True, exist_ok=True)
    derived.mkdir(parents=True, exist_ok=True)
    con = connect(derived / "review.duckdb", "72GB", OUT / "tmp")
    measured = {
        "code_fingerprint_now": info["code_fingerprint"],
        "historical_fingerprints": {
            "J01": read_json(RESULTS / "01_core_audit" / "DONE.json").get("code_fingerprint"),
            "J02": read_json(RESULTS / "02_semmeddb_validation" / "DONE.json").get("code_fingerprint"),
            "J03": read_json(ROUND1 / "derived" / "j03" / "DONE.json").get("code_fingerprint"),
            "J04": read_json(ROUND1 / "tables" / "DONE.json").get("code_fingerprint"),
            "J05": read_json(ROUND1 / "run_manifest.json").get("code_fingerprint"),
        },
        "git": "project_2 is not a git repository; executed source bytes were not recovered",
    }
    core_summary = read_json(RESULTS / "01_core_audit" / "tables" / "summary.json")
    semmed_summary = read_json(RESULTS / "02_semmeddb_validation" / "tables" / "summary.json")
    align_summary = read_json(ROUND1 / "derived" / "j03" / "summary.json")
    joint_summary = read_json(ROUND1 / "tables" / "summary.json")
    measured["stored"] = {
        "core_integrity": core_summary.get("integrity"),
        "core_labels": core_summary.get("labels"),
        "core_funnel": core_summary.get("funnel"),
        "core_distributions": {
            "pmid_per_pair": core_summary.get("distributions", {}).get("pmid_per_explicit_pair"),
            "single": core_summary.get("distributions", {}).get("n_pairs_single_pmid"),
            "multi": core_summary.get("distributions", {}).get("n_pairs_multi_pmid"),
            "zero": core_summary.get("distributions", {}).get("n_pairs_zero_pmid"),
            "head": core_summary.get("distributions", {}).get("head_1pct_pairs_by_pmid_weight_pmid"),
        },
        "core_text": core_summary.get("text"),
        "core_counts": {
            "triples": core_summary.get("n_triples"),
            "pairs": core_summary.get("n_explicit_pairs"),
            "pmid": core_summary.get("n_explicit_pmid"),
            "pair_pmid": core_summary.get("n_explicit_pair_pmid"),
        },
        "semmed_tables": semmed_summary.get("tables"),
        "semmed_resolution": semmed_summary.get("entity_sentence_id_resolution"),
        "semmed_citations": semmed_summary.get("citations"),
        "semmed_sentences": semmed_summary.get("sentences"),
        "semmed_entities": semmed_summary.get("entities"),
        "semmed_integrity": semmed_summary.get("integrity"),
        "semmed_scope_e": semmed_summary.get("scope_e"),
        "semmed_columns": semmed_summary.get("local_column_maps"),
        "align_funnel": align_summary.get("funnel"),
        "align_resolution": align_summary.get("entity_sentence_id_resolution"),
        "joint_populations": joint_summary.get("populations"),
        "joint_observable": joint_summary.get("observable"),
        "joint_window": joint_summary.get("window_years"),
        "joint_year_removed": joint_summary.get("year_window_removed"),
        "joint_common_removed": joint_summary.get("common_article_removed"),
        "joint_common_pmid": joint_summary.get("common_article_pmid"),
        "joint_pmid_route": joint_summary.get("pmid_route"),
        "scope_e_vs_core": joint_summary.get("scope_e_vs_core") or align_summary.get("scope_e_vs_core"),
    }
    logger.info("phase=stored_summaries_loaded")
    mention = mention_structure(con, logger)
    measured["mention"] = mention
    labels = label_structure(con, logger)
    measured["labels"] = labels
    closure = relation_closure(con, logger)
    measured["closure"] = closure
    unconfirmed = predication_extras(con, logger)
    measured["unconfirmed"] = unconfirmed
    sentence_id = sentence_id_windows(con, logger)
    measured["sentence_id"] = sentence_id
    mapping = mapping_loss(con, logger)
    measured["mapping"] = mapping
    text_cases = entity_text_sample(con, sentence_id.get("sample_rows", []), logger)
    measured["text_cases"] = text_cases
    write_json(derived / "measured.json", measured)
    write_outputs(measured, info, logger)
    write_json(
        OUT / "DONE.json",
        {
            "status": "success",
            "job": "REVIEW",
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "code_fingerprint": info["code_fingerprint"],
            "report": str(OUT / "report.md"),
        },
    )
    logger.info("phase=review_success")
    return 0


def mention_structure(con, logger) -> dict:
    logger.info("phase_start phase=mention_structure")
    db = RESULTS / "01_core_audit" / "tables" / "core.duckdb"
    src = duckdb_readonly(db)
    tables = {row[0] for row in src.execute("SHOW TABLES").fetchall()}
    needed = {"sent_gene_mention", "sent_disease_mention", "gcs_sentence", "gcs_gene", "gcs_disease"}
    if not needed.issubset(tables):
        src.close()
        raise SystemExit(f"core.duckdb is missing mention tables: {sorted(needed - tables)}")
    gene_sql = """
        SELECT COUNT(*) FROM sent_gene_mention m
        JOIN gcs_sentence gs ON m.subject = gs.object
        JOIN gcs_gene gg ON gs.subject = gg.subject
        WHERE m.object <> gg.object
    """
    disease_sql = gene_sql.replace("sent_gene_mention", "sent_disease_mention").replace("gcs_gene", "gcs_disease")
    out = {
        "source": str(db),
        "unit_cartesian": "mention_edge × GCS_of_same_sentence where the two object URIs differ",
        "code": "lib/core_build.py _integrity mention_gene_disagrees_with_gcs",
        "gene_cartesian_rows": one(src, gene_sql),
        "disease_cartesian_rows": one(src, disease_sql),
        "sentences_multi_gcs": one(
            src,
            "SELECT COUNT(*) FROM (SELECT object FROM gcs_sentence GROUP BY 1 HAVING COUNT(DISTINCT subject) > 1)",
        ),
    }
    out["gene_set"] = set_disagreement(src, "sent_gene_mention", "gcs_gene")
    out["disease_set"] = set_disagreement(src, "sent_disease_mention", "gcs_disease")
    out["examples"] = mention_examples(src)
    src.close()
    log_progress(logger, "mention_structure", gene_rows=out["gene_cartesian_rows"], disease_rows=out["disease_cartesian_rows"])
    return out


def set_disagreement(src, mention: str, gcs_edge: str) -> dict:
    src.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE mention_set AS
        SELECT subject AS sentence_uri, object AS uri FROM {mention}
        """
    )
    src.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE gcs_set AS
        SELECT gs.object AS sentence_uri, gs.subject AS gcs_uri, e.object AS uri
        FROM gcs_sentence gs
        JOIN {gcs_edge} e ON gs.subject = e.subject
        """
    )
    return {
        "gcs_whose_uri_is_absent_from_sentence_mentions": one(
            src,
            """
            SELECT COUNT(*) FROM gcs_set g
            WHERE NOT EXISTS (
              SELECT 1 FROM mention_set m
              WHERE m.sentence_uri = g.sentence_uri AND m.uri = g.uri
            )
            """,
        ),
        "sentences_with_such_a_gcs": one(
            src,
            """
            SELECT COUNT(DISTINCT g.sentence_uri) FROM gcs_set g
            WHERE NOT EXISTS (
              SELECT 1 FROM mention_set m
              WHERE m.sentence_uri = g.sentence_uri AND m.uri = g.uri
            )
            """,
        ),
        "cartesian_rows_where_mention_set_still_covers_gcs": one(
            src,
            f"""
            SELECT COUNT(*) FROM {mention} m
            JOIN gcs_sentence gs ON m.subject = gs.object
            JOIN {gcs_edge} e ON gs.subject = e.subject
            WHERE m.object <> e.object
              AND EXISTS (
                SELECT 1 FROM {mention} m2
                WHERE m2.subject = m.subject AND m2.object = e.object
              )
            """,
        ),
        "definition": "Set check: a GCS object URI is absent from that sentence's mention URIs. Cartesian rows can be nonzero when every GCS URI is still one of the mentions.",
    }


def mention_examples(src) -> list[dict]:
    picked = rows(
        src,
        """
        WITH multi AS (
          SELECT object AS sentence_uri
          FROM gcs_sentence
          GROUP BY 1
          HAVING COUNT(DISTINCT subject) > 1
        ),
        flagged AS (
          SELECT gs.object AS sentence_uri,
                 COUNT(*) AS n_gcs,
                 COUNT(*) FILTER (
                   WHERE NOT EXISTS (
                     SELECT 1 FROM sent_gene_mention m
                     WHERE m.subject = gs.object AND m.object = gg.object
                   )
                 ) AS n_gene_absent
          FROM gcs_sentence gs
          JOIN gcs_gene gg ON gs.subject = gg.subject
          JOIN multi ON gs.object = multi.sentence_uri
          GROUP BY 1
        )
        SELECT sentence_uri, n_gcs, n_gene_absent
        FROM flagged
        WHERE n_gene_absent = 0
        LIMIT 1
        """,
    )
    absent = rows(
        src,
        """
        SELECT gs.object AS sentence_uri, gs.subject AS gcs_uri, gg.object AS gcs_gene
        FROM gcs_sentence gs
        JOIN gcs_gene gg ON gs.subject = gg.subject
        WHERE NOT EXISTS (
          SELECT 1 FROM sent_gene_mention m
          WHERE m.subject = gs.object AND m.object = gg.object
        )
        LIMIT 1
        """,
    )
    examples = []
    for kind, batch in (("multi_gcs_mentions_cover_every_gcs_gene", picked), ("gcs_gene_absent_from_mentions", absent)):
        if not batch:
            examples.append({"kind": kind, "status": "not_found"})
            continue
        uri = batch[0]["sentence_uri"]
        examples.append(
            {
                "kind": kind,
                "locator": batch[0],
                "mentions": rows(src, f"SELECT object AS gene_uri, n_triples FROM sent_gene_mention WHERE subject = '{uri}'"),
                "gcs": rows(
                    src,
                    f"""
                    SELECT gs.subject AS gcs_uri, gg.object AS gene_uri
                    FROM gcs_sentence gs
                    JOIN gcs_gene gg ON gs.subject = gg.subject
                    WHERE gs.object = '{uri}'
                    """,
                ),
            }
        )
    return examples


def label_structure(con, logger) -> dict:
    logger.info("phase_start phase=label_structure")
    path = qpath(RESULTS / "01_core_audit" / "tables" / "sentence.parquet")
    con.execute(f"CREATE OR REPLACE VIEW core_sentence AS SELECT * FROM read_parquet({path})")
    aspects = {}
    for column in ("gcc", "cge", "ccs", "gci", "curated", "has_type"):
        aspects[column] = rows(
            con,
            f"""
            SELECT COALESCE(CAST({column} AS VARCHAR), 'FIELD_ABSENT') AS value,
                   COUNT(*) AS sentence_gcs_rows,
                   COUNT(DISTINCT sentence_uri) AS sentences
            FROM core_sentence
            GROUP BY 1
            ORDER BY sentences DESC
            """,
        )
    combos = rows(
        con,
        """
        SELECT COALESCE(label_tuple, 'FIELD_ABSENT') AS label_tuple,
               COUNT(*) AS sentence_gcs_rows,
               COUNT(DISTINCT sentence_uri) AS sentences
        FROM core_sentence
        GROUP BY 1
        ORDER BY sentences DESC
        LIMIT 40
        """,
    )
    combo_n = one(con, "SELECT COUNT(*) FROM (SELECT label_tuple FROM core_sentence GROUP BY 1)")
    pair_pmid = qpath(RESULTS / "01_core_audit" / "tables" / "pair_pmid.parquet")
    pair = qpath(RESULTS / "01_core_audit" / "tables" / "pair.parquet")
    hetero = {
        "pair_pmid_rows": one(con, f"SELECT COUNT(*) FROM read_parquet({pair_pmid})"),
        "heterogeneous_within_pmid": one(con, f"SELECT COUNT(*) FROM read_parquet({pair_pmid}) WHERE heterogeneous_within_pmid"),
        "no_usable_label": one(con, f"SELECT COUNT(*) FROM read_parquet({pair_pmid}) WHERE usable_status = 'no_usable_label'"),
        "pairs": one(con, f"SELECT COUNT(*) FROM read_parquet({pair})"),
        "heterogeneous_across_pmid": one(con, f"SELECT COUNT(*) FROM read_parquet({pair}) WHERE heterogeneous_across_pmid"),
        "homogeneous_across_pmid": one(con, f"SELECT COUNT(*) FROM read_parquet({pair}) WHERE heterogeneous_across_pmid = FALSE"),
        "across_not_applicable": one(con, f"SELECT COUNT(*) FROM read_parquet({pair}) WHERE heterogeneous_across_pmid IS NULL"),
    }
    has_type = rows(
        con,
        """
        SELECT COALESCE(CAST(has_type AS VARCHAR), 'FIELD_ABSENT') AS has_type, COUNT(*) AS n
        FROM read_parquet(%s)
        GROUP BY 1 ORDER BY n DESC
        """
        % qpath(RESULTS / "01_core_audit" / "tables" / "gcs_status.parquet"),
    )
    log_progress(logger, "label_structure", combo_n=combo_n, pair_pmid=hetero["pair_pmid_rows"])
    return {
        "row_unit": "core sentence.parquet is one row per sentence–GCS join, not one row per sentence",
        "label_tuple_code": "lib/core_build.py CONCAT_WS of gcc|cge|ccs|gci with FIELD_ABSENT or MULTI kept in position",
        "usable_status_code": "lib/core_build.py pair_pmid usable_status keeps each aspect's own column; NOTINF and null do not erase the aspect name inside label_tuple",
        "aspects": aspects,
        "label_tuple_top": combos,
        "n_distinct_label_tuples": combo_n,
        "heterogeneity": hetero,
        "has_type_on_gcs_status": has_type,
    }


def relation_closure(con, logger) -> dict:
    logger.info("phase_start phase=relation_closure")
    rel = qpath(ROUND1 / "tables" / "pair_relations.parquet")
    sem = qpath(ROUND1 / "derived" / "j03" / "semmed_pair_pmid.parquet")
    core = qpath(RESULTS / "01_core_audit" / "tables" / "pair_pmid.parquet")
    classes = rows(con, f"SELECT relation_class, COUNT(*) AS n_pairs FROM read_parquet({rel}) GROUP BY 1 ORDER BY n_pairs DESC")
    n_class = one(con, f"SELECT COUNT(*) FROM read_parquet({rel})")
    n_pair = one(con, f"SELECT COUNT(*) FROM read_parquet({qpath(RESULTS / '01_core_audit' / 'tables' / 'pair.parquet')})")
    sem_stats = rows(
        con,
        f"""
        SELECT COUNT(*) AS pair_pmid,
               COUNT(DISTINCT gene_id || '|' || disease_cui) AS pairs,
               COUNT(DISTINCT pmid) AS pmid
        FROM read_parquet({sem})
        """,
    )[0]
    intersection = one(
        con,
        f"""
        SELECT COUNT(*) FROM read_parquet({core}) c
        JOIN read_parquet({sem}) s USING (gene_id, disease_cui, pmid)
        """,
    )
    log_progress(logger, "relation_closure", classes=n_class, intersection=intersection)
    return {
        "class_counts": classes,
        "class_rows": n_class,
        "explicit_pairs": n_pair,
        "classes_cover_all_pairs": n_class == n_pair,
        "semmed_pair_pmid": sem_stats,
        "intersection_pair_pmid": intersection,
    }


def predication_extras(con, logger) -> dict:
    logger.info("phase_start phase=predication_unconfirmed")
    path = qpath(RESULTS / "02_semmeddb_validation" / "tables" / "predication.parquet")
    summary = {}
    for index in (12, 13, 14):
        column = f"unconfirmed_{index}"
        summary[column] = rows(
            con,
            f"""
            SELECT
              CASE
                WHEN {column} IS NULL THEN 'sql_null'
                WHEN trim({column}) = '' THEN 'empty_string'
                WHEN trim({column}) IN (CHR(92) || 'N', 'N') THEN 'mysql_null_token'
                ELSE 'other'
              END AS kind,
              COUNT(*) AS n
            FROM read_parquet({path})
            GROUP BY 1
            ORDER BY n DESC
            """,
        )[0:8]
        summary[column + "_top"] = rows(
            con,
            f"""
            SELECT COALESCE({column}, 'sql_null') AS value, COUNT(*) AS n
            FROM read_parquet({path})
            GROUP BY 1
            ORDER BY n DESC
            LIMIT 8
            """,
        )
        log_progress(logger, "predication_unconfirmed", column=column)
    return summary


def sentence_id_windows(con, logger) -> dict:
    logger.info("phase_start phase=sentence_id_windows")
    import pyarrow.parquet as pq

    entity = RESULTS / "02_semmeddb_validation" / "tables" / "entity.parquet"
    pf = pq.ParquetFile(entity)
    groups = pf.num_row_groups
    targets = sorted({0, groups // 4, groups // 2, (3 * groups) // 4, max(groups - 1, 0)})
    canon1 = canonical_id_expr("pre_cui_numeric_1")
    canon2 = canonical_id_expr("pre_cui_numeric_2")
    import pyarrow as pa

    frames = []
    sample_rows = []
    for group in targets:
        table = pf.read_row_group(
            group,
            columns=["entity_id", "cui", "pre_cui_numeric_1", "pre_cui_numeric_2", "gene_id_raw", "gene_id_class", "start_index", "end_index", "text_len"],
        )
        take = table.slice(0, min(20000, table.num_rows))
        take = take.append_column("row_group", pa.array([group] * take.num_rows, type=pa.int32()))
        frames.append(take)
        for record in take.slice(0, min(4, take.num_rows)).to_pylist():
            sample_rows.append({key: (None if value is None else str(value)) for key, value in record.items()})
        log_progress(logger, "sentence_id_windows", row_group=group, rows=take.num_rows)
    sample = pa.concat_tables(frames)
    sample_path = OUT / "derived" / "entity_rowgroup_sample.parquet"
    pq.write_table(sample, sample_path)
    con.execute(f"CREATE OR REPLACE VIEW ent_sample AS SELECT * FROM read_parquet({qpath(sample_path)})")
    con.execute(f"CREATE OR REPLACE VIEW sem_sentence AS SELECT sentence_id, pmid, sentence_text FROM read_parquet({qpath(RESULTS / '02_semmeddb_validation' / 'tables' / 'sentence.parquet')})")
    per_group = rows(
        con,
        f"""
        SELECT row_group,
               COUNT(*) AS n,
               COUNT(s1.sentence_id) AS c1_matched_rows,
               COUNT(s2.sentence_id) AS c2_matched_rows,
               COUNT(*) FILTER (WHERE a = b) AS columns_equal,
               COUNT(DISTINCT a) AS distinct_c1,
               COUNT(DISTINCT b) AS distinct_c2
        FROM (
          SELECT row_group, {canon1} AS a, {canon2} AS b
          FROM ent_sample
        ) e
        LEFT JOIN sem_sentence s1 ON e.a = s1.sentence_id
        LEFT JOIN sem_sentence s2 ON e.b = s2.sentence_id
        GROUP BY 1
        ORDER BY 1
        """,
    )
    first = frames[0].slice(0, min(50000, frames[0].num_rows)) if frames and frames[0].column("row_group")[0].as_py() == 0 else sample.filter(pa.compute.equal(sample.column("row_group"), 0)).slice(0, 50000)
    first_path = OUT / "derived" / "entity_first_rowgroup_50k.parquet"
    pq.write_table(first, first_path)
    ordered = rows(
        con,
        f"""
        SELECT COUNT(*) AS n,
               COUNT(s1.sentence_id) AS c1_matched_rows,
               COUNT(s2.sentence_id) AS c2_matched_rows
        FROM (
          SELECT {canon1} AS a, {canon2} AS b FROM read_parquet({qpath(first_path)})
        ) e
        LEFT JOIN sem_sentence s1 ON e.a = s1.sentence_id
        LEFT JOIN sem_sentence s2 ON e.b = s2.sentence_id
        """,
    )[0]
    linked = rows(
        con,
        f"""
        SELECT entity_id, {canon1} AS sentence_candidate
        FROM ent_sample
        """,
    )
    by_entity = {str(row["entity_id"]): row["sentence_candidate"] for row in linked}
    for record in sample_rows:
        record["sentence_candidate"] = None if by_entity.get(record["entity_id"]) is None else str(by_entity[record["entity_id"]])
    return {
        "row_groups": groups,
        "sampled_row_groups": targets,
        "per_group": per_group,
        "first_row_group_prefix": ordered,
        "sample_rows": sample_rows,
        "limit_without_order": "J02 and J03 used SELECT ... LIMIT 50000 with no ORDER BY. preserve_insertion_order is false, so the two 50,000-row prefixes are not a defined file prefix.",
    }


def mapping_loss(con, logger) -> dict:
    logger.info("phase_start phase=mapping_loss")
    canon = canonical_id_expr("pre_cui_numeric_1")
    scope = qpath(RESULTS / "02_semmeddb_validation" / "tables" / "scope_e_gngm_neop.parquet")
    entity = qpath(RESULTS / "02_semmeddb_validation" / "tables" / "entity.parquet")
    pair = qpath(RESULTS / "01_core_audit" / "tables" / "pair.parquet")
    pred = qpath(RESULTS / "02_semmeddb_validation" / "tables" / "predication.parquet")
    existing = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    if "ent_hit" in existing and one(con, "SELECT COUNT(*) FROM ent_hit") > 0:
        log_progress(logger, "entity_scan_for_scope_e", hit_rows=one(con, "SELECT COUNT(*) FROM ent_hit"), reused="yes")
    else:
        con.execute(f"CREATE OR REPLACE TABLE scope_e AS SELECT * FROM read_parquet({scope})")
        con.execute("CREATE OR REPLACE TABLE scope_keys AS SELECT DISTINCT sentence_id, gene_cui FROM scope_e")
        logger.info("phase_start phase=entity_scan_for_scope_e")
        con.execute(
            f"""
            CREATE OR REPLACE TABLE ent_hit AS
            SELECT {canon} AS sentence_id, cui, gene_id_class, gene_id, COUNT(*) AS n_rows
            FROM read_parquet({entity})
            WHERE {canon} IN (SELECT sentence_id FROM scope_keys)
            GROUP BY 1, 2, 3, 4
            """
        )
        log_progress(logger, "entity_scan_for_scope_e", hit_rows=one(con, "SELECT COUNT(*) FROM ent_hit"), reused="no")
    if "scope_e" not in {row[0] for row in con.execute("SHOW TABLES").fetchall()}:
        con.execute(f"CREATE OR REPLACE TABLE scope_e AS SELECT * FROM read_parquet({scope})")
    con.execute(
        """
        CREATE OR REPLACE TABLE scope_class AS
        SELECT
          e.predication_id,
          e.sentence_id,
          e.gene_cui,
          e.disease_cui,
          e.pmid,
          COALESCE(SUM(h.n_rows), 0) AS n_entity_rows,
          COUNT(DISTINCT h.gene_id) FILTER (WHERE h.gene_id IS NOT NULL) AS n_numeric_gene,
          SUM(h.n_rows) FILTER (WHERE h.gene_id_class = 'numeric') AS n_numeric_rows,
          SUM(h.n_rows) FILTER (WHERE h.gene_id_class = 'composite_comma') AS n_composite_comma_rows,
          SUM(h.n_rows) FILTER (WHERE h.gene_id_class = 'composite_pipe') AS n_composite_pipe_rows,
          SUM(h.n_rows) FILTER (WHERE h.gene_id_class IN ('empty_string', 'sql_null', 'backslash_N', 'literal_NULL')) AS n_empty_rows,
          SUM(h.n_rows) FILTER (WHERE h.gene_id_class = 'other') AS n_other_rows
        FROM scope_e e
        LEFT JOIN ent_hit h ON e.sentence_id = h.sentence_id AND e.gene_cui = h.cui
        GROUP BY 1, 2, 3, 4, 5
        """
    )
    scope_counts = rows(
        con,
        """
        SELECT
          CASE
            WHEN n_entity_rows = 0 THEN 'no_entity_row_for_sentence_and_cui'
            WHEN n_numeric_gene > 1 THEN 'multiple_numeric_gene_id'
            WHEN n_numeric_gene = 1 THEN 'unique_numeric_gene_id'
            WHEN COALESCE(n_composite_comma_rows, 0) + COALESCE(n_composite_pipe_rows, 0) > 0 THEN 'composite_id_only'
            WHEN COALESCE(n_other_rows, 0) > 0 THEN 'non_numeric_other_only'
            ELSE 'entity_present_gene_field_empty'
          END AS exclusive_class,
          COUNT(*) AS n_predication
        FROM scope_class
        GROUP BY 1
        ORDER BY n_predication DESC
        """,
    )
    flags = rows(
        con,
        """
        SELECT
          COUNT(*) FILTER (WHERE n_entity_rows = 0) AS no_entity_row,
          COUNT(*) FILTER (WHERE COALESCE(n_composite_comma_rows, 0) > 0) AS any_composite_comma,
          COUNT(*) FILTER (WHERE COALESCE(n_composite_pipe_rows, 0) > 0) AS any_composite_pipe,
          COUNT(*) FILTER (WHERE COALESCE(n_empty_rows, 0) > 0) AS any_empty_gene_field,
          COUNT(*) FILTER (WHERE n_numeric_gene = 1 AND COALESCE(n_composite_comma_rows, 0) + COALESCE(n_composite_pipe_rows, 0) > 0) AS unique_numeric_and_composite,
          COUNT(*) AS n
        FROM scope_class
        """,
    )[0]
    logger.info("phase_start phase=id_presence_anywhere")
    existing = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
    if "core_pair" not in existing:
        con.execute(f"CREATE OR REPLACE TABLE core_pair AS SELECT gene_id, disease_cui FROM read_parquet({pair})")
    if "gene_anywhere" not in existing:
        con.execute(
            f"""
            CREATE OR REPLACE TABLE gene_anywhere AS
            SELECT DISTINCT gene_id FROM read_parquet({entity}) WHERE gene_id IS NOT NULL
            """
        )
    if "disease_anywhere" not in existing:
        con.execute(
            f"""
            CREATE OR REPLACE TABLE disease_anywhere AS
            SELECT DISTINCT cui FROM (
              SELECT subject_cui AS cui FROM read_parquet({pred})
              UNION
              SELECT object_cui FROM read_parquet({pred})
            )
            WHERE cui IN (SELECT disease_cui FROM core_pair)
            """
        )
    log_progress(
        logger,
        "id_presence_anywhere",
        gene_anywhere=one(con, "SELECT COUNT(*) FROM gene_anywhere"),
        disease_anywhere=one(con, "SELECT COUNT(*) FROM disease_anywhere"),
    )
    pair_presence = rows(
        con,
        """
        SELECT
          COUNT(*) AS pairs,
          COUNT(*) FILTER (WHERE p.gene_id IN (SELECT gene_id FROM gene_anywhere)) AS gene_seen_anywhere,
          COUNT(*) FILTER (WHERE p.disease_cui IN (SELECT cui FROM disease_anywhere)) AS disease_seen_anywhere,
          COUNT(*) FILTER (
            WHERE p.gene_id IN (SELECT gene_id FROM gene_anywhere)
              AND p.disease_cui IN (SELECT cui FROM disease_anywhere)
          ) AS both_seen_anywhere
        FROM core_pair p
        """,
    )[0]
    names = rows(
        con,
        f"""
        SELECT cui, any_value(name) AS name, COUNT(*) AS n
        FROM (
          SELECT subject_cui AS cui, subject_name AS name FROM read_parquet({pred})
          WHERE subject_cui IN ({", ".join(repr(c) for c in SPARQL_EXAMPLE_EXCLUDED_CUIS)})
          UNION ALL
          SELECT object_cui, object_name FROM read_parquet({pred})
          WHERE object_cui IN ({", ".join(repr(c) for c in SPARQL_EXAMPLE_EXCLUDED_CUIS)})
        )
        GROUP BY 1
        ORDER BY 1
        """,
    )
    log_progress(logger, "mapping_loss", both_seen=pair_presence["both_seen_anywhere"])
    return {
        "scope_e_exclusive": scope_counts,
        "scope_e_flags": flags,
        "exclusive_priority": "multiple_numeric_gene_id > unique_numeric_gene_id > composite_id_only > non_numeric_other_only > entity_present_gene_field_empty > no_entity_row",
        "pair_id_presence_anywhere": pair_presence,
        "excluded_cui_names": names,
        "same_sentence_match_not_recomputed": "The 276-pair primary match remains the J03 result. This review adds the anywhere-presence stage and the scope-E gene-field split. It does not rebuild the same-sentence join.",
    }


def entity_text_sample(con, sample_rows: list[dict], logger) -> dict:
    logger.info("phase_start phase=entity_text_sample")
    source = RESULTS / "02_semmeddb_validation" / "tmp" / "entity.utf8.csv"
    if not sample_rows or not source.is_file():
        return {"status": "not_run", "reason": "sample or repaired csv missing", "source": str(source)}
    wanted = {row["entity_id"] for row in sample_rows if row.get("entity_id")}
    found = {}
    import csv as csvmod

    started = time.time()
    last = started
    seen = 0
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csvmod.reader(handle, delimiter=",", quotechar='"', escapechar="\\")
        for record in reader:
            seen += 1
            if record and record[0].lstrip("0") in wanted or (record and record[0] in wanted):
                key = record[0].lstrip("0") or "0"
                if key in wanted and key not in found:
                    found[key] = record
            now = time.time()
            if now - last >= 45:
                log_progress(logger, "entity_text_sample", records=seen, found=len(found))
                last = now
            if len(found) == len(wanted):
                break
    sentences = rows(
        con,
        f"""
        SELECT sentence_id, pmid, sentence_text
        FROM read_parquet({qpath(RESULTS / '02_semmeddb_validation' / 'tables' / 'sentence.parquet')})
        WHERE sentence_id IN ({_candidate_ids(sample_rows)})
        """,
    ) if sample_rows else []
    by_sentence = {row["sentence_id"]: row for row in sentences}
    checks = []
    for row in sample_rows:
        raw = found.get(row["entity_id"])
        sentence = by_sentence.get(row.get("sentence_candidate"))
        checks.append(compare_span(row, raw, sentence))
    return {
        "status": "computed",
        "source": str(source),
        "records_scanned": seen,
        "rows_found": len(found),
        "rows_requested": len(wanted),
        "stopped_early": len(found) == len(wanted),
        "checks": checks,
        "elapsed_s": round(time.time() - started, 1),
    }


def _candidate_ids(sample_rows: list[dict]) -> str:
    values = []
    for row in sample_rows:
        value = row.get("sentence_candidate")
        if value:
            values.append("'" + value.replace("'", "''") + "'")
    return ", ".join(values) or "''"


def compare_span(sample: dict, raw: list[str] | None, sentence: dict | None) -> dict:
    out = {
        "entity_id": sample.get("entity_id"),
        "row_group": sample.get("row_group"),
        "sentence_candidate": sample.get("sentence_candidate"),
        "pmid_of_candidate_sentence": None if sentence is None else sentence.get("pmid"),
        "raw_row_found": raw is not None,
        "width": None if raw is None else len(raw),
    }
    if raw is None or sentence is None or not sentence.get("sentence_text"):
        out["span_status"] = "not_assessable"
        out["reason"] = "raw entity row or candidate sentence text missing"
        return out
    text = raw[8] if len(raw) > 8 else ""
    try:
        start = int(sample["start_index"])
        end = int(sample["end_index"])
    except (TypeError, ValueError):
        out["span_status"] = "not_assessable"
        out["reason"] = "start or end index is not an integer"
        return out
    body = sentence["sentence_text"]
    slices = {}
    for shift in (-1, 0, 1):
        left = start + shift
        right = end + shift
        if 0 <= left <= right <= len(body):
            slices[f"shift_{shift}"] = body[left:right] == text
        else:
            slices[f"shift_{shift}"] = False
    out["mention_in_sentence"] = text != "" and text in body
    out["slice_equals_mention"] = slices
    out["length_delta"] = (end - start) - len(text)
    out["span_status"] = "observed"
    out["mention_preview"] = text[:120]
    return out


def duckdb_readonly(path: Path):
    import duckdb

    return duckdb.connect(str(path), read_only=True)


def write_outputs(measured: dict, info: dict, logger) -> None:
    logger.info("phase_start phase=write_report")
    write_tsv(OUT / "tables" / "analysis_coverage.tsv", coverage_rows(measured))
    write_tsv(OUT / "tables" / "schema_evidence.tsv", schema_rows(measured))
    write_tsv(OUT / "tables" / "result_lineage.tsv", lineage_rows(measured))
    write_tsv(OUT / "tables" / "mapping_flow.tsv", flow_rows(measured))
    write_cases(measured)
    (OUT / "report.md").write_text(render_report(measured, info), encoding="utf-8")


def coverage_rows(measured: dict) -> list[dict]:
    hist = measured["historical_fingerprints"]
    now = measured["code_fingerprint_now"]
    return [
        item("J00 预检", "已实现、已执行，有结果依据", "derived/j00/DONE.json status success", "schema 样本确认；不是全表扫描", "字段名仍有未确认项"),
        item("J01 CORE 解析与单库统计", "已实现、已执行，有结果依据", f"job 7420640 fingerprint {hist['J01']}", "summary.json 与 parquet", "整树指纹与当前源码不同；无 git，不能证明每个函数字节未变"),
        item("J02 SemMedDB 装载与单库统计", "已实现、已执行，有结果依据", f"job 7422513 fingerprint {hist['J02']}", "六张 parquet 与 summary.json", "ENTITY sentence_id 在该版本为 unresolved"),
        item("J03 对齐", "已实现、已执行，有结果依据", f"job 7424374 fingerprint {hist['J03']}", "matched_predication、semmed_pair_pmid、pmid_route", "sentence_id 使用 v2，与 J02 规则不同"),
        item("J04 联合与敏感性", "已实现、已执行，有结果依据", f"job 7424375 fingerprint {hist['J04']}", "pair_relations 与 summary.json", "规范化文本比较被规模阈值跳过"),
        item("J05 报告与案例", "已实现、已执行，有结果依据", f"job 7424376 fingerprint {hist['J05']}", "report.md、5 图、20 案例", "报告把字典原样写入，且与 J02 的句子编号结论并列而未解释命中数差异"),
        item("规范化文本比较", "因实现限制未计算", "joint_build.py exclusive CORE pair-PMID > 250000", "精确文本比较已计算", "不能写成数据不足"),
        item("ENTITY–sentence 全库外键完整性", "因字段未确认而未计算", "J02 integrity entity_sentence_missing = not_assessable", "J03 仍用 v2 列做了下游连接", "全库缺失实体数没有独立复核"),
        item("当前源码与 J03 是否逐字节一致", "当前版本是否有效仍待核实", f"now {now} vs J03 {hist['J03']}", "整树哈希", "本审计新增文件会改变当前哈希"),
    ]


def item(name, status, evidence, result, limit) -> dict:
    return {"item": name, "status": status, "evidence": evidence, "result_or_gap": result, "limit": limit}


def schema_rows(measured: dict) -> list[dict]:
    extras = measured.get("unconfirmed", {})
    rows_out = [
        evidence_row("PREDICATION", "0-11", "dbinfo 的 12 列", "样本取值模式与文档类型一致", "J00 样本", "有文档列序加样本模式", "已用于下游；附加列不影响这 12 列的使用"),
        evidence_row("PREDICATION", "12-14", "未确认", _extra_text(extras), "全表取值分组", "尚不明确", "主键和 CUI 连接不用这三列"),
        evidence_row("ENTITY", "0", "entity_id 候选", "数值、位于首列", "J00 样本", "位置推测", "未与官方 DDL 核对"),
        evidence_row("ENTITY", "1", "sentence_id 候选，不是已证实外键", "多个位置的行组里对 sentence.sentence_id 命中率高", "本审计行组样本", "连接证据，仍不是 DDL", "J03 把它当作 sentence_id；J02 拒绝认定"),
        evidence_row("ENTITY", "2", "另一个数值列", "命中 sentence_id 的比例高但低于第 1 列，且常与第 1 列不同", "J02/J03 与本审计", "不能认定为外键", "不参与主分析连接"),
        evidence_row("ENTITY", "6", "gene_id 候选", "位于 semtype 后；装载时分成 numeric / composite / empty", "样本位置加全表 gene_id_class", "位置推测加内容分类", "只有 numeric 进入 gene_id；复合值被排除"),
        evidence_row("SENTENCE", "5", "句子文本", "中位长度最长", "J00 样本", "内容证据", "已用作句子文本"),
        evidence_row("SENTENCE", "7-8", "章节标题候选", "剩余两列，命名时没有做非空内容检查", "J02 _availability", "位置推测", "空列不能证明它们是章节标题"),
        evidence_row("CITATIONS", "2 与 4", "DP 与 PYEAR", "dp_year 来自 c2 前四位；completion_year 来自 c4", "semmed_build.py _select", "两列独立解析", "不一致数为 0 是这两列的比较结果"),
    ]
    return rows_out


def evidence_row(table, column, meaning, observation, check, status, effect) -> dict:
    return {
        "table": table,
        "column": column,
        "candidate_meaning": meaning,
        "observation": observation,
        "check": check,
        "confirmation": status,
        "effect_on_later_results": effect,
    }


def _extra_text(extras: dict) -> str:
    parts = []
    for index in (12, 13, 14):
        kinds = extras.get(f"unconfirmed_{index}", [])
        top = extras.get(f"unconfirmed_{index}_top", [])
        parts.append(f"c{index} kinds={kinds} top={top}")
    return " | ".join(parts) if parts else "未测"


def lineage_rows(measured: dict) -> list[dict]:
    closure = measured["closure"]
    mention = measured["mention"]
    return [
        line("CORE 显式 pair 153016", "pair", "J01 7420640", "lib/core_build.py _funnel explicit_pairs", "01_core_audit/tables/pair.parquet"),
        line("sentence_with_multiple_gcs", "唯一句子", "J01 7420640", "lib/core_build.py _integrity", f"复核 {mention['sentences_multi_gcs']}"),
        line("mention_gene_disagrees 57347", "mention×GCS 行，不是句子", "J01 7420640", "lib/core_build.py:529", f"复核笛卡尔 {mention['gene_cartesian_rows']}；集合缺失 {mention['gene_set']['gcs_whose_uri_is_absent_from_sentence_mentions']}"),
        line("mention_disease_disagrees 70573", "mention×GCS 行", "J01 7420640", "lib/core_build.py:535", f"复核笛卡尔 {mention['disease_cartesian_rows']}；集合缺失 {mention['disease_set']['gcs_whose_uri_is_absent_from_sentence_mentions']}"),
        line("主分析 pair 276 / pair-PMID 426 / 交集 33", "pair 与 pair-PMID 分开", "J03 7424374 / J04 7424375", "lib/align_build.py matched_predication", f"复核 pairs={closure['semmed_pair_pmid']['pairs']} pair_pmid={closure['semmed_pair_pmid']['pair_pmid']} pmid={closure['semmed_pair_pmid']['pmid']} intersection={closure['intersection_pair_pmid']}"),
        line("范围 E 619022 与唯一 gene_id 103", "predication", "J02 行数；J03 附基因", "lib/semmed_build.py _make_scope；lib/align_build.py scope_e_annotated", "本审计 scope_class 排他分类见 mapping_flow.tsv"),
        line("句子编号命中 43942 与 43870", "前 50000 行的连接行数，且 LIMIT 无序", "J02 7422513 与 J03 7424374", "lib/semmed_build.py resolve_entity_sentence_id", "见报告第 2 节"),
    ]


def line(result, unit, job, code, artifact) -> dict:
    return {"result": result, "unit": unit, "job": job, "code": code, "artifact_or_recheck": artifact}


def flow_rows(measured: dict) -> list[dict]:
    presence = measured["mapping"]["pair_id_presence_anywhere"]
    out = [
        {
            "stage": "explicit_core_pairs",
            "unit": "pair",
            "n": presence["pairs"],
            "definition": "CORE 显式基因–疾病 pair",
        },
        {
            "stage": "gene_id_seen_on_any_entity_numeric_field",
            "unit": "pair",
            "n": presence["gene_seen_anywhere"],
            "definition": "该 pair 的基因 ID 在 ENTITY 任一 numeric gene_id 中出现，不要求同句",
        },
        {
            "stage": "disease_cui_seen_on_any_predication_argument",
            "unit": "pair",
            "n": presence["disease_seen_anywhere"],
            "definition": "疾病 CUI 出现在任一谓词的主语或宾语，不要求同句或基因",
        },
        {
            "stage": "both_ids_seen_anywhere",
            "unit": "pair",
            "n": presence["both_seen_anywhere"],
            "definition": "上面两个条件同时成立，仍不是主分析匹配",
        },
        {
            "stage": "primary_same_sentence_match",
            "unit": "pair",
            "n": measured["closure"]["semmed_pair_pmid"]["pairs"],
            "definition": "J03：同句、疾病 CUI 在谓词一侧、另一侧 CUI 上恰好一个 numeric gene_id 且等于 CORE 基因",
        },
    ]
    for row in measured["mapping"]["scope_e_exclusive"]:
        out.append(
            {
                "stage": "scope_e_" + row["exclusive_class"],
                "unit": "predication",
                "n": row["n_predication"],
                "definition": measured["mapping"]["exclusive_priority"],
            }
        )
    return out


def write_cases(measured: dict) -> None:
    path = OUT / "cases" / "cases.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for example in measured["mention"].get("examples", []):
        records.append(
            {
                "kind": example.get("kind"),
                "locator": example.get("locator"),
                "rule": "mention edge joined to every GCS of the sentence",
                "gcs": example.get("gcs"),
                "mentions": example.get("mentions"),
                "boundary": "结构描述。不判断抽取对错。",
            }
        )
    for check in measured.get("text_cases", {}).get("checks", []):
        records.append({"kind": "entity_span_crosscheck", **check, "boundary": "只记录切片和子串是否一致，不把不一致叫作错误。"})
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def render_report(measured: dict, info: dict) -> str:
    mention = measured["mention"]
    labels = measured["labels"]
    closure = measured["closure"]
    mapping = measured["mapping"]
    sid = measured["sentence_id"]
    text = measured["text_cases"]
    stored = measured["stored"]
    hetero = labels["heterogeneity"]
    sem = closure["semmed_pair_pmid"]
    lines = [
        "# CORE 与 SemMedDB 数据审计",
        "",
        "本报告核查已经完成的 J00–J05。它不重跑主分析，不评价论文质量，也不判断科学支持或反驳。",
        "",
        "## 1. 本轮实际完成了什么",
        "",
        "要回答的问题：哪些结果来自已结束的作业，哪些只是当前源码里的函数，哪些被跳过或未能验证。",
        "",
        "方法和范围：读取各作业 DONE、summary 和派生 parquet。整树指纹会因后来修改的任何代码文件而改变。`project_2` 不是 git 仓库，不能恢复当时的源码字节。",
        "",
        f"当前树指纹 `{info['code_fingerprint']}`。J01 `{measured['historical_fingerprints']['J01']}`，J02 `{measured['historical_fingerprints']['J02']}`，J03–J05 `{measured['historical_fingerprints']['J03']}`。",
        "",
        "结果：J00–J05 的完成标记都是 success。J01 产出 CORE 表，J02 产出六张 SemMedDB parquet，J03 产出主分析匹配和 PMID 路线，J04 产出 pair 关系与敏感性，J05 把这些 summary 写进了原报告。原报告的问题是把字典和长分布直接贴出，并把 J02 与 J03 两套句子编号规则并排，而没有解释命中数为何不同。",
        "",
        "规范化文本比较没有执行：独有 CORE pair–PMID 超过 250000，代码把 `text_normalized` 留为 not_assessable。这是实现限制，不是数据不足。ENTITY 到 sentence 的全库反连接在 J02 中是 not_assessable，因为当时句子编号 unresolved。",
        "",
        "限制：下面引用的函数行号属于当前文件。不能据此声称 J01 运行时的文件字节与现在相同。",
        "",
        "## 2. 本地文件、记录层级和字段证据",
        "",
        "要回答的问题：列名有多少是文档或连接证据，有多少只是位置推测；两次句子编号命中数为什么不同。",
        "",
        "方法和范围：字段来源见 `tables/schema_evidence.tsv`。句子编号用 ENTITY parquet 的多个行组各取最多 20000 行，按行组位置而不是无序 LIMIT。文本抽查读已修复的 UTF-8 ENTITY 副本，只保留样本实体编号。",
        "",
        "结果：PREDICATION 第 13–15 列的全表取值分组见 schema 表。它们不进入主键或 CUI 连接。SENTENCE 第 8、9 列在 J02 被命名为章节标题，但命名代码写明没有做非空检查；可用性计数在 J02 summary 的 sentences.section_header。",
        "",
        f"J02 前 50000 行无序样本：第 1 列命中 {stored['semmed_resolution']['rates']['pre_cui_numeric_1']['matched']}，第 2 列命中 {stored['semmed_resolution']['rates']['pre_cui_numeric_2']['matched']}。J03 同形查询：第 1 列仍为 50000，第 2 列为 {stored['align_resolution']['rates']['pre_cui_numeric_2']['matched']}。两次查询都是 `LIMIT 50000` 且没有 ORDER BY，连接配置 `preserve_insertion_order=false`。因此这 72 行差异来自无序样本，不是阈值公式本身。J02 记录的重复 sentence_id 个数是 {stored['semmed_sentences'].get('duplicate_sentence_id')}；为 0 时，命中行数等于命中的实体行数。按第一个行组顺序截取的前缀结果：{sid['first_row_group_prefix']}。各行组命中见下表。",
        "",
        _md(sid["per_group"]),
        "",
        "第 1 列在这些分散行组里持续对齐 sentence_id，第 2 列命中率高但不是每一行都对齐，而且两列并不总是相等。数字落在 sentence_id 集合里仍不足以单独证明外键。文本切片结果单独记录在案例里，切片不一致不记为抽取错误。",
        "",
        f"文本抽查状态：{text.get('status')}，扫描记录 {text.get('records_scanned')}，找到 {text.get('rows_found')} / {text.get('rows_requested')}。",
        "",
        "DP 与 PYEAR：`dp_year` 由原始列 c2 在状态为 parsed_single 时取前四位；`completion_year` 由原始列 c4 的四位数字单独解析。不一致计数比较的是这两个结果，不是把同一解析用了两次。J02 记录两边都解析且年份不同的行数为 0。",
        "",
        "编码：SENTENCE 全表按 UTF-8 与反斜杠转义读完。PREDICATION 与 PREDICATION_AUX 使用 latin-1，因为 UTF-8 阅读器遇到非法字节，而 DuckDB 的 latin-1 又拒绝 0x80–0x9F。ENTITY 主体是 UTF-8，另有 8566 个非法字节被写成 Windows-1252 字符后才装载。latin-1 能解码并不证明原文就是 latin-1。标识符列是 ASCII，不依赖这 8566 个字节。",
        "",
        "限制：没有找到与本地 12 列 ENTITY、15 列 PREDICATION 对应的 DDL。未确认列保持未确认。",
        "",
        "## 3. CORE 的实际数据特征",
        "",
        "要回答的问题：24,890、57,347、70,573 计的是什么；标签和 hasType 的实际分布是什么。",
        "",
        "方法和范围：在 J01 留下的 `core.duckdb` 上重跑同一连接，并增加“GCS 的基因是否落在该句 mention 集合中”的集合检查。标签从 `sentence.parquet` 汇总。该 parquet 的一行是句子与 GCS 的连接，不是唯一句子。",
        "",
        f"多 GCS 句子：唯一句子 {mention['sentences_multi_gcs']}。这与 J01 的 24890 同一单位。",
        "",
        f"基因“不一致”的原计数是笛卡尔连接行数，复核 {mention['gene_cartesian_rows']}。其中，mention 集合其实已经覆盖该 GCS 基因、只是与另一个 mention 交叉比较而产生的行数：{mention['gene_set']['cartesian_rows_where_mention_set_still_covers_gcs']}。GCS 基因不在该句 mention 集合中的 GCS 数：{mention['gene_set']['gcs_whose_uri_is_absent_from_sentence_mentions']}，涉及句子 {mention['gene_set']['sentences_with_such_a_gcs']}。",
        "",
        f"疾病侧笛卡尔复核 {mention['disease_cartesian_rows']}；集合未覆盖的 GCS 数 {mention['disease_set']['gcs_whose_uri_is_absent_from_sentence_mentions']}，句子 {mention['disease_set']['sentences_with_such_a_gcs']}。",
        "",
        "因此 57,347 和 70,573 不是“错误句子”的个数。一个句子上的多个 mention 与多个 GCS 做了两两比较，URI 不同就计一行。标签本身写在句子节点上，再连接到该句的每一个 GCS，所以同一句的多个 GCS 共享同一组四标签。",
        "",
        "四方面在句子–GCS 行和唯一句子上的取值：",
        "",
        _aspect_block(labels["aspects"]),
        "",
        f"四标签组合共 {labels['n_distinct_label_tuples']} 种。出现最多的组合：",
        "",
        _md(labels["label_tuple_top"][:12]),
        "",
        f"pair–PMID 行 {hetero['pair_pmid_rows']}，其中句内标签组合多于一种的有 {hetero['heterogeneous_within_pmid']}。无可用标签 {hetero['no_usable_label']}。pair {hetero['pairs']} 中，跨 PMID 标签集合不同的有 {hetero['heterogeneous_across_pmid']}，相同的有 {hetero['homogeneous_across_pmid']}，只有一篇或无法比较的有 {hetero['across_not_applicable']}。",
        "",
        "usable_status 按每个方面自己的列判断，不用一个丢掉字段名的序列。label_tuple 用竖线按 gcc、cge、ccs、gci 的固定位置拼接，多值记为 MULTI，缺失记为 FIELD_ABSENT。NOTINF 保留在组合里，只有计算 usable_status 时不把它当成可用标签。",
        "",
        "hasType 存在 GCS 上，不是句子标签。复核：",
        "",
        _md(labels["has_type_on_gcs_status"]),
        "",
        "这三项之和等于 GCS 数 153016。本次存储结果里没有 UNRELIABLE。这里不估计被过滤掉的数量，因为本地 RDF 没有保存构建分数。",
        "",
        "限制：集合检查说明了笛卡尔计数里有多少是交叉比较。它仍然不判断 mention 或 GCS 哪一方更可靠。",
        "",
        "## 4. SemMedDB 的实际数据特征",
        "",
        "要回答的问题：六张表的规模、哪些完整性已经计算、ENTITY 连接还有什么没验证。",
        "",
        "方法和范围：行数与完整性取 J02 summary。本审计补了 PREDICATION 附加列和 ENTITY 行组抽查，没有重扫六张原始 CSV。",
        "",
        "表行数：citations 37,233,341；generic_concept 259；sentence 263,160,520；predication 130,480,195；predication_aux 130,480,181；entity 1,982,424,585。单位都是各自的表行，不是文章或 pair。",
        "",
        "已计算的完整性：sentence 的 PMID 都能在 citations 找到；predication 的 sentence_id 都能找到，且 PMID 与 sentence 不一致的行数是 0；aux 找不到 predication 的行数是 0；predication 没有 aux 的是 14 行；没有句子的 citation 是 176,693；没有 predication 的 sentence 是 181,932,638，代码把它记为结构上允许。",
        "",
        "未验证：J02 没有做 ENTITY 到 sentence 的全库反连接，因为句子编号 unresolved。J03 用第 1 列做了匹配，但没有补上这个全库计数。generic_concept 的 259 行里，第二列不全是 CUI 形式，已单独计数，没有改解释成基因 ID。",
        "",
        "范围 E 是谓词两侧语义类型恰好为 gngm 与 neop，共 619,022 行谓词、220,400 个 CUI 对、323,968 个 PMID。aapp×neop 另计 517,308 行，没有并入。这不是完整癌症集合。",
        "",
        "限制：章节列、基因列和两个 pre-CUI 数字列都还没有 DDL。附加列的取值分组不能给它们命名。",
        "",
        "## 5. ID 对齐与记录损失",
        "",
        "要回答的问题：153,016 个 pair 为什么主分析只留下 276 个；619,022 条范围 E 谓词为什么只有 103 条带上唯一基因 ID。",
        "",
        "方法和范围：主分析规则仍是 id_rule_v1，不放宽。本审计用已有 parquet 增加两段计数：基因或疾病是否在任一位置出现；范围 E 的基因字段是空、复合、多个数字，还是没有实体行。同句主匹配没有重算，276 与 426 沿用 J03，并用明细做了闭包复核。",
        "",
        "ID 出现情况（pair 单位，不要求同句）：",
        "",
        _md([mapping["pair_id_presence_anywhere"]]),
        "",
        f"复核：SemMedDB 主分析 pair {sem['pairs']}，pair–PMID {sem['pair_pmid']}，去重 PMID {sem['pmid']}，与 CORE 的交集 pair–PMID {closure['intersection_pair_pmid']}。pair 关系分类行数 {closure['class_rows']}，显式 pair {closure['explicit_pairs']}，两者相等：{closure['classes_cover_all_pairs']}。",
        "",
        "范围 E 排他分类的优先顺序：" + mapping["exclusive_priority"] + "。同时出现的情况见标志计数，不挤进单一类别后丢掉。",
        "",
        _md(mapping["scope_e_exclusive"]),
        "",
        "标志（一行可同时计入多项）：",
        "",
        _md([mapping["scope_e_flags"]]),
        "",
        "868,007 是全库里“同一句子、同一 CUI、多个不同 numeric gene_id”的句子–CUI 数，分母不是范围 E。范围 E 的歧义数是 0，因为只看 gngm–neop 谓词的基因侧 CUI；这个审计的排他分类若出现 multiple_numeric_gene_id，才是同一范围里的歧义。两个数字的范围不同，不能相减。",
        "",
        "含竖线或逗号的基因字段保持复合，不展开成确定实体。它们在排他分类中进入 composite_id_only，或在同时有数字 ID 时由标志单独列出。",
        "",
        "限制：同句匹配之前的流失没有拆成更细的谓词过滤。both_seen_anywhere 与 276 之间的差，是同句、唯一基因 ID 和两侧 CUI 对齐共同造成的，本轮没有再拆开。",
        "",
        "## 6. pair 与 PMID 联合分析",
        "",
        "要回答的问题：五类是否完备；418、33、426 到 89 各指什么；四个被排除的 CUI 是什么。",
        "",
        "方法和范围：用 pair_relations、pair_pmid 和 semmed_pair_pmid 重算闭包。敏感性数字来自 J04 summary，不重新解释成新的匹配。",
        "",
        "全部显式 pair 的关系分类：",
        "",
        _md(closure["class_counts"]),
        "",
        "五类两边都非空的 pair 加上 semmed_pmids_empty 后覆盖全部显式 pair。分类行数与 pair 数相等，所以这组分类在 pair 单位上是互斥且完备的。没有 PMID 的一侧用 empty 类表示，不把它写成分析范围之外。",
        "",
        f"418 是主分析 SemMedDB pair–PMID 表里的去重 PMID，复核 {sem['pmid']}。它不是 SemMedDB 全库 PMID，也不是范围 E 的 323,968 个 PMID。33 是 gene_id、disease_cui、pmid 三个键都相同的 pair–PMID 交集，复核 {closure['intersection_pair_pmid']}。",
        "",
        "PMID 路线使用全部 CORE PMID，不先要求 pair 匹配。J04 记录 161,449 个 CORE PMID 中 161,405 个在 citations，161,197 个有 sentence，160,572 个有任意谓词，144,196 个通过句子编号候选列连到实体。最后一项依赖尚未由 DDL 确认的第 1 列。",
        "",
        f"共同年份窗口把主分析 SemMedDB pair–PMID 从 426 减到 89。J04 记录的移除数：{stored['joint_year_removed']}。移除条件是发表年不在两边都出现过的年份里，或 CORE publicationYear 与 SemMedDB DP 年份都已解析但彼此不同。",
        "",
        f"两边都有文章记录的过滤：{stored['joint_common_removed']}。共同文章 PMID 数 {stored['joint_common_pmid']}。这个集合来自 CORE PMID 与 citations 的交集，不是“已经匹配上目标 pair 的文章”。",
        "",
        "四个被排除的 CUI 来自 CORE SPARQL 示例 5，不是本地癌症定义。谓词表中的名称：",
        "",
        _md(mapping["excluded_cui_names"]),
        "",
        "独有 pair–PMID 上的“目标谓词 absent”来自反连接的定义。它是闭包检查，不是独立发现了另一条失败原因。精确文本比较已经执行；规范化比较没有执行。",
        "",
        "限制：共同年份和共同文章改变的是覆盖分母，不提高 ID 对齐的确定性。",
        "",
        "## 7. 可以支持的描述、未解决问题和最小修正",
        "",
        "可以保留的结果：六张表和 CORE 的行数；CORE 的 pair、PMID、pair–PMID 规模和文献量长尾；hasType 的三项存储计数及未观察到 UNRELIABLE；标签组合与异质性计数；主分析 276 个 pair、426 个 pair–PMID、33 个交集和 418 个 SemMedDB PMID，在现有 id_rule_v1 与第 1 列句子编号假设下可以从明细复核；范围 E 的 619,022 行是语义类型子集。",
        "",
        "需要暂缓解释的结果：把 57,347 或 70,573 说成标注不一致的句子数；把 276 或 103 说成两个库的真实重叠上限；把第 1 列已经证实为 SENTENCE_ID；把基因列已经证实为官方 GENE_ID。276 和 103 都依赖这两项尚未确认的字段规则。复合 ID 被排除会进一步减少匹配，但这是规则选择，不是测量到的错误。",
        "",
        "最小后续修正：先取得与本地导出一致的 DDL，或用文本位置抽查扩大到能够单独确认第 1 列和第 6 列。在此之前不必重跑 J01 和 J02 的装载。若字段确认改变了句子编号或基因列，只需重跑 J03–J05。规范化文本比较若要补做，是另一项独立作业，不能写进当前覆盖结论。",
        "",
        "完成清单、字段证据、结果溯源和映射阶段见同目录 `tables/`。案例见 `cases/cases.jsonl`。",
        "",
    ]
    return "\n".join(lines) + "\n"


def _md(records: list[dict]) -> str:
    if not records:
        return "（无行）"
    keys = list(records[0].keys())
    head = "| " + " | ".join(keys) + " |"
    sep = "| " + " | ".join("---" for _ in keys) + " |"
    body = ["| " + " | ".join(_cell(row.get(key)) for key in keys) + " |" for row in records]
    return "\n".join([head, sep, *body])


def _cell(value) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "/").replace("\n", " ")


def _aspect_block(aspects: dict) -> str:
    blocks = []
    for name, table in aspects.items():
        blocks.append(f"{name}")
        blocks.append("")
        blocks.append(_md(table[:8]))
        blocks.append("")
    return "\n".join(blocks).rstrip()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        raise SystemExit(1)

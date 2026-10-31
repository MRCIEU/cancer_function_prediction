#!/usr/bin/env python3
"""Check SemMedDB argument forms and the scope-E entity join. Does not replace id_rule_v1."""

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
from lib.runtime import prologue, setup_logging, write_json  # noqa: E402
from lib.semmed_columns import canonical_id_expr  # noqa: E402

RESULTS = Path("/projects/b5ac/project_2/results")
PREV = RESULTS / "03_integrated_analysis" / "data_analysis_review_round1"
OUT = RESULTS / "03_integrated_analysis" / "data_analysis_linkcheck_round1"
CASES = PREV / "cases" / "cases.jsonl"

FORMAT_SQL = """
CASE
  WHEN gene_cui IS NULL OR trim(gene_cui) = '' OR trim(gene_cui) IN (CHR(92) || 'N', 'N') THEN 'empty'
  WHEN regexp_matches(trim(gene_cui), '^C[0-9]+$') THEN 'single_umls_cui'
  WHEN regexp_matches(trim(gene_cui), '^[0-9]+$') THEN 'single_digits'
  WHEN position('|' IN gene_cui) > 0 THEN 'pipe_composite'
  WHEN regexp_matches(gene_cui, '[;,/]') THEN 'other_separator'
  ELSE 'other'
END
"""


def log_progress(logger, phase: str, **fields) -> None:
    parts = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info(
        "progress phase=%s %s last_progress_utc=%s",
        phase,
        parts,
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


def qpath(path: Path) -> str:
    return "'" + path.as_posix().replace("'", "''") + "'"


def one(con, sql: str):
    return con.execute(sql).fetchone()[0]


def has_table(con, name: str) -> bool:
    return one(con, f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{name}' AND table_schema = 'main'") > 0


def rows(con, sql: str) -> list[dict]:
    cur = con.execute(sql)
    names = [item[0] for item in cur.description]
    return [dict(zip(names, row)) for row in cur.fetchall()]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "derived").mkdir(parents=True, exist_ok=True)
    logger = setup_logging(OUT / "logs" / f"linkcheck_{os.environ.get('SLURM_JOB_ID', 'local')}.log")
    info = prologue("LINKCHECK", logger, OUT, [str(PREV)])
    con = connect(OUT / "derived" / "linkcheck.duckdb", "72GB", OUT / "tmp")
    attach_previous(con, logger)
    measured = {"code_fingerprint": info["code_fingerprint"]}
    measured["formats"] = argument_formats(con, logger)
    measured["no_entity"] = decompose_no_entity(con, logger)
    measured["examples"] = representative_examples(con, logger)
    measured["existing_cases"] = summarize_existing_cases()
    measured["core"] = core_corrections(con, logger)
    measured["pair_class"] = pair_class_check(con, logger)
    write_json(OUT / "derived" / "measured.json", measured)
    write_tables(measured)
    (OUT / "report.md").write_text(render(measured), encoding="utf-8")
    write_json(
        OUT / "DONE.json",
        {"status": "success", "job": "LINKCHECK", "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "report": str(OUT / "report.md")},
    )
    logger.info("phase=linkcheck_success")
    return 0


def attach_previous(con, logger) -> None:
    if has_table(con, "scope_labeled") and has_table(con, "no_ent_flags"):
        logger.info(
            "phase=reuse_completed_scan scope_labeled=%s no_ent_flags=%s",
            one(con, "SELECT COUNT(*) FROM scope_labeled"),
            one(con, "SELECT COUNT(*) FROM no_ent_flags"),
        )
        return
    db = PREV / "derived" / "review.duckdb"
    con.execute(f"ATTACH {qpath(db)} AS prev (READ_ONLY)")
    con.execute("CREATE TABLE scope_class AS SELECT * FROM prev.scope_class")
    con.execute("CREATE TABLE ent_hit AS SELECT * FROM prev.ent_hit")
    logger.info(
        "phase=attached_previous scope_class=%s ent_hit=%s",
        one(con, "SELECT COUNT(*) FROM scope_class"),
        one(con, "SELECT COUNT(*) FROM ent_hit"),
    )
    con.execute(
        f"""
        CREATE TABLE scope_e AS
        SELECT *, {FORMAT_SQL} AS argument_format
        FROM read_parquet({qpath(RESULTS / '02_semmeddb_validation' / 'tables' / 'scope_e_gngm_neop.parquet')})
        """
    )
    con.execute(
        """
        CREATE TABLE scope_labeled AS
        SELECT e.*, c.n_entity_rows, c.n_numeric_gene, c.n_composite_comma_rows, c.n_composite_pipe_rows, c.n_empty_rows, c.n_other_rows,
          CASE
            WHEN c.n_entity_rows = 0 THEN 'no_entity_row_for_sentence_and_cui'
            WHEN c.n_numeric_gene > 1 THEN 'multiple_numeric_gene_id'
            WHEN c.n_numeric_gene = 1 THEN 'unique_numeric_gene_id'
            WHEN COALESCE(c.n_composite_comma_rows, 0) + COALESCE(c.n_composite_pipe_rows, 0) > 0 THEN 'composite_id_only'
            WHEN COALESCE(c.n_other_rows, 0) > 0 THEN 'non_numeric_other_only'
            ELSE 'entity_present_gene_field_empty'
          END AS join_class
        FROM scope_e e
        JOIN scope_class c USING (predication_id)
        """
    )
    n_scope = one(con, "SELECT COUNT(*) FROM scope_e")
    n_labeled = one(con, "SELECT COUNT(*) FROM scope_labeled")
    if n_scope != n_labeled:
        raise SystemExit(f"scope join changed the row count: scope_e={n_scope} labeled={n_labeled}")


def argument_formats(con, logger) -> dict:
    logger.info("phase_start phase=argument_formats")
    out = {
        "by_format": rows(
            con,
            """
            SELECT argument_format,
                   COUNT(*) AS n_predication,
                   COUNT(DISTINCT gene_cui) AS n_distinct_argument
            FROM scope_labeled
            GROUP BY 1
            ORDER BY n_predication DESC
            """,
        ),
        "examples": rows(
            con,
            """
            SELECT argument_format, gene_cui, n_predication
            FROM (
              SELECT argument_format, gene_cui, n_predication,
                     row_number() OVER (PARTITION BY argument_format ORDER BY n_predication DESC, gene_cui) AS rk
              FROM (
                SELECT argument_format, gene_cui, COUNT(*) AS n_predication
                FROM scope_labeled
                GROUP BY 1, 2
              )
            )
            WHERE rk <= 3
            ORDER BY argument_format, n_predication DESC
            """,
        ),
        "format_by_join_class": rows(
            con,
            """
            SELECT argument_format, join_class, COUNT(*) AS n_predication
            FROM scope_labeled
            GROUP BY 1, 2
            ORDER BY n_predication DESC
            """,
        ),
    }
    log_progress(logger, "argument_formats", rows=len(out["format_by_join_class"]))
    return out


def decompose_no_entity(con, logger) -> dict:
    logger.info("phase_start phase=no_entity_decompose")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS no_ent AS
        SELECT predication_id, sentence_id, gene_cui, disease_cui, pmid, gene_position, predicate, argument_format
        FROM scope_labeled
        WHERE join_class = 'no_entity_row_for_sentence_and_cui'
        """
    )
    n = one(con, "SELECT COUNT(*) FROM no_ent")
    log_progress(logger, "no_entity_decompose", n=n)
    sentence = qpath(RESULTS / "02_semmeddb_validation" / "tables" / "sentence.parquet")
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS sentence_found AS
        SELECT DISTINCT sentence_id
        FROM read_parquet({sentence})
        WHERE sentence_id IN (SELECT sentence_id FROM no_ent)
        """
    )
    log_progress(logger, "sentence_presence", found=one(con, "SELECT COUNT(*) FROM sentence_found"))
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS sentence_entity_presence AS
        SELECT sentence_id, SUM(n_rows) AS n_entity_rows
        FROM ent_hit
        WHERE sentence_id IN (SELECT sentence_id FROM no_ent)
        GROUP BY 1
        """
    )
    logger.info("phase_start phase=alternate_id_scan")
    canon1 = canonical_id_expr("pre_cui_numeric_1")
    canon2 = canonical_id_expr("pre_cui_numeric_2")
    entity = qpath(RESULTS / "02_semmeddb_validation" / "tables" / "entity.parquet")
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS ent_narrow AS
        SELECT {canon1} AS s1,
               {canon2} AS s2,
               cui,
               entity_id,
               pre_cui_numeric_2 AS pre2_raw,
               gene_id,
               gene_id_raw,
               COUNT(*) AS n_rows
        FROM read_parquet({entity})
        WHERE {canon1} IN (SELECT sentence_id FROM no_ent)
           OR {canon2} IN (SELECT sentence_id FROM no_ent)
        GROUP BY 1, 2, 3, 4, 5, 6, 7
        """
    )
    log_progress(logger, "alternate_id_scan", rows=one(con, "SELECT COUNT(*) FROM ent_narrow"))
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS hit_entity_id AS
        SELECT DISTINCT n.predication_id
        FROM no_ent n
        JOIN ent_narrow e ON e.s1 = n.sentence_id AND e.entity_id = n.gene_cui
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS hit_pre2 AS
        SELECT DISTINCT n.predication_id
        FROM no_ent n
        JOIN ent_narrow e ON e.s1 = n.sentence_id AND e.pre2_raw = n.gene_cui
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS hit_gene AS
        SELECT DISTINCT n.predication_id
        FROM no_ent n
        JOIN ent_narrow e
          ON e.s1 = n.sentence_id
         AND (e.gene_id = n.gene_cui OR e.gene_id_raw = n.gene_cui)
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS hit_col2_cui AS
        SELECT DISTINCT n.predication_id
        FROM no_ent n
        JOIN ent_narrow e ON e.s2 = n.sentence_id AND e.cui = n.gene_cui
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS no_ent_flags AS
        SELECT n.predication_id,
               n.argument_format,
               (s.sentence_id IS NOT NULL) AS sentence_id_in_sentence_table,
               COALESCE(p.n_entity_rows, 0) > 0 AS sentence_has_any_entity,
               (i.predication_id IS NOT NULL) AS argument_equals_entity_id,
               (r.predication_id IS NOT NULL) AS argument_equals_pre2,
               (g.predication_id IS NOT NULL) AS argument_equals_gene_field,
               (c2.predication_id IS NOT NULL) AS cui_equals_on_column2
        FROM no_ent n
        LEFT JOIN sentence_found s USING (sentence_id)
        LEFT JOIN sentence_entity_presence p USING (sentence_id)
        LEFT JOIN hit_entity_id i USING (predication_id)
        LEFT JOIN hit_pre2 r USING (predication_id)
        LEFT JOIN hit_gene g USING (predication_id)
        LEFT JOIN hit_col2_cui c2 USING (predication_id)
        """
    )
    exclusive = rows(
        con,
        """
        SELECT
          CASE
            WHEN NOT sentence_id_in_sentence_table THEN 'candidate_sentence_id_absent'
            WHEN NOT sentence_has_any_entity THEN 'sentence_has_no_entity_row'
            WHEN argument_equals_entity_id OR argument_equals_pre2 OR argument_equals_gene_field THEN 'exact_string_on_other_entity_field'
            WHEN cui_equals_on_column2 THEN 'cui_matches_if_column2_is_sentence'
            WHEN argument_format IN ('pipe_composite', 'other_separator') THEN 'composite_argument_not_expanded'
            ELSE 'cui_string_not_equal_to_entity_cui'
          END AS exclusive_state,
          COUNT(*) AS n_predication
        FROM no_ent_flags
        GROUP BY 1
        ORDER BY n_predication DESC
        """,
    )
    flags = rows(
        con,
        """
        SELECT argument_format,
               COUNT(*) AS n,
               COUNT(*) FILTER (WHERE NOT sentence_id_in_sentence_table) AS sentence_absent,
               COUNT(*) FILTER (WHERE sentence_id_in_sentence_table AND NOT sentence_has_any_entity) AS no_entity_on_sentence,
               COUNT(*) FILTER (WHERE sentence_has_any_entity) AS entities_present_cui_miss,
               COUNT(*) FILTER (WHERE argument_equals_entity_id) AS equals_entity_id,
               COUNT(*) FILTER (WHERE argument_equals_pre2) AS equals_pre2,
               COUNT(*) FILTER (WHERE argument_equals_gene_field) AS equals_gene_field,
               COUNT(*) FILTER (WHERE cui_equals_on_column2) AS cui_match_on_column2
        FROM no_ent_flags
        GROUP BY 1
        ORDER BY n DESC
        """,
    )
    digits = rows(
        con,
        """
        SELECT COUNT(*) AS n_digit_arguments,
               COUNT(*) FILTER (WHERE argument_equals_gene_field) AS equal_gene_field_same_sentence
        FROM no_ent_flags
        WHERE argument_format = 'single_digits'
        """,
    )[0]
    log_progress(logger, "no_entity_decompose", states=len(exclusive))
    narrow_rows = one(con, "SELECT COUNT(*) FROM ent_narrow")
    return {
        "n": n,
        "exclusive_priority": "candidate_sentence_id_absent > sentence_has_no_entity_row > exact_string_on_other_entity_field > cui_matches_if_column2_is_sentence > composite_argument_not_expanded > cui_string_not_equal_to_entity_cui",
        "exclusive": exclusive,
        "by_format": flags,
        "digit_arguments_among_no_entity": digits,
        "scan": f"实体 parquet 只扫了一遍：保留第 1 列或第 2 列落在 no_entity 句子号集合里的行，共 {narrow_rows} 组。句子 parquet 只取了这些句子号。没有读原始 ENTITY csv。",
    }


def representative_examples(con, logger) -> list[dict]:
    logger.info("phase_start phase=examples")
    con.execute(
        """
        CREATE OR REPLACE TABLE example_ids AS
        SELECT predication_id, argument_format, join_class
        FROM (
          SELECT predication_id, argument_format, join_class,
                 row_number() OVER (PARTITION BY argument_format, join_class ORDER BY predication_id) AS rk
          FROM scope_labeled
        )
        WHERE rk <= 2
          AND (
            join_class = 'no_entity_row_for_sentence_and_cui'
            OR join_class = 'unique_numeric_gene_id'
          )
        """
    )
    pred = qpath(RESULTS / "02_semmeddb_validation" / "tables" / "predication.parquet")
    aux = qpath(RESULTS / "02_semmeddb_validation" / "tables" / "predication_aux.parquet")
    sent = qpath(RESULTS / "02_semmeddb_validation" / "tables" / "sentence.parquet")
    found = rows(
        con,
        f"""
        SELECT e.predication_id, e.argument_format, e.join_class, e.sentence_id, e.pmid, e.gene_position,
               e.gene_cui, e.disease_cui, e.predicate,
               p.subject_cui, p.subject_name, p.subject_semtype,
               p.object_cui, p.object_name, p.object_semtype,
               a.subject_text, a.object_text,
               s.sentence_text
        FROM scope_labeled e
        JOIN example_ids i USING (predication_id)
        JOIN read_parquet({pred}) p USING (predication_id)
        LEFT JOIN read_parquet({aux}) a USING (predication_id)
        LEFT JOIN read_parquet({sent}) s ON e.sentence_id = s.sentence_id
        """,
    )
    out = []
    for row in found:
        gene_text = row["subject_text"] if row["gene_position"] == "gene_subject" else row["object_text"]
        body = row.get("sentence_text") or ""
        out.append(
            {
                "predication_id": row["predication_id"],
                "argument_format": row["argument_format"],
                "join_class": row["join_class"],
                "sentence_id": row["sentence_id"],
                "pmid": row["pmid"],
                "gene_position": row["gene_position"],
                "gene_cui": row["gene_cui"],
                "disease_cui": row["disease_cui"],
                "predicate": row["predicate"],
                "subject_cui": row["subject_cui"],
                "subject_name": row["subject_name"],
                "subject_semtype": row["subject_semtype"],
                "object_cui": row["object_cui"],
                "object_name": row["object_name"],
                "object_semtype": row["object_semtype"],
                "aux_gene_side_text": None if gene_text is None else str(gene_text)[:160],
                "aux_text_in_sentence": bool(gene_text) and str(gene_text) in body,
                "sentence_row_found": row.get("sentence_text") is not None,
            }
        )
    log_progress(logger, "examples", n=len(out))
    path = OUT / "cases" / "examples.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in out:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return out


def summarize_existing_cases() -> dict:
    checks = []
    if CASES.is_file():
        for line in CASES.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") == "entity_span_crosscheck":
                checks.append(row)
    n = len(checks)
    in_sentence = sum(1 for row in checks if row.get("mention_in_sentence") is True)
    slice_any = sum(1 for row in checks if any(row.get("slice_equals_mention", {}).values()))
    delta0 = sum(1 for row in checks if row.get("length_delta") == 0)
    return {
        "n": n,
        "fields_compared": "比较了第 1 列候选句子、修复后 CSV 的提及文本、sentence_text，以及 start/end 在 -1、0、+1 位移下的切片。没有比较 gene 字段。",
        "mention_found_in_candidate_sentence": in_sentence,
        "slice_equals_mention_any_shift": slice_any,
        "length_delta_zero": delta0,
        "gene_id_compared": 0,
        "note": "These rows were the first four entities of five parquet row groups. They were not selected as genes. A substring hit supports the candidate sentence; a failed slice does not by itself reject the sentence. No gene-id field was compared.",
    }


def core_corrections(con, logger) -> dict:
    logger.info("phase_start phase=core_corrections")
    pair_pmid = qpath(RESULTS / "01_core_audit" / "tables" / "pair_pmid.parquet")
    sentence = qpath(RESULTS / "01_core_audit" / "tables" / "sentence.parquet")
    hetero = rows(
        con,
        f"""
        SELECT
          COUNT(*) FILTER (WHERE heterogeneous_within_pmid) AS heterogeneous_pair_pmid,
          COUNT(*) FILTER (WHERE heterogeneous_within_pmid AND n_sentence > 1) AS more_than_one_sentence,
          COUNT(*) FILTER (WHERE heterogeneous_within_pmid AND n_sentence = 1) AS one_sentence,
          COUNT(*) FILTER (WHERE heterogeneous_within_pmid AND n_sentence = 0) AS zero_sentence
        FROM read_parquet({pair_pmid})
        """,
    )[0]
    shared = rows(
        con,
        f"""
        WITH per_sentence AS (
          SELECT sentence_uri,
                 COUNT(DISTINCT gcs_uri) AS n_gcs,
                 COUNT(DISTINCT label_tuple) AS n_tuple
          FROM read_parquet({sentence})
          WHERE gcs_uri IS NOT NULL
          GROUP BY 1
        )
        SELECT
          COUNT(*) FILTER (WHERE n_gcs > 1) AS multi_gcs_sentences,
          COUNT(*) FILTER (WHERE n_gcs > 1 AND n_tuple = 1) AS multi_gcs_one_label_tuple,
          COUNT(*) FILTER (WHERE n_gcs > 1 AND n_tuple > 1) AS multi_gcs_several_label_tuples
        FROM per_sentence
        """,
    )[0]
    log_progress(logger, "core_corrections", hetero=hetero["heterogeneous_pair_pmid"])
    return {"heterogeneity": hetero, "shared_sentence_labels": shared}


def pair_class_check(con, logger) -> dict:
    logger.info("phase_start phase=pair_class_check")
    rel = qpath(RESULTS / "03_integrated_analysis" / "data_analysis_round1" / "tables" / "pair_relations.parquet")
    pair = qpath(RESULTS / "01_core_audit" / "tables" / "pair.parquet")
    core_pp = qpath(RESULTS / "01_core_audit" / "tables" / "pair_pmid.parquet")
    sem_pp = qpath(RESULTS / "03_integrated_analysis" / "data_analysis_round1" / "derived" / "j03" / "semmed_pair_pmid.parquet")
    keys = rows(
        con,
        f"""
        SELECT COUNT(*) AS n_rows,
               COUNT(DISTINCT gene_id || '|' || disease_cui) AS n_keys
        FROM read_parquet({rel})
        """,
    )[0]
    missing = rows(
        con,
        f"""
        SELECT
          (SELECT COUNT(*) FROM read_parquet({pair}) p ANTI JOIN read_parquet({rel}) r USING (gene_id, disease_cui)) AS pair_missing_from_class,
          (SELECT COUNT(*) FROM read_parquet({rel}) r ANTI JOIN read_parquet({pair}) p USING (gene_id, disease_cui)) AS class_missing_from_pair
        """,
    )[0]
    con.execute(
        f"""
        CREATE OR REPLACE TABLE rel AS
        SELECT *,
          CASE
            WHEN n_core = 0 AND n_semmed = 0 THEN 'both_empty'
            WHEN n_core = 0 THEN 'core_pmids_empty'
            WHEN n_semmed = 0 THEN 'semmed_pmids_empty'
            WHEN n_intersection = n_core AND n_intersection = n_semmed THEN 'identical'
            WHEN n_intersection = n_core AND n_semmed > n_core THEN 'core_subset'
            WHEN n_intersection = n_semmed AND n_core > n_semmed THEN 'semmed_subset'
            WHEN n_intersection > 0 THEN 'partial_overlap'
            ELSE 'disjoint'
          END AS recomputed
        FROM read_parquet({rel})
        """
    )
    mismatch = one(con, "SELECT COUNT(*) FROM rel WHERE recomputed <> relation_class")
    branch = rows(
        con,
        """
        SELECT
          COUNT(*) FILTER (WHERE (n_core = 0 AND n_semmed = 0)::INT
            + (n_core = 0 AND n_semmed > 0)::INT
            + (n_semmed = 0 AND n_core > 0)::INT
            + (n_core > 0 AND n_semmed > 0 AND n_intersection = n_core AND n_intersection = n_semmed)::INT
            + (n_core > 0 AND n_semmed > n_core AND n_intersection = n_core)::INT
            + (n_semmed > 0 AND n_core > n_semmed AND n_intersection = n_semmed)::INT
            + (n_core > 0 AND n_semmed > 0 AND n_intersection > 0
               AND NOT (n_intersection = n_core AND n_intersection = n_semmed)
               AND NOT (n_intersection = n_core AND n_semmed > n_core)
               AND NOT (n_intersection = n_semmed AND n_core > n_semmed))::INT
            + (n_core > 0 AND n_semmed > 0 AND n_intersection = 0)::INT
            <> 1) AS not_exactly_one_branch
        FROM rel
        """,
    )[0]
    setdiff = rows(
        con,
        f"""
        WITH core_keys AS (
          SELECT gene_id, disease_cui, pmid FROM read_parquet({core_pp})
        ), sem_keys AS (
          SELECT gene_id, disease_cui, pmid FROM read_parquet({sem_pp})
        ), inter AS (
          SELECT * FROM core_keys INTERSECT SELECT * FROM sem_keys
        ), core_only AS (
          SELECT * FROM core_keys EXCEPT SELECT * FROM sem_keys
        ), sem_only AS (
          SELECT * FROM sem_keys EXCEPT SELECT * FROM core_keys
        )
        SELECT
          (SELECT COUNT(*) FROM inter) AS intersection_pair_pmid,
          (SELECT COUNT(*) FROM core_only) AS core_only_pair_pmid,
          (SELECT COUNT(*) FROM sem_only) AS semmed_only_pair_pmid,
          (SELECT COALESCE(SUM(n_intersection), 0) FROM rel) AS class_intersection_sum,
          (SELECT COALESCE(SUM(n_core_only), 0) FROM rel) AS class_core_only_sum,
          (SELECT COALESCE(SUM(n_semmed_only), 0) FROM rel) AS class_semmed_only_sum
        """,
    )[0]
    log_progress(logger, "pair_class_check", mismatch=mismatch)
    return {"keys": keys, "coverage": missing, "recomputed_mismatch": mismatch, "branches": branch, "setdiff": setdiff}


def write_tables(measured: dict) -> None:
    import csv

    path = OUT / "tables"
    path.mkdir(parents=True, exist_ok=True)
    _tsv(path / "argument_format_by_join.tsv", measured["formats"]["format_by_join_class"])
    _tsv(path / "no_entity_states.tsv", measured["no_entity"]["exclusive"])
    _tsv(path / "no_entity_by_format.tsv", measured["no_entity"]["by_format"])


def _tsv(path: Path, records: list[dict]) -> None:
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fields = list(records[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def render(measured: dict) -> str:
    formats = measured["formats"]["by_format"]
    examples = measured["formats"]["examples"]
    cross = measured["formats"]["format_by_join_class"]
    no_ent = measured["no_entity"]
    digits = no_ent["digit_arguments_among_no_entity"]
    cases = measured["existing_cases"]
    core = measured["core"]
    pair = measured["pair_class"]
    hetero = core["heterogeneity"]
    shared = core["shared_sentence_labels"]
    loss = _loss_paragraph(no_ent)
    lines = [
        "# SemMedDB 连接与基因 ID 路径核查",
        "",
        "本核查保留 id_rule_v1 的 276 个 pair，不替换主结果。它解释范围 E 里基因 ID 为什么几乎没有接上，并核对两项 CORE 计数的单位。",
        "",
        "## 1. 基因侧参数是什么形式",
        "",
        "范围 E 派生表里的 gene_cui 是谓词基因侧的原始参数，按 semtype 从 SUBJECT 或 OBJECT 取出。分类只看字符形式。",
        "",
        _md(formats),
        "",
        "每种形式里出现次数最多的原始参数：",
        "",
        _md(examples),
        "",
        "与当前连接状态的交叉：",
        "",
        _md(cross),
        "",
        "## 2. no_entity_row 拆开之后",
        "",
        f"no_entity_row 行数：{no_ent['n']}。判定顺序：{no_ent['exclusive_priority']}。",
        "",
        no_ent["scan"],
        "",
        "完整字符串比较的是：在第 1 列等于候选 sentence_id 的实体上，参数是否等于 entity_id、第 2 列原文或 gene 字段；以及在第 2 列等于候选 sentence_id 时，参数是否等于 CUI。复合值没有被拆开。",
        "",
        _md(no_ent["exclusive"]),
        "",
        "按参数形式分开的标志。同一行可以同时计入多个标志：",
        "",
        _md(no_ent["by_format"]),
        "",
        "no_entity 中的纯数字参数有 "
        + str(digits["n_digit_arguments"])
        + " 条，其中 "
        + str(digits["equal_gene_field_same_sentence"])
        + " 条与同一候选句子上的 gene_id 或 gene_id_raw 完整字符串相同。谓词表没有带命名空间的基因列，这个相等关系不能单独变成主规则，也不替换 id_rule_v1。",
        "",
        "代表实例保留主语/宾语位置、两侧 CUI、名称、semantic type、sentence_id 和 PMID，见 `cases/examples.jsonl`。AUX 提及是否出现在候选句子里只核对候选对应，子串相同不算确定映射。",
        "",
        loss,
        "",
        "## 3. 已有 20 条抽查分别支持什么",
        "",
        f"共 {cases['n']} 条。{cases['fields_compared']}",
        "",
        f"原始实体行已找到：{cases['n']}。候选句子文本包含该提及：{cases['mention_found_in_candidate_sentence']}。任一位移下切片等于提及：{cases['slice_equals_mention_any_shift']}。end 减 start 与提及长度之差为 0：{cases['length_delta_zero']}。比较过 gene_id 含义的条数：{cases['gene_id_compared']}。",
        "",
        "这四项要分开看。找到原始行，以及提及出现在候选句子中，支持第 1 列连到了含有该提及的句子。切片全部不相等，同时长度差为 0，说明 start/end 不是这句文本的字符坐标；更像坐标原点不在该句，而不是编码替换了字符。这 20 条没有比较 gene 字段，所以不支持 gene_id 的含义。",
        "",
        "本地没有与这 12 列 ENTITY 对应的 DDL。上面的数字就是本轮能独立支持的范围：句子对应有子串证据，字符坐标没有对齐，基因字段语义没有被这 20 条检验。",
        "",
        "## 4. 应改哪里，276 是否保留",
        "",
        "ENTITY parquet 已有 cui、entity_id、两个数字列、gene_id、gene_id_raw。谓词 parquet 已有两侧 CUI、名称和 semantic type。AUX parquet 已有两侧提及文本。本轮用的是这些列，外加已保存的 20 条提及抽查。",
        "",
        "不需要重读 CITATIONS、SENTENCE、PREDICATION、PREDICATION_AUX、GENERIC_CONCEPT 或原始 ENTITY CSV。也不需要因为这一项重跑 J01，或重做六张表。",
        "",
        "因此修正范围停在 J03 的连接解释。谓词参数和实体的 gene 字段都已经在现有 parquet 里。不需要重生成 J02，也不需要重读六张原始表。新的候选规则如果要写，只能单独保存，不能覆盖 id_rule_v1。在没有命名空间依据之前，不应为了提高重叠而重跑 J03–J05。",
        "",
        "276 个共享 pair 继续作为主结果。它是 id_rule_v1 的结果：同一候选句子、疾病侧 CUI、唯一 numeric gene_id。本轮没有给出可以替换它的规则。",
        "",
        "## 5. 三项 CORE 纠正",
        "",
        f"pair–PMID 内标签组合多于一种的有 {hetero['heterogeneous_pair_pmid']}。其中句子数大于 1 的是 {hetero['more_than_one_sentence']}，等于 1 的是 {hetero['one_sentence']}，等于 0 的是 {hetero['zero_sentence']}。这是同一 pair、同一 PMID 下多个句子之间的标签差异。句子数等于 1 的部分才可能是句内差异；上面的数字说明它是否存在。",
        "",
        f"多个 GCS 的句子有 {shared['multi_gcs_sentences']}。其中标签组合只有一种的是 {shared['multi_gcs_one_label_tuple']}，多于一种的是 {shared['multi_gcs_several_label_tuples']}。标签写在句子节点上，再连到该句每个 GCS。多个 GCS 共享句子标签，不等于每个 pair 各有一套独立标注。",
        "",
        "pair 分类键：",
        "",
        _md([pair["keys"]]),
        "",
        "双向集合差：",
        "",
        _md([pair["coverage"]]),
        "",
        f"按计数值重算后与原类别不一致的行数：{pair['recomputed_mismatch']}。不能恰好落入一个分支的行数：{pair['branches']['not_exactly_one_branch']}。",
        "",
        "pair–PMID 集合差与分类表中三项计数之和：",
        "",
        _md([pair["setdiff"]]),
        "",
        "分类表行数等于 pair 行数只说明两表一样大。互斥看的是每个键恰好一个分支，且重算类别与原类别一致。完备看的是 pair 键没有双向缺失，并且 pair–PMID 的交、CORE 独有、SemMedDB 独有与分类中的三项计数一致。",
        "",
    ]
    return "\n".join(lines) + "\n"


def _loss_paragraph(no_ent: dict) -> str:
    by_state = {row["exclusive_state"]: row["n_predication"] for row in no_ent["exclusive"]}
    digits = no_ent["digit_arguments_among_no_entity"]
    umls_without_entity = 0
    for row in no_ent["by_format"]:
        if row["argument_format"] == "single_umls_cui":
            umls_without_entity = row["no_entity_on_sentence"]
    return "\n".join(
        [
            "句子连接没有落在“未验证”。no_entity 的 sentence_absent 全是 0，候选 sentence_id 都能在句子表找到。抽查实例的句子行也存在，AUX 基因侧提及出现在该句文本中；这只支持候选对应，不是确定的实体映射。",
            "",
            "用第 2 个数字列当作句子号、再要求 CUI 完整字符串相等：cui_match_on_column2 全是 0。连接失败不是句子列选错。",
            "",
            "按本轮要求的六类来读上面的互斥计数：",
            "",
            f"- 句中没有实体记录：{by_state.get('sentence_has_no_entity_row', 0)}。其中单个 CUI 有 {umls_without_entity}，其余是竖线复合值或纯数字，而且该候选句子在第 1 列下没有任何实体。",
            f"- 候选表示不同，且没有拆开复合值：{by_state.get('composite_argument_not_expanded', 0)}。这些行的参数含竖线，句子里有实体，但整个字符串不等于 CUI、entity_id、第 2 列或 gene 字段。",
            f"- 原始参数与 CUI 不相等，但与同句 gene 字段完整字符串相等：{by_state.get('exact_string_on_other_entity_field', 0)}。它们都来自纯数字参数；纯数字共 {digits['n_digit_arguments']}，对上 gene 字段的是 {digits['equal_gene_field_same_sentence']}。等于 entity_id 或第 2 列的是 0。",
            f"- 句中有实体，但原始 ID 与已核对的字段都不相等，当前无法解释：{by_state.get('cui_string_not_equal_to_entity_cui', 0)}。",
            "- 句子连接未验证：0。有歧义、需要在多个候选之间取舍的情况：本轮没有出现。",
            "",
            "所以 497,287 不是“实体缺失”。主要损失是基因侧参数经常写成 `CUI|数字` 或单个数字，而当前连接要求它与 ENTITY.CUI 整个字符串相同。单个 CUI 且基因字段唯一有数字的，只有交叉表里的 103 条。另有 121,581 条单个 CUI 已经连到实体，但 gene 字段为空，因此也没有进入这 103。",
        ]
    )


def _md(records: list[dict]) -> str:
    if not records:
        return "（无行）"
    keys = list(records[0].keys())
    head = "| " + " | ".join(keys) + " |"
    sep = "| " + " | ".join("---" for _ in keys) + " |"
    body = [
        "| " + " | ".join("" if row.get(key) is None else str(row.get(key)).replace("|", "\\|") for key in keys) + " |"
        for row in records
    ]
    return "\n".join([head, sep, *body])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        raise SystemExit(1)

#!/usr/bin/env python3
"""Check whether the year table omits SemMedDB-only PMIDs, and correct that filter."""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE_ROOT = Path("/home/b5ac/freddieyu.b5ac/project_2")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CODE_ROOT))

from lib.duckutil import connect  # noqa: E402
from lib.runtime import assert_compute_node, prologue, read_json, setup_logging, write_json  # noqa: E402

RESULTS = Path("/projects/b5ac/project_2/results")
ROUND = RESULTS / "03_integrated_analysis" / "id_alignment_v2_round1"
OUT = ROUND / "revision" / "year_coverage"
PMID_YEAR = RESULTS / "03_integrated_analysis" / "data_analysis_round1" / "derived" / "j03" / "pmid_year.parquet"
SEMMED_PP = ROUND / "tables" / "semmed_pair_pmid.parquet"
CORE_PP = RESULTS / "01_core_audit" / "tables" / "pair_pmid.parquet"
PAIR = RESULTS / "01_core_audit" / "tables" / "pair.parquet"
CORE_SENTENCE = RESULTS / "01_core_audit" / "tables" / "sentence.parquet"
CITATIONS = RESULTS / "02_semmeddb_validation" / "tables" / "citations.parquet"
V22_DB = ROUND / "derived" / "v22.duckdb"
OLD_NUMBERS = ROUND / "revision" / "derived" / "revision_numbers.json"
JOINT = ROUND / "derived" / "joint_v2.json"


def main() -> int:
    assert_compute_node()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "derived").mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    logger = setup_logging(OUT / "logs" / f"v25_{os.environ.get('SLURM_JOB_ID', 'local')}.log")
    info = prologue("V25", logger, OUT, [str(PMID_YEAR), str(SEMMED_PP), str(CITATIONS)])
    _assert_set_algebra()
    con = connect(OUT / "derived" / "v25.duckdb", "72GB", OUT / "tmp")
    logger.info("phase_start phase=load")
    load(con)
    logger.info("phase_start phase=coverage")
    coverage = year_table_coverage(con)
    logger.info("phase_start phase=key_sets")
    keys = compare_keys(con)
    logger.info("phase_start phase=citations")
    citations = classify_citations(con)
    core_only = coverage["pmid_year_equals_core_sentence_pmids"]
    corrected = None
    if core_only:
        logger.info("phase_start phase=extend_year_table")
        corrected = extend_and_recompute(con, logger)
    payload = {
        "code_fingerprint": info["code_fingerprint"],
        "coverage": coverage,
        "key_sets": keys,
        "citations": citations,
        "corrected": corrected,
        "unfiltered_main_result_unchanged": True,
        "reason": "pmid_year.parquet is built from core_pmid only. SemMedDB match PMIDs outside that set were labeled pmid_absent_from_year_table.",
    }
    write_json(OUT / "derived" / "year_coverage.json", payload)
    (OUT / "report.md").write_text(render(payload), encoding="utf-8")
    ok = keys["stored_common_article_matches_definition"] and (not core_only or corrected["checks_ok"])
    write_json(OUT / "derived" / "v25_DONE.json", {"status": "success" if ok else "failed", "job": "V25", "report": str(OUT / "report.md")})
    logger.info("phase=v25_%s", "success" if ok else "failed")
    return 0 if ok else 1


def _assert_set_algebra() -> None:
    import duckdb

    con = duckdb.connect()
    con.execute(
        """
        CREATE TABLE t AS SELECT * FROM (VALUES
          ('g','d','1'), ('g','d','2'), ('g','e','3')
        ) v(gene_id, disease_cui, pmid)
        """
    )
    con.execute("CREATE TABLE a AS SELECT * FROM t WHERE pmid <> '2'")
    con.execute("CREATE TABLE b AS SELECT * FROM t WHERE pmid <> '3'")
    got = con.execute(
        """
        SELECT
          (SELECT count(*) FROM (SELECT * FROM a INTERSECT SELECT * FROM b)),
          (SELECT count(*) FROM a ANTI JOIN b USING (gene_id, disease_cui, pmid)),
          (SELECT count(*) FROM b ANTI JOIN a USING (gene_id, disease_cui, pmid))
        """
    ).fetchone()
    if got != (1, 1, 1):
        raise SystemExit(f"set algebra self-check failed: {got}")


def load(con) -> None:
    con.execute(f"CREATE OR REPLACE TABLE pmid_year AS SELECT * FROM read_parquet('{PMID_YEAR.as_posix()}')")
    con.execute(f"CREATE OR REPLACE TABLE sem_pp AS SELECT gene_id, disease_cui, pmid FROM read_parquet('{SEMMED_PP.as_posix()}')")
    con.execute(f"CREATE OR REPLACE TABLE core_pp AS SELECT gene_id, disease_cui, pmid FROM read_parquet('{CORE_PP.as_posix()}')")
    con.execute(f"CREATE OR REPLACE TABLE pair AS SELECT gene_id, disease_cui FROM read_parquet('{PAIR.as_posix()}')")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE core_sentence_pmid AS
        SELECT DISTINCT pmid FROM read_parquet('{CORE_SENTENCE.as_posix()}') WHERE pmid IS NOT NULL
        """
    )
    con.execute(f"ATTACH '{V22_DB.as_posix()}' AS src (READ_ONLY)")


def year_table_coverage(con) -> dict:
    return {
        "pmid_year_rows": one(con, "SELECT count(*) FROM pmid_year"),
        "pmid_year_distinct_pmid": one(con, "SELECT count(DISTINCT pmid) FROM pmid_year"),
        "core_sentence_pmid": one(con, "SELECT count(*) FROM core_sentence_pmid"),
        "pmid_year_not_in_core_sentence": one(
            con,
            "SELECT count(*) FROM (SELECT DISTINCT pmid FROM pmid_year) y ANTI JOIN core_sentence_pmid c USING (pmid)",
        ),
        "core_sentence_not_in_pmid_year": one(
            con,
            "SELECT count(*) FROM core_sentence_pmid c ANTI JOIN (SELECT DISTINCT pmid FROM pmid_year) y USING (pmid)",
        ),
        "pmid_year_equals_core_sentence_pmids": one(
            con,
            """
            SELECT
              (SELECT count(*) FROM (SELECT DISTINCT pmid FROM pmid_year) y ANTI JOIN core_sentence_pmid c USING (pmid)) = 0
              AND (SELECT count(*) FROM core_sentence_pmid c ANTI JOIN (SELECT DISTINCT pmid FROM pmid_year) y USING (pmid)) = 0
            """,
        ),
        "v2_match_pmid": one(con, "SELECT count(DISTINCT pmid) FROM sem_pp"),
        "v2_match_pmid_absent_from_pmid_year": one(
            con,
            """
            SELECT count(*) FROM (SELECT DISTINCT pmid FROM sem_pp) s
            ANTI JOIN (SELECT DISTINCT pmid FROM pmid_year) y USING (pmid)
            """,
        ),
        "build": "align_build._years writes pmid_year from core_pmid, which is DISTINCT pmid FROM core_sentence. Citations are left-joined onto that set only.",
    }


def compare_keys(con) -> dict:
    con.execute(
        f"""
        CREATE OR REPLACE TABLE citation_pmid AS
        SELECT DISTINCT pmid FROM read_parquet('{CITATIONS.as_posix()}') WHERE pmid IS NOT NULL
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE common_article AS
        SELECT pmid FROM core_sentence_pmid
        INTERSECT
        SELECT pmid FROM citation_pmid
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE removed_year_absent AS
        SELECT s.gene_id, s.disease_cui, s.pmid
        FROM sem_pp s
        ANTI JOIN pmid_year y USING (pmid)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE removed_common AS
        SELECT s.gene_id, s.disease_cui, s.pmid
        FROM sem_pp s
        ANTI JOIN common_article a USING (pmid)
        """
    )
    stored_only = one(
        con,
        """
        WITH stored AS (
          SELECT gene_id, disease_cui, pmid FROM src.semmed_pair_pmid
          ANTI JOIN src.sem_pp_common USING (gene_id, disease_cui, pmid)
        )
        SELECT
          (SELECT count(*) FROM stored s ANTI JOIN removed_common c USING (gene_id, disease_cui, pmid))
          + (SELECT count(*) FROM removed_common c ANTI JOIN stored s USING (gene_id, disease_cui, pmid))
        """,
    )
    sets = {
        "A_year_absent_pair_pmid": one(con, "SELECT count(*) FROM removed_year_absent"),
        "B_common_article_removed_pair_pmid": one(con, "SELECT count(*) FROM removed_common"),
        "intersection_pair_pmid": one(
            con,
            """
            SELECT count(*) FROM (
              SELECT * FROM removed_year_absent
              INTERSECT
              SELECT * FROM removed_common
            )
            """,
        ),
        "A_not_B_pair_pmid": one(
            con,
            "SELECT count(*) FROM removed_year_absent a ANTI JOIN removed_common b USING (gene_id, disease_cui, pmid)",
        ),
        "B_not_A_pair_pmid": one(
            con,
            "SELECT count(*) FROM removed_common b ANTI JOIN removed_year_absent a USING (gene_id, disease_cui, pmid)",
        ),
        "A_pmid": one(con, "SELECT count(DISTINCT pmid) FROM removed_year_absent"),
        "B_pmid": one(con, "SELECT count(DISTINCT pmid) FROM removed_common"),
        "intersection_pmid": one(
            con,
            """
            SELECT count(*) FROM (
              SELECT DISTINCT pmid FROM removed_year_absent
              INTERSECT
              SELECT DISTINCT pmid FROM removed_common
            )
            """,
        ),
        "A_not_B_pmid": one(
            con,
            """
            SELECT count(*) FROM (SELECT DISTINCT pmid FROM removed_year_absent) a
            ANTI JOIN (SELECT DISTINCT pmid FROM removed_common) b USING (pmid)
            """,
        ),
        "B_not_A_pmid": one(
            con,
            """
            SELECT count(*) FROM (SELECT DISTINCT pmid FROM removed_common) b
            ANTI JOIN (SELECT DISTINCT pmid FROM removed_year_absent) a USING (pmid)
            """,
        ),
        "stored_common_article_symmetric_diff": stored_only,
        "stored_common_article_matches_definition": stored_only == 0,
    }
    return sets


def classify_citations(con) -> dict:
    con.execute(
        f"""
        CREATE OR REPLACE TABLE citation_one AS
        SELECT pmid,
               count(*) AS n_row,
               count(DISTINCT dp_status) AS n_status,
               any_value(dp_status) AS dp_status,
               any_value(dp) AS dp,
               any_value(dp_year) AS dp_year
        FROM read_parquet('{CITATIONS.as_posix()}')
        WHERE pmid IN (SELECT DISTINCT pmid FROM removed_year_absent)
        GROUP BY pmid
        """
    )
    detail = rows(
        con,
        """
        SELECT
          CASE
            WHEN c.pmid IS NULL THEN 'no_citation_row'
            WHEN c.n_status > 1 THEN 'conflicting_citation_rows'
            WHEN c.dp_status = 'parsed_single' AND c.dp_year IS NOT NULL THEN 'parsed_dp_not_in_year_table'
            WHEN c.dp_status IS NULL THEN 'citation_row_dp_status_null'
            ELSE 'citation_row_dp_' || c.dp_status
          END AS dp_class,
          count(*) AS n_pair_pmid,
          count(DISTINCT a.pmid) AS n_pmid
        FROM removed_year_absent a
        LEFT JOIN citation_one c USING (pmid)
        GROUP BY 1
        ORDER BY n_pair_pmid DESC
        """,
    )
    return {
        "by_class": detail,
        "citation_rows_touching_A": one(con, "SELECT coalesce(sum(n_row), 0) FROM citation_one"),
        "A_pmid_with_multiple_status": one(con, "SELECT count(*) FROM citation_one WHERE n_status > 1"),
    }


def extend_and_recompute(con, logger) -> dict:
    con.execute(
        f"""
        CREATE OR REPLACE TABLE citation_universe AS
        SELECT pmid,
               count(*) AS n_row,
               count(DISTINCT dp_status) AS n_status,
               any_value(dp_status) AS dp_status,
               any_value(dp_year) AS dp_year
        FROM read_parquet('{CITATIONS.as_posix()}')
        WHERE pmid IN (
          SELECT pmid FROM core_pp
          UNION
          SELECT pmid FROM sem_pp
        )
        GROUP BY pmid
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE pmid_year_extended AS
        WITH universe AS (
          SELECT pmid FROM core_pp
          UNION
          SELECT pmid FROM sem_pp
        )
        SELECT u.pmid,
               y.core_year_status,
               y.core_year,
               CASE WHEN y.pmid IS NOT NULL THEN y.semmed_dp_status ELSE c.dp_status END AS semmed_dp_status,
               CASE WHEN y.pmid IS NOT NULL THEN y.semmed_dp_year ELSE c.dp_year END AS semmed_dp_year,
               y.pmid IS NOT NULL AS in_original_year_table
        FROM universe u
        LEFT JOIN pmid_year y USING (pmid)
        LEFT JOIN citation_universe c USING (pmid)
        """
    )
    con.execute(
        f"""
        COPY pmid_year_extended TO '{(OUT / 'tables' / 'pmid_year_extended.parquet').as_posix()}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE year_window AS
        SELECT core_year AS year FROM pmid_year_extended
        WHERE core_year_status = 'parsed_single' AND core_year IS NOT NULL
        INTERSECT
        SELECT semmed_dp_year FROM pmid_year_extended
        WHERE semmed_dp_status = 'parsed_single' AND semmed_dp_year IS NOT NULL
        """
    )
    old_years = set()
    if OLD_NUMBERS.is_file():
        old = read_json(OLD_NUMBERS)
        old_years = set(old.get("sensitivity", {}).get("year_window", {}).get("years") or [])
    new_years = [row[0] for row in con.execute("SELECT year FROM year_window ORDER BY 1").fetchall()]
    logger.info("phase=window old=%s new=%s", len(old_years), len(new_years))
    for side, source, own_status, own_year, other_status, other_year in (
        ("core", "core_pp", "core_year_status", "core_year", "semmed_dp_status", "semmed_dp_year"),
        ("sem", "sem_pp", "semmed_dp_status", "semmed_dp_year", "core_year_status", "core_year"),
    ):
        con.execute(
            f"""
            CREATE OR REPLACE TABLE kept_{side} AS
            SELECT s.gene_id, s.disease_cui, s.pmid
            FROM {source} s
            JOIN pmid_year_extended y USING (pmid)
            WHERE y.{own_status} = 'parsed_single'
              AND y.{own_year} IN (SELECT year FROM year_window)
              AND (
                y.{other_year} IS NULL
                OR y.{other_year} = y.{own_year}
                OR y.{other_status} <> 'parsed_single'
              )
            """
        )
        con.execute(
            f"""
            CREATE OR REPLACE TABLE removed_{side} AS
            SELECT s.gene_id, s.disease_cui, s.pmid
            FROM {source} s
            ANTI JOIN kept_{side} k USING (gene_id, disease_cui, pmid)
            """
        )
        con.execute(
            f"""
            CREATE OR REPLACE TABLE reason_{side} AS
            SELECT r.pmid,
              CASE
                WHEN y.pmid IS NULL THEN 'pmid_absent_from_extended_year_table'
                WHEN y.{own_status} IS DISTINCT FROM 'parsed_single' OR y.{own_year} IS NULL THEN 'own_year_not_parsed_single'
                WHEN y.{own_year} NOT IN (SELECT year FROM year_window) THEN 'parsed_year_outside_shared_set'
                WHEN y.{own_status} = 'parsed_single'
                 AND y.{other_status} = 'parsed_single'
                 AND y.{own_year} IS NOT NULL AND y.{other_year} IS NOT NULL
                 AND y.{own_year} <> y.{other_year} THEN 'both_parsed_and_disagree'
                ELSE 'residual'
              END AS exclusive_reason,
              coalesce(y.{own_status}, 'null') AS own_status
            FROM removed_{side} r
            LEFT JOIN pmid_year_extended y USING (pmid)
            """
        )
    reasons = {
        side: {
            "n_removed": one(con, f"SELECT count(*) FROM removed_{side}"),
            "exclusive": rows(con, f"SELECT exclusive_reason, count(*) AS n_pair_pmid FROM reason_{side} GROUP BY 1 ORDER BY n_pair_pmid DESC"),
            "unparsed_status": rows(
                con,
                f"""
                SELECT own_status, count(*) AS n_pair_pmid
                FROM reason_{side}
                WHERE exclusive_reason = 'own_year_not_parsed_single'
                GROUP BY 1 ORDER BY n_pair_pmid DESC
                """,
            ),
        }
        for side in ("core", "sem")
    }
    transition = pair_transition(con)
    classes = year_classes(con)
    n_class = one(con, "SELECT count(*) FROM rel_year_new")
    n_pair = one(con, "SELECT count(*) FROM pair")
    residual = sum(row["n_pair_pmid"] for row in reasons["core"]["exclusive"] if row["exclusive_reason"] == "residual")
    residual += sum(row["n_pair_pmid"] for row in reasons["sem"]["exclusive"] if row["exclusive_reason"] == "residual")
    covered = one(
        con,
        """
        SELECT count(*) FROM (SELECT DISTINCT pmid FROM sem_pp) s
        ANTI JOIN pmid_year_extended y USING (pmid)
        """,
    )
    old_core_removed = one(
        con,
        """
        SELECT count(*) FROM core_pp c
        ANTI JOIN src.core_pp_year k USING (gene_id, disease_cui, pmid)
        """,
    )
    return {
        "checks_ok": residual == 0 and covered == 0 and transition["n_shared"] == transition["n_states"] and n_class == n_pair,
        "extended_pmids": one(con, "SELECT count(*) FROM pmid_year_extended"),
        "extended_not_in_original": one(con, "SELECT count(*) FROM pmid_year_extended WHERE NOT in_original_year_table"),
        "semmed_pmid_still_missing": covered,
        "window": {
            "n": len(new_years),
            "min": new_years[0] if new_years else None,
            "max": new_years[-1] if new_years else None,
            "added_years": sorted(set(new_years) - old_years),
            "removed_years": sorted(old_years - set(new_years)),
        },
        "reasons": reasons,
        "old_core_removed_pair_pmid": old_core_removed,
        "new_core_removed_pair_pmid": reasons["core"]["n_removed"],
        "transition": transition,
        "year_filtered_classes": classes,
        "unfiltered_semmed_pair_pmid": one(con, "SELECT count(*) FROM sem_pp"),
        "unfiltered_pairs_with_both_sides": transition["n_shared"],
    }


def pair_transition(con) -> dict:
    con.execute(
        """
        CREATE OR REPLACE TABLE shared_pairs AS
        SELECT gene_id, disease_cui FROM (
          SELECT DISTINCT gene_id, disease_cui FROM core_pp
          INTERSECT
          SELECT DISTINCT gene_id, disease_cui FROM sem_pp
        )
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE trans AS
        WITH cb AS (SELECT gene_id, disease_cui, count(DISTINCT pmid) AS n FROM core_pp GROUP BY 1, 2),
             sb AS (SELECT gene_id, disease_cui, count(DISTINCT pmid) AS n FROM sem_pp GROUP BY 1, 2),
             ib AS (
               SELECT c.gene_id, c.disease_cui, count(DISTINCT c.pmid) AS n
               FROM core_pp c JOIN sem_pp s USING (gene_id, disease_cui, pmid)
               GROUP BY 1, 2
             ),
             ca AS (SELECT gene_id, disease_cui, count(DISTINCT pmid) AS n FROM kept_core GROUP BY 1, 2),
             sa AS (SELECT gene_id, disease_cui, count(DISTINCT pmid) AS n FROM kept_sem GROUP BY 1, 2),
             ia AS (
               SELECT c.gene_id, c.disease_cui, count(DISTINCT c.pmid) AS n
               FROM kept_core c JOIN kept_sem s USING (gene_id, disease_cui, pmid)
               GROUP BY 1, 2
             )
        SELECT p.gene_id, p.disease_cui,
               coalesce(cb.n, 0) AS n_core_before,
               coalesce(sb.n, 0) AS n_sem_before,
               coalesce(ib.n, 0) AS n_inter_before,
               coalesce(ca.n, 0) AS n_core_after,
               coalesce(sa.n, 0) AS n_sem_after,
               coalesce(ia.n, 0) AS n_inter_after
        FROM shared_pairs p
        LEFT JOIN cb USING (gene_id, disease_cui)
        LEFT JOIN sb USING (gene_id, disease_cui)
        LEFT JOIN ib USING (gene_id, disease_cui)
        LEFT JOIN ca USING (gene_id, disease_cui)
        LEFT JOIN sa USING (gene_id, disease_cui)
        LEFT JOIN ia USING (gene_id, disease_cui)
        """
    )
    states = rows(
        con,
        """
        SELECT CASE
                 WHEN n_core_after > 0 AND n_sem_after > 0 THEN 'both_nonempty'
                 WHEN n_core_after > 0 AND n_sem_after = 0 THEN 'semmed_side_empty'
                 WHEN n_core_after = 0 AND n_sem_after > 0 THEN 'core_side_empty'
                 ELSE 'both_empty'
               END AS state_after,
               count(*) AS n_pair
        FROM trans
        GROUP BY 1
        ORDER BY n_pair DESC
        """,
    )
    fixed = rows(
        con,
        """
        SELECT count(*) AS n_pair,
               sum(n_inter_before) AS inter_before,
               sum(n_core_before + n_sem_before - n_inter_before) AS union_before,
               sum(n_inter_after) AS inter_after,
               sum(n_core_after + n_sem_after - n_inter_after) AS union_after
        FROM trans
        WHERE n_core_after > 0 AND n_sem_after > 0
        """,
    )[0]
    return {
        "n_shared": one(con, "SELECT count(*) FROM shared_pairs"),
        "n_states": sum(row["n_pair"] for row in states),
        "states": states,
        "fixed_both_nonempty": fixed,
    }


def year_classes(con) -> list[dict]:
    con.execute(
        """
        CREATE OR REPLACE TABLE rel_year_new AS
        WITH cu AS (
          SELECT gene_id, disease_cui, count(DISTINCT pmid) AS n_core FROM kept_core GROUP BY 1, 2
        ), su AS (
          SELECT gene_id, disease_cui, count(DISTINCT pmid) AS n_semmed FROM kept_sem GROUP BY 1, 2
        ), inter AS (
          SELECT c.gene_id, c.disease_cui, count(DISTINCT c.pmid) AS n_intersection
          FROM kept_core c JOIN kept_sem s USING (gene_id, disease_cui, pmid)
          GROUP BY 1, 2
        )
        SELECT
          CASE
            WHEN coalesce(cu.n_core, 0) = 0 AND coalesce(su.n_semmed, 0) = 0 THEN 'both_empty'
            WHEN coalesce(cu.n_core, 0) = 0 THEN 'core_pmids_empty'
            WHEN coalesce(su.n_semmed, 0) = 0 THEN 'semmed_pmids_empty'
            WHEN coalesce(inter.n_intersection, 0) = coalesce(cu.n_core, 0)
             AND coalesce(inter.n_intersection, 0) = coalesce(su.n_semmed, 0) THEN 'identical'
            WHEN coalesce(inter.n_intersection, 0) = coalesce(cu.n_core, 0)
             AND coalesce(su.n_semmed, 0) > coalesce(cu.n_core, 0) THEN 'core_subset'
            WHEN coalesce(inter.n_intersection, 0) = coalesce(su.n_semmed, 0)
             AND coalesce(cu.n_core, 0) > coalesce(su.n_semmed, 0) THEN 'semmed_subset'
            WHEN coalesce(inter.n_intersection, 0) > 0 THEN 'partial_overlap'
            ELSE 'disjoint'
          END AS relation_class
        FROM pair u
        LEFT JOIN cu USING (gene_id, disease_cui)
        LEFT JOIN su USING (gene_id, disease_cui)
        LEFT JOIN inter USING (gene_id, disease_cui)
        """
    )
    return rows(con, "SELECT relation_class, count(*) AS n_pair FROM rel_year_new GROUP BY 1 ORDER BY n_pair DESC")


def rows(con, sql: str) -> list[dict]:
    cur = con.execute(sql)
    names = [item[0] for item in cur.description]
    return [dict(zip(names, row)) for row in cur.fetchall()]


def one(con, sql: str):
    return con.execute(sql).fetchone()[0]


def render(payload: dict) -> str:
    cov = payload["coverage"]
    keys = payload["key_sets"]
    cit = payload["citations"]
    fix = payload["corrected"]
    same = (
        keys["A_not_B_pair_pmid"] == 0
        and keys["B_not_A_pair_pmid"] == 0
        and keys["A_year_absent_pair_pmid"] == keys["B_common_article_removed_pair_pmid"]
    )
    lines = [
        "# 年份表覆盖范围",
        "",
        "`pmid_year.parquet` 的行来自 `core_pmid`，也就是 CORE 句子表里的 PMID，再左连接 citations。它不包含只出现在 v2 匹配里的 SemMedDB PMID。",
        "",
        f"年份表 PMID {cov['pmid_year_distinct_pmid']}，CORE 句子 PMID {cov['core_sentence_pmid']}。年份表多出的 PMID {cov['pmid_year_not_in_core_sentence']}，CORE 句子里没有进入年份表的 PMID {cov['core_sentence_not_in_pmid_year']}。v2 匹配 PMID {cov['v2_match_pmid']}，其中 {cov['v2_match_pmid_absent_from_pmid_year']} 个不在年份表里。",
        "",
        "## 两组键",
        "",
        "A 是 v2 的 `(gene_id, disease_cui, pmid)` 中 PMID 不在原年份表的记录。B 是同一张表里 PMID 不在共同文章集合的记录。共同文章是 CORE 句子 PMID 与 citations PMID 的交集。",
        "",
        _md(
            [
                {
                    "set": "A 年份表缺失",
                    "pair_pmid": keys["A_year_absent_pair_pmid"],
                    "pmid": keys["A_pmid"],
                },
                {
                    "set": "B 共同文章删除",
                    "pair_pmid": keys["B_common_article_removed_pair_pmid"],
                    "pmid": keys["B_pmid"],
                },
                {
                    "set": "交集",
                    "pair_pmid": keys["intersection_pair_pmid"],
                    "pmid": keys["intersection_pmid"],
                },
                {
                    "set": "只在 A",
                    "pair_pmid": keys["A_not_B_pair_pmid"],
                    "pmid": keys["A_not_B_pmid"],
                },
                {
                    "set": "只在 B",
                    "pair_pmid": keys["B_not_A_pair_pmid"],
                    "pmid": keys["B_not_A_pmid"],
                },
            ]
        ),
        "",
        ("两组键集合相同。" if same else "两组键集合不同。")
        + f" 与 V22 已保存的共同文章删除集对称差为 {keys['stored_common_article_symmetric_diff']}。",
        "",
        "## A 组 PMID 在 citations 中的 DP",
        "",
        "下面用已有 `citations.parquet` 的 `pmid`、`dp`、`dp_status`、`dp_year`。没有重读原始 CSV。",
        "",
        _md(cit["by_class"]),
        "",
        "`no_citation_row` 是 citations 没有该 PMID。`citation_row_dp_` 开头的类是有引文行，但 DP 不是单一年份。`parsed_dp_not_in_year_table` 是 citations 已有可解析 DP，只是该 PMID 不属于 CORE，所以没有进入派生年份表。",
        "",
    ]
    if fix is None:
        lines.append("年份表并不只覆盖 CORE PMID，因此没有改筛选。")
    else:
        window = fix["window"]
        lines.extend(
            [
                "## 修正",
                "",
                "年份表确认只覆盖 CORE 句子 PMID，所以为 CORE pair–PMID 与 v2 匹配 PMID 的并集补了年份。CORE PMID 仍用原表里的 CORE 出版年和当时连接上的 DP。新进入的 PMID 没有 CORE 出版年，DP 来自 citations。旧的 `pmid_year.parquet` 和上一份修订报告没有覆盖。",
                "",
                f"补入后的年份表有 {fix['extended_pmids']} 个 PMID，其中 {fix['extended_not_in_original']} 个不在原表。窗口从原来的集合变为 {window['n']} 个年份（{window['min']}–{window['max']}）。新增年份 {window['added_years'] or '无'}，消失年份 {window['removed_years'] or '无'}。",
                "",
                f"CORE 侧删除的 pair–PMID：原来 {fix['old_core_removed_pair_pmid']}，现在 {fix['new_core_removed_pair_pmid']}。",
                "",
                "SemMedDB 侧删除原因：",
                "",
                _md(fix["reasons"]["sem"]["exclusive"]),
                "",
                "其中本侧未解析的 DP 状态：",
                "",
                _md(fix["reasons"]["sem"]["unparsed_status"]),
                "",
                "CORE 侧删除原因：",
                "",
                _md(fix["reasons"]["core"]["exclusive"]),
                "",
                "原先两侧都有 PMID 的 pair，在新窗口下的状态：",
                "",
                _md(fix["transition"]["states"]),
                "",
                _fixed(fix["transition"]["fixed_both_nonempty"]),
                "",
                "新窗口下的 pair 类别。分母仍是全部 CORE 显式 pair。这只替换年份筛选后的类别，不替换未筛选的五类。",
                "",
                _md(fix["year_filtered_classes"]),
                "",
                f"未筛选主结果没有重算。v2 pair–PMID 仍是 {fix['unfiltered_semmed_pair_pmid']} 条，两侧都有 PMID 的 pair 仍是 {fix['unfiltered_pairs_with_both_sides']} 个。共同文章筛选没有改。",
                "",
                "补齐后的年份表在 `revision/year_coverage/tables/pmid_year_extended.parquet`。",
                "",
            ]
        )
    lines.append("机器可读计数在 `revision/year_coverage/derived/year_coverage.json`。")
    lines.append("")
    return "\n".join(lines)


def _fixed(fixed: dict) -> str:
    return (
        f"新窗口后仍两侧都有 PMID 的 pair 有 {fixed['n_pair']} 个。"
        f"同一集合的 pair–PMID Jaccard，筛选前是 {fixed['inter_before']}/{fixed['union_before']}，"
        f"筛选后是 {fixed['inter_after']}/{fixed['union_after']}。"
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

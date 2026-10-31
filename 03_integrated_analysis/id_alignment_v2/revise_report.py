#!/usr/bin/env python3
"""V24: funnel, sensitivity transitions, and the revised v2 report. Does not rescan ENTITY."""

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
from lib.schema import SPARQL_EXAMPLE_EXCLUDED_CUIS  # noqa: E402

RESULTS = Path("/projects/b5ac/project_2/results")
ROUND = RESULTS / "03_integrated_analysis" / "id_alignment_v2_round1"
REV = ROUND / "revision"
V22_DB = ROUND / "derived" / "v22.duckdb"
PROVENANCE = REV / "derived" / "provenance.json"
CUIS = list(SPARQL_EXAMPLE_EXCLUDED_CUIS)


def main() -> int:
    assert_compute_node()
    done = read_json(REV / "derived" / "v23_DONE.json")
    if done.get("status") != "success":
        raise SystemExit(f"V23 is not complete: {done}")
    provenance = read_json(PROVENANCE)
    logger = setup_logging(REV / "logs" / f"v24_{os.environ.get('SLURM_JOB_ID', 'local')}.log")
    info = prologue("V24", logger, REV, [str(PROVENANCE), str(V22_DB)])
    con = connect(REV / "derived" / "v24.duckdb", "72GB", REV / "tmp")
    con.execute(f"ATTACH '{V22_DB.as_posix()}' AS src (READ_ONLY)")
    checks: list[dict] = []
    logger.info("phase_start phase=funnel")
    funnel = build_funnel(con, checks)
    logger.info("phase_start phase=scope_sets")
    scope = build_scope(con, checks)
    logger.info("phase_start phase=rule_execution")
    rules = rule_execution(con, checks)
    logger.info("phase_start phase=sensitivity")
    sensitivity = build_sensitivity(con, checks, logger)
    logger.info("phase_start phase=observable_arguments")
    observable = build_observable(con, checks, logger)
    payload = {
        "code_fingerprint": info["code_fingerprint"],
        "provenance_verdict": provenance["label_audit"]["verdict"],
        "funnel": funnel,
        "scope": scope,
        "rules": rules,
        "sensitivity": sensitivity,
        "observable": observable,
        "checks": checks,
    }
    write_json(REV / "derived" / "revision_numbers.json", payload)
    failed = [item for item in checks if not item["ok"]]
    (REV / "report.md").write_text(render(provenance, payload), encoding="utf-8")
    status = "success" if not failed else "failed"
    write_json(REV / "derived" / "v24_DONE.json", {"status": status, "job": "V24", "report": str(REV / "report.md"), "n_failed_checks": len(failed)})
    logger.info("phase=v24_%s failed_checks=%s", status, len(failed))
    return 0 if status == "success" else 1


def rows(con, sql: str) -> list[dict]:
    cur = con.execute(sql)
    names = [item[0] for item in cur.description]
    return [dict(zip(names, row)) for row in cur.fetchall()]


def one(con, sql: str):
    return con.execute(sql).fetchone()[0]


def check(checks: list[dict], name: str, ok: bool, detail) -> None:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def build_funnel(con, checks: list[dict]) -> dict:
    status = rows(
        con,
        """
        SELECT population, mapping_status, count(*) AS n
        FROM src.mapped
        GROUP BY 1, 2
        ORDER BY 1, 3 DESC
        """,
    )
    core_n = sum(row["n"] for row in status if row["population"] == "core_anchored")
    scope_n = sum(row["n"] for row in status if row["population"] == "scope_e")
    accepted_n = sum(row["n"] for row in status if row["population"] == "core_anchored" and row["mapping_status"] == "accepted")
    grain = one(
        con,
        """
        SELECT count(*) - count(DISTINCT (population, predication_id, disease_cui, gene_side))
        FROM src.mapped
        WHERE population = 'core_anchored'
        """,
    )
    joined = one(
        con,
        """
        SELECT count(*)
        FROM src.mapped m
        JOIN src.pair p ON m.accepted_gene_id = p.gene_id AND m.disease_cui = p.disease_cui
        WHERE m.population = 'core_anchored' AND m.mapping_status = 'accepted'
        """,
    )
    unmatched = one(
        con,
        """
        SELECT count(*)
        FROM src.mapped m
        ANTI JOIN src.pair p ON m.accepted_gene_id = p.gene_id AND m.disease_cui = p.disease_cui
        WHERE m.population = 'core_anchored' AND m.mapping_status = 'accepted'
        """,
    )
    routes = rows(
        con,
        """
        SELECT m.mapping_route,
               count(*) AS n_accepted_argument,
               count(*) FILTER (WHERE p.gene_id IS NOT NULL) AS n_argument_on_core_pair
        FROM src.mapped m
        LEFT JOIN src.pair p ON m.accepted_gene_id = p.gene_id AND m.disease_cui = p.disease_cui
        WHERE m.population = 'core_anchored' AND m.mapping_status = 'accepted'
        GROUP BY 1
        ORDER BY n_accepted_argument DESC
        """,
    )
    match_rows = one(con, "SELECT count(*) FROM src.v2_match")
    pairs = one(con, "SELECT count(*) FROM (SELECT DISTINCT gene_id, disease_cui FROM src.semmed_pair_pmid)")
    pair_pmid = one(con, "SELECT count(*) FROM src.semmed_pair_pmid")
    pmids = one(con, "SELECT count(DISTINCT pmid) FROM src.semmed_pair_pmid")
    null_pmid = one(con, "SELECT count(*) FROM src.v2_match WHERE pmid IS NULL")
    distinct_pair_from_match = one(
        con,
        "SELECT count(*) FROM (SELECT DISTINCT accepted_gene_id, disease_cui FROM src.v2_match)",
    )
    distinct_pp_from_match = one(
        con,
        "SELECT count(*) FROM (SELECT DISTINCT accepted_gene_id, disease_cui, pmid FROM src.v2_match)",
    )
    check(checks, "core_anchored_grain", grain == 0, grain)
    check(checks, "accepted_split_sums", joined + unmatched == accepted_n, {"joined": joined, "unmatched": unmatched, "accepted": accepted_n})
    check(checks, "accepted_match_equals_join", match_rows == joined, {"match": match_rows, "joined": joined})
    check(checks, "null_pmid_in_match", null_pmid == 0, null_pmid)
    check(checks, "pairs_from_match", distinct_pair_from_match == pairs, {"match": distinct_pair_from_match, "pair_pmid_table": pairs})
    check(checks, "pair_pmid_from_match", distinct_pp_from_match == pair_pmid, {"match": distinct_pp_from_match, "table": pair_pmid})
    return {
        "status": status,
        "core_anchored_rows": core_n,
        "scope_e_rows": scope_n,
        "accepted_arguments": accepted_n,
        "accepted_on_core_pair": joined,
        "accepted_gene_not_core_pair": unmatched,
        "routes": routes,
        "pairs": pairs,
        "pair_pmid": pair_pmid,
        "pmids": pmids,
        "unit_note": "513477 and the CORE-pair join are both core_anchored argument rows. 30910, 137221 and the PMID count are aggregations of the joined rows, not further exclusions.",
    }


def build_scope(con, checks: list[dict]) -> dict:
    con.execute(
        """
        CREATE OR REPLACE TABLE main_pairs AS
        SELECT DISTINCT gene_id, disease_cui FROM src.semmed_pair_pmid
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE e_shared AS
        SELECT e.gene_id, e.disease_cui
        FROM src.scope_e_gene_pairs e
        JOIN src.pair p USING (gene_id, disease_cui)
        """
    )
    n_main = one(con, "SELECT count(*) FROM main_pairs")
    n_e = one(con, "SELECT count(*) FROM src.scope_e_gene_pairs")
    n_shared = one(con, "SELECT count(*) FROM e_shared")
    n_e_only = one(
        con,
        """
        SELECT count(*) FROM src.scope_e_gene_pairs e
        ANTI JOIN src.pair p USING (gene_id, disease_cui)
        """,
    )
    n_both = one(
        con,
        """
        SELECT count(*) FROM main_pairs m JOIN e_shared e USING (gene_id, disease_cui)
        """,
    )
    n_main_only = one(
        con,
        """
        SELECT count(*) FROM main_pairs m ANTI JOIN e_shared e USING (gene_id, disease_cui)
        """,
    )
    n_shared_not_main = one(
        con,
        """
        SELECT count(*) FROM e_shared e ANTI JOIN main_pairs m USING (gene_id, disease_cui)
        """,
    )
    main_only_sem = con.execute(
        """
        WITH flagged AS (
          SELECT m.gene_id, m.disease_cui,
                 bool_or(
                   v.argument_semtype = 'gngm'
                   AND (
                     (v.gene_side = 'object' AND v.subject_semtype = 'neop')
                     OR (v.gene_side = 'subject' AND v.object_semtype = 'neop')
                   )
                 ) AS gngm_neop
          FROM main_pairs m
          ANTI JOIN e_shared e USING (gene_id, disease_cui)
          JOIN src.v2_match v ON v.accepted_gene_id = m.gene_id AND v.disease_cui = m.disease_cui
          GROUP BY 1, 2
        )
        SELECT count(*) FILTER (WHERE gngm_neop), count(*) FROM flagged
        """
    ).fetchone()
    examples = rows(
        con,
        """
        SELECT e.gene_id, e.disease_cui
        FROM e_shared e
        ANTI JOIN main_pairs m USING (gene_id, disease_cui)
        LIMIT 8
        """,
    )
    check(checks, "scope_e_pairs_split", n_shared + n_e_only == n_e, {"shared": n_shared, "only": n_e_only, "e": n_e})
    check(checks, "main_pairs_split", n_both + n_main_only == n_main, {"both": n_both, "main_only": n_main_only, "main": n_main})
    check(checks, "main_only_rows_match", main_only_sem[1] == n_main_only, {"seen": main_only_sem[1], "main_only": n_main_only})
    check(checks, "published_scope_shared", n_shared == 27028, n_shared)
    return {
        "n_main_pairs": n_main,
        "n_scope_e_pairs": n_e,
        "n_scope_e_in_core_pairs": n_shared,
        "n_scope_e_not_in_core_pairs": n_e_only,
        "n_main_and_scope_e_shared": n_both,
        "n_main_not_in_scope_e_shared": n_main_only,
        "n_scope_e_shared_not_in_main": n_shared_not_main,
        "main_only_with_gngm_neop_row": main_only_sem[0],
        "main_only_pairs_seen": main_only_sem[1],
        "scope_shared_not_in_main_examples": examples,
        "scope_e_only_population": "Accepted pairs from predications whose two semtype fields are exactly gngm and neop. The anti-join to CORE explicit pairs uses gene_id, disease_cui. This is not the full SemMedDB cancer set and not the core_anchored match.",
    }


def rule_execution(con, checks: list[dict]) -> dict:
    summary = rows(
        con,
        """
        SELECT
          count(*) FILTER (WHERE mapping_status IS NULL) AS null_status,
          count(*) FILTER (WHERE mapping_status = 'accepted' AND entity_check = 'conflicts') AS accepted_marked_conflict,
          count(*) FILTER (WHERE mapping_status = 'conflict' AND accepted_gene_id IS NOT NULL) AS conflict_with_accepted_id,
          count(*) FILTER (WHERE mapping_status = 'ambiguous_multiple_gene_ids' AND accepted_gene_id IS NOT NULL) AS ambiguous_with_accepted_id,
          count(*) FILTER (WHERE mapping_status = 'generic_concept_not_forced' AND accepted_gene_id IS NOT NULL) AS generic_with_accepted_id
        FROM src.mapped
        """,
    )[0]
    accepted_entity = rows(
        con,
        """
        SELECT population, mapping_route, entity_check, count(*) AS n
        FROM src.mapped
        WHERE mapping_status = 'accepted'
        GROUP BY 1, 2, 3
        ORDER BY 1, 4 DESC
        """,
    )
    conflict_examples = rows(
        con,
        """
        SELECT population, argument, argument_format, candidate_gene_ids, entity_check, count(*) AS n
        FROM src.mapped
        WHERE mapping_status = 'conflict'
        GROUP BY 1, 2, 3, 4, 5
        ORDER BY n DESC
        LIMIT 8
        """,
    )
    check(checks, "no_null_status", summary["null_status"] == 0, summary["null_status"])
    check(checks, "accepted_not_labeled_conflict", summary["accepted_marked_conflict"] == 0, summary["accepted_marked_conflict"])
    check(checks, "conflict_has_no_accepted_id", summary["conflict_with_accepted_id"] == 0, summary["conflict_with_accepted_id"])
    check(checks, "ambiguous_has_no_accepted_id", summary["ambiguous_with_accepted_id"] == 0, summary["ambiguous_with_accepted_id"])
    check(checks, "generic_has_no_accepted_id", summary["generic_with_accepted_id"] == 0, summary["generic_with_accepted_id"])
    return {"summary": summary, "accepted_entity_check": accepted_entity, "conflict_examples": conflict_examples}


def build_sensitivity(con, checks: list[dict], logger) -> dict:
    con.execute(
        """
        CREATE OR REPLACE TABLE year_window AS
        SELECT core_year AS year
        FROM src.pmid_year
        WHERE core_year_status = 'parsed_single' AND core_year IS NOT NULL
        INTERSECT
        SELECT semmed_dp_year
        FROM src.pmid_year
        WHERE semmed_dp_status = 'parsed_single' AND semmed_dp_year IS NOT NULL
        """
    )
    years = [row[0] for row in con.execute("SELECT year FROM year_window ORDER BY 1").fetchall()]
    gaps = [[years[i - 1], years[i]] for i in range(1, len(years)) if years[i] - years[i - 1] > 1]
    logger.info("phase=year_window n=%s", len(years))
    for side, own_status, own_year, other_status, other_year, kept in (
        ("core", "core_year_status", "core_year", "semmed_dp_status", "semmed_dp_year", "src.core_pp_year"),
        ("semmed", "semmed_dp_status", "semmed_dp_year", "core_year_status", "core_year", "src.sem_pp_year"),
    ):
        con.execute(
            f"""
            CREATE OR REPLACE TABLE rebuilt_{side} AS
            SELECT c.gene_id, c.disease_cui, c.pmid
            FROM src.{'core_pair_pmid' if side == 'core' else 'semmed_pair_pmid'} c
            JOIN src.pmid_year y USING (pmid)
            WHERE y.{own_status} = 'parsed_single'
              AND y.{own_year} IN (SELECT year FROM year_window)
              AND (y.{other_year} IS NULL OR y.{other_year} = y.{own_year} OR y.{other_status} <> 'parsed_single')
            """
        )
        missing = one(
            con,
            f"""
            SELECT count(*) FROM {kept} k
            ANTI JOIN rebuilt_{side} r USING (gene_id, disease_cui, pmid)
            """,
        )
        extra = one(
            con,
            f"""
            SELECT count(*) FROM rebuilt_{side} r
            ANTI JOIN {kept} k USING (gene_id, disease_cui, pmid)
            """,
        )
        check(checks, f"year_keep_matches_stored_{side}", missing == 0 and extra == 0, {"missing": missing, "extra": extra})
        con.execute(
            f"""
            CREATE OR REPLACE TABLE removed_{side} AS
            SELECT c.gene_id, c.disease_cui, c.pmid
            FROM src.{'core_pair_pmid' if side == 'core' else 'semmed_pair_pmid'} c
            ANTI JOIN {kept} k USING (gene_id, disease_cui, pmid)
            """
        )
        con.execute(
            f"""
            CREATE OR REPLACE TABLE reason_{side} AS
            SELECT r.gene_id, r.disease_cui, r.pmid,
              CASE
                WHEN y.pmid IS NULL THEN 'pmid_absent_from_year_table'
                WHEN y.{own_status} IS DISTINCT FROM 'parsed_single' OR y.{own_year} IS NULL THEN 'own_year_not_parsed_single'
                WHEN y.{own_year} NOT IN (SELECT year FROM year_window) THEN 'parsed_year_outside_shared_set'
                WHEN y.{own_status} = 'parsed_single'
                 AND y.{other_status} = 'parsed_single'
                 AND y.{own_year} IS NOT NULL
                 AND y.{other_year} IS NOT NULL
                 AND y.{own_year} <> y.{other_year} THEN 'both_parsed_and_disagree'
                ELSE 'residual'
              END AS exclusive_reason,
              coalesce(y.{own_status}, 'null') AS own_status,
              (y.pmid IS NULL) AS flag_absent,
              (y.{own_status} IS DISTINCT FROM 'parsed_single' OR y.{own_year} IS NULL) AS flag_own_unparsed,
              (y.{own_status} = 'parsed_single' AND y.{own_year} IS NOT NULL AND y.{own_year} NOT IN (SELECT year FROM year_window)) AS flag_outside,
              (y.{own_status} = 'parsed_single' AND y.{other_status} = 'parsed_single'
                AND y.{own_year} IS NOT NULL AND y.{other_year} IS NOT NULL AND y.{own_year} <> y.{other_year}) AS flag_disagree
            FROM removed_{side} r
            LEFT JOIN src.pmid_year y USING (pmid)
            """
        )
    year_reasons = {
        side: {
            "exclusive": rows(con, f"SELECT exclusive_reason, count(*) AS n FROM reason_{side} GROUP BY 1 ORDER BY n DESC"),
            "own_status_when_unparsed": rows(
                con,
                f"""
                SELECT own_status, count(*) AS n
                FROM reason_{side}
                WHERE exclusive_reason = 'own_year_not_parsed_single'
                GROUP BY 1 ORDER BY n DESC
                """,
            ),
            "flag_cross": rows(
                con,
                f"""
                SELECT flag_absent, flag_own_unparsed, flag_outside, flag_disagree, count(*) AS n
                FROM reason_{side}
                GROUP BY 1, 2, 3, 4
                ORDER BY n DESC
                """,
            ),
            "n_removed": one(con, f"SELECT count(*) FROM removed_{side}"),
        }
        for side in ("core", "semmed")
    }
    for side in ("core", "semmed"):
        exclusive_sum = sum(row["n"] for row in year_reasons[side]["exclusive"])
        residual = sum(row["n"] for row in year_reasons[side]["exclusive"] if row["exclusive_reason"] == "residual")
        check(checks, f"year_reasons_cover_{side}", exclusive_sum == year_reasons[side]["n_removed"] and residual == 0, {"sum": exclusive_sum, "removed": year_reasons[side]["n_removed"], "residual": residual})
    transitions = {}
    for name, core_kept, sem_kept in (
        ("year", "src.core_pp_year", "src.sem_pp_year"),
        ("common_article", "src.core_pp_common", "src.sem_pp_common"),
    ):
        transitions[name] = pair_transition(con, core_kept, sem_kept)
        n_states = sum(row["n_pair"] for row in transitions[name]["states"])
        check(checks, f"transition_covers_shared_{name}", n_states == transitions[name]["n_shared_before"], n_states)
    article = {
        "n_common_article_pmid": one(con, "SELECT count(*) FROM src.common_article"),
        "pair_intersection_pmid_not_in_common_article": one(
            con,
            """
            SELECT count(*) FROM (
              SELECT pmid FROM src.core_pair_pmid
              INTERSECT
              SELECT pmid FROM src.semmed_pair_pmid
            ) i
            ANTI JOIN src.common_article a USING (pmid)
            """,
        ),
        "common_article_pmid_not_in_v2_match": one(
            con,
            """
            SELECT count(*) FROM src.common_article a
            ANTI JOIN (SELECT DISTINCT pmid FROM src.semmed_pair_pmid) s USING (pmid)
            """,
        ),
    }
    cui_sql = ", ".join("'" + cui + "'" for cui in CUIS)
    cui = rows(
        con,
        f"""
        SELECT
          count(*) FILTER (WHERE disease_cui IN ({cui_sql}) AND relation_class = 'both_empty') AS excluded_labeled_both_empty,
          count(*) FILTER (WHERE disease_cui IN ({cui_sql}) AND relation_class <> 'both_empty') AS excluded_other_class,
          count(*) FILTER (WHERE disease_cui NOT IN ({cui_sql}) AND relation_class = 'both_empty') AS true_both_empty,
          count(*) AS n_pairs
        FROM src.rel_exclude_example_cuis
        """,
    )[0]
    unfiltered_both = one(con, "SELECT count(*) FROM src.rel_all WHERE relation_class = 'both_empty'")
    class_counts = rows(con, "SELECT relation_class, count(*) AS n_pair FROM src.rel_all GROUP BY 1 ORDER BY n_pair DESC")
    exclude_counts = rows(
        con,
        f"""
        SELECT CASE WHEN disease_cui IN ({cui_sql}) THEN 'excluded' ELSE relation_class END AS relation_class,
               count(*) AS n_pair
        FROM src.rel_exclude_example_cuis
        GROUP BY 1
        ORDER BY n_pair DESC
        """,
    )
    check(checks, "excluded_not_mixed_with_both_empty", cui["excluded_other_class"] == 0 and cui["true_both_empty"] == 0 and unfiltered_both == 0, cui)
    micro = one(
        con,
        """
        SELECT count(*) FROM (
          SELECT gene_id, disease_cui, pmid FROM src.core_pair_pmid
          INTERSECT
          SELECT gene_id, disease_cui, pmid FROM src.semmed_pair_pmid
        )
        """,
    )
    dedup = {
        "n_core": one(con, "SELECT count(DISTINCT pmid) FROM src.core_pair_pmid"),
        "n_semmed": one(con, "SELECT count(DISTINCT pmid) FROM src.semmed_pair_pmid"),
        "n_intersection": one(
            con,
            """
            SELECT count(*) FROM (
              SELECT DISTINCT pmid FROM src.core_pair_pmid
              INTERSECT
              SELECT DISTINCT pmid FROM src.semmed_pair_pmid
            )
            """,
        ),
    }
    return {
        "year_window": {"n": len(years), "min": years[0] if years else None, "max": years[-1] if years else None, "gaps": gaps, "years": years},
        "year_reasons": year_reasons,
        "transitions": transitions,
        "article": article,
        "cui": cui,
        "unfiltered_both_empty": unfiltered_both,
        "class_counts": class_counts,
        "exclude_counts": exclude_counts,
        "pair_pmid_intersection": micro,
        "deduplicated_pmid": dedup,
    }


def pair_transition(con, core_kept: str, sem_kept: str) -> dict:
    con.execute(
        f"""
        CREATE OR REPLACE TABLE shared_pairs AS
        SELECT gene_id, disease_cui FROM (
          SELECT DISTINCT gene_id, disease_cui FROM src.core_pair_pmid
          INTERSECT
          SELECT DISTINCT gene_id, disease_cui FROM src.semmed_pair_pmid
        )
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE trans AS
        WITH cb AS (
          SELECT gene_id, disease_cui, count(DISTINCT pmid) AS n FROM src.core_pair_pmid GROUP BY 1, 2
        ), sb AS (
          SELECT gene_id, disease_cui, count(DISTINCT pmid) AS n FROM src.semmed_pair_pmid GROUP BY 1, 2
        ), ib AS (
          SELECT c.gene_id, c.disease_cui, count(DISTINCT c.pmid) AS n
          FROM src.core_pair_pmid c
          JOIN src.semmed_pair_pmid s USING (gene_id, disease_cui, pmid)
          GROUP BY 1, 2
        ), ca AS (
          SELECT gene_id, disease_cui, count(DISTINCT pmid) AS n FROM {core_kept} GROUP BY 1, 2
        ), sa AS (
          SELECT gene_id, disease_cui, count(DISTINCT pmid) AS n FROM {sem_kept} GROUP BY 1, 2
        ), ia AS (
          SELECT c.gene_id, c.disease_cui, count(DISTINCT c.pmid) AS n
          FROM {core_kept} c
          JOIN {sem_kept} s USING (gene_id, disease_cui, pmid)
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
               sum(n_core_after + n_sem_after - n_inter_after) AS union_after,
               avg(n_inter_before * 1.0 / nullif(n_core_before + n_sem_before - n_inter_before, 0)) AS mean_jaccard_before,
               avg(n_inter_after * 1.0 / nullif(n_core_after + n_sem_after - n_inter_after, 0)) AS mean_jaccard_after
        FROM trans
        WHERE n_core_after > 0 AND n_sem_after > 0
        """,
    )[0]
    return {"n_shared_before": one(con, "SELECT count(*) FROM shared_pairs"), "states": states, "fixed_both_nonempty": fixed}


def build_observable(con, checks: list[dict], logger) -> dict:
    probe = one(con, "SELECT candidate_gene_ids FROM src.mapped WHERE candidate_gene_ids IS NOT NULL AND candidate_gene_ids <> '[]' LIMIT 1")
    logger.info("phase=candidate_probe %s", probe)
    con.execute(
        """
        CREATE OR REPLACE TABLE focus AS
        SELECT gene_id, disease_cui, pmid, semmed_gene, semmed_article_record
        FROM src.core_only_state
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE acc_core AS
        SELECT DISTINCT pmid, accepted_gene_id AS gene_id
        FROM src.mapped
        WHERE population = 'core_anchored' AND mapping_status = 'accepted' AND accepted_gene_id IS NOT NULL
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE acc_scope AS
        SELECT DISTINCT pmid, accepted_gene_id AS gene_id
        FROM src.mapped
        WHERE population = 'scope_e' AND mapping_status = 'accepted' AND accepted_gene_id IS NOT NULL
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE pmid_core AS
        SELECT DISTINCT pmid FROM src.mapped WHERE population = 'core_anchored'
        """
    )
    candidate_ok = True
    try:
        con.execute(
            """
            CREATE OR REPLACE TABLE cand_core AS
            SELECT DISTINCT pmid, gene_id
            FROM (
              SELECT pmid, unnest(from_json(candidate_gene_ids, '["VARCHAR"]')) AS gene_id
              FROM src.mapped
              WHERE population = 'core_anchored'
                AND candidate_gene_ids IS NOT NULL
                AND candidate_gene_ids NOT IN ('[]', 'null')
            )
            WHERE gene_id IS NOT NULL
            """
        )
    except Exception as exc:
        candidate_ok = False
        logger.info("phase=candidate_parse_failed %s", exc)
    if candidate_ok:
        state_sql = """
            CASE
              WHEN a.gene_id IS NOT NULL THEN 'accepted_gene_id'
              WHEN c.gene_id IS NOT NULL THEN 'candidate_only'
              WHEN p.pmid IS NOT NULL THEN 'pmid_in_core_anchored_input_gene_not_in_argument'
              ELSE 'pmid_not_in_core_anchored_input'
            END
        """
        cand_join = "LEFT JOIN cand_core c ON f.pmid = c.pmid AND f.gene_id = c.gene_id"
    else:
        state_sql = """
            CASE
              WHEN a.gene_id IS NOT NULL THEN 'accepted_gene_id'
              WHEN p.pmid IS NOT NULL THEN 'pmid_in_core_anchored_input_candidate_not_parsed'
              ELSE 'pmid_not_in_core_anchored_input'
            END
        """
        cand_join = ""
    con.execute(
        f"""
        CREATE OR REPLACE TABLE focus_flags AS
        SELECT f.gene_id, f.disease_cui, f.pmid, f.semmed_gene, f.semmed_article_record,
               {state_sql} AS core_anchored_argument,
               CASE WHEN s.gene_id IS NOT NULL THEN 'accepted_gene_id' ELSE 'not_accepted_in_scope_e' END AS scope_e_argument
        FROM focus f
        LEFT JOIN acc_core a ON f.pmid = a.pmid AND f.gene_id = a.gene_id
        {cand_join}
        LEFT JOIN pmid_core p ON f.pmid = p.pmid
        LEFT JOIN acc_scope s ON f.pmid = s.pmid AND f.gene_id = s.gene_id
        """
    )
    cross = rows(
        con,
        """
        SELECT semmed_gene, core_anchored_argument, count(*) AS n
        FROM focus_flags
        GROUP BY 1, 2
        ORDER BY 1, 3 DESC
        """,
    )
    scope_cross = rows(
        con,
        """
        SELECT semmed_gene, scope_e_argument, count(*) AS n
        FROM focus_flags
        GROUP BY 1, 2
        ORDER BY 1, 3 DESC
        """,
    )
    n_focus = one(con, "SELECT count(*) FROM focus")
    n_flags = one(con, "SELECT count(*) FROM focus_flags")
    check(checks, "observable_grain", n_focus == n_flags == 303941, {"focus": n_focus, "flags": n_flags})
    return {
        "candidate_parsed": candidate_ok,
        "candidate_probe": probe,
        "n_core_only_pair_pmid": n_focus,
        "entity_by_core_anchored_argument": cross,
        "entity_by_scope_e_argument": scope_cross,
        "entity_query": "core_only_state.semmed_gene. Citation must exist, then ENTITY.gene_id is sought on sentences of that PMID. sentence_id is the J03 candidate column pre_cui_numeric_1. absent_in_available_records means that join missed. It does not mean the gene is absent from SemMedDB.",
        "argument_query": "core_anchored rows of mapped.parquet for the same PMID. accepted_gene_id is the normalized identifier. candidate_only means the parsed candidate list contains the gene and the row was not accepted as that gene.",
    }


def render(provenance: dict, payload: dict) -> str:
    funnel = payload["funnel"]
    scope = payload["scope"]
    sens = payload["sensitivity"]
    obs = payload["observable"]
    rules = payload["rules"]
    failed = [item for item in payload["checks"] if not item["ok"]]
    lines = [
        "# id_alignment_v2 结果核查",
        "",
        "这份报告读的是已保存的 v2 parquet 和 V22 的 DuckDB 表，没有重扫 ENTITY，也没有重写 `id_alignment_v2_round1/report.md`。重叠、五类关系和 observable 的原计数没有重算；本轮新增的是漏斗、筛选转移、年份原因和参数存在性。",
        "",
        "## 主要发现及适用范围",
        "",
        f"主结果仍是 v2 中 `mapping_status = accepted`、且 `(accepted_gene_id, disease_cui)` 等于 CORE 显式 pair 的记录。共 {funnel['pairs']} 个 pair、{funnel['pair_pmid']} 条 pair–PMID、{funnel['pmids']} 个去重 PMID。历史 id_rule_v1 的 276 个 pair 留在原目录。",
        "",
        "适用范围是 CORE 显式 pair 与 SemMedDB 谓词参数的标识对齐。范围 E 是 gngm–neop 这个操作性子集，不代替主结果。这里不做文献质量、科学支持或排序。",
        "",
        f"存储的 `joint_v2.json` 把比例字段写成了 `id_rule_v1`，范围标签写成了 `E_gngm_neop_with_same_sentence_numeric_gene_id`。V23 核对过行数：pair–PMID {provenance['counts']['semmed_pair_pmid_rows']} 与 joint 的 SemMedDB 侧一致，scope E pair {provenance['counts']['scope_e_pairs']} 与 joint 的 `n_pairs` 一致。结论是元数据标签过期，计算用的是 v2 表。`lib/joint_build.py` 此后把调用者传入的规则名写入比例字段；已经落盘的 JSON 没有改。",
        "",
        "## 输入与映射规则",
        "",
        "规则版本 `id_alignment_v2`。阈值在 `rules.py` 的 `THRESHOLDS`，并写入 `derived/rules_v2.json`。启用前固定，不按结果回调。",
        "",
        "| 阈值 | 值 | 用途 |",
        "| --- | --- | --- |",
        "| ids_per_quartile | 100 | 频率四分位内的样本量 |",
        "| min_presence_current_or_history | 0.90 | 路径 A 启用：样本 ID 在 gene_info 或 gene_history |",
        "| min_current_name_agreement | 0.70 | 路径 A 启用：当前 ID 的名称精确词一致 |",
        "| max_conflict_when_entity_gene_present | 0.05 | 路径 C 启用：CUI 对应实体基因与数字冲突的比例 |",
        "| min_rows_for_conflict_rate | 50 | 冲突比例至少需要的可比行数 |",
        "| coordinate_adopt_match / coordinate_reject_alternative | 0.90 / 0.50 | 坐标规则，不改写基因 ID |",
        "",
        "NCBI 的存在性和名称核对只在分层样本上做。样本是 gngm 或 aapp 参数里的纯数字，以及一条竖线形式，每个频率四分位最多 100 个不同参数。全表 accepted 行没有再逐条查 gene_info 或名称。历史 ID 计入样本存在性，替代 ID 只记在样本上，不写入 `accepted_gene_id`。",
        "",
        "逐条接受条件在 `rules.decide`，V21 对 `mapped.parquet` 的每一行调用了它：",
        "",
        "- 路径 A：参数是一个数字，语义类型含 gngm 或 aapp。接受这个数字。没有实体行、或其他实体基因对不上，都只改变 `entity_check`，不取消接受。路径 A 不做冲突排除。",
        "- 路径 B：参数是一个 CUI。泛化概念不取值。该 CUI 在候选句子上没有实体行、基因字段为空、或多个数字基因 ID，分别保持原状态。恰好一个数字基因 ID 才接受。",
        "- 路径 C：一个 CUI 加一个数字，或重复同一数字。多个不同数字保持候选集，状态为 ambiguous，不取第一个。CUI 所连实体基因若与该数字不同，状态为 conflict，`accepted_gene_id` 为空。没有实体基因时仍接受该数字。",
        "",
        "下面的计数来自已保存的 `mapped.parquet`，用来核对上述规则是否写进了结果，不是第二次实体扫描，也不是人工准确率。",
        "",
        _md([rules["summary"]]),
        "",
        "已接受行的 `entity_check`：",
        "",
        _md(rules["accepted_entity_check"]),
        "",
        "名称不一致的样本（精确词，不是子串；不一致不等于这条映射被判错，生产映射也没有因名称把它排除）：",
        "",
        _md(provenance["samples"]["name_disagreements"]),
        "",
        "只在 gene_history 中的样本。替代 ID 没有替换原 ID：",
        "",
        _md(provenance["samples"]["history_only"]),
        "",
        "冲突样本是路径 C 的 200 个分层参数里、能在候选句子上找到该 CUI 的实体基因、且基因 ID 与数字不同的参数。处置是不接受。",
        "",
        _md(provenance["conflicts"]),
        "",
        "全表 conflict 行的高频参数：",
        "",
        _md(rules["conflict_examples"]),
        "",
        "## 映射漏斗",
        "",
        "core_anchored 的进入条件：谓词的 `subject_cui` 或 `object_cui` 等于某个 CORE `disease_cui`，另一侧作为基因参数；`pmid` 非空；两侧 CUI 不相等。一行的键是 `population, predication_id, disease_cui, gene_side`。同一个谓词可以因两个不同的 CORE 疾病各占一行。",
        "",
        f"core_anchored 共 {funnel['core_anchored_rows']} 行。其中 `mapping_status = accepted` 的是 {funnel['accepted_arguments']} 行。这是 SemMedDB 参数被接受为一个基因 ID，还没有要求该基因和该疾病构成 CORE pair。",
        "",
        f"这 {funnel['accepted_arguments']} 行里，{funnel['accepted_on_core_pair']} 行的 `(accepted_gene_id, disease_cui)` 能对上 CORE 显式 pair，并写入 `tables/accepted_match.parquet`。其余 {funnel['accepted_gene_not_core_pair']} 行被排除，原因只有一个：接受的基因 ID 与这个疾病 CUI 不是 CORE 显式 pair。两边单位都是 core_anchored 参数行，所以可以相减。",
        "",
        f"从 {funnel['accepted_on_core_pair']} 行往下不再删除记录。按 `(gene_id, disease_cui)` 聚合得 {funnel['pairs']} 个 pair；按 `(gene_id, disease_cui, pmid)` 聚合得 {funnel['pair_pmid']} 条 pair–PMID；再按 `pmid` 去重得 {funnel['pmids']} 个 PMID。一个 PMID 可以对应多个 pair，所以 137221 与 113380 不能相减后当成排除量。",
        "",
        "状态计数（参数行）：",
        "",
        _md(funnel["status"]),
        "",
        "accepted 参数按路径，以及其中对上 CORE pair 的行数：",
        "",
        _md(funnel["routes"]),
        "",
        "字段位置：`tables/mapped.parquet` 的 `population`、`mapping_status`、`accepted_gene_id`、`disease_cui`；`tables/accepted_match.parquet`；`tables/semmed_pair_pmid.parquet` 的 `gene_id`、`disease_cui`、`pmid`。",
        "",
        "## 范围 E 与主结果",
        "",
        f"范围 E 的接受 pair 有 {scope['n_scope_e_pairs']} 个，键是 `accepted_gene_id, disease_cui`。其中 {scope['n_scope_e_in_core_pairs']} 个也是 CORE 显式 pair，{scope['n_scope_e_not_in_core_pairs']} 个不是。后者只存在于 gngm–neop 且映射被接受的子集，文件是 `tables/scope_e_gene_pairs.parquet` 对 `pair.parquet` 的反连接。",
        "",
        f"主结果的 {scope['n_main_pairs']} 个 pair 包含全部 {scope['n_scope_e_in_core_pairs']} 个范围 E 共享 pair，另外 {scope['n_main_not_in_scope_e_shared']} 个只出现在主结果中。这 {scope['n_main_not_in_scope_e_shared']} 个 pair 里，accepted_match 带有精确 gngm–neop 语义类型的有 {scope['main_only_with_gngm_neop_row']} 个，其余来自其他语义类型。范围 E 共享 pair 中没有进入主结果的有 {scope['n_scope_e_shared_not_in_main']} 个。",
        "",
        "主结果的入口是“疾病侧 CUI 等于某个 CORE 疾病”，范围 E 的入口是“两侧语义类型恰好为 gngm 与 neop”。两边都接受之后才按基因 ID 和疾病 CUI 比较。",
        "",
        "## 五类 PMID 集合",
        "",
        "分母是全部 153016 个 CORE 显式 pair。五类要求两侧 PMID 集合都非空：identical、core_subset、semmed_subset、partial_overlap、disjoint。一侧为空时记为 semmed_pmids_empty 或 core_pmids_empty，两侧都空记为 both_empty。未筛选时 both_empty 为 "
        + str(sens["unfiltered_both_empty"])
        + "。五类 pair 之和就是两侧都有 PMID 的 30910 个共享 pair。",
        "",
        _md(sens["class_counts"]),
        "",
        "两种单位：",
        "",
        f"- pair–PMID：交集 {sens['pair_pmid_intersection']}。同一 PMID 出现在两个 pair 里会计两次。CORE 侧 pair–PMID 与 SemMedDB 侧 pair–PMID 的独有量在 `joint_v2.json` 的 `pair_pmid_micro_all_explicit_pairs`，本轮对交集做了复核。",
        f"- 去重 PMID：CORE {sens['deduplicated_pmid']['n_core']}，SemMedDB {sens['deduplicated_pmid']['n_semmed']}，交集 {sens['deduplicated_pmid']['n_intersection']}。一个 PMID 只计一次，不管它连接多少 pair。",
        "",
        "文章覆盖看的是 PMID 是否有文章或句子记录。关联覆盖看的是这个基因–疾病 pair 的 PMID 集合是否相交。去重 PMID 交集 31412 大于 pair–PMID 交集，因为有的 PMID 在两侧出现，但带着的 pair 不同。",
        "",
        "## 共同文章与共同年份",
        "",
        "共同文章是 `core_sentence` 与 `citations` 都有记录的 PMID，定义在 `joint_build._sensitivity` 的 `common_article` 表。它不要求目标 pair 已经匹配。"
        f"两侧 pair–PMID 表都出现过的 PMID 有 {sens['deduplicated_pmid']['n_intersection']} 个，其中 "
        + str(sens["article"]["pair_intersection_pmid_not_in_common_article"])
        + " 个不在这个文章集合里。文章集合另外有 "
        + str(sens["article"]["common_article_pmid_not_in_v2_match"])
        + f" 个 PMID 不在 v2 匹配中。文章集合大小是 {sens['article']['n_common_article_pmid']}。因此它是文章记录的交集，不是匹配 pair 的 PMID 集合。",
        "",
        "年份窗口是两侧都至少出现过一次的、状态为 parsed_single 的年份集合，不是逐篇 PMID 先求交。CORE 年用 `publicationYear` 的整数解析：空、非整数、非正数为未解析，同一 PMID 多个不同正整数为 multi_value。SemMedDB 用 citations 的 DP：前四位为单年则 parsed_single；PYEAR 是完成日期，不作为出版年。窗口有 "
        + str(sens["year_window"]["n"])
        + f" 个年份，从 {sens['year_window']['min']} 到 {sens['year_window']['max']}，相邻缺口 {len(sens['year_window']['gaps'])} 处。完整列表在 `revision/derived/revision_numbers.json` 的 `sensitivity.year_window.years`。",
        "",
        "一篇保留下来的 pair–PMID 必须：本侧年份是 parsed_single 且落在窗口内；并且对侧年份为空、与本侧相同，或对侧不是 parsed_single。两侧都是 parsed_single 且年份不同，则两侧都删除。",
        "",
        "互斥归类顺序是：年份表没有该 PMID；本侧不是 parsed_single；本侧年份不在窗口；两侧都已解析且不一致。一行只进入第一个成立的类。PMID 不在年份表时，本侧状态为空，交叉表里的未解析标志也会是真；互斥类仍记为年份表没有该 PMID。窗外和不一致可以同时为真，互斥表里它记在窗外。",
        "",
        "CORE pair–PMID 删除原因：",
        "",
        _md(sens["year_reasons"]["core"]["exclusive"]),
        "",
        "其中本侧未解析的状态：",
        "",
        _md(sens["year_reasons"]["core"]["own_status_when_unparsed"]),
        "",
        "CORE 删除行的原因交叉：",
        "",
        _md(sens["year_reasons"]["core"]["flag_cross"]),
        "",
        "SemMedDB pair–PMID 删除原因：",
        "",
        _md(sens["year_reasons"]["semmed"]["exclusive"]),
        "",
        "其中本侧未解析的状态：",
        "",
        _md(sens["year_reasons"]["semmed"]["own_status_when_unparsed"]),
        "",
        "SemMedDB 删除行的原因交叉：",
        "",
        _md(sens["year_reasons"]["semmed"]["flag_cross"]),
        "",
        "下面的转移只含筛选前两侧都非空的 30910 个 pair。筛选前状态都是 both_nonempty。",
        "",
        "年份筛选后：",
        "",
        _md(sens["transitions"]["year"]["states"]),
        "",
        _jaccard_line("年份", sens["transitions"]["year"]["fixed_both_nonempty"]),
        "",
        "共同文章筛选后：",
        "",
        _md(sens["transitions"]["common_article"]["states"]),
        "",
        _jaccard_line("共同文章", sens["transitions"]["common_article"]["fixed_both_nonempty"]),
        "",
        "Jaccard 的分子是该固定 pair 集合的 pair–PMID 交集之和，分母是并集之和。平均 Jaccard 是同一批 pair 上逐 pair Jaccard 的算术平均。筛选前的对照使用这批 pair 在筛选前的 PMID，不是另外一批 pair。",
        "",
        "疾病 CUI 排除使用 CORE SPARQL 示例 5 的四个 CUI："
        + ", ".join(CUIS)
        + "。原表把这些 pair 标成 both_empty，是因为过滤后两侧 PMID 都变成 0，不是因为它们本来没有 PMID。未筛选的 both_empty 为 "
        + str(sens["unfiltered_both_empty"])
        + f"。本轮把这 {sens['cui']['excluded_labeled_both_empty']} 个 pair 标为 excluded。落在排除名单之外、又被标成 both_empty 的 pair 有 {sens['cui']['true_both_empty']} 个；排除名单之内、类别不是 both_empty 的有 {sens['cui']['excluded_other_class']} 个。五类计数因此不变，变的是这一类的名称。",
        "",
        _md(sens["exclude_counts"]),
        "",
        "## 差异诊断里的基因存在性",
        "",
        "CORE 独有 pair–PMID 有 "
        + str(obs["n_core_only_pair_pmid"])
        + " 条，沿用 `tables/core_only_state.parquet`。",
        "",
        "ENTITY 列 `semmed_gene` 的查询是：该 PMID 先要有 SemMedDB 引文记录，再在该 PMID 的句子上找 `ENTITY.gene_id`。句子编号用的是 J03 的候选列 `pre_cui_numeric_1`。`absent_in_available_records` 表示这次连接没有命中。它不表示 SemMedDB 全库没有这个基因。引文不存在时，这一列是 `not_assessable`。",
        "",
        "参数列看的是同一 PMID 在 `mapped.parquet` 的 core_anchored 行。`accepted_gene_id` 是规则接受的基因 ID。`candidate_only` 表示解析出的候选里有这个基因，但该行没有把它接受为这个基因。这个范围只包括一侧 CUI 等于某个 CORE 疾病的谓词，不覆盖其余谓词。",
        "",
        "ENTITY 状态与 core_anchored 参数状态：",
        "",
        _md(obs["entity_by_core_anchored_argument"]),
        "",
        "同一批 pair–PMID 在范围 E 已接受参数中的状态。`not_accepted_in_scope_e` 包含不在该子集里，以及在子集里但没有被接受为这个基因。",
        "",
        _md(obs["entity_by_scope_e_argument"]),
        "",
        "## 已解决和仍无法确定",
        "",
        "已核对：v2 重叠计数与 v2 pair–PMID 表一致；旧规则名和范围 E 旧标签没有参与选表；accepted 行没有带 conflict 标记；conflict 与 ambiguous 没有写入 accepted_gene_id；年份保留表可以用书面条件重建；CUI 排除与本来无数据的 both_empty 可以分开。",
        "",
        "仍无法确定：路径 A 的 NCBI 名称和存在性没有逐条用于全部 accepted 记录，样本一致率不是全库准确率。坐标规则没有用来改写 ID。候选句子列仍是 J03 的 `pre_cui_numeric_1`，第 2 列对照不能证明它在全库都对。ENTITY 未命中的查询范围写在上一节，不能外推成全库没有该基因。多个不同数字、泛化概念和 conflict 仍不进入主结果。",
        "",
        "守恒检查失败数：" + str(len(failed)) + "。明细在 `revision/derived/revision_numbers.json` 的 `checks`。",
        "",
    ]
    if failed:
        lines.extend(["未通过的检查：", "", _md(failed), ""])
    return "\n".join(lines)


def _jaccard_line(label: str, fixed: dict) -> str:
    return (
        f"{label}筛选后仍两侧都非空的 pair 有 {fixed['n_pair']} 个。"
        f"这同一集合的 pair–PMID Jaccard，筛选前是 {fixed['inter_before']}/{fixed['union_before']}，"
        f"筛选后是 {fixed['inter_after']}/{fixed['union_after']}。"
        f"逐 pair 平均 Jaccard 筛选前是 {fixed['mean_jaccard_before']}，筛选后是 {fixed['mean_jaccard_after']}。"
    )


def _md(records: list[dict]) -> str:
    if not records:
        return "（无行）"
    keys = list(records[0].keys())
    head = "| " + " | ".join(keys) + " |"
    sep = "| " + " | ".join("---" for _ in keys) + " |"
    body = [
        "| " + " | ".join("" if row.get(key) is None else str(row.get(key)).replace("|", "\\|").replace("\n", " ") for key in keys) + " |"
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

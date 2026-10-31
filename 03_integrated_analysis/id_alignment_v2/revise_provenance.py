#!/usr/bin/env python3
"""V23: trace v2 inputs and decide whether stored labels describe the computation."""

from __future__ import annotations

import json
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
from rules import RULE_VERSION, THRESHOLDS, parse_argument  # noqa: E402

RESULTS = Path("/projects/b5ac/project_2/results")
ROUND = RESULTS / "03_integrated_analysis" / "id_alignment_v2_round1"
REV = ROUND / "revision"
TABLES = ROUND / "tables"
V22_DB = ROUND / "derived" / "v22.duckdb"
V20_DB = ROUND / "derived" / "v20.duckdb"
SAMPLES = ROUND / "cases" / "identifier_samples.json"
RULES = ROUND / "derived" / "rules_v2.json"
JOINT = ROUND / "derived" / "joint_v2.json"
V21 = ROUND / "derived" / "v21_summary.json"
J03 = RESULTS / "03_integrated_analysis" / "data_analysis_round1" / "derived" / "j03" / "summary.json"


def main() -> int:
    assert_compute_node()
    REV.mkdir(parents=True, exist_ok=True)
    (REV / "derived").mkdir(parents=True, exist_ok=True)
    logger = setup_logging(REV / "logs" / f"v23_{os.environ.get('SLURM_JOB_ID', 'local')}.log")
    info = prologue("V23", logger, REV, [str(RULES), str(JOINT), str(V21)])
    rules = read_json(RULES)
    joint = read_json(JOINT)
    v21 = read_json(V21)
    j03 = read_json(J03)
    con = connect(REV / "derived" / "v23.duckdb", "72GB", REV / "tmp")
    counts = parquet_counts(con)
    logger.info("phase=parquet_counts %s", counts)
    label = label_verdict(joint, counts)
    logger.info("phase=label_verdict %s", label["verdict"])
    samples = sample_cases()
    conflicts = conflict_cases(logger)
    payload = {
        "code_fingerprint": info["code_fingerprint"],
        "rule_version": rules.get("rule_version"),
        "expected_rule_version": RULE_VERSION,
        "thresholds": rules.get("thresholds"),
        "thresholds_match_code": rules.get("thresholds") == THRESHOLDS,
        "enabled": rules.get("enabled"),
        "ncbi": {
            "available": (rules.get("ncbi") or {}).get("available"),
            "info_sha256": (rules.get("ncbi") or {}).get("info_sha256"),
            "history_sha256": (rules.get("ncbi") or {}).get("history_sha256"),
            "n_current": (rules.get("ncbi") or {}).get("n_current"),
            "n_discontinued": (rules.get("ncbi") or {}).get("n_discontinued"),
            "replacement_policy": (rules.get("ncbi") or {}).get("replacement_policy"),
        },
        "entity_sentence_column": (j03.get("entity_sentence_id_resolution") or {}).get("column"),
        "sparql_example_excluded_cuis": list(SPARQL_EXAMPLE_EXCLUDED_CUIS),
        "inputs": input_catalog(),
        "counts": counts,
        "published": {
            "v21_accepted_core_predications": v21["counts"]["accepted_core_predications"],
            "v21_accepted_core_pairs": v21["counts"]["accepted_core_pairs"],
            "v21_accepted_core_pair_pmid": v21["counts"]["accepted_core_pair_pmid"],
            "v21_scope_e_accepted_pairs": v21["counts"]["scope_e_accepted_pairs"],
            "joint_semmed_pair_pmid": joint["populations"]["all_available"]["pair_pmid_micro_all_explicit_pairs"]["n_semmed"],
            "joint_dedup_semmed_pmid": joint["populations"]["all_available"]["deduplicated_pmid"]["n_semmed"],
            "joint_scope_e_pairs": joint["scope_e_only_pairs"]["n_pairs"],
            "joint_scope_e_shared": joint["scope_e_only_pairs"]["n_shared_gene_disease_id"],
            "joint_rule_field": joint["populations"]["all_available"]["pair_pmid_micro_all_explicit_pairs"]["jaccard"]["rule"],
            "joint_scope_label": joint["scope_e_only_pairs"]["scope"],
        },
        "label_audit": label,
        "samples": samples,
        "conflicts": conflicts,
        "acceptance_scope": {
            "ncbi_name_and_presence": "Stratified sample of genomic arguments only. Not applied to each accepted row.",
            "historical_id": "Presence includes gene_history. The replacement id is stored on the sample row and is not written into accepted_gene_id.",
            "conflict_exclusion": "Path C compares the single digit with ENTITY gene ids linked to the CUI on the candidate sentence. Path A does not. Path B accepts only when that CUI has exactly one numeric gene id.",
        },
    }
    mismatches = consistency_errors(payload)
    payload["consistency_errors"] = mismatches
    write_json(REV / "derived" / "provenance.json", payload)
    status = "success" if not mismatches and label["verdict"] == "label_stale_computation_is_v2" else "failed"
    write_json(REV / "derived" / "v23_DONE.json", {"status": status, "job": "V23", "provenance": str(REV / "derived" / "provenance.json")})
    logger.info("phase=v23_%s errors=%s", status, mismatches)
    return 0 if status == "success" else 1


def input_catalog() -> list[dict]:
    return [
        {
            "result": "core_anchored arguments and scope E arguments",
            "path": str(TABLES / "mapped.parquet"),
            "rule": "id_alignment_v2",
            "entry": "core_anchored: subject_cui or object_cui equals a CORE disease_cui, the other side is the gene argument, pmid is not null, and the two CUIs differ. scope_e: gngm-neop predications from scope_e_gngm_neop.parquet, gene side taken from gene_position.",
            "dedup_key": "One row per population, predication_id, disease_cui and gene_side. Not deduplicated further in mapped.",
        },
        {
            "result": "accepted CORE match",
            "path": str(TABLES / "accepted_match.parquet"),
            "rule": "id_alignment_v2",
            "entry": "mapped population=core_anchored, mapping_status=accepted, and (accepted_gene_id, disease_cui) equals a CORE explicit pair.",
            "dedup_key": "predication_id, disease_cui, gene_side, after the pair join. Pair key is gene_id, disease_cui.",
        },
        {
            "result": "v2 pair–PMID",
            "path": str(TABLES / "semmed_pair_pmid.parquet"),
            "rule": "id_alignment_v2",
            "entry": "accepted_match grouped where pmid is not null.",
            "dedup_key": "gene_id, disease_cui, pmid.",
        },
        {
            "result": "overlap, sensitivity, observable",
            "path": str(V22_DB),
            "rule": "Counts were produced by run_joint on the v2 semmed_pair_pmid view. The ratio.rule string in joint_v2.json was hardcoded at the time of that run.",
            "entry": "Universe is every CORE explicit pair. Five classes require both PMID sets to be non-empty.",
            "dedup_key": "Pair relations use gene_id, disease_cui. Micro coverage sums pair–PMID counts. Deduplicated PMID counts a PMID once.",
        },
        {
            "result": "scope E accepted pairs",
            "path": str(TABLES / "scope_e_gene_pairs.parquet"),
            "rule": "id_alignment_v2",
            "entry": "mapped population=scope_e and mapping_status=accepted. Not restricted to CORE diseases.",
            "dedup_key": "accepted_gene_id, disease_cui.",
        },
    ]


def parquet_counts(con) -> dict:
    def n(path: Path, where: str = "TRUE") -> int:
        return con.execute(f"SELECT count(*) FROM read_parquet('{path.as_posix()}') WHERE {where}").fetchone()[0]

    mapped = TABLES / "mapped.parquet"
    accepted = TABLES / "accepted_match.parquet"
    sem = TABLES / "semmed_pair_pmid.parquet"
    scope = TABLES / "scope_e_gene_pairs.parquet"
    pair = RESULTS / "01_core_audit" / "tables" / "pair.parquet"
    core_pp = RESULTS / "01_core_audit" / "tables" / "pair_pmid.parquet"
    return {
        "mapped_rows": n(mapped),
        "core_anchored_rows": n(mapped, "population = 'core_anchored'"),
        "core_anchored_accepted_rows": n(mapped, "population = 'core_anchored' AND mapping_status = 'accepted'"),
        "scope_e_rows": n(mapped, "population = 'scope_e'"),
        "scope_e_accepted_rows": n(mapped, "population = 'scope_e' AND mapping_status = 'accepted'"),
        "accepted_match_rows": n(accepted),
        "accepted_match_distinct_argument": con.execute(
            f"""
            SELECT count(*) FROM (
              SELECT DISTINCT predication_id, disease_cui, gene_side
              FROM read_parquet('{accepted.as_posix()}')
            )
            """
        ).fetchone()[0],
        "semmed_pair_pmid_rows": n(sem),
        "semmed_pairs": con.execute(
            f"SELECT count(*) FROM (SELECT DISTINCT gene_id, disease_cui FROM read_parquet('{sem.as_posix()}'))"
        ).fetchone()[0],
        "semmed_pmids": con.execute(
            f"SELECT count(*) FROM (SELECT DISTINCT pmid FROM read_parquet('{sem.as_posix()}'))"
        ).fetchone()[0],
        "scope_e_pairs": n(scope),
        "core_pairs": n(pair),
        "core_pairs_distinct": con.execute(
            f"SELECT count(*) FROM (SELECT DISTINCT gene_id, disease_cui FROM read_parquet('{pair.as_posix()}'))"
        ).fetchone()[0],
        "core_pair_pmid_rows": n(core_pp),
    }


def label_verdict(joint: dict, counts: dict) -> dict:
    micro = joint["populations"]["all_available"]["pair_pmid_micro_all_explicit_pairs"]
    scope = joint["scope_e_only_pairs"]
    rule_field = micro["jaccard"]["rule"]
    count_match = (
        micro["n_semmed"] == counts["semmed_pair_pmid_rows"]
        and joint["populations"]["all_available"]["deduplicated_pmid"]["n_semmed"] == counts["semmed_pmids"]
        and scope["n_pairs"] == counts["scope_e_pairs"]
    )
    if count_match and rule_field == "id_rule_v1":
        verdict = "label_stale_computation_is_v2"
    elif count_match and rule_field == "id_alignment_v2":
        verdict = "label_and_computation_are_v2"
    else:
        verdict = "computation_not_v2"
    return {
        "verdict": verdict,
        "stored_rule_field": rule_field,
        "stored_scope_label": scope["scope"],
        "scope_count_matches_v2_pairs": scope["n_pairs"] == counts["scope_e_pairs"],
        "overlap_count_matches_v2_pair_pmid": micro["n_semmed"] == counts["semmed_pair_pmid_rows"],
        "caller": "summarize_v2.rerun_joint passes rule id_alignment_v2 and points semmed_pair_pmid and scope_e_gene_pairs at the v2 parquet files.",
        "what_was_hardcoded": "lib/joint_build._population_summary wrote id_rule_v1 into every ratio at the time V22 ran. _scope_e_only wrote E_gngm_neop_with_same_sentence_numeric_gene_id. Neither string selected the input table.",
        "scope_computation": "COUNT of scope_e_gene_pairs, which is the distinct accepted_gene_id, disease_cui from scope_e rows with mapping_status=accepted. The old same-sentence numeric-gene predicate is not reapplied.",
    }


def sample_cases() -> dict:
    raw = json.loads(SAMPLES.read_text(encoding="utf-8"))
    rows = list(raw.get("single_digits") or []) + list(raw.get("pipe") or [])
    disagree = [
        {
            "kind": row.get("kind"),
            "argument": row.get("argument"),
            "argument_format": row.get("argument_format"),
            "example_name": row.get("example_name"),
            "symbol": row.get("symbol"),
            "n": row.get("n"),
            "in_current_gene_info": row.get("in_current_gene_info"),
            "history_only": row.get("history_only"),
        }
        for row in rows
        if row.get("in_current_gene_info") and row.get("example_name") and not row.get("name_agrees")
    ]
    history = [
        {
            "kind": row.get("kind"),
            "argument": row.get("argument"),
            "example_name": row.get("example_name"),
            "replacement_gene_id": row.get("replacement_gene_id"),
            "n": row.get("n"),
        }
        for row in rows
        if row.get("history_only")
    ]
    return {"name_disagreements": disagree, "history_only": history, "conflict_summary": (raw.get("conflicts") or {}).get("summary")}


def conflict_cases(logger) -> list[dict]:
    import duckdb

    con = duckdb.connect(str(V20_DB), read_only=True)
    fetched = con.execute(
        """
        SELECT o.argument, o.predication_id, o.sentence_id, e.cui, e.gene_id
        FROM conflict_one o
        LEFT JOIN conflict_entity e
          ON o.sentence_id = e.sentence_id
         AND e.gene_id IS NOT NULL
        """
    ).fetchall()
    grouped: dict[str, dict] = {}
    for argument, predication_id, sentence_id, cui, gene_id in fetched:
        parsed = parse_argument(argument)
        item = grouped.setdefault(
            argument,
            {
                "argument": argument,
                "predication_id": predication_id,
                "sentence_id": sentence_id,
                "cui": parsed.get("cui"),
                "digit": parsed["candidate_gene_ids"][0] if len(parsed["candidate_gene_ids"]) == 1 else None,
                "genes": [],
            },
        )
        if gene_id and cui == parsed.get("cui") and gene_id not in item["genes"]:
            item["genes"].append(gene_id)
    out = []
    for item in grouped.values():
        digit = item["digit"]
        linked = item["genes"]
        if not digit or not any(gene != digit for gene in linked):
            continue
        out.append(
            {
                "argument": item["argument"],
                "predication_id": item["predication_id"],
                "sentence_id": item["sentence_id"],
                "candidate": digit,
                "entity_gene_ids": linked,
                "disposition": "Sample conflict. Path C leaves accepted_gene_id empty when this pattern is applied to a row.",
            }
        )
    logger.info("phase=conflict_cases n=%s", len(out))
    con.close()
    return out


def consistency_errors(payload: dict) -> list[str]:
    errors = []
    counts = payload["counts"]
    published = payload["published"]
    if payload["rule_version"] != RULE_VERSION:
        errors.append("rules_v2.json rule_version is not id_alignment_v2")
    if not payload["thresholds_match_code"]:
        errors.append("rules_v2.json thresholds differ from rules.THRESHOLDS")
    if counts["core_pairs"] != counts["core_pairs_distinct"]:
        errors.append("CORE pair grain is not unique on gene_id, disease_cui")
    if counts["accepted_match_rows"] != counts["accepted_match_distinct_argument"]:
        errors.append("accepted_match multiplies argument rows")
    pairs = [
        (counts["core_anchored_accepted_rows"], 513477, "core accepted arguments"),
        (counts["accepted_match_rows"], published["v21_accepted_core_predications"], "accepted_match vs v21"),
        (counts["semmed_pairs"], published["v21_accepted_core_pairs"], "v2 pairs"),
        (counts["semmed_pair_pmid_rows"], published["v21_accepted_core_pair_pmid"], "v2 pair-pmid"),
        (counts["semmed_pmids"], published["joint_dedup_semmed_pmid"], "deduplicated pmid"),
        (counts["scope_e_pairs"], published["v21_scope_e_accepted_pairs"], "scope E pairs"),
        (published["joint_semmed_pair_pmid"], counts["semmed_pair_pmid_rows"], "joint micro vs parquet"),
        (published["joint_scope_e_pairs"], counts["scope_e_pairs"], "joint scope count vs parquet"),
    ]
    for got, expect, name in pairs:
        if got != expect:
            errors.append(f"{name}: {got} != {expect}")
    if payload["label_audit"]["verdict"] != "label_stale_computation_is_v2":
        errors.append("label verdict is " + payload["label_audit"]["verdict"])
    return errors


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        raise SystemExit(1)

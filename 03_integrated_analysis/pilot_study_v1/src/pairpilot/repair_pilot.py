#!/usr/bin/env python3
"""Revision of child run 20261015T221200Z_b20cont. Prior responses are read, not rewritten."""

from __future__ import annotations

import json
import os
import sys
import traceback
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pairpilot.comparison import (  # noqa: E402
    build_candidates,
    cluster_candidates,
    gap_from_comparison,
    is_analysis_finding,
    within_document_pairs,
)
from pairpilot.config import digest, load_bundle  # noqa: E402
from pairpilot.experiments.e1 import condition_key, equal_article, source_metrics  # noqa: E402
from pairpilot.experiments.e5 import allocate_candidates  # noqa: E402
from pairpilot.experiments.e6 import align_replicates, structured_key  # noqa: E402
from pairpilot.ledger import LIMITS, SharedLedger, available_usd  # noqa: E402
from pairpilot.sampling import rank  # noqa: E402
from pairpilot.store import query_rows, read_json, write_json, write_parquet  # noqa: E402

PRIOR = Path("/projects/b5ac/project_2/results/03_integrated_analysis/pilot_study_v1/runs/20261015T221200Z_b20cont")
PREP = Path("/projects/b5ac/project_2/results/03_integrated_analysis/pilot_study_v1/runs/20261015T204633Z_0d9b9724")
RUNS = PRIOR.parent
RULE_VERSION = "comparison_gap_v2"
E6_PAIRS = {"3604__C0007131", "3090__C0006142"}
EXPECTED_M = {
    "8318__C2239176": 3,
    "3090__C0006142": 4,
    "3958__C0007137": 2,
    "1401__C0242379": 2,
    "84868__C0026764": 3,
    "768__C0376358": 5,
    "3604__C0007131": 2,
    "2625__C3539878": 4,
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _login() -> bool:
    return os.uname().nodename.startswith("login")


def _con():
    from pairpilot.store import connect
    tmp = Path(os.environ.get("SLURM_TMPDIR") or f"/tmp/{os.environ.get('USER', 'pilot')}/pairpilot")
    return connect(threads=max(1, int(os.environ.get("SLURM_CPUS_PER_TASK") or 2)), memory="16GB", tmp=tmp)


def _run() -> Path:
    run_id = os.environ["PILOT_CHILD_RUN"]
    path = RUNS / run_id
    if not path.exists():
        raise SystemExit(f"revision run is missing: {path}")
    return path


def _read_jsonl(path: Path) -> list[dict]:
    """Iterate records by newline. splitlines() also breaks on U+2028 inside JSON strings."""
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _cell(value) -> str:
    return " ".join(str(value or "").split()).replace("|", "/")


def _short(value, limit: int = 220) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


def _median(values: list[int]):
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def init_run(run_id: str) -> Path:
    run = RUNS / run_id
    run.mkdir(parents=False, exist_ok=True)
    for name in ("audit", "tables", "experiments", "reports", "requests", "responses", "jobs"):
        (run / name).mkdir(exist_ok=True)
    if not (run / "manifest.json").exists():
        write_json(run / "manifest.json", {
            "run_id": run_id,
            "parent_run_id": PRIOR.name,
            "preparation_run_id": PREP.name,
            "created_utc": _now(),
            "configuration_revision": RULE_VERSION,
            "revision_reason": "v1 turned every insufficient comparison into a candidate gap and repeated E6 calls once per finding. v2 keeps those rows as comparison states and only clusters direction conflicts that share reported fields.",
            "parent_not_rewritten": True,
            "status": "incomplete",
        })
    ledger_path = run / "audit" / "ledger.json"
    if not ledger_path.exists():
        prior = json.loads((PRIOR / "audit" / "ledger.json").read_text(encoding="utf-8"))
        available = available_usd(prior)
        spent = float(prior["spent_usd"])
        unique_hold = None if available is None else LIMITS["max_usd"] - spent - available
        seeded = {
            "spent_usd": spent,
            "reserved_usd": 0.0,
            "billing_unknown_usd": unique_hold,
            "attempts": prior.get("attempts") or 0,
            "input_tokens": prior.get("input_tokens") or 0,
            "output_tokens": prior.get("output_tokens") or 0,
            "canary_spent_usd": prior.get("canary_spent_usd") or 0.0,
            "canary_reserved_usd": 0.0,
            "canary_attempts": prior.get("canary_attempts") or 0,
            "canary_input_tokens": prior.get("canary_input_tokens") or 0,
            "canary_output_tokens": prior.get("canary_output_tokens") or 0,
            "reservations": {},
            "opening_reconciliation": {
                "source_ledger": str(PRIOR / "audit" / "ledger.json"),
                "at": _now(),
                "legacy_open_reservation": prior.get("reservations"),
                "legacy_reserved_usd": prior.get("reserved_usd"),
                "legacy_billing_unknown_usd": prior.get("billing_unknown_usd"),
                "counted_once_usd": unique_hold,
                "reason": "The open reservation and billing_unknown amounts match one timed-out request and are the same hold.",
            },
        }
        write_json(ledger_path, seeded)
        write_json(PRIOR / "audit" / "ledger_reconciliation_rev2.json", {
            "at": _now(),
            "revision_run_id": run_id,
            "counted_once_usd": unique_hold,
            "available_usd_after_overlap_fix": available,
            "ledger_json_not_modified": True,
        })
    return run


def _error_kind(raw: str) -> str:
    text = raw or ""
    if "credit_balance_exhausted" in text or "insufficient_quota" in text:
        return "quota_exhausted"
    if "timed out" in text.lower():
        return "timeout_or_transport"
    return "other"


def stage_r1(run_id: str) -> int:
    run = init_run(run_id)
    con = _con()
    findings = query_rows(con, f"SELECT * FROM read_parquet('{(PRIOR / 'tables' / 'findings.parquet').as_posix()}')")
    analysis = [row for row in findings if is_analysis_finding(row)]
    rules = load_bundle()["pilot"]["required_comparability_fields"]
    config_hash = digest({"rule": RULE_VERSION, "fields": rules})
    comparisons = build_candidates(analysis, config_hash, rules)
    within = within_document_pairs(analysis, config_hash, rules)
    by_id = {row["finding_id"]: row for row in analysis}
    marks = []
    for row in comparisons:
        gap = gap_from_comparison(row, config_hash, by_id)
        if gap:
            gap["comparison_id"] = row["comparison_id"]
            marks.append(gap)
    clustered = cluster_candidates(marks)
    write_parquet(run / "tables" / "comparisons.parquet", comparisons)
    write_parquet(run / "tables" / "within_document.parquet", within)
    write_parquet(run / "tables" / "candidate_marks.parquet", marks)
    write_parquet(run / "tables" / "candidate_questions.parquet", clustered)

    old_gaps = query_rows(con, f"SELECT candidate_id, gap_state, pair_id FROM read_parquet('{(PRIOR / 'tables' / 'candidate_gaps.parquet').as_posix()}')")
    old_comps = query_rows(con, f"SELECT condition_relation, result_relation FROM read_parquet('{(PRIOR / 'tables' / 'comparisons.parquet').as_posix()}')")
    requests = _read_jsonl(PRIOR / "requests" / "requests.jsonl")
    responses = _read_jsonl(PRIOR / "responses" / "responses.jsonl")
    paired = list(zip(requests, responses))
    task_counts = Counter(row.get("task") for row in requests)
    summary = {
        "finding_audit": {
            "n_rows": len(findings),
            "n_no_finding": sum(1 for row in findings if row.get("status") == "no_finding"),
            "n_analysis_finding": len(analysis),
            "n_background_excluded": sum(1 for row in findings if row.get("discourse_role") == "background"),
            "n_citation_located": sum(1 for row in findings if row.get("status") == "citation_located"),
            "question_type": dict(Counter(row.get("question_type") or "" for row in analysis)),
            "unit": "finding_row",
        },
        "comparison_audit": {
            "old_comparison_rows": len(old_comps),
            "old_gap_rows": len(old_gaps),
            "old_unique_candidate_id": len({row.get("candidate_id") for row in old_gaps}),
            "old_gap_state": dict(Counter(row.get("gap_state") for row in old_gaps)),
            "old_origin": "deterministic_rule_v1",
            "old_model_compare_requests": task_counts.get("compare_findings", 0),
            "old_synthesis_requests": task_counts.get("synthesize_pair", 0),
            "revised_cross_pmid": len(comparisons),
            "revised_within_document": len(within),
            "revised_condition": dict(Counter(row["condition_relation"] for row in comparisons)),
            "revised_result": dict(Counter(row["result_relation"] for row in comparisons)),
            "direction_marks": len(marks),
            "clustered_candidates": len(clustered),
            "origin": "deterministic_rule",
            "rule_version": RULE_VERSION,
            "config_hash": config_hash,
        },
        "request_inventory": {
            "n_attempt": len(requests),
            "n_response": len(responses),
            "by_task": dict(task_counts),
            "by_status": dict(Counter(row.get("status") for row in requests)),
            "note": "An attempt is one API call. A logical unit can be repeated once per finding under the old E6 loop.",
        },
    }
    links = query_rows(con, f"SELECT pair_id, pmid, source FROM read_parquet('{(PREP / 'inputs' / 'pair_pmids.parquet').as_posix()}')")
    names = query_rows(con, f"SELECT * FROM read_parquet('{(PRIOR / 'tables' / 'pair_names.parquet').as_posix()}')")
    name_of = {row["pair_id"]: row for row in names}
    by_pair = {}
    for row in links:
        slot = by_pair.setdefault(row["pair_id"], {"CORE": set(), "SemMedDB": set()})
        slot[row["source"]].add(str(row["pmid"]))
    pair_rows = []
    metric_rows = []
    draws_out = {}
    condition_rows = []
    for pair, slot in sorted(by_pair.items()):
        own = [row for row in analysis if row["pair_id"] == pair]
        rel = [row for row in comparisons if row["pair_id"] == pair]
        metric_rows.extend(source_metrics(run_id, pair, own, rel, slot["CORE"], slot["SemMedDB"]))
        draw = equal_article(slot["CORE"], slot["SemMedDB"], own, rel, 20261015)
        draws_out[pair] = {"status": draw["status"], "m": draw["m"]}
        if draw["status"] == "computed":
            draws_out[pair]["core_condition_median"] = _median([item["core"]["n_condition"] for item in draw["draws"]])
            draws_out[pair]["semmed_condition_median"] = _median([item["semmed"]["n_condition"] for item in draw["draws"]])
            draws_out[pair]["core_comparable_median"] = _median([item["core"]["n_comparable"] for item in draw["draws"]])
            draws_out[pair]["semmed_comparable_median"] = _median([item["semmed"]["n_comparable"] for item in draw["draws"]])
        keys = {}
        for source, pmids in (("CORE", slot["CORE"]), ("SemMedDB", slot["SemMedDB"]), ("Union", slot["CORE"] | slot["SemMedDB"])):
            keys[source] = sorted({" | ".join(condition_key(row)) for row in own if str(row["pmid"]) in pmids})
            for key in keys[source]:
                condition_rows.append({"pair_id": pair, "source": source, "condition_key": key})
        info = name_of.get(pair, {})
        pair_rows.append({
            "pair_id": pair,
            "gene_id": str(info.get("gene_id") or pair.split("__")[0]),
            "disease_cui": str(info.get("disease_cui") or pair.split("__")[-1]),
            "raw_gene_mentions": info.get("gene_name") or "",
            "raw_disease_mentions": info.get("disease_name") or "",
            "gene_name_source": info.get("gene_name_source") or "",
            "disease_name_source": info.get("disease_name_source") or "",
            "stratum": info.get("stratum") or "",
            "n_core_pmid": len(slot["CORE"]),
            "n_semmed_pmid": len(slot["SemMedDB"]),
            "n_union_pmid": len(slot["CORE"] | slot["SemMedDB"]),
            "n_intersection_pmid": len(slot["CORE"] & slot["SemMedDB"]),
            "n_core_only_pmid": len(slot["CORE"] - slot["SemMedDB"]),
            "n_semmed_only_pmid": len(slot["SemMedDB"] - slot["CORE"]),
            "n_analysis_finding": len(own),
            "n_no_finding": sum(1 for row in findings if row.get("pair_id") == pair and row.get("status") == "no_finding"),
            "question_type": dict(Counter(row.get("question_type") for row in own)),
            "n_comparison": len(rel),
            "n_insufficient": sum(1 for row in rel if row["condition_relation"] == "insufficient_context"),
            "n_different_question": sum(1 for row in rel if row["condition_relation"] == "different_question"),
            "n_known_difference": sum(1 for row in rel if row["condition_relation"] == "known_context_difference"),
            "n_comparable": sum(1 for row in rel if row["condition_relation"] == "comparable_on_reported_fields"),
            "n_different_direction": sum(1 for row in rel if row["result_relation"] == "different_reported_direction"),
            "n_similar_direction": sum(1 for row in rel if row["result_relation"] == "similar_reported_direction"),
            "n_candidate": sum(1 for row in clustered if row["pair_id"] == pair),
            "core_only_conditions": [key for key in keys["CORE"] if key not in set(keys["SemMedDB"])][:12],
            "semmed_only_conditions": [key for key in keys["SemMedDB"] if key not in set(keys["CORE"])][:12],
            "n_core_condition": len(keys["CORE"]),
            "n_semmed_condition": len(keys["SemMedDB"]),
            "n_union_condition": len(keys["Union"]),
            "equal_article": draws_out[pair],
            "expected_m": EXPECTED_M.get(pair),
        })
    write_parquet(run / "tables" / "experiment_metrics.parquet", metric_rows)
    write_parquet(run / "tables" / "condition_keys.parquet", condition_rows)
    write_json(run / "experiments" / "e1_draw_summary.json", draws_out)

    e2_calls = query_rows(con, f"SELECT * FROM read_parquet('{(PRIOR / 'tables' / 'e2_calls.parquet').as_posix()}')")
    direct = [resp for _, resp in paired if resp.get("task") == "direct_pair"]
    knowledge = [resp for _, resp in paired if resp.get("task") == "knowledge_pair"]
    d_rows = [row for row in e2_calls if row.get("config") == "D"]
    k_rows = [row for row in e2_calls if row.get("config") == "K"]
    e2_out = []
    for call, resp in zip(d_rows, direct):
        parsed = resp.get("parsed") if isinstance(resp.get("parsed"), dict) else {}
        relations = parsed.get("relations") if isinstance(parsed.get("relations"), list) else []
        gaps = parsed.get("candidate_gaps") if isinstance(parsed.get("candidate_gaps"), list) else []
        e2_out.append({
            "pair_id": call.get("pair_id"),
            "config": "D",
            "status": resp.get("status"),
            "n_relation": len(relations),
            "n_gap": len(gaps),
            "example": _relation_text(relations[0] if relations else {}),
        })
    for call, resp in zip(k_rows, knowledge):
        parsed = resp.get("parsed") if isinstance(resp.get("parsed"), dict) else {}
        found = parsed.get("findings") if isinstance(parsed.get("findings"), list) else []
        e2_out.append({
            "pair_id": call.get("pair_id"),
            "pmid": str(call.get("pmid") or ""),
            "config": "K",
            "status": resp.get("status"),
            "n_finding": len(found),
            "n_located_quote": sum(1 for item in found if isinstance(item, dict) and item.get("evidence")),
        })
    write_parquet(run / "tables" / "e2_alignment.parquet", e2_out)
    for row in pair_rows:
        row["d_relations"] = next((item["n_relation"] for item in e2_out if item["config"] == "D" and item["pair_id"] == row["pair_id"]), None)
        row["d_example"] = next((item["example"] for item in e2_out if item["config"] == "D" and item["pair_id"] == row["pair_id"]), "")
        k_for = [item for item in e2_out if item["config"] == "K" and item["pair_id"] == row["pair_id"]]
        row["k_calls"] = len(k_for)
        row["k_findings"] = sum(item["n_finding"] for item in k_for)
        f_for_k = [item for item in analysis if item["pair_id"] == row["pair_id"] and str(item["pmid"]) in {item2["pmid"] for item2 in k_for}]
        row["f_findings_on_k_papers"] = len(f_for_k)

    frozen = read_json(PREP / "inputs" / "frozen_subsets.json")
    e3_items = frozen.get("E3") or []
    scope_s = [resp for _, resp in paired if resp.get("task") == "assess_scope_s"]
    scope_a = [resp for _, resp in paired if resp.get("task") == "assess_scope_a"]
    e3_rows = []
    for index, item in enumerate(e3_items):
        left = scope_s[index] if index < len(scope_s) else {}
        right = scope_a[index] if index < len(scope_a) else {}
        e3_rows.append(_scope_row(item, left, right))
    write_parquet(run / "tables" / "scope_transfer.parquet", e3_rows)

    e5 = _e5(run, con, clustered, {str(row["pmid"]) for row in links})
    e6 = _e6(findings, paired)
    summary["pairs"] = pair_rows
    summary["e2"] = {
        "d_calls": len(d_rows),
        "d_responses": len(direct),
        "k_calls": len(k_rows),
        "k_responses": len(knowledge),
        "k_compare_requests": 0,
        "k_synthesis_requests": 0,
        "aligned": len(d_rows) == len(direct) and len(k_rows) == len(knowledge),
    }
    summary["e3"] = {
        "n_item": len(e3_items),
        "n_s": len(scope_s),
        "n_a": len(scope_a),
        "n_transfer": len(e3_rows),
        "s_not_judged": sum(1 for row in e3_rows if row["s_result"] == "not_judged"),
        "a_reported": sum(1 for row in e3_rows if row["a_result"] == "reported"),
        "both_not_judged": sum(1 for row in e3_rows if row["s_result"] == "not_judged" and row["a_result"] == "not_judged"),
    }
    summary["e5"] = e5
    summary["e6"] = e6
    summary["stages"] = _stage_table(summary)
    summary["cases"] = _cases(pair_rows)
    summary["ledger"] = _ledger_view(run)
    write_json(run / "audit" / "summary.json", summary)
    write_json(run / "audit" / "stage_snapshot.json", summary["stages"])
    _write_report(run, summary)
    return 0


def _relation_text(relation: dict) -> str:
    if not isinstance(relation, dict):
        return ""
    for key in ("interpretation", "statement", "comparison", "topic"):
        value = relation.get(key)
        if isinstance(value, str) and value.strip():
            return _short(value)
    return ""


def _scope_row(item: dict, left: dict, right: dict) -> dict:
    def label(resp: dict, field: str) -> str:
        parsed = resp.get("parsed") if isinstance(resp.get("parsed"), dict) else {}
        value = parsed.get(field)
        if value in (None, "", "not_judged"):
            return "not_judged"
        if isinstance(value, dict) and not any(value.values()):
            return "not_judged"
        return "reported"

    parsed_a = right.get("parsed") if isinstance(right.get("parsed"), dict) else {}
    differences = parsed_a.get("differences")
    if isinstance(differences, list):
        diff_text = "; ".join(_short(item.get("dimension") if isinstance(item, dict) else item, 80) for item in differences[:4])
    else:
        diff_text = _short(differences, 160)
    return {
        "pair_id": item.get("pair_id"),
        "pmid": str(item.get("pmid")),
        "s_status": left.get("status") or "missing_response",
        "a_status": right.get("status") or "missing_response",
        "s_result": label(left, "result_scope"),
        "s_conclusion": label(left, "conclusion_scope"),
        "a_result": label(right, "result_scope"),
        "a_conclusion": label(right, "conclusion_scope"),
        "a_differences": diff_text,
    }


def _e5(run: Path, con, clustered: list[dict], known: set[str]) -> dict:
    old = []
    path = PRIOR / "tables" / "search_results.parquet"
    if path.exists():
        old = query_rows(con, f"SELECT * FROM read_parquet('{path.as_posix()}')")
    for row in old:
        row["protocol_status"] = "historical_off_protocol"
    write_parquet(run / "tables" / "search_results_historical.parquet", old)
    ordered = sorted(clustered, key=lambda row: rank(20261015, "e5-rev2", row["candidate_id"]))[:3]
    write_json(run / "audit" / "e5_frozen_candidates.json", ordered)
    cache = {}

    def hits_for(candidate, query):
        key = query["text"][:300]
        if key not in cache:
            cache[key] = _esearch(key)
        rows = cache[key]
        if rows and rows[0].get("status") == "fetch_failed":
            return []
        return rows

    selected = allocate_candidates(ordered, hits_for, known) if ordered else []
    write_parquet(run / "tables" / "search_results.parquet", [row for row in selected if row.get("pmid")])
    new = [row for row in selected if row.get("status") == "retrieved" and not row.get("already_in_packet")]
    return {
        "old_rows": len(old),
        "old_new_flags": sum(1 for row in old if not row.get("already_in_packet")),
        "old_unique_pmid": len({str(row.get("pmid")) for row in old}),
        "old_candidate_ids": len({row.get("candidate_id") for row in old}),
        "old_status": "historical_off_protocol",
        "old_reason": "v1 gaps collapsed to one candidate id and one candidate took every new PMID. Those rows were not deep-read.",
        "revised_candidates": len(ordered),
        "revised_retrieved_new": len(new),
        "revised_by_candidate": dict(Counter(row.get("candidate_id") for row in new)),
        "deep_read": "not_started",
        "review": "not_started",
        "fetch_failed_queries": sum(1 for rows in cache.values() if rows and rows[0].get("status") == "fetch_failed"),
    }


def _esearch(term: str) -> list[dict]:
    query = urllib.parse.urlencode({"db": "pubmed", "term": term, "retmax": 50, "retmode": "json", "sort": "relevance"})
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + query
    if os.environ.get("NCBI_API_KEY"):
        url += "&api_key=" + urllib.parse.quote(os.environ["NCBI_API_KEY"])
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "pairpilot/1.0"}), timeout=40) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return [{"pmid": "", "status": "fetch_failed", "error": type(exc).__name__}]
    return [{"pmid": pmid, "retrieved_at": _now()} for pmid in (payload.get("esearchresult") or {}).get("idlist") or []]


def _e6(findings: list[dict], paired: list[tuple]) -> dict:
    base = [row for row in findings if row.get("pair_id") in E6_PAIRS and row.get("question_type")]
    attempts = [(req, resp) for req, resp in paired if req.get("task") == "extract_findings" and req.get("replicate_id") in (1, 2)]
    expected = []
    for replicate in (1, 2):
        for row in base:
            expected.append((replicate, row))
    mapped = len(attempts) == len(expected)
    per_paper = {}
    if mapped:
        for (replicate, row), (req, resp) in zip(expected, attempts):
            key = (row["pair_id"], str(row["pmid"]), replicate)
            slot = per_paper.setdefault(key, {"attempts": 0, "ok": 0, "parsed": None})
            slot["attempts"] += 1
            if resp.get("status") == "ok" and slot["parsed"] is None:
                slot["ok"] += 1
                slot["parsed"] = resp.get("parsed")
            elif resp.get("status") == "ok":
                slot["ok"] += 1
    papers = {(row["pair_id"], str(row["pmid"])) for row in base}
    alignments = []
    for pair, pmid in sorted(papers):
        baseline = [row for row in base if row["pair_id"] == pair and str(row["pmid"]) == pmid and is_analysis_finding(row)]
        for replicate in (1, 2):
            slot = per_paper.get((pair, pmid, replicate), {})
            parsed = slot.get("parsed") if isinstance(slot.get("parsed"), dict) else {}
            repeat = parsed.get("findings") if isinstance(parsed.get("findings"), list) else []
            repeat = [row for row in repeat if isinstance(row, dict)]
            aligned = align_replicates(baseline, repeat) if repeat else {"matched": 0, "only_baseline": len({structured_key(row) for row in baseline}), "only_repeat": 0, "span_changed": 0}
            alignments.append({"pair_id": pair, "pmid": pmid, "replicate_id": replicate, "attempts": slot.get("attempts", 0), "ok_attempts": slot.get("ok", 0), **aligned, "used": bool(repeat)})
    kinds = Counter(_error_kind(resp.get("raw") or "") for req, resp in attempts if resp.get("status") != "ok")
    perturbations = [resp for req, resp in paired if str(req.get("task") or "").startswith("perturb_")]
    return {
        "pairs": sorted(E6_PAIRS),
        "n_pmid": len(papers),
        "logical_replicates": len(papers) * 2,
        "finding_rows_used_as_calls": len(base),
        "attempts": len(attempts),
        "attempt_map_matches_finding_loop": mapped,
        "attempt_status": dict(Counter(resp.get("status") for _, resp in attempts)),
        "failure_kind": dict(kinds),
        "papers_with_one_success": {
            "replicate_1": sum(1 for row in alignments if row["replicate_id"] == 1 and row["used"]),
            "replicate_2": sum(1 for row in alignments if row["replicate_id"] == 2 and row["used"]),
            "denominator_papers": len(papers),
        },
        "alignment_used": [row for row in alignments if row["used"]],
        "perturbation_attempts": len(perturbations),
        "perturbation_ok": sum(1 for row in perturbations if row.get("status") == "ok"),
        "perturbation_kind": dict(Counter(_error_kind(row.get("raw") or "") for row in perturbations if row.get("status") != "ok")),
        "stability": "not_computed",
        "reason": "Calls were issued once per finding rather than once per paper. Later calls stopped because the provider account reported credit_balance_exhausted. Perturbations have no successful response.",
    }


def _stage_table(summary: dict) -> dict:
    e6 = summary["e6"]
    e5 = summary["e5"]
    return {
        "extract": {"state": "completed_reused", "expected": 70, "succeeded": 70, "unit": "pmid"},
        "compare_rule": {"state": "revised", "expected": "all analysis finding pairs", "succeeded": summary["comparison_audit"]["revised_cross_pmid"], "unit": "cross_pmid_comparison"},
        "compare_model": {"state": "not_started", "reason": "No compare_findings request exists. Semantic residue was not sent to the model."},
        "synthesize": {"state": "partial", "succeeded": summary["comparison_audit"]["old_synthesis_requests"], "expected": 8, "unit": "pair", "reason": "Only the canary synthesis was called."},
        "E1": {"state": "revised_from_cache", "reason": "Counts were recomputed from reused findings and the revised rule. No new model calls."},
        "E2_D": {"state": "completed_reused" if summary["e2"]["aligned"] else "partial", "expected": 8, "succeeded": summary["e2"]["d_responses"], "unit": "pair_call"},
        "E2_K": {"state": "partial", "expected": "4 small pairs, extract plus compare plus synthesis", "succeeded_extract_calls": summary["e2"]["k_responses"], "compare": 0, "synthesis": 0},
        "E3": {"state": "completed_reused" if summary["e3"]["n_transfer"] == summary["e3"]["n_item"] else "partial", "expected": summary["e3"]["n_item"], "succeeded": summary["e3"]["n_transfer"], "unit": "pmid_s_to_a"},
        "E4": {"state": "disabled_deferred"},
        "E5": {"state": "partial", "retrieval_revised_candidates": e5["revised_candidates"], "deep_read": e5["deep_read"], "review": e5["review"], "historical": e5["old_status"]},
        "E6": {"state": "failed", "logical_replicates": e6["logical_replicates"], "attempts": e6["attempts"], "perturbation_ok": e6["perturbation_ok"], "reason": e6["reason"]},
        "open_model": {"state": "disabled"},
        "report": {"state": "incomplete"},
    }


def _cases(pairs: list[dict]) -> dict:
    ranked = sorted(pairs, key=lambda row: (-row["n_candidate"], -row["n_different_direction"], row["pair_id"]))
    change = ranked[0]["pair_id"] if ranked else ""
    low_pool = [row for row in pairs if row["pair_id"] != change]
    low = sorted(low_pool, key=lambda row: (row["n_union_condition"] - max(row["n_core_condition"], row["n_semmed_condition"]), row["n_candidate"], row["pair_id"]))
    return {
        "rule": "Change case: most clustered direction-conflict candidates, then most different-direction marks, then pair id. Low-gain case: smallest union-minus-larger-side condition count among the others, then pair id.",
        "change": change,
        "low": low[0]["pair_id"] if low else "",
    }


def _ledger_view(run: Path) -> dict:
    state = json.loads((run / "audit" / "ledger.json").read_text(encoding="utf-8"))
    available = available_usd(state)
    return {
        "authorized_usd": LIMITS["max_usd"],
        "canary_cap_usd": LIMITS["canary_max_usd"],
        "settled_usd": state.get("spent_usd"),
        "in_flight_reserved_usd": state.get("reserved_usd"),
        "unknown_billing_reserved_usd": state.get("billing_unknown_usd"),
        "available_usd": available,
        "available_status": "ok" if available is not None else "unavailable",
        "attempts": state.get("attempts"),
        "attempt_cap": LIMITS["max_attempts"],
        "input_tokens": state.get("input_tokens"),
        "output_tokens": state.get("output_tokens"),
        "updated_utc": _now(),
        "price_source": "https://developers.openai.com/api/docs/models/gpt-6.1-sol",
        "price_retrieved_utc": "2026-10-15",
    }


def _write_report(run: Path, summary: dict) -> None:
    probe = read_json(run / "audit" / "probe.json") if (run / "audit" / "probe.json").exists() else {}
    synth = read_json(run / "audit" / "synthesis.json") if (run / "audit" / "synthesis.json").exists() else {}
    grounded = read_json(run / "audit" / "synthesis_grounded.json") if (run / "audit" / "synthesis_grounded.json").exists() else {}
    ledger = _ledger_view(run)
    lines = [
        "# CORE–SemMedDB pilot 修订",
        "",
        f"修订 run `{run.name}`，直接上一级是 `{PRIOR.name}`，准备 run 是 `{PREP.name}`。旧响应、旧报告和旧账本文件没有改写。",
        "",
        "这里的模型输出和规则输出都还没有人工科学核对。引用能定位只说明字符串对得上，不说明原文支持。结构有效只说明字段齐全。",
        "",
        "## 1. 样本、完成范围和这次修订改变了什么",
        "",
        "实际样本仍是 8 个 pair、70 个唯一 PMID。identical 的 4–20 篇两档没有候选，分母没有改回 10。",
        "",
        "主提取沿用上一级的 70 篇结果。293 行里，2 行是文档级 `no_finding`，5 行是 background，其余进入分析的 finding 是 "
        f"{summary['finding_audit']['n_analysis_finding']} 行。`no_finding` 不再进入比较或缺口分母。",
        "",
        f"上一级 5812 条比较和 {summary['comparison_audit']['old_gap_rows']} 条缺口来自规则函数 `build_candidates` / `gap_from_comparison`，"
        "不是模型判断，也没有比较缓存可继承。请求清单里 `compare_findings` 为 "
        f"{summary['comparison_audit']['old_model_compare_requests']} 次，`synthesize_pair` 只有 "
        f"{summary['comparison_audit']['old_synthesis_requests']} 次，而且是 canary。",
        "",
        f"修订规则 `{RULE_VERSION}` 重新计算了 {summary['comparison_audit']['revised_cross_pmid']} 条跨 PMID 比较。"
        f"其中条件关系是 {summary['comparison_audit']['revised_condition']}。"
        f"方向冲突的原始标记 {summary['comparison_audit']['direction_marks']} 条，按问题、干预、对照和结局合并后是 "
        f"{summary['comparison_audit']['clustered_candidates']} 个候选问题。缺字段仍保留为比较状态，不再每条都升级成研究缺口。",
        "",
        "配置变化只影响比较和缺口的定义。已成功的提取响应没有重写，也没有当成新的模型判断。",
        "",
    ]
    lines.extend(_pair_table(summary["pairs"]))
    lines.extend(_e1_section(summary))
    _refresh_d_examples(summary)
    lines.extend(_e2_section(summary))
    lines.extend(_e3_section(summary))
    lines.extend(_e5_section(summary, probe))
    lines.extend(_e6_section(summary))
    gap_rows = _grounded_gap_rows(run)
    write_parquet(run / "tables" / "grounded_gaps.parquet", gap_rows)
    lines.extend(_direction_section(summary, probe, synth, grounded, gap_rows))
    lines.extend(_money_section(ledger, summary, probe, synth))
    text = "\n".join(lines).rstrip() + "\n"
    (run / "reports").mkdir(exist_ok=True)
    (run / "reports" / "report.md").write_text(text, encoding="utf-8")
    _pair_pages(run, summary)
    manifest = read_json(run / "manifest.json")
    manifest["status"] = "incomplete"
    manifest["updated_utc"] = _now()
    write_json(run / "manifest.json", manifest)


def _pair_table(pairs: list[dict]) -> list[str]:
    lines = ["| pair | 分层 | 基因 ID / 原始提及 | 疾病 ID / 原始提及 | CORE / SemMedDB / 并集 PMID | 分析 finding |", "| --- | --- | --- | --- | --- | --- |"]
    for row in pairs:
        lines.append("| " + " | ".join([
            _cell(row["pair_id"]),
            _cell(row["stratum"]),
            _cell(f"{row['gene_id']} / {row['raw_gene_mentions']}"),
            _cell(f"{row['disease_cui']} / {row['raw_disease_mentions']}"),
            f"{row['n_core_pmid']} / {row['n_semmed_pmid']} / {row['n_union_pmid']}",
            str(row["n_analysis_finding"]),
        ]) + " |")
    lines.append("")
    lines.append("名称来自 CORE 句子提及或 SemMedDB `argument_name`。HCC、SCC、NSCLC 这些缩写保持为原始提及，不当作已经核对过的疾病规范名。")
    lines.append("")
    return lines


def _e1_section(summary: dict) -> list[str]:
    lines = ["## 2. E1：第二个库增加了什么", ""]
    lines.append("文章数来自 pair–PMID 表。条件数是提取字段组成的元组，同一自由文本的不同写法会算成不同元组。并集文章变多时，元组数量增加本身只是集合变大，不是方法有效的证据。")
    lines.append("")
    lines.append("| pair | 仅 CORE / 仅 SemMedDB PMID | CORE / SemMedDB / Union 条件 | Union 相对较多一侧增加的条件 | 可比较 / 信息不足 | 等文章量 m | 重采样后条件中位数 CORE / SemMedDB |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in summary["pairs"]:
        draw = row["equal_article"]
        added = row["n_union_condition"] - max(row["n_core_condition"], row["n_semmed_condition"])
        med = f"{draw.get('core_condition_median')} / {draw.get('semmed_condition_median')}" if draw.get("status") == "computed" else draw.get("status")
        flag = "" if row["expected_m"] == draw.get("m") else "，与预期 m 不一致"
        lines.append("| " + " | ".join([
            _cell(row["pair_id"]),
            f"{row['n_core_only_pmid']} / {row['n_semmed_only_pmid']}",
            f"{row['n_core_condition']} / {row['n_semmed_condition']} / {row['n_union_condition']}",
            str(added),
            f"{row['n_comparable']} / {row['n_insufficient']}",
            f"{draw.get('m')}{flag}",
            str(med),
        ]) + " |")
    lines.append("")
    lines.append("分母是该 pair 的分析 finding 或比较行，单位分别是条件元组、比较行和 PMID。来源表是本修订 run 的 `tables/condition_keys.parquet`、`tables/comparisons.parquet` 和 `experiments/e1_draw_summary.json`。50 次重采样没有新的模型请求。")
    lines.append("")
    return lines


def _e2_section(summary: dict) -> list[str]:
    e2 = summary["e2"]
    lines = ["## 3. E2：D、F、K 实际改变了什么", ""]
    lines.append(f"D 有 {e2['d_calls']} 次调用和 {e2['d_responses']} 条响应，顺序与 `e2_calls` 一致：{e2['aligned']}。K 有 {e2['k_calls']} 次提取响应，只覆盖四个 small pair；large pair 的 K 是 not_applicable，不是失败。K 的比较和综合请求数是 0，所以还不能和 D、F 对齐到同一种最终问题合同。")
    lines.append("")
    lines.append("| pair | F 分析 finding | D 关系数 | D 例子 | K 调用 / K finding | 同一批 K 文章上的 F finding |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in summary["pairs"]:
        lines.append("| " + " | ".join([
            _cell(row["pair_id"]),
            str(row["n_analysis_finding"]),
            str(row.get("d_relations")),
            _cell(row.get("d_example")),
            f"{row.get('k_calls')} / {row.get('k_findings')}",
            str(row.get("f_findings_on_k_papers")),
        ]) + " |")
    lines.append("")
    lines.append("D 的例子是模型英文表述，不是已核对结论。K finding 数高于或低于 F，只说明输入里多了 KG 原文字段后的输出数量变化，不表示改对了。来源表是 `tables/e2_alignment.parquet`。")
    lines.append("")
    return lines


def _e3_section(summary: dict) -> list[str]:
    e3 = summary["e3"]
    lines = ["## 4. E3：来源句到摘要的范围", ""]
    lines.append(f"冻结清单 {e3['n_item']} 个 PMID。保存的来源句响应 {e3['n_s']} 条，摘要响应 {e3['n_a']} 条，转移表 {e3['n_transfer']} 行。")
    lines.append(f"来源句结果范围为 not_judged 的有 {e3['s_not_judged']} / {e3['n_transfer']}。摘要结果范围有报告字段的有 {e3['a_reported']} / {e3['n_transfer']}。两边都无法判断的有 {e3['both_not_judged']} / {e3['n_transfer']}。")
    lines.append("")
    lines.append("来源句没有作者结论时保持无法判断。摘要多出来的上下文不是更准确的证明，也没有比较不同文本版本的长度效应。逐篇字段在 `tables/scope_transfer.parquet`。")
    lines.append("")
    return lines


def _e5_section(summary: dict, probe: dict) -> list[str]:
    e5 = summary["e5"]
    review = e5["review"]
    if probe.get("status") == "quota_exhausted":
        review = "blocked_provider_quota"
    lines = ["## 5. E5：候选、查询和复核", ""]
    lines.append(f"上一级冻结了 3 行，但它们是同一 `candidate_id` 的重复缺字段标记，实际检索候选数是 {e5['old_candidate_ids']}。历史检索行 {e5['old_rows']}，其中新 PMID 标记 {e5['old_new_flags']}，唯一 PMID {e5['old_unique_pmid']}。这些行标为 `{e5['old_status']}`：{e5['old_reason']}")
    lines.append("")
    if e5["revised_candidates"] == 0:
        lines.append("修订规则没有产生具备同一已报告问题和不同方向的候选，因此没有新的 E5 检索。深读和复核是 not_applicable，不是还在排队。旧的 30 篇仍然不能补成这个缺口。")
    else:
        lines.append(f"按修订后的聚类和 seed 20261015 重新冻结 {e5['revised_candidates']} 个候选。每个候选两条查询、每条最多 5 篇新摘要。本次检索到的新 PMID 共 {e5['revised_retrieved_new']}，分候选 {e5['revised_by_candidate']}。")
        lines.append(f"深读状态：{e5['deep_read']}。复核状态：{review}。没有深读和复核，E5 不算完成。")
    lines.append("")
    return lines


def _e6_section(summary: dict) -> list[str]:
    e6 = summary["e6"]
    used = e6["alignment_used"]
    matched = sum(row["matched"] for row in used)
    only_base = sum(row["only_baseline"] for row in used)
    only_repeat = sum(row["only_repeat"] for row in used)
    lines = ["## 6. E6：重复和扰动", ""]
    lines.append(f"计划是两个固定 pair、每篇两次新提取。有分析 finding 的论文是 {e6['n_pmid']} 篇，逻辑提取数是 {e6['logical_replicates']}。旧循环按 finding 行调用，finding 行 {e6['finding_rows_used_as_calls']}，两次重复共 {e6['attempts']} 个 attempt。映射是否吻合：{e6['attempt_map_matches_finding_loop']}。")
    lines.append(f"attempt 状态：{e6['attempt_status']}。失败类型：{e6['failure_kind']}。至少有一次成功提取的论文数，replicate 1 是 {e6['papers_with_one_success']['replicate_1']} / {e6['n_pmid']}，replicate 2 是 {e6['papers_with_one_success']['replicate_2']} / {e6['n_pmid']}。")
    lines.append(f"在有成功响应的论文上，结构化键 matched {matched}，只在主运行 {only_base}，只在重复 {only_repeat}。失败和没有响应的论文没有算进不一致。这只是提取字段的一致，不是缺口稳定性。")
    lines.append(f"扰动 attempt {e6['perturbation_attempts']}，成功 {e6['perturbation_ok']}。失败类型 {e6['perturbation_kind']}。稳定性指标{e6['stability']}。")
    lines.append("")
    lines.append(e6["reason"])
    lines.append("")
    return lines


def _refresh_d_examples(summary: dict) -> None:
    responses = _read_jsonl(PRIOR / "responses" / "responses.jsonl")
    direct = [row.get("parsed") for row in responses if row.get("task") == "direct_pair"]
    by_pair = {row["pair_id"]: row for row in summary["pairs"]}
    ordered = sorted(by_pair)
    for pair_id, parsed in zip(ordered, direct):
        relations = parsed.get("relations") if isinstance(parsed, dict) else None
        if relations:
            by_pair[pair_id]["d_example"] = _relation_text(relations[0])


def _grounded_gap_rows(run: Path) -> list[dict]:
    rows = []
    meta = read_json(run / "audit" / "synthesis_grounded.json") if (run / "audit" / "synthesis_grounded.json").exists() else {}
    expected = [row.get("pair_id") or "" for row in meta.get("rows") or []]
    seen = 0
    for response in _read_jsonl(run / "responses" / "responses.jsonl"):
        if response.get("task") != "synthesize_pair_grounded" or response.get("status") != "ok":
            continue
        parsed = response.get("parsed") if isinstance(response.get("parsed"), dict) else {}
        pair_id = parsed.get("pair_id") or (expected[seen] if seen < len(expected) else "")
        seen += 1
        for gap in parsed.get("candidate_gaps") or []:
            if not isinstance(gap, dict):
                continue
            quote = ""
            evidence = gap.get("evidence_quotes")
            if isinstance(evidence, list) and evidence:
                quote = evidence[0] if isinstance(evidence[0], str) else str(evidence[0])
            rows.append({
                "pair_id": pair_id,
                "gap_state": gap.get("gap_state") or "",
                "question": _short(gap.get("question"), 300),
                "missing_dimension": _short(gap.get("missing_dimension"), 240),
                "quote": _short(quote, 240),
                "origin": "model_judgment",
                "input": "finding_quotes_and_known_context_differences",
            })
    return rows


def _grounded_gap_lines(rows: list[dict]) -> list[str]:
    if not rows:
        return []
    lines = ["", "带引句综合返回的候选问题如下。英文保持原文。来源表是 `tables/grounded_gaps.parquet`。", "", "| pair | gap_state | 问题 |", "| --- | --- | --- |"]
    for row in rows:
        lines.append("| " + " | ".join([_cell(row["pair_id"]), _cell(row["gap_state"]), _cell(row["question"])]) + " |")
    lines.append("")
    return lines


def _direction_section(summary: dict, probe: dict, synth: dict, grounded: dict, gap_rows: list[dict]) -> list[str]:
    audit = summary["comparison_audit"]
    finding = summary["finding_audit"]
    other = finding["question_type"].get("other", 0)
    insufficient = audit["revised_condition"].get("insufficient_context", 0)
    total = audit["revised_cross_pmid"] or 1
    known_diff = audit["revised_condition"].get("known_context_difference", 0)
    contrast = max(summary["pairs"], key=lambda row: (row["n_known_difference"], row["pair_id"]))
    others = [row for row in summary["pairs"] if row["pair_id"] != contrast["pair_id"]]
    low = min(others, key=lambda row: (row["n_union_condition"] - max(row["n_core_condition"], row["n_semmed_condition"]), row["pair_id"]))
    lines = ["## 7. 三个方向是否值得继续", ""]
    lines.append(f"8 个 pair 的聚类方向冲突候选都是 {audit['clustered_candidates']}。没有 pair 可以代表“发现了方向冲突”。已报告字段不相同最多的是 `{contrast['pair_id']}`，known_context_difference {contrast['n_known_difference']} 条；这表示规则看到了不同条件，不是科学矛盾。条件元组增量最小的是 `{low['pair_id']}`：Union {low['n_union_condition']}，较多一侧 {max(low['n_core_condition'], low['n_semmed_condition'])}。")
    lines.append("")
    lines.append(f"B，跨文章比较。分析 finding 里 question_type 为 other 的有 {other} / {finding['n_analysis_finding']}。修订后的跨 PMID 比较里，信息不足是 {insufficient} / {total}，已知条件不同是 {known_diff} / {total}，可比较是 {audit['revised_condition'].get('comparable_on_reported_fields', 0)} / {total}。缺字段和 other 问题都不是研究缺口。规则没有留下可合并的方向冲突候选。继续 B 之前，需要收窄提取，让预后、表达差异和扰动之外的记录不再占满比较对；不值得对这 {total} 对再做模型逐对判断。")
    if synth.get("n_ok") and not synth.get("input_has_quotes"):
        lines.append(f"另有 {synth.get('n_ok')} / {synth.get('n_expected')} 个 pair 的综合调用只收到计数，没有原文引句。它们不能当作有依据的候选，也不能进入 E5。")
    if grounded.get("n_ok") is not None:
        lines.append(f"带引句的综合完成 {grounded.get('n_ok')} / {grounded.get('n_expected')} 个 pair。各 pair 返回的候选数是 {[(row.get('pair_id'), row.get('n_gap')) for row in grounded.get('rows') or []]}。空列表是允许的科学输出。这些仍然不是人工核对后的研究缺口。")
        lines.extend(_grounded_gap_lines(gap_rows))
    lines.append("")
    e3 = summary["e3"]
    lines.append(f"A，摘要结果到结论范围。{e3['s_not_judged']} / {e3['n_transfer']} 的来源句结果范围无法判断，{e3['a_reported']} / {e3['n_transfer']} 的摘要响应写出了结果范围字段。这说明来源句经常带不出作者结论，但这 16 对还没有人工核对，不能推断全文缺少实验，也不能把摘要更长当成更准确。下一步最多是核对这 16 对转移表，而不是扩大模型批次。")
    lines.append("")
    lines.append("E5，用补充检索检验候选。旧的 30 篇不能进入主分析，因为它们来自失效的缺口定义，而且名额没有按候选分开。修订后没有通过结构门槛的方向冲突候选，所以没有新的深读，也没有“保持 / 细化 / 受到挑战 / 撤回”的复核。这个方向目前应暂缓，不能说候选已经被文献检验。")
    lines.append("")
    lines.append(f"正式实验前需要的少量人工核对是：{known_diff} 条已知条件不同是否真是不同问题；16 对来源句和摘要的范围差异是否被模型读对。E6 的扰动没有成功响应。账户额度探针后来通过了，但扰动还没有按修正后的每篇一次重跑。")
    lines.append("")
    return lines


def _money_section(ledger: dict, summary: dict, probe: dict, synth: dict) -> list[str]:
    available = ledger["available_usd"]
    available_text = "unavailable" if available is None else f"{available:.4f}"
    lines = ["## 8. 费用、状态和未完成项", ""]
    lines.append(f"授权总额 {ledger['authorized_usd']} 美元，canary 上限 {ledger['canary_cap_usd']} 美元，包含在总额内。累计结算 {ledger['settled_usd']} 美元。未结算的在途预留 {ledger['in_flight_reserved_usd']} 美元。计费不明预留 {ledger['unknown_billing_reserved_usd']} 美元，这一笔来自同一次超时，没有和在途预留相加两次。保守可用 {available_text} 美元。更新时间 {ledger['updated_utc']}。")
    lines.append(f"已用 attempt {ledger['attempts']} / {ledger['attempt_cap']}。输入 token {ledger['input_tokens']}，输出 token {ledger['output_tokens']}。价格来源 {ledger['price_source']}，记录日期 {ledger['price_retrieved_utc']}。")
    lines.append("")
    lines.append("阶段状态：")
    for name, row in summary["stages"].items():
        lines.append(f"- {name}：{row.get('state')}")
    if probe:
        lines.append(f"- provider_probe：{probe.get('status')}，passed={probe.get('passed')}")
    else:
        lines.append("- provider_probe：not_started")
    if synth:
        lines.append(f"- revised_synthesis：ok {synth.get('n_ok')} / {synth.get('n_expected')}")
    lines.append("")
    lines.append("仍未完成：K 的比较和综合、E6 按每篇一次的完整重复，以及全部依据扰动。5812 对比较没有逐对送模型，因为其中没有可比较的方向冲突。E5 没有新候选，所以没有深读。E4 和开放模型是主动关闭。上一级把 E6 写成 completed、把报告写成 complete，与这些缺项不符；本报告保持 incomplete。")
    lines.append("")
    lines.append("返回的服务端模型名没有写进上一级响应日志，日志只保留了生成文本。调用配置写的是 gpt-6.1-sol、标准层、不发送 temperature。")
    lines.append("")
    return lines


def _pair_pages(run: Path, summary: dict) -> None:
    folder = run / "reports" / "pairs"
    folder.mkdir(exist_ok=True)
    cases = summary["cases"]
    for row in summary["pairs"]:
        role = "关系标记最多的案例" if row["pair_id"] == cases["change"] else ("条件增量最小的案例" if row["pair_id"] == cases["low"] else "同样模板")
        text = "\n".join([
            f"# {row['gene_id']} / {row['disease_cui']}",
            "",
            f"pair `{row['pair_id']}`，分层 {row['stratum']}。{role}。",
            f"基因原始提及：{row['raw_gene_mentions'] or '无'}。来源：{row['gene_name_source']}。",
            f"疾病原始提及：{row['raw_disease_mentions'] or '无'}。来源：{row['disease_name_source']}。这些不是已核对规范名。",
            f"PMID：CORE {row['n_core_pmid']}，SemMedDB {row['n_semmed_pmid']}，交集 {row['n_intersection_pmid']}，并集 {row['n_union_pmid']}。",
            f"分析 finding {row['n_analysis_finding']}，no_finding {row['n_no_finding']}。问题类型 {row['question_type']}。",
            f"比较 {row['n_comparison']}：信息不足 {row['n_insufficient']}，问题不同 {row['n_different_question']}，已知条件不同 {row['n_known_difference']}，可比较 {row['n_comparable']}。",
            f"方向相似 {row['n_similar_direction']}，方向不同 {row['n_different_direction']}，聚类候选 {row['n_candidate']}。",
            f"条件元组 CORE / SemMedDB / Union = {row['n_core_condition']} / {row['n_semmed_condition']} / {row['n_union_condition']}。",
            f"等文章量 m={row['equal_article'].get('m')}，预期 {row['expected_m']}。",
            f"D 关系数 {row.get('d_relations')}。K 调用 {row.get('k_calls')}，K finding {row.get('k_findings')}。",
            "",
            "零结果和失败与其他 pair 使用同一模板。科学判断尚未人工核对。",
            "",
        ])
        (folder / f"{row['pair_id']}.md").write_text(text, encoding="utf-8")


def stage_probe(run_id: str) -> int:
    run = _run()
    from pairpilot.continue_pilot import _post
    ledger = SharedLedger(run / "audit" / "ledger.json")
    reservation = "probe-repair"
    try:
        ledger.reserve(reservation, 500, 200, "main")
    except Exception as exc:
        write_json(run / "audit" / "probe.json", {"passed": False, "status": "budget_blocked", "detail": str(exc), "at": _now()})
        return 2
    result = _post("gpt-6.1-sol", "Return one JSON object.", '{"ping":true}')
    status = result.get("status")
    if status == "quota_exhausted":
        ledger.release(reservation)
        write_json(run / "audit" / "probe.json", {"passed": False, "status": status, "at": _now(), "model_response_id": result.get("model_response_id")})
        return 2
    if status != "ok":
        if status in {"timeout_or_transport", "http_5xx"}:
            ledger.keep_unknown(reservation)
        else:
            ledger.release(reservation)
        write_json(run / "audit" / "probe.json", {"passed": False, "status": status, "at": _now()})
        return 2
    usage = result.get("usage") or {}
    ledger.settle(reservation, int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0), int(usage.get("cached_input_tokens") or 0))
    write_json(run / "audit" / "probe.json", {"passed": True, "status": "ok", "at": _now(), "model_response_id": result.get("model_response_id")})
    return 0


def stage_synth_grounded(run_id: str) -> int:
    """One synthesis per pair, with finding quotes. Count-only synthesis is not reused."""
    run = _run()
    probe = read_json(run / "audit" / "probe.json") if (run / "audit" / "probe.json").exists() else {}
    if not probe.get("passed"):
        write_json(run / "audit" / "synthesis_grounded.json", {"n_ok": 0, "n_expected": 8, "status": "blocked_probe", "input_has_quotes": True, "at": _now()})
        return 2
    from pairpilot.continue_pilot import _call, _prompt
    summary = read_json(run / "audit" / "summary.json")
    con = _con()
    findings = query_rows(con, f"SELECT * FROM read_parquet('{(PRIOR / 'tables' / 'findings.parquet').as_posix()}')")
    comparisons = query_rows(con, f"SELECT * FROM read_parquet('{(run / 'tables' / 'comparisons.parquet').as_posix()}')")
    by_id = {row["finding_id"]: row for row in findings}
    prompt = _prompt("synthesize_pair.txt")
    ok = 0
    rows = []
    for pair in summary["pairs"]:
        own = [row for row in findings if row.get("pair_id") == pair["pair_id"] and is_analysis_finding(row)][:12]
        diffs = [row for row in comparisons if row.get("pair_id") == pair["pair_id"] and row.get("condition_relation") == "known_context_difference"][:8]
        packet = {
            "pair_id": pair["pair_id"],
            "instruction": "Use only these findings and comparisons. candidate_gaps may be empty. Do not invent PMIDs, quotes, or studies.",
            "findings": [{
                "finding_id": row.get("finding_id"),
                "pmid": str(row.get("pmid")),
                "question_type": row.get("question_type"),
                "study_system": row.get("study_system"),
                "exposure_or_intervention": row.get("exposure_or_intervention"),
                "comparator": row.get("comparator"),
                "outcome": row.get("outcome"),
                "result_direction": row.get("result_direction"),
                "significance": row.get("significance"),
                "quote": _short(row.get("quote"), 240),
            } for row in own],
            "known_context_differences": [{
                "comparison_id": row.get("comparison_id"),
                "missing_fields": row.get("missing_fields"),
                "different_fields": row.get("different_fields"),
                "pmid_a": row.get("pmid_a"),
                "pmid_b": row.get("pmid_b"),
                "quote_a": _short((by_id.get(row.get("finding_a")) or {}).get("quote"), 180),
                "quote_b": _short((by_id.get(row.get("finding_b")) or {}).get("quote"), 180),
            } for row in diffs],
        }
        result = _call(run, "main", "synthesize_pair_grounded", json.dumps(packet, ensure_ascii=False), prompt)
        parsed = result.get("parsed") if isinstance(result.get("parsed"), dict) else {}
        gaps = parsed.get("candidate_gaps") if isinstance(parsed.get("candidate_gaps"), list) else []
        rows.append({"pair_id": pair["pair_id"], "status": result.get("status"), "n_gap": len(gaps)})
        if result.get("status") == "ok":
            ok += 1
        if result.get("status") in {"quota_exhausted", "not_evaluated_budget"}:
            break
    write_json(run / "audit" / "synthesis_grounded.json", {
        "n_ok": ok, "n_expected": 8, "input_has_quotes": True, "rows": rows, "at": _now(),
    })
    return 0 if ok == 8 else 2


def stage_synth(run_id: str) -> int:
    run = _run()
    probe = read_json(run / "audit" / "probe.json") if (run / "audit" / "probe.json").exists() else {}
    if not probe.get("passed"):
        write_json(run / "audit" / "synthesis.json", {"n_ok": 0, "n_expected": 8, "status": "blocked_probe", "at": _now()})
        return 2
    from pairpilot.continue_pilot import _call, _prompt
    summary = read_json(run / "audit" / "summary.json")
    prompt = _prompt("synthesize_pair.txt")
    ok = 0
    rows = []
    for pair in summary["pairs"]:
        packet = {
            "pair_id": pair["pair_id"],
            "counts": {key: pair[key] for key in ("n_analysis_finding", "n_comparable", "n_insufficient", "n_different_direction", "n_candidate")},
            "question_type": pair["question_type"],
        }
        result = _call(run, "main", "synthesize_pair", json.dumps(packet, ensure_ascii=False), prompt)
        rows.append({"pair_id": pair["pair_id"], "status": result.get("status")})
        if result.get("status") == "ok":
            ok += 1
        if result.get("status") == "quota_exhausted":
            break
    write_json(run / "audit" / "synthesis.json", {"n_ok": ok, "n_expected": 8, "rows": rows, "at": _now()})
    return 0 if ok == 8 else 2


def stage_report(run_id: str) -> int:
    run = _run()
    summary = read_json(run / "audit" / "summary.json")
    _write_report(run, summary)
    return 0


STAGES = {
    "r1": stage_r1,
    "probe": stage_probe,
    "synth": stage_synth,
    "synth_grounded": stage_synth_grounded,
    "report": stage_report,
}


def main() -> int:
    if _login():
        raise SystemExit("refusing to run the revision on the login node")
    stage = sys.argv[1]
    run_id = os.environ["PILOT_CHILD_RUN"]
    try:
        if stage == "init":
            init_run(run_id)
            return 0
        return STAGES[stage](run_id)
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

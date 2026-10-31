#!/usr/bin/env python3
"""diagnostic_v1 child of rev2. comparison_gap_v2 tables are read, not rewritten."""

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

from pairpilot.comparison import build_candidates, is_analysis_finding  # noqa: E402
from pairpilot.config import digest, load_bundle  # noqa: E402
from pairpilot.diagnostic import (  # noqa: E402
    VERSION,
    comparison_role,
    diagnostic_dimension,
    e5_block_reasons,
    eligible_pair,
    entity_identity,
    freeze_e5,
    logical_task_id,
    normalize_tuple,
    select_comparisons,
    trace_comparison,
    view_allows,
)
from pairpilot.experiments.e1 import condition_key  # noqa: E402
from pairpilot.experiments.e5 import select_new  # noqa: E402
from pairpilot.ledger import LIMITS, available_usd  # noqa: E402
from pairpilot.repair_pilot import _esearch, _read_jsonl  # noqa: E402
from pairpilot.store import append_jsonl, connect, query_rows, read_json, write_json, write_parquet  # noqa: E402

REV2 = Path("/projects/b5ac/project_2/results/03_integrated_analysis/pilot_study_v1/runs/20261016T005500Z_rev2")
PRIOR = Path("/projects/b5ac/project_2/results/03_integrated_analysis/pilot_study_v1/runs/20261015T221200Z_b20cont")
PREP = Path("/projects/b5ac/project_2/results/03_integrated_analysis/pilot_study_v1/runs/20261015T204633Z_0d9b9724")
RUNS = REV2.parent
DIAG_PAIRS = ("8318__C2239176", "768__C0376358", "3604__C0007131")
COMPARE_PAIRS = {"8318__C2239176", "768__C0376358"}
E6_PAIRS = ("3604__C0007131", "3090__C0006142")
SEED = 20261015


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _login() -> bool:
    return os.uname().nodename.startswith("login")


def _con():
    tmp = Path(os.environ.get("SLURM_TMPDIR") or f"/tmp/{os.environ.get('USER', 'pilot')}/pairpilot")
    return connect(threads=max(1, int(os.environ.get("SLURM_CPUS_PER_TASK") or 2)), memory="16GB", tmp=tmp)


def _run() -> Path:
    path = RUNS / os.environ["PILOT_CHILD_RUN"]
    if not path.exists():
        raise SystemExit(f"diagnostic run is missing: {path}")
    return path


def _short(value, limit: int = 240) -> str:
    return " ".join(str(value or "").split())[:limit]


def _cell(value) -> str:
    return _short(value, 180).replace("|", "/")


def init_run(run_id: str) -> Path:
    run = RUNS / run_id
    run.mkdir(parents=False, exist_ok=True)
    for name in ("audit", "tables", "experiments", "reports", "requests", "responses", "jobs"):
        (run / name).mkdir(exist_ok=True)
    if not (run / "manifest.json").exists():
        write_json(run / "manifest.json", {
            "run_id": run_id,
            "parent_run_id": REV2.name,
            "model_run_id": PRIOR.name,
            "preparation_run_id": PREP.name,
            "science_version": VERSION,
            "baseline_rule": "comparison_gap_v2",
            "created_utc": _now(),
            "status": "incomplete",
            "pilot_status": "incomplete",
            "input_registry": {
                "findings": str(PRIOR / "tables" / "findings.parquet"),
                "v2_comparisons": str(REV2 / "tables" / "comparisons.parquet"),
                "grounded_gaps": str(REV2 / "tables" / "grounded_gaps.parquet"),
                "documents": str(PREP / "inputs" / "documents.parquet"),
                "pair_pmids": str(PREP / "inputs" / "pair_pmids.parquet"),
                "responses": str(PRIOR / "responses" / "responses.jsonl"),
            },
        })
    ledger = run / "audit" / "ledger.json"
    if not ledger.exists():
        prior = json.loads((REV2 / "audit" / "ledger.json").read_text(encoding="utf-8"))
        available = available_usd(prior)
        spent = float(prior["spent_usd"])
        hold = None if available is None else LIMITS["max_usd"] - spent - available
        write_json(ledger, {
            "spent_usd": spent,
            "reserved_usd": 0.0,
            "billing_unknown_usd": hold,
            "attempts": prior.get("attempts") or 0,
            "input_tokens": prior.get("input_tokens") or 0,
            "output_tokens": prior.get("output_tokens") or 0,
            "canary_spent_usd": prior.get("canary_spent_usd") or 0.0,
            "canary_reserved_usd": 0.0,
            "canary_attempts": prior.get("canary_attempts") or 0,
            "canary_input_tokens": prior.get("canary_input_tokens") or 0,
            "canary_output_tokens": prior.get("canary_output_tokens") or 0,
            "reservations": {},
            "opening_source": str(REV2 / "audit" / "ledger.json"),
            "opened_utc": _now(),
        })
    return run


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return query_rows(_con(), f"SELECT * FROM read_parquet('{path.as_posix()}')")


def _stopped(run: Path) -> bool:
    return (run / "audit" / "provider_stop.json").exists()


def _call_task(run: Path, task: str, user: str, prompt_name: str, logical_id: str, max_output: int) -> dict:
    from pairpilot.continue_pilot import _call, _prompt
    if _stopped(run):
        return {"status": "stopped_after_quota", "parsed": None}
    prompt = _prompt(prompt_name)
    result = _call(run, "main", task, user, prompt, max_output=max_output)
    if result.get("status") == "unparsed":
        result = _call(run, "main", task + "_repair", user + "\nReturn one JSON object only.", prompt, replicate_id=logical_id + ":repair", max_output=max_output)
    append_jsonl(run / "audit" / "task_log.jsonl", {
        "logical_task_id": logical_id, "task": task, "status": result.get("status"),
        "returned_model": result.get("returned_model") or "unknown", "at": _now(),
    })
    if result.get("status") == "quota_exhausted":
        write_json(run / "audit" / "provider_stop.json", {"at": _now(), "status": "quota_exhausted"})
    return result


def _done_ids(run: Path) -> set[str]:
    ids = set()
    path = run / "audit" / "task_log.jsonl"
    if not path.exists():
        return ids
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") == "ok":
            ids.add(row.get("logical_task_id"))
    return ids


def stage_j0(run_id: str) -> int:
    run = init_run(run_id)
    findings = _rows(PRIOR / "tables" / "findings.parquet")
    roles = Counter(comparison_role(row) for row in findings)
    analysis = [row for row in findings if is_analysis_finding(row)]
    original = [row for row in findings if row.get("question_type") and row.get("discourse_role") != "background"]
    rules = load_bundle()["pilot"]["required_comparability_fields"]
    config_hash = digest({"rule": "comparison_gap_v2", "fields": rules})
    v2_rows = build_candidates(analysis, config_hash, rules)
    original_rows = build_candidates(original, config_hash, rules)
    v2_ids = {row["comparison_id"] for row in v2_rows}
    original_ids = {row["comparison_id"] for row in original_rows}
    traces = []
    by_id = {row["finding_id"]: row for row in analysis}
    branch_counts = Counter()
    pair_insufficient = Counter()
    for row in v2_rows:
        left = by_id[row["finding_a"]]
        right = by_id[row["finding_b"]]
        traced = trace_comparison(left, right, rules)
        traced.update({
            "comparison_id": row["comparison_id"],
            "pair_id": row["pair_id"],
            "finding_a": row["finding_id"] if False else row["finding_a"],
            "finding_b": row["finding_b"],
            "pmid_a": row["pmid_a"],
            "pmid_b": row["pmid_b"],
        })
        traces.append(traced)
        branch_counts[traced["deciding_branch"]] += 1
        if traced["condition_relation"] == "insufficient_context":
            pair_insufficient[row["pair_id"]] += 1
    write_parquet(run / "tables" / "baseline_rule_trace.parquet", traces)
    others = []
    for row in analysis:
        if row.get("question_type") != "other":
            continue
        others.append({
            "finding_id": row["finding_id"], "pair_id": row["pair_id"], "pmid": str(row["pmid"]),
            "discourse_role": row.get("discourse_role"), "study_system": row.get("study_system"),
            "exposure_or_intervention": row.get("exposure_or_intervention"),
            "comparator": row.get("comparator"), "outcome": row.get("outcome"),
            "quote": _short(row.get("quote"), 300), "located": row.get("located"),
        })
    write_parquet(run / "tables" / "other_inventory.parquet", others)
    raw_keys = {" | ".join(condition_key(row)) for row in analysis}
    norm_keys = {normalize_tuple(key) for key in raw_keys}
    links = _rows(PREP / "inputs" / "pair_pmids.parquet")
    diag_links = [row for row in links if row["pair_id"] in DIAG_PAIRS]
    tasks = []
    for row in sorted(diag_links, key=lambda item: (item["pair_id"], str(item["pmid"]), item["source"])):
        tasks.append({
            "logical_task_id": logical_task_id(row["pair_id"], str(row["pmid"])),
            "pair_id": row["pair_id"], "pmid": str(row["pmid"]), "source": row["source"],
        })
    unique_tasks = {}
    for row in tasks:
        unique_tasks.setdefault(row["logical_task_id"], row)
    task_list = list(unique_tasks.values())
    write_json(run / "audit" / "p1_tasks.json", task_list)
    _e3(run)
    _deferred(run, links)
    summary = {
        "n_raw_rows": len(findings),
        "roles": dict(roles),
        "n_analysis_v2": len(analysis),
        "n_original_filter": len(original),
        "finding_id_sets_equal": {row["finding_id"] for row in analysis} == {row["finding_id"] for row in original},
        "n_v2_comparisons": len(v2_rows),
        "n_original_filter_comparisons": len(original_rows),
        "comparison_id_sets_equal": v2_ids == original_ids,
        "why_5812": "Both generators drop empty question_type and background before pairing. The two no_finding rows have no question_type, so they never entered either comparison set.",
        "branches": dict(branch_counts),
        "insufficient_by_pair": dict(pair_insufficient),
        "n_other": len(others),
        "baseline_raw_tuple_count": len(raw_keys),
        "punctuation_normalized_tuple_count": len(norm_keys),
        "diagnostic_pair_pmid_rows": len(diag_links),
        "diagnostic_unique_pmids": len({str(row["pmid"]) for row in diag_links}),
        "p1_logical_tasks": len(task_list),
    }
    write_json(run / "audit" / "p0_summary.json", summary)
    write_json(run / "audit" / "expected_manifest.json", {
        "version": VERSION,
        "seed": SEED,
        "p1_max_calls": len(task_list),
        "p2_max_finding_pairs": 24,
        "p2_max_per_pair": 12,
        "review_max_pair_batches": 8,
        "e5_max_candidates": 3,
        "e5_max_new_abstracts": 30,
        "k_e6": "deferred_pending_protocol",
        "max_output_p1": 6000,
        "max_output_p2": 8000,
    })
    return 0


def _e3(run: Path) -> None:
    frozen = read_json(PREP / "inputs" / "frozen_subsets.json").get("E3") or []
    responses = _read_jsonl(PRIOR / "responses" / "responses.jsonl")
    source = [row for row in responses if row.get("task") == "assess_scope_s"]
    abstract = [row for row in responses if row.get("task") == "assess_scope_a"]
    rows = []
    for index, item in enumerate(frozen):
        left = source[index] if index < len(source) else {}
        right = abstract[index] if index < len(abstract) else {}
        rows.append({
            "pair_id": item.get("pair_id"),
            "pmid": str(item.get("pmid")),
            "s_result_field": _field_state(left, "result_scope"),
            "s_conclusion_field": _field_state(left, "conclusion_scope"),
            "s_evidence": _evidence_state(left),
            "s_judgement": _judgement_state(left),
            "a_result_field": _field_state(right, "result_scope"),
            "a_conclusion_field": _field_state(right, "conclusion_scope"),
            "a_evidence": _evidence_state(right),
            "a_judgement": _judgement_state(right),
            "text_version_relation": "not_recorded",
        })
    write_parquet(run / "tables" / "scope_transfer_revised.parquet", rows)
    write_json(run / "audit" / "e3_summary.json", {
        "n": len(rows),
        "s_result_nonempty": sum(1 for row in rows if row["s_result_field"] == "nonempty"),
        "s_result_not_judged": sum(1 for row in rows if row["s_result_field"] == "not_judged"),
        "s_result_not_recorded": sum(1 for row in rows if row["s_result_field"] == "not_recorded"),
        "a_result_nonempty": sum(1 for row in rows if row["a_result_field"] == "nonempty"),
        "a_judgement_not_judged": sum(1 for row in rows if row["a_judgement"] == "not_judged"),
        "note": "Field presence is not locatability and not a sufficient judgement. These columns are not an improvement rate.",
    })


def _field_state(response: dict, field: str) -> str:
    parsed = response.get("parsed") if isinstance(response.get("parsed"), dict) else None
    if not parsed or field not in parsed:
        return "not_recorded"
    value = parsed.get(field)
    if value in (None, "", "not_judged"):
        return "not_judged"
    if isinstance(value, dict):
        return "nonempty" if any(value.values()) else "empty"
    if isinstance(value, str) and value.strip():
        return "nonempty"
    return "present"


def _evidence_state(response: dict) -> str:
    parsed = response.get("parsed") if isinstance(response.get("parsed"), dict) else None
    if not parsed or "evidence" not in parsed:
        return "not_recorded"
    return "recorded" if parsed.get("evidence") else "empty"


def _judgement_state(response: dict) -> str:
    parsed = response.get("parsed") if isinstance(response.get("parsed"), dict) else None
    if not parsed:
        return "not_recorded"
    differences = parsed.get("differences")
    if differences in (None, "", "not_judged"):
        return "not_judged"
    if isinstance(differences, list) and differences:
        return "recorded"
    if isinstance(differences, str) and differences.strip():
        return "recorded"
    return "not_recorded"


def _deferred(run: Path, links: list[dict]) -> None:
    small = [row["pair_id"] for row in _rows(REV2 / "tables" / "pair_names.parquet") if str(row.get("stratum") or "").endswith("small")]
    e6 = [row for row in links if row["pair_id"] in E6_PAIRS]
    pmids = sorted({(row["pair_id"], str(row["pmid"])) for row in e6})
    write_json(run / "audit" / "k_e6_manifest.json", {
        "k_status": "deferred_pending_protocol",
        "k_existing_extract_calls": 19,
        "k_pairs": small,
        "k_note": "A later K run on diagnostic_v1 cannot be compared with old F as if only the KG input differed.",
        "e6_status": "deferred_pending_protocol",
        "e6_logical_units": [{"pair_id": pair, "pmid": pmid, "replicates": [1, 2]} for pair, pmid in pmids],
        "e6_logical_extracts": len(pmids) * 2,
        "e6_prior_failure": "quota_exhausted on finding-level calls; not relabeled",
        "rough_cost_if_run_later_usd": "not a reservation; two extracts per listed PMID plus at most 24 perturbation calls",
    })


def _documents() -> dict[str, dict]:
    return {str(row["pmid"]): row for row in _rows(PREP / "inputs" / "documents.parquet")}


def _aliases() -> dict[str, dict]:
    path = REV2 / "tables" / "pair_names.parquet"
    if not path.exists():
        path = PRIOR / "tables" / "pair_names.parquet"
    return {row["pair_id"]: row for row in _rows(path)}


def _p1_user(task: dict, doc: dict, alias: dict) -> str:
    return json.dumps({
        "pair_id": task["pair_id"],
        "pmid": task["pmid"],
        "gene_id": alias.get("gene_id"),
        "disease_cui": alias.get("disease_cui"),
        "raw_gene_mentions": alias.get("gene_name") or alias.get("raw_gene_mentions"),
        "raw_disease_mentions": alias.get("disease_name") or alias.get("raw_disease_mentions"),
        "mention_source": alias.get("gene_name_source"),
        "title": doc.get("title") or "",
        "abstract": doc.get("text") or "",
        "text_version_id": doc.get("text_version_id") or "",
    }, ensure_ascii=False)


def _run_p1(run: Path, tasks: list[dict]) -> int:
    docs = _documents()
    names = _aliases()
    done = _done_ids(run)
    for task in tasks:
        if task["logical_task_id"] in done:
            continue
        doc = docs.get(task["pmid"])
        if not doc:
            append_jsonl(run / "audit" / "task_log.jsonl", {"logical_task_id": task["logical_task_id"], "status": "text_unavailable", "at": _now()})
            continue
        result = _call_task(run, "diagnostic_reread", _p1_user(task, doc, names.get(task["pair_id"], {})), "diagnostic_reread.txt", task["logical_task_id"], 6000)
        _store_reread(run, task, doc, result)
        if result.get("status") in {"quota_exhausted", "not_evaluated_budget", "stopped_after_quota"}:
            return 2
    return 0


def _store_reread(run: Path, task: dict, doc: dict, result: dict) -> None:
    parsed = result.get("parsed") if isinstance(result.get("parsed"), dict) else {}
    findings = parsed.get("findings") if isinstance(parsed.get("findings"), list) else []
    rows = []
    claims = []
    if result.get("status") == "ok" and not findings:
        rows.append({"logical_task_id": task["logical_task_id"], "pair_id": task["pair_id"], "pmid": task["pmid"], "status": "no_finding", "science_version": VERSION})
    for index, finding in enumerate(findings):
        if not isinstance(finding, dict):
            continue
        quote = ""
        evidence = finding.get("evidence") or []
        if evidence and isinstance(evidence[0], dict):
            quote = evidence[0].get("quote") or ""
        entity = entity_identity(" ".join(str(finding.get(name) or "") for name in ("measured_entity", "intervened_entity", "reagent_or_construct")))
        flat = {
            "logical_task_id": task["logical_task_id"],
            "pair_id": task["pair_id"],
            "pmid": task["pmid"],
            "local": index,
            "status": "parsed",
            "science_version": VERSION,
            "text_version_id": doc.get("text_version_id") or "",
            "primary_question_type": finding.get("primary_question_type") or "",
            "raw_question": _short(finding.get("raw_question"), 300),
            "discourse_role": finding.get("discourse_role") or "",
            "target_relation": finding.get("target_relation") or "",
            "measured_entity": finding.get("measured_entity") or "",
            "intervened_entity": finding.get("intervened_entity") or "",
            "reagent_or_construct": finding.get("reagent_or_construct") or "",
            "entity_kind": entity["kind"],
            "entity_gene_id": entity["gene_id"],
            "outcome": finding.get("outcome") or "",
            "comparator": finding.get("comparator") or "",
            "measurement_compartment": finding.get("measurement_compartment") or "",
            "result_direction": finding.get("result_direction") or "",
            "quote": _short(quote, 300),
        }
        if finding.get("discourse_role") in {"author_interpretation", "recommendation"}:
            claims.append(flat)
        rows.append(flat)
    if result.get("status") != "ok":
        rows.append({"logical_task_id": task["logical_task_id"], "pair_id": task["pair_id"], "pmid": task["pmid"], "status": result.get("status"), "science_version": VERSION})
    existing = _rows(run / "tables" / "diagnostic_findings.parquet")
    write_parquet(run / "tables" / "diagnostic_findings.parquet", existing + rows)
    if claims:
        write_parquet(run / "tables" / "diagnostic_claims.parquet", _rows(run / "tables" / "diagnostic_claims.parquet") + claims)


def stage_j1(run_id: str) -> int:
    run = _run()
    tasks = read_json(run / "audit" / "p1_tasks.json")
    if not tasks:
        return 2
    code = _run_p1(run, tasks[:1])
    ok = tasks[0]["logical_task_id"] in _done_ids(run)
    write_json(run / "audit" / "p1_interface.json", {"passed": ok, "logical_task_id": tasks[0]["logical_task_id"], "at": _now()})
    return 0 if ok else 2


def stage_j2(run_id: str) -> int:
    run = _run()
    if not read_json(run / "audit" / "p1_interface.json").get("passed"):
        return 2
    tasks = read_json(run / "audit" / "p1_tasks.json")
    code = _run_p1(run, tasks[1:])
    _align(run)
    return code


def _align(run: Path) -> None:
    old = [row for row in _rows(PRIOR / "tables" / "findings.parquet") if row.get("pair_id") in DIAG_PAIRS and is_analysis_finding(row)]
    new = [row for row in _rows(run / "tables" / "diagnostic_findings.parquet") if row.get("status") == "parsed"]
    rows = []
    for item in old:
        same = [row for row in new if row.get("pair_id") == item["pair_id"] and str(row.get("pmid")) == str(item["pmid"])]
        quote = _short(item.get("quote"), 80)
        match = next((row for row in same if quote and quote[:40] in (row.get("quote") or "")), None)
        if match is None and same:
            match = same[0]
            reason = "unresolved"
        elif match is None:
            reason = "unresolved"
        else:
            reason = _alignment_reason(item, match)
        rows.append({
            "old_finding_id": item["finding_id"],
            "pmid": str(item["pmid"]),
            "pair_id": item["pair_id"],
            "new_local": None if match is None else match.get("local"),
            "reason": reason,
            "old_question_type": item.get("question_type"),
            "new_primary_question_type": None if match is None else match.get("primary_question_type"),
        })
    write_parquet(run / "tables" / "finding_alignment.parquet", rows)
    write_json(run / "audit" / "alignment_summary.json", {"n_old": len(old), "reasons": dict(Counter(row["reason"] for row in rows))})


def _alignment_reason(old: dict, new: dict) -> str:
    if not old.get("outcome") and new.get("outcome"):
        return "candidate_extraction_omission"
    if old.get("question_type") == "other" and new.get("primary_question_type") not in {"", "other", "unclear"}:
        return "representation_or_rule_issue"
    if new.get("primary_question_type") in {"other", "unclear"} and not new.get("outcome"):
        return "not_found_in_abstract_by_model"
    return "unresolved"


def _rebuild_findings(run: Path) -> None:
    """The saved reread is authoritative. The first flat table omitted status and reporting fields."""
    text_fields = (
        "primary_question_type", "raw_question", "discourse_role", "target_relation",
        "measured_entity", "intervened_entity", "reagent_or_construct", "molecular_form",
        "study_system", "disease_site_or_subtype", "exposure_or_intervention",
        "comparator", "outcome", "measurement_compartment", "result_direction",
        "significance", "causal_wording", "author_conclusion",
    )
    status_fields = (
        "comparator", "outcome", "measurement_compartment", "result_direction",
        "measured_entity", "intervened_entity", "reagent_or_construct",
    )
    rows = []
    claims = []
    entities = []
    seen = set()
    for record in _read_jsonl(run / "audit" / "reread_units.jsonl"):
        logical_id = record.get("logical_task_id") or ""
        if not logical_id or logical_id in seen or "|supplemental" in logical_id:
            continue
        seen.add(logical_id)
        parsed = record.get("parsed") if isinstance(record.get("parsed"), dict) else {}
        findings = parsed.get("findings") if isinstance(parsed.get("findings"), list) else []
        if not findings:
            rows.append({
                "logical_task_id": logical_id, "pair_id": record.get("pair_id"), "pmid": str(record.get("pmid")),
                "local": 0, "status": "no_finding", "science_version": VERSION,
                "text_version_id": record.get("text_version_id") or "",
            })
            continue
        for index, finding in enumerate(findings):
            if not isinstance(finding, dict):
                continue
            evidence = finding.get("evidence") if isinstance(finding.get("evidence"), list) else []
            quote = evidence[0].get("quote") if evidence and isinstance(evidence[0], dict) else ""
            mention = " ".join(str(finding.get(name) or "") for name in ("measured_entity", "intervened_entity", "reagent_or_construct"))
            entity = entity_identity(mention)
            species = finding.get("species")
            flat = {
                "logical_task_id": logical_id,
                "pair_id": record.get("pair_id"),
                "pmid": str(record.get("pmid")),
                "local": index,
                "status": "parsed",
                "science_version": VERSION,
                "text_version_id": record.get("text_version_id") or "",
                "species": ", ".join(species) if isinstance(species, list) else (species or ""),
                "sample_size": _short(finding.get("sample_size"), 80),
                "quote": _short(quote, 500),
                "entity_kind": entity["kind"],
                "entity_gene_id": entity["gene_id"],
            }
            for name in text_fields:
                flat[name] = _short(finding.get(name), 400)
            for name in status_fields:
                flat[f"{name}_reporting_status"] = finding.get(f"{name}_reporting_status") or ""
            rows.append(flat)
            entities.append({
                "pair_id": flat["pair_id"], "pmid": flat["pmid"], "local": index,
                "mention": _short(mention, 200), "entity_kind": entity["kind"], "entity_gene_id": entity["gene_id"],
            })
            if finding.get("discourse_role") in {"author_interpretation", "recommendation"} or finding.get("author_conclusion"):
                claims.append(flat)
    write_parquet(run / "tables" / "diagnostic_findings.parquet", rows)
    write_parquet(run / "tables" / "diagnostic_entities.parquet", entities)
    if claims:
        write_parquet(run / "tables" / "diagnostic_claims.parquet", claims)
    _align(run)


def stage_j3(run_id: str) -> int:
    run = _run()
    _rebuild_findings(run)
    findings = [row for row in _rows(run / "tables" / "diagnostic_findings.parquet") if row.get("status") == "parsed" and row.get("pair_id") in COMPARE_PAIRS]
    for row in findings:
        row["question_type"] = row.get("primary_question_type")
    traces = _rows(run / "tables" / "baseline_rule_trace.parquet")
    insufficient_papers = {
        tuple(sorted((str(row["pmid_a"]), str(row["pmid_b"]))))
        for row in traces if row.get("condition_relation") == "insufficient_context" and row.get("pair_id") in COMPARE_PAIRS
    }
    candidates = []
    for i, left in enumerate(findings):
        for right in findings[i + 1:]:
            if not eligible_pair(left, right):
                continue
            dims = {field: diagnostic_dimension(left, right, field) for field in ("outcome", "comparator", "measurement_compartment")}
            paper = tuple(sorted((str(left["pmid"]), str(right["pmid"]))))
            if paper in insufficient_papers and (left.get("quote") or right.get("quote")):
                layer = "old_insufficient_with_text"
            elif "known_difference" in dims.values():
                layer = "known_difference"
            elif left.get("result_direction") and right.get("result_direction"):
                layer = "apparent_direction"
            else:
                layer = "other_same_question"
            key = f"{left['logical_task_id']}:{left.get('local')}|{right['logical_task_id']}:{right.get('local')}"
            candidates.append({
                "candidate_key": key,
                "comparison_id": key,
                "pair_id": left["pair_id"],
                "pmid_a": str(left["pmid"]),
                "pmid_b": str(right["pmid"]),
                "finding_a": f"{left['pmid']}:{left.get('local')}",
                "finding_b": f"{right['pmid']}:{right.get('local')}",
                "primary_question_type": left.get("primary_question_type"),
                "layer": layer,
                "quote_a": left.get("quote") or "",
                "quote_b": right.get("quote") or "",
            })
    selected = select_comparisons(candidates, SEED)
    for row in candidates:
        row["selected"] = any(item["candidate_key"] == row["candidate_key"] for item in selected)
    write_parquet(run / "tables" / "comparison_candidates.parquet", candidates)
    write_parquet(run / "tables" / "comparison_selected.parquet", selected)
    if not selected:
        write_json(run / "audit" / "p2_summary.json", {"n_candidates": len(candidates), "n_selected": 0, "status": "no_eligible_pair"})
        return 0
    docs = _documents()
    groups = {}
    for row in selected:
        groups.setdefault((row["pair_id"], row["pmid_a"], row["pmid_b"]), []).append(row)
    judgements = []
    for (pair, pmid_a, pmid_b), rows in groups.items():
        by_local = {(str(item["pmid"]), str(item.get("local"))): item for item in findings if item["pair_id"] == pair}
        packets = []
        for row in rows:
            packets.append({
                "comparison_id": row["comparison_id"],
                "finding_a": by_local.get((row["pmid_a"], row["finding_a"].split(":")[-1]), {}),
                "finding_b": by_local.get((row["pmid_b"], row["finding_b"].split(":")[-1]), {}),
            })
        user = json.dumps({
            "pair_id": pair,
            "abstract_a": {"pmid": pmid_a, "text": (docs.get(pmid_a) or {}).get("text") or ""},
            "abstract_b": {"pmid": pmid_b, "text": (docs.get(pmid_b) or {}).get("text") or ""},
            "finding_pairs": packets,
        }, ensure_ascii=False)
        result = _call_task(run, "diagnostic_compare", user, "diagnostic_compare.txt", f"p2|{pair}|{pmid_a}|{pmid_b}", 8000)
        parsed = result.get("parsed") if isinstance(result.get("parsed"), dict) else {}
        returned = {item.get("comparison_id") or item.get("candidate_key"): item for item in (parsed.get("comparisons") or []) if isinstance(item, dict)}
        for row in rows:
            item = returned.get(row["candidate_key"], {})
            judgements.append({
                "candidate_key": row["candidate_key"],
                "pair_id": pair,
                "pmid_a": pmid_a,
                "pmid_b": pmid_b,
                "primary_question_type": row.get("primary_question_type") or "",
                "status": result.get("status"),
                "can_describe_relation": item.get("can_describe_relation"),
                "can_compare_direction": item.get("can_compare_direction"),
                "can_compare_magnitude": item.get("can_compare_magnitude"),
                "direction_relation": item.get("direction_relation") or "",
                "shared_question": _short(item.get("shared_question"), 300),
                "relation_description": _short(item.get("relation_description"), 400),
                "dimensions": item.get("dimensions") if isinstance(item.get("dimensions"), dict) else {},
                "missing_information": _short(item.get("missing_information"), 400),
                "why_it_blocks": _short(item.get("why_it_blocks"), 400),
                "inference_limit": _short(item.get("inference_limit"), 300),
            })
        if result.get("status") in {"quota_exhausted", "not_evaluated_budget", "stopped_after_quota"}:
            break
    write_parquet(run / "tables" / "diagnostic_comparisons.parquet", judgements)
    _source_views(run, findings, judgements)
    write_json(run / "audit" / "p2_summary.json", {
        "n_candidates": len(candidates),
        "n_selected": len(selected),
        "n_judged": sum(1 for row in judgements if row["status"] == "ok"),
        "describe": sum(1 for row in judgements if row.get("can_describe_relation") in (True, "true")),
        "direction": sum(1 for row in judgements if row.get("can_compare_direction") in (True, "true")),
        "magnitude": sum(1 for row in judgements if row.get("can_compare_magnitude") in (True, "true")),
    })
    return 0 if judgements and judgements[-1]["status"] not in {"quota_exhausted", "not_evaluated_budget"} else 2


def _source_views(run: Path, findings: list[dict], judgements: list[dict]) -> None:
    links = _rows(PREP / "inputs" / "pair_pmids.parquet")
    rows = []
    for pair in COMPARE_PAIRS:
        core = {str(row["pmid"]) for row in links if row["pair_id"] == pair and row["source"] == "CORE"}
        sem = {str(row["pmid"]) for row in links if row["pair_id"] == pair and row["source"] == "SemMedDB"}
        own = [row for row in findings if row["pair_id"] == pair]
        for name, pmids in (("CORE", core), ("SemMedDB", sem), ("Union", core | sem)):
            usable = [row for row in judgements if row["pair_id"] == pair and view_allows(row, pmids) and row["status"] == "ok"]
            rows.append({
                "pair_id": pair, "view": name, "n_pmid": len(pmids),
                "n_finding": sum(1 for row in own if str(row["pmid"]) in pmids),
                "n_selected_comparison_in_view": len(usable),
                "n_can_describe": sum(1 for row in usable if row.get("can_describe_relation") in (True, "true")),
            })
    write_parquet(run / "tables" / "diagnostic_source_views.parquet", rows)


def stage_j4(run_id: str) -> int:
    run = _run()
    gaps = _rows(REV2 / "tables" / "grounded_gaps.parquet")
    prepared = []
    for index, row in enumerate(gaps):
        entity = entity_identity(row.get("question") or "")
        prepared.append({
            "candidate_id": f"g{index:02d}",
            "pair_id": row.get("pair_id"),
            "target_question": row.get("question"),
            "missing_dimension": row.get("missing_dimension"),
            "quote": row.get("quote"),
            "gap_state": row.get("gap_state"),
            "search_question": row.get("question"),
            "evidence_refs": row.get("quote"),
            "entity_conflict": entity["kind"] == "ligand_tnfsf9" and row.get("pair_id") == "3604__C0007131",
            "only_original_methods_omission": False,
        })
    by_pair = {}
    for row in prepared:
        by_pair.setdefault(row["pair_id"], []).append(row)
    docs = _documents()
    links = _rows(PREP / "inputs" / "pair_pmids.parquet")
    reviews = []
    for pair, rows in sorted(by_pair.items()):
        pmids = sorted({str(item["pmid"]) for item in links if item["pair_id"] == pair})
        packet = [{"pmid": pmid, "text": (docs.get(pmid) or {}).get("text") or ""} for pmid in pmids]
        user = json.dumps({"pair_id": pair, "candidates": rows, "abstracts": packet}, ensure_ascii=False)
        if len(user) > 45000:
            user = json.dumps({"pair_id": pair, "candidates": rows, "abstracts": packet[:8], "input_truncated_to_registered_prefix": True}, ensure_ascii=False)
        result = _call_task(run, "candidate_review", user, "candidate_review.txt", f"review|{pair}", 5000)
        parsed = result.get("parsed") if isinstance(result.get("parsed"), dict) else {}
        returned = {item.get("candidate_id"): item for item in (parsed.get("reviews") or []) if isinstance(item, dict)}
        for row in rows:
            review = returned.get(row["candidate_id"], {})
            merged = dict(row)
            merged["searchability"] = review.get("searchability") or ""
            merged["only_original_methods_omission"] = bool(review.get("only_original_methods_omission"))
            merged["entity_conflict"] = bool(row["entity_conflict"] or review.get("entity_conflict"))
            merged["search_question"] = review.get("search_question") or row["search_question"]
            merged["target_question"] = review.get("target_question") or row["target_question"]
            merged["review_status"] = result.get("status")
            merged["block_reasons"] = ",".join(e5_block_reasons(merged))
            reviews.append(merged)
        if result.get("status") in {"quota_exhausted", "not_evaluated_budget", "stopped_after_quota"}:
            break
    write_parquet(run / "tables" / "candidate_review.parquet", reviews)
    eligible = [row for row in reviews if not e5_block_reasons(row)]
    frozen = freeze_e5(eligible, SEED, 3)
    write_json(run / "audit" / "e5_frozen.json", frozen)
    write_json(run / "audit" / "review_summary.json", {
        "n_input": len(gaps),
        "n_reviewed": len(reviews),
        "n_eligible": len(eligible),
        "n_frozen": len(frozen),
        "blocked": dict(Counter(reason for row in reviews for reason in (row.get("block_reasons") or "").split(",") if reason)),
    })
    if not frozen:
        write_json(run / "audit" / "e5_status.json", {"status": "eligibility_empty", "at": _now()})
    return 0


def _retire_invalid_search(run: Path) -> int:
    """The first search used the pair id as a PubMed term. Keep those rows, and do not treat them as evidence."""
    path = run / "experiments" / "e5_search.parquet"
    if not path.exists():
        return 0
    rows = _rows(path)
    if not rows or not any("__C" in str(row.get("query") or "") for row in rows):
        archived = _rows(run / "experiments" / "e5_search_invalid_alias.parquet")
        return len({str(row.get("pmid")) for row in archived if row.get("status") == "included" and row.get("pmid")})
    for name in ("e5_search", "e5_extracts", "e5_reviews"):
        source = run / "experiments" / f"{name}.parquet"
        target = run / "experiments" / f"{name}_invalid_alias.parquet"
        if source.exists() and not target.exists():
            source.replace(target)
    write_json(run / "audit" / "e5_query_correction.json", {
        "reason": "pair_names were read from rev2, where that table is absent, so the query used the pair id",
        "invalid_included": sum(1 for row in rows if row.get("status") == "included"),
        "at": _now(),
    })
    return sum(1 for row in rows if row.get("status") == "included")


def _alias_parts(value: str) -> list[str]:
    seen = []
    for part in str(value or "").split("|"):
        part = " ".join(part.split())
        if part and part.lower() not in {item.lower() for item in seen}:
            seen.append(part)
    return seen


def _or_terms(parts: list[str]) -> str:
    rendered = []
    for part in parts:
        rendered.append(f'"{part}"' if " " in part else part)
    return " OR ".join(rendered)


def _content_terms(question: str) -> list[str]:
    """Keep only short terms that already occur in the frozen question."""
    low = question.lower()
    preferred = (
        "tnbc", "methylation", "silencing", "hbv", "hypoxia", "stroma", "stromal",
        "ihc", "protein", "compartment", "localization",
    )
    return [term for term in preferred if term in low][:4]


def _literal_queries(gene_parts: list[str], disease_parts: list[str], question: str) -> list[dict]:
    genes = _or_terms(gene_parts[:4])
    diseases = _or_terms(disease_parts[:4])
    broad = f"({genes}) AND ({diseases})"
    extra = _or_terms(_content_terms(question))
    specific = f"{broad} AND ({extra})" if extra else broad
    return [
        {"query_id": "specific", "text": specific, "query_form": "alias_and_disease"},
        {"query_id": "broad", "text": broad, "query_form": "alias_and_disease"},
    ]


def stage_j5(run_id: str) -> int:
    run = _run()
    if not (run / "tables" / "candidate_review.parquet").exists():
        write_json(run / "audit" / "e5_status.json", {"status": "eligibility_empty", "reason": "review_table_missing", "at": _now()})
        return 0
    reviews = _rows(run / "tables" / "candidate_review.parquet")
    for row in reviews:
        if row.get("searchability") in {"only_original_paper_detail", "not_searchable"}:
            row["only_original_methods_omission"] = True
        if row.get("review_status") not in {"ok", "truncated"}:
            row["search_question"] = ""
    frozen = freeze_e5([row for row in reviews if not e5_block_reasons(row)], SEED, 3)
    write_json(run / "audit" / "e5_frozen.json", frozen)
    if not frozen:
        write_json(run / "audit" / "e5_status.json", {"status": "eligibility_empty", "n_reviewed": len(reviews), "at": _now()})
        return 0
    if (run / "audit" / "e5_short_query.json").exists():
        return 0
    names = _aliases()
    invalid_n = _retire_invalid_search(run)
    known = {str(row["pmid"]) for row in _rows(PREP / "inputs" / "pair_pmids.parquet")}
    for row in _rows(run / "experiments" / "e5_search_invalid_alias.parquet"):
        if row.get("pmid"):
            known.add(str(row["pmid"]))
    prior = _rows(run / "experiments" / "e5_search.parquet")
    long_clause = run / "experiments" / "e5_search_long_clause.parquet"
    if prior and not long_clause.exists() and any(not (row.get("n_hit") or 0) and len(str(row.get("query") or "")) > 120 for row in prior):
        write_parquet(long_clause, prior)
    if long_clause.exists():
        prior = _rows(long_clause)
    selected = [row for row in prior if (row.get("n_hit") or 0) > 0 or row.get("status") in {"included", "already_in_baseline"}]
    for row in selected:
        if row.get("pmid"):
            known.add(str(row["pmid"]))
    failed = {
        (row.get("candidate_id"), row.get("query_id"))
        for row in prior
        if row.get("status") == "query_recorded" and not (row.get("n_hit") or 0)
    }
    already_new = sum(1 for row in selected if row.get("status") == "included")
    remaining_slots = max(0, 30 - invalid_n - already_new)
    for candidate in frozen:
        gained = sum(1 for row in selected if row.get("candidate_id") == candidate["candidate_id"] and row.get("status") == "included")
        alias_row = names.get(candidate.get("pair_id"), {})
        gene_parts = _alias_parts(alias_row.get("gene_name"))
        disease_parts = _alias_parts(alias_row.get("disease_name"))
        question = " ".join(str(candidate.get("search_question") or candidate.get("target_question") or "").split())
        if not gene_parts or not disease_parts:
            selected.append({"candidate_id": candidate["candidate_id"], "query_id": "", "pmid": "", "status": "query_not_formed"})
            continue
        for query in _literal_queries(gene_parts, disease_parts, question):
            if failed and (candidate["candidate_id"], query["query_id"]) not in failed:
                continue
            hits = _esearch(query["text"][:300])
            if hits and hits[0].get("status") == "fetch_failed":
                selected.append({
                    "candidate_id": candidate["candidate_id"], "query_id": query["query_id"], "pmid": "",
                    "status": "fetch_failed", "query": query["text"][:300], "n_hit": 0, "query_form": query["query_form"],
                })
                continue
            room = min(5, 10 - gained, remaining_slots)
            selected.append({
                "candidate_id": candidate["candidate_id"], "query_id": query["query_id"], "pmid": "",
                "status": "query_recorded", "query": query["text"][:300], "n_hit": len(hits), "top50": min(len(hits), 50),
                "alias_source": alias_row.get("gene_name_source") or "", "query_form": query["query_form"],
            })
            if room <= 0:
                continue
            for hit in select_new(hits[:50], known, room):
                item = dict(hit)
                item.update({
                    "candidate_id": candidate["candidate_id"],
                    "pair_id": candidate.get("pair_id"),
                    "query_id": query["query_id"],
                    "query": query["text"][:300],
                    "query_form": query["query_form"],
                })
                if item.get("already_in_packet"):
                    item["status"] = "already_in_baseline"
                else:
                    item["status"] = "included"
                    known.add(str(item["pmid"]))
                    gained += 1
                    remaining_slots -= 1
                selected.append(item)
    write_parquet(run / "experiments" / "e5_search.parquet", selected)
    included = [row for row in selected if row.get("status") == "included" and row.get("pmid")]
    prior_extracts = {str(row.get("pmid")): row for row in _rows(run / "experiments" / "e5_extracts.parquet")}
    pending = [row for row in included if str(row["pmid"]) not in prior_extracts or prior_extracts[str(row["pmid"])].get("status") not in {"ok", "truncated"}]
    texts = _efetch([row["pmid"] for row in pending])
    extracts = [prior_extracts[str(row["pmid"])] for row in included if str(row["pmid"]) in prior_extracts and prior_extracts[str(row["pmid"])].get("status") in {"ok", "truncated"}]
    for row in pending:
        doc = texts.get(str(row["pmid"]))
        if not doc:
            extracts.append({"pmid": row["pmid"], "candidate_id": row["candidate_id"], "pair_id": row.get("pair_id"), "status": "fetch_failed"})
            continue
        user = json.dumps({"pair_id": row.get("pair_id") or "", "pmid": row["pmid"], "title": doc.get("title"), "abstract": doc.get("abstract"), "supplemental_search": True}, ensure_ascii=False)
        result = _call_task(run, "diagnostic_reread", user, "diagnostic_reread.txt", f"e5|{row['pmid']}", 5000)
        extracts.append({
            "pmid": row["pmid"], "candidate_id": row["candidate_id"], "pair_id": row.get("pair_id"),
            "status": result.get("status"), "supplemental_search": True,
            "title": doc.get("title") or "", "abstract": _short(doc.get("abstract"), 1200),
        })
        if result.get("status") in {"quota_exhausted", "stopped_after_quota"}:
            write_parquet(run / "experiments" / "e5_extracts.parquet", extracts)
            write_json(run / "audit" / "e5_status.json", {"status": "stopped_after_quota", "n_included": len(included), "at": _now()})
            return 2
    write_parquet(run / "experiments" / "e5_extracts.parquet", extracts)
    review_rows = []
    for candidate in frozen:
        packet = [row for row in extracts if row.get("candidate_id") == candidate["candidate_id"]]
        user = json.dumps({"candidate": candidate, "new_extracts": packet}, ensure_ascii=False)
        result = _call_task(run, "e5_review", user, "e5_review.txt", f"e5review|{candidate['candidate_id']}|shortquery", 4000)
        parsed = result.get("parsed") if isinstance(result.get("parsed"), dict) else {}
        evidence = parsed.get("evidence") if isinstance(parsed.get("evidence"), list) else []
        review_rows.append({
            "candidate_id": candidate["candidate_id"],
            "pair_id": candidate.get("pair_id"),
            "status": result.get("status"),
            "new_material_relation": parsed.get("new_material_relation") or "",
            "candidate_status": parsed.get("candidate_status") or "",
            "reason": _short(parsed.get("reason"), 500),
            "evidence": json.dumps(evidence, ensure_ascii=False)[:2000],
            "n_extracts_supplied": len(packet),
            "n_extracts_truncated": sum(1 for row in packet if row.get("status") == "truncated"),
        })
    write_parquet(run / "experiments" / "e5_reviews.parquet", review_rows)
    write_json(run / "audit" / "e5_status.json", {
        "status": "reviewed" if review_rows else "search_only",
        "n_retrieved_rows": len(selected),
        "n_included": len(included),
        "n_extract_ok": sum(1 for row in extracts if row.get("status") == "ok"),
        "n_extract_truncated": sum(1 for row in extracts if row.get("status") == "truncated"),
        "n_reviewed": sum(1 for row in review_rows if row.get("status") == "ok"),
        "invalid_alias_fetches": invalid_n,
        "short_query_slots_left": remaining_slots,
        "zero_hit_queries_replaced": sorted(f"{pair}|{qid}" for pair, qid in failed),
        "at": _now(),
    })
    write_json(run / "audit" / "e5_short_query.json", {"at": _now(), "replaced": sorted(f"{pair}|{qid}" for pair, qid in failed)})
    return 0


def _efetch(pmids: list[str]) -> dict[str, dict]:
    out = {}
    clean = [pmid for pmid in pmids if pmid]
    for start in range(0, len(clean), 10):
        chunk = clean[start:start + 10]
        query = urllib.parse.urlencode({"db": "pubmed", "id": ",".join(chunk), "retmode": "xml"})
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + query
        if os.environ.get("NCBI_API_KEY"):
            url += "&api_key=" + urllib.parse.quote(os.environ["NCBI_API_KEY"])
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "pairpilot/1.0"}), timeout=60) as response:
                xml = response.read().decode("utf-8", errors="replace")
        except Exception:
            continue
        for pmid, title, abstract in _split_pubmed(xml):
            out[pmid] = {"title": title, "abstract": abstract}
    return out


def _split_pubmed(xml: str) -> list[tuple[str, str, str]]:
    rows = []
    for block in xml.split("<PubmedArticle"):
        if "<PMID" not in block:
            continue
        pmid = _tag(block, "PMID")
        title = _tag(block, "ArticleTitle")
        abstract = " ".join(_all_tags(block, "AbstractText"))
        if pmid:
            rows.append((pmid, title, abstract))
    return rows


def _tag(text: str, name: str) -> str:
    start = text.find(f"<{name}")
    if start < 0:
        return ""
    start = text.find(">", start)
    end = text.find(f"</{name}>", start)
    if start < 0 or end < 0:
        return ""
    return " ".join(text[start + 1:end].split())


def _all_tags(text: str, name: str) -> list[str]:
    rows = []
    cursor = 0
    while True:
        start = text.find(f"<{name}", cursor)
        if start < 0:
            break
        start = text.find(">", start)
        end = text.find(f"</{name}>", start)
        if start < 0 or end < 0:
            break
        rows.append(" ".join(text[start + 1:end].split()))
        cursor = end + len(name)
    return rows


def _source_lines(run: Path) -> list[str]:
    path = run / "tables" / "diagnostic_source_views.parquet"
    if not path.exists():
        return ["来源视图尚未生成。"]
    lines = []
    for row in _rows(path):
        lines.append(
            f"{row.get('pair_id')} / {row.get('view')}：PMID {row.get('n_pmid')}，finding {row.get('n_finding')}，"
            f"选中且两边都在该视图的比较 {row.get('n_selected_comparison_in_view')}，其中模型描述了关系 {row.get('n_can_describe')}。"
        )
    lines.append("不在某个单库视图里，表示比较用到了该库没有的文章；没有被选进这 24 对的 finding 仍是未选中，不是科学上的零。")
    return lines


def _e3_counts(run: Path) -> dict:
    path = run / "tables" / "scope_transfer_revised.parquet"
    if not path.exists():
        return {}
    rows = _rows(path)
    return {
        "n": len(rows),
        "s_result_nonempty": sum(1 for row in rows if row.get("s_result_field") == "nonempty"),
        "s_result_not_judged": sum(1 for row in rows if row.get("s_result_field") == "not_judged"),
        "s_conclusion_nonempty": sum(1 for row in rows if row.get("s_conclusion_field") == "nonempty"),
        "a_result_nonempty": sum(1 for row in rows if row.get("a_result_field") == "nonempty"),
        "a_conclusion_nonempty": sum(1 for row in rows if row.get("a_conclusion_field") == "nonempty"),
        "text_version_match": "not_recorded",
    }


def _e5_lines(run: Path) -> list[str]:
    lines = []
    note = read_json(run / "audit" / "e5_query_correction.json") if (run / "audit" / "e5_query_correction.json").exists() else {}
    if note:
        lines.append(f"第一次检索把 pair id 当成了 PubMed 词，{note.get('invalid_included')} 篇因此下载的文章已归档，不作为候选证据。")
    search = run / "experiments" / "e5_search.parquet"
    if search.exists():
        hits = [f"{row.get('candidate_id')}/{row.get('query_id')}={row.get('n_hit')}" for row in _rows(search) if row.get("status") == "query_recorded"]
        lines.append("各条已记录查询的命中数：" + "，".join(hits) + "。命中 0 只说明这条查询没有返回，不能解释成文献里没有该问题。")
    short = read_json(run / "audit" / "e5_short_query.json") if (run / "audit" / "e5_short_query.json").exists() else {}
    if short:
        lines.append(f"整句 AND 没有命中的查询改成基因别名 AND 疾病别名后补跑：{short.get('replaced')}。已成功的 broad 查询和已下载摘要保留。")
    extract_path = run / "experiments" / "e5_extracts.parquet"
    truncated = [str(row.get("pmid")) for row in _rows(extract_path) if row.get("status") == "truncated"] if extract_path.exists() else []
    if truncated:
        lines.append("PMID " + "、".join(truncated) + " 的复读在 5000 输出 token 处截断。这是工程失败，不把它当成这篇摘要没有发现。")
    path = run / "experiments" / "e5_reviews.parquet"
    if not path.exists():
        return lines or ["补充检索复核尚未生成。"]
    for row in _rows(path):
        lines.append(
            f"{row.get('candidate_id')} / {row.get('pair_id')}：新材料 {row.get('new_material_relation') or 'not_recorded'}，"
            f"候选状态 {row.get('candidate_status') or 'not_recorded'}。供给摘要 {row.get('n_extracts_supplied')}，截断 {row.get('n_extracts_truncated') or 0}。{_clip(row.get('reason') or '', 360)}"
        )
        if row.get("evidence") not in (None, "", "[]"):
            lines.append(f"模型引用：{_clip(row.get('evidence'), 360)}")
    return lines


def _clip(text: str, limit: int = 320) -> str:
    raw = " ".join(str(text or "").split())
    if len(raw) <= limit:
        return raw
    cut = raw[:limit].rsplit(" ", 1)[0]
    return (cut or raw[:limit]) + " ..."


def _comparison_examples(run: Path) -> list[str]:
    path = run / "tables" / "diagnostic_comparisons.parquet"
    if not path.exists():
        return ["比较表尚未生成。"]
    rows = _rows(path)
    lines = []
    described = next((row for row in rows if row.get("can_compare_direction") in (True, "true") and row.get("relation_description")), None)
    blocked = next((row for row in rows if row.get("can_describe_relation") in (True, "true") and row.get("can_compare_direction") not in (True, "true") and row.get("status") == "ok"), None)
    for label, row in (("方向可以对照的例子", described), ("关系可描述但方向不能比较的例子", blocked)):
        if not row:
            lines.append(f"{label}：表里没有对应行。")
            continue
        lines.append(
            f"{label}：pair {row.get('pair_id')}，PMID {row.get('pmid_a')} 与 {row.get('pmid_b')}，问题类型 {row.get('primary_question_type')}。"
            f"关系：{_clip(row.get('relation_description') or row.get('shared_question') or '', 280)} "
            f"方向 {row.get('direction_relation') or 'not_assessed'}。缺的信息：{_clip(row.get('missing_information') or 'not_recorded', 240)}。"
        )
    return lines


def stage_j6(run_id: str) -> int:
    run = _run()
    p0 = read_json(run / "audit" / "p0_summary.json") if (run / "audit" / "p0_summary.json").exists() else read_json(run / "audit" / "p0.json")
    p0.setdefault("n_v2_comparisons", p0.get("n_cross_v2"))
    p0.setdefault("n_original_filter_comparisons", p0.get("n_cross_original"))
    p0.setdefault("comparison_id_sets_equal", p0.get("cross_id_sets_equal"))
    p0.setdefault("branches", p0.get("branch"))
    p0.setdefault("insufficient_by_pair", p0.get("condition_by_pair"))
    p0.setdefault("diagnostic_unique_pmids", p0.get("n_diagnostic_pmid"))
    p0.setdefault("diagnostic_pair_pmid_rows", p0.get("n_diagnostic_link"))
    p0.setdefault("p1_logical_tasks", p0.get("n_diagnostic_pmid"))
    p0.setdefault("baseline_raw_tuple_count", p0.get("raw_tuple_count"))
    e3 = read_json(run / "audit" / "e3_summary.json") if (run / "audit" / "e3_summary.json").exists() else {}
    p2 = read_json(run / "audit" / "p2_summary.json") if (run / "audit" / "p2_summary.json").exists() else {}
    review = read_json(run / "audit" / "review_summary.json") if (run / "audit" / "review_summary.json").exists() else {}
    e5 = read_json(run / "audit" / "e5_status.json") if (run / "audit" / "e5_status.json").exists() else {}
    align = read_json(run / "audit" / "alignment_summary.json") if (run / "audit" / "alignment_summary.json").exists() else {}
    ledger_state = json.loads((run / "audit" / "ledger.json").read_text(encoding="utf-8")) if (run / "audit" / "ledger.json").exists() else {}
    available = available_usd(ledger_state) if ledger_state else None
    findings = _rows(run / "tables" / "diagnostic_findings.parquet")
    entity_counts = Counter((row.get("entity_kind") or "unresolved") for row in findings if row.get("pair_id") == "3604__C0007131" and row.get("status") == "parsed")
    hidden = 0
    trace_path = run / "tables" / "baseline_rule_trace.parquet"
    if trace_path.exists():
        for row in _rows(trace_path):
            observed = str(row.get("observed_known_difference_fields") or "")
            if row.get("deciding_branch") in {"question_type_other_or_unknown", "required_field_unknown_in_record"} and observed not in {"", "[]", "null"}:
                hidden += 1
    lines = [
        "# diagnostic_v1",
        "",
        f"child `{run.name}`，parent `{REV2.name}`。comparison_gap_v2 没有被覆盖。整个 pilot 仍是 incomplete。",
        "",
        "模型输出没有经过人工科学核对。没有可比较关系或没有合格候选都是合法结果。",
        "",
        "## 1. 原规则为什么大量信息不足",
        "",
        f"原始提取 {p0.get('n_raw_rows')} 行。角色划分：{p0.get('roles')}。",
        f"v2 分析 finding {p0.get('n_analysis_v2')} 行，原先有 question_type 且不是 background 的是 {p0.get('n_original_filter')} 行。两边比较数是 {p0.get('n_v2_comparisons')} 和 {p0.get('n_original_filter_comparisons')}，比较 ID 集合相同：{p0.get('comparison_id_sets_equal')}。进入配对的 finding 是同一批。{p0.get('why_5812')}",
        f"决定分支：{p0.get('branches')}。4600 条先被 other 或 unknown 问题类型拦住，430 条是必填栏在旧记录里为空；两者相加是 5030 条信息不足。其中 {hidden} 条在决定分支已经是信息不足之后，记录里仍有已知字段差异。13 条作者解释没有被 v2 排除，仍留在 5812 对里。这些都不是已经证实摘要没有信息。",
        f"信息不足按 pair：{p0.get('insufficient_by_pair')}。分母是各 pair 的比较行，不能把合计比例当成每个 pair 的比例。",
        "",
        "## 2. other",
        "",
        f"全量 other 清单 {p0.get('n_other')} 条，表 `tables/other_inventory.parquet`。这里只按旧字段审计，没有把词频当成新分类。三个诊断 pair 的复读不能外推到这 {p0.get('n_other')} 条。",
        f"原始条件元组 {p0.get('baseline_raw_tuple_count')}，只合并大小写、空格和标点后是 {p0.get('punctuation_normalized_tuple_count')}。没有按科学含义合并。",
        "",
        "## 3. TNFRSF9 实体边界",
        "",
        f"诊断复读里，3604 这一 pair 的实体种类计数：{dict(entity_counts)}。CD137-Fc 记为干预构建体，TNFSF9/CD137L 记为 8744，不并入 3604。计数为 0 只表示这批解析行里没有该类，不是原文绝对没有。",
        "",
        "## 4. CDC45 与 CA IX 的关系描述",
        "",
        f"P2 候选 {p2.get('n_candidates')}，选中 {p2.get('n_selected')}，模型返回成功 {p2.get('n_judged')}。可以描述关系 {p2.get('describe')}，可以比较方向 {p2.get('direction')}，可以比较数值 {p2.get('magnitude')}。未选中的候选保持未选中，不记为信息不足。这 24 对上限不是全覆盖。",
        "TNFRSF9 本轮不做跨文章比较。",
        *_comparison_examples(run),
        "",
        "## 5. 两库材料",
        "",
        "来源视图在 `tables/diagnostic_source_views.parquet`。一个比较只有两篇 PMID 都在该视图里才计入。这只说明材料是否在场，不是新增有效证据。",
        *_source_lines(run),
        "",
        "## 6. 21 个候选和 E5",
        "",
        f"审核输入 {review.get('n_input')}，完成审核 {review.get('n_reviewed')}，合格 {review.get('n_eligible')}，冻结 {review.get('n_frozen')}。拒绝原因：{review.get('blocked')}。",
        f"E5 状态：{e5 or 'not_started'}。eligibility_empty 表示没有合格候选，不是已经检索验证。",
        *_e5_lines(run),
        "",
        "## 7. E3",
        "",
        f"E3 共 {(e3 or _e3_counts(run)).get('n')} 对。来源句 result 非空 {(e3 or _e3_counts(run)).get('s_result_nonempty')}，not_judged {(e3 or _e3_counts(run)).get('s_result_not_judged')}，conclusion 非空 {(e3 or _e3_counts(run)).get('s_conclusion_nonempty')}。摘要 result 非空 {(e3 or _e3_counts(run)).get('a_result_nonempty')}，conclusion 非空 {(e3 or _e3_counts(run)).get('a_conclusion_nonempty')}。缺字段记 not_recorded，不是 sufficient。文本版本关系是 not_recorded。这不是改善率。",
        "",
        "## 8. 下一步",
        "",
        "值得保留的是：把 other 和作者解释从可比较结果里分开。24 对里 16 对可以描述关系但不能比较方向，数值比较是 0。不值得做的是再对 5812 对逐条调用模型。TNFRSF9 需要先核对配体和融合蛋白，再决定是否比较。K 和 E6 仍是 deferred_pending_protocol。",
        "",
        "## 9. 费用和缺项",
        "",
        f"授权 {LIMITS['max_usd']} 美元。已结算 {ledger_state.get('spent_usd')}。计费不明 {ledger_state.get('billing_unknown_usd')}。在途 {ledger_state.get('reserved_usd')}。可用 {available if available is not None else 'unavailable'}。attempt {ledger_state.get('attempts')} / {LIMITS['max_attempts']}。输入 token {ledger_state.get('input_tokens')}，输出 token {ledger_state.get('output_tokens')} / {LIMITS['max_output_tokens']}。",
        f"旧新对齐原因：{align or 'not_started'}。诊断 pair–PMID 行 {p0.get('diagnostic_pair_pmid_rows')}，唯一 PMID {p0.get('diagnostic_unique_pmids')}，论文级任务 {p0.get('p1_logical_tasks')}。",
        "",
        "旧 pilot 的 K 比较/综合和 E6 扰动仍未完成。本轮诊断完成不等于 pilot 完成。",
        "",
    ]
    (run / "reports").mkdir(exist_ok=True)
    (run / "reports" / "report.md").write_text("\n".join(lines), encoding="utf-8")
    for pair, title in (("8318__C2239176", "CDC45"), ("768__C0376358", "CA IX"), ("3604__C0007131", "TNFRSF9")):
        bits = [f"# {title}", "", f"pair `{pair}`。下面是诊断复读里的原文引用，不是人工核对后的结论。", ""]
        for row in _rows(run / "tables" / "diagnostic_findings.parquet"):
            if row.get("pair_id") != pair or not row.get("quote"):
                continue
            bits.append(f"- PMID {row.get('pmid')} / {row.get('primary_question_type')} / {row.get('entity_kind') or 'entity_not_classified'}: \"{row.get('quote')}\"")
            if len(bits) > 8:
                break
        (run / "reports" / f"{pair}.md").write_text("\n".join(bits) + "\n", encoding="utf-8")
    manifest = read_json(run / "manifest.json")
    e5_done = e5.get("status") in {"reviewed", "eligibility_empty", "search_only"} and (run / "audit" / "e5_short_query.json").exists()
    manifest["status"] = "diagnostic_round_completed" if p2 and review and e5_done else "incomplete"
    manifest["pilot_status"] = "incomplete"
    manifest["updated_utc"] = _now()
    write_json(run / "manifest.json", manifest)
    return 0


STAGES = {"j0": stage_j0, "j1": stage_j1, "j2": stage_j2, "j3": stage_j3, "j4": stage_j4, "j5": stage_j5, "j6": stage_j6}


def main() -> int:
    if _login():
        raise SystemExit("refusing to run the diagnostic round on the login node")
    stage = sys.argv[1]
    try:
        if stage == "test":
            import unittest
            suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
            result = unittest.TextTestRunner(verbosity=1).run(suite)
            return 0 if result.wasSuccessful() else 1
        return STAGES[stage](os.environ["PILOT_CHILD_RUN"])
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

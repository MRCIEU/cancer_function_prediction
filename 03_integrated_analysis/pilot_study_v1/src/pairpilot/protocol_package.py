#!/usr/bin/env python3
"""Offline protocol package. It does not call a model or a search client."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pairpilot.diagnostic_round import _rows  # noqa: E402
from pairpilot.query_ast import class_counts, paper_pair_key, source_membership  # noqa: E402
from pairpilot.store import read_json, write_json, write_parquet  # noqa: E402
from pairpilot.texts import split_sentences  # noqa: E402

VERSION = "evidence_dossier_protocol_v1"
PROMPT_STATUS = "untested_protocol_v1"
E6 = Path("/projects/b5ac/project_2/results/03_integrated_analysis/pilot_study_v1/runs/20261016T020912Z_e6diag")
CLOSE1 = Path("/projects/b5ac/project_2/results/03_integrated_analysis/pilot_study_v1/runs/20261016T012650Z_close1")
DIAG1 = Path("/projects/b5ac/project_2/results/03_integrated_analysis/pilot_study_v1/runs/20261016T013500Z_diag1")
REV2 = Path("/projects/b5ac/project_2/results/03_integrated_analysis/pilot_study_v1/runs/20261016T005500Z_rev2")
PRIOR = Path("/projects/b5ac/project_2/results/03_integrated_analysis/pilot_study_v1/runs/20261015T221200Z_b20cont")
PREP = Path("/projects/b5ac/project_2/results/03_integrated_analysis/pilot_study_v1/runs/20261015T204633Z_0d9b9724")
RUNS = E6.parent
PACKAGE = ROOT
FORMAL_CLASSES = ("core_subset", "semmed_subset", "partial_overlap", "disjoint")

INFORMATION_STATUS = {
    "reported",
    "not_reported_in_text",
    "unassessable",
    "not_applicable",
    "not_recorded_in_legacy",
    "technical_failure",
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _login() -> bool:
    return os.uname().nodename.startswith("login")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _norm(value) -> str:
    return " ".join(str(value or "").casefold().split())


def project_legacy_status(value, field_existed: bool, technical_failure: bool = False) -> str:
    """Old missing fields stay legacy-missing. Failures stay failures."""
    if technical_failure:
        return "technical_failure"
    if not field_existed:
        return "not_recorded_in_legacy"
    text = _norm(value)
    if text in {"", "none", "null"}:
        return "not_recorded_in_legacy"
    if text in {"unassessable", "not_assessed", "unclear"}:
        return "unassessable"
    if text in {"not_applicable", "na"}:
        return "not_applicable"
    return "reported"


def field_link(ref: dict) -> str:
    if ref.get("supported_field"):
        return str(ref["supported_field"])
    return "field_link_unknown"


def align_findings(left: list[dict], right: list[dict]):
    """Match on entity text and cited sentences. Direction is not required."""
    used = set()
    pairs = []
    left_only = []
    for index, row in enumerate(left):
        best = None
        best_score = 0
        for other_index, other in enumerate(right):
            if other_index in used:
                continue
            shared = len(set(row.get("sentences") or []) & set(other.get("sentences") or []))
            same_entity = bool(row.get("entity")) and _norm(row.get("entity")) == _norm(other.get("entity"))
            score = shared * 2 + int(same_entity)
            if score > best_score:
                best = other_index
                best_score = score
        if best is None:
            left_only.append(index)
        else:
            used.add(best)
            pairs.append((index, best))
    right_only = [index for index in range(len(right)) if index not in used]
    return pairs, left_only, right_only


def unit_counts(cases: list[dict]) -> dict:
    slots = [(case["case_id"], case["pmid_a"]) for case in cases] + [(case["case_id"], case["pmid_b"]) for case in cases]
    return {
        "cases": len(cases),
        "paper_pairs": len({paper_pair_key(case["pmid_a"], case["pmid_b"]) for case in cases}),
        "unique_pmids": len({pmid for case in cases for pmid in (case["pmid_a"], case["pmid_b"])}),
        "abstract_slots": len(slots),
        "finding_items": sum(int(case.get("finding_items") or 0) for case in cases),
    }


def formal_draw(frame: list[dict], excluded_pairs: set[str], excluded_pmids: set[str], seed: int) -> dict:
    """Describe a future draw. Callers that pass real pairs must not treat the result as frozen by this round."""
    chosen = []
    notes = []
    used = set(excluded_pairs)
    for relation in FORMAL_CLASSES:
        pool = []
        for row in frame:
            pmids = {str(pmid) for pmid in row.get("pmids") or []}
            if row.get("relation_class") != relation or row.get("pair_id") in used:
                continue
            if pmids & excluded_pmids:
                continue
            pool.append(row)
        pool.sort(key=lambda row: hashlib.sha256(f"{seed}|formal|{relation}|{row['pair_id']}".encode()).hexdigest())
        if not pool:
            notes.append({"relation_class": relation, "status": "stratum_empty"})
            continue
        pick = pool[0]
        used.add(pick["pair_id"])
        chosen.append({"relation_class": relation, "pair_id": pick["pair_id"], "status": "algorithm_output_not_a_frozen_sample"})
    return {"chosen": chosen, "notes": notes, "frozen": False}


def metric_rows() -> list[dict]:
    common = {
        "human_labels": "required",
        "current_value": "not_evaluated",
        "missing_case_status": "no_such_case",
        "unknown_value_status": "not_evaluated",
    }
    specs = [
        ("finding_support", "finding", "findings_with_this_support_label", "emitted_findings_in_the_frozen_set", "full, partial, contradiction, insufficient, and not_evaluated stay separate; the rate uses adjudicated findings and also shows the full denominator"),
        ("quote_localization", "evidence_ref", "quotes_located_in_the_supplied_text", "emitted_quotes", "localization is not semantic support"),
        ("question_role_comparison", "finding_or_case", "cells_of_the_reference_by_system_table", "adjudicated_units_plus_unassessable_and_not_evaluated", "unassessable is not a negative class; small samples do not force Macro-F1"),
        ("critical_direction_errors", "case", "system_allows_but_reference_rejects plus system_rejects_but_reference_allows", "adjudicated_cases", "both error directions are shown"),
        ("omission", "target_result_on_preselected_abstracts", "unmatched_reference_results", "reference_results_on_the_four_preselected_abstracts", "only those abstracts; split and merge stay unmatched rather than row-aligned"),
        ("source_material", "case", "cases_in_each_source_class", "frozen_cases_for_that_pair", "material complementarity, not a single-library model comparison"),
        ("source_after_review", "adjudicated_case", "cases_judged_to_have_a_reasonable_basis", "adjudicated_cases_in_that_source_class", "article gain, comparison opportunity, and reviewed comparison are different quantities"),
        ("abstention", "case", "system_judgements, problematic_judgements, and rejected_but_reference_allows", "all_frozen_cases", "abstention is not counted as reliability"),
        ("e6_diagnostic", "paper_or_fixed_case", "not_applicable", "the fourteen papers and four cases already run", "exploratory only; no end-to-end reliability and no ERASER probability score"),
    ]
    rows = []
    for name, unit, numerator, denominator, supports in specs:
        rows.append({
            "metric": name,
            "unit": unit,
            "numerator": numerator,
            "denominator": denominator,
            "required_labels": common["human_labels"] if name != "e6_diagnostic" else "none_for_the_saved_diagnostic",
            "missing_handling": "keep unassessable, technical_failure, and not_evaluated distinct; do not fill 0",
            "strata": "pair, unique PMID, paper pair, and question case reported separately",
            "can_support": supports,
            "cannot_support": "a universal error rate, a time saving, or a clinical benefit",
            "current_value": "not_evaluated" if name != "e6_diagnostic" else "see_evidence_to_design",
        })
    return rows


def _prompt_guard(text: str) -> None:
    if "STATUS: untested_protocol_v1" not in text:
        raise RuntimeError("protocol prompt is missing its untested status")
    if "api.openai" in text or "esearch" in text:
        raise RuntimeError("protocol prompt contains a client call")


def _run_dir() -> Path:
    run = RUNS / os.environ["PILOT_CHILD_RUN"]
    if not run.exists():
        raise SystemExit(f"protocol run is missing: {run}")
    return run


def assert_output_path(path: Path) -> None:
    resolved = path.resolve()
    for locked in (E6, CLOSE1, DIAG1, REV2, PRIOR, PREP):
        root = locked.resolve()
        if resolved == root or root in resolved.parents:
            raise RuntimeError(f"refusing to write inside a source run: {path}")


def init_run(run_id: str) -> Path:
    run = RUNS / run_id
    assert_output_path(run)
    run.mkdir(parents=False, exist_ok=True)
    for name in ("audit", "tables", "annotation/calibration", "annotation/templates", "jobs", "reports"):
        (run / name).mkdir(parents=True, exist_ok=True)
    if not (run / "manifest.json").exists():
        write_json(run / "manifest.json", {
            "run_id": run_id,
            "parent_run_id": E6.name,
            "science_version": VERSION,
            "prompt_status": PROMPT_STATUS,
            "formal_study_status": "not_started",
            "pilot_status_historical": "incomplete",
            "protocol_package_status": "incomplete",
            "schema_validation_status": "not_run",
            "created_utc": _now(),
            "execution": {
                "model_client": "disabled",
                "search_client": "disabled",
                "target_new_requests": 0,
                "target_new_input_tokens": 0,
                "target_new_output_tokens": 0,
                "target_new_usd": 0,
            },
        })
    return run


def _ledger_snapshot() -> dict:
    ledger = json.loads((E6 / "audit" / "ledger.json").read_text(encoding="utf-8"))
    return {
        "source": str(E6 / "audit" / "ledger.json"),
        "sha256": _sha256(E6 / "audit" / "ledger.json"),
        "spent_usd": ledger.get("spent_usd"),
        "billing_unknown_usd": ledger.get("billing_unknown_usd"),
        "reserved_usd": ledger.get("reserved_usd"),
        "attempts": ledger.get("attempts"),
        "input_tokens": ledger.get("input_tokens"),
        "output_tokens": ledger.get("output_tokens"),
        "read_utc": _now(),
    }


def stage_j0(run_id: str) -> int:
    import unittest
    run = init_run(run_id)
    suite = unittest.defaultTestLoader.discover(str(PACKAGE / "tests"), pattern="test_protocol_package.py")
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    opening = _ledger_snapshot()
    write_json(run / "audit" / "ledger_opening.json", opening)
    hashes = {}
    for path in (
        E6 / "manifest.json",
        E6 / "tables" / "e6_extraction_repeat.parquet",
        E6 / "tables" / "e6_judgment_transition.parquet",
        CLOSE1 / "audit" / "source_summary.json",
        DIAG1 / "audit" / "p0.json",
        PREP / "inputs" / "selected_pairs.parquet",
        PREP / "inputs" / "pair_pmids.parquet",
    ):
        hashes[str(path)] = _sha256(path) if path.exists() else "missing"
    write_json(run / "audit" / "input_hashes.json", hashes)
    manifest = read_json(run / "manifest.json")
    manifest["schema_validation_status"] = "tests_failed" if result.failures or result.errors else "unit_tests_passed"
    manifest["updated_utc"] = _now()
    write_json(run / "manifest.json", manifest)
    write_json(run / "jobs" / "j0.json", {"utc": _now(), "tests_failed": len(result.failures), "tests_errors": len(result.errors)})
    return 0 if result.wasSuccessful() else 1


def _sentence_ids(text: str, quotes: list[str]) -> list:
    sentences = split_sentences(text)
    found = []
    for quote in quotes:
        start = text.find(quote) if quote else -1
        if start < 0:
            found.append(f"unlocated:{_norm(quote)[:48]}")
            continue
        end = start + len(quote)
        for sentence in sentences:
            if sentence["start"] < end and start < sentence["end"] and sentence["start"] not in found:
                found.append(sentence["start"])
    return found


def _findings_of(parsed, text: str = "") -> list[dict]:
    findings = (parsed or {}).get("findings") if isinstance(parsed, dict) else None
    if not isinstance(findings, list):
        return []
    rows = []
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        quotes = [item.get("quote") for item in (finding.get("evidence") or []) if isinstance(item, dict) and item.get("quote")]
        rows.append({
            "entity": finding.get("measured_entity") or finding.get("intervened_entity") or "",
            "question": finding.get("primary_question_type") or "",
            "role": finding.get("discourse_role") or "",
            "direction": finding.get("result_direction") or "",
            "outcome": finding.get("outcome") or "",
            "quote": (quotes[0] if quotes else ""),
            "sentences": _sentence_ids(text, quotes) if text else quotes,
        })
    return rows


def _paper_views(logs: list[dict], documents: dict[str, str] | None = None) -> list[dict]:
    grouped = {}
    for row in logs:
        if row.get("replicate") not in {"r0", "r1", "r2"} or row.get("status") not in {"ok", "truncated"}:
            continue
        key = (row.get("pair_id"), str(row.get("pmid")))
        slot = grouped.setdefault(key, {})
        current = slot.get(row["replicate"])
        if current is None or (row.get("at") or "") < (current.get("at") or ""):
            slot[row["replicate"]] = row
    documents = documents or {}
    views = []
    for (pair_id, pmid), replicates in sorted(grouped.items()):
        text = documents.get(pmid) or ""
        items = {name: _findings_of((replicates.get(name) or {}).get("parsed"), text) for name in ("r0", "r1", "r2")}
        present = {name: name in replicates and isinstance((replicates[name].get("parsed") or {}).get("findings") if isinstance(replicates[name].get("parsed"), dict) else None, list) for name in ("r0", "r1", "r2")}
        pairs, left_only, right_only = align_findings(items["r0"], items["r1"]) if present["r0"] and present["r1"] else ([], [], [])
        views.append({
            "pair_id": pair_id,
            "pmid": pmid,
            "n": {name: len(items[name]) if present[name] else None for name in ("r0", "r1", "r2")},
            "present": present,
            "matched": len(pairs),
            "left_unmatched": len(left_only),
            "right_unmatched": len(right_only),
            "pairs": [
                {
                    "question": [items["r0"][i]["question"], items["r1"][j]["question"]],
                    "role": [items["r0"][i]["role"], items["r1"][j]["role"]],
                    "entity": [items["r0"][i]["entity"], items["r1"][j]["entity"]],
                    "direction_same": _norm(items["r0"][i]["direction"]) == _norm(items["r1"][j]["direction"]),
                    "outcome_same": _norm(items["r0"][i]["outcome"]) == _norm(items["r1"][j]["outcome"]),
                    "entity_same": _norm(items["r0"][i]["entity"]) == _norm(items["r1"][j]["entity"]),
                    "role_same": _norm(items["r0"][i]["role"]) == _norm(items["r1"][j]["role"]),
                    "question_same": _norm(items["r0"][i]["question"]) == _norm(items["r1"][j]["question"]),
                    "quote": [items["r0"][i]["quote"], items["r1"][j]["quote"]],
                    "direction": [items["r0"][i]["direction"], items["r1"][j]["direction"]],
                    "outcome": [items["r0"][i]["outcome"], items["r1"][j]["outcome"]],
                }
                for i, j in pairs
            ],
        })
    return views


def _expect(measured, expected, unit: str) -> str:
    return "matches_prior_note" if measured == expected else f"differs prior={expected} measured={measured} unit={unit}"


def _evidence_rows(views: list[dict], judgments: list[dict], source: dict, p0: dict) -> list[dict]:
    two = sum(1 for row in views if row["present"]["r0"] and row["present"]["r1"])
    three = sum(1 for row in views if row["present"]["r0"] and row["present"]["r1"] and row["present"]["r2"])
    count_diff = sum(1 for row in views if row["present"]["r0"] and row["present"]["r1"] and row["n"]["r0"] != row["n"]["r1"])
    zero_flip = sum(1 for row in views if row["present"]["r0"] and row["present"]["r1"] and (row["n"]["r0"] == 0) != (row["n"]["r1"] == 0))
    matched = sum(row["matched"] for row in views)
    unmatched = sum(row["left_unmatched"] + row["right_unmatched"] for row in views)
    pairs = [item for row in views for item in row["pairs"]]
    question = sum(not item["question_same"] for item in pairs)
    role = sum(not item["role_same"] for item in pairs)
    entity = sum(not item["entity_same"] for item in pairs)
    direction = sum(not item["direction_same"] for item in pairs)
    outcome = sum(not item["outcome_same"] for item in pairs)
    by_case = {}
    for row in judgments:
        by_case.setdefault(row["comparison_id"], {})[row["condition"]] = row
    direction_stable = 0
    for conditions in by_case.values():
        labels = [((conditions.get(name) or {}).get("direction_pattern")) for name in ("full_0", "full_1", "full_2")]
        direction_stable += len(set(labels)) == 1 and all(labels)
    slots = []
    for case_id, conditions in by_case.items():
        parts = [part for part in str(case_id).split("|") if part.isdigit()]
        if len(parts) >= 2:
            slots.extend(parts[:2])
    branch = p0.get("branch") or {}
    insufficient = int(branch.get("question_type_other_or_unknown") or 0) + int(branch.get("required_field_unknown_in_record") or 0)
    question_pmids = sorted({row["pmid"] for row in views for item in row["pairs"] if not item["question_same"]})
    outside = {"E": 0, "D": 0, "R": 0}
    measurement_varies = 0
    replaced = 0
    for conditions in by_case.values():
        full_measures = {(conditions.get(name) or {}).get("measurement_relation") for name in ("full_0", "full_1", "full_2")}
        full_measures.discard(None)
        measurement_varies += len(full_measures) > 1
        for name in ("E", "D", "R"):
            outside[name] += bool((conditions.get(name) or {}).get("outside_full_set"))
        replaced += bool((conditions.get("D") or {}).get("citation_not_in_deleted_set"))
    plan_path = E6 / "experiments" / "e6" / "replicate_plan.json"
    plan = read_json(plan_path) if plan_path.exists() else {}
    rows = [
        ("e6_two_replicates", f"{two}/{len(views)} papers have r0 and r1", _expect(two, 14, "paper"), "program_fact", "adopted_definition", "Use only replicates produced under one saved contract.", "A third call is not inferred from an older extract."),
        ("e6_three_replicates", f"{three}/{len(views)} papers have a third replicate", _expect(three, 0, "paper"), "program_fact", "adopted_definition", "Do not backfill a third replicate.", "diag1 responses are not this round's r0."),
        ("e6_r0_rebuild", f"baseline {plan.get('baseline')}; first {plan.get('first_new_call')}; second {plan.get('second_new_call')}", _expect(plan.get("baseline"), "new_pair_names", "plan"), "program_fact", "adopted_definition", "Keep the new pair-name contract for both calls.", plan.get("reason") or "plan missing"),
        ("e6_zero_state", f"zero/nonzero flips {zero_flip}/{len(views)}", _expect(zero_flip, 0, "paper"), "program_fact", "study_recommendation", "Keep zero findings as their own state.", "No paper in this set changed empty versus nonempty."),
        ("e6_count_and_alignment", f"count differs {count_diff}/{len(views)}; matched {matched}; unmatched slots {unmatched}", _expect((count_diff, matched, unmatched), (4, 31, 5), "paper_and_finding"), "program_fact", "adopted_definition", "Report both sides and leave split or merge unmatched.", "matched/(matched+unmatched) is not a coverage rate."),
        ("e6_closed_fields", f"question {question}/{len(pairs)}; role {role}/{len(pairs)}; entity string {entity}/{len(pairs)}; pmids {question_pmids}", _expect((question, role, entity), (4, 1, 11), "matched_finding"), "program_fact", "study_recommendation", "Define expression versus prognosis by the outcome, not by the word expression.", "Entity string differences are not automatic identity errors."),
        ("e6_free_text", f"direction string {direction}/{len(pairs)}; outcome string {outcome}/{len(pairs)}", _expect((direction, outcome), (26, 22), "matched_finding"), "program_fact", "adopted_definition", "Keep free text beside any structured direction and do not auto-normalize it.", "String inequality is not a polarity flip."),
        ("e6_judgement_cases", f"{len(by_case)} cases; full-condition direction stable in {direction_stable}/{len(by_case)}; full measurement already varies in {measurement_varies}/{len(by_case)}", _expect((len(by_case), direction_stable), (4, 4), "case"), "unverified_model_output", "study_recommendation", "Separate direction, measurement, and population. Do not score the four cases.", "All four saved cases are expression questions. E outside {outside['E']}, D outside {outside['D']}, R outside {outside['R']}, D replacement {replaced}/{len(by_case)}."),
        ("e6_slots", f"abstract slots {len(slots)}; unique PMIDs {len(set(slots))}", _expect((len(slots), len(set(slots))), (8, 7), "slot_and_pmid"), "program_fact", "adopted_definition", "Count cases, paper pairs, unique PMIDs, and slots separately.", "PMID 27225200 occupies more than one slot."),
        ("source_24", json.dumps({"flags": source.get("flags") or {}, "by_pair": source.get("by_pair") or {}}, ensure_ascii=False), "saved_source_summary", "unverified_model_output", "pending_human_check", "Human review must include comparisons the model refused.", "The 24/8/0 figures are model labels. Source classes describe material, not model accuracy."),
        ("old_cross_pairs", f"cross rows {p0.get('n_cross_v2')}; other-or-unknown {branch.get('question_type_other_or_unknown')}; missing record field {branch.get('required_field_unknown_in_record')}; sum {insufficient}", _expect(insufficient, 5030, "old_comparison_row"), "program_fact", "adopted_definition", "Do not rerun or relabel the 5812 rows as abstract absence.", "4600+430 is an old rule branch."),
        ("e3_version", f"n {p0.get('e3_n')}; source result nonempty {p0.get('e3_s_result_nonempty')}; abstract result nonempty {p0.get('e3_a_result_nonempty')}", "text_version_relation not_recorded", "program_fact", "adopted_definition", "Link a source sentence to an abstract sentence only when the string or offset is verified.", "E3 is not an accuracy gain."),
        ("e5_scope", "three frozen candidates have a bounded close1 re-review", "close1 source_summary and report", "method_interpretation", "adopted_definition", "Keep E5 as exploratory refinement.", "A refined candidate is not a confirmed research gap."),
        ("tnfrsf9_boundary", "diagnostic entity rows separate ligand_tnfsf9 gene 8744, intervention_construct, and receptor_tnfrsf9 gene 3604", "diag1 diagnostic_entities", "unverified_model_output", "adopted_definition", "Keep receptor, ligand, and construct as different objects.", "The saved labels are not a new identity adjudication."),
    ]
    output = []
    for item in rows:
        output.append({
            "observation_id": item[0],
            "observation": item[1],
            "comparison_with_prior_note": item[2],
            "source": "e6diag task_log and judgment table; close1 source_summary; diag1 p0 and entities",
            "unit": item[2],
            "evidence_kind": item[3],
            "decision_status": item[4],
            "design_decision": item[5],
            "limit": item[6],
        })
    return output


def _legacy_rows() -> list[dict]:
    path = DIAG1 / "tables" / "diagnostic_findings.parquet"
    rows = []
    for row in _rows(path):
        rows.append({
            "source_run": DIAG1.name,
            "source_key": f"{row.get('pair_id')}|{row.get('pmid')}|{row.get('local')}",
            "legacy_question_type": row.get("primary_question_type") or row.get("question_type") or "",
            "legacy_role": row.get("discourse_role") or "",
            "supported_field": project_legacy_status(None, field_existed=False),
            "field_link": "field_link_unknown",
            "projection_rule": "legacy finding has no per-field evidence link; none is inferred",
            "verified_finding": False,
        })
    return rows


def _collect_pmids(path: Path) -> tuple[list[str], str]:
    if not path.exists():
        return [], "missing"
    if path.suffix == ".json":
        return [], "no_pmid_field"
    try:
        rows = _rows(path)
    except Exception:
        return [], "unreadable"
    if not rows or "pmid" not in rows[0]:
        return [], "no_pmid_field"
    return [str(row["pmid"]) for row in rows if row.get("pmid")], "pmid_column"


def _exclusions() -> list[dict]:
    rows = []
    content = set()
    for row in _rows(PREP / "inputs" / "pair_pmids.parquet"):
        pmid = str(row["pmid"])
        content.add(pmid)
        rows.append({
            "unit_type": "pmid",
            "unit_id": pmid,
            "pair_id": row["pair_id"],
            "exposure": "content_exposed",
            "source_run": PREP.name,
            "source_file": "inputs/pair_pmids.parquet",
            "reason": "frozen pilot abstract set",
        })
    for row in _rows(PREP / "inputs" / "selected_pairs.parquet"):
        rows.append({
            "unit_type": "pair",
            "unit_id": row["pair_id"],
            "pair_id": row["pair_id"],
            "exposure": "content_exposed",
            "source_run": PREP.name,
            "source_file": "inputs/selected_pairs.parquet",
            "reason": f"development pair {row.get('relation_class')} {row.get('band')}",
        })
    scan = [
        CLOSE1 / "experiments" / "supplemental_findings.parquet",
        DIAG1 / "experiments" / "e5_extracts.parquet",
        DIAG1 / "experiments" / "e5_extracts_invalid_alias.parquet",
        DIAG1 / "experiments" / "e5_search.parquet",
        DIAG1 / "experiments" / "e5_search_invalid_alias.parquet",
        PRIOR / "tables" / "search_results.parquet",
    ]
    for path in scan:
        pmids, status = _collect_pmids(path)
        if status != "pmid_column":
            rows.append({
                "unit_type": "file",
                "unit_id": path.name,
                "pair_id": "",
                "exposure": "residual_leakage_risk",
                "source_run": path.parent.parent.name,
                "source_file": str(path),
                "reason": status,
            })
            continue
        for pmid in sorted(set(pmids)):
            exposure = "content_exposed" if pmid in content or "extract" in path.name or "supplemental" in path.name else "pmid_listed_only"
            rows.append({
                "unit_type": "pmid",
                "unit_id": pmid,
                "pair_id": "",
                "exposure": exposure,
                "source_run": path.parent.parent.name,
                "source_file": str(path.relative_to(path.parents[2])) if len(path.parents) > 2 else path.name,
                "reason": "listed in a saved experiment table",
            })
    return rows


def _casebook(views: list[dict], judgments: list[dict]) -> str:
    lines = [
        "# 诊断案例",
        "",
        "这些案例来自已经保存的输出，模型标签是 non_gold。可自动看出的是行数、字段字符串和条件标签。语义是否相同留给以后的人工核对，这里不下判。",
        "",
    ]
    changed = [row for row in views if row["present"]["r0"] and row["present"]["r1"] and row["n"]["r0"] != row["n"]["r1"]]
    lines.append("## 1. finding 数不同和未对齐")
    lines.append("")
    lines.append("单位是一篇文章的两次提取。未对齐是对齐后一侧多出来的 finding，不是覆盖率分母里的自动错误。")
    lines.append("")
    for row in changed:
        lines.append(f"- PMID {row['pmid']}（{row['pair_id']}）：r0 {row['n']['r0']} 条，r1 {row['n']['r1']} 条，对齐 {row['matched']}，仅 r0 {row['left_unmatched']}，仅 r1 {row['right_unmatched']}。")
    lines.append("")
    lines.append("按来源句把这些行收成一对一，可能把一句里的两个结果合掉。这里保持未对齐。")
    lines.append("")
    lines.append("## 2. 表达和预后类型字符串不同")
    lines.append("")
    for row in views:
        for item in row["pairs"]:
            if item["question_same"]:
                continue
            lines.append(f"- PMID {row['pmid']}：{item['question'][0]} → {item['question'][1]}。实体 r0：{item['entity'][0][:160]}。实体 r1：{item['entity'][1][:160]}。")
            lines.append(f"  依据 r0：{item['quote'][0][:240]}")
            lines.append(f"  依据 r1：{item['quote'][1][:240]}")
            lines.append("  这是闭合字段的类型字符串变化，不是已经裁定谁对。")
    lines.append("")
    lines.append("## 3. 实体字符串和 TNFRSF9 边界")
    lines.append("")
    shown = 0
    for row in views:
        for item in row["pairs"]:
            if item["entity_same"] or shown >= 2:
                continue
            lines.append(f"- PMID {row['pmid']} 实体字符串不同。r0：{item['entity'][0][:180]}。r1：{item['entity'][1][:180]}。字符串不同不自动是身份错误。")
            shown += 1
    lines.append("TNFRSF9 pair 3604__C0007131 的诊断实体表，按 PMID、角色和基因号原样列出。这是已保存的模型角色，不是新的身份裁定。")
    seen = set()
    for row in _rows(DIAG1 / "tables" / "diagnostic_entities.parquet"):
        if row.get("pair_id") != "3604__C0007131":
            continue
        key = (str(row.get("pmid")), row.get("entity_kind"), str(row.get("mention") or "")[:80])
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- PMID {row.get('pmid')} {row.get('entity_kind')} gene {row.get('entity_gene_id')}：{str(row.get('mention') or '')[:160]}")
    lines.append("")
    lines.append("## 4–7. 四个 E6 比较案例")
    lines.append("")
    lines.append("每个案例三次完整摘要的方向，以及 E、D、R 各一次。不计算总稳定性。四个都是表达或分布问题，不能推断预后比较。")
    lines.append("")
    variants = read_json(E6 / "experiments" / "e6" / "variants.json")
    by_id = {row["comparison_id"]: row for row in variants}
    for case_id, conditions in by_case_order(judgments):
        short = " vs ".join(part for part in case_id.split("|") if part.isdigit())
        labels = {name: (conditions.get(name) or {}).get("direction_pattern") for name in ("full_0", "full_1", "full_2", "E", "D", "R")}
        measure = {name: (conditions.get(name) or {}).get("measurement_relation") for name in labels}
        population = {name: (conditions.get(name) or {}).get("population_or_system_relation") for name in labels}
        lines.append(f"### {short}")
        lines.append("")
        lines.append(f"方向 {labels}。测量 {measure}。人群或系统 {population}。")
        replacement = (conditions.get("D") or {}).get("replacement_quote") or ""
        lines.append(f"D 的字段连接是 field_link_unknown。替换引用原文：{replacement or '无'}。这条引用没有被认证为支持方向。")
        stored = (by_id.get(case_id) or {}).get("stored") or {}
        for pmid, side in stored.items():
            lines.append(f"PMID {pmid}：引用句 {side.get('cited_n')}，删除字符 {side.get('deleted_chars')}，R={side.get('R_status')}，token 差 {side.get('R_token_gap')}。")
        lines.append("")
    lines.append("## 8. 报告方向相同但测量字符串不同")
    lines.append("")
    lines.extend(_prognostic_lines())
    lines.append("")
    return "\n".join(lines)


def _prognostic_lines() -> list[str]:
    comparisons = {row["candidate_key"]: row for row in _rows(DIAG1 / "tables" / "diagnostic_comparisons.parquet")}
    found = []
    for row in _rows(CLOSE1 / "tables" / "comparison_casebook.parquet"):
        comp = comparisons.get(row.get("comparison_id"), {})
        if comp.get("primary_question_type") != "prognostic_association" or row.get("can_compare_direction") is not True:
            continue
        if str(row.get("pmid_a")) == "32395522" and str(row.get("pmid_b")) == "33622998":
            found.append(row)
    if not found:
        return ["没有找到 32395522 vs 33622998 的预后案例，不另造例子。"]
    row = found[0]
    return [
        f"close1 casebook 的 CDC45 {row.get('pmid_a')} vs {row.get('pmid_b')}：模型 direction_relation={row.get('direction_relation')}，can_compare_direction={row.get('can_compare_direction')}。",
        f"measured_a：{row.get('measured_a')}。measured_b：{row.get('measured_b')}。",
        f"study_system_a：{row.get('study_system_a')}。study_system_b：{row.get('study_system_b')}。",
        f"outcome_a：{str(row.get('outcome_a') or '')[:240]}。outcome_b：{str(row.get('outcome_b') or '')[:240]}。",
        "这些是未验证的模型字段。相同的方向标签加上不同的测量字符串，还不能裁定测量或人群是否相同。",
    ]


def _source_check() -> dict:
    membership = {}
    for row in _rows(PREP / "inputs" / "pair_pmids.parquet"):
        slot = membership.setdefault(row["pair_id"], {"core": set(), "semmed": set()})
        if row.get("source") == "CORE":
            slot["core"].add(str(row["pmid"]))
        elif row.get("source") == "SemMedDB":
            slot["semmed"].add(str(row["pmid"]))
    comparisons = {row["candidate_key"]: row for row in _rows(DIAG1 / "tables" / "diagnostic_comparisons.parquet")}
    classified = []
    for row in _rows(CLOSE1 / "tables" / "comparison_casebook.parquet"):
        pair = row.get("pair_id")
        if pair not in {"8318__C2239176", "768__C0376358"}:
            continue
        sets = membership.get(pair) or {"core": set(), "semmed": set()}
        label = source_membership({str(row["pmid_a"]), str(row["pmid_b"])}, sets["core"], sets["semmed"])
        classified.append({
            "pair_id": pair,
            "pmid_a": str(row["pmid_a"]),
            "pmid_b": str(row["pmid_b"]),
            "mutually_exclusive_source_class": label["mutually_exclusive_source_class"],
            "question": (comparisons.get(row["comparison_id"]) or {}).get("primary_question_type"),
        })
    by_pair = {}
    saved = json.loads((CLOSE1 / "audit" / "source_summary.json").read_text(encoding="utf-8")).get("by_pair") or {}
    for pair in ("8318__C2239176", "768__C0376358"):
        subset = [row for row in classified if row["pair_id"] == pair]
        counts = class_counts(subset)
        prior = saved.get(pair) or {}
        measured = {
            "n": len(subset),
            "paper_pairs": len({paper_pair_key(row["pmid_a"], row["pmid_b"]) for row in subset}),
            "both_individually": counts["both_individually"],
            "core_only": counts["core_only"],
            "semmed_only": counts["semmed_only"],
            "union_required": counts["union_required"],
            "questions": sorted({row["question"] for row in subset if row.get("question")}),
        }
        measured["matches_saved_summary"] = all(measured[key] == prior.get(key) for key in ("both_individually", "core_only", "semmed_only", "union_required")) and measured["paper_pairs"] == prior.get("n_unique_paper_pairs")
        by_pair[pair] = measured
    return {"recomputed": by_pair}


def by_case_order(judgments: list[dict]):
    grouped = {}
    for row in judgments:
        grouped.setdefault(row["comparison_id"], {})[row["condition"]] = row
    return grouped.items()


def stage_j1(run_id: str) -> int:
    run = _run_dir()
    logs = _read_jsonl(E6 / "experiments" / "e6" / "task_log.jsonl")
    documents = {str(row["pmid"]): row.get("text") or "" for row in _rows(PREP / "inputs" / "documents.parquet")}
    views = _paper_views(logs, documents)
    judgments = _rows(E6 / "tables" / "e6_judgment_transition.parquet")
    source = json.loads((CLOSE1 / "audit" / "source_summary.json").read_text(encoding="utf-8"))
    p0 = json.loads((DIAG1 / "audit" / "p0.json").read_text(encoding="utf-8"))
    evidence = _evidence_rows(views, judgments, source, p0)
    source_check = _source_check()
    evidence.append({
        "observation_id": "source_recompute",
        "observation": json.dumps(source_check["recomputed"], ensure_ascii=False),
        "comparison_with_prior_note": "recomputed_from_pair_pmids_and_casebook",
        "source": "prep pair_pmids; close1 casebook; diag1 comparisons",
        "unit": "comparison_case",
        "evidence_kind": "program_fact",
        "decision_status": "adopted_definition",
        "design_decision": "Classify a case from the target pair's own PMID sets.",
        "limit": "A PMID in another pair does not change this pair's class. The saved source_summary remains the prior note.",
    })
    write_json(run / "audit" / "source_recompute.json", source_check)
    write_parquet(run / "tables" / "evidence_to_design.parquet", evidence)
    write_json(run / "audit" / "paper_alignment.json", views)
    legacy = _legacy_rows()
    write_parquet(run / "tables" / "legacy_projection.parquet", legacy)
    exclusions = _exclusions()
    write_parquet(run / "tables" / "development_exclusions.parquet", exclusions)
    write_json(run / "audit" / "formal_draw_algorithm.json", {
        "frozen": False,
        "function": "formal_draw",
        "classes": list(FORMAL_CLASSES),
        "seed": 20261015,
        "per_pair_abstract_cap": 6,
        "study_abstract_cap": 24,
        "per_pair_case_cap": 6,
        "study_case_cap": 24,
        "preselected_full_inventory_abstracts": 4,
        "second_reader_cases": 8,
        "selected_pairs": [],
        "reason": "The algorithm is saved and unit-tested. It was not applied to the live frame.",
    })
    (run / "reports" / "casebook.md").write_text(_casebook(views, judgments), encoding="utf-8")
    write_json(run / "jobs" / "j1.json", {
        "utc": _now(),
        "evidence_rows": len(evidence),
        "legacy_rows": len(legacy),
        "exclusion_rows": len(exclusions),
        "papers": len(views),
    })
    return 0


def _copy_text(name: str, destination: Path) -> str:
    text = (PACKAGE / "protocol" / name).read_text(encoding="utf-8")
    destination.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _templates(run: Path) -> None:
    blank = {
        "template_status": "blank",
        "labels_filled": False,
        "records": [],
        "fields": json.loads((PACKAGE / "schemas" / "annotation_v1.json").read_text(encoding="utf-8")),
        "note": "Do not import historical model labels into this template.",
    }
    write_json(run / "annotation" / "templates" / "comparison_annotation.json", blank)
    write_json(run / "annotation" / "templates" / "finding_annotation.json", blank)
    calibration = {
        "non_gold": True,
        "use": "guide_calibration_only",
        "source_casebook": "reports/casebook.md",
        "labels_are_reference": False,
    }
    write_json(run / "annotation" / "calibration" / "index.json", calibration)


def _schema_check() -> dict:
    prompts = []
    for name in ("extract_protocol_v1.txt", "compare_protocol_v1.txt"):
        text = (PACKAGE / "prompts" / name).read_text(encoding="utf-8")
        _prompt_guard(text)
        prompts.append(name)
    for name in ("finding_protocol_v1.json", "comparison_protocol_v1.json", "field_evidence_v1.json", "annotation_v1.json"):
        body = json.loads((PACKAGE / "schemas" / name).read_text(encoding="utf-8"))
        if "untested" not in body.get("status", "") and "blank" not in body.get("status", ""):
            raise RuntimeError(name)
        if "not_recorded_in_legacy" in json.dumps(body) or name.startswith("field") or name.startswith("annotation"):
            continue
    finding = json.loads((PACKAGE / "schemas" / "finding_protocol_v1.json").read_text(encoding="utf-8"))
    if "not_recorded_in_legacy" not in finding["information_status"]:
        raise RuntimeError("legacy status missing")
    if "technical_failure" not in finding["information_status"]:
        raise RuntimeError("technical failure missing")
    return {"prompts": prompts, "status": "passed"}


def _report(run: Path, opening: dict, closing: dict) -> str:
    evidence = _rows(run / "tables" / "evidence_to_design.parquet") if (run / "tables" / "evidence_to_design.parquet").exists() else []
    diffs = [row["observation_id"] for row in evidence if str(row.get("comparison_with_prior_note", "")).startswith("differs")]
    spent_same = opening.get("spent_usd") == closing.get("spent_usd") and opening.get("attempts") == closing.get("attempts")
    requests = list((run / "requests").glob("*")) if (run / "requests").exists() else []
    return "\n".join([
        "# evidence_dossier_protocol_v1",
        "",
        f"child `{run.name}`，parent `{E6.name}`。这是协议和评价材料，不是正式实验。",
        "",
        "正式对象是生物医学研究者和支持文献分析的信息人员。输入是一个 gene–cancer pair 和冻结文献。输出是可追溯的证据比较档案。RQ1 问摘要里的发现是否保住实体、结局和条件。RQ2 问跨文章比较的描述、方向和数值是否分开而且理由具体。RQ3 问两库在材料和经核对的比较上增加了什么。",
        "",
        "finding 是一条实证结果或一条单独的作者解释。比较案例是文章对加中性问题。字段证据必须指向 supported_field；旧的通用引用保持 field_link_unknown。两个 KG 以原始记录和 pair–PMID 连接进入档案，来源四分类只说明材料能否形成，不说明判断正确。",
        "",
        "依赖人工的指标在 `tables/metric_spec.parquet`，当前值是 not_evaluated。E6 的次数和四案转移在 evidence_to_design，不能换成准确率。",
        "",
        f"与先前笔记不一致的观察：{diffs or '无'}。",
        "",
        "下一次真正运行前要先做规范试用，再冻结独立样本，再跑生产合同，再做有限人工评价。本轮的 prompt 仍是 untested_protocol_v1，正式研究是 not_started。",
        "",
        f"父账本 `{opening.get('source')}`。开读 sha 与回读 sha 相同：{opening.get('sha256') == closing.get('sha256')}。开读花费 {opening.get('spent_usd')}，回读花费 {closing.get('spent_usd')}，attempt {opening.get('attempts')} → {closing.get('attempts')}，未知计费 {opening.get('billing_unknown_usd')} → {closing.get('billing_unknown_usd')}。花费和 attempt 未变：{spent_same}。本 child 的 requests 文件数 {len(requests)}。这个目录没有模型请求，所以本轮新增请求、token 和费用记 0；若父账本发生变化，差额记为其他进程，不并进本轮。",
        "",
        "protocol_package_status 见 manifest。历史 pilot 仍是 incomplete。K、E4、开放模型和 E6 没有重新启动。",
        "",
    ])


def stage_j2(run_id: str) -> int:
    run = _run_dir()
    opening = read_json(run / "audit" / "ledger_opening.json")
    closing = _ledger_snapshot()
    write_json(run / "audit" / "ledger_closing.json", closing)
    schema = _schema_check()
    write_json(run / "audit" / "schema_check.json", schema)
    write_parquet(run / "tables" / "metric_spec.parquet", metric_rows())
    hashes = {
        "protocol": _copy_text("protocol.md", run / "reports" / "protocol.md"),
        "annotation_guide": _copy_text("annotation_guide.md", run / "reports" / "annotation_guide.md"),
        "implementation_plan": _copy_text("implementation_plan.md", run / "reports" / "implementation_plan.md"),
    }
    _templates(run)
    complete = all((run / path).exists() for path in (
        "tables/evidence_to_design.parquet",
        "tables/legacy_projection.parquet",
        "tables/development_exclusions.parquet",
        "tables/metric_spec.parquet",
        "reports/casebook.md",
        "reports/protocol.md",
        "reports/annotation_guide.md",
        "reports/implementation_plan.md",
        "annotation/templates/comparison_annotation.json",
    ))
    (run / "reports" / "report.md").write_text(_report(run, opening, closing), encoding="utf-8")
    new_requests = 0
    if (run / "requests").exists():
        new_requests = sum(1 for _ in (run / "requests").glob("*"))
    same_ledger = opening.get("sha256") == closing.get("sha256")
    manifest = read_json(run / "manifest.json")
    manifest.update({
        "prompt_status": PROMPT_STATUS,
        "formal_study_status": "not_started",
        "pilot_status_historical": "incomplete",
        "protocol_package_status": "protocol_package_completed" if complete and new_requests == 0 else "partial",
        "schema_validation_status": schema["status"],
        "template_sha256": hashes,
        "new_model_requests_observed": new_requests,
        "parent_ledger_unchanged": same_ledger,
        "updated_utc": _now(),
    })
    write_json(run / "manifest.json", manifest)
    write_json(run / "jobs" / "j2.json", {"utc": _now(), "complete": complete, "new_requests": new_requests, "ledger_unchanged": same_ledger})
    return 0 if complete and new_requests == 0 else 2


STAGES = {"j0": stage_j0, "j1": stage_j1, "j2": stage_j2}


def main() -> int:
    if _login():
        raise SystemExit("refusing to build the protocol package on the login node")
    if os.environ.get("PILOT_ALLOW_MODEL") == "1":
        raise SystemExit("this package refuses a model-enabled run")
    try:
        return STAGES[sys.argv[1]](os.environ["PILOT_CHILD_RUN"])
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

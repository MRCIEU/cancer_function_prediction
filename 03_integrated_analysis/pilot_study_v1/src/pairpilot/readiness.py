#!/usr/bin/env python3
"""Protocol v2 readiness. Formal sampling stays closed. Calibration answers stay blank."""

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
from pairpilot.ledger import LIMITS, estimate_usd  # noqa: E402
from pairpilot.query_ast import paper_pair_key  # noqa: E402
from pairpilot.store import read_json, write_json, write_parquet  # noqa: E402

VERSION = "evidence_dossier_protocol_v2"
PROMPT_STATUS = "technical_check_pending_calibration"
PARENT = Path("/projects/b5ac/project_2/results/03_integrated_analysis/pilot_study_v1/runs/20261017T020726Z_protocol1")
E6 = Path("/projects/b5ac/project_2/results/03_integrated_analysis/pilot_study_v1/runs/20261016T020912Z_e6diag")
CLOSE1 = Path("/projects/b5ac/project_2/results/03_integrated_analysis/pilot_study_v1/runs/20261016T012650Z_close1")
DIAG1 = Path("/projects/b5ac/project_2/results/03_integrated_analysis/pilot_study_v1/runs/20261016T013500Z_diag1")
PREP = Path("/projects/b5ac/project_2/results/03_integrated_analysis/pilot_study_v1/runs/20261015T204633Z_0d9b9724")
PRIOR = Path("/projects/b5ac/project_2/results/03_integrated_analysis/pilot_study_v1/runs/20261015T221200Z_b20cont")
RUNS = PARENT.parent
PACKAGE = ROOT
SEED = 20261017
ROUND_USD = 2.0
ROUND_ATTEMPTS = 12
ROUND_INPUT = 100_000
ROUND_OUTPUT = 60_000
EXTRACT_CAP = 6000
COMPARE_CAP = 4000
FIELD_LIMITS = {
    "shared_question": 300,
    "relation_description": 400,
    "missing_information": 400,
    "why_it_blocks": 400,
    "inference_limit": 300,
}
QUESTIONS = {
    "Q_EXPR": "两篇摘要分别报告了目标在相关癌症样本/模型中的哪些表达、丰度或定位结果；这些结果能在什么层面相互对照？",
    "Q_PROG": "两篇摘要分别报告了目标表达与患者预后之间的什么关系；这些关系能在什么层面相互对照？",
}
CANARY_PAPERS = (
    ("8318__C2239176", "32395522"),
    ("8318__C2239176", "33622998"),
    ("768__C0376358", "27225200"),
    ("768__C0376358", "35711477"),
)
CANARY_COMPARES = (
    ("Q_PROG", "8318__C2239176", "32395522", "33622998"),
    ("Q_EXPR", "768__C0376358", "27225200", "35711477"),
)
CALIBRATION = (
    {"case_code": "CAL01", "pair_id": "8318__C2239176", "pmids": ("32395522", "33622998"), "question_id": "Q_PROG"},
    {"case_code": "CAL02", "pair_id": "768__C0376358", "pmids": ("27225200", "35711477"), "question_id": "Q_EXPR"},
    {"case_code": "CAL03", "pair_id": "768__C0376358", "pmids": ("20038959", "31790614"), "question_id": "Q_EXPR"},
    {"case_code": "CAL04", "pair_id": "3604__C0007131", "pmids": ("25631633",), "question_id": "ENTITY"},
)
A_FORBIDDEN = (
    "union_required", "non_gold", "phase_b", "historical_model", "direction_relation",
    "source_class", "can_compare", "evidence_quotes", "CD137L", "CD137-Fc", "cd137l", "cd137-fc",
)
LAYERS = ("shared", "core_only", "semmed_only")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _login() -> bool:
    return os.uname().nodename.startswith("login")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def collapsed(value) -> str:
    return " ".join(str(value or "").split())


def public_text(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def truncation_status(stored, full, limit: int) -> str:
    if full is None:
        return "response_missing"
    stored_c = collapsed(stored)
    full_c = collapsed(full)
    if stored_c == full_c:
        return "not_truncated"
    # The exporter collapses whitespace and then slices. A second collapse can drop one trailing space.
    if stored_c and full_c.startswith(stored_c) and len(stored_c) >= limit - 1 and len(full_c) > len(stored_c):
        return "exporter_truncation"
    return "text_differs"


def locate_all(text: str, quote: str) -> list[dict]:
    if not text or not quote:
        return []
    spans = []
    start = 0
    while len(spans) < 20:
        idx = text.find(quote, start)
        if idx < 0:
            break
        spans.append({"start": idx, "end": idx + len(quote)})
        start = idx + 1
    return spans


def span_record(text: str, quote: str, text_hash: str) -> dict:
    spans = locate_all(text, quote)
    if not spans:
        status = "zero"
    elif len(spans) == 1:
        status = "unique"
    else:
        status = "multiple"
    return {
        "text_sha256": text_hash,
        "match_count": len(spans),
        "match_status": status,
        "offsets": spans,
        "semantic_support": "not_assessed",
    }


def question_id_for(legacy_type: str) -> str:
    if legacy_type == "expression_or_distribution":
        return "Q_EXPR"
    if legacy_type == "prognostic_association":
        return "Q_PROG"
    return "mapping_unresolved"


def case_identity(pair_id: str, pmid_a: str, pmid_b: str, question_id: str, version_a: str, version_b: str) -> str:
    if question_id not in QUESTIONS:
        return ""
    paper = paper_pair_key(pmid_a, pmid_b)
    versions = "|".join(sorted((version_a or "", version_b or "")))
    return hashlib.sha256(f"{pair_id}|{paper}|{question_id}|q_v2|{versions}".encode()).hexdigest()[:24]


def allocate_abstracts(layers: dict, seed: int, cap: int = 6) -> dict:
    remaining = {name: list(layers.get(name) or []) for name in LAYERS}
    for name, values in remaining.items():
        values.sort(key=lambda pmid: hashlib.sha256(f"{seed}|{name}|{pmid}".encode()).hexdigest())
    picked = []

    def take(name: str, step: str) -> None:
        if len(picked) >= cap or not remaining[name]:
            return
        picked.append({"pmid": remaining[name].pop(0), "layer": name, "step": step})

    for name in LAYERS:
        if remaining[name]:
            take(name, "first_of_nonempty_layer")
    while len(picked) < cap:
        cursor = 0 if not picked else (LAYERS.index(picked[-1]["layer"]) + 1) % len(LAYERS)
        moved = False
        for offset in range(len(LAYERS)):
            name = LAYERS[(cursor + offset) % len(LAYERS)]
            if remaining[name]:
                take(name, "round_robin")
                moved = True
                break
        if not moved:
            break
    seen = set()
    unique = []
    for item in picked:
        if item["pmid"] in seen:
            continue
        seen.add(item["pmid"])
        unique.append(item)
    core_n = sum(1 for item in unique if item["layer"] in {"shared", "core_only"})
    sem_n = sum(1 for item in unique if item["layer"] in {"shared", "semmed_only"})
    return {
        "picked": unique[:cap],
        "library_core": core_n,
        "library_semmed": sem_n,
        "unique_pmids": len(unique[:cap]),
        "meets_start_plan": core_n >= 2 and sem_n >= 2,
        "empty_layers": [name for name in LAYERS if not (layers.get(name) or [])],
    }


def filter_pair_articles(pair_id: str, pmids: list[str], dev_pairs: set[str], exposed: set[str]) -> dict:
    if pair_id in dev_pairs:
        return {"status": "development_pair_excluded", "remaining": [], "removed": list(pmids)}
    removed = [pmid for pmid in pmids if pmid in exposed]
    remaining = [pmid for pmid in pmids if pmid not in exposed]
    return {"status": "exposed_articles_removed", "remaining": remaining, "removed": removed}


def blank_answer() -> dict:
    return {
        "workflow_status": "pending",
        "discourse_role_reference": None,
        "segmentation_assessability": None,
        "segmentation_flags": None,
        "identity": None,
        "role_form": None,
        "text_support": None,
        "citation_support": None,
        "comparison_scope": None,
        "direction_pattern": None,
        "notes": None,
    }


def a_pack_leaks(text: str) -> list[str]:
    lowered = text.casefold()
    return [token for token in A_FORBIDDEN if token.casefold() in lowered]


def formal_draw_live(*_args, **_kwargs):
    raise RuntimeError("formal sampling is not started")


def technical_label(call_ok: bool, check: dict) -> str:
    if not call_ok:
        return "technical_failed"
    if check.get("zero") or check.get("missing_field"):
        return "technical_partial"
    return "technical_pass"


def readiness_label(repair: str, annotation: str, canary: str) -> str:
    if repair == "completed" and annotation == "prepared_blank" and canary == "technical_pass":
        return "ready_for_calibration"
    return "partial"


def extract_logical(pair_id: str, pmid: str) -> str:
    return f"{VERSION}|extract|{pair_id}|{pmid}"


def _quotes_nonempty(quotes) -> bool:
    if not isinstance(quotes, dict):
        return False
    for value in quotes.values():
        if isinstance(value, list) and any(isinstance(item, str) and item.strip() for item in value):
            return True
    return False


def round_block_reasons(opening: dict, state: dict, next_input: int, next_output: int, next_usd: float) -> list[str]:
    reasons = []
    if int(state["attempts"]) - int(opening["attempts"]) + 1 > ROUND_ATTEMPTS:
        reasons.append("round_attempts")
    if int(state["input_tokens"]) - int(opening["input_tokens"]) + next_input > ROUND_INPUT:
        reasons.append("round_input")
    if int(state["output_tokens"]) - int(opening["output_tokens"]) + next_output > ROUND_OUTPUT:
        reasons.append("round_output")
    spent_now = float(state["spent_usd"]) + float(state["reserved_usd"]) + float(state["billing_unknown_usd"])
    spent_open = float(opening["spent_usd"]) + float(opening["reserved_usd"]) + float(opening["billing_unknown_usd"])
    if spent_now - spent_open + next_usd > ROUND_USD:
        reasons.append("round_usd")
    canary_left = int(LIMITS["canary_attempts"]) - int(state.get("canary_attempts") or 0)
    if canary_left < 1:
        reasons.append("canary_attempts")
    return reasons


def cache_key(task: str, user: str, prompt: str, logical: str, cap: int, contract: str = VERSION) -> str:
    body = {
        "task": task,
        "contract": contract,
        "user_sha256": hashlib.sha256(user.encode()).hexdigest(),
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "logical_id": logical,
        "max_completion_tokens": cap,
        "model_id": "gpt-6.1-sol",
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def assert_output_path(path: Path) -> None:
    resolved = path.resolve()
    for locked in (PARENT, E6, CLOSE1, DIAG1, PREP, PRIOR):
        root = locked.resolve()
        if resolved == root or root in resolved.parents:
            raise RuntimeError(f"refusing to write inside a source run: {path}")


def _run() -> Path:
    run = RUNS / os.environ["PILOT_CHILD_RUN"]
    if not run.exists():
        raise SystemExit(f"readiness run is missing: {run}")
    return run


def init_run(run_id: str) -> Path:
    run = RUNS / run_id
    assert_output_path(run)
    run.mkdir(parents=False, exist_ok=True)
    for name in ("audit", "tables", "config", "annotation/calibration_v2/phase_a", "annotation/calibration_v2/phase_b", "technical_canary", "jobs", "reports", "requests", "responses"):
        (run / name).mkdir(parents=True, exist_ok=True)
    if not (run / "manifest.json").exists():
        write_json(run / "manifest.json", {
            "run_id": run_id,
            "parent_run_id": PARENT.name,
            "science_version": VERSION,
            "prompt_status": PROMPT_STATUS,
            "formal_sampling_status": "not_started",
            "formal_study_status": "not_started",
            "human_calibration_status": "pending_reviewer",
            "pilot_status_historical": "incomplete",
            "created_utc": _now(),
        })
    ledger = run / "audit" / "ledger.json"
    if not ledger.exists():
        prior = json.loads((E6 / "audit" / "ledger.json").read_text(encoding="utf-8"))
        prior["reservations"] = {}
        prior["reserved_usd"] = 0.0
        prior["canary_reserved_usd"] = 0.0
        prior["opening_source"] = str(E6 / "audit" / "ledger.json")
        prior["opened_utc"] = _now()
        write_json(ledger, prior)
    return run


def _ledger_view(path: Path) -> dict:
    state = json.loads(path.read_text(encoding="utf-8"))
    return {
        "source": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "spent_usd": state.get("spent_usd"),
        "billing_unknown_usd": state.get("billing_unknown_usd"),
        "reserved_usd": state.get("reserved_usd"),
        "attempts": state.get("attempts"),
        "input_tokens": state.get("input_tokens"),
        "output_tokens": state.get("output_tokens"),
        "canary_attempts": state.get("canary_attempts"),
        "canary_spent_usd": state.get("canary_spent_usd"),
        "canary_input_tokens": state.get("canary_input_tokens"),
        "canary_output_tokens": state.get("canary_output_tokens"),
    }


def stage_j0(run_id: str) -> int:
    import unittest
    run = init_run(run_id)
    suite = unittest.defaultTestLoader.discover(str(PACKAGE / "tests"), pattern="test_readiness.py")
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    opening = _ledger_view(run / "audit" / "ledger.json")
    parent = _ledger_view(E6 / "audit" / "ledger.json")
    write_json(run / "audit" / "ledger_opening.json", opening)
    write_json(run / "audit" / "parent_ledger.json", parent)
    room = round_block_reasons(opening, json.loads((run / "audit" / "ledger.json").read_text(encoding="utf-8")), 0, 0, 0.0)
    write_json(run / "audit" / "round_budget.json", {
        "round_usd": ROUND_USD,
        "round_attempts": ROUND_ATTEMPTS,
        "round_input": ROUND_INPUT,
        "round_output": ROUND_OUTPUT,
        "canary_attempts_already": opening.get("canary_attempts"),
        "canary_attempt_cap": LIMITS["canary_attempts"],
        "blocked_before_start": room,
    })
    write_json(run / "jobs" / "j0.json", {"utc": _now(), "tests_failed": len(result.failures), "tests_errors": len(result.errors)})
    return 0 if result.wasSuccessful() else 1


def _index_comparisons() -> dict[str, dict]:
    found = {}
    for row in _read_jsonl(DIAG1 / "responses" / "responses.jsonl"):
        if row.get("task") != "diagnostic_compare":
            continue
        parsed = row.get("parsed") if isinstance(row.get("parsed"), dict) else {}
        for item in parsed.get("comparisons") or []:
            if not isinstance(item, dict):
                continue
            key = item.get("comparison_id") or item.get("candidate_key")
            if key and key not in found:
                found[key] = item
    return found


def _audit_rows() -> list[dict]:
    stored = _rows(DIAG1 / "tables" / "diagnostic_comparisons.parquet")
    full = _index_comparisons()
    docs = {str(row["pmid"]): row for row in _rows(PREP / "inputs" / "documents.parquet")}
    rows = []
    for row in stored:
        key = row["candidate_key"]
        source = full.get(key)
        record = {
            "legacy_comparison_id": key,
            "response_found": bool(source),
            "still_missing": "",
        }
        missing = []
        if source is None:
            missing.append("full_response")
        for field, limit in FIELD_LIMITS.items():
            action = truncation_status(row.get(field), None if source is None else source.get(field), limit)
            record[f"{field}_action"] = action
            record[f"{field}_stored_len"] = len(collapsed(row.get(field)))
            record[f"{field}_full_len"] = None if source is None else len(collapsed(source.get(field)))
        quotes = source.get("evidence_quotes") if source else None
        nonempty = _quotes_nonempty(quotes)
        record["comparison_quotes_in_response"] = nonempty
        record["comparison_quotes_in_parquet"] = "evidence_quotes" in row
        if source is None:
            record["comparison_quote_action"] = "response_missing"
        elif nonempty and "evidence_quotes" not in row:
            record["comparison_quote_action"] = "omitted_from_parquet"
        else:
            record["comparison_quote_action"] = "not_present_in_response"
        record["field_link"] = "field_link_unknown"
        if quotes and not isinstance(quotes, (dict, list)):
            record["comparison_quote_action"] = "unparsed_shape"
        for side, pmid_field in (("a", "pmid_a"), ("b", "pmid_b")):
            pmid = str(row.get(pmid_field) or "")
            doc = docs.get(pmid) or {}
            text = doc.get("text") or ""
            digest = hashlib.sha256(text.encode()).hexdigest()
            shown = ""
            if isinstance(quotes, dict):
                bucket = quotes.get(f"abstract_{side}") or []
                if isinstance(bucket, list) and bucket:
                    shown = bucket[0] if isinstance(bucket[0], str) else ""
            located = span_record(text, shown, digest) if shown else {"match_count": 0, "match_status": "no_comparison_quote", "semantic_support": "not_assessed", "text_sha256": digest, "offsets": []}
            record[f"span_{side}"] = located["match_status"]
            record[f"span_count_{side}"] = located["match_count"]
        if record["span_a"] == "no_comparison_quote" and record["span_b"] == "no_comparison_quote":
            missing.append("comparison_quote")
        record["still_missing"] = ",".join(missing)
        record["semantic_support"] = "not_assessed"
        rows.append(record)
    return rows


def _recovered_payload(audit: list[dict]) -> list[dict]:
    full = _index_comparisons()
    docs = {str(row["pmid"]): row for row in _rows(PREP / "inputs" / "documents.parquet")}
    payloads = []
    for row in audit:
        source = full.get(row["legacy_comparison_id"]) or {}
        quotes = source.get("evidence_quotes") if isinstance(source.get("evidence_quotes"), dict) else {}
        located = {}
        for side in ("a", "b"):
            bucket = quotes.get(f"abstract_{side}") or []
            quote = bucket[0] if isinstance(bucket, list) and bucket and isinstance(bucket[0], str) else ""
            payloads_side = []
            # The comparison row does not repeat pmids here; locate later in case materials.
            located[side] = quote
        payloads.append({
            "legacy_comparison_id": row["legacy_comparison_id"],
            "shared_question": source.get("shared_question"),
            "relation_description": source.get("relation_description"),
            "missing_information": source.get("missing_information"),
            "why_it_blocks": source.get("why_it_blocks"),
            "inference_limit": source.get("inference_limit"),
            "evidence_quotes": quotes,
            "field_link": "field_link_unknown",
            "semantic_support": "not_assessed",
            "quote_preview_a": located["a"],
            "quote_preview_b": located["b"],
        })
    return payloads


def _case_map() -> list[dict]:
    selected = {row["comparison_id"]: row for row in _rows(DIAG1 / "tables" / "comparison_selected.parquet")}
    findings = {(row["pair_id"], str(row["pmid"]), str(row["local"])): row for row in _rows(DIAG1 / "tables" / "diagnostic_findings.parquet")}
    mapped = []
    for row in _rows(DIAG1 / "tables" / "diagnostic_comparisons.parquet"):
        meta = selected.get(row["candidate_key"]) or {}
        qid = question_id_for(row.get("primary_question_type") or "")
        left = findings.get((row["pair_id"], str(row["pmid_a"]), str(meta.get("finding_a") or "").split(":")[-1]), {})
        right = findings.get((row["pair_id"], str(row["pmid_b"]), str(meta.get("finding_b") or "").split(":")[-1]), {})
        case_id = case_identity(row["pair_id"], str(row["pmid_a"]), str(row["pmid_b"]), qid, left.get("text_version_id") or "", right.get("text_version_id") or "")
        mapped.append({
            "legacy_comparison_id": row["candidate_key"],
            "pair_id": row["pair_id"],
            "pmid_a": str(row["pmid_a"]),
            "pmid_b": str(row["pmid_b"]),
            "paper_pair": paper_pair_key(str(row["pmid_a"]), str(row["pmid_b"])),
            "legacy_question_type": row.get("primary_question_type") or "",
            "question_id": qid if qid in QUESTIONS else "mapping_unresolved",
            "new_case_id": case_id,
            "finding_a": meta.get("finding_a") or "",
            "finding_b": meta.get("finding_b") or "",
            "mapping_status": "mapped" if case_id else "mapping_unresolved",
        })
    return mapped


def _labels(pair_id: str) -> dict:
    names = {row["pair_id"]: row for row in _rows(PRIOR / "tables" / "pair_names.parquet")}
    row = names.get(pair_id) or {}
    gene_names = [part.strip() for part in str(row.get("gene_name") or "").split("|")]
    if pair_id == "3604__C0007131":
        gene_names = [part for part in gene_names if part and "cd137l" not in part.casefold() and "cd137-fc" not in part.casefold() and "4-1bbl" not in part.casefold()]
        for extra in ("TNFRSF9", "CD137", "4-1BB"):
            if extra not in gene_names:
                gene_names.append(extra)
    return {"gene_label": gene_names[0] if gene_names else pair_id, "target_names": gene_names, "disease_label": row.get("disease_name") or ""}


def _documents() -> dict[str, dict]:
    return {str(row["pmid"]): row for row in _rows(PREP / "inputs" / "documents.parquet")}


def _write_calibration(run: Path, recovered: dict[str, dict]) -> dict:
    docs = _documents()
    phase_a = run / "annotation" / "calibration_v2" / "phase_a"
    phase_b = run / "annotation" / "calibration_v2" / "phase_b"
    leaks = []
    ready = []
    for spec in CALIBRATION:
        labels = _labels(spec["pair_id"])
        papers = []
        missing = []
        for pmid in spec["pmids"]:
            doc = docs.get(pmid)
            if not doc or not (doc.get("text") or doc.get("title")):
                missing.append(pmid)
                continue
            papers.append({"pmid": pmid, "title": doc.get("title") or "", "abstract": doc.get("text") or "", "text_version_id": doc.get("text_version_id") or ""})
        question = QUESTIONS.get(spec["question_id"]) or "这篇摘要里，与目标基因相关的测量对象是什么，它是什么形式？"
        packet = {
            "case_code": spec["case_code"],
            "pair_id": spec["pair_id"],
            "gene_label": labels["gene_label"],
            "target_names": labels["target_names"],
            "disease_label": labels["disease_label"],
            "question_id": spec["question_id"],
            "question": question,
            "papers": papers,
            "answer": blank_answer(),
            "missing_inputs": missing,
        }
        wrapper = json.dumps({
            "case_code": spec["case_code"],
            "gene_label": labels["gene_label"],
            "target_names": labels["target_names"],
            "disease_label": labels["disease_label"],
            "question": question,
        }, ensure_ascii=False)
        leaks.extend(f"{spec['case_code']}:{token}" for token in a_pack_leaks(wrapper))
        (phase_a / f"{spec['case_code']}.json").write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
        lines = [f"# {spec['case_code']}", "", f"目标：{labels['gene_label']}；疾病记录名：{labels['disease_label']}", "", f"问题 {spec['question_id']}：{question}", ""]
        for paper in papers:
            lines.extend([f"## PMID {paper['pmid']}", "", paper["title"], "", paper["abstract"], ""])
        lines.append("答案表尚未填写。workflow_status=pending。")
        form = "\n".join(lines)
        (phase_a / f"{spec['case_code']}.md").write_text(form, encoding="utf-8")
        (phase_a / f"{spec['case_code']}.answers.jsonl").write_text("", encoding="utf-8")
        historical = _historical_for(spec, recovered, docs)
        (phase_b / f"{spec['case_code']}.json").write_text(json.dumps(historical, ensure_ascii=False, indent=2), encoding="utf-8")
        ready.append({"case_code": spec["case_code"], "papers": len(papers), "missing": missing})
    (run / "annotation" / "calibration_v2" / "instructions.md").write_text((PACKAGE / "protocol" / "calibration_instructions_v2.md").read_text(encoding="utf-8"), encoding="utf-8")
    return {"cases": ready, "leaks": leaks}


def _historical_for(spec: dict, recovered: dict[str, dict], docs: dict[str, dict]) -> dict:
    related = []
    if len(spec["pmids"]) == 2:
        wanted = set(spec["pmids"])
        for row in _rows(DIAG1 / "tables" / "diagnostic_comparisons.parquet"):
            if row["pair_id"] == spec["pair_id"] and {str(row["pmid_a"]), str(row["pmid_b"])} == wanted:
                full = recovered.get(row["candidate_key"]) or {}
                related.append({
                    "legacy_comparison_id": row["candidate_key"],
                    "legacy_question_type": row.get("primary_question_type"),
                    "direction_relation": row.get("direction_relation"),
                    "relation_description": full.get("relation_description") or row.get("relation_description"),
                    "evidence_quotes": full.get("evidence_quotes") or {},
                    "field_link": "field_link_unknown",
                    "label": "historical_model_output",
                    "non_gold": True,
                })
    findings = []
    for row in _read_jsonl(DIAG1 / "audit" / "reread_units.jsonl"):
        if row.get("pair_id") == spec["pair_id"] and str(row.get("pmid")) in set(spec["pmids"]):
            findings.append({"pmid": str(row.get("pmid")), "parsed": row.get("parsed"), "label": "historical_model_output", "non_gold": True})
    return {"case_code": spec["case_code"], "label": "historical_model_output", "non_gold": True, "comparisons": related, "extractions": findings}


def stage_j1(run_id: str) -> int:
    run = _run()
    audit = _audit_rows()
    write_parquet(run / "tables" / "evidence_export_audit.parquet", audit)
    recovered_rows = _recovered_payload(audit)
    recovered = {row["legacy_comparison_id"]: row for row in recovered_rows}
    with (run / "audit" / "recovered_comparisons.jsonl").open("w", encoding="utf-8") as handle:
        for row in recovered_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    mapped = _case_map()
    write_parquet(run / "tables" / "legacy_case_map.parquet", mapped)
    calibration = _write_calibration(run, recovered)
    write_json(run / "config" / "sampling_v2.json", {
        "executed": False,
        "seed": SEED,
        "formal_draw_live": "disabled",
        "cap_per_pair": 6,
        "unique_pmid_cap": 24,
        "case_cap": 24,
        "library_minimum": 2,
        "layer_order": list(LAYERS),
        "rule": "one from each nonempty layer, then round robin; shared counts toward both libraries once",
        "direction_quota": "removed",
    })
    for name, dest in (
        ("protocol_v2.md", "protocol.md"),
        ("annotation_guide_v2.md", "annotation_guide.md"),
        ("implementation_plan_v2.md", "implementation_plan.md"),
    ):
        (run / "reports" / dest).write_text((PACKAGE / "protocol" / name).read_text(encoding="utf-8"), encoding="utf-8")
    truncated = sum(1 for row in audit if row.get("relation_description_action") == "exporter_truncation")
    omitted = sum(1 for row in audit if row.get("comparison_quote_action") == "omitted_from_parquet")
    write_json(run / "audit" / "changes.json", {
        "contract": VERSION,
        "relation_exporter_truncation": f"{truncated}/{len(audit)}",
        "comparison_quotes_omitted": f"{omitted}/{len(audit)}",
        "new_case_mapped": sum(1 for row in mapped if row["mapping_status"] == "mapped"),
        "mapping_unresolved": sum(1 for row in mapped if row["mapping_status"] == "mapping_unresolved"),
        "calibration_leaks": calibration["leaks"],
        "formal_sampling": "not_started",
    })
    write_json(run / "jobs" / "j1.json", {"utc": _now(), "audit_rows": len(audit), "leaks": calibration["leaks"]})
    return 0 if not calibration["leaks"] else 2


def _stopped(run: Path) -> bool:
    return (run / "audit" / "round_stop.json").exists()


def _cache_get(run: Path, key: str):
    path = run / "technical_canary" / "cache" / f"{key}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _cache_put(run: Path, key: str, value: dict) -> None:
    path = run / "technical_canary" / "cache" / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _invoke(run: Path, task: str, user: str, prompt: str, logical: str, cap: int) -> dict:
    key = cache_key(task, user, prompt, logical, cap)
    cached = _cache_get(run, key)
    if cached and cached.get("status") in {"ok", "truncated"} and isinstance(cached.get("parsed"), dict):
        cached["cache_hit"] = True
        return cached
    if _stopped(run):
        return {"status": "stopped_after_budget", "parsed": None}
    opening = read_json(run / "audit" / "ledger_opening.json")
    state = json.loads((run / "audit" / "ledger.json").read_text(encoding="utf-8"))
    next_input = max(1, len(prompt + user) // 4 + 300)
    reasons = round_block_reasons(opening, state, next_input, cap, estimate_usd(next_input, cap))
    if reasons:
        write_json(run / "audit" / "round_stop.json", {"at": _now(), "reasons": reasons, "logical_id": logical})
        return {"status": "not_evaluated_budget", "parsed": None, "reasons": reasons}
    input_path = run / "technical_canary" / "inputs" / f"{hashlib.sha256(logical.encode()).hexdigest()[:16]}.json"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(json.dumps({
        "logical_id": logical, "task": task, "contract": VERSION, "max_completion_tokens": cap,
        "model_id": "gpt-6.1-sol", "service_tier": "standard", "temperature": "omitted",
        "prompt": prompt, "user": user,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    from pairpilot.continue_pilot import _call
    result = _call(run, "canary", task, user, prompt, replicate_id=logical, max_output=cap)
    repair_flag = run / "audit" / "format_repair.json"
    if result.get("status") == "unparsed" and not repair_flag.exists():
        failed = run / "technical_canary" / "first_failure.json"
        failed.write_text(json.dumps({"logical_id": logical, "status": result.get("status"), "raw": result.get("raw")}, ensure_ascii=False), encoding="utf-8")
        write_json(repair_flag, {"at": _now(), "logical_id": logical, "kept_first_status": result.get("status")})
        repair_user = user + "\nReturn one JSON object only."
        result = _call(run, "canary", task + "_repair", repair_user, prompt, replicate_id=logical + ":repair", max_output=cap)
        result["repair"] = True
    if isinstance(result.get("parsed"), dict) and result.get("status") in {"ok", "truncated"}:
        result["logical_id"] = logical
        _cache_put(run, key, result)
    if result.get("status") in {"quota_exhausted", "not_evaluated_budget"}:
        write_json(run / "audit" / "round_stop.json", {"at": _now(), "status": result.get("status"), "logical_id": logical})
    return result


def _extract_user(pair_id: str, pmid: str) -> str:
    doc = _documents().get(pmid) or {}
    labels = _labels(pair_id)
    return json.dumps({
        "pair_id": pair_id,
        "gene_label": labels["gene_label"],
        "target_names": labels["target_names"],
        "disease_label": labels["disease_label"],
        "pmid": pmid,
        "title": doc.get("title") or "",
        "abstract": doc.get("text") or "",
        "text_version_id": doc.get("text_version_id") or "",
    }, ensure_ascii=False)


def _check_quotes(parsed: dict, texts: dict[str, str]) -> dict:
    refs = []
    if isinstance(parsed.get("findings"), list):
        for finding in parsed["findings"]:
            if isinstance(finding, dict):
                for item in finding.get("evidence") or []:
                    if isinstance(item, dict):
                        refs.append(item)
    for item in parsed.get("evidence") or []:
        if isinstance(item, dict):
            refs.append(item)
    for component in parsed.get("components") or []:
        if isinstance(component, dict):
            for item in component.get("evidence") or []:
                if isinstance(item, dict):
                    refs.append(item)
    located = []
    for item in refs:
        quote = item.get("quote") or ""
        pmid = str(item.get("pmid") or "")
        haystack = texts.get(pmid) or "\n".join(texts.values())
        record = span_record(haystack, quote, hashlib.sha256(haystack.encode()).hexdigest())
        record["supported_field"] = item.get("supported_field") or "missing"
        record["pmid"] = pmid
        located.append(record)
    return {
        "n_quotes": len(refs),
        "zero": sum(1 for row in located if row["match_status"] == "zero"),
        "multiple": sum(1 for row in located if row["match_status"] == "multiple"),
        "unique": sum(1 for row in located if row["match_status"] == "unique"),
        "missing_field": sum(1 for row in located if row["supported_field"] == "missing"),
        "semantic_support": "not_assessed",
        "details": located,
    }


def stage_j2(run_id: str) -> int:
    run = _run()
    prompt = (PACKAGE / "prompts" / "extract_protocol_v2.txt").read_text(encoding="utf-8")
    first = True
    for pair_id, pmid in CANARY_PAPERS:
        logical = f"{VERSION}|extract|{pair_id}|{pmid}"
        if first and _stopped(run):
            _mark(run, logical, {"status": "not_started"})
            continue
        user = _extract_user(pair_id, pmid)
        doc = _documents().get(pmid) or {}
        result = _invoke(run, "readiness_extract_v2", user, prompt, logical, EXTRACT_CAP)
        check = _check_quotes(result.get("parsed") or {}, {pmid: (doc.get("title") or "") + "\n" + (doc.get("text") or "")})
        status = result.get("status")
        passed = status in {"ok", "truncated"} and isinstance(result.get("parsed"), dict) and isinstance((result.get("parsed") or {}).get("findings"), list)
        _mark(run, logical, {"status": status, "technical": technical_label(passed, check), "quote_check": check, "returned_model": result.get("returned_model"), "cache_hit": result.get("cache_hit", False)})
        if first and not passed:
            write_json(run / "audit" / "round_stop.json", {"at": _now(), "reason": "first_extract_failed", "logical_id": logical})
            break
        first = False
        if status in {"quota_exhausted", "not_evaluated_budget", "stopped_after_budget"}:
            break
    return 0


def _mark(run: Path, logical: str, payload: dict) -> None:
    path = run / "technical_canary" / "units.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"logical_id": logical, "at": _now(), **payload}, ensure_ascii=False) + "\n")


def _unit_ok(run: Path, logical: str) -> dict | None:
    rows = [row for row in _read_jsonl(run / "technical_canary" / "units.jsonl") if row.get("logical_id") == logical and row.get("technical") in {"technical_pass", "technical_partial"}]
    return rows[-1] if rows else None


def stage_j3(run_id: str) -> int:
    run = _run()
    prompt = (PACKAGE / "prompts" / "compare_protocol_v2.txt").read_text(encoding="utf-8")
    for question_id, pair_id, pmid_a, pmid_b in CANARY_COMPARES:
        logical = f"{VERSION}|compare|{question_id}|{paper_pair_key(pmid_a, pmid_b)}"
        left = _unit_ok(run, f"{VERSION}|extract|{pair_id}|{pmid_a}")
        right = _unit_ok(run, f"{VERSION}|extract|{pair_id}|{pmid_b}")
        if not left or not right or _stopped(run):
            _mark(run, logical, {"status": "blocked_dependency", "technical": "technical_failed"})
            continue
        docs = _documents()
        findings = []
        for pmid in (pmid_a, pmid_b):
            findings.extend(_findings_from_cache(run, f"{VERSION}|extract|{pair_id}|{pmid}", pmid))
        user = json.dumps({
            "pair_id": pair_id,
            "gene_label": _labels(pair_id)["gene_label"],
            "disease_label": _labels(pair_id)["disease_label"],
            "question_id": question_id,
            "question": QUESTIONS[question_id],
            "abstracts": [
                {"pmid": pmid, "title": (docs.get(pmid) or {}).get("title") or "", "text": (docs.get(pmid) or {}).get("text") or "", "text_version_id": (docs.get(pmid) or {}).get("text_version_id") or ""}
                for pmid in (pmid_a, pmid_b)
            ],
            "findings_to_check": findings,
            "instruction": "These findings are not facts. Report conflicts with the abstracts.",
        }, ensure_ascii=False)
        result = _invoke(run, "readiness_compare_v2", user, prompt, logical, COMPARE_CAP)
        texts = {pmid: ((docs.get(pmid) or {}).get("title") or "") + "\n" + ((docs.get(pmid) or {}).get("text") or "") for pmid in (pmid_a, pmid_b)}
        check = _check_quotes(result.get("parsed") or {}, texts)
        passed = result.get("status") in {"ok", "truncated"} and isinstance(result.get("parsed"), dict)
        _mark(run, logical, {"status": result.get("status"), "technical": technical_label(passed, check), "quote_check": check, "returned_model": result.get("returned_model")})
    return 0


def _findings_from_cache(run: Path, logical: str, pmid: str) -> list[dict]:
    cache_dir = run / "technical_canary" / "cache"
    if not cache_dir.exists():
        return []
    for path in cache_dir.glob("*.json"):
        body = json.loads(path.read_text(encoding="utf-8"))
        if body.get("logical_id") != logical:
            continue
        parsed = body.get("parsed") if isinstance(body.get("parsed"), dict) else {}
        rows = []
        for index, finding in enumerate(parsed.get("findings") or []):
            if isinstance(finding, dict):
                rows.append({"finding_id": f"{logical}|{index}", "pmid": pmid, "finding": finding})
        return rows
    return []


def stage_j4(run_id: str) -> int:
    run = _run()
    units = _read_jsonl(run / "technical_canary" / "units.jsonl") if (run / "technical_canary" / "units.jsonl").exists() else []
    by_id = {}
    for row in units:
        by_id[row["logical_id"]] = row
    expected = [f"{VERSION}|extract|{pair}|{pmid}" for pair, pmid in CANARY_PAPERS]
    expected += [f"{VERSION}|compare|{qid}|{paper_pair_key(a, b)}" for qid, _pair, a, b in CANARY_COMPARES]
    labels = [(by_id.get(key) or {}).get("technical") or "not_run" for key in expected]
    if labels and all(item == "technical_pass" for item in labels):
        canary = "technical_pass"
    elif any(item in {"technical_pass", "technical_partial"} for item in labels):
        canary = "technical_partial"
    elif (run / "audit" / "round_stop.json").exists():
        canary = "blocked"
    else:
        canary = "technical_failed"
    leaks = []
    changes = read_json(run / "audit" / "changes.json") if (run / "audit" / "changes.json").exists() else {}
    leaks = changes.get("calibration_leaks") or []
    repair = "completed" if (run / "tables" / "evidence_export_audit.parquet").exists() else "missing"
    annotation = "prepared_blank" if (run / "annotation" / "calibration_v2" / "phase_a" / "CAL01.json").exists() and not leaks else "needs_fix"
    ready = readiness_label(repair, annotation, canary) == "ready_for_calibration"
    opening = read_json(run / "audit" / "ledger_opening.json")
    closing = _ledger_view(run / "audit" / "ledger.json")
    parent = _ledger_view(E6 / "audit" / "ledger.json")
    write_json(run / "audit" / "ledger_closing.json", closing)
    manifest = read_json(run / "manifest.json")
    manifest.update({
        "artifact_repair_status": repair,
        "annotation_package_status": annotation,
        "technical_canary_status": canary,
        "human_calibration_status": "pending_reviewer",
        "formal_sampling_status": "not_started",
        "formal_study_status": "not_started",
        "pilot_status_historical": "incomplete",
        "readiness_status": "ready_for_calibration" if ready else "partial",
        "prompt_status": PROMPT_STATUS,
        "parent_ledger_unchanged": parent["sha256"] == read_json(run / "audit" / "parent_ledger.json")["sha256"],
        "updated_utc": _now(),
    })
    write_json(run / "manifest.json", manifest)
    (run / "reports" / "report.md").write_text(_report(run, manifest, opening, closing, by_id, expected), encoding="utf-8")
    write_json(run / "jobs" / "j4.json", {"utc": _now(), "canary": canary, "passed": labels.count("technical_pass"), "expected": len(expected)})
    return 0 if canary in {"technical_pass", "technical_partial", "blocked"} or repair == "completed" else 2


def _report(run: Path, manifest: dict, opening: dict, closing: dict, by_id: dict, expected: list[str]) -> str:
    lines = [
        "# readiness",
        "",
        f"child `{run.name}`，parent `{PARENT.name}`。这是协议修订、空白试用包和小范围技术检查。",
        "",
        f"导出修复：{manifest.get('artifact_repair_status')}。试用包：{manifest.get('annotation_package_status')}。技术检查：{manifest.get('technical_canary_status')}。人工校准：pending_reviewer。正式抽样和正式实验：not_started。",
        "",
        "六个技术单元：",
    ]
    for key in expected:
        row = by_id.get(key) or {"status": "not_run", "technical": "not_run"}
        check = row.get("quote_check") or {}
        lines.append(f"- {key}：调用 {row.get('status')}，技术 {row.get('technical')}，引用 {check.get('n_quotes')}，唯一 {check.get('unique')}，多处 {check.get('multiple')}，未命中 {check.get('zero')}。语义支持未评估。")
    lines.extend([
        "",
        f"父账本未改写：{manifest.get('parent_ledger_unchanged')}。本 child 开读花费 {opening.get('spent_usd')}，回读 {closing.get('spent_usd')}；attempt {opening.get('attempts')} → {closing.get('attempts')}；输出 token {opening.get('output_tokens')} → {closing.get('output_tokens')}；未知计费 {opening.get('billing_unknown_usd')} → {closing.get('billing_unknown_usd')}。",
        "",
        "phase_a 在 annotation/calibration_v2/phase_a。答案文件是空的。phase_b 只在 A 锁定之后使用。",
        "",
        "下一步只接真实的四例试用记录。没有记录之前不冻结样本，也不再调用模型。",
        "",
    ])
    return "\n".join(lines)


STAGES = {"j0": stage_j0, "j1": stage_j1, "j2": stage_j2, "j3": stage_j3, "j4": stage_j4}


def main() -> int:
    if _login():
        raise SystemExit("refusing to run readiness on the login node")
    try:
        return STAGES[sys.argv[1]](os.environ["PILOT_CHILD_RUN"])
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

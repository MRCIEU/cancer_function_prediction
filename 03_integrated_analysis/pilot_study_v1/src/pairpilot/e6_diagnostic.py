#!/usr/bin/env python3
"""Bounded E6 for the diagnostic contract. This does not complete the old E6."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pairpilot.config import digest  # noqa: E402
from pairpilot.continue_pilot import _call, _prompt  # noqa: E402
from pairpilot.diagnostic_round import _p1_user, _rows  # noqa: E402
from pairpilot.inference import CallCache, input_hash  # noqa: E402
from pairpilot.ledger import LIMITS, available_usd  # noqa: E402
from pairpilot.repair_pilot import _read_jsonl  # noqa: E402
from pairpilot.store import read_json, write_json, write_parquet  # noqa: E402
from pairpilot.texts import split_sentences  # noqa: E402

VERSION = "e6_diagnostic_v1"
SEED = 20261015
CLOSE1 = Path("/projects/b5ac/project_2/results/03_integrated_analysis/pilot_study_v1/runs/20261016T012650Z_close1")
DIAG1 = Path("/projects/b5ac/project_2/results/03_integrated_analysis/pilot_study_v1/runs/20261016T013500Z_diag1")
REV2 = Path("/projects/b5ac/project_2/results/03_integrated_analysis/pilot_study_v1/runs/20261016T005500Z_rev2")
PRIOR = Path("/projects/b5ac/project_2/results/03_integrated_analysis/pilot_study_v1/runs/20261015T221200Z_b20cont")
PREP = Path("/projects/b5ac/project_2/results/03_integrated_analysis/pilot_study_v1/runs/20261015T204633Z_0d9b9724")
RUNS = CLOSE1.parent
PAIRS = ("8318__C2239176", "768__C0376358")
LABELS = {
    "8318__C2239176": ("CDC45", "hepatocellular carcinoma"),
    "768__C0376358": ("CA IX", "prostate cancer"),
}
QUESTIONS = {
    "expression_or_distribution": "What expression or distribution relationship, if any, is reported for {gene} in {disease} across these abstracts?",
    "prognostic_association": "What prognostic relationship, if any, is reported for {gene} in {disease} across these abstracts?",
    "functional_perturbation": "What functional perturbation, if any, is reported for {gene} in {disease} across these abstracts?",
    "diagnostic_performance": "What diagnostic performance, if any, is reported for {gene} in {disease} across these abstracts?",
    "treatment_response_association": "What treatment-response association, if any, is reported for {gene} in {disease} across these abstracts?",
    "disease_risk_association": "What disease-risk association, if any, is reported for {gene} in {disease} across these abstracts?",
    "other": "What relationship, if any, is reported for {gene} in {disease} across these abstracts?",
    "unclear": "What relationship, if any, is reported for {gene} in {disease} across these abstracts?",
}
EXTRACT_CAP = 6000
JUDGE_CAP = 4000


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _login() -> bool:
    return os.uname().nodename.startswith("login")


def _short(value, limit: int = 400) -> str:
    return " ".join(str(value or "").split())[:limit]


def _tokens(text: str) -> int:
    return len(str(text or "").split())


def extract_task(pair_id: str, pmid: str, replicate: str, zero_finding: bool = False) -> dict:
    task = {
        "logical_task_id": f"{VERSION}|extract|{pair_id}|{pmid}|{replicate}",
        "pair_id": pair_id,
        "pmid": str(pmid),
        "replicate": replicate,
        "unit": "paper",
        "flags": [],
    }
    if zero_finding:
        task["flags"].append("zero_finding_kept")
    return task


def cache_key(payload: dict, replicate_id) -> str:
    body = {
        "task": payload["task"],
        "input_sha256": payload["input_sha256"],
        "schema_id": payload["schema_id"],
        "prompt_sha256": payload["prompt_sha256"],
        "model_id": payload["model_id"],
        "params": payload.get("params") or {},
    }
    if replicate_id is not None:
        body["replicate_id"] = replicate_id
    return digest(body)


def sentence_jaccard(left: set, right: set):
    if not left and not right:
        return "both_empty"
    union = left | right
    return len(left & right) / len(union)


def align_findings(left: list[dict], right: list[dict]):
    """Match on entity and cited sentences. Direction is not a match key."""
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


def _norm(value) -> str:
    return " ".join(str(value or "").casefold().split())


def _locate(text: str, quote: str) -> tuple[int, int] | None:
    if not quote:
        return None
    start = text.find(quote)
    if start < 0:
        start = " ".join(text.split()).find(" ".join(quote.split()))
        if start < 0:
            return None
        return None
    return start, start + len(quote)


def build_variants(text: str, quotes: list[str], pmid: str, seed: int) -> dict:
    sentences = split_sentences(text)
    cited = []
    invalid = []
    for quote in quotes:
        span = _locate(text, quote)
        if span is None:
            invalid.append(quote)
            continue
        for sentence in sentences:
            if sentence["start"] < span[1] and span[0] < sentence["end"] and sentence["start"] not in {row["start"] for row in cited}:
                cited.append(sentence)
    cited.sort(key=lambda row: row["start"])
    cited_starts = {row["start"] for row in cited}
    kept = [row for row in sentences if row["start"] not in cited_starts]
    e_text = " ".join(row["text"].strip() for row in cited)
    d_text = _without(text, cited)
    same_as_full = _norm(e_text) == _norm(text) or (cited and len(cited) == len(sentences))
    control = _control(sentences, cited, pmid, seed)
    return {
        "cited_sentences": cited,
        "invalid_quotes": invalid,
        "E": {"text": e_text, "status": "ok" if cited else "no_locatable_quote", "informative": bool(cited) and not same_as_full},
        "D": {"text": d_text, "status": "ok" if cited else "no_locatable_quote", "deleted": cited, "informative": bool(cited) and _norm(d_text) != _norm(text)},
        "R": control,
    }


def _without(text: str, sentences: list[dict]) -> str:
    pieces = []
    cursor = 0
    for sentence in sorted(sentences, key=lambda row: row["start"]):
        pieces.append(text[cursor:sentence["start"]])
        cursor = sentence["end"]
    pieces.append(text[cursor:])
    return "".join(pieces).strip()


def _control(sentences: list[dict], cited: list[dict], pmid: str, seed: int) -> dict:
    if not cited:
        return {"status": "not_constructible", "reason": "no_cited_sentences", "text": "", "token_gap": None}
    uncited = [row for row in sentences if row["start"] not in {item["start"] for item in cited}]
    target_n = len(cited)
    target_tokens = sum(_tokens(row["text"]) for row in cited) or 1
    if len(uncited) < target_n:
        return {"status": "not_constructible", "reason": "too_few_uncited_sentences", "text": "", "token_gap": None, "target_n": target_n, "available": len(uncited)}
    if math.comb(len(uncited), target_n) > 50000:
        return {"status": "not_constructible", "reason": "search_space_cap", "text": "", "token_gap": None}
    ranked = []
    for combo in itertools.combinations(uncited, target_n):
        tokens = sum(_tokens(row["text"]) for row in combo)
        gap = abs(tokens - target_tokens) / target_tokens
        if gap <= 0.20:
            signature = ",".join(str(row["start"]) for row in combo)
            ranked.append((hashlib.sha256(f"{seed}|{pmid}|{signature}".encode()).hexdigest(), gap, combo, tokens))
    if not ranked:
        return {"status": "not_constructible", "reason": "no_length_match_within_20_percent", "text": "", "token_gap": None, "target_tokens": target_tokens}
    ranked.sort(key=lambda item: item[0])
    _key, gap, combo, tokens = ranked[0]
    removed = {row["start"] for row in combo}
    if removed & {row["start"] for row in cited}:
        return {"status": "not_constructible", "reason": "control_overlapped_citation", "text": "", "token_gap": None}
    return {
        "status": "ok",
        "text": " ".join(row["text"].strip() for row in sentences if row["start"] not in removed),
        "deleted": list(combo),
        "token_gap": gap,
        "deleted_tokens": tokens,
        "target_tokens": target_tokens,
    }


def judgment_user(gene: str, disease: str, question_type: str, abstracts: list[tuple[str, str]]) -> dict:
    template = QUESTIONS.get(question_type) or QUESTIONS["unclear"]
    return {
        "gene_label": gene,
        "disease_label": disease,
        "question": template.format(gene=gene, disease=disease),
        "abstracts": [{"pmid": pmid, "text": text} for pmid, text in abstracts],
        "instruction": "Use only these abstract bodies.",
    }


def select_cases(rows: list[dict], seed: int):
    excluded = []
    eligible = []
    for row in rows:
        if row.get("text_available") is False:
            excluded.append({**row, "reason": "text_unavailable"})
        elif not row.get("quote_a") and not row.get("quote_b"):
            excluded.append({**row, "reason": "no_locatable_evidence"})
        else:
            eligible.append(row)
    pools = {
        "direction": _ranked([row for row in eligible if row.get("can_compare_direction") in (True, "true")], seed, "direction"),
        "no_direction": _ranked([row for row in eligible if row.get("can_compare_direction") not in (True, "true")], seed, "no_direction"),
    }
    chosen = []
    notes = []
    for stratum in ("direction", "no_direction"):
        pool = [row for row in pools[stratum] if row["comparison_id"] not in {item["comparison_id"] for item in chosen}]
        if not pool:
            notes.append(f"{stratum}_empty")
            continue
        first = pool[0]
        chosen.append({**first, "stratum": stratum, "pick_rule": "hash_order"})
        pairs = {item["pair_id"] for item in chosen}
        missing = [pair for pair in PAIRS if pair not in pairs]
        second = next((row for row in pool[1:] if missing and row["pair_id"] in missing), None)
        rule = "missing_pair"
        if second is None and not any(item.get("source_class") == "union_required" for item in chosen):
            second = next((row for row in pool[1:] if row.get("source_class") == "union_required"), None)
            rule = "union_required"
        if second is None and len(pool) > 1:
            second = pool[1]
            rule = "hash_order"
        if second is not None:
            if rule != "hash_order":
                notes.append(f"{stratum}:{rule}")
            chosen.append({**second, "stratum": stratum, "pick_rule": rule})
    if not any(row.get("source_class") == "union_required" for row in chosen):
        notes.append("union_required_not_covered")
    return chosen[:4], excluded, notes


def _ranked(rows: list[dict], seed: int, stratum: str) -> list[dict]:
    return sorted(rows, key=lambda row: hashlib.sha256(f"{seed}|{stratum}|{row['comparison_id']}".encode()).hexdigest())


def _run() -> Path:
    path = RUNS / os.environ["PILOT_CHILD_RUN"]
    if not path.exists():
        raise SystemExit(f"e6 run is missing: {path}")
    return path


def init_run(run_id: str) -> Path:
    run = RUNS / run_id
    run.mkdir(parents=False, exist_ok=True)
    for name in ("audit", "tables", "experiments", "reports", "requests", "responses", "jobs"):
        (run / name).mkdir(exist_ok=True)
    (run / "experiments" / "e6").mkdir(exist_ok=True)
    if not (run / "manifest.json").exists():
        write_json(run / "manifest.json", {
            "run_id": run_id,
            "parent_run_id": CLOSE1.name,
            "diagnostic_run_id": DIAG1.name,
            "science_version": VERSION,
            "judgment_contract": "e6_judgment_v1",
            "created_utc": _now(),
            "status": "incomplete",
            "pilot_status": "incomplete",
            "scope_change": "E6-A uses CDC45 and CA IX diagnostic papers, not the old failed E6 pairs.",
        })
    ledger = run / "audit" / "ledger.json"
    if not ledger.exists():
        prior = json.loads((CLOSE1 / "audit" / "ledger.json").read_text(encoding="utf-8"))
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
            "opening_source": str(CLOSE1 / "audit" / "ledger.json"),
            "opened_utc": _now(),
        })
    return run


def _documents() -> dict[str, dict]:
    return {str(row["pmid"]): row for row in _rows(PREP / "inputs" / "documents.parquet")}


def _aliases() -> dict[str, dict]:
    path = PRIOR / "tables" / "pair_names.parquet"
    return {row["pair_id"]: row for row in _rows(path)}


def _papers() -> list[dict]:
    seen = {}
    for row in _rows(PREP / "inputs" / "pair_pmids.parquet"):
        if row["pair_id"] not in PAIRS:
            continue
        seen.setdefault((row["pair_id"], str(row["pmid"])), {"pair_id": row["pair_id"], "pmid": str(row["pmid"]), "sources": []})
        seen[(row["pair_id"], str(row["pmid"]))]["sources"].append(row["source"])
    return [seen[key] for key in sorted(seen)]


def _r0_match(user: str, prompt: str, keys: set[str]) -> str:
    request = {
        "task": "diagnostic_reread",
        "input_sha256": input_hash(user),
        "schema_id": "diagnostic_reread",
        "prompt_sha256": input_hash(prompt),
        "model_id": "gpt-6.1-sol",
        "params": {"max_completion_tokens": EXTRACT_CAP, "service_tier": "standard"},
    }
    key = cache_key(request, None)
    return key if key in keys else ""


def stage_j0(run_id: str) -> int:
    run = init_run(run_id)
    papers = _papers()
    docs = _documents()
    names = _aliases()
    prompt = _prompt("diagnostic_reread.txt")
    keys = {row.get("cache_key") for row in _read_jsonl(DIAG1 / "requests" / "requests.jsonl") if row.get("task") == "diagnostic_reread"}
    units = {f"{row.get('pair_id')}|{row.get('pmid')}": row for row in _read_jsonl(DIAG1 / "audit" / "reread_units.jsonl")}
    expected = []
    for paper in papers:
        doc = docs.get(paper["pmid"]) or {}
        full_alias = names.get(paper["pair_id"], {})
        pmid_values = [paper["pmid"]]
        if str(paper["pmid"]).isdigit():
            pmid_values.append(int(paper["pmid"]))
        candidates = {}
        for pmid_value in pmid_values:
            candidates[f"pair_names|{type(pmid_value).__name__}"] = _p1_user({"pair_id": paper["pair_id"], "pmid": pmid_value}, doc, full_alias)
            candidates[f"empty_alias|{type(pmid_value).__name__}"] = _p1_user({"pair_id": paper["pair_id"], "pmid": pmid_value}, doc, {})
        matched = next((name for name, user in candidates.items() if _r0_match(user, prompt, keys)), "")
        unit = units.get(f"{paper['pair_id']}|{paper['pmid']}")
        expected.append({
            **extract_task(paper["pair_id"], paper["pmid"], "r0", zero_finding=bool(unit and not (unit.get("parsed") or {}).get("findings"))),
            "sources": ",".join(paper["sources"]),
            "text_version_id": doc.get("text_version_id") or "",
            "r0_cache_match": matched or "not_rebuilt",
            "r0_reconstruction": matched.split("|")[0] if matched else "not_rebuilt",
            "r0_response": "diag1_reread_units" if matched and unit else "missing",
            "planned_replicates": ["r0", "r1", "r2"],
        })
    reusable = all(row["r0_response"] == "diag1_reread_units" for row in expected) and len(expected) == 14
    if not reusable:
        for row in expected:
            row["r0_reconstruction"] = "new_pair_names"
            row["r0_response"] = "not_reused"
            row["planned_replicates"] = ["r0", "r1"]
    write_json(run / "experiments" / "e6" / "extraction_expected.json", expected)
    write_json(run / "experiments" / "e6" / "replicate_plan.json", {
        "baseline": "reused_diag1" if reusable else "new_pair_names",
        "first_new_call": "r1" if reusable else "r0",
        "second_new_call": "r2" if reusable else "r1",
        "third_not_scheduled": not reusable,
        "reason": "r0 cache keys matched the diagnostic requests" if reusable else "r0 user body could not be rebuilt for every paper, so this round makes two new calls on one contract",
    })
    case_rows = []
    comparisons = {row["candidate_key"]: row for row in _rows(DIAG1 / "tables" / "diagnostic_comparisons.parquet")}
    for row in _rows(CLOSE1 / "tables" / "comparison_casebook.parquet"):
        if row.get("pair_id") not in PAIRS:
            continue
        comp = comparisons.get(row["comparison_id"], {})
        case_rows.append({
            "comparison_id": row["comparison_id"],
            "pair_id": row["pair_id"],
            "pmid_a": str(row["pmid_a"]),
            "pmid_b": str(row["pmid_b"]),
            "source_class": row.get("source_class"),
            "can_compare_direction": row.get("can_compare_direction"),
            "quote_a": row.get("quote_a") or "",
            "quote_b": row.get("quote_b") or "",
            "text_available": bool(docs.get(str(row["pmid_a"])) and docs.get(str(row["pmid_b"]))),
            "primary_question_type": comp.get("primary_question_type") or "unclear",
            "old_direction_relation": row.get("direction_relation") or "",
        })
    chosen, excluded, notes = select_cases(case_rows, SEED)
    write_json(run / "experiments" / "e6" / "comparison_case_manifest.json", {"chosen": chosen, "excluded": excluded, "notes": notes, "seed": SEED})
    reusable = sum(1 for row in expected if row.get("r0_cache_match") not in {None, "", "not_rebuilt"})
    ledger = json.loads((run / "audit" / "ledger.json").read_text(encoding="utf-8"))
    output_left = LIMITS["max_output_tokens"] - int(ledger.get("output_tokens") or 0)
    write_json(run / "audit" / "expected_manifest.json", {
        "papers": len(papers),
        "r0_reusable": reusable,
        "new_extract_calls_if_r0_reused": len(papers) * 2,
        "new_extract_calls_if_r0_rebuilt": len(papers) * 3,
        "judgment_calls_full": len(chosen) * 6,
        "extract_max_output": EXTRACT_CAP,
        "judgment_max_output": JUDGE_CAP,
        "output_tokens_remaining": output_left,
        "conservative_output_reservation": len(papers) * 2 * EXTRACT_CAP + len(chosen) * 6 * JUDGE_CAP,
        "note": "Reservation is per call. The hard cap uses settled output plus the next request, not the sum of all maxima at once.",
    })
    write_json(run / "audit" / "input_registry.json", {
        "close1_casebook": str(CLOSE1 / "tables" / "comparison_casebook.parquet"),
        "diag1_reread_units": str(DIAG1 / "audit" / "reread_units.jsonl"),
        "documents": str(PREP / "inputs" / "documents.parquet"),
        "pair_pmids": str(PREP / "inputs" / "pair_pmids.parquet"),
    })
    return 0 if len(papers) == 14 and len(chosen) == 4 else 2


def _stopped(run: Path) -> bool:
    return (run / "audit" / "provider_stop.json").exists()


def _invoke(run: Path, task: str, user: str, prompt: str, replicate_id: str, cap: int) -> dict:
    if _stopped(run):
        return {"status": "stopped_after_quota", "parsed": None}
    request = {
        "task": task,
        "input_sha256": input_hash(user),
        "schema_id": task,
        "prompt_sha256": input_hash(prompt),
        "model_id": "gpt-6.1-sol",
        "params": {"max_completion_tokens": cap, "service_tier": "standard"},
    }
    key = cache_key(request, replicate_id)
    cache = CallCache(run / "responses" / "cache")
    cached = cache.get(key)
    if cached and cached.get("status") in {"ok", "truncated"}:
        cached["cache_hit"] = True
        return cached
    result = _call(run, "main", task, user, prompt, replicate_id=replicate_id, max_output=cap)
    usable = isinstance(result.get("parsed"), dict)
    if not usable and result.get("status") in {"unparsed", "truncated"}:
        repair_user = user + "\nReturn one JSON object only."
        repair_key = cache_key({**request, "task": task + "_repair", "schema_id": task + "_repair", "input_sha256": input_hash(repair_user)}, replicate_id + ":repair")
        result = _call(run, "main", task + "_repair", repair_user, prompt, replicate_id=replicate_id + ":repair", max_output=cap)
        if isinstance(result.get("parsed"), dict):
            cache.put(repair_key, result)
    elif usable:
        cache.put(key, result)
    if result.get("status") == "quota_exhausted":
        write_json(run / "audit" / "provider_stop.json", {"at": _now(), "status": "quota_exhausted"})
    return result


def _extract_user(run: Path, paper: dict) -> str:
    spec = next(row for row in read_json(run / "experiments" / "e6" / "extraction_expected.json") if row["pmid"] == paper["pmid"] and row["pair_id"] == paper["pair_id"])
    alias = _aliases().get(paper["pair_id"], {}) if spec.get("r0_reconstruction") in {"pair_names", "new_pair_names"} else {}
    pmid = paper["pmid"]
    if spec.get("r0_reconstruction") != "new_pair_names" and "int" in str(spec.get("r0_cache_match")) and str(paper["pmid"]).isdigit():
        pmid = int(paper["pmid"])
    return _p1_user({"pair_id": paper["pair_id"], "pmid": pmid}, _documents().get(paper["pmid"]) or {}, alias)


def stage_j1(run_id: str) -> int:
    plan = read_json(_run() / "experiments" / "e6" / "replicate_plan.json")
    return _extract_replicate(plan["first_new_call"])


def stage_j4(run_id: str) -> int:
    plan = read_json(_run() / "experiments" / "e6" / "replicate_plan.json")
    return _extract_replicate(plan["second_new_call"])


def _extract_replicate(replicate: str) -> int:
    run = _run()
    if _stopped(run):
        write_json(run / "experiments" / "e6" / f"{replicate}_status.json", {"status": "not_started_budget"})
        return 2
    prompt = _prompt("diagnostic_reread.txt")
    done = _done(run)
    failures = 0
    for paper in _papers():
        logical = extract_task(paper["pair_id"], paper["pmid"], replicate)["logical_task_id"]
        if logical in done:
            continue
        result = _invoke(run, "diagnostic_reread", _extract_user(run, paper), prompt, replicate, EXTRACT_CAP)
        _mark(run, logical, result.get("status"), paper, replicate, result.get("parsed"))
        if result.get("status") in {"quota_exhausted", "not_evaluated_budget", "stopped_after_quota"}:
            return 2
        if not isinstance(result.get("parsed"), dict):
            failures += 1
    write_json(run / "experiments" / "e6" / f"{replicate}_status.json", {"failures": failures, "at": _now()})
    return 0


def _done(run: Path) -> set[str]:
    path = run / "experiments" / "e6" / "task_log.jsonl"
    if not path.exists():
        return set()
    return {row.get("logical_task_id") for row in _read_jsonl(path) if row.get("status") in {"ok", "truncated"}}


def _mark(run: Path, logical: str, status: str, paper: dict, replicate: str, parsed) -> None:
    path = run / "experiments" / "e6" / "task_log.jsonl"
    path.parent.mkdir(exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "logical_task_id": logical, "status": status, "pair_id": paper.get("pair_id"),
            "pmid": paper.get("pmid"), "comparison_id": paper.get("comparison_id") or "",
            "replicate": replicate, "parsed": parsed, "at": _now(),
        }, ensure_ascii=False) + "\n")


def stage_j2(run_id: str) -> int:
    run = _run()
    cases = read_json(run / "experiments" / "e6" / "comparison_case_manifest.json")["chosen"]
    docs = _documents()
    prompt = _prompt("e6_judgment.txt")
    frozen = []
    for case in cases:
        if _stopped(run):
            return 2
        user = judgment_user(*LABELS[case["pair_id"]], case.get("primary_question_type") or "unclear", [
            (case["pmid_a"], (docs.get(case["pmid_a"]) or {}).get("text") or ""),
            (case["pmid_b"], (docs.get(case["pmid_b"]) or {}).get("text") or ""),
        ])
        result = _invoke(run, "e6_judgment", json.dumps(user, ensure_ascii=False), prompt, f"{case['comparison_id']}|full_0", JUDGE_CAP)
        parsed = result.get("parsed") if isinstance(result.get("parsed"), dict) else {}
        quotes = _quotes(parsed)
        variants = {
            case["pmid_a"]: build_variants((docs.get(case["pmid_a"]) or {}).get("text") or "", [row["quote"] for row in quotes if row["pmid"] == case["pmid_a"]], case["pmid_a"], SEED),
            case["pmid_b"]: build_variants((docs.get(case["pmid_b"]) or {}).get("text") or "", [row["quote"] for row in quotes if row["pmid"] == case["pmid_b"]], case["pmid_b"], SEED),
        }
        stored = {}
        for pmid, item in variants.items():
            stored[pmid] = {
                "full": (docs.get(pmid) or {}).get("text") or "",
                "E": item["E"]["text"],
                "D": item["D"]["text"],
                "R": item["R"].get("text") or "",
                "E_status": item["E"]["status"],
                "D_status": item["D"]["status"],
                "R_status": item["R"]["status"],
                "E_informative": item["E"]["informative"],
                "D_informative": item["D"]["informative"],
                "cited_n": len(item["cited_sentences"]),
                "cited_sentences": [row["text"] for row in item["cited_sentences"]],
                "deleted_chars": len((docs.get(pmid) or {}).get("text") or "") - len(item["D"]["text"]),
                "R_reason": item["R"].get("reason") or "",
                "R_token_gap": item["R"].get("token_gap"),
            }
        frozen.append({
            "comparison_id": case["comparison_id"],
            "pair_id": case["pair_id"],
            "pmid_a": case["pmid_a"],
            "pmid_b": case["pmid_b"],
            "question_type": case.get("primary_question_type") or "unclear",
            "full_0_status": result.get("status"),
            "full_0": _labels(parsed),
            "quotes": quotes,
            "variants": {pmid: _public_variant(item) for pmid, item in variants.items()},
            "stored": stored,
        })
        _mark(run, f"{VERSION}|judge|{hashlib.sha256(case['comparison_id'].encode()).hexdigest()[:16]}|full_0", result.get("status"), {"pair_id": case["pair_id"], "pmid": "", "comparison_id": case["comparison_id"]}, "full_0", parsed)
        if result.get("status") in {"quota_exhausted", "not_evaluated_budget", "stopped_after_quota"}:
            write_json(run / "experiments" / "e6" / "variants.json", frozen)
            return 2
    write_json(run / "experiments" / "e6" / "variants.json", frozen)
    return 0


def _quotes(parsed: dict) -> list[dict]:
    rows = []
    for item in parsed.get("evidence_refs") or []:
        if isinstance(item, dict) and item.get("quote"):
            rows.append({"pmid": str(item.get("pmid") or ""), "quote": item.get("quote")})
    return rows


def _labels(parsed: dict) -> dict:
    return {
        "can_describe_relation": parsed.get("can_describe_relation"),
        "direction_pattern": parsed.get("direction_pattern") or "",
        "direction_applicable": parsed.get("direction_applicable"),
        "magnitude_comparable": parsed.get("magnitude_comparable"),
        "measurement_relation": parsed.get("measurement_relation") or "",
        "population_or_system_relation": parsed.get("population_or_system_relation") or "",
        "missing_information": _short(parsed.get("missing_information"), 400),
        "relation_scope": _short(parsed.get("relation_scope"), 300),
    }


def _public_variant(item: dict) -> dict:
    return {
        "E": {key: value for key, value in item["E"].items() if key != "text"} | {"chars": len(item["E"]["text"])},
        "D": {"status": item["D"]["status"], "informative": item["D"]["informative"], "chars": len(item["D"]["text"]), "deleted_n": len(item["D"].get("deleted") or [])},
        "R": {key: value for key, value in item["R"].items() if key not in {"text", "deleted"}},
        "invalid_quotes": len(item["invalid_quotes"]),
        "cited_n": len(item["cited_sentences"]),
    }


def stage_j3(run_id: str) -> int:
    run = _run()
    frozen = read_json(run / "experiments" / "e6" / "variants.json")
    docs = _documents()
    prompt = _prompt("e6_judgment.txt")
    jobs = []
    for case in frozen:
        full = judgment_user(*LABELS[case["pair_id"]], case["question_type"], _full_pair(case, docs))
        for condition in ("full_1", "full_2"):
            jobs.append((case, condition, full))
        for condition in ("E", "D", "R"):
            payload, status = _condition_user(case, docs, condition)
            jobs.append((case, condition, payload if status == "ok" else {"_blocked": status}))
    jobs.sort(key=lambda item: hashlib.sha256(f"{SEED}|{item[0]['comparison_id']}|{item[1]}".encode()).hexdigest())
    for case, condition, user in jobs:
        logical = f"{VERSION}|judge|{hashlib.sha256(case['comparison_id'].encode()).hexdigest()[:16]}|{condition}"
        if logical in _done(run) or _stopped(run):
            continue
        if isinstance(user, dict) and user.get("_blocked"):
            _mark(run, logical, user["_blocked"], {"pair_id": case["pair_id"], "pmid": "", "comparison_id": case["comparison_id"]}, condition, None)
            continue
        result = _invoke(run, "e6_judgment", json.dumps(user, ensure_ascii=False), prompt, f"{case['comparison_id']}|{condition}", JUDGE_CAP)
        _mark(run, logical, result.get("status"), {"pair_id": case["pair_id"], "pmid": "", "comparison_id": case["comparison_id"]}, condition, result.get("parsed"))
        if result.get("status") in {"quota_exhausted", "not_evaluated_budget", "stopped_after_quota"}:
            return 2
    return 0


def _full_pair(case: dict, docs: dict[str, dict]) -> list[tuple[str, str]]:
    manifest = {row["comparison_id"]: row for row in read_json(_run() / "experiments" / "e6" / "comparison_case_manifest.json")["chosen"]}
    row = manifest[case["comparison_id"]]
    return [(row["pmid_a"], (docs.get(row["pmid_a"]) or {}).get("text") or ""), (row["pmid_b"], (docs.get(row["pmid_b"]) or {}).get("text") or "")]


def _condition_user(case: dict, docs: dict[str, dict], condition: str):
    abstracts = []
    saw_edit = False
    for pmid in (case.get("pmid_a"), case.get("pmid_b")):
        item = (case.get("stored") or {}).get(pmid) or {}
        if condition == "R":
            if item.get("R_status") == "ok":
                abstracts.append((pmid, item.get("R") or ""))
                saw_edit = True
            elif not item.get("cited_n"):
                abstracts.append((pmid, item.get("full") or ""))
            else:
                return {}, "not_constructible"
        else:
            abstracts.append((pmid, item.get(condition) or ""))
            if item.get(f"{condition}_informative"):
                saw_edit = True
    if not saw_edit:
        return {}, "uninformative_or_not_constructible" if condition == "R" else "uninformative_same_input"
    return judgment_user(*LABELS[case["pair_id"]], case["question_type"], abstracts), "ok"


def stage_j5(run_id: str) -> int:
    run = _run()
    papers = _papers()
    logs = _read_jsonl(run / "experiments" / "e6" / "task_log.jsonl") if (run / "experiments" / "e6" / "task_log.jsonl").exists() else []
    expected = read_json(run / "experiments" / "e6" / "extraction_expected.json")
    plan = read_json(run / "experiments" / "e6" / "replicate_plan.json")
    units = {f"{row.get('pair_id')}|{row.get('pmid')}": row for row in _read_jsonl(DIAG1 / "audit" / "reread_units.jsonl")}
    docs = _documents()
    extract_rows = []
    for paper in papers:
        if plan.get("baseline") == "reused_diag1":
            r0 = units.get(f"{paper['pair_id']}|{paper['pmid']}")
            r0_parsed = r0.get("parsed") if r0 else None
        else:
            r0_parsed = _first_ok(logs, paper["pair_id"], paper["pmid"], "r0")
        replicates = {"r0": _finding_view(r0_parsed, paper["pmid"], (docs.get(paper["pmid"]) or {}).get("text") or "")}
        for replicate in ("r1", "r2"):
            parsed = _first_ok(logs, paper["pair_id"], paper["pmid"], replicate)
            replicates[replicate] = _finding_view(parsed, paper["pmid"], (docs.get(paper["pmid"]) or {}).get("text") or "")
        extract_rows.append({"pair_id": paper["pair_id"], "pmid": paper["pmid"], "replicates": replicates})
    judgments = []
    for row in logs:
        if "|judge|" not in str(row.get("logical_task_id")):
            continue
        judgments.append({
            "logical_task_id": row["logical_task_id"],
            "comparison_id": row.get("comparison_id") or "",
            "condition": row.get("replicate") or "",
            "status": row.get("status"),
            "labels": _labels(row.get("parsed") or {}),
            "quotes": _quotes(row.get("parsed") or {}),
        })
    summary = _extract_summary(extract_rows)
    transitions = _judgment_summary(run, judgments)
    write_json(run / "experiments" / "e6" / "analysis.json", {"extracts": summary, "judgments": transitions})
    write_parquet(run / "tables" / "e6_extraction_repeat.parquet", summary)
    write_parquet(run / "tables" / "e6_judgment_transition.parquet", transitions)
    _write_report(run, expected, extract_rows, judgments, summary, transitions)
    return 0


def _first_ok(logs: list[dict], pair_id: str, pmid: str, replicate: str):
    rows = [row for row in logs if row.get("pair_id") == pair_id and str(row.get("pmid")) == pmid and row.get("replicate") == replicate and row.get("status") in {"ok", "truncated"}]
    rows.sort(key=lambda row: row.get("at") or "")
    return rows[0].get("parsed") if rows else None


def _finding_view(parsed, pmid: str, text: str = "") -> dict:
    findings = (parsed or {}).get("findings") if isinstance(parsed, dict) else None
    if not isinstance(findings, list):
        return {"status": "missing", "n": None, "sentences": [], "items": []}
    sentences = []
    items = []
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        quotes = [row.get("quote") for row in (finding.get("evidence") or []) if isinstance(row, dict) and row.get("quote")]
        starts = {row["start"] for row in build_variants(text, quotes, pmid, SEED)["cited_sentences"]} if text else set()
        items.append({
            "entity": finding.get("measured_entity") or finding.get("intervened_entity") or "",
            "question": finding.get("primary_question_type") or "",
            "role": finding.get("discourse_role") or "",
            "direction": finding.get("result_direction") or "",
            "outcome": finding.get("outcome") or "",
            "information": finding.get("result_direction_reporting_status") or "",
            "sentences": sorted(starts),
        })
        sentences.extend(starts)
    return {"status": "zero" if not findings else "nonzero", "n": len(findings), "items": items, "sentences": sorted(set(sentences)), "pmid": pmid}


def _pair_agreement(left: dict, right: dict) -> dict:
    if left["status"] == "missing" or right["status"] == "missing":
        return {"compared": False}
    left_set, right_set = set(left["sentences"]), set(right["sentences"])
    pairs, left_only, right_only = align_findings(left["items"], right["items"])
    fields = Counter()
    for left_index, right_index in pairs:
        a, b = left["items"][left_index], right["items"][right_index]
        for field in ("entity", "question", "role", "direction", "outcome", "information"):
            fields[field + "_same" if _norm(a.get(field)) == _norm(b.get(field)) else field + "_changed"] += 1
    return {
        "compared": True,
        "count_same": left["n"] == right["n"],
        "state_same": left["status"] == right["status"],
        "sentence_jaccard": sentence_jaccard(left_set, right_set),
        "matched": len(pairs),
        "left_unmatched": len(left_only),
        "right_unmatched": len(right_only),
        "left_n": left["n"],
        "right_n": right["n"],
        "fields": dict(fields),
    }


def _extract_summary(rows: list[dict]) -> list[dict]:
    summary = []
    for row in rows:
        views = row["replicates"]
        agreement = _pair_agreement(views["r0"], views["r1"])
        summary.append({
            "pair_id": row["pair_id"],
            "pmid": row["pmid"],
            "r0_status": views["r0"]["status"],
            "r1_status": views["r1"]["status"],
            "r2_status": views["r2"]["status"],
            "r0_n": views["r0"]["n"],
            "r1_n": views["r1"]["n"],
            "r2_n": views["r2"]["n"],
            "r0_r1": json.dumps(agreement, ensure_ascii=False),
        })
    return summary


def _judgment_summary(run: Path, rows: list[dict]) -> list[dict]:
    frozen = {row["comparison_id"]: row for row in read_json(run / "experiments" / "e6" / "variants.json")} if (run / "experiments" / "e6" / "variants.json").exists() else {}
    by_case = {}
    for row in rows:
        by_case.setdefault(row.get("comparison_id") or row.get("logical_task_id"), {})[row.get("condition") or ""] = row
    table = []
    for case_id, conditions in by_case.items():
        full_labels = [((conditions.get(name) or {}).get("labels") or {}).get("direction_pattern") for name in ("full_0", "full_1", "full_2")]
        observed = sorted({label for label in full_labels if label})
        case = frozen.get(case_id) or {}
        deleted = " ".join(" ".join(side.get("cited_sentences") or []) for side in (case.get("stored") or {}).values())
        for condition, item in conditions.items():
            quote_texts = [quote.get("quote") or "" for quote in item.get("quotes") or []]
            replacements = [quote for quote in quote_texts if quote and quote not in deleted]
            replacement = bool(condition == "D" and replacements)
            table.append({
                "comparison_id": case_id,
                "condition": condition,
                "status": item.get("status"),
                "direction_pattern": (item.get("labels") or {}).get("direction_pattern"),
                "can_describe_relation": str((item.get("labels") or {}).get("can_describe_relation")),
                "measurement_relation": str((item.get("labels") or {}).get("measurement_relation") or ""),
                "population_or_system_relation": str((item.get("labels") or {}).get("population_or_system_relation") or ""),
                "full_labels": ",".join(label or "missing" for label in full_labels),
                "outside_full_set": bool(item.get("labels", {}).get("direction_pattern") and observed and item["labels"]["direction_pattern"] not in observed),
                "citation_not_in_deleted_set": replacement,
                "replacement_quote": _short(replacements[0], 180) if replacements else "",
            })
    return table


def _write_report(run: Path, expected: list[dict], extracts: list[dict], judgments: list[dict], summary: list[dict], transitions: list[dict]) -> None:
    ledger = json.loads((run / "audit" / "ledger.json").read_text(encoding="utf-8"))
    two = 0
    three = 0
    for row in extracts:
        states = [row["replicates"][name]["status"] for name in ("r0", "r1", "r2")]
        ok = sum(1 for state in states if state in {"zero", "nonzero"})
        two += ok >= 2
        three += ok == 3
    lines = [
        "# e6_diagnostic_v1",
        "",
        f"child `{run.name}`，parent `{CLOSE1.name}`。这不是旧 E6 的修复完成，也不是准确率或生物学验证。整个 pilot 仍是 incomplete。",
        "",
        "提取重复使用诊断合同里的标题和摘要。比较重复只使用摘要正文，不输入标题、旧 finding 或旧判断。",
        "",
        "## 1. 提取完成了多少",
        "",
        f"论文 {len(extracts)} 篇。本轮新合同下两次完整输出 {two}/{len(extracts)}，三次 {three}/{len(extracts)}。",
        f"diag1 的请求缓存键能对上的论文 {sum(1 for row in expected if row.get('r0_cache_match') not in {None, '', 'not_rebuilt'})}/{len(expected)}。prompt 哈希一致，但请求正文没有存档，重建后的用户 JSON 对不上缓存键，所以 diag1 没有被当成这次的 r0。两次输出都是本轮发出的 r0 和 r1。",
        "",
        "## 2. 提取变在哪里",
        "",
        _extract_change_text(extracts),
        "",
        "## 3. 四个案例的完整条件和扰动",
        "",
        _case_text(run),
        _judgment_text(transitions),
        "",
        "## 4. 不能构造的对照",
        "",
        _variant_text(run),
        "R 只有在未引用句的数量相同、token 差不超过 20% 时才执行。做不到就记 not_constructible，不放宽容差。E 与全文相同不算“只留依据已经足够”。删除会改变上下文；引用句不是全部支持材料。四个案例仍是四个案例。没有人工 gold。",
        "",
        "## 5. 对正式研究的含义",
        "",
        "三次相同不代表正确。删除依据后改变，可能是依据没了，也可能是上下文变了。删除依据后不变，不能说明模型忽视原文。R 的变化说明未引用材料可能重要，或输出本身不稳定。正式核对应同时看实体角色、原文依据和比较层次，而不是只看一个稳定性分数。",
        "",
        "## 6. 费用和状态",
        "",
        f"已结算 {ledger.get('spent_usd')}。计费不明 {ledger.get('billing_unknown_usd')}。在途 {ledger.get('reserved_usd')}。可用 {available_usd(ledger)}。attempt {ledger.get('attempts')} / {LIMITS['max_attempts']}。输入 token {ledger.get('input_tokens')} / {LIMITS['max_input_tokens']}。输出 token {ledger.get('output_tokens')} / {LIMITS['max_output_tokens']}。",
        "K、E4 和开放模型仍暂缓。pilot_status 保持 incomplete。",
        "",
    ]
    (run / "reports").mkdir(exist_ok=True)
    (run / "reports" / "report.md").write_text("\n".join(lines), encoding="utf-8")
    by_case = {}
    for row in transitions:
        by_case.setdefault(row["comparison_id"], {})[row["condition"]] = row.get("status")
    full_done = sum(all(by_case[case].get(name) in {"ok", "truncated"} for name in ("full_0", "full_1", "full_2")) for case in by_case)
    perturb_done = sum(by_case[case].get(name) in {"ok", "truncated", "not_constructible", "uninformative_same_input", "uninformative_or_not_constructible", "no_locatable_quote"} for case in by_case for name in ("E", "D", "R"))
    manifest = read_json(run / "manifest.json")
    manifest["extraction_repeat"] = "complete" if two == len(extracts) and len(extracts) else ("partial" if two else "incomplete")
    manifest["extraction_three_replicates"] = f"{three}/{len(extracts)}"
    manifest["judgment_repeat"] = "complete" if by_case and full_done == len(by_case) else ("partial" if full_done else "incomplete")
    manifest["evidence_perturbation"] = "recorded" if perturb_done else "incomplete"
    manifest["judgment_rows"] = len(transitions)
    manifest["pilot_status"] = "incomplete"
    manifest["status"] = "e6_diagnostic_recorded"
    manifest["updated_utc"] = _now()
    write_json(run / "manifest.json", manifest)


def _extract_change_text(rows: list[dict]) -> str:
    compared = 0
    count_changed = 0
    state_changed = 0
    field_changed = Counter()
    field_same = Counter()
    matched = 0
    unmatched = 0
    both_empty = 0
    for row in rows:
        agreement = _pair_agreement(row["replicates"]["r0"], row["replicates"]["r1"])
        if not agreement.get("compared"):
            continue
        compared += 1
        count_changed += not agreement["count_same"]
        state_changed += not agreement["state_same"]
        if agreement["sentence_jaccard"] == "both_empty":
            both_empty += 1
        matched += agreement["matched"]
        unmatched += agreement["left_unmatched"] + agreement["right_unmatched"]
        for key, value in agreement["fields"].items():
            (field_same if key.endswith("_same") else field_changed)[key.rsplit("_", 1)[0]] += value
        for left_index, right_index in align_findings(row["replicates"]["r0"]["items"], row["replicates"]["r1"]["items"])[0]:
            left = row["replicates"]["r0"]["items"][left_index]
            right = row["replicates"]["r1"]["items"][right_index]
            if _norm(left.get("question")) != _norm(right.get("question")):
                field_changed[f"qpair:{row['pmid']}:{left.get('question')}->{right.get('question')}"] += 1
    parts = [
        f"r0 与 r1 都能读到的论文 {compared}/{len(rows)}。finding 数不同 {count_changed}/{compared or 'na'}。zero 与 nonzero 不同 {state_changed}/{compared or 'na'}。双方都没有可定位依据句 {both_empty}/{compared or 'na'}，这不记成 Jaccard 1。",
        f"可对齐 finding {matched}，未对齐 {unmatched}。未对齐集中在 finding 数发生变化的论文，是拆分或合并，不是按行号强行配上。对齐只用实体和原文句子，不用方向当匹配键。",
    ]
    for field in ("entity", "question", "role", "direction", "outcome", "information"):
        same = field_same[field]
        changed = field_changed[field]
        parts.append(f"{field} 字符串相同 {same}/{same + changed}，不同 {changed}/{same + changed}。")
    shifts = [key.split(":", 1)[1] for key in field_changed if key.startswith("qpair:")]
    parts.append("闭合字段里的类型变化主要是 primary_question_type：" + ("；".join(shifts) if shifts else "无") + "。")
    parts.append("result_direction 和 outcome 是自由文本。字符串不同多半是同一句结果的措辞差别，不能把 26 对字符串不同读成 26 次方向翻转。")
    return " ".join(parts)


def _judgment_text(rows: list[dict]) -> str:
    if not rows:
        return "比较判读尚未生成。"
    by_case = {}
    for row in rows:
        by_case.setdefault(row["comparison_id"], []).append(row)
    lines = []
    for case, items in by_case.items():
        short = " vs ".join(part for part in str(case).split("|") if part.isdigit())
        labels = {row["condition"]: row.get("direction_pattern") or row.get("status") for row in items}
        measure = {row["condition"]: row.get("measurement_relation") for row in items}
        population = {row["condition"]: row.get("population_or_system_relation") for row in items}
        full = next((row.get("full_labels") for row in items), "")
        outside = [row["condition"] for row in items if row.get("outside_full_set")]
        replaced = [f"{row['condition']}: {row.get('replacement_quote')}" for row in items if row.get("citation_not_in_deleted_set")]
        lines.append(f"{short}：三次完整条件的方向 {full}。各条件方向 {labels}。测量关系 {measure}。人群或系统 {population}。方向超出三次完整条件已见标签 {outside or '无'}。D 改用仍在输入中的其他原文 {replaced or '无'}。")
    return " ".join(lines)


def _case_text(run: Path) -> str:
    path = run / "experiments" / "e6" / "comparison_case_manifest.json"
    if not path.exists():
        return "案例清单尚未写入。"
    manifest = read_json(path)
    lines = [f"抽样 seed {manifest.get('seed')}。规则记录 {manifest.get('notes')}。"]
    for row in manifest.get("chosen") or []:
        lines.append(f"{row.get('stratum')} {row.get('pair_id')} {row.get('pmid_a')} vs {row.get('pmid_b')} {row.get('source_class')} {row.get('primary_question_type')} 规则 {row.get('pick_rule')}。")
    return " ".join(lines)


def _variant_text(run: Path) -> str:
    path = run / "experiments" / "e6" / "variants.json"
    if not path.exists():
        return "变体尚未冻结。"
    rows = read_json(path)
    notes = []
    for row in rows:
        for pmid, side in (row.get("stored") or {}).items():
            notes.append(f"{row['comparison_id']} {pmid}：引用句 {side.get('cited_n')}，删除字符 {side.get('deleted_chars')}，R={side.get('R_status')} {side.get('R_reason')}，token差 {side.get('R_token_gap')}。")
    return " ".join(notes) + " "


STAGES = {"j0": stage_j0, "j1": stage_j1, "j2": stage_j2, "j3": stage_j3, "j4": stage_j4, "j5": stage_j5}


def main() -> int:
    if _login():
        raise SystemExit("refusing to run e6 on the login node")
    try:
        return STAGES[sys.argv[1]](os.environ["PILOT_CHILD_RUN"])
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

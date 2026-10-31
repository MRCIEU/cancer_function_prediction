#!/usr/bin/env python3
"""Child-run continuation. The parent run's inputs and report are not rewritten."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, "/home/b5ac/freddieyu.b5ac/project_2")

from pairpilot.comparison import build_candidates, gap_from_comparison  # noqa: E402
from pairpilot.config import PACKAGE, digest, load_bundle, prompt_hashes  # noqa: E402
from pairpilot.contracts import exact_span, locate_quote  # noqa: E402
from pairpilot.experiments.e1 import equal_article, source_metrics  # noqa: E402
from pairpilot.experiments.e2 import direct_payload, extract_payload, knowledge_payload  # noqa: E402
from pairpilot.experiments.e5 import queries_for, select_new  # noqa: E402
from pairpilot.experiments.e6 import align_replicates, control_deletion, delete_span  # noqa: E402
from pairpilot.inference import CallCache, input_hash  # noqa: E402
from pairpilot.ledger import LIMITS, PRICE, BudgetBlocked, SharedLedger  # noqa: E402
from pairpilot.sampling import rank  # noqa: E402
from pairpilot.store import append_jsonl, connect, query_rows, read_json, write_json, write_parquet  # noqa: E402

PARENT = Path("/projects/b5ac/project_2/results/03_integrated_analysis/pilot_study_v1/runs/20261015T204633Z_0d9b9724")
RUNS = PARENT.parent
REL = Path("/projects/b5ac/project_2/results/03_integrated_analysis/id_alignment_v2_round1/tables/pair_relations.parquet")
SENTENCE = Path("/projects/b5ac/project_2/results/01_core_audit/tables/sentence.parquet")
MATCH = Path("/projects/b5ac/project_2/results/03_integrated_analysis/id_alignment_v2_round1/tables/accepted_match.parquet")
CANARY_PAIR = "84868__C0026764"
EXPECTED = {
    "n_pair": 8,
    "core_pair_pmid": 47,
    "semmed_pair_pmid": 36,
    "intersection": 13,
    "core_only": 34,
    "semmed_only": 23,
    "union_pair_pmid": 70,
    "unique_pmid": 70,
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _login() -> bool:
    return os.uname().nodename.startswith("login")


def _child() -> Path:
    run_id = os.environ["PILOT_CHILD_RUN"]
    path = RUNS / run_id
    if not path.exists():
        raise SystemExit(f"child run is missing: {path}")
    return path


def _status(run: Path, stage: str, state: str, detail: dict | None = None) -> None:
    path = run / "audit" / "stage_status.json"
    current = read_json(path) if path.exists() else {}
    current[stage] = {"state": state, "at": _now(), "detail": detail or {}}
    write_json(path, current)


def _con():
    tmp = Path(os.environ.get("SLURM_TMPDIR") or f"/tmp/{os.environ.get('USER', 'pilot')}/pairpilot")
    threads = max(1, int(os.environ.get("SLURM_CPUS_PER_TASK") or 2))
    return connect(threads=threads, memory="16GB", tmp=tmp)


def init_child(run_id: str) -> Path:
    run = RUNS / run_id
    if run.exists():
        return run
    run.mkdir(parents=False)
    for name in ("audit", "requests", "responses", "tables", "experiments", "reports", "logs", "jobs"):
        (run / name).mkdir()
    bundle = load_bundle()
    execution = {
        "parent_run_id": PARENT.name,
        "parent_inputs": str(PARENT / "inputs"),
        "rule_version": "id_alignment_v2",
        "science_config_hash": digest(bundle["pilot"]),
        "prompt_hashes": prompt_hashes(),
        "model_id": "gpt-6.1-sol",
        "service_tier": "standard",
        "temperature_sent": False,
        "tools_enabled": False,
        "open_model": False,
        "e4": "disabled_deferred",
        "parent_e4_status_unchanged": "skipped_unavailable_input",
        "scale": {
            "pairs": 8,
            "abstracts": 70,
            "planned_pairs": 10,
            "k_pairs": 4,
            "e3_pmids": 16,
            "e5_candidates": 3,
            "e5_new_abstracts": 30,
            "e6_pairs": 2,
            "e6_perturbations": 6,
        },
        "budget": LIMITS,
        "price": PRICE,
        "authorization_source": LIMITS["authorization_source"],
    }
    write_json(run / "execution_config.json", execution)
    write_json(run / "manifest.json", {
        "run_id": run_id,
        "parent_run_id": PARENT.name,
        "created_utc": _now(),
        "code_package": str(PACKAGE),
        "execution_hash": digest(execution),
        "status": "created",
        "inherited_stages": ["preflight", "prepare", "semmed_sentence_recovery"],
        "parent_not_rewritten": True,
    })
    _status(run, "child", "created", {})
    SharedLedger(run / "audit" / "ledger.json")
    return run


def stage_r0(run_id: str) -> int:
    run = init_child(run_id)
    con = _con()
    bands = query_rows(
        con,
        f"""
        SELECT
          CASE
            WHEN n_union = 1 THEN '1'
            WHEN n_union BETWEEN 2 AND 3 THEN '2-3'
            WHEN n_union BETWEEN 4 AND 8 THEN '4-8'
            WHEN n_union BETWEEN 9 AND 20 THEN '9-20'
            ELSE '>20'
          END AS band,
          count(*) AS n_identical
        FROM read_parquet('{REL.as_posix()}')
        WHERE relation_class = 'identical'
          AND n_core = n_semmed
          AND n_core = n_union
        GROUP BY 1
        ORDER BY 1
        """,
    )
    holdout = query_rows(
        con,
        f"""
        SELECT
          CASE
            WHEN n_union = 1 THEN '1'
            WHEN n_union BETWEEN 2 AND 3 THEN '2-3'
            WHEN n_union BETWEEN 4 AND 8 THEN '4-8'
            WHEN n_union BETWEEN 9 AND 20 THEN '9-20'
            ELSE '>20'
          END AS band,
          count(*) AS n_after_holdout
        FROM read_parquet('{REL.as_posix()}')
        WHERE relation_class = 'identical'
          AND n_core = n_semmed
          AND n_core = n_union
        GROUP BY 1
        """,
    )
    selected = query_rows(con, f"SELECT * FROM read_parquet('{(PARENT / 'inputs' / 'selected_pairs.parquet').as_posix()}')")
    links = query_rows(con, f"SELECT pair_id, pmid, source FROM read_parquet('{(PARENT / 'inputs' / 'pair_pmids.parquet').as_posix()}')")
    by_pair = {}
    for row in links:
        slot = by_pair.setdefault(row["pair_id"], {"CORE": set(), "SemMedDB": set()})
        slot[row["source"]].add(str(row["pmid"]))
    core = set()
    sem = set()
    core_rows = sem_rows = 0
    for slot in by_pair.values():
        core |= slot["CORE"]
        sem |= slot["SemMedDB"]
        core_rows += len(slot["CORE"])
        sem_rows += len(slot["SemMedDB"])
    both = set()
    for slot in by_pair.values():
        both |= slot["CORE"] & slot["SemMedDB"]
    counts = {
        "n_pair": len(selected),
        "core_pair_pmid": core_rows,
        "semmed_pair_pmid": sem_rows,
        "intersection": sum(len(slot["CORE"] & slot["SemMedDB"]) for slot in by_pair.values()),
        "core_only": sum(len(slot["CORE"] - slot["SemMedDB"]) for slot in by_pair.values()),
        "semmed_only": sum(len(slot["SemMedDB"] - slot["CORE"]) for slot in by_pair.values()),
        "union_pair_pmid": sum(len(slot["CORE"] | slot["SemMedDB"]) for slot in by_pair.values()),
        "unique_pmid": len(core | sem),
    }
    names = _names(con, selected)
    write_json(run / "audit" / "r0_identical.json", {
        "definition": "For identical, n_core = n_semmed = n_union. A union of 4-20 already implies at least two PMIDs on each side.",
        "eval_pair_list": "none_found",
        "exclusion_sequence": [
            {"step": "relation_class_identical", "removed_by_later_filter": 0},
            {"step": "holdout_or_eval_id", "removed": 0, "reason": "no frozen evaluation list"},
        ],
        "band_counts": bands,
        "after_holdout": holdout,
        "bands_4_8_and_9_20_empty": not any(row["band"] in {"4-8", "9-20"} and row["n_identical"] for row in bands),
        "selected_pairs_kept": True,
        "sample_changed": False,
    })
    write_json(run / "audit" / "r0_sample.json", {"counts": counts, "expected": EXPECTED, "match": counts == EXPECTED})
    write_parquet(run / "tables" / "pair_names.parquet", names)
    _status(run, "r0", "completed" if counts == EXPECTED else "sample_count_mismatch", counts)
    if counts != EXPECTED:
        return 2
    if any(row["relation_class"] == "identical" for row in selected):
        _status(run, "r0", "selected_pair_class_unexpected", {})
        return 2
    return 0


def _names(con, selected: list[dict]) -> list[dict]:
    ids = ", ".join("'" + row["pair_id"].replace("'", "") + "'" for row in selected)
    core = query_rows(
        con,
        f"""
        SELECT p.pair_id,
               string_agg(DISTINCT s.gene_mention, ' | ') AS gene_mentions,
               string_agg(DISTINCT s.disease_mention, ' | ') AS disease_mentions
        FROM read_parquet('{(PARENT / 'inputs' / 'selected_pairs.parquet').as_posix()}') p
        JOIN read_parquet('{SENTENCE.as_posix()}') s
          ON CAST(s.gene_id AS VARCHAR) = CAST(p.gene_id AS VARCHAR)
         AND CAST(s.disease_cui AS VARCHAR) = CAST(p.disease_cui AS VARCHAR)
        WHERE p.pair_id IN ({ids})
        GROUP BY 1
        """,
    )
    sem = query_rows(
        con,
        f"""
        SELECT p.pair_id,
               string_agg(DISTINCT m.argument_name, ' | ') AS argument_names,
               string_agg(DISTINCT m.mapping_route, ' | ') AS mapping_routes,
               count(*) AS n_match_row
        FROM read_parquet('{(PARENT / 'inputs' / 'selected_pairs.parquet').as_posix()}') p
        JOIN read_parquet('{MATCH.as_posix()}') m
          ON CAST(m.accepted_gene_id AS VARCHAR) = CAST(p.gene_id AS VARCHAR)
         AND CAST(m.disease_cui AS VARCHAR) = CAST(p.disease_cui AS VARCHAR)
        GROUP BY 1
        """,
    )
    core_map = {row["pair_id"]: row for row in core}
    sem_map = {row["pair_id"]: row for row in sem}
    kg = query_rows(con, f"SELECT source, gene_id, disease_cui, text_status, sentence_id FROM read_parquet('{(PARENT / 'inputs' / 'kg_records.parquet').as_posix()}')")
    semmed_text = query_rows(con, f"SELECT sentence_id, sentence_text FROM read_parquet('{(PARENT / 'inputs' / 'semmed_sentences.parquet').as_posix()}')")
    text_ids = {str(row["sentence_id"]) for row in semmed_text if row.get("sentence_text")}
    out = []
    for row in selected:
        c = core_map.get(row["pair_id"], {})
        s = sem_map.get(row["pair_id"], {})
        gene_name = c.get("gene_mentions") or s.get("argument_names") or ""
        disease_name = c.get("disease_mentions") or ""
        source = "core_sentence.gene_mention" if c.get("gene_mentions") else ("accepted_match.argument_name" if s.get("argument_names") else "not_in_available_fields")
        disease_source = "core_sentence.disease_mention" if c.get("disease_mentions") else "not_in_available_fields"
        linked = [item for item in kg if str(item.get("gene_id")) == str(row["gene_id"]) and str(item.get("disease_cui")) == str(row["disease_cui"])]
        out.append({
            "pair_id": row["pair_id"],
            "stratum": row["stratum"],
            "gene_id": row["gene_id"],
            "disease_cui": row["disease_cui"],
            "gene_name": gene_name,
            "gene_name_source": source,
            "disease_name": disease_name,
            "disease_name_source": disease_source,
            "n_core_records": sum(1 for item in linked if item["source"] == "CORE"),
            "n_semmed_records": sum(1 for item in linked if item["source"] == "SemMedDB"),
            "n_core_text": sum(1 for item in linked if item["source"] == "CORE" and item.get("text_status") == "source_sentence"),
            "n_semmed_text": sum(1 for item in linked if item["source"] == "SemMedDB" and item.get("text_status") == "source_sentence"),
            "mapping_routes": s.get("mapping_routes") or "",
            "n_semmed_text_recovered": sum(
                1 for item in linked
                if item.get("source") == "SemMedDB" and str(item.get("sentence_id") or "") in text_ids
            ),
        })
    return out


def _docs() -> list[dict]:
    con = _con()
    return query_rows(con, f"SELECT * FROM read_parquet('{(PARENT / 'inputs' / 'documents.parquet').as_posix()}')")


def _links() -> list[dict]:
    con = _con()
    return query_rows(con, f"SELECT * FROM read_parquet('{(PARENT / 'inputs' / 'pair_pmids.parquet').as_posix()}')")


def _prompt(name: str) -> str:
    return (PACKAGE / "prompts" / name).read_text(encoding="utf-8")


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4 + 300)


def _call(run: Path, scope: str, task: str, user: str, prompt: str, replicate_id=None, max_output: int | None = None) -> dict:
    ledger = SharedLedger(run / "audit" / "ledger.json")
    cache = CallCache(run / "responses" / "cache")
    model_id = "gpt-6.1-sol"
    cap = int(max_output or LIMITS["max_output_tokens_per_call"])
    request = {
        "task": task,
        "input_sha256": input_hash(user),
        "schema_id": task,
        "prompt_sha256": input_hash(prompt),
        "model_id": model_id,
        "params": {"max_completion_tokens": cap, "service_tier": "standard"},
        "replicate_id": replicate_id,
    }
    key = cache.key(request)
    if replicate_id is None:
        cached = cache.get(key)
        if cached:
            cached["cache_hit"] = True
            return cached
    if len(user) > 48000:
        return {"status": "not_evaluated_context", "parsed": None, "raw": "", "usage": {}}
    reservation_id = hashlib.sha256(f"{key}|{time.time_ns()}".encode()).hexdigest()[:16]
    try:
        held = ledger.reserve(reservation_id, _estimate_tokens(prompt + user), cap, scope)
    except BudgetBlocked as exc:
        return {"status": "not_evaluated_budget", "parsed": None, "raw": str(exc), "usage": {}}
    result = _post(model_id, prompt, user, cap)
    if result["status"] in {"timeout_or_transport", "http_5xx"}:
        ledger.keep_unknown(reservation_id)
        result["billing"] = "billing_unknown"
    elif result["status"] in {"http_400", "quota_exhausted"}:
        ledger.release(reservation_id)
    else:
        usage = result.get("usage") or {}
        ledger.settle(
            reservation_id,
            int(usage.get("input_tokens") or 0),
            int(usage.get("output_tokens") or 0),
            int(usage.get("cached_input_tokens") or 0),
        )
    result["cache_hit"] = False
    result["reservation_usd"] = held["reserved_usd"]
    append_jsonl(run / "requests" / "requests.jsonl", {
        "task": task, "scope": scope, "cache_key": key, "replicate_id": replicate_id,
        "requested_model": model_id, "returned_model": result.get("returned_model") or "unknown",
        "prompt_sha256": request["prompt_sha256"], "schema_id": task,
        "max_completion_tokens": cap,
        "status": result["status"], "model_response_id": result.get("model_response_id"),
        "usage": result.get("usage") or {}, "billing": result.get("billing"),
    })
    append_jsonl(run / "responses" / "responses.jsonl", {
        "task": task, "scope": scope, "status": result["status"], "raw": result.get("raw") or "",
        "parsed": result.get("parsed"), "model_response_id": result.get("model_response_id"),
    })
    if result["status"] == "ok" and replicate_id is None:
        cache.put(key, result)
    return result


def _post(model_id: str, prompt: str, user: str, max_output: int | None = None) -> dict:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return {"status": "openai_key_missing", "parsed": None, "raw": "", "usage": {}}
    body = {
        "model": model_id,
        "max_completion_tokens": int(max_output or LIMITS["max_output_tokens_per_call"]),
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user},
        ],
    }
    http = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(http, timeout=180) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:800]
        if "insufficient_quota" in detail or "credit_balance_exhausted" in detail:
            status = "quota_exhausted"
        elif exc.code == 400:
            status = "http_400"
        elif exc.code >= 500:
            status = "http_5xx"
        else:
            status = "http_error"
        return {"status": status, "parsed": None, "raw": detail, "usage": {}, "model_response_id": None}
    except Exception as exc:
        return {"status": "timeout_or_transport", "parsed": None, "raw": str(exc)[:300], "usage": {}, "model_response_id": None}
    choice = (payload.get("choices") or [{}])[0]
    raw = ((choice.get("message") or {}).get("content")) or ""
    usage = payload.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    parsed = _parse_json(raw)
    status = "truncated" if choice.get("finish_reason") == "length" else ("ok" if parsed is not None else "unparsed")
    return {
        "status": status,
        "parsed": parsed,
        "raw": raw,
        "usage": {
            "input_tokens": usage.get("prompt_tokens") or 0,
            "output_tokens": usage.get("completion_tokens") or 0,
            "cached_input_tokens": details.get("cached_tokens") or 0,
            "reasoning_tokens": ((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0),
        },
        "model_response_id": payload.get("id"),
        "returned_model": payload.get("model") or "unknown",
    }


def _parse_json(raw: str):
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[-1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return None
    return None


def _pair_for_pmid(links: list[dict]) -> dict[str, str]:
    out = {}
    for row in links:
        out.setdefault(str(row["pmid"]), row["pair_id"])
    return out


def _evidence_text(doc: dict) -> str:
    return ((doc.get("title") or "") + "\n\n" + (doc.get("text") or "")).strip()


def _findings_from(result: dict, doc: dict, pair: str, run_id: str) -> list[dict]:
    text = _evidence_text(doc)
    rows = []
    findings = ((result.get("parsed") or {}).get("findings") or []) if result.get("status") == "ok" else []
    if result.get("status") == "ok" and not findings:
        rows.append({
            "pair_id": pair, "pmid": str(doc["pmid"]), "local": 0, "finding_id": f"{run_id}:{pair}:{doc['pmid']}:none",
            "status": "no_finding", "discourse_role": "", "question_type": "", "located": False,
        })
    for local, finding in enumerate(findings):
        quote = ""
        evidence = finding.get("evidence") or []
        if evidence and isinstance(evidence[0], dict):
            quote = evidence[0].get("quote") or ""
        located = locate_quote(text, quote) if quote else None
        ok = bool(located and exact_span(text, located[0], located[1], quote))
        rows.append({
            "pair_id": pair,
            "pmid": str(doc["pmid"]),
            "local": local,
            "finding_id": f"{run_id}:{pair}:{doc['pmid']}:{local}",
            "question_type": finding.get("question_type") or "unknown",
            "study_system": finding.get("study_system") or "unknown",
            "exposure_or_intervention": finding.get("exposure_or_intervention") or "",
            "comparator": finding.get("comparator") or "",
            "outcome": finding.get("outcome") or "",
            "measurement_context": finding.get("measurement_context") or "",
            "perturbation_direction": finding.get("perturbation_direction") or "",
            "result_direction": finding.get("result_direction") or "",
            "significance": finding.get("significance") or "not_reported",
            "discourse_role": finding.get("discourse_role") or "unclear",
            "quote": quote,
            "evidence_start": located[0] if ok else None,
            "evidence_end": located[1] if ok else None,
            "located": ok,
            "status": "citation_located" if ok else "citation_not_located",
            "text_version_id": doc.get("text_version_id") or "",
        })
    if result.get("status") != "ok":
        rows.append({
            "pair_id": pair, "pmid": str(doc["pmid"]), "local": 0,
            "finding_id": f"{run_id}:{pair}:{doc['pmid']}:failed",
            "status": result.get("status"), "discourse_role": "", "question_type": "", "located": False,
        })
    return rows


def stage_canary(run_id: str) -> int:
    run = _child()
    if (run / "audit" / "canary_passed.json").exists():
        _status(run, "canary", "completed_cached", {})
        return 0
    links = _links()
    core = {str(row["pmid"]) for row in links if row["pair_id"] == CANARY_PAIR and row["source"] == "CORE"}
    sem = {str(row["pmid"]) for row in links if row["pair_id"] == CANARY_PAIR and row["source"] == "SemMedDB"}
    core_only = sorted(core - sem, key=lambda pmid: rank(20261015, "canary-core", pmid))
    sem_only = sorted(sem - core, key=lambda pmid: rank(20261015, "canary-sem", pmid))
    if not core_only or not sem_only:
        _status(run, "canary", "blocked_no_split_pmid", {})
        return 2
    chosen = [core_only[0], sem_only[0]]
    write_json(run / "audit" / "canary_selection.json", {"pair_id": CANARY_PAIR, "pmids": chosen, "seed": 20261015})
    docs = {str(row["pmid"]): row for row in _docs()}
    prompt = _prompt("extract_findings.txt")
    rows = []
    for pmid in chosen:
        doc = docs[pmid]
        user = json.dumps(extract_payload(
            {"pair_id": CANARY_PAIR, "gene_id": "84868", "disease_cui": "C0026764", "aliases": []},
            {"pmid": pmid, "title": doc.get("title"), "text": doc.get("text"), "text_version_id": doc.get("text_version_id")},
        ), ensure_ascii=False)
        result = _call(run, "canary", "extract_findings", user + "\n\nTEXT:\n" + _evidence_text(doc), prompt)
        if result["status"] == "unparsed":
            result = _call(run, "canary", "extract_findings_repair", user + "\n\nReturn one JSON object only.\n" + _evidence_text(doc), prompt)
        rows.extend(_findings_from(result, doc, CANARY_PAIR, run_id))
    write_parquet(run / "tables" / "canary_findings.parquet", rows)
    rules = load_bundle()["pilot"]["required_comparability_fields"]
    usable = [row for row in rows if row.get("question_type")]
    comparisons = build_candidates(usable, "canary", rules) if len({row["pmid"] for row in usable}) > 1 else []
    write_parquet(run / "tables" / "canary_comparisons.parquet", comparisons or [{"comparison_id": "none", "status": "no_comparison_formed", "pair_id": CANARY_PAIR}])
    synth = _call(run, "canary", "synthesize_pair", json.dumps({"findings": usable, "comparisons": comparisons}, ensure_ascii=False)[:20000], _prompt("synthesize_pair.txt"))
    passed = all(row.get("status") not in {"http_400", "http_error", "openai_key_missing", "timeout_or_transport", "unparsed"} for row in rows)
    passed = passed and synth.get("status") in {"ok", "no_finding", "not_evaluated_budget"}
    write_json(run / "audit" / "canary_passed.json", {"passed": passed, "n_finding_row": len(rows), "synthesis_status": synth.get("status")})
    _status(run, "canary", "completed" if passed else "failed", {"passed": passed})
    return 0 if passed else 2


def stage_extract(run_id: str) -> int:
    run = _child()
    if not (run / "audit" / "canary_passed.json").exists() or not read_json(run / "audit" / "canary_passed.json").get("passed"):
        _status(run, "extract", "blocked_canary", {})
        return 2
    done = {row["pmid"] for row in _read_optional(run / "tables" / "findings.parquet")}
    links = _links()
    pair_of = _pair_for_pmid(links)
    docs = _docs()
    prompt = _prompt("extract_findings.txt")
    rows = _read_optional(run / "tables" / "findings.parquet")
    for doc in docs:
        pmid = str(doc["pmid"])
        if pmid in done:
            continue
        pair = pair_of.get(pmid)
        if not pair:
            continue
        gene, cui = pair.split("__", 1)
        user = json.dumps(extract_payload(
            {"pair_id": pair, "gene_id": gene, "disease_cui": cui, "aliases": []},
            doc,
        ), ensure_ascii=False)
        result = _call(run, "main", "extract_findings", user + "\n\nTEXT:\n" + _evidence_text(doc), prompt)
        rows.extend(_findings_from(result, doc, pair, run_id))
        write_parquet(run / "tables" / "findings.parquet", rows)
        if result.get("status") == "not_evaluated_budget":
            _status(run, "extract", "stopped_budget", {"n": len(rows)})
            return 0
    _status(run, "extract", "completed", {"n": len(rows)})
    return 0


def stage_compare(run_id: str) -> int:
    run = _child()
    findings = [row for row in _read_optional(run / "tables" / "findings.parquet") if row.get("question_type")]
    rules = load_bundle()["pilot"]["required_comparability_fields"]
    comparisons = build_candidates(findings, read_json(run / "manifest.json")["execution_hash"], rules)
    write_parquet(run / "tables" / "comparisons.parquet", comparisons)
    gaps = [gap for gap in (gap_from_comparison(row, "child") for row in comparisons) if gap]
    write_parquet(run / "tables" / "candidate_gaps.parquet", gaps)
    _status(run, "compare", "completed", {"n_comparison": len(comparisons), "n_gap": len(gaps)})
    return 0


def stage_e1(run_id: str) -> int:
    run = _child()
    findings = [row for row in _read_optional(run / "tables" / "findings.parquet") if row.get("discourse_role") != "background"]
    comparisons = _read_optional(run / "tables" / "comparisons.parquet")
    links = _links()
    by_pair = {}
    for row in links:
        slot = by_pair.setdefault(row["pair_id"], {"CORE": set(), "SemMedDB": set()})
        slot[row["source"]].add(str(row["pmid"]))
    metrics = []
    draws = {}
    for pair, slot in by_pair.items():
        metrics.extend(source_metrics(run_id, pair, findings, comparisons, slot["CORE"], slot["SemMedDB"]))
        draws[pair] = equal_article(slot["CORE"], slot["SemMedDB"], findings, comparisons, 20261015)
    write_parquet(run / "tables" / "experiment_metrics.parquet", metrics)
    write_json(run / "experiments" / "e1_draws.json", draws)
    _status(run, "E1", "completed", {"n_metric": len(metrics)})
    return 0


def stage_e2(run_id: str) -> int:
    run = _child()
    links = _links()
    docs = {str(row["pmid"]): row for row in _docs()}
    by_pair = {}
    for row in links:
        by_pair.setdefault(row["pair_id"], set()).add(str(row["pmid"]))
    prompt = _prompt("direct_pair_analysis.txt")
    rows = []
    for pair, pmids in sorted(by_pair.items()):
        packet = [docs[pmid] for pmid in sorted(pmids) if pmid in docs]
        gene, cui = pair.split("__", 1)
        user = json.dumps(direct_payload({"pair_id": pair, "gene_id": gene, "disease_cui": cui, "aliases": []}, packet), ensure_ascii=False)
        result = _call(run, "main", "direct_pair", user, prompt)
        rows.append({"pair_id": pair, "config": "D", "status": result.get("status"), "n_call": 0 if result.get("cache_hit") else 1})
        if result.get("status") == "not_evaluated_budget":
            break
    small = [row["pair_id"] for row in query_rows(_con(), f"SELECT pair_id FROM read_parquet('{(PARENT / 'inputs' / 'selected_pairs.parquet').as_posix()}') WHERE band = 'small'")]
    kg = query_rows(_con(), f"SELECT * FROM read_parquet('{(PARENT / 'inputs' / 'kg_records.parquet').as_posix()}')")
    for pair in small:
        if any(row.get("status") == "not_evaluated_budget" for row in rows):
            break
        gene, cui = pair.split("__", 1)
        records = [row for row in kg if f"{row.get('gene_id')}__{row.get('disease_cui')}" == pair][:8]
        for pmid in sorted(by_pair.get(pair, [])):
            doc = docs.get(pmid)
            if not doc:
                continue
            user = json.dumps(knowledge_payload(
                {"pair_id": pair, "gene_id": gene, "disease_cui": cui, "aliases": []}, doc, records,
            ), ensure_ascii=False)
            result = _call(run, "main", "knowledge_pair", user, _prompt("extract_findings.txt"))
            rows.append({"pair_id": pair, "pmid": pmid, "config": "K", "status": result.get("status")})
            if result.get("status") == "not_evaluated_budget":
                break
    write_parquet(run / "tables" / "e2_calls.parquet", rows)
    _status(run, "E2", "completed" if rows and rows[-1].get("status") != "not_evaluated_budget" else "stopped_budget", {"n": len(rows)})
    return 0


def stage_e3(run_id: str) -> int:
    run = _child()
    frozen = read_json(PARENT / "inputs" / "frozen_subsets.json")
    chosen = frozen["E3"][:16]
    docs = {str(row["pmid"]): row for row in _docs()}
    kg = query_rows(_con(), f"SELECT * FROM read_parquet('{(PARENT / 'inputs' / 'kg_records.parquet').as_posix()}')")
    sem_text = {str(row["sentence_id"]): row.get("sentence_text") or "" for row in query_rows(_con(), f"SELECT * FROM read_parquet('{(PARENT / 'inputs' / 'semmed_sentences.parquet').as_posix()}')")}
    prompt = _prompt("assess_scope.txt")
    rows = []
    for item in chosen:
        pair, pmid = item["pair_id"], str(item["pmid"])
        doc = docs.get(pmid)
        if not doc:
            rows.append({"pair_id": pair, "pmid": pmid, "status": "text_unavailable"})
            continue
        sentences = []
        for record in kg:
            if f"{record.get('gene_id')}__{record.get('disease_cui')}" != pair or str(record.get("pmid")) != pmid:
                continue
            text = record.get("text") or sem_text.get(str(record.get("sentence_id") or ""))
            if text:
                sentences.append({"source": record.get("source"), "text": text})
        if not sentences:
            rows.append({"pair_id": pair, "pmid": pmid, "arm": "S", "status": "source_sentence_missing"})
        else:
            result = _call(run, "main", "assess_scope_s", json.dumps(sentences, ensure_ascii=False), prompt)
            rows.append({"pair_id": pair, "pmid": pmid, "arm": "S", "status": result.get("status")})
        result_a = _call(run, "main", "assess_scope_a", _evidence_text(doc), prompt)
        rows.append({"pair_id": pair, "pmid": pmid, "arm": "A", "status": result_a.get("status")})
        if result_a.get("status") == "not_evaluated_budget":
            break
    write_parquet(run / "tables" / "scope_assessments.parquet", rows)
    _status(run, "E3", "completed", {"n": len(rows)})
    return 0


def stage_e5(run_id: str) -> int:
    run = _child()
    gaps = _read_optional(run / "tables" / "candidate_gaps.parquet")
    ranked = sorted(gaps, key=lambda row: rank(20261015, "e5", row.get("candidate_id") or ""))[:3]
    write_json(run / "audit" / "e5_frozen_candidates.json", ranked)
    known = {str(row["pmid"]) for row in _links()}
    found = []
    for candidate in ranked:
        for query in queries_for(candidate, []):
            hits = _esearch(query["text"])
            chosen = select_new(hits[:50], known, 5)
            for hit in chosen:
                hit["query_id"] = query["query_id"]
                hit["candidate_id"] = candidate.get("candidate_id")
                if not hit.get("already_in_packet"):
                    known.add(str(hit["pmid"]))
            found.extend(chosen)
            if sum(1 for row in found if not row.get("already_in_packet")) >= 30:
                break
    write_parquet(run / "tables" / "search_results.parquet", found)
    _status(run, "E5", "completed", {"n_hit": len(found), "n_candidate": len(ranked)})
    return 0


def _esearch(term: str) -> list[dict]:
    query = urllib.parse.urlencode({"db": "pubmed", "term": term, "retmax": 50, "retmode": "json", "sort": "relevance"})
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + query
    if os.environ.get("NCBI_API_KEY"):
        url += "&api_key=" + os.environ["NCBI_API_KEY"]
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "pairpilot/1.0"}), timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []
    return [{"pmid": pmid, "retrieved_at": _now()} for pmid in (payload.get("esearchresult") or {}).get("idlist") or []]


def stage_e6(run_id: str) -> int:
    run = _child()
    frozen = read_json(PARENT / "inputs" / "frozen_subsets.json")
    pairs = set(frozen["E6"])
    base = [row for row in _read_optional(run / "tables" / "findings.parquet") if row.get("pair_id") in pairs and row.get("question_type")]
    docs = {str(row["pmid"]): row for row in _docs()}
    prompt = _prompt("extract_findings.txt")
    repeats = []
    for replicate in (1, 2):
        for row in base:
            doc = docs.get(str(row["pmid"]))
            if not doc:
                continue
            result = _call(run, "main", "extract_findings", _evidence_text(doc), prompt, replicate_id=replicate)
            repeats.append({"replicate_id": replicate, "pmid": row["pmid"], "status": result.get("status"), "cache_hit": result.get("cache_hit")})
            if result.get("status") == "not_evaluated_budget":
                break
    located = [row for row in base if row.get("located")][:6]
    perturbations = []
    for row in located:
        doc = docs.get(str(row["pmid"]))
        if not doc:
            continue
        text = _evidence_text(doc)
        start, end = row.get("evidence_start"), row.get("evidence_end")
        if start is None or end is None:
            perturbations.append({"finding_id": row["finding_id"], "status": "not_applicable"})
            continue
        control = control_deletion(text, int(start), int(end))
        variants = {
            "full": text,
            "quote_only": text[int(start):int(end)],
            "deleted_quote": delete_span(text, int(start), int(end)),
        }
        if control:
            variants["deleted_control"] = control[0]
        else:
            perturbations.append({"finding_id": row["finding_id"], "arm": "deleted_control", "status": "not_applicable"})
        for arm, body in variants.items():
            result = _call(run, "main", "perturb_" + arm, body, prompt, replicate_id=f"{row['finding_id']}:{arm}")
            perturbations.append({"finding_id": row["finding_id"], "arm": arm, "status": result.get("status")})
    write_json(run / "experiments" / "e6_alignment.json", {
        "replicates": repeats,
        "perturbations": perturbations,
        "alignment_note": align_replicates(base, base),
    })
    _status(run, "E6", "completed", {"n_replicate": len(repeats), "n_perturbation": len(perturbations)})
    _status(run, "E4", "disabled_deferred", {"parent_status": "skipped_unavailable_input"})
    _status(run, "open_model", "disabled", {})
    return 0


def stage_report(run_id: str) -> int:
    run = _child()
    status = read_json(run / "audit" / "stage_status.json") if (run / "audit" / "stage_status.json").exists() else {}
    ledger = SharedLedger(run / "audit" / "ledger.json").snapshot() if (run / "audit" / "ledger.json").exists() else {}
    identical = read_json(run / "audit" / "r0_identical.json") if (run / "audit" / "r0_identical.json").exists() else {}
    names = _read_optional(run / "tables" / "pair_names.parquet")
    required = ("canary", "extract", "compare", "E1", "E2", "E3", "E5", "E6")
    complete = all(status.get(name, {}).get("state") == "completed" for name in required)
    lines = [
        "# CORE–SemMedDB pilot 续跑",
        "",
        f"child run `{run.name}`，parent `{PARENT.name}`。准备阶段沿用 parent，本报告只统计本轮写出的表。",
        "",
        "这是待验证的模型输出，不是已证实的生物学结论。引用定位率不是科学支持率。",
        "",
        "## 样本",
        "",
        "原计划 10 个 pair。identical 的 PMID 并集在 4–8 和 9–20 两档没有候选；对 identical 来说，并集达到 4 已经包含两侧至少 2 篇。没有另加评估集排除。实际分析分母是 8 个 pair、70 篇摘要。",
        "",
    ]
    if identical.get("band_counts"):
        lines.append("identical 分档：" + "，".join(f"{row['band']}={row['n_identical']}" for row in identical["band_counts"]))
        lines.append("")
    lines.append("| pair | 名称 | 分层 |")
    lines.append("| --- | --- | --- |")
    for row in names:
        label = " / ".join(part for part in (row.get("gene_name"), row.get("disease_name")) if part) or "名称字段为空"
        lines.append(f"| {row['pair_id']} | {label} | {row['stratum']} |")
    lines.extend(["", "## 阶段与费用", ""])
    for name in ("r0", "canary", "extract", "compare", "E1", "E2", "E3", "E4", "E5", "E6"):
        lines.append(f"- {name}：{status.get(name, {}).get('state', 'not_started')}")
    if ledger:
        lines.append("")
        lines.append(
            f"已结算约 {ledger.get('spent_usd', 0):.4f} 美元，预留 {ledger.get('reserved_usd', 0):.4f}，"
            f"计费不明 {ledger.get('billing_unknown_usd', 0):.4f}，剩余额度约 {ledger.get('remaining_usd', 0):.4f}。"
            f"尝试 {ledger.get('attempts', 0)} 次。总额上限 {LIMITS['max_usd']} 美元，canary 上限 {LIMITS['canary_max_usd']} 美元。"
        )
    lines.extend([
        "",
        "E4 在 parent 中是输入不足；本轮记为 disabled_deferred，没有新建索引。开放模型对照关闭。",
        "",
        "E1 的条件增量只描述当前结构化输出，不表示新增有效证据。等文章量重采样是这一样本的敏感性分布。",
        "",
        "仍需正式人工验证：finding 是否被原文支持、比较和缺口是否成立、来源句与摘要的范围差异是否成立。",
        "",
        f"报告状态：{'complete' if complete else 'incomplete'}。",
        "",
    ])
    (run / "reports" / "report.md").write_text("\n".join(lines), encoding="utf-8")
    folder = run / "reports" / "pairs"
    folder.mkdir(exist_ok=True)
    for row in names:
        (folder / f"{row['pair_id']}.md").write_text(
            "\n".join([
                f"# {row.get('gene_name') or row['gene_id']} / {row.get('disease_name') or row['disease_cui']}",
                "",
                f"ID：{row['pair_id']}。分层：{row['stratum']}。",
                f"基因名来源：{row.get('gene_name_source')}。疾病名来源：{row.get('disease_name_source')}。",
                f"CORE 记录 {row.get('n_core_records')}，SemMedDB 记录 {row.get('n_semmed_records')}。",
                f"映射路径：{row.get('mapping_routes') or '无'}。",
                "",
                "零结果和失败与有 finding 的 pair 使用同一模板。科学判断仍未人工验证。",
                "",
            ]),
            encoding="utf-8",
        )
    manifest = read_json(run / "manifest.json")
    manifest["status"] = "complete" if complete else "incomplete"
    write_json(run / "manifest.json", manifest)
    _status(run, "report", "complete" if complete else "incomplete", {})
    return 0


def _read_optional(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return query_rows(_con(), f"SELECT * FROM read_parquet('{path.as_posix()}')")


STAGES = {
    "r0": stage_r0,
    "canary": stage_canary,
    "extract": stage_extract,
    "compare": stage_compare,
    "e1": stage_e1,
    "e2": stage_e2,
    "e3": stage_e3,
    "e5": stage_e5,
    "e6": stage_e6,
    "report": stage_report,
}


def main() -> int:
    if _login():
        raise SystemExit("refusing to run the continuation on the login node")
    stage = sys.argv[1]
    run_id = os.environ["PILOT_CHILD_RUN"]
    try:
        if stage == "init":
            init_child(run_id)
            return 0
        return STAGES[stage](run_id)
    except Exception:
        traceback.print_exc()
        try:
            _status(_child(), stage, "failed", {})
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

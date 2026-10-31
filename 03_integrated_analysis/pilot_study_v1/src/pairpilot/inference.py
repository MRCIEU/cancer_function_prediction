"""Model adapters. Tests use the synthetic adapter and never load a model."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

from pairpilot.config import budget_state, digest
from pairpilot.store import append_jsonl

KG_DENY = (
    "hasType", "predicate", "mapping_route", "relation_class", "NOTINF",
    "novelty", "entity_score", "semtype", "NEG_", "BIOMARKER", "ONCOGENE",
)
F_ALLOW = {
    "pair_id", "gene_id", "disease_cui", "aliases", "pmid", "title",
    "abstract", "sections", "text_version_id", "finding", "findings",
}


class BudgetBlocked(RuntimeError):
    pass


class CallCache:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def key(self, payload: dict) -> str:
        body = {
            "task": payload["task"],
            "input_sha256": payload["input_sha256"],
            "schema_id": payload["schema_id"],
            "prompt_sha256": payload["prompt_sha256"],
            "model_id": payload["model_id"],
            "params": payload.get("params") or {},
        }
        if payload.get("replicate_id") is not None:
            body["replicate_id"] = payload["replicate_id"]
        return digest(body)

    def get(self, key: str) -> dict | None:
        path = self.directory / f"{key}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, key: str, value: dict) -> None:
        path = self.directory / f"{key}.json"
        partial = path.with_suffix(".json.partial")
        partial.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        os_replace(partial, path)


def os_replace(src: Path, dst: Path) -> None:
    os.replace(src, dst)


def input_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def assert_no_kg_labels(payload: dict) -> None:
    text = json.dumps(payload, ensure_ascii=False)
    leaked = [token for token in KG_DENY if token in text]
    if leaked:
        raise ValueError(f"KG labels in a D/F payload: {leaked}")


def allowlist_payload(raw: dict) -> dict:
    return {key: raw[key] for key in raw if key in F_ALLOW}


class UsageLedger:
    def __init__(self, bundle: dict, path: Path):
        self.state = budget_state(bundle)
        self.path = path
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        if path.exists():
            prior = json.loads(path.read_text(encoding="utf-8"))
            self.calls = prior.get("calls") or 0
            self.input_tokens = prior.get("input_tokens") or 0
            self.output_tokens = prior.get("output_tokens") or 0

    def reserve(self) -> None:
        if not self.state["approved"]:
            raise BudgetBlocked(self.state["reason"])
        limit = self.state["max_api_calls"]
        if limit is not None and self.calls + 1 > limit:
            raise BudgetBlocked("not_evaluated_budget")

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        if self.state["max_input_tokens"] is not None and self.input_tokens > self.state["max_input_tokens"]:
            raise BudgetBlocked("not_evaluated_budget")
        if self.state["max_output_tokens"] is not None and self.output_tokens > self.state["max_output_tokens"]:
            raise BudgetBlocked("not_evaluated_budget")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }), encoding="utf-8")


class SyntheticAdapter:
    """Deterministic stand-in. It does not judge real science."""

    def complete(self, request: dict) -> dict:
        task = request["task"]
        if task == "extract_findings":
            content = {
                "findings": [{
                    "question_type": "prognostic_association",
                    "study_system": "human",
                    "study_system_detail": "adults",
                    "cancer_context": "the supplied cancer",
                    "exposure_or_intervention": "high expression",
                    "comparator": "low expression",
                    "outcome": "overall survival",
                    "timepoint": "",
                    "measurement_context": "tumor tissue",
                    "sample_size": "",
                    "sample_size_unit": "",
                    "adjustment_factors": "",
                    "perturbation_direction": "",
                    "result_direction": "higher_risk",
                    "effect": "",
                    "effect_unit": "",
                    "interval": "",
                    "p_value": "",
                    "significance": "not_reported",
                    "result_text": "reported association",
                    "discourse_role": "own_result",
                    "reporting_status": {"sample_size": "not_reported"},
                    "assessability": "insufficient",
                    "evidence": [{"quote": request.get("anchor_quote") or ""}],
                }],
            }
        elif task == "fail_truncate":
            return {"status": "truncated", "parsed": None, "raw": "{", "usage": {"input_tokens": 1, "output_tokens": 1}}
        else:
            content = {"status": "not_judged", "evidence": []}
        return {
            "status": "ok",
            "parsed": content,
            "raw": json.dumps(content),
            "usage": {"input_tokens": 10, "output_tokens": 20},
            "model_response_id": "synthetic",
        }


class OpenAIAdapter:
    def __init__(self, model_id: str, max_output_tokens: int, temperature: float):
        self.model_id = model_id
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature

    def complete(self, request: dict) -> dict:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise BudgetBlocked("openai_key_missing")
        body = {
            "model": self.model_id,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_output_tokens,
            "messages": [
                {"role": "system", "content": request["prompt"]},
                {"role": "user", "content": request["user"]},
            ],
        }
        data = json.dumps(body).encode("utf-8")
        http = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(http, timeout=120) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            return {"status": "http_error", "parsed": None, "raw": detail, "usage": {}, "model_response_id": None}
        choice = (payload.get("choices") or [{}])[0]
        raw = ((choice.get("message") or {}).get("content")) or ""
        finish = choice.get("finish_reason")
        usage = payload.get("usage") or {}
        parsed = None
        status = "ok"
        if finish == "length":
            status = "truncated"
        else:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                status = "unparsed"
        return {
            "status": status,
            "parsed": parsed,
            "raw": raw,
            "usage": {
                "input_tokens": usage.get("prompt_tokens") or 0,
                "output_tokens": usage.get("completion_tokens") or 0,
            },
            "model_response_id": payload.get("id"),
        }


def run_request(adapter, cache: CallCache, ledger: UsageLedger, request: dict, run_dir: Path, retries: int = 3) -> dict:
    """Network retries do not include format repair. Replicates bypass the primary cache."""
    if request.get("over_context"):
        return {"status": "not_evaluated_context", "parsed": None, "raw": "", "usage": {}}
    key = cache.key(request)
    replicate = request.get("replicate_id") is not None
    if not replicate:
        cached = cache.get(key)
        if cached:
            cached["cache_hit"] = True
            return cached
    attempt = 0
    last = None
    while attempt < retries:
        attempt += 1
        try:
            ledger.reserve()
            last = adapter.complete(request)
            ledger.add(
                int((last.get("usage") or {}).get("input_tokens") or 0),
                int((last.get("usage") or {}).get("output_tokens") or 0),
            )
        except BudgetBlocked as exc:
            last = {"status": "not_evaluated_budget", "parsed": None, "raw": str(exc), "usage": {}}
            break
        except Exception as exc:
            last = {"status": "network_error", "parsed": None, "raw": str(exc)[:300], "usage": {}}
            time.sleep(min(8, 2 ** (attempt - 1)))
            continue
        if last["status"] != "network_error":
            break
    last["attempt"] = attempt
    last["cache_hit"] = False
    record = {
        "task": request["task"],
        "cache_key": key,
        "replicate_id": request.get("replicate_id"),
        "status": last["status"],
        "model_response_id": last.get("model_response_id"),
        "usage": last.get("usage") or {},
    }
    append_jsonl(run_dir / "requests" / "requests.jsonl", record)
    append_jsonl(run_dir / "responses" / "responses.jsonl", {
        **record,
        "raw": last.get("raw") or "",
        "parsed": last.get("parsed"),
    })
    if last["status"] == "ok" and not replicate:
        cache.put(key, last)
    return last

"""Cross-job API budget. Reservations are conservative and shared through a file lock."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

PRICE = {
    "source": "https://developers.openai.com/api/docs/models/gpt-6.1-sol",
    "pricing_page": "https://developers.openai.com/api/docs/pricing",
    "retrieved_utc": "2026-10-15",
    "service_tier": "standard",
    "input_usd_per_million": 2.0,
    "cached_input_usd_per_million": 0.10,
    "output_usd_per_million": 10.0,
    "long_context_input_tokens": 272000,
    "long_context_input_multiplier": 2.0,
    "long_context_output_multiplier": 1.5,
    "note": "Fast and Ultrafast are not used. Output price includes billed reasoning tokens.",
}

LIMITS = {
    "authorization_source": "user message 2026-10-15, section 用户确认后发送的执行授权",
    "max_usd": 20.0,
    "canary_max_usd": 3.0,
    "max_attempts": 1000,
    "max_input_tokens": 3000000,
    "max_output_tokens": 1000000,
    "canary_attempts": 12,
    "canary_input_tokens": 200000,
    "canary_output_tokens": 200000,
    "max_output_tokens_per_call": 16000,
}


class BudgetBlocked(RuntimeError):
    pass


def estimate_usd(input_tokens: int, output_tokens: int, cached_input_tokens: int = 0) -> float:
    cached = min(max(cached_input_tokens, 0), input_tokens)
    fresh = input_tokens - cached
    long = input_tokens > PRICE["long_context_input_tokens"]
    input_mult = PRICE["long_context_input_multiplier"] if long else 1.0
    output_mult = PRICE["long_context_output_multiplier"] if long else 1.0
    return (
        fresh * PRICE["input_usd_per_million"] * input_mult
        + cached * PRICE["cached_input_usd_per_million"] * input_mult
        + output_tokens * PRICE["output_usd_per_million"] * output_mult
    ) / 1_000_000


def same_error_should_stop(last_kind: str, run_length: int, kind: str, limit: int = 3):
    if kind != last_kind:
        return False, kind, 1
    run_length += 1
    return run_length >= limit, kind, run_length


def available_usd(state: dict):
    """Unique remaining dollars. Missing fields stay unavailable instead of becoming zero."""
    needed = ("spent_usd", "reserved_usd", "billing_unknown_usd")
    if any(key not in state for key in needed):
        return None
    try:
        spent = float(state["spent_usd"])
        reserved = float(state["reserved_usd"])
        unknown = float(state["billing_unknown_usd"])
    except (TypeError, ValueError):
        return None
    open_usd = float(sum(float(item.get("usd") or 0) for item in (state.get("reservations") or {}).values()))
    overlap = 0.0
    if open_usd > 0 and abs(reserved - open_usd) < 1e-6 and abs(unknown - open_usd) < 1e-6:
        overlap = open_usd
    return LIMITS["max_usd"] - spent - reserved - unknown + overlap


def _blank() -> dict:
    return {
        "spent_usd": 0.0,
        "reserved_usd": 0.0,
        "billing_unknown_usd": 0.0,
        "attempts": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "canary_spent_usd": 0.0,
        "canary_reserved_usd": 0.0,
        "canary_attempts": 0,
        "canary_input_tokens": 0,
        "canary_output_tokens": 0,
        "reservations": {},
    }


class SharedLedger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = path.with_suffix(".lock")
        if not path.exists():
            self._write(_blank())

    def _locked(self):
        handle = self.lock_path.open("a+")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return handle

    def _read(self) -> dict:
        if not self.path.exists():
            return _blank()
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, state: dict) -> None:
        partial = self.path.with_suffix(".json.partial")
        partial.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(partial, self.path)

    def snapshot(self) -> dict:
        handle = self._locked()
        try:
            return self._read()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def reserve(self, reservation_id: str, input_tokens: int, output_tokens: int, scope: str) -> dict:
        cost = estimate_usd(input_tokens, output_tokens)
        handle = self._locked()
        try:
            state = self._read()
            self._check(state, cost, input_tokens, output_tokens, scope)
            state["reserved_usd"] += cost
            state["attempts"] += 1
            if scope == "canary":
                state["canary_reserved_usd"] += cost
                state["canary_attempts"] += 1
            state["reservations"][reservation_id] = {
                "usd": cost,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "scope": scope,
            }
            self._write(state)
            return {"reservation_id": reservation_id, "reserved_usd": cost}
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def settle(self, reservation_id: str, input_tokens: int, output_tokens: int, cached_input_tokens: int = 0) -> dict:
        actual = estimate_usd(input_tokens, output_tokens, cached_input_tokens)
        handle = self._locked()
        try:
            state = self._read()
            reserved = state["reservations"].pop(reservation_id, None)
            if reserved is None:
                raise KeyError(reservation_id)
            state["reserved_usd"] = max(0.0, state["reserved_usd"] - reserved["usd"])
            state["spent_usd"] += actual
            state["input_tokens"] += input_tokens
            state["output_tokens"] += output_tokens
            if reserved["scope"] == "canary":
                state["canary_reserved_usd"] = max(0.0, state["canary_reserved_usd"] - reserved["usd"])
                state["canary_spent_usd"] += actual
                state["canary_input_tokens"] += input_tokens
                state["canary_output_tokens"] += output_tokens
            self._write(state)
            return {"spent_usd": actual, "state": self._public(state)}
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def keep_unknown(self, reservation_id: str) -> None:
        """Move one in-flight reservation into unknown billing. It is not counted twice."""
        handle = self._locked()
        try:
            state = self._read()
            reserved = state["reservations"].pop(reservation_id, None)
            if reserved is None:
                return
            state["reserved_usd"] = max(0.0, state["reserved_usd"] - reserved["usd"])
            state["billing_unknown_usd"] += reserved["usd"]
            self._write(state)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def release(self, reservation_id: str) -> None:
        handle = self._locked()
        try:
            state = self._read()
            reserved = state["reservations"].pop(reservation_id, None)
            if reserved is None:
                return
            state["reserved_usd"] = max(0.0, state["reserved_usd"] - reserved["usd"])
            state["attempts"] = max(0, state["attempts"] - 1)
            if reserved["scope"] == "canary":
                state["canary_reserved_usd"] = max(0.0, state["canary_reserved_usd"] - reserved["usd"])
                state["canary_attempts"] = max(0, state["canary_attempts"] - 1)
            self._write(state)
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def _check(self, state: dict, cost: float, input_tokens: int, output_tokens: int, scope: str) -> None:
        committed = state["spent_usd"] + state["reserved_usd"] + state["billing_unknown_usd"]
        if committed + cost > LIMITS["max_usd"]:
            raise BudgetBlocked("usd_cap")
        if state["attempts"] + 1 > LIMITS["max_attempts"]:
            raise BudgetBlocked("attempt_cap")
        if state["input_tokens"] + input_tokens > LIMITS["max_input_tokens"]:
            raise BudgetBlocked("input_token_cap")
        if state["output_tokens"] + output_tokens > LIMITS["max_output_tokens"]:
            raise BudgetBlocked("output_token_cap")
        if scope == "canary":
            canary = state["canary_spent_usd"] + state["canary_reserved_usd"]
            if canary + cost > LIMITS["canary_max_usd"]:
                raise BudgetBlocked("canary_usd_cap")
            if state["canary_attempts"] + 1 > LIMITS["canary_attempts"]:
                raise BudgetBlocked("canary_attempt_cap")
            if state["canary_input_tokens"] + input_tokens > LIMITS["canary_input_tokens"]:
                raise BudgetBlocked("canary_input_cap")
            if state["canary_output_tokens"] + output_tokens > LIMITS["canary_output_tokens"]:
                raise BudgetBlocked("canary_output_cap")

    @staticmethod
    def _public(state: dict) -> dict:
        remaining = available_usd(state)
        return {
            "spent_usd": state["spent_usd"],
            "reserved_usd": state["reserved_usd"],
            "billing_unknown_usd": state["billing_unknown_usd"],
            "remaining_usd": remaining,
            "attempts": state["attempts"],
            "input_tokens": state["input_tokens"],
            "output_tokens": state["output_tokens"],
            "canary_spent_usd": state["canary_spent_usd"],
            "canary_attempts": state["canary_attempts"],
            "max_usd": LIMITS["max_usd"],
            "canary_max_usd": LIMITS["canary_max_usd"],
        }

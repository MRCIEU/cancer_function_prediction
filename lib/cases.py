"""Deterministic exploratory case selection. Not an error-rate sample."""

from __future__ import annotations

import hashlib


def _rank(seed: str, key: str) -> bytes:
    return hashlib.blake2b(f"{seed}|{key}".encode("utf-8"), digest_size=8).digest()


def _tertile(items: list[dict]) -> dict[str, list[dict]]:
    ordered = sorted(items, key=lambda row: (row["volume"], row["key"]))
    n = len(ordered)
    if n == 0:
        return {"low": [], "mid": [], "high": []}
    if n < 3:
        return {"low": ordered, "mid": [], "high": []}
    a = n // 3
    b = 2 * n // 3
    return {"low": ordered[:a], "mid": ordered[a:b], "high": ordered[b:]}


def select_cases(pools: dict[str, list[dict]], seed: str, target_n: int = 20) -> dict:
    """pools map a stratum name to rows with key and volume.

    Preference order: one from each requested stratum, then low/mid/high volume
    inside strata that still have candidates. Empty strata are reported and skipped.
    """
    selected = []
    reasons = []
    empty = sorted(name for name, rows in pools.items() if not rows)
    for name in empty:
        reasons.append({"stratum": name, "action": "absent", "pool_size": 0})
    present = [name for name in pools if pools[name]]
    used = set()

    def take(name: str, bucket: list[dict], why: str) -> None:
        choices = sorted(bucket, key=lambda row: _rank(seed, row["key"]))
        for row in choices:
            ident = (name, row["key"])
            if ident in used:
                continue
            used.add(ident)
            selected.append({**row, "stratum": name, "reason": why})
            return

    for name in present:
        take(name, pools[name], "one_per_stratum")
        if len(selected) >= target_n:
            break
    if len(selected) < target_n:
        for name in present:
            parts = _tertile(pools[name])
            for band in ("low", "mid", "high"):
                if len(selected) >= target_n:
                    break
                take(name, parts[band], f"volume_{band}")
    for name in present:
        reasons.append({"stratum": name, "action": "sampled", "pool_size": len(pools[name]), "selected": sum(1 for row in selected if row["stratum"] == name)})
    return {"seed": seed, "target_n": target_n, "selected": selected[:target_n], "strata": reasons, "note": "Exploratory reading sample. Do not estimate a corpus error rate from it."}

"""Scope comparison between source sentences and a full abstract."""

from __future__ import annotations


def group_source_text(records: list[dict]) -> list[dict]:
    """Keep original order inside a source. Do not join sources into one paragraph."""
    grouped = {}
    for row in records:
        grouped.setdefault(row.get("source") or "unknown", []).append(row)
    blocks = []
    for source in sorted(grouped):
        rows = sorted(grouped[source], key=lambda item: (item.get("ordinal") or 0, item.get("native_record_id") or ""))
        blocks.append({
            "source": source,
            "sentences": [row.get("text") or "" for row in rows],
        })
    return blocks


def scope_change(source_judgment: str, abstract_judgment: str) -> str:
    if source_judgment == "not_judged" or abstract_judgment == "not_judged":
        return "not_judged"
    if source_judgment == abstract_judgment:
        return "unchanged"
    return "changed"

"""Robustness diagnostics. Replicates must not reuse the primary response cache."""

from __future__ import annotations


def structured_key(row: dict) -> tuple:
    return (
        row.get("question_type"),
        row.get("study_system"),
        row.get("exposure_or_intervention"),
        row.get("comparator"),
        row.get("outcome"),
        row.get("result_direction"),
        row.get("significance"),
    )


def align_replicates(baseline: list[dict], repeat: list[dict]) -> dict:
    base = {structured_key(row): row for row in baseline}
    other = {structured_key(row): row for row in repeat}
    matched = sorted(set(base) & set(other))
    return {
        "matched": len(matched),
        "only_baseline": len(set(base) - set(other)),
        "only_repeat": len(set(other) - set(base)),
        "span_changed": sum(
            1 for key in matched
            if (base[key].get("evidence_start"), base[key].get("evidence_end"))
            != (other[key].get("evidence_start"), other[key].get("evidence_end"))
        ),
    }


def delete_span(text: str, start: int, end: int) -> str:
    return text[:start] + text[end:]


def control_deletion(text: str, start: int, end: int) -> tuple[str, int, int] | None:
    """Delete an equal-length span that does not overlap the cited span."""
    width = end - start
    if width <= 0:
        return None
    if start >= width:
        return delete_span(text, 0, width), 0, width
    tail = len(text) - end
    if tail >= width:
        return delete_span(text, end, end + width), end, end + width
    return None

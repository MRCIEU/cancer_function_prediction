"""Source increment and equal-count article draws. No extra model calls."""

from __future__ import annotations

import hashlib

from pairpilot.comparison import filter_comparisons
from pairpilot.contracts import metric_row, ratio


def article_sets(core: set[str], semmed: set[str]) -> dict:
    core = {str(item) for item in core}
    semmed = {str(item) for item in semmed}
    return {
        "CORE": core,
        "SemMedDB": semmed,
        "Union": core | semmed,
        "intersection": core & semmed,
    }


def condition_key(row: dict) -> tuple:
    return (
        row.get("question_type") or "unknown",
        row.get("study_system") or "unknown",
        _token(row.get("exposure_or_intervention")),
        _token(row.get("comparator")),
        _token(row.get("outcome")),
    )


def _token(value) -> str:
    text = " ".join(str(value or "").strip().lower().split())
    return text or "unknown"


def view_stats(findings: list[dict], comparisons: list[dict], pmids: set[str]) -> dict:
    pmids = {str(item) for item in pmids}
    kept = [row for row in findings if str(row["pmid"]) in pmids]
    rels = filter_comparisons(comparisons, pmids)
    conditions = {condition_key(row) for row in kept if row.get("discourse_role") != "background"}
    return {
        "n_pmid": len(pmids),
        "n_finding": len(kept),
        "n_condition": len(conditions),
        "n_comparison": len(rels),
        "n_insufficient": sum(1 for row in rels if row["condition_relation"] == "insufficient_context"),
        "n_comparable": sum(1 for row in rels if row["condition_relation"] == "comparable_on_reported_fields"),
    }


def increment(union_value: int, side_value: int) -> dict:
    return {"absolute": union_value - side_value, "numerator": union_value - side_value, "denominator": side_value}


def draw(items: list[str], k: int, seed: int, tag: str) -> list[str]:
    ranked = sorted(items, key=lambda item: hashlib.sha256(f"{seed}|{tag}|{item}".encode("utf-8")).hexdigest())
    return ranked[:k]


def equal_article(core: set[str], semmed: set[str], findings, comparisons, seed: int, draws: int = 50) -> dict:
    core = sorted(str(item) for item in core)
    semmed = sorted(str(item) for item in semmed)
    m = min(5, len(core), len(semmed))
    if m < 2:
        return {"status": "not_applicable", "m": m, "draws": []}
    rows = []
    for rep in range(draws):
        left = set(draw(core, m, seed, f"core:{rep}"))
        right = set(draw(semmed, m, seed, f"sem:{rep}"))
        rows.append({
            "replicate": rep,
            "overlap": len(left & right),
            "core": view_stats(findings, comparisons, left),
            "semmed": view_stats(findings, comparisons, right),
        })
    return {"status": "computed", "m": m, "draws": rows}


def source_metrics(run_id: str, pair: str, findings, comparisons, core: set[str], semmed: set[str]) -> list[dict]:
    sets = article_sets(core, semmed)
    stats = {name: view_stats(findings, comparisons, pmids) for name, pmids in sets.items() if name != "intersection"}
    rows = []
    for name, values in stats.items():
        for metric, value in values.items():
            rows.append(metric_row(
                run_id=run_id, experiment="E1", config=name, pair_id=pair, metric=metric,
                numerator=value, denominator=None, unit="count", value=value, status="computed",
                population_scope="selected_pair", source_table="tables/findings.parquet",
            ))
    for side in ("CORE", "SemMedDB"):
        added = increment(stats["Union"]["n_condition"], stats[side]["n_condition"])
        rows.append(metric_row(
            run_id=run_id, experiment="E1", config=f"Union_minus_{side}", pair_id=pair,
            metric="condition_increment", numerator=added["numerator"], denominator=added["denominator"],
            unit="condition_per_side_condition", value=ratio(added["numerator"], added["denominator"]),
            population_scope="selected_pair", source_table="tables/findings.parquet",
        ))
    return rows

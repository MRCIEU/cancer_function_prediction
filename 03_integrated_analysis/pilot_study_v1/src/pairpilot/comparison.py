"""Context and result relations. Unknown is never treated as equality."""

from __future__ import annotations

from pairpilot.contracts import UNKNOWN, candidate_id, comparison_id

FIELD_MAP = {
    "study_system": "study_system",
    "comparator": "comparator",
    "outcome": "outcome",
    "measurement_context": "measurement_context",
    "exposure_or_intervention": "exposure_or_intervention",
    "perturbation_direction": "perturbation_direction",
}


def _known(value) -> bool:
    return value not in UNKNOWN and str(value).strip() not in UNKNOWN


def _norm(value) -> str:
    return " ".join(str(value).strip().lower().split())


def required_fields(question_type: str, rules: dict) -> list[str]:
    if question_type in {"other", "unknown"}:
        return []
    fields = list(rules.get("all") or [])
    fields.extend(rules.get(question_type) or [])
    seen = []
    for field in fields:
        if field not in seen:
            seen.append(field)
    return seen


def context_relation(left: dict, right: dict, rules: dict) -> dict:
    q1 = left.get("question_type") or "unknown"
    q2 = right.get("question_type") or "unknown"
    if q1 != q2 and q1 not in {"unknown", "other"} and q2 not in {"unknown", "other"}:
        return {"condition_relation": "different_question", "checked": ["question_type"], "missing": []}
    if q1 in {"other", "unknown"} or q2 in {"other", "unknown"}:
        return {"condition_relation": "insufficient_context", "checked": ["question_type"], "missing": ["question_type"]}
    fields = required_fields(q1, rules)
    missing = []
    different = []
    checked = []
    for field in fields:
        checked.append(field)
        a = left.get(field)
        b = right.get(field)
        if not _known(a) or not _known(b):
            missing.append(field)
        elif _norm(a) != _norm(b):
            different.append(field)
    if missing:
        return {"condition_relation": "insufficient_context", "checked": checked, "missing": missing}
    if different:
        return {"condition_relation": "known_context_difference", "checked": checked, "missing": [], "different": different}
    return {"condition_relation": "comparable_on_reported_fields", "checked": checked, "missing": []}


def result_relation(left: dict, right: dict, condition: str) -> str:
    if condition != "comparable_on_reported_fields":
        return "not_assessed"
    d1 = left.get("result_direction")
    d2 = right.get("result_direction")
    if not _known(d1) or not _known(d2):
        return "not_assessed"
    # Significance is stored separately and does not reverse a direction.
    if _norm(d1) == _norm(d2):
        return "similar_reported_direction"
    return "different_reported_direction"


FAILED_STATUS = {
    "no_finding",
    "http_error",
    "http_400",
    "http_5xx",
    "quota_exhausted",
    "timeout_or_transport",
    "unparsed",
    "not_evaluated_budget",
    "openai_key_missing",
}


def is_analysis_finding(row: dict) -> bool:
    """A document-level empty result is not a finding and cannot enter a comparison."""
    if row.get("status") in FAILED_STATUS:
        return False
    if not row.get("question_type"):
        return False
    if row.get("discourse_role") == "background":
        return False
    return True


def _comparison_row(left: dict, right: dict, config_hash: str, rules: dict, scope: str) -> dict:
    condition = context_relation(left, right, rules)
    result = result_relation(left, right, condition["condition_relation"])
    return {
        "comparison_id": comparison_id(left["finding_id"], right["finding_id"], config_hash),
        "pair_id": left["pair_id"],
        "finding_a": left["finding_id"],
        "finding_b": right["finding_id"],
        "pmid_a": str(left["pmid"]),
        "pmid_b": str(right["pmid"]),
        "condition_relation": condition["condition_relation"],
        "result_relation": result,
        "checked_fields": ",".join(condition.get("checked") or []),
        "missing_fields": ",".join(condition.get("missing") or []),
        "different_fields": ",".join(condition.get("different") or []),
        "config_hash": config_hash,
        "origin": "deterministic_rule",
        "rule_version": "comparison_gap_v2",
        "comparison_scope": scope,
    }


def build_candidates(findings: list[dict], config_hash: str, rules: dict) -> list[dict]:
    by_pair = {}
    for row in findings:
        if not is_analysis_finding(row):
            continue
        by_pair.setdefault(row["pair_id"], []).append(row)
    out = []
    for rows in by_pair.values():
        rows = sorted(rows, key=lambda item: item["finding_id"])
        for i, left in enumerate(rows):
            for right in rows[i + 1:]:
                if left["pmid"] == right["pmid"]:
                    continue
                out.append(_comparison_row(left, right, config_hash, rules, "cross_pmid"))
    return out


def within_document_pairs(findings: list[dict], config_hash: str, rules: dict) -> list[dict]:
    by_paper = {}
    for row in findings:
        if not is_analysis_finding(row):
            continue
        by_paper.setdefault((row["pair_id"], str(row["pmid"])), []).append(row)
    out = []
    for rows in by_paper.values():
        rows = sorted(rows, key=lambda item: item["finding_id"])
        for i, left in enumerate(rows):
            for right in rows[i + 1:]:
                out.append(_comparison_row(left, right, config_hash, rules, "within_document"))
    return out


def filter_comparisons(comparisons: list[dict], pmids: set[str]) -> list[dict]:
    pmids = {str(item) for item in pmids}
    return [
        row for row in comparisons
        if str(row["pmid_a"]) in pmids and str(row["pmid_b"]) in pmids
    ]


def gap_from_comparison(row: dict, config_hash: str, findings_by_id: dict | None = None) -> dict | None:
    """A missing field is a comparison state, not by itself a research-gap candidate."""
    if row.get("comparison_scope") == "within_document":
        return None
    if row.get("condition_relation") != "comparable_on_reported_fields":
        return None
    if row.get("result_relation") != "different_reported_direction":
        return None
    left = (findings_by_id or {}).get(row.get("finding_a"), {})
    right = (findings_by_id or {}).get(row.get("finding_b"), {})
    question = left.get("question_type") or right.get("question_type") or ""
    if question in {"", "other", "unknown"}:
        return None
    parts = {
        "pair_id": row["pair_id"],
        "question_type": question,
        "exposure_or_intervention": left.get("exposure_or_intervention") or "",
        "comparator": left.get("comparator") or "",
        "outcome": left.get("outcome") or "",
        "missing_dimension": "reported_direction",
    }
    if not _known(parts["outcome"]):
        return None
    return {
        "candidate_id": candidate_id(parts, config_hash + "|cross_direction"),
        "pair_id": row["pair_id"],
        "gap_state": "candidate_cross_study_gap",
        "question_type": question,
        "exposure_or_intervention": parts["exposure_or_intervention"],
        "comparator": parts["comparator"],
        "outcome": parts["outcome"],
        "missing_dimension": "reported_direction",
        "why_it_matters": "Two findings report different directions on fields the rule treated as the same reported context.",
        "search_question": " ".join(str(parts[key]) for key in ("exposure_or_intervention", "comparator", "outcome")),
        "alternative_explanations": "Unreported population, assay, or wording differences can remain inside fields the rule treated as matching.",
        "unresolved": "This is a structural candidate. It is not a verified contradiction or a field-wide research gap.",
        "finding_ids": f"{row['finding_a']},{row['finding_b']}",
        "population_scope": "selected_pair_abstracts",
        "origin": "deterministic_rule",
        "rule_version": "comparison_gap_v2",
    }


def cluster_candidates(marks: list[dict]) -> list[dict]:
    grouped = {}
    for row in marks:
        grouped.setdefault(row["candidate_id"], []).append(row)
    out = []
    for candidate_id, rows in grouped.items():
        base = dict(rows[0])
        ids = []
        for row in rows:
            for item in str(row.get("finding_ids") or "").split(","):
                if item and item not in ids:
                    ids.append(item)
        base["finding_ids"] = ",".join(ids)
        base["n_source_marks"] = len(rows)
        base["source_comparison_ids"] = ",".join(row.get("comparison_id", "") for row in rows if row.get("comparison_id"))
        out.append(base)
    return out

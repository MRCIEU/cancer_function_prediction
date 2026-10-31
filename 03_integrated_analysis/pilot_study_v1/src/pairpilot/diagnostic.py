"""diagnostic_v1 rules. These do not rewrite comparison_gap_v2."""

from __future__ import annotations

import hashlib

from pairpilot.comparison import context_relation, is_analysis_finding
from pairpilot.contracts import UNKNOWN

VERSION = "diagnostic_v1"
OWN_RESULT = {"own_result"}
CLAIM_ROLES = {"author_interpretation", "recommendation"}
QUESTION_TYPES = {
    "expression_or_distribution",
    "prognostic_association",
    "disease_risk_association",
    "diagnostic_performance",
    "treatment_response_association",
    "functional_perturbation",
    "other",
    "unclear",
}
RECEPTOR = ("tnfrsf9", "cd137", "4-1bb")
LIGAND = ("tnfsf9", "cd137l", "4-1bbl")
CONSTRUCT = ("cd137-fc", "4-1bb-fc", "cd137 fc", "4-1bb fc")


def _known(value) -> bool:
    return value not in UNKNOWN and str(value or "").strip() not in UNKNOWN


def _norm(value) -> str:
    return " ".join(str(value or "").strip().lower().split())


def logical_task_id(pair_id: str, pmid: str, replicate_id: int = 0) -> str:
    return f"{VERSION}|{pair_id}|{pmid}|rep{replicate_id}"


def comparison_role(row: dict) -> str:
    role = row.get("discourse_role") or ""
    if row.get("status") == "no_finding" or not row.get("question_type"):
        return "not_a_finding"
    if role == "background":
        return "excluded_background"
    if role in CLAIM_ROLES:
        return "author_claim"
    if role in OWN_RESULT:
        return "comparable_result"
    return "not_comparable_pending"


def entity_identity(text: str) -> dict:
    """TNFRSF9 and TNFSF9 stay distinct. A fusion construct is not a target measurement."""
    low = _norm(text).replace(" ", "")
    spaced = _norm(text)
    if any(mark.replace(" ", "") in low for mark in CONSTRUCT):
        kind = "intervention_construct"
        gene_id = ""
    elif any(mark in low for mark in LIGAND):
        kind = "ligand_tnfsf9"
        gene_id = "8744"
    elif any(mark in low for mark in RECEPTOR):
        kind = "receptor_tnfrsf9"
        gene_id = "3604"
    else:
        kind = "unresolved"
        gene_id = ""
    return {
        "mention": text,
        "kind": kind,
        "gene_id": gene_id,
        "same_as_target_3604": kind == "receptor_tnfrsf9",
        "note": spaced,
    }


def trace_comparison(left: dict, right: dict, rules: dict) -> dict:
    """Explain the v2 branch without treating a missing field as abstract absence."""
    relation = context_relation(left, right, rules)
    q1 = left.get("question_type") or "unknown"
    q2 = right.get("question_type") or "unknown"
    branch = "comparable_on_reported_fields"
    if q1 != q2 and q1 not in {"unknown", "other"} and q2 not in {"unknown", "other"}:
        branch = "different_question"
    elif q1 in {"other", "unknown"} or q2 in {"other", "unknown"}:
        branch = "question_type_other_or_unknown"
    else:
        missing = relation.get("missing") or []
        different = relation.get("different") or []
        if missing:
            branch = "required_field_unknown_in_record"
        elif different:
            branch = "known_field_difference"
    observed = []
    for field in relation.get("checked") or []:
        if field == "question_type":
            continue
        a, b = left.get(field), right.get(field)
        if _known(a) and _known(b) and _norm(a) != _norm(b):
            observed.append(field)
    return {
        "condition_relation": relation["condition_relation"],
        "deciding_branch": branch,
        "question_type_a": q1,
        "question_type_b": q2,
        "missing_fields": ",".join(relation.get("missing") or []),
        "observed_known_difference_fields": ",".join(observed),
        "failure_is_not_abstract_absence": branch == "required_field_unknown_in_record",
    }


def diagnostic_dimension(left: dict, right: dict, field: str) -> str:
    left_status = left.get(f"{field}_reporting_status")
    right_status = right.get(f"{field}_reporting_status")
    if left_status == "not_applicable" or right_status == "not_applicable":
        return "not_applicable"
    if not _known(left.get(field)) or not _known(right.get(field)):
        return "unknown"
    if _norm(left.get(field)) == _norm(right.get(field)):
        return "same_on_reported_information"
    return "known_difference"


def eligible_pair(left: dict, right: dict) -> bool:
    if str(left.get("pmid")) == str(right.get("pmid")):
        return False
    if left.get("pair_id") != right.get("pair_id"):
        return False
    if comparison_role(left) != "comparable_result" or comparison_role(right) != "comparable_result":
        return False
    q1 = left.get("primary_question_type") or left.get("question_type")
    q2 = right.get("primary_question_type") or right.get("question_type")
    if not q1 or not q2 or q1 != q2 or q1 in {"other", "unclear"}:
        return False
    return True


def select_comparisons(candidates: list[dict], seed: int, per_pair: int = 12, total: int = 24) -> list[dict]:
    """Rotate across layers so the first layer cannot take the whole diagnostic quota."""
    layers = ("old_insufficient_with_text", "known_difference", "apparent_direction", "other_same_question")
    buckets = {}
    for layer in layers:
        pool = [row for row in candidates if row.get("layer") == layer]
        pool.sort(key=lambda row: hashlib.sha256(f"{seed}|{layer}|{row.get('candidate_key')}".encode()).hexdigest())
        buckets[layer] = pool
    chosen = []
    used_pairs = {}
    seen_papers = set()

    def take(layer: str, allow_repeat: bool):
        picked = None
        rest = []
        for row in buckets[layer]:
            if picked is not None:
                rest.append(row)
                continue
            if used_pairs.get(row.get("pair_id"), 0) >= per_pair:
                continue
            paper_pair = tuple(sorted((str(row.get("pmid_a")), str(row.get("pmid_b")))))
            if not allow_repeat and paper_pair in seen_papers:
                rest.append(row)
                continue
            picked = row
            seen_papers.add(paper_pair)
        buckets[layer] = rest
        return picked

    for allow_repeat in (False, True):
        while len(chosen) < total:
            moved = False
            for layer in layers:
                row = take(layer, allow_repeat)
                if row is None:
                    continue
                chosen.append({**row, "selected": True, "selection_layer": layer})
                used_pairs[row.get("pair_id")] = used_pairs.get(row.get("pair_id"), 0) + 1
                moved = True
                if len(chosen) >= total:
                    break
            if not moved:
                break
    return chosen


def e5_block_reasons(candidate: dict) -> list[str]:
    """A direction conflict is not required. An original-paper methods hole is not searchable here."""
    reasons = []
    if not _known(candidate.get("target_question")):
        reasons.append("no_target_question")
    if not (candidate.get("evidence_refs") or candidate.get("quote")):
        reasons.append("no_evidence")
    if not _known(candidate.get("search_question")):
        reasons.append("no_search_question")
    if candidate.get("entity_conflict"):
        reasons.append("entity_conflict")
    if candidate.get("only_original_methods_omission"):
        reasons.append("original_study_detail_not_searchable")
    if candidate.get("duplicate_question_key"):
        reasons.append("duplicate_question")
    return reasons


def question_key(candidate: dict) -> str:
    return "|".join(_norm(candidate.get(name)) for name in ("pair_id", "target_question", "missing_dimension"))


def freeze_e5(candidates: list[dict], seed: int, limit: int = 3) -> list[dict]:
    """At most one candidate per pair, at most three, in hash order."""
    seen = set()
    pool = []
    for row in candidates:
        if e5_block_reasons(row):
            continue
        key = question_key(row)
        if key in seen:
            continue
        seen.add(key)
        pool.append(row)
    pool.sort(key=lambda row: hashlib.sha256(f"{seed}|e5|{row.get('candidate_id')}|{question_key(row)}".encode()).hexdigest())
    chosen = []
    used_pairs = set()
    for row in pool:
        if len(chosen) >= limit:
            break
        if row.get("pair_id") in used_pairs:
            continue
        used_pairs.add(row.get("pair_id"))
        chosen.append(row)
    return chosen


def view_allows(comparison: dict, pmids: set[str]) -> bool:
    return str(comparison.get("pmid_a")) in pmids and str(comparison.get("pmid_b")) in pmids


def normalize_tuple(value: str) -> str:
    chars = []
    for char in str(value or "").lower():
        if char.isalnum() or char.isspace():
            chars.append(char)
        else:
            chars.append(" ")
    return " ".join("".join(chars).split())


def is_analysis_for_v2(row: dict) -> bool:
    return is_analysis_finding(row)

"""query_builder_v2. Scientific PubMed queries are an AND of concept groups."""

from __future__ import annotations

import re

VERSION = "query_builder_v2"
PAIR_ID = re.compile(r"\d+__C\d+")
BARE_CUI = re.compile(r"^C\d+$")
CLASSES = ("both_individually", "core_only", "semmed_only", "union_required")


class QueryRejected(ValueError):
    pass


def _term(text: str, role: str, source: str, field: str, original: str, normalization: str) -> dict:
    cleaned = " ".join(str(text or "").split())
    if not cleaned or PAIR_ID.search(cleaned) or BARE_CUI.fullmatch(cleaned):
        raise QueryRejected(f"scientific query rejected term: {cleaned}")
    if field not in {"", "tiab", "mesh", "uid"}:
        raise QueryRejected(f"unsupported field: {field}")
    return {
        "text": cleaned,
        "role": role,
        "source": source,
        "field": field,
        "original": original,
        "normalization": normalization,
    }


def group(role: str, terms: list[dict]) -> dict:
    if not terms:
        raise QueryRejected(f"empty concept group: {role}")
    if any(term["role"] != role for term in terms):
        raise QueryRejected("a concept group cannot mix roles")
    return {"role": role, "operator": "OR", "terms": terms}


def and_query(kind: str, groups: list[dict], uid_pmids: list[str] | None = None) -> dict:
    roles = [item["role"] for item in groups]
    if "gene" not in roles or "disease" not in roles:
        raise QueryRejected("a scientific query needs a gene group AND a disease group")
    if kind == "broad" and roles != ["gene", "disease"]:
        raise QueryRejected("a broad query is only gene AND disease")
    if kind == "specific" and roles != ["gene", "disease", "condition"]:
        raise QueryRejected("a specific query adds one condition group")
    if uid_pmids is not None and any(not str(pmid).isdigit() for pmid in uid_pmids):
        raise QueryRejected("uid check accepts PubMed ids only")
    return {"version": VERSION, "kind": kind, "operator": "AND", "groups": groups, "uid_pmids": list(uid_pmids or [])}


def _render_term(term: dict) -> str:
    text = term["text"]
    literal = f'"{text}"' if " " in text else text
    if term["field"]:
        return f"{literal}[{term['field']}]"
    return literal


def render(query: dict) -> str:
    parts = []
    for item in query["groups"]:
        parts.append("(" + " OR ".join(_render_term(term) for term in item["terms"]) + ")")
    text = " AND ".join(parts)
    if query.get("uid_pmids"):
        uids = " OR ".join(f"{pmid}[uid]" for pmid in query["uid_pmids"])
        text = f"({text}) AND ({uids})"
    if PAIR_ID.search(text):
        raise QueryRejected("pair id entered a scientific query")
    return text


def source_membership(pmids: set[str], core: set[str], semmed: set[str]) -> dict:
    """Material presence only. This is not a claim that one library reproduced the judgement."""
    needed = {str(pmid) for pmid in pmids}
    missing = sorted(pmid for pmid in needed if pmid not in core and pmid not in semmed)
    in_core = bool(needed) and needed <= core
    in_semmed = bool(needed) and needed <= semmed
    in_union = bool(needed) and not missing
    if not needed or missing:
        klass = "unresolved"
    elif in_core and in_semmed:
        klass = "both_individually"
    elif in_core:
        klass = "core_only"
    elif in_semmed:
        klass = "semmed_only"
    else:
        klass = "union_required"
    return {
        "available_in_core": in_core,
        "available_in_semmed": in_semmed,
        "available_in_union": in_union,
        "mutually_exclusive_source_class": klass,
        "missing_pmids": ",".join(missing),
    }


def pmid_source(pmid: str, core: set[str], semmed: set[str]) -> str:
    in_core = str(pmid) in core
    in_semmed = str(pmid) in semmed
    if in_core and in_semmed:
        return "shared"
    if in_core:
        return "core_only"
    if in_semmed:
        return "semmed_only"
    return "outside_both"


def class_counts(rows: list[dict]) -> dict:
    counts = {name: 0 for name in (*CLASSES, "unresolved")}
    for row in rows:
        counts[row["mutually_exclusive_source_class"]] += 1
    classified = [row for row in rows if row["mutually_exclusive_source_class"] in CLASSES]
    total = len(classified)
    both = counts["both_individually"]
    core_n = both + counts["core_only"]
    semmed_n = both + counts["semmed_only"]
    implied_union = total - (core_n + semmed_n - both)
    exclusive = sum(counts[name] for name in CLASSES)
    return {
        **counts,
        "n_classified": total,
        "n_rows": len(rows),
        "exclusive_cover": exclusive == total and counts["unresolved"] == len(rows) - total,
        "core_available": core_n,
        "semmed_available": semmed_n,
        "intersection_x": both,
        "implied_union_required": implied_union,
    }


def paper_pair_key(pmid_a: str, pmid_b: str) -> str:
    left, right = sorted((str(pmid_a), str(pmid_b)))
    return f"{left}|{right}"


def truncation_record(count, returned: int, retmax: int = 50) -> dict:
    count_known = isinstance(count, int)
    return {
        "hit_count": count if count_known else None,
        "hit_count_status": "returned" if count_known else "unknown",
        "retmax": retmax,
        "returned_pmid_n": returned,
        "truncated": bool(count_known and count > retmax),
    }


def choose_new(ranked: list[str], blocked: set[str], limit: int) -> dict:
    """Top-list order. Already seen PMIDs stay linked but do not take a new slot."""
    included = []
    already = []
    not_taken = []
    for pmid in ranked:
        pmid = str(pmid)
        if not pmid:
            continue
        if pmid in blocked or pmid in included:
            already.append(pmid)
            continue
        if len(included) >= limit:
            not_taken.append(pmid)
            continue
        included.append(pmid)
    return {"included": included, "already_present": already, "not_taken_in_top": not_taken}


def candidate_outcome(query_status: str, retrieval_status: str, model_state: str) -> str:
    if query_status != "valid":
        return "not_assessable_query_failure"
    if retrieval_status in {"not_started", "fetch_failed", "no_new_material", "zero_hit"}:
        return "no_new_material_in_bounded_search" if retrieval_status != "not_started" else "not_started"
    return model_state or "not_recorded"

"""Workflow payloads. D and F never receive KG labels. K keeps raw labels unchanged."""

from __future__ import annotations

from pairpilot.inference import F_ALLOW, allowlist_payload, assert_no_kg_labels


def direct_payload(pair: dict, documents: list[dict]) -> dict:
    payload = allowlist_payload({
        "pair_id": pair["pair_id"],
        "gene_id": pair["gene_id"],
        "disease_cui": pair["disease_cui"],
        "aliases": pair.get("aliases") or [],
        "abstract": [
            {"pmid": doc["pmid"], "title": doc.get("title") or "", "text": doc.get("text") or ""}
            for doc in documents
        ],
    })
    assert_no_kg_labels(payload)
    return payload


def extract_payload(pair: dict, document: dict) -> dict:
    payload = allowlist_payload({
        "pair_id": pair["pair_id"],
        "gene_id": pair["gene_id"],
        "disease_cui": pair["disease_cui"],
        "aliases": pair.get("aliases") or [],
        "pmid": document["pmid"],
        "title": document.get("title") or "",
        "abstract": document.get("text") or "",
        "text_version_id": document.get("text_version_id") or "",
    })
    assert_no_kg_labels(payload)
    return payload


def knowledge_payload(pair: dict, document: dict, records: list[dict]) -> dict:
    """K may see raw attributes. It must not rewrite them into a single gold label."""
    payload = extract_payload(pair, document)
    payload["source_records"] = [
        {
            "native_record_id": row.get("native_record_id"),
            "source": row.get("source"),
            "predicate": row.get("predicate"),
            "negated": row.get("negated"),
            "has_type": row.get("has_type"),
            "label_tuple": row.get("label_tuple"),
            "text": row.get("text"),
        }
        for row in records
    ]
    return payload


def output_delta(left: dict, right: dict) -> dict:
    return {
        "relations_left": left.get("n_relation") or 0,
        "relations_right": right.get("n_relation") or 0,
        "gaps_left": left.get("n_gap") or 0,
        "gaps_right": right.get("n_gap") or 0,
        "located_left": left.get("n_located") or 0,
        "located_right": right.get("n_located") or 0,
        "unlocated_left": left.get("n_unlocated") or 0,
        "unlocated_right": right.get("n_unlocated") or 0,
        "calls_left": left.get("n_call") or 0,
        "calls_right": right.get("n_call") or 0,
    }

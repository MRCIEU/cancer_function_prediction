"""Identifiers, span checks, and output-contract validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

UNKNOWN = {"", None, "unknown", "not_reported", "unclear", "not_assessed"}
SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"


def pair_id(gene_id: str, disease_cui: str) -> str:
    return f"{gene_id}__{disease_cui}"


def pair_key(gene_id: str, disease_cui: str) -> str:
    return f"{str(gene_id).strip()}|{str(disease_cui).strip()}"


def text_version_id(source: str, payload: str) -> str:
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{source}:{digest}"


def finding_id(run_id: str, pair: str, pmid: str, local: int) -> str:
    return f"{run_id}:{pair}:{pmid}:{local}"


def evidence_id(version: str, start: int, end: int) -> str:
    return f"{version}:{start}:{end}"


def comparison_id(left: str, right: str, config_hash: str) -> str:
    ordered = "__".join(sorted([left, right]))
    return hashlib.sha256(f"{config_hash}|{ordered}".encode("utf-8")).hexdigest()[:24]


def candidate_id(parts: dict, config_hash: str) -> str:
    body = "|".join(str(parts.get(key) or "") for key in (
        "pair_id", "question_type", "exposure_or_intervention", "comparator", "outcome", "missing_dimension"
    ))
    return hashlib.sha256(f"{config_hash}|{body}".encode("utf-8")).hexdigest()[:24]


def schema(name: str) -> dict:
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


def exact_span(text: str, start: int, end: int, quote: str) -> bool:
    if not isinstance(start, int) or not isinstance(end, int):
        return False
    if start < 0 or end < start or end > len(text):
        return False
    return text[start:end] == quote


def locate_quote(text: str, quote: str) -> tuple[int, int] | None:
    if not quote:
        return None
    start = text.find(quote)
    if start < 0:
        return None
    return start, start + len(quote)


def metric_row(**kwargs) -> dict:
    denominator = kwargs.get("denominator")
    value = kwargs.get("value")
    status = kwargs.get("status") or "computed"
    if denominator == 0:
        value = None
        status = "not_applicable"
    row = {
        "run_id": kwargs.get("run_id"),
        "experiment": kwargs["experiment"],
        "config": kwargs["config"],
        "pair_id": kwargs.get("pair_id"),
        "metric": kwargs["metric"],
        "numerator": kwargs.get("numerator"),
        "denominator": denominator,
        "unit": kwargs.get("unit"),
        "value": value,
        "status": status,
        "population_scope": kwargs.get("population_scope"),
        "source_table": kwargs.get("source_table"),
    }
    return row


def ratio(numerator, denominator):
    if denominator in (0, None):
        return None
    return numerator / denominator

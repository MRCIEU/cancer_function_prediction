"""Gene and cancer identifier resolution.

Rules are the study design. Display names are never used as a match key.
"""

from __future__ import annotations

import re

CUI_RE = re.compile(r"^C\d+$")
GENE_RE = re.compile(r"^\d+$")
COMPOSITE_RE = re.compile(r"^(C\d+)\|(\d+)$")
MULTI_GENE_RE = re.compile(r"^\d+(,\d+)+$")

GENE_SEMTYPES = {"gngm", "aapp"}
CANCER_SEMTYPE = "neop"
HUMAN_TAX = "9606"


def semantic_tokens(value: str | None) -> set[str]:
    if not value:
        return set()
    return {token.strip() for token in str(value).split(",") if token.strip() and token.strip() not in {"\\N", "NULL"}}


def endpoint_class(tokens: set[str]) -> str:
    gene = bool(tokens & GENE_SEMTYPES)
    cancer = CANCER_SEMTYPE in tokens
    if gene and cancer:
        return "ambiguous_semtype"
    if gene:
        return "gene"
    if cancer:
        return "cancer"
    return "other"


def parse_endpoint_identifier(raw: str | None) -> dict:
    text = "" if raw is None else str(raw).strip()
    if text in {"", "\\N", "NULL", "null", "None"}:
        return {"pattern": "missing", "cui": None, "embedded_gene_ids": []}
    if CUI_RE.fullmatch(text):
        return {"pattern": "cui", "cui": text, "embedded_gene_ids": []}
    if GENE_RE.fullmatch(text):
        return {"pattern": "numeric_gene", "cui": None, "embedded_gene_ids": [text.lstrip("0") or "0"]}
    match = COMPOSITE_RE.fullmatch(text)
    if match:
        gene = match.group(2).lstrip("0") or "0"
        return {"pattern": "cui_pipe_gene", "cui": match.group(1), "embedded_gene_ids": [gene]}
    if "|" in text or "," in text:
        return {"pattern": "unresolved_identifier_pattern", "cui": None, "embedded_gene_ids": []}
    return {"pattern": "unresolved_identifier_pattern", "cui": None, "embedded_gene_ids": []}


def parse_entity_gene_cell(raw: str | None) -> dict:
    text = "" if raw is None else str(raw).strip()
    if text in {"", "\\N", "NULL", "null", "None"}:
        return {"kind": "missing", "gene_ids": []}
    if GENE_RE.fullmatch(text):
        return {"kind": "single", "gene_ids": [text.lstrip("0") or "0"]}
    if MULTI_GENE_RE.fullmatch(text):
        return {
            "kind": "composite_mention",
            "gene_ids": [part.lstrip("0") or "0" for part in text.split(",")],
        }
    return {"kind": "other", "gene_ids": []}


def resolve_cui_gene_ids(gene_ids: set[str], tax_by_gene: dict[str, str]) -> dict:
    """Resolve one CUI from the set of single-valued ENTITY gene identifiers."""
    cleaned = {gene for gene in gene_ids if gene}
    if not cleaned:
        return {"status": "no_gene_id", "gene_id": None, "route": "entity_gene_id"}
    unknown = sorted(gene for gene in cleaned if gene not in tax_by_gene)
    if unknown:
        return {
            "status": "unresolved_unknown_gene_id",
            "gene_id": None,
            "route": "entity_gene_id",
            "unknown_gene_ids": unknown,
        }
    if len(cleaned) > 1:
        return {
            "status": "ambiguous_multiple_gene_ids",
            "gene_id": None,
            "route": "entity_gene_id",
            "gene_ids": sorted(cleaned),
            "tax_ids": sorted({tax_by_gene[gene] for gene in cleaned}),
        }
    only = next(iter(cleaned))
    tax = tax_by_gene[only]
    if tax != HUMAN_TAX:
        return {
            "status": "unresolved_nonhuman",
            "gene_id": None,
            "route": "entity_gene_id",
            "observed_gene_id": only,
            "tax_id": tax,
        }
    return {"status": "resolved", "gene_id": only, "route": "entity_gene_id", "tax_id": tax}


def resolve_gene_endpoint(raw: str | None, cui_resolution: dict | None, tax_by_gene: dict[str, str]) -> dict:
    """Combine a predication identifier with the entity-table CUI resolution."""
    parsed = parse_endpoint_identifier(raw)
    pattern = parsed["pattern"]
    if pattern in {"missing", "unresolved_identifier_pattern"}:
        return {"status": pattern, "gene_id": None, "route": "predication_identifier", "cui": parsed["cui"]}
    embedded = parsed["embedded_gene_ids"]
    if any(gene not in tax_by_gene for gene in embedded):
        return {
            "status": "unresolved_unknown_gene_id",
            "gene_id": None,
            "route": "predication_identifier",
            "cui": parsed["cui"],
        }
    if any(tax_by_gene[gene] != HUMAN_TAX for gene in embedded):
        return {
            "status": "unresolved_nonhuman",
            "gene_id": None,
            "route": "predication_identifier",
            "cui": parsed["cui"],
            "tax_id": tax_by_gene[embedded[0]] if len(embedded) == 1 else None,
        }
    embedded_human = embedded
    cui_status = None if cui_resolution is None else cui_resolution.get("status")
    cui_gene = None if cui_resolution is None else cui_resolution.get("gene_id")

    if pattern == "numeric_gene":
        if len(embedded_human) != 1:
            return {"status": "ambiguous_multiple_gene_ids", "gene_id": None, "route": "predication_identifier"}
        return {
            "status": "resolved",
            "gene_id": embedded_human[0],
            "route": "predication_identifier_numeric",
            "cui": None,
            "tax_id": HUMAN_TAX,
        }

    if pattern == "cui" or pattern == "cui_pipe_gene":
        if cui_status == "ambiguous_multiple_gene_ids":
            return {
                "status": "ambiguous_multiple_gene_ids",
                "gene_id": None,
                "route": "entity_gene_id",
                "cui": parsed["cui"],
            }
        if pattern == "cui":
            if cui_status == "resolved":
                return {
                    "status": "resolved",
                    "gene_id": cui_gene,
                    "route": "entity_gene_id",
                    "cui": parsed["cui"],
                    "tax_id": HUMAN_TAX,
                }
            return {
                "status": cui_status or "no_gene_id",
                "gene_id": None,
                "route": "entity_gene_id",
                "cui": parsed["cui"],
            }
        # Composite CUI|GeneID. Do not let the embedded identifier break an ambiguity.
        embedded_gene = embedded_human[0]
        if cui_status == "resolved" and cui_gene == embedded_gene:
            return {
                "status": "resolved",
                "gene_id": embedded_gene,
                "route": "predication_identifier_and_entity_gene_id",
                "cui": parsed["cui"],
                "tax_id": HUMAN_TAX,
            }
        if cui_status == "resolved" and cui_gene != embedded_gene:
            return {
                "status": "unresolved_route_conflict",
                "gene_id": None,
                "route": "predication_identifier_and_entity_gene_id",
                "cui": parsed["cui"],
            }
        if cui_status in {None, "no_gene_id", "not_in_entity"}:
            return {
                "status": "resolved",
                "gene_id": embedded_gene,
                "route": "predication_identifier_composite",
                "cui": parsed["cui"],
                "tax_id": HUMAN_TAX,
            }
        return {
            "status": cui_status or "no_gene_id",
            "gene_id": None,
            "route": "predication_identifier_composite",
            "cui": parsed["cui"],
        }
    return {"status": "unresolved_identifier_pattern", "gene_id": None, "route": "predication_identifier"}


def informative_core_tuple(cge: str, ccs: str, gci: str, gcc: str, has_type: str) -> bool:
    return any(
        (
            cge in {"UP", "DOWN"},
            ccs in {"PROGRESSION", "REGRESSION"},
            gci in {"CAUSALITY", "OBSERVATION"},
            gcc in {"EXPRESSION", "OTHER"},
            has_type in {"BIOMARKER", "ONCOGENE", "TSG"},
        )
    )


def classify_attribute_sets(left: set[tuple], right: set[tuple]) -> str:
    if not left or not right:
        return "insufficient_information"
    if left == right:
        return "comparable_attributes"
    return "different_attributes"


def core_has_type_path_eligible(has_type: str | None) -> bool:
    if has_type is None or has_type == "":
        return True
    return has_type in {"BIOMARKER", "ONCOGENE", "TSG"}

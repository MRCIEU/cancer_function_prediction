"""Explicit identifier rules taken from the CORE builder and SemMedDB dbinfo.

Rules apply only when the URI namespace matches. Unmatched strings stay unverified.
"""

from __future__ import annotations

import re

from lib.schema import ARTICLE_NS, DISEASE_NS, GENE_NS

_CUI_RE = re.compile(r"^C\d{7,8}$")
_DIGITS_RE = re.compile(r"^\d+$")


def _local(uri: str, namespace: str) -> str | None:
    if uri.startswith(namespace):
        local = uri[len(namespace) :]
    else:
        http = "http://" + namespace[len("https://") :] if namespace.startswith("https://") else None
        https = "https://" + namespace[len("http://") :] if namespace.startswith("http://") else None
        local = None
        for alt in (http, https):
            if alt and uri.startswith(alt):
                local = uri[len(alt) :]
                break
        if local is None:
            return None
    return local.strip().strip("/")


def parse_core_gene(uri: str) -> dict:
    local = _local(uri, GENE_NS)
    if local is None:
        return {
            "raw_id": uri,
            "namespace": "unmatched",
            "normalized_id": None,
            "system": "ncbigene",
            "status": "no_mapping",
        }
    scheme = "explicit_namespace" if uri.startswith(GENE_NS) else "scheme_variant_explicit"
    if _DIGITS_RE.fullmatch(local):
        return {
            "raw_id": uri,
            "namespace": GENE_NS,
            "normalized_id": local,
            "system": "ncbigene",
            "status": scheme,
        }
    if "|" in local:
        return {
            "raw_id": uri,
            "namespace": GENE_NS,
            "normalized_id": None,
            "system": "ncbigene",
            "status": "composite_unexpanded",
        }
    return {
        "raw_id": uri,
        "namespace": GENE_NS,
        "normalized_id": None,
        "system": "ncbigene",
        "status": "namespace_explicit_but_not_numeric",
    }


def parse_core_disease(uri: str) -> dict:
    local = _local(uri, DISEASE_NS)
    if local is None:
        return {
            "raw_id": uri,
            "namespace": "unmatched",
            "normalized_id": None,
            "system": "umls_cui",
            "status": "no_mapping",
        }
    if _CUI_RE.fullmatch(local):
        return {
            "raw_id": uri,
            "namespace": DISEASE_NS,
            "normalized_id": local,
            "system": "umls_cui",
            "status": "explicit_namespace",
        }
    if local.startswith("C") and local[1:].isdigit():
        return {
            "raw_id": uri,
            "namespace": DISEASE_NS,
            "normalized_id": local,
            "system": "umls_cui",
            "status": "explicit_namespace_pattern_unusual",
        }
    return {
        "raw_id": uri,
        "namespace": DISEASE_NS,
        "normalized_id": None,
        "system": "umls_cui",
        "status": "namespace_explicit_but_localname_not_cui_pattern",
    }


def parse_core_pmid(uri: str) -> dict:
    local = _local(uri, ARTICLE_NS)
    if local is None:
        return {
            "raw_id": uri,
            "namespace": "unmatched",
            "normalized_id": None,
            "system": "pmid",
            "status": "no_mapping",
        }
    scheme = "explicit_namespace" if uri.startswith(ARTICLE_NS) else "scheme_variant_explicit"
    if _DIGITS_RE.fullmatch(local):
        return {
            "raw_id": uri,
            "namespace": ARTICLE_NS,
            "normalized_id": local.lstrip("0") or "0",
            "system": "pmid",
            "status": scheme,
        }
    return {
        "raw_id": uri,
        "namespace": ARTICLE_NS,
        "normalized_id": None,
        "system": "pmid",
        "status": "namespace_explicit_but_not_numeric",
    }


def classify_gene_id_token(raw: str | None) -> str:
    if raw is None:
        return "sql_null"
    if raw == "":
        return "empty_string"
    if raw == "\\N":
        return "backslash_N"
    if raw in {"NULL", "null", "None"}:
        return "literal_NULL"
    if "|" in raw:
        return "composite_pipe"
    if _DIGITS_RE.fullmatch(raw):
        return "numeric"
    return "other"


def numeric_gene_id(raw: str | None) -> str | None:
    if classify_gene_id_token(raw) == "numeric":
        return raw.lstrip("0") or "0"
    return None


def usable_id_status(status: str) -> bool:
    return status in {
        "explicit_namespace",
        "scheme_variant_explicit",
        "explicit_namespace_pattern_unusual",
    }

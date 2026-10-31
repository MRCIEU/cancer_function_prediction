"""Identifier parsing for id_alignment_v2.

Acceptance thresholds are fixed here, before the validation job sees the rates.
"""

from __future__ import annotations

import json
import re

RULE_VERSION = "id_alignment_v2"
GENOMIC_SEMTYPES = frozenset({"gngm", "aapp"})
_CUI = re.compile(r"^C[0-9]+$")

# Official statements used to name the identifier. Local CSV column order is not taken from these files.
DOCUMENT_EVIDENCE = [
    {
        "id": "semrep_xml_v1_8",
        "url": "https://github.com/LHNCBC/SemRep/blob/master/doc/SemRep.v1.8_XML_output_desc.txt",
        "retrieved": "2026-10-15",
        "statements": [
            "Entity.entrezID is a unique identifier from EntrezGene for genomic concepts.",
            "Entity.entrezName is the EntrezGene name.",
            "Entity.cui is a UMLS CUI or any other unique source identifier.",
            "Entity.begin is the 0-based character offset of the first character of the mention.",
            "Entity.end is the character offset for the end of the mention. The text does not say whether that offset is inclusive.",
        ],
    },
    {
        "id": "semmeddb_dbinfo",
        "url": "https://lhncbc.nlm.nih.gov/temp/SemRep_SemMedDB_SKR/dbinfo.html",
        "retrieved": "2026-10-15",
        "statements": [
            "ENTITY.GENE_ID is the EntrezGene ID of the entity.",
            "ENTITY.START_INDEX is the first character position of the entity in the document.",
            "ENTITY.END_INDEX is the last character position of the entity in the document.",
            "SENTENCE.SENT_START_INDEX is the first character of the sentence in the MEDLINE citation text.",
            "SENTENCE.SENT_END_INDEX is the last character of the sentence in that text.",
            "PREDICATION_AUX subject and object start/end indexes are character positions in the document.",
            "PREDICATION.SUBJECT_CUI is described as the CUI. A pipe-separated value is not defined on this page.",
        ],
    },
    {
        "id": "semrep_fielded_pdf",
        "url": "https://github.com/LHNCBC/SemRep/blob/master/doc/SemRep_full_fielded_output.pdf",
        "retrieved": "2026-10-15",
        "statements": [
            "Recorded as the fielded-output description. It is not used as the column order of the local CSV.",
        ],
    },
]

THRESHOLDS = {
    "ids_per_quartile": 100,
    "min_presence_current_or_history": 0.90,
    "min_current_name_agreement": 0.70,
    "max_conflict_when_entity_gene_present": 0.05,
    "min_rows_for_conflict_rate": 50,
    "coordinate_adopt_match": 0.90,
    "coordinate_reject_alternative": 0.50,
}


def norm_digits(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text.isdigit():
        return None
    stripped = text.lstrip("0")
    return stripped if stripped else "0"


def semtype_is_genomic(semtype: str | None) -> bool:
    if not semtype:
        return False
    tokens = [part.strip() for part in str(semtype).split(",") if part.strip()]
    return any(token in GENOMIC_SEMTYPES for token in tokens)


def parse_argument(raw: str | None) -> dict:
    if raw is None or str(raw).strip() in {"", r"\N", "N"}:
        return {"argument_format": "empty", "parts": [], "candidate_gene_ids": [], "cui": None}
    text = str(raw).strip()
    chunks = text.split("|") if "|" in text else [text]
    parts = []
    for ordinal, chunk in enumerate(chunks, start=1):
        if chunk.isdigit():
            kind = "digits"
        elif _CUI.fullmatch(chunk):
            kind = "umls_cui"
        else:
            kind = "other"
        parts.append({"ord": ordinal, "text": chunk, "kind": kind, "norm": norm_digits(chunk) if kind == "digits" else None})
    candidates: list[str] = []
    for part in parts:
        if part["norm"] and part["norm"] not in candidates:
            candidates.append(part["norm"])
    cuis = [part["text"] for part in parts if part["kind"] == "umls_cui"]
    if "|" not in text:
        if len(parts) == 1 and parts[0]["kind"] == "digits":
            fmt = "single_digits"
        elif len(parts) == 1 and parts[0]["kind"] == "umls_cui":
            fmt = "single_umls_cui"
        else:
            fmt = "other"
    elif len(candidates) >= 2:
        fmt = "pipe_multiple_distinct_digits"
    elif len(candidates) == 1 and len(cuis) == 1 and all(part["kind"] in {"digits", "umls_cui"} for part in parts):
        fmt = "pipe_cui_and_one_digit"
    elif len(candidates) == 1 and not cuis and all(part["kind"] == "digits" for part in parts):
        fmt = "pipe_repeated_same_digit"
    else:
        fmt = "pipe_other"
    cui = cuis[0] if len(cuis) == 1 else None
    return {"argument_format": fmt, "parts": parts, "candidate_gene_ids": candidates, "cui": cui}


def _split_ids(csv: str | None) -> list[str]:
    if not csv:
        return []
    out = []
    for part in str(csv).split(","):
        norm = norm_digits(part)
        if norm and norm not in out:
            out.append(norm)
    return out


def decide(
    argument: str | None,
    semtype: str | None,
    n_sentence_entity_rows: int | None,
    direct_supported: bool | None,
    cui_n_entity_rows: int | None,
    cui_gene_csv: str | None,
    is_generic: bool | None,
    path_a: bool,
    path_b: bool,
    path_c: bool,
) -> dict:
    parsed = parse_argument(argument)
    fmt = parsed["argument_format"]
    candidates = list(parsed["candidate_gene_ids"])
    cui_genes = _split_ids(cui_gene_csv)
    n_sentence = int(n_sentence_entity_rows or 0)
    n_cui_rows = int(cui_n_entity_rows or 0)
    result = {
        "argument_format": fmt,
        "parts": parsed["parts"],
        "candidate_gene_ids": candidates,
        "accepted_gene_id": None,
        "mapping_route": "none",
        "mapping_status": "unmapped_no_gene_id",
        "entity_check": "not_used",
        "rule_version": RULE_VERSION,
    }
    if fmt == "single_digits":
        if not semtype_is_genomic(semtype):
            result["mapping_status"] = "semtype_not_genomic"
            return result
        if not path_a:
            result["mapping_status"] = "rule_not_enabled"
            return result
        gene = candidates[0]
        if direct_supported:
            entity_check = "supports"
        elif n_sentence == 0:
            entity_check = "no_entity_row"
        else:
            entity_check = "no_matching_entity_gene"
        result.update(accepted_gene_id=gene, mapping_route="A_direct_entrez", mapping_status="accepted", entity_check=entity_check)
        return result
    if fmt == "single_umls_cui":
        if is_generic:
            result["mapping_status"] = "generic_concept_not_forced"
            result["candidate_gene_ids"] = cui_genes
            return result
        if n_cui_rows == 0:
            result["mapping_status"] = "no_entity_row"
            result["entity_check"] = "no_entity_row"
            return result
        if not cui_genes:
            result["mapping_status"] = "gene_field_empty"
            result["entity_check"] = "gene_field_empty"
            return result
        if len(cui_genes) > 1:
            result["candidate_gene_ids"] = cui_genes
            result["mapping_status"] = "ambiguous_multiple_gene_ids"
            result["entity_check"] = "multiple"
            return result
        if not path_b:
            result["candidate_gene_ids"] = cui_genes
            result["mapping_status"] = "rule_not_enabled"
            result["entity_check"] = "supports"
            return result
        result.update(
            candidate_gene_ids=cui_genes,
            accepted_gene_id=cui_genes[0],
            mapping_route="B_entity_gene_id",
            mapping_status="accepted",
            entity_check="supports",
        )
        return result
    if fmt == "pipe_multiple_distinct_digits":
        result["mapping_status"] = "ambiguous_multiple_gene_ids"
        return result
    if fmt in {"pipe_cui_and_one_digit", "pipe_repeated_same_digit"}:
        if not semtype_is_genomic(semtype):
            result["mapping_status"] = "semtype_not_genomic"
            return result
        digit = candidates[0]
        conflicts = any(gene != digit for gene in cui_genes)
        if conflicts:
            merged = []
            for gene in [digit, *cui_genes]:
                if gene not in merged:
                    merged.append(gene)
            result["candidate_gene_ids"] = merged
            result["mapping_status"] = "conflict"
            result["entity_check"] = "conflicts"
            return result
        routes = []
        if path_c:
            routes.append("C_composite_entrez")
        if cui_genes == [digit] and path_b:
            routes.append("B_entity_gene_id")
        if not routes:
            result["mapping_status"] = "rule_not_enabled"
            result["entity_check"] = _entity_label(n_cui_rows, cui_genes, direct_supported, fmt)
            return result
        result.update(
            accepted_gene_id=digit,
            mapping_route="+".join(routes),
            mapping_status="accepted",
            entity_check=_entity_label(n_cui_rows, cui_genes, direct_supported, fmt),
        )
        return result
    if fmt == "empty":
        result["mapping_status"] = "unmapped_no_gene_id"
    else:
        result["mapping_status"] = "other_format"
    return result


def _entity_label(n_cui_rows: int, cui_genes: list[str], direct_supported: bool | None, fmt: str) -> str:
    if fmt == "pipe_repeated_same_digit":
        if direct_supported:
            return "supports"
        return "no_entity_row" if n_cui_rows == 0 else "no_matching_entity_gene"
    if n_cui_rows == 0:
        return "no_entity_row"
    if not cui_genes:
        return "gene_field_empty"
    return "supports"


def decide_json(
    argument: str | None,
    semtype: str | None,
    n_sentence_entity_rows: int | None,
    direct_supported: bool | None,
    cui_n_entity_rows: int | None,
    cui_gene_csv: str | None,
    is_generic: bool | None,
    path_a: bool,
    path_b: bool,
    path_c: bool,
) -> str:
    return json.dumps(
        decide(
            argument,
            semtype,
            n_sentence_entity_rows,
            direct_supported,
            cui_n_entity_rows,
            cui_gene_csv,
            is_generic,
            bool(path_a),
            bool(path_b),
            bool(path_c),
        ),
        ensure_ascii=False,
    )


def name_agrees(observed_name: str | None, symbol: str | None, synonyms: str | None) -> bool:
    """Exact token agreement. This is a check, not a mapping rule."""
    if not observed_name or not symbol or len(symbol) < 2:
        return False
    observed_tokens = _tokens(observed_name)
    authority = {symbol.casefold()}
    if synonyms and synonyms not in {"-", ""}:
        authority.update(part.casefold() for part in synonyms.split("|") if len(part) >= 2)
    return any(token in authority for token in observed_tokens)


def _tokens(text: str) -> set[str]:
    tokens = set()
    for chunk in str(text).split("|"):
        piece = chunk.strip()
        if piece:
            tokens.add(piece.casefold())
        for word in piece.split():
            if word:
                tokens.add(word.casefold())
    return tokens


def slice_document(text: str, sentence_start: int, mention_start: int, mention_end: int, end_inclusive: bool) -> str | None:
    """Convert document offsets into the sentence string. One rule for every row."""
    if text is None:
        return None
    left = mention_start - sentence_start
    right = mention_end - sentence_start + (1 if end_inclusive else 0)
    if left < 0 or right < left or right > len(text):
        return None
    return text[left:right]

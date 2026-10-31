"""Assign SemMedDB column names from a bounded sample.

Documented dbinfo order is used only when the sample width matches and the
sampled values agree with that order. Extra or relocated fields stay unnamed
until a content rule identifies them. This module does not read the full files.
"""

from __future__ import annotations

import re
from statistics import median

from lib.schema import SEMMED_TABLES

_DIGITS = re.compile(r"^[0-9]+$")
_CUI = re.compile(r"^C[0-9]+$")
_SEMTYPE = re.compile(r"^[a-z]{4}$")
_SEMTYPE_FIELD = re.compile(r"^[a-z]{4}(,[a-z]{4})*$")
_PREDICATE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_TIMESTAMP = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}$")


def canonical_id_expr(column: str) -> str:
    """Same numeric-id rule as the SemMedDB load: blank and \\N are null, leading zeros drop."""
    return f"""CASE
      WHEN {column} IS NULL THEN NULL
      WHEN trim({column}) = '' OR trim({column}) = CHR(92) || 'N' OR trim({column}) IN ('NULL', 'null', 'None') THEN NULL
      WHEN regexp_matches(trim({column}), '^[0-9]+$') THEN
        CASE WHEN ltrim(trim({column}), '0') = '' THEN '0' ELSE ltrim(trim({column}), '0') END
      ELSE NULL
    END"""


def assign_local_columns(table: str, rows: list[list[str]]) -> dict:
    documented = list(SEMMED_TABLES[table]["columns"])
    if not rows:
        return {"status": "empty_sample", "documented_width": len(documented)}
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        return {"status": "unstable_width_inside_sample", "documented_width": len(documented)}
    if table == "SENTENCE":
        assigned = _sentence(rows)
    elif table == "PREDICATION":
        assigned = _predication(rows, documented)
    elif table == "ENTITY":
        assigned = _entity(rows)
    elif table == "PREDICATION_AUX":
        assigned = _documented_if_width_and_spotcheck(rows, documented, _aux_spotcheck)
    elif table == "CITATIONS":
        assigned = _documented_if_width_and_spotcheck(rows, documented, _citation_spotcheck)
    elif table == "GENERIC_CONCEPT":
        assigned = _documented_if_width_and_spotcheck(rows, documented, lambda rows_: len(rows_[0]) == 3)
    else:
        assigned = {"status": "no_rule"}
    assigned["documented_columns"] = documented
    assigned["documented_width"] = len(documented)
    assigned["observed_width"] = width
    assigned["width_matches_documented"] = width == len(documented)
    return assigned


def _documented_if_width_and_spotcheck(rows, documented, spotcheck) -> dict:
    if len(rows[0]) != len(documented) or not spotcheck(rows):
        return {"status": "documented_order_not_supported_by_sample"}
    return {
        "status": "assigned",
        "index": {name: i for i, name in enumerate(documented)},
        "unconfirmed_indexes": [],
        "evidence": "sample_width_matches_dbinfo_and_spotcheck",
    }


def _citation_spotcheck(rows) -> bool:
    return _frac(rows, 0, lambda v: bool(_DIGITS.match(v))) >= 0.99 and _frac(rows, 4, lambda v: bool(re.match(r"^[0-9]{4}$", v))) >= 0.9


def _aux_spotcheck(rows) -> bool:
    if len(rows[0]) != 18:
        return False
    return _frac(rows, 17, lambda v: bool(_TIMESTAMP.match(v))) >= 0.9 and _frac(rows, 2, lambda v: not _DIGITS.match(v)) >= 0.5


def _sentence(rows) -> dict:
    if _frac(rows, 0, lambda v: bool(_DIGITS.match(v))) < 0.99:
        return {"status": "sentence_id_column_not_numeric"}
    if _frac(rows, 1, lambda v: bool(_DIGITS.match(v))) < 0.99:
        return {"status": "pmid_column_not_numeric"}
    if _frac(rows, 2, lambda v: v in {"ti", "ab"}) < 0.9:
        return {"status": "type_column_not_ti_ab"}
    text_idx = _longest_text_column(rows)
    if text_idx is None:
        return {"status": "sentence_text_column_not_identified"}
    index = {"sentence_id": 0, "pmid": 1, "sent_type": 2, "sent_number": 3, "sentence_text": text_idx}
    span = _best_span(rows, text_idx, excluded=set(index.values()))
    evidence = ["c0_numeric_sentence_id", "c1_numeric_pmid", "c2_ti_ab", f"sentence_text_at_{text_idx}_by_median_length"]
    if span is None:
        return {"status": "sentence_span_columns_not_identified", "partial_index": index}
    index["sent_start_index"] = span["start"]
    index["sent_end_index"] = span["end"]
    evidence.append(f"span_c{span['start']}_c{span['end']}_agreement_{span['agreement']:.3f}_delta_{span['delta_mode']}")
    used = set(index.values())
    leftovers = [i for i in range(len(rows[0])) if i not in used]
    if len(leftovers) == 2:
        index["section_header"] = leftovers[0]
        index["normalized_section_header"] = leftovers[1]
        evidence.append("two_leftover_columns_named_as_section_headers_without_nonempty_content_check")
    else:
        return {
            "status": "sentence_section_columns_not_a_pair",
            "partial_index": index,
            "leftovers": leftovers,
            "evidence": evidence,
        }
    return {"status": "assigned", "index": index, "unconfirmed_indexes": [], "evidence": evidence, "span": span}


def _predication(rows, documented) -> dict:
    checks = {
        "c3_predicate": _frac(rows, 3, lambda v: bool(_PREDICATE.match(v))),
        "c4_subject_cui": _frac(rows, 4, lambda v: bool(_CUI.match(v))),
        "c6_subject_semtype": _frac(rows, 6, lambda v: bool(_SEMTYPE.match(v))),
        "c7_subject_novelty": _frac(rows, 7, lambda v: v in {"0", "1"}),
        "c8_object_cui": _frac(rows, 8, lambda v: bool(_CUI.match(v))),
        "c10_object_semtype": _frac(rows, 10, lambda v: bool(_SEMTYPE.match(v))),
        "c11_object_novelty": _frac(rows, 11, lambda v: v in {"0", "1"}),
    }
    if len(rows[0]) < 12 or any(v < 0.8 for v in checks.values()):
        return {"status": "predication_prefix_not_confirmed", "checks": checks}
    index = {name: i for i, name in enumerate(documented)}
    extra = list(range(len(documented), len(rows[0])))
    return {
        "status": "assigned",
        "index": index,
        "unconfirmed_indexes": extra,
        "evidence": "first_12_columns_match_dbinfo_value_patterns",
        "checks": checks,
    }


def _entity(rows) -> dict:
    width = len(rows[0])
    cui_i = _dominant_column(rows, lambda v: bool(_CUI.match(v)), min_nonempty=0.8, min_match=0.9)
    sem_i = _dominant_column(rows, lambda v: bool(_SEMTYPE_FIELD.match(v)), min_nonempty=0.8, min_match=0.95)
    if cui_i is None or sem_i is None:
        return {
            "status": "entity_cui_or_semtype_not_unique",
            "cui_column": cui_i,
            "semtype_column": sem_i,
            "column_fractions": _column_fractions(rows),
        }
    if sem_i != cui_i + 2:
        return {"status": "entity_name_column_not_between_cui_and_semtype", "cui": cui_i, "semtype": sem_i}
    name_i = cui_i + 1
    text_idx = _longest_text_column(rows, excluded={cui_i, sem_i, name_i}, min_median=3)
    if text_idx is None:
        return {
            "status": "entity_text_column_not_identified",
            "cui": cui_i,
            "semtype": sem_i,
            "text_scores": _text_scores(rows, {cui_i, sem_i, name_i}, 3),
        }
    span = _best_span(rows, text_idx, excluded={cui_i, sem_i, name_i, text_idx})
    if span is None:
        return {"status": "entity_span_columns_not_identified"}
    gene_i = sem_i + 1
    gene_name_i = sem_i + 2
    if gene_name_i >= width or gene_name_i == text_idx:
        return {"status": "entity_gene_columns_not_adjacent_to_semtype"}
    index = {
        "cui": cui_i,
        "name": cui_i + 1,
        "semtype": sem_i,
        "gene_id": gene_i,
        "gene_name": gene_name_i,
        "text": text_idx,
        "start_index": span["start"],
        "end_index": span["end"],
    }
    if _frac(rows, 0, lambda v: bool(_DIGITS.match(v))) < 0.99 or 0 in index.values():
        return {"status": "entity_id_column_not_confirmed"}
    index["entity_id"] = 0
    used = set(index.values())
    pre = [i for i in range(width) if i not in used and i < cui_i and _frac(rows, i, lambda v: v == "" or bool(_DIGITS.match(v))) >= 0.99]
    extras = [i for i in range(width) if i not in used and i not in pre]
    score_cols = [
        i
        for i in extras
        if _frac(rows, i, lambda v: bool(_DIGITS.match(v)) and 0 <= int(v) <= 1000) >= 0.9
    ]
    evidence = [
        f"cui_at_{cui_i}",
        f"semtype_at_{sem_i}",
        f"text_at_{text_idx}",
        f"span_{span['start']}_{span['end']}_agreement_{span['agreement']:.3f}",
        "gene_id_is_the_column_immediately_after_semtype",
        "pre_cui numeric columns are not named sentence_id or pmid until a sentence join",
    ]
    for n, i in enumerate(pre, start=1):
        index[f"pre_cui_numeric_{n}"] = i
    unconfirmed = [i for i in extras if i not in score_cols]
    if len(score_cols) != 1:
        return {"status": "entity_score_column_not_unique", "score_cols": score_cols, "partial_index": index}
    index["score"] = score_cols[0]
    return {
        "status": "assigned",
        "index": index,
        "unconfirmed_indexes": unconfirmed,
        "evidence": evidence,
        "span": span,
        "sentence_id_status": "pending_sentence_join",
    }


def _text_scores(rows, excluded: set[int], min_median: float) -> list[dict]:
    scored = []
    for i in range(len(rows[0])):
        if i in excluded:
            continue
        lengths = [len(row[i]) for row in rows]
        med = float(median(lengths)) if lengths else 0.0
        letters = _frac(rows, i, lambda v: any(ch.isalpha() for ch in v))
        if med >= min_median and letters >= 0.5:
            scored.append({"index": i, "median_len": med, "letter_rate": round(letters, 4)})
    scored.sort(key=lambda item: item["median_len"], reverse=True)
    return scored


def _longest_text_column(rows, excluded: set[int] | None = None, min_median: float = 12) -> int | None:
    scored = _text_scores(rows, excluded or set(), min_median)
    if not scored:
        return None
    if len(scored) > 1 and scored[0]["median_len"] < scored[1]["median_len"] * 1.5:
        return None
    return scored[0]["index"]


def _best_span(rows, text_idx: int, excluded: set[int]) -> dict | None:
    width = len(rows[0])
    int_cols = [
        i
        for i in range(width)
        if i not in excluded and i != text_idx and _frac(rows, i, lambda v: bool(_DIGITS.match(v))) >= 0.9
    ]
    ranked = []
    for start in int_cols:
        for end in int_cols:
            if start == end:
                continue
            used = 0
            ok = 0
            deltas = {}
            for row in rows:
                if not (_DIGITS.match(row[start]) and _DIGITS.match(row[end])):
                    continue
                used += 1
                delta = int(row[end]) - int(row[start]) - len(row[text_idx])
                deltas[delta] = deltas.get(delta, 0) + 1
                if delta in {-1, 0, 1}:
                    ok += 1
            if not used:
                continue
            agreement = ok / used
            if agreement < 0.8:
                continue
            mode = max(deltas, key=deltas.get)
            ranked.append({"start": start, "end": end, "agreement": agreement, "used": used, "delta_mode": mode})
    if not ranked:
        return None
    ranked.sort(key=lambda item: (-item["agreement"], item["start"], item["end"]))
    if len(ranked) > 1 and ranked[0]["agreement"] - ranked[1]["agreement"] < 0.2:
        return None
    return ranked[0]


def _dominant_column(rows, pred, min_nonempty: float, min_match: float) -> int | None:
    """Pick the column where non-empty values match pred, if one column is clearly ahead."""
    scored = []
    for i in range(len(rows[0])):
        values = [row[i] for row in rows]
        nonempty = [v for v in values if v != ""]
        nonempty_rate = len(nonempty) / len(values) if values else 0.0
        match = (sum(1 for v in nonempty if pred(v)) / len(nonempty)) if nonempty else 0.0
        if nonempty_rate >= min_nonempty and match >= min_match:
            scored.append((match, nonempty_rate, i))
    if len(scored) == 1:
        return scored[0][2]
    if len(scored) > 1:
        scored.sort(reverse=True)
        if scored[0][0] >= scored[1][0] + 0.2:
            return scored[0][2]
    return None


def _column_fractions(rows) -> list[dict]:
    out = []
    for i in range(len(rows[0])):
        values = [row[i] for row in rows]
        nonempty = [v for v in values if v != ""]
        examples = []
        for value in nonempty:
            if value not in examples:
                examples.append(value[:40])
            if len(examples) == 5:
                break
        out.append(
            {
                "index": i,
                "nonempty_rate": round(len(nonempty) / len(values), 4) if values else 0,
                "cui_match_among_nonempty": round(sum(1 for v in nonempty if _CUI.match(v)) / len(nonempty), 4) if nonempty else 0,
                "semtype_match_among_nonempty": round(sum(1 for v in nonempty if _SEMTYPE.match(v)) / len(nonempty), 4) if nonempty else 0,
                "examples": examples,
            }
        )
    return out


def _frac(rows, index: int, pred) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if pred(row[index])) / len(rows)

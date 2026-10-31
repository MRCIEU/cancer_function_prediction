"""PMID-set relations and coverage ratios.

The five overlap classes require both sets to be non-empty.
Empty sides are separate states and are not called disjoint or identical.
"""

from __future__ import annotations

FIVE = ("identical", "core_subset", "semmed_subset", "partial_overlap", "disjoint")


def classify_pmid_sets(core: set, semmed: set) -> str:
    if not core and not semmed:
        return "both_empty"
    if not core:
        return "core_pmids_empty"
    if not semmed:
        return "semmed_pmids_empty"
    if core == semmed:
        return "identical"
    if core < semmed:
        return "core_subset"
    if semmed < core:
        return "semmed_subset"
    if core & semmed:
        return "partial_overlap"
    return "disjoint"


def ratio(numerator: int, denominator: int, unit: str, filters: str, rule: str) -> dict:
    if denominator == 0:
        return {
            "numerator": numerator,
            "denominator": denominator,
            "value": None,
            "status": "not_applicable",
            "unit": unit,
            "filters": filters,
            "rule": rule,
        }
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator,
        "status": "computed",
        "unit": unit,
        "filters": filters,
        "rule": rule,
    }


def overlap_metrics(core: set, semmed: set, unit: str, filters: str, rule: str) -> dict:
    inter = core & semmed
    union = core | semmed
    c_only = core - semmed
    s_only = semmed - core
    return {
        "class": classify_pmid_sets(core, semmed),
        "n_core": len(core),
        "n_semmed": len(semmed),
        "n_intersection": len(inter),
        "n_union": len(union),
        "n_core_only": len(c_only),
        "n_semmed_only": len(s_only),
        "jaccard": ratio(len(inter), len(union), unit, filters, rule),
        "core_covered_by_semmed": ratio(len(inter), len(core), unit, filters, rule),
        "semmed_covered_by_core": ratio(len(inter), len(semmed), unit, filters, rule),
        "semmed_increment_over_core": ratio(len(s_only), len(core), unit, filters, rule),
        "core_increment_over_semmed": ratio(len(c_only), len(semmed), unit, filters, rule),
        "absolute_semmed_increment": len(s_only),
        "absolute_core_increment": len(c_only),
    }


def micro_coverage(rows: list[dict], rule: str, filters: str) -> dict:
    """Pair-PMID micro coverage. A PMID in two pairs is counted twice."""
    n_c = sum(r["n_core"] for r in rows)
    n_s = sum(r["n_semmed"] for r in rows)
    n_i = sum(r["n_intersection"] for r in rows)
    n_u = sum(r["n_union"] for r in rows)
    n_co = sum(r["n_core_only"] for r in rows)
    n_so = sum(r["n_semmed_only"] for r in rows)
    unit = "pair_pmid_micro"
    return {
        "weight": "pair_pmid_micro",
        "n_pairs": len(rows),
        "n_core": n_c,
        "n_semmed": n_s,
        "n_intersection": n_i,
        "n_union": n_u,
        "n_core_only": n_co,
        "n_semmed_only": n_so,
        "jaccard": ratio(n_i, n_u, unit, filters, rule),
        "core_covered_by_semmed": ratio(n_i, n_c, unit, filters, rule),
        "semmed_covered_by_core": ratio(n_i, n_s, unit, filters, rule),
        "semmed_increment_over_core": ratio(n_so, n_c, unit, filters, rule),
        "core_increment_over_semmed": ratio(n_co, n_s, unit, filters, rule),
    }


def global_pmid_coverage(core_union: set, semmed_union: set, rule: str, filters: str) -> dict:
    metrics = overlap_metrics(core_union, semmed_union, "deduplicated_pmid", filters, rule)
    metrics["weight"] = "deduplicated_pmid"
    return metrics


# First match wins. Raw atomic columns are retained beside this label.
CORE_ONLY_SUMMARY_ORDER = (
    "consistency_error_pair_predication_present",
    "pmid_absent_in_semmed_article_records",
    "pmid_present_sentence_absent",
    "sentence_present_gene_absent",
    "sentence_present_disease_absent",
    "entities_in_article_but_not_same_sentence",
    "same_sentence_entities_without_pair_predication",
    "not_assessable",
    "other_observed_pattern",
)

SEMMED_ONLY_SUMMARY_ORDER = (
    "consistency_error_core_pair_pmid_present",
    "pmid_absent_in_core_article_records",
    "pmid_present_sentence_absent",
    "sentence_present_gene_absent",
    "sentence_present_disease_absent",
    "entities_in_article_but_not_same_sentence",
    "same_sentence_entities_without_core_pair_record",
    "not_assessable",
    "other_observed_pattern",
)


def summarize_exclusive(flags: dict, order: tuple[str, ...]) -> str:
    for name in order:
        if flags.get(name) is True:
            return name
    if any(v == "not_assessable" for v in flags.values()):
        return "not_assessable"
    return "other_observed_pattern"

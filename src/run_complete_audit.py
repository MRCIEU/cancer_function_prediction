"""Complete the descriptive path-evidence audit.

The run reuses the fixed joint graph and the completed path-evidence tables.
It does not train a model or read a test set.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from src.common import StageTimer, ensure_dir, normalise_pmid, peak_rss_bytes, write_json
from src.complete_audit import (
    SOURCE_CATEGORIES,
    coverage_summary,
    fill_sequence_table,
    gain_category,
    intermediate_cancer_fraction,
    marginal_from_sequences,
    pair_dependence,
    path_increase_split,
    plain_adjacency,
    select_concentration_sample,
    sequence_group,
    summarise_sequences,
    three_edge_count_matrix,
    three_edge_source_sequences,
    two_edge_count_matrix,
    two_edge_source_sequences,
)
from src.mapping import informative_core_tuple
from src.path_count import exact_two_intermediate
from src.run_path_evidence import _as_bool, _connection

GRAPH_RUN = Path("/home/b5ac/freddieyu.b5ac/projects/project_2/results/joint_graph_and_evidence_audit_20261116T0404Z")
PRIOR_AUDIT = Path("/home/b5ac/freddieyu.b5ac/projects/project_2/results/path_evidence_audit_20261116T1310Z")
EXPECTED_PATHS_ONE = 18414593
EXPECTED_PATHS_TWO = 14437428485
EXPECTED_SHARED = 24160
EXPECTED_SEMMED_GAIN_PAIRS = 18782
EXPECTED_SEMMED_GAIN_LINKS = 69944
EXPECTED_SEMMED_GAIN_PMIDS = 61891
EXPECTED_SHARED_PMID_PAIRS = 9268
EXPECTED_SHARED_PMID_LINKS = 11120
EXPECTED_CORE_VIEW = 3343619438
EXPECTED_CORE_GAIN_RECORD = 1680974137
EXPECTED_TWO_SEMMED_ALONE = 12201666
EXPECTED_TWO_REQUIRES_BOTH = 6212927
EXPECTED_TWO_BOTH_EDGE = 2838820
EXPECTED_MEAN_ONE = 0.209133
EXPECTED_MEAN_TWO = 0.977243
CORE_COLOR = "#2171b5"
SEMMED_COLOR = "#ef6548"
SHARED_COLOR = "#737373"
JOINT_COLOR = "#6a51a3"
PMID_PATH_CAP = 100000


def _write_tsv(path: Path, rows: list[dict], columns: list[str]) -> None:
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join("" if row.get(column) is None else str(row.get(column)) for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_profiles(path: Path) -> list[dict]:
    columns = [
        "family",
        "endpoint_a",
        "endpoint_b",
        "support",
        "core_info",
        "sem_predicate",
        "n_core_sentences",
        "n_semmed_traversal",
        "n_semmed_predications",
        "n_pmid_core",
        "n_pmid_semmed",
        "n_pmid_union",
        "n_core_informative_tuples",
        "weight_t1_pos1",
        "weight_t1_pos2",
        "weight_t2_pos1",
        "weight_t2_pos2",
        "weight_t2_pos3",
    ]
    return pq.read_table(path, columns=columns).to_pylist()


def labelled_from_profiles(profiles: list[dict], mode: str) -> tuple[dict[str, dict[str, str]], dict[str, set[str]]]:
    """Build gene–cancer categories and undirected gene–gene neighbours.

    ``mode`` is joint, core, or semmed. Gene–gene edges exist only in the SemMedDB and joint views.
    """
    genes: dict[str, dict[str, str]] = defaultdict(dict)
    neighbors: dict[str, set[str]] = defaultdict(set)
    for row in profiles:
        support = row["support"]
        if support not in SOURCE_CATEGORIES:
            continue
        if row["family"] == "gene_cancer":
            if mode == "core" and support == "semmed_only":
                continue
            if mode == "semmed" and support == "core_only":
                continue
            genes[row["endpoint_b"]][row["endpoint_a"]] = support if mode == "joint" else (
                "both_sources" if support == "both_sources" else support
            )
        elif row["family"] == "gene_gene" and mode != "core":
            left, right = row["endpoint_a"], row["endpoint_b"]
            neighbors[left].add(right)
            neighbors[right].add(left)
    return genes, neighbors


def direct_mask(genes: dict[str, set[str]], candidates: list[str], queries: list[str]) -> np.ndarray:
    index = {gene: column for column, gene in enumerate(candidates)}
    mask = np.zeros((len(queries), len(candidates)), dtype=bool)
    for row, query in enumerate(queries):
        for gene in genes.get(query, ()):
            column = index.get(gene)
            if column is not None:
                mask[row, column] = True
    return mask


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    total = weights.sum()
    if total <= 0 or values.size == 0:
        return 0.0
    cumulative = np.cumsum(weights)
    position = quantile * total
    index = int(np.searchsorted(cumulative, position, side="left"))
    index = min(index, values.size - 1)
    return float(values[index])


def edge_position_summary(profiles: list[dict]) -> list[dict]:
    positions = {
        "weight_t1_pos1": ("one_intermediate_path", "position_1_gene_gene"),
        "weight_t1_pos2": ("one_intermediate_path", "position_2_gene_cancer"),
        "weight_t2_pos1": ("two_intermediate_path", "position_1_candidate_to_intermediate_cancer"),
        "weight_t2_pos2": ("two_intermediate_path", "position_2_intermediate_cancer_to_intermediate_gene"),
        "weight_t2_pos3": ("two_intermediate_path", "position_3_intermediate_gene_to_query_cancer"),
    }
    rows = []
    for column, (template, position) in positions.items():
        selected = [row for row in profiles if int(row[column]) > 0]
        weights = np.asarray([int(row[column]) for row in selected], dtype=np.int64)
        distinct = len(selected)
        occurrence = int(weights.sum()) if selected else 0
        support = Counter(row["support"] for row in selected)
        support_w = Counter()
        for row, weight in zip(selected, weights):
            support_w[row["support"]] += int(weight)
        for measure, value_name in (
            ("core_sentences", "n_core_sentences"),
            ("semmed_traversal_predications", "n_semmed_traversal"),
            ("pmid_union", "n_pmid_union"),
            ("pmid_core", "n_pmid_core"),
            ("pmid_semmed", "n_pmid_semmed"),
            ("core_informative_tuples", "n_core_informative_tuples"),
        ):
            values = np.asarray([int(row[value_name]) for row in selected], dtype=np.float64)
            scope = selected
            scoped_weights = weights
            if measure.startswith("core") or measure == "pmid_core":
                scope = [row for row in selected if row["family"] != "gene_gene"]
                scoped_weights = np.asarray([int(row[column]) for row in scope], dtype=np.int64)
                values = np.asarray([int(row[value_name]) for row in scope], dtype=np.float64)
            rows.append(
                {
                    "template": template,
                    "position": position,
                    "measure": measure,
                    "distinct_edges": distinct,
                    "occurrence_weight": occurrence,
                    "records_in_distribution": len(scope),
                    "occurrence_in_distribution": int(scoped_weights.sum()) if scope else 0,
                    "zero_distinct": int((values == 0).sum()) if values.size else 0,
                    "median_distinct": _weighted_quantile(values, np.ones(values.size), 0.5) if values.size else 0,
                    "q1_distinct": _weighted_quantile(values, np.ones(values.size), 0.25) if values.size else 0,
                    "q3_distinct": _weighted_quantile(values, np.ones(values.size), 0.75) if values.size else 0,
                    "p90_distinct": _weighted_quantile(values, np.ones(values.size), 0.9) if values.size else 0,
                    "median_occurrence": _weighted_quantile(values, scoped_weights.astype(np.float64), 0.5) if values.size else 0,
                    "q1_occurrence": _weighted_quantile(values, scoped_weights.astype(np.float64), 0.25) if values.size else 0,
                    "q3_occurrence": _weighted_quantile(values, scoped_weights.astype(np.float64), 0.75) if values.size else 0,
                    "p90_occurrence": _weighted_quantile(values, scoped_weights.astype(np.float64), 0.9) if values.size else 0,
                    "core_only_distinct": support["core_only"],
                    "both_sources_distinct": support["both_sources"],
                    "semmed_only_distinct": support["semmed_only"],
                    "core_only_occurrence": support_w["core_only"],
                    "both_sources_occurrence": support_w["both_sources"],
                    "semmed_only_occurrence": support_w["semmed_only"],
                }
            )
    return rows


def overlap_summary(path: Path) -> list[dict]:
    rows = pq.read_table(path).to_pylist()
    grouped = {}
    for row in rows:
        if row["support_left"] != "ALL" or row["overlap_band"]:
            continue
        key = (row["template"], row["positions"], row["comparison"], row["weighting"])
        item = grouped.setdefault(key, {"neither": 0, "one_sided": 0, "both_known": 0})
        item[row["availability"]] = int(row["count"])
    bands = {}
    for row in rows:
        if row["support_left"] != "ALL" or not row["overlap_band"]:
            continue
        key = (row["template"], row["positions"], row["comparison"], row["weighting"], row["overlap_band"])
        bands[key] = int(row["count"])
    out = []
    for key, item in sorted(grouped.items()):
        template, positions, comparison, weighting = key
        both = item["both_known"]
        zero = bands.get((*key, "0"), 0)
        shared = both - zero
        out.append(
            {
                "template": template,
                "positions": positions,
                "comparison": comparison,
                "weighting": weighting,
                "neither": item["neither"],
                "one_sided": item["one_sided"],
                "both_known": both,
                "zero_overlap": zero,
                "any_overlap": shared,
                "any_overlap_rate": shared / both if both else 0.0,
                "denominator_pairs": item["neither"] + item["one_sided"] + both,
            }
        )
    return out


def combination_rows(path: Path) -> list[dict]:
    return [
        row
        for row in pq.read_table(path).to_pylist()
        if row["table"] == "support_category"
    ]


def scan_pmids(graph_dir: Path, keep: set[tuple[str, str, str]]) -> dict[tuple[str, str, str], dict[str, set[str]]]:
    relations = {
        row["relation_id"]: row
        for row in pq.read_table(
            graph_dir / "relations.parquet",
            columns=["relation_id", "resource", "subject_entity_id", "object_entity_id", "predicate", "negation", "connectivity_family"],
        ).to_pylist()
    }
    links = defaultdict(list)
    for row in pq.read_table(graph_dir / "relation_evidence.parquet", columns=["relation_id", "evidence_id"]).to_pylist():
        links[row["evidence_id"]].append(row["relation_id"])
    found = {key: {"core": set(), "sem": set(), "sem_traversal": set()} for key in keep}
    columns = ["evidence_id", "resource", "pmid", "path_eligible", "negation"]
    evidence = pq.ParquetFile(graph_dir / "evidence_records.parquet")
    for batch in evidence.iter_batches(batch_size=200000, columns=columns):
        for row in batch.to_pylist():
            pmid = normalise_pmid(row.get("pmid")) or ""
            if not pmid:
                continue
            for relation_id in links.get(row["evidence_id"], ()):
                relation = relations.get(relation_id)
                if relation is None:
                    continue
                key = _connection(relation)
                if key not in found:
                    continue
                bucket = found[key]
                if row["resource"] == "core":
                    bucket["core"].add(pmid)
                else:
                    bucket["sem"].add(pmid)
                    negated = _as_bool(relation.get("negation")) or _as_bool(row.get("negation"))
                    if _as_bool(row.get("path_eligible")) and not negated:
                        bucket["sem_traversal"].add(pmid)
    return found


def shared_gain_tables(pair_rows: list[dict], pmid_sets: dict[tuple[str, str], dict[str, set[str]]]) -> tuple[list[dict], dict, dict]:
    categories = Counter()
    core_added = set()
    semmed_added = set()
    shared_pmids = set()
    summary = Counter()
    position_rows = []
    for row in pair_rows:
        gene, cancer = row["gene_id"], row["cancer_cui"]
        sets = pmid_sets.get((gene, cancer), {"core": set(), "sem": set()})
        core = sets["core"]
        semmed = sets["sem"]
        if len(core) != row["n_core_pmid"] or len(semmed) != row["n_semmed_pmid"] or len(core & semmed) != row["n_intersection"]:
            summary["pmid_identity_mismatches"] += 1
        category = gain_category(len(core - semmed), len(semmed - core))
        categories[category] += 1
        core_added |= core - semmed
        semmed_added |= semmed - core
        shared_pmids |= core & semmed
        summary["pairs"] += 1
        summary["core_added_links"] += len(core - semmed)
        summary["semmed_added_links"] += len(semmed - core)
        summary["shared_links"] += len(core & semmed)
        summary["pairs_core_adds"] += 1 if core - semmed else 0
        summary["pairs_semmed_adds"] += 1 if semmed - core else 0
        summary["pairs_shared_pmid"] += 1 if core & semmed else 0
        summary["pairs_missing_core_pmid"] += 1 if row["n_core_pmid"] == 0 else 0
        summary["pairs_missing_semmed_pmid"] += 1 if row["n_semmed_pmid"] == 0 else 0
        summary[f"core_added_band_{_band(len(core - semmed))}"] += 1
        summary[f"semmed_added_band_{_band(len(semmed - core))}"] += 1
    summary["distinct_core_added_pmids"] = len(core_added)
    summary["distinct_semmed_added_pmids"] = len(semmed_added)
    summary["distinct_shared_pmids"] = len(shared_pmids)
    summary.update(categories)
    positions = (
        ("one_intermediate_path", "position_2_gene_cancer", "weight_t1_pos2"),
        ("two_intermediate_path", "position_1_candidate_to_intermediate_cancer", "weight_t2_pos1"),
        ("two_intermediate_path", "position_2_intermediate_cancer_to_intermediate_gene", "weight_t2_pos2"),
        ("two_intermediate_path", "position_3_intermediate_gene_to_query_cancer", "weight_t2_pos3"),
    )
    for template, position, column in positions:
        total_weight = sum(int(row[column]) for row in pair_rows)
        core_weight = sum(int(row[column]) for row in pair_rows if row["n_core_only"] > 0)
        semmed_weight = sum(int(row[column]) for row in pair_rows if row["n_semmed_only"] > 0)
        position_rows.append(
            {
                "template": template,
                "position": position,
                "occurrence_weight": total_weight,
                "occurrence_core_adds_pmid": core_weight,
                "fraction_core_adds_pmid": core_weight / total_weight if total_weight else 0.0,
                "occurrence_semmed_adds_pmid": semmed_weight,
                "fraction_semmed_adds_pmid": semmed_weight / total_weight if total_weight else 0.0,
                "weighted_core_added_links": sum(int(row["n_core_only"]) * int(row[column]) for row in pair_rows),
                "weighted_semmed_added_links": sum(int(row["n_semmed_only"]) * int(row[column]) for row in pair_rows),
            }
        )
    return position_rows, summary, {"categories": categories}


def _band(value: int) -> str:
    if value <= 0:
        return "0"
    if value == 1:
        return "1"
    if value <= 5:
        return "2-5"
    if value <= 10:
        return "6-10"
    if value <= 25:
        return "11-25"
    return "26+"


def field_summary(detail_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    link_fields = Counter()
    pair_fields = defaultdict(set)
    combinations = Counter()
    pair_combinations = defaultdict(set)
    informative_links = 0
    missing_links = 0
    for row in detail_rows:
        pair = (row["gene_id"], row["cancer_cui"])
        core_parts = [part for part in row["core_attributes"].split(" || ") if part]
        traversal = [part for part in row["semmed_traversal_details"].split(" || ") if part]
        negated = [part for part in row["semmed_negated_details"].split(" || ") if part]
        other = [part for part in row["semmed_other_details"].split(" || ") if part]
        combo = (row["core_attributes"], row["semmed_traversal_details"], row["semmed_negated_details"], row["semmed_other_details"])
        combinations[combo] += 1
        pair_combinations[combo].add(pair)
        link_has_informative = False
        link_has_missing = False
        for part in core_parts:
            fields = part.split("|")
            if len(fields) != 5:
                continue
            names = ("cge", "ccs", "gci", "gcc", "has_type")
            for name, value in zip(names, fields):
                label = value if value else "missing"
                if value in {"", "NOTINF"}:
                    label = "missing_or_NOTINF"
                    link_has_missing = True
                link_fields[("core", name, label)] += 1
                pair_fields[("core", name, label)].add(pair)
            if informative_core_tuple(*fields):
                link_has_informative = True
        if link_has_informative:
            informative_links += 1
        if link_has_missing or not core_parts:
            missing_links += 1
        for kind, parts in (("traversal", traversal), ("negated", negated), ("other", other)):
            if parts:
                link_fields[("semmed_record_class", kind, "present")] += 1
                pair_fields[("semmed_record_class", kind, "present")].add(pair)
            for part in parts:
                predicate, subject, _object, negation = (part.split("|") + ["", "", "", ""])[:4]
                orientation = "subject_is_gene" if subject.startswith("gene:") else "subject_is_cancer" if subject.startswith("cancer:") else "other_orientation"
                link_fields[("semmed", "predicate", predicate or "missing")] += 1
                link_fields[("semmed", "orientation", orientation)] += 1
                link_fields[("semmed", "negation", negation or "missing")] += 1
                pair_fields[("semmed", "predicate", predicate or "missing")].add(pair)
                pair_fields[("semmed", "orientation", orientation)].add(pair)
                pair_fields[("semmed", "negation", negation or "missing")].add(pair)
    field_rows = [
        {
            "resource": key[0],
            "field": key[1],
            "value": key[2],
            "pair_pmid_links": link_fields[key],
            "distinct_pairs": len(pair_fields[key]),
            "link_denominator": len(detail_rows),
            "multi_label": "yes",
        }
        for key in sorted(link_fields)
    ]
    combo_rows = [
        {
            "core_attributes": key[0],
            "semmed_traversal_details": key[1],
            "semmed_negated_details": key[2],
            "semmed_other_details": key[3],
            "pair_pmid_links": value,
            "distinct_pairs": len(pair_combinations[key]),
            "link_denominator": len(detail_rows),
            "multi_label": "no_within_link_after_deduplication",
        }
        for key, value in combinations.most_common(30)
    ]
    field_rows.append(
        {
            "resource": "core",
            "field": "informative_tuple",
            "value": "any_informative_tuple_on_link",
            "pair_pmid_links": informative_links,
            "distinct_pairs": "",
            "link_denominator": len(detail_rows),
            "multi_label": "link_level",
        }
    )
    field_rows.append(
        {
            "resource": "core",
            "field": "missing_attribute",
            "value": "any_missing_or_NOTINF_on_link",
            "pair_pmid_links": missing_links,
            "distinct_pairs": "",
            "link_denominator": len(detail_rows),
            "multi_label": "link_level",
        }
    )
    return field_rows, combo_rows


def concentration_for_sample(sample, counts, max_cancer, genes, pmids, candidates, queries, cap: int) -> list[dict]:
    cand_of = {gene: {cancer for cancer, members in genes.items() if gene in members} for gene in candidates}
    rows = []
    for query_row, column, band, status in sample:
        query = queries[query_row]
        candidate = candidates[column]
        total = int(counts[query_row, column])
        cancer_fraction = float(max_cancer[query_row, column] / total) if total else 0.0
        gene_counts = Counter()
        pmid_counts = Counter()
        computed_pmid = total <= cap
        query_genes = genes.get(query, set())
        for gene in query_genes:
            if gene == candidate:
                continue
            for cancer in cand_of.get(candidate, ()):
                if cancer == query or gene not in genes.get(cancer, ()):
                    continue
                gene_counts[gene] += 1
        if computed_pmid:
            for cancer in cand_of.get(candidate, ()):
                if cancer == query:
                    continue
                both = genes.get(cancer, set()) & query_genes
                both.discard(candidate)
                if not both:
                    continue
                left = pmids.get(("gene_cancer", candidate, cancer), set())
                for gene in both:
                    touched = set(left)
                    touched |= pmids.get(("gene_cancer", gene, cancer), set())
                    touched |= pmids.get(("gene_cancer", gene, query), set())
                    for pmid in touched:
                        pmid_counts[pmid] += 1
        max_gene = max(gene_counts.values()) if gene_counts else 0
        max_pmid = max(pmid_counts.values()) if pmid_counts else 0
        rows.append(
            {
                "query_cui": query,
                "gene_id": candidate,
                "path_count_band": band,
                "direct_status": status,
                "complete_paths": total,
                "distinct_intermediate_genes": len(gene_counts),
                "max_intermediate_gene_fraction": max_gene / total if total else 0.0,
                "max_intermediate_cancer_fraction": cancer_fraction,
                "distinct_pmids": len(pmid_counts) if computed_pmid else "",
                "max_pmid_fraction": (max_pmid / total if total else 0.0) if computed_pmid else "",
                "pmid_scope": "exact" if computed_pmid else f"not_computed_above_{cap}_paths",
            }
        )
    return rows


def two_edge_pmid_concentration(genes, neighbors, pmids, candidates, queries) -> dict:
    """Exact PMID concentration. Each two-edge path is one intermediate gene."""
    fractions = []
    distinct_counts = []
    cand_set = set(candidates)
    for query in queries:
        per_candidate = defaultdict(list)
        for gene in genes.get(query, ()):
            edge_pmids = pmids.get(("gene_cancer", gene, query), set())
            for partner in neighbors.get(gene, ()):
                if partner not in cand_set or partner == gene:
                    continue
                touched = set(edge_pmids)
                left, right = (partner, gene) if partner <= gene else (gene, partner)
                touched |= pmids.get(("gene_gene", left, right), set())
                per_candidate[partner].append(touched)
        for sets in per_candidate.values():
            total = len(sets)
            counter = Counter()
            for touched in sets:
                for pmid in touched:
                    counter[pmid] += 1
            distinct_counts.append(len(counter))
            fractions.append((max(counter.values()) / total) if counter and total else 0.0)
    values = np.asarray(fractions, dtype=np.float64)
    distinct = np.asarray(distinct_counts, dtype=np.float64)
    return {
        "reached_pairs": len(fractions),
        "median_max_pmid_fraction": float(np.median(values)) if values.size else 0.0,
        "median_distinct_pmids": float(np.median(distinct)) if distinct.size else 0.0,
        "fraction_at_least_0_5": int((values >= 0.5).sum()) if values.size else 0,
    }


def render_report(summary: dict) -> str:
    gain = summary["gain"]
    groups = summary["three_groups"]
    two = summary["two_groups"]
    cover = summary["coverage"]
    return f"""# CORE and SemMedDB path evidence

This note completes the descriptive audit of the fixed joint graph. A shared PMID is shared article provenance. Equal CORE attributes are agreement of recorded fields. Neither comparison is evidence of the same experiment.

## What each resource adds on the 24,160 shared pairs

The fixed universe contains {gain['pairs']} gene–cancer pairs recorded by both resources. CORE adds at least one PMID on {gain['pairs_core_adds']} pairs ({gain['pairs_core_adds'] / gain['pairs']:.3f}). SemMedDB adds at least one PMID on {gain['pairs_semmed_adds']} pairs ({gain['pairs_semmed_adds'] / gain['pairs']:.3f}). At least one PMID is already on both sides for {gain['pairs_shared_pmid']} pairs.

The four gain categories, each pair counted once, are: only CORE adds PMIDs {gain['only_core_adds_pmids']}; only SemMedDB adds PMIDs {gain['only_semmed_adds_pmids']}; both add PMIDs {gain['both_add_pmids']}; neither adds PMIDs {gain['neither_adds_pmids']}.

Added pair–PMID links are {gain['core_added_links']} from CORE and {gain['semmed_added_links']} from SemMedDB. Those links represent {gain['distinct_core_added_pmids']} distinct PMIDs on the CORE side and {gain['distinct_semmed_added_pmids']} distinct PMIDs on the SemMedDB side. A PMID can be exclusive for one pair and exclusive in the other direction for a different pair, so the two distinct totals are separate. Shared pair–PMID links: {gain['shared_links']}, from {gain['distinct_shared_pmids']} distinct PMIDs. Pairs with no known CORE PMID: {gain['pairs_missing_core_pmid']}. Pairs with no known SemMedDB PMID: {gain['pairs_missing_semmed_pmid']}. These missing-coverage counts are separate from pairs whose two observed PMID sets are disjoint.

On the {gain['shared_links']} shared links, CORE sentence attributes and SemMedDB predicate, subject/object orientation, and negation are summarised in `analysis/shared_article_field_summary.tsv`. Field rows are multi-label. Combination rows count each link once after the field values on that link have been deduplicated. Occurrence-weighted gains for these same pairs are in `analysis/shared_pair_gain_by_position.tsv`. Weighted pair–PMID contributions multiply an added-link count by a path occurrence weight; they are not unique publication totals.

## When a three-edge path needs both resources

The three-edge table contains {groups['complete_paths']} complete paths. Only CORE can traverse {groups['only_core']} ({groups['only_core'] / groups['complete_paths']:.3f}). Only SemMedDB can traverse {groups['only_semmed']} ({groups['only_semmed'] / groups['complete_paths']:.3f}). Either resource can traverse {groups['either_resource']} ({groups['either_resource'] / groups['complete_paths']:.3f}). The complete path requires both resources for {groups['requires_both_resources']} ({groups['requires_both_resources'] / groups['complete_paths']:.3f}).

The CORE view contains {groups['core_view']} paths. The SemMedDB view contains {groups['semmed_view']} paths. SemMedDB adds {groups['paths_added_by_semmed']} paths relative to CORE. Of those, {groups['only_semmed']} can be traversed by SemMedDB alone and {groups['requires_both_resources']} require both resources. Existing CORE paths that gain a SemMedDB record on at least one edge: {groups['core_paths_gaining_semmed_record']}.

The two-edge template has no CORE gene–gene edge, so a complete two-edge path is never available in CORE alone. SemMedDB alone can traverse {two['semmed_alone']} paths, of which {two['semmed_also_core_record']} also have a CORE record on the gene–cancer edge. The other {two['requires_both']} paths require both resources.

## Candidate pairs reached by those paths

The candidate–query universe has {cover['universe']} pairs. Two-edge paths reach {cover['two_joint_reached']} pairs in the joint graph (mean coverage {cover['two_joint_mean']:.3f} across cancer queries) and {cover['two_semmed_reached']} in the SemMedDB view. The CORE view reaches {cover['two_core_reached']} because the gene–gene edge is outside CORE. Relative to SemMedDB, the joint graph adds {cover['two_added_vs_semmed']} pairs.

Three-edge paths reach {cover['three_joint_reached']} pairs in the joint graph (mean coverage {cover['three_joint_mean']:.3f}), {cover['three_core_reached']} in the CORE view, and {cover['three_semmed_reached']} in the SemMedDB view. The joint graph adds {cover['three_added_vs_core']} pairs relative to CORE and {cover['three_added_vs_semmed']} pairs relative to SemMedDB.

Among joint-reachable pairs with no eligible direct record, the four reachability categories are in `analysis/candidate_coverage.tsv`. Path totals split into paths that reach a new pair and paths added to an already reachable pair are in `analysis/path_increase_split.tsv`.

Across all reached three-edge pairs, the median fraction of paths passing through the most-used intermediate cancer is {cover['three_cancer_fraction_median']:.3f}. Two-edge PMID concentration is exact. Three-edge intermediate-gene and PMID concentration uses one deterministic pair per query, path-count band, and direct-record status. Pairs above {PMID_PATH_CAP} paths keep the exact cancer-concentration fraction and leave the PMID fraction unmarked. Sample size and capped pairs: {summary['sample_size']} selected, {summary['sample_capped']} above the path cap.

## Edge and adjacent-edge evidence

Distinct-edge and occurrence-weighted summaries of sentence, predication, and PMID counts are in `analysis/edge_position_summary.tsv`. Medians and quartiles use the exact counts stored on each edge. CORE sentence summaries at the gene–gene position are outside CORE's scope and are not treated as zero.

Adjacent source combinations and PMID overlap, including one-sided coverage and zero overlap of two observed PMID sets, are in `analysis/adjacent_overlap_summary.tsv`. The three-edge position 2–3 overlap rate stays higher than position 1–2 under both distinct and occurrence-weighted counting. The stored CORE comparison at positions 1–2 remains comparable attributes, different attributes, or insufficient information. That table does not identify whether a difference is in expression, cancer status, or relation nature.

## Method

Three-edge source sequences multiply, at each eligible middle edge, the number of eligible initial edges of each source category by the number of eligible final edges of each source category. Candidate coverage uses bitsets over genes and stores one path count per candidate–query pair. PMID membership for the shared pairs was read again from the graph and checked against the stored counts. Figures use the summary tables in this directory.
"""


def _save(fig, figures: Path, name: str) -> None:
    fig.savefig(figures / f"{name}.png", dpi=160, bbox_inches="tight")
    fig.savefig(figures / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_figures(analysis: Path, figures: Path, summary: dict) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    gain = summary["gain"]
    direct = summary["direct"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    panel_a = [
        ("CORE only", direct["core_only"], CORE_COLOR),
        ("Shared", direct["shared"], SHARED_COLOR),
        ("SemMedDB only", direct["semmed_only"], SEMMED_COLOR),
    ]
    panel_b = [
        ("CORE added", gain["core_added_links"], CORE_COLOR),
        ("Shared PMID", gain["shared_links"], SHARED_COLOR),
        ("SemMedDB added", gain["semmed_added_links"], SEMMED_COLOR),
    ]
    for axis, rows, title, denominator in (
        (axes[0], panel_a, "A  Direct pairs", direct["joint"]),
        (axes[1], panel_b, "B  Pair–PMID links on shared pairs", gain["pairs"]),
    ):
        labels = [row[0] for row in rows][::-1]
        values = [row[1] for row in rows][::-1]
        colors = [row[2] for row in rows][::-1]
        axis.barh(labels, values, color=colors)
        for index, value in enumerate(values):
            axis.text(value, index, f" {value:,}", va="center", fontsize=8)
        axis.set_title(f"{title}\nDenominator {denominator:,}")
        axis.set_xlabel("Count")
        axis.set_facecolor("white")
    fig.suptitle("Publications and pairs added by each resource", fontsize=12)
    fig.tight_layout()
    _save(fig, figures, "resource_additions")
    _write_tsv(
        analysis / "plot_resource_additions.tsv",
        [{"panel": "direct_pairs", "category": name, "count": count} for name, count, _ in panel_a]
        + [{"panel": "shared_pair_pmid_links", "category": name, "count": count} for name, count, _ in panel_b],
        ["panel", "category", "count"],
    )

    groups = summary["three_groups"]
    two = summary["two_groups"]
    no_direct = summary["no_direct_categories"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    path_rows = {
        "Two-edge paths": [two["only_core"], two["only_semmed"], two["either_resource"], two["requires_both"]],
        "Three-edge paths": [groups["only_core"], groups["only_semmed"], groups["either_resource"], groups["requires_both_resources"]],
    }
    pair_rows = {
        "Two-edge pairs": no_direct["two"],
        "Three-edge pairs": no_direct["three"],
    }
    colors = [CORE_COLOR, SEMMED_COLOR, SHARED_COLOR, JOINT_COLOR]
    names = ["Only CORE", "Only SemMedDB", "Either resource", "Joint graph required"]
    for axis, block, title in (
        (axes[0], path_rows, "A  Complete paths"),
        (axes[1], pair_rows, "B  Reachable pairs with no direct edge"),
    ):
        bottoms = np.zeros(len(block))
        labels = list(block)
        for index, (name, color) in enumerate(zip(names, colors)):
            values = np.asarray([block[label][index] / sum(block[label]) if sum(block[label]) else 0 for label in labels])
            axis.bar(labels, values, bottom=bottoms, color=color, label=name)
            bottoms += values
        for index, label in enumerate(labels):
            axis.text(index, 1.02, f"n={sum(block[label]):,}", ha="center", va="bottom", fontsize=7)
        axis.set_ylim(0, 1.18)
        axis.set_ylabel("Share of the bar total")
        axis.set_title(title)
        axis.set_facecolor("white")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Source dependence of paths and of pairs with no direct record", fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, figures, "joint_graph_necessity")

    combos = summary["combination_plot"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
    order = SOURCE_CATEGORIES
    for column, (template, positions, title) in enumerate(
        (
            ("one_intermediate_path", "positions_1_2", "Two-edge 1–2"),
            ("two_intermediate_path", "positions_1_2", "Three-edge 1–2"),
            ("two_intermediate_path", "positions_2_3", "Three-edge 2–3"),
        )
    ):
        for row_index, weighting in enumerate(("distinct_count", "path_occurrence_count")):
            matrix = np.zeros((3, 3))
            for item in combos:
                if item["template"] == template and item["positions"] == positions and item["weighting"] == weighting:
                    matrix[order.index(item["left"]), order.index(item["right"])] = item["percent"]
            axis = axes[row_index, column]
            image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=summary["heatmap_max"])
            axis.set_xticks(range(3), ["CORE", "Both", "SemMed"], fontsize=7)
            axis.set_yticks(range(3), ["CORE", "Both", "SemMed"], fontsize=7)
            axis.set_title(f"{title}\n{weighting.replace('_', ' ')}", fontsize=9)
            for i in range(3):
                for j in range(3):
                    axis.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=axes, fraction=0.03, pad=0.02, label="Percent of denominator")
    fig.suptitle("Resource combination on adjacent edges", fontsize=12)
    _save(fig, figures, "adjacent_resource_combinations")

    overlap = [row for row in summary["overlap_plot"] if row["comparison"] == "union"]
    fig, axis = plt.subplots(figsize=(8.4, 4.2))
    labels = []
    for index, row in enumerate(overlap):
        if row["weighting"] != "distinct_count":
            continue
        labels.append(f"{row['template'].replace('_', ' ')}\n{row['positions']}")
    distinct = [row["any_overlap_rate"] * 100 for row in overlap if row["weighting"] == "distinct_count"]
    weighted = [row["any_overlap_rate"] * 100 for row in overlap if row["weighting"] == "path_occurrence_count"]
    positions = np.arange(len(labels))
    axis.plot(positions - 0.05, distinct, "o", color=SHARED_COLOR, label="Distinct adjacent pairs")
    axis.plot(positions + 0.05, weighted, "o", color=JOINT_COLOR, label="Occurrence in complete paths")
    for index, (left, right) in enumerate(zip(distinct, weighted)):
        axis.text(index - 0.05, left, f" {left:.2f}%", fontsize=7, va="bottom")
        axis.text(index + 0.05, right, f" {right:.2f}%", fontsize=7, va="bottom")
    axis.set_xticks(positions, labels, fontsize=8)
    axis.set_ylabel("Percent with at least one shared PMID")
    axis.set_title("PMID overlap among adjacent edges with known PMIDs on both sides")
    axis.legend(frameon=False)
    axis.set_facecolor("white")
    fig.tight_layout()
    _save(fig, figures, "adjacent_pmid_overlap")


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete the CORE–SemMedDB path evidence audit.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--graph-run", type=Path, default=GRAPH_RUN)
    parser.add_argument("--prior-audit", type=Path, default=PRIOR_AUDIT)
    parser.add_argument("--sample-queries", type=int, default=0, help="0 means every query. A positive value keeps the highest-degree queries plus the smallest ones.")
    args = parser.parse_args()
    timer = StageTimer()
    run_dir = ensure_dir(args.run_dir)
    analysis = ensure_dir(run_dir / "analysis")
    figures = ensure_dir(run_dir / "figures")
    ensure_dir(run_dir / "logs")
    prior_manifest = args.prior_audit / "manifest.json"
    profiles = load_profiles(args.prior_audit / "analysis" / "edge_profiles.parquet")
    candidates = sorted(
        row["gene_id"]
        for row in pq.read_table(args.graph_run / "analysis" / "candidate_genes.parquet", columns=["gene_id", "status"]).to_pylist()
        if row["status"] == "resolved_human"
    )
    queries = sorted(row["cancer_cui"] for row in pq.read_table(args.graph_run / "analysis" / "cancer_queries.parquet", columns=["cancer_cui"]).to_pylist())
    candidate_set = set(candidates)
    query_set = set(queries)
    timer.mark("load_profiles", profiles=len(profiles), candidates=len(candidates), queries=len(queries))

    labelled = {}
    neighbors = {}
    for mode in ("joint", "core", "semmed"):
        labelled[mode], neighbors[mode] = labelled_from_profiles(profiles, mode)
    sequences = three_edge_source_sequences(labelled["joint"], candidate_set, query_set)
    sequence_rows = fill_sequence_table(sequences)
    three_groups = summarise_sequences(sequence_rows)
    two_sequences = two_edge_source_sequences(labelled["joint"], neighbors["joint"], candidate_set, query_set)
    two_groups = {
        "only_core": 0,
        "only_semmed": int(two_sequences[("semmed_only", "semmed_only")] + two_sequences[("semmed_only", "both_sources")]),
        "either_resource": 0,
        "requires_both": int(two_sequences[("semmed_only", "core_only")]),
        "semmed_alone": int(two_sequences[("semmed_only", "semmed_only")] + two_sequences[("semmed_only", "both_sources")]),
        "semmed_also_core_record": int(two_sequences[("semmed_only", "both_sources")]),
        "complete_paths": int(sum(two_sequences.values())),
    }
    bitset_total = sum(
        row["n_paths"]
        for row in exact_two_intermediate(plain_adjacency(labelled["joint"]), candidate_set, queries, sorted(set(labelled["joint"]) | query_set)).values()
    )
    timer.mark("source_sequences", three=three_groups["complete_paths"], two=two_groups["complete_paths"], bitset=int(bitset_total))

    coverage = {}
    matrices = {}
    max_cancers = {}
    directs = {}
    for mode in ("joint", "core", "semmed"):
        plain = plain_adjacency(labelled[mode])
        counts, max_cancer, _n_cancers, total = three_edge_count_matrix(plain, candidates, queries)
        direct = direct_mask(plain, candidates, queries)
        matrices[mode] = counts
        max_cancers[mode] = max_cancer
        directs[mode] = direct
        coverage[f"three_{mode}"] = coverage_summary(counts, direct)
        coverage[f"three_{mode}"]["complete_paths_check"] = total
        two_counts = two_edge_count_matrix(plain, neighbors[mode], candidates, queries)
        coverage[f"two_{mode}"] = coverage_summary(two_counts, direct)
        if mode != "joint":
            del two_counts
        else:
            matrices["two_joint"] = two_counts
    timer.mark("coverage", three_joint=coverage["three_joint"]["complete_paths"], two_joint=coverage["two_joint"]["complete_paths"])

    published = {}
    with (args.graph_run / "analysis" / "path_coverage_by_query.tsv").open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        for line in handle:
            values = dict(zip(header, line.rstrip("\n").split("\t")))
            published[values["query_cui"]] = values
    query_mismatches = 0
    per_query_rows = []
    for index, query in enumerate(queries):
        observed_paths = int(matrices["joint"][index].sum())
        observed_reached = int((matrices["joint"][index] > 0).sum())
        source = published.get(query, {})
        if int(source.get("n_paths_two_intermediate", -1)) != observed_paths or int(source.get("n_reached_two_intermediate", -1)) != observed_reached:
            query_mismatches += 1
        per_query_rows.append(
            {
                "query_cui": query,
                "candidates": len(candidates),
                "two_edge_joint_reached": int((matrices["two_joint"][index] > 0).sum()),
                "two_edge_semmed_reached": int((coverage["two_semmed"]["per_query_reached"][index])),
                "three_edge_joint_reached": observed_reached,
                "three_edge_core_reached": int((matrices["core"][index] > 0).sum()),
                "three_edge_semmed_reached": int((matrices["semmed"][index] > 0).sum()),
                "three_edge_joint_paths": observed_paths,
                "three_edge_core_paths": int(matrices["core"][index].sum()),
                "three_edge_semmed_paths": int(matrices["semmed"][index].sum()),
                "direct_joint": int(directs["joint"][index].sum()),
            }
        )
    no_direct = ~directs["joint"]
    two_semmed = two_edge_count_matrix(plain_adjacency(labelled["semmed"]), neighbors["semmed"], candidates, queries)
    dependence = {
        "three_all": pair_dependence(matrices["core"], matrices["semmed"], matrices["joint"], np.ones_like(directs["joint"], dtype=bool)),
        "three_no_direct": pair_dependence(matrices["core"], matrices["semmed"], matrices["joint"], no_direct),
    }
    dependence["two_all"] = pair_dependence(np.zeros_like(two_semmed), two_semmed, matrices["two_joint"], np.ones_like(two_semmed, dtype=bool))
    dependence["two_no_direct"] = pair_dependence(np.zeros_like(two_semmed), two_semmed, matrices["two_joint"], no_direct)
    increase = {
        "three_vs_core": path_increase_split(matrices["core"], matrices["joint"]),
        "three_vs_semmed": path_increase_split(matrices["semmed"], matrices["joint"]),
        "two_vs_semmed": path_increase_split(two_semmed, matrices["two_joint"]),
        "two_vs_core": path_increase_split(np.zeros_like(matrices["two_joint"]), matrices["two_joint"]),
    }
    cancer_fraction = intermediate_cancer_fraction(matrices["joint"], max_cancers["joint"])
    sample = select_concentration_sample(matrices["joint"], directs["joint"], candidates)
    if args.sample_queries > 0:
        degree = sorted(((len(labelled["joint"][query]), query) for query in queries if query in labelled["joint"]), reverse=True)
        keep_queries = {query for _degree, query in degree[: args.sample_queries]}
        keep_queries.update(query for _degree, query in degree[-min(2, len(degree)) :])
        sample = [item for item in sample if queries[item[0]] in keep_queries]
    timer.mark("coverage_checks", query_mismatches=query_mismatches, sample=len(sample))

    position_summary = edge_position_summary(profiles)
    overlap_rows = overlap_summary(args.prior_audit / "analysis" / "adjacent_article_overlap.parquet")
    combo_rows = combination_rows(args.prior_audit / "analysis" / "adjacent_edge_combinations.parquet")
    sequence_marginal = marginal_from_sequences(sequence_rows, "position_1", "position_2")
    marginal_mismatches = []
    for row in combo_rows:
        if row["template"] == "two_intermediate_path" and row["positions"] == "positions_1_2" and row["weighting"] == "path_occurrence_count":
            observed = sequence_marginal[(row["left"], row["right"])]
            if observed != int(row["count"]):
                marginal_mismatches.append({"left": row["left"], "right": row["right"], "sequence": observed, "table": int(row["count"])})
    timer.mark("existing_tables", marginal_mismatches=len(marginal_mismatches))

    shared_rows = pq.read_table(args.prior_audit / "analysis" / "shared_pair_pmids.parquet").to_pylist()
    detail_rows = pq.read_table(args.prior_audit / "analysis" / "shared_article_relation_details.parquet").to_pylist()
    keep = set()
    for row in profiles:
        if row["support"] in SOURCE_CATEGORIES:
            keep.add((row["family"], row["endpoint_a"], row["endpoint_b"]))
    pmid_buckets = scan_pmids(args.graph_run / "graph", keep)
    pair_sets = {(gene, cancer): pmid_buckets.get(("gene_cancer", gene, cancer), {"core": set(), "sem": set(), "sem_traversal": set()}) for gene, cancer in ((row["gene_id"], row["cancer_cui"]) for row in shared_rows)}
    gain_positions, gain_summary, _categories = shared_gain_tables(shared_rows, pair_sets)
    field_rows, combo_field_rows = field_summary(detail_rows)
    qualifying_pmids = {}
    for key, bucket in pmid_buckets.items():
        qualifying_pmids[key] = set(bucket["core"]) | set(bucket["sem_traversal"])
    sample_rows = concentration_for_sample(
        sample,
        matrices["joint"],
        max_cancers["joint"],
        plain_adjacency(labelled["joint"]),
        qualifying_pmids,
        candidates,
        queries,
        PMID_PATH_CAP,
    )
    two_pmid = two_edge_pmid_concentration(
        plain_adjacency(labelled["joint"]),
        neighbors["joint"],
        qualifying_pmids,
        candidates,
        queries,
    )
    timer.mark("pmid_and_concentration", shared_mismatches=gain_summary["pmid_identity_mismatches"], sample=len(sample_rows))

    direct_joint_pairs = int(directs["joint"].sum())
    direct_core_pairs = int(directs["core"].sum())
    direct_sem_pairs = int(directs["semmed"].sum())
    direct_shared = int((directs["core"] & directs["semmed"]).sum())
    failures = []
    def check(name, observed, expected):
        if int(observed) != int(expected):
            failures.append({"name": name, "observed": int(observed), "expected": int(expected)})
    check("shared_pairs", gain_summary["pairs"], EXPECTED_SHARED)
    check("semmed_gain_pairs", gain_summary["pairs_semmed_adds"], EXPECTED_SEMMED_GAIN_PAIRS)
    check("semmed_gain_links", gain_summary["semmed_added_links"], EXPECTED_SEMMED_GAIN_LINKS)
    check("semmed_gain_pmids", gain_summary["distinct_semmed_added_pmids"], EXPECTED_SEMMED_GAIN_PMIDS)
    check("shared_pmid_pairs", gain_summary["pairs_shared_pmid"], EXPECTED_SHARED_PMID_PAIRS)
    check("shared_pmid_links", gain_summary["shared_links"], EXPECTED_SHARED_PMID_LINKS)
    check("pmid_identity", gain_summary["pmid_identity_mismatches"], 0)
    check("three_paths", three_groups["complete_paths"], EXPECTED_PATHS_TWO)
    check("bitset_paths", bitset_total, EXPECTED_PATHS_TWO)
    check("core_view", three_groups["core_view"], EXPECTED_CORE_VIEW)
    check("core_gain_record", three_groups["core_paths_gaining_semmed_record"], EXPECTED_CORE_GAIN_RECORD)
    check("two_paths", two_groups["complete_paths"], EXPECTED_PATHS_ONE)
    check("two_semmed_alone", two_groups["semmed_alone"], EXPECTED_TWO_SEMMED_ALONE)
    check("two_requires_both", two_groups["requires_both"], EXPECTED_TWO_REQUIRES_BOTH)
    check("two_both_edge", two_groups["semmed_also_core_record"], EXPECTED_TWO_BOTH_EDGE)
    check("matrix_three", coverage["three_joint"]["complete_paths"], EXPECTED_PATHS_TWO)
    check("matrix_two", coverage["two_joint"]["complete_paths"], EXPECTED_PATHS_ONE)
    check("query_rows", query_mismatches, 0)
    check("marginals", len(marginal_mismatches), 0)
    check("direct_shared", direct_shared, EXPECTED_SHARED)
    if abs(coverage["two_joint"]["mean_coverage"] - EXPECTED_MEAN_ONE) > 0.00001:
        failures.append({"name": "mean_two_edge_coverage", "observed": coverage["two_joint"]["mean_coverage"], "expected": EXPECTED_MEAN_ONE})
    if abs(coverage["three_joint"]["mean_coverage"] - EXPECTED_MEAN_TWO) > 0.00001:
        failures.append({"name": "mean_three_edge_coverage", "observed": coverage["three_joint"]["mean_coverage"], "expected": EXPECTED_MEAN_TWO})

    _write_tsv(analysis / "source_sequences.tsv", sequence_rows, ["position_1", "position_2", "position_3", "sequence_group", "complete_paths"])
    _write_tsv(analysis / "source_sequence_groups.tsv", [{"group": key, "complete_paths": value} for key, value in three_groups.items()], ["group", "complete_paths"])
    _write_tsv(analysis / "two_edge_source_sequences.tsv", [{"position_1": key[0], "position_2": key[1], "sequence_group": sequence_group(key), "complete_paths": value} for key, value in sorted(two_sequences.items())], ["position_1", "position_2", "sequence_group", "complete_paths"])
    coverage_rows = []
    band_rows = []
    for name, item in coverage.items():
        coverage_rows.append(
            {
                "view": name,
                "universe_pairs": item["universe_pairs"],
                "reached_pairs": item["reached_pairs"],
                "unreached_pairs": item["unreached_pairs"],
                "complete_paths": item["complete_paths"],
                "pooled_coverage": item["pooled_coverage"],
                "mean_coverage": item["mean_coverage"],
                "reached_path_median": item["reached_path_median"],
                "reached_path_q1": item["reached_path_q1"],
                "reached_path_q3": item["reached_path_q3"],
                "direct_and_reached": item["direct_and_reached"],
                "no_direct_and_reached": item["no_direct_and_reached"],
                "no_direct_pairs": item["no_direct_pairs"],
            }
        )
        for band, count in item["path_count_band"].items():
            band_rows.append({"view": name, "band": band, "pairs": count})
    _write_tsv(analysis / "candidate_coverage.tsv", coverage_rows, list(coverage_rows[0]))
    _write_tsv(analysis / "path_count_bands.tsv", band_rows, ["view", "band", "pairs"])
    _write_tsv(analysis / "candidate_coverage_by_query.tsv", per_query_rows, list(per_query_rows[0]))
    _write_tsv(
        analysis / "candidate_dependence.tsv",
        [{"population": name, "category": key, "pairs": value} for name, counter in dependence.items() for key, value in counter.items()],
        ["population", "category", "pairs"],
    )
    _write_tsv(
        analysis / "path_increase_split.tsv",
        [{"comparison": name, **values} for name, values in increase.items()],
        ["comparison", "paths_on_newly_reached_pairs", "paths_added_on_already_reached_pairs", "newly_reached_pairs", "already_reached_pairs", "path_increase"],
    )
    _write_tsv(
        analysis / "intermediate_cancer_concentration.tsv",
        [{"measure": key, "value": value} for key, value in cancer_fraction.items()]
        + [{"measure": f"two_edge_pmid_{key}", "value": value} for key, value in two_pmid.items()],
        ["measure", "value"],
    )
    _write_tsv(analysis / "concentration_sample.tsv", sample_rows, list(sample_rows[0]) if sample_rows else ["query_cui"])
    _write_tsv(analysis / "edge_position_summary.tsv", position_summary, list(position_summary[0]))
    _write_tsv(analysis / "adjacent_overlap_summary.tsv", overlap_rows, list(overlap_rows[0]))
    _write_tsv(analysis / "shared_pair_gain_by_position.tsv", gain_positions, list(gain_positions[0]))
    _write_tsv(analysis / "shared_pair_gain_summary.tsv", [{"measure": key, "value": value} for key, value in sorted(gain_summary.items())], ["measure", "value"])
    _write_tsv(analysis / "shared_article_field_summary.tsv", field_rows, ["resource", "field", "value", "pair_pmid_links", "distinct_pairs", "link_denominator", "multi_label"])
    _write_tsv(analysis / "shared_article_combinations.tsv", combo_field_rows, ["core_attributes", "semmed_traversal_details", "semmed_negated_details", "semmed_other_details", "pair_pmid_links", "distinct_pairs", "link_denominator", "multi_label"])

    heatmap_max = 0.0
    combination_plot = []
    for row in combo_rows:
        denominator = int(row["denominator"]) or 1
        percent = 100 * int(row["count"]) / denominator
        heatmap_max = max(heatmap_max, percent)
        combination_plot.append({**row, "percent": percent})
    report_summary = {
        "gain": {key: int(value) if not isinstance(value, float) else value for key, value in gain_summary.items()},
        "three_groups": {key: int(value) for key, value in three_groups.items()},
        "two_groups": two_groups,
        "coverage": {
            "universe": len(candidates) * len(queries),
            "two_joint_reached": coverage["two_joint"]["reached_pairs"],
            "two_semmed_reached": coverage["two_semmed"]["reached_pairs"],
            "two_core_reached": coverage["two_core"]["reached_pairs"],
            "two_joint_mean": coverage["two_joint"]["mean_coverage"],
            "two_added_vs_semmed": dependence["two_all"]["joint_only_reach"],
            "three_joint_reached": coverage["three_joint"]["reached_pairs"],
            "three_core_reached": coverage["three_core"]["reached_pairs"],
            "three_semmed_reached": coverage["three_semmed"]["reached_pairs"],
            "three_joint_mean": coverage["three_joint"]["mean_coverage"],
            "three_added_vs_core": coverage["three_joint"]["reached_pairs"] - coverage["three_core"]["reached_pairs"],
            "three_added_vs_semmed": coverage["three_joint"]["reached_pairs"] - coverage["three_semmed"]["reached_pairs"],
            "three_cancer_fraction_median": cancer_fraction["median_fraction"],
        },
        "direct": {
            "core_only": direct_core_pairs - direct_shared,
            "shared": direct_shared,
            "semmed_only": direct_sem_pairs - direct_shared,
            "joint": direct_joint_pairs,
        },
        "no_direct_categories": {
            "two": [
                dependence["two_no_direct"]["core_only_reach"],
                dependence["two_no_direct"]["semmed_only_reach"],
                dependence["two_no_direct"]["either_resource_reach"],
                dependence["two_no_direct"]["joint_only_reach"],
            ],
            "three": [
                dependence["three_no_direct"]["core_only_reach"],
                dependence["three_no_direct"]["semmed_only_reach"],
                dependence["three_no_direct"]["either_resource_reach"],
                dependence["three_no_direct"]["joint_only_reach"],
            ],
        },
        "sample_size": len(sample_rows),
        "sample_capped": sum(1 for row in sample_rows if row["pmid_scope"] != "exact"),
        "combination_plot": combination_plot,
        "heatmap_max": heatmap_max,
        "overlap_plot": [row for row in overlap_rows if row["comparison"] == "union"],
    }
    plot_figures(analysis, figures, report_summary)
    (run_dir / "report.md").write_text(render_report(report_summary), encoding="utf-8")
    validation = {
        "status": "failed" if failures else "passed",
        "failures": failures,
        "marginal_mismatches": marginal_mismatches[:10],
        "anchors": {
            "three_paths": three_groups["complete_paths"],
            "core_view": three_groups["core_view"],
            "requires_both": three_groups["requires_both_resources"],
            "two_mean_coverage": coverage["two_joint"]["mean_coverage"],
            "three_mean_coverage": coverage["three_joint"]["mean_coverage"],
        },
        "sample": {
            "rule": "smallest gene id within each query, path-count band, and direct-record status",
            "queries": "all" if args.sample_queries == 0 else args.sample_queries,
            "selected_pairs": len(sample_rows),
            "pmid_path_cap": PMID_PATH_CAP,
            "capped_pairs": report_summary["sample_capped"],
            "two_edge_pmid_concentration": "exact",
            "three_edge_intermediate_cancer_concentration": "exact",
        },
        "stages": timer.stages,
        "peak_rss_bytes": peak_rss_bytes(),
    }
    write_json(run_dir / "validation.json", validation)
    write_json(
        run_dir / "manifest.json",
        {
            "task": "path_evidence_audit_completion",
            "status": "failed" if failures else "completed",
            "graph_run": str(args.graph_run),
            "prior_path_evidence_audit": str(args.prior_audit),
            "prior_manifest_status": "completed",
            "prior_manifest": str(prior_manifest),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "recovered_from_prior_tables": [
                "edge_profiles.parquet",
                "edge_position_distributions source via edge profiles",
                "adjacent_edge_combinations.parquet",
                "adjacent_article_overlap.parquet",
                "shared_pair_pmids.parquet",
                "shared_article_relation_details.parquet",
            ],
            "calculated_exactly": [
                "bidirectional PMID sets",
                "three-edge source sequences",
                "candidate pair path counts",
                "intermediate-cancer concentration",
                "two-edge PMID concentration",
            ],
            "sampled": ["three-edge intermediate-gene and PMID concentration"],
            "stages": timer.stages,
        },
    )
    print(f"status={'failed' if failures else 'completed'} failures={len(failures)} run_dir={run_dir}", flush=True)
    if failures:
        print(failures[:8], flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

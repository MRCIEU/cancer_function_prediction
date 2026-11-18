"""Source sequences, bidirectional PMID gains, and candidate-pair coverage.

Counts are exact. Three-edge paths are aggregated by grouped extension counts.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from src.path_count import exact_two_intermediate
from src.path_evidence import brute_path_weights

SOURCE_CATEGORIES = ("core_only", "both_sources", "semmed_only")
SEQUENCE_GROUPS = (
    "only_core",
    "only_semmed",
    "either_resource",
    "requires_both_resources",
)


def sequence_group(categories: tuple[str, ...]) -> str:
    has_core = "core_only" in categories
    has_semmed = "semmed_only" in categories
    if has_core and has_semmed:
        return "requires_both_resources"
    if has_core:
        return "only_core"
    if has_semmed:
        return "only_semmed"
    return "either_resource"


def three_edge_source_sequences(
    genes_of_cancer: dict[str, dict[str, str]],
    candidates: set[str],
    queries: set[str],
) -> Counter:
    """Count complete three-edge paths by the source category of each edge.

    ``genes_of_cancer[cancer][gene]`` is ``core_only``, ``both_sources``, or ``semmed_only``.
    Repeated genes and repeated cancers are excluded. Initial and final edges are independent
    once those exclusions are applied, so the path count is the product of the grouped sizes.
    """
    queries_of: dict[str, set[str]] = defaultdict(set)
    for cancer, genes in genes_of_cancer.items():
        if cancer in queries:
            for gene in genes:
                queries_of[gene].add(cancer)
    sequences: Counter = Counter()
    for cancer, genes in genes_of_cancer.items():
        candidate_categories = {gene: category for gene, category in genes.items() if gene in candidates}
        if not candidate_categories:
            continue
        left_counts: Counter = Counter(candidate_categories.values())
        for gene, middle in genes.items():
            right_counts: Counter = Counter()
            for query in queries_of.get(gene, ()):
                if query == cancer:
                    continue
                right_counts[genes_of_cancer[query][gene]] += 1
            if not right_counts:
                continue
            available = left_counts.copy()
            own = candidate_categories.get(gene)
            if own is not None:
                available[own] -= 1
                if available[own] == 0:
                    del available[own]
            for left, n_left in available.items():
                if n_left <= 0:
                    continue
                for right, n_right in right_counts.items():
                    sequences[(left, middle, right)] += n_left * n_right
    return sequences


def brute_source_sequences(
    genes_of_cancer: dict[str, dict[str, str]],
    candidates: set[str],
    queries: set[str],
) -> Counter:
    sequences: Counter = Counter()
    plain = {cancer: set(genes) for cancer, genes in genes_of_cancer.items()}
    for cancer, genes in genes_of_cancer.items():
        for query in queries:
            if query == cancer or query not in plain:
                continue
            both = set(genes) & plain[query]
            for candidate, left in genes.items():
                if candidate not in candidates:
                    continue
                for gene in both:
                    if candidate == gene:
                        continue
                    sequences[(left, genes[gene], genes_of_cancer[query][gene])] += 1
    return sequences


def two_edge_source_sequences(
    genes_of_cancer: dict[str, dict[str, str]],
    gene_neighbors: dict[str, set[str]],
    candidates: set[str],
    queries: set[str],
) -> Counter:
    """Two-edge sequences. The gene–gene edge is SemMedDB-only by the graph rules."""
    sequences: Counter = Counter()
    for cancer, genes in genes_of_cancer.items():
        if cancer not in queries:
            continue
        for gene, category in genes.items():
            partners = [item for item in gene_neighbors.get(gene, ()) if item in candidates and item != gene]
            if partners:
                sequences[("semmed_only", category)] += len(partners)
    return sequences


def fill_sequence_table(sequences: Counter) -> list[dict]:
    rows = []
    for left in SOURCE_CATEGORIES:
        for middle in SOURCE_CATEGORIES:
            for right in SOURCE_CATEGORIES:
                key = (left, middle, right)
                count = int(sequences.get(key, 0))
                rows.append(
                    {
                        "position_1": left,
                        "position_2": middle,
                        "position_3": right,
                        "sequence_group": sequence_group(key),
                        "complete_paths": count,
                    }
                )
    return rows


def summarise_sequences(rows: list[dict]) -> dict:
    groups = Counter()
    total = 0
    for row in rows:
        groups[row["sequence_group"]] += row["complete_paths"]
        total += row["complete_paths"]
    core_view = groups["only_core"] + groups["either_resource"]
    semmed_view = groups["only_semmed"] + groups["either_resource"]
    core_only_edges = sum(
        row["complete_paths"]
        for row in rows
        if row["position_1"] == row["position_2"] == row["position_3"] == "core_only"
    )
    return {
        "complete_paths": total,
        "only_core": groups["only_core"],
        "only_semmed": groups["only_semmed"],
        "either_resource": groups["either_resource"],
        "requires_both_resources": groups["requires_both_resources"],
        "core_view": core_view,
        "semmed_view": semmed_view,
        "paths_added_by_semmed": total - core_view,
        "paths_added_by_core": total - semmed_view,
        "core_paths_gaining_semmed_record": core_view - core_only_edges,
        "all_core_only_edges": core_only_edges,
        "semmed_paths_gaining_core_record": semmed_view
        - sum(
            row["complete_paths"]
            for row in rows
            if "core_only" not in (row["position_1"], row["position_2"], row["position_3"])
            and "both_sources" not in (row["position_1"], row["position_2"], row["position_3"])
        ),
    }


def gain_category(core_only: int, semmed_only: int) -> str:
    core_adds = core_only > 0
    semmed_adds = semmed_only > 0
    if core_adds and semmed_adds:
        return "both_add_pmids"
    if core_adds:
        return "only_core_adds_pmids"
    if semmed_adds:
        return "only_semmed_adds_pmids"
    return "neither_adds_pmids"


def count_band_extended(value: int) -> str:
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
    if value <= 100:
        return "26-100"
    if value <= 1000:
        return "101-1000"
    if value <= 10000:
        return "1001-10000"
    return "10001+"


def three_edge_count_matrix(
    genes_of_cancer: dict[str, set[str]],
    candidates: list[str],
    queries: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return path counts, max single-cancer contribution, and intermediate-cancer counts.

    Rows follow ``queries``. Columns follow ``candidates``.
    """
    n_queries = len(queries)
    n_candidates = len(candidates)
    counts = np.zeros((n_queries, n_candidates), dtype=np.int64)
    max_cancer = np.zeros((n_queries, n_candidates), dtype=np.int32)
    n_cancers = np.zeros((n_queries, n_candidates), dtype=np.int16)
    if n_queries == 0 or n_candidates == 0:
        return counts, max_cancer, n_cancers, 0
    cand_index = {gene: index for index, gene in enumerate(candidates)}
    cancer_ids = sorted(set(genes_of_cancer) | set(queries))
    cancer_index = {cancer: index for index, cancer in enumerate(cancer_ids)}
    gene_ids = sorted({gene for genes in genes_of_cancer.values() for gene in genes})
    gene_index = {gene: index for index, gene in enumerate(gene_ids)}
    n_genes = len(gene_ids)
    n_words = (n_genes + 63) // 64
    packed = np.zeros((len(cancer_ids), n_words), dtype=np.uint64)
    on_query = np.zeros((n_queries, n_candidates), dtype=bool)
    cand_lists: list[np.ndarray] = []
    for cancer in cancer_ids:
        genes = genes_of_cancer.get(cancer, set())
        row = cancer_index[cancer]
        local = []
        for gene in genes:
            packed[row, gene_index[gene] // 64] |= np.uint64(1) << np.uint64(gene_index[gene] % 64)
            local_index = cand_index.get(gene)
            if local_index is not None:
                local.append(local_index)
        cand_lists.append(np.asarray(local, dtype=np.int32))
    for query_row, query in enumerate(queries):
        cancer_row = cancer_index.get(query)
        if cancer_row is None:
            continue
        for local_index in cand_lists[cancer_row]:
            on_query[query_row, local_index] = True
    query_rows = np.asarray([cancer_index.get(query, -1) for query in queries], dtype=np.int32)
    total = 0
    for cancer_row, cancer in enumerate(cancer_ids):
        local = cand_lists[cancer_row]
        if local.size == 0:
            continue
        both = packed & packed[cancer_row]
        both_count = np.bitwise_count(both).sum(axis=1).astype(np.int64)
        present = query_rows >= 0
        added = np.zeros(n_queries, dtype=np.int64)
        added[present] = both_count[query_rows[present]]
        same = np.fromiter((query == cancer for query in queries), dtype=bool, count=n_queries)
        added[same] = 0
        if not np.any(added):
            continue
        block = np.broadcast_to(added[:, None], (n_queries, local.size)).copy()
        subtract = on_query[:, local].copy()
        subtract[same, :] = False
        block -= subtract.astype(np.int64)
        counts[:, local] += block
        positive = block > 0
        # Integer-array indexing returns a copy, so the update is written back explicitly.
        n_cancers[:, local] = n_cancers[:, local] + positive.astype(np.int16)
        max_cancer[:, local] = np.maximum(max_cancer[:, local], block.astype(np.int32))
        total += int(block.sum())
    return counts, max_cancer, n_cancers, total


def two_edge_count_matrix(
    genes_of_cancer: dict[str, set[str]],
    gene_neighbors: dict[str, set[str]],
    candidates: list[str],
    queries: list[str],
) -> np.ndarray:
    counts = np.zeros((len(queries), len(candidates)), dtype=np.int64)
    cand_index = {gene: index for index, gene in enumerate(candidates)}
    for query_row, query in enumerate(queries):
        for gene in genes_of_cancer.get(query, ()):
            for partner in gene_neighbors.get(gene, ()):
                column = cand_index.get(partner)
                if column is None or partner == gene:
                    continue
                counts[query_row, column] += 1
    return counts


def coverage_summary(counts: np.ndarray, direct: np.ndarray) -> dict:
    reached = counts > 0
    n_pairs = int(counts.size)
    n_reached = int(reached.sum())
    per_query = reached.sum(axis=1).astype(np.int64)
    n_queries = counts.shape[0]
    n_candidates = counts.shape[1] if counts.ndim == 2 else 0
    positive = counts[reached]
    return {
        "universe_pairs": n_pairs,
        "reached_pairs": n_reached,
        "unreached_pairs": n_pairs - n_reached,
        "complete_paths": int(counts.sum()),
        "pooled_coverage": n_reached / n_pairs if n_pairs else 0.0,
        "mean_coverage": float(per_query.mean() / n_candidates) if n_queries and n_candidates else 0.0,
        "per_query_reached": per_query,
        "path_count_band": _band_counts(counts),
        "reached_path_median": float(np.median(positive)) if positive.size else 0.0,
        "reached_path_q1": float(np.percentile(positive, 25)) if positive.size else 0.0,
        "reached_path_q3": float(np.percentile(positive, 75)) if positive.size else 0.0,
        "direct_and_reached": int((reached & direct).sum()),
        "no_direct_and_reached": int((reached & ~direct).sum()),
        "no_direct_pairs": int((~direct).sum()),
    }


def _band_counts(counts: np.ndarray) -> Counter:
    bands: Counter = Counter()
    flat = counts.ravel()
    # Exact band assignment without materialising a Python int per pair when possible.
    bands["0"] = int((flat == 0).sum())
    bands["1"] = int((flat == 1).sum())
    bands["2-5"] = int(((flat >= 2) & (flat <= 5)).sum())
    bands["6-10"] = int(((flat >= 6) & (flat <= 10)).sum())
    bands["11-25"] = int(((flat >= 11) & (flat <= 25)).sum())
    bands["26-100"] = int(((flat >= 26) & (flat <= 100)).sum())
    bands["101-1000"] = int(((flat >= 101) & (flat <= 1000)).sum())
    bands["1001-10000"] = int(((flat >= 1001) & (flat <= 10000)).sum())
    bands["10001+"] = int((flat > 10000).sum())
    return bands


def pair_dependence(core: np.ndarray, semmed: np.ndarray, joint: np.ndarray, mask: np.ndarray) -> Counter:
    """Classify pairs inside ``mask`` by single-resource and joint reachability."""
    core_hit = (core > 0) & mask
    semmed_hit = (semmed > 0) & mask
    joint_hit = (joint > 0) & mask
    out = Counter()
    out["core_only_reach"] = int((core_hit & ~semmed_hit & joint_hit).sum())
    out["semmed_only_reach"] = int((semmed_hit & ~core_hit & joint_hit).sum())
    out["either_resource_reach"] = int((core_hit & semmed_hit & joint_hit).sum())
    out["joint_only_reach"] = int((joint_hit & ~core_hit & ~semmed_hit).sum())
    out["joint_reached"] = int(joint_hit.sum())
    out["core_reached"] = int((core_hit & mask).sum())
    out["semmed_reached"] = int((semmed_hit & mask).sum())
    return out


def path_increase_split(reference: np.ndarray, joint: np.ndarray) -> dict:
    new_pair = (reference == 0) & (joint > 0)
    already = reference > 0
    added_new = int(joint[new_pair].sum())
    added_existing = int((joint[already] - reference[already]).sum())
    return {
        "paths_on_newly_reached_pairs": added_new,
        "paths_added_on_already_reached_pairs": added_existing,
        "newly_reached_pairs": int(new_pair.sum()),
        "already_reached_pairs": int(already.sum()),
        "path_increase": added_new + added_existing,
    }


def intermediate_cancer_fraction(counts: np.ndarray, max_cancer: np.ndarray) -> dict:
    reached = counts > 0
    fraction = np.zeros(counts.shape, dtype=np.float64)
    fraction[reached] = max_cancer[reached] / counts[reached]
    values = fraction[reached]
    return {
        "reached_pairs": int(reached.sum()),
        "median_fraction": float(np.median(values)) if values.size else 0.0,
        "q1_fraction": float(np.percentile(values, 25)) if values.size else 0.0,
        "q3_fraction": float(np.percentile(values, 75)) if values.size else 0.0,
        "fraction_at_least_0_5": int((values >= 0.5).sum()) if values.size else 0,
        "fraction_exactly_1": int((values == 1).sum()) if values.size else 0,
    }


def select_concentration_sample(counts: np.ndarray, direct: np.ndarray, candidates: list[str]) -> list[tuple[int, int, str, str]]:
    """One deterministic pair per query, path-count band, and direct-record status.

    The selected candidate is the smallest gene identifier in the cell.
    """
    bands = (
        (1, 1, "1"),
        (2, 10, "2-10"),
        (11, 100, "11-100"),
        (101, 1000, "101-1000"),
        (1001, 10000, "1001-10000"),
        (10001, 10**18, "10001+"),
    )
    chosen = []
    order = np.argsort(np.asarray(candidates))
    for query_row in range(counts.shape[0]):
        row = counts[query_row]
        direct_row = direct[query_row]
        for low, high, label in bands:
            for has_direct, status in ((True, "direct_record"), (False, "no_direct_record")):
                mask = (row >= low) & (row <= high) & (direct_row if has_direct else ~direct_row)
                if not np.any(mask):
                    continue
                for column in order:
                    if mask[column]:
                        chosen.append((query_row, int(column), label, status))
                        break
    return chosen


def marginal_from_sequences(rows: list[dict], left_name: str, right_name: str) -> Counter:
    counts: Counter = Counter()
    for row in rows:
        counts[(row[left_name], row[right_name])] += row["complete_paths"]
    return counts


def plain_adjacency(labelled: dict[str, dict[str, str]]) -> dict[str, set[str]]:
    return {cancer: set(genes) for cancer, genes in labelled.items()}


def expected_small_total(labelled, candidates, queries) -> int:
    plain = plain_adjacency(labelled)
    return brute_path_weights(plain, {}, candidates, queries)["two_total"]

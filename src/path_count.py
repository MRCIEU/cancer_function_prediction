"""Exact counts for the two short path templates.

one_intermediate_path has two edges.
two_intermediate_path has three edges.
A canonical entity tuple is counted once. Paths do not follow asserted direction.
"""

from __future__ import annotations

import numpy as np


def _as_set_map(adjacency: dict) -> dict[int, set[int]]:
    return {int(key): set(int(item) for item in values) for key, values in adjacency.items()}


def brute_one_intermediate(
    gene_neighbors: dict[int, set[int]],
    genes_of_cancer: dict[int, set[int]],
    candidates: set[int],
    queries: set[int],
) -> dict[int, dict]:
    """Template: candidate — intermediate gene — query cancer."""
    out = {}
    for query in queries:
        linked = genes_of_cancer.get(query, set())
        reached = set()
        n_paths = 0
        for gene in linked:
            partners = gene_neighbors.get(gene, set()) & candidates
            partners.discard(gene)
            if not partners:
                continue
            n_paths += len(partners)
            reached |= partners
        out[query] = {"n_paths": n_paths, "reached": reached}
    return out


def brute_two_intermediate(
    genes_of_cancer: dict[int, set[int]],
    candidates: set[int],
    queries: set[int],
    cancers: set[int] | None = None,
) -> dict[int, dict]:
    """Template: candidate — intermediate cancer — intermediate gene — query cancer."""
    universe = set(genes_of_cancer) if cancers is None else set(cancers)
    out = {}
    for query in queries:
        genes_q = genes_of_cancer.get(query, set())
        reached = set()
        n_paths = 0
        for cancer in universe:
            if cancer == query:
                continue
            genes_k = genes_of_cancer.get(cancer, set())
            both = genes_k & genes_q
            cands = genes_k & candidates
            if not both or not cands:
                continue
            n_paths += len(cands) * len(both) - len(cands & both)
            if len(both) >= 2:
                reached |= cands
            elif len(both) == 1:
                reached |= cands - both
        out[query] = {"n_paths": n_paths, "reached": reached}
    return out


def _pack(index_sets: list[set[int]], n_bits: int) -> np.ndarray:
    n_words = (n_bits + 63) // 64
    packed = np.zeros((len(index_sets), n_words), dtype=np.uint64)
    for row, items in enumerate(index_sets):
        for item in items:
            packed[row, item // 64] |= np.uint64(1) << np.uint64(item % 64)
    return packed


def _popcount_rows(bits: np.ndarray) -> np.ndarray:
    return np.bitwise_count(bits).sum(axis=1).astype(np.int64)


def exact_two_intermediate(
    genes_of_cancer: dict[int, set[int]],
    candidates: set[int],
    queries: list[int],
    cancer_ids: list[int],
) -> dict[int, dict]:
    """Exact template-2 counts for every query, using bitsets over genes."""
    if not cancer_ids:
        return {query: {"n_paths": 0, "n_reached": 0} for query in queries}
    gene_ids = sorted({gene for genes in genes_of_cancer.values() for gene in genes})
    gene_index = {gene: index for index, gene in enumerate(gene_ids)}
    n_genes = len(gene_ids)
    cancer_index = {cancer: index for index, cancer in enumerate(cancer_ids)}
    sets = []
    for cancer in cancer_ids:
        sets.append({gene_index[gene] for gene in genes_of_cancer.get(cancer, set()) if gene in gene_index})
    packed = _pack(sets, n_genes)
    candidate_bits = {gene_index[gene] for gene in candidates if gene in gene_index}
    candidate_mask = _pack([candidate_bits], n_genes)[0]
    cand_packed = packed & candidate_mask
    cand_counts = _popcount_rows(cand_packed)
    query_rows = [cancer_index[query] for query in queries if query in cancer_index]
    n_paths = {query: 0 for query in queries}
    reached_words = (n_genes + 63) // 64
    reached = {query: np.zeros(reached_words, dtype=np.uint64) for query in queries}
    query_lookup = {cancer_index[query]: query for query in queries if query in cancer_index}
    for cancer_row, cancer in enumerate(cancer_ids):
        both = packed & packed[cancer_row]
        both_count = _popcount_rows(both)
        inter = _popcount_rows(cand_packed[cancer_row] & packed)
        # Broadcast is expressed per cancer row against every cancer, then kept for queries.
        contribution = cand_counts[cancer_row] * both_count - inter
        for other_row, query in query_lookup.items():
            if other_row == cancer_row:
                continue
            n_paths[query] += int(contribution[other_row])
            count = int(both_count[other_row])
            if count >= 2:
                reached[query] |= cand_packed[cancer_row]
            elif count == 1:
                reached[query] |= cand_packed[cancer_row] & ~both[other_row]
    out = {}
    for query in queries:
        if query not in cancer_index:
            out[query] = {"n_paths": 0, "n_reached": 0}
            continue
        out[query] = {
            "n_paths": int(n_paths[query]),
            "n_reached": int(np.bitwise_count(reached[query]).sum()),
        }
    return out


def direct_candidate_counts(
    genes_of_cancer: dict[int, set[int]],
    candidates: set[int],
    queries: list[int],
) -> dict[int, int]:
    return {query: len(genes_of_cancer.get(query, set()) & candidates) for query in queries}

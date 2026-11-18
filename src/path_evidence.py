"""Occurrence weights and evidence summaries for the two short path templates.

one_intermediate_path has two edges.
two_intermediate_path has three edges.
Weights count eligible complete paths. They do not materialise those paths.
"""

from __future__ import annotations

from collections import Counter, defaultdict


BANDS = ("0", "1", "2-5", "6-10", "11-25", "26+")
EXPECTED_ONE_INTERMEDIATE_PATHS = 18414593
EXPECTED_TWO_INTERMEDIATE_PATHS = 14437428485
EXPECTED_SHARED_DIRECT_PAIRS = 24160


def count_band(value: int) -> str:
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


def intersection_size(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    if not left or not right:
        return 0
    if len(left) == 1 and len(right) == 1:
        return 1 if left[0] == right[0] else 0
    i = j = found = 0
    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            found += 1
            i += 1
            j += 1
        elif left[i] < right[j]:
            i += 1
        else:
            j += 1
    return found


def _canonical_gene_pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left <= right else (right, left)


def queries_by_gene(genes_of_cancer: dict[str, set[str]], queries: set[str]) -> dict[str, set[str]]:
    found: dict[str, set[str]] = defaultdict(set)
    for cancer, genes in genes_of_cancer.items():
        if cancer in queries:
            for gene in genes:
                found[gene].add(cancer)
    return found


def template_weights(
    genes_of_cancer: dict[str, set[str]],
    gene_neighbors: dict[str, set[str]],
    candidates: set[str],
    queries: set[str],
) -> dict:
    """Return per-edge occurrence weights and the two complete-path totals."""
    queries_of = queries_by_gene(genes_of_cancer, queries)
    one_gene_gene: dict[tuple[str, str], int] = defaultdict(int)
    one_gene_cancer: dict[tuple[str, str], int] = defaultdict(int)
    one_total = 0
    for gene, query_set in queries_of.items():
        partners = [item for item in gene_neighbors.get(gene, ()) if item in candidates and item != gene]
        if not partners or not query_set:
            continue
        n_partners = len(partners)
        n_queries = len(query_set)
        one_total += n_partners * n_queries
        for partner in partners:
            one_gene_gene[_canonical_gene_pair(gene, partner)] += n_queries
        for query in query_set:
            one_gene_cancer[(gene, query)] += n_partners

    right_of: dict[tuple[str, str], int] = {}
    left_of: dict[tuple[str, str], int] = {}
    sum_right: dict[str, int] = {}
    cands_of: dict[str, set[str]] = {}
    for cancer, genes in genes_of_cancer.items():
        cands = genes & candidates
        cands_of[cancer] = cands
        total_right = 0
        for gene in genes:
            n_right = len(queries_of.get(gene, ())) - (1 if cancer in queries_of.get(gene, ()) else 0)
            n_left = len(cands) - (1 if gene in cands else 0)
            if n_right:
                right_of[(gene, cancer)] = n_right
                total_right += n_right
            if n_left:
                left_of[(gene, cancer)] = n_left
        sum_right[cancer] = total_right

    two_pos1: dict[tuple[str, str], int] = {}
    two_pos2: dict[tuple[str, str], int] = {}
    two_pos3: dict[tuple[str, str], int] = defaultdict(int)
    two_total = 0
    for cancer, genes in genes_of_cancer.items():
        total_right = sum_right.get(cancer, 0)
        if not total_right:
            continue
        for gene in cands_of[cancer]:
            weight = total_right - right_of.get((gene, cancer), 0)
            if weight:
                two_pos1[(gene, cancer)] = weight
        for gene in genes:
            n_left = left_of.get((gene, cancer), 0)
            n_right = right_of.get((gene, cancer), 0)
            if n_left and n_right:
                two_pos2[(gene, cancer)] = n_left * n_right
                two_total += n_left * n_right
            if n_left:
                for query in queries_of.get(gene, ()):
                    if query != cancer:
                        two_pos3[(gene, query)] += n_left
    return {
        "one_gene_gene": one_gene_gene,
        "one_gene_cancer": one_gene_cancer,
        "two_pos1": two_pos1,
        "two_pos2": two_pos2,
        "two_pos3": two_pos3,
        "one_total": one_total,
        "two_total": two_total,
        "queries_of": queries_of,
        "right_of": right_of,
        "left_of": left_of,
    }


def brute_path_weights(
    genes_of_cancer: dict[str, set[str]],
    gene_neighbors: dict[str, set[str]],
    candidates: set[str],
    queries: set[str],
) -> dict:
    """Enumerate complete paths on a small graph and tally the same edge weights."""
    queries_of = queries_by_gene(genes_of_cancer, queries)
    one_gene_gene: Counter = Counter()
    one_gene_cancer: Counter = Counter()
    two_pos1: Counter = Counter()
    two_pos2: Counter = Counter()
    two_pos3: Counter = Counter()
    one_paths = 0
    for gene, query_set in queries_of.items():
        for partner in gene_neighbors.get(gene, ()):
            if partner not in candidates or partner == gene:
                continue
            for query in query_set:
                one_paths += 1
                one_gene_gene[_canonical_gene_pair(gene, partner)] += 1
                one_gene_cancer[(gene, query)] += 1
    two_paths = 0
    for cancer, genes in genes_of_cancer.items():
        for query in queries:
            if query == cancer:
                continue
            both = genes & genes_of_cancer.get(query, set())
            for candidate in genes & candidates:
                for gene in both:
                    if candidate == gene:
                        continue
                    two_paths += 1
                    two_pos1[(candidate, cancer)] += 1
                    two_pos2[(gene, cancer)] += 1
                    two_pos3[(gene, query)] += 1
    return {
        "one_total": one_paths,
        "two_total": two_paths,
        "one_gene_gene": one_gene_gene,
        "one_gene_cancer": one_gene_cancer,
        "two_pos1": two_pos1,
        "two_pos2": two_pos2,
        "two_pos3": two_pos3,
    }


def attribute_agreement(left_sig: int, right_sig: int) -> str:
    """Compare informative CORE tuple-set signatures. Zero means the set is empty."""
    if left_sig == 0 or right_sig == 0:
        return "insufficient_information"
    if left_sig == right_sig:
        return "comparable_attributes"
    return "different_attributes"


class OverlapAccumulator:
    """Exact overlap tallies for one PMID-set comparison."""

    def __init__(self) -> None:
        self.distinct = 0
        self.weighted = 0
        self.neither_distinct = 0
        self.neither_weighted = 0
        self.one_sided_distinct = 0
        self.one_sided_weighted = 0
        self.both_distinct = 0
        self.both_weighted = 0
        self.band_distinct = [0, 0, 0, 0, 0, 0]
        self.band_weighted = [0, 0, 0, 0, 0, 0]
        self.exact_distinct = [0] * 32
        self.exact_weighted = [0] * 32
        self.exact_distinct_high: dict[int, int] = {}
        self.exact_weighted_high: dict[int, int] = {}

    def add(self, availability: str, size: int, weight: int) -> None:
        self.distinct += 1
        self.weighted += weight
        if availability == "neither":
            self.neither_distinct += 1
            self.neither_weighted += weight
            return
        if availability == "one_sided":
            self.one_sided_distinct += 1
            self.one_sided_weighted += weight
            return
        self.both_distinct += 1
        self.both_weighted += weight
        index = _band_index(size)
        self.band_distinct[index] += 1
        self.band_weighted[index] += weight
        if size < len(self.exact_distinct):
            self.exact_distinct[size] += 1
            self.exact_weighted[size] += weight
        else:
            self.exact_distinct_high[size] = self.exact_distinct_high.get(size, 0) + 1
            self.exact_weighted_high[size] = self.exact_weighted_high.get(size, 0) + weight

    def merge(self, other: "OverlapAccumulator") -> None:
        self.distinct += other.distinct
        self.weighted += other.weighted
        self.neither_distinct += other.neither_distinct
        self.neither_weighted += other.neither_weighted
        self.one_sided_distinct += other.one_sided_distinct
        self.one_sided_weighted += other.one_sided_weighted
        self.both_distinct += other.both_distinct
        self.both_weighted += other.both_weighted
        for index in range(6):
            self.band_distinct[index] += other.band_distinct[index]
            self.band_weighted[index] += other.band_weighted[index]
        for index, value in enumerate(other.exact_distinct):
            self.exact_distinct[index] += value
            self.exact_weighted[index] += other.exact_weighted[index]
        for size, value in other.exact_distinct_high.items():
            self.exact_distinct_high[size] = self.exact_distinct_high.get(size, 0) + value
        for size, value in other.exact_weighted_high.items():
            self.exact_weighted_high[size] = self.exact_weighted_high.get(size, 0) + value


def _band_index(size: int) -> int:
    if size <= 1:
        return size
    if size <= 5:
        return 2
    if size <= 10:
        return 3
    if size <= 25:
        return 4
    return 5


def _touch(acc: OverlapAccumulator, left: tuple[int, ...], right: tuple[int, ...], weight: int) -> str:
    availability, size = pmid_availability(left, right)
    acc.add(availability, size, weight)
    return availability


def empty_overlap_result() -> dict:
    return {
        "union": OverlapAccumulator(),
        "core_core": OverlapAccumulator(),
        "core_semmed": OverlapAccumulator(),
        "semmed_core": OverlapAccumulator(),
        "semmed_semmed": OverlapAccumulator(),
        "joint_distinct": Counter(),
        "joint_weighted": Counter(),
    }


def accumulate_pair(result: dict, left: dict, right: dict, weight: int) -> None:
    availability = _touch(result["union"], left["all_pmids"], right["all_pmids"], weight)
    _touch(result["core_core"], left["core_pmids"], right["core_pmids"], weight)
    _touch(result["core_semmed"], left["core_pmids"], right["sem_pmids"], weight)
    _touch(result["semmed_core"], left["sem_pmids"], right["core_pmids"], weight)
    _touch(result["semmed_semmed"], left["sem_pmids"], right["sem_pmids"], weight)
    key = (left["support"], right["support"], availability)
    result["joint_distinct"][key] += 1
    result["joint_weighted"][key] += weight


def merge_overlap(target: dict, source: dict) -> None:
    for name in ("union", "core_core", "core_semmed", "semmed_core", "semmed_semmed"):
        target[name].merge(source[name])
    target["joint_distinct"].update(source["joint_distinct"])
    target["joint_weighted"].update(source["joint_weighted"])


def overlap_template2_positions_1_2(nodes: list[dict]) -> dict:
    """PMID overlap for candidate edge × intermediate-gene edge at one cancer."""
    result = empty_overlap_result()
    candidates = [node for node in nodes if node["cand"]]
    extenders = [node for node in nodes if node["extend"]]
    for right in extenders:
        weight = right["extend"]
        for left in candidates:
            if left["gene"] == right["gene"]:
                continue
            accumulate_pair(result, left, right, weight)
    return result


def pmid_availability(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[str, int]:
    if left and right:
        return "both_known", intersection_size(left, right)
    if left or right:
        return "one_sided", 0
    return "neither", 0


def add_overlap(store: dict, comparison: str, left: tuple[int, ...], right: tuple[int, ...], weight: int) -> None:
    availability, size = pmid_availability(left, right)
    key = (comparison, availability)
    row = store.setdefault(key, {"distinct": 0, "weighted": 0, "bands_distinct": Counter(), "bands_weighted": Counter(), "exact_distinct": Counter(), "exact_weighted": Counter()})
    row["distinct"] += 1
    row["weighted"] += weight
    if availability == "both_known":
        band = count_band(size)
        row["bands_distinct"][band] += 1
        row["bands_weighted"][band] += weight
        row["exact_distinct"][size] += 1
        row["exact_weighted"][size] += weight


def summarize_overlap(pairs: list[tuple[dict, dict, int]]) -> dict:
    """Aggregate PMID overlap for explicit adjacent pairs.

    Each item is ``(left_edge, right_edge, path_weight)``.
    """
    store: dict = {}
    support: Counter = Counter()
    support_w: Counter = Counter()
    for left, right, weight in pairs:
        support[(left["support"], right["support"])] += 1
        support_w[(left["support"], right["support"])] += weight
        add_overlap(store, "union", left["all_pmids"], right["all_pmids"], weight)
        add_overlap(store, "core_core", left["core_pmids"], right["core_pmids"], weight)
        add_overlap(store, "core_semmed", left["core_pmids"], right["sem_pmids"], weight)
        add_overlap(store, "semmed_core", left["sem_pmids"], right["core_pmids"], weight)
        add_overlap(store, "semmed_semmed", left["sem_pmids"], right["sem_pmids"], weight)
    return {"comparisons": store, "support_distinct": support, "support_weighted": support_w}


def grouped_crosstab(nodes: list[dict], left_key: str, right_key: str) -> tuple[Counter, Counter]:
    """Cross-tabulate edge attributes for template-2 positions 1–2 at one cancer.

    ``nodes`` each have cand, extend, and the two attribute keys.
    Self-pairs, where the candidate gene is also the intermediate gene, are removed.
    """
    distinct: Counter = Counter()
    weighted: Counter = Counter()
    left_counts: Counter = Counter()
    right_counts: Counter = Counter()
    right_weights: Counter = Counter()
    for node in nodes:
        if node["cand"]:
            left_counts[node[left_key]] += 1
        if node["extend"]:
            right_counts[node[right_key]] += 1
            right_weights[node[right_key]] += node["extend"]
    for left, n_left in left_counts.items():
        for right, n_right in right_counts.items():
            distinct[(left, right)] += n_left * n_right
        for right, weight in right_weights.items():
            weighted[(left, right)] += n_left * weight
    for node in nodes:
        if node["cand"] and node["extend"]:
            key = (node[left_key], node[right_key])
            distinct[key] -= 1
            weighted[key] -= node["extend"]
    return distinct, weighted

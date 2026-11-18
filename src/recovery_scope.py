"""Development-split rules and closed-form path scores.

The functions in this module do not read a formal test partition. They describe
the fresh development cohort used when protocol metadata registers no holdout
for this graph version.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.association_recovery import _stable_rank

PREDICATES = (
    "AFFECTS",
    "ASSOCIATED_WITH",
    "AUGMENTS",
    "CAUSES",
    "COEXISTS_WITH",
    "CONVERTS_TO",
    "DISRUPTS",
    "INHIBITS",
    "INTERACTS_WITH",
    "PREDISPOSES",
    "PREVENTS",
    "PRODUCES",
    "STIMULATES",
    "compared_with",
    "higher_than",
    "lower_than",
)

CORE_ATTRIBUTES = (
    "cge_up",
    "cge_down",
    "ccs_progression",
    "ccs_regression",
    "gci_causality",
    "gci_observation",
    "gcc_expression",
    "gcc_other",
    "type_biomarker",
    "type_oncogene",
    "type_tsg",
)

POSITIONS = ("t1", "t2_12", "t2_23")
LITERATURE = (
    "support_pmids",
    "shared_articles",
    "coverage",
    "age",
    "direct_collapsed",
    "direct_ltr",
    "direct_rtl",
    "direct_ambiguous",
    "cocite",
    "shared_references",
)

STRUCTURAL = (
    "two_edge_paths",
    "three_edge_paths",
    "two_edge_degree_weighted",
    "three_edge_degree_weighted",
    "candidate_degree",
    "candidate_publications",
    "query_degree",
    "query_publications",
)

# Prespecified feature blocks. Direct citations start at D, not in the baseline.
BASELINE_FIELDS = STRUCTURAL + tuple(f"{pos}_{name}" for pos in POSITIONS for name in ("support_pmids", "shared_articles", "coverage", "age"))
RELATION_FIELDS = CORE_ATTRIBUTES + tuple(f"predicate_{name}" for name in PREDICATES)
DIRECT_FIELDS = tuple(f"{pos}_direct_collapsed" for pos in POSITIONS)
DIRECTION_FIELDS = tuple(f"{pos}_direct_{role}" for pos in POSITIONS for role in ("ltr", "rtl", "ambiguous"))
COCITE_FIELDS = tuple(f"{pos}_cocite" for pos in POSITIONS)
REFERENCE_FIELDS = tuple(f"{pos}_shared_references" for pos in POSITIONS)
ORIENTATION_FIELDS = tuple(
    f"orientation_{name}_{side}" for name in PREDICATES for side in ("with", "against")
)

FEATURE_BLOCKS = {
    "baseline": BASELINE_FIELDS,
    "T": BASELINE_FIELDS + RELATION_FIELDS,
    "D": BASELINE_FIELDS + RELATION_FIELDS + DIRECT_FIELDS,
    "D_direction": BASELINE_FIELDS + RELATION_FIELDS + DIRECT_FIELDS + DIRECTION_FIELDS,
    "D_cocite": BASELINE_FIELDS + RELATION_FIELDS + DIRECT_FIELDS + COCITE_FIELDS,
    "D_references": BASELINE_FIELDS + RELATION_FIELDS + DIRECT_FIELDS + REFERENCE_FIELDS,
    "D_both_witnesses": BASELINE_FIELDS + RELATION_FIELDS + DIRECT_FIELDS + COCITE_FIELDS + REFERENCE_FIELDS,
    "D_orientation": BASELINE_FIELDS + RELATION_FIELDS + DIRECT_FIELDS + ORIENTATION_FIELDS,
    "combined": (
        BASELINE_FIELDS
        + RELATION_FIELDS
        + DIRECT_FIELDS
        + DIRECTION_FIELDS
        + COCITE_FIELDS
        + REFERENCE_FIELDS
        + ORIENTATION_FIELDS
    ),
}

CONTRASTS = (
    ("relation_types", "baseline", "T"),
    ("direct_citations", "T", "D"),
    ("citation_direction", "D", "D_direction"),
    ("cocitation", "D", "D_cocite"),
    ("shared_references", "D", "D_references"),
    ("complementarity_cocite", "D_cocite", "D_both_witnesses"),
    ("complementarity_references", "D_references", "D_both_witnesses"),
    ("predicate_direction", "D", "D_orientation"),
)

EXPECTED_CORE_PAIRS = 133863
EXPECTED_BOTH = 24160
EXPECTED_CORE_ONLY = 109703
EXPECTED_TWO_EDGE = 18414593
EXPECTED_THREE_EDGE = 14437428485
COHORT_SIZE = 24
CONTROL_RATIO = 10
PRIMARY_SEED = 20261116
SEEDS = (20261116, 20261117, 20261118)


def evenly_spaced(rows: list[dict], count: int) -> list[dict]:
    if len(rows) <= count:
        return list(rows)
    indexes = [round(i * (len(rows) - 1) / (count - 1)) for i in range(count)]
    return [rows[index] for index in indexes]


def select_development_cancers(rows: list[dict], count: int = COHORT_SIZE) -> dict:
    """Choose cancers from coverage and CORE abundance.

    A row needs ``cancer``, ``n_core``, ``paths``, and ``two_reached``. The
    path cap keeps three-edge aggregation inside a closed form. Relaxation of
    that cap is recorded when a band cannot fill the cohort.
    """
    bands = ((20, 35), (36, 55), (56, 80))
    relaxations = []
    path_caps = (5_000_000, 20_000_000, 50_000_000)
    chosen: list[dict] = []
    used: set[str] = set()
    cap_used = path_caps[-1]
    for cap in path_caps:
        cap_used = cap
        eligible = [
            row for row in rows
            if 20 <= int(row["n_core"]) <= 80 and 100_000 <= int(row["paths"]) <= cap and int(row["two_reached"]) >= 30
        ]
        chosen = []
        used = set()
        for low, high in bands:
            pool = sorted(
                [row for row in eligible if low <= int(row["n_core"]) <= high],
                key=lambda row: row["cancer"],
            )
            for row in evenly_spaced(pool, count // len(bands)):
                if row["cancer"] not in used:
                    chosen.append(row)
                    used.add(row["cancer"])
        if len(chosen) < count:
            rest = sorted([row for row in eligible if row["cancer"] not in used], key=lambda row: row["cancer"])
            for row in rest:
                if len(chosen) >= count:
                    break
                chosen.append(row)
                used.add(row["cancer"])
        if len(chosen) >= count:
            if cap != path_caps[0]:
                relaxations.append(f"three-edge path cap raised to {cap}")
            break
        relaxations.append(f"path cap {cap} produced {len(chosen)} cancers")
    chosen = chosen[:count]
    return {
        "cancers": [row["cancer"] for row in chosen],
        "rows": chosen,
        "path_cap": cap_used,
        "relaxations": relaxations,
    }


def stratified_targets(pairs: list[dict], n_take: int, seed: int) -> list[dict]:
    """Draw CORE positives in proportion to the SemMedDB overlap flag."""
    if n_take >= len(pairs):
        return list(pairs)
    shared = [row for row in pairs if row["semmed_direct"]]
    core_only = [row for row in pairs if not row["semmed_direct"]]
    raw = n_take * len(shared) / len(pairs) if pairs else 0
    n_shared = int(np.floor(raw))
    if raw - n_shared >= 0.5:
        n_shared += 1
    n_shared = min(n_shared, len(shared), n_take)
    n_only = min(n_take - n_shared, len(core_only))
    if n_shared + n_only < n_take:
        extra = n_take - n_shared - n_only
        if len(shared) - n_shared >= extra:
            n_shared += extra
        else:
            n_only += extra

    def take(group: list[dict], size: int) -> list[dict]:
        ranked = sorted(group, key=lambda row: (_stable_rank(seed, row["target_id"]), row["target_id"]))
        return ranked[:size]

    chosen = take(shared, n_shared) + take(core_only, n_only)
    chosen_ids = {row["target_id"] for row in chosen}
    if len(chosen) < n_take:
        rest = [row for row in pairs if row["target_id"] not in chosen_ids]
        chosen.extend(take(rest, n_take - len(chosen)))
    return chosen


def targets_per_cancer(n_core: int) -> int:
    """Keep at least half of a cancer's CORE pairs as background evidence."""
    return max(0, min(24, n_core // 2))


def candidate_neighbors(gene: str, neighbors: dict[str, set[str]], candidates: set[str]) -> list[str]:
    return [item for item in neighbors.get(gene, ()) if item in candidates and item != gene]


def two_edge_scores(
    genes_of: dict[str, set[str]],
    neighbors: dict[str, set[str]],
    query: str,
    candidates: set[str],
    index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return path counts, degree-weighted counts, and uniform walk mass."""
    counts = np.zeros(len(index), dtype=np.float64)
    weighted = np.zeros(len(index), dtype=np.float64)
    walk = np.zeros(len(index), dtype=np.float64)
    genes = list(genes_of.get(query, ()))
    if not genes:
        return counts, weighted, walk
    hop = 1.0 / len(genes)
    for gene in genes:
        partners = candidate_neighbors(gene, neighbors, candidates)
        if not partners:
            continue
        share = 1.0 / len(partners)
        walk_share = hop * share
        for partner in partners:
            slot = index[partner]
            counts[slot] += 1
            weighted[slot] += share
            walk[slot] += walk_share
    return counts, weighted, walk


def three_edge_scores(
    genes_of: dict[str, set[str]],
    cancers_of: dict[str, set[str]],
    query: str,
    candidates: set[str],
    index: dict[str, int],
    candidate_slots: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed form for counts, degree weights, and the uniform three-edge walk.

    Intermediate genes may be outside the candidate set. The candidate endpoint
    cannot repeat the intermediate gene, and the intermediate cancer cannot be
    the query. Degree weight is one over the number of genes on the intermediate
    cancer.
    """
    counts = np.zeros(len(index), dtype=np.float64)
    weighted = np.zeros(len(index), dtype=np.float64)
    walk = np.zeros(len(index), dtype=np.float64)
    query_genes = genes_of.get(query, ())
    n_query = len(query_genes)
    if n_query == 0:
        return counts, weighted, walk
    hop = 1.0 / n_query
    for gene in query_genes:
        cancers = [cancer for cancer in cancers_of.get(gene, ()) if cancer != query]
        if not cancers:
            continue
        cancer_share = 1.0 / len(cancers)
        for cancer in cancers:
            slots = candidate_slots.get(cancer)
            if slots is None or len(slots) == 0:
                continue
            gene_is_candidate = gene in index and gene in genes_of[cancer]
            n_cands = len(slots) - (1 if gene_is_candidate else 0)
            if n_cands <= 0:
                continue
            degree = float(len(genes_of[cancer]))
            counts[slots] += 1.0
            weighted[slots] += 1.0 / degree
            step = hop * cancer_share / n_cands
            walk[slots] += step
            if gene_is_candidate:
                slot = index[gene]
                counts[slot] -= 1.0
                weighted[slot] -= 1.0 / degree
                walk[slot] -= step
    return counts, weighted, walk


def three_edge_path_count(
    genes_of: dict[str, set[str]],
    query: str,
    candidates: set[str],
    index: dict[str, int],
    candidate_slots: dict[str, np.ndarray],
) -> np.ndarray:
    """Count C-K-G-Q paths. G is any gene on both cancers and is not C."""
    counts = np.zeros(len(index), dtype=np.float64)
    weighted = np.zeros(len(index), dtype=np.float64)
    query_genes = genes_of.get(query, set())
    if not query_genes:
        return counts, weighted
    for cancer, genes in genes_of.items():
        if cancer == query:
            continue
        both = genes & query_genes
        if not both:
            continue
        slots = candidate_slots.get(cancer)
        if slots is None or len(slots) == 0:
            continue
        degree = float(len(genes))
        both_count = len(both)
        counts[slots] += both_count
        weighted[slots] += both_count / degree
        for gene in both:
            if gene in index and gene in genes:
                counts[index[gene]] -= 1
                weighted[index[gene]] -= 1.0 / degree
    return counts, weighted


def brute_two_edge(genes_of, neighbors, query, candidates):
    scores = defaultdict(float)
    weighted = defaultdict(float)
    walk = defaultdict(float)
    genes = list(genes_of.get(query, ()))
    if not genes:
        return scores, weighted, walk
    hop = 1.0 / len(genes)
    for gene in genes:
        partners = [item for item in neighbors.get(gene, ()) if item in candidates and item != gene]
        if not partners:
            continue
        share = 1.0 / len(partners)
        for partner in partners:
            scores[partner] += 1
            weighted[partner] += share
            walk[partner] += hop * share
    return scores, weighted, walk


def brute_three_edge(genes_of, cancers_of, query, candidates):
    counts = defaultdict(float)
    walk = defaultdict(float)
    query_genes = list(genes_of.get(query, ()))
    if not query_genes:
        return counts, walk
    hop = 1.0 / len(query_genes)
    for gene in query_genes:
        cancers = [cancer for cancer in cancers_of.get(gene, ()) if cancer != query]
        if not cancers:
            continue
        for cancer in cancers:
            cands = [item for item in genes_of[cancer] if item in candidates and item != gene]
            if not cands:
                continue
            step = hop / len(cancers) / len(cands)
            for cand in cands:
                counts[cand] += 1
                walk[cand] += step
    return counts, walk


def combine_templates(two_edge: np.ndarray, three_edge: np.ndarray) -> np.ndarray:
    """Fixed rule: sum of log1p template scores."""
    return np.log1p(np.maximum(two_edge, 0)) + np.log1p(np.maximum(three_edge, 0))


def candidate_slot_table(genes_of: dict[str, set[str]], index: dict[str, int]) -> dict[str, np.ndarray]:
    table = {}
    for cancer, genes in genes_of.items():
        slots = [index[gene] for gene in genes if gene in index]
        table[cancer] = np.asarray(slots, dtype=np.int32)
    return table


class LazyTuples:
    """Present sorted PMID arrays as tuples, built once per PMID."""

    def __init__(self, arrays: dict[int, np.ndarray]):
        self.arrays = arrays
        self.cache: dict[int, tuple[int, ...]] = {}

    def get(self, key, default=()):
        pmid = int(key)
        cached = self.cache.get(pmid)
        if cached is not None:
            return cached
        array = self.arrays.get(pmid)
        if array is None:
            return default
        cached = tuple(int(item) for item in array)
        self.cache[pmid] = cached
        return cached


def aggregate_singleton_block(
    pmids: list[int],
    outgoing: dict[int, np.ndarray],
    incoming: dict[int, np.ndarray],
    extenders: list[int],
    candidates: list[int],
) -> dict[str, np.ndarray]:
    """Sum adjacent-edge motifs from each candidate to each extender.

    Every gene has one PMID. A link with both articles in both sets cannot
    occur when those PMIDs differ, so ambiguous counts stay zero. Self pairs
    are excluded. PMIDs must be unique across the genes in this block.
    """
    ext_set = set(extenders)
    owner: dict[int, int] = {}
    for index, pmid in enumerate(pmids):
        owner[int(pmid)] = index
    cites = [set() for _ in pmids]
    for index, pmid in enumerate(pmids):
        for target in outgoing.get(int(pmid), ()):
            other = owner.get(int(target))
            if other is not None and other != index:
                cites[index].add(other)
    names = (
        "support_pmids",
        "shared_articles",
        "direct_ltr",
        "direct_rtl",
        "direct_ambiguous",
        "direct_collapsed",
        "cocite",
        "shared_references",
    )
    out = {name: np.zeros(len(candidates), dtype=np.float64) for name in names}
    ext_counts: dict[int, int] = defaultdict(int)
    ref_counts: dict[int, int] = defaultdict(int)
    for index in extenders:
        for witness in incoming.get(int(pmids[index]), ()):
            ext_counts[int(witness)] += 1
        for witness in outgoing.get(int(pmids[index]), ()):
            ref_counts[int(witness)] += 1
    for slot, index in enumerate(candidates):
        if index in ext_set and len(extenders) == 1 and extenders[0] == index:
            continue
        partners = [item for item in extenders if item != index]
        if not partners:
            continue
        pmid = int(pmids[index])
        out["support_pmids"][slot] = float(2 * len(partners) - sum(1 for item in partners if pmids[item] == pmid))
        out["shared_articles"][slot] = float(sum(1 for item in partners if pmids[item] == pmid))
        out["direct_ltr"][slot] = float(len(cites[index].intersection(partners)))
        out["direct_rtl"][slot] = float(sum(1 for item in partners if index in cites[item]))
        out["direct_ambiguous"][slot] = 0.0
        out["direct_collapsed"][slot] = out["direct_ltr"][slot] + out["direct_rtl"][slot]
        inc = incoming.get(pmid, ())
        cocite = 0
        for witness in inc:
            cocite += ext_counts.get(int(witness), 0)
        if index in ext_set:
            cocite -= len(inc)
        out["cocite"][slot] = float(cocite)
        refs = outgoing.get(pmid, ())
        shared = 0
        for witness in refs:
            shared += ref_counts.get(int(witness), 0)
        if index in ext_set:
            shared -= len(refs)
        out["shared_references"][slot] = float(shared)
    return out


def citation_between(left, right, outgoing: LazyTuples, incoming: LazyTuples) -> dict | None:
    """Motifs for one adjacent pair. Missing PMID support returns None."""
    from src.citation_features import direct_citation_roles, intersection_size, third_party_motifs

    left_ids = tuple(sorted({int(item) for item in left}))
    right_ids = tuple(sorted({int(item) for item in right}))
    if not left_ids or not right_ids:
        return None
    roles = direct_citation_roles(left_ids, right_ids, outgoing)
    cocite = third_party_motifs(left_ids, right_ids, incoming, set())
    references = third_party_motifs(left_ids, right_ids, outgoing, set())
    return {
        "support_pmids": len(set(left_ids) | set(right_ids)),
        "shared_articles": intersection_size(left_ids, right_ids),
        "direct_collapsed": roles["direction_collapsed"],
        "direct_ltr": roles["exclusive_left_to_right"],
        "direct_rtl": roles["exclusive_right_to_left"],
        "direct_ambiguous": roles["ambiguous_position"],
        "cocite": cocite["witnesses"],
        "shared_references": references["witnesses"],
    }


def _papers_distinct(left_papers: set[int], right_papers: set[int]) -> bool:
    if not left_papers or not right_papers:
        return False
    if len(left_papers) > 1 or len(right_papers) > 1:
        return True
    return next(iter(left_papers)) != next(iter(right_papers))


def build_motif_block(pmid_sets: list[tuple[int, ...]], outgoing: dict, incoming: dict) -> dict:
    """Index one cancer's edges so subset motif sums do not revisit each pair."""
    sets = [set(int(item) for item in papers) for papers in pmid_sets]
    pmid_genes: dict[int, list[int]] = defaultdict(list)
    for gene, papers in enumerate(sets):
        for pmid in papers:
            pmid_genes[pmid].append(gene)
    on_cancer = set(pmid_genes)
    out_sets: dict[int, list[int]] = {}
    incoming_on: dict[int, set[int]] = defaultdict(set)
    for pmid in on_cancer:
        targets = []
        seen = set()
        for target in outgoing.get(pmid, ()):
            target = int(target)
            if target == pmid or target not in on_cancer or target in seen:
                continue
            seen.add(target)
            targets.append(target)
            incoming_on[target].add(pmid)
        out_sets[pmid] = targets

    def grouped(neighbors: dict) -> dict[int, dict[int, set[int]]]:
        partial: dict[int, dict[int, set[int]]] = defaultdict(dict)
        for pmid, genes in pmid_genes.items():
            for witness in neighbors.get(pmid, ()):
                witness = int(witness)
                if witness == pmid:
                    continue
                bucket = partial[witness]
                for gene in genes:
                    papers = bucket.get(gene)
                    if papers is None:
                        bucket[gene] = {pmid}
                    else:
                        papers.add(pmid)
        return partial

    return {
        "sets": sets,
        "pmid_genes": pmid_genes,
        "out_sets": out_sets,
        "incoming_on": incoming_on,
        "cocite": grouped(incoming),
        "references": grouped(outgoing),
    }


def sum_adjacent_motifs(block: dict, extenders: list[int], candidates: list[int]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Sum citation_between over extender edges for each candidate.

    An empty PMID set makes that pair unobserved. A candidate is not paired
    with itself. Returned arrays follow ``candidates``.
    """
    names = (
        "support_pmids",
        "shared_articles",
        "direct_ltr",
        "direct_rtl",
        "direct_ambiguous",
        "direct_collapsed",
        "cocite",
        "shared_references",
    )
    sets = block["sets"]
    observed = np.zeros(len(candidates), dtype=np.float64)
    sums = {name: np.zeros(len(candidates), dtype=np.float64) for name in names}
    ext_list = [gene for gene in extenders if sets[gene]]
    if not ext_list:
        return observed, sums
    ext_nonempty = set(ext_list)
    ext_size_sum = float(sum(len(sets[gene]) for gene in ext_list))
    pmid_ext_count: dict[int, int] = defaultdict(int)
    for gene in ext_list:
        for pmid in sets[gene]:
            pmid_ext_count[pmid] += 1
    cand_pos = {gene: pos for pos, gene in enumerate(candidates)}
    for gene, pos in cand_pos.items():
        papers = sets[gene]
        if not papers:
            continue
        self_ext = gene in ext_nonempty
        n_partners = len(ext_list) - (1 if self_ext else 0)
        if n_partners <= 0:
            continue
        observed[pos] = n_partners
        shared = 0
        for pmid in papers:
            shared += pmid_ext_count.get(pmid, 0) - (1 if self_ext else 0)
        sum_sizes = ext_size_sum - (len(papers) if self_ext else 0)
        sums["shared_articles"][pos] = shared
        sums["support_pmids"][pos] = len(papers) * n_partners + sum_sizes - shared
    pmid_genes = block["pmid_genes"]
    out_sets = block["out_sets"]
    incoming_on = block["incoming_on"]
    for gene, pos in cand_pos.items():
        papers = sets[gene]
        if not papers:
            continue
        ltr = rtl = ambiguous = 0
        for source in papers:
            for target in out_sets.get(source, ()):
                target_on_self = target in papers
                for other in pmid_genes.get(target, ()):
                    if other not in ext_nonempty or other == gene:
                        continue
                    if source in sets[other] and target_on_self:
                        ambiguous += 1
                    else:
                        ltr += 1
        for target in papers:
            for source in incoming_on.get(target, ()):
                for other in pmid_genes.get(source, ()):
                    if other not in ext_nonempty or other == gene:
                        continue
                    # Ambiguous links also start on the candidate and are counted above.
                    if source in papers and target in sets[other]:
                        continue
                    rtl += 1
        sums["direct_ltr"][pos] = ltr
        sums["direct_rtl"][pos] = rtl
        sums["direct_ambiguous"][pos] = ambiguous
        sums["direct_collapsed"][pos] = ltr + rtl + ambiguous
    _add_witness_sums(block["cocite"], block["pmid_genes"], ext_nonempty, cand_pos, sums["cocite"])
    _add_witness_sums(block["references"], block["pmid_genes"], ext_nonempty, cand_pos, sums["shared_references"])
    return observed, sums


def _add_witness_sums(grouped: dict, pmid_genes: dict, ext_nonempty: set[int], cand_pos: dict[int, int], out: np.ndarray) -> None:
    for by_gene in grouped.values():
        simple = True
        for papers in by_gene.values():
            if len(papers) != 1 or len(pmid_genes[next(iter(papers))]) != 1:
                simple = False
                break
        if simple:
            ext_hits = [gene for gene in by_gene if gene in ext_nonempty]
            n_ext = len(ext_hits)
            for gene in ext_hits:
                pos = cand_pos.get(gene)
                if pos is not None and n_ext:
                    out[pos] += n_ext - 1
            if n_ext:
                for gene in by_gene:
                    if gene in ext_nonempty:
                        continue
                    pos = cand_pos.get(gene)
                    if pos is not None:
                        out[pos] += n_ext
            continue
        for gene, left in by_gene.items():
            pos = cand_pos.get(gene)
            if pos is None:
                continue
            for other, right in by_gene.items():
                if other == gene or other not in ext_nonempty:
                    continue
                if _papers_distinct(left, right):
                    out[pos] += 1

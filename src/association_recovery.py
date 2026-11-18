"""Development-only rules for withheld gene–cancer association recovery.

These functions do not read a formal test split. A caller must supply an
already cleared development allowlist.
"""

from __future__ import annotations

from collections import defaultdict


def connected_pmid_groups(records: list[dict]) -> list[list[dict]]:
    """Keep targets that share a supporting PMID in one group.

    A record is ``{"target_id", "cancer", "gene", "pmids"}``.
    """
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for record in records:
        target = record["target_id"]
        find(target)
        for pmid in record["pmids"]:
            union(target, f"pmid:{pmid}")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[find(record["target_id"])].append(record)
    return list(grouped.values())


def assign_partitions(records: list[dict], fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)) -> dict[str, str]:
    """Assign whole PMID groups to train, tune, and development validation.

    Larger groups are placed first. The assignment fills the partition whose
    current share is furthest below its target, then prefers the cancer that
    is least represented in that partition.
    """
    names = ("train", "tune", "development_validation")
    groups = connected_pmid_groups(records)
    groups.sort(key=lambda group: (-len(group), min(record["target_id"] for record in group)))
    total = max(len(records), 1)
    counts = {name: 0 for name in names}
    cancer_counts = {name: defaultdict(int) for name in names}
    assignment = {}
    for group in groups:
        def deficit(name: str) -> float:
            return fractions[names.index(name)] - counts[name] / total

        def cancer_load(name: str) -> int:
            return sum(cancer_counts[name][record["cancer"]] for record in group)

        choice = max(names, key=lambda name: (deficit(name), -cancer_load(name), name))
        for record in group:
            assignment[record["target_id"]] = choice
            counts[choice] += 1
            cancer_counts[choice][record["cancer"]] += 1
    return assignment


def sample_controls(positives: list[str], eligible: list[str], direct_genes: set[str], ratio: int, seed: int) -> list[str]:
    """Draw uniform controls from genes with no original direct record.

    Masking must not move a formerly direct gene into this pool. Sampling is
    without replacement and stable for a fixed seed.
    """
    pool = [gene for gene in eligible if gene not in direct_genes and gene not in set(positives)]
    pool.sort()
    wanted = min(len(pool), ratio * len(positives))
    if wanted <= 0:
        return []
    ranked = sorted(pool, key=lambda gene: (_stable_rank(seed, gene), gene))
    return ranked[:wanted]


def _stable_rank(seed: int, gene: str) -> int:
    total = seed
    for character in gene:
        total = (total * 1315423911 + ord(character)) % 2**32
    return total


def apply_mask(edges: list[dict], citations: list[tuple[int, int]], target_pairs: set[tuple[str, str]], masked_pmids: set[int]) -> dict:
    """Remove target-pair edges, masked-PMID evidence, and incident citations."""
    kept_edges = []
    for edge in edges:
        pair = (edge["gene"], edge["cancer"])
        if pair in target_pairs:
            continue
        remaining = [pmid for pmid in edge["pmids"] if pmid not in masked_pmids]
        if not remaining and edge["pmids"]:
            continue
        kept = dict(edge)
        kept["pmids"] = remaining
        kept_edges.append(kept)
    kept_citations = [
        (citing, cited)
        for citing, cited in citations
        if citing not in masked_pmids and cited not in masked_pmids
    ]
    return {"edges": kept_edges, "citations": kept_citations}


def citation_roles_partition(exclusive_left: int, exclusive_right: int, ambiguous: int, collapsed: int) -> bool:
    return exclusive_left + exclusive_right + ambiguous == collapsed


def recall_at_k(rankings: dict[str, list[str]], positives: dict[str, set[str]], k: int = 50) -> dict:
    """Macro Recall@k across cancers. Cancers with no positives are omitted."""
    per_cancer = []
    recovered = 0
    total = 0
    for cancer in sorted(positives):
        held = positives[cancer]
        if not held:
            continue
        top = set(rankings.get(cancer, [])[:k])
        hits = len(top & held)
        recovered += hits
        total += len(held)
        per_cancer.append({"cancer": cancer, "positives": len(held), "recovered": hits, "recall": hits / len(held)})
    macro = sum(row["recall"] for row in per_cancer) / len(per_cancer) if per_cancer else 0.0
    return {
        "macro_recall": macro,
        "cancers": len(per_cancer),
        "recovered": recovered,
        "positives": total,
        "per_cancer": per_cancer,
    }


def rank_with_tie_break(rows: list[dict], score_key: str = "score") -> list[str]:
    """Higher score first. Equal scores break by gene identifier, independent of labels."""
    ordered = sorted(rows, key=lambda row: (-float(row[score_key]), row["gene"]))
    return [row["gene"] for row in ordered]

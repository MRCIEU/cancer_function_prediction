"""Joint article-relation patterns for one ordered adjacent edge pair.

The citation audit stored each relation's marginal count. This script rebuilds
the four presence flags on the same edge pair and checks those marginals before
writing the 16-pattern table.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from src.citation_features import classify_prepared_links, intersection_size, witness_matrices
from src.path_evidence import template_weights
from src.run_citation_audit import Edge, _both_index, _index_state
from src.run_complete_audit import labelled_from_profiles

GRAPH_RUN = Path("/home/b5ac/freddieyu.b5ac/projects/project_2/results/joint_graph_and_evidence_audit_20261116T0404Z")
PATH_AUDIT = Path("/home/b5ac/freddieyu.b5ac/projects/project_2/results/path_evidence_audit_20261116T1310Z")
CITATION = Path("/home/b5ac/freddieyu.b5ac/projects/project_2/results/citation_path_evidence_audit_20261116T1647Z")
OUTGOING: dict[int, np.ndarray] = {}
INCOMING: dict[int, np.ndarray] = {}
GC: list[Edge] = []
GG: list[Edge] = []
INSIDE = np.zeros(1, dtype=np.uint8)
POSITIONS = {
    "gene_t1": "2-edge 1–2",
    "cancer_12": "3-edge 1–2",
    "gene_23": "3-edge 2–3",
}
TEMPLATES = {
    "gene_t1": ("one_intermediate_path", "positions_1_2"),
    "cancer_12": ("two_intermediate_path", "positions_1_2"),
    "gene_23": ("two_intermediate_path", "positions_2_3"),
}
SOURCES = ("core_only", "both_sources", "semmed_only")
SOURCE_CODE = {name: index for index, name in enumerate(SOURCES)}


def _index_maps(path: Path) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    table = pq.read_table(path, columns=["citing_pmid", "cited_pmid"])
    citing = table.column("citing_pmid").to_numpy()
    cited = table.column("cited_pmid").to_numpy()

    def grouped(keys: np.ndarray, values: np.ndarray) -> dict[int, np.ndarray]:
        order = np.argsort(keys, kind="mergesort")
        keys = keys[order]
        values = values[order]
        change = np.flatnonzero(keys[1:] != keys[:-1]) + 1
        starts = np.concatenate((np.asarray([0], dtype=np.int64), change))
        ends = np.concatenate((change, np.asarray([keys.size], dtype=np.int64)))
        grouped_map: dict[int, np.ndarray] = {}
        for start, end in zip(starts, ends):
            grouped_map[int(keys[start])] = np.unique(values[start:end])
        return grouped_map

    print(f"[index] {citing.size} directed links", flush=True)
    return grouped(citing, cited), grouped(cited, citing)


def _prepare() -> dict:
    profiles = pq.read_table(
        PATH_AUDIT / "analysis" / "edge_profiles.parquet",
        columns=["family", "endpoint_a", "endpoint_b", "support"],
    ).to_pylist()
    labelled, neighbors = labelled_from_profiles(profiles, "joint")
    genes = {cancer: set(members) for cancer, members in labelled.items()}
    candidates = {
        row["gene_id"]
        for row in pq.read_table(GRAPH_RUN / "analysis" / "candidate_genes.parquet", columns=["gene_id", "status"]).to_pylist()
        if row["status"] == "resolved_human"
    }
    queries = {row["cancer_cui"] for row in pq.read_table(GRAPH_RUN / "analysis" / "cancer_queries.parquet", columns=["cancer_cui"]).to_pylist()}
    weights = template_weights(genes, neighbors, candidates, queries)
    if int(weights["one_total"]) != 18_414_593 or int(weights["two_total"]) != 14_437_428_485:
        raise RuntimeError(f"path totals {weights['one_total']} {weights['two_total']} do not match the audit")
    pmids: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    membership = pq.read_table(
        CITATION / "analysis" / "edge_pmid_membership.parquet",
        columns=["family", "endpoint_a", "endpoint_b", "pmid", "qualifying"],
    )
    qualifying_column = membership.column("qualifying").to_numpy(zero_copy_only=False)
    family = membership.column("family").to_pylist()
    endpoint_a = membership.column("endpoint_a").to_pylist()
    endpoint_b = membership.column("endpoint_b").to_pylist()
    pmid = membership.column("pmid").to_pylist()
    support: set[int] = set()
    for index, is_qualifying in enumerate(qualifying_column):
        if not bool(is_qualifying):
            continue
        value = int(pmid[index])
        support.add(value)
        pmids[(family[index], endpoint_a[index], endpoint_b[index])].add(value)
    GC.clear()
    GG.clear()
    for profile in profiles:
        if profile["support"] == "no_traversal_support":
            continue
        edge = Edge()
        edge.family = profile["family"]
        edge.gene = profile["endpoint_a"]
        edge.cancer = profile["endpoint_b"]
        edge.q = tuple(sorted(pmids.get((edge.family, edge.gene, edge.cancer), ())))
        edge.qset = frozenset(edge.q)
        edge.source = SOURCE_CODE.get(profile["support"], -1)
        if edge.family == "gene_cancer":
            edge.extend = int(weights["right_of"].get((edge.gene, edge.cancer), 0))
            edge.n_left = int(weights["left_of"].get((edge.gene, edge.cancer), 0))
            edge.candidate = int(edge.gene in candidates)
            edge.query_edge = int(edge.cancer in queries)
            GC.append(edge)
        elif profile["support"] == "semmed_only":
            GG.append(edge)
    print(f"[edges] gene-cancer {len(GC)} gene-gene {len(GG)} qualifying-pmids {len(support)}", flush=True)
    return {"weights": weights, "candidates": candidates, "support": support}


def _attach(support: set[int]) -> None:
    for edge in GC + GG:
        links = []
        seen = set()
        n_out = n_in = 0
        for pmid in edge.q:
            targets = OUTGOING.get(pmid)
            if targets is not None and targets.size:
                n_out += 1
                for target in targets:
                    target = int(target)
                    if target == pmid or target not in support:
                        continue
                    pair = (pmid, target)
                    if pair in seen:
                        continue
                    seen.add(pair)
                    links.append(pair)
            incoming = INCOMING.get(pmid)
            if incoming is not None and incoming.size:
                n_in += 1
        edge.links = tuple(links)
        edge.n_out = n_out
        edge.n_in = n_in


def _specs(weights: dict, candidates: set[str]) -> list[tuple]:
    gc_index = {}
    gg_index = {}
    by_cancer = defaultdict(list)
    by_gene = defaultdict(list)
    for index, edge in enumerate(GC):
        gc_index[(edge.gene, edge.cancer)] = index
        by_cancer[edge.cancer].append(index)
        by_gene[edge.gene].append(index)
    for index, edge in enumerate(GG):
        left, right = (edge.gene, edge.cancer) if edge.gene <= edge.cancer else (edge.cancer, edge.gene)
        gg_index[(left, right)] = index
    specs = []
    missing = 0
    for cancer, indices in by_cancer.items():
        specs.append(("cancer_12", cancer, indices))
    for gene, indices in by_gene.items():
        if len(indices) >= 2:
            specs.append(("gene_23", gene, indices))
    for gene, query_set in weights["queries_of"].items():
        partners = [item for item in weights["neighbors"].get(gene, ()) if item in candidates and item != gene]
        if not partners or not query_set:
            continue
        left_ids = []
        for partner in partners:
            left, right = (partner, gene) if partner <= gene else (gene, partner)
            found = gg_index.get((left, right))
            if found is None:
                missing += 1
            else:
                left_ids.append(found)
        right_ids = []
        for query in query_set:
            found = gc_index.get((gene, query))
            if found is None:
                missing += 1
            else:
                right_ids.append(found)
        if left_ids and right_ids:
            specs.append(("gene_t1", gene, left_ids, right_ids))
    if missing:
        raise RuntimeError(f"{missing} adjacent edges were missing from the rebuilt index")
    # template_weights does not return neighbors. The gene-gene loop above needs them.
    return specs


def _pair_iterator(kind: str, edges: list[Edge], split: int):
    if kind == "gene_t1":
        return ((left, right, 1) for left in range(split) for right in range(split, len(edges)))
    if kind == "cancer_12":
        candidates = [index for index, edge in enumerate(edges) if edge.candidate]
        extenders = [index for index, edge in enumerate(edges) if edge.extend]
        return (
            (left, right, edges[right].extend)
            for left in candidates
            for right in extenders
            if edges[left].gene != edges[right].gene
        )
    lefts = [index for index, edge in enumerate(edges) if edge.n_left]
    rights = [index for index, edge in enumerate(edges) if edge.query_edge]
    return (
        (left, right, edges[left].n_left)
        for left in lefts
        for right in rights
        if edges[left].cancer != edges[right].cancer
    )


def pattern_spec(spec: tuple) -> dict:
    started = time.time()
    kind = spec[0]
    if kind == "gene_t1":
        _kind, group_id, left_ids, right_ids = spec
        edges = [GG[index] for index in left_ids] + [GC[index] for index in right_ids]
        split = len(left_ids)
        has_block = split > 0 and split < len(edges)
    elif kind == "cancer_12":
        _kind, group_id, indices = spec
        edges = [GC[index] for index in indices]
        split = 0
        has_block = any(edge.candidate for edge in edges) and any(edge.extend for edge in edges)
    else:
        _kind, group_id, indices = spec
        edges = [GC[index] for index in indices]
        split = 0
        has_block = any(edge.n_left for edge in edges) and any(edge.query_edge for edge in edges)
    cocite = shared = None
    if has_block and len(edges) >= 2:
        pmid_lists = [edge.q for edge in edges]
        cocite = witness_matrices(pmid_lists, INCOMING, INSIDE, False, 0)
        shared = witness_matrices(pmid_lists, OUTGOING, INSIDE, False, 0)
    distinct = [0] * 16
    weighted = [0] * 16
    stats = {name: 0 for name in ("pairs", "pairs_w", "both", "both_w", "same", "same_w", "direct", "direct_w", "cocite", "cocite_w", "ref", "ref_w", "decidable", "decidable_w", "unknown", "unknown_w", "outside", "outside_w")}
    source_pattern = np.zeros((3, 3, 16), dtype=np.int64)
    source_pattern_w = np.zeros((3, 3, 16), dtype=np.int64)
    source_all = np.zeros((3, 3), dtype=np.int64)
    source_all_w = np.zeros((3, 3), dtype=np.int64)
    source_outside = np.zeros((3, 3), dtype=np.int64)
    source_outside_w = np.zeros((3, 3), dtype=np.int64)
    source_qualifying = np.zeros((3, 3), dtype=np.int64)
    source_qualifying_w = np.zeros((3, 3), dtype=np.int64)
    source_unknown = np.zeros((3, 3), dtype=np.int64)
    source_unknown_w = np.zeros((3, 3), dtype=np.int64)
    for left_index, right_index, weight in _pair_iterator(kind, edges, split):
        left = edges[left_index]
        right = edges[right_index]
        left_source = left.source
        right_source = right.source
        if left_source < 0 or right_source < 0:
            raise RuntimeError(f"missing source on {left.family} {left.gene} {left.cancer}")
        weight = int(weight)
        stats["pairs"] += 1
        stats["pairs_w"] += weight
        source_all[left_source, right_source] += 1
        source_all_w[left_source, right_source] += weight
        if not left.q or not right.q:
            stats["outside"] += 1
            stats["outside_w"] += weight
            source_outside[left_source, right_source] += 1
            source_outside_w[left_source, right_source] += weight
            continue
        stats["both"] += 1
        stats["both_w"] += weight
        source_qualifying[left_source, right_source] += 1
        source_qualifying_w[left_source, right_source] += weight
        same = intersection_size(left.q, right.q) > 0
        if left.links or right.links:
            exclusive_ltr, exclusive_rtl, ambiguous = classify_prepared_links(left.qset, right.qset, left.links, right.links)
        else:
            exclusive_ltr = exclusive_rtl = ambiguous = 0
        direct = bool(exclusive_ltr or exclusive_rtl or ambiguous)
        cocite_present = bool(cocite is not None and int(cocite["witnesses"][left_index, right_index]))
        ref_present = bool(shared is not None and int(shared["witnesses"][left_index, right_index]))
        if same:
            stats["same"] += 1
            stats["same_w"] += weight
        if direct:
            stats["direct"] += 1
            stats["direct_w"] += weight
        if cocite_present:
            stats["cocite"] += 1
            stats["cocite_w"] += weight
        if ref_present:
            stats["ref"] += 1
            stats["ref_w"] += weight
        direct_known = direct or (
            _index_state(left.n_out, len(left.q)) == "complete" and _index_state(right.n_out, len(right.q)) == "complete"
        )
        cocite_known = cocite_present or _both_index(left.n_in, len(left.q), right.n_in, len(right.q)) == "complete"
        ref_known = ref_present or _both_index(left.n_out, len(left.q), right.n_out, len(right.q)) == "complete"
        if direct_known and cocite_known and ref_known:
            pattern = int(same) | (int(direct) << 1) | (int(cocite_present) << 2) | (int(ref_present) << 3)
            distinct[pattern] += 1
            weighted[pattern] += weight
            source_pattern[left_source, right_source, pattern] += 1
            source_pattern_w[left_source, right_source, pattern] += weight
            stats["decidable"] += 1
            stats["decidable_w"] += weight
        else:
            stats["unknown"] += 1
            stats["unknown_w"] += weight
            source_unknown[left_source, right_source] += 1
            source_unknown_w[left_source, right_source] += weight
    return {
        "kind": kind,
        "group": group_id,
        "seconds": round(time.time() - started, 3),
        "edges": len(edges),
        "distinct": distinct,
        "weighted": weighted,
        "stats": stats,
        "source_pattern": source_pattern,
        "source_pattern_w": source_pattern_w,
        "source_all": source_all,
        "source_all_w": source_all_w,
        "source_outside": source_outside,
        "source_outside_w": source_outside_w,
        "source_qualifying": source_qualifying,
        "source_qualifying_w": source_qualifying_w,
        "source_unknown": source_unknown,
        "source_unknown_w": source_unknown_w,
    }


def _expected() -> dict[tuple[str, str], tuple[int, int]]:
    rows = {}
    with (CITATION / "analysis" / "adjacent_citation_summary.tsv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows[(row["template"], row["positions"], row["measure"])] = (int(row["distinct_count"]), int(row["path_occurrence_count"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    global OUTGOING, INCOMING
    prepared = _prepare()
    # Neighbors are recovered from the gene-gene edges just built.
    neighbors = defaultdict(set)
    for edge in GG:
        neighbors[edge.gene].add(edge.cancer)
        neighbors[edge.cancer].add(edge.gene)
    prepared["weights"]["neighbors"] = neighbors
    specs = _specs(prepared["weights"], prepared["candidates"])
    print(f"[groups] {len(specs)}", flush=True)
    OUTGOING, INCOMING = _index_maps(CITATION / "analysis" / "directed_citation_links.parquet")
    _attach(prepared["support"])
    print("[links] attached", flush=True)
    totals = {kind: {"distinct": [0] * 16, "weighted": [0] * 16, "stats": defaultdict(int)} for kind in POSITIONS}
    started = time.time()
    workers = max(1, args.workers)
    if workers == 1:
        results = (pattern_spec(spec) for spec in specs)
        pool = None
    else:
        pool = mp.get_context("fork").Pool(workers)
        results = pool.imap_unordered(pattern_spec, specs, chunksize=1)
    try:
        for number, part in enumerate(results, start=1):
            bucket = totals[part["kind"]]
            for index, amount in enumerate(part["distinct"]):
                bucket["distinct"][index] += amount
                bucket["weighted"][index] += part["weighted"][index]
            for name, amount in part["stats"].items():
                bucket["stats"][name] += amount
            if number % 500 == 0 or part["edges"] >= 2000:
                print(f"[groups] {number}/{len(specs)} {part['kind']} edges={part['edges']} seconds={part['seconds']} elapsed={round(time.time()-started,1)}", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    expected = _expected()
    failures = []
    rows = []
    for kind, position in POSITIONS.items():
        template, slot = TEMPLATES[kind]
        stats = totals[kind]["stats"]
        checks = {
            "pairs": "pairs",
            "both": "both",
            "shared_q_pairs": "same",
            "collapsed_pairs": "direct",
            "cocite_pairs": "cocite",
            "ref_pairs": "ref",
        }
        for measure, local in checks.items():
            observed = (stats[local], stats[local + "_w"])
            wanted = expected[(template, slot, measure)]
            if observed != wanted:
                failures.append({"kind": kind, "measure": measure, "observed": list(observed), "expected": list(wanted)})
        if sum(totals[kind]["distinct"]) != stats["decidable"] or sum(totals[kind]["weighted"]) != stats["decidable_w"]:
            failures.append({"kind": kind, "measure": "pattern_total", "observed": [sum(totals[kind]["distinct"]), sum(totals[kind]["weighted"])], "expected": [stats["decidable"], stats["decidable_w"]]})
        for pattern in range(16):
            rows.append(
                {
                    "position": position,
                    "pattern": pattern,
                    "same_article": pattern & 1,
                    "direct_citation": (pattern >> 1) & 1,
                    "co_citation": (pattern >> 2) & 1,
                    "shared_reference": (pattern >> 3) & 1,
                    "distinct_count": totals[kind]["distinct"][pattern],
                    "weighted_count": totals[kind]["weighted"][pattern],
                    "decidable_distinct": stats["decidable"],
                    "decidable_weighted": stats["decidable_w"],
                    "unknown_distinct": stats["unknown"],
                    "unknown_weighted": stats["unknown_w"],
                    "outside_distinct": stats["outside"],
                    "outside_weighted": stats["outside_w"],
                    "all_pairs_distinct": stats["pairs"],
                    "all_pairs_weighted": stats["pairs_w"],
                }
            )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with (args.out_dir / "figure6_article_combinations.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    status = {"status": "passed" if not failures else "failed", "failures": failures}
    (args.out_dir / "figure6_validation.json").write_text(__import__("json").dumps(status, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(1)
    print("[done] article-link combinations", flush=True)


if __name__ == "__main__":
    main()

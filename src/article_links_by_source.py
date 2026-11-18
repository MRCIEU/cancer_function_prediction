"""Article-link patterns crossed with the source of each adjacent edge.

The pair walk is the Figure 6 combination count. Each ordered pair also keeps
the joint-graph support of its left and right edges. N is the count of pairs
whose two edges both have a qualifying PMID. D is the subset for which all
four article relations are decidable. The mosaic uses D.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.article_link_combinations import (
    GG,
    POSITIONS,
    SOURCES,
    TEMPLATES,
    _attach,
    _expected,
    _index_maps,
    _prepare,
    _specs,
    pattern_spec,
)
from src.plot_figure6_by_source import plot_figure
import src.article_link_combinations as combinations

REFERENCE = Path(
    "/home/b5ac/freddieyu.b5ac/projects/project_2/results/scientific_figures_20261117T0219Z/data/figure6_article_combinations.tsv"
)
CITATION = combinations.CITATION
TRIPLE_PATTERN = 14
TRIPLE_SHARE_TO_SEPARATE = 0.05


def _category(pattern: int, separate_triple: bool) -> str:
    if pattern == 0:
        return "No observed link"
    if pattern == 8:
        return "Shared reference only"
    if pattern == 12:
        return "Co-citation + shared reference"
    if pattern == TRIPLE_PATTERN and separate_triple:
        return "Citation + co-citation + shared reference"
    if pattern & 1:
        return "Any shared PMID"
    return "Other observed links"


def _bits(pattern: int) -> str:
    return "".join(str((pattern >> shift) & 1) for shift in range(4))


def _write_tsv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _reference_counts() -> dict[str, dict[int, int]]:
    counts: dict[str, dict[int, int]] = defaultdict(dict)
    with REFERENCE.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            counts[row["position"]][int(row["pattern"])] = int(row["distinct_count"])
    return counts


def _accumulate(workers: int) -> tuple[dict, dict]:
    prepared = _prepare()
    neighbors = defaultdict(set)
    for edge in GG:
        neighbors[edge.gene].add(edge.cancer)
        neighbors[edge.cancer].add(edge.gene)
    prepared["weights"]["neighbors"] = neighbors
    specs = _specs(prepared["weights"], prepared["candidates"])
    print(f"[groups] {len(specs)}", flush=True)
    combinations.OUTGOING, combinations.INCOMING = _index_maps(CITATION / "analysis" / "directed_citation_links.parquet")
    _attach(prepared["support"])
    print("[links] attached", flush=True)
    totals = {kind: {"distinct": [0] * 16, "weighted": [0] * 16, "stats": defaultdict(int)} for kind in POSITIONS}
    source = {
        kind: {
            "pattern": np.zeros((3, 3, 16), dtype=np.int64),
            "pattern_w": np.zeros((3, 3, 16), dtype=np.int64),
            "all": np.zeros((3, 3), dtype=np.int64),
            "all_w": np.zeros((3, 3), dtype=np.int64),
            "outside": np.zeros((3, 3), dtype=np.int64),
            "outside_w": np.zeros((3, 3), dtype=np.int64),
            "qualifying": np.zeros((3, 3), dtype=np.int64),
            "qualifying_w": np.zeros((3, 3), dtype=np.int64),
            "unknown": np.zeros((3, 3), dtype=np.int64),
            "unknown_w": np.zeros((3, 3), dtype=np.int64),
        }
        for kind in POSITIONS
    }
    started = time.time()
    pool = None
    if workers == 1:
        results = (pattern_spec(spec) for spec in specs)
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
            block = source[part["kind"]]
            for key in block:
                block[key] += part[{"pattern": "source_pattern", "pattern_w": "source_pattern_w", "all": "source_all", "all_w": "source_all_w", "outside": "source_outside", "outside_w": "source_outside_w", "qualifying": "source_qualifying", "qualifying_w": "source_qualifying_w", "unknown": "source_unknown", "unknown_w": "source_unknown_w"}[key]]
            if number % 500 == 0 or part["edges"] >= 2000:
                print(
                    f"[groups] {number}/{len(specs)} {part['kind']} edges={part['edges']} seconds={part['seconds']} elapsed={round(time.time() - started, 1)}",
                    flush=True,
                )
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return totals, source


def _check(totals: dict, source: dict) -> tuple[list[dict], dict]:
    expected = _expected()
    reference = _reference_counts()
    failures = []
    shares = {}
    for kind, position in POSITIONS.items():
        template, slot = TEMPLATES[kind]
        stats = totals[kind]["stats"]
        block = source[kind]
        checks = {
            "pairs": "pairs",
            "both": "both",
            "shared_q_pairs": "same",
            "collapsed_pairs": "direct",
            "cocite_pairs": "cocite",
            "ref_pairs": "ref",
        }
        for measure, local in checks.items():
            observed = (int(stats[local]), int(stats[local + "_w"]))
            wanted = expected[(template, slot, measure)]
            if observed != wanted:
                failures.append({"kind": kind, "measure": measure, "observed": list(observed), "expected": list(wanted)})
        pattern_sum = block["pattern"].sum(axis=(0, 1))
        pattern_sum_w = block["pattern_w"].sum(axis=(0, 1))
        for pattern in range(16):
            if int(pattern_sum[pattern]) != int(totals[kind]["distinct"][pattern]):
                failures.append({"kind": kind, "measure": f"source_pattern_{pattern}", "observed": int(pattern_sum[pattern]), "expected": int(totals[kind]["distinct"][pattern])})
            if int(pattern_sum[pattern]) != reference[position][pattern]:
                failures.append({"kind": kind, "measure": f"reference_pattern_{pattern}", "observed": int(pattern_sum[pattern]), "expected": reference[position][pattern]})
        decidable = int(block["pattern"].sum())
        qualifying = int(block["qualifying"].sum())
        unknown = int(block["unknown"].sum())
        outside = int(block["outside"].sum())
        all_pairs = int(block["all"].sum())
        if decidable != int(stats["decidable"]) or qualifying != int(stats["both"]) or unknown != int(stats["unknown"]) or outside != int(stats["outside"]) or all_pairs != int(stats["pairs"]):
            failures.append({"kind": kind, "measure": "source_margin", "observed": [all_pairs, qualifying, decidable, unknown, outside], "expected": [int(stats["pairs"]), int(stats["both"]), int(stats["decidable"]), int(stats["unknown"]), int(stats["outside"])]})
        if qualifying != decidable + unknown or all_pairs != qualifying + outside:
            failures.append({"kind": kind, "measure": "partition", "observed": [all_pairs, qualifying, decidable, unknown, outside], "expected": "all = N + outside and N = D + unknown"})
        if int(pattern_sum_w.sum()) != int(stats["decidable_w"]):
            failures.append({"kind": kind, "measure": "weighted_pattern_total", "observed": int(pattern_sum_w.sum()), "expected": int(stats["decidable_w"])})
        for left in range(3):
            for right in range(3):
                counted = int(block["pattern"][left, right].sum())
                if counted != int(block["qualifying"][left, right] - block["unknown"][left, right]):
                    failures.append({"kind": kind, "measure": "cell_decidable", "observed": [SOURCES[left], SOURCES[right], counted], "expected": int(block["qualifying"][left, right] - block["unknown"][left, right])})
        if kind == "gene_t1":
            leaked = int(block["all"][:2].sum())
            if leaked:
                failures.append({"kind": kind, "measure": "two_edge_left_source", "observed": leaked, "expected": 0})
        shares[position] = (int(pattern_sum[TRIPLE_PATTERN]) / decidable) if decidable else 0.0
    separate_triple = max(shares.values(), default=0.0) >= TRIPLE_SHARE_TO_SEPARATE
    return failures, {"triple_pattern_share": shares, "separate_triple": separate_triple}


def _rows(source: dict, separate_triple: bool) -> tuple[list[dict], list[dict], list[dict]]:
    cross = []
    coverage = []
    mapping = []
    for pattern in range(16):
        mapping.append(
            {
                "pattern": pattern,
                "bits": _bits(pattern),
                "same_article": pattern & 1,
                "direct_citation": (pattern >> 1) & 1,
                "co_citation": (pattern >> 2) & 1,
                "shared_reference": (pattern >> 3) & 1,
                "category": _category(pattern, separate_triple),
                "bit_order": "same_article, direct_citation, co_citation, shared_reference",
            }
        )
    for kind, position in POSITIONS.items():
        block = source[kind]
        position_all = int(block["all"].sum())
        position_n = int(block["qualifying"].sum())
        position_d = int(block["pattern"].sum())
        coverage.append(
            {
                "position": position,
                "left_source": "ALL",
                "right_source": "ALL",
                "all_pairs_distinct": position_all,
                "all_pairs_weighted": int(block["all_w"].sum()),
                "outside_distinct": int(block["outside"].sum()),
                "outside_weighted": int(block["outside_w"].sum()),
                "qualifying_both_distinct": position_n,
                "qualifying_both_weighted": int(block["qualifying_w"].sum()),
                "decidable_distinct": position_d,
                "decidable_weighted": int(block["pattern_w"].sum()),
                "unknown_distinct": int(block["unknown"].sum()),
                "unknown_weighted": int(block["unknown_w"].sum()),
                "coverage": (position_d / position_n) if position_n else "",
                "plot_status": "position_total",
                "qualification": "N counts distinct ordered pairs with a qualifying PMID on both edges. Pairs missing a qualifying PMID are outside, not in N.",
            }
        )
        for left, left_name in enumerate(SOURCES):
            for right, right_name in enumerate(SOURCES):
                all_pairs = int(block["all"][left, right])
                outside = int(block["outside"][left, right])
                qualifying = int(block["qualifying"][left, right])
                unknown = int(block["unknown"][left, right])
                decidable = int(block["pattern"][left, right].sum())
                if qualifying == 0 and all_pairs == 0:
                    status = "empty"
                elif decidable == 0:
                    status = "no_decidable_pairs"
                else:
                    status = "plotted"
                coverage.append(
                    {
                        "position": position,
                        "left_source": left_name,
                        "right_source": right_name,
                        "all_pairs_distinct": all_pairs,
                        "all_pairs_weighted": int(block["all_w"][left, right]),
                        "outside_distinct": outside,
                        "outside_weighted": int(block["outside_w"][left, right]),
                        "qualifying_both_distinct": qualifying,
                        "qualifying_both_weighted": int(block["qualifying_w"][left, right]),
                        "decidable_distinct": decidable,
                        "decidable_weighted": int(block["pattern_w"][left, right].sum()),
                        "unknown_distinct": unknown,
                        "unknown_weighted": int(block["unknown_w"][left, right]),
                        "coverage": (decidable / qualifying) if qualifying else "",
                        "plot_status": status,
                        "qualification": "N counts distinct ordered pairs with a qualifying PMID on both edges. Pairs missing a qualifying PMID are outside, not in N.",
                    }
                )
                for pattern in range(16):
                    cross.append(
                        {
                            "position": position,
                            "left_source": left_name,
                            "right_source": right_name,
                            "pattern": pattern,
                            "bits": _bits(pattern),
                            "same_article": pattern & 1,
                            "direct_citation": (pattern >> 1) & 1,
                            "co_citation": (pattern >> 2) & 1,
                            "shared_reference": (pattern >> 3) & 1,
                            "category": _category(pattern, separate_triple),
                            "distinct_count": int(block["pattern"][left, right, pattern]),
                            "weighted_count": int(block["pattern_w"][left, right, pattern]),
                            "decidable_distinct": decidable,
                            "qualifying_both_distinct": qualifying,
                            "position_decidable_distinct": position_d,
                        }
                    )
    return cross, coverage, mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    totals, source = _accumulate(max(1, args.workers))
    failures, decision = _check(totals, source)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = args.out_dir / "data"
    data.mkdir(parents=True, exist_ok=True)
    cross, coverage, mapping = _rows(source, bool(decision["separate_triple"]))
    _write_tsv(data / "figure6_source_cross.tsv", cross)
    _write_tsv(data / "figure6_source_coverage.tsv", coverage)
    _write_tsv(data / "figure6_source_category_map.tsv", mapping)
    status = {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "reference": str(REFERENCE),
        "n_definition": "Distinct ordered adjacent pairs with a qualifying PMID on both edges.",
        "d_definition": "Subset of N for which same article, direct citation, co-citation, and shared reference are all decidable.",
        "triple_pattern_share": decision["triple_pattern_share"],
        "separate_triple": decision["separate_triple"],
        "figure4_reconciliation": "Citation marginals on the qualifying-PMID denominator were checked against adjacent_citation_summary.tsv. Figure 4 keeps pairs with partial coverage in its citation denominator and uses the union PMID denominator for same article, so its percentages are not the mosaic shares.",
    }
    (data / "figure6_source_validation.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(1)
    plot_figure(args.out_dir)
    print("[done] article links by source", flush=True)


if __name__ == "__main__":
    main()

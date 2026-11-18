"""Pair-equal and path-equal source sequences for descriptive figures.

The published sequence tables count every complete path, including pairs that
already have a direct record. This script keeps that population for validation,
then restricts both weights to joint-reachable pairs with no direct record.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from src.complete_audit import (
    sequence_group,
    three_edge_source_sequences,
    two_edge_source_sequences,
)
from src.run_complete_audit import labelled_from_profiles

STATE_INDEX = {"core_only": 0, "both_sources": 1, "semmed_only": 2}
STATE_NAME = ("core_only", "both_sources", "semmed_only")
GRAPH_RUN = Path("/home/b5ac/freddieyu.b5ac/projects/project_2/results/joint_graph_and_evidence_audit_20261116T0404Z")
PATH_AUDIT = Path("/home/b5ac/freddieyu.b5ac/projects/project_2/results/path_evidence_audit_20261116T1310Z")
COMPLETION = Path("/home/b5ac/freddieyu.b5ac/projects/project_2/results/path_evidence_audit_completion_20261116T1402Z")


def _sequences(length: int) -> list[tuple[str, ...]]:
    if length == 2:
        return [("semmed_only", state) for state in STATE_NAME]
    rows = []
    for left in STATE_NAME:
        for middle in STATE_NAME:
            for right in STATE_NAME:
                rows.append((left, middle, right))
    return rows


def _compatible(sequence: tuple[str, ...], forbidden: str) -> bool:
    return forbidden not in sequence


def _load_scope() -> tuple[dict, dict, set[str], set[str]]:
    profiles = pq.read_table(
        PATH_AUDIT / "analysis" / "edge_profiles.parquet",
        columns=["family", "endpoint_a", "endpoint_b", "support"],
    ).to_pylist()
    genes, neighbors = labelled_from_profiles(profiles, "joint")
    candidates = {
        row["gene_id"]
        for row in pq.read_table(
            GRAPH_RUN / "analysis" / "candidate_genes.parquet", columns=["gene_id", "status"]
        ).to_pylist()
        if row["status"] == "resolved_human"
    }
    queries = {
        row["cancer_cui"]
        for row in pq.read_table(GRAPH_RUN / "analysis" / "cancer_queries.parquet", columns=["cancer_cui"]).to_pylist()
    }
    return genes, neighbors, candidates, queries


def _candidate_order(candidates: set[str]) -> list[str]:
    return sorted(candidates, key=lambda gene: (len(gene), gene))


def _prepare(genes: dict, candidates: set[str], queries: set[str]):
    ordered = _candidate_order(candidates)
    slot_of = {gene: index for index, gene in enumerate(ordered)}
    cancers_of: dict[str, list[tuple[str, int]]] = {}
    direct_slots: dict[str, np.ndarray] = {}
    groups: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for cancer, members in genes.items():
        by_state: list[list[int]] = [[], [], []]
        direct: list[int] = []
        for gene, support in members.items():
            state = STATE_INDEX[support]
            cancers_of.setdefault(gene, []).append((cancer, state))
            slot = slot_of.get(gene)
            if slot is None:
                continue
            by_state[state].append(slot)
            if cancer in queries:
                direct.append(slot)
        groups[cancer] = tuple(np.asarray(slots, dtype=np.int32) for slots in by_state)
        if cancer in queries:
            direct_slots[cancer] = np.asarray(direct, dtype=np.int32)
    return ordered, cancers_of, groups, direct_slots


def _empty_accumulator(length: int) -> dict:
    sequences = _sequences(length)
    return {
        "sequences": sequences,
        "path": np.zeros(len(sequences), dtype=np.int64),
        "pair_mass": np.zeros(len(sequences), dtype=np.float64),
        "pair_support": np.zeros(len(sequences), dtype=np.int64),
        "paths": 0,
        "pairs": 0,
        "reach": {"core_only": 0, "semmed_only": 0, "either": 0, "joint_only": 0},
    }


def _consume_block(counts: np.ndarray, direct: np.ndarray, sequences: list[tuple[str, ...]], acc: dict) -> None:
    if direct.size:
        counts[direct, :] = 0
    totals = counts.sum(axis=1)
    keep = np.flatnonzero(totals > 0)
    if keep.size == 0:
        return
    kept = counts[keep]
    kept_totals = totals[keep]
    acc["paths"] += int(kept_totals.sum())
    acc["pairs"] += int(keep.size)
    acc["path"] += kept.sum(axis=0)
    shares = kept / kept_totals[:, None]
    acc["pair_mass"] += shares.sum(axis=0)
    acc["pair_support"] += (kept > 0).sum(axis=0)
    core_ok = np.zeros(keep.size, dtype=bool)
    semmed_ok = np.zeros(keep.size, dtype=bool)
    for index, sequence in enumerate(sequences):
        column = kept[:, index] > 0
        if _compatible(sequence, "semmed_only"):
            core_ok |= column
        if _compatible(sequence, "core_only"):
            semmed_ok |= column
    acc["reach"]["either"] += int((core_ok & semmed_ok).sum())
    acc["reach"]["core_only"] += int((core_ok & ~semmed_ok).sum())
    acc["reach"]["semmed_only"] += int((~core_ok & semmed_ok).sum())
    acc["reach"]["joint_only"] += int((~core_ok & ~semmed_ok).sum())


def _three_edge_query(
    query: str,
    genes,
    cancers_of,
    groups,
    slot_of,
    counts: np.ndarray,
) -> None:
    block: dict[str, np.ndarray] = {}
    sub_slot: list[int] = []
    sub_seq: list[int] = []
    for gene, support in genes.get(query, {}).items():
        right_state = STATE_INDEX[support]
        for cancer, middle_state in cancers_of.get(gene, ()):
            if cancer == query:
                continue
            arr = block.get(cancer)
            if arr is None:
                arr = np.zeros((3, 3), dtype=np.int32)
                block[cancer] = arr
            arr[middle_state, right_state] += 1
            slot = slot_of.get(gene)
            if slot is not None:
                sub_slot.append(slot)
                sub_seq.append(middle_state * 9 + middle_state * 3 + right_state)
    for cancer, arr in block.items():
        flat = arr.ravel()
        for state, slots in enumerate(groups.get(cancer, ())):
            if slots.size == 0:
                continue
            base = state * 9
            for offset, amount in enumerate(flat):
                if amount:
                    counts[slots, base + offset] += int(amount)
    if sub_slot:
        np.add.at(counts, (np.asarray(sub_slot, dtype=np.int32), np.asarray(sub_seq, dtype=np.int32)), -1)


def accumulate_three(genes, slot_of, cancers_of, groups, direct_slots, queries, mask_direct: bool) -> dict:
    counts = np.zeros((len(slot_of), 27), dtype=np.int64)
    acc = _empty_accumulator(3)
    for number, query in enumerate(sorted(queries), start=1):
        counts.fill(0)
        _three_edge_query(query, genes, cancers_of, groups, slot_of, counts)
        if int(counts.min()) < 0:
            raise RuntimeError(f"negative three-edge count at query {query}")
        direct = direct_slots.get(query, np.asarray([], dtype=np.int32)) if mask_direct else np.asarray([], dtype=np.int32)
        _consume_block(counts, direct, acc["sequences"], acc)
        if number % 200 == 0:
            print(f"[three-edge] {number} queries, pairs={acc['pairs']}", flush=True)
    return acc


def accumulate_two(genes, neighbors, slot_of, direct_slots, queries, mask_direct: bool) -> dict:
    counts = np.zeros((len(slot_of), 3), dtype=np.int64)
    acc = _empty_accumulator(2)
    for number, query in enumerate(sorted(queries), start=1):
        counts.fill(0)
        for gene, support in genes.get(query, {}).items():
            state = STATE_INDEX[support]
            for partner in neighbors.get(gene, ()):
                slot = slot_of.get(partner)
                if slot is None or partner == gene:
                    continue
                counts[slot, state] += 1
        direct = direct_slots.get(query, np.asarray([], dtype=np.int32)) if mask_direct else np.asarray([], dtype=np.int32)
        _consume_block(counts, direct, acc["sequences"], acc)
        if number % 400 == 0:
            print(f"[two-edge] {number} queries, pairs={acc['pairs']}", flush=True)
    return acc


def _rows(acc: dict, population: str) -> list[dict]:
    paths = int(acc["paths"])
    pairs = int(acc["pairs"])
    rows = []
    for index, sequence in enumerate(acc["sequences"]):
        path_count = int(acc["path"][index])
        pair_mass = float(acc["pair_mass"][index])
        path_prop = (path_count / paths) if paths else 0.0
        pair_prop = (pair_mass / pairs) if pairs else 0.0
        record = {
            "population": population,
            "length": len(sequence),
            "position_1": sequence[0],
            "position_2": sequence[1],
            "position_3": sequence[2] if len(sequence) == 3 else "",
            "sequence_group": sequence_group(sequence),
            "path_count": path_count,
            "pairs_with_sequence": int(acc["pair_support"][index]),
            "path_proportion": path_prop,
            "pair_proportion": pair_prop,
            "difference_percentage_points": (path_prop - pair_prop) * 100.0,
            "denominator_paths": paths,
            "denominator_pairs": pairs,
        }
        rows.append(record)
    return rows


def _published_three(genes, candidates, queries) -> Counter:
    return three_edge_source_sequences(genes, candidates, queries)


def _published_two(genes, neighbors, candidates, queries) -> Counter:
    return two_edge_source_sequences(genes, neighbors, candidates, queries)


def _check(failures: list[dict], name: str, observed, expected) -> None:
    if int(observed) != int(expected):
        failures.append({"name": name, "observed": int(observed), "expected": int(expected)})


def _dependence_expected() -> dict[str, dict[str, int]]:
    expected: dict[str, dict[str, int]] = {}
    path = COMPLETION / "analysis" / "candidate_dependence.tsv"
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        population, category, pairs = line.split("\t")
        expected.setdefault(population, {})[category] = int(pairs)
    return expected


def _synthetic_check() -> None:
    genes = {
        "K": {"g1": "core_only", "g2": "semmed_only", "h": "both_sources"},
        "Q": {"h": "core_only", "g1": "semmed_only"},
    }
    candidates = {"g1", "g2"}
    queries = {"Q"}
    published = three_edge_source_sequences(genes, candidates, queries)
    ordered, cancers_of, groups, direct_slots = _prepare(genes, candidates, queries)
    slot_of = {gene: index for index, gene in enumerate(ordered)}
    counts = np.zeros((len(ordered), 27), dtype=np.int64)
    _three_edge_query("Q", genes, cancers_of, groups, slot_of, counts)
    if int(counts.min()) < 0:
        raise RuntimeError("synthetic three-edge count went negative")
    total = int(counts.sum())
    if total != int(sum(published.values())) or total != 3:
        raise RuntimeError(f"synthetic three-edge total {total}, published {dict(published)}")
    # g1 has a direct record and is removed; g2 keeps both of its paths.
    _consume_block(counts.copy(), direct_slots["Q"], _sequences(3), acc := _empty_accumulator(3))
    if acc["pairs"] != 1 or acc["paths"] != 2:
        raise RuntimeError(f"synthetic no-direct mask failed: {acc['pairs']} pairs, {acc['paths']} paths")


def write_tsv(path: Path, rows: list[dict]) -> None:
    columns = list(rows[0])
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join(str(row[column]) for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    print("[check] synthetic", flush=True)
    _synthetic_check()
    print("[load] profiles", flush=True)
    genes, neighbors, candidates, queries = _load_scope()
    ordered, cancers_of, groups, direct_slots = _prepare(genes, candidates, queries)
    slot_of = {gene: index for index, gene in enumerate(ordered)}
    failures: list[dict] = []
    print("[validate] published sequence counters", flush=True)
    published_three = _published_three(genes, candidates, queries)
    published_two = _published_two(genes, neighbors, candidates, queries)
    _check(failures, "published_three_paths", sum(published_three.values()), 14_437_428_485)
    _check(failures, "published_two_paths", sum(published_two.values()), 18_414_593)
    print("[accumulate] unmasked", flush=True)
    three_all = accumulate_three(genes, slot_of, cancers_of, groups, direct_slots, queries, False)
    two_all = accumulate_two(genes, neighbors, slot_of, direct_slots, queries, False)
    for index, sequence in enumerate(three_all["sequences"]):
        _check(failures, "three_seq_" + "_".join(sequence), three_all["path"][index], published_three.get(sequence, 0))
    for index, sequence in enumerate(two_all["sequences"]):
        _check(failures, "two_seq_" + "_".join(sequence), two_all["path"][index], published_two.get(sequence, 0))
    expected = _dependence_expected()
    for length, acc, masked_name, all_name in (
        (3, three_all, "three_no_direct", "three_all"),
        (2, two_all, "two_no_direct", "two_all"),
    ):
        _check(failures, f"{all_name}_paths", acc["paths"], 14_437_428_485 if length == 3 else 18_414_593)
        _check(failures, f"{all_name}_pairs", acc["pairs"], expected[all_name]["joint_reached"])
        for reach_name, column in (
            ("core_only", "core_only_reach"),
            ("semmed_only", "semmed_only_reach"),
            ("either", "either_resource_reach"),
            ("joint_only", "joint_only_reach"),
        ):
            _check(failures, f"{all_name}_{reach_name}", acc["reach"][reach_name], expected[all_name][column])
    print("[accumulate] no direct record", flush=True)
    three_open = accumulate_three(genes, slot_of, cancers_of, groups, direct_slots, queries, True)
    two_open = accumulate_two(genes, neighbors, slot_of, direct_slots, queries, True)
    for length, acc, all_name in ((3, three_open, "three_no_direct"), (2, two_open, "two_no_direct")):
        _check(failures, f"{all_name}_pairs", acc["pairs"], expected[all_name]["joint_reached"])
        for reach_name, column in (
            ("core_only", "core_only_reach"),
            ("semmed_only", "semmed_only_reach"),
            ("either", "either_resource_reach"),
            ("joint_only", "joint_only_reach"),
        ):
            _check(failures, f"{all_name}_{reach_name}", acc["reach"][reach_name], expected[all_name][column])
        path_share = float(acc["path"].sum()) / acc["paths"] if acc["paths"] else 0
        pair_share = float(acc["pair_mass"].sum()) / acc["pairs"] if acc["pairs"] else 0
        if abs(path_share - 1) > 1e-9 or abs(pair_share - 1) > 1e-8:
            failures.append({"name": f"{all_name}_weight_total", "path_share": path_share, "pair_share": pair_share})
    rows = _rows(two_open, "no_direct") + _rows(three_open, "no_direct")
    write_tsv(out / "source_pattern_weights.tsv", rows)
    validation = {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "two_edge_no_direct": {"paths": two_open["paths"], "pairs": two_open["pairs"], "reach": two_open["reach"]},
        "three_edge_no_direct": {"paths": three_open["paths"], "pairs": three_open["pairs"], "reach": three_open["reach"]},
        "two_edge_all": {"paths": two_all["paths"], "pairs": two_all["pairs"], "reach": two_all["reach"]},
        "three_edge_all": {"paths": three_all["paths"], "pairs": three_all["pairs"], "reach": three_all["reach"]},
        "scope": {
            "candidates": len(candidates),
            "queries": len(queries),
            "rule": "Joint-view eligible edges. Repeated genes and repeated cancers excluded. No-direct rows drop a candidate–query pair when an eligible direct edge exists.",
        },
    }
    (out / "source_pattern_validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(1)
    print("[done] source patterns", flush=True)


if __name__ == "__main__":
    main()

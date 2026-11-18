"""Closed-form development scores against enumerated paths."""

from __future__ import annotations

import unittest

import numpy as np

from src.association_recovery import citation_roles_partition
from src.recovery_scope import (
    LazyTuples,
    aggregate_singleton_block,
    brute_three_edge,
    brute_two_edge,
    build_motif_block,
    candidate_slot_table,
    citation_between,
    sum_adjacent_motifs,
    select_development_cancers,
    stratified_targets,
    targets_per_cancer,
    three_edge_path_count,
    three_edge_scores,
    two_edge_scores,
)


class RecoveryScopeTests(unittest.TestCase):
    def setUp(self):
        self.candidates = {"C1", "C2", "C3"}
        self.index = {"C1": 0, "C2": 1, "C3": 2}
        self.genes_of = {
            "Q": {"G", "C1"},
            "K": {"C1", "C2", "G"},
            "M": {"C3", "G"},
        }
        self.neighbors = {
            "G": {"C1", "C2"},
            "C1": {"G", "C3"},
            "C2": {"G"},
            "C3": {"C1"},
        }
        self.cancers_of = {
            "G": {"Q", "K", "M"},
            "C1": {"Q", "K"},
            "C2": {"K"},
            "C3": {"M"},
        }
        self.slots = candidate_slot_table(self.genes_of, self.index)

    def test_two_edge_matches_enumeration(self):
        counts, weighted, walk = two_edge_scores(self.genes_of, self.neighbors, "Q", self.candidates, self.index)
        raw, raw_weighted, raw_walk = brute_two_edge(self.genes_of, self.neighbors, "Q", self.candidates)
        for gene, slot in self.index.items():
            self.assertEqual(counts[slot], raw[gene])
            self.assertAlmostEqual(weighted[slot], raw_weighted[gene])
            self.assertAlmostEqual(walk[slot], raw_walk[gene])

    def test_three_edge_matches_enumeration(self):
        counts, weighted, walk = three_edge_scores(
            self.genes_of, self.cancers_of, "Q", self.candidates, self.index, self.slots
        )
        raw, raw_walk = brute_three_edge(self.genes_of, self.cancers_of, "Q", self.candidates)
        other, _weighted = three_edge_path_count(self.genes_of, "Q", self.candidates, self.index, self.slots)
        for gene, slot in self.index.items():
            self.assertEqual(counts[slot], raw[gene])
            self.assertEqual(other[slot], raw[gene])
            self.assertAlmostEqual(walk[slot], raw_walk[gene])
            self.assertGreaterEqual(weighted[slot], 0)

    def test_repeated_intermediate_is_excluded(self):
        counts, _weighted, _walk = three_edge_scores(
            self.genes_of, self.cancers_of, "Q", self.candidates, self.index, self.slots
        )
        # C1 is on K and on Q. Paths C1-K-C1-Q are not allowed, and C1-Q-*-Q repeats the query.
        self.assertGreaterEqual(counts[self.index["C2"]], 1)
        self.assertNotEqual(counts[self.index["C1"]], counts[self.index["C2"]] + len(self.genes_of["K"]))

    def test_cohort_rule_is_deterministic(self):
        rows = []
        for number in range(30):
            rows.append({
                "cancer": f"C{number:04d}",
                "n_core": 20 + (number % 50),
                "paths": 200_000 + number,
                "two_reached": 40,
            })
        first = select_development_cancers(rows, count=6)
        second = select_development_cancers(rows, count=6)
        self.assertEqual(first["cancers"], second["cancers"])
        self.assertLessEqual(len(first["cancers"]), 6)

    def test_target_draw_keeps_background(self):
        self.assertEqual(targets_per_cancer(20), 10)
        self.assertEqual(targets_per_cancer(80), 24)
        pairs = [
            {"target_id": f"{gene}|Q", "gene": gene, "cancer": "Q", "semmed_direct": gene in {"1", "2"}}
            for gene in [str(item) for item in range(10)]
        ]
        chosen = stratified_targets(pairs, 4, seed=5)
        self.assertEqual(len(chosen), 4)
        self.assertEqual(chosen, stratified_targets(pairs, 4, seed=5))

    def test_citation_direction_partition(self):
        outgoing = LazyTuples({
            1: np.array([2, 3], dtype=np.int64),
            2: np.array([1], dtype=np.int64),
        })
        incoming = LazyTuples({
            1: np.array([9], dtype=np.int64),
            2: np.array([9, 10], dtype=np.int64),
        })
        motifs = citation_between((1,), (2,), outgoing, incoming)
        self.assertIsNotNone(motifs)
        self.assertTrue(citation_roles_partition(
            motifs["direct_ltr"], motifs["direct_rtl"], motifs["direct_ambiguous"], motifs["direct_collapsed"]
        ))
        self.assertGreaterEqual(motifs["cocite"], 1)
        self.assertIsNone(citation_between((), (2,), outgoing, incoming))

    def test_singleton_block_matches_pairwise_motifs(self):
        pmids = [10, 20, 30]
        outgoing = {
            10: np.array([20], dtype=np.int64),
            20: np.array([10, 99], dtype=np.int64),
            30: np.array([99], dtype=np.int64),
        }
        incoming = {
            10: np.array([7], dtype=np.int64),
            20: np.array([7, 8], dtype=np.int64),
            30: np.array([8], dtype=np.int64),
        }
        got = aggregate_singleton_block(pmids, outgoing, incoming, [1, 2], [0, 1])
        view_out = LazyTuples(outgoing)
        view_in = LazyTuples(incoming)
        manual = {"direct_ltr": 0, "direct_rtl": 0, "direct_collapsed": 0, "cocite": 0, "shared_references": 0}
        for partner in (1, 2):
            motifs = citation_between((pmids[0],), (pmids[partner],), view_out, view_in)
            for name in manual:
                manual[name] += motifs[name]
        self.assertEqual(got["direct_ltr"][0], manual["direct_ltr"])
        self.assertEqual(got["direct_rtl"][0], manual["direct_rtl"])
        self.assertEqual(got["direct_collapsed"][0], manual["direct_collapsed"])
        self.assertEqual(got["cocite"][0], manual["cocite"])
        self.assertEqual(got["shared_references"][0], manual["shared_references"])
        self.assertEqual(got["direct_ambiguous"][0], 0)
        self.assertTrue(citation_roles_partition(
            int(got["direct_ltr"][0]), int(got["direct_rtl"][0]), int(got["direct_ambiguous"][0]), int(got["direct_collapsed"][0])
        ))

    def test_motif_sums_match_pairwise_calls(self):
        pmid_sets = [(10, 11), (10,), (20, 30), (), (40,), (11, 40)]
        outgoing = {
            10: np.array([11, 20, 10], dtype=np.int64),
            11: np.array([40], dtype=np.int64),
            20: np.array([10, 99], dtype=np.int64),
            30: np.array([99, 40], dtype=np.int64),
            40: np.array([30], dtype=np.int64),
            99: np.array([10], dtype=np.int64),
        }
        incoming = {
            10: np.array([7, 20], dtype=np.int64),
            11: np.array([7, 10], dtype=np.int64),
            20: np.array([7, 8], dtype=np.int64),
            30: np.array([8, 40], dtype=np.int64),
            40: np.array([8, 11], dtype=np.int64),
        }
        extenders = [0, 2, 3, 5]
        candidates = [0, 1, 2, 4, 5]
        block = build_motif_block(pmid_sets, outgoing, incoming)
        observed, sums = sum_adjacent_motifs(block, extenders, candidates)
        view_out = LazyTuples({key: np.asarray(value) for key, value in outgoing.items()})
        view_in = LazyTuples({key: np.asarray(value) for key, value in incoming.items()})
        for pos, candidate in enumerate(candidates):
            manual_obs = 0
            manual = {name: 0 for name in sums}
            for extender in extenders:
                if extender == candidate:
                    continue
                motifs = citation_between(pmid_sets[candidate], pmid_sets[extender], view_out, view_in)
                if motifs is None:
                    continue
                manual_obs += 1
                for name in manual:
                    manual[name] += motifs[name]
            self.assertEqual(observed[pos], manual_obs)
            for name in manual:
                self.assertEqual(sums[name][pos], manual[name], name)
            self.assertTrue(citation_roles_partition(
                int(sums["direct_ltr"][pos]),
                int(sums["direct_rtl"][pos]),
                int(sums["direct_ambiguous"][pos]),
                int(sums["direct_collapsed"][pos]),
            ))


if __name__ == "__main__":
    unittest.main()

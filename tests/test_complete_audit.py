"""Small-graph checks for source sequences and candidate path counts."""

from __future__ import annotations

import unittest

import numpy as np

from src.complete_audit import (
    brute_source_sequences,
    gain_category,
    sequence_group,
    summarise_sequences,
    three_edge_count_matrix,
    three_edge_source_sequences,
    two_edge_count_matrix,
    two_edge_source_sequences,
    fill_sequence_table,
)
from src.path_count import brute_two_intermediate


class SequenceTests(unittest.TestCase):
    def test_grouped_sequences_match_enumeration(self):
        labelled = {
            "K": {"C1": "core_only", "C2": "both_sources", "G": "semmed_only"},
            "Q1": {"G": "both_sources"},
            "Q2": {"G": "core_only", "C1": "semmed_only"},
        }
        candidates = {"C1", "C2"}
        queries = {"Q1", "Q2"}
        exact = three_edge_source_sequences(labelled, candidates, queries)
        brute = brute_source_sequences(labelled, candidates, queries)
        self.assertEqual(exact, brute)
        self.assertGreater(sum(exact.values()), 0)
        self.assertEqual(sequence_group(("core_only", "semmed_only", "both_sources")), "requires_both_resources")
        self.assertEqual(sequence_group(("both_sources", "both_sources", "both_sources")), "either_resource")
        rows = fill_sequence_table(exact)
        self.assertEqual(len(rows), 27)
        self.assertEqual(sum(row["complete_paths"] for row in rows), sum(exact.values()))

    def test_two_edge_sequences_use_semmed_gene_gene_edges(self):
        labelled = {"Q": {"G": "core_only", "H": "both_sources"}}
        neighbors = {"G": {"C"}, "C": {"G"}, "H": {"C"}}
        sequences = two_edge_source_sequences(labelled, neighbors, {"C"}, {"Q"})
        self.assertEqual(sequences[("semmed_only", "core_only")], 1)
        self.assertEqual(sequences[("semmed_only", "both_sources")], 1)
        summary = summarise_sequences(
            [
                {
                    "position_1": "semmed_only",
                    "position_2": "core_only",
                    "position_3": "core_only",
                    "sequence_group": "requires_both_resources",
                    "complete_paths": 4,
                }
            ]
        )
        self.assertEqual(summary["requires_both_resources"], 4)


class CoverageTests(unittest.TestCase):
    def test_matrix_matches_brute_force_and_excludes_endpoints(self):
        genes = {
            "K": {"C1", "C2", "G"},
            "Q1": {"G"},
            "Q2": {"G", "C1"},
        }
        candidates = ["C1", "C2"]
        queries = ["Q1", "Q2", "K"]
        counts, max_cancer, n_cancers, total = three_edge_count_matrix(genes, candidates, queries)
        brute = brute_two_intermediate(genes, set(candidates), set(queries))
        self.assertEqual(total, sum(row["n_paths"] for row in brute.values()))
        self.assertEqual(int(counts.sum()), total)
        for query in queries:
            reached = {candidates[index] for index in np.flatnonzero(counts[queries.index(query)])}
            self.assertEqual(reached, brute[query]["reached"])
        self.assertTrue(np.all(max_cancer <= counts))
        self.assertTrue(np.all(max_cancer[counts > 0] > 0))
        self.assertTrue(np.all(n_cancers[counts == 0] == 0))
        self.assertTrue(np.all(n_cancers[counts > 0] > 0))

    def test_two_edge_matrix_counts_candidate_neighbors(self):
        genes = {"Q": {"G", "C"}}
        neighbors = {"G": {"C", "D"}, "C": {"G"}}
        counts = two_edge_count_matrix(genes, neighbors, ["C", "D"], ["Q"])
        self.assertEqual(int(counts[0, 0]), 1)
        self.assertEqual(int(counts[0, 1]), 1)

    def test_gain_categories(self):
        self.assertEqual(gain_category(1, 0), "only_core_adds_pmids")
        self.assertEqual(gain_category(0, 2), "only_semmed_adds_pmids")
        self.assertEqual(gain_category(1, 2), "both_add_pmids")
        self.assertEqual(gain_category(0, 0), "neither_adds_pmids")


if __name__ == "__main__":
    unittest.main()

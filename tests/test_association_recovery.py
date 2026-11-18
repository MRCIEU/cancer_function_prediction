"""Protocol checks for development recovery. No formal test labels are used."""

from __future__ import annotations

import unittest

from src.association_recovery import (
    apply_mask,
    assign_partitions,
    citation_roles_partition,
    rank_with_tie_break,
    recall_at_k,
    sample_controls,
)


class RecoveryProtocolTests(unittest.TestCase):
    def test_shared_pmid_targets_stay_in_one_partition(self):
        records = [
            {"target_id": "a", "cancer": "C1", "gene": "1", "pmids": [10]},
            {"target_id": "b", "cancer": "C2", "gene": "2", "pmids": [10]},
            {"target_id": "c", "cancer": "C1", "gene": "3", "pmids": [11]},
            {"target_id": "d", "cancer": "C2", "gene": "4", "pmids": [12]},
            {"target_id": "e", "cancer": "C3", "gene": "5", "pmids": [13]},
        ]
        assignment = assign_partitions(records)
        self.assertEqual(assignment["a"], assignment["b"])
        self.assertEqual(set(assignment), {"a", "b", "c", "d", "e"})
        self.assertTrue(set(assignment.values()) <= {"train", "tune", "development_validation"})

    def test_controls_use_the_original_direct_pool(self):
        controls = sample_controls(["1"], ["1", "2", "3", "4"], {"2"}, 10, seed=7)
        self.assertNotIn("1", controls)
        self.assertNotIn("2", controls)
        self.assertEqual(controls, sample_controls(["1"], ["1", "2", "3", "4"], {"2"}, 10, seed=7))

    def test_mask_removes_pairs_pmids_and_citation_roles(self):
        masked = apply_mask(
            edges=[
                {"gene": "1", "cancer": "C1", "pmids": [10]},
                {"gene": "9", "cancer": "C2", "pmids": [10, 11]},
                {"gene": "8", "cancer": "C2", "pmids": [12]},
            ],
            citations=[(10, 12), (13, 10), (12, 14), (15, 16)],
            target_pairs={("1", "C1")},
            masked_pmids={10},
        )
        remaining = {(edge["gene"], edge["cancer"]): edge["pmids"] for edge in masked["edges"]}
        self.assertNotIn(("1", "C1"), remaining)
        self.assertEqual(remaining[("9", "C2")], [11])
        self.assertEqual(remaining[("8", "C2")], [12])
        self.assertEqual(masked["citations"], [(12, 14), (15, 16)])

    def test_direction_totals_and_recall(self):
        self.assertTrue(citation_roles_partition(3, 4, 1, 8))
        self.assertFalse(citation_roles_partition(3, 4, 1, 7))
        result = recall_at_k({"C1": ["9", "1", "2"], "C2": ["3"]}, {"C1": {"1"}, "C2": {"4"}}, k=2)
        self.assertEqual(result["recovered"], 1)
        self.assertEqual(result["positives"], 2)
        self.assertAlmostEqual(result["macro_recall"], 0.5)
        self.assertEqual(rank_with_tie_break([{"gene": "b", "score": 1}, {"gene": "a", "score": 1}]), ["a", "b"])


if __name__ == "__main__":
    unittest.main()

"""Synthetic checks for evaluation lists, citation motifs, and path budgets."""

from __future__ import annotations

import unittest

import numpy as np

from src.input_preparation import (
    CITATION_CONFIRMED_ZERO,
    CITATION_OBSERVED_LINK,
    CITATION_UNKNOWN,
    PATH_BUDGET,
    apply_score_mask,
    citation_motif,
    classify_list_membership,
    evaluate_masked_scores,
    update_topk,
)


class InputPreparationTests(unittest.TestCase):
    def test_lists_keep_hidden_targets_and_record_why_they_differ(self) -> None:
        hidden = classify_list_membership(True, False, "development_validation")
        self.assertTrue(hidden["in_primary"])
        self.assertTrue(hidden["historical"]["development_validation"])
        self.assertFalse(hidden["historical"]["train"])
        visible = classify_list_membership(True, True, None)
        self.assertFalse(visible["in_primary"])
        self.assertFalse(visible["historical"]["development_validation"])
        removed = classify_list_membership(True, False, None)
        self.assertTrue(removed["in_primary"])
        self.assertFalse(removed["historical"]["development_validation"])
        self.assertEqual(removed["reason"], "unmasked_direct_removed_by_mask")
        with self.assertRaises(ValueError):
            classify_list_membership(True, True, "train")

    def test_two_masks_read_one_score_matrix(self) -> None:
        genes = ["1", "2", "10"]
        scores = np.array([[1.0, 1.0, 0.0]])
        full = np.ones((1, 3), dtype=bool)
        primary = np.array([[True, False, True]])
        positives = {"Q": {"10"}}
        full_metric = evaluate_masked_scores(scores, genes, full, positives, ["Q"], ks=(2,))
        primary_metric = evaluate_masked_scores(apply_score_mask(scores, primary), genes, primary, positives, ["Q"], ks=(2,))
        self.assertEqual(full_metric["recall_at_2"]["recovered"], 0)
        self.assertEqual(primary_metric["recall_at_2"]["recovered"], 1)
        self.assertTrue(np.isnan(apply_score_mask(scores, primary)[0, 1]))

    def test_citation_direction_witness_and_unknown(self) -> None:
        indexed = {1, 2, 10, 20}
        direct = citation_motif((1,), (2,), {1: (2,)}, {}, indexed)
        self.assertEqual(direct["state"], CITATION_OBSERVED_LINK)
        self.assertEqual(direct["direct_ltr"], 1)
        self.assertEqual(direct["direct_collapsed"], 1)
        cocite = citation_motif((1,), (2,), {}, {1: (10,), 2: (10,)}, indexed)
        self.assertEqual(cocite["cocite_witnesses"], 1)
        shared = citation_motif((1,), (2,), {1: (20,), 2: (20,)}, {}, indexed)
        self.assertEqual(shared["shared_reference_witnesses"], 1)
        none = citation_motif((1,), (2,), {}, {}, indexed)
        self.assertEqual(none["state"], CITATION_CONFIRMED_ZERO)
        unknown = citation_motif((), (2,), {}, {}, indexed)
        self.assertEqual(unknown["state"], CITATION_UNKNOWN)
        overlap = citation_motif((1, 2), (2,), {}, {}, indexed)
        self.assertEqual(overlap["shared_articles"], 1)

    def test_path_budget_keeps_the_stronger_label_free_path(self) -> None:
        values = np.full((2, PATH_BUDGET), -1.0)
        keys_a = np.full((2, PATH_BUDGET), 2**31 - 1, dtype=np.int32)
        keys_b = np.full((2, PATH_BUDGET), 2**31 - 1, dtype=np.int32)
        counts = np.zeros(2, dtype=np.int32)
        slots = np.array([0, 1], dtype=np.int32)
        update_topk(values, keys_a, keys_b, counts, slots, np.array([0.2, 0.2]), 5, 1)
        update_topk(values, keys_a, keys_b, counts, np.array([0]), np.array([0.4]), 3, 1)
        update_topk(values, keys_a, keys_b, counts, np.array([0]), np.array([0.4]), 2, 1)
        self.assertEqual(int(counts[0]), 3)
        self.assertEqual(int(keys_a[0, 0]), 2)
        self.assertAlmostEqual(float(values[0, 0]), 0.4)
        self.assertAlmostEqual(float(values[1, 0]), 0.2)


if __name__ == "__main__":
    unittest.main()

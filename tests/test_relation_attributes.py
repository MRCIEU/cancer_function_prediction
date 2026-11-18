"""Hand-enumerated checks for DWPC weights and attribute splits."""

from __future__ import annotations

import unittest

import numpy as np

from src.relation_attributes import DAMPING, assign_development_roles, normalise_core, score_query


class RelationAttributeTests(unittest.TestCase):
    def test_core_tokens_keep_uninformative_distinct(self) -> None:
        self.assertEqual(normalise_core("cge", "UP"), "up")
        self.assertEqual(normalise_core("cge", "NOTINF"), "uninformative")
        self.assertEqual(normalise_core("ccs", ""), "uninformative")
        self.assertEqual(normalise_core("gci", "CAUSALITY"), "causality")

    def test_dwpc_and_attribute_split_on_two_paths(self) -> None:
        # Candidate c reaches query Q by c—g—Q and by c—K—g—Q.
        genes_of = {"Q": {"g"}, "K": {"c", "g"}}
        neighbors = {"c": {"g"}, "g": {"c"}}
        cancers_of = {"g": {"Q", "K"}, "c": {"K"}}
        index = {"c": 0, "g": 1}
        slots = {"K": np.array([0, 1], dtype=np.int32), "Q": np.array([1], dtype=np.int32)}
        deg_gg = {"c": 1, "g": 1}
        deg_gc = {"c": 1, "g": 2}
        edge_pmids = {
            ("gene_gene", "c", "g"): (1,),
            ("gene_cancer", "g", "Q"): (2,),
            ("gene_cancer", "c", "K"): (3,),
            ("gene_cancer", "g", "K"): (2, 4),
        }
        attributes = {
            ("gene_cancer", "g", "Q"): {"core": {"cge": {"up", "down"}, "ccs": set(), "gci": set()}, "predicates": {"CAUSES"}},
            ("gene_cancer", "c", "K"): {"core": {"cge": {"up"}, "ccs": set(), "gci": set()}, "predicates": set()},
            ("gene_gene", "c", "g"): {"core": None, "predicates": {"INTERACTS_WITH"}},
        }
        scored = score_query("Q", genes_of, neighbors, cancers_of, index, slots, deg_gg, deg_gc, edge_pmids, {}, attributes, np.zeros(2))
        two = (1.0 ** -DAMPING) * (1.0 ** -DAMPING) * (2.0 ** -DAMPING) * (1.0 ** -DAMPING)
        self.assertAlmostEqual(scored["two_edge_dwpc"][0], two)
        self.assertAlmostEqual(scored["two_edge_paths"][0], 1.0)
        self.assertAlmostEqual(scored["two_edge_prior_weight"][0], 1.0)
        three = (1.0 ** -DAMPING) * (2.0 ** -0.8) * (2.0 ** -0.8) * (1.0 ** -DAMPING)
        self.assertAlmostEqual(scored["three_edge_dwpc"][0], three)
        self.assertAlmostEqual(scored["three_edge_prior_weight"][0], 0.5)
        self.assertAlmostEqual(scored["t1_e2_cge_up"][0], 0.5)
        self.assertAlmostEqual(scored["t1_e2_cge_down"][0], 0.5)
        self.assertAlmostEqual(scored["t1_e2_cge_observed"][0], 1.0)
        self.assertAlmostEqual(scored["t2_e1_cge_up"][0], 1.0)
        self.assertAlmostEqual(scored["t2_e1_cge_down"][0], 0.0)
        self.assertTrue(np.isnan(scored["t2_e1_ccs_progression"][0]))
        self.assertAlmostEqual(scored["t1_e1_pred_INTERACTS_WITH"][0], 1.0)
        self.assertAlmostEqual(scored["t1_shared_pmid"][0], 0.0)
        self.assertAlmostEqual(scored["t2_e23_shared_pmid"][0], 1.0)

    def test_previous_targets_stay_out_of_validation(self) -> None:
        records = [
            {"target_id": "a", "cancer": "C1", "gene": "1", "pmids": [10]},
            {"target_id": "b", "cancer": "C1", "gene": "2", "pmids": [10]},
            {"target_id": "c", "cancer": "C2", "gene": "3", "pmids": [11]},
        ]
        roles = assign_development_roles(records, {10})
        self.assertEqual(roles["a"], "train")
        self.assertEqual(roles["b"], "train")
        self.assertNotEqual(roles["c"], "train")


if __name__ == "__main__":
    unittest.main()

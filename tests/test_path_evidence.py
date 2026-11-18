"""Weights, overlap, and provenance fixtures for the path evidence audit."""

from __future__ import annotations

import unittest

from src.run_path_evidence import derive_profiles
from src.path_evidence import (
    brute_path_weights,
    grouped_crosstab,
    overlap_template2_positions_1_2,
    template_weights,
)


class WeightTests(unittest.TestCase):
    def test_unequal_extensions_and_excluded_endpoint(self):
        genes = {
            "K": {"C1", "C2", "G"},
            "Q1": {"G"},
            "Q2": {"G", "C1"},
            "Q3": {"G"},
        }
        neighbors = {"C1": {"G"}, "G": {"C1", "C2"}, "C2": {"G"}}
        candidates = {"C1", "C2"}
        queries = {"Q1", "Q2", "Q3"}
        exact = template_weights(genes, neighbors, candidates, queries)
        brute = brute_path_weights(genes, neighbors, candidates, queries)
        self.assertEqual(exact["two_total"], brute["two_total"])
        self.assertGreater(exact["two_total"], 0)
        self.assertEqual(exact["two_pos2"][("G", "K")], brute["two_pos2"][("G", "K")])
        self.assertNotEqual(exact["two_pos2"][("G", "K")], exact["two_pos1"][("C1", "K")])
        self.assertNotIn(("C1", "K", "C1"), [])
        for key, value in brute["two_pos1"].items():
            self.assertEqual(exact["two_pos1"].get(key, 0), value)
        for key, value in brute["two_pos3"].items():
            self.assertEqual(exact["two_pos3"].get(key, 0), value)
        self.assertEqual(exact["one_total"], brute["one_total"])
        self.assertEqual(dict(exact["one_gene_gene"]), dict(brute["one_gene_gene"]))

    def test_same_cancer_and_same_gene_are_excluded(self):
        genes = {"Q": {"C", "G"}, "K": {"C"}}
        exact = template_weights(genes, {}, {"C"}, {"Q"})
        self.assertEqual(exact["two_total"], 0)

    def test_position_sums_match_complete_paths(self):
        genes = {
            "K": {"A", "B", "C"},
            "Q": {"B", "C"},
            "R": {"A"},
        }
        neighbors = {"A": {"B"}, "B": {"A", "C"}, "C": {"B"}}
        candidates = {"A", "B"}
        queries = {"Q", "R"}
        exact = template_weights(genes, neighbors, candidates, queries)
        self.assertEqual(sum(exact["two_pos1"].values()), exact["two_total"])
        self.assertEqual(sum(exact["two_pos2"].values()), exact["two_total"])
        self.assertEqual(sum(exact["two_pos3"].values()), exact["two_total"])
        self.assertEqual(sum(exact["one_gene_gene"].values()), exact["one_total"])
        self.assertEqual(sum(exact["one_gene_cancer"].values()), exact["one_total"])


class ProfileTests(unittest.TestCase):
    def test_repeated_records_missing_pmid_and_supplementary_negation(self):
        connections = {
            ("gene_cancer", "10", "C0001"): {
                "core": {
                    "sentence-a": ("123", "UP", "NOTINF", "", "", "BIOMARKER"),
                    "sentence-b": ("123", "DOWN", "", "", "", ""),
                    "sentence-c": ("", "NOTINF", "", "", "", ""),
                },
                "sem": {
                    "pred-a": ("123", "CAUSES", "gene:10", "cancer:C0001", False, True),
                    "pred-b": ("123", "AFFECTS", "gene:10", "cancer:C0001", False, True),
                    "pred-c": ("", "TREATS", "gene:10", "cancer:C0001", False, False),
                    "pred-d": ("999", "CAUSES", "cancer:C0001", "gene:10", True, False),
                },
            }
        }
        profiles, shared = derive_profiles(connections, {"10"}, {"C0001"})
        profile = profiles[0]
        self.assertEqual(profile["n_core_sentences"], 3)
        self.assertEqual(profile["n_pmid_core"], 1)
        self.assertEqual(profile["n_core_records_without_pmid"], 1)
        self.assertEqual(profile["n_semmed_predications"], 4)
        self.assertEqual(profile["n_semmed_traversal"], 2)
        self.assertEqual(profile["n_semmed_negated"], 1)
        self.assertEqual(profile["n_pmid_semmed"], 2)
        self.assertEqual(profile["n_pmid_traversal"], 1)
        self.assertEqual(profile["n_pmid_intersection"], 1)
        self.assertEqual(profile["n_semmed_records_without_pmid"], 1)
        self.assertEqual(profile["support"], "both_sources")
        self.assertEqual(len(shared), 1)
        collapsed = {
            ("gene_cancer", "10", "C0001"): {
                "core": {"sentence-a": connections[("gene_cancer", "10", "C0001")]["core"]["sentence-a"]},
                "sem": {"pred-a": connections[("gene_cancer", "10", "C0001")]["sem"]["pred-a"]},
            }
        }
        again, _ = derive_profiles(collapsed, {"10"}, {"C0001"})
        self.assertEqual(again[0]["n_pmid_core"], 1)
        self.assertEqual(again[0]["n_pmid_intersection"], 1)


class OverlapTests(unittest.TestCase):
    def test_shared_pmid_missing_pmid_and_self_gene_excluded(self):
        nodes = [
            {
                "gene": "C",
                "cand": True,
                "extend": 2,
                "support": "both_sources",
                "all_pmids": (1, 2),
                "core_pmids": (1,),
                "sem_pmids": (2,),
            },
            {
                "gene": "G",
                "cand": True,
                "extend": 3,
                "support": "semmed_only",
                "all_pmids": (2,),
                "core_pmids": (),
                "sem_pmids": (2,),
            },
            {
                "gene": "H",
                "cand": False,
                "extend": 0,
                "support": "core_only",
                "all_pmids": (),
                "core_pmids": (),
                "sem_pmids": (),
            },
        ]
        result = overlap_template2_positions_1_2(nodes)
        union = result["union"]
        self.assertEqual(union.distinct, 2)
        self.assertEqual(union.weighted, 2 + 3)
        self.assertEqual(union.both_distinct, 2)
        self.assertEqual(union.one_sided_distinct, 0)
        self.assertEqual(result["core_core"].one_sided_distinct, 2)
        self.assertEqual(result["core_semmed"].both_distinct, 1)
        self.assertEqual(result["core_semmed"].exact_distinct[0], 1)
        self.assertEqual(result["semmed_semmed"].exact_weighted[1], 5)

    def test_grouped_crosstab_removes_the_excluded_endpoint(self):
        nodes = [
            {"cand": True, "extend": 4, "support": "both_sources", "sig": 1},
            {"cand": True, "extend": 1, "support": "core_only", "sig": 2},
            {"cand": False, "extend": 5, "support": "semmed_only", "sig": 1},
        ]
        distinct, weighted = grouped_crosstab(nodes, "support", "support")
        self.assertEqual(sum(distinct.values()), 2 * 3 - 2)
        self.assertEqual(sum(weighted.values()), (4 + 1 + 5) * 2 - 4 - 1)


if __name__ == "__main__":
    unittest.main()

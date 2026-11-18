"""Hand checks for same-sentence triples and traversal direction."""

from __future__ import annotations

import unittest

import numpy as np

from src.relation_representation import (
    accumulate_fine,
    canonical_direction,
    classify_core_record,
    collision_summary,
    combination_share,
    combo_index,
    direction_share,
    in_multi_group,
    marginal_collision_groups,
    marginal_from_joint,
    quant_signature,
    undirected_from_directions,
)
from src.relation_attributes import GC_PREDICATES


class RelationRepresentationTests(unittest.TestCase):
    def test_same_sentence_binding_and_missing_states(self) -> None:
        complete = classify_core_record("S1", "UP", "PROGRESSION", "OBSERVATION")
        self.assertEqual(complete["status"], "traceable_complete")
        self.assertEqual(complete["combination"], ("UP", "PROGRESSION", "OBSERVATION"))
        partial = classify_core_record("S2", "UP", "", "CAUSALITY")
        self.assertEqual(partial["status"], "partial")
        self.assertIsNone(partial["combination"])
        unknown = classify_core_record("S3", "UP", "NOTINF", "NOTINF")
        self.assertEqual(unknown["status"], "traceable_complete")
        self.assertTrue(unknown["explicit_notinf"])
        missing = classify_core_record("", "", "", "")
        self.assertEqual(missing["status"], "untraceable")
        self.assertIsNone(missing["combination"])

    def test_duplicate_combination_is_one_share(self) -> None:
        triple = ("UP", "PROGRESSION", "OBSERVATION")
        share = combination_share({triple, triple})
        self.assertAlmostEqual(float(share.sum()), 1.0)
        self.assertAlmostEqual(float(share[combo_index(triple)]), 1.0)

    def test_joint_projects_to_its_own_marginal(self) -> None:
        left = ("UP", "PROGRESSION", "OBSERVATION")
        right = ("DOWN", "REGRESSION", "CAUSALITY")
        share = combination_share({left, right})
        marginal = marginal_from_joint(share)
        self.assertAlmostEqual(float(marginal.sum()), 3.0)
        self.assertAlmostEqual(float(marginal[3]), 0.5)
        self.assertAlmostEqual(float(marginal[2]), 0.5)

    def test_direction_splits_and_restores_the_predicate(self) -> None:
        origin, destination = "gene:1", "cancer:C1"
        self.assertEqual(canonical_direction("gene:1", "cancer:C1", origin, destination), "along")
        self.assertEqual(canonical_direction("cancer:C1", "gene:1", origin, destination), "against")
        self.assertEqual(canonical_direction("gene:9", "cancer:C9", origin, destination), "unresolved")
        directions = {"CAUSES": {"along", "against"}, "AFFECTS": {"along"}}
        forward = direction_share(directions, GC_PREDICATES, False)
        backward = direction_share(directions, GC_PREDICATES, True)
        self.assertAlmostEqual(float(forward.sum()), 1.0)
        restored = undirected_from_directions(forward)
        self.assertAlmostEqual(float(restored[GC_PREDICATES.index("CAUSES")]), 0.5)
        self.assertAlmostEqual(float(restored[GC_PREDICATES.index("AFFECTS")]), 0.5)
        cause = GC_PREDICATES.index("CAUSES") * 3
        self.assertAlmostEqual(float(forward[cause]), 0.25)
        self.assertAlmostEqual(float(forward[cause + 1]), 0.25)
        self.assertAlmostEqual(float(backward[cause]), float(forward[cause + 1]))
        self.assertAlmostEqual(float(backward[cause + 1]), float(forward[cause]))

    def test_two_candidates_with_the_same_marginal_stay_split_by_the_joint(self) -> None:
        index = {"c1": 0, "c2": 1, "g": 2}
        genes_of = {"Q": {"g"}, "K1": {"c1", "g"}, "K2": {"c2", "g"}}
        cancers_of = {"g": {"Q", "K1", "K2"}, "c1": {"K1"}, "c2": {"K2"}}
        slots = {
            "K1": np.array([0, 2], dtype=np.int32),
            "K2": np.array([1, 2], dtype=np.int32),
            "Q": np.array([2], dtype=np.int32),
        }
        deg_gc = {"c1": 1, "c2": 1, "g": 3}
        left = {("UP", "PROGRESSION", "OBSERVATION"), ("DOWN", "REGRESSION", "CAUSALITY")}
        right = {("UP", "REGRESSION", "OBSERVATION"), ("DOWN", "PROGRESSION", "CAUSALITY")}
        edges = {
            ("gene_cancer", "c1", "K1"): _edge(left),
            ("gene_cancer", "c2", "K2"): _edge(right),
            ("gene_cancer", "g", "K1"): _edge(set()),
            ("gene_cancer", "g", "K2"): _edge(set()),
            ("gene_cancer", "g", "Q"): _edge(set()),
        }
        matrices = {}
        for cancer, gene in (("K1", "c1"), ("K2", "c2")):
            matrices[(cancer, "combo")] = _slot_matrix(slots[cancer], {0: edges[("gene_cancer", gene, cancer)]["combo_share"]})
            matrices[(cancer, "marginal")] = _slot_matrix(slots[cancer], {0: edges[("gene_cancer", gene, cancer)]["marginal_share"]}, width=12)
            matrices[(cancer, "direction")] = np.zeros((len(slots[cancer]), len(GC_PREDICATES) * 3))
        scored = accumulate_fine("Q", genes_of, {}, cancers_of, index, slots, {}, deg_gc, edges, matrices)
        self.assertGreater(scored["three_edge_dwpc"][0], 0)
        self.assertAlmostEqual(float(scored["three_edge_dwpc"][0]), float(scored["three_edge_dwpc"][1]))
        fine = []
        coarse = []
        for slot in (0, 1):
            denominator = scored["complete_dwpc"]["t2_e1"][slot]
            joint = scored["combo"]["t2_e1"][slot] / denominator
            margin = scored["marginal"]["t2_e1"][slot] / denominator
            self.assertAlmostEqual(float(np.nansum(joint)), 1.0)
            self.assertTrue(np.allclose(margin, marginal_from_joint(np.nan_to_num(joint)), atol=1e-12))
            fine.append(quant_signature(joint, 6))
            coarse.append(quant_signature(margin, 6))
        summary = collision_summary(coarse, fine)
        self.assertEqual(summary["candidates_in_multi"], 2)
        self.assertEqual(summary["multi_fine_groups"], 1)
        self.assertEqual(in_multi_group(coarse, fine), [True, True])
        groups = marginal_collision_groups([
            (_marginal_key(left), tuple(sorted(left))),
            (_marginal_key(right), tuple(sorted(right))),
        ])
        self.assertEqual(groups["connections_in_those_groups"], 2)


def _edge(combinations: set[tuple[str, str, str]]) -> dict:
    share = combination_share(combinations)
    return {"combo_share": share, "marginal_share": marginal_from_joint(share), "directions": {}}


def _slot_matrix(slots: np.ndarray, rows: dict[int, np.ndarray], width: int = 64) -> np.ndarray:
    matrix = np.zeros((len(slots), width), dtype=np.float64)
    for local, share in rows.items():
        matrix[local, : len(share)] = share
    return matrix


def _marginal_key(combinations: set[tuple[str, str, str]]) -> tuple:
    share = combination_share(combinations)
    marginal = marginal_from_joint(share)
    return quant_signature(marginal, 6)


if __name__ == "__main__":
    unittest.main()

"""Small citation graphs for adjacent-edge motifs and traversal alignment."""

from __future__ import annotations

import unittest

from src.run_citation_audit import Edge, consume, fresh, value
from src.citation_features import (
    classify_prepared_links,
    direct_citation_roles,
    pair_features,
    third_party_motifs,
    traversal_alignment,
    witness_matrices,
)


class CitationFeatureTests(unittest.TestCase):
    def test_direct_orientation_and_ambiguous_overlap(self):
        outgoing = {10: (20,), 20: (10,)}
        roles = direct_citation_roles((10,), (20,), outgoing)
        self.assertEqual(roles["exclusive_left_to_right"], 1)
        self.assertEqual(roles["exclusive_right_to_left"], 1)
        self.assertEqual(roles["ambiguous_position"], 0)
        self.assertEqual(roles["direction_collapsed"], 2)
        overlap = direct_citation_roles((10, 20), (10, 20), {10: (20,)})
        self.assertEqual(overlap["ambiguous_position"], 1)
        self.assertEqual(overlap["direction_collapsed"], 1)
        self.assertEqual(overlap["exclusive_left_to_right"], 0)

    def test_external_cocitation_shared_reference_and_false_shared_pmid(self):
        incoming = {1: (9,), 2: (9,)}
        outgoing = {1: (7,), 2: (7,)}
        support = {1, 2}
        cocite = third_party_motifs((1,), (2,), incoming, support)
        self.assertEqual(cocite["witnesses"], 1)
        self.assertEqual(cocite["external_witnesses"], 1)
        self.assertEqual(cocite["article_pairs"], 1)
        shared = third_party_motifs((1,), (2,), outgoing, support)
        self.assertEqual(shared["witnesses"], 1)
        self.assertEqual(shared["external_witnesses"], 1)
        false = third_party_motifs((1,), (1,), {1: (9,)}, support)
        self.assertEqual(false["witnesses"], 0)
        repeated = third_party_motifs((1,), (2,), {1: (9, 9), 2: (9, 9)}, support)
        self.assertEqual(repeated["witnesses"], 1)

    def test_reverse_traversal_keeps_original_endpoints(self):
        self.assertEqual(traversal_alignment("gene:10", "cancer:C1", "gene:10", "cancer:C1"), "with_traversal")
        self.assertEqual(traversal_alignment("gene:10", "cancer:C1", "cancer:C1", "gene:10"), "against_traversal")
        self.assertEqual(traversal_alignment("gene:10", "gene:20", "gene:20", "gene:10"), "against_traversal")

    def test_prepared_links_count_an_overlapping_link_once(self):
        left = (1, 2)
        right = (2, 3)
        link = ((2, 3),)
        exclusive_ltr, exclusive_rtl, ambiguous = classify_prepared_links(left, right, link, link)
        self.assertEqual((exclusive_ltr, exclusive_rtl, ambiguous), (1, 0, 0))
        both = ((1, 2),)
        exclusive_ltr, exclusive_rtl, ambiguous = classify_prepared_links((1, 2), (1, 2), both, both)
        self.assertEqual((exclusive_ltr, exclusive_rtl, ambiguous), (0, 0, 1))
        roles = direct_citation_roles((1, 2), (1, 2), {1: (2,)})
        self.assertEqual(exclusive_ltr + exclusive_rtl + ambiguous, roles["direction_collapsed"])

    def test_witness_matrix_matches_pairwise_and_ignores_duplicate_rows(self):
        edges = [(1,), (2,), (1,)]
        incoming = {1: (9, 9, 8), 2: (9,)}
        outgoing = {1: (7,), 2: (7, 4)}
        inside = [False] * 12
        for pmid in (1, 2):
            inside[pmid] = True
        cocite = witness_matrices(edges, incoming, inside, True, 1000)
        shared = witness_matrices(edges, outgoing, inside, True, 1000)
        support = {1, 2}
        for left in range(3):
            for right in range(3):
                if left == right:
                    continue
                expected = pair_features(edges[left], edges[right], outgoing, incoming, support)
                self.assertEqual(int(cocite["witnesses"][left, right]), expected["cocitation_witnesses"])
                self.assertEqual(int(cocite["external"][left, right]), expected["cocitation_external_witnesses"])
                self.assertEqual(cocite["article_counts"].get((left, right), 0), expected["cocitation_article_pairs"])
                self.assertEqual(int(shared["witnesses"][left, right]), expected["shared_reference_witnesses"])
                self.assertEqual(shared["article_counts"].get((left, right), 0), expected["shared_reference_article_pairs"])
        self.assertEqual(int(cocite["witnesses"][0, 2]), 0)
        self.assertEqual(int(cocite["external"][0, 1]), 1)

    def test_block_witness_counts_match_the_pairwise_rule(self):
        edges = [(1,), (1, 2), (2,), (1,), (3, 4)]
        incoming = {1: (9,), 2: (9,), 3: (8,), 4: (8,)}
        inside = [False] * 12
        for pmid in (1, 2, 3, 4):
            inside[pmid] = True
        slow = witness_matrices(edges, incoming, inside, True, 100000)
        fast = witness_matrices(edges, incoming, inside, False, 100000)
        self.assertTrue((slow["witnesses"] == fast["witnesses"]).all())
        self.assertTrue((slow["external"] == fast["external"]).all())
        self.assertEqual(int(fast["witnesses"][0, 3]), 0)
        self.assertEqual(int(fast["witnesses"][0, 2]), 1)
        self.assertEqual(int(fast["witnesses"][1, 3]), 1)
        self.assertEqual(int(fast["external"][1, 0]), 1)

    def test_adjacent_counter_matches_link_roles_and_keeps_supplementary_overlap_separate(self):
        left = Edge()
        right = Edge()
        left.q = (1,)
        right.q = (2,)
        left.qset = frozenset(left.q)
        right.qset = frozenset(right.q)
        left.links = ((1, 2),)
        right.links = ()
        left.all_interned = (1, 5)
        right.all_interned = (2, 5)
        left.n_out = left.n_in = left.n_meta = 1
        right.n_out = right.n_in = right.n_meta = 1
        values = fresh()
        consume(values, left, right, 4, None, None, 0, 1)
        self.assertEqual(value(values, "ltr_links"), 1)
        self.assertEqual(value(values, "collapsed_links"), 1)
        self.assertEqual(value(values, "ambiguous_links"), 0)
        self.assertEqual(value(values, "shared_q_pairs"), 0)
        self.assertEqual(value(values, "all_overlap_pairs"), 1)
        self.assertEqual(value(values, "pairs", True), 4)
        self.assertEqual(value(values, "witness_skipped_pairs"), 1)
        overlap = Edge()
        overlap.q = (1, 2)
        overlap.qset = frozenset(overlap.q)
        overlap.links = ((1, 2),)
        overlap.all_interned = (1, 2)
        overlap.n_out = 1
        overlap.n_in = 0
        overlap.n_meta = 2
        values = fresh()
        consume(values, overlap, overlap, 1, None, None, 0, 0)
        self.assertEqual(value(values, "ambiguous_links"), 1)
        self.assertEqual(value(values, "ltr_links") + value(values, "rtl_links") + value(values, "ambiguous_links"), value(values, "collapsed_links"))

    def test_pair_features_do_not_treat_one_shared_pmid_as_a_third_party_motif(self):
        features = pair_features((5,), (5,), outgoing={5: (8,)}, incoming={5: (9,)}, support={5})
        self.assertEqual(features["shared_pmids"], 1)
        self.assertEqual(features["cocitation_witnesses"], 0)
        self.assertEqual(features["shared_reference_witnesses"], 0)
        self.assertEqual(features["direction_collapsed"], 0)


if __name__ == "__main__":
    unittest.main()

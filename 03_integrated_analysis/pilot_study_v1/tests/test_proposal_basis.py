import unittest

from pairpilot.proposal_basis import (
    claim_ids,
    reuse_key,
    select_year_both,
    source_membership,
    status_label,
    unit_counts,
)


class ProposalBasisTests(unittest.TestCase):
    def test_reused_pmid_is_one_article(self):
        self.assertEqual(reuse_key("run-a", "1"), reuse_key("run-b", "1"))

    def test_units_are_not_interchangeable(self):
        rows = [
            {"new_case_id": "c1", "paper_pair": "1|2", "question_id": "Q_EXPR"},
            {"new_case_id": "c1", "paper_pair": "1|2", "question_id": "Q_EXPR"},
        ]
        counts = unit_counts(rows)
        self.assertEqual(counts["mapping_rows"], 2)
        self.assertEqual(counts["unique_case_ids"], 1)
        self.assertEqual(counts["component_rows"], 2)

    def test_source_class_uses_this_pair_only(self):
        # PMID 9 is in some other pair's CORE set, but not in this pair.
        result = source_membership(["1", "2"], {"1"}, {"2"})
        self.assertEqual(result["mutually_exclusive_source_class"], "union_required")
        other = source_membership(["1", "2"], {"1", "2", "9"}, {"1"})
        self.assertEqual(other["mutually_exclusive_source_class"], "core_only")

    def test_statuses_stay_distinct(self):
        self.assertNotEqual(status_label("not_evaluated"), status_label("zero"))
        self.assertNotEqual(status_label("technical_failure"), status_label("not_recorded"))

    def test_stale_year_does_not_replace_current(self):
        chosen = select_year_both(27193, 18307)
        self.assertEqual(chosen["used"], 27193)
        self.assertNotEqual(chosen["used"], 18307)

    def test_claim_ids_are_traceable(self):
        self.assertIn("C-DATA-CORE", claim_ids("见 [C-DATA-CORE]"))


if __name__ == "__main__":
    unittest.main()

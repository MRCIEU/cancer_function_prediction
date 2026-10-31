"""Program checks for diagnostic_v1. They do not measure scientific accuracy."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pairpilot.diagnostic import (  # noqa: E402
    comparison_role,
    diagnostic_dimension,
    e5_block_reasons,
    eligible_pair,
    entity_identity,
    freeze_e5,
    logical_task_id,
    select_comparisons,
    view_allows,
)
from pairpilot.ledger import same_error_should_stop  # noqa: E402


def _row(**extra):
    base = {
        "pair_id": "g__C",
        "pmid": "1",
        "question_type": "expression_or_distribution",
        "primary_question_type": "expression_or_distribution",
        "discourse_role": "own_result",
        "status": "citation_located",
        "study_system": "human",
        "comparator": "low",
        "outcome": "expression",
    }
    base.update(extra)
    return base


class DiagnosticTests(unittest.TestCase):
    def test_roles_and_one_task_per_paper(self):
        self.assertEqual(comparison_role({"status": "no_finding", "question_type": "", "discourse_role": ""}), "not_a_finding")
        self.assertEqual(comparison_role(_row(discourse_role="background")), "excluded_background")
        self.assertEqual(comparison_role(_row(discourse_role="author_interpretation")), "author_claim")
        self.assertEqual(comparison_role(_row()), "comparable_result")
        self.assertEqual(logical_task_id("g__C", "1"), logical_task_id("g__C", "1"))
        self.assertNotEqual(logical_task_id("g__C", "1", 1), logical_task_id("g__C", "1", 0))

    def test_questions_unknown_and_entity(self):
        expression = _row(pmid="1")
        survival = _row(pmid="2", primary_question_type="prognostic_association", question_type="prognostic_association")
        self.assertFalse(eligible_pair(expression, survival))
        self.assertEqual(diagnostic_dimension(_row(outcome=""), _row(pmid="2", outcome="os"), "outcome"), "unknown")
        self.assertEqual(diagnostic_dimension(_row(outcome_reporting_status="not_applicable"), _row(pmid="2"), "outcome"), "not_applicable")
        self.assertNotEqual(diagnostic_dimension(_row(outcome=""), _row(pmid="2", outcome=""), "outcome"), "same_on_reported_information")
        self.assertEqual(entity_identity("CD137-Fc")["kind"], "intervention_construct")
        self.assertEqual(entity_identity("TNFSF9")["gene_id"], "8744")
        self.assertEqual(entity_identity("TNFRSF9 protein")["gene_id"], "3604")
        self.assertFalse(entity_identity("CD137L")["same_as_target_3604"])

    def test_e5_selection_and_views(self):
        searchable = {"candidate_id": "a", "pair_id": "p1", "target_question": "link CRP tissue and blood", "quote": "CRP", "search_question": "CRP tissue blood"}
        methods = dict(searchable, candidate_id="b", only_original_methods_omission=True)
        self.assertEqual(e5_block_reasons(searchable), [])
        self.assertIn("original_study_detail_not_searchable", e5_block_reasons(methods))
        rows = [
            dict(searchable, candidate_id=str(i), pair_id=f"p{i%2}", target_question=f"q{i}")
            for i in range(6)
        ]
        frozen = freeze_e5(rows, 20261015, 3)
        self.assertLessEqual(len(frozen), 3)
        self.assertEqual(len({row["pair_id"] for row in frozen}), len(frozen))
        candidates = [
            {"pair_id": "p", "pmid_a": str(i), "pmid_b": str(i + 10), "layer": "known_difference", "candidate_key": f"k{i}"}
            for i in range(20)
        ]
        selected = select_comparisons(candidates, 20261015, per_pair=12, total=24)
        self.assertLessEqual(len(selected), 12)
        self.assertFalse(view_allows({"pmid_a": "1", "pmid_b": "9"}, {"1"}))
        self.assertNotEqual("not_selected", "insufficient_context")
        stop, _, _ = same_error_should_stop("quota_exhausted", 2, "quota_exhausted", 3)
        self.assertTrue(stop)


if __name__ == "__main__":
    unittest.main()

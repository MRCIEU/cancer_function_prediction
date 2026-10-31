import json
import unittest
from pathlib import Path

from pairpilot.readiness import (
    a_pack_leaks,
    allocate_abstracts,
    blank_answer,
    cache_key,
    case_identity,
    extract_logical,
    filter_pair_articles,
    formal_draw_live,
    locate_all,
    public_text,
    readiness_label,
    round_block_reasons,
    span_record,
    truncation_status,
)

PACKAGE = Path(__file__).resolve().parents[1]


class ReadinessTests(unittest.TestCase):
    def test_long_text_is_not_cut(self):
        long_text = "结果" * 250 + "αβγ"
        self.assertEqual(public_text(long_text), long_text)
        self.assertGreater(len(public_text(long_text)), 400)

    def test_quote_location_does_not_fill_support(self):
        text = "The protein was high. The protein was high."
        none = span_record(text, "missing sentence", "hash")
        many = span_record(text, "The protein was high.", "hash")
        self.assertEqual(none["match_status"], "zero")
        self.assertEqual(many["match_status"], "multiple")
        self.assertEqual(len(locate_all(text, "The protein was high.")), 2)
        self.assertEqual(none["semantic_support"], "not_assessed")
        self.assertEqual(many["semantic_support"], "not_assessed")

    def test_truncation_is_an_export_action(self):
        full = "word " * 120
        stored = " ".join(full.split())[:400]
        self.assertEqual(truncation_status(stored, full, 400), "exporter_truncation")
        self.assertEqual(truncation_status(None, None, 400), "response_missing")
        self.assertEqual(truncation_status(full, full, 400), "not_truncated")

    def test_annotation_axes_are_independent(self):
        answer = blank_answer()
        answer["discourse_role_reference"] = "own_result"
        answer["segmentation_assessability"] = "assessable"
        answer["segmentation_flags"] = ["over_split", "over_merged"]
        answer["identity"] = "matches_task_object"
        answer["role_form"] = "form_error"
        answer["text_support"] = "full_support"
        answer["citation_support"] = "citation_missing"
        self.assertEqual(answer["text_support"], "full_support")
        self.assertNotEqual(answer["citation_support"], "insufficient_text")
        self.assertIsNone(answer["comparison_scope"])
        schema = json.loads((PACKAGE / "schemas" / "annotation_v2.json").read_text(encoding="utf-8"))
        self.assertIn("citation_missing", schema["citation_support"])
        self.assertNotIn("not_reported_in_abstract", schema["text_support"])
        self.assertNotEqual("not_recorded_in_legacy", "not_reported_in_abstract")
        self.assertNotEqual("pending", "insufficient_text")

    def test_case_identity_keeps_paper_order_out_of_the_evidence(self):
        left = case_identity("p", "2", "1", "Q_EXPR", "va", "vb")
        right = case_identity("p", "1", "2", "Q_EXPR", "vb", "va")
        self.assertEqual(left, right)
        self.assertEqual(case_identity("p", "1", "2", "other", "va", "vb"), "")
        self.assertNotEqual(extract_logical("p", "1"), extract_logical("p", "1|finding-9"))

    def test_round_robin_keeps_scarce_layers(self):
        drawn = allocate_abstracts(
            {"shared": ["s1", "s2", "s3", "s4", "s5"], "core_only": ["c1"], "semmed_only": ["m1"]},
            20261017,
            6,
        )
        early = [item["layer"] for item in drawn["picked"][:3]]
        self.assertIn("core_only", early)
        self.assertIn("semmed_only", early)
        self.assertTrue(drawn["meets_start_plan"])
        short = allocate_abstracts({"shared": ["s1"], "core_only": [], "semmed_only": []}, 20261017, 6)
        self.assertFalse(short["meets_start_plan"])
        self.assertIn("core_only", short["empty_layers"])

    def test_one_exposed_article_does_not_drop_the_pair(self):
        kept = filter_pair_articles("new", ["1", "2"], {"old"}, {"1"})
        self.assertEqual(kept["remaining"], ["2"])
        blocked = filter_pair_articles("old", ["9"], {"old"}, set())
        self.assertEqual(blocked["status"], "development_pair_excluded")
        self.assertEqual(blocked["remaining"], [])

    def test_live_formal_draw_is_closed(self):
        with self.assertRaises(RuntimeError):
            formal_draw_live([{"pair_id": "x"}])

    def test_phase_a_rejects_model_hints(self):
        self.assertEqual(a_pack_leaks("PMID 1\nThe protein was measured."), [])
        self.assertTrue(a_pack_leaks("see phase_b for the gold direction_relation"))

    def test_budget_and_cache_are_separate_from_an_old_contract(self):
        opening = {"attempts": 10, "input_tokens": 0, "output_tokens": 0, "spent_usd": 1, "reserved_usd": 0, "billing_unknown_usd": 0, "canary_attempts": 3}
        state = dict(opening)
        state["output_tokens"] = 59000
        self.assertIn("round_output", round_block_reasons(opening, state, 10, 6000, 0.01))
        self.assertNotEqual(
            cache_key("readiness_extract_v2", "{}", "p", "unit", 6000, "evidence_dossier_protocol_v1"),
            cache_key("readiness_extract_v2", "{}", "p", "unit", 6000, "evidence_dossier_protocol_v2"),
        )

    def test_technical_failure_is_not_calibration(self):
        self.assertEqual(readiness_label("completed", "prepared_blank", "technical_failed"), "partial")
        self.assertEqual(readiness_label("completed", "prepared_blank", "technical_pass"), "ready_for_calibration")


if __name__ == "__main__":
    unittest.main()

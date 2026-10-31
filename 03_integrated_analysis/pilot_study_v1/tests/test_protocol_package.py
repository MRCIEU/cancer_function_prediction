"""Offline checks for the protocol package. They do not score scientific accuracy."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pairpilot.protocol_package import (  # noqa: E402
    E6,
    align_findings,
    assert_output_path,
    field_link,
    formal_draw,
    metric_rows,
    project_legacy_status,
    unit_counts,
)
from pairpilot.query_ast import source_membership  # noqa: E402


class ProtocolPackageTests(unittest.TestCase):
    def test_legacy_missing_is_not_abstract_absence_and_failure_stays_failure(self):
        self.assertEqual(project_legacy_status(None, field_existed=False), "not_recorded_in_legacy")
        self.assertNotEqual(project_legacy_status("", field_existed=False), "not_reported_in_text")
        self.assertEqual(project_legacy_status("higher", field_existed=True, technical_failure=True), "technical_failure")
        self.assertNotEqual(project_legacy_status("higher", field_existed=True, technical_failure=True), "unassessable")

    def test_alignment_ignores_direction_and_keeps_split(self):
        pairs, left_only, right_only = align_findings(
            [{"entity": "CDC45", "sentences": [0], "direction": "higher"}],
            [
                {"entity": "CDC45", "sentences": [0], "direction": "lower"},
                {"entity": "other", "sentences": [2], "direction": "higher"},
            ],
        )
        self.assertEqual(pairs, [(0, 0)])
        self.assertEqual(right_only, [1])
        self.assertEqual(left_only, [])
        split_pairs, split_left, _split_right = align_findings(
            [
                {"entity": "expression", "sentences": [1], "direction": "up"},
                {"entity": "survival", "sentences": [1], "direction": "up"},
            ],
            [{"entity": "expression", "sentences": [1], "direction": "down"}],
        )
        self.assertEqual(split_pairs, [(0, 0)])
        self.assertEqual(split_left, [1])

    def test_denominators_and_field_link(self):
        cases = [
            {"case_id": "a", "pmid_a": "1", "pmid_b": "2", "finding_items": 3},
            {"case_id": "b", "pmid_a": "2", "pmid_b": "3", "finding_items": 1},
            {"case_id": "c", "pmid_a": "4", "pmid_b": "5", "finding_items": 0},
            {"case_id": "d", "pmid_a": "6", "pmid_b": "7", "finding_items": 2},
        ]
        counts = unit_counts(cases)
        self.assertEqual(counts["cases"], 4)
        self.assertEqual(counts["paper_pairs"], 4)
        self.assertEqual(counts["unique_pmids"], 7)
        self.assertEqual(counts["abstract_slots"], 8)
        self.assertEqual(counts["finding_items"], 6)
        self.assertEqual(field_link({"quote": "A sentence."}), "field_link_unknown")
        self.assertEqual(field_link({"quote": "A sentence.", "supported_field": "direction"}), "direction")

    def test_source_class_uses_the_target_pair_sets(self):
        core = {"1", "2"}
        semmed = set()
        other_library = {"1", "2", "9"}
        own = source_membership({"1", "2"}, core, semmed)
        self.assertEqual(own["mutually_exclusive_source_class"], "core_only")
        leaked = source_membership({"1", "2"}, core, other_library)
        self.assertEqual(leaked["mutually_exclusive_source_class"], "both_individually")
        self.assertNotEqual(own["mutually_exclusive_source_class"], leaked["mutually_exclusive_source_class"])

    def test_formal_draw_is_not_frozen_and_respects_exclusions(self):
        frame = [
            {"relation_class": "core_subset", "pair_id": "p1", "pmids": ["10"]},
            {"relation_class": "core_subset", "pair_id": "p2", "pmids": ["11"]},
            {"relation_class": "semmed_subset", "pair_id": "p3", "pmids": ["10"]},
            {"relation_class": "semmed_subset", "pair_id": "p4", "pmids": ["12"]},
            {"relation_class": "partial_overlap", "pair_id": "p5", "pmids": ["13"]},
            {"relation_class": "disjoint", "pair_id": "p6", "pmids": ["14"]},
        ]
        drawn = formal_draw(frame, {"p5"}, {"10"}, 20261015)
        chosen = {row["pair_id"] for row in drawn["chosen"]}
        self.assertFalse(drawn["frozen"])
        self.assertNotIn("p1", chosen)
        self.assertNotIn("p3", chosen)
        self.assertNotIn("p5", chosen)
        self.assertIn("p2", chosen)
        self.assertIn("p4", chosen)
        self.assertIn("p6", chosen)

    def test_metrics_are_not_filled_and_source_runs_are_read_only(self):
        for row in metric_rows():
            self.assertNotEqual(row["current_value"], "0")
            self.assertNotEqual(row["current_value"], 0)
        with self.assertRaises(RuntimeError):
            assert_output_path(E6 / "reports" / "report.md")
        source = Path(__file__).resolve().parents[1].joinpath("src/pairpilot/protocol_package.py").read_text(encoding="utf-8")
        self.assertNotIn("https://api.openai.com", source)
        self.assertNotIn("esearch.fcgi", source)
        self.assertNotIn("continue_pilot", source)

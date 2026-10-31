"""Program checks for the closeout round. They do not measure scientific accuracy."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pairpilot.query_ast import (  # noqa: E402
    QueryRejected,
    and_query,
    candidate_outcome,
    choose_new,
    class_counts,
    group,
    paper_pair_key,
    render,
    source_membership,
    truncation_record,
    _term,
)


def _gene(*texts):
    return group("gene", [_term(text, "gene", "test", "tiab", text, "as_written") for text in texts])


def _disease(*texts):
    return group("disease", [_term(text, "disease", "test", "tiab", text, "as_written") for text in texts])


class CloseoutTests(unittest.TestCase):
    def test_source_classes_are_exclusive_and_count_intersections(self):
        core = {"1", "2", "3"}
        semmed = {"2", "3", "4"}
        rows = [
            source_membership({"1", "2"}, core, semmed),
            source_membership({"2", "3"}, core, semmed),
            source_membership({"4"}, core, semmed),
            source_membership({"1", "4"}, core, semmed),
            source_membership({"9"}, core, semmed),
        ]
        self.assertEqual(
            [row["mutually_exclusive_source_class"] for row in rows],
            ["core_only", "both_individually", "semmed_only", "union_required", "unresolved"],
        )
        counts = class_counts(rows)
        self.assertTrue(counts["exclusive_cover"])
        self.assertEqual(counts["intersection_x"], 1)
        self.assertEqual(counts["implied_union_required"], 1)
        self.assertEqual(paper_pair_key("4", "1"), paper_pair_key("1", "4"))
        self.assertEqual(len({paper_pair_key("1", "4"), paper_pair_key("4", "1")}), 1)

    def test_outside_pmid_is_not_a_single_library_reproduction(self):
        row = source_membership({"1", "8"}, {"1"}, {"1"})
        self.assertEqual(row["mutually_exclusive_source_class"], "unresolved")
        self.assertFalse(row["available_in_core"])
        self.assertIn("8", row["missing_pmids"])

    def test_query_requires_gene_and_disease_groups(self):
        broad = and_query("broad", [_gene("HIC1", "HIC-1"), _disease("breast cancer")])
        text = render(broad)
        self.assertIn(" AND ", text)
        self.assertTrue(text.startswith("(HIC1[tiab] OR"))
        self.assertIn('"breast cancer"[tiab]', text)
        self.assertLess(text.index("HIC1"), text.index("breast cancer"))
        with self.assertRaises(QueryRejected):
            and_query("broad", [_disease("breast cancer", "TNBC")])
        specific = and_query("specific", [
            _gene("CDC45"),
            _disease("HCC"),
            group("condition", [_term("HBV", "condition", "candidate", "tiab", "HBV", "as_written")]),
        ])
        rendered = render(specific)
        self.assertEqual(rendered.count(" AND "), 2)
        self.assertIn("HBV[tiab]", rendered)

    def test_pair_id_is_rejected_and_uid_check_is_separate(self):
        with self.assertRaises(QueryRejected):
            _term("3090__C0006142", "gene", "internal", "tiab", "3090__C0006142", "none")
        with self.assertRaises(QueryRejected):
            _term("C0006142", "disease", "cui", "", "C0006142", "none")
        broad = and_query("broad", [_gene("CA9", "CAIX"), _disease("prostate cancer")])
        checked = and_query("broad", broad["groups"], ["20038959", "27225200"])
        text = render(checked)
        self.assertIn("20038959[uid]", text)
        self.assertIn(" AND ", text)
        self.assertNotIn("__", text)

    def test_truncation_and_slots_and_candidate_state(self):
        record = truncation_record(418, 50, 50)
        self.assertEqual(record["hit_count"], 418)
        self.assertTrue(record["truncated"])
        self.assertNotEqual(record["hit_count"], record["returned_pmid_n"])
        unknown = truncation_record(None, 10, 50)
        self.assertEqual(unknown["hit_count_status"], "unknown")
        self.assertFalse(unknown["truncated"])
        first = choose_new(["1", "2", "3", "4"], {"9"}, 2)
        second = choose_new(["2", "5", "6"], set(first["included"]) | {"9"}, 2)
        self.assertEqual(first["included"], ["1", "2"])
        self.assertEqual(second["included"], ["5", "6"])
        self.assertIn("2", second["already_present"])
        self.assertEqual(candidate_outcome("unresolved", "zero_hit", "still_open_in_examined_sample"), "not_assessable_query_failure")
        self.assertEqual(candidate_outcome("valid", "zero_hit", "still_open_in_examined_sample"), "no_new_material_in_bounded_search")
        self.assertEqual(candidate_outcome("valid", "bounded_with_new_material", "still_open_in_examined_sample"), "still_open_in_examined_sample")
        self.assertNotEqual(candidate_outcome("valid", "not_started", ""), candidate_outcome("valid", "zero_hit", ""))

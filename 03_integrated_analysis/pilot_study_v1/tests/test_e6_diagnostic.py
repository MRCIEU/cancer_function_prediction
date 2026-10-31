"""Program checks for e6_diagnostic_v1. They do not measure scientific accuracy."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pairpilot.e6_diagnostic import (  # noqa: E402
    align_findings,
    build_variants,
    cache_key,
    extract_task,
    judgment_user,
    select_cases,
    sentence_jaccard,
)
from pairpilot.texts import split_sentences  # noqa: E402


class E6DiagnosticTests(unittest.TestCase):
    def test_one_paper_task_covers_many_findings_and_zero(self):
        first = extract_task("8318__C2239176", "1", "r1")
        second = extract_task("8318__C2239176", "1", "r1")
        other = extract_task("8318__C2239176", "1", "r2")
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        manifest = [extract_task("768__C0376358", pmid, "r1") for pmid in ("9", "8")]
        manifest.append(extract_task("768__C0376358", "7", "r1"))
        self.assertEqual(len(manifest), 3)
        self.assertIn("zero_finding_kept", extract_task("768__C0376358", "7", "r0", zero_finding=True)["flags"])

    def test_replicates_do_not_share_a_cache_and_old_loop_is_isolated(self):
        base = {"task": "diagnostic_reread", "input_sha256": "a", "schema_id": "diagnostic_reread", "prompt_sha256": "p", "model_id": "gpt-6.1-sol", "params": {"max_completion_tokens": 6000}}
        self.assertNotEqual(cache_key(base, "r1"), cache_key(base, "r2"))
        self.assertNotEqual(cache_key(base, None), cache_key(base, "r1"))
        self.assertEqual(cache_key(base, "r1"), cache_key(base, "r1"))
        old = dict(base, task="extract_findings", schema_id="extract_findings")
        self.assertNotEqual(cache_key(old, 1), cache_key(base, "r1"))

    def test_sentence_edits_keep_offsets_and_do_not_leak(self):
        text = "Alpha is here. Beta was cited. Gamma remains."
        sentences = split_sentences(text)
        cited = build_variants(text, ["Beta was cited."], "11", 20261015)["cited_sentences"]
        self.assertEqual(cited[0]["text"], "Beta was cited.")
        variants = build_variants(text, ["Beta was cited."], "11", 20261015)
        self.assertNotIn("Beta was cited.", variants["D"]["text"])
        self.assertIn("Beta was cited.", variants["R"]["text"])
        self.assertEqual(variants["E"]["text"].strip(), "Beta was cited.")
        self.assertNotIn("title", variants["E"])
        user = judgment_user("CDC45", "hepatocellular carcinoma", "prognostic_association", [("11", variants["E"]["text"])])
        self.assertNotIn("title", user)
        self.assertNotIn("condition", user)
        self.assertNotIn("finding", user)
        self.assertTrue(variants["D"]["deleted"][0]["start"] < variants["D"]["deleted"][0]["end"])

    def test_overlapping_unicode_quotes_expand_to_sentences(self):
        text = "甲蛋白升高。乙指标下降。丙保持。"
        variants = build_variants(text, ["甲蛋白升高。乙指标"], "1", 20261015)
        cited = [row["text"] for row in variants["cited_sentences"]]
        self.assertEqual(cited, ["甲蛋白升高。", "乙指标下降。"])
        self.assertIn("丙保持。", variants["D"]["text"])
        self.assertNotIn("甲蛋白升高。", variants["D"]["text"])
        self.assertEqual(variants["R"]["status"], "not_constructible")

    def test_control_is_not_forced_and_empty_sets_are_separate(self):
        text = "One. Two."
        variants = build_variants(text, ["One."], "9", 20261015)
        self.assertIn(variants["R"]["status"], {"ok", "not_constructible"})
        if variants["R"]["status"] == "ok":
            self.assertLessEqual(variants["R"]["token_gap"], 0.20)
        blocked = build_variants("Only one sentence here.", ["Only one sentence here."], "3", 20261015)
        self.assertEqual(blocked["R"]["status"], "not_constructible")
        self.assertEqual(blocked["E"]["informative"], False)
        self.assertEqual(sentence_jaccard(set(), set()), "both_empty")
        self.assertNotEqual(sentence_jaccard(set(), set()), 1)
        pairs, left_only, right_only = align_findings(
            [{"entity": "CDC45", "sentences": [0], "direction": "higher", "outcome": "os"}],
            [{"entity": "CDC45", "sentences": [0], "direction": "lower", "outcome": "os"}, {"entity": "other", "sentences": [2], "direction": "higher", "outcome": "stage"}],
        )
        self.assertEqual(pairs, [(0, 0)])
        self.assertEqual(right_only, [1])
        self.assertEqual(left_only, [])

    def test_case_selection_keeps_direction_strata(self):
        rows = []
        for index, direction, pair, source in (
            (0, True, "8318__C2239176", "both_individually"),
            (1, True, "8318__C2239176", "semmed_only"),
            (2, True, "768__C0376358", "union_required"),
            (3, False, "768__C0376358", "core_only"),
            (4, False, "8318__C2239176", "both_individually"),
            (5, False, "768__C0376358", "union_required"),
        ):
            rows.append({
                "comparison_id": f"c{index}",
                "pair_id": pair,
                "source_class": source,
                "can_compare_direction": direction,
                "quote_a": "q",
                "quote_b": "q",
                "text_available": True,
            })
        chosen, excluded, notes = select_cases(rows, 20261015)
        self.assertEqual(len(chosen), 4)
        self.assertEqual(sum(1 for row in chosen if row["can_compare_direction"] is True), 2)
        self.assertEqual(sum(1 for row in chosen if row["can_compare_direction"] is False), 2)
        self.assertEqual(excluded, [])
        self.assertGreaterEqual(len({row["pair_id"] for row in chosen}), 1)
        self.assertTrue(notes is not None)

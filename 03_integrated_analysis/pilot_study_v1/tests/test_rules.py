"""Synthetic checks for rules that would change a scientific reading."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pairpilot.comparison import build_candidates, context_relation, filter_comparisons  # noqa: E402
from pairpilot.config import load_bundle  # noqa: E402
from pairpilot.contracts import exact_span, finding_id, metric_row  # noqa: E402
from pairpilot.experiments.e1 import article_sets, equal_article, view_stats  # noqa: E402
from pairpilot.experiments.e2 import direct_payload, extract_payload, knowledge_payload  # noqa: E402
from pairpilot.experiments.e3 import group_source_text, scope_change  # noqa: E402
from pairpilot.experiments.e4 import path_degree_weight, positive_paths, rank_paths, topk_overlap, typed_degree  # noqa: E402
from pairpilot.experiments.e5 import queries_for, select_new  # noqa: E402
from pairpilot.experiments.e6 import align_replicates, control_deletion, delete_span  # noqa: E402
from pairpilot.inference import CallCache, F_ALLOW, assert_no_kg_labels  # noqa: E402
from pairpilot.sampling import select_pairs  # noqa: E402
from pairpilot.stages import synthetic_chain  # noqa: E402
from pairpilot.texts import normalize_view, split_sentences  # noqa: E402


RULES = load_bundle()["pilot"]["required_comparability_fields"]


def _finding(local, pmid, **extra):
    row = {
        "finding_id": f"f{local}",
        "pair_id": "g__C1",
        "pmid": pmid,
        "question_type": "prognostic_association",
        "study_system": "human",
        "comparator": "low expression",
        "outcome": "overall survival",
        "exposure_or_intervention": "high expression",
        "result_direction": "higher_risk",
        "significance": "not_significant",
        "discourse_role": "own_result",
    }
    row.update(extra)
    return row


class Rules(unittest.TestCase):
    def test_shared_pmid_counts_once_and_pairs_stay(self):
        sets = article_sets({"1", "2"}, {"2", "3"})
        self.assertEqual(sets["Union"], {"1", "2", "3"})
        self.assertEqual(len(sets["intersection"]), 1)
        left = [_finding(1, "1", pair_id="a"), _finding(2, "1", pair_id="b")]
        self.assertEqual({row["pair_id"] for row in left}, {"a", "b"})

    def test_identical_pmid_sets_match(self):
        findings = [_finding(1, "1"), _finding(2, "2")]
        comparisons = build_candidates(findings, "cfg", RULES)
        one = view_stats(findings, comparisons, {"1", "2"})
        two = view_stats(findings, comparisons, {"2", "1"})
        self.assertEqual(one, two)

    def test_source_view_excludes_the_other_source(self):
        findings = [_finding(1, "1"), _finding(2, "9")]
        comparisons = build_candidates(findings, "cfg", RULES)
        kept = filter_comparisons(comparisons, {"1"})
        self.assertEqual(kept, [])
        chain = synthetic_chain(Path("."))
        self.assertNotIn("9", chain["core_view_pmids"])
        self.assertTrue(chain["same_union"])

    def test_unknown_is_not_equality_and_significance_is_separate(self):
        known = _finding(1, "1")
        missing = _finding(2, "2", comparator=None)
        relation = context_relation(known, missing, RULES)
        self.assertEqual(relation["condition_relation"], "insufficient_context")
        same = context_relation(known, _finding(3, "3", significance="significant"), RULES)
        self.assertEqual(same["condition_relation"], "comparable_on_reported_fields")
        built = build_candidates([known, _finding(3, "3", result_direction="lower_risk", significance="not_significant")], "cfg", RULES)
        self.assertEqual(built[0]["result_relation"], "different_reported_direction")

    def test_same_sentence_keeps_two_findings(self):
        a = finding_id("run", "g__C1", "1", 0)
        b = finding_id("run", "g__C1", "1", 1)
        self.assertNotEqual(a, b)

    def test_spans_unicode_and_invalid_quotes(self):
        text = "αβ 结果。重复。αβ 结果。"
        spans = split_sentences(text)
        self.assertGreaterEqual(len(spans), 2)
        self.assertTrue(exact_span(text, spans[0]["start"], spans[0]["end"], spans[0]["text"]))
        self.assertFalse(exact_span(text, 0, 2, "结果"))
        self.assertNotEqual(normalize_view("A  B"), "A  B")
        combined = "a\u0301"
        self.assertEqual(len(combined), 2)
        self.assertTrue(exact_span(combined, 0, 2, combined))

    def test_payload_isolation(self):
        pair = {"pair_id": "g__C1", "gene_id": "1", "disease_cui": "C1", "aliases": ["GENE"]}
        doc = {"pmid": "1", "title": "T", "text": "abstract", "text_version_id": "v"}
        direct_payload(pair, [doc])
        extract_payload(pair, doc)
        kept = knowledge_payload(pair, doc, [{
            "native_record_id": "r", "source": "SemMedDB", "predicate": "AFFECTS",
            "negated": False, "has_type": None, "label_tuple": "UP|OBSERVATION", "text": "sentence",
        }])
        self.assertEqual(kept["source_records"][0]["predicate"], "AFFECTS")
        self.assertEqual(kept["source_records"][0]["label_tuple"], "UP|OBSERVATION")
        with self.assertRaises(ValueError):
            assert_no_kg_labels({"predicate": "TREATS"})
        self.assertIn("pmid", F_ALLOW)
        self.assertNotIn("predicate", F_ALLOW)

    def test_cache_and_replicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = CallCache(Path(tmp))
            base = {
                "task": "extract_findings", "input_sha256": "a", "schema_id": "finding.v1",
                "prompt_sha256": "p", "model_id": "m", "params": {"temperature": 0},
            }
            other = dict(base, model_id="other")
            repeat = dict(base, replicate_id=1)
            again = dict(base, replicate_id=2)
            self.assertNotEqual(cache.key(base), cache.key(other))
            self.assertNotEqual(cache.key(base), cache.key(repeat))
            self.assertNotEqual(cache.key(repeat), cache.key(again))
            cache.put(cache.key(base), {"status": "ok"})
            self.assertIsNone(cache.get(cache.key(repeat)))

    def test_budget_and_metric_zero_denominator(self):
        row = metric_row(experiment="E1", config="CORE", metric="share", numerator=1, denominator=0, unit="pmid")
        self.assertEqual(row["status"], "not_applicable")
        self.assertIsNone(row["value"])
        self.assertNotEqual(row["status"], "computed")

    def test_paths_and_degree(self):
        edges = [
            {"start": "X", "end": "Z", "predicate": "interacts", "direction": "forward", "negated": False, "pmids": ["1", "1", "2"]},
            {"start": "Z", "end": "Y", "predicate": "affects", "direction": "forward", "negated": False, "pmids": ["2"]},
            {"start": "X", "end": "Z", "predicate": "interacts", "direction": "forward", "negated": True, "pmids": ["9"]},
            {"start": "Z", "end": "Q", "predicate": "affects", "direction": "forward", "negated": False, "pmids": ["3"]},
        ]
        paths = positive_paths(edges, "X", "Y")
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0]["n_pmid_edge_1"], 2)
        self.assertEqual(paths[0]["n_pmid_intersection"], 1)
        self.assertEqual(typed_degree(edges, "Z", "affects", "forward"), 2)
        weighted, literature = rank_paths(paths, edges, 0.4)
        self.assertAlmostEqual(weighted[0]["degree_weight"], path_degree_weight(2, 0.4))
        overlap = topk_overlap(weighted, literature, 1)
        self.assertEqual(overlap["overlap"], 1)

    def test_sampling_is_stable_and_does_not_relax(self):
        rows = []
        for relation, union in (("identical", 5), ("identical", 12)):
            rows.append({
                "gene_id": f"g{relation}{union}", "disease_cui": f"C{union}", "relation_class": relation,
                "n_core": 2, "n_semmed": 2, "n_union": union, "n_intersection": 1,
            })
        bands = {"small": [4, 8], "large": [9, 20]}
        first = select_pairs(rows, 20261015, bands)
        second = select_pairs(rows, 20261015, bands)
        self.assertEqual(first["selected"], second["selected"])
        self.assertTrue(any(item["kind"] == "stratum_empty" for item in first["deviations"]))
        self.assertEqual(sum(1 for item in first["selected"] if item["relation_class"] == "identical"), 2)

    def test_perturbation_control_and_search_queries(self):
        text = "before target after"
        start, end = text.index("target"), text.index("target") + len("target")
        deleted = delete_span(text, start, end)
        self.assertNotIn("target", deleted)
        control = control_deletion(text, start, end)
        self.assertIsNotNone(control)
        self.assertIn("target", control[0])
        queries = queries_for({"pair_id": "g__C1", "missing_dimension": "stage", "search_question": "survival"}, ["GENE"])
        self.assertEqual(len(queries), 2)
        blob = " ".join(item["text"] for item in queries).lower()
        self.assertNotIn("not ", blob)
        chosen = select_new([{"pmid": "1"}, {"pmid": "2"}], {"1"}, 1)
        self.assertTrue(any(row.get("already_in_packet") for row in chosen))
        self.assertEqual(sum(1 for row in chosen if not row.get("already_in_packet")), 1)

    def test_scope_groups_do_not_merge_sources(self):
        blocks = group_source_text([
            {"source": "SemMedDB", "ordinal": 1, "native_record_id": "b", "text": "second"},
            {"source": "CORE", "ordinal": 0, "native_record_id": "a", "text": "first"},
            {"source": "SemMedDB", "ordinal": 0, "native_record_id": "c", "text": "earlier"},
        ])
        self.assertEqual(blocks[1]["sentences"], ["earlier", "second"])
        self.assertEqual(scope_change("not_judged", "scope_difference"), "not_judged")

    def test_replicate_alignment_keeps_wording_apart_from_keys(self):
        base = [_finding(1, "1")]
        repeat = [_finding(1, "1", result_text="different words")]
        aligned = align_replicates(base, repeat)
        self.assertEqual(aligned["matched"], 1)
        self.assertEqual(aligned["only_baseline"], 0)

    def test_resume_hash_and_draw_repeat(self):
        draw = equal_article({"1", "2", "3", "4"}, {"5", "6", "7", "8"}, [], [], 20261015, draws=50)
        self.assertEqual(draw["status"], "computed")
        self.assertEqual(len(draw["draws"]), 50)
        again = equal_article({"1", "2", "3", "4"}, {"5", "6", "7", "8"}, [], [], 20261015, draws=50)
        self.assertEqual(draw, again)
        short = equal_article({"1"}, {"1", "2"}, [], [], 1)
        self.assertEqual(short["status"], "not_applicable")


if __name__ == "__main__":
    unittest.main()

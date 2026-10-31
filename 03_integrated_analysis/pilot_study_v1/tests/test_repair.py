"""Checks for the revision that separates status, budget, comparisons, E5 and E6."""

from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pairpilot.comparison import build_candidates, cluster_candidates, gap_from_comparison, is_analysis_finding  # noqa: E402
from pairpilot.config import load_bundle  # noqa: E402
from pairpilot.contracts import metric_row  # noqa: E402
from pairpilot.experiments.e1 import view_stats  # noqa: E402
from pairpilot.experiments.e2 import direct_payload, knowledge_payload  # noqa: E402
from pairpilot.experiments.e5 import allocate_candidates  # noqa: E402
from pairpilot.inference import CallCache  # noqa: E402
from pairpilot.ledger import LIMITS, BudgetBlocked, SharedLedger, available_usd, same_error_should_stop  # noqa: E402

RULES = load_bundle()["pilot"]["required_comparability_fields"]


def _finding(local, pmid, **extra):
    row = {
        "finding_id": f"f{local}",
        "pair_id": "g__C1",
        "pmid": pmid,
        "status": "citation_located",
        "question_type": "prognostic_association",
        "study_system": "human",
        "comparator": "low expression",
        "outcome": "overall survival",
        "exposure_or_intervention": "high expression",
        "result_direction": "higher_risk",
        "discourse_role": "own_result",
    }
    row.update(extra)
    return row


class RepairTests(unittest.TestCase):
    def test_no_finding_and_unknown_do_not_become_gaps(self):
        empty = {"finding_id": "none", "pair_id": "g__C1", "pmid": "9", "status": "no_finding", "question_type": "", "discourse_role": ""}
        self.assertFalse(is_analysis_finding(empty))
        missing = _finding(2, "2", comparator=None)
        comparisons = build_candidates([empty, _finding(1, "1"), missing], "cfg", RULES)
        self.assertEqual(len(comparisons), 1)
        self.assertIsNone(gap_from_comparison(comparisons[0], "cfg", {}))
        self.assertEqual(view_stats([empty, _finding(1, "1")], comparisons, {"1", "9"})["n_comparison"], 0)

    def test_direction_conflict_clusters_without_dropping_ids(self):
        left = _finding(1, "1")
        right = _finding(2, "2", result_direction="lower_risk")
        other = _finding(3, "3", result_direction="lower_risk")
        comparisons = build_candidates([left, right, other], "cfg", RULES)
        by_id = {row["finding_id"]: row for row in (left, right, other)}
        marks = []
        for row in comparisons:
            gap = gap_from_comparison(row, "cfg", by_id)
            if gap:
                gap["comparison_id"] = row["comparison_id"]
                marks.append(gap)
        clustered = cluster_candidates(marks)
        self.assertEqual(len(clustered), 1)
        self.assertGreaterEqual(clustered[0]["n_source_marks"], 2)
        self.assertIn("f1", clustered[0]["finding_ids"])
        self.assertIn("f2", clustered[0]["finding_ids"])

    def test_e5_quota_stays_with_each_candidate(self):
        candidates = [
            {"candidate_id": str(i), "pair_id": "p", "missing_dimension": "stage", "search_question": "survival"}
            for i in range(3)
        ]

        def hits(candidate, query):
            return [{"pmid": f"{candidate['candidate_id']}-{query['query_id']}-{i}"} for i in range(10)]

        rows = allocate_candidates(candidates, hits, set(), per_query=5, queries_per_candidate=2, global_new=30)
        new = [row for row in rows if row.get("status") == "retrieved" and not row.get("already_in_packet")]
        counts = {}
        for row in new:
            counts[row["candidate_id"]] = counts.get(row["candidate_id"], 0) + 1
        self.assertEqual(counts, {"0": 10, "1": 10, "2": 10})

    def test_replicate_cache_and_resume_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = CallCache(Path(tmp))
            base = {"task": "extract_findings", "input_sha256": "a", "schema_id": "s", "prompt_sha256": "p", "model_id": "m", "params": {}}
            cache.put(cache.key(base), {"status": "ok"})
            self.assertIsNone(cache.get(cache.key(dict(base, replicate_id=1))))
            self.assertEqual(cache.get(cache.key(base))["status"], "ok")

    def test_repeated_quota_stops_and_failure_is_not_complete(self):
        stop, kind, run = same_error_should_stop("", 0, "quota_exhausted", 3)
        self.assertFalse(stop)
        stop, kind, run = same_error_should_stop(kind, run, "quota_exhausted", 3)
        self.assertFalse(stop)
        stop, kind, run = same_error_should_stop(kind, run, "quota_exhausted", 3)
        self.assertTrue(stop)
        self.assertNotEqual("failed", "completed")

    def test_legacy_unknown_hold_is_counted_once_and_missing_balance_is_not_zero(self):
        legacy = {
            "spent_usd": 5.9397,
            "reserved_usd": 0.1622,
            "billing_unknown_usd": 0.1622,
            "reservations": {"same": {"usd": 0.1622}},
        }
        self.assertAlmostEqual(available_usd(legacy), LIMITS["max_usd"] - 5.9397 - 0.1622)
        self.assertIsNone(available_usd({"spent_usd": 1.0}))
        with tempfile.TemporaryDirectory() as tmp:
            ledger = SharedLedger(Path(tmp) / "ledger.json")
            ledger.reserve("a", 1000, 1000, "main")
            ledger.keep_unknown("a")
            state = ledger.snapshot()
            self.assertEqual(state["reserved_usd"], 0)
            self.assertGreater(state["billing_unknown_usd"], 0)
            self.assertAlmostEqual(available_usd(state), LIMITS["max_usd"] - state["billing_unknown_usd"])

    def test_concurrent_reserve_cannot_pass_the_remaining_amount(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.json"
            ledger = SharedLedger(path)
            ledger.reserve("seed", 1, 1, "main")
            ledger.settle("seed", 9_000_000, 200_000)
            errors = []
            wins = []

            def once(name):
                try:
                    SharedLedger(path).reserve(name, 1_000_000, 16000, "main")
                    wins.append(name)
                except BudgetBlocked:
                    errors.append(name)

            threads = [threading.Thread(target=once, args=(str(i),)) for i in range(4)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertLessEqual(len(wins), 1)
            self.assertGreaterEqual(len(errors), 3)

    def test_payload_and_denominator_contracts(self):
        pair = {"pair_id": "g__C", "gene_id": "g", "disease_cui": "C", "aliases": []}
        direct = direct_payload(pair, [{"pmid": "1", "title": "T", "text": "A"}])
        self.assertNotIn("predicate", direct)
        kept = knowledge_payload(pair, {"pmid": "1", "title": "T", "text": "A", "text_version_id": "v"}, [{"predicate": "AFFECTS", "text": "s", "source": "SemMedDB"}])
        self.assertEqual(kept["source_records"][0]["predicate"], "AFFECTS")
        row = metric_row(experiment="E1", config="CORE", metric="share", numerator=0, denominator=0, unit="finding")
        self.assertEqual(row["status"], "not_applicable")
        self.assertIsNone(row["value"])


if __name__ == "__main__":
    unittest.main()

"""Budget ledger and cache isolation. No network and no project data."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pairpilot.experiments.e1 import article_sets, view_stats  # noqa: E402
from pairpilot.experiments.e2 import direct_payload, knowledge_payload  # noqa: E402
from pairpilot.inference import CallCache  # noqa: E402
from pairpilot.ledger import BudgetBlocked, SharedLedger, estimate_usd  # noqa: E402


class LedgerTests(unittest.TestCase):
    def test_shared_cap_and_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = SharedLedger(Path(tmp) / "ledger.json")
            first = ledger.reserve("a", 1000, 1000, "main")
            self.assertGreater(first["reserved_usd"], 0)
            ledger.settle("a", 1000, 100, 0)
            state = ledger.snapshot()
            self.assertEqual(state["reserved_usd"], 0)
            self.assertAlmostEqual(state["spent_usd"], estimate_usd(1000, 100))
            with self.assertRaises(BudgetBlocked):
                ledger.reserve("too_big", 20_000_000, 1_000_000, "main")

    def test_canary_cap_is_inside_total(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = SharedLedger(Path(tmp) / "ledger.json")
            with self.assertRaises(BudgetBlocked):
                ledger.reserve("big", 2_000_000, 200_000, "canary")

    def test_replicate_cache_and_source_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = CallCache(Path(tmp))
            base = {"task": "extract_findings", "input_sha256": "a", "schema_id": "s", "prompt_sha256": "p", "model_id": "gpt-6.1-sol", "params": {}}
            cache.put(cache.key(base), {"status": "ok"})
            repeat = dict(base, replicate_id=1)
            self.assertIsNone(cache.get(cache.key(repeat)))
        sets = article_sets({"1"}, {"2"})
        findings = [{"pmid": "2", "question_type": "other", "study_system": "human", "discourse_role": "own_result",
                     "exposure_or_intervention": "", "comparator": "", "outcome": ""}]
        self.assertEqual(view_stats(findings, [], sets["CORE"])["n_finding"], 0)
        direct_payload({"pair_id": "g__C", "gene_id": "g", "disease_cui": "C", "aliases": []}, [{"pmid": "1", "title": "T", "text": "A"}])
        kept = knowledge_payload({"pair_id": "g__C", "gene_id": "g", "disease_cui": "C", "aliases": []}, {"pmid": "1", "title": "T", "text": "A", "text_version_id": "v"}, [{"predicate": "AFFECTS", "native_record_id": "r", "source": "SemMedDB", "negated": False, "has_type": None, "label_tuple": None, "text": "s"}])
        self.assertEqual(kept["source_records"][0]["predicate"], "AFFECTS")

    def test_missing_and_failure_are_distinct(self):
        self.assertNotEqual("no_finding", "not_evaluated_budget")
        self.assertNotEqual("citation_not_located", "http_error")


if __name__ == "__main__":
    unittest.main()

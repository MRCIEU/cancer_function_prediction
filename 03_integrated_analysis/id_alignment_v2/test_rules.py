#!/usr/bin/env python3
"""Rule fixtures for id_alignment_v2. Run on a compute node."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE_ROOT = Path("/home/b5ac/freddieyu.b5ac/project_2")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CODE_ROOT))

from rules import decide, name_agrees, parse_argument, slice_document  # noqa: E402


ENABLED = {"path_a": True, "path_b": True, "path_c": True}


def run(argument, semtype="gngm", n_sentence=1, direct=False, cui_rows=0, cui_genes="", generic=False, **flags):
    opts = dict(ENABLED)
    opts.update(flags)
    return decide(argument, semtype, n_sentence, direct, cui_rows, cui_genes, generic, opts["path_a"], opts["path_b"], opts["path_c"])


class ParseTests(unittest.TestCase):
    def test_single_digit_keeps_original_and_normalizes_candidates(self):
        parsed = parse_argument("007157")
        self.assertEqual(parsed["argument_format"], "single_digits")
        self.assertEqual(parsed["candidate_gene_ids"], ["7157"])
        self.assertEqual(parsed["parts"][0]["text"], "007157")

    def test_single_cui(self):
        self.assertEqual(parse_argument("C0079419")["argument_format"], "single_umls_cui")

    def test_one_cui_and_one_digit_preserves_order(self):
        parsed = parse_argument("C0079419|7157")
        self.assertEqual(parsed["argument_format"], "pipe_cui_and_one_digit")
        self.assertEqual([part["text"] for part in parsed["parts"]], ["C0079419", "7157"])
        self.assertEqual(parsed["candidate_gene_ids"], ["7157"])

    def test_digit_then_cui_is_the_same_format(self):
        parsed = parse_argument("7157|C0079419")
        self.assertEqual(parsed["argument_format"], "pipe_cui_and_one_digit")
        self.assertEqual([part["ord"] for part in parsed["parts"]], [1, 2])

    def test_multiple_digits_keep_every_distinct_id_in_order(self):
        parsed = parse_argument("C0085828|2353|2354|3725")
        self.assertEqual(parsed["argument_format"], "pipe_multiple_distinct_digits")
        self.assertEqual(parsed["candidate_gene_ids"], ["2353", "2354", "3725"])

    def test_repeated_same_digit_is_one_candidate(self):
        parsed = parse_argument("7157|07157")
        self.assertEqual(parsed["argument_format"], "pipe_repeated_same_digit")
        self.assertEqual(parsed["candidate_gene_ids"], ["7157"])


class DecideTests(unittest.TestCase):
    def test_path_a_does_not_require_entity_cui(self):
        got = run("7157", n_sentence=0, direct=False)
        self.assertEqual(got["accepted_gene_id"], "7157")
        self.assertEqual(got["mapping_route"], "A_direct_entrez")
        self.assertEqual(got["entity_check"], "no_entity_row")

    def test_path_a_object_side_uses_the_same_rule(self):
        got = run("3815", semtype="aapp", direct=True)
        self.assertEqual(got["mapping_status"], "accepted")

    def test_path_a_disabled_is_not_an_accepted_id(self):
        got = run("7157", path_a=False)
        self.assertIsNone(got["accepted_gene_id"])
        self.assertEqual(got["mapping_status"], "rule_not_enabled")

    def test_non_genomic_digit_is_not_accepted(self):
        got = run("7157", semtype="orch")
        self.assertEqual(got["mapping_status"], "semtype_not_genomic")

    def test_path_b_unique_entity_gene(self):
        got = run("C0079419", cui_rows=2, cui_genes="7157")
        self.assertEqual(got["accepted_gene_id"], "7157")
        self.assertEqual(got["mapping_route"], "B_entity_gene_id")

    def test_missing_entity_and_empty_gene_stay_distinct(self):
        missing = run("C0079419", cui_rows=0, cui_genes="")
        empty = run("C0079419", cui_rows=4, cui_genes="")
        self.assertEqual(missing["mapping_status"], "no_entity_row")
        self.assertEqual(empty["mapping_status"], "gene_field_empty")
        self.assertNotEqual(missing["mapping_status"], empty["mapping_status"])

    def test_generic_cui_is_not_forced(self):
        got = run("C0017337", cui_rows=3, cui_genes="7157", generic=True)
        self.assertIsNone(got["accepted_gene_id"])
        self.assertEqual(got["mapping_status"], "generic_concept_not_forced")

    def test_multiple_entity_genes_stay_ambiguous(self):
        got = run("C0085828", cui_rows=2, cui_genes="2353,2354")
        self.assertIsNone(got["accepted_gene_id"])
        self.assertEqual(got["candidate_gene_ids"], ["2353", "2354"])
        self.assertEqual(got["mapping_status"], "ambiguous_multiple_gene_ids")

    def test_composite_agrees_and_dedupes_routes(self):
        got = run("C0079419|7157", cui_rows=1, cui_genes="7157", direct=True)
        self.assertEqual(got["accepted_gene_id"], "7157")
        self.assertEqual(got["mapping_route"], "C_composite_entrez+B_entity_gene_id")

    def test_composite_conflict_accepts_nothing(self):
        got = run("C0079419|7157", cui_rows=1, cui_genes="9999")
        self.assertIsNone(got["accepted_gene_id"])
        self.assertEqual(got["mapping_status"], "conflict")
        self.assertEqual(got["candidate_gene_ids"], ["7157", "9999"])

    def test_multiple_digits_are_not_expanded_and_first_is_not_chosen(self):
        got = run("1523|4791|4940")
        self.assertIsNone(got["accepted_gene_id"])
        self.assertEqual(got["candidate_gene_ids"], ["1523", "4791", "4940"])
        self.assertNotEqual(got["accepted_gene_id"], "1523")

    def test_composite_without_entity_can_use_the_direct_id(self):
        got = run("C0079419|7157", n_sentence=0, cui_rows=0, cui_genes="")
        self.assertEqual(got["accepted_gene_id"], "7157")
        self.assertEqual(got["entity_check"], "no_entity_row")
        self.assertEqual(got["mapping_route"], "C_composite_entrez")

    def test_name_check_is_exact_token_not_a_substring(self):
        self.assertTrue(name_agrees("TP53 gene|TP53", "TP53", "-"))
        self.assertFalse(name_agrees("TP53BP1", "TP53", "-"))
        self.assertTrue(name_agrees("p53", "TP53", "p53|P53"))

    def test_coordinate_conversion_is_uniform(self):
        sentence = "xxTP53yy"
        # Document sentence starts at 100. Mention TP53 occupies [102, 106) if end is exclusive.
        self.assertEqual(slice_document(sentence, 100, 102, 106, end_inclusive=False), "TP53")
        self.assertEqual(slice_document(sentence, 100, 102, 105, end_inclusive=True), "TP53")
        self.assertNotEqual(slice_document(sentence, 100, 102, 106, end_inclusive=True), "TP53")


class DuckDBParityTests(unittest.TestCase):
    def test_udf_matches_python_on_both_directions(self):
        import duckdb

        con = duckdb.connect()
        con.create_function("decide_map", _udf, return_type="VARCHAR", null_handling="special")
        con.execute(
            """
            CREATE TABLE fixture (
              row_id VARCHAR, argument VARCHAR, semtype VARCHAR,
              n_sentence INTEGER, direct_supported BOOLEAN, cui_rows INTEGER,
              cui_genes VARCHAR, is_generic BOOLEAN, path_a BOOLEAN, path_b BOOLEAN, path_c BOOLEAN
            )
            """
        )
        rows = [
            ("subj", "7157", "gngm", 0, False, 0, "", False, True, True, True),
            ("obj", "3815", "aapp", 2, True, 0, "", False, True, True, True),
            ("b", "C0079419", "gngm", 2, False, 2, "7157", False, True, True, True),
            ("multi", "C1|2353|2354", "gngm", 1, False, 1, "2353,2354", False, True, True, True),
            ("conflict", "C0079419|7157", "gngm", 1, False, 1, "9999", False, True, True, True),
            ("generic", "C0017337", "gngm", 1, False, 1, "7157", True, True, True, True),
            ("off", "7157", "gngm", 0, False, 0, "", False, False, True, True),
            ("nullgenes", "7157", "gngm", 0, False, 0, None, False, True, True, True),
        ]
        con.executemany("INSERT INTO fixture VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
        got = con.execute(
            """
            SELECT row_id, decide_map(argument, semtype, n_sentence, direct_supported, cui_rows, cui_genes, is_generic, path_a, path_b, path_c)
            FROM fixture ORDER BY row_id
            """
        ).fetchall()
        expect = {row[0]: json.loads(_udf(*row[1:])) for row in rows}
        for row_id, payload in got:
            self.assertEqual(json.loads(payload)["mapping_status"], expect[row_id]["mapping_status"])
            self.assertEqual(json.loads(payload)["accepted_gene_id"], expect[row_id]["accepted_gene_id"])


def _udf(argument, semtype, n_sentence, direct_supported, cui_rows, cui_genes, is_generic, path_a, path_b, path_c):
    return json.dumps(
        decide(argument, semtype, n_sentence, direct_supported, cui_rows, cui_genes, is_generic, path_a, path_b, path_c),
        ensure_ascii=False,
    )


class DedupeTests(unittest.TestCase):
    def test_same_pair_from_two_predications_keeps_both_routes(self):
        import duckdb

        con = duckdb.connect()
        con.execute(
            """
            CREATE TABLE mapped AS
            SELECT * FROM (VALUES
              ('1', '7157', 'C0027651', 'A_direct_entrez'),
              ('2', '7157', 'C0027651', 'B_entity_gene_id')
            ) t(predication_id, gene_id, disease_cui, mapping_route)
            """
        )
        pairs = con.execute("SELECT COUNT(*) FROM (SELECT DISTINCT gene_id, disease_cui FROM mapped)").fetchone()[0]
        routes = con.execute(
            "SELECT string_agg(mapping_route, ',' ORDER BY mapping_route) FROM mapped"
        ).fetchone()[0]
        self.assertEqual(pairs, 1)
        self.assertEqual(con.execute("SELECT COUNT(*) FROM mapped").fetchone()[0], 2)
        self.assertIn("A_direct_entrez", routes)
        self.assertIn("B_entity_gene_id", routes)


if __name__ == "__main__":
    raise SystemExit(unittest.main(verbosity=2))

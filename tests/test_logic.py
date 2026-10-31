import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from lib import labels, relations, stats, textnorm, years
from lib.cases import select_cases
from lib.id_rules import classify_gene_id_token, parse_core_disease, parse_core_gene, parse_core_pmid
from lib.runtime import CODE_ROOT, load_config


class CsvTests(unittest.TestCase):
    def test_multiline_quotes_and_null_token_keep_first_row(self):
        text = '"PMID"\n"1","a","line1\nline2","\\N"\n'
        # The first row is data even if it looks short; a real header-like check is separate.
        raw = '1,"see ""note""","line1\nline2",\\N\n2,b,c,d\n'
        rows = list(csv.reader(io.StringIO(raw)))
        self.assertEqual(rows[0][0], "1")
        self.assertEqual(rows[0][1], 'see "note"')
        self.assertEqual(rows[0][2], "line1\nline2")
        self.assertEqual(rows[0][3], "\\N")
        self.assertEqual(len(rows), 2)
        self.assertNotIn("header", text)


class RelationTests(unittest.TestCase):
    def test_five_classes_and_empty_sides(self):
        self.assertEqual(relations.classify_pmid_sets(set(), set()), "both_empty")
        self.assertEqual(relations.classify_pmid_sets(set(), {"1"}), "core_pmids_empty")
        self.assertEqual(relations.classify_pmid_sets({"1"}, set()), "semmed_pmids_empty")
        self.assertEqual(relations.classify_pmid_sets({"1"}, {"1"}), "identical")
        self.assertEqual(relations.classify_pmid_sets({"1"}, {"1", "2"}), "core_subset")
        self.assertEqual(relations.classify_pmid_sets({"1", "2"}, {"1"}), "semmed_subset")
        self.assertEqual(relations.classify_pmid_sets({"1", "2"}, {"2", "3"}), "partial_overlap")
        self.assertEqual(relations.classify_pmid_sets({"1"}, {"2"}), "disjoint")

    def test_zero_denominator_is_not_zero(self):
        got = relations.ratio(0, 0, "pmid", "none", "id_rule_v1")
        self.assertIsNone(got["value"])
        self.assertEqual(got["status"], "not_applicable")
        self.assertEqual(got["numerator"], 0)
        metrics = relations.overlap_metrics(set(), {"1"}, "pmid", "x", "id_rule_v1")
        self.assertEqual(metrics["class"], "core_pmids_empty")
        self.assertEqual(metrics["core_covered_by_semmed"]["status"], "not_applicable")
        self.assertEqual(metrics["absolute_semmed_increment"], 1)

    def test_increment_can_exceed_one(self):
        metrics = relations.overlap_metrics({"1"}, {"1", "2", "3"}, "pmid", "x", "id_rule_v1")
        self.assertEqual(metrics["semmed_increment_over_core"]["value"], 2.0)
        self.assertEqual(metrics["jaccard"]["numerator"], 1)
        self.assertEqual(metrics["jaccard"]["denominator"], 3)


class LabelTests(unittest.TestCase):
    def test_notinf_is_not_missing_and_exclusion_is_unusable(self):
        raw = labels.label_tuple("EXPRESSION", "NOTINF", None, "OBSERVATION")
        self.assertEqual(raw[1], "NOTINF")
        self.assertEqual(raw[2], "FIELD_ABSENT")
        het = labels.heterogeneous([raw, labels.label_tuple("EXPRESSION", "UP", "PROGRESSION", "OBSERVATION")])
        self.assertTrue(het["heterogeneous"])
        only_notinf = labels.heterogeneous_excluding_notinf([labels.label_tuple("NOTINF", "NOTINF", "NOTINF", "NOTINF")])
        self.assertEqual(only_notinf["status"], "no_usable_label")
        self.assertIsNone(only_notinf["heterogeneous"])


class IdTests(unittest.TestCase):
    def test_builder_namespaces(self):
        gene = parse_core_gene("https://www.ncbi.nlm.nih.gov/gene/7157")
        self.assertEqual(gene["normalized_id"], "7157")
        self.assertEqual(gene["status"], "explicit_namespace")
        disease = parse_core_disease("http://linkedlifedata.com/resource/umls/id/C0006826")
        self.assertEqual(disease["normalized_id"], "C0006826")
        pmid = parse_core_pmid("https://pubmed.ncbi.nlm.nih.gov/12345/")
        self.assertEqual(pmid["normalized_id"], "12345")
        odd = parse_core_disease("http://example.org/C0006826")
        self.assertEqual(odd["status"], "no_mapping")
        self.assertIsNone(odd["normalized_id"])

    def test_composite_gene_id_is_not_split(self):
        self.assertEqual(classify_gene_id_token("1|2"), "composite_pipe")
        self.assertEqual(classify_gene_id_token("\\N"), "backslash_N")
        self.assertEqual(classify_gene_id_token("7157"), "numeric")


class TextTests(unittest.TestCase):
    def test_normalization_keeps_negation(self):
        left = "Gene is not up-regulated."
        right = "Gene is not  up-regulated."
        got = textnorm.compare_texts(left, right)
        self.assertTrue(got["match"])
        self.assertEqual(got["method"], "nfkc_whitespace")
        self.assertIn("not", textnorm.normalize_text(left))
        exact = textnorm.compare_texts(left, left)
        self.assertEqual(exact["method"], "exact")


class YearTests(unittest.TestCase):
    def test_dp_and_pyear_stay_separate(self):
        self.assertEqual(years.parse_dp("2009 Dec")["years"], [2009])
        self.assertEqual(years.parse_dp("1998-1999")["status"], "multi_year_range")
        self.assertEqual(years.parse_dp("1998-1999")["years"], [1998, 1999])
        completion = years.parse_completion_pyear("2010")
        self.assertEqual(completion["status"], "parsed_completion_year")
        self.assertNotEqual(completion["status"], "parsed_single")
        self.assertEqual(years.parse_core_year("0")["status"], "non_positive")


class StatsTests(unittest.TestCase):
    def test_nearest_rank(self):
        values = list(range(1, 11))
        got = stats.summarize_numeric(values, "fixture", "count")
        self.assertEqual(got["p25"], 3)
        self.assertEqual(got["p50"], 5)
        self.assertEqual(got["p90"], 9)
        self.assertEqual(got["p99"], 10)
        self.assertEqual(got["n_zero"], 0)
        empty = stats.summarize_numeric([], "empty", "count")
        self.assertEqual(empty["status"], "not_applicable")
        self.assertIsNone(empty["p50"])

    def test_head_share(self):
        got = stats.head_share([1, 1, 1, 10], 0.25)
        self.assertEqual(got["n_head"], 1)
        self.assertEqual(got["weight_numerator"], 10)
        self.assertEqual(got["weight_denominator"], 13)


class CaseTests(unittest.TestCase):
    def test_selection_is_deterministic_and_reports_empty(self):
        pools = {
            "identical": [{"key": "a", "volume": 1}, {"key": "b", "volume": 5}, {"key": "c", "volume": 9}],
            "disjoint": [],
        }
        first = select_cases(pools, "20241014", 2)
        second = select_cases(pools, "20241014", 2)
        self.assertEqual(first["selected"], second["selected"])
        self.assertEqual(first["strata"][0]["action"], "absent")
        self.assertTrue(first["selected"])


class ConfigTests(unittest.TestCase):
    def test_preflight_caps(self):
        cfg = json.loads((CODE_ROOT / "config" / "analysis_config.json").read_text(encoding="utf-8"))
        self.assertLessEqual(cfg["preflight"]["max_records"], 5000)
        self.assertLessEqual(cfg["preflight"]["max_bytes"], 64 * 1024 * 1024)
        self.assertLessEqual(cfg["preflight"]["max_triples"], 5000)
        self.assertEqual(load_config()["id_rule_version"], "id_rule_v1")


class RdfTests(unittest.TestCase):
    def test_turtle_multiline_literal(self):
        from lib.rdf_stream import stream_to_callback

        ttl = (
            "@prefix core: <http://gda.dei.unipd.it/cecore/ontology/> .\n"
            "<http://gda.dei.unipd.it/cecore/resource/GCS#abc> "
            "core:expressedBy <https://www.ncbi.nlm.nih.gov/gene/7157> ;\n"
            "    core:hasContent \"\"\"alpha\nbeta\"\"\" .\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.ttl"
            path.write_text(ttl, encoding="utf-8")
            rows = []
            stream_to_callback(path, rows.extend, batch_size=1)
        preds = {row[1] for row in rows}
        self.assertIn("http://gda.dei.unipd.it/cecore/ontology/expressedBy", preds)
        contents = [row[2] for row in rows if row[1].endswith("hasContent")]
        self.assertEqual(contents, ["alpha\nbeta"])


class DuckDbTests(unittest.TestCase):
    def test_sql_percentile_matches_python(self):
        import duckdb

        from lib.stats import summarize_groups_sql

        con = duckdb.connect()
        con.execute("CREATE TABLE groups(n INTEGER)")
        con.executemany("INSERT INTO groups VALUES (?)", [(n,) for n in range(1, 11)])
        got = summarize_groups_sql(con, "groups", "n", "fixture", "count")
        py = stats.summarize_numeric(list(range(1, 11)), "fixture", "count")
        for key in ("p25", "p50", "p75", "p90", "p95", "p99", "min", "max"):
            self.assertEqual(got[key], py[key])


class ColumnMapTests(unittest.TestCase):
    def test_sample_layouts_do_not_follow_dbinfo_positions(self):
        from lib.semmed_columns import assign_local_columns

        sentence = "Fluoride-selective colorimetric sensor based on thiourea binding site and anthraquinone reporter."
        sentence_rows = [["6", "16530473", "ti", "1", "21", sentence, "119", "", ""]]
        sent = assign_local_columns("SENTENCE", sentence_rows)
        self.assertEqual(sent["status"], "assigned")
        self.assertEqual(sent["index"]["sentence_text"], 5)
        self.assertEqual(sent["index"]["sent_start_index"], 4)
        self.assertEqual(sent["index"]["sent_end_index"], 6)
        self.assertNotEqual(sent["index"]["sentence_text"], 8)

        entity_rows = [
            ["454796", "774955", "2784046", "C0162783", "Prefrontal Cortex", "bpoc", "", "", "prefrontal cortex", "888", "33", "50"],
            ["454797", "774852", "2724035", "C0597134", "oral bacteria", "bact", "", "", "oral bacteria", "1000", "34", "47"],
            ["454798", "52631916", "1793713", "C0242485", "Measurement", "ftcn", "", "", "measurements", "1000", "24", "36"],
        ]
        entity = assign_local_columns("ENTITY", entity_rows)
        self.assertEqual(entity["status"], "assigned", entity.get("status"))
        self.assertEqual(entity["index"]["cui"], 3)
        self.assertEqual(entity["index"]["semtype"], 5)
        self.assertEqual(entity["index"]["gene_id"], 6)
        self.assertEqual(entity["index"]["text"], 8)
        self.assertEqual(entity["index"]["start_index"], 10)
        self.assertEqual(entity["index"]["end_index"], 11)
        self.assertEqual(entity["index"]["pre_cui_numeric_1"], 1)
        self.assertEqual(entity["index"]["pre_cui_numeric_2"], 2)
        self.assertNotIn("sentence_id", entity["index"])

        pred_rows = [[
            "10592604", "16", "16530475", "PROCESS_OF", "C0003725", "Arboviruses", "virs", "1",
            "C0999630", "Lepus capensis", "mamm", "1", "\\N", "\\N", "\\N",
        ]]
        pred = assign_local_columns("PREDICATION", pred_rows)
        self.assertEqual(pred["status"], "assigned")
        self.assertEqual(pred["index"]["PREDICATE"], 3)
        self.assertEqual(pred["unconfirmed_indexes"], [12, 13, 14])


if __name__ == "__main__":
    unittest.main()

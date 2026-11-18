"""Fixture checks for parsing, mapping, path counts, and provenance."""

from __future__ import annotations

import unittest

from src.core_parse import parse_core_turtle
from src.mapping import (
    classify_attribute_sets,
    parse_endpoint_identifier,
    resolve_cui_gene_ids,
    resolve_gene_endpoint,
)
from src.path_count import brute_one_intermediate, brute_two_intermediate, exact_two_intermediate
from src.turtle_stream import iter_triples


CORE_FIXTURE = """
@prefix ceonto: <http://gda.dei.unipd.it/cecore/ontology/> .
@prefix cegcs: <http://gda.dei.unipd.it/cecore/resource/GCS#> .
@prefix cesent: <http://gda.dei.unipd.it/cecore/resource/Sentence#> .
@prefix ncbi: <https://www.ncbi.nlm.nih.gov/gene/> .
@prefix pubmed: <https://pubmed.ncbi.nlm.nih.gov/> .
@prefix umls: <http://linkedlifedata.com/resource/umls/id/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

cegcs:alpha a ceonto:GCS ;
    ceonto:expressedBy ncbi:7157 ;
    ceonto:hasType "TSG"^^xsd:string ;
    ceonto:involves umls:C0006826 ;
    ceonto:supportedBy cesent:s1, cesent:s2 .

cesent:s1 a <http://semanticscience.org/resource/SIO_000113> ;
    ceonto:CGELabel "UP"^^xsd:string ;
    ceonto:CCSLabel "PROGRESSION"^^xsd:string ;
    ceonto:GCILabel "CAUSALITY"^^xsd:string ;
    ceonto:GCCLabel "EXPRESSION"^^xsd:string ;
    ceonto:extractedFrom pubmed:111 ;
    ceonto:hasContent "This sentence is prose and must not be stored."^^xsd:string ;
    ceonto:mentionsGene ncbi:7157 ;
    ceonto:mentionsDisease umls:C0006826 .

cesent:s2 a <http://semanticscience.org/resource/SIO_000113> ;
    ceonto:CGELabel "DOWN"^^xsd:string ;
    ceonto:CCSLabel "NOTINF"^^xsd:string ;
    ceonto:GCILabel "NOTINF"^^xsd:string ;
    ceonto:GCCLabel "EXPRESSION"^^xsd:string ;
    ceonto:extractedFrom pubmed:222 ;
    ceonto:mentionsGene ncbi:7157 ;
    ceonto:mentionsDisease umls:C0006826 .

pubmed:111 ceonto:publicationYear "2001"^^xsd:int ;
    ceonto:title "A bibliographic title"^^xsd:string ;
    ceonto:cites pubmed:333 ;
    ceonto:abstract "Abstract prose must not be stored."^^xsd:string .
"""


class TurtleTests(unittest.TestCase):
    def test_long_string_is_one_literal(self):
        text = '''@prefix ex: <http://example.org/> .
ex:s ex:hasContent """line one; still one.
line \\"two\\".
"""^^ex:string ;
    ex:label "kept" .
'''
        triples = list(iter_triples(text))
        self.assertEqual(
            triples,
            [
                ("http://example.org/s", "http://example.org/hasContent", ""),
                ("http://example.org/s", "http://example.org/label", "kept"),
            ],
        )

    def test_gene_info_header_hash_is_taxonomy(self):
        import gzip
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from src.prepare_data import parse_gene_info

        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "gene_info.gz"
            path.write_bytes(
                gzip.compress(b"#tax_id\tGeneID\tSymbol\ttype_of_gene\n9606\t7157\tTP53\tprotein-coding\n10090\t22059\tTrp53\tprotein-coding\n")
            )
            stats = parse_gene_info(path, root, {"7157", "22059"})
            import pyarrow.parquet as pq

            rows = {row["gene_id"]: row for row in pq.read_table(root / "gene_info_matched.parquet").to_pylist()}
        self.assertEqual(stats["matched_gene_ids"], 2)
        self.assertEqual(rows["7157"]["tax_id"], "9606")
        self.assertEqual(rows["22059"]["tax_id"], "10090")

    def test_semmed_backslash_quote_keeps_null_marker(self):
        import gzip
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from src.prepare_data import ENTITY_COLUMNS, PREDICATE_COLUMNS, open_csv_gz

        with TemporaryDirectory() as directory:
            root = Path(directory)
            pred = root / "pred.csv.gz"
            ent = root / "ent.csv.gz"
            pred.write_bytes(
                gzip.compress(
                    b'"1","2","3","INTERACTS_WITH","C1","gene","gngm","1","C2","\\"U\\" lymphocyte","cell","1",\\N,\\N,\\N\n'
                )
            )
            ent.write_bytes(
                gzip.compress(
                    b'"9","8","7","C0450302","0.014\\"","qnco","","","0.014","901","1","2"\n'
                    b'"10","8","7","C0450303","bad\xffbyte","qnco","","","bad\xffbyte","901","1","2"\n'
                )
            )
            pred_row = next(iter(open_csv_gz(pred, PREDICATE_COLUMNS))).to_pylist()[0]
            ent_rows = []
            for batch in open_csv_gz(ent, ENTITY_COLUMNS):
                ent_rows.extend(batch.to_pylist())
        self.assertEqual(pred_row["object_name"], '"U" lymphocyte')
        self.assertIsNone(pred_row["trailing_1"])
        self.assertEqual(ent_rows[0]["name"], '0.014"')
        self.assertEqual(ent_rows[1]["name"], "bad\ufffdbyte")

    def test_utf8_characters_survive_small_reads(self):
        from src.prepare_data import _BackslashQuoteFile

        class OneByte:
            def __init__(self, data: bytes):
                self.data = data
                self.index = 0

            def read(self, size: int = -1) -> bytes:
                if self.index >= len(self.data):
                    return b""
                out = self.data[self.index : self.index + 1]
                self.index += 1
                return out

            def close(self) -> None:
                return None

        stream = _BackslashQuoteFile(OneByte(b'caf\xc3\xa9 \\"x" \xff'))
        collected = b""
        while True:
            block = stream.read(3)
            if not block:
                break
            collected += block
        self.assertEqual(collected.decode("utf-8"), 'café ""x" \ufffd')

    def test_quotes_and_prefixes(self):
        text = '@prefix ex: <http://example.org/> . ex:s ex:p "a, b. \\"c\\""^^ex:string .'
        triples = list(iter_triples(text))
        self.assertEqual(triples, [("http://example.org/s", "http://example.org/p", 'a, b. "c"')])

    def test_core_fixture_keeps_two_sentences_and_drops_prose(self):
        parsed = parse_core_turtle(CORE_FIXTURE)
        self.assertEqual(parsed["counts"]["content_literals_skipped"], 1)
        self.assertEqual(parsed["counts"]["abstract_literals_skipped"], 1)
        self.assertEqual(parsed["gcs"]["alpha"]["gene_ids"], ["7157"])
        self.assertEqual(parsed["gcs"]["alpha"]["cancer_cuis"], ["C0006826"])
        self.assertEqual(parsed["gcs"]["alpha"]["sentence_ids"], ["s1", "s2"])
        self.assertEqual(parsed["sentences"]["s1"]["cge_label"], "UP")
        self.assertEqual(parsed["sentences"]["s2"]["ccs_label"], "NOTINF")
        self.assertNotIn("hasContent", parsed["sentences"]["s1"])
        self.assertEqual(parsed["sentences"]["s1"]["pmid"], "111")
        self.assertEqual(parsed["citation_links"], [{"citing_pmid": "111", "cited_pmid": "333", "source": "core_cites"}])
        self.assertEqual(parsed["articles"]["111"]["years"], ["2001"])


class MappingTests(unittest.TestCase):
    def test_does_not_resolve_mixed_species_cui(self):
        result = resolve_cui_gene_ids({"7157", "22059"}, {"7157": "9606", "22059": "10090"})
        self.assertEqual(result["status"], "ambiguous_multiple_gene_ids")
        self.assertIsNone(result["gene_id"])

    def test_numeric_identifier_requires_human_gene_info(self):
        resolved = resolve_gene_endpoint("7157", None, {"7157": "9606"})
        self.assertEqual(resolved["status"], "resolved")
        self.assertEqual(resolved["gene_id"], "7157")
        mouse = resolve_gene_endpoint("22059", None, {"22059": "10090"})
        self.assertEqual(mouse["status"], "unresolved_nonhuman")

    def test_route_conflict_is_not_forced(self):
        cui = {"status": "resolved", "gene_id": "7157"}
        conflict = resolve_gene_endpoint("C0000001|9999", cui, {"9999": "9606", "7157": "9606"})
        self.assertEqual(conflict["status"], "unresolved_route_conflict")
        agree = resolve_gene_endpoint("C0000001|7157", cui, {"7157": "9606"})
        self.assertEqual(agree["status"], "resolved")
        self.assertEqual(agree["route"], "predication_identifier_and_entity_gene_id")

    def test_ambiguous_cui_is_not_broken_by_embedded_gene(self):
        cui = {"status": "ambiguous_multiple_gene_ids", "gene_id": None}
        result = resolve_gene_endpoint("C0000001|7157", cui, {"7157": "9606"})
        self.assertEqual(result["status"], "ambiguous_multiple_gene_ids")

    def test_negation_is_not_part_of_the_identifier_parser(self):
        self.assertEqual(parse_endpoint_identifier("C0006826")["pattern"], "cui")

    def test_attribute_sets(self):
        shared = {("UP", "PROGRESSION", "CAUSALITY", "EXPRESSION", "TSG")}
        other = {("DOWN", "PROGRESSION", "CAUSALITY", "EXPRESSION", "TSG")}
        self.assertEqual(classify_attribute_sets(shared, set()), "insufficient_information")
        self.assertEqual(classify_attribute_sets(shared, shared), "comparable_attributes")
        self.assertEqual(classify_attribute_sets(shared, other), "different_attributes")


class PathTests(unittest.TestCase):
    def test_templates_count_each_tuple_once_and_match(self):
        # Genes 0 and 1 are candidates. Gene 2 is only an intermediate.
        neighbors = {0: {1, 2}, 1: {0}, 2: {0}}
        genes_of_cancer = {0: {0, 1}, 1: {1, 2}, 2: {0}}
        candidates = {0, 1}
        queries = {1, 2}
        one = brute_one_intermediate(neighbors, genes_of_cancer, candidates, queries)
        # Query 1 genes {1,2}. Candidate partners of 1: {0}. Partners of 2: {0}.
        self.assertEqual(one[1]["n_paths"], 2)
        self.assertEqual(one[1]["reached"], {0})
        # Query 2 genes {0}. Partners of 0 that are candidates, excluding itself: {1,2} & candidates = {1}.
        self.assertEqual(one[2]["n_paths"], 1)
        self.assertEqual(one[2]["reached"], {1})
        brute = brute_two_intermediate(genes_of_cancer, candidates, queries, cancers={0, 1, 2})
        exact = exact_two_intermediate(genes_of_cancer, candidates, [1, 2], [0, 1, 2])
        for query in queries:
            self.assertEqual(exact[query]["n_paths"], brute[query]["n_paths"])
            self.assertEqual(exact[query]["n_reached"], len(brute[query]["reached"]))
        # Query 1, intermediate cancer 0, shared gene 1, candidates on cancer 0 are 0 and 1.
        # Only candidate 0 differs from shared gene 1. One path: 0 — cancer 0 — gene 1 — query 1.
        self.assertEqual(brute[1]["n_paths"], 1)
        self.assertEqual(brute[1]["reached"], {0})

    def test_self_loop_and_same_cancer_are_excluded(self):
        genes_of_cancer = {5: {1, 1}}
        result = brute_two_intermediate(genes_of_cancer, {1}, {5}, cancers={5})
        self.assertEqual(result[5]["n_paths"], 0)
        self.assertEqual(result[5]["reached"], set())

    def test_random_graphs_agree(self):
        graphs = [
            ({0: {0, 1}, 1: {1, 2}, 2: {2}}, {0, 2}, [0, 2], [0, 1, 2]),
            ({0: {3}, 1: {3, 4}, 2: {4}}, {3, 4}, [2], [0, 1, 2]),
            ({7: {1, 2, 3}, 8: {2, 3}, 9: {1}}, {1, 2, 3}, [8, 9], [7, 8, 9]),
        ]
        for genes_of_cancer, candidates, queries, cancers in graphs:
            brute = brute_two_intermediate(genes_of_cancer, set(candidates), set(queries), set(cancers))
            exact = exact_two_intermediate(genes_of_cancer, set(candidates), queries, cancers)
            for query in queries:
                self.assertEqual(exact[query]["n_paths"], brute[query]["n_paths"], msg=str(genes_of_cancer))
                self.assertEqual(exact[query]["n_reached"], len(brute[query]["reached"]), msg=str(genes_of_cancer))


if __name__ == "__main__":
    unittest.main()

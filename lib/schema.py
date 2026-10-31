"""Field dictionaries grounded in local schema and official descriptions.

Observed widths are not filled here. Jobs overwrite confirmation after preflight.
"""

from __future__ import annotations

CE_NS = "http://gda.dei.unipd.it/cecore/ontology/"
GENE_NS = "https://www.ncbi.nlm.nih.gov/gene/"
DISEASE_NS = "http://linkedlifedata.com/resource/umls/id/"
ARTICLE_NS = "https://pubmed.ncbi.nlm.nih.gov/"
SENTENCE_NS = "http://gda.dei.unipd.it/cecore/resource/Sentence#"
GCS_NS = "http://gda.dei.unipd.it/cecore/resource/GCS#"

TYPE_GENE = "http://purl.obolibrary.org/obo/SO_0000704"
TYPE_DISEASE = "https://www.ebi.ac.uk/ols/ontologies/doid/Disease"
TYPE_GCS = CE_NS + "GCS"
TYPE_SENTENCE = "http://semanticscience.org/resource/SIO_000113"
TYPE_SENTENCE_EQUIV = "http://purl.obolibrary.org/obo/NCIT_C62638"
TYPE_ARTICLE = "http://purl.obolibrary.org/obo/NCIT_C47902"
TYPE_DATABASE = "http://purl.obolibrary.org/obo/NCIT_C15426"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

# Directions the public builder writes (src/core/cecore.py). Inverses may be absent.
CORE_PREDICATES = {
    "expressedBy": CE_NS + "expressedBy",
    "involves": CE_NS + "involves",
    "supportedBy": CE_NS + "supportedBy",
    "hasType": CE_NS + "hasType",
    "mentionsGene": CE_NS + "mentionsGene",
    "mentionsDisease": CE_NS + "mentionsDisease",
    "hasContent": CE_NS + "hasContent",
    "extractedFrom": CE_NS + "extractedFrom",
    "includedIn": CE_NS + "includedIn",
    "CGELabel": CE_NS + "CGELabel",
    "CCSLabel": CE_NS + "CCSLabel",
    "GCILabel": CE_NS + "GCILabel",
    "GCCLabel": CE_NS + "GCCLabel",
    "CGE": CE_NS + "CGE",
    "CCS": CE_NS + "CCS",
    "GCI": CE_NS + "GCI",
    "GCC": CE_NS + "GCC",
    "GSAStartPosition": CE_NS + "GSAStartPosition",
    "GSAEndPosition": CE_NS + "GSAEndPosition",
    "GSAMention": CE_NS + "GSAMention",
    "DSAStartPosition": CE_NS + "DSAStartPosition",
    "DSAEndPosition": CE_NS + "DSAEndPosition",
    "DSAMention": CE_NS + "DSAMention",
    "curatedAnnotation": CE_NS + "curatedAnnotation",
    "title": CE_NS + "title",
    "abstract": CE_NS + "abstract",
    "hasVenue": CE_NS + "hasVenue",
    "publicationYear": CE_NS + "publicationYear",
    "cites": CE_NS + "cites",
    "hasCitation": CE_NS + "hasCitation",
}

OWL_LABEL_ENUM = {
    "CCSLabel": ["NOTINF", "PROGRESSION", "REGRESSION"],
    "CGELabel": ["DOWN", "UP"],
    "GCCLabel": ["EXPRESSION", "OTHER"],
    "GCILabel": ["CAUSALITY", "NOTINF", "OBSERVATION"],
    "hasType": ["BIOMARKER", "ONCOGENE", "TSG"],
}

# Builder (utils.py tag_gcs2type) can also store UNRELIABLE, which is outside the OWL oneOf.
BUILDER_EXTRA_HASTYPE = ["UNRELIABLE"]

# SPARQL example 5 excludes these CUIs from a "most discussed diseases" query.
# This is an example-query filter, not a local cancer definition.
SPARQL_EXAMPLE_EXCLUDED_CUIS = ["C0086692", "C0027651", "C0006826", "C1306459"]

# Official dbinfo.html column order for the post-VER30 schema used by VER43.
# confirmation_status stays unconfirmed until preflight measures widths.
SEMMED_TABLES = {
    "CITATIONS": {
        "filename": "semmedVER43_2024_R_CITATIONS.23327.csv",
        "columns": ["PMID", "ISSN", "DP", "EDAT", "PYEAR"],
        "primary_key": ["PMID"],
        "notes": {
            "DP": "dbinfo: publication date for the citation. Year is parsed from this field.",
            "EDAT": "dbinfo: date the citation was added to PubMed. Not a publication date.",
            "PYEAR": "dbinfo: completion date for the citation. Not used as publication year.",
        },
    },
    "GENERIC_CONCEPT": {
        "filename": "semmedVER43_2024_R_GENERIC_CONCEPT.23327.csv",
        "columns": ["CONCEPT_ID", "CUI", "PREFERRED_NAME"],
        "primary_key": ["CONCEPT_ID"],
        "notes": {
            "CUI": "dbinfo: UMLS CUI of a concept treated as generic. Novelty is 0 when the concept is in this table.",
        },
    },
    "SENTENCE": {
        "filename": "semmedVER43_2024_R_SENTENCE.23327.csv",
        "columns": [
            "SENTENCE_ID",
            "PMID",
            "TYPE",
            "NUMBER",
            "SENT_START_INDEX",
            "SENT_END_INDEX",
            "SECTION_HEADER",
            "NORMALIZED_SECTION_HEADER",
            "SENTENCE",
        ],
        "primary_key": ["SENTENCE_ID"],
        "foreign_keys": [{"columns": ["PMID"], "ref": "CITATIONS.PMID"}],
        "notes": {
            "TYPE": "dbinfo: ti title, ab abstract.",
            "SENTENCE": "Sentence string. A set of these is not treated as a reconstructed full abstract.",
        },
    },
    "PREDICATION": {
        "filename": "semmedVER43_2024_R_PREDICATION.23327.csv",
        "columns": [
            "PREDICATION_ID",
            "SENTENCE_ID",
            "PMID",
            "PREDICATE",
            "SUBJECT_CUI",
            "SUBJECT_NAME",
            "SUBJECT_SEMTYPE",
            "SUBJECT_NOVELTY",
            "OBJECT_CUI",
            "OBJECT_NAME",
            "OBJECT_SEMTYPE",
            "OBJECT_NOVELTY",
        ],
        "primary_key": ["PREDICATION_ID"],
        "foreign_keys": [
            {"columns": ["SENTENCE_ID"], "ref": "SENTENCE.SENTENCE_ID"},
            {"columns": ["PMID"], "ref": "CITATIONS.PMID"},
        ],
        "notes": {
            "SUBJECT_NOVELTY": "Download page: 1 if the concept is not in GENERIC_CONCEPT, else 0. Not research credibility.",
            "PREDICATE": "Raw predicate string. A NEG_ prefix is recorded as a string flag, not a scientific judgment.",
        },
    },
    "PREDICATION_AUX": {
        "filename": "semmedVER43_2024_R_PREDICATION_AUX.23327.csv",
        "columns": [
            "PREDICATION_AUX_ID",
            "PREDICATION_ID",
            "SUBJECT_TEXT",
            "SUBJECT_DIST",
            "SUBJECT_MAXDIST",
            "SUBJECT_START_INDEX",
            "SUBJECT_END_INDEX",
            "SUBJECT_SCORE",
            "INDICATOR_TYPE",
            "PREDICATE_START_INDEX",
            "PREDICATE_END_INDEX",
            "OBJECT_TEXT",
            "OBJECT_DIST",
            "OBJECT_MAXDIST",
            "OBJECT_START_INDEX",
            "OBJECT_END_INDEX",
            "OBJECT_SCORE",
            "CURR_TIMESTAMP",
        ],
        "primary_key": ["PREDICATION_AUX_ID"],
        "foreign_keys": [{"columns": ["PREDICATION_ID"], "ref": "PREDICATION.PREDICATION_ID"}],
        "notes": {
            "relation": "dbinfo text says 1-to-1 with PREDICATION. The download page lists 14 fewer AUX rows than PREDICATION rows. Local measurement decides; a row-count difference alone is not called corruption.",
            "SUBJECT_SCORE": "dbinfo: confidence of the string-to-concept mapping. Not study credibility.",
            "CURR_TIMESTAMP": "dbinfo: timestamp for the record. Not article publication time.",
        },
    },
    "ENTITY": {
        "filename": "semmedVER43_2024_R_ENTITY.23327.csv",
        "columns": [
            "ENTITY_ID",
            "SENTENCE_ID",
            "CUI",
            "NAME",
            "TYPE",
            "GENE_ID",
            "GENE_NAME",
            "TEXT",
            "START_INDEX",
            "END_INDEX",
            "SCORE",
        ],
        "primary_key": ["ENTITY_ID"],
        "foreign_keys": [{"columns": ["SENTENCE_ID"], "ref": "SENTENCE.SENTENCE_ID"}],
        "notes": {
            "GENE_ID": "dbinfo: Entrez Gene ID of the entity. Used only when the value is a single numeric token.",
            "TYPE": "dbinfo: semantic type of the entity.",
            "SCORE": "dbinfo: confidence of the string-to-concept mapping.",
            "START_INDEX": "dbinfo: character position in the document, not a sentence-relative CORE offset.",
        },
    },
}

# Operational semantic-type subset. Not a complete cancer collection.
SCOPE_E_GENE_SEMTYPE = "gngm"
SCOPE_E_DISEASE_SEMTYPE = "neop"
SCOPE_E_PROTEIN_SEMTYPE = "aapp"

NULL_TOKENS = ("", "\\N", "NULL", "null", "None")


def semmed_path(semmed_dir, table: str):
    from pathlib import Path

    return Path(semmed_dir) / SEMMED_TABLES[table]["filename"]

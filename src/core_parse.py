"""Parse CORE Turtle into records without retaining article or sentence prose."""

from __future__ import annotations

from src.mapping import core_has_type_path_eligible
from src.turtle_stream import iter_triples

ONTO = "http://gda.dei.unipd.it/cecore/ontology/"
GENE_NS = "https://www.ncbi.nlm.nih.gov/gene/"
DISEASE_NS = "http://linkedlifedata.com/resource/umls/id/"
ARTICLE_NS = "https://pubmed.ncbi.nlm.nih.gov/"
SENTENCE_NS = "http://gda.dei.unipd.it/cecore/resource/Sentence#"
GCS_NS = "http://gda.dei.unipd.it/cecore/resource/GCS#"

SENTENCE_LITERALS = {
    "CCSLabel": "ccs_label",
    "CGELabel": "cge_label",
    "GCCLabel": "gcc_label",
    "GCILabel": "gci_label",
    "DSAStartPosition": "disease_start",
    "DSAEndPosition": "disease_end",
    "GSAStartPosition": "gene_start",
    "GSAEndPosition": "gene_end",
    "DSAMention": "disease_mention",
    "GSAMention": "gene_mention",
    "curatedAnnotation": "curated_annotation",
}


def _local(iri: str, namespace: str) -> str | None:
    if iri.startswith(namespace):
        return iri[len(namespace) :]
    return None


def _pmid(value: str) -> str | None:
    local = _local(value, ARTICLE_NS)
    if local is None:
        text = value.strip()
        if text.isdigit():
            local = text
        else:
            return None
    if not local.isdigit():
        return None
    return local.lstrip("0") or "0"


def blank_sentence() -> dict:
    return {
        "gcs_id": None,
        "pmid": None,
        "cge_label": "",
        "ccs_label": "",
        "gci_label": "",
        "gcc_label": "",
        "disease_mention": "",
        "gene_mention": "",
        "disease_start": "",
        "disease_end": "",
        "gene_start": "",
        "gene_end": "",
        "curated_annotation": "",
        "mentions_gene": None,
        "mentions_cancer": None,
    }


def parse_core_turtle(text: str) -> dict:
    gcs: dict[str, dict] = {}
    sentences: dict[str, dict] = {}
    articles: dict[str, dict] = {}
    citation_links: list[dict] = []
    counts = {
        "triples": 0,
        "content_literals_skipped": 0,
        "abstract_literals_skipped": 0,
        "gcs_multiple_gene": 0,
        "gcs_multiple_cancer": 0,
        "unparsed_citation_values": 0,
    }

    def ensure_gcs(gcs_id: str) -> dict:
        return gcs.setdefault(
            gcs_id,
            {"gene_ids": [], "cancer_cuis": [], "has_type": "", "sentence_ids": []},
        )

    def ensure_sentence(sentence_id: str) -> dict:
        return sentences.setdefault(sentence_id, blank_sentence())

    def ensure_article(pmid: str) -> dict:
        return articles.setdefault(
            pmid,
            {"years": [], "venues": [], "titles": [], "source": "core"},
        )

    for subject, predicate, obj in iter_triples(text):
        counts["triples"] += 1
        if predicate == ONTO + "hasContent":
            counts["content_literals_skipped"] += 1
            continue
        if predicate == ONTO + "abstract":
            counts["abstract_literals_skipped"] += 1
            continue
        gcs_id = _local(subject, GCS_NS)
        sentence_id = _local(subject, SENTENCE_NS)
        pmid = _local(subject, ARTICLE_NS)
        if gcs_id is not None:
            row = ensure_gcs(gcs_id)
            if predicate == ONTO + "expressedBy":
                gene = _local(obj, GENE_NS)
                if gene and gene.isdigit():
                    gene = gene.lstrip("0") or "0"
                    if row["gene_ids"] and gene not in row["gene_ids"]:
                        counts["gcs_multiple_gene"] += 1
                    if gene not in row["gene_ids"]:
                        row["gene_ids"].append(gene)
            elif predicate == ONTO + "involves":
                cui = _local(obj, DISEASE_NS)
                if cui:
                    if row["cancer_cuis"] and cui not in row["cancer_cuis"]:
                        counts["gcs_multiple_cancer"] += 1
                    if cui not in row["cancer_cuis"]:
                        row["cancer_cuis"].append(cui)
            elif predicate == ONTO + "hasType":
                row["has_type"] = obj
            elif predicate == ONTO + "supportedBy":
                supported = _local(obj, SENTENCE_NS)
                if supported and supported not in row["sentence_ids"]:
                    row["sentence_ids"].append(supported)
                    ensure_sentence(supported)["gcs_id"] = gcs_id
            continue
        if sentence_id is not None:
            row = ensure_sentence(sentence_id)
            local = predicate[len(ONTO) :] if predicate.startswith(ONTO) else ""
            if local in SENTENCE_LITERALS:
                row[SENTENCE_LITERALS[local]] = obj
            elif predicate == ONTO + "extractedFrom":
                row["pmid"] = _pmid(obj)
            elif predicate == ONTO + "mentionsGene":
                gene = _local(obj, GENE_NS)
                if gene and gene.isdigit():
                    row["mentions_gene"] = gene.lstrip("0") or "0"
            elif predicate == ONTO + "mentionsDisease":
                row["mentions_cancer"] = _local(obj, DISEASE_NS)
            continue
        if pmid is not None and pmid.isdigit():
            pmid = pmid.lstrip("0") or "0"
            row = ensure_article(pmid)
            if predicate == ONTO + "publicationYear" and obj.isdigit():
                if obj not in row["years"]:
                    row["years"].append(obj)
            elif predicate == ONTO + "hasVenue" and obj not in row["venues"]:
                row["venues"].append(obj)
            elif predicate == ONTO + "title" and obj not in row["titles"]:
                row["titles"].append(obj)
            elif predicate in {ONTO + "cites", ONTO + "hasCitation"}:
                cited = _pmid(obj)
                if cited is None:
                    counts["unparsed_citation_values"] += 1
                elif predicate == ONTO + "cites":
                    citation_links.append({"citing_pmid": pmid, "cited_pmid": cited, "source": "core_cites"})
                else:
                    citation_links.append({"citing_pmid": cited, "cited_pmid": pmid, "source": "core_hasCitation"})
    for gcs_id, row in gcs.items():
        row["path_eligible_has_type"] = core_has_type_path_eligible(row["has_type"])
        for sentence_id in row["sentence_ids"]:
            ensure_sentence(sentence_id)["gcs_id"] = gcs_id
    counts["n_gcs"] = len(gcs)
    counts["n_sentences"] = len(sentences)
    counts["n_articles"] = len(articles)
    counts["n_citation_links"] = len(citation_links)
    return {
        "gcs": gcs,
        "sentences": sentences,
        "articles": articles,
        "citation_links": citation_links,
        "counts": counts,
    }

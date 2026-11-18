"""Build the provenance-preserving joint graph from parsed records."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.common import SCRATCH_ROOT, ensure_dir, load_yaml, normalise_pmid, write_json
from src.mapping import HUMAN_TAX, core_has_type_path_eligible, parse_endpoint_identifier, resolve_gene_endpoint

EVIDENCE_COLUMNS = [
    "evidence_id",
    "resource",
    "native_id",
    "parent_record_id",
    "pmid",
    "sentence_id",
    "path_eligible",
    "predicate",
    "negation",
    "gcs_has_type",
    "cge_label",
    "ccs_label",
    "gci_label",
    "gcc_label",
    "curated_annotation",
    "subject_semtype",
    "object_semtype",
    "subject_novelty",
    "object_novelty",
    "resolution_status",
]


class GraphWriter:
    def __init__(self, path: Path, columns: list[str]):
        self.path = path
        self.columns = columns
        self.rows: list[dict] = []
        self.writer = None

    def add(self, row: dict) -> None:
        self.rows.append({column: row.get(column, "") for column in self.columns})
        if len(self.rows) >= 50000:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        table = pa.Table.from_pylist(self.rows)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="snappy")
        self.writer.write_table(table)
        self.rows.clear()

    def close(self) -> None:
        self.flush()
        if self.writer is not None:
            self.writer.close()
            return
        empty = pa.table({name: pa.array([], type=pa.string()) for name in self.columns})
        pq.write_table(empty, self.path)


def load_lookup(path: Path) -> tuple[dict, dict, dict]:
    rows = pq.read_table(path).to_pylist()
    tax = {row["gene_id"]: row["tax_id"] for row in rows}
    symbol = {row["gene_id"]: row["symbol"] for row in rows}
    gene_type = {row["gene_id"]: row["type_of_gene"] for row in rows}
    return tax, symbol, gene_type


def load_cui_resolution(path: Path) -> dict[str, dict]:
    out = {}
    for row in pq.read_table(path).to_pylist():
        out[row["cui"]] = {
            "status": row["status"],
            "gene_id": row["gene_id"] or None,
            "route": row["route"],
        }
    return out


def publication_year_from_dp(dp: str) -> str:
    for token in (dp or "").replace("-", " ").split():
        if len(token) == 4 and token.isdigit():
            return token
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build joint graph parquet tables.")
    parser.add_argument("--scratch", type=Path, default=SCRATCH_ROOT / "prepared")
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--design", type=Path, default=Path("config/study_design.yaml"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    design = load_yaml(args.design)
    entity_stats_path = args.scratch / "entity_stats.json"
    if entity_stats_path.exists():
        import json

        entity_stats = json.loads(entity_stats_path.read_text(encoding="utf-8"))
        truncated = "truncated_by_dry_run_cap" in entity_stats
        if entity_stats.get("gene_id_column_status") != "confirmed" and not truncated:
            raise RuntimeError("ENTITY gene-identifier column was not confirmed; graph build stopped")
    graph_dir = ensure_dir(args.run_dir / "graph")
    tax, symbol, gene_type = load_lookup(args.scratch / "gene_info_matched.parquet")
    cui_resolution = load_cui_resolution(args.alignment / "cui_gene_resolution.parquet")
    core_gene_status = {
        row["gene_id"]: row["status"] for row in pq.read_table(args.alignment / "core_gene_resolution.parquet").to_pylist()
    }
    names = {}
    name_path = args.scratch / "cui_preferred_names.parquet"
    if name_path.exists():
        for row in pq.read_table(name_path).to_pylist():
            names[row["cui"]] = row["preferred_name"]
    allow_gene = set(design["semmeddb_predications"]["gene_gene_allowed_predicates"])
    allow_cancer = set(design["semmeddb_predications"]["gene_cancer_allowed_predicates"])
    candidate_gene_ids = {gene_id for gene_id, status in core_gene_status.items() if status == "resolved_human"}
    stats = Counter()
    relations: dict[tuple, dict] = {}
    evidence = GraphWriter(graph_dir / "evidence_records.parquet", EVIDENCE_COLUMNS)
    junction = GraphWriter(
        graph_dir / "relation_evidence.parquet",
        ["relation_id", "evidence_id", "path_eligible"],
    )
    entity_seen: dict[str, dict] = {}
    articles: dict[str, dict] = {}
    evidence_seq = 0

    def entity_gene(gene_id: str) -> str:
        entity_id = f"gene:{gene_id}"
        entity_seen.setdefault(
            entity_id,
            {
                "entity_id": entity_id,
                "canonical_id": gene_id,
                "original_identifiers": f"ncbigene:{gene_id}",
                "display_label": symbol.get(gene_id, gene_id),
                "entity_type": "gene",
                "species": "Homo sapiens" if tax.get(gene_id) == HUMAN_TAX else tax.get(gene_id, ""),
                "type_of_gene": gene_type.get(gene_id, ""),
            },
        )
        return entity_id

    def entity_cancer(cui: str) -> str:
        entity_id = f"cancer:{cui}"
        entity_seen.setdefault(
            entity_id,
            {
                "entity_id": entity_id,
                "canonical_id": cui,
                "original_identifiers": f"umls:{cui}",
                "display_label": names.get(cui, cui),
                "entity_type": "cancer",
                "species": "",
                "type_of_gene": "",
            },
        )
        return entity_id

    def touch_article(pmid: str, source: str) -> None:
        if not pmid:
            return
        row = articles.setdefault(
            pmid,
            {"pmid": pmid, "sources": set(), "years": set(), "venues": set(), "titles": set(), "issn": "", "dp": "", "edat": "", "pyear": ""},
        )
        row["sources"].add(source)

    def add_relation(key: tuple, fields: dict) -> dict:
        row = relations.get(key)
        if row is None:
            row = {"relation_id": f"REL{len(relations) + 1:09d}", "n_evidence": 0, "n_path_eligible_evidence": 0, **fields}
            relations[key] = row
            stats["canonical_relationships"] += 1
        return row

    def add_evidence(row: dict, relation_rows: list[tuple[dict, bool]]) -> None:
        nonlocal evidence_seq
        evidence_seq += 1
        evidence_id = f"EV{evidence_seq:09d}"
        row["evidence_id"] = evidence_id
        evidence.add(row)
        stats["evidence_records"] += 1
        stats[f"evidence_{row['resource']}"] += 1
        if row["path_eligible"] == "true":
            stats["evidence_path_eligible"] += 1
        touch_article(row.get("pmid", ""), row["resource"])
        for relation, eligible in relation_rows:
            relation["n_evidence"] += 1
            if eligible:
                relation["n_path_eligible_evidence"] += 1
            junction.add(
                {
                    "relation_id": relation["relation_id"],
                    "evidence_id": evidence_id,
                    "path_eligible": "true" if eligible else "false",
                }
            )

    gcs_rows = pq.read_table(args.scratch / "core_gcs.parquet").to_pylist()
    sentence_rows = pq.read_table(args.scratch / "core_sentences.parquet").to_pylist()
    by_gcs = defaultdict(list)
    for sentence in sentence_rows:
        if sentence.get("gcs_id"):
            by_gcs[sentence["gcs_id"]].append(sentence)
    stats["core_gcs_input"] = len(gcs_rows)
    stats["core_sentence_input"] = len(sentence_rows)
    for gcs in gcs_rows:
        gene_ids = [gene for gene in str(gcs["gene_ids"]).split("|") if gene]
        cancer_cuis = [cui for cui in str(gcs["cancer_cuis"]).split("|") if cui]
        if len(gene_ids) > 1 or len(cancer_cuis) > 1:
            stats["core_multi_pair_gcs"] += 1
        has_type = gcs.get("has_type") or ""
        type_ok = bool(gcs.get("path_eligible_has_type")) and core_has_type_path_eligible(has_type)
        if not type_ok:
            stats["core_has_type_excluded"] += 1
        pairs = []
        for gene_id in gene_ids:
            status = core_gene_status.get(gene_id, "unresolved_unknown_gene_id")
            if status != "resolved_human":
                stats[f"core_gene_{status}"] += 1
                continue
            for cui in cancer_cuis:
                if not cui.startswith("C"):
                    stats["core_cancer_unresolved"] += 1
                    continue
                pairs.append((gene_id, cui))
        if not pairs:
            stats["core_gcs_without_resolved_pair"] += 1
            continue
        stats["core_gcs_mapped"] += 1
        linked = []
        for gene_id, cui in pairs:
            eligible = type_ok
            relation = add_relation(
                ("gene_cancer_association", f"gene:{gene_id}", f"cancer:{cui}", "gene_cancer_association", False),
                {
                    "resource": "core",
                    "edge_family": "gene_cancer_association",
                    "subject_entity_id": entity_gene(gene_id),
                    "object_entity_id": entity_cancer(cui),
                    "predicate": "gene_cancer_association",
                    "negation": "false",
                    "asserted_direction": "gene_subject_cancer_object",
                },
            )
            linked.append((relation, eligible))
        sentences = by_gcs.get(gcs["gcs_id"], [])
        if not sentences:
            stats["core_gcs_without_sentence"] += 1
            add_evidence(
                {
                    "resource": "core",
                    "native_id": gcs["gcs_id"],
                    "parent_record_id": gcs["gcs_id"],
                    "pmid": "",
                    "sentence_id": "",
                    "path_eligible": "true" if type_ok else "false",
                    "predicate": "gene_cancer_association",
                    "negation": "false",
                    "gcs_has_type": has_type,
                    "resolution_status": "resolved",
                },
                linked,
            )
            continue
        for sentence in sentences:
            pmid = normalise_pmid(sentence.get("pmid")) or ""
            if sentence.get("mentions_gene") and sentence["mentions_gene"] not in gene_ids:
                stats["core_sentence_gene_mention_mismatch"] += 1
            if sentence.get("mentions_cancer") and sentence["mentions_cancer"] not in cancer_cuis:
                stats["core_sentence_cancer_mention_mismatch"] += 1
            add_evidence(
                {
                    "resource": "core",
                    "native_id": sentence["sentence_id"],
                    "parent_record_id": gcs["gcs_id"],
                    "pmid": pmid,
                    "sentence_id": sentence["sentence_id"],
                    "path_eligible": "true" if type_ok else "false",
                    "predicate": "gene_cancer_association",
                    "negation": "false",
                    "gcs_has_type": has_type,
                    "cge_label": sentence.get("cge_label") or "",
                    "ccs_label": sentence.get("ccs_label") or "",
                    "gci_label": sentence.get("gci_label") or "",
                    "gcc_label": sentence.get("gcc_label") or "",
                    "curated_annotation": str(sentence.get("curated_annotation") or ""),
                    "resolution_status": "resolved",
                },
                linked,
            )
    seen_pred = set()
    pred_path = args.scratch / "semmed_predications.parquet"
    if pred_path.exists():
        parquet = pq.ParquetFile(pred_path)
        for batch in parquet.iter_batches(batch_size=50000):
            for row in batch.to_pylist():
                stats["semmed_rows_read"] += 1
                native = str(row["predication_id"])
                if native in seen_pred:
                    stats["semmed_duplicate_predication_id"] += 1
                    continue
                seen_pred.add(native)
                subject = resolve_endpoint(row["subject_raw"], row["subject_class"], cui_resolution, tax)
                obj = resolve_endpoint(row["object_raw"], row["object_class"], cui_resolution, tax)
                stats[f"semmed_subject_{subject['status']}"] += 1
                stats[f"semmed_object_{obj['status']}"] += 1
                if subject["status"] != "resolved" or obj["status"] != "resolved":
                    stats["semmed_unresolved_endpoint"] += 1
                    continue
                if subject["entity_id"] == obj["entity_id"]:
                    stats["semmed_self_loop"] += 1
                    continue
                if row["family"] == "gene_gene" and subject["local_id"] not in candidate_gene_ids and obj["local_id"] not in candidate_gene_ids:
                    # Declared graph boundary: a gene-gene edge is retained only when
                    # at least one endpoint is a candidate gene. The row is counted here.
                    stats["semmed_gene_gene_excluded_no_candidate"] += 1
                    continue
                if subject["kind"] == "gene":
                    entity_gene(subject["local_id"])
                else:
                    entity_cancer(subject["local_id"])
                if obj["kind"] == "gene":
                    entity_gene(obj["local_id"])
                else:
                    entity_cancer(obj["local_id"])
                family = row["family"]
                base = row["base_predicate"]
                negated = row["negation"] is True or str(row["negation"]).lower() == "true"
                allowed = base in (allow_gene if family == "gene_gene" else allow_cancer)
                eligible = allowed and not negated
                if not allowed:
                    stats["semmed_predicate_outside_allow_list"] += 1
                if negated:
                    stats["semmed_negated_retained"] += 1
                relation = add_relation(
                    ("semantic_predication", subject["entity_id"], obj["entity_id"], row["predicate"], negated),
                    {
                        "resource": "semmeddb",
                        "edge_family": "semantic_predication",
                        "subject_entity_id": subject["entity_id"],
                        "object_entity_id": obj["entity_id"],
                        "predicate": row["predicate"],
                        "negation": "true" if negated else "false",
                        "asserted_direction": "subject_to_object",
                        "base_predicate": base,
                        "connectivity_family": family,
                    },
                )
                add_evidence(
                    {
                        "resource": "semmeddb",
                        "native_id": native,
                        "parent_record_id": str(row.get("sentence_id") or ""),
                        "pmid": normalise_pmid(row.get("pmid")) or "",
                        "sentence_id": str(row.get("sentence_id") or ""),
                        "path_eligible": "true" if eligible else "false",
                        "predicate": row["predicate"],
                        "negation": "true" if negated else "false",
                        "subject_semtype": row.get("subject_semtype") or "",
                        "object_semtype": row.get("object_semtype") or "",
                        "subject_novelty": row.get("subject_novelty") or "",
                        "object_novelty": row.get("object_novelty") or "",
                        "resolution_status": "resolved",
                    },
                    [(relation, eligible)],
                )
                stats["semmed_mapped_records"] += 1
    evidence.close()
    junction.close()
    relation_rows = []
    connectivity = set()
    for row in relations.values():
        if row["n_path_eligible_evidence"]:
            left = row["subject_entity_id"]
            right = row["object_entity_id"]
            family = row.get("connectivity_family") or ("gene_cancer" if row["resource"] == "core" else "gene_gene")
            if row["resource"] == "core":
                family = "gene_cancer"
            connectivity.add((family, tuple(sorted((left, right)))))
        relation_rows.append(
            {
                "relation_id": row["relation_id"],
                "resource": row["resource"],
                "edge_family": row["edge_family"],
                "subject_entity_id": row["subject_entity_id"],
                "object_entity_id": row["object_entity_id"],
                "predicate": row["predicate"],
                "negation": row["negation"],
                "asserted_direction": row["asserted_direction"],
                "base_predicate": row.get("base_predicate") or row["predicate"],
                "connectivity_family": row.get("connectivity_family") or "gene_cancer",
                "n_evidence": row["n_evidence"],
                "n_path_eligible_evidence": row["n_path_eligible_evidence"],
            }
        )
    pq.write_table(pa.Table.from_pylist(relation_rows), graph_dir / "relations.parquet")
    pq.write_table(pa.Table.from_pylist(list(entity_seen.values())), graph_dir / "entities.parquet")
    for pmid, row in list(articles.items()):
        row["sources"] = "|".join(sorted(row["sources"]))
        row["years"] = "|".join(sorted(row["years"]))
        row["venues"] = "|".join(row["venues"])
        row["titles"] = "|".join(row["titles"])
    core_articles = {row["pmid"]: row for row in pq.read_table(args.scratch / "core_articles.parquet").to_pylist()}
    citation_path = args.scratch / "semmed_citations_matched.parquet"
    semmed_articles = {}
    if citation_path.exists():
        semmed_articles = {row["pmid"]: row for row in pq.read_table(citation_path).to_pylist()}
    article_rows = []
    for pmid, row in articles.items():
        core = core_articles.get(pmid, {})
        semmed = semmed_articles.get(pmid, {})
        years = set(filter(None, (row["years"].split("|") if row["years"] else [])))
        if core.get("years"):
            years.update(part for part in str(core["years"]).split("|") if part)
        dp_year = publication_year_from_dp(semmed.get("dp", ""))
        if dp_year:
            years.add(dp_year)
        sources = set(row["sources"].split("|")) if row["sources"] else set()
        if core:
            sources.add("core")
        if semmed:
            sources.add("semmeddb")
        article_rows.append(
            {
                "pmid": pmid,
                "sources": "|".join(sorted(sources)),
                "publication_years": "|".join(sorted(years)),
                "publication_year_from_dp": dp_year,
                "venues": core.get("venues", ""),
                "titles": core.get("titles", ""),
                "issn": semmed.get("issn", ""),
                "dp": semmed.get("dp", ""),
                "edat": semmed.get("edat", ""),
                "pyear_completion": semmed.get("pyear", ""),
            }
        )
    pq.write_table(pa.Table.from_pylist(article_rows), graph_dir / "articles.parquet")
    link_path = args.scratch / "core_citation_links.parquet"
    link_rows = pq.read_table(link_path).to_pylist() if link_path.exists() else []
    valid_links = []
    for link in link_rows:
        citing = normalise_pmid(link.get("citing_pmid"))
        cited = normalise_pmid(link.get("cited_pmid"))
        if not citing or not cited or citing == cited:
            stats["citation_links_rejected"] += 1
            continue
        valid_links.append({"citing_pmid": citing, "cited_pmid": cited, "source": link.get("source", "core"), "citation_location": ""})
    if valid_links:
        pq.write_table(pa.Table.from_pylist(valid_links), graph_dir / "citations.parquet")
    stats["entities"] = len(entity_seen)
    stats["relations"] = len(relation_rows)
    stats["connectivity_edges_with_eligible_evidence"] = len(connectivity)
    stats["articles"] = len(article_rows)
    stats["citation_links"] = len(valid_links)
    write_json(graph_dir / "build_stats.json", dict(stats))
    print(f"graph entities={stats['entities']} relations={stats['relations']} evidence={stats['evidence_records']}", flush=True)


def resolve_endpoint(raw: str, endpoint_class_name: str, cui_resolution: dict, tax: dict) -> dict:
    if endpoint_class_name == "cancer":
        parsed = parse_endpoint_identifier(raw)
        if parsed["pattern"] == "cui":
            return {"status": "resolved", "kind": "cancer", "local_id": parsed["cui"], "entity_id": f"cancer:{parsed['cui']}"}
        return {"status": "unresolved_cancer_identifier", "kind": "cancer", "local_id": None, "entity_id": None}
    if endpoint_class_name != "gene":
        return {"status": "not_gene_or_cancer", "kind": endpoint_class_name, "local_id": None, "entity_id": None}
    parsed = parse_endpoint_identifier(raw)
    cui_row = cui_resolution.get(parsed["cui"]) if parsed["cui"] else None
    resolved = resolve_gene_endpoint(raw, cui_row, tax)
    if resolved["status"] != "resolved":
        return {"status": resolved["status"], "kind": "gene", "local_id": None, "entity_id": None}
    gene_id = resolved["gene_id"]
    return {"status": "resolved", "kind": "gene", "local_id": gene_id, "entity_id": f"gene:{gene_id}", "route": resolved["route"]}


if __name__ == "__main__":
    main()

"""Count the two short path templates and write the audit tables.

one_intermediate_path has two edges.
two_intermediate_path has three edges.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from src.common import ensure_dir, write_json
from src.mapping import classify_attribute_sets, informative_core_tuple
from src.path_count import direct_candidate_counts, exact_two_intermediate


def as_bool(value) -> bool:
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def local_id(entity_id: str) -> tuple[str, str]:
    kind, _, value = entity_id.partition(":")
    return kind, value


def signature(tuples: set[tuple]) -> str:
    if not tuples:
        return ""
    return " || ".join("|".join(part) for part in sorted(tuples))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract evidence-path coverage and representation tables.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    args = parser.parse_args()
    analysis = ensure_dir(args.run_dir / "analysis")
    graph = args.run_dir / "graph"
    relations = {row["relation_id"]: row for row in pq.read_table(graph / "relations.parquet").to_pylist()}
    links = defaultdict(list)
    for row in pq.read_table(graph / "relation_evidence.parquet").to_pylist():
        links[row["evidence_id"]].append(row)
    candidates = []
    for row in pq.read_table(args.alignment / "core_gene_resolution.parquet").to_pylist():
        if row["status"] == "resolved_human":
            candidates.append(row)
    candidate_ids = {row["gene_id"] for row in candidates}
    queries = []
    seen_queries = set()
    for row in pq.read_table(args.scratch / "core_gcs.parquet").to_pylist():
        for cui in str(row["cancer_cuis"]).split("|"):
            if cui and cui not in seen_queries:
                seen_queries.add(cui)
                queries.append(cui)
    details = {}

    def detail(key):
        return details.setdefault(
            key,
            {
                "core_records": 0,
                "semmed_allowed": 0,
                "predicates": set(),
                "extra_predicates": set(),
                "orientations": set(),
                "negated": 0,
                "pmids_core": set(),
                "pmids_semmed": set(),
                "core_tuples": set(),
                "core_records_unknown_only": 0,
            },
        )

    evidence_file = pq.ParquetFile(graph / "evidence_records.parquet")
    for batch in evidence_file.iter_batches(batch_size=50000):
        for evidence in batch.to_pylist():
            for link in links.get(evidence["evidence_id"], []):
                relation = relations[link["relation_id"]]
                subject_kind, subject_local = local_id(relation["subject_entity_id"])
                object_kind, object_local = local_id(relation["object_entity_id"])
                if relation["resource"] == "core" or relation.get("connectivity_family") == "gene_cancer":
                    gene = subject_local if subject_kind == "gene" else object_local
                    cancer = object_local if object_kind == "cancer" else subject_local
                    if subject_kind == object_kind:
                        continue
                    key = ("gene_cancer", gene, cancer)
                else:
                    key = ("gene_gene", *sorted((subject_local, object_local)))
                item = detail(key)
                eligible = as_bool(link["path_eligible"])
                pmid = evidence.get("pmid") or ""
                if relation["resource"] == "core":
                    item["core_records"] += 1
                    tuple_value = (
                        evidence.get("cge_label") or "",
                        evidence.get("ccs_label") or "",
                        evidence.get("gci_label") or "",
                        evidence.get("gcc_label") or "",
                        evidence.get("gcs_has_type") or "",
                    )
                    if informative_core_tuple(*tuple_value):
                        item["core_tuples"].add(tuple_value)
                    else:
                        item["core_records_unknown_only"] += 1
                    if pmid:
                        item["pmids_core"].add(pmid)
                else:
                    predicate = relation.get("base_predicate") or relation.get("predicate")
                    item["extra_predicates"].add(relation.get("predicate") or predicate)
                    if as_bool(relation.get("negation")):
                        item["negated"] += 1
                    elif eligible:
                        item["semmed_allowed"] += 1
                        item["predicates"].add(predicate)
                        item["orientations"].add(relation["subject_entity_id"])
                        if pmid:
                            item["pmids_semmed"].add(pmid)
    genes_joint = defaultdict(set)
    genes_core = defaultdict(set)
    genes_core_only = defaultdict(set)
    genes_sem = defaultdict(set)
    neighbors = defaultdict(set)
    for key, item in details.items():
        core = item["core_records"] > 0
        semmed = item["semmed_allowed"] > 0
        if key[0] == "gene_cancer" and (core or semmed):
            _, gene, cancer = key
            genes_joint[cancer].add(gene)
            if core:
                genes_core[cancer].add(gene)
            if semmed:
                genes_sem[cancer].add(gene)
            if core and not semmed:
                genes_core_only[cancer].add(gene)
        elif key[0] == "gene_gene" and semmed and key[1] != key[2]:
            neighbors[key[1]].add(key[2])
            neighbors[key[2]].add(key[1])
    query_ids = sorted(seen_queries)
    cancer_ids = sorted(set(query_ids) | set(genes_joint))
    candidate_list = sorted(candidate_ids)
    one_rows = {}
    for query in query_ids:
        reached = set()
        n_paths = 0
        by_source = Counter()
        multi_predicate_paths = 0
        for gene in genes_joint.get(query, ()):
            partners = neighbors.get(gene, set()) & candidate_ids
            partners.discard(gene)
            if not partners:
                continue
            n_paths += len(partners)
            reached |= partners
            item = details[("gene_cancer", gene, query)]
            if item["core_records"] and item["semmed_allowed"]:
                source = "both_sources"
            elif item["core_records"]:
                source = "core_only"
            else:
                source = "semmed_only"
            by_source[source] += len(partners)
            if len(item["predicates"]) > 1:
                multi_predicate_paths += len(partners)
        one_rows[query] = {
            "n_paths": n_paths,
            "reached": reached,
            "by_source": by_source,
            "multi_predicate_paths": multi_predicate_paths,
        }
    two_joint = exact_two_intermediate(genes_joint, candidate_ids, query_ids, cancer_ids)
    two_core = exact_two_intermediate(genes_core, candidate_ids, query_ids, cancer_ids)
    two_core_only = exact_two_intermediate(genes_core_only, candidate_ids, query_ids, cancer_ids)
    direct_joint = direct_candidate_counts(genes_joint, candidate_ids, query_ids)
    direct_core = direct_candidate_counts(genes_core, candidate_ids, query_ids)
    direct_sem = direct_candidate_counts(genes_sem, candidate_ids, query_ids)
    queries_of_gene = defaultdict(set)
    cancers_of_gene = defaultdict(set)
    for cancer, genes in genes_joint.items():
        for gene in genes:
            cancers_of_gene[gene].add(cancer)
            if cancer in seen_queries:
                queries_of_gene[gene].add(cancer)
    cand_neighbors = {}
    for gene, partners in neighbors.items():
        cand_neighbors[gene] = partners & candidate_ids - {gene}
    used_gene_gene = set()
    used_gene_cancer = set()
    for gene, partners in cand_neighbors.items():
        if queries_of_gene.get(gene):
            for partner in partners:
                used_gene_gene.add(tuple(sorted((gene, partner))))
            for query in queries_of_gene.get(gene, ()):
                used_gene_cancer.add((gene, query))
    for cancer, genes in genes_joint.items():
        reachers = [gene for gene in genes if queries_of_gene.get(gene, set()) - {cancer}]
        cands = [gene for gene in genes if gene in candidate_ids]
        reacher_set = set(reachers)
        if reacher_set and cands:
            for cand in cands:
                if reacher_set - {cand}:
                    used_gene_cancer.add((cand, cancer))
            for gene in reachers:
                if any(cand != gene for cand in cands):
                    used_gene_cancer.add((gene, cancer))
                    for query in queries_of_gene.get(gene, ()):
                        if query != cancer:
                            used_gene_cancer.add((gene, query))
    attribute_counts = Counter()
    for cancer, genes in genes_joint.items():
        weights = {}
        for gene in genes:
            n_queries = len(queries_of_gene.get(gene, set()) - {cancer})
            if n_queries:
                weights[gene] = n_queries
        if not weights:
            continue
        cand_group = Counter(signature(details.get(("gene_cancer", gene, cancer), {}).get("core_tuples", set())) for gene in genes if gene in candidate_ids)
        weight_group = Counter()
        for gene, weight in weights.items():
            weight_group[signature(details.get(("gene_cancer", gene, cancer), {}).get("core_tuples", set()))] += weight
        for left, n_left in cand_group.items():
            for right, n_right in weight_group.items():
                left_set = {left} if left else set()
                right_set = {right} if right else set()
                attribute_counts[classify_attribute_sets(left_set, right_set)] += n_left * n_right
        for gene, weight in weights.items():
            if gene in candidate_ids:
                sig = signature(details.get(("gene_cancer", gene, cancer), {}).get("core_tuples", set()))
                sig_set = {sig} if sig else set()
                attribute_counts[classify_attribute_sets(sig_set, sig_set)] -= weight
    n_candidates = len(candidate_ids)
    lines = ["level\tquery_cui\tmetric\tnumerator\tdenominator\tvalue"]

    def add_metric(level, query, metric, numerator, denominator):
        value = "" if not denominator else f"{numerator / denominator:.6f}"
        lines.append(f"{level}\t{query}\t{metric}\t{numerator}\t{denominator}\t{value}")

    totals = Counter()
    reached_fraction_one = []
    reached_fraction_two = []
    by_query_path = ["query_cui\tn_candidates\tn_direct_either\tn_direct_core\tn_direct_semmed\tn_reached_one_intermediate\tn_paths_one_intermediate\tn_reached_two_intermediate\tn_paths_two_intermediate\tn_paths_two_intermediate_core_view\tn_paths_two_intermediate_newly_available\tn_candidates_without_one_intermediate_path\tn_candidates_without_two_intermediate_path"]
    for query in query_ids:
        one = one_rows[query]
        two_paths = two_joint[query]["n_paths"]
        two_reached = two_joint[query]["n_reached"]
        core_paths = two_core[query]["n_paths"]
        add_metric("query", query, "candidates_reached_one_intermediate_path", len(one["reached"]), n_candidates)
        add_metric("query", query, "candidates_reached_two_intermediate_path", two_reached, n_candidates)
        add_metric("query", query, "direct_eligible_pairs", direct_joint[query], n_candidates)
        reached_fraction_one.append(len(one["reached"]) / n_candidates if n_candidates else 0)
        reached_fraction_two.append(two_reached / n_candidates if n_candidates else 0)
        totals["paths_one"] += one["n_paths"]
        totals["reached_one"] += len(one["reached"])
        totals["paths_two"] += two_paths
        totals["reached_two"] += two_reached
        totals["paths_two_core"] += core_paths
        totals["paths_two_core_exclusive"] += two_core_only[query]["n_paths"]
        totals["direct_joint"] += direct_joint[query]
        totals["direct_core"] += direct_core[query]
        totals["direct_sem"] += direct_sem[query]
        totals["one_without_path"] += n_candidates - len(one["reached"])
        totals["two_without_path"] += n_candidates - two_reached
        for source, count in one["by_source"].items():
            totals[f"one_{source}"] += count
        totals["one_multi_predicate_paths"] += one["multi_predicate_paths"]
        indirect_only = len(one["reached"] - (genes_joint.get(query, set()) & candidate_ids))
        totals["one_indirect_only"] += indirect_only
        by_query_path.append(
            "\t".join(
                map(
                    str,
                    [
                        query,
                        n_candidates,
                        direct_joint[query],
                        direct_core[query],
                        direct_sem[query],
                        len(one["reached"]),
                        one["n_paths"],
                        two_reached,
                        two_paths,
                        core_paths,
                        two_paths - core_paths,
                        n_candidates - len(one["reached"]),
                        n_candidates - two_reached,
                    ],
                )
            )
        )
    n_queries = len(query_ids)
    add_metric("aggregate", "ALL", "cancer_queries", n_queries, n_queries)
    add_metric("aggregate", "ALL", "candidate_genes", n_candidates, n_candidates)
    add_metric("aggregate", "ALL", "one_intermediate_paths", totals["paths_one"], totals["paths_one"])
    add_metric("aggregate", "ALL", "one_intermediate_paths_core_only_view", 0, totals["paths_one"])
    add_metric("aggregate", "ALL", "one_intermediate_paths_newly_available", totals["paths_one"], totals["paths_one"])
    add_metric("aggregate", "ALL", "one_intermediate_paths_gene_cancer_leg_core_only", totals["one_core_only"], totals["paths_one"])
    add_metric("aggregate", "ALL", "one_intermediate_paths_gene_cancer_leg_semmed_only", totals["one_semmed_only"], totals["paths_one"])
    add_metric("aggregate", "ALL", "one_intermediate_paths_gene_cancer_leg_both_sources", totals["one_both_sources"], totals["paths_one"])
    add_metric("aggregate", "ALL", "one_intermediate_paths_multiple_predicates_on_gene_cancer_leg", totals["one_multi_predicate_paths"], totals["paths_one"])
    add_metric("aggregate", "ALL", "two_intermediate_paths_joint", totals["paths_two"], totals["paths_two"])
    add_metric("aggregate", "ALL", "two_intermediate_paths_core_only_view", totals["paths_two_core"], totals["paths_two"])
    add_metric("aggregate", "ALL", "two_intermediate_paths_newly_available", totals["paths_two"] - totals["paths_two_core"], totals["paths_two"])
    add_metric("aggregate", "ALL", "two_intermediate_paths_records_changed", totals["paths_two_core"] - totals["paths_two_core_exclusive"], totals["paths_two_core"])
    add_metric("aggregate", "ALL", "direct_candidate_query_pairs", totals["direct_joint"], n_candidates * n_queries if n_queries else 0)
    add_metric("aggregate", "ALL", "direct_core_candidate_query_pairs", totals["direct_core"], n_candidates * n_queries if n_queries else 0)
    add_metric("aggregate", "ALL", "direct_semmed_candidate_query_pairs", totals["direct_sem"], n_candidates * n_queries if n_queries else 0)
    add_metric("aggregate", "ALL", "mean_fraction_candidates_reached_one_intermediate", sum(reached_fraction_one), n_queries)
    add_metric("aggregate", "ALL", "mean_fraction_candidates_reached_two_intermediate", sum(reached_fraction_two), n_queries)
    (analysis / "path_coverage.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (analysis / "path_coverage_by_query.tsv").write_text("\n".join(by_query_path) + "\n", encoding="utf-8")

    def edge_audit(kind: str, keys: set) -> dict:
        counts = Counter()
        pmid_sizes = []
        for key in keys:
            if kind == "gene_gene":
                item = details.get(("gene_gene", key[0], key[1]))
            else:
                item = details.get(("gene_cancer", key[0], key[1]))
            if item is None:
                continue
            counts["connections"] += 1
            if len(item["predicates"]) > 1:
                counts["multiple_allowed_predicates"] += 1
            if len(item["orientations"]) > 1:
                counts["multiple_orientations"] += 1
            if item["negated"]:
                counts["negated_record_also_present"] += 1
            extra = item["extra_predicates"] - {f"NEG_{predicate}" for predicate in item["predicates"]} - set(item["predicates"])
            if extra:
                counts["additional_recorded_predicates"] += 1
            if kind == "gene_cancer":
                if item["core_records"]:
                    counts["core_record_present"] += 1
                if len(item["core_tuples"]) > 1:
                    counts["multiple_core_attribute_tuples"] += 1
                if item["core_records"] and not item["core_tuples"]:
                    counts["core_records_unknown_only"] += 1
                if item["core_records"] and item["semmed_allowed"]:
                    counts["both_sources"] += 1
                elif item["core_records"]:
                    counts["core_only"] += 1
                elif item["semmed_allowed"]:
                    counts["semmed_only"] += 1
            pmid_sizes.append(len(item["pmids_core"] | item["pmids_semmed"]))
        pmid_sizes.sort()
        median = pmid_sizes[len(pmid_sizes) // 2] if pmid_sizes else 0
        return {"counts": counts, "median_pmids": median, "n": len(pmid_sizes)}

    gene_gene_audit = edge_audit("gene_gene", used_gene_gene)
    gene_cancer_audit = edge_audit("gene_cancer", used_gene_cancer)
    rep_lines = ["edge_family\tmeasure\tnumerator\tdenominator"]
    for family, audit in (("gene_gene", gene_gene_audit), ("gene_cancer", gene_cancer_audit)):
        denominator = audit["counts"]["connections"]
        for measure, numerator in sorted(audit["counts"].items()):
            rep_lines.append(f"{family}\t{measure}\t{numerator}\t{denominator if measure != 'connections' else denominator}")
        rep_lines.append(f"{family}\tmedian_distinct_pmids\t{audit['median_pmids']}\t{audit['n']}")
    for label, numerator in sorted(attribute_counts.items()):
        rep_lines.append(f"two_intermediate_path\t{label}\t{numerator}\t{totals['paths_two']}")
    (analysis / "representation_audit.tsv").write_text("\n".join(rep_lines) + "\n", encoding="utf-8")

    core_pmids = set()
    sem_pmids = set()
    core_pairs = set()
    sem_pairs = set()
    for key, item in details.items():
        if key[0] != "gene_cancer":
            continue
        pair = (key[1], key[2])
        if item["core_records"]:
            core_pairs.add(pair)
            core_pmids |= item["pmids_core"]
        if item["semmed_allowed"]:
            sem_pairs.add(pair)
            sem_pmids |= item["pmids_semmed"]
    overlap_lines = ["level\tunit\tn_core\tn_semmed\tn_intersection\tn_core_only\tn_semmed_only\tn_union"]

    def overlap(unit, core_set, sem_set):
        overlap_lines.append(
            "\t".join(
                map(
                    str,
                    [
                        "eligible_graph",
                        unit,
                        len(core_set),
                        len(sem_set),
                        len(core_set & sem_set),
                        len(core_set - sem_set),
                        len(sem_set - core_set),
                        len(core_set | sem_set),
                    ],
                )
            )
        )

    overlap("source_article_pmid", core_pmids, sem_pmids)
    overlap("gene_cancer_pair", core_pairs, sem_pairs)
    (analysis / "source_overlap.tsv").write_text("\n".join(overlap_lines) + "\n", encoding="utf-8")

    degrees = []
    for cancer, genes in genes_joint.items():
        degrees.append(("cancer", cancer, len(genes)))
    for gene, cancers in cancers_of_gene.items():
        degrees.append(("gene", gene, len(cancers)))
    degrees.sort(key=lambda item: item[2], reverse=True)
    degree_lines = ["entity_type\tcanonical_id\tdegree\tincluded_in_exact_counts"]
    for row in degrees[:25]:
        degree_lines.append(f"{row[0]}\t{row[1]}\t{row[2]}\tyes")
    (analysis / "high_degree_nodes.tsv").write_text("\n".join(degree_lines) + "\n", encoding="utf-8")

    examples = find_examples(details, cand_neighbors, genes_joint, candidate_ids, query_ids)
    example_lines = ["example_class\ttemplate\tcandidate_gene\tintermediate_gene\tintermediate_cancer\tquery_cancer\tleg\tresource\tnative_hint\tpmid\tpredicate\tnegation\tcore_attributes"]
    example_lines.extend(examples)
    (analysis / "path_examples.tsv").write_text("\n".join(example_lines) + "\n", encoding="utf-8")
    import pyarrow as pa

    pq.write_table(pa.Table.from_pylist(candidates), analysis / "candidate_genes.parquet")
    cancer_rows = [{"cancer_cui": cui, "entity_id": f"cancer:{cui}", "display_label": cui} for cui in query_ids]
    name_path = args.scratch / "cui_preferred_names.parquet"
    if name_path.exists():
        labels = {row["cui"]: row["preferred_name"] for row in pq.read_table(name_path).to_pylist()}
        for row in cancer_rows:
            row["display_label"] = labels.get(row["cancer_cui"], row["cancer_cui"])
    pq.write_table(pa.Table.from_pylist(cancer_rows), analysis / "cancer_queries.parquet")
    write_json(
        analysis / "path_summary.json",
        {
            "n_candidates": n_candidates,
            "n_queries": n_queries,
            "totals": {key: int(value) for key, value in totals.items()},
            "attribute_counts": {key: int(value) for key, value in attribute_counts.items()},
            "gene_gene_used": int(gene_gene_audit["counts"]["connections"]),
            "gene_cancer_used": int(gene_cancer_audit["counts"]["connections"]),
            "high_degree_top": [{"entity_type": row[0], "canonical_id": row[1], "degree": row[2]} for row in degrees[:10]],
            "counting": "exact",
            "one_intermediate_edges": 2,
            "two_intermediate_edges": 3,
        },
    )
    print(f"paths one={totals['paths_one']} two={totals['paths_two']} queries={n_queries} candidates={n_candidates}", flush=True)


def find_examples(details, cand_neighbors, genes_joint, candidates, queries) -> list[str]:
    rows = []
    found = None
    for query in queries:
        for gene in sorted(genes_joint.get(query, ())):
            partners = sorted(cand_neighbors.get(gene, set()))
            if not partners:
                continue
            found = (partners[0], gene, query)
            break
        if found:
            break
    if found:
        candidate, gene, query = found
        rows.extend(example_rows("smallest_one_intermediate_path", "one_intermediate_path", candidate, gene, "", query, details))
    multi = None
    for key, item in sorted(details.items()):
        if key[0] == "gene_cancer" and len(item["predicates"]) > 1 and item["semmed_allowed"]:
            multi = key
            break
    if multi:
        _, gene, cancer = multi
        rows.append(
            "\t".join(
                [
                    "multiple_predicates_on_one_connection",
                    "connectivity_edge",
                    "",
                    gene,
                    "",
                    cancer,
                    "gene_cancer",
                    "semmeddb",
                    "",
                    "|".join(sorted(item for item in list(details[multi]["pmids_semmed"])[:5])),
                    "|".join(sorted(details[multi]["predicates"])),
                    "false",
                    "",
                ]
            )
        )
    negated = None
    for key, item in sorted(details.items()):
        if item["negated"] and item["semmed_allowed"]:
            negated = key
            break
    if negated:
        rows.append(
            "\t".join(
                [
                    "negated_record_retained_beside_affirmative_edge",
                    "connectivity_edge",
                    "",
                    negated[1],
                    "",
                    negated[2] if len(negated) > 2 else "",
                    negated[0],
                    "semmeddb",
                    "",
                    "",
                    "|".join(sorted(details[negated]["extra_predicates"])),
                    "true",
                    "",
                ]
            )
        )
    return rows


def example_rows(example_class, template, candidate, gene, cancer, query, details) -> list[str]:
    gene_gene = details.get(("gene_gene", *sorted((candidate, gene))))
    gene_cancer = details.get(("gene_cancer", gene, query))
    rows = []
    if gene_gene:
        rows.append(
            "\t".join(
                [
                    example_class,
                    template,
                    candidate,
                    gene,
                    cancer,
                    query,
                    "candidate_gene_to_intermediate_gene",
                    "semmeddb",
                    "",
                    "|".join(sorted(list(gene_gene["pmids_semmed"])[:5])),
                    "|".join(sorted(gene_gene["predicates"])),
                    "false",
                    "",
                ]
            )
        )
    if gene_cancer:
        rows.append(
            "\t".join(
                [
                    example_class,
                    template,
                    candidate,
                    gene,
                    cancer,
                    query,
                    "intermediate_gene_to_query_cancer",
                    "core" if gene_cancer["core_records"] else "semmeddb",
                    "",
                    "|".join(sorted(list((gene_cancer["pmids_core"] or gene_cancer["pmids_semmed"]))[:5])),
                    "|".join(sorted(gene_cancer["predicates"])) or "gene_cancer_association",
                    "false",
                    "|".join(signature_one(item) for item in list(gene_cancer["core_tuples"])[:3]),
                ]
            )
        )
    return rows


def signature_one(value: tuple) -> str:
    return "|".join(value)


if __name__ == "__main__":
    main()

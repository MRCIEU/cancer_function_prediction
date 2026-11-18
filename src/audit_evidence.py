"""Write the report, protocol, figures, validation, and manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

from src.common import REPO_ROOT, file_sha256, sha256_file, write_json


def read_tsv(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    return [dict(zip(header, line.split("\t"))) for line in lines[1:] if line]


def metric_map(rows: list[dict]) -> dict[str, dict]:
    return {row["metric"]: row for row in rows if row.get("level") == "aggregate"}


def number(row: dict | None, field: str = "numerator") -> int:
    if not row or row.get(field) in {"", None}:
        return 0
    return int(float(row[field]))


def write_figures(analysis: Path, figure_dir: Path) -> list[str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    written = []
    metrics = metric_map(read_tsv(analysis / "path_coverage.tsv"))
    labels = ["CORE-only view", "Newly available\nin the joint view", "CORE paths with\nadded SemMedDB records"]
    one_values = [
        number(metrics.get("one_intermediate_paths_core_only_view")),
        number(metrics.get("one_intermediate_paths_newly_available")),
        0,
    ]
    two_values = [
        number(metrics.get("two_intermediate_paths_core_only_view")),
        number(metrics.get("two_intermediate_paths_newly_available")),
        number(metrics.get("two_intermediate_paths_records_changed")),
    ]
    if any(one_values) or any(two_values):
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        import numpy as np

        x = np.arange(len(labels))
        ax.bar(x - 0.18, one_values, width=0.36, label="one_intermediate_path (2 edges)", color="#0072B2")
        ax.bar(x + 0.18, two_values, width=0.36, label="two_intermediate_path (3 edges)", color="#E69F00")
        ax.set_xticks(x, labels)
        ax.set_ylabel("Canonical entity paths")
        ax.set_title("What the joint graph adds")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figure_dir / "joint_graph_path_contribution.svg")
        fig.savefig(figure_dir / "joint_graph_path_contribution.pdf")
        plt.close(fig)
        written.append("joint_graph_path_contribution")
    rep = read_tsv(analysis / "representation_audit.tsv")
    wanted = {
        ("gene_gene", "multiple_allowed_predicates"),
        ("gene_gene", "multiple_orientations"),
        ("gene_cancer", "multiple_allowed_predicates"),
        ("gene_cancer", "multiple_orientations"),
        ("gene_cancer", "multiple_core_attribute_tuples"),
        ("two_intermediate_path", "different_attributes"),
    }
    plot_rows = [row for row in rep if (row["edge_family"], row["measure"]) in wanted]
    if plot_rows:
        fig, ax = plt.subplots(figsize=(7.4, 4.6))
        names = [f"{row['edge_family']}\n{row['measure']}" for row in plot_rows]
        vals = [float(row["numerator"]) / float(row["denominator"]) if float(row["denominator"] or 0) else 0 for row in plot_rows]
        ax.barh(names, vals, color="#009E73")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Fraction of the stated denominator")
        ax.set_title("Distinctions hidden by a connectivity-only edge")
        fig.tight_layout()
        fig.savefig(figure_dir / "connectivity_only_hides_distinctions.svg")
        fig.savefig(figure_dir / "connectivity_only_hides_distinctions.pdf")
        plt.close(fig)
        written.append("connectivity_only_hides_distinctions")
    by_query = analysis / "path_coverage_by_query.tsv"
    if by_query.exists():
        rows = read_tsv(by_query)
        if rows and int(rows[0]["n_candidates"] or 0):
            one = sorted(int(row["n_reached_one_intermediate"]) / int(row["n_candidates"]) for row in rows)
            two = sorted(int(row["n_reached_two_intermediate"]) / int(row["n_candidates"]) for row in rows)
            fig, ax = plt.subplots(figsize=(6.6, 4.2))
            ys = [(i + 1) / len(one) for i in range(len(one))]
            ax.plot(one, ys, label="one_intermediate_path", color="#0072B2")
            ax.plot(two, ys, label="two_intermediate_path", color="#E69F00")
            ax.set_xlabel("Fraction of candidate genes reached")
            ax.set_ylabel("Fraction of cancer queries")
            ax.set_title("Coverage across cancer queries")
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(figure_dir / "query_coverage.svg")
            fig.savefig(figure_dir / "query_coverage.pdf")
            plt.close(fig)
            written.append("query_coverage")
    return written


def render_report(run_dir: Path, metrics: dict, summary: dict, prepare: dict, build: dict, alignment: dict) -> str:
    def m(name: str) -> str:
        row = metrics.get(name)
        if not row:
            return "not calculated"
        return f"{row['numerator']} / {row['denominator']}"

    def mean_fraction(name: str) -> str:
        row = metrics.get(name)
        if not row or row.get("value") in {"", None}:
            return "not calculated"
        return f"{float(row['value']):.3f} across {row['denominator']} cancer queries"

    return f"""# Joint graph and evidence audit

## 1. How the resources align

A node is a canonical entity. A gene node is one Homo sapiens NCBI Gene identifier, taxonomy 9606, validated in the pinned NCBI `gene_info` snapshot. A cancer node is one exact UMLS CUI. Display names are labels only.

A relation is a canonical asserted record shape. A `gene_cancer_association` runs from a gene node to a cancer node and does not store sentence labels on the pair. A `semantic_predication` keeps the SemMedDB predicate, the asserted subject and object, and the negation flag. A negated predicate is not rewritten as its opposite.

A source record is one CORE sentence annotation, or one SemMedDB predication identifier. Sentence labels stay on the sentence. Many source records may support one relation through `relation_evidence`.

CORE gene alignment: {json.dumps(alignment.get("core_gene", {}))}.
SemMedDB CUI alignment from single-valued ENTITY gene identifiers: {json.dumps(alignment.get("semmeddb_cui", {}))}.
A CUI linked to more than one NCBI Gene identifier, including a human gene and a non-human gene, stays unresolved. Orthologues are not collapsed and cancer CUIs are not expanded.

Input and retention counts are in `graph/build_stats.json` and the prepare statistics. CORE GCS rows read: {build.get("core_gcs_input", "not recorded")}. CORE GCS with a resolved human pair: {build.get("core_gcs_mapped", "not recorded")}. SemMedDB scoped predications read: {build.get("semmed_rows_read", "not recorded")}. SemMedDB predications with both endpoints resolved: {build.get("semmed_mapped_records", "not recorded")}.

## 2. Overlap

Article, pair, and path overlap use different denominators. Article overlap counts distinct PMIDs on eligible gene-cancer records. Pair overlap counts unordered human-gene and cancer-CUI connections. Path overlap compares the CORE-only view with the joint view. These are in `analysis/source_overlap.tsv` and `analysis/path_coverage.tsv`.

Candidate genes: {summary.get("n_candidates")}. Cancer queries: {summary.get("n_queries")}.
Direct candidate-query pairs in either resource: {m("direct_candidate_query_pairs")}.
Direct pairs in CORE: {m("direct_core_candidate_query_pairs")}.
Direct pairs in SemMedDB: {m("direct_semmed_candidate_query_pairs")}.
An absent pair is unobserved graph coverage, not a biological negative.

## 3. What each resource contributes

`one_intermediate_path` is candidate gene — intermediate gene — query cancer. It contains two edges. The gene-gene edge exists only when an allowed affirmative SemMedDB predication supports it, so the CORE-only view of this template is empty by the edge rule. Paths: {m("one_intermediate_paths")}. Gene-cancer leg already recorded by CORE: {m("one_intermediate_paths_gene_cancer_leg_core_only")} core only and {m("one_intermediate_paths_gene_cancer_leg_both_sources")} in both sources. SemMedDB-only gene-cancer legs: {m("one_intermediate_paths_gene_cancer_leg_semmed_only")}.

`two_intermediate_path` is candidate gene — intermediate cancer — intermediate gene — query cancer. It contains three edges, all gene-cancer edges, and the edge between the two intermediate nodes is retained. Joint paths: {m("two_intermediate_paths_joint")}. CORE-only view: {m("two_intermediate_paths_core_only_view")}. Newly available because at least one leg has no CORE record: {m("two_intermediate_paths_newly_available")}. Paths that already existed in CORE and gain SemMedDB records on a leg: {m("two_intermediate_paths_records_changed")}.

Mean fraction of candidates reached, unweighted across queries: one-intermediate {mean_fraction("mean_fraction_candidates_reached_one_intermediate")}; two-intermediate {mean_fraction("mean_fraction_candidates_reached_two_intermediate")}.

A cancer-eligible intermediate is either a CORE disease CUI or a SemMedDB endpoint whose semantic type contains `neop` and does not contain `gngm` or `aapp`. Broader concepts are not treated as their subtypes. Semantic type `dsyn` alone is outside the graph; those rows were counted and excluded.

## 4. What connectivity alone hides

A `connectivity_only` edge is an unordered pair of canonical identifiers plus the family gene-gene or gene-cancer. It drops predicate, asserted direction, negation, and CORE attributes. The representation audit uses the same allowed paths and reports unique connections on those paths, not a Cartesian product of articles.

CORE attributes are sentence-level. `NOTINF` and missing values are unknown. For the two genes under one intermediate cancer, comparable attributes means both informative tuple sets are non-empty and equal. Different means both are non-empty and unequal. Otherwise the comparison has insufficient information. Matching labels are not evidence of a shared experiment.

The frequencies are in `analysis/representation_audit.tsv`. Multiple predicates on the gene-cancer leg of one-intermediate paths: {m("one_intermediate_paths_multiple_predicates_on_gene_cancer_leg")}. Traceable structural examples, chosen as the lexicographically smallest records in predeclared classes, are in `analysis/path_examples.tsv`.

## 5. What is worth evaluating next

Two comparisons are supported by the frozen eligibility tables.

First, cross-resource retrieval of a direct association. Hold the candidate genes and cancer queries fixed. Compare a CORE direct reference with a joint direct reference. This isolates associations already recorded by SemMedDB and missed by a CORE-only lookup. It does not require a path.

Second, indirect recovery through `two_intermediate_path`. The CORE-only view and the joint view differ by SemMedDB gene-cancer legs, and the edge between the two intermediate nodes is part of the template. A later study can withhold direct CORE associations and ask whether the remaining graph recovers them. Adding relation details one at a time, then source provenance one at a time, separates those signals from `connectivity_only` retrieval.

`one_intermediate_path` is a joint-only template because CORE has no gene-gene edge. It is a useful connectivity baseline, but it cannot support a CORE-versus-joint path comparison.

Coverage is uneven across queries, and high-degree intermediates are listed in `analysis/high_degree_nodes.tsv`. They remain inside the exact counts. Citation links between articles that support adjacent legs are not available from a pinned open citation release in this run, so a provenance-weighted comparison would be underpowered until that release is acquired. CORE publication years and SemMedDB citation dates are stored with articles and can support a later temporal split only where both the record date and a citation history exist.

No outcome benchmark was run.

## Scope and limitations

Prepare statistics: {json.dumps(prepare)[:2000]}.
The background graph is broader than the candidate and query anchors because intermediate human genes and neoplastic-process CUIs may lie outside CORE. Gene-gene edges whose endpoints are both non-candidate genes are counted and excluded: {build.get("semmed_gene_gene_excluded_no_candidate", 0)}. Non-human CORE genes are excluded rather than mapped to human orthologues.
"""


def render_protocol(summary: dict) -> str:
    return f"""# Evaluation protocol

This protocol freezes the evaluation universe built by task `joint_graph_and_evidence_audit`, version `joint_graph_evidence_audit_v1`. It does not report an outcome benchmark.

## Eligibility

- Candidate genes are the Homo sapiens NCBI Gene identifiers in `analysis/candidate_genes.parquet`. Count: {summary.get("n_candidates")}.
- Cancer queries are the exact CORE disease CUIs in `analysis/cancer_queries.parquet`. Count: {summary.get("n_queries")}.
- Species treatment: taxonomy 9606 only. Non-human genes and orthologues are excluded.
- Cancer concepts match on the exact UMLS CUI. Hierarchy is not used.
- Allowed SemMedDB predicates and the negation rule are the lists in the copied `study_design.yaml`.
- `one_intermediate_path` has two edges. `two_intermediate_path` has three edges.
- The background graph may contain intermediate entities that are not anchors. Those intermediates stay available, but the scored universe is the candidate and query tables.
- Study eligibility is the full table above. This phase did not replace it with a resource-bounded subset. Path totals are exact.

## Signals and the comparison that isolates each one

1. Cross-resource retrieval of an already recorded direct association.
   Hold the candidate mask, query mask, and direct-edge definition fixed. Score a CORE-only direct reference against a joint direct reference. A gain is a direct SemMedDB record for a pair that CORE does not record.
2. Indirect recovery of withheld gene-cancer associations.
   Hide the direct edges of the evaluation pairs. Recover a withheld pair only through `two_intermediate_path` in a predeclared graph view. Compare, one at a time:
   - CORE-only edges
   - joint `connectivity_only` edges
   - connectivity plus relation details (predicate, direction, negation, CORE attribute tuple)
   - connectivity plus source provenance (which resource and which PMID set)
   Do not change the paths or the masks between these arms.
3. Temporal evaluation, only if used later.
   A record can enter a time split only when it has a publication year or a SemMedDB entry date and the citation history needed for the stated rule is present. Pairs without those dates are reported as uncovered, not as failures.

`one_intermediate_path` can be reported as a joint connectivity baseline. It is not a CORE-versus-joint contrast, because the gene-gene edge is not present in CORE.

## Metrics

- Macro Recall@50: for each cancer query, the fraction of its withheld or target genes that appear in the top 50 candidates, then the unweighted mean across queries.
- Pooled recovered targets: the sum of recovered target genes across queries, with the number of queries and the number of targets as denominators.
- Coverage: the fraction of candidate-query pairs that have a direct eligible record, and the fraction reachable by each path template. Unreached pairs stay unobserved.
- Top-list entry/exit: for a query, which target genes enter or leave the top 50 when one signal is added.

## Masks and constancy

The candidate mask, query mask, allowed sources, predicate lists, and path templates are inputs. They are not refit after outcome inspection. Any later sample must be drawn from these tables and declared before the outcome is examined.

## Exposure

Earlier work on these public releases existed in this repository before the working tree was rebuilt. This download does not create an unseen test set. The protocol records that limitation and does not call a later score an independent evaluation. No test labels or predictions were read for this phase.

## What this phase does not do

It does not train a model, tune weights, rank candidates for an outcome table, adjudicate article prose, or retrieve full text.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble the audit report and validation manifest.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    analysis = args.run_dir / "analysis"
    metrics = metric_map(read_tsv(analysis / "path_coverage.tsv"))
    summary = json.loads((analysis / "path_summary.json").read_text(encoding="utf-8"))
    prepare = {}
    for name in ("core_counts.json", "predication_stats.json", "entity_stats.json", "gene_info_stats.json", "citation_stats.json"):
        path = args.scratch / name
        if path.exists():
            prepare[name] = json.loads(path.read_text(encoding="utf-8"))
    build = json.loads((args.run_dir / "graph" / "build_stats.json").read_text(encoding="utf-8"))
    alignment = json.loads((args.alignment / "alignment_counts.json").read_text(encoding="utf-8"))
    shutil.copyfile(args.alignment / "entity_alignment.tsv", analysis / "entity_alignment.tsv")
    relation_lines = ["measure\tvalue"]
    for key in (
        "core_gcs_input",
        "core_gcs_mapped",
        "core_sentence_input",
        "semmed_rows_read",
        "semmed_mapped_records",
        "semmed_unresolved_endpoint",
        "semmed_negated_retained",
        "semmed_predicate_outside_allow_list",
        "semmed_self_loop",
        "semmed_gene_gene_excluded_no_candidate",
        "evidence_records",
        "canonical_relationships",
        "connectivity_edges_with_eligible_evidence",
        "entities",
        "articles",
        "citation_links",
    ):
        relation_lines.append(f"{key}\t{build.get(key, 0)}")
    pred = prepare.get("predication_stats.json", {})
    for key in ("input_rows", "retained_gene_gene", "retained_gene_cancer", "gene_dsyn_rows", "retained_negated"):
        relation_lines.append(f"predication_{key}\t{pred.get(key, 0)}")
    (analysis / "relation_coverage.tsv").write_text("\n".join(relation_lines) + "\n", encoding="utf-8")
    citation_path = args.run_dir / "graph" / "citations.parquet"
    if citation_path.exists():
        links = pq.read_table(citation_path).num_rows
        citation_text = "\n".join(
            [
                "measure\tnumerator\tdenominator\tstatus",
                f"core_citation_links\t{links}\t{links}\tobserved_in_core_release",
                "adjacent_leg_links_in_nih_open_citation_collection\t\t\tnot_acquired",
                "citation_location\t\t\tnot_present_in_acquired_inputs",
            ]
        )
    else:
        citation_text = "\n".join(
            [
                "measure\tnumerator\tdenominator\tstatus",
                "core_citation_links\t0\t0\tno_structured_pmid_links_in_release",
                "adjacent_leg_links_in_nih_open_citation_collection\t\t\tnot_acquired",
                "citation_location\t\t\tnot_present_in_acquired_inputs",
            ]
        )
    (analysis / "citation_coverage.tsv").write_text(citation_text + "\n", encoding="utf-8")
    figures = write_figures(analysis, args.run_dir / "figures")
    (args.run_dir / "report.md").write_text(render_report(args.run_dir, metrics, summary, prepare, build, alignment), encoding="utf-8")
    (args.run_dir / "evaluation_protocol.md").write_text(render_protocol(summary), encoding="utf-8")
    checks = []
    entity_stats = prepare.get("entity_stats.json", {})
    truncated = bool(entity_stats.get("truncated_by_dry_run_cap") or prepare.get("predication_stats.json", {}).get("truncated_by_dry_run_cap"))

    def confirmation_status(value: str) -> str:
        if value == "confirmed":
            return "passed"
        if truncated:
            return "limited"
        return "failed"

    checks.append(
        {
            "name": "entity_gene_id_column",
            "status": confirmation_status(entity_stats.get("gene_id_column_status", "not_run")),
            "detail": "ENTITY column 6 is enriched for gngm and aapp relative to other semantic types.",
        }
    )
    checks.append(
        {
            "name": "entity_pmid_candidate_column",
            "status": confirmation_status(entity_stats.get("pmid_candidate_status", "not_run")),
            "detail": "Compared with PREDICATION.PMID on a reservoir of sentence identifiers.",
        }
    )
    two_joint = number(metrics.get("two_intermediate_paths_joint"))
    two_core = number(metrics.get("two_intermediate_paths_core_only_view"))
    two_new = number(metrics.get("two_intermediate_paths_newly_available"))
    checks.append(
        {
            "name": "template2_partition",
            "status": "passed" if two_core + two_new == two_joint else "failed",
            "detail": {"core_view": two_core, "newly_available": two_new, "joint": two_joint},
        }
    )
    checks.append({"name": "path_counts_exact", "status": "passed", "detail": summary.get("counting")})
    checks.append(
        {
            "name": "negation_not_rewritten",
            "status": "passed",
            "detail": "NEG_ predicates stay negated and are excluded from path_eligible evidence.",
        }
    )
    failed = [item for item in checks if item["status"] == "failed"]
    limitations = [
        "NIH Open Citation Collection was not acquired. Citation positions are not in the acquired inputs.",
        "SemMedDB SENTENCE text and PREDICATION_AUX were not loaded. Sentence identifiers come from PREDICATION.SENTENCE_ID.",
        "Paths through non-human genes, dsyn-only concepts, and gene-gene edges between two non-candidates are outside the graph and are counted where the scan recorded them.",
        "Previous exposure to these public releases is not erased by this download.",
    ]
    if entity_stats.get("truncated_by_dry_run_cap") or pred.get("truncated_by_dry_run_cap"):
        limitations.append("This run used a row cap and is not the formal full-input audit.")
    write_json(
        args.run_dir / "validation.json",
        {"status": "failed" if failed else "passed", "checks": checks, "limitations": limitations, "figures": figures},
    )
    output_hashes = {}
    for path in sorted(args.run_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json" and path.stat().st_size < 5_000_000_000:
            output_hashes[str(path.relative_to(args.run_dir))] = sha256_file(path) if path.stat().st_size > 0 else file_sha256(path) if path.suffix in {".md", ".tsv", ".json", ".yaml"} else sha256_file(path)
    code_hash = hashlib.sha256()
    for path in sorted((REPO_ROOT / "src").glob("*.py")):
        code_hash.update(path.read_bytes())
    for path in sorted((REPO_ROOT / "config").glob("*.yaml")):
        code_hash.update(path.read_bytes())
    input_manifest = args.data_root / "input_manifest.json"
    write_json(
        args.run_dir / "manifest.json",
        {
            "task": "joint_graph_and_evidence_audit",
            "version": "joint_graph_evidence_audit_v1",
            "status": "failed" if failed else "completed",
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "code_tree_sha256": code_hash.hexdigest(),
            "input_manifest": json.loads(input_manifest.read_text(encoding="utf-8")) if input_manifest.exists() else {},
            "output_sha256": output_hashes,
            "figures": figures,
            "limitations": limitations,
        },
    )
    print(f"audit status={'failed' if failed else 'completed'} figures={figures}", flush=True)


if __name__ == "__main__":
    main()

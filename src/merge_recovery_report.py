"""Combine the six development-model jobs into one report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from src.common import write_json
from src.run_association_recovery import evaluate_method

METHODS = ("path_counts", "random_walk", "ridge", "trees", "rgcn", "pathnet")
PRIMARY = {
    "path_counts": "degree_weighted_paths",
    "random_walk": "random_walk",
    "ridge": "ridge_combined_seed_20261116",
    "trees": "trees_combined_seed_20261116",
    "rgcn": "rgcn_features",
    "pathnet": "pathnet_features",
}


def _load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _macro(run_dir: Path, method: str, score_name: str, context: dict) -> float | None:
    score_path = run_dir / "analysis" / "models" / method / "dev_scores.npz"
    ranking_path = run_dir / "analysis" / "models" / method / f"{score_name}_rankings.parquet"
    if not score_path.exists() or not ranking_path.exists():
        return None
    scores = np.load(score_path)[score_name]
    table = pq.read_table(ranking_path, columns=["cancer_cui", "gene_id", "rank", "label"])
    cancers = context["cancers"]
    gene_ids = context["gene_ids"]
    gene_index = {gene: index for index, gene in enumerate(gene_ids)}
    eligible = np.zeros((len(cancers), len(gene_ids)), dtype=bool)
    positives = {cancer: set() for cancer in cancers}
    rankings = {cancer: [] for cancer in cancers}
    rows = table.to_pydict()
    ordered = sorted(zip(rows["cancer_cui"], rows["gene_id"], rows["rank"], rows["label"]), key=lambda item: (item[0], item[2]))
    for cancer, gene, _rank, label in ordered:
        rankings[cancer].append(gene)
        eligible[cancers.index(cancer), gene_index[gene]] = True
        if int(label) == 1:
            positives[cancer].add(gene)
    from_scores = evaluate_method(scores, gene_ids, eligible, positives, cancers)["macro_recall"]
    from_rank = evaluate_method(
        _grid_from_rankings(rankings, cancers, gene_ids),
        gene_ids,
        eligible,
        positives,
        cancers,
    )["macro_recall"]
    return from_scores if abs(from_scores - from_rank) < 1e-9 else None


def _grid_from_rankings(rankings, cancers, gene_ids):
    grid = np.zeros((len(cancers), len(gene_ids)))
    index = {gene: slot for slot, gene in enumerate(gene_ids)}
    for query, cancer in enumerate(cancers):
        for rank, gene in enumerate(rankings[cancer]):
            grid[query, index[gene]] = -rank
    return grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    context_path = args.run_dir / "analysis" / "preparation_context.json"
    if not context_path.exists():
        write_json(args.run_dir / "validation.json", {
            "status": "preparation_incomplete",
            "formal_test_files_opened": False,
            "models_fit": False,
        })
        (args.run_dir / "report.md").write_text(
            "# Development association recovery\n\nPreparation did not finish, so the six model jobs had no shared features.\n",
            encoding="utf-8",
        )
        print("preparation_incomplete", args.run_dir)
        return
    context = json.loads(context_path.read_text(encoding="utf-8"))
    prep = _load(args.run_dir / "preparation_validation.json") or {}
    method_metrics = {method: _load(args.run_dir / "analysis" / "models" / method / "metrics.json") for method in METHODS}
    reproduced = {}
    for method, score_name in PRIMARY.items():
        if method_metrics.get(method):
            reproduced[method] = _macro(args.run_dir, method, score_name, context)
    lines = [
        "# Development association recovery",
        "",
        f"Run directory: `{args.run_dir}`",
        "",
        "The development split was generated for this graph version. Protocol metadata records `outcome_benchmark: not_run_in_this_phase`, and no split manifest or reserved-target file is registered. Previously reserved targets were not opened. The formal test set stayed sealed.",
        "",
        f"Extracted CORE direct pairs: {context['summary']['core_direct_pairs']} ({context['summary']['both_resources']} also in SemMedDB, {context['summary']['core_only']} CORE only, {context['summary']['missing_core_pmid_provenance']} without a CORE PMID).",
        f"Selected development positives: {context['split']['selected_targets']} across {len(context['cancers'])} cancers. Roles: {context['split']['role_counts']}. Fractions: {context['split']['role_fractions']}.",
        f"PMID groups: {context['split']['pmid_groups']}. Largest group: {context['split']['largest_pmid_group']}.",
        "",
        "## Primary macro Recall@50",
        "",
        "| Method | Score | Macro Recall@50 | Recovered | Positives |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    completed = 0
    for method, score_name in PRIMARY.items():
        payload = method_metrics.get(method)
        if not payload or score_name not in payload.get("metrics", {}):
            lines.append(f"| {method} | {score_name} | unavailable |  |  |")
            continue
        completed += 1
        item = payload["metrics"][score_name]
        lines.append(
            f"| {method} | {score_name} | {item['macro_recall_at_50']:.4f} | {item['recovered']} | {item['positives']} |"
        )
    lines.extend(["", "## Feature contrasts on the primary seed", ""])
    for family in ("ridge", "trees"):
        payload = method_metrics.get(family) or {}
        lines.append(f"### {family}")
        lines.append("")
        lines.append("| Question | Left | Right | Left recall | Right recall | Delta |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: |")
        for row in payload.get("contrasts", []):
            lines.append(
                f"| {row['question']} | {row['left']} | {row['right']} | {row['left_macro_recall_at_50']:.4f} | {row['right_macro_recall_at_50']:.4f} | {row['delta']:.4f} |"
            )
        lines.append("")
    lines.extend([
        "## Checks",
        "",
        f"Unmasked path totals: {context.get('unmasked_path_totals')}.",
        f"Masked PMIDs absent from edges: {context.get('masked_pmids_absent_from_edges')}. Citation index clear: {context.get('citation_index_excludes_masked_pmids')}. Target pairs absent: {context.get('target_pairs_absent_from_graph')}.",
        f"Direction gaps (left-to-right + right-to-left + ambiguous − collapsed): {context.get('direction_gaps')}.",
        f"Saved rankings reproduce the primary metrics: {reproduced}.",
        "",
        "Six model jobs read the same masked features, the same development candidate lists, and the same training controls. LambdaMART was not fit. Neural scores use a six-epoch grid selected on the tuning partition. The comparison stops at development validation.",
        "",
    ])
    (args.run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    status = "passed" if completed == len(METHODS) and all(value is not None for value in reproduced.values()) and prep.get("status") == "prepared" else "partial"
    write_json(args.run_dir / "validation.json", {
        "status": status,
        "methods_completed": completed,
        "rankings_reproduce_primary_metrics": reproduced,
        "preparation": prep,
        "formal_test_files_opened": False,
    })
    write_json(args.run_dir / "manifest.json", {
        "task": "association_recovery_development",
        "run_dir": str(args.run_dir),
        "extracted_core_pairs": context["summary"]["core_direct_pairs"],
        "selected_targets": context["split"]["selected_targets"],
        "selected_cancers": len(context["cancers"]),
        "methods_completed": completed,
        "development_provenance": context["split"]["provenance"],
        "formal_test_files_opened": False,
    })
    print(status, args.run_dir)


if __name__ == "__main__":
    main()

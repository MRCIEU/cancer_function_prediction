"""Fit one development recovery method from the shared masked features.

Each method is a separate job. This script does not rebuild the graph and does
not open a formal test partition.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from src.common import StageTimer, write_json
from src.recovery_models import (
    RelationGCN,
    TemplatePathNet,
    cancer_weights,
    fit_logistic,
    fit_trees,
    macro_from_scores,
    prepare_features,
)
from src.recovery_scope import CONTRASTS, FEATURE_BLOCKS, POSITIONS, PRIMARY_SEED, SEEDS, combine_templates
from src.run_association_recovery import evaluate_method


def load_shared(run_dir: Path):
    features = dict(np.load(run_dir / "analysis" / "features.npz"))
    context_path = run_dir / "analysis" / "preparation_context.json"
    import json
    context = json.loads(context_path.read_text(encoding="utf-8"))
    samples = pq.read_table(run_dir / "analysis" / "training_samples.parquet").to_pydict()
    return features, context, samples


def subset(samples, role: str, seed: int | None, cancers: list[str]):
    keep = []
    for index, sample_role in enumerate(samples["role"]):
        if sample_role != role:
            continue
        if seed is not None and int(samples["seed"][index]) != int(seed):
            continue
        keep.append(index)
    return {
        "cancer": np.asarray([samples["cancer_cui"][index] for index in keep]),
        "gene": np.asarray([samples["gene_id"][index] for index in keep]),
        "label": np.asarray([samples["label"][index] for index in keep], dtype=np.int64),
    }


def matrix_from(features, columns, cancers, gene_index, cancer_ids, gene_ids):
    query = np.asarray([cancers.index(str(cancer)) for cancer in cancer_ids], dtype=np.int32)
    gene = np.asarray([gene_index[str(item)] for item in gene_ids], dtype=np.int32)
    if not columns:
        return np.zeros((len(query), 0))
    return np.column_stack([features[name][query, gene] for name in columns])


def scores_to_grid(flat_scores, cancer_ids, gene_ids, cancers, gene_index, shape):
    grid = np.full(shape, np.nan, dtype=np.float64)
    for score, cancer, gene in zip(flat_scores, cancer_ids, gene_ids):
        grid[cancers.index(str(cancer)), gene_index[str(gene)]] = float(score)
    return grid


def eligible_grid(cancer_ids, gene_ids, cancers, n_gene, gene_index):
    grid = np.zeros((len(cancers), n_gene), dtype=bool)
    for cancer, gene in zip(cancer_ids, gene_ids):
        grid[cancers.index(str(cancer)), gene_index[str(gene)]] = True
    return grid


def positives_from(samples, cancers):
    found = {cancer: set() for cancer in cancers}
    dev = subset(samples, "development_validation", None, cancers)
    for cancer, gene, label in zip(dev["cancer"], dev["gene"], dev["label"]):
        if int(label) == 1:
            found[str(cancer)].add(str(gene))
    return found


def metric_bundle(grid, eligible, positives, cancers, gene_ids):
    metric = evaluate_method(grid, gene_ids, eligible, positives, cancers)
    return {
        "macro_recall_at_50": metric["macro_recall"],
        "recovered": metric["recovered"],
        "positives": metric["positives"],
        "cancers": metric["cancers"],
        "per_cancer": metric["per_cancer"],
    }


def citation_observable(features, positives, cancers, gene_index) -> set[tuple[str, str]]:
    fields = [
        f"{position}_{name}"
        for position in POSITIONS
        for name in ("support_pmids", "direct_collapsed", "cocite", "shared_references")
    ]
    found = set()
    for cancer, genes in positives.items():
        query = cancers.index(cancer)
        for gene in genes:
            slot = gene_index[gene]
            if any(np.isfinite(features[field][query, slot]) for field in fields):
                found.add((cancer, gene))
    return found


def write_rankings(path: Path, grid, eligible, cancers, gene_ids, labels_by_pair):
    rows_cancer = []
    rows_gene = []
    rows_score = []
    rows_rank = []
    rows_label = []
    gene_index = {gene: index for index, gene in enumerate(gene_ids)}
    for query, cancer in enumerate(cancers):
        slots = np.flatnonzero(eligible[query])
        order = sorted(slots, key=lambda slot: (-(grid[query, slot] if np.isfinite(grid[query, slot]) else -1e300), gene_ids[slot]))
        for rank, slot in enumerate(order, start=1):
            gene = gene_ids[slot]
            rows_cancer.append(cancer)
            rows_gene.append(gene)
            value = grid[query, gene_index[gene]]
            rows_score.append(None if not np.isfinite(value) else float(value))
            rows_rank.append(rank)
            rows_label.append(int(labels_by_pair.get((cancer, gene), 0)))
    pq.write_table(pa.table({
        "cancer_cui": rows_cancer,
        "gene_id": rows_gene,
        "score": rows_score,
        "rank": rows_rank,
        "label": rows_label,
    }), path)


def subgroup_metrics(grid, eligible, positives, semmed, observable, cancers, gene_ids):
    groups = {
        "all": positives,
        "core_only": {cancer: {gene for gene in genes if gene not in set(semmed.get(cancer, []))} for cancer, genes in positives.items()},
        "shared_with_semmed": {cancer: {gene for gene in genes if gene in set(semmed.get(cancer, []))} for cancer, genes in positives.items()},
        "citation_observable": {cancer: {gene for gene in genes if (cancer, gene) in observable} for cancer, genes in positives.items()},
    }
    output = {}
    for name, held in groups.items():
        metric = evaluate_method(grid, gene_ids, eligible, held, cancers)
        output[name] = {
            "macro_recall_at_50": metric["macro_recall"],
            "recovered": metric["recovered"],
            "positives": metric["positives"],
            "cancers": metric["cancers"],
        }
    return output


def observable_targets(features, positives):
    fields = [f"{position}_{name}" for position in POSITIONS for name in ("support_pmids", "direct_collapsed", "cocite", "shared_references")]
    found = set()
    # Positives are small. Scan the feature rows for those genes via the caller-supplied coordinates.
    return fields, found


def run_structural(features, samples, context, method: str):
    cancers = context["cancers"]
    gene_ids = context["gene_ids"]
    gene_index = {gene: index for index, gene in enumerate(gene_ids)}
    dev = subset(samples, "development_validation", None, cancers)
    eligible = eligible_grid(dev["cancer"], dev["gene"], cancers, len(gene_ids), gene_index)
    positives = positives_from(samples, cancers)
    observable = citation_observable(features, positives, cancers, gene_index)
    if method == "path_counts":
        grids = {
            "degree_weighted_paths": combine_templates(features["two_edge_degree_weighted"], features["three_edge_degree_weighted"]),
            "two_edge_paths": features["two_edge_paths"],
            "three_edge_joint": features["three_edge_paths"],
            "three_edge_core": features["_core_counts"],
            "three_edge_semmed": features["_sem_counts"],
            "degree_control": features["candidate_degree"],
            "publication_control": features["candidate_publications"],
        }
    else:
        grids = {"random_walk": combine_templates(features["_walk_two"], features["_walk_three"])}
    metrics = {}
    for name, grid in grids.items():
        metrics[name] = metric_bundle(grid, eligible, positives, cancers, gene_ids)
        metrics[name]["subgroups"] = subgroup_metrics(grid, eligible, positives, context["target_semmed_direct"], observable, cancers, gene_ids)
    return grids, metrics, eligible, positives


def fit_block_family(features, samples, context, family: str, timer: StageTimer):
    cancers = context["cancers"]
    gene_ids = context["gene_ids"]
    gene_index = {gene: index for index, gene in enumerate(gene_ids)}
    dev = subset(samples, "development_validation", None, cancers)
    tune = subset(samples, "tune", None, cancers)
    eligible = eligible_grid(dev["cancer"], dev["gene"], cancers, len(gene_ids), gene_index)
    positives = positives_from(samples, cancers)
    observable = citation_observable(features, positives, cancers, gene_index)
    scale = family == "ridge"
    grids = {}
    chosen = {}
    for seed in SEEDS:
        blocks = list(FEATURE_BLOCKS) if seed == PRIMARY_SEED else ["combined"]
        train = subset(samples, "train", seed, cancers)
        for block in blocks:
            columns = list(FEATURE_BLOCKS[block])
            train_x = matrix_from(features, columns, cancers, gene_index, train["cancer"], train["gene"])
            tune_x = matrix_from(features, columns, cancers, gene_index, tune["cancer"], tune["gene"])
            if family == "ridge":
                fitted = fit_logistic(train_x, train["label"], train["cancer"], train["gene"], tune_x, tune["label"], tune["cancer"], tune["gene"])
                chosen_key = {"C": fitted["C"], "tune_recall_at_50": fitted["metric"]["macro_recall"]}
            else:
                fitted = fit_trees(train_x, train["label"], train["cancer"], train["gene"], tune_x, tune["label"], tune["cancer"], tune["gene"], seed)
                chosen_key = {
                    "max_depth": fitted["max_depth"],
                    "learning_rate": fitted["learning_rate"],
                    "tune_recall_at_50": fitted["metric"]["macro_recall"],
                }
            dev_x = matrix_from(features, columns, cancers, gene_index, dev["cancer"], dev["gene"])
            predicted = prepare_features(train_x, [dev_x], scale=scale)[1]
            if family == "ridge":
                flat = fitted["model"].decision_function(predicted)
            else:
                flat = fitted["model"].predict_proba(predicted)[:, 1]
            name = f"{family}_{block}_seed_{seed}"
            grids[name] = scores_to_grid(flat, dev["cancer"], dev["gene"], cancers, gene_index, (len(cancers), len(gene_ids)))
            chosen[name] = chosen_key
            timer.mark(name, **{key: value for key, value in chosen_key.items() if key != "tune_recall_at_50"}, tune_recall=chosen_key["tune_recall_at_50"])
    metrics = {}
    for name, grid in grids.items():
        metrics[name] = metric_bundle(grid, eligible, positives, cancers, gene_ids)
        metrics[name]["selection"] = chosen[name]
        metrics[name]["subgroups"] = subgroup_metrics(grid, eligible, positives, context["target_semmed_direct"], observable, cancers, gene_ids)
    contrasts = []
    if family == "ridge" or family == "trees":
        for question, left, right in CONTRASTS:
            left_name = f"{family}_{left}_seed_{PRIMARY_SEED}"
            right_name = f"{family}_{right}_seed_{PRIMARY_SEED}"
            contrasts.append({
                "question": question,
                "left": left,
                "right": right,
                "left_macro_recall_at_50": metrics[left_name]["macro_recall_at_50"],
                "right_macro_recall_at_50": metrics[right_name]["macro_recall_at_50"],
                "delta": metrics[right_name]["macro_recall_at_50"] - metrics[left_name]["macro_recall_at_50"],
            })
    return grids, metrics, contrasts, eligible, positives


def _pair_batch(hidden, scorer, cancer_ids, gene_ids, gene_index, cancer_node, extra):
    gene_nodes = torch.tensor([gene_index[str(gene)] for gene in gene_ids], dtype=torch.long)
    cancer_nodes = torch.tensor([cancer_node[str(cancer)] for cancer in cancer_ids], dtype=torch.long)
    pieces = [hidden[gene_nodes], hidden[cancer_nodes]]
    if extra is not None:
        pieces.append(torch.tensor(extra, dtype=torch.float32))
    return scorer(torch.cat(pieces, dim=1)).squeeze(1)


def run_rgcn(features, samples, context, run_dir: Path, with_features: bool, timer: StageTimer):
    pack = torch.load(run_dir / "analysis" / "graph_pack.pt", weights_only=False)
    cancers = context["cancers"]
    gene_ids = context["gene_ids"]
    gene_index = {gene: index for index, gene in enumerate(gene_ids)}
    columns = list(FEATURE_BLOCKS["combined"]) if with_features else []
    train = subset(samples, "train", PRIMARY_SEED, cancers)
    tune = subset(samples, "tune", None, cancers)
    dev = subset(samples, "development_validation", None, cancers)
    train_extra = prepare_features(matrix_from(features, columns, cancers, gene_index, train["cancer"], train["gene"]), [], scale=True)[0] if columns else None
    tune_extra = prepare_features(matrix_from(features, columns, cancers, gene_index, train["cancer"], train["gene"]), [matrix_from(features, columns, cancers, gene_index, tune["cancer"], tune["gene"])], scale=True)[1] if columns else None
    dev_extra = prepare_features(matrix_from(features, columns, cancers, gene_index, train["cancer"], train["gene"]), [matrix_from(features, columns, cancers, gene_index, dev["cancer"], dev["gene"])], scale=True)[1] if columns else None
    encoder = RelationGCN(pack["n_predicates"], 3, 32)
    scorer = torch.nn.Linear(encoder.hidden * 2 + (0 if train_extra is None else train_extra.shape[1]), 1)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(scorer.parameters()), lr=0.01)
    node_features = torch.zeros(pack["n_nodes"], 3)
    node_features[: len(gene_ids), 0] = torch.log1p(torch.tensor(features["candidate_degree"][0], dtype=torch.float32))
    node_features[: len(gene_ids), 1] = torch.log1p(torch.tensor(features["candidate_publications"][0], dtype=torch.float32))
    node_features[: len(gene_ids), 2] = 1
    weights = torch.tensor(cancer_weights(train["cancer"], train["label"]), dtype=torch.float32)
    labels = torch.tensor(train["label"], dtype=torch.float32)
    best = None
    for epoch in range(6):
        encoder.train()
        optimizer.zero_grad()
        hidden = encoder(node_features, pack["relations"])
        logits = _pair_batch(hidden, scorer, train["cancer"], train["gene"], gene_index, pack["cancer_node"], train_extra)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, weight=weights)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            hidden = encoder(node_features, pack["relations"])
            tune_scores = _pair_batch(hidden, scorer, tune["cancer"], tune["gene"], gene_index, pack["cancer_node"], tune_extra).numpy()
            tune_metric = macro_from_scores(tune_scores, tune["cancer"], tune["gene"], tune["label"])
        if best is None or tune_metric["macro_recall"] > best["tune_recall"]:
            best = {
                "epoch": epoch + 1,
                "tune_recall": tune_metric["macro_recall"],
                "encoder": {key: value.detach().clone() for key, value in encoder.state_dict().items()},
                "scorer": {key: value.detach().clone() for key, value in scorer.state_dict().items()},
            }
    encoder.load_state_dict(best["encoder"])
    scorer.load_state_dict(best["scorer"])
    with torch.no_grad():
        hidden = encoder(node_features, pack["relations"])
        flat = _pair_batch(hidden, scorer, dev["cancer"], dev["gene"], gene_index, pack["cancer_node"], dev_extra).numpy()
    grid = scores_to_grid(flat, dev["cancer"], dev["gene"], cancers, gene_index, (len(cancers), len(gene_ids)))
    timer.mark("rgcn" if not with_features else "rgcn_features", epoch=best["epoch"], tune_recall=best["tune_recall"])
    return grid, {"epoch": best["epoch"], "tune_recall_at_50": best["tune_recall"]}


def run_pathnet(features, samples, context, run_dir: Path, with_features: bool, timer: StageTimer):
    pack = torch.load(run_dir / "analysis" / "graph_pack.pt", weights_only=False)
    cancers = context["cancers"]
    gene_ids = context["gene_ids"]
    gene_index = {gene: index for index, gene in enumerate(gene_ids)}
    columns = list(FEATURE_BLOCKS["combined"]) if with_features else []
    train = subset(samples, "train", PRIMARY_SEED, cancers)
    tune = subset(samples, "tune", None, cancers)
    dev = subset(samples, "development_validation", None, cancers)
    train_raw = matrix_from(features, columns, cancers, gene_index, train["cancer"], train["gene"]) if columns else None
    model = TemplatePathNet(pack["n_predicates"])
    extra_width = 0 if train_raw is None else prepare_features(train_raw, [], scale=True)[0].shape[1]
    scorer = torch.nn.Linear(2 + extra_width, 1)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(scorer.parameters()), lr=0.05)
    labels = torch.tensor(train["label"], dtype=torch.float32)
    weights = torch.tensor(cancer_weights(train["cancer"], train["label"]), dtype=torch.float32)
    best = None

    def extras(split):
        if train_raw is None:
            return None
        raw = matrix_from(features, columns, cancers, gene_index, split["cancer"], split["gene"])
        return prepare_features(train_raw, [raw], scale=True)[1]

    def logits_for(two, three, split, extra):
        query_slots = torch.tensor([cancers.index(str(cancer)) for cancer in split["cancer"]], dtype=torch.long)
        gene_slots = torch.tensor([gene_index[str(gene)] for gene in split["gene"]], dtype=torch.long)
        mass = torch.stack([
            torch.log1p(two[query_slots, gene_slots].clamp(min=0)),
            torch.log1p(three[query_slots, gene_slots].clamp(min=0)),
        ], dim=1)
        pieces = [mass]
        if extra is not None:
            pieces.append(torch.tensor(extra, dtype=torch.float32))
        return scorer(torch.cat(pieces, dim=1)).squeeze(1)

    train_extra = extras(train)
    for epoch in range(6):
        optimizer.zero_grad()
        two, three = model(pack)
        logits = logits_for(two, three, train, train_extra)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, weight=weights)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            two, three = model(pack)
            tune_scores = logits_for(two, three, tune, extras(tune)).numpy()
            tune_metric = macro_from_scores(tune_scores, tune["cancer"], tune["gene"], tune["label"])
        if best is None or tune_metric["macro_recall"] > best["tune_recall"]:
            best = {
                "epoch": epoch + 1,
                "tune_recall": tune_metric["macro_recall"],
                "model": {key: value.detach().clone() for key, value in model.state_dict().items()},
                "scorer": {key: value.detach().clone() for key, value in scorer.state_dict().items()},
            }
    model.load_state_dict(best["model"])
    scorer.load_state_dict(best["scorer"])
    with torch.no_grad():
        two, three = model(pack)
        flat = logits_for(two, three, dev, extras(dev)).numpy()
    grid = scores_to_grid(flat, dev["cancer"], dev["gene"], cancers, gene_index, (len(cancers), len(gene_ids)))
    timer.mark("pathnet" if not with_features else "pathnet_features", epoch=best["epoch"], tune_recall=best["tune_recall"])
    return grid, {"epoch": best["epoch"], "tune_recall_at_50": best["tune_recall"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--method", required=True, choices=["path_counts", "random_walk", "ridge", "trees", "rgcn", "pathnet"])
    args = parser.parse_args()
    timer = StageTimer()
    features, context, samples = load_shared(args.run_dir)
    out = args.run_dir / "analysis" / "models" / args.method
    out.mkdir(parents=True, exist_ok=True)
    cancers = context["cancers"]
    gene_ids = context["gene_ids"]
    contrasts = []
    selection = {}
    if args.method in {"path_counts", "random_walk"}:
        grids, metrics, eligible, positives = run_structural(features, samples, context, args.method)
    elif args.method in {"ridge", "trees"}:
        grids, metrics, contrasts, eligible, positives = fit_block_family(features, samples, context, args.method, timer)
    else:
        dev = subset(samples, "development_validation", None, cancers)
        gene_index = {gene: index for index, gene in enumerate(gene_ids)}
        eligible = eligible_grid(dev["cancer"], dev["gene"], cancers, len(gene_ids), gene_index)
        positives = positives_from(samples, cancers)
        observable = citation_observable(features, positives, cancers, gene_index)
        if args.method == "rgcn":
            plain, plain_info = run_rgcn(features, samples, context, args.run_dir, False, timer)
            featured, featured_info = run_rgcn(features, samples, context, args.run_dir, True, timer)
        else:
            plain, plain_info = run_pathnet(features, samples, context, args.run_dir, False, timer)
            featured, featured_info = run_pathnet(features, samples, context, args.run_dir, True, timer)
        grids = {args.method: plain, f"{args.method}_features": featured}
        selection = {args.method: plain_info, f"{args.method}_features": featured_info}
        metrics = {}
        for name, grid in grids.items():
            metrics[name] = metric_bundle(grid, eligible, positives, cancers, gene_ids)
            metrics[name]["selection"] = selection[name]
            metrics[name]["subgroups"] = subgroup_metrics(grid, eligible, positives, context["target_semmed_direct"], observable, cancers, gene_ids)
    labels = {}
    dev = subset(samples, "development_validation", None, cancers)
    for cancer, gene, label in zip(dev["cancer"], dev["gene"], dev["label"]):
        labels[(str(cancer), str(gene))] = int(label)
    for name, grid in grids.items():
        write_rankings(out / f"{name}_rankings.parquet", grid, eligible, cancers, gene_ids, labels)
    np.savez(out / "dev_scores.npz", **{name: value for name, value in grids.items()})
    payload = {
        "method": args.method,
        "metrics": metrics,
        "contrasts": contrasts,
        "stages": timer.stages,
        "formal_test_files_opened": False,
    }
    write_json(out / "metrics.json", payload)
    print(args.method, "complete", {name: item["macro_recall_at_50"] for name, item in metrics.items()})


if __name__ == "__main__":
    main()

"""Pointwise models and template-constrained graph scorers for development recovery."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.association_recovery import rank_with_tie_break, recall_at_k


LOGISTIC_C = (0.01, 0.1, 1.0, 10.0)
TREE_GRID = ((2, 0.05), (2, 0.1), (3, 0.05), (3, 0.1))


def prepare_features(train: np.ndarray, later: list[np.ndarray], scale: bool) -> list[np.ndarray]:
    """Median-impute missing values and optional scaling, both fit on training rows.

    A missingness indicator keeps an unobserved citation field distinct from a
    confirmed zero. Indicators are determined by the row, not by a label.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(train, axis=0)
    median = np.where(np.isnan(median), 0.0, median)

    def transform(matrix: np.ndarray, fitted: StandardScaler | None) -> np.ndarray:
        missing = np.isnan(matrix)
        filled = np.where(missing, median, matrix)
        base = fitted.transform(filled) if fitted is not None else filled
        return np.hstack([base, missing.astype(np.float64)])

    scaler = None
    if scale:
        missing = np.isnan(train)
        scaler = StandardScaler()
        scaler.fit(np.where(missing, median, train))
    return [transform(matrix, scaler) for matrix in [train, *later]]


def cancer_weights(cancer_ids: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Give each cancer the same total weight in a pointwise loss."""
    weights = np.ones(len(cancer_ids), dtype=np.float64)
    for cancer in np.unique(cancer_ids):
        mask = cancer_ids == cancer
        weights[mask] = 1.0 / max(int(mask.sum()), 1)
    return weights


def macro_from_scores(scores: np.ndarray, cancer_ids: np.ndarray, gene_ids: np.ndarray, labels: np.ndarray) -> dict:
    rankings = {}
    positives = {}
    for cancer in np.unique(cancer_ids):
        mask = cancer_ids == cancer
        rows = [
            {"gene": str(gene), "score": float(score)}
            for gene, score in zip(gene_ids[mask], scores[mask])
        ]
        rankings[str(cancer)] = rank_with_tie_break(rows)
        positives[str(cancer)] = {str(gene) for gene, label in zip(gene_ids[mask], labels[mask]) if int(label) == 1}
    return recall_at_k(rankings, positives, k=50)


def fit_logistic(train_x, train_y, train_cancer, train_gene, tune_x, tune_y, tune_cancer, tune_gene):
    prepared = prepare_features(train_x, [tune_x], scale=True)
    train_p, tune_p = prepared
    weights = cancer_weights(train_cancer, train_y)
    best = None
    for penalty in LOGISTIC_C:
        model = LogisticRegression(C=float(penalty), solver="lbfgs", max_iter=400)
        model.fit(train_p, train_y, sample_weight=weights)
        metric = macro_from_scores(model.decision_function(tune_p), tune_cancer, tune_gene, tune_y)
        if best is None or metric["macro_recall"] > best["metric"]["macro_recall"]:
            best = {"model": model, "metric": metric, "C": float(penalty), "train": train_p}
    return best


def fit_trees(train_x, train_y, train_cancer, train_gene, tune_x, tune_y, tune_cancer, tune_gene, seed: int):
    prepared = prepare_features(train_x, [tune_x], scale=False)
    train_p, tune_p = prepared
    weights = cancer_weights(train_cancer, train_y)
    best = None
    for depth, rate in TREE_GRID:
        model = HistGradientBoostingClassifier(
            max_depth=int(depth),
            learning_rate=float(rate),
            max_iter=40,
            early_stopping=False,
            random_state=int(seed),
        )
        model.fit(train_p, train_y, sample_weight=weights)
        metric = macro_from_scores(model.predict_proba(tune_p)[:, 1], tune_cancer, tune_gene, tune_y)
        if best is None or metric["macro_recall"] > best["metric"]["macro_recall"]:
            best = {
                "model": model,
                "metric": metric,
                "max_depth": int(depth),
                "learning_rate": float(rate),
                "train": train_p,
            }
    return best


def transform_like(reference_train: np.ndarray, matrices: list[np.ndarray], scale: bool) -> list[np.ndarray]:
    return prepare_features(reference_train, matrices, scale=scale)[1:]


class RelationGCN(torch.nn.Module):
    """Two-layer relational encoder. One message per source, relation, and target."""

    def __init__(self, n_relations: int, in_dim: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.first = torch.nn.ModuleList([torch.nn.Linear(in_dim, hidden, bias=False) for _ in range(n_relations)])
        self.first_loop = torch.nn.Linear(in_dim, hidden, bias=False)
        self.second = torch.nn.ModuleList([torch.nn.Linear(hidden, hidden, bias=False) for _ in range(n_relations)])
        self.second_loop = torch.nn.Linear(hidden, hidden, bias=False)

    def _mix(self, values, relations, layers, loop):
        mixed = loop(values)
        degree = torch.ones(values.shape[0], device=values.device)
        for relation, (src, dst) in enumerate(relations):
            if src.numel() == 0:
                continue
            messages = layers[relation](values[src])
            bucket = torch.zeros_like(mixed)
            bucket.index_add_(0, dst, messages)
            mixed = mixed + bucket
            degree.index_add_(0, dst, torch.ones(dst.shape[0], device=values.device))
        return mixed / degree.clamp(min=1).unsqueeze(1)

    def forward(self, features, relations):
        hidden = torch.relu(self._mix(features, relations, self.first, self.first_loop))
        return self._mix(hidden, relations, self.second, self.second_loop)


class TemplatePathNet(torch.nn.Module):
    """Query-conditioned propagation limited to the two audited templates."""

    def __init__(self, n_predicates: int):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.zeros(n_predicates))

    def edge_weight(self, predicate_hot: torch.Tensor) -> torch.Tensor:
        return predicate_hot @ torch.nn.functional.softplus(self.theta)

    def forward(self, pack: dict) -> tuple[torch.Tensor, torch.Tensor]:
        return propagate_templates(
            self.edge_weight(pack["gene_cancer_hot"]),
            self.edge_weight(pack["gene_gene_hot"]),
            pack["gc_gene"],
            pack["gc_cancer"],
            pack["gg_src"],
            pack["gg_dst"],
            pack["query_cancer"],
            pack["candidate_mask"],
            int(pack["n_gene"]),
            int(pack["n_cancer"]),
        )


def propagate_templates(
    gc_w: torch.Tensor,
    gg_w: torch.Tensor,
    gc_gene: torch.Tensor,
    gc_cancer: torch.Tensor,
    gg_src: torch.Tensor,
    gg_dst: torch.Tensor,
    query_cancer: torch.Tensor,
    candidate_mask: torch.Tensor,
    n_gene: int,
    n_cancer: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Propagate one mass per query through the two-edge and three-edge templates.

    Equal edge weights reproduce the uniform neighbour walk. A candidate cannot
    repeat the intermediate gene, and the intermediate cancer cannot be the query.
    """
    n_query = int(query_cancer.shape[0])
    same = gc_cancer.unsqueeze(0) == query_cancer.unsqueeze(1)
    raw = torch.zeros(n_query, n_gene, dtype=gc_w.dtype)
    matched = torch.nonzero(same, as_tuple=False)
    if matched.numel():
        raw.index_put_((matched[:, 0], gc_gene[matched[:, 1]]), gc_w[matched[:, 1]], accumulate=True)
    start = raw / raw.sum(dim=1, keepdim=True).clamp(min=1e-8)

    degree = torch.zeros(n_gene, dtype=gg_w.dtype)
    if gg_src.numel():
        degree.index_add_(0, gg_src, gg_w)
        share = gg_w / degree[gg_src].clamp(min=1e-8)
        two = torch.zeros(n_query, n_gene, dtype=gc_w.dtype)
        two.index_add_(1, gg_dst, start[:, gg_src] * share.unsqueeze(0))
    else:
        two = torch.zeros(n_query, n_gene, dtype=gc_w.dtype)

    total = torch.zeros(n_gene, dtype=gc_w.dtype)
    total.index_add_(0, gc_gene, gc_w)
    remain = (total.unsqueeze(0) - raw).clamp(min=0)
    candidate_total = torch.zeros(n_cancer, dtype=gc_w.dtype)
    candidate_edges = candidate_mask[gc_gene]
    if bool(candidate_edges.any()):
        candidate_total.index_add_(0, gc_cancer[candidate_edges], gc_w[candidate_edges])
    cancer_coeff = torch.zeros(n_query, n_cancer, dtype=gc_w.dtype)
    other = torch.nonzero(~same, as_tuple=False)
    three = torch.zeros(n_query, n_gene, dtype=gc_w.dtype)
    if other.numel():
        query_index = other[:, 0]
        edge_index = other[:, 1]
        gene = gc_gene[edge_index]
        cancer = gc_cancer[edge_index]
        exclude = candidate_total[cancer] - torch.where(candidate_mask[gene], gc_w[edge_index], 0)
        hop = torch.where(
            remain[query_index, gene] > 0,
            start[query_index, gene] * gc_w[edge_index] / remain[query_index, gene],
            0,
        )
        coeff = torch.where(exclude > 0, hop / exclude, 0)
        cancer_coeff.index_put_((query_index, cancer), coeff, accumulate=True)
        own = torch.nonzero(candidate_edges, as_tuple=False).flatten()
        if own.numel():
            own_gene = gc_gene[own]
            own_cancer = gc_cancer[own]
            three.index_add_(1, own_gene, gc_w[own].unsqueeze(0) * cancer_coeff[:, own_cancer])
            own_exclude = candidate_total[own_cancer] - gc_w[own]
            own_hop = torch.where(
                remain[:, own_gene] > 0,
                start[:, own_gene] * gc_w[own].unsqueeze(0) / remain[:, own_gene],
                0,
            )
            own_coeff = torch.where(own_exclude > 0, own_hop / own_exclude, 0)
            is_query = own_cancer.unsqueeze(0) == query_cancer.unsqueeze(1)
            own_coeff = own_coeff.masked_fill(is_query, 0)
            three.index_add_(1, own_gene, -(gc_w[own].unsqueeze(0) * own_coeff))
    return two, three.clamp(min=0)

"""Frozen evaluation lists, missingness codes, and synthetic checks.

These helpers do not read a formal test file and do not score development
candidates. Production aggregation stays in the preparation runner.
"""

from __future__ import annotations

import numpy as np

from src.association_recovery import rank_with_tie_break, recall_at_k
from src.citation_features import direct_citation_roles, intersection_size
from src.relation_attributes import GG_PREDICATES
from src.relation_representation import DIRECTION_NAMES, FIELD_TOKENS, SYMMETRIC_PREDICATES

PATH_BUDGET = 8
ROLES = ("train", "tune", "development_validation")
MISSING_NO_PATH = 0
MISSING_PATH_WITHOUT_RECORD = 1
MISSING_UNJUDGEABLE = 2
MISSING_OBSERVED = 3
CITATION_NO_PATH = 0
CITATION_UNKNOWN = 1
CITATION_CONFIRMED_ZERO = 2
CITATION_OBSERVED_LINK = 3
ARTICLE_PAIRS_NOT_APPLICABLE = 0
ARTICLE_PAIRS_ENUMERATED = 1
ARTICLE_PAIRS_NOT_ENUMERATED = 2
ARTICLE_PAIR_CAP = 64


def combo_column_names(position: str) -> list[str]:
    names = []
    for cge in FIELD_TOKENS["cge"]:
        for ccs in FIELD_TOKENS["ccs"]:
            for gci in FIELD_TOKENS["gci"]:
                names.append(f"{position}_combo_{cge or 'EMPTY'}_{ccs or 'EMPTY'}_{gci or 'EMPTY'}")
    return names


def marginal_column_names(position: str) -> list[str]:
    names = []
    for field, tokens in FIELD_TOKENS.items():
        for token in tokens:
            names.append(f"{position}_marginal_{field}_{token or 'EMPTY'}")
    return names


def direction_column_names(position: str, predicates: tuple[str, ...]) -> list[str]:
    return [f"{position}_dir_{name}_{direction}" for name in predicates for direction in DIRECTION_NAMES]


def asymmetric_predicates(predicates: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(name for name in predicates if name not in SYMMETRIC_PREDICATES)


def classify_list_membership(is_unmasked_direct: bool, is_visible_direct: bool, target_role: str | None) -> dict:
    """Return membership in the full, primary, and historical evaluation lists.

    The primary list removes a candidate only when the masked graph still
    shows a direct-eligible edge. The historical list removes every unmasked
    direct gene except the hidden target of the partition being scored.
    """
    if is_visible_direct and target_role is not None:
        raise ValueError("a hidden target is still a visible direct edge")
    historical = {role: not (is_unmasked_direct and target_role != role) for role in ROLES}
    reason = "background"
    if target_role is not None:
        reason = f"hidden_{target_role}"
    elif is_visible_direct:
        reason = "visible_direct_after_mask"
    elif is_unmasked_direct:
        reason = "unmasked_direct_removed_by_mask"
    return {
        "in_full": True,
        "in_primary": not is_visible_direct,
        "historical": historical,
        "reason": reason,
    }


def apply_score_mask(scores: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """Keep one score matrix readable under either frozen list. Ineligible cells become missing."""
    masked = np.array(scores, dtype=np.float64, copy=True)
    masked[~np.asarray(eligible, dtype=bool)] = np.nan
    return masked


def evaluate_masked_scores(scores: np.ndarray, genes: list[str], eligible: np.ndarray, positives: dict[str, set[str]], cancers: list[str], ks: tuple[int, ...] = (50, 100)) -> dict:
    """Score rows are already computed. This only applies a list and the frozen metrics."""
    results = {}
    entry_exit = []
    previous = {}
    for k in ks:
        rankings = {}
        held = {}
        for row, cancer in enumerate(cancers):
            keep = np.asarray(eligible[row], dtype=bool)
            rows = []
            for slot in np.flatnonzero(keep):
                value = float(scores[row, slot])
                rows.append({"gene": genes[slot], "score": -1e300 if np.isnan(value) else value})
            rankings[cancer] = rank_with_tie_break(rows)
            held[cancer] = positives.get(cancer, set())
            if k == 50:
                current = set(rankings[cancer][:50])
                prior = previous.get(cancer, set())
                if prior:
                    entry_exit.append({
                        "cancer": cancer,
                        "entered": sorted(current - prior),
                        "left": sorted(prior - current),
                    })
                previous[cancer] = current
        metric = recall_at_k(rankings, held, k)
        results[f"recall_at_{k}"] = {key: metric[key] for key in ("macro_recall", "cancers", "recovered", "positives")}
    results["top50_entry_exit"] = entry_exit
    return results


def citation_motif(left: tuple[int, ...], right: tuple[int, ...], outgoing: dict, incoming: dict, indexed_pmids: set[int] | None) -> dict:
    """Motifs for one adjacent pair after the blocked-PMID filter has already been applied.

    A side with no PMID is unknown. A PMID absent from the snapshot metadata is
    unknown. Both sides indexed and no link is a confirmed zero.
    """
    left_ids = tuple(sorted({int(item) for item in left if int(item) > 0}))
    right_ids = tuple(sorted({int(item) for item in right if int(item) > 0}))
    if not left_ids or not right_ids:
        return {"state": CITATION_UNKNOWN, "article_pairs_state": ARTICLE_PAIRS_NOT_APPLICABLE}
    if indexed_pmids is not None and (any(pmid not in indexed_pmids for pmid in left_ids) or any(pmid not in indexed_pmids for pmid in right_ids)):
        return {"state": CITATION_UNKNOWN, "article_pairs_state": ARTICLE_PAIRS_NOT_APPLICABLE}
    roles = direct_citation_roles(left_ids, right_ids, outgoing)
    cocite = _witnesses(left_ids, right_ids, incoming)
    references = _witnesses(left_ids, right_ids, outgoing)
    collapsed = roles["direction_collapsed"]
    linked = collapsed > 0 or cocite["witnesses"] > 0 or references["witnesses"] > 0
    return {
        "state": CITATION_OBSERVED_LINK if linked else CITATION_CONFIRMED_ZERO,
        "support_pmids": len(set(left_ids) | set(right_ids)),
        "shared_articles": intersection_size(left_ids, right_ids),
        "direct_collapsed": roles["direction_collapsed"],
        "direct_ltr": roles["exclusive_left_to_right"],
        "direct_rtl": roles["exclusive_right_to_left"],
        "direct_ambiguous": roles["ambiguous_position"],
        "cocite_witnesses": cocite["witnesses"],
        "shared_reference_witnesses": references["witnesses"],
        "article_pairs_state": cocite["article_pairs_state"],
    }


def _witnesses(left: tuple[int, ...], right: tuple[int, ...], neighbors: dict) -> dict:
    left_set = set(left)
    right_set = set(right)
    papers = left_set | right_set
    by_witness: dict[int, set[int]] = {}
    for paper in papers:
        for witness in neighbors.get(paper, ()):
            witness = int(witness)
            if witness == paper:
                continue
            by_witness.setdefault(witness, set()).add(paper)
    witnesses = 0
    enumerate_pairs = len(left_set) * len(right_set) <= ARTICLE_PAIR_CAP
    for cited in by_witness.values():
        on_left = cited & left_set
        on_right = cited & right_set
        if not on_left or not on_right:
            continue
        if len(on_left) == 1 and len(on_right) == 1 and next(iter(on_left)) == next(iter(on_right)):
            continue
        witnesses += 1
    return {
        "witnesses": witnesses,
        "article_pairs_state": ARTICLE_PAIRS_ENUMERATED if enumerate_pairs else ARTICLE_PAIRS_NOT_ENUMERATED,
    }


def update_topk(values: np.ndarray, keys_a: np.ndarray, keys_b: np.ndarray, counts: np.ndarray, slots: np.ndarray, weight: np.ndarray, key_a: int, key_b: int) -> None:
    """Insert one shared path weight into a per-candidate top-k table.

    Higher weight wins. Equal weights keep the smaller intermediate identifiers.
    Unused slots are negative and sort after every real path. The rule does not read labels.
    """
    weight = np.asarray(weight, dtype=np.float64)
    worst = values[slots, -1]
    worst_a = keys_a[slots, -1]
    worst_b = keys_b[slots, -1]
    improve = (weight > worst) | ((weight == worst) & ((key_a < worst_a) | ((key_a == worst_a) & (key_b < worst_b))))
    chosen = np.flatnonzero(improve)
    if len(chosen) == 0:
        return
    take = slots[chosen]
    values[take, -1] = weight[chosen]
    keys_a[take, -1] = key_a
    keys_b[take, -1] = key_b
    block_v = values[take]
    block_a = keys_a[take]
    block_b = keys_b[take]
    order = np.argsort(np.rec.fromarrays((-block_v, block_a, block_b), names="w,a,b"), axis=1)
    values[take] = np.take_along_axis(block_v, order, axis=1)
    keys_a[take] = np.take_along_axis(block_a, order, axis=1)
    keys_b[take] = np.take_along_axis(block_b, order, axis=1)
    counts[take] = np.minimum(counts[take] + 1, values.shape[1])


def symmetric_direction_names() -> tuple[str, ...]:
    return tuple(name for name in GG_PREDICATES if name in SYMMETRIC_PREDICATES)

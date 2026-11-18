"""Audit coarse relation features against same-sentence and directed encodings.

The path weights are the DWPC values from the relation-attribute run. This
module does not fit a model and does not rank candidates.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.relation_attributes import (
    DAMPING,
    GC_PREDICATES,
    GG_PREDICATES,
    _power,
)

FIELD_TOKENS = {
    "cge": ("", "NOTINF", "DOWN", "UP"),
    "ccs": ("", "NOTINF", "REGRESSION", "PROGRESSION"),
    "gci": ("", "NOTINF", "OBSERVATION", "CAUSALITY"),
}
N_TOKEN = 4
N_COMBO = N_TOKEN ** 3
N_MARGINAL = N_TOKEN * 3
DIRECTION_NAMES = ("along", "against", "unresolved")
SYMMETRIC_PREDICATES = frozenset({"INTERACTS_WITH", "COEXISTS_WITH", "ASSOCIATED_WITH", "compared_with"})
QUANT_PRIMARY = 6
QUANT_SENSITIVITY = 4
GC_POSITIONS = ("t1_e2", "t2_e1", "t2_e2", "t2_e3")
GG_POSITIONS = ("t1_e1",)
POSITIONS = GG_POSITIONS + GC_POSITIONS


def combo_index(triple: tuple[str, str, str]) -> int:
    """Stable index for one raw ``(CGE, CCS, GCI)`` triple. Unknown tokens return -1."""
    try:
        return (
            FIELD_TOKENS["cge"].index(triple[0]) * 16
            + FIELD_TOKENS["ccs"].index(triple[1]) * 4
            + FIELD_TOKENS["gci"].index(triple[2])
        )
    except ValueError:
        return -1


def classify_core_record(sentence_id: str, cge: str, ccs: str, gci: str) -> dict:
    """Classify one CORE source row.

    A traceable complete tuple has a sentence identifier and a non-empty token
    in every field. ``NOTINF`` is an explicit token. An empty token makes the
    row partial. A row with no sentence identifier is not a same-sentence instance.
    """
    tokens = (cge or "", ccs or "", gci or "")
    empty = any(token == "" for token in tokens)
    notinf = any(token == "NOTINF" for token in tokens)
    if not sentence_id:
        return {"status": "untraceable", "combination": None, "explicit_notinf": notinf, "empty_label": empty}
    if empty or combo_index(tokens) < 0:
        return {"status": "partial", "combination": None, "explicit_notinf": notinf, "empty_label": empty or combo_index(tokens) < 0}
    return {
        "status": "traceable_complete",
        "combination": tokens,
        "explicit_notinf": notinf,
        "empty_label": False,
        "informative_complete": not notinf,
    }


def canonical_direction(subject: str, object_: str, origin: str, destination: str) -> str:
    """Orientation of one assertion relative to one canonical endpoint order."""
    if subject == origin and object_ == destination:
        return "along"
    if subject == destination and object_ == origin:
        return "against"
    return "unresolved"


def swap_direction(direction: str) -> str:
    if direction == "along":
        return "against"
    if direction == "against":
        return "along"
    return direction


def combination_share(combinations: set[tuple[str, str, str]]) -> np.ndarray:
    """Equal split of one path across distinct complete triples."""
    share = np.zeros(N_COMBO, dtype=np.float64)
    usable = [combo for combo in combinations if combo_index(combo) >= 0]
    if not usable:
        return share
    piece = 1.0 / len(usable)
    for combo in usable:
        share[combo_index(combo)] += piece
    return share


def marginal_from_joint(share: np.ndarray) -> np.ndarray:
    """Project a combination distribution onto the three raw field margins."""
    marginal = np.zeros(N_MARGINAL, dtype=np.float64)
    for index, value in enumerate(share):
        if value == 0.0:
            continue
        cge, rem = divmod(index, 16)
        ccs, gci = divmod(rem, 4)
        marginal[cge] += value
        marginal[4 + ccs] += value
        marginal[8 + gci] += value
    return marginal


def direction_share(directions: dict[str, set[str]], predicates: tuple[str, ...], reversed_traversal: bool) -> np.ndarray:
    """Give each predicate its undirected share, then split that share across observed directions."""
    present = [name for name in predicates if directions.get(name)]
    share = np.zeros(len(predicates) * 3, dtype=np.float64)
    if not present:
        return share
    each = 1.0 / len(present)
    for name in present:
        observed = {swap_direction(item) if reversed_traversal else item for item in directions[name]}
        piece = each / len(observed)
        base = predicates.index(name) * 3
        for direction in observed:
            share[base + DIRECTION_NAMES.index(direction)] += piece
    return share


def undirected_from_directions(share: np.ndarray, width: int = 3) -> np.ndarray:
    """Sum direction bins back to one weight per predicate."""
    array = np.asarray(share)
    if array.ndim == 1:
        return array.reshape(-1, width).sum(axis=1)
    return array.reshape(array.shape[0], -1, width).sum(axis=2)


def quant_signature(vector: np.ndarray | None, decimals: int) -> tuple:
    """Quantized signature. Missing entries stay distinct from observed zeros."""
    if vector is None:
        return ("absent",)
    rounded = np.round(vector, decimals)
    missing = np.isnan(vector)
    parts = []
    for index, value in enumerate(rounded):
        if missing[index]:
            parts.append((index, "missing"))
        elif value != 0:
            parts.append((index, float(value)))
    return tuple(parts) if parts else (("empty",),)


def collision_summary(coarse: list[tuple], fine: list[tuple]) -> dict:
    """Count coarse groups that contain more than one fine signature."""
    grouped: dict[tuple, set] = defaultdict(set)
    sizes: dict[tuple, int] = defaultdict(int)
    for left, right in zip(coarse, fine):
        grouped[left].add(right)
        sizes[left] += 1
    multi = [key for key, values in grouped.items() if len(values) > 1]
    return {
        "coarse_groups": len(grouped),
        "multi_fine_groups": len(multi),
        "candidates_in_multi": int(sum(sizes[key] for key in multi)),
        "fine_signatures_in_multi": int(sum(len(grouped[key]) for key in multi)),
        "candidates": len(coarse),
        "fine_signatures": len(set(fine)),
    }


def in_multi_group(coarse: list[tuple], fine: list[tuple]) -> list[bool]:
    grouped: dict[tuple, set] = defaultdict(set)
    for left, right in zip(coarse, fine):
        grouped[left].add(right)
    multi = {key for key, values in grouped.items() if len(values) > 1}
    return [key in multi for key in coarse]


def marginal_collision_groups(records: list[tuple[tuple, tuple]]) -> dict:
    """Group connections that share a combination-marginal and differ in the joint.

    Each item is ``(marginal_key, joint_key)``. Multiple combinations on one
    connection are not themselves a collision.
    """
    grouped: dict[tuple, set] = defaultdict(set)
    sizes: dict[tuple, int] = defaultdict(int)
    for marginal, joint in records:
        grouped[marginal].add(joint)
        sizes[marginal] += 1
    multi = [key for key, values in grouped.items() if len(values) > 1]
    return {
        "marginal_groups": len(grouped),
        "groups_with_distinct_joints": len(multi),
        "connections_in_those_groups": int(sum(sizes[key] for key in multi)),
    }


def accumulate_fine(
    query: str,
    genes_of: dict[str, set[str]],
    neighbors: dict[str, set[str]],
    cancers_of: dict[str, set[str]],
    index: dict[str, int],
    candidate_slots: dict[str, np.ndarray],
    deg_gg: dict[str, int],
    deg_gc: dict[str, int],
    edges: dict[tuple[str, str, str], dict],
    matrices: dict[tuple[str, str], np.ndarray],
    touched: dict[str, set] | None = None,
) -> dict[str, np.ndarray]:
    """DWPC-weighted fine shares for one query, using the same path weights as ``score_query``."""
    n_gene = len(index)
    combo = {position: np.zeros((n_gene, N_COMBO), dtype=np.float64) for position in GC_POSITIONS}
    marginal = {position: np.zeros((n_gene, N_MARGINAL), dtype=np.float64) for position in GC_POSITIONS}
    direction = {
        position: np.zeros((n_gene, (len(GG_PREDICATES) if position == "t1_e1" else len(GC_PREDICATES)) * 3), dtype=np.float64)
        for position in POSITIONS
    }
    complete_dwpc = {position: np.zeros(n_gene, dtype=np.float64) for position in GC_POSITIONS}
    directed_dwpc = {position: np.zeros(n_gene, dtype=np.float64) for position in POSITIONS}
    complete_paths = {position: np.zeros(n_gene, dtype=np.int64) for position in GC_POSITIONS}
    directed_paths = {position: np.zeros(n_gene, dtype=np.int64) for position in POSITIONS}
    two_dwpc = np.zeros(n_gene, dtype=np.float64)
    three_dwpc = np.zeros(n_gene, dtype=np.float64)
    two_paths = np.zeros(n_gene, dtype=np.int64)
    three_paths = np.zeros(n_gene, dtype=np.int64)
    q_factor = _power(len(genes_of.get(query, ())), DAMPING)
    slot_gc = np.zeros(n_gene, dtype=np.float64)
    slot_gg = np.zeros(n_gene, dtype=np.float64)
    for gene, slot in index.items():
        slot_gc[slot] = _power(deg_gc.get(gene, 0), DAMPING)
        slot_gg[slot] = _power(deg_gg.get(gene, 0), DAMPING)
    reverse = {slot: gene for gene, slot in index.items()}

    def add_combo(position: str, slots: np.ndarray, weights: np.ndarray, edge: dict | None, key: tuple | None = None) -> None:
        if touched is not None and key is not None:
            touched[position].add(key)
        if edge is None or edge["combo_share"] is None:
            return
        share = edge["combo_share"]
        margin = edge["marginal_share"]
        complete_dwpc[position][slots] += weights
        complete_paths[position][slots] += 1
        nonzero = np.flatnonzero(share)
        for column in nonzero:
            combo[position][slots, column] += weights * share[column]
            # Marginal is filled from the joint projection stored on the edge.
        nonzero_m = np.flatnonzero(margin)
        for column in nonzero_m:
            marginal[position][slots, column] += weights * margin[column]

    def add_direction(position: str, slots: np.ndarray, weights: np.ndarray, edge: dict | None, reversed_traversal: bool, key: tuple | None = None) -> None:
        if touched is not None and key is not None:
            touched[position].add(key)
        if edge is None or not edge["directions"]:
            return
        predicates = GG_PREDICATES if position == "t1_e1" else GC_PREDICATES
        share = direction_share(edge["directions"], predicates, reversed_traversal)
        if not np.any(share):
            return
        directed_dwpc[position][slots] += weights
        directed_paths[position][slots] += 1
        nonzero = np.flatnonzero(share)
        for column in nonzero:
            direction[position][slots, column] += weights * share[column]

    for gene in genes_of.get(query, ()):
        partners = [item for item in neighbors.get(gene, ()) if item in index and item != gene]
        if not partners:
            continue
        base = _power(deg_gg.get(gene, 0), DAMPING) * _power(deg_gc.get(gene, 0), DAMPING) * q_factor
        partner_slots = np.array([index[item] for item in partners], dtype=np.int32)
        weights = base * slot_gg[partner_slots]
        two_dwpc[partner_slots] += weights
        two_paths[partner_slots] += 1
        right = edges.get(("gene_cancer", gene, query))
        right_key = ("gene_cancer", gene, query)
        add_combo("t1_e2", partner_slots, weights, right, right_key)
        add_direction("t1_e2", partner_slots, weights, right, False, right_key)
        forward = []
        backward = []
        for partner, slot, weight in zip(partners, partner_slots, weights):
            ordered = (partner, gene) if partner <= gene else (gene, partner)
            bucket = forward if ordered[0] == partner else backward
            bucket.append((slot, weight, edges.get(("gene_gene", ordered[0], ordered[1]))))
        for reversed_traversal, bucket in ((False, forward), (True, backward)):
            by_edge: dict[int, list] = defaultdict(list)
            for slot, weight, edge in bucket:
                by_edge[id(edge)].append((slot, weight, edge))
            for group in by_edge.values():
                edge = group[0][2]
                slots = np.array([item[0] for item in group], dtype=np.int32)
                grouped_weights = np.array([item[1] for item in group], dtype=np.float64)
                add_direction("t1_e1", slots, grouped_weights, edge, reversed_traversal, None if edge is None else edge.get("key"))

    pending: dict[str, list] = defaultdict(list)
    for gene in genes_of.get(query, ()):
        g_term = _power(deg_gc.get(gene, 0), 0.8)
        if g_term == 0:
            continue
        query_edge = edges.get(("gene_cancer", gene, query))
        g_slot = index.get(gene)
        for cancer in cancers_of.get(gene, ()):
            if cancer == query:
                continue
            slots = candidate_slots.get(cancer)
            if slots is None or len(slots) == 0:
                continue
            degree = float(len(genes_of.get(cancer, ())))
            if degree <= 0:
                continue
            local = np.arange(len(slots)) if g_slot is None else np.flatnonzero(slots != g_slot)
            if len(local) == 0:
                continue
            keep = slots[local]
            weights = g_term * _power(degree, 0.8) * q_factor * slot_gc[keep]
            three_dwpc[keep] += weights
            three_paths[keep] += 1
            mid_key = ("gene_cancer", gene, cancer)
            mid = edges.get(mid_key)
            add_combo("t2_e2", keep, weights, mid, mid_key)
            add_direction("t2_e2", keep, weights, mid, True, mid_key)
            query_key = ("gene_cancer", gene, query)
            add_combo("t2_e3", keep, weights, query_edge, query_key)
            add_direction("t2_e3", keep, weights, query_edge, False, query_key)
            pending[cancer].append((keep, local, weights))

    seen_position1 = set()
    for cancer, items in pending.items():
        combo_matrix = matrices.get((cancer, "combo"))
        marginal_matrix = matrices.get((cancer, "marginal"))
        direction_matrix = matrices.get((cancer, "direction"))
        if touched is not None and cancer not in seen_position1:
            seen_position1.add(cancer)
            slots = candidate_slots.get(cancer)
            if slots is not None:
                for slot in slots.tolist():
                    touched["t2_e1"].add(("gene_cancer", reverse[int(slot)], cancer))
        if combo_matrix is None:
            continue
        for keep, local, weights in items:
            if combo_matrix is not None and np.any(combo_matrix):
                block = combo_matrix[local]
                present = np.flatnonzero(block.sum(axis=0))
                if len(present):
                    complete_dwpc["t2_e1"][keep] += weights * (block.sum(axis=1) > 0)
                    complete_paths["t2_e1"][keep] += (block.sum(axis=1) > 0).astype(np.int64)
                    for column in present:
                        combo["t2_e1"][keep, column] += weights * block[:, column]
            if marginal_matrix is not None and np.any(marginal_matrix):
                block = marginal_matrix[local]
                present = np.flatnonzero(block.sum(axis=0))
                for column in present:
                    marginal["t2_e1"][keep, column] += weights * block[:, column]
            if direction_matrix is not None and np.any(direction_matrix):
                block = direction_matrix[local]
                present = np.flatnonzero(block.sum(axis=0))
                if len(present):
                    directed_dwpc["t2_e1"][keep] += weights * (block.sum(axis=1) > 0)
                    directed_paths["t2_e1"][keep] += (block.sum(axis=1) > 0).astype(np.int64)
                    for column in present:
                        direction["t2_e1"][keep, column] += weights * block[:, column]
    return {
        "combo": combo,
        "marginal": marginal,
        "direction": direction,
        "complete_dwpc": complete_dwpc,
        "directed_dwpc": directed_dwpc,
        "complete_paths": complete_paths,
        "directed_paths": directed_paths,
        "two_edge_dwpc": two_dwpc,
        "three_edge_dwpc": three_dwpc,
        "two_edge_paths": two_paths,
        "three_edge_paths": three_paths,
        "reverse": reverse,
    }


def shares_from_mass(mass: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide accumulated mass by the observable DWPC. Missing stays NaN."""
    shares = np.full(mass.shape, np.nan, dtype=np.float64)
    present = denominator > 0
    if mass.ndim == 1:
        shares[present] = mass[present] / denominator[present]
        return shares
    shares[present] = mass[present] / denominator[present, None]
    return shares


def judge_representation(payload: dict) -> dict:
    """Recommend whether a later fixed-path refit is supported. This is not a gain estimate."""
    reasons = []
    queries = int(payload["queries_with_collision"])
    candidates = int(payload["candidate_pairs_in_collision"])
    targets = int(payload["validation_targets_in_collision"])
    top3 = float(payload["top3_query_share_of_colliding_pairs"])
    position = float(payload["largest_position_share_of_colliding_rows"])
    encoding = float(payload["encoding_sensitivity_retention"])
    quantization = float(payload["quantization_sensitivity_retention"])
    spread = queries >= 10 and top3 < 0.5 and position < 0.8
    covered = targets >= 10
    batch = candidates >= 100
    stable = encoding >= 0.5 and quantization >= 0.5
    if spread and covered and batch and stable:
        decision = "compare_in_a_fixed_path_refit"
        reasons.append("The finer encoding separates candidates in many development queries, including hidden validation targets, and the separation remains after the encoding and quantization checks.")
    elif candidates >= 20 and (queries < 10 or top3 >= 0.5 or position >= 0.8):
        decision = "scoped_comparison_only"
        reasons.append("The separation is visible, and it is concentrated in a small set of queries or in one edge position.")
    else:
        decision = "full_refit_not_supported"
        reasons.append("The remaining separation is too small, too concentrated, or mostly produced by coverage, symmetric argument order, or quantization.")
    if encoding < 0.5:
        reasons.append("Collapsing explicit unknown tokens, or restricting to asymmetric predicates, removes most of the separation.")
    if quantization < 0.5:
        reasons.append("A coarser quantization removes most of the separation.")
    return {"decision": decision, "reasons": reasons}

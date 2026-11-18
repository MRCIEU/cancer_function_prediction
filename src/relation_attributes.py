"""Attribute-matched features for masked gene–cancer ranking.

Path weights use two named scores. ``prior_degree_weight`` is the earlier
development score: a two-hop path contributes one over the intermediate
gene's candidate-neighbor count, and a three-hop path contributes one over
the number of genes on the intermediate cancer. ``dwpc`` is the standard
degree-weighted path count with damping 0.4. A path weight is the product,
over its edges, of each endpoint degree raised to -0.4. Degree is counted
on the masked graph for that edge's type. Query degree is constant within
a cancer and does not change within-cancer ranks; it is retained so the
formula matches the edge product.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.association_recovery import _stable_rank, connected_pmid_groups
from src.recovery_scope import evenly_spaced

DAMPING = 0.4
GG_PREDICATES = (
    "INTERACTS_WITH",
    "STIMULATES",
    "INHIBITS",
    "AFFECTS",
    "DISRUPTS",
    "AUGMENTS",
    "PRODUCES",
    "COEXISTS_WITH",
    "ASSOCIATED_WITH",
    "CONVERTS_TO",
    "compared_with",
    "higher_than",
    "lower_than",
)
GC_PREDICATES = (
    "AFFECTS",
    "ASSOCIATED_WITH",
    "CAUSES",
    "PREDISPOSES",
    "PREVENTS",
    "DISRUPTS",
    "AUGMENTS",
    "STIMULATES",
    "INHIBITS",
    "PRODUCES",
    "COEXISTS_WITH",
    "compared_with",
    "higher_than",
    "lower_than",
)
CORE_LEVELS = {
    "cge": ("up", "down", "uninformative"),
    "ccs": ("progression", "regression", "uninformative"),
    "gci": ("causality", "observation", "uninformative"),
}
CORE_MAP = {
    "cge": {"UP": "up", "DOWN": "down", "NOTINF": "uninformative", "": "uninformative"},
    "ccs": {"PROGRESSION": "progression", "REGRESSION": "regression", "NOTINF": "uninformative", "": "uninformative"},
    "gci": {"CAUSALITY": "causality", "OBSERVATION": "observation", "NOTINF": "uninformative", "": "uninformative"},
}
GC_POSITIONS = ("t1_e2", "t2_e1", "t2_e2", "t2_e3")
GG_POSITIONS = ("t1_e1",)
SEEDS = (20261116, 20261117, 20261118)
COHORT_SEED = 20261119
CONTROL_RATIO = 10
COHORT_LADDER = ((100, 40), (120, 50), (140, 60))


def _power(degree: float, exponent: float) -> float:
    if degree <= 0:
        return 0.0
    return float(degree) ** (-exponent)


def feature_names() -> dict[str, list[str]]:
    baseline = [
        "two_edge_paths",
        "three_edge_paths",
        "three_edge_core_paths",
        "three_edge_semmed_paths",
        "two_edge_prior_weight",
        "three_edge_prior_weight",
        "two_edge_dwpc",
        "three_edge_dwpc",
        "gene_gg_degree",
        "gene_gc_degree",
        "gene_publications",
        "has_two_edge",
        "has_three_edge",
        "t1_shared_pmid",
        "t2_e12_shared_pmid",
        "t2_e23_shared_pmid",
        "t1_pmid_count",
        "t2_pmid_count",
        "t1_age",
        "t2_age",
    ]
    for position in GG_POSITIONS + GC_POSITIONS:
        baseline.append(f"{position}_predicate_observed")
    for position in GC_POSITIONS:
        for field in CORE_LEVELS:
            baseline.append(f"{position}_{field}_observed")
    core = []
    for position in GC_POSITIONS:
        for field, levels in CORE_LEVELS.items():
            for level in levels:
                core.append(f"{position}_{field}_{level}")
    predicates = []
    for position in GG_POSITIONS:
        for name in GG_PREDICATES:
            predicates.append(f"{position}_pred_{name}")
    for position in GC_POSITIONS:
        for name in GC_PREDICATES:
            predicates.append(f"{position}_pred_{name}")
    return {
        "baseline": baseline,
        "core_attributes": baseline + core,
        "semmed_predicates": baseline + predicates,
        "both": baseline + core + predicates,
        "core_only": core,
        "predicate_only": predicates,
        "log1p": [
            "two_edge_paths",
            "three_edge_paths",
            "three_edge_core_paths",
            "three_edge_semmed_paths",
            "two_edge_prior_weight",
            "three_edge_prior_weight",
            "two_edge_dwpc",
            "three_edge_dwpc",
            "gene_gg_degree",
            "gene_gc_degree",
            "gene_publications",
            "t1_pmid_count",
            "t2_pmid_count",
        ],
    }


def normalise_core(field: str, label: str) -> str:
    mapping = CORE_MAP[field]
    token = mapping.get(label if label is not None else "", "other")
    return token


def select_cancers(rows: list[dict], count: int) -> list[dict]:
    """Spread cancers across PMID-abundance quartiles and path coverage.

    A row needs ``cancer``, ``n_pmid``, and ``paths``. The draw is
    deterministic. Cancers with fewer than 20 PMID-backed positives or
    fewer than 100,000 three-hop paths are outside the planning pool.
    """
    pool = [row for row in rows if int(row["n_pmid"]) >= 20 and int(row["paths"]) >= 100_000]
    pool.sort(key=lambda row: (int(row["n_pmid"]), int(row["paths"]), row["cancer"]))
    if len(pool) < count:
        pool = [row for row in rows if int(row["n_pmid"]) >= 16]
        pool.sort(key=lambda row: (int(row["n_pmid"]), int(row["paths"]), row["cancer"]))
    if not pool:
        return []
    quartiles = [[] for _ in range(4)]
    for index, row in enumerate(pool):
        quartiles[min(3, index * 4 // len(pool))].append(row)
    chosen = []
    used = set()
    per_band = max(1, count // 4)
    for band in quartiles:
        band.sort(key=lambda row: (int(row["paths"]), row["cancer"]))
        for row in evenly_spaced(band, min(per_band, len(band))):
            if row["cancer"] not in used:
                chosen.append(row)
                used.add(row["cancer"])
    if len(chosen) < count:
        rest = [row for row in pool if row["cancer"] not in used]
        rest.sort(key=lambda row: (_stable_rank(COHORT_SEED, row["cancer"]), row["cancer"]))
        for row in rest:
            if len(chosen) >= count:
                break
            chosen.append(row)
    return chosen[:count]


def sample_positives(pairs: list[dict], n_take: int, seed: int) -> list[dict]:
    """Stratify on the SemMedDB direct-record flag. Pairs need ``target_id``."""
    if n_take >= len(pairs):
        return list(pairs)
    shared = [row for row in pairs if row["semmed_direct"]]
    only = [row for row in pairs if not row["semmed_direct"]]
    raw = n_take * len(shared) / len(pairs) if pairs else 0
    n_shared = int(np.floor(raw + 0.5))
    n_shared = min(n_shared, len(shared), n_take)
    n_only = min(n_take - n_shared, len(only))

    def take(group: list[dict], size: int) -> list[dict]:
        ranked = sorted(group, key=lambda row: (_stable_rank(seed, row["target_id"]), row["target_id"]))
        return ranked[:size]

    chosen = take(shared, n_shared) + take(only, n_only)
    return chosen


def assign_development_roles(records: list[dict], blocked_pmids: set[int]) -> dict[str, str]:
    """Put PMID groups that touch the previous development run into train.

    Remaining groups fill train, tune, and development validation toward
    60/20/20 of the whole selected cohort.
    """
    groups = connected_pmid_groups(records)
    blocked = []
    free = []
    for group in groups:
        pmids = {int(pmid) for record in group for pmid in record["pmids"]}
        if pmids & blocked_pmids:
            blocked.append(group)
        else:
            free.append(group)
    assignment = {}
    counts = {"train": 0, "tune": 0, "development_validation": 0}
    for group in blocked:
        for record in group:
            assignment[record["target_id"]] = "train"
            counts["train"] += 1
    total = max(len(records), 1)
    fractions = {"train": 0.6, "tune": 0.2, "development_validation": 0.2}
    free.sort(key=lambda group: (-len(group), min(record["target_id"] for record in group)))
    names = ("train", "tune", "development_validation")
    for group in free:
        choice = max(names, key=lambda name: (fractions[name] - counts[name] / total, name == "development_validation", name))
        for record in group:
            assignment[record["target_id"]] = choice
            counts[choice] += 1
    return assignment


def build_edge_attributes(connections: dict, masked_pmids: set[int]) -> dict[tuple[str, str, str], dict]:
    """Distinct surviving field values on each edge.

    A blocked PMID removes every attached record. A record with no PMID is
    kept. Negated SemMedDB predicates are counted and omitted. Predicate
    orientation is collapsed to the base predicate.
    """
    prepared = {}
    negated = 0
    for key, bucket in connections.items():
        core_values = {field: set() for field in CORE_LEVELS}
        core_present = False
        predicates = set()
        for pmid, cge, ccs, gci, gcc, has_type in bucket["core"].values():
            number = int(pmid) if str(pmid).isdigit() else 0
            if number and number in masked_pmids:
                continue
            core_present = True
            labels = {"cge": cge, "ccs": ccs, "gci": gci}
            for field, label in labels.items():
                core_values[field].add(normalise_core(field, label or ""))
        for pmid, predicate, _subject, _object, is_negated, is_traversal in bucket["sem"].values():
            number = int(pmid) if str(pmid).isdigit() else 0
            if number and number in masked_pmids:
                continue
            if is_negated:
                negated += 1
                continue
            if not is_traversal:
                continue
            family = key[0]
            allowed = GG_PREDICATES if family == "gene_gene" else GC_PREDICATES
            if predicate in allowed:
                predicates.add(predicate)
        if core_present or predicates:
            prepared[key] = {"core": core_values if core_present else None, "predicates": predicates, "negated_excluded": negated}
    return prepared


def _split_add(store: dict[str, np.ndarray], observed: np.ndarray, slot: int, weight: float, values: set[str], prefix: str) -> None:
    if weight == 0 or not values:
        return
    share = weight / len(values)
    observed[slot] += weight
    for value in values:
        column = store.get(f"{prefix}_{value}")
        if column is not None:
            column[slot] += share


def position1_tables(
    attributes: dict[tuple[str, str, str], dict],
    edge_pmids: dict[tuple[str, str, str], tuple[int, ...]],
    index: dict[str, int],
    candidate_slots: dict[str, np.ndarray],
) -> dict[str, dict]:
    """Share vectors for the candidate–cancer edge, aligned with each cancer's slots."""
    reverse = {slot: gene for gene, slot in index.items()}
    pred_names = [f"t2_e1_pred_{name}" for name in GC_PREDICATES]
    core_names = [f"t2_e1_{field}_{level}" for field, levels in CORE_LEVELS.items() for level in levels]
    tables = {}
    for cancer, slots in candidate_slots.items():
        width = len(slots)
        n_pmid = np.zeros(width, dtype=np.float64)
        pmids = []
        fractions = {name: np.zeros(width, dtype=np.float64) for name in pred_names + core_names}
        core_obs = {field: np.zeros(width, dtype=np.float64) for field in CORE_LEVELS}
        pred_obs = np.zeros(width, dtype=np.float64)
        for local, slot in enumerate(slots.tolist()):
            gene = reverse[int(slot)]
            key = ("gene_cancer", gene, cancer)
            pmid_values = edge_pmids.get(key, ())
            n_pmid[local] = float(len(pmid_values))
            pmids.append(set(pmid_values))
            attr = attributes.get(key)
            if not attr:
                continue
            if attr["predicates"]:
                pred_obs[local] = 1.0
                share = 1.0 / len(attr["predicates"])
                for predicate in attr["predicates"]:
                    fractions[f"t2_e1_pred_{predicate}"][local] = share
            if attr["core"] is not None:
                for field, values in attr["core"].items():
                    if not values or field not in core_obs:
                        continue
                    core_obs[field][local] = 1.0
                    share = 1.0 / len(values)
                    for value in values:
                        name = f"t2_e1_{field}_{value}"
                        if name in fractions:
                            fractions[name][local] = share
        tables[cancer] = {
            "n_pmid": n_pmid,
            "pmids": pmids,
            "fractions": fractions,
            "core_obs": core_obs,
            "pred_obs": pred_obs,
        }
    return tables


def score_query(
    query: str,
    genes_of: dict[str, set[str]],
    neighbors: dict[str, set[str]],
    cancers_of: dict[str, set[str]],
    index: dict[str, int],
    candidate_slots: dict[str, np.ndarray],
    deg_gg: dict[str, int],
    deg_gc: dict[str, int],
    edge_pmids: dict[tuple[str, str, str], tuple[int, ...]],
    ages: dict[tuple[str, str, str], float | None],
    attributes: dict[tuple[str, str, str], dict],
    publications: np.ndarray,
    tables: dict[str, dict] | None = None,
) -> dict[str, np.ndarray]:
    """Structural scores and attribute contributions for one query."""
    names = feature_names()
    n_gene = len(index)
    columns = {name: np.zeros(n_gene, dtype=np.float64) for name in names["both"]}
    obs_weight = {f"{position}_{field}": np.zeros(n_gene) for position in GC_POSITIONS for field in CORE_LEVELS}
    obs_weight.update({f"{position}_predicate": np.zeros(n_gene) for position in GG_POSITIONS + GC_POSITIONS})
    obs_weight["t1_age_w"] = np.zeros(n_gene)
    obs_weight["t2_age_w"] = np.zeros(n_gene)
    dwpc_mass = {"t1": np.zeros(n_gene), "t2": np.zeros(n_gene)}
    q_factor = _power(len(genes_of.get(query, ())), DAMPING)
    slot_gc = np.array([_power(deg_gc.get(gene, 0), DAMPING) for gene in index], dtype=np.float64)
    # ``index`` insertion order is not slot order. Rebuild by slot.
    slot_gc = np.zeros(n_gene)
    slot_gg = np.zeros(n_gene)
    for gene, slot in index.items():
        slot_gc[slot] = _power(deg_gc.get(gene, 0), DAMPING)
        slot_gg[slot] = _power(deg_gg.get(gene, 0), DAMPING)
    gg_degree = np.zeros(n_gene)
    gc_degree = np.zeros(n_gene)
    for gene, slot in index.items():
        gg_degree[slot] = float(deg_gg.get(gene, 0))
        gc_degree[slot] = float(deg_gc.get(gene, 0))
    if tables is None:
        tables = position1_tables(attributes, edge_pmids, index, candidate_slots)

    def age_of(key: tuple[str, str, str]) -> float | None:
        return ages.get(key)

    for gene in genes_of.get(query, ()):
        partners = [item for item in neighbors.get(gene, ()) if item in index and item != gene]
        if not partners:
            continue
        prior = 1.0 / len(partners)
        base = _power(deg_gg.get(gene, 0), DAMPING) * _power(deg_gc.get(gene, 0), DAMPING) * q_factor
        right_key = ("gene_cancer", gene, query)
        right_attr = attributes.get(right_key)
        right_pmids = set(edge_pmids.get(right_key, ()))
        right_n = float(len(right_pmids))
        right_age = age_of(right_key)
        for partner in partners:
            slot = index[partner]
            weight = base * slot_gg[slot]
            columns["two_edge_paths"][slot] += 1.0
            columns["two_edge_prior_weight"][slot] += prior
            columns["two_edge_dwpc"][slot] += weight
            dwpc_mass["t1"][slot] += weight
            ordered = (partner, gene) if partner <= gene else (gene, partner)
            left_key = ("gene_gene", ordered[0], ordered[1])
            left_pmids = set(edge_pmids.get(left_key, ()))
            columns["t1_pmid_count"][slot] += weight * (len(left_pmids) + right_n)
            if left_pmids & right_pmids:
                columns["t1_shared_pmid"][slot] += weight
            known_ages = [item for item in (age_of(left_key), right_age) if item is not None]
            if known_ages:
                columns["t1_age"][slot] += weight * float(sum(known_ages) / len(known_ages))
                obs_weight["t1_age_w"][slot] += weight
            left_attr = attributes.get(left_key)
            if left_attr and left_attr["predicates"]:
                _split_add(columns, obs_weight["t1_e1_predicate"], slot, weight, left_attr["predicates"], "t1_e1_pred")
            if right_attr and right_attr["predicates"]:
                _split_add(columns, obs_weight["t1_e2_predicate"], slot, weight, right_attr["predicates"], "t1_e2_pred")
            if right_attr and right_attr["core"] is not None:
                for field, values in right_attr["core"].items():
                    _split_add(columns, obs_weight[f"t1_e2_{field}"], slot, weight, values, f"t1_e2_{field}")

    for gene in genes_of.get(query, ()):
        g_term = _power(deg_gc.get(gene, 0), 0.8)
        if g_term == 0:
            continue
        query_key = ("gene_cancer", gene, query)
        query_attr = attributes.get(query_key)
        query_pmids = set(edge_pmids.get(query_key, ()))
        query_n = float(len(query_pmids))
        query_age = age_of(query_key)
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
            base = g_term * _power(degree, 0.8) * q_factor
            weights = base * slot_gc[keep]
            columns["three_edge_paths"][keep] += 1.0
            columns["three_edge_prior_weight"][keep] += 1.0 / degree
            columns["three_edge_dwpc"][keep] += weights
            dwpc_mass["t2"][keep] += weights
            mid_key = ("gene_cancer", gene, cancer)
            mid_attr = attributes.get(mid_key)
            mid_pmids = set(edge_pmids.get(mid_key, ()))
            mid_n = float(len(mid_pmids))
            if query_pmids & mid_pmids:
                columns["t2_e23_shared_pmid"][keep] += weights
            known = [item for item in (query_age, age_of(mid_key)) if item is not None]
            if known:
                columns["t2_age"][keep] += weights * (float(sum(known)) / len(known))
                obs_weight["t2_age_w"][keep] += weights
            if query_attr and query_attr["predicates"]:
                share = weights / len(query_attr["predicates"])
                obs_weight["t2_e3_predicate"][keep] += weights
                for predicate in query_attr["predicates"]:
                    columns[f"t2_e3_pred_{predicate}"][keep] += share
            if mid_attr and mid_attr["predicates"]:
                share = weights / len(mid_attr["predicates"])
                obs_weight["t2_e2_predicate"][keep] += weights
                for predicate in mid_attr["predicates"]:
                    columns[f"t2_e2_pred_{predicate}"][keep] += share
            if query_attr and query_attr["core"] is not None:
                for field, values in query_attr["core"].items():
                    if not values:
                        continue
                    share = weights / len(values)
                    obs_weight[f"t2_e3_{field}"][keep] += weights
                    for value in values:
                        name = f"t2_e3_{field}_{value}"
                        if name in columns:
                            columns[name][keep] += share
            if mid_attr and mid_attr["core"] is not None:
                for field, values in mid_attr["core"].items():
                    if not values:
                        continue
                    share = weights / len(values)
                    obs_weight[f"t2_e2_{field}"][keep] += weights
                    for value in values:
                        name = f"t2_e2_{field}_{value}"
                        if name in columns:
                            columns[name][keep] += share
            columns.setdefault("_pos1_cancer", {})
            columns["_pos1_cancer"].setdefault(cancer, []).append((weights, local, keep, mid_n, mid_pmids, query_n))

    _finish_position1(columns, obs_weight, tables)
    _finalize_shares(columns, obs_weight, dwpc_mass, publications, gg_degree, gc_degree)
    columns.pop("_pos1_cancer", None)
    return columns


def _finish_position1(columns, obs_weight, tables: dict[str, dict]) -> None:
    bundles = columns.get("_pos1_cancer", {})
    for cancer, items in bundles.items():
        table = tables.get(cancer)
        if table is None:
            continue
        for weights, local, keep, mid_n, mid_pmids, query_n in items:
            n_pmid = table["n_pmid"][local]
            columns["t2_pmid_count"][keep] += weights * (n_pmid + mid_n + query_n)
            if mid_pmids:
                shared = np.fromiter(
                    (1.0 if table["pmids"][int(item)] & mid_pmids else 0.0 for item in local),
                    dtype=np.float64,
                    count=len(local),
                )
                columns["t2_e12_shared_pmid"][keep] += weights * shared
            pred_obs = table["pred_obs"][local]
            if np.any(pred_obs):
                obs_weight["t2_e1_predicate"][keep] += weights * pred_obs
            for field, mask in table["core_obs"].items():
                observed = mask[local]
                if np.any(observed):
                    obs_weight[f"t2_e1_{field}"][keep] += weights * observed
            for name, fractions in table["fractions"].items():
                piece = fractions[local]
                if np.any(piece):
                    columns[name][keep] += weights * piece


def _finalize_shares(columns, obs_weight, dwpc_mass, publications, deg_gg, deg_gc_gene) -> None:
    columns["gene_gg_degree"][:] = deg_gg
    columns["gene_gc_degree"][:] = deg_gc_gene
    columns["gene_publications"][:] = publications
    columns["has_two_edge"] = (columns["two_edge_paths"] > 0).astype(np.float64)
    columns["has_three_edge"] = (columns["three_edge_paths"] > 0).astype(np.float64)
    for name, mass in (("t1_shared_pmid", dwpc_mass["t1"]), ("t2_e12_shared_pmid", dwpc_mass["t2"]), ("t2_e23_shared_pmid", dwpc_mass["t2"])):
        share = np.full(mass.shape, np.nan)
        present = mass > 0
        share[present] = columns[name][present] / mass[present]
        columns[name] = share
    for name, mass_name in (("t1_pmid_count", "t1"), ("t2_pmid_count", "t2")):
        mass = dwpc_mass[mass_name]
        present = mass > 0
        values = np.full(mass.shape, np.nan)
        values[present] = columns[name][present] / mass[present]
        columns[name] = values
    for name, key in (("t1_age", "t1_age_w"), ("t2_age", "t2_age_w")):
        mass = obs_weight.get(key)
        values = np.full(columns["two_edge_paths"].shape, np.nan)
        if mass is not None:
            present = mass > 0
            values[present] = columns[name][present] / mass[present]
        columns[name] = values
    names = feature_names()
    for feature in names["core_only"] + names["predicate_only"]:
        if feature.startswith("t1_e1_"):
            mass_key = "t1_e1_predicate"
        elif "_pred_" in feature:
            mass_key = feature.split("_pred_")[0] + "_predicate"
        else:
            position, field, _level = feature.rsplit("_", 2)
            mass_key = f"{position}_{field}"
        mass = obs_weight[mass_key]
        present = mass > 0
        values = np.full(mass.shape, np.nan)
        values[present] = columns[feature][present] / mass[present]
        columns[feature] = values
    for position in GG_POSITIONS + GC_POSITIONS:
        mass = obs_weight[f"{position}_predicate"]
        total = dwpc_mass["t1"] if position.startswith("t1") else dwpc_mass["t2"]
        values = np.zeros(total.shape)
        present = total > 0
        values[present] = mass[present] / total[present]
        columns[f"{position}_predicate_observed"] = values
    for position in GC_POSITIONS:
        total = dwpc_mass["t1"] if position.startswith("t1") else dwpc_mass["t2"]
        for field in CORE_LEVELS:
            mass = obs_weight[f"{position}_{field}"]
            values = np.zeros(total.shape)
            present = total > 0
            values[present] = mass[present] / total[present]
            columns[f"{position}_{field}_observed"] = values


def apply_log1p(matrix: np.ndarray, names: list[str]) -> np.ndarray:
    logged = np.array(matrix, copy=True)
    wanted = set(feature_names()["log1p"])
    for index, name in enumerate(names):
        if name in wanted:
            column = logged[:, index]
            present = ~np.isnan(column)
            column[present] = np.log1p(np.maximum(column[present], 0))
            logged[:, index] = column
    return logged


def paired_interval(deltas: np.ndarray, draws: int = 2000, seed: int = 20261119) -> dict:
    """Percentile interval for the mean of per-cancer paired differences."""
    if len(deltas) == 0:
        return {"mean": None, "low": None, "high": None, "draws": 0}
    generator = np.random.default_rng(seed)
    means = np.empty(draws)
    for index in range(draws):
        sample = generator.choice(deltas, size=len(deltas), replace=True)
        means[index] = float(sample.mean())
    return {
        "mean": float(deltas.mean()),
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
        "draws": draws,
        "cancers": int(len(deltas)),
    }


def fuse_log1p(two_edge: np.ndarray, three_edge: np.ndarray) -> np.ndarray:
    """Earlier fixed fusion: sum of log1p template scores."""
    return np.log1p(np.maximum(two_edge, 0)) + np.log1p(np.maximum(three_edge, 0))


def share_conservation(store: dict[str, np.ndarray], atol: float = 1e-3) -> dict:
    """Where a field is observed, its value shares sum to one."""
    groups: dict[str, list[str]] = {}
    names = feature_names()
    for feature in names["core_only"] + names["predicate_only"]:
        if "_pred_" in feature:
            parent = feature.split("_pred_")[0] + "_predicate"
        else:
            parent = feature.rsplit("_", 1)[0]
        groups.setdefault(parent, []).append(feature)
    worst = 0.0
    checked = 0
    for parent, features in groups.items():
        observed = store.get(f"{parent}_observed")
        if observed is None:
            continue
        total = np.zeros(observed.shape, dtype=np.float64)
        for feature in features:
            total += np.nan_to_num(store[feature], nan=0.0)
        mask = observed > 0
        if not np.any(mask):
            continue
        gap = float(np.max(np.abs(total[mask] - 1.0)))
        worst = max(worst, gap)
        checked += int(np.sum(mask))
        if gap > atol:
            raise RuntimeError(f"{parent} shares do not conserve observable weight (gap {gap})")
    return {"groups": len(groups), "observed_cells": checked, "max_gap": worst}

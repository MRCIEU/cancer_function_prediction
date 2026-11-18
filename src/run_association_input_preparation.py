"""Build frozen development inputs for gene–cancer ranking.

The run writes evaluation lists, scoped relation columns, citation columns,
and model packages. It does not fit a model, score development candidates,
or open a formal test file.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.common import RESULTS_ROOT, peak_rss_bytes, sha256_file, sha256_text, write_json
from src.input_preparation import (
    ARTICLE_PAIRS_NOT_ENUMERATED,
    CITATION_CONFIRMED_ZERO,
    CITATION_NO_PATH,
    CITATION_OBSERVED_LINK,
    CITATION_UNKNOWN,
    MISSING_NO_PATH,
    MISSING_OBSERVED,
    MISSING_PATH_WITHOUT_RECORD,
    PATH_BUDGET,
    ROLES,
    asymmetric_predicates,
    citation_motif,
    classify_list_membership,
    combo_column_names,
    direction_column_names,
    marginal_column_names,
    update_topk,
)
from src.relation_attributes import (
    DAMPING,
    GC_PREDICATES,
    GG_PREDICATES,
    _power,
    feature_names,
)
from src.relation_representation import (
    SYMMETRIC_PREDICATES,
    accumulate_fine,
    collision_summary,
    in_multi_group,
    marginal_from_joint,
    quant_signature,
    shares_from_mass,
    undirected_from_directions,
)
from src.recovery_scope import build_motif_block, candidate_slot_table, sum_adjacent_motifs
from src.run_association_recovery import (
    CITATION_AUDIT,
    build_graphs,
    load_citation_index,
    load_membership,
    load_profiles,
    load_universe,
)
from src.run_relation_representation import _matrices, _stream, _views

SOURCE_RUN = RESULTS_ROOT / "relation_attributes_development_20261117T1544Z"
PREFLIGHT_RUN = RESULTS_ROOT / "association_recovery_preflight_20261117T2053Z"
AUDIT_RUN = RESULTS_ROOT / "relation_representation_audit_20261117T1917Z"
MOTIF_NAMES = (
    "direct_collapsed",
    "direct_ltr",
    "direct_rtl",
    "direct_ambiguous",
    "cocite_witnesses",
    "shared_reference_witnesses",
)
CITATION_POSITIONS = ("t1", "t2_12", "t2_23")


def _tsv(path: Path, rows: list[dict], columns: list[str]) -> None:
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join("" if row.get(column) is None else str(row[column]) for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _peak_gb() -> float:
    return peak_rss_bytes() / (1024 ** 3)


def _load_targets() -> list[dict]:
    table = pq.read_table(
        SOURCE_RUN / "analysis" / "development_targets.parquet",
        columns=["gene_id", "cancer_cui", "role", "pmids"],
    )
    rows = []
    for gene, cancer, role, pmids in zip(
        table.column("gene_id").to_pylist(),
        table.column("cancer_cui").to_pylist(),
        table.column("role").to_pylist(),
        table.column("pmids").to_pylist(),
    ):
        rows.append({"gene_id": str(gene), "cancer_cui": str(cancer), "role": str(role), "pmids": [int(item) for item in pmids]})
    return rows


def _identity(records: list[dict], gene_ids: list[str]) -> dict:
    roles = Counter(row["role"] for row in records)
    pairs = sorted((row["gene_id"], row["cancer_cui"], row["role"]) for row in records)
    store = SOURCE_RUN / "analysis" / "feature_store.npz"
    targets = SOURCE_RUN / "analysis" / "development_targets.parquet"
    return {
        "role_counts": dict(roles),
        "role_counts_match": dict(roles) == {"train": 3124, "tune": 1041, "development_validation": 1041},
        "validation_cancers": len({row["cancer_cui"] for row in records if row["role"] == "development_validation"}),
        "pair_sha256": sha256_text("\n".join(f"{gene}|{cancer}|{role}" for gene, cancer, role in pairs)),
        "development_targets_sha256": sha256_file(targets),
        "feature_store_sha256": sha256_file(store),
        "gene_count": len(gene_ids),
        "gene_universe_sha256": sha256_text("\n".join(gene_ids)),
        "preflight_training_rows": str(PREFLIGHT_RUN / "analysis" / "training_rows.parquet"),
        "preflight_training_rows_sha256": sha256_file(PREFLIGHT_RUN / "analysis" / "training_rows.parquet"),
    }


def _write_lists(run_dir: Path, records: list[dict], gene_ids: list[str], cancers: list[str], graph: dict) -> dict:
    index = {gene: slot for slot, gene in enumerate(gene_ids)}
    cancer_row = {cancer: row for row, cancer in enumerate(cancers)}
    n_gene = len(gene_ids)
    primary = np.ones((len(cancers), n_gene), dtype=bool)
    historical = {role: np.ones((len(cancers), n_gene), dtype=bool) for role in ROLES}
    targets = defaultdict(dict)
    for row in records:
        targets[row["cancer_cui"]][row["gene_id"]] = row["role"]
    leaked = []
    difference_rows = []
    for cancer in cancers:
        row = cancer_row[cancer]
        direct = set(graph["direct"].get(cancer, ()))
        visible = set(graph["genes_of"].get(cancer, ()))
        role_of = targets.get(cancer, {})
        counts = Counter()
        for gene, slot in index.items():
            role = role_of.get(gene)
            is_direct = gene in direct
            is_visible = gene in visible and is_direct
            if role is not None and gene in visible:
                leaked.append(f"{gene}|{cancer}")
                continue
            membership = classify_list_membership(is_direct, is_visible, role)
            primary[row, slot] = membership["in_primary"]
            for partition, keep in membership["historical"].items():
                historical[partition][row, slot] = keep
            counts[membership["reason"]] += 1
        hidden = {role: 0 for role in ROLES}
        for gene, role in role_of.items():
            hidden[role] += 1
            if not primary[row, index[gene]] or not historical[role][row, index[gene]]:
                leaked.append(f"dropped-target:{gene}|{cancer}|{role}")
        difference_rows.append({
            "cancer_cui": cancer,
            "full_candidates": n_gene,
            "primary_candidates": int(primary[row].sum()),
            "historical_validation_candidates": int(historical["development_validation"][row].sum()),
            "historical_only_removed": int((primary[row] & ~historical["development_validation"][row]).sum()),
            "primary_only_removed": int((~primary[row] & historical["development_validation"][row]).sum()),
            "visible_directs": counts["visible_direct_after_mask"],
            "unmasked_directs_removed_by_mask": counts["unmasked_direct_removed_by_mask"],
            "train_targets": hidden["train"],
            "tune_targets": hidden["tune"],
            "validation_targets": hidden["development_validation"],
        })
    if leaked:
        raise RuntimeError("evaluation list dropped or exposed a hidden target: " + ",".join(leaked[:8]))
    np.savez(
        run_dir / "analysis" / "evaluation_masks.npz",
        cancers=np.asarray(cancers),
        gene_ids=np.asarray(gene_ids),
        primary=primary,
        historical_train=historical["train"],
        historical_tune=historical["tune"],
        historical_validation=historical["development_validation"],
    )
    _tsv(run_dir / "analysis" / "evaluation_list_differences.tsv", difference_rows, list(difference_rows[0]))
    return {
        "hidden_targets_retained": True,
        "leaked_visible_targets": 0,
        "mean_primary_candidates": float(np.mean([row["primary_candidates"] for row in difference_rows])),
        "mean_historical_validation_candidates": float(np.mean([row["historical_validation_candidates"] for row in difference_rows])),
        "cancers_with_list_difference": int(sum(row["historical_only_removed"] > 0 for row in difference_rows)),
        "direct_source": "Unmasked joint-graph edge with at least one CORE sentence or one SemMedDB traversal. The label table is not the source of the exclusion.",
    }


def _write_adjacency(run_dir: Path, graph: dict, gene_ids: list[str]) -> dict:
    index = {gene: slot for slot, gene in enumerate(gene_ids)}
    indptr = [0]
    indices = []
    traced = None
    for slot, gene in enumerate(gene_ids):
        destinations = sorted(index[item] for item in graph["neighbors"].get(gene, ()) if item in index and item != gene)
        indices.extend(destinations)
        indptr.append(len(indices))
        if traced is None and destinations:
            other = gene_ids[destinations[0]]
            ordered = (gene, other) if gene <= other else (other, gene)
            key = ("gene_gene", ordered[0], ordered[1])
            traced = {"gene": gene, "neighbor": other, "edge_key": list(key), "in_masked_edge_pmids": key in graph["edge_pmids"]}
    cancer_ids = sorted(graph["genes_of"])
    cancer_index = {cancer: slot for slot, cancer in enumerate(cancer_ids)}
    gc_ptr = [0]
    gc_idx = []
    for gene in gene_ids:
        linked = sorted(cancer_index[cancer] for cancer, genes in graph["genes_of"].items() if gene in genes)
        gc_idx.extend(linked)
        gc_ptr.append(len(gc_idx))
    np.savez(
        run_dir / "analysis" / "walk_adjacency.npz",
        gene_ids=np.asarray(gene_ids),
        gene_gene_indptr=np.asarray(indptr, dtype=np.int64),
        gene_gene_indices=np.asarray(indices, dtype=np.int32),
        cancer_ids=np.asarray(cancer_ids),
        gene_cancer_indptr=np.asarray(gc_ptr, dtype=np.int64),
        gene_cancer_indices=np.asarray(gc_idx, dtype=np.int32),
    )
    return {"traced_adjacency": traced, "gene_gene_edges_directed": len(indices), "gene_cancer_links": len(gc_idx), "cancers_in_graph": len(cancer_ids)}


def _numeric(run_dir: Path, graph: dict, views: dict, matrices: dict, gene_ids: list[str], cancers: list[str], records: list[dict], skip_collisions: bool = False) -> dict:
    index = {gene: slot for slot, gene in enumerate(gene_ids)}
    slots = candidate_slot_table(graph["genes_of"], index)
    cancers_of = defaultdict(set)
    deg_gc = defaultdict(int)
    for cancer, genes in graph["genes_of"].items():
        for gene in genes:
            cancers_of[gene].add(cancer)
            deg_gc[gene] += 1
    deg_gg = {gene: len(partners) for gene, partners in graph["neighbors"].items()}
    stored = np.load(SOURCE_RUN / "analysis" / "feature_store.npz")
    stored_index = {str(cancer): row for row, cancer in enumerate(stored["cancers"].tolist())}
    n_cancer = len(cancers)
    n_gene = len(gene_ids)
    direction = np.full((n_cancer, n_gene, len(GG_PREDICATES) * 3), np.nan, dtype=np.float32)
    asymmetric = np.full((n_cancer, n_gene, len(asymmetric_predicates(GG_PREDICATES)) * 3), np.nan, dtype=np.float32)
    combo = np.full((n_cancer, n_gene, 64), np.nan, dtype=np.float32)
    marginal = np.full((n_cancer, n_gene, 12), np.nan, dtype=np.float32)
    secondary = np.full((n_cancer, n_gene, 64), np.nan, dtype=np.float32)
    denominator = {name: np.zeros((n_cancer, n_gene), dtype=np.float32) for name in ("t1_e1_directed_dwpc", "t2_e1_complete_dwpc", "t1_e2_complete_dwpc", "two_edge_dwpc", "three_edge_dwpc")}
    state = {name: np.zeros((n_cancer, n_gene), dtype=np.uint8) for name in ("t1_e1_direction_state", "t2_e1_combo_state")}
    gaps = []
    traces = []
    collision_targets = {"t2_e1": 0, "t1_e1": 0}
    covered_targets = {"t2_e1": 0, "t1_e1": 0}
    hidden = defaultdict(dict)
    for row in records:
        hidden[row["cancer_cui"]][row["gene_id"]] = row["role"]
    asymmetric_index = [offset for offset, name in enumerate(GG_PREDICATES) if name not in SYMMETRIC_PREDICATES]
    for query_number, cancer in enumerate(cancers):
        fine = accumulate_fine(cancer, graph["genes_of"], graph["neighbors"], cancers_of, index, slots, deg_gg, deg_gc, views, matrices, None)
        row = stored_index[cancer]
        gap = max(
            float(np.max(np.abs(fine["two_edge_dwpc"].astype(np.float32) - stored["two_edge_dwpc"][row]))),
            float(np.max(np.abs(fine["three_edge_dwpc"].astype(np.float32) - stored["three_edge_dwpc"][row]))),
        )
        if gap > 1e-4:
            raise RuntimeError(f"DWPC diverged from the frozen feature store at {cancer}: {gap}")
        directed = shares_from_mass(fine["direction"]["t1_e1"], fine["directed_dwpc"]["t1_e1"])
        undirected = undirected_from_directions(np.nan_to_num(directed))
        predicate = np.vstack([stored[f"t1_e1_pred_{name}"][row] for name in GG_PREDICATES]).T
        observed = fine["directed_dwpc"]["t1_e1"] > 0
        if np.any(observed):
            restore = float(np.nanmax(np.abs(np.nan_to_num(undirected[observed]) - np.nan_to_num(predicate[observed]))))
        else:
            restore = 0.0
        joint = shares_from_mass(fine["combo"]["t2_e1"], fine["complete_dwpc"]["t2_e1"])
        margin = shares_from_mass(fine["marginal"]["t2_e1"], fine["complete_dwpc"]["t2_e1"])
        complete = fine["complete_dwpc"]["t2_e1"] > 0
        projection = 0.0
        if np.any(complete):
            projected = np.vstack([marginal_from_joint(np.nan_to_num(joint[slot])) for slot in np.flatnonzero(complete)])
            projection = float(np.max(np.abs(projected - np.nan_to_num(margin[complete]))))
        if restore > 1e-4 or projection > 1e-4:
            raise RuntimeError(f"representation restoration failed at {cancer}: direction {restore}, combination {projection}")
        gaps.append({"cancer": cancer, "dwpc_gap": gap, "direction_restoration": restore, "combination_projection": projection})
        direction[query_number] = directed
        for column, predicate_index in enumerate(asymmetric_index):
            asymmetric[query_number, :, column * 3: column * 3 + 3] = directed[:, predicate_index * 3: predicate_index * 3 + 3]
        combo[query_number] = joint
        marginal[query_number] = margin
        secondary[query_number] = shares_from_mass(fine["combo"]["t1_e2"], fine["complete_dwpc"]["t1_e2"])
        denominator["t1_e1_directed_dwpc"][query_number] = fine["directed_dwpc"]["t1_e1"]
        denominator["t2_e1_complete_dwpc"][query_number] = fine["complete_dwpc"]["t2_e1"]
        denominator["t1_e2_complete_dwpc"][query_number] = fine["complete_dwpc"]["t1_e2"]
        denominator["two_edge_dwpc"][query_number] = fine["two_edge_dwpc"]
        denominator["three_edge_dwpc"][query_number] = fine["three_edge_dwpc"]
        state["t1_e1_direction_state"][query_number] = np.where(fine["two_edge_dwpc"] <= 0, MISSING_NO_PATH, np.where(fine["directed_dwpc"]["t1_e1"] > 0, MISSING_OBSERVED, MISSING_PATH_WITHOUT_RECORD)).astype(np.uint8)
        state["t2_e1_combo_state"][query_number] = np.where(fine["three_edge_dwpc"] <= 0, MISSING_NO_PATH, np.where(fine["complete_dwpc"]["t2_e1"] > 0, MISSING_OBSERVED, MISSING_PATH_WITHOUT_RECORD)).astype(np.uint8)
        if not skip_collisions:
            _count_collisions(cancer, index, hidden, directed, predicate, joint, margin, complete, observed, collision_targets, covered_targets)
        _trace(traces, cancer, gene_ids, {
            "complete_two_hop_direction": observed,
            "complete_three_hop_combination": complete,
            "no_two_hop_path": fine["two_edge_dwpc"] <= 0,
            "no_three_hop_path": fine["three_edge_dwpc"] <= 0,
            "direction_against_exceeds_along": np.nansum(directed[:, 1::3], axis=1) > np.nansum(directed[:, 0::3], axis=1),
        })
        if (query_number + 1) % 10 == 0 or query_number < 2:
            print(f"[numeric] {query_number + 1}/{n_cancer} {cancer} rss_gb={_peak_gb():.2f}", flush=True)
    payload = {
        "cancers": np.asarray(cancers),
        "gene_ids": np.asarray(gene_ids),
        "t1_e1_direction": direction,
        "t1_e1_direction_names": np.asarray(direction_column_names("t1_e1", GG_PREDICATES)),
        "t1_e1_asymmetric": asymmetric,
        "t1_e1_asymmetric_names": np.asarray(direction_column_names("t1_e1", asymmetric_predicates(GG_PREDICATES))),
        "t2_e1_combo": combo,
        "t2_e1_combo_names": np.asarray(combo_column_names("t2_e1")),
        "t2_e1_marginal": marginal,
        "t2_e1_marginal_names": np.asarray(marginal_column_names("t2_e1")),
        "t1_e2_combo": secondary,
        "t1_e2_combo_names": np.asarray(combo_column_names("t1_e2")),
        **denominator,
        **state,
    }
    np.savez_compressed(run_dir / "analysis" / "scoped_relation_features.npz", **payload)
    stored.close()
    shapes = {key: list(value.shape) for key, value in payload.items() if isinstance(value, np.ndarray) and value.ndim >= 2}
    return {"shapes": shapes, **{
        "max_dwpc_gap": max(row["dwpc_gap"] for row in gaps),
        "max_direction_restoration": max(row["direction_restoration"] for row in gaps),
        "max_combination_projection": max(row["combination_projection"] for row in gaps),
        "validation_targets_with_t1_e1_direction": covered_targets["t1_e1"],
        "validation_targets_in_t1_e1_direction_collision": collision_targets["t1_e1"],
        "validation_targets_with_t2_e1_combo": covered_targets["t2_e1"],
        "validation_targets_in_t2_e1_combo_collision": collision_targets["t2_e1"],
        "traces": traces,
    }}


def _trace(traces: list[dict], cancer: str, gene_ids: list[str], conditions: dict[str, np.ndarray]) -> None:
    seen = {row["kind"] for row in traces}
    for kind, mask in conditions.items():
        if kind in seen or not np.any(mask):
            continue
        traces.append({"kind": kind, "cancer_cui": cancer, "gene_id": gene_ids[int(np.flatnonzero(mask)[0])]})


def _count_collisions(cancer, index, hidden, directed, predicate, joint, margin, complete, observed, collision_targets, covered_targets) -> None:
    if np.any(observed):
        slots = np.flatnonzero(observed).tolist()
        coarse = [quant_signature(predicate[slot], 6) for slot in slots]
        fine = [quant_signature(directed[slot], 6) for slot in slots]
        flags = in_multi_group(coarse, fine)
        summary = collision_summary(coarse, fine)
        del summary
    else:
        slots, flags = [], []
    flagged = {slot for slot, flag in zip(slots, flags) if flag}
    combo_slots = np.flatnonzero(complete).tolist()
    if combo_slots:
        coarse_c = [quant_signature(margin[slot], 6) for slot in combo_slots]
        fine_c = [quant_signature(joint[slot], 6) for slot in combo_slots]
        combo_flags = {slot for slot, flag in zip(combo_slots, in_multi_group(coarse_c, fine_c)) if flag}
    else:
        combo_flags = set()
    for gene, role in hidden.get(cancer, {}).items():
        if role != "development_validation":
            continue
        slot = index[gene]
        if observed[slot]:
            covered_targets["t1_e1"] += 1
            collision_targets["t1_e1"] += int(slot in flagged)
        if complete[slot]:
            covered_targets["t2_e1"] += 1
            collision_targets["t2_e1"] += int(slot in combo_flags)


def _path_samples(run_dir: Path, graph: dict, views: dict, gene_ids: list[str], cancers: list[str]) -> dict:
    index = {gene: slot for slot, gene in enumerate(gene_ids)}
    reverse = {slot: gene for gene, slot in index.items()}
    cancers_of = defaultdict(set)
    deg_gc = defaultdict(int)
    for cancer, genes in graph["genes_of"].items():
        for gene in genes:
            cancers_of[gene].add(cancer)
            deg_gc[gene] += 1
    deg_gg = {gene: len(graph["neighbors"].get(gene, ())) for gene in gene_ids}
    cancer_codes = {cancer: code for code, cancer in enumerate(sorted(set(cancers) | set(graph["genes_of"])))}
    code_cancer = {code: cancer for cancer, code in cancer_codes.items()}
    extra_ids = sorted({gene for genes in graph["genes_of"].values() for gene in genes if gene not in index})
    extra_code = {gene: position for position, gene in enumerate(extra_ids)}
    slot_table = candidate_slot_table(graph["genes_of"], index)
    slot_gc = np.zeros(len(gene_ids), dtype=np.float64)
    for gene, slot in index.items():
        slot_gc[slot] = _power(deg_gc.get(gene, 0), DAMPING)
    n_gene = len(gene_ids)
    coverage = {"two_retained_dwpc": 0.0, "two_total_dwpc": 0.0, "three_retained_dwpc": 0.0, "three_total_dwpc": 0.0, "empty_two": 0, "empty_three": 0, "retained_rows": 0}
    same_path = np.zeros((len(cancers), n_gene), dtype=np.float32)
    writer = None
    for query_number, cancer in enumerate(cancers):
        q_factor = _power(len(graph["genes_of"].get(cancer, ())), DAMPING)
        values = np.full((n_gene, PATH_BUDGET), -1.0)
        keys_a = np.full((n_gene, PATH_BUDGET), np.iinfo(np.int32).max, dtype=np.int32)
        keys_b = np.full((n_gene, PATH_BUDGET), np.iinfo(np.int32).max, dtype=np.int32)
        counts = np.zeros(n_gene, dtype=np.int32)
        total = np.zeros(n_gene, dtype=np.float64)
        for gene in graph["genes_of"].get(cancer, ()):
            partners = [item for item in graph["neighbors"].get(gene, ()) if item in index and item != gene]
            if not partners:
                continue
            base = _power(deg_gg.get(gene, 0), DAMPING) * _power(deg_gc.get(gene, 0), DAMPING) * q_factor
            partner_slots = np.array([index[item] for item in partners], dtype=np.int32)
            weights = base * np.array([_power(deg_gg.get(item, 0), DAMPING) for item in partners])
            total[partner_slots] += weights
            gene_code = index[gene] if gene in index else -(extra_code[gene] + 1)
            update_topk(values, keys_a, keys_b, counts, partner_slots, weights, int(gene_code), 0)
        cancer_rows = []
        _collect_paths(cancer_rows, coverage, same_path[query_number], "two_hop", cancer, values, keys_a, keys_b, counts, total, reverse, code_cancer, extra_ids, views)
        values.fill(-1.0)
        keys_a.fill(np.iinfo(np.int32).max)
        keys_b.fill(np.iinfo(np.int32).max)
        counts.fill(0)
        total.fill(0.0)
        for gene in graph["genes_of"].get(cancer, ()):
            g_term = _power(deg_gc.get(gene, 0), 0.8)
            if g_term == 0:
                continue
            g_slot = index.get(gene)
            gene_code = index[gene] if gene in index else -(extra_code[gene] + 1)
            for other in cancers_of.get(gene, ()):
                if other == cancer:
                    continue
                member_slots = slot_table.get(other)
                if member_slots is None or len(member_slots) == 0:
                    continue
                local = member_slots if g_slot is None else member_slots[member_slots != g_slot]
                if len(local) == 0:
                    continue
                degree = float(len(graph["genes_of"].get(other, ())))
                weights = g_term * _power(degree, 0.8) * q_factor * slot_gc[local]
                total[local] += weights
                update_topk(values, keys_a, keys_b, counts, local.astype(np.int32), weights, cancer_codes[other], int(gene_code))
        _collect_paths(cancer_rows, coverage, None, "three_hop", cancer, values, keys_a, keys_b, counts, total, reverse, code_cancer, extra_ids, views)
        if cancer_rows:
            batch = pa.Table.from_pylist(cancer_rows)
            if writer is None:
                writer = pq.ParquetWriter(run_dir / "analysis" / "sampled_paths.parquet", batch.schema)
            writer.write_table(batch)
        print(f"[paths] {query_number + 1}/{len(cancers)} {cancer} rows={coverage['retained_rows']} rss_gb={_peak_gb():.2f}", flush=True)
    if writer is not None:
        writer.close()
    np.save(run_dir / "analysis" / "sampled_twohop_same_path_mass.npy", same_path)
    coverage["same_path_feature"] = "sampled_twohop_asymmetric_predicate_and_complete_core is accumulated only on the retained two-hop paths"
    coverage["path_budget"] = PATH_BUDGET
    coverage["selection"] = "highest DWPC, then smaller intermediate identifiers; labels are not read"
    return coverage


def _resolve_gene(code: int, reverse: dict[int, str], extra_ids: list[str]) -> str:
    if code >= 0:
        return reverse[code]
    return extra_ids[-code - 1]


def _collect_paths(rows, coverage, same_path, template, cancer, values, keys_a, keys_b, counts, total, reverse, code_cancer, extra_ids, views) -> None:
    key = "two" if template == "two_hop" else "three"
    for slot, count in enumerate(counts.tolist()):
        coverage[f"{key}_total_dwpc"] += float(total[slot])
        if count == 0:
            coverage[f"empty_{key}"] += 1
            continue
        for rank in range(count):
            weight = float(values[slot, rank])
            coverage[f"{key}_retained_dwpc"] += weight
            coverage["retained_rows"] += 1
            first = int(keys_a[slot, rank])
            second = int(keys_b[slot, rank])
            gene = reverse[slot]
            if template == "two_hop":
                intermediate = _resolve_gene(first, reverse, extra_ids)
                ordered = (gene, intermediate) if gene <= intermediate else (intermediate, gene)
                left = views.get(("gene_gene", ordered[0], ordered[1]))
                right = views.get(("gene_cancer", intermediate, cancer))
                asymmetric = bool(left and any(name not in SYMMETRIC_PREDICATES and left["directions"].get(name) for name in GG_PREDICATES))
                complete = bool(right and right["combo_share"] is not None)
                if asymmetric and complete and same_path is not None:
                    same_path[slot] += weight
                intermediate_b = ""
                intermediate_a = intermediate
            else:
                intermediate_a = code_cancer[first]
                intermediate_b = _resolve_gene(second, reverse, extra_ids) if second != np.iinfo(np.int32).max else ""
            rows.append({
                "cancer_cui": cancer,
                "gene_id": gene,
                "template": template,
                "rank": rank,
                "intermediate_a": intermediate_a,
                "intermediate_b": intermediate_b,
                "dwpc": weight,
                "path_budget": PATH_BUDGET,
            })


def _citation(run_dir: Path, graph: dict, gene_ids: list[str], cancers: list[str], masked: set[int]) -> dict:
    needed = {pmid for pmids in graph["edge_pmids"].values() for pmid in pmids}
    outgoing, incoming, _citing, _cited = load_citation_index(masked, needed)
    indexed = _indexed_pmids(needed)
    ready = {}
    for key, pmids in graph["edge_pmids"].items():
        papers = tuple(sorted(int(item) for item in pmids if int(item) > 0))
        ready[key] = papers if papers and all(pmid in indexed for pmid in papers) else ()
    n_cancer = len(cancers)
    n_gene = len(gene_ids)
    sums = {f"{position}_{name}": np.zeros((n_cancer, n_gene), dtype=np.float32) for position in CITATION_POSITIONS for name in MOTIF_NAMES}
    observed = {position: np.zeros((n_cancer, n_gene), dtype=np.float32) for position in CITATION_POSITIONS}
    paths = {position: np.zeros((n_cancer, n_gene), dtype=np.float32) for position in CITATION_POSITIONS}
    states = {position: np.zeros((n_cancer, n_gene), dtype=np.uint8) for position in CITATION_POSITIONS}
    distinct = {position: np.zeros((n_cancer, n_gene), dtype=np.float32) for position in ("t1", "t2")}
    index = {gene: slot for slot, gene in enumerate(gene_ids)}
    cancers_of = defaultdict(set)
    for cancer, genes in graph["genes_of"].items():
        for gene in genes:
            cancers_of[gene].add(cancer)
    cache = {}
    block_cache = {}
    for query_number, cancer in enumerate(cancers):
        _citation_two_hop(query_number, cancer, graph, index, ready, outgoing, incoming, indexed, cache, sums, observed, paths, distinct)
        _citation_three_hop(query_number, cancer, graph, index, cancers_of, ready, outgoing, incoming, indexed, cache, sums, observed, paths, distinct, block_cache)
        cache.clear()
        for position in CITATION_POSITIONS:
            present = paths[position][query_number] > 0
            known = observed[position][query_number] > 0
            linked = np.zeros(n_gene, dtype=bool)
            for name in ("direct_collapsed", "cocite_witnesses", "shared_reference_witnesses"):
                linked |= sums[f"{position}_{name}"][query_number] > 0
            state = np.full(n_gene, CITATION_NO_PATH, dtype=np.uint8)
            state[present & ~known] = CITATION_UNKNOWN
            state[known & ~linked] = CITATION_CONFIRMED_ZERO
            state[known & linked] = CITATION_OBSERVED_LINK
            states[position][query_number] = state
        if (query_number + 1) % 10 == 0 or query_number < 2:
            print(f"[citation] {query_number + 1}/{n_cancer} {cancer} rss_gb={_peak_gb():.2f}", flush=True)
    means = {}
    for position in CITATION_POSITIONS:
        denominator = observed[position]
        for name in MOTIF_NAMES:
            mean = np.full_like(sums[f"{position}_{name}"], np.nan)
            take = denominator > 0
            mean[take] = sums[f"{position}_{name}"][take] / denominator[take]
            means[f"{position}_{name}_mean"] = mean
    for position, path_name in (("t1", "t1"), ("t2", "t2_23")):
        mean = np.full_like(distinct[position], np.nan)
        take = paths[path_name] > 0
        mean[take] = distinct[position][take] / paths[path_name][take]
        means[f"{position}_distinct_supporting_pmids_mean"] = mean
    np.savez_compressed(
        run_dir / "analysis" / "citation_features.npz",
        cancers=np.asarray(cancers),
        gene_ids=np.asarray(gene_ids),
        **means,
        **{f"{position}_observed_paths": observed[position] for position in CITATION_POSITIONS},
        **{f"{position}_paths": paths[position] for position in CITATION_POSITIONS},
        **{f"{position}_state": states[position] for position in CITATION_POSITIONS},
        **{f"{position}_distinct_pmid_mass": distinct[position] for position in ("t1", "t2")},
        article_pair_cap=np.asarray([64]),
        article_pair_note=np.asarray(["Witness counts are stored. Article-pair enumeration stops above 64 endpoint pairs and is not a model column."]),
    )
    return {
        "citation_columns": len(means),
        "article_pairs_above_cap_state": ARTICLE_PAIRS_NOT_ENUMERATED,
        "unit": "path occurrence, then divided by the number of adjacent pairs whose PMIDs are both in the citation snapshot",
    }


def _motif(left, right, outgoing, incoming, indexed, cache):
    key = (left, right)
    found = cache.get(key)
    if found is None:
        found = citation_motif(left, right, outgoing, incoming, indexed)
        cache[key] = found
    return found


def _citation_two_hop(query_number, cancer, graph, index, ready, outgoing, incoming, indexed, cache, sums, observed, paths, distinct) -> None:
    for gene in graph["genes_of"].get(cancer, ()):
        right = ready.get(("gene_cancer", gene, cancer), ())
        partners = [item for item in graph["neighbors"].get(gene, ()) if item in index and item != gene]
        for partner in partners:
            slot = index[partner]
            paths["t1"][query_number, slot] += 1
            ordered = (partner, gene) if partner <= gene else (gene, partner)
            left_key = ("gene_gene", ordered[0], ordered[1])
            left = ready.get(left_key, ())
            raw_left = graph["edge_pmids"].get(left_key, ())
            raw_right = graph["edge_pmids"].get(("gene_cancer", gene, cancer), ())
            distinct["t1"][query_number, slot] += len(set(raw_left) | set(raw_right))
            motif = _motif(left, right, outgoing, incoming, indexed, cache)
            if motif["state"] == CITATION_UNKNOWN:
                continue
            observed["t1"][query_number, slot] += 1
            for name in MOTIF_NAMES:
                sums[f"t1_{name}"][query_number, slot] += motif[name]


def _citation_three_hop(query_number, cancer, graph, index, cancers_of, ready, outgoing, incoming, indexed, cache, sums, observed, paths, distinct, block_cache) -> None:
    """One motif block per intermediate cancer, summed across all extenders at once."""
    extenders_of = defaultdict(list)
    for gene in graph["genes_of"].get(cancer, ()):
        for other in cancers_of.get(gene, ()):
            if other != cancer:
                extenders_of[other].append(gene)
    for other, extenders in extenders_of.items():
        cached = block_cache.get(other)
        if cached is None:
            ordered = [item for item in graph["genes_of"].get(other, ()) if item in index]
            local = {item: pos for pos, item in enumerate(ordered)}
            papers = [ready.get(("gene_cancer", item, other), ()) for item in ordered]
            cached = {"ordered": ordered, "local": local, "block": build_motif_block(papers, outgoing, incoming) if ordered else None}
            block_cache[other] = cached
        ordered = cached["ordered"]
        local = cached["local"]
        block = cached["block"]
        if not ordered or block is None:
            continue
        extender_local = [local[gene] for gene in extenders if gene in local]
        candidate_genes = [item for item in ordered if item not in set(extenders)]
        # A candidate may also be an extender on another path; sum_adjacent_motifs excludes self pairs.
        candidate_genes = ordered
        candidate_local = list(range(len(ordered)))
        for gene in extenders:
            members = [item for item in ordered if item != gene]
            if not members:
                continue
            slots = np.array([index[item] for item in members], dtype=np.int32)
            paths["t2_23"][query_number, slots] += 1
            paths["t2_12"][query_number, slots] += 1
            raw_mid = graph["edge_pmids"].get(("gene_cancer", gene, other), ())
            raw_query = graph["edge_pmids"].get(("gene_cancer", gene, cancer), ())
            distinct["t2"][query_number, slots] += len(set(raw_mid) | set(raw_query))
            motif = _motif(ready.get(("gene_cancer", gene, other), ()), ready.get(("gene_cancer", gene, cancer), ()), outgoing, incoming, indexed, cache)
            if motif["state"] != CITATION_UNKNOWN:
                observed["t2_23"][query_number, slots] += 1
                for name in MOTIF_NAMES:
                    sums[f"t2_23_{name}"][query_number, slots] += motif[name]
        if not extender_local:
            continue
        seen, motif_sums = sum_adjacent_motifs(block, extender_local, candidate_local)
        for local_pos, gene_id in enumerate(candidate_genes):
            if seen[local_pos] <= 0:
                continue
            slot = index[gene_id]
            observed["t2_12"][query_number, slot] += seen[local_pos]
            for source, target in (
                ("direct_collapsed", "direct_collapsed"),
                ("direct_ltr", "direct_ltr"),
                ("direct_rtl", "direct_rtl"),
                ("direct_ambiguous", "direct_ambiguous"),
                ("cocite", "cocite_witnesses"),
                ("shared_references", "shared_reference_witnesses"),
            ):
                sums[f"t2_12_{target}"][query_number, slot] += motif_sums[source][local_pos]


def _indexed_pmids(needed: set[int]) -> set[int]:
    found = set()
    parquet = pq.ParquetFile(CITATION_AUDIT / "analysis" / "endpoint_metadata.parquet")
    for batch in parquet.iter_batches(columns=["pmid"], batch_size=200_000):
        for pmid in batch.column(0).to_numpy():
            pmid = int(pmid)
            if pmid in needed:
                found.add(pmid)
    return found


def _graph_package(run_dir: Path, graph: dict, views: dict, gene_ids: list[str], cancers: list[str]) -> dict:
    rows = []
    predicate_rows = []
    for key, pmids in graph["edge_pmids"].items():
        family, left, right = key
        view = views.get(key, {})
        rows.append({"family": family, "endpoint_a": left, "endpoint_b": right, "inverse": 0, "pmids": len(pmids), "derived_from": ""})
        rows.append({"family": family, "endpoint_a": right, "endpoint_b": left, "inverse": 1, "pmids": len(pmids), "derived_from": "|".join(key)})
        for predicate, directions in view.get("directions", {}).items():
            predicate_rows.append({"family": family, "endpoint_a": left, "endpoint_b": right, "predicate": predicate, "directions": ",".join(sorted(directions)), "inverse": 0})
            swapped = [("against" if item == "along" else "along" if item == "against" else item) for item in directions]
            predicate_rows.append({"family": family, "endpoint_a": right, "endpoint_b": left, "predicate": predicate, "directions": ",".join(sorted(set(swapped))), "inverse": 1})
    pq.write_table(pa.Table.from_pylist(rows), run_dir / "analysis" / "structural_edges.parquet")
    pq.write_table(pa.Table.from_pylist(predicate_rows) if predicate_rows else pa.table({"family": [], "predicate": []}), run_dir / "analysis" / "predicate_edges.parquet")
    nodes = [{"node_type": "gene", "node_id": gene, "candidate_slot": slot} for slot, gene in enumerate(gene_ids)]
    extra = sorted({gene for genes in graph["genes_of"].values() for gene in genes if gene not in set(gene_ids)})
    nodes.extend({"node_type": "gene", "node_id": gene, "candidate_slot": -1} for gene in extra)
    nodes.extend({"node_type": "cancer", "node_id": cancer, "candidate_slot": -1} for cancer in sorted(graph["genes_of"]))
    pq.write_table(pa.Table.from_pylist(nodes), run_dir / "analysis" / "graph_nodes.parquet")
    return {
        "structural_edges": len(rows),
        "predicate_edge_rows": len(predicate_rows),
        "nodes": len(nodes),
        "query_cancers": len(cancers),
        "supervision": "label index, separate from background edges",
        "inverse_rule": "Inverse rows swap the stored endpoints of a masked edge and swap along with against. They add no PMID.",
    }


def _libraries() -> dict:
    found = {}
    for name in ("xgboost", "lightgbm"):
        try:
            module = __import__(name)
            found[name] = getattr(module, "__version__", "imported")
        except Exception as exc:  # noqa: BLE001
            found[name] = f"unavailable: {exc.__class__.__name__}"
    return found


def _lowest_columns() -> list[dict]:
    stored = np.load(SOURCE_RUN / "analysis" / "feature_store.npz")
    rows = []
    for name in feature_names()["both"]:
        if name not in stored.files:
            continue
        values = stored[name]
        finite = np.isfinite(values)
        rows.append({"column": name, "finite_fraction": float(finite.mean()), "nonzero_fraction": float(np.nan_to_num(values).astype(bool).mean())})
    stored.close()
    rows.sort(key=lambda row: row["finite_fraction"])
    return rows[:8]


def _study_design() -> dict:
    return {
        "task": "Rank candidate human genes within one exact cancer CUI and recover masked CORE direct records.",
        "sample": "NCBI Gene ID x cancer CUI",
        "primary_metric": "macro Recall@50 over cancers that have a hidden target in the scored partition",
        "secondary_metrics": ["Recall@100", "pooled hits", "top-50 entries and exits"],
        "tie_break": "higher score, then gene identifier",
        "evaluation_lists": {
            "full": "all 13,571 candidate genes, including genes with no path, no attribute, or no PMID",
            "primary": "full list minus candidates that still have a direct-eligible edge in the masked joint graph",
            "historical": "full list minus unmasked direct genes, except the hidden targets of the partition being scored",
        },
        "direct_eligibility": "An unmasked gene-cancer edge with a CORE sentence or a SemMedDB traversal. Masked visibility is the surviving edge.",
        "mask": "Drop the 5,206 target edges. Drop a PMID on any remaining edge when it supports a selected target. Inverse edges and auxiliary triples are copied from that masked graph.",
        "citation_mask": "Drop a citation when either PMID is blocked. Co-citation and shared-reference witnesses use the same filtered snapshot. Witnesses outside CORE and SemMedDB are allowed.",
        "paths": {
            "two_hop": "candidate gene -> intermediate gene -> query cancer",
            "three_hop": "candidate gene -> intermediate cancer -> intermediate gene -> query cancer",
            "weights": "raw count, historical one-over-degree, and DWPC with damping 0.4 share the masked graph",
        },
        "missingness": {
            "0": "no path",
            "1": "path without an eligible record, or citation unknown because a PMID is absent",
            "2": "record present but the field is not judgeable",
            "3": "observed, including an observed zero",
            "citation_states": "0 no path, 1 unknown, 2 confirmed zero in the snapshot, 3 observed link",
        },
        "transforms_not_fit": {
            "fit_on": "training rows of one seed only, after this preparation",
            "shares": "no log1p; median imputation and a missingness indicator",
            "counts": "log1p, then the same imputation",
            "constant_columns": "drop only after they are constant on the training rows",
            "excluded_from_numeric_features": ["gene_id", "cancer_cui", "pmid", "native_id", "role"],
        },
        "scoped_builds": {
            "semmed_direction": "two-hop first edge, candidate gene to intermediate gene",
            "core_combination": "three-hop first edge, candidate gene to intermediate cancer; two-hop final edge is secondary",
            "path_budget": PATH_BUDGET,
        },
    }


def _report(run_dir: Path, summary: dict) -> None:
    lines = [
        "# Frozen inputs for development gene–cancer ranking",
        "",
        "The task is to rank candidate human genes within one exact cancer CUI and recover CORE direct records whose edges and PMIDs are masked. No model was fit and no development candidate was scored.",
        "",
        "## Evaluation lists",
        "",
        "The historical list removes every gene with an unmasked direct edge, except the hidden targets of the partition being scored. That direct set comes from the joint graph: a gene–cancer edge with a CORE sentence or a SemMedDB traversal. It is not read from the label table. The earlier Recall@50 used this list.",
        "The primary list starts from the same 13,571 genes and removes a gene only when that direct edge is still visible after the mask. Hidden targets are not visible; a visible hidden target would have stopped the run. Genes whose unmasked direct edge disappeared because its PMIDs were blocked stay in the primary list and stay out of the historical list. The two Recall@50 values are therefore not the same denominator. Both lists, and the full list, are stored in `analysis/evaluation_masks.npz` and can be applied to one later score matrix.",
        "",
        f"Primary-list size, mean across cohort cancers: {summary['lists']['mean_primary_candidates']:.1f}. Historical validation list, mean: {summary['lists']['mean_historical_validation_candidates']:.1f}. Cancers where those lists differ: {summary['lists']['cancers_with_list_difference']}.",
        "",
        "## Inputs added in this run",
        "",
        "SemMedDB direction is a numeric matrix on the two-hop gene–gene edge. Direction bins sum back to the frozen undirected predicate weight. An asymmetric-predicate copy sets the four symmetric predicates aside; those four are argument-order diagnostics, not a biological causal direction. Traversal direction is alignment with the path, not a causal claim.",
        "CORE same-sentence combinations are a numeric matrix on the three-hop edge from the candidate gene to the intermediate cancer. The matched single-field margins use the same complete records, the same split, and the same denominator. The older per-field features remain in the relation-attribute feature store and use every surviving field value, including rows that do not form a complete triple. The two-hop final edge has a secondary combination matrix and is not the primary contrast.",
        "Citation columns use the same masked PMID list. A link is removed when either PMID is blocked. Values are means over path occurrences whose two PMIDs are in the citation snapshot. Confirmed zeros, unknown coverage, and observed links have separate states. Witness counts are stored; article-pair enumeration is capped and is not a model column. Background degree, publication count, shared-PMID fraction, and the older PMID-count columns stay in the existing feature store.",
        "",
        "Fixed references and the 12 legacy column sets read the existing feature store. Ridge and trees read those columns plus the new matrices through column lists, with the preflight training rows. LambdaMART uses the same rows and query groups. Random-walk adjacency is the masked gene–gene and gene–cancer graph, which includes neighbors outside the two fixed templates. Path-set rows keep at most eight paths per template, chosen by DWPC without reading labels; empty candidates remain in the coverage table. Graph files contain the masked structural edges, their inverse rows, and predicate rows on those same endpoints. Supervision stays in the label index.",
        "",
        "## Comparison status",
        "",
        "See `analysis/comparison_status.tsv`. A readable input is not an estimate of Recall@50.",
        "",
        f"Preparation summary: {json.dumps(summary['flags'])}",
        "",
    ]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_catalogs(analysis: Path, numeric: dict, citation: dict, paths: dict, package: dict, libraries: dict) -> None:
    _tsv(analysis / "trace_examples.tsv", numeric.get("traces", []), ["kind", "cancer_cui", "gene_id"])
    _tsv(analysis / "model_input_readout.tsv", [
        {"model": "fixed_references_and_legacy_12", "status": "runnable", "input": str(SOURCE_RUN / "analysis" / "feature_store.npz"), "shape": "140 x 13571 x selected columns", "note": "Existing store. Column lists select the 12 legacy specifications."},
        {"model": "ridge_trees_lambdamart", "status": "runnable" if "unavailable" not in " ".join(libraries.values()) else "runnable_for_ridge_and_trees", "input": str(PREFLIGHT_RUN / "analysis" / "training_rows.parquet"), "shape": "103092 training rows; query groups by cancer", "note": f"Libraries: {json.dumps(libraries)}. No ranker was constructed."},
        {"model": "semmed_direction_t1_e1", "status": "scoped_runnable", "input": str(analysis / "scoped_relation_features.npz"), "shape": str(numeric.get("shapes", {}).get("t1_e1_direction")), "note": "Two-hop first edge only. Asymmetric copy is a sensitivity."},
        {"model": "core_combination_t2_e1", "status": "scoped_runnable", "input": str(analysis / "scoped_relation_features.npz"), "shape": str(numeric.get("shapes", {}).get("t2_e1_combo")), "note": "Three-hop first edge. t1_e2_combo is the secondary edge."},
        {"model": "citation", "status": "runnable" if citation.get("status") != "blocked" else "blocked", "input": str(analysis / "citation_features.npz"), "shape": "140 x 13571", "note": citation.get("unit", citation.get("error", ""))},
        {"model": "random_walk", "status": "runnable", "input": str(analysis / "walk_adjacency.npz"), "shape": "masked gene-gene and gene-cancer adjacency", "note": "The walk sees every surviving neighbor, including edges outside the two fixed templates. No walk scores were written."},
        {"model": "path_set", "status": "scoped_runnable" if paths.get("status") != "blocked" else "blocked", "input": str(analysis / "sampled_paths.parquet"), "shape": f"budget {PATH_BUDGET}", "note": paths.get("selection", paths.get("error", ""))},
        {"model": "complex_rgcn_biopathnet", "status": "runnable_as_whole_method", "input": str(analysis / "structural_edges.parquet"), "shape": str(package.get("structural_edges")), "note": "Structural edges plus inverses are the message-passing neighborhood. Predicate rows label the same endpoints. This is wider than the tabular column set."},
    ], ["model", "status", "input", "shape", "note"])
    rows = []
    for name, status, changed, reason in (
        ("legacy_12_column_sets", "runnable", "which of the 12 stored column sets is read", "One feature store and one column list."),
        ("path_count_versus_dwpc", "runnable", "path weight", "Both weights are already in the feature store."),
        ("historical_versus_primary_list", "runnable", "evaluation mask", "The score matrix is not changed. Historical Recall@50 is not the primary-list denominator."),
        ("t1_e1_undirected_versus_directed", "scoped_runnable", "direction columns on the gene-gene edge", "Scope is the two-hop first edge. Restoration was checked against the stored undirected weights."),
        ("t2_e1_marginal_versus_combination", "scoped_runnable", "combination columns on the first three-hop edge", "The matched margins use the same complete records and denominator. The older field features remain a historical reference."),
        ("citation_increments", "runnable" if citation.get("status") != "blocked" else "blocked", "one citation column group", "Provenance columns already in the feature store stay fixed. New citation means use path occurrence, not DWPC."),
        ("sampled_path_versus_full_path_baseline", "scoped_runnable" if paths.get("status") != "blocked" else "blocked", "path sample versus all paths", "The cap changes the path set, so this is not a feature-only contrast."),
        ("graph_model_versus_tabular", "runnable_as_whole_method", "neighborhood and relation inventory", "Graph models see structural neighbors beyond the two templates."),
        ("same_path_core_and_predicate", "scoped_runnable" if paths.get("status") != "blocked" else "blocked", "co-occurrence on retained two-hop paths", "Independent edge shares are not multiplied. The feature exists only on the DWPC sample."),
    ):
        rows.append({"comparison": name, "status": status, "held_fixed": "masked graph, 5,206 targets, candidate index", "changed": changed, "reason": reason})
    _tsv(analysis / "comparison_status.tsv", rows, ["comparison", "status", "held_fixed", "changed", "reason"])
    dictionary = [
        {"column": "t1_e1_dir_*", "group": "semmed_direction", "unit": "share of directed DWPC", "missing": "NaN when the two-hop edge has no affirmative allow-listed predicate; state 0 is no path", "transform_later": "no log1p; fit imputer on training rows"},
        {"column": "t2_e1_combo_*", "group": "core_same_sentence", "unit": "share of complete-tuple DWPC", "missing": "NaN when no complete same-sentence tuple is observed", "transform_later": "no log1p; same missing rule as the matched marginal"},
        {"column": "t2_e1_marginal_*", "group": "core_matched_marginal", "unit": "projection of the combination share", "missing": "same denominator as t2_e1_combo", "transform_later": "fit on training rows only"},
        {"column": "t1_*_mean and t2_12_*_mean and t2_23_*_mean", "group": "citation", "unit": "path occurrence", "missing": "state 1 unknown, state 2 confirmed zero, state 3 observed link", "transform_later": "log1p counts after the mean is formed; do not treat unknown as zero"},
        {"column": "t1_distinct_supporting_pmids_mean", "group": "provenance", "unit": "distinct PMIDs on the two two-hop edges, averaged over paths", "missing": "NaN when the candidate has no two-hop path", "transform_later": "log1p"},
        {"column": "t2_distinct_supporting_pmids_mean", "group": "provenance", "unit": "distinct PMIDs on the intermediate-gene edges only", "missing": "NaN when the candidate has no three-hop path", "transform_later": "log1p"},
        {"column": "gene_publications, t1_shared_pmid, t1_pmid_count", "group": "existing_provenance", "unit": "see the relation-attribute dictionary; shared PMID and PMID count are DWPC-weighted", "missing": "NaN when the template DWPC is zero", "transform_later": "already stored in feature_store.npz"},
        {"column": "sampled path dwpc", "group": "path_sample", "unit": "DWPC of at most 8 paths", "missing": "a candidate with no retained path is absent from the path table and remains in the coverage counts", "transform_later": "do not compare to the full-path model without recording the cap"},
    ]
    _tsv(analysis / "feature_dictionary.tsv", dictionary, ["column", "group", "unit", "missing", "transform_later"])


def run(run_dir: Path) -> None:
    analysis = run_dir / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    (run_dir / "study_design.yaml").write_text(yaml.safe_dump(_study_design(), sort_keys=False), encoding="utf-8")
    records = _load_targets()
    gene_ids, candidates, queries = load_universe()
    identity = _identity(records, gene_ids)
    if not identity["role_counts_match"] or identity["validation_cancers"] != 131 or identity["gene_count"] != 13571:
        raise RuntimeError("frozen split identity failed: " + json.dumps(identity))
    if identity["pair_sha256"] != "c3d25c4c46cc04c11ced0bfcbd83e42c428844b45c614d9cd3bd18acb9599d93":
        raise RuntimeError("target-pair hash does not match the representation audit")
    training_rows = pq.ParquetFile(identity["preflight_training_rows"]).metadata.num_rows
    if training_rows != 103092:
        raise RuntimeError(f"frozen training rows are {training_rows}, expected 103092")
    cancers = sorted({row["cancer_cui"] for row in records})
    profiles = load_profiles()
    membership = load_membership()
    target_pairs = {(row["gene_id"], row["cancer_cui"]) for row in records}
    masked = {pmid for row in records for pmid in row["pmids"]}
    graph = build_graphs(profiles, membership, candidates, queries, target_pairs, masked)
    lists = _write_lists(run_dir, records, gene_ids, cancers, graph)
    adjacency = _write_adjacency(run_dir, graph, gene_ids)
    if not adjacency["traced_adjacency"]["in_masked_edge_pmids"]:
        raise RuntimeError("traced walk edge is not in the masked edge list")
    print(f"[graph] edges={len(graph['edge_pmids'])} rss_gb={_peak_gb():.2f}", flush=True)
    raw, stream_audit = _stream(set(graph["edge_pmids"]), masked, target_pairs)
    views, _old = _views(raw)
    index = {gene: slot for slot, gene in enumerate(gene_ids)}
    reverse = {slot: gene for gene, slot in index.items()}
    matrices = _matrices(views, candidate_slot_table(graph["genes_of"], index), reverse)
    print(f"[stream] record_edges={stream_audit.get('edges_with_records')} rss_gb={_peak_gb():.2f}", flush=True)
    numeric = _numeric(run_dir, graph, views, matrices, gene_ids, cancers, records, skip_collisions=True)
    prior_checkpoint = analysis / "checkpoint_numeric.json"
    if prior_checkpoint.exists():
        prior = json.loads(prior_checkpoint.read_text())
        for key in (
            "validation_targets_with_t1_e1_direction",
            "validation_targets_in_t1_e1_direction_collision",
            "validation_targets_with_t2_e1_combo",
            "validation_targets_in_t2_e1_combo_collision",
            "traces",
        ):
            if key in prior:
                numeric[key] = prior[key]
    write_json(analysis / "checkpoint_numeric.json", numeric)
    try:
        paths = _path_samples(run_dir, graph, views, gene_ids, cancers)
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        paths = {"status": "blocked", "error": f"{exc.__class__.__name__}: {exc}"}
    try:
        citation = _citation(run_dir, graph, gene_ids, cancers, masked)
    except Exception as exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        citation = {"status": "blocked", "error": f"{exc.__class__.__name__}: {exc}"}
    package = _graph_package(run_dir, graph, views, gene_ids, cancers)
    libraries = _libraries()
    _write_catalogs(analysis, numeric, citation, paths, package, libraries)
    lowest = _lowest_columns()
    flags = {
        "formal_test_files_opened": False,
        "models_fit": False,
        "candidate_scores_written": False,
        "stream": {key: int(value) for key, value in stream_audit.items()},
        "numeric": numeric,
        "citation": citation,
        "paths": {key: paths[key] for key in paths if key != "same_path_feature"},
        "libraries": libraries,
    }
    write_json(run_dir / "validation.json", {"identity": identity, "lists": lists, "adjacency": adjacency, "flags": flags, "lowest_finite_columns": lowest})
    write_json(run_dir / "manifest.json", {
        "sources": {
            "development_split": str(SOURCE_RUN),
            "preflight": str(PREFLIGHT_RUN),
            "representation_audit": str(AUDIT_RUN),
            "citation_audit": str(CITATION_AUDIT),
        },
        "hashes": {key: identity[key] for key in identity if key.endswith("sha256")},
        "inputs": {
            "evaluation_masks": str(analysis / "evaluation_masks.npz"),
            "feature_store": str(SOURCE_RUN / "analysis" / "feature_store.npz"),
            "training_rows": identity["preflight_training_rows"],
            "scoped_relation_features": str(analysis / "scoped_relation_features.npz"),
            "citation_features": str(analysis / "citation_features.npz"),
            "sampled_paths": str(analysis / "sampled_paths.parquet"),
            "walk_adjacency": str(analysis / "walk_adjacency.npz"),
            "structural_edges": str(analysis / "structural_edges.parquet"),
            "predicate_edges": str(analysis / "predicate_edges.parquet"),
            "graph_nodes": str(analysis / "graph_nodes.parquet"),
        },
        "cache_key": {
            "graph": "joint_graph_and_evidence_audit_20261116T0404Z",
            "targets_sha256": identity["development_targets_sha256"],
            "pair_sha256": identity["pair_sha256"],
            "damping": DAMPING,
            "path_budget": PATH_BUDGET,
            "direction_scope": "t1_e1",
            "combination_scope": "t2_e1",
        },
    })
    _report(run_dir, {"lists": lists, "flags": flags, "package": package})
    print("[done] input preparation", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.run_dir)


if __name__ == "__main__":
    main()

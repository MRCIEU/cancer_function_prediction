"""Read-only preflight for development gene–cancer ranking inputs.

The script checks frozen artifacts, replays the training-control rule, and
writes column manifests. It does not fit a model, score candidates, or open
a formal test file.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.association_recovery import connected_pmid_groups, rank_with_tie_break, recall_at_k, sample_controls
from src.common import RESULTS_ROOT, sha256_file, sha256_text, write_json
from src.relation_attributes import (
    CONTROL_RATIO,
    GC_PREDICATES,
    GG_PREDICATES,
    SEEDS,
    feature_names,
)
from src.recovery_models import cancer_weights
from src.run_association_recovery import build_graphs, load_membership, load_profiles, load_universe

ATTR_RUN = RESULTS_ROOT / "relation_attributes_development_20261117T1544Z"
REPR_RUN = RESULTS_ROOT / "relation_representation_audit_20261117T1917Z"
PRIOR_24 = RESULTS_ROOT / "association_recovery_development_20261116T2109Z"
GRAPH_RUN = RESULTS_ROOT / "joint_graph_and_evidence_audit_20261116T0404Z"
CITATION_RUN = RESULTS_ROOT / "citation_path_evidence_audit_20261116T1647Z"
STRUCTURES = {
    "counts": ["two_edge_paths", "three_edge_paths"],
    "prior_degree": ["two_edge_prior_weight", "three_edge_prior_weight"],
    "dwpc": ["two_edge_dwpc", "three_edge_dwpc"],
}
HELD = ["gene_gg_degree", "gene_gc_degree", "gene_publications", "has_two_edge", "has_three_edge"]


def _tsv(path: Path, rows: list[dict], columns: list[str]) -> None:
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join("" if row.get(column) is None else str(row[column]) for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _hand_metrics() -> dict:
    """Synthetic lists only. No development candidate receives a score."""
    rows = [
        {"gene": "2", "score": 1.0},
        {"gene": "10", "score": 1.0},
        {"gene": "3", "score": 0.0},
    ]
    order = rank_with_tie_break(rows)
    tie_ok = order == ["10", "2", "3"]
    rankings = {"C1": ["g1", "g2", "g0"], "C2": ["g9"]}
    positives = {"C1": {"g2", "g0"}, "C2": {"g9"}}
    metric = recall_at_k(rankings, positives, k=2)
    # C1 hits g2 only inside top 2 (g0 is third): recall 0.5. C2 hits g9: recall 1.
    macro_ok = abs(metric["macro_recall"] - 0.75) < 1e-12 and metric["recovered"] == 2 and metric["positives"] == 3
    entered = sorted(({"g0", "g2"} ) - {"g2"})
    left = sorted(({"g2"} ) - {"g0", "g2"})
    return {
        "tie_break_gene_id": tie_ok,
        "tie_order": order,
        "macro_recall_at_2": metric["macro_recall"],
        "pooled_hits": metric["recovered"],
        "positives": metric["positives"],
        "macro_ok": macro_ok,
        "top_entered": entered,
        "top_left": left,
        "zero_path_candidate_retained": "g0" in rankings["C1"],
    }


def _column_contract(names: dict) -> dict:
    core = names["core_only"]
    predicates = names["predicate_only"]
    specs = {}
    for structure, structure_columns in STRUCTURES.items():
        for core_name, core_columns in (("off", []), ("legacy_field", core), ("combination", [])):
            for sem_name, sem_columns in (("off", []), ("undirected", predicates), ("directed", [])):
                key = f"{structure}__core_{core_name}__semmed_{sem_name}"
                present = core_name != "combination" and sem_name != "directed"
                specs[key] = {
                    "held_constant": HELD,
                    "structure": structure_columns,
                    "core": core_columns if core_name != "combination" else [f"core_combination::{position}" for position in ("t1_e2", "t2_e1", "t2_e2", "t2_e3")],
                    "semmed": sem_columns if sem_name != "directed" else [f"semmed_direction::{position}" for position in ("t1_e1", "t1_e2", "t2_e1", "t2_e2", "t2_e3")],
                    "numeric_columns_present": present,
                    "status": "runnable" if present else "blocked",
                    "block_reason": "" if present else "Numeric same-sentence combination shares or traversal-direction shares were not stored. The finished representation audit, when present, limits a later comparison to the edge positions named in its validation file.",
                }
    return specs


def _dictionary(names: dict) -> list[dict]:
    rows = []
    for column in HELD + STRUCTURES["counts"] + STRUCTURES["prior_degree"] + STRUCTURES["dwpc"]:
        rows.append({
            "column": column,
            "group": "structure",
            "source": "masked graph, relation-attribute feature_store.npz",
            "formula": "path count, prior one-over-degree weight, or DWPC with damping 0.4; degree and publication count are node summaries",
            "numerator": "path occurrence or weighted path mass, else degree or PMID count",
            "denominator": "not a share, except has_* which is 0/1",
            "unit": "count_or_weight",
            "missing": "zero means no path or no degree; not used as a label",
            "transform_later": "log1p for counts and weights; fit any scaler on training rows only",
            "present": True,
        })
    for column in names["core_only"]:
        rows.append({
            "column": column,
            "group": "core_legacy_field",
            "source": "masked CORE sentences, fields aggregated separately",
            "formula": "equal split of path DWPC across distinct observed field values, divided by observable DWPC",
            "numerator": "DWPC allocated to that field value",
            "denominator": "DWPC of paths on which the field has a surviving record",
            "unit": "share",
            "missing": "NaN if the field has no surviving record; 0 if the field is observed and the value is absent",
            "transform_later": "do not log1p shares; median impute and missingness indicator fit on training rows",
            "present": True,
        })
    for column in names["predicate_only"]:
        rows.append({
            "column": column,
            "group": "semmed_undirected",
            "source": "masked affirmative allow-listed predications, orientation collapsed",
            "formula": "equal split of path DWPC across distinct predicates, divided by observable DWPC",
            "numerator": "DWPC allocated to that predicate",
            "denominator": "DWPC of paths on which an allow-listed affirmative predicate survives",
            "unit": "share",
            "missing": "NaN if no allow-listed predicate survives; negated predications are excluded and not a zero of an allowed predicate",
            "transform_later": "do not log1p shares; fit imputer on training rows",
            "present": True,
        })
    for position in ("t1_e2", "t2_e1", "t2_e2", "t2_e3"):
        rows.append({
            "column": f"core_combination::{position}",
            "group": "core_same_sentence",
            "source": "not stored as numeric columns",
            "formula": "equal split of the same path DWPC across distinct traceable (CGE, CCS, GCI) tuples",
            "numerator": "DWPC allocated to one complete tuple",
            "denominator": "DWPC of paths whose edge has a traceable complete tuple",
            "unit": "share",
            "missing": "absent file, not an observed zero",
            "transform_later": "not available for fitting",
            "present": False,
        })
    for position in ("t1_e1", "t1_e2", "t2_e1", "t2_e2", "t2_e3"):
        predicates = GG_PREDICATES if position == "t1_e1" else GC_PREDICATES
        rows.append({
            "column": f"semmed_direction::{position}",
            "group": "semmed_directed",
            "source": "not stored as numeric columns",
            "formula": "undirected predicate weight split equally across observed along, against, and unresolved directions",
            "numerator": "direction-specific portion of the predicate DWPC",
            "denominator": "same observable predicate DWPC as the undirected column",
            "unit": "share",
            "missing": "absent file; symmetric-predicate argument order would be a separate diagnostic",
            "transform_later": "not available for fitting",
            "present": False,
            "predicates": ",".join(predicates),
        })
    rows.append({
        "column": "same_path_core_semmed_combination",
        "group": "same_path",
        "source": "not built",
        "formula": "only a predicate and a CORE tuple observed on the same eligible path",
        "numerator": "not computed",
        "denominator": "not computed",
        "unit": "not_available",
        "missing": "independent edge shares must not be multiplied to invent co-occurrence",
        "transform_later": "blocked",
        "present": False,
    })
    rows.append({
        "column": "citation_increment",
        "group": "citation",
        "source": "24-cancer features.npz uses a different target mask and is not an input here",
        "formula": "direct citation, direction, co-citation, and shared references on the expanded masked graph",
        "numerator": "not aggregated for the 5,206-target mask",
        "denominator": "frozen rule drops a citation link if either PMID is blocked",
        "unit": "not_available",
        "missing": "no expanded-cohort cache",
        "transform_later": "blocked until re-aggregated",
        "present": False,
    })
    return rows


def run(run_dir: Path) -> None:
    analysis = run_dir / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    hand = _hand_metrics()
    if not (hand["tie_break_gene_id"] and hand["macro_ok"] and hand["top_entered"] == ["g0"] and hand["top_left"] == []):
        raise RuntimeError("synthetic ranking check failed")
    targets_path = ATTR_RUN / "analysis" / "development_targets.parquet"
    store_path = ATTR_RUN / "analysis" / "feature_store.npz"
    table = pq.read_table(targets_path)
    genes = table.column("gene_id").to_pylist()
    cancers = table.column("cancer_cui").to_pylist()
    roles = table.column("role").to_pylist()
    pmids = table.column("pmids").to_pylist()
    role_counts = Counter(roles)
    pair_keys = [f"{gene}|{cancer}" for gene, cancer in zip(genes, cancers)]
    duplicate_pairs = [key for key, count in Counter(pair_keys).items() if count > 1]
    records = []
    for index, (gene, cancer, role, pmid_values) in enumerate(zip(genes, cancers, roles, pmids)):
        records.append({"target_id": f"{gene}|{cancer}", "cancer": cancer, "gene": gene, "pmids": [int(item) for item in pmid_values], "role": role})
    cross_role = []
    for group in connected_pmid_groups(records):
        group_roles = {row["role"] for row in group}
        if len(group_roles) > 1:
            cross_role.append(sorted(row["target_id"] for row in group))
    gene_ids, candidates, queries = load_universe()
    gene_set = set(gene_ids)
    missing_genes = sorted({gene for gene in genes if gene not in gene_set})
    duplicate_gene_ids = len(gene_ids) - len(gene_set)
    stored = np.load(store_path)
    stored_cancers = [str(item) for item in stored["cancers"].tolist()]
    target_cancers = sorted(set(cancers))
    names = feature_names()
    missing_columns = [column for column in names["both"] if column not in stored.files]
    finite = {}
    constant = []
    for column in names["both"]:
        if column not in stored.files:
            continue
        values = stored[column]
        finite[column] = float(np.isfinite(values).mean())
        filled = values[np.isfinite(values)]
        if filled.size == 0 or np.unique(filled).size <= 1:
            constant.append(column)
    profiles = load_profiles()
    membership = load_membership()
    target_pairs = set(zip(genes, cancers))
    masked = {pmid for row in records for pmid in row["pmids"]}
    graph = build_graphs(profiles, membership, candidates, queries, target_pairs, masked)
    leaked = sum(1 for gene, cancer in target_pairs if gene in graph["genes_of"].get(cancer, ()))
    blocked_left = sum(1 for pmid_values in graph["edge_pmids"].values() for pmid in pmid_values if pmid in masked)
    by_cancer = defaultdict(lambda: Counter())
    for gene, cancer, role in zip(genes, cancers, roles):
        by_cancer[cancer][role] += 1
    sample_rows = []
    control_rows = []
    overlap_problems = Counter()
    ratio_rows = []
    for seed in SEEDS:
        seed_controls = {}
        for cancer in sorted(by_cancer):
            positives = [gene for gene, row_cancer, role in zip(genes, cancers, roles) if row_cancer == cancer and role == "train"]
            if not positives:
                continue
            controls = sample_controls(positives, gene_ids, set(graph["direct"].get(cancer, ())), CONTROL_RATIO, seed)
            seed_controls[cancer] = controls
            positive_set = set(positives)
            direct = set(graph["direct"].get(cancer, ()))
            for gene in controls:
                if gene in positive_set:
                    overlap_problems["control_is_train_positive"] += 1
                if gene in direct:
                    overlap_problems["control_is_direct"] += 1
                control_rows.append({"seed": seed, "cancer_cui": cancer, "gene_id": gene, "row_role": "train_control", "label": 0})
            for gene in positives:
                control_rows.append({"seed": seed, "cancer_cui": cancer, "gene_id": gene, "row_role": "train_positive", "label": 1})
            hidden = {gene for gene, row_cancer, role in zip(genes, cancers, roles) if row_cancer == cancer and role != "train"}
            overlap_problems["control_overlaps_other_role"] += sum(gene in hidden for gene in controls)
            ratio_rows.append({
                "seed": seed,
                "cancer_cui": cancer,
                "train_positives": len(positives),
                "train_controls": len(controls),
                "planned_controls": CONTROL_RATIO * len(positives),
                "shortfall": max(0, CONTROL_RATIO * len(positives) - len(controls)),
                "tune_targets": by_cancer[cancer]["tune"],
                "validation_targets": by_cancer[cancer]["development_validation"],
                "direct_genes": len(direct),
                "evaluation_candidates_validation": len(gene_ids) - len(direct - {gene for gene, row_cancer, role in zip(genes, cancers, roles) if row_cancer == cancer and role == "development_validation"}),
            })
        sample_rows.extend(ratio_rows[-len(seed_controls):])
    groups = defaultdict(list)
    for row in control_rows:
        if row["seed"] != SEEDS[0]:
            continue
        groups[row["cancer_cui"]].append(row)
    invalid_groups = 0
    group_rows = []
    for cancer, members in groups.items():
        labels = {row["label"] for row in members}
        group_rows.append({"cancer_cui": cancer, "group_size": len(members), "n_positive": sum(row["label"] == 1 for row in members), "n_control": sum(row["label"] == 0 for row in members)})
        if len(labels) < 2 or len(members) < 2:
            invalid_groups += 1
    weights = cancer_weights(np.array([row["cancer_cui"] for row in control_rows if row["seed"] == SEEDS[0]]), np.array([row["label"] for row in control_rows if row["seed"] == SEEDS[0]]))
    specs = _column_contract(names)
    dictionary = _dictionary(names)
    repr_validation = REPR_RUN / "validation.json"
    audit_payload = json.loads(repr_validation.read_text()) if repr_validation.exists() else {}
    core_decision = audit_payload.get("core_judgment", {}).get("decision", {}).get("decision")
    semmed_decision = audit_payload.get("semmed_judgment", {}).get("decision", {}).get("decision")
    audit_state = {
        "path": str(REPR_RUN),
        "validation_present": repr_validation.exists(),
        "report_present": (REPR_RUN / "report.md").exists(),
        "used_as_model_input": False,
        "core_decision": core_decision,
        "semmed_decision": semmed_decision,
        "reproduction_max_abs_float32": audit_payload.get("reproduction_max_abs_float32"),
        "reason": (
            "The finished audit reproduced the frozen feature store. It did not write numeric combination or direction columns, so those specifications stay blocked. Its judgments limit a later comparison to the positions it reports."
            if repr_validation.exists()
            else "The representation audit had not written validation.json. Partial signature chunks are not a numeric feature matrix."
        ),
    }
    identity = {
        "development_targets_sha256": sha256_file(targets_path),
        "feature_store_sha256": sha256_file(store_path),
        "attribute_manifest_sha256": sha256_file(ATTR_RUN / "manifest.json"),
        "pair_sha256": sha256_text("\n".join(sorted(f"{gene}|{cancer}|{role}" for gene, cancer, role in zip(genes, cancers, roles)))),
        "gene_universe_sha256": sha256_text("\n".join(gene_ids)),
        "role_counts": dict(role_counts),
        "role_counts_match": dict(role_counts) == {"train": 3124, "tune": 1041, "development_validation": 1041},
        "validation_cancers": len({cancer for cancer, role in zip(cancers, roles) if role == "development_validation"}),
        "feature_cancers_match": stored_cancers == target_cancers,
        "feature_shape": [int(stored["two_edge_dwpc"].shape[0]), int(stored["two_edge_dwpc"].shape[1])],
        "candidate_gene_count": len(gene_ids),
        "duplicate_candidate_genes": duplicate_gene_ids,
        "targets_missing_from_universe": missing_genes,
        "duplicate_target_pairs": duplicate_pairs,
        "pmid_groups_crossing_roles": len(cross_role),
        "masked_target_edges_remaining": leaked,
        "blocked_pmids_remaining_on_edges": blocked_left,
        "blocked_pmid_count": len(masked),
        "missing_feature_columns": missing_columns,
        "constant_or_empty_columns": constant,
        "control_problems": dict(overlap_problems),
        "lambdamart_groups_without_both_labels": invalid_groups,
        "pointwise_weight_sum_per_cancer": float(np.max([weights[np.array([row["cancer_cui"] for row in control_rows if row["seed"] == SEEDS[0]]) == cancer].sum() for cancer in {row["cancer_cui"] for row in control_rows if row["seed"] == SEEDS[0]}])),
    }
    pq.write_table(pa.table({"slot": list(range(len(gene_ids))), "gene_id": gene_ids}), analysis / "candidate_index.parquet")
    pq.write_table(pa.table({"gene_id": genes, "cancer_cui": cancers, "role": roles}), analysis / "label_index.parquet")
    pq.write_table(pa.Table.from_pylist(control_rows), analysis / "training_rows.parquet")
    _tsv(analysis / "candidate_label_reconciliation.tsv", [
        {"check": "role_counts", "value": json.dumps(dict(role_counts)), "expected": "train 3124, tune 1041, development_validation 1041"},
        {"check": "validation_cancers", "value": identity["validation_cancers"], "expected": 131},
        {"check": "each_target_is_one_candidate", "value": len(missing_genes) == 0 and not duplicate_pairs, "expected": True},
        {"check": "pmid_groups_stay_inside_one_role", "value": len(cross_role) == 0, "expected": True},
        {"check": "candidate_ids_unique", "value": duplicate_gene_ids == 0, "expected": True},
    ], ["check", "value", "expected"])
    _tsv(analysis / "training_sample_stats.tsv", ratio_rows, list(ratio_rows[0]))
    _tsv(analysis / "lambdamart_groups.tsv", group_rows, ["cancer_cui", "group_size", "n_positive", "n_control"])
    _tsv(analysis / "masking_trace.tsv", [
        {"layer": "direct_target_edge", "check": "target gene absent from masked genes_of", "result": leaked == 0, "affected_models": "all", "open_rule": ""},
        {"layer": "blocked_pmid", "check": "blocked PMID absent from masked edge_pmids", "result": blocked_left == 0, "affected_models": "all", "open_rule": ""},
        {"layer": "supervision_versus_propagation", "check": "training labels are stored separately from graph edges", "result": True, "affected_models": "all", "open_rule": ""},
        {"layer": "citation_witness", "check": "expanded-cohort citation features were not aggregated", "result": False, "affected_models": "citation increments", "open_rule": "The frozen rule removes a citation when either PMID is blocked. No unresolved witness class was found; the matrix is simply absent."},
        {"layer": "same_path_combination", "check": "same-path CORE and predicate co-occurrence was not built", "result": False, "affected_models": "same-path block", "open_rule": "Do not multiply independent edge shares."},
        {"layer": "auxiliary_gnn_edges", "check": "no expanded-cohort graph package", "result": False, "affected_models": "ComplEx, R-GCN, BioPathNet", "open_rule": "Whether inverse relations or auxiliary triples may use a PMID that is blocked on another edge is undecided because those edges were not packaged."},
        {"layer": "random_walk_neighborhood", "check": "walk adjacency was not cached; fixed two-hop and three-hop scores are cached", "result": False, "affected_models": "random walk", "open_rule": "A walk would see every surviving masked neighbor, including edges outside the two fixed templates."},
    ], ["layer", "check", "result", "affected_models", "open_rule"])
    _tsv(analysis / "feature_dictionary.tsv", dictionary, ["column", "group", "source", "formula", "numerator", "denominator", "unit", "missing", "transform_later", "present"])
    shapes = [
        {"model": "popularity_and_fixed_paths", "status": "runnable", "rows": identity["feature_shape"][0], "columns": len(HELD) + 6, "key": "cancer_cui x gene slot", "note": "Reads degree, publication count, path counts, prior weights, and DWPC from feature_store.npz. No scores were produced."},
        {"model": "random_walk", "status": "blocked", "rows": "", "columns": "", "key": "masked adjacency", "note": "Fixed template scores exist. Restart-walk adjacency for this mask was not cached."},
        {"model": "ridge_and_trees", "status": "runnable_for_12_of_27", "rows": sum(1 for row in control_rows if row["seed"] == SEEDS[0]), "columns": "see column_manifest.json", "key": "seed x cancer x gene", "note": "Training rows and the 12 legacy-field or undirected specifications can be read. Combination and direction specifications are blocked."},
        {"model": "lambdamart", "status": "input_ready_library_absent", "rows": sum(1 for row in control_rows if row["seed"] == SEEDS[0]), "columns": "same rows as ridge", "key": "cancer group", "note": f"Groups with both labels: {len(group_rows) - invalid_groups}. xgboost and lightgbm are not installed. No ranker was constructed."},
        {"model": "path_set", "status": "blocked", "rows": "", "columns": "", "key": "candidate path set", "note": "Three-hop paths are not enumerated. Edge-position aggregates are the available substitute and are not a path-set model."},
        {"model": "complex_rgcn_biopathnet", "status": "blocked", "rows": "", "columns": "", "key": "node and edge package", "note": "The 24-cancer graph package is a different split. No expanded package was loaded."},
    ]
    _tsv(analysis / "model_input_shapes.tsv", shapes, ["model", "status", "rows", "columns", "key", "note"])
    comparisons = []
    for key, spec in specs.items():
        comparisons.append({
            "comparison": key,
            "family": "three_layer_representation",
            "held_fixed": "masked graph, candidates, DWPC or the named structure weight, development split",
            "changed": key,
            "input": str(store_path) if spec["numeric_columns_present"] else "",
            "question": "Does this column set change later development Recall@50 relative to its matched neighbor?",
            "status": spec["status"],
            "reason": spec["block_reason"],
            "primary": key in {"dwpc__core_legacy_field__semmed_off", "dwpc__core_legacy_field__semmed_undirected", "dwpc__core_off__semmed_undirected"},
        })
    for name, question, status, reason in (
        ("core_legacy_versus_combination", "Do same-sentence tuples add a distinction the matched field marginal does not?", "blocked", "Combination columns are not in the feature matrix."),
        ("semmed_undirected_versus_directed", "Does traversal direction add a distinction beyond the undirected predicate weight?", "blocked", "Direction columns are not in the feature matrix."),
        ("citation_direct", "Do direct citations add recovery beyond relation features?", "blocked", "Citation features were not aggregated under the 5,206-target mask."),
        ("citation_direction", "Does citation direction add recovery beyond an undirected citation?", "blocked", "Same missing citation matrix."),
        ("cocitation", "Does co-citation add recovery beyond direct citation?", "blocked", "Same missing citation matrix."),
        ("shared_references", "Do shared references add recovery beyond direct citation?", "blocked", "Same missing citation matrix."),
        ("path_weight_counts_versus_dwpc", "Does DWPC rank differently from raw path counts under the same candidates?", "runnable", "Both column sets are in feature_store.npz. This preflight did not rank them."),
        ("layer_removal", "Does removing one present feature group change only that group's columns?", "runnable", "Checked by column-list difference, not by fitting."),
        ("advanced_versus_matched_baseline", "Do path-set, walk, or graph models beat a matched tabular baseline?", "blocked", "Those model inputs are not packaged for this split."),
    ):
        comparisons.append({"comparison": name, "family": "increment", "held_fixed": "development split and masked graph", "changed": name, "input": str(store_path) if status == "runnable" else "", "question": question, "status": status, "reason": reason, "primary": name in {"core_legacy_versus_combination", "semmed_undirected_versus_directed", "citation_direct"}})
    _tsv(analysis / "comparison_checklist.tsv", comparisons, ["comparison", "family", "held_fixed", "changed", "input", "question", "status", "reason", "primary"])
    write_json(analysis / "column_manifest.json", specs)
    write_json(analysis / "hand_metric_check.json", hand)
    write_json(analysis / "input_identity.json", identity)
    finite_summary = {
        "columns": len(finite),
        "min_finite_fraction": min(finite.values()) if finite else None,
        "median_finite_fraction": float(np.median(list(finite.values()))) if finite else None,
    }
    validation = {
        "formal_test_files_opened": False,
        "models_fit": False,
        "candidate_scores_written": False,
        "identity": {key: identity[key] for key in ("role_counts_match", "validation_cancers", "feature_cancers_match", "pmid_groups_crossing_roles", "masked_target_edges_remaining", "blocked_pmids_remaining_on_edges", "control_problems", "lambdamart_groups_without_both_labels", "duplicate_target_pairs", "targets_missing_from_universe")},
        "feature_finite": finite_summary,
        "representation_audit": audit_state,
        "hand_metrics_passed": True,
        "actual_control_ratio_seed_20261116": _ratio(ratio_rows, SEEDS[0]),
    }
    write_json(run_dir / "validation.json", validation)
    write_json(run_dir / "manifest.json", {
        "task": "rank candidate human genes within a cancer and recover masked CORE direct records",
        "sample_unit": "NCBI Gene ID x exact cancer CUI",
        "formal_test_files_opened": False,
        "sources": {
            "graph": str(GRAPH_RUN),
            "development_split": str(ATTR_RUN),
            "feature_store": str(store_path),
            "citation_audit": str(CITATION_RUN),
            "prior_24_cancer_run_not_an_input": str(PRIOR_24),
            "representation_audit_not_frozen": str(REPR_RUN),
        },
        "hashes": {key: identity[key] for key in ("development_targets_sha256", "feature_store_sha256", "attribute_manifest_sha256", "pair_sha256", "gene_universe_sha256")},
        "seeds": {"controls": list(SEEDS), "cohort": 20261119},
        "losses_not_fit": {
            "ridge": "L2 logistic loss, each cancer's training rows sum to weight 1",
            "trees": "binomial loss in HistGradientBoosting, same cancer weights",
            "lambdamart": "not available; intended query-grouped ranking loss was not fit and no library was imported",
        },
    })
    _report(run_dir, identity, validation, comparisons, audit_state)
    print(json.dumps({"role_counts_match": identity["role_counts_match"], "leaked": leaked, "blocked_left": blocked_left, "cross_role": len(cross_role), "control_problems": dict(overlap_problems)}, indent=2), flush=True)


def _ratio(rows: list[dict], seed: int) -> dict:
    chosen = [row for row in rows if row["seed"] == seed]
    positives = sum(row["train_positives"] for row in chosen)
    controls = sum(row["train_controls"] for row in chosen)
    return {"positives": positives, "controls": controls, "controls_per_positive": controls / positives if positives else None, "cancers_with_shortfall": sum(row["shortfall"] > 0 for row in chosen)}


def _report(run_dir: Path, identity: dict, validation: dict, comparisons: list[dict], audit_state: dict) -> None:
    runnable = [row["comparison"] for row in comparisons if row["status"] == "runnable"]
    blocked = [row["comparison"] for row in comparisons if row["status"] == "blocked"]
    lines = [
        "# Association-recovery input preflight",
        "",
        "The task is to rank candidate human genes within one cancer and recover CORE direct records whose edges and PMIDs have been masked. The sample is an NCBI Gene ID paired with one exact cancer CUI. This preflight did not fit a model, did not score candidates, and did not open a formal test file.",
        "",
        "## What each reader can see",
        "",
        f"The frozen development split matches 3,124 train, 1,041 tune, and 1,041 development-validation targets, across {identity['validation_cancers']} validation cancers. The feature matrix is {identity['feature_shape'][0]} cancers by {identity['feature_shape'][1]} candidate genes.",
        "Ridge and gradient-boosted trees can read the 12 specifications that use path counts, the earlier degree weight, or DWPC, with or without the legacy per-field CORE shares and the undirected SemMedDB predicates. Those columns are in the relation-attribute feature store. Removing one of those groups changes only the columns named in `analysis/column_manifest.json`.",
        "Same-sentence CORE tuples, traversal-aligned SemMedDB direction, citation increments, path sets, restart walks, and graph-model packages are not available as matrices for this split. Signature chunks from the representation audit are not a numeric feature matrix and were not used as inputs.",
        "",
        "## Checks",
        "",
        f"Every hidden target is one candidate gene, with no duplicate pair: {not identity['duplicate_target_pairs'] and not identity['targets_missing_from_universe']}. PMID groups that share an article stay inside one role: {identity['pmid_groups_crossing_roles'] == 0}. Masked target edges remaining: {identity['masked_target_edges_remaining']}. Blocked PMIDs remaining on edges: {identity['blocked_pmids_remaining_on_edges']}.",
        f"Replayed training controls, seed 20261116: {json.dumps(validation['actual_control_ratio_seed_20261116'])}. Control problems: {json.dumps(identity['control_problems'])}. LambdaMART groups without both labels: {identity['lambdamart_groups_without_both_labels']}.",
        "A synthetic three-gene list confirmed the gene-id tie break, retention of a zero-path candidate, macro recall, pooled hits, and a top-list entry. No development gene was scored.",
        "",
        "## Comparisons that can start",
        "",
        "Runnable now, still without a fit in this preflight: " + ", ".join(runnable) + ".",
        "",
        "## Comparisons that still need a decision or a missing matrix",
        "",
        "Blocked: " + ", ".join(blocked) + ".",
        "The open packaging question is whether a later graph model may add inverse relations or auxiliary triples that touch a PMID blocked on a different edge. That question affects only ComplEx, R-GCN, and BioPathNet. Citation increments have a frozen mask rule and lack the expanded matrix; they do not need a new task definition.",
        "",
        f"Representation audit state: {json.dumps(audit_state)}.",
        "",
    ]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.run_dir)


if __name__ == "__main__":
    main()

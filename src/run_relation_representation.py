"""Audit coarse CORE and SemMedDB features on the frozen development graph.

The run reproduces the saved relation-attribute features, then compares them
with same-sentence triples and traversal-aligned predicate direction. It does
not fit a model, rank candidates, or open the formal test partition.
"""

from __future__ import annotations

import argparse
import json
import resource
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.common import RESULTS_ROOT, normalise_pmid, sha256_file, sha256_text, write_json
from src.relation_attributes import (
    CORE_LEVELS,
    GC_PREDICATES,
    GG_PREDICATES,
    feature_names,
    normalise_core,
    position1_tables,
    score_query,
)
from src.relation_representation import (
    GC_POSITIONS,
    N_COMBO,
    POSITIONS,
    QUANT_PRIMARY,
    QUANT_SENSITIVITY,
    SYMMETRIC_PREDICATES,
    accumulate_fine,
    canonical_direction,
    classify_core_record,
    combination_share,
    direction_share,
    judge_representation,
    marginal_collision_groups,
    marginal_from_joint,
    quant_signature,
    shares_from_mass,
    undirected_from_directions,
)
from src.recovery_scope import candidate_slot_table
from src.run_association_recovery import (
    GRAPH_RUN,
    build_graphs,
    load_membership,
    load_profiles,
    load_universe,
)
from src.run_path_evidence import _as_bool, _connection, _entity

SOURCE_RUN = RESULTS_ROOT / "relation_attributes_development_20261117T1544Z"
GRAPH_DIR = GRAPH_RUN / "graph"


def _peak_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def _write_tsv(path: Path, rows: list[dict], columns: list[str]) -> None:
    lines = ["\t".join(columns)]
    for row in rows:
        lines.append("\t".join("" if row.get(column) is None else str(row.get(column)) for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _study_design() -> dict:
    return {
        "study": {
            "task": "relation_representation_audit",
            "trains_models": False,
            "opens_formal_test": False,
            "ranks_candidates": False,
        },
        "literature": {
            "marchesin_2023": {
                "doi": "10.1093/database/baad061",
                "supports": "CORE CGE, CCS, and GCI are aspects of one gene-cancer relation and were used jointly when inferring a gene role. This audit counts same-sentence (CGE, CCS, GCI) tuples that the per-field share discards. It does not infer new roles.",
            },
            "bakal_2018": {
                "doi": "10.1016/j.jbi.2018.05.003",
                "supports": "SemMedDB paths can be found without direction and then annotated with the original predicate and subject-object order. This audit keeps the same paths and changes only the assertion encoding.",
            },
            "himmelstein_baranzini_2015": {
                "doi": "10.1371/journal.pcbi.1004259",
                "supports": "Degree-weighted path counts are the weight used to compare representations. The audit reuses the damping-0.4 DWPC from the relation-attribute run and does not refit weights.",
            },
            "sosa_2024": {
                "doi": "10.1186/s13326-024-00308-z",
                "supports": "Topology and relation semantics can be intervened on separately. This audit freezes the masked graph and the path set, then changes only the relation encoding.",
            },
        },
        "frozen_from": str(SOURCE_RUN),
        "paths": {
            "two_hop": "candidate gene -> intermediate gene -> query cancer",
            "three_hop": "candidate gene -> intermediate cancer -> intermediate gene -> query cancer",
        },
        "core_complete_tuple": "A source row with a sentence identifier and a non-empty token in CGE, CCS, and GCI. NOTINF is explicit. An empty token is partial. No sentence identifier is untraceable.",
        "direction": "along traversal, against traversal, or unresolved, judged from the assertion's original subject and object. The undirected predicate weight is split equally across observed direction categories.",
        "symmetric_predicates": sorted(SYMMETRIC_PREDICATES),
        "quantization_decimals": {"primary": QUANT_PRIMARY, "sensitivity": QUANT_SENSITIVITY},
        "structural_sensitivity": "Approximate. Groups also share log1p(DWPC) rounded to 1 decimal.",
    }


def _verify_inputs(run_dir: Path) -> dict:
    targets = SOURCE_RUN / "analysis" / "development_targets.parquet"
    store = SOURCE_RUN / "analysis" / "feature_store.npz"
    manifest = SOURCE_RUN / "manifest.json"
    missing = [str(path) for path in (targets, store, manifest, GRAPH_DIR / "evidence_records.parquet") if not path.exists()]
    if missing:
        raise FileNotFoundError("missing frozen inputs: " + ", ".join(missing))
    table = pq.read_table(targets)
    roles = Counter(table.column("role").to_pylist())
    cancers = sorted(set(table.column("cancer_cui").to_pylist()))
    pairs = sorted(zip(table.column("gene_id").to_pylist(), table.column("cancer_cui").to_pylist(), table.column("role").to_pylist()))
    stored = np.load(store)
    stored_cancers = [str(item) for item in stored["cancers"].tolist()]
    feature_rows = int(stored["two_edge_dwpc"].shape[0])
    feature_columns = int(stored["two_edge_dwpc"].shape[1])
    del stored
    identity = {
        "development_targets_sha256": sha256_file(targets),
        "feature_store_sha256": sha256_file(store),
        "manifest_sha256": sha256_file(manifest),
        "role_counts": dict(roles),
        "cancers": len(cancers),
        "validation_cancers": len({cancer for _gene, cancer, role in pairs if role == "development_validation"}),
        "pair_sha256": sha256_text("\n".join(f"{gene}|{cancer}|{role}" for gene, cancer, role in pairs)),
        "feature_cancers_match_targets": stored_cancers == cancers,
        "feature_rows": feature_rows,
        "feature_columns": feature_columns,
    }
    expected = {"train": 3124, "tune": 1041, "development_validation": 1041}
    identity["role_counts_match"] = dict(roles) == expected and identity["validation_cancers"] == 131 and identity["feature_cancers_match_targets"]
    write_json(run_dir / "analysis" / "input_identity.json", identity)
    if not identity["role_counts_match"]:
        raise RuntimeError("frozen development split does not match the expected ids: " + json.dumps(identity))
    return {"table": table, "cancers": cancers, "identity": identity}


def _empty_edge() -> dict:
    return {
        "combinations": {},
        "directions": {},
        "negated": 0,
        "ineligible": 0,
        "orientable": 0,
        "unresolved": 0,
        "core_records": 0,
        "core_status": Counter(),
        "explicit_notinf": 0,
        "empty_label": 0,
        "old_core": {field: set() for field in CORE_LEVELS},
        "old_predicates": set(),
        "core_present": False,
    }


def _stream(surviving: set[tuple[str, str, str]], masked: set[int], target_pairs: set[tuple[str, str]]) -> tuple[dict, dict]:
    relations = {
        row["relation_id"]: row
        for row in pq.read_table(
            GRAPH_DIR / "relations.parquet",
            columns=["relation_id", "subject_entity_id", "object_entity_id", "predicate", "base_predicate", "negation", "connectivity_family"],
        ).to_pylist()
    }
    links = defaultdict(list)
    evidence = pq.read_table(GRAPH_DIR / "relation_evidence.parquet", columns=["relation_id", "evidence_id"])
    for relation_id, evidence_id in zip(evidence.column("relation_id").to_pylist(), evidence.column("evidence_id").to_pylist()):
        links[evidence_id].append(relation_id)
    edges: dict[tuple[str, str, str], dict] = {}
    seen: dict[tuple, set] = defaultdict(set)
    audit = Counter()
    parquet = pq.ParquetFile(GRAPH_DIR / "evidence_records.parquet")
    columns = ["evidence_id", "resource", "native_id", "pmid", "sentence_id", "path_eligible", "predicate", "negation", "cge_label", "ccs_label", "gci_label"]
    for batch in parquet.iter_batches(batch_size=200_000, columns=columns):
        rows = batch.to_pydict()
        for index, evidence_id in enumerate(rows["evidence_id"]):
            for relation_id in links.get(evidence_id, ()):
                relation = relations.get(relation_id)
                if relation is None:
                    continue
                key = _connection(relation)
                if key is None:
                    audit["unmapped_relation"] += 1
                    continue
                if key[0] == "gene_cancer" and (key[1], key[2]) in target_pairs:
                    audit["target_edge_records_excluded"] += 1
                    continue
                if key not in surviving:
                    audit["removed_edge_records_excluded"] += 1
                    continue
                pmid_text = normalise_pmid(rows["pmid"][index]) or ""
                pmid_number = int(pmid_text) if pmid_text.isdigit() else 0
                if pmid_number and pmid_number in masked:
                    audit["blocked_pmid_records_excluded"] += 1
                    continue
                native = str(rows["native_id"][index] or evidence_id)
                if native in seen[key]:
                    audit["duplicate_native_skipped"] += 1
                    continue
                seen[key].add(native)
                edge = edges.get(key)
                if edge is None:
                    edge = _empty_edge()
                    edges[key] = edge
                if rows["resource"][index] == "core":
                    classified = classify_core_record(rows["sentence_id"][index] or "", rows["cge_label"][index] or "", rows["ccs_label"][index] or "", rows["gci_label"][index] or "")
                    edge["core_records"] += 1
                    edge["core_present"] = True
                    edge["core_status"][classified["status"]] += 1
                    edge["explicit_notinf"] += int(classified["explicit_notinf"])
                    edge["empty_label"] += int(classified["empty_label"])
                    labels = {"cge": rows["cge_label"][index] or "", "ccs": rows["ccs_label"][index] or "", "gci": rows["gci_label"][index] or ""}
                    for field, label in labels.items():
                        edge["old_core"][field].add(normalise_core(field, label))
                    combo = classified["combination"]
                    if combo is not None:
                        slot = edge["combinations"].setdefault(combo, {"n": 0, "pmids": set(), "native_id": native, "pmid": pmid_text})
                        slot["n"] += 1
                        if pmid_text:
                            slot["pmids"].add(pmid_text)
                    audit[f"core_{classified['status']}"] += 1
                else:
                    predicate = relation.get("base_predicate") or relation.get("predicate") or ""
                    negated = _as_bool(relation.get("negation")) or _as_bool(rows["negation"][index])
                    allowed = predicate in (GG_PREDICATES if key[0] == "gene_gene" else GC_PREDICATES)
                    traversal = _as_bool(rows["path_eligible"][index]) and not negated
                    if negated:
                        edge["negated"] += 1
                        audit["semmed_negated"] += 1
                        continue
                    if not traversal or not allowed:
                        edge["ineligible"] += 1
                        audit["semmed_ineligible"] += 1
                        continue
                    subject = relation["subject_entity_id"]
                    object_ = relation["object_entity_id"]
                    if key[0] == "gene_cancer":
                        origin, destination = f"gene:{key[1]}", f"cancer:{key[2]}"
                    else:
                        origin, destination = f"gene:{key[1]}", f"gene:{key[2]}"
                    direction = canonical_direction(subject, object_, origin, destination)
                    bucket = edge["directions"].setdefault(predicate, {})
                    detail = bucket.setdefault(direction, {"n": 0, "native_id": native, "pmid": pmid_text, "subject": subject, "object": object_})
                    detail["n"] += 1
                    edge["old_predicates"].add(predicate)
                    if direction == "unresolved":
                        edge["unresolved"] += 1
                        audit["semmed_unresolved"] += 1
                    else:
                        edge["orientable"] += 1
                        audit["semmed_orientable"] += 1
    audit["edges_with_records"] = len(edges)
    return edges, dict(audit)


def _views(raw: dict[tuple[str, str, str], dict]) -> tuple[dict, dict]:
    views = {}
    old = {}
    for key, edge in raw.items():
        share = combination_share(set(edge["combinations"]))
        directions = {predicate: set(items) for predicate, items in edge["directions"].items()}
        views[key] = {
            "key": key,
            "combo_share": None if not edge["combinations"] else share,
            "marginal_share": None if not edge["combinations"] else marginal_from_joint(share),
            "directions": directions,
        }
        old[key] = {
            "core": edge["old_core"] if edge["core_present"] else None,
            "predicates": set(edge["old_predicates"]),
        }
    return views, old


def _matrices(views: dict, slots: dict[str, np.ndarray], reverse: dict[int, str]) -> dict:
    matrices = {}
    width = len(GC_PREDICATES) * 3
    for cancer, gene_slots in slots.items():
        combo = np.zeros((len(gene_slots), N_COMBO), dtype=np.float64)
        marginal = np.zeros((len(gene_slots), 12), dtype=np.float64)
        direction = np.zeros((len(gene_slots), width), dtype=np.float64)
        for local, slot in enumerate(gene_slots.tolist()):
            edge = views.get(("gene_cancer", reverse[int(slot)], cancer))
            if edge is None:
                continue
            if edge["combo_share"] is not None:
                combo[local] = edge["combo_share"]
                marginal[local] = edge["marginal_share"]
            if edge["directions"]:
                direction[local] = direction_share(edge["directions"], GC_PREDICATES, False)
        matrices[(cancer, "combo")] = combo
        matrices[(cancer, "marginal")] = marginal
        matrices[(cancer, "direction")] = direction
    return matrices


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    cast = left.astype(np.float32)
    gap = np.abs(cast - right)
    missing = np.isnan(cast) & np.isnan(right)
    gap = np.where(missing, 0.0, gap)
    if np.isnan(gap).any():
        return float("inf")
    return float(gap.max()) if gap.size else 0.0


def _vector(scored: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    return np.vstack([scored[name] for name in names])


def _core_names(position: str) -> list[str]:
    return [f"{position}_{field}_{level}" for field, levels in CORE_LEVELS.items() for level in levels]


def _predicate_names(position: str) -> list[str]:
    predicates = GG_PREDICATES if position == "t1_e1" else GC_PREDICATES
    return [f"{position}_pred_{name}" for name in predicates]


def _drop_notinf(share: np.ndarray) -> np.ndarray:
    kept = share.copy()
    for index, value in enumerate(share):
        if value == 0:
            continue
        cge, rem = divmod(index, 16)
        ccs, gci = divmod(rem, 4)
        if 1 in (cge, ccs, gci):
            kept[index] = 0.0
    total = float(kept.sum())
    if total > 0:
        kept /= total
    return kept


def _signature_rows(cancer: str, gene_ids: list[str], position: str, comparison: str, coarse: list[tuple], fine: list[tuple], slots: np.ndarray, path_count: np.ndarray, dwpc: np.ndarray) -> list[dict]:
    flags = []
    grouped = defaultdict(set)
    for left, right in zip(coarse, fine):
        grouped[left].add(right)
    multi = {key for key, values in grouped.items() if len(values) > 1}
    rows = []
    for local, slot in enumerate(slots.tolist()):
        if path_count[slot] <= 0:
            continue
        rows.append({
            "cancer": cancer,
            "gene_id": gene_ids[int(slot)],
            "position": position,
            "comparison": comparison,
            "quantization_decimals": QUANT_PRIMARY,
            "coarse_signature": json.dumps(coarse[local]),
            "fine_signature": json.dumps(fine[local]),
            "in_multi_fine_group": coarse[local] in multi,
            "path_count": int(path_count[slot]),
            "dwpc": float(dwpc[slot]),
        })
    return rows


def _collect_signatures(position: str, coarse_matrix: np.ndarray, fine_matrix: np.ndarray, active: np.ndarray, decimals: int) -> tuple[list[int], list[tuple], list[tuple]]:
    slots = np.flatnonzero(active)
    coarse = [quant_signature(coarse_matrix[:, slot], decimals) for slot in slots]
    fine = [quant_signature(fine_matrix[:, slot], decimals) for slot in slots]
    return slots.tolist(), coarse, fine


def run(run_dir: Path, max_cancers: int) -> None:
    analysis = run_dir / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    (analysis / "candidate_chunks").mkdir(exist_ok=True)
    (run_dir / "study_design.yaml").write_text(yaml.safe_dump(_study_design(), sort_keys=False), encoding="utf-8")
    identity = _verify_inputs(run_dir)
    gene_ids, candidates, queries = load_universe()
    if len(gene_ids) != identity["identity"]["feature_columns"]:
        raise RuntimeError("candidate-gene order does not match the feature store width")
    profiles = load_profiles()
    membership = load_membership()
    records = identity["table"].to_pylist()
    target_pairs = {(row["gene_id"], row["cancer_cui"]) for row in records}
    masked = set()
    for row in records:
        masked.update(int(item) for item in row["pmids"])
    cancers = identity["cancers"][:max_cancers] if max_cancers else identity["cancers"]
    graph = build_graphs(profiles, membership, candidates, queries, target_pairs, masked)
    leaked = [pair for pair in target_pairs if pair[0] in graph["genes_of"].get(pair[1], ())]
    if leaked:
        raise RuntimeError(f"masked target edges re-entered the graph: {len(leaked)}")
    raw, stream_audit = _stream(set(graph["edge_pmids"]), masked, target_pairs)
    views, old_attributes = _views(raw)
    index = {gene: slot for slot, gene in enumerate(gene_ids)}
    reverse = {slot: gene for gene, slot in index.items()}
    slots = candidate_slot_table(graph["genes_of"], index)
    print(f"[graph] edges={len(graph['edge_pmids'])} record_edges={len(raw)} rss_gb={_peak_gb():.2f}", flush=True)
    matrices = _matrices(views, slots, reverse)
    tables = position1_tables(old_attributes, graph["edge_pmids"], index, slots)
    stored = np.load(SOURCE_RUN / "analysis" / "feature_store.npz")
    cancers_of = defaultdict(set)
    deg_gc = defaultdict(int)
    for cancer, genes in graph["genes_of"].items():
        for gene in genes:
            cancers_of[gene].add(cancer)
            deg_gc[gene] += 1
    deg_gg = {gene: len(partners) for gene, partners in graph["neighbors"].items()}
    stored_index = {str(cancer): row for row, cancer in enumerate(stored["cancers"].tolist())}
    names = feature_names()
    reproduction = []
    restoration = []
    collision_rows = []
    target_rows = []
    chunk = []
    touched = {position: set() for position in POSITIONS}
    path_totals = Counter()
    per_cancer_collisions = []
    examples = []
    hidden = defaultdict(set)
    for row in records:
        hidden[row["cancer_cui"]].add((row["gene_id"], row["role"]))
    for query_number, cancer in enumerate(cancers):
        scored = score_query(cancer, graph["genes_of"], graph["neighbors"], cancers_of, index, slots, deg_gg, deg_gc, graph["edge_pmids"], graph["ages"], old_attributes, np.zeros(len(gene_ids)), tables)
        fine = accumulate_fine(cancer, graph["genes_of"], graph["neighbors"], cancers_of, index, slots, deg_gg, deg_gc, views, matrices, touched)
        dwpc_gap = max(abs(float(np.max(np.abs(fine["two_edge_dwpc"] - scored["two_edge_dwpc"])))), abs(float(np.max(np.abs(fine["three_edge_dwpc"] - scored["three_edge_dwpc"])))))
        if dwpc_gap > 1e-6:
            raise RuntimeError(f"fine DWPC diverged from the saved implementation at {cancer}: {dwpc_gap}")
        row_index = stored_index[cancer]
        gaps = {}
        for name in ("two_edge_paths", "three_edge_paths", "two_edge_dwpc", "three_edge_dwpc", *names["core_only"], *names["predicate_only"]):
            gaps[name] = _max_abs(scored[name], stored[name][row_index])
        worst = max(gaps.values())
        reproduction.append({"cancer": cancer, "max_abs_float32": worst})
        if worst > 1e-4:
            raise RuntimeError(f"saved features were not reproduced at {cancer}: {worst}")
        path_totals["two_edge_paths"] += int(scored["two_edge_paths"].sum())
        path_totals["three_edge_paths"] += int(scored["three_edge_paths"].sum())
        path_totals["two_edge_dwpc"] += float(scored["two_edge_dwpc"].sum())
        path_totals["three_edge_dwpc"] += float(scored["three_edge_dwpc"].sum())
        for position in GC_POSITIONS:
            path_totals[f"{position}_complete_paths"] += int(fine["complete_paths"][position].sum())
            path_totals[f"{position}_complete_dwpc"] += float(fine["complete_dwpc"][position].sum())
        for position in POSITIONS:
            path_totals[f"{position}_directed_paths"] += int(fine["directed_paths"][position].sum())
            path_totals[f"{position}_directed_dwpc"] += float(fine["directed_dwpc"][position].sum())
            observed = np.flatnonzero(fine["directed_dwpc"][position] > 0)
            if len(observed):
                restored = undirected_from_directions(shares_from_mass(fine["direction"][position], fine["directed_dwpc"][position])[observed])
                predicate = np.vstack([scored[name] for name in _predicate_names(position)]).T[observed]
                gap = float(np.nanmax(np.abs(np.nan_to_num(restored) - np.nan_to_num(predicate))))
                restoration.append({"cancer": cancer, "position": position, "kind": "semmed_direction", "max_abs": gap})
                if gap > 1e-5:
                    raise RuntimeError(f"direction shares do not restore predicate weight at {cancer} {position}: {gap}")
        for position in GC_POSITIONS:
            observed = np.flatnonzero(fine["complete_dwpc"][position] > 0)
            if len(observed) == 0:
                continue
            joint = shares_from_mass(fine["combo"][position], fine["complete_dwpc"][position])
            margin = shares_from_mass(fine["marginal"][position], fine["complete_dwpc"][position])
            projected = np.vstack([marginal_from_joint(np.nan_to_num(joint[slot])) for slot in observed])
            gap = float(np.max(np.abs(projected - np.nan_to_num(margin[observed]))))
            mass_gap = float(np.max(np.abs(np.nansum(joint[observed], axis=1) - 1.0)))
            restoration.append({"cancer": cancer, "position": position, "kind": "core_joint", "max_abs": max(gap, mass_gap)})
            if max(gap, mass_gap) > 1e-5:
                raise RuntimeError(f"joint projection failed at {cancer} {position}: {gap} {mass_gap}")
        cancer_collision = {"cancer": cancer}
        core_hit = set()
        sem_hit = set()
        for position in GC_POSITIONS:
            template = "two_hop" if position.startswith("t1") else "three_hop"
            old = _vector(scored, _core_names(position))
            joint = shares_from_mass(fine["combo"][position], fine["complete_dwpc"][position]).T
            margin = shares_from_mass(fine["marginal"][position], fine["complete_dwpc"][position]).T
            active = scored["two_edge_paths" if position.startswith("t1") else "three_edge_paths"] > 0
            _slots_old, coarse_old, fine_old = _collect_signatures(position, old, joint, active, QUANT_PRIMARY)
            summary_old = _summarise(coarse_old, fine_old)
            matched_active = fine["complete_dwpc"][position] > 0
            slots_m, coarse_m, fine_m = _collect_signatures(position, margin, joint, matched_active, QUANT_PRIMARY)
            summary_m = _summarise(coarse_m, fine_m)
            _sens = _collect_signatures(position, margin, joint, matched_active, QUANT_SENSITIVITY)
            summary_q = _summarise(_sens[1], _sens[2])
            collapsed = joint.copy()
            for slot in np.flatnonzero(matched_active):
                collapsed[:, slot] = _drop_notinf(np.nan_to_num(joint[:, slot]))
            summary_enc = _summarise(*_collect_signatures(position, margin, collapsed, matched_active, QUANT_PRIMARY)[1:])
            structural = []
            structural_fine = []
            for slot, coarse_key, fine_key in zip(slots_m, coarse_m, fine_m):
                structural.append((round(float(np.log1p(scored["three_edge_dwpc" if position.startswith("t2") else "two_edge_dwpc"][slot])), 1), coarse_key))
                structural_fine.append(fine_key)
            summary_struct = _summarise(structural, structural_fine)
            for comparison, summary in (
                ("old_versus_joint", summary_old),
                ("matched_joint", summary_m),
                ("quantization_1e-4", summary_q),
                ("notinf_collapsed", summary_enc),
                ("structural_bin_approximate", summary_struct),
            ):
                collision_rows.append({"resource": "core", "scope": "development_query", "template": template, "position": position, "comparison": comparison, "quantization": "1e-4" if comparison == "quantization_1e-4" else "1e-6", "cancer": cancer, **summary, "eligible_candidates": len(gene_ids)})
            cancer_collision[position] = summary_m["candidates_in_multi"]
            for slot, flag in zip(slots_m, _in_multi(coarse_m, fine_m)):
                if flag:
                    core_hit.add(slot)
            if summary_m["candidates_in_multi"] and len(examples) < 12:
                _add_candidate_example(examples, cancer, gene_ids, position, slots_m, coarse_m, fine_m, raw, "core_same_marginal_different_joint")
            chunk.extend(_signature_rows(cancer, gene_ids, position, "matched_joint", coarse_m, fine_m, np.array(slots_m), fine["complete_paths"][position], fine["complete_dwpc"][position]))
        for position in POSITIONS:
            template = "two_hop" if position.startswith("t1") else "three_hop"
            predicates = _vector(scored, _predicate_names(position))
            directed = shares_from_mass(fine["direction"][position], fine["directed_dwpc"][position]).T
            active = fine["directed_dwpc"][position] > 0
            slots_d, coarse_d, fine_d = _collect_signatures(position, predicates, directed, active, QUANT_PRIMARY)
            summary_d = _summarise(coarse_d, fine_d)
            summary_q = _summarise(*_collect_signatures(position, predicates, directed, active, QUANT_SENSITIVITY)[1:])
            asymmetric = _asymmetric_share(fine["direction"][position], fine["directed_dwpc"][position], position)
            summary_a = _summarise(*_collect_signatures(position, predicates, asymmetric, active, QUANT_PRIMARY)[1:])
            for comparison, summary in (("old_versus_direction", summary_d), ("quantization_1e-4", summary_q), ("asymmetric_predicates", summary_a)):
                collision_rows.append({"resource": "semmeddb", "scope": "development_query", "template": template, "position": position, "comparison": comparison, "quantization": "1e-4" if comparison == "quantization_1e-4" else "1e-6", "cancer": cancer, **summary, "eligible_candidates": len(gene_ids)})
            cancer_collision[f"sem_{position}"] = summary_d["candidates_in_multi"]
            for slot, flag in zip(slots_d, _in_multi(coarse_d, fine_d)):
                if flag:
                    sem_hit.add(slot)
            if summary_d["candidates_in_multi"] and len(examples) < 24:
                _add_candidate_example(examples, cancer, gene_ids, position, slots_d, coarse_d, fine_d, raw, "semmed_same_predicate_different_direction")
        for resource, comparison, count in (
            ("core", "matched_joint_unique", len(core_hit)),
            ("semmeddb", "direction_unique", len(sem_hit)),
        ):
            collision_rows.append({"resource": resource, "scope": "development_query", "template": "either", "position": "any", "comparison": comparison, "quantization": "1e-6", "cancer": cancer, "coarse_groups": 0, "multi_fine_groups": 0, "candidates_in_multi": count, "fine_signatures_in_multi": 0, "candidates": count, "fine_signatures": 0, "eligible_candidates": len(gene_ids)})
        per_cancer_collisions.append(cancer_collision)
        for gene, role in hidden.get(cancer, ()):
            slot = index[gene]
            row = _target_row(cancer, gene, role, slot, scored, fine)
            row["in_core_matched_collision"] = int(slot in core_hit)
            row["in_semmed_direction_collision"] = int(slot in sem_hit)
            target_rows.append(row)
        if len(chunk) >= 200_000:
            _flush_chunk(analysis, query_number, chunk)
            chunk = []
        if (query_number + 1) % 10 == 0 or query_number < 3:
            print(f"[audit] {query_number + 1}/{len(cancers)} {cancer} rss_gb={_peak_gb():.2f}", flush=True)
    if chunk:
        _flush_chunk(analysis, len(cancers), chunk)
    _write_outputs(run_dir, raw, stream_audit, path_totals, collision_rows, target_rows, per_cancer_collisions, examples, reproduction, restoration, touched, leaked, masked, identity["identity"], max_cancers)
    print("[done] relation representation audit", flush=True)


def _summarise(coarse: list[tuple], fine: list[tuple]) -> dict:
    from src.relation_representation import collision_summary

    summary = collision_summary(coarse, fine)
    return summary


def _asymmetric_share(mass: np.ndarray, denominator: np.ndarray, position: str) -> np.ndarray:
    """Direction shares with symmetric predicates removed, laid out as features by gene."""
    share = shares_from_mass(mass, denominator)
    predicates = GG_PREDICATES if position == "t1_e1" else GC_PREDICATES
    out = share.copy()
    for index, name in enumerate(predicates):
        if name in SYMMETRIC_PREDICATES:
            out[:, index * 3: index * 3 + 3] = np.nan
    return out.T


def _in_multi(coarse: list[tuple], fine: list[tuple]) -> list[bool]:
    from src.relation_representation import in_multi_group

    return in_multi_group(coarse, fine)


def _add_candidate_example(examples: list[dict], cancer: str, gene_ids: list[str], position: str, slots: list[int], coarse: list[tuple], fine: list[tuple], raw: dict, kind: str) -> None:
    grouped = defaultdict(list)
    for slot, left, right in zip(slots, coarse, fine):
        grouped[left].append((slot, right))
    for left, members in sorted(grouped.items(), key=lambda item: json.dumps(item[0])):
        signatures = {json.dumps(item[1]) for item in members}
        if len(signatures) < 2:
            continue
        for slot, right in sorted(members, key=lambda item: gene_ids[item[0]])[:2]:
            gene = gene_ids[slot]
            examples.append({
                "kind": kind,
                "cancer": cancer,
                "gene": gene,
                "template": "two_hop" if position.startswith("t1") else "three_hop",
                "position": position,
                "coarse": json.dumps(left),
                "fine": json.dumps(right),
                "source_keys": _source_keys(raw, gene, cancer),
            })
        return


def _source_keys(raw: dict, gene: str, cancer: str) -> str:
    edge = raw.get(("gene_cancer", gene, cancer))
    if edge is None:
        return ""
    keys = []
    for combo, detail in sorted(edge["combinations"].items()):
        keys.append(f"{detail['native_id']}|{detail['pmid']}|{'/'.join(combo)}")
    for predicate, directions in sorted(edge["directions"].items()):
        for direction, detail in sorted(directions.items()):
            keys.append(f"{detail['native_id']}|{detail['pmid']}|{predicate}|{direction}|{detail['subject']}->{detail['object']}")
    return ";".join(keys[:6])


def _target_row(cancer: str, gene: str, role: str, slot: int, scored: dict, fine: dict) -> dict:
    has_two = int(scored["two_edge_paths"][slot] > 0)
    has_three = int(scored["three_edge_paths"][slot] > 0)
    complete = {position: int(fine["complete_dwpc"][position][slot] > 0) for position in GC_POSITIONS}
    directed = {position: int(fine["directed_dwpc"][position][slot] > 0) for position in POSITIONS}
    return {
        "role": role,
        "cancer": cancer,
        "gene": gene,
        "has_two_hop": has_two,
        "has_three_hop": has_three,
        "has_any_path": int(has_two or has_three),
        "complete_tuple_any_position": int(any(complete.values())),
        "directed_predicate_any_position": int(any(directed.values())),
        **{f"complete_{position}": value for position, value in complete.items()},
        **{f"directed_{position}": value for position, value in directed.items()},
    }


def _flush_chunk(analysis: Path, index: int, rows: list[dict]) -> None:
    if not rows:
        return
    pq.write_table(pa.Table.from_pylist(rows), analysis / "candidate_chunks" / f"part_{index:04d}.parquet")


def _write_outputs(run_dir, raw, stream_audit, path_totals, collision_rows, target_rows, per_cancer, examples, reproduction, restoration, touched, leaked, masked, identity, max_cancers) -> None:
    analysis = run_dir / "analysis"
    record_rows = _record_rows(raw, stream_audit)
    edge_rows = _edge_rows(raw, path_totals, touched)
    _write_tsv(analysis / "record_coverage.tsv", record_rows, ["resource", "status", "numerator", "denominator", "denominator_definition", "unit"])
    _write_tsv(analysis / "edge_position_summary.tsv", edge_rows, ["scope", "template", "position", "measure", "numerator", "denominator", "denominator_definition", "unit"])
    _write_tsv(analysis / "candidate_collisions.tsv", collision_rows, ["resource", "scope", "template", "position", "comparison", "quantization", "cancer", "coarse_groups", "multi_fine_groups", "candidates_in_multi", "fine_signatures_in_multi", "candidates", "fine_signatures", "eligible_candidates"])
    target_summary = _target_summary(target_rows)
    _write_tsv(analysis / "development_target_coverage.tsv", target_summary, ["role", "measure", "numerator", "denominator", "denominator_definition"])
    _write_tsv(analysis / "traceable_examples.tsv", examples, ["kind", "cancer", "gene", "template", "position", "coarse", "fine", "source_keys"])
    _write_tsv(analysis / "data_dictionary.tsv", _dictionary(), ["table", "column", "meaning"])
    core_judgment, semmed_judgment = _judgments(collision_rows, target_rows)
    validation = {
        "formal_test_files_opened": False,
        "models_trained": False,
        "rankings_written": False,
        "input_identity": identity,
        "partial_cancer_limit": max_cancers or None,
        "targets_absent_as_direct_edges": len(leaked) == 0,
        "blocked_pmids": len(masked),
        "stream": stream_audit,
        "reproduction_max_abs_float32": max(row["max_abs_float32"] for row in reproduction) if reproduction else None,
        "restoration_max_abs": max(row["max_abs"] for row in restoration) if restoration else None,
        "peak_rss_gb": round(_peak_gb(), 3),
        "core_judgment": core_judgment,
        "semmed_judgment": semmed_judgment,
    }
    write_json(run_dir / "validation.json", validation)
    write_json(run_dir / "manifest.json", {
        "source_run": str(SOURCE_RUN),
        "graph": str(GRAPH_RUN),
        "formal_test_files_opened": False,
        "input_identity": identity,
        "literature": _study_design()["literature"],
    })
    _write_report(run_dir, record_rows, edge_rows, collision_rows, target_summary, examples, validation)


def _record_rows(raw: dict, stream_audit: dict) -> list[dict]:
    core_records = sum(edge["core_records"] for edge in raw.values())
    rows = []
    for status in ("traceable_complete", "partial", "untraceable"):
        rows.append({"resource": "core", "status": status, "numerator": stream_audit.get(f"core_{status}", 0), "denominator": core_records, "denominator_definition": "deduplicated CORE source rows on surviving masked edges", "unit": "source_record"})
    rows.append({"resource": "core", "status": "explicit_notinf_flag", "numerator": sum(edge["explicit_notinf"] for edge in raw.values()), "denominator": core_records, "denominator_definition": "deduplicated CORE source rows on surviving masked edges; a row may also have another status", "unit": "source_record"})
    rows.append({"resource": "core", "status": "empty_label_flag", "numerator": sum(edge["empty_label"] for edge in raw.values()), "denominator": core_records, "denominator_definition": "deduplicated CORE source rows on surviving masked edges; a row may also have another status", "unit": "source_record"})
    multi = 0
    comparable_connections = 0
    for edge in raw.values():
        if edge["combinations"]:
            comparable_connections += 1
        if len(edge["combinations"]) > 1:
            multi += 1
    rows.append({"resource": "core", "status": "connections_with_multiple_complete_tuples", "numerator": multi, "denominator": comparable_connections, "denominator_definition": "surviving gene-cancer connections with at least one traceable complete tuple", "unit": "connection"})
    marginal_records = []
    for edge in raw.values():
        if len(edge["combinations"]) < 1:
            continue
        share = combination_share(set(edge["combinations"]))
        marginal_records.append((quant_signature(marginal_from_joint(share), QUANT_PRIMARY), tuple(sorted(edge["combinations"]))))
    grouped = marginal_collision_groups(marginal_records)
    rows.append({"resource": "core", "status": "connections_same_marginal_different_joint", "numerator": grouped["connections_in_those_groups"], "denominator": len(marginal_records), "denominator_definition": "connections with a traceable complete tuple; numerator is connections whose marginal matches a different joint", "unit": "connection"})
    sem_orientable = stream_audit.get("semmed_orientable", 0)
    sem_unresolved = stream_audit.get("semmed_unresolved", 0)
    sem_negated = stream_audit.get("semmed_negated", 0)
    sem_ineligible = stream_audit.get("semmed_ineligible", 0)
    sem_denominator = sem_orientable + sem_unresolved + sem_negated + sem_ineligible
    for status, numerator in (("orientable", sem_orientable), ("unresolved", sem_unresolved), ("negated", sem_negated), ("ineligible", sem_ineligible)):
        rows.append({"resource": "semmeddb", "status": status, "numerator": numerator, "denominator": sem_denominator, "denominator_definition": "deduplicated SemMedDB predications on surviving masked edges", "unit": "predication"})
    both = 0
    both_symmetric = 0
    both_assertions = 0
    groups = 0
    for edge in raw.values():
        for predicate, directions in edge["directions"].items():
            groups += 1
            kinds = {name for name in directions if name in {"along", "against"}}
            if len(kinds) > 1:
                both += 1
                both_assertions += sum(directions[name]["n"] for name in kinds)
                if predicate in SYMMETRIC_PREDICATES:
                    both_symmetric += 1
    rows.append({"resource": "semmeddb", "status": "connection_predicate_both_directions", "numerator": both, "denominator": groups, "denominator_definition": "surviving connection-predicate groups with an affirmative allow-listed predication", "unit": "connection_predicate"})
    rows.append({"resource": "semmeddb", "status": "both_directions_on_symmetric_predicates", "numerator": both_symmetric, "denominator": both, "denominator_definition": "connection-predicate groups that have both along and against traversal", "unit": "connection_predicate"})
    rows.append({"resource": "semmeddb", "status": "assertions_in_both_direction_groups", "numerator": both_assertions, "denominator": sem_orientable, "denominator_definition": "orientable affirmative allow-listed predications", "unit": "predication"})
    rows.append({"resource": "core", "status": "blocked_or_removed_records", "numerator": stream_audit.get("blocked_pmid_records_excluded", 0) + stream_audit.get("target_edge_records_excluded", 0) + stream_audit.get("removed_edge_records_excluded", 0), "denominator": stream_audit.get("blocked_pmid_records_excluded", 0) + stream_audit.get("target_edge_records_excluded", 0) + stream_audit.get("removed_edge_records_excluded", 0) + core_records + sem_denominator, "denominator_definition": "records seen while streaming; numerator is excluded from every later table", "unit": "source_record"})
    return rows


def _edge_rows(raw, path_totals, touched) -> list[dict]:
    rows = []
    for family in ("gene_gene", "gene_cancer"):
        members = [edge for key, edge in raw.items() if key[0] == family]
        with_records = len(members)
        complete = sum(1 for edge in members if edge["combinations"])
        directed = sum(1 for edge in members if any(set(edge["directions"].get(predicate, {})) & {"along", "against"} for predicate in edge["directions"]))
        rows.append({"scope": "masked_graph", "template": "", "position": family, "measure": "connections_with_eligible_records", "numerator": with_records, "denominator": with_records, "denominator_definition": "connections retained in edge_pmids that still have a surviving source row", "unit": "connection"})
        if family == "gene_cancer":
            rows.append({"scope": "masked_graph", "template": "", "position": family, "measure": "connections_with_complete_tuple", "numerator": complete, "denominator": with_records, "denominator_definition": "gene-cancer connections with a surviving source row", "unit": "connection"})
        rows.append({"scope": "masked_graph", "template": "", "position": family, "measure": "connections_with_orientable_predicate", "numerator": directed, "denominator": with_records, "denominator_definition": "connections with a surviving source row", "unit": "connection"})
    for position in POSITIONS:
        template = "two_hop" if position.startswith("t1") else "three_hop"
        total_paths = path_totals["two_edge_paths"] if position.startswith("t1") else path_totals["three_edge_paths"]
        total_dwpc = path_totals["two_edge_dwpc"] if position.startswith("t1") else path_totals["three_edge_dwpc"]
        rows.append({"scope": "development_queries", "template": template, "position": position, "measure": "connections_used", "numerator": len(touched[position]), "denominator": len(touched[position]), "denominator_definition": "distinct connections traversed at this position by a development-query path", "unit": "connection"})
        for measure, key in (("complete_tuple_paths", f"{position}_complete_paths"), ("directed_predicate_paths", f"{position}_directed_paths")):
            if key not in path_totals:
                continue
            rows.append({"scope": "development_queries", "template": template, "position": position, "measure": measure, "numerator": path_totals[key], "denominator": total_paths, "denominator_definition": "path occurrences summed over candidates in the audited development queries", "unit": "path_occurrence"})
        for measure, key in (("complete_tuple_dwpc", f"{position}_complete_dwpc"), ("directed_predicate_dwpc", f"{position}_directed_dwpc")):
            if key not in path_totals:
                continue
            rows.append({"scope": "development_queries", "template": template, "position": position, "measure": measure, "numerator": f"{path_totals[key]:.6f}", "denominator": f"{total_dwpc:.6f}", "denominator_definition": "DWPC mass summed over candidates in the audited development queries", "unit": "dwpc_mass"})
    rows.append({"scope": "development_queries", "template": "two_hop", "position": "", "measure": "path_occurrences", "numerator": path_totals["two_edge_paths"], "denominator": path_totals["two_edge_paths"], "denominator_definition": "all two-hop path occurrences for audited queries, including paths with no attribute", "unit": "path_occurrence"})
    rows.append({"scope": "development_queries", "template": "three_hop", "position": "", "measure": "path_occurrences", "numerator": path_totals["three_edge_paths"], "denominator": path_totals["three_edge_paths"], "denominator_definition": "all three-hop path occurrences for audited queries, including paths with no attribute", "unit": "path_occurrence"})
    return rows


def _target_summary(rows: list[dict]) -> list[dict]:
    output = []
    for role in ("train", "tune", "development_validation"):
        chosen = [row for row in rows if row["role"] == role]
        denominator = {"train": 3124, "tune": 1041, "development_validation": 1041}[role]
        observable = sum(row["has_any_path"] for row in chosen)
        for measure, value in (
            ("targets_in_audited_queries", len(chosen)),
            ("targets_with_any_path", observable),
            ("targets_with_complete_tuple", sum(row["complete_tuple_any_position"] for row in chosen)),
            ("targets_with_directed_predicate", sum(row["directed_predicate_any_position"] for row in chosen)),
            ("targets_in_core_matched_collision", sum(row.get("in_core_matched_collision", 0) for row in chosen)),
            ("targets_in_semmed_direction_collision", sum(row.get("in_semmed_direction_collision", 0) for row in chosen)),
        ):
            output.append({"role": role, "measure": measure, "numerator": value, "denominator": denominator, "denominator_definition": "all development targets in this role, including targets with no remaining path"})
        output.append({"role": role, "measure": "targets_with_complete_tuple_given_a_path", "numerator": sum(row["complete_tuple_any_position"] for row in chosen), "denominator": observable, "denominator_definition": "targets in this role with at least one remaining path"})
    return output


def _judgments(collision_rows: list[dict], target_rows: list[dict]) -> tuple[dict, dict]:
    def pack(resource: str, comparison: str, unique_comparison: str, sensitivity: str) -> dict:
        primary = [row for row in collision_rows if row["comparison"] == comparison and row["resource"] == resource]
        unique = [row for row in collision_rows if row["comparison"] == unique_comparison and row["resource"] == resource]
        queries = {row["cancer"] for row in unique if row["candidates_in_multi"]}
        pairs = sum(row["candidates_in_multi"] for row in unique)
        by_query = Counter()
        by_position = Counter()
        for row in unique:
            by_query[row["cancer"]] += row["candidates_in_multi"]
        for row in primary:
            by_position[row["position"]] += row["candidates_in_multi"]
        top3 = sum(count for _cancer, count in by_query.most_common(3))
        position_total = sum(by_position.values())
        largest_position = max(by_position.values()) if by_position else 0
        family_positions = {(row["cancer"], row["position"]) for row in primary}
        position_pairs = sum(row["candidates_in_multi"] for row in primary)
        quant_pairs = sum(row["candidates_in_multi"] for row in collision_rows if row["resource"] == resource and row["comparison"] == "quantization_1e-4" and (row["cancer"], row["position"]) in family_positions)
        sens_pairs = sum(row["candidates_in_multi"] for row in collision_rows if row["resource"] == resource and row["comparison"] == sensitivity and (row["cancer"], row["position"]) in family_positions)
        return {
            "queries_with_collision": len(queries),
            "candidate_pairs_in_collision": pairs,
            "validation_targets_in_collision": 0,
            "top3_query_share_of_colliding_pairs": (top3 / pairs) if pairs else 0.0,
            "largest_position_share_of_colliding_rows": (largest_position / position_total) if position_total else 0.0,
            "encoding_sensitivity_retention": (sens_pairs / position_pairs) if position_pairs else 0.0,
            "quantization_sensitivity_retention": (quant_pairs / position_pairs) if position_pairs else 0.0,
            "positions": dict(by_position),
            "top_queries": by_query.most_common(5),
        }
    core = pack("core", "matched_joint", "matched_joint_unique", "notinf_collapsed")
    semmed = pack("semmeddb", "old_versus_direction", "direction_unique", "asymmetric_predicates")
    validation_cancers = {row["cancer"] for row in target_rows if row["role"] == "development_validation"}
    core["validation_targets_in_collision"] = sum(row.get("in_core_matched_collision", 0) for row in target_rows if row["role"] == "development_validation")
    semmed["validation_targets_in_collision"] = sum(row.get("in_semmed_direction_collision", 0) for row in target_rows if row["role"] == "development_validation")
    core["validation_observable_targets"] = sum(1 for row in target_rows if row["role"] == "development_validation" and row["has_any_path"])
    semmed["validation_observable_targets"] = core["validation_observable_targets"]
    core["decision"] = judge_representation(core)
    semmed["decision"] = judge_representation(semmed)
    core["validation_cancers"] = len(validation_cancers)
    return core, semmed


def _dictionary() -> list[dict]:
    return [
        {"table": "record_coverage", "column": "denominator", "meaning": "The denominator is stated in denominator_definition. Explicit NOTINF and empty-label rows are flags and can overlap a status."},
        {"table": "edge_position_summary", "column": "unit", "meaning": "connection is a deduplicated masked edge. path_occurrence counts each candidate path once. dwpc_mass sums the same paths' degree-weighted counts."},
        {"table": "candidate_collisions", "column": "candidates_in_multi", "meaning": "Candidate-query pairs whose quantized coarse signature is shared by at least one other fine signature in the same query and position. eligible_candidates is the full 13,571-gene denominator."},
        {"table": "candidate_collisions", "column": "comparison", "meaning": "matched_joint uses the marginal of the same complete tuples. old_versus_joint compares the saved per-field features with the joint. old_versus_direction compares undirected predicates with traversal direction. structural_bin_approximate also bins log1p(DWPC)."},
        {"table": "development_target_coverage", "column": "denominator", "meaning": "Role totals are 3124, 1041, and 1041. The conditional row uses targets that still have a path."},
        {"table": "traceable_examples", "column": "source_keys", "meaning": "native source id, PMID, and the stored tuple or subject-object ids. No sentence text is reconstructed."},
    ]


def _write_report(run_dir, record_rows, edge_rows, collision_rows, target_summary, examples, validation) -> None:
    def pick(resource: str, status: str):
        for row in record_rows:
            if row["resource"] == resource and row["status"] == status:
                return row
        return None
    core_records = pick("core", "traceable_complete")
    core_connections = pick("core", "connections_same_marginal_different_joint")
    sem_records = pick("semmeddb", "orientable")
    sem_connections = pick("semmeddb", "connection_predicate_both_directions")
    lines = [
        "# Relation representation audit",
        "",
        "This audit freezes the masked development graph, the development split, and the DWPC weights. It does not train a model or write a ranking. The formal test partition was not opened.",
        "",
        "## CORE",
        "",
        f"Comparable same-sentence tuples: {core_records['numerator']} of {core_records['denominator']} surviving CORE source rows.",
        f"Connections whose field marginal matches a different joint: {core_connections['numerator']} of {core_connections['denominator']} connections with a complete tuple.",
        f"Judgment: {validation['core_judgment']['decision']['decision']}. " + " ".join(validation["core_judgment"]["decision"]["reasons"]),
        f"Candidate-query pairs inside a matched collision: {validation['core_judgment']['candidate_pairs_in_collision']} across {validation['core_judgment']['queries_with_collision']} queries. Largest position share {validation['core_judgment']['largest_position_share_of_colliding_rows']:.2f}. Top-three query share {validation['core_judgment']['top3_query_share_of_colliding_pairs']:.2f}.",
        f"Top positions: {validation['core_judgment']['positions']}. Top queries: {validation['core_judgment']['top_queries']}.",
        "",
        "## SemMedDB",
        "",
        f"Orientable affirmative allow-listed predications: {sem_records['numerator']} of {sem_records['denominator']} surviving predications.",
        f"Connection-predicate groups with both traversal directions: {sem_connections['numerator']} of {sem_connections['denominator']}.",
        f"Judgment: {validation['semmed_judgment']['decision']['decision']}. " + " ".join(validation["semmed_judgment"]["decision"]["reasons"]),
        f"Candidate-query pairs inside a direction collision: {validation['semmed_judgment']['candidate_pairs_in_collision']} across {validation['semmed_judgment']['queries_with_collision']} queries. Largest position share {validation['semmed_judgment']['largest_position_share_of_colliding_rows']:.2f}. Top-three query share {validation['semmed_judgment']['top3_query_share_of_colliding_pairs']:.2f}.",
        f"Top positions: {validation['semmed_judgment']['positions']}. Top queries: {validation['semmed_judgment']['top_queries']}.",
        "",
        "## Checks",
        "",
        f"Saved feature reproduction, maximum absolute float32 gap: {validation['reproduction_max_abs_float32']}. Direction and joint restoration gap: {validation['restoration_max_abs']}. Masked target edges absent: {validation['targets_absent_as_direct_edges']}.",
        "Matched collisions use one quantization of 1e-6. The 1e-4 rows and the NOTINF-collapsed or asymmetric-predicate rows are sensitivities. Structural bins are approximate.",
        "A positive decision means the finer encoding separates candidates widely enough to justify one later fixed-path comparison. It is not an estimate of Recall@50.",
        "",
        "Traceable examples: " + str(len(examples)) + ".",
        "",
    ]
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-cancers", type=int, default=0)
    args = parser.parse_args()
    run(args.run_dir, args.max_cancers)


if __name__ == "__main__":
    main()

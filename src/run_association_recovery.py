"""Development association recovery on the masked CORE–SemMedDB graph.

The formal test partition is not read. Protocol metadata for this graph
version does not register a holdout, so the run draws a development split
from the extracted CORE candidate pool and stops at development validation.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml

from src.association_recovery import assign_partitions, connected_pmid_groups, rank_with_tie_break, recall_at_k, sample_controls
from src.common import REPO_ROOT, RESULTS_ROOT, StageTimer, ensure_dir, read_json, write_json
from src.path_evidence import template_weights
from src.recovery_models import (
    RelationGCN,
    TemplatePathNet,
    fit_logistic,
    fit_trees,
    macro_from_scores,
    prepare_features,
    propagate_templates,
)
from src.recovery_scope import (
    COHORT_SIZE,
    CONTRASTS,
    CONTROL_RATIO,
    CORE_ATTRIBUTES,
    EXPECTED_BOTH,
    EXPECTED_CORE_ONLY,
    EXPECTED_CORE_PAIRS,
    EXPECTED_THREE_EDGE,
    EXPECTED_TWO_EDGE,
    FEATURE_BLOCKS,
    POSITIONS,
    PREDICATES,
    PRIMARY_SEED,
    SEEDS,
    LazyTuples,
    build_motif_block,
    candidate_slot_table,
    citation_between,
    sum_adjacent_motifs,
    combine_templates,
    select_development_cancers,
    stratified_targets,
    targets_per_cancer,
    three_edge_scores,
    two_edge_scores,
)

GRAPH_RUN = RESULTS_ROOT / "joint_graph_and_evidence_audit_20261116T0404Z"
PATH_AUDIT = RESULTS_ROOT / "path_evidence_audit_20261116T1310Z"
COMPLETION = RESULTS_ROOT / "path_evidence_audit_completion_20261116T1402Z"
CITATION_AUDIT = RESULTS_ROOT / "citation_path_evidence_audit_20261116T1647Z"
REFERENCE_YEAR = 2024
PREDICATE_INDEX = {name: index for index, name in enumerate(PREDICATES)}
CORE_WIDTH = len(CORE_ATTRIBUTES)
PREDICATE_WIDTH = len(PREDICATES)
VECTOR_WIDTH = CORE_WIDTH + PREDICATE_WIDTH * 5


def _edge_key(family: str, left: str, right: str) -> tuple[str, str, str]:
    return (family, left, right)


def _vector_views(vector: np.ndarray) -> dict[str, np.ndarray]:
    cursor = CORE_WIDTH
    collapsed = vector[cursor: cursor + PREDICATE_WIDTH]
    cursor += PREDICATE_WIDTH
    with_ab = vector[cursor: cursor + PREDICATE_WIDTH]
    cursor += PREDICATE_WIDTH
    against_ab = vector[cursor: cursor + PREDICATE_WIDTH]
    cursor += PREDICATE_WIDTH
    with_ba = vector[cursor: cursor + PREDICATE_WIDTH]
    cursor += PREDICATE_WIDTH
    against_ba = vector[cursor: cursor + PREDICATE_WIDTH]
    return {
        "core": vector[:CORE_WIDTH],
        "collapsed": collapsed,
        "with_ab": with_ab,
        "against_ab": against_ab,
        "with_ba": with_ba,
        "against_ba": against_ba,
    }


def _add_core_label(vector: np.ndarray, label: str, names: tuple[str, str]) -> None:
    if label == names[0]:
        vector[CORE_ATTRIBUTES.index(names[0])] += 1
    elif label == names[1]:
        vector[CORE_ATTRIBUTES.index(names[1])] += 1


def load_universe() -> tuple[list[str], set[str], set[str]]:
    genes = pq.read_table(GRAPH_RUN / "analysis" / "candidate_genes.parquet", columns=["gene_id", "status"])
    gene_ids = [gene for gene, status in zip(genes.column("gene_id").to_pylist(), genes.column("status").to_pylist()) if status == "resolved_human"]
    queries = set(pq.read_table(GRAPH_RUN / "analysis" / "cancer_queries.parquet", columns=["cancer_cui"]).column("cancer_cui").to_pylist())
    return gene_ids, set(gene_ids), queries


def load_profiles():
    columns = [
        "family", "endpoint_a", "endpoint_b", "support", "n_core_sentences", "n_semmed_traversal",
        "n_pmid_core", "n_pmid_traversal", "n_core_records_without_pmid", "n_semmed_records_without_pmid",
    ]
    table = pq.read_table(PATH_AUDIT / "analysis" / "edge_profiles.parquet", columns=columns)
    return table.to_pydict()


def load_membership() -> dict[tuple[str, str, str], dict[str, list[int]]]:
    table = pq.read_table(
        CITATION_AUDIT / "analysis" / "edge_pmid_membership.parquet",
        columns=["family", "endpoint_a", "endpoint_b", "pmid", "in_core", "in_semmed_traversal"],
    )
    grouped: dict[tuple[str, str, str], dict[str, list[int]]] = defaultdict(lambda: {"core": [], "trav": [], "all": []})
    families = table.column("family").to_pylist()
    left = table.column("endpoint_a").to_pylist()
    right = table.column("endpoint_b").to_pylist()
    pmids = table.column("pmid").to_numpy()
    in_core = table.column("in_core").to_numpy()
    in_trav = table.column("in_semmed_traversal").to_numpy()
    for index, family in enumerate(families):
        bucket = grouped[_edge_key(family, left[index], right[index])]
        pmid = int(pmids[index])
        bucket["all"].append(pmid)
        if bool(in_core[index]):
            bucket["core"].append(pmid)
        if bool(in_trav[index]):
            bucket["trav"].append(pmid)
    return grouped


def extract_core_pairs(profiles, membership, candidates: set[str], queries: set[str]) -> tuple[list[dict], dict]:
    rows = []
    semmed_direct = 0
    for index, family in enumerate(profiles["family"]):
        if family != "gene_cancer":
            continue
        gene = profiles["endpoint_a"][index]
        cancer = profiles["endpoint_b"][index]
        if gene not in candidates or cancer not in queries:
            continue
        traversal = int(profiles["n_semmed_traversal"][index])
        if traversal > 0:
            semmed_direct += 1
        if int(profiles["n_core_sentences"][index]) <= 0:
            continue
        bucket = membership.get(_edge_key(family, gene, cancer), {"core": [], "trav": [], "all": []})
        core_pmids = sorted(set(bucket["core"]))
        sem_pmids = sorted(set(bucket["trav"]))
        rows.append({
            "gene_id": gene,
            "cancer_cui": cancer,
            "target_id": f"{gene}|{cancer}",
            "core_pmids": core_pmids,
            "core_pmid_count": len(core_pmids),
            "core_records_without_pmid": int(profiles["n_core_records_without_pmid"][index]),
            "missing_core_pmid_provenance": len(core_pmids) == 0,
            "semmed_direct": traversal > 0,
            "semmed_pmids": sem_pmids,
            "semmed_pmid_count": len(sem_pmids),
            "pmids": sorted(set(core_pmids) | set(sem_pmids)),
        })
    rows.sort(key=lambda row: (row["cancer_cui"], row["gene_id"]))
    both = sum(1 for row in rows if row["semmed_direct"])
    summary = {
        "core_direct_pairs": len(rows),
        "both_resources": both,
        "core_only": len(rows) - both,
        "missing_core_pmid_provenance": sum(1 for row in rows if row["missing_core_pmid_provenance"]),
        "semmed_direct_pairs_in_candidate_query_universe": semmed_direct,
        "anchors_match": len(rows) == EXPECTED_CORE_PAIRS and both == EXPECTED_BOTH and len(rows) - both == EXPECTED_CORE_ONLY,
    }
    return rows, summary


def write_candidate_table(path: Path, rows: list[dict]) -> None:
    table = pa.Table.from_pylist([{
        "gene_id": row["gene_id"],
        "cancer_cui": row["cancer_cui"],
        "core_pmids": row["core_pmids"],
        "core_pmid_count": row["core_pmid_count"],
        "core_records_without_pmid": row["core_records_without_pmid"],
        "missing_core_pmid_provenance": row["missing_core_pmid_provenance"],
        "semmed_direct": row["semmed_direct"],
        "semmed_pmids": row["semmed_pmids"],
        "semmed_pmid_count": row["semmed_pmid_count"],
    } for row in rows])
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def coverage_by_cancer() -> dict[str, dict]:
    found = {}
    handle = (COMPLETION / "analysis" / "candidate_coverage_by_query.tsv").open(encoding="utf-8")
    header = handle.readline().rstrip("\n").split("\t")
    for line in handle:
        values = dict(zip(header, line.rstrip("\n").split("\t")))
        found[values["query_cui"]] = {
            "paths": int(values["three_edge_joint_paths"]),
            "two_reached": int(values["two_edge_joint_reached"]),
            "direct_joint": int(values["direct_joint"]),
        }
    handle.close()
    return found


def build_split(rows: list[dict], coverage: dict[str, dict], run_dir: Path) -> dict:
    by_cancer = defaultdict(list)
    for row in rows:
        by_cancer[row["cancer_cui"]].append(row)
    cohort_rows = []
    for cancer, pairs in by_cancer.items():
        stats = coverage.get(cancer, {"paths": 0, "two_reached": 0})
        cohort_rows.append({
            "cancer": cancer,
            "n_core": len(pairs),
            "paths": stats["paths"],
            "two_reached": stats["two_reached"],
        })
    cohort = select_development_cancers(cohort_rows, COHORT_SIZE)
    selected_pairs = []
    background = {}
    for cancer in cohort["cancers"]:
        pairs = by_cancer[cancer]
        n_take = targets_per_cancer(len(pairs))
        chosen = stratified_targets(pairs, n_take, PRIMARY_SEED)
        selected_pairs.extend(chosen)
        background[cancer] = len(pairs) - len(chosen)
    records = [{
        "target_id": row["target_id"],
        "cancer": row["cancer_cui"],
        "gene": row["gene_id"],
        "pmids": row["pmids"],
        "semmed_direct": row["semmed_direct"],
    } for row in selected_pairs]
    assignment = assign_partitions(records)
    groups = connected_pmid_groups(records)
    role_counts = defaultdict(int)
    for role in assignment.values():
        role_counts[role] += 1
    total = max(len(records), 1)
    manifest = {
        "provenance": "fresh_development_split",
        "formal_holdout_registered_for_this_graph": False,
        "evidence": [
            "config/study_design.yaml sets evaluation.outcome_benchmark to not_run_in_this_phase.",
            "No development split manifest, allowlist, or reserved-target count is present in the audit manifests.",
            "The committed analysis configuration has an audit case sample, not a gene-cancer recovery partition.",
            "A new graph version was not treated as permission to open previously reserved targets. No such target file is registered for this version.",
        ],
        "selected_cancers": cohort["cancers"],
        "path_cap": cohort["path_cap"],
        "relaxations": cohort["relaxations"],
        "extracted_core_pairs": len(rows),
        "selected_targets": len(records),
        "background_pairs_by_cancer": background,
        "role_counts": dict(role_counts),
        "role_fractions": {role: role_counts[role] / total for role in ("train", "tune", "development_validation")},
        "pmid_groups": len(groups),
        "largest_pmid_group": max((len(group) for group in groups), default=0),
        "roles_are_disjoint": len(assignment) == len(records) and len(role_counts) <= 3,
        "pmid_groups_stay_inside_one_role": all(len({assignment[item["target_id"]] for item in group}) == 1 for group in groups),
        "shared_with_semmed": sum(1 for row in records if row["semmed_direct"]),
        "core_only": sum(1 for row in records if not row["semmed_direct"]),
        "seeds": list(SEEDS),
        "control_ratio": CONTROL_RATIO,
    }
    by_id = {row["target_id"]: row for row in selected_pairs}
    target_rows = []
    for record in records:
        source = by_id[record["target_id"]]
        target_rows.append({
            "gene_id": record["gene"],
            "cancer_cui": record["cancer"],
            "role": assignment[record["target_id"]],
            "semmed_direct": bool(source["semmed_direct"]),
            "pmids": source["pmids"],
            "pmid_group_size": next(len(group) for group in groups if any(item["target_id"] == record["target_id"] for item in group)),
        })
    pq.write_table(pa.Table.from_pylist(target_rows), run_dir / "analysis" / "development_targets.parquet")
    write_json(run_dir / "analysis" / "development_manifest.json", manifest)
    return {"manifest": manifest, "records": records, "assignment": assignment, "targets": target_rows}


def write_study_design(path: Path, split: dict, summary: dict) -> None:
    payload = {
        "task": "association_recovery_development",
        "status": "frozen_before_model_scores",
        "formal_test": "sealed_not_opened",
        "development_provenance": split["manifest"]["provenance"],
        "extracted_core_pairs": summary,
        "cohort": split["manifest"]["selected_cancers"],
        "selected_targets": split["manifest"]["selected_targets"],
        "role_counts": split["manifest"]["role_counts"],
        "seeds": list(SEEDS),
        "primary_seed": PRIMARY_SEED,
        "control_ratio": CONTROL_RATIO,
        "combination_rule": "log1p(two_edge_score) + log1p(three_edge_score)",
        "degree_weight": "two-edge uses the intermediate gene's candidate-neighbour degree; three-edge uses the intermediate cancer's gene degree",
        "tree_model": "sklearn HistGradientBoostingClassifier, pointwise. LambdaMART is not fit because xgboost and lightgbm are not installed.",
        "ridge_model": "L2 logistic regression",
        "combined_feature_block": "D plus citation direction, both witness groups, and predicate orientation",
        "contrasts": [list(item) for item in CONTRASTS],
        "feature_blocks": {name: list(columns) for name, columns in FEATURE_BLOCKS.items()},
        "reference_year_for_age": REFERENCE_YEAR,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _survives(n_core, n_trav, n_core_without, n_sem_without, n_pmid_trav, remaining_core, remaining_trav, drop_pair: bool) -> tuple[bool, bool]:
    if drop_pair:
        return False, False
    core_ok = n_core > 0 and (bool(remaining_core) or n_core_without > 0)
    sem_ok = n_trav > 0 and (bool(remaining_trav) or (n_sem_without > 0 and n_trav > n_pmid_trav))
    return core_ok, sem_ok


def build_graphs(profiles, membership, candidates: set[str], queries: set[str], target_pairs: set[tuple[str, str]], masked_pmids: set[int]):
    genes_of = defaultdict(set)
    core_of = defaultdict(set)
    sem_of = defaultdict(set)
    neighbors = defaultdict(set)
    direct = defaultdict(set)
    edge_pmids = {}
    ages = {}
    unmasked_genes = defaultdict(set)
    unmasked_neighbors = defaultdict(set)
    for index, family in enumerate(profiles["family"]):
        left = profiles["endpoint_a"][index]
        right = profiles["endpoint_b"][index]
        key = _edge_key(family, left, right)
        bucket = membership.get(key, {"core": [], "trav": [], "all": []})
        core_pmids = {int(item) for item in bucket["core"]}
        trav_pmids = {int(item) for item in bucket["trav"]}
        qualifying = core_pmids | trav_pmids
        support = profiles["support"][index]
        if family == "gene_cancer" and support != "no_traversal_support":
            unmasked_genes[right].add(left)
        if family == "gene_gene" and support != "no_traversal_support":
            unmasked_neighbors[left].add(right)
            unmasked_neighbors[right].add(left)
        drop = family == "gene_cancer" and (left, right) in target_pairs
        remaining_core = {pmid for pmid in core_pmids if pmid not in masked_pmids}
        remaining_trav = {pmid for pmid in trav_pmids if pmid not in masked_pmids}
        core_ok, sem_ok = _survives(
            int(profiles["n_core_sentences"][index]),
            int(profiles["n_semmed_traversal"][index]),
            int(profiles["n_core_records_without_pmid"][index]),
            int(profiles["n_semmed_records_without_pmid"][index]),
            int(profiles["n_pmid_traversal"][index]),
            remaining_core,
            remaining_trav,
            drop,
        )
        if family == "gene_cancer" and right in queries and (int(profiles["n_core_sentences"][index]) > 0 or int(profiles["n_semmed_traversal"][index]) > 0):
            if left in candidates:
                direct[right].add(left)
        if not core_ok and not sem_ok:
            continue
        kept = tuple(sorted((remaining_core | remaining_trav)))
        edge_pmids[key] = kept
        if family == "gene_cancer":
            genes_of[right].add(left)
            if core_ok:
                core_of[right].add(left)
            if sem_ok:
                sem_of[right].add(left)
        else:
            neighbors[left].add(right)
            neighbors[right].add(left)
    return {
        "genes_of": dict(genes_of),
        "core_of": dict(core_of),
        "sem_of": dict(sem_of),
        "neighbors": dict(neighbors),
        "direct": dict(direct),
        "edge_pmids": edge_pmids,
        "ages": ages,
        "unmasked_genes": dict(unmasked_genes),
        "unmasked_neighbors": dict(unmasked_neighbors),
    }


def attach_years(graph, run_timer: StageTimer) -> None:
    needed = {pmid for pmids in graph["edge_pmids"].values() for pmid in pmids}
    years = {}
    parquet = pq.ParquetFile(CITATION_AUDIT / "analysis" / "endpoint_metadata.parquet")
    for batch in parquet.iter_batches(columns=["pmid", "publication_year", "year_missing"], batch_size=200_000):
        pmid_values = batch.column(0).to_numpy()
        year_values = batch.column(1).to_numpy()
        missing = batch.column(2).to_numpy(zero_copy_only=False)
        for pmid, year, flag in zip(pmid_values, year_values, missing):
            pmid = int(pmid)
            if pmid in needed and not bool(flag) and int(year) > 0:
                years[pmid] = int(year)
    for key, pmids in graph["edge_pmids"].items():
        observed = sorted(REFERENCE_YEAR - years[pmid] for pmid in pmids if pmid in years)
        graph["ages"][key] = observed[len(observed) // 2] if observed else None
    run_timer.mark("publication_years", pmids=len(needed), years=len(years))


def load_assertion_vectors(graph) -> dict[tuple[str, str, str], np.ndarray]:
    vectors = {key: np.zeros(VECTOR_WIDTH, dtype=np.float64) for key in graph["edge_pmids"]}
    parquet = pq.ParquetFile(CITATION_AUDIT / "analysis" / "assertion_orientations.parquet")
    columns = [
        "family", "endpoint_a", "endpoint_b", "resource", "predicate", "negation", "record_path_eligible",
        "alignment_a_to_b", "alignment_b_to_a", "cge_label", "ccs_label", "gci_label", "gcc_label", "gcs_has_type",
    ]
    for batch in parquet.iter_batches(columns=columns, batch_size=100_000):
        rows = batch.to_pydict()
        for index, family in enumerate(rows["family"]):
            key = _edge_key(family, rows["endpoint_a"][index], rows["endpoint_b"][index])
            vector = vectors.get(key)
            if vector is None or rows["negation"][index] == "true":
                continue
            if rows["resource"][index] == "core":
                _add_core_label(vector, rows["cge_label"][index], ("cge_up", "cge_down"))
                _add_core_label(vector, rows["ccs_label"][index], ("ccs_progression", "ccs_regression"))
                _add_core_label(vector, rows["gci_label"][index], ("gci_causality", "gci_observation"))
                _add_core_label(vector, rows["gcc_label"][index], ("gcc_expression", "gcc_other"))
                has_type = rows["gcs_has_type"][index]
                if has_type == "BIOMARKER":
                    vector[CORE_ATTRIBUTES.index("type_biomarker")] += 1
                elif has_type == "ONCOGENE":
                    vector[CORE_ATTRIBUTES.index("type_oncogene")] += 1
                elif has_type == "TSG":
                    vector[CORE_ATTRIBUTES.index("type_tsg")] += 1
                continue
            predicate = rows["predicate"][index]
            slot = PREDICATE_INDEX.get(predicate)
            if slot is None or not rows["record_path_eligible"][index]:
                continue
            vector[CORE_WIDTH + slot] += 1
            for alignment, offset in ((rows["alignment_a_to_b"][index], 1), (rows["alignment_b_to_a"][index], 3)):
                if alignment == "with_traversal":
                    vector[CORE_WIDTH + offset * PREDICATE_WIDTH + slot] += 1
                elif alignment == "against_traversal":
                    vector[CORE_WIDTH + (offset + 1) * PREDICATE_WIDTH + slot] += 1
    return vectors


def load_citation_index(masked_pmids: set[int], needed_pmids: set[int]):
    table = pq.read_table(CITATION_AUDIT / "analysis" / "directed_citation_links.parquet")
    citing = table.column("citing_pmid").to_numpy().astype(np.int64, copy=False)
    cited = table.column("cited_pmid").to_numpy().astype(np.int64, copy=False)
    masked = np.fromiter(masked_pmids, dtype=np.int64, count=len(masked_pmids)) if masked_pmids else np.array([], dtype=np.int64)
    if len(masked):
        keep = ~(np.isin(citing, masked) | np.isin(cited, masked))
        citing = citing[keep]
        cited = cited[keep]
    needed = np.fromiter(needed_pmids, dtype=np.int64, count=len(needed_pmids))
    out_keep = np.isin(citing, needed)
    in_keep = np.isin(cited, needed)

    def grouped(src, dst, keep):
        src = src[keep]
        dst = dst[keep]
        order = np.argsort(src, kind="mergesort")
        src = src[order]
        dst = dst[order]
        unique, starts = np.unique(src, return_index=True)
        starts = list(starts) + [len(src)]
        return {int(pmid): dst[starts[index]: starts[index + 1]] for index, pmid in enumerate(unique)}

    outgoing = grouped(citing, cited, out_keep)
    incoming = grouped(cited, citing, in_keep)
    return outgoing, incoming, citing, cited


def _relation_slice(vector: np.ndarray, direction: str) -> np.ndarray:
    views = _vector_views(vector)
    if direction == "ab":
        return np.concatenate([views["core"], views["collapsed"], views["with_ab"], views["against_ab"]])
    return np.concatenate([views["core"], views["collapsed"], views["with_ba"], views["against_ba"]])


def _blank_features(n_query: int, n_gene: int) -> dict[str, np.ndarray]:
    features = {}
    for name in FEATURE_BLOCKS["combined"]:
        if name.startswith("two_edge") or name.startswith("three_edge") or name.startswith("candidate_") or name.startswith("query_") or name.startswith("predicate_") or name.startswith("orientation_") or name.startswith("cge_") or name.startswith("ccs_") or name.startswith("gci_") or name.startswith("gcc_") or name.startswith("type_"):
            features[name] = np.zeros((n_query, n_gene), dtype=np.float64)
        else:
            features[name] = np.full((n_query, n_gene), np.nan, dtype=np.float64)
    return features


def _finalize_observation(total_paths: np.ndarray, observed: np.ndarray, values: np.ndarray) -> np.ndarray:
    result = np.full(total_paths.shape, np.nan, dtype=np.float64)
    confirmed = (total_paths > 0) & (observed == total_paths)
    positive = (observed > 0) & (observed < total_paths) & (values > 0)
    result[confirmed | positive] = values[confirmed | positive]
    return result


def _load_feature_checkpoint(path: Path):
    if not path.exists():
        return None, []
    loaded = np.load(path, allow_pickle=True)
    done = [str(item) for item in loaded["__done_cancers"].tolist()]
    features = {key: np.array(loaded[key], copy=True) for key in loaded.files if key != "__done_cancers"}
    return features, done


def _save_feature_checkpoint(path: Path, features: dict, done: list[str]) -> None:
    temporary = path.with_name("partial_features_writing.npz")
    payload = dict(features)
    payload["__done_cancers"] = np.asarray(done)
    np.savez(temporary, **payload)
    temporary.replace(path)


def fill_features(graph, vectors, outgoing, incoming, gene_ids: list[str], cancers: list[str], candidates: set[str], timer: StageTimer, checkpoint: Path | None = None):
    index = {gene: slot for slot, gene in enumerate(gene_ids)}
    n_gene = len(gene_ids)
    n_query = len(cancers)
    features = _blank_features(n_query, n_gene)
    done: list[str] = []
    if checkpoint is not None:
        loaded, done = _load_feature_checkpoint(checkpoint)
        if loaded is not None:
            features = loaded
            print(f"[query_features] resumed {len(done)}/{n_query}", flush=True)
    done_set = set(done)
    for name in ("_core_counts", "_sem_counts", "_walk_two", "_walk_three"):
        features.setdefault(name, np.zeros((n_query, n_gene)))
    motif_cache: dict = {}
    cancers_of = defaultdict(set)
    for cancer, genes in graph["genes_of"].items():
        for gene in genes:
            cancers_of[gene].add(cancer)
    slots = {
        "joint": candidate_slot_table(graph["genes_of"], index),
        "core": candidate_slot_table(graph["core_of"], index),
        "sem": candidate_slot_table(graph["sem_of"], index),
    }
    degree = np.zeros(n_gene, dtype=np.float64)
    publications = np.zeros(n_gene, dtype=np.float64)
    gene_pmids = defaultdict(set)
    for (family, left, right), pmids in graph["edge_pmids"].items():
        if family == "gene_gene":
            for gene in (left, right):
                if gene in index:
                    degree[index[gene]] += 1
                gene_pmids[gene].update(pmids)
        elif right in set(cancers) or True:
            if left in index:
                degree[index[left]] += 1
            gene_pmids[left].update(pmids)
    for gene, pmids in gene_pmids.items():
        if gene in index:
            publications[index[gene]] = float(len(pmids))
    out_view = LazyTuples(outgoing)
    in_view = LazyTuples(incoming)
    for query_slot, cancer in enumerate(cancers):
        if cancer in done_set:
            print(f"[query_features] cancer={cancer} resumed", flush=True)
            continue
        two_counts, two_weighted, two_walk = two_edge_scores(graph["genes_of"], graph["neighbors"], cancer, candidates, index)
        three_counts, three_weighted, three_walk = three_edge_scores(
            graph["genes_of"], cancers_of, cancer, candidates, index, slots["joint"]
        )
        core_counts, _core_weighted, _core_walk = three_edge_scores(
            graph["core_of"], _cancers_from(graph["core_of"]), cancer, candidates, index, slots["core"]
        )
        sem_counts, _sem_weighted, _sem_walk = three_edge_scores(
            graph["sem_of"], _cancers_from(graph["sem_of"]), cancer, candidates, index, slots["sem"]
        )
        features["two_edge_paths"][query_slot] = two_counts
        features["three_edge_paths"][query_slot] = three_counts
        features["two_edge_degree_weighted"][query_slot] = two_weighted
        features["three_edge_degree_weighted"][query_slot] = three_weighted
        features["candidate_degree"][query_slot] = degree
        features["candidate_publications"][query_slot] = publications
        features["query_degree"][query_slot] = float(len(graph["genes_of"].get(cancer, ())))
        query_pmids = set()
        for gene in graph["genes_of"].get(cancer, ()):
            query_pmids.update(graph["edge_pmids"].get(("gene_cancer", gene, cancer), ()))
        features["query_publications"][query_slot] = float(len(query_pmids))
        _fill_two_edge(
            features, query_slot, cancer, graph, vectors, out_view, in_view, index, two_counts
        )
        _fill_three_edge(
            features, query_slot, cancer, graph, vectors, outgoing, incoming, out_view, in_view, index, candidates, motif_cache
        )
        features["_core_counts"][query_slot] = core_counts
        features["_sem_counts"][query_slot] = sem_counts
        features["_walk_two"][query_slot] = two_walk
        features["_walk_three"][query_slot] = three_walk
        done.append(cancer)
        if checkpoint is not None:
            _save_feature_checkpoint(checkpoint, features, done)
        print(
            f"[query_features] cancer={cancer} completed={len(done)}/{n_query} two_edge_paths={int(two_counts.sum())}",
            flush=True,
        )
        timer.mark(
            "query_features",
            cancer=cancer,
            two_edge_paths=int(two_counts.sum()),
            three_edge_paths=int(round(float(three_counts.sum()))),
        )
    return features, index


def _cancers_from(genes_of: dict[str, set[str]]) -> dict[str, set[str]]:
    found = defaultdict(set)
    for cancer, genes in genes_of.items():
        for gene in genes:
            found[gene].add(cancer)
    return found


def _accumulate_relation(target: dict[str, np.ndarray], query_slot: int, slots: np.ndarray, vector: np.ndarray) -> None:
    if len(slots) == 0:
        return
    _scatter_relation_matrix(target, query_slot, slots, np.asarray(vector, dtype=np.float64).reshape(1, -1))


def _scatter_relation_matrix(target: dict[str, np.ndarray], query_slot: int, slots: np.ndarray, matrix: np.ndarray) -> None:
    if len(slots) == 0:
        return
    core = matrix[:, :CORE_WIDTH]
    collapsed = matrix[:, CORE_WIDTH: CORE_WIDTH + PREDICATE_WIDTH]
    with_part = matrix[:, CORE_WIDTH + PREDICATE_WIDTH: CORE_WIDTH + 2 * PREDICATE_WIDTH]
    against = matrix[:, CORE_WIDTH + 2 * PREDICATE_WIDTH: CORE_WIDTH + 3 * PREDICATE_WIDTH]
    for offset, name in enumerate(CORE_ATTRIBUTES):
        values = core[:, offset]
        if np.any(values):
            target[name][query_slot, slots] += values
    for offset, name in enumerate(PREDICATES):
        collapsed_values = collapsed[:, offset]
        with_values = with_part[:, offset]
        against_values = against[:, offset]
        if np.any(collapsed_values):
            target[f"predicate_{name}"][query_slot, slots] += collapsed_values
        if np.any(with_values):
            target[f"orientation_{name}_with"][query_slot, slots] += with_values
        if np.any(against_values):
            target[f"orientation_{name}_against"][query_slot, slots] += against_values


def _add_uniform_relation(target: dict[str, np.ndarray], query_slot: int, slots: np.ndarray, piece: np.ndarray) -> None:
    if len(slots) == 0 or not np.any(piece):
        return
    _scatter_relation_matrix(target, query_slot, slots, np.broadcast_to(piece, (len(slots), piece.shape[0])).copy())


def _fill_two_edge(features, query_slot, cancer, graph, vectors, outgoing, incoming, index, expected_counts):
    n_gene = expected_counts.shape[0]
    path_n = np.zeros(n_gene)
    obs_n = np.zeros(n_gene)
    sums = {name: np.zeros(n_gene) for name in (
        "support_pmids", "shared_articles", "direct_collapsed", "direct_ltr", "direct_rtl", "direct_ambiguous", "cocite", "shared_references"
    )}
    age_sum = np.zeros(n_gene)
    age_n = np.zeros(n_gene)
    for gene in graph["genes_of"].get(cancer, ()):
        right_key = ("gene_cancer", gene, cancer)
        right_pmids = graph["edge_pmids"].get(right_key, ())
        right_vector = vectors.get(right_key)
        partners = [item for item in graph["neighbors"].get(gene, ()) if item in index and item != gene]
        for partner in partners:
            slot = index[partner]
            path_n[slot] += 1
            ordered = (partner, gene) if partner <= gene else (gene, partner)
            left_key = ("gene_gene", ordered[0], ordered[1])
            left_pmids = graph["edge_pmids"].get(left_key, ())
            direction = "ab" if partner == ordered[0] else "ba"
            if vectors.get(left_key) is not None:
                _accumulate_relation(features, query_slot, np.array([slot]), _relation_slice(vectors[left_key], direction))
            if right_vector is not None:
                _accumulate_relation(features, query_slot, np.array([slot]), _relation_slice(right_vector, "ab"))
            motifs = citation_between(left_pmids, right_pmids, outgoing, incoming)
            if motifs is not None:
                obs_n[slot] += 1
                for name in sums:
                    sums[name][slot] += motifs[name]
            ages = [graph["ages"].get(left_key), graph["ages"].get(right_key)]
            known = [age for age in ages if age is not None]
            if known:
                age_sum[slot] += float(sum(known) / len(known))
                age_n[slot] += 1
    if not np.allclose(path_n, expected_counts):
        raise RuntimeError(f"Two-edge path count mismatch for {cancer}")
    _write_position(features, query_slot, "t1", path_n, obs_n, sums, age_sum, age_n)


def _write_position(features, query_slot, position, path_n, obs_n, sums, age_sum, age_n):
    for name, values in sums.items():
        features[f"{position}_{name}"][query_slot] = _finalize_observation(path_n, obs_n, values)
    coverage = np.full(path_n.shape, np.nan)
    present = path_n > 0
    coverage[present] = obs_n[present] / path_n[present]
    features[f"{position}_coverage"][query_slot] = coverage
    age = np.full(path_n.shape, np.nan)
    aged = age_n > 0
    age[aged] = age_sum[aged] / age_n[aged]
    features[f"{position}_age"][query_slot] = age


def _fill_three_edge(features, query_slot, cancer, graph, vectors, outgoing, incoming, out_view, in_view, index, candidates, motif_cache):
    n_gene = len(index)
    path12 = np.zeros(n_gene)
    obs12 = np.zeros(n_gene)
    sum12 = {name: np.zeros(n_gene) for name in (
        "support_pmids", "shared_articles", "direct_collapsed", "direct_ltr", "direct_rtl", "direct_ambiguous", "cocite", "shared_references"
    )}
    path23 = np.zeros(n_gene)
    obs23 = np.zeros(n_gene)
    sum23 = {name: np.zeros(n_gene) for name in sum12}
    age12_sum = np.zeros(n_gene)
    age12_n = np.zeros(n_gene)
    age23_sum = np.zeros(n_gene)
    age23_n = np.zeros(n_gene)
    query_genes = graph["genes_of"].get(cancer, set())
    cancers_of = _cancers_from(graph["genes_of"])
    touched = defaultdict(list)
    for gene in query_genes:
        for other in cancers_of.get(gene, ()):
            if other != cancer and gene in graph["genes_of"].get(other, ()):
                touched[other].append(gene)
    relation_width = CORE_WIDTH + 3 * PREDICATE_WIDTH
    for other, extenders in touched.items():
        all_slots = np.array([index[gene] for gene in graph["genes_of"].get(other, ()) if gene in index], dtype=np.int32)
        if len(all_slots) == 0:
            continue
        path23[all_slots] += len(extenders)
        for gene in extenders:
            if gene in index:
                path23[index[gene]] -= 1
        total = np.zeros(relation_width)
        order = []
        rows = []
        for gene in extenders:
            piece = np.zeros(relation_width)
            edge_vector = vectors.get(("gene_cancer", gene, other))
            query_vector = vectors.get(("gene_cancer", gene, cancer))
            if edge_vector is not None:
                piece += _relation_slice(edge_vector, "ba")
            if query_vector is not None:
                piece += _relation_slice(query_vector, "ab")
            if not np.any(piece):
                continue
            total += piece
            if gene in index:
                order.append(index[gene])
                rows.append(piece)
        if np.any(total):
            _add_uniform_relation(features, query_slot, all_slots, total)
        if rows:
            _scatter_relation_matrix(features, query_slot, np.asarray(order, dtype=np.int32), -np.vstack(rows))
        for gene in extenders:
            edge_key = ("gene_cancer", gene, other)
            query_key = ("gene_cancer", gene, cancer)
            receivers = all_slots if gene not in index else all_slots[all_slots != index[gene]]
            motifs = citation_between(
                graph["edge_pmids"].get(edge_key, ()),
                graph["edge_pmids"].get(query_key, ()),
                out_view,
                in_view,
            )
            if motifs is not None and len(receivers):
                obs23[receivers] += 1
                for name in sum23:
                    sum23[name][receivers] += motifs[name]
            ages = [graph["ages"].get(edge_key), graph["ages"].get(query_key)]
            known = [age for age in ages if age is not None]
            if known and len(receivers):
                age23_sum[receivers] += float(sum(known) / len(known))
                age23_n[receivers] += 1
        _fill_position_12(
            features, query_slot, other, extenders, graph, vectors, outgoing, incoming, index, candidates,
            path12, obs12, sum12, age12_sum, age12_n, motif_cache,
        )
    _write_position(features, query_slot, "t2_12", path12, obs12, sum12, age12_sum, age12_n)
    _write_position(features, query_slot, "t2_23", path23, obs23, sum23, age23_sum, age23_n)


def _fill_position_12(features, query_slot, cancer, extenders, graph, vectors, outgoing, incoming, index, candidates, path_n, obs_n, sums, age_sum, age_n, motif_cache):
    all_genes = list(graph["genes_of"].get(cancer, ()))
    genes = [gene for gene in all_genes if gene in index and gene in candidates]
    if not all_genes or not extenders or not genes:
        return
    extender_set = set(extenders)
    block = motif_cache.get(cancer)
    if block is None:
        pmid_sets = [graph["edge_pmids"].get(("gene_cancer", gene, cancer), ()) for gene in all_genes]
        block = build_motif_block(pmid_sets, outgoing, incoming)
        motif_cache[cancer] = block
        if len(motif_cache) > 12:
            motif_cache.pop(next(iter(motif_cache)))
    else:
        motif_cache.pop(cancer)
        motif_cache[cancer] = block
    local_candidates = [pos for pos, gene in enumerate(all_genes) if gene in index and gene in candidates]
    local_extenders = [pos for pos, gene in enumerate(all_genes) if gene in extender_set]
    observed, motif_sums = sum_adjacent_motifs(block, local_extenders, local_candidates)
    n_ext = len(extenders)
    slots = np.empty(len(genes), dtype=np.int32)
    relation = np.zeros((len(genes), CORE_WIDTH + 3 * PREDICATE_WIDTH))
    total = np.zeros(relation.shape[1])
    self_rows = []
    for other in extenders:
        vector = vectors.get(("gene_cancer", other, cancer))
        if vector is None:
            continue
        piece = _relation_slice(vector, "ba")
        total += piece
        if other in index and other in candidates:
            self_rows.append((other, piece))
    if np.any(total):
        relation += total
    gene_row = {}
    for row, gene in enumerate(genes):
        slot = index[gene]
        gene_row[gene] = row
        slots[row] = slot
        partners = n_ext - (1 if gene in extender_set else 0)
        path_n[slot] += partners
        obs_n[slot] += observed[row]
        for name in sums:
            sums[name][slot] += motif_sums[name][row]
        own = vectors.get(("gene_cancer", gene, cancer))
        if own is not None and partners:
            relation[row] += _relation_slice(own, "ab") * partners
    for gene, piece in self_rows:
        relation[gene_row[gene]] -= piece
    _scatter_relation_matrix(features, query_slot, slots, relation)
    ages = np.full(len(genes), np.nan)
    for row, gene in enumerate(genes):
        age = graph["ages"].get(("gene_cancer", gene, cancer))
        if age is not None:
            ages[row] = float(age)
    for other in extenders:
        age_g = graph["ages"].get(("gene_cancer", other, cancer))
        mask = np.ones(len(genes), dtype=bool)
        if other in gene_row:
            mask[gene_row[other]] = False
        chosen = np.flatnonzero(mask)
        if len(chosen) == 0:
            continue
        own_age = ages[chosen]
        chosen_slots = slots[chosen]
        if age_g is None:
            known = ~np.isnan(own_age)
            age_sum[chosen_slots[known]] += own_age[known]
            age_n[chosen_slots[known]] += 1
        else:
            known = ~np.isnan(own_age)
            age_sum[chosen_slots[known]] += (own_age[known] + float(age_g)) / 2.0
            age_n[chosen_slots[known]] += 1
            missing = ~known
            age_sum[chosen_slots[missing]] += float(age_g)
            age_n[chosen_slots[missing]] += 1


def path_totals(genes_of, neighbors, candidates, queries) -> dict[str, int]:
    totals = template_weights(genes_of, neighbors, candidates, queries)
    return {"two_edge": int(totals["one_total"]), "three_edge": int(totals["two_total"])}


def ranking_matrix(scores: np.ndarray, genes: list[str], eligible: np.ndarray) -> list[str]:
    rows = []
    for slot in np.flatnonzero(eligible):
        value = float(scores[slot])
        rows.append({"gene": genes[slot], "score": -1e300 if np.isnan(value) else value})
    return rank_with_tie_break(rows)


def evaluate_method(score_rows: np.ndarray, genes: list[str], eligible: np.ndarray, positives: dict[str, set[str]], cancers: list[str]) -> dict:
    rankings = {}
    held = {}
    for slot, cancer in enumerate(cancers):
        rankings[cancer] = ranking_matrix(score_rows[slot], genes, eligible[slot])
        held[cancer] = positives.get(cancer, set())
    return recall_at_k(rankings, held, 50)


def top_set(ranking: list[str], k: int = 50) -> set[str]:
    return set(ranking[:k])


def train_rows(features, columns, cancers, roles, graph, gene_ids, seed: int, role: str):
    index = {gene: slot for slot, gene in enumerate(gene_ids)}
    xs = []
    ys = []
    cancer_ids = []
    genes = []
    eligible = np.zeros((len(cancers), len(gene_ids)), dtype=bool)
    for query_slot, cancer in enumerate(cancers):
        positives = roles[role].get(cancer, set())
        if role == "train":
            chosen = list(positives) + sample_controls(
                list(positives), gene_ids, graph["direct"].get(cancer, set()), CONTROL_RATIO, seed + query_slot
            )
        else:
            blocked = graph["direct"].get(cancer, set()) - positives
            chosen = [gene for gene in gene_ids if gene not in blocked]
        for gene in chosen:
            slot = index[gene]
            eligible[query_slot, slot] = True
            xs.append([features[name][query_slot, slot] for name in columns])
            ys.append(1 if gene in positives else 0)
            cancer_ids.append(cancer)
            genes.append(gene)
    return (
        np.asarray(xs, dtype=np.float64),
        np.asarray(ys, dtype=np.int64),
        np.asarray(cancer_ids),
        np.asarray(genes),
        eligible,
    )


def score_all(model, raw_train, matrices, scale: bool, probability: bool) -> list[np.ndarray]:
    transformed = prepare_features(raw_train, matrices, scale=scale)
    outputs = []
    for matrix in transformed[1:]:
        if probability:
            outputs.append(model.predict_proba(matrix)[:, 1])
        else:
            outputs.append(model.decision_function(matrix))
    return outputs


def pack_graph(graph, vectors, gene_ids, cancers):
    extra_genes = sorted({gene for genes in graph["genes_of"].values() for gene in genes if gene not in set(gene_ids)})
    all_genes = list(gene_ids) + extra_genes
    gene_index = {gene: slot for slot, gene in enumerate(all_genes)}
    cancer_list = sorted(graph["genes_of"])
    cancer_index = {cancer: slot for slot, cancer in enumerate(cancer_list)}
    gc_gene = []
    gc_cancer = []
    gc_hot = []
    gg_src = []
    gg_dst = []
    gg_hot = []
    width = PREDICATE_WIDTH + 1
    for (family, left, right), _pmids in graph["edge_pmids"].items():
        vector = vectors.get((family, left, right))
        hot = np.zeros(width, dtype=np.float32)
        if vector is not None:
            hot[:PREDICATE_WIDTH] = (vector[CORE_WIDTH: CORE_WIDTH + PREDICATE_WIDTH] > 0).astype(np.float32)
        if family == "gene_cancer":
            if int(np.sum(hot)) == 0 and left not in graph["core_of"].get(right, ()):
                continue
            if left in graph["core_of"].get(right, ()):
                hot[-1] = 1
            gc_gene.append(gene_index[left])
            gc_cancer.append(cancer_index[right])
            gc_hot.append(hot)
        else:
            if int(np.sum(hot)) == 0:
                continue
            for src, dst in ((left, right), (right, left)):
                if dst in gene_index and dst in set(gene_ids) and src in gene_index and src != dst:
                    gg_src.append(gene_index[src])
                    gg_dst.append(gene_index[dst])
                    gg_hot.append(hot.copy())
    candidate_mask = torch.zeros(len(all_genes), dtype=torch.bool)
    candidate_mask[: len(gene_ids)] = True
    pack = {
        "gene_cancer_hot": torch.tensor(np.asarray(gc_hot, dtype=np.float32) if gc_hot else np.zeros((0, width), np.float32)),
        "gene_gene_hot": torch.tensor(np.asarray(gg_hot, dtype=np.float32) if gg_hot else np.zeros((0, width), np.float32)),
        "gc_gene": torch.tensor(gc_gene, dtype=torch.long),
        "gc_cancer": torch.tensor(gc_cancer, dtype=torch.long),
        "gg_src": torch.tensor(gg_src, dtype=torch.long),
        "gg_dst": torch.tensor(gg_dst, dtype=torch.long),
        "query_cancer": torch.tensor([cancer_index[cancer] for cancer in cancers], dtype=torch.long),
        "candidate_mask": candidate_mask,
        "n_gene": len(all_genes),
        "n_cancer": len(cancer_list),
        "n_predicates": width,
    }
    relations = [[] for _ in range(width)]
    for hot, gene, cancer in zip(gc_hot, gc_gene, gc_cancer):
        cancer_node = len(all_genes) + cancer
        for relation, flag in enumerate(hot):
            if flag:
                relations[relation].append((gene, cancer_node))
                relations[relation].append((cancer_node, gene))
    for hot, src, dst in zip(gg_hot, gg_src, gg_dst):
        for relation, flag in enumerate(hot):
            if flag:
                relations[relation].append((src, dst))
    pack["relations"] = [
        (
            torch.tensor([item[0] for item in pairs], dtype=torch.long),
            torch.tensor([item[1] for item in pairs], dtype=torch.long),
        )
        if pairs else (torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))
        for pairs in relations
    ]
    pack["n_nodes"] = len(all_genes) + len(cancer_list)
    pack["cancer_node"] = {cancer: len(all_genes) + cancer_index[cancer] for cancer in cancers}
    pack["gene_ids"] = gene_ids
    pack["cancers"] = cancers
    return pack


def _roles_from_targets(targets: list[dict]) -> dict[str, dict[str, set[str]]]:
    roles = {"train": defaultdict(set), "tune": defaultdict(set), "development_validation": defaultdict(set)}
    for row in targets:
        roles[row["role"]][row["cancer_cui"]].add(row["gene_id"])
    return roles


def _direction_gap(features: dict[str, np.ndarray]) -> dict:
    gaps = {}
    for position in POSITIONS:
        left = features[f"{position}_direct_ltr"]
        right = features[f"{position}_direct_rtl"]
        ambiguous = features[f"{position}_direct_ambiguous"]
        collapsed = features[f"{position}_direct_collapsed"]
        finite = np.isfinite(left) & np.isfinite(right) & np.isfinite(ambiguous) & np.isfinite(collapsed)
        gaps[position] = None if not finite.any() else float(np.max(np.abs(left[finite] + right[finite] + ambiguous[finite] - collapsed[finite])))
    return gaps


def _write_samples(path: Path, cancers: list[str], gene_ids: list[str], roles, direct: dict[str, set[str]]) -> dict:
    cancer_col = []
    gene_col = []
    role_col = []
    seed_col = []
    label_col = []
    ratios = []
    for query_slot, cancer in enumerate(cancers):
        direct_genes = direct.get(cancer, set())
        for role in ("tune", "development_validation"):
            positives = roles[role].get(cancer, set())
            blocked = direct_genes - positives
            for gene in gene_ids:
                if gene in blocked:
                    continue
                cancer_col.append(cancer)
                gene_col.append(gene)
                role_col.append(role)
                seed_col.append(0)
                label_col.append(1 if gene in positives else 0)
        for seed in SEEDS:
            positives = sorted(roles["train"].get(cancer, set()))
            controls = sample_controls(positives, gene_ids, direct_genes, CONTROL_RATIO, seed + query_slot)
            ratios.append({
                "cancer": cancer,
                "seed": int(seed),
                "positives": len(positives),
                "controls": len(controls),
                "ratio": (len(controls) / len(positives)) if positives else None,
            })
            for gene in positives + controls:
                cancer_col.append(cancer)
                gene_col.append(gene)
                role_col.append("train")
                seed_col.append(int(seed))
                label_col.append(1 if gene in roles["train"].get(cancer, set()) else 0)
    table = pa.table({
        "cancer_cui": cancer_col,
        "gene_id": gene_col,
        "role": role_col,
        "seed": seed_col,
        "label": label_col,
    })
    pq.write_table(table, path)
    return {"rows": len(cancer_col), "control_ratios": ratios}


def prepare(run_dir: Path) -> None:
    timer = StageTimer()
    analysis = ensure_dir(run_dir / "analysis")
    ensure_dir(run_dir / "logs")
    gene_ids, candidates, queries = load_universe()
    gene_ids = sorted(gene_ids)
    profiles = load_profiles()
    membership = load_membership()
    timer.mark("inputs_loaded", candidates=len(gene_ids), queries=len(queries))
    rows, summary = extract_core_pairs(profiles, membership, candidates, queries)
    write_candidate_table(analysis / "core_positive_candidates.parquet", rows)
    timer.mark("core_positive_candidates", **summary)
    if not summary["anchors_match"]:
        write_json(run_dir / "validation.json", {"status": "failed_core_anchor_mismatch", "summary": summary, "formal_test_files_opened": False})
        raise SystemExit("CORE positive anchors do not match the published direct-pair counts")
    manifest_path = analysis / "development_manifest.json"
    targets_path = analysis / "development_targets.parquet"
    if manifest_path.exists() and targets_path.exists():
        split = {"manifest": read_json(manifest_path), "targets": pq.read_table(targets_path).to_pylist()}
        timer.mark("development_split", targets=split["manifest"]["selected_targets"], cancers=len(split["manifest"]["selected_cancers"]), resumed=True)
    else:
        split = build_split(rows, coverage_by_cancer(), run_dir)
        write_study_design(run_dir / "study_design.yaml", split, summary)
        timer.mark("development_split", targets=split["manifest"]["selected_targets"], cancers=len(split["manifest"]["selected_cancers"]))
    cancers = split["manifest"]["selected_cancers"]
    target_pairs = {(row["gene_id"], row["cancer_cui"]) for row in split["targets"]}
    masked_pmids = {int(pmid) for row in rows if (row["gene_id"], row["cancer_cui"]) in target_pairs for pmid in row["pmids"]}
    graph = build_graphs(profiles, membership, candidates, queries, target_pairs, masked_pmids)
    timer.mark("masked_graph", edges=len(graph["edge_pmids"]), masked_pmids=len(masked_pmids))
    unmasked = path_totals(graph["unmasked_genes"], graph["unmasked_neighbors"], candidates, queries)
    if unmasked["two_edge"] != EXPECTED_TWO_EDGE or unmasked["three_edge"] != EXPECTED_THREE_EDGE:
        write_json(run_dir / "validation.json", {"status": "failed_path_anchor_mismatch", "unmasked_totals": unmasked, "formal_test_files_opened": False})
        raise SystemExit(f"Unmasked path totals do not match the audit: {unmasked}")
    timer.mark("unmasked_path_anchors", **unmasked)
    attach_years(graph, timer)
    vectors = load_assertion_vectors(graph)
    timer.mark("assertion_vectors", edges=len(vectors))
    needed = {pmid for pmids in graph["edge_pmids"].values() for pmid in pmids}
    outgoing, incoming, citing, cited = load_citation_index(masked_pmids, needed)
    masked_array = np.fromiter(masked_pmids, dtype=np.int64, count=len(masked_pmids)) if masked_pmids else np.zeros(0, dtype=np.int64)
    citation_clear = True if len(masked_array) == 0 else not bool(np.isin(citing, masked_array).any() or np.isin(cited, masked_array).any())
    edge_clear = all(not (set(pmids) & masked_pmids) for pmids in graph["edge_pmids"].values())
    pairs_clear = all(gene not in graph["genes_of"].get(cancer, ()) for gene, cancer in target_pairs)
    timer.mark("citation_index", outgoing_pmids=len(outgoing), incoming_pmids=len(incoming), citation_clear=citation_clear)
    features, _index = fill_features(
        graph, vectors, outgoing, incoming, gene_ids, cancers, candidates, timer, analysis / "partial_features.npz"
    )
    gaps = _direction_gap(features)
    np.savez(analysis / "features.npz", **features)
    pack = pack_graph(graph, vectors, gene_ids, cancers)
    torch.save(pack, analysis / "graph_pack.pt")
    roles = _roles_from_targets(split["targets"])
    sample_summary = _write_samples(analysis / "training_samples.parquet", cancers, gene_ids, roles, graph["direct"])
    context = {
        "cancers": cancers,
        "gene_ids": gene_ids,
        "roles": {role: {cancer: sorted(genes) for cancer, genes in mapping.items()} for role, mapping in roles.items()},
        "direct_genes": {cancer: sorted(genes) for cancer, genes in graph["direct"].items() if cancer in set(cancers)},
        "target_semmed_direct": {row["cancer_cui"]: [] for row in split["targets"]},
        "summary": summary,
        "split": split["manifest"],
        "unmasked_path_totals": unmasked,
        "masked_pmid_count": len(masked_pmids),
        "masked_edge_count": len(graph["edge_pmids"]),
        "citation_index_excludes_masked_pmids": citation_clear,
        "masked_pmids_absent_from_edges": edge_clear,
        "target_pairs_absent_from_graph": pairs_clear,
        "direction_gaps": gaps,
        "training_samples": sample_summary,
        "stages": timer.stages,
        "formal_test_files_opened": False,
    }
    for row in split["targets"]:
        context["target_semmed_direct"].setdefault(row["cancer_cui"], [])
        if row["semmed_direct"]:
            context["target_semmed_direct"][row["cancer_cui"]].append(row["gene_id"])
    write_json(analysis / "preparation_context.json", context)
    write_json(run_dir / "preparation_validation.json", {
        "status": "prepared",
        "anchors_match": True,
        "unmasked_path_totals": unmasked,
        "citation_index_excludes_masked_pmids": citation_clear,
        "masked_pmids_absent_from_edges": edge_clear,
        "target_pairs_absent_from_graph": pairs_clear,
        "direction_gaps": gaps,
        "roles_disjoint": split["manifest"]["roles_are_disjoint"],
        "pmid_groups_stay_inside_one_role": split["manifest"]["pmid_groups_stay_inside_one_role"],
        "formal_test_files_opened": False,
        "models_fit": False,
    })
    timer.mark("preparation_complete")
    print("prepared", run_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.run_dir)


if __name__ == "__main__":
    main()



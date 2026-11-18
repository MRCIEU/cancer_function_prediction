"""Expanded development test of CORE attributes and SemMedDB predicates.

The formal test partition is not read. Citation links are not model inputs.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.association_recovery import rank_with_tie_break, recall_at_k, sample_controls
from src.common import RESULTS_ROOT, StageTimer, write_json
from src.path_evidence import template_weights
from src.recovery_models import cancer_weights, fit_logistic, fit_trees, macro_from_scores, prepare_features
from src.recovery_scope import (
    EXPECTED_BOTH,
    EXPECTED_CORE_ONLY,
    EXPECTED_CORE_PAIRS,
    EXPECTED_THREE_EDGE,
    EXPECTED_TWO_EDGE,
    candidate_slot_table,
    three_edge_scores,
)
from src.relation_attributes import (
    COHORT_LADDER,
    COHORT_SEED,
    CONTROL_RATIO,
    SEEDS,
    apply_log1p,
    assign_development_roles,
    build_edge_attributes,
    feature_names,
    fuse_log1p,
    paired_interval,
    position1_tables,
    sample_positives,
    score_query,
    select_cancers,
    share_conservation,
)
from src.run_association_recovery import (
    GRAPH_RUN,
    PATH_AUDIT,
    attach_years,
    build_graphs,
    coverage_by_cancer,
    extract_core_pairs,
    load_membership,
    load_profiles,
    load_universe,
)
from src.run_path_evidence import _as_bool, _connection

PRIOR_RUN = RESULTS_ROOT / "association_recovery_development_20261116T2109Z"
SPECS = ("baseline", "core_attributes", "semmed_predicates", "both")


def _peak_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def _prior_block() -> tuple[set[tuple[str, str]], set[int], set[str]]:
    table = pq.read_table(PRIOR_RUN / "analysis" / "development_targets.parquet")
    genes = table.column("gene_id").to_pylist()
    cancers = table.column("cancer_cui").to_pylist()
    pairs = set(zip(genes, cancers))
    pmids = set()
    for values in table.column("pmids").to_pylist():
        pmids.update(int(item) for item in values)
    return pairs, pmids, set(cancers)


def _write_design(path: Path, cohort: dict) -> None:
    path.write_text(
        "\n".join(
            [
                "# Relation attributes for gene–cancer ranking",
                "",
                "The question is whether sentence-level CORE attributes, and separately SemMedDB predicate types, improve recovery of masked CORE gene–cancer associations beyond a structural baseline. The formal test set stays sealed. Citation links, co-citation, shared references, predicate direction, and neural path models are outside this comparison.",
                "",
                "## Literature",
                "",
                "Himmelstein and Baranzini, 2015, DOI 10.1371/journal.pcbi.1004259. Degree-weighted path counts and regularized logistic regression are the precedents for ranking disease-associated genes from paths. This run fits the logistic weights on the project training split and uses a damping exponent of 0.4 for the DWPC reference.",
                "",
                "Himmelstein et al., 2017, DOI 10.7554/eLife.26726. Specific path information is contrasted with node connectivity. Gene degree and publication count are reported as separate reference rankings, and both remain in the common baseline.",
                "",
                "Marchesin et al., 2023, DOI 10.1093/database/baad061. CORE change of gene expression (CGE), change of cancer status (CCS), and gene–cancer interaction (GCI) stay on the sentence records. Informative values, explicit NOTINF or empty labels, and absent records are kept distinct. GCC and gene-type labels are not part of this primary representation.",
                "",
                "Bakal et al., 2018, DOI 10.1016/j.jbi.2018.05.003. SemMedDB predicate types are the matched addition to the same baseline. Subject and object orientation is collapsed. Negated predicates are excluded and counted.",
                "",
                "Friedman, 2001, DOI 10.1214/aos/1013203451. Shallow gradient-boosted trees test whether an attribute's contribution depends on other inputs, including edge position and coverage. Tree depth and learning rate are chosen on the common baseline.",
                "",
                "## Frozen choices",
                "",
                f"Cohort seed {COHORT_SEED}. Training-control seeds {list(SEEDS)}. Background ratio {CONTROL_RATIO}. The cohort ladder, applied before any model score is computed, is {list(COHORT_LADDER)}. Rare predicates are not binned: every allow-listed predicate keeps its own column. Count-like inputs use log1p. Shares do not. L2 penalties are 0.01, 0.1, 1, and 10. Tree settings are depth 2 or 3, learning rate 0.05 or 0.1, and 40 boosting rounds. The penalty and the tree setting are chosen on each seed's baseline tuning Recall@50 and then reused for that seed's other three specifications.",
                "",
                "Two path scores are named separately. `prior_degree_weight` is the earlier development score: a two-hop path contributes one over the intermediate gene's candidate-neighbor count, and a three-hop path contributes one over the gene count of the intermediate cancer. `dwpc` multiplies, for every edge on the path, each endpoint's type-specific degree raised to -0.4. The fixed fusion used for the DWPC reference ranking is the earlier rule, log1p(two-hop) + log1p(three-hop). The logistic and tree models instead receive the two templates as separate features.",
                "",
                "Attribute features divide a path's DWPC equally across the distinct observed values on that edge and position. Identical labels are counted once. The model feature is that contribution divided by the DWPC of paths on which the field is observable. A value that is absent while the field is observable is an observed zero. A field with no surviving record is missing, not zero.",
                "",
                "## Achieved cohort",
                "",
                f"Selected positives: {cohort['n_selected']}. Train {cohort['role_counts'].get('train', 0)}, tune {cohort['role_counts'].get('tune', 0)}, development validation {cohort['role_counts'].get('development_validation', 0)}.",
                f"Validation cancers: {cohort['validation_cancers']}. Largest PMID group: {cohort['largest_pmid_group']}.",
                f"Ladder step used: {cohort['ladder_step']}. Deviations: {cohort['deviations'] or 'none'}.",
                f"Pairs without a CORE PMID, excluded from the primary cohort: {cohort['missing_core_pmid']}.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def build_cohort(run_dir: Path) -> dict:
    timer = StageTimer()
    gene_ids, candidates, queries = load_universe()
    profiles = load_profiles()
    membership = load_membership()
    rows, summary = extract_core_pairs(profiles, membership, candidates, queries)
    if not summary["anchors_match"]:
        raise RuntimeError(f"CORE pair anchor mismatch {summary}")
    prior_pairs, prior_pmids, prior_cancers = _prior_block()
    coverage = coverage_by_cancer()
    by_cancer = defaultdict(list)
    for row in rows:
        if row["core_pmid_count"] <= 0:
            continue
        by_cancer[row["cancer_cui"]].append(row)
    catalog = []
    for cancer, pairs in by_cancer.items():
        stats = coverage.get(cancer, {"paths": 0, "two_reached": 0})
        catalog.append({"cancer": cancer, "n_pmid": len(pairs), "paths": stats["paths"], "two_reached": stats["two_reached"]})
    chosen_rows = []
    step = None
    deviations = []
    for n_cancers, per_cancer in COHORT_LADDER:
        cancers = select_cancers(catalog, n_cancers)
        selected = []
        for row in cancers:
            pairs = by_cancer[row["cancer"]]
            take = min(len(pairs) // 2, per_cancer)
            if take <= 0:
                continue
            selected.extend(sample_positives(pairs, take, COHORT_SEED))
        records = [{
            "target_id": row["target_id"],
            "cancer": row["cancer_cui"],
            "gene": row["gene_id"],
            "pmids": row["pmids"],
            "semmed_direct": row["semmed_direct"],
        } for row in selected]
        roles = assign_development_roles(records, prior_pmids)
        n_val = sum(1 for role in roles.values() if role == "development_validation")
        val_cancers = len({record["cancer"] for record in records if roles[record["target_id"]] == "development_validation"})
        step = {"cancers": n_cancers, "per_cancer": per_cancer, "selected": len(records), "validation": n_val, "validation_cancers": val_cancers}
        chosen_rows = selected
        if n_val >= 900 and val_cancers >= 80:
            break
        deviations.append(f"ladder {n_cancers}/{per_cancer} produced validation {n_val} across {val_cancers} cancers")
    records = [{
        "target_id": row["target_id"],
        "cancer": row["cancer_cui"],
        "gene": row["gene_id"],
        "pmids": row["pmids"],
        "semmed_direct": row["semmed_direct"],
        "core_pmids": row["core_pmids"],
        "semmed_pmids": row["semmed_pmids"],
    } for row in chosen_rows]
    roles = assign_development_roles(records, prior_pmids)
    role_counts = defaultdict(int)
    for role in roles.values():
        role_counts[role] += 1
    fractions = {name: role_counts[name] / max(len(records), 1) for name in ("train", "tune", "development_validation")}
    if abs(fractions["development_validation"] - 0.2) > 0.05:
        deviations.append(f"validation fraction {fractions['development_validation']:.3f}")
    groups = {}
    for record in records:
        groups.setdefault(tuple(sorted(set(record["pmids"]))), 0)
    # Largest connected group, not the raw PMID tuple.
    from src.association_recovery import connected_pmid_groups
    largest = max((len(group) for group in connected_pmid_groups(records)), default=0)
    val_cancers = sorted({record["cancer"] for record in records if roles[record["target_id"]] == "development_validation"})
    overlap = [(record["gene"], record["cancer"]) for record in records if (record["gene"], record["cancer"]) in prior_pairs and roles[record["target_id"]] == "development_validation"]
    if overlap:
        raise RuntimeError(f"{len(overlap)} previously used targets entered validation")
    cohort = {
        "anchors": summary,
        "missing_core_pmid": summary["missing_core_pmid_provenance"],
        "n_selected": len(records),
        "role_counts": dict(role_counts),
        "fractions": fractions,
        "validation_cancers": len(val_cancers),
        "largest_pmid_group": largest,
        "ladder_step": step,
        "deviations": deviations,
        "prior_targets": len(prior_pairs),
        "prior_cancers": len(prior_cancers),
        "prior_targets_excluded_from_validation": True,
        "formal_test_files_opened": False,
        "elapsed_s": round(time.time() - timer.started, 3),
    }
    analysis = run_dir / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    table_rows = []
    for record in records:
        table_rows.append({
            "gene_id": record["gene"],
            "cancer_cui": record["cancer"],
            "role": roles[record["target_id"]],
            "semmed_direct": record["semmed_direct"],
            "pmids": record["pmids"],
            "previously_used": (record["gene"], record["cancer"]) in prior_pairs,
        })
    pq.write_table(pa.Table.from_pylist(table_rows), analysis / "development_targets.parquet")
    write_json(analysis / "cohort.json", cohort)
    _write_design(run_dir / "design.md", cohort)
    manifest = {
        "graph_runs": {
            "joint": str(GRAPH_RUN),
            "paths": str(PATH_AUDIT),
            "prior_development": str(PRIOR_RUN),
        },
        "anchors": summary,
        "expected_paths": {"two_edge": EXPECTED_TWO_EDGE, "three_edge": EXPECTED_THREE_EDGE},
        "seeds": {"cohort": COHORT_SEED, "controls": list(SEEDS)},
        "formal_test_files_opened": False,
        "citation_links_in_model": False,
        "cache_key": {
            "graph": "joint_graph_and_evidence_audit_20261116T0404Z",
            "split_seed": COHORT_SEED,
            "damping": 0.4,
            "predicates": "study_design allow lists, collapsed orientation",
            "core_fields": ["cge", "ccs", "gci"],
        },
        "cohort": cohort,
    }
    write_json(run_dir / "manifest.json", manifest)
    print(json.dumps({"selected": len(records), "roles": dict(role_counts), "validation_cancers": len(val_cancers), "step": step}, indent=2), flush=True)
    return {"records": table_rows, "summary": summary, "gene_ids": gene_ids, "candidates": candidates, "queries": queries, "profiles": profiles, "membership": membership}


def stream_attributes(graph_dir: Path, masked_pmids: set[int]) -> dict:
    """Distinct surviving values, without retaining every evidence tuple."""
    from src.common import normalise_pmid

    relations = {
        row["relation_id"]: row
        for row in pq.read_table(
            graph_dir / "relations.parquet",
            columns=["relation_id", "resource", "subject_entity_id", "object_entity_id", "predicate", "base_predicate", "negation", "connectivity_family"],
        ).to_pylist()
    }
    links = defaultdict(list)
    evidence = pq.read_table(graph_dir / "relation_evidence.parquet", columns=["relation_id", "evidence_id"])
    for relation_id, evidence_id in zip(evidence.column("relation_id").to_pylist(), evidence.column("evidence_id").to_pylist()):
        links[evidence_id].append(relation_id)
    connections = {}
    parquet = pq.ParquetFile(graph_dir / "evidence_records.parquet")
    columns = ["evidence_id", "resource", "native_id", "pmid", "path_eligible", "predicate", "negation", "cge_label", "ccs_label", "gci_label"]
    for batch in parquet.iter_batches(batch_size=200_000, columns=columns):
        rows = batch.to_pydict()
        for index, evidence_id in enumerate(rows["evidence_id"]):
            for relation_id in links.get(evidence_id, ()):
                relation = relations.get(relation_id)
                if relation is None:
                    continue
                key = _connection(relation)
                if key is None:
                    continue
                bucket = connections.get(key)
                if bucket is None:
                    bucket = {"core": {}, "sem": {}}
                    connections[key] = bucket
                pmid = normalise_pmid(rows["pmid"][index]) or ""
                native = str(rows["native_id"][index] or evidence_id)
                if rows["resource"][index] == "core":
                    bucket["core"].setdefault(native, (pmid, rows["cge_label"][index] or "", rows["ccs_label"][index] or "", rows["gci_label"][index] or "", "", ""))
                else:
                    predicate = relation.get("base_predicate") or relation.get("predicate") or ""
                    negated = _as_bool(relation.get("negation")) or _as_bool(rows["negation"][index])
                    traversal = _as_bool(rows["path_eligible"][index]) and not negated
                    bucket["sem"].setdefault(native, (pmid, predicate, "", "", negated, traversal))
    print(f"[attributes] edges={len(connections)}", flush=True)
    return build_edge_attributes(connections, masked_pmids)


def _graph_bundle(payload: dict, records: list[dict]):
    target_pairs = {(row["gene_id"], row["cancer_cui"]) for row in records}
    masked = set()
    for row in records:
        masked.update(int(item) for item in row["pmids"])
    graph = build_graphs(payload["profiles"], payload["membership"], payload["candidates"], payload["queries"], target_pairs, masked)
    attach_years(graph, StageTimer())
    attributes = stream_attributes(GRAPH_RUN / "graph", masked)
    return graph, attributes, target_pairs, masked


def _score_cancers(graph, attributes, gene_ids, cancers: list[str]) -> dict[str, np.ndarray]:
    index = {gene: slot for slot, gene in enumerate(gene_ids)}
    slots = candidate_slot_table(graph["genes_of"], index)
    cancers_of = defaultdict(set)
    deg_gc = defaultdict(int)
    for cancer, genes in graph["genes_of"].items():
        for gene in genes:
            cancers_of[gene].add(cancer)
            deg_gc[gene] += 1
    deg_gg = {gene: len(partners) for gene, partners in graph["neighbors"].items()}
    publications = np.zeros(len(gene_ids))
    gene_pmids = defaultdict(set)
    for (family, left, right), pmids in graph["edge_pmids"].items():
        if family == "gene_gene":
            gene_pmids[left].update(pmids)
            gene_pmids[right].update(pmids)
        else:
            gene_pmids[left].update(pmids)
    for gene, pmids in gene_pmids.items():
        slot = index.get(gene)
        if slot is not None:
            publications[slot] = float(len(pmids))
    names = feature_names()["both"]
    store = {name: np.zeros((len(cancers), len(gene_ids)), dtype=np.float32) for name in names}
    core_slots = candidate_slot_table(graph["core_of"], index)
    sem_slots = candidate_slot_table(graph["sem_of"], index)
    core_cancers = defaultdict(set)
    sem_cancers = defaultdict(set)
    for cancer, genes in graph["core_of"].items():
        for gene in genes:
            core_cancers[gene].add(cancer)
    for cancer, genes in graph["sem_of"].items():
        for gene in genes:
            sem_cancers[gene].add(cancer)
    tables = position1_tables(attributes, graph["edge_pmids"], index, slots)
    print(f"[attributes] position1_tables={len(tables)} rss_gb={_peak_gb():.2f}", flush=True)
    started = time.time()
    for query_slot, cancer in enumerate(cancers):
        one_started = time.time()
        scored = score_query(
            cancer,
            graph["genes_of"],
            graph["neighbors"],
            cancers_of,
            index,
            slots,
            deg_gg,
            deg_gc,
            graph["edge_pmids"],
            graph["ages"],
            attributes,
            publications,
            tables,
        )
        core_counts, _weighted, _walk = three_edge_scores(graph["core_of"], core_cancers, cancer, set(index), index, core_slots)
        sem_counts, _sem_weighted, _sem_walk = three_edge_scores(graph["sem_of"], sem_cancers, cancer, set(index), index, sem_slots)
        scored["three_edge_core_paths"] = core_counts
        scored["three_edge_semmed_paths"] = sem_counts
        for name in names:
            store[name][query_slot] = scored[name].astype(np.float32)
        if query_slot < 3 or (query_slot + 1) % 10 == 0:
            print(f"[score] {query_slot + 1}/{len(cancers)} {cancer} seconds={time.time() - one_started:.1f} elapsed={time.time() - started:.1f} rss_gb={_peak_gb():.2f}", flush=True)
    return store


def _matrix(store: dict[str, np.ndarray], spec: str, query_index: dict[str, int], rows: list[tuple[str, str]]) -> np.ndarray:
    names = feature_names()[spec]
    chosen = np.stack([store[name] for name in names], axis=-1)
    out = np.empty((len(rows), len(names)), dtype=np.float64)
    for index, (cancer, gene) in enumerate(rows):
        out[index] = chosen[query_index[cancer], int(gene)]
    return apply_log1p(out, names)


def _metrics(scores: np.ndarray, cancers: np.ndarray, genes: np.ndarray, labels: np.ndarray) -> dict:
    base = macro_from_scores(scores, cancers, genes, labels)
    per = []
    recovered10 = 0
    rr_sum = 0.0
    n_pos = 0
    for cancer in sorted(set(cancers.tolist())):
        mask = cancers == cancer
        order = rank_with_tie_break([
            {"gene": str(gene), "score": float(score)}
            for gene, score in zip(genes[mask], scores[mask])
        ])
        positives = [str(gene) for gene, label in zip(genes[mask], labels[mask]) if int(label) == 1]
        top10 = set(order[:10])
        top50 = set(order[:50])
        hits10 = len(top10 & set(positives))
        hits50 = len(top50 & set(positives))
        recovered10 += hits10
        ranks = {gene: rank + 1 for rank, gene in enumerate(order)}
        reciprocal = [1.0 / ranks[gene] for gene in positives]
        rr_sum += float(sum(reciprocal))
        n_pos += len(positives)
        per.append({
            "cancer": str(cancer),
            "positives": len(positives),
            "recovered_at_10": hits10,
            "recovered_at_50": hits50,
            "recall_at_10": hits10 / len(positives) if positives else None,
            "recall_at_50": hits50 / len(positives) if positives else None,
            "mrr": float(np.mean(reciprocal)) if reciprocal else None,
        })
    macro10 = float(np.mean([row["recall_at_10"] for row in per])) if per else 0.0
    macro_mrr = float(np.mean([row["mrr"] for row in per])) if per else 0.0
    return {
        "macro_recall_at_50": base["macro_recall"],
        "macro_recall_at_10": macro10,
        "macro_mrr": macro_mrr,
        "recovered_at_50": base["recovered"],
        "recovered_at_10": recovered10,
        "positives": base["positives"],
        "cancers": base["cancers"],
        "mean_reciprocal_rank_pooled": rr_sum / n_pos if n_pos else None,
        "per_cancer": per,
    }


def _eval_rows(graph, gene_ids, records, role: str) -> list[tuple[str, str, int]]:
    hidden = defaultdict(set)
    for row in records:
        if row["role"] == role:
            hidden[row["cancer_cui"]].add(row["gene_id"])
    index = {gene: slot for slot, gene in enumerate(gene_ids)}
    rows = []
    for cancer, positives in hidden.items():
        blocked = set(graph["direct"].get(cancer, ())) - positives
        for gene, slot in index.items():
            if gene in blocked:
                continue
            rows.append((cancer, slot, 1 if gene in positives else 0))
    return rows


def fit_models(run_dir: Path, store: dict[str, np.ndarray], graph, gene_ids, records: list[dict]) -> dict:
    cancers = sorted({row["cancer_cui"] for row in records})
    query_index = {cancer: index for index, cancer in enumerate(cancers)}
    index = {gene: slot for slot, gene in enumerate(gene_ids)}
    analysis = run_dir / "analysis"
    results = {"seeds": {}, "contrasts": {}, "formal_test_files_opened": False}
    score_cache = {}
    for seed in SEEDS:
        train_rows = []
        train_y = []
        for row in records:
            if row["role"] != "train":
                continue
            train_rows.append((row["cancer_cui"], index[row["gene_id"]]))
            train_y.append(1)
        by_cancer = defaultdict(list)
        for row in records:
            if row["role"] == "train":
                by_cancer[row["cancer_cui"]].append(row["gene_id"])
        for cancer, positives in by_cancer.items():
            controls = sample_controls(positives, gene_ids, set(graph["direct"].get(cancer, ())), CONTROL_RATIO, seed)
            for gene in controls:
                train_rows.append((cancer, index[gene]))
                train_y.append(0)
        tune_rows_src = [row for row in records if row["role"] == "tune"]
        tune_eval = _eval_rows(graph, gene_ids, records, "tune")
        val_eval = _eval_rows(graph, gene_ids, records, "development_validation")
        # Baseline matrices for tuning. Full-list tune evaluation is separate from the positive-only tune labels used by fit_*.
        # fit_logistic ranks only the rows it is given. Pass the full tune candidate list so Recall@50 matches the protocol.
        def pack(spec: str, pairs: list[tuple[str, int]], labels: list[int]):
            matrix = _matrix(store, spec, query_index, [(cancer, slot) for cancer, slot in pairs])
            cancer_ids = np.array([cancer for cancer, _slot in pairs])
            gene_ids_row = np.array([gene_ids[slot] for _cancer, slot in pairs])
            return matrix, np.array(labels), cancer_ids, gene_ids_row

        tune_pairs = [(cancer, slot) for cancer, slot, _label in tune_eval]
        tune_labels = [label for _cancer, _slot, label in tune_eval]
        val_pairs = [(cancer, slot) for cancer, slot, _label in val_eval]
        val_labels = [label for _cancer, _slot, label in val_eval]
        base_train = pack("baseline", train_rows, train_y)
        base_tune = pack("baseline", tune_pairs, tune_labels)
        logistic = fit_logistic(base_train[0], base_train[1], base_train[2], base_train[3], base_tune[0], base_tune[1], base_tune[2], base_tune[3])
        trees = fit_trees(base_train[0], base_train[1], base_train[2], base_train[3], base_tune[0], base_tune[1], base_tune[2], base_tune[3], seed)
        chosen = {"logistic_C": logistic["C"], "tree_depth": trees["max_depth"], "tree_learning_rate": trees["learning_rate"]}
        seed_metrics = {"settings": chosen, "methods": {}}
        for method, kind in (("ridge", "logistic"), ("trees", "tree")):
            seed_metrics["methods"][method] = {}
            for spec in SPECS:
                train = pack(spec, train_rows, train_y)
                tune = pack(spec, tune_pairs, tune_labels)
                val = pack(spec, val_pairs, val_labels)
                if kind == "logistic":
                    prepared = prepare_features(train[0], [tune[0], val[0]], scale=True)
                    model = type(logistic["model"])(C=float(chosen["logistic_C"]), solver="lbfgs", max_iter=400)
                    model.fit(prepared[0], train[1], sample_weight=cancer_weights(train[2], train[1]))
                    val_scores = model.decision_function(prepared[2])
                    if spec == "baseline" or spec == "both":
                        named = feature_names()[spec]
                        coefficients = {
                            "intercept": float(model.intercept_[0]),
                            "coefficients": [float(value) for value in model.coef_.ravel()],
                            "features": named + [f"{name}__missing" for name in named],
                            "note": "The second half of the coefficient vector is the missingness indicator for the named feature.",
                        }
                        (analysis / f"coefficients_{method}_{spec}_seed_{seed}.json").write_text(json.dumps(coefficients), encoding="utf-8")
                else:
                    from sklearn.ensemble import HistGradientBoostingClassifier
                    prepared = prepare_features(train[0], [tune[0], val[0]], scale=False)
                    model = HistGradientBoostingClassifier(
                        max_depth=int(chosen["tree_depth"]),
                        learning_rate=float(chosen["tree_learning_rate"]),
                        max_iter=40,
                        early_stopping=False,
                        random_state=int(seed),
                    )
                    model.fit(prepared[0], train[1], sample_weight=cancer_weights(train[2], train[1]))
                    val_scores = model.predict_proba(prepared[2])[:, 1]
                metric = _metrics(val_scores, val[2], val[3], val[1])
                seed_metrics["methods"][method][spec] = {key: value for key, value in metric.items() if key != "per_cancer"}
                seed_metrics["methods"][method][spec]["per_cancer"] = metric["per_cancer"]
                score_cache[(seed, method, spec)] = {
                    "cancer": val[2],
                    "gene": val[3],
                    "label": val[1],
                    "score": val_scores,
                }
                print(f"[fit] seed={seed} {method} {spec} recall50={metric['macro_recall_at_50']:.4f}", flush=True)
        results["seeds"][str(seed)] = seed_metrics
    results["contrasts"] = _contrasts(results)
    results["concentration"] = _concentration(results)
    results["ranking_check"] = _ranking_check(score_cache, results)
    _write_outputs(run_dir, results, score_cache)
    return results


def _contrasts(results: dict) -> dict:
    found = {}
    for method in ("ridge", "trees"):
        found[method] = {}
        for left, right, name in (
            ("baseline", "core_attributes", "core_versus_baseline"),
            ("baseline", "semmed_predicates", "semmed_versus_baseline"),
            ("core_attributes", "both", "both_versus_core"),
            ("semmed_predicates", "both", "both_versus_semmed"),
        ):
            per_seed = []
            cancer_delta = defaultdict(list)
            for seed, payload in results["seeds"].items():
                left_rows = {row["cancer"]: row["recall_at_50"] for row in payload["methods"][method][left]["per_cancer"]}
                right_rows = {row["cancer"]: row["recall_at_50"] for row in payload["methods"][method][right]["per_cancer"]}
                per_seed.append({
                    "seed": int(seed),
                    "left": payload["methods"][method][left]["macro_recall_at_50"],
                    "right": payload["methods"][method][right]["macro_recall_at_50"],
                    "delta": payload["methods"][method][right]["macro_recall_at_50"] - payload["methods"][method][left]["macro_recall_at_50"],
                })
                for cancer in left_rows:
                    cancer_delta[cancer].append(right_rows[cancer] - left_rows[cancer])
            averaged = np.array([float(np.mean(values)) for values in cancer_delta.values()])
            found[method][name] = {"seeds": per_seed, "interval": paired_interval(averaged)}
    return found


def _concentration(results: dict) -> dict:
    """Whether a Recall@50 gain is spread across cancers or sits in a few queries."""
    summary = {}
    for method, contrasts in results["contrasts"].items():
        summary[method] = {}
        for name in contrasts:
            left_name, right_name = {
                "core_versus_baseline": ("baseline", "core_attributes"),
                "semmed_versus_baseline": ("baseline", "semmed_predicates"),
                "both_versus_core": ("core_attributes", "both"),
                "both_versus_semmed": ("semmed_predicates", "both"),
            }[name]
            per_cancer = defaultdict(list)
            for seed_payload in results["seeds"].values():
                left_rows = {row["cancer"]: row["recall_at_50"] for row in seed_payload["methods"][method][left_name]["per_cancer"]}
                right_rows = {row["cancer"]: row["recall_at_50"] for row in seed_payload["methods"][method][right_name]["per_cancer"]}
                for cancer, value in left_rows.items():
                    per_cancer[cancer].append(right_rows[cancer] - value)
            averaged = np.array([float(np.mean(values)) for values in per_cancer.values()])
            positive = averaged[averaged > 0]
            order = np.argsort(averaged)[::-1]
            top = averaged[order[: max(1, len(averaged) // 10)]]
            summary[method][name] = {
                "cancers": int(len(averaged)),
                "cancers_improved": int(np.sum(averaged > 0)),
                "cancers_unchanged": int(np.sum(averaged == 0)),
                "top_decile_share_of_positive_mass": float(top[top > 0].sum() / positive.sum()) if positive.sum() else 0.0,
            }
    return summary


def _reference_rankings(store: dict[str, np.ndarray], graph, gene_ids: list[str], records: list[dict]) -> dict:
    cancers = sorted({row["cancer_cui"] for row in records})
    query_index = {cancer: index for index, cancer in enumerate(cancers)}
    index = {gene: slot for slot, gene in enumerate(gene_ids)}
    rows = _eval_rows(graph, gene_ids, records, "development_validation")
    cancer_ids = np.array([cancer for cancer, _slot, _label in rows])
    gene_row = np.array([gene_ids[slot] for _cancer, slot, _label in rows])
    labels = np.array([label for _cancer, _slot, label in rows])
    query_slots = np.array([query_index[cancer] for cancer, _slot, _label in rows])
    gene_slots = np.array([slot for _cancer, slot, _label in rows])
    two = store["two_edge_dwpc"][query_slots, gene_slots]
    three = store["three_edge_dwpc"][query_slots, gene_slots]
    prior_two = store["two_edge_prior_weight"][query_slots, gene_slots]
    prior_three = store["three_edge_prior_weight"][query_slots, gene_slots]
    hidden = [row for row in records if row["role"] == "development_validation"]
    zero_paths = 0
    for row in hidden:
        slot = index[row["gene_id"]]
        query_slot = query_index[row["cancer_cui"]]
        if store["has_two_edge"][query_slot, slot] == 0 and store["has_three_edge"][query_slot, slot] == 0:
            zero_paths += 1
    found = {}
    for name, scores in (
        ("dwpc_log1p_fusion", fuse_log1p(two, three)),
        ("prior_degree_log1p_fusion", fuse_log1p(prior_two, prior_three)),
        ("gene_gg_degree", store["gene_gg_degree"][query_slots, gene_slots]),
        ("gene_publications", store["gene_publications"][query_slots, gene_slots]),
    ):
        metric = _metrics(np.asarray(scores, dtype=np.float64), cancer_ids, gene_row, labels)
        found[name] = {key: value for key, value in metric.items() if key != "per_cancer"}
    found["validation_targets_with_zero_paths"] = zero_paths
    found["validation_targets"] = len(hidden)
    return found


def _mask_summary(graph, masked: set[int]) -> dict:
    def tally(genes: dict, neighbors: dict) -> dict:
        return {
            "gene_cancer_edges": int(sum(len(members) for members in genes.values())),
            "gene_gene_edges": int(sum(len(partners) for partners in neighbors.values()) // 2),
        }
    remaining = {pmid for pmids in graph["edge_pmids"].values() for pmid in pmids}
    return {
        "before": tally(graph["unmasked_genes"], graph["unmasked_neighbors"]),
        "after": tally(graph["genes_of"], graph["neighbors"]),
        "blocked_pmids": len(masked),
        "pmids_remaining_on_edges": len(remaining),
    }


def _ranking_check(score_cache: dict, results: dict) -> dict:
    mismatches = []
    for (seed, method, spec), payload in score_cache.items():
        recomputed = _metrics(payload["score"], payload["cancer"], payload["gene"], payload["label"])
        stored = results["seeds"][str(seed)]["methods"][method][spec]["macro_recall_at_50"]
        if abs(recomputed["macro_recall_at_50"] - stored) > 1e-12:
            mismatches.append({"seed": seed, "method": method, "spec": spec})
    return {"matches": not mismatches, "mismatches": mismatches}


def _write_outputs(run_dir: Path, results: dict, score_cache: dict) -> None:
    analysis = run_dir / "analysis"
    # Drop per-cancer lists from the compact metrics file after writing the long table.
    long_rows = []
    for seed, payload in results["seeds"].items():
        for method, specs in payload["methods"].items():
            for spec, metric in specs.items():
                for row in metric["per_cancer"]:
                    long_rows.append({"seed": int(seed), "method": method, "specification": spec, **row})
    pq.write_table(pa.Table.from_pylist(long_rows), analysis / "per_cancer_metrics.parquet")
    compact = json.loads(json.dumps(results))
    for payload in compact["seeds"].values():
        for specs in payload["methods"].values():
            for metric in specs.values():
                metric.pop("per_cancer", None)
    write_json(analysis / "model_metrics.json", compact)
    entries = []
    for (seed, method, spec), payload in score_cache.items():
        if spec == "baseline" or seed != SEEDS[0]:
            continue
        base = score_cache[(seed, method, "baseline")]
        for cancer in sorted(set(payload["cancer"].tolist())):
            mask = payload["cancer"] == cancer
            order = rank_with_tie_break([{"gene": str(gene), "score": float(score)} for gene, score in zip(payload["gene"][mask], payload["score"][mask])])
            base_mask = base["cancer"] == cancer
            base_order = rank_with_tie_break([{"gene": str(gene), "score": float(score)} for gene, score in zip(base["gene"][base_mask], base["score"][base_mask])])
            positives = {str(gene) for gene, label in zip(payload["gene"][mask], payload["label"][mask]) if int(label) == 1}
            entered = sorted((set(order[:50]) & positives) - set(base_order[:50]))
            left = sorted((set(base_order[:50]) & positives) - set(order[:50]))
            for gene in entered:
                entries.append({"seed": seed, "method": method, "specification": spec, "cancer": str(cancer), "gene": gene, "change": "entered_top_50"})
            for gene in left:
                entries.append({"seed": seed, "method": method, "specification": spec, "cancer": str(cancer), "gene": gene, "change": "left_top_50"})
    if entries:
        pq.write_table(pa.Table.from_pylist(entries), analysis / "top50_entries_exits.parquet")
    write_json(analysis / "ranking_reproduction.json", results["ranking_check"])


def _report(run_dir: Path, results: dict, checks: dict) -> None:
    lines = ["# Relation-attribute development results", ""]
    lines.append(f"Cohort and masking checks: {json.dumps(checks['summary'])}.")
    lines.append("")
    lines.append("## DWPC reconstruction")
    lines.append("")
    lines.append("Unmasked two-hop and three-hop totals were reconciled to the completion audit before masking. `prior_degree_weight` keeps the earlier one-over-degree score. `dwpc` uses damping 0.4 on the product of endpoint degrees. The reference ranking fuses the two DWPC templates with log1p(two) + log1p(three). Learned models do not use that fusion; they see the templates as separate features.")
    lines.append("")
    lines.append("## Primary contrasts")
    lines.append("")
    lines.append("Macro Recall@50 gives one vote to each validation cancer. The interval resamples cancers after averaging the paired difference across the three control seeds.")
    lines.append("")
    for method, contrasts in results["contrasts"].items():
        lines.append(f"### {method}")
        lines.append("")
        for name, payload in contrasts.items():
            interval = payload["interval"]
            lines.append(
                f"- {name}: mean difference {interval['mean']:.4f} (95% interval {interval['low']:.4f} to {interval['high']:.4f}, {interval['cancers']} cancers)."
            )
        lines.append("")
    lines.append("A positive difference means the added feature group recovered a larger share of hidden targets inside the top 50.")
    lines.append("")
    if results.get("references"):
        lines.append("## Fixed reference rankings")
        lines.append("")
        for name, metric in results["references"].items():
            if not isinstance(metric, dict) or "macro_recall_at_50" not in metric:
                continue
            lines.append(
                f"- {name}: macro Recall@50 {metric['macro_recall_at_50']:.4f}, Recall@10 {metric['macro_recall_at_10']:.4f}, MRR {metric['macro_mrr']:.4f}, recovered {metric['recovered_at_50']}/{metric['positives']}."
            )
        lines.append("")
        lines.append(f"Validation targets with no remaining two-hop or three-hop path: {results['references'].get('validation_targets_with_zero_paths')} of {results['references'].get('validation_targets')}.")
        lines.append("")
    if results.get("concentration"):
        lines.append("## Where the differences sit")
        lines.append("")
        for method, contrasts in results["concentration"].items():
            for name, payload in contrasts.items():
                lines.append(
                    f"- {method} {name}: {payload['cancers_improved']} of {payload['cancers']} cancers improved; the highest decile holds {payload['top_decile_share_of_positive_mass']:.2f} of the positive paired mass."
                )
        lines.append("")
    lines.append("The comparison supports keeping a feature group when its matched difference stays positive across the cancer resample and appears in more than one seed. Complementarity is the both-versus-one-group contrast.")
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(run_dir: Path, stage: str, pilot: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = build_cohort(run_dir)
    if stage == "cohort":
        return
    records = pq.read_table(run_dir / "analysis" / "development_targets.parquet").to_pylist()
    if pilot:
        keep = []
        seen = []
        for row in records:
            if row["cancer_cui"] not in seen:
                if len(seen) >= pilot:
                    continue
                seen.append(row["cancer_cui"])
            if row["cancer_cui"] in seen:
                keep.append(row)
        records = keep
    graph, attributes, target_pairs, masked = _graph_bundle(payload, records)
    cancers = sorted({row["cancer_cui"] for row in records})
    # Unmasked reconciliation uses the full graph only on the full stage.
    checks = {"formal_test_files_opened": False, "summary": {}}
    if not pilot:
        genes = {cancer: set(members) for cancer, members in graph["unmasked_genes"].items()}
        weights = template_weights(genes, graph["unmasked_neighbors"], payload["candidates"], payload["queries"])
        checks["unmasked_paths"] = {"two_edge": int(weights["one_total"]), "three_edge": int(weights["two_total"])}
        checks["unmasked_paths_match"] = checks["unmasked_paths"]["two_edge"] == EXPECTED_TWO_EDGE and checks["unmasked_paths"]["three_edge"] == EXPECTED_THREE_EDGE
    leaked = [pair for pair in target_pairs if pair[0] in graph["genes_of"].get(pair[1], ())]
    checks["targets_absent_as_direct_edges"] = len(leaked) == 0
    blocked_still_present = [pmid for pmids in graph["edge_pmids"].values() for pmid in pmids if pmid in masked]
    checks["blocked_pmids_absent"] = len(blocked_still_present) == 0
    checks["mask"] = _mask_summary(graph, masked)
    write_json(run_dir / "analysis" / "feature_dictionary.json", feature_names())
    print(f"[mask] cancers={len(cancers)} leaked_edges={len(leaked)} blocked_pmids_remaining={len(blocked_still_present)} rss_gb={_peak_gb():.2f}", flush=True)
    store = _score_cancers(graph, attributes, payload["gene_ids"], cancers)
    checks["share_conservation"] = share_conservation(store)
    print(f"[shares] {json.dumps(checks['share_conservation'])}", flush=True)
    if pilot:
        write_json(run_dir / "analysis" / "pilot_checks.json", checks)
        print("[pilot] features complete", flush=True)
        return
    np.savez_compressed(run_dir / "analysis" / "feature_store.npz", **{name: values for name, values in store.items()}, cancers=np.array(cancers))
    results = fit_models(run_dir, store, graph, payload["gene_ids"], records)
    results["references"] = _reference_rankings(store, graph, payload["gene_ids"], records)
    write_json(run_dir / "analysis" / "reference_rankings.json", results["references"])
    write_json(run_dir / "analysis" / "concentration.json", results["concentration"])
    checks["summary"] = {
        "targets_absent": checks["targets_absent_as_direct_edges"],
        "blocked_pmids_absent": checks["blocked_pmids_absent"],
        "unmasked_paths_match": checks.get("unmasked_paths_match"),
        "rankings_reproduce_metrics": results["ranking_check"]["matches"],
        "peak_rss_gb": round(_peak_gb(), 3),
    }
    checks["formal_test_files_opened"] = False
    write_json(run_dir / "validation.json", checks)
    _report(run_dir, results, checks)
    print("[done] relation attributes", json.dumps(checks["summary"]), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("cohort", "full", "pilot"), default="full")
    parser.add_argument("--pilot-cancers", type=int, default=4)
    args = parser.parse_args()
    run(args.run_dir, "cohort" if args.stage == "cohort" else args.stage, args.pilot_cancers if args.stage == "pilot" else 0)


if __name__ == "__main__":
    main()

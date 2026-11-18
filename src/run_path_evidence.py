"""Descriptive evidence audit on the fixed joint-graph paths.

The run reuses the completed joint graph. It does not refit eligibility rules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.common import StageTimer, ensure_dir, normalise_pmid, peak_rss_bytes, write_json
from src.mapping import informative_core_tuple
from src.path_count import exact_two_intermediate
from src.path_evidence import (
    BANDS,
    EXPECTED_ONE_INTERMEDIATE_PATHS,
    EXPECTED_SHARED_DIRECT_PAIRS,
    EXPECTED_TWO_INTERMEDIATE_PATHS,
    accumulate_pair,
    attribute_agreement,
    count_band,
    empty_overlap_result,
    grouped_crosstab,
    merge_overlap,
    overlap_template2_positions_1_2,
    template_weights,
)

INPUT_RUN = Path(
    "/home/b5ac/freddieyu.b5ac/projects/project_2/results/joint_graph_and_evidence_audit_20261116T0404Z"
)
UNKNOWN_FIELDS = {"", "NOTINF"}
NODE_PACKS: list[list[dict]] = []


def _entity(entity_id: str) -> tuple[str, str]:
    kind, _, value = entity_id.partition(":")
    return kind, value


def _connection(relation: dict) -> tuple[str, str, str] | None:
    subject_kind, subject = _entity(relation["subject_entity_id"])
    object_kind, object_ = _entity(relation["object_entity_id"])
    family = relation.get("connectivity_family") or ""
    if family == "gene_cancer" or {subject_kind, object_kind} == {"gene", "cancer"}:
        gene = subject if subject_kind == "gene" else object_
        cancer = object_ if object_kind == "cancer" else subject
        if subject_kind == object_kind:
            return None
        return ("gene_cancer", gene, cancer)
    if subject_kind == "gene" and object_kind == "gene" and subject != object_:
        left, right = (subject, object_) if subject <= object_ else (object_, subject)
        return ("gene_gene", left, right)
    return None


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def load_connections(graph_dir: Path) -> dict[tuple[str, str, str], dict]:
    relations = {
        row["relation_id"]: row
        for row in pq.read_table(
            graph_dir / "relations.parquet",
            columns=[
                "relation_id",
                "resource",
                "subject_entity_id",
                "object_entity_id",
                "predicate",
                "base_predicate",
                "negation",
                "connectivity_family",
            ],
        ).to_pylist()
    }
    links: dict[str, list[str]] = defaultdict(list)
    for row in pq.read_table(graph_dir / "relation_evidence.parquet", columns=["relation_id", "evidence_id"]).to_pylist():
        links[row["evidence_id"]].append(row["relation_id"])
    connections: dict[tuple[str, str, str], dict] = {}
    evidence = pq.ParquetFile(graph_dir / "evidence_records.parquet")
    columns = [
        "evidence_id",
        "resource",
        "native_id",
        "pmid",
        "path_eligible",
        "predicate",
        "negation",
        "gcs_has_type",
        "cge_label",
        "ccs_label",
        "gci_label",
        "gcc_label",
    ]
    for batch in evidence.iter_batches(batch_size=200000, columns=columns):
        for row in batch.to_pylist():
            for relation_id in links.get(row["evidence_id"], ()):
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
                pmid = normalise_pmid(row.get("pmid")) or ""
                native = str(row.get("native_id") or row["evidence_id"])
                if row["resource"] == "core":
                    bucket["core"].setdefault(
                        native,
                        (
                            pmid,
                            row.get("cge_label") or "",
                            row.get("ccs_label") or "",
                            row.get("gci_label") or "",
                            row.get("gcc_label") or "",
                            row.get("gcs_has_type") or "",
                        ),
                    )
                else:
                    predicate = relation.get("base_predicate") or relation.get("predicate") or ""
                    negated = _as_bool(relation.get("negation")) or _as_bool(row.get("negation"))
                    traversal = _as_bool(row.get("path_eligible")) and not negated
                    subject_kind, subject = _entity(relation["subject_entity_id"])
                    object_kind, object_ = _entity(relation["object_entity_id"])
                    bucket["sem"].setdefault(
                        native,
                        (pmid, predicate, f"{subject_kind}:{subject}", f"{object_kind}:{object_}", negated, traversal),
                    )
    return connections


def _pmid_tuple(values: set[str], interned: dict[str, int]) -> tuple[int, ...]:
    numbers = []
    for pmid in values:
        if not pmid:
            continue
        number = interned.get(pmid)
        if number is None:
            number = len(interned) + 1
            interned[pmid] = number
        numbers.append(number)
    numbers.sort()
    return tuple(numbers)


def derive_profiles(connections: dict, candidates: set[str], queries: set[str]) -> tuple[list[dict], dict]:
    interned: dict[str, int] = {}
    signatures: dict[frozenset, int] = {frozenset(): 0}
    profiles = []
    shared_raw = {}
    for key, raw in connections.items():
        family, left, right = key
        core_rows = list(raw["core"].values())
        sem_rows = list(raw["sem"].values())
        observed = []
        informative = []
        missing_fields = 0
        core_pmids = set()
        core_without_pmid = 0
        for pmid, cge, ccs, gci, gcc, has_type in core_rows:
            observed.append((cge, ccs, gci, gcc, has_type))
            if informative_core_tuple(cge, ccs, gci, gcc, has_type):
                informative.append((cge, ccs, gci, gcc, has_type))
            missing_fields += sum(token in UNKNOWN_FIELDS for token in (cge, ccs, gci, gcc, has_type))
            if pmid:
                core_pmids.add(pmid)
            else:
                core_without_pmid += 1
        sem_pmids = set()
        negated_pmids = set()
        traversal_pmids = set()
        sem_without_pmid = 0
        traversal = []
        negated = []
        other = []
        for pmid, predicate, subject, object_, is_negated, is_traversal in sem_rows:
            combo = (predicate, subject, object_, "negated" if is_negated else "affirmative")
            if is_negated:
                negated.append(combo)
            elif is_traversal:
                traversal.append(combo)
            else:
                other.append(combo)
            if not pmid:
                sem_without_pmid += 1
                continue
            sem_pmids.add(pmid)
            if is_negated:
                negated_pmids.add(pmid)
            if is_traversal:
                traversal_pmids.add(pmid)
        info_key = frozenset(informative)
        info_sig = signatures.get(info_key)
        if info_sig is None:
            info_sig = len(signatures)
            signatures[info_key] = info_sig
        core_scope = "out_of_scope" if family == "gene_gene" else "in_scope"
        has_core = bool(core_rows)
        has_traversal = bool(traversal)
        if family == "gene_gene":
            support = "semmed_only" if has_traversal else "no_traversal_support"
        elif has_core and has_traversal:
            support = "both_sources"
        elif has_core:
            support = "core_only"
        elif has_traversal:
            support = "semmed_only"
        else:
            support = "no_traversal_support"
        if has_core and any(informative):
            core_info = "informative"
        elif has_core:
            core_info = "unknown_only"
        elif core_scope == "out_of_scope":
            core_info = "out_of_scope"
        else:
            core_info = "no_core_record"
        profile = {
            "family": family,
            "endpoint_a": left,
            "endpoint_b": right,
            "core_scope": core_scope,
            "support": support,
            "core_info": core_info,
            "sem_predicate": "present" if traversal else "absent",
            "sem_negation": "present" if negated else "absent",
            "info_sig": info_sig if core_scope == "in_scope" else 0,
            "n_core_sentences": len(core_rows),
            "n_core_observed_tuples": len(set(observed)),
            "n_core_informative_tuples": len(info_key),
            "n_missing_or_notinf_fields": missing_fields,
            "n_core_records_unknown_only": sum(1 for item in observed if item not in info_key) if has_core else 0,
            "n_semmed_predications": len(sem_rows),
            "n_semmed_traversal": len(traversal),
            "n_semmed_negated": len(negated),
            "n_semmed_other": len(other),
            "n_semmed_traversal_combos": len(set(traversal)),
            "n_semmed_all_combos": len(set(traversal) | set(negated) | set(other)),
            "n_pmid_core": len(core_pmids),
            "n_pmid_semmed": len(sem_pmids),
            "n_pmid_traversal": len(traversal_pmids),
            "n_pmid_intersection": len(core_pmids & sem_pmids),
            "n_pmid_core_only": len(core_pmids - sem_pmids),
            "n_pmid_semmed_only": len(sem_pmids - core_pmids),
            "n_pmid_union": len(core_pmids | sem_pmids),
            "n_core_records_without_pmid": core_without_pmid,
            "n_semmed_records_without_pmid": sem_without_pmid,
            "n_pmid_negated": len(negated_pmids),
            "core_pmids": _pmid_tuple(core_pmids, interned),
            "sem_pmids": _pmid_tuple(sem_pmids, interned),
            "all_pmids": _pmid_tuple(core_pmids | sem_pmids, interned),
            "traversal_labels": tuple(sorted({f"{item[0]}|{item[1].split(':', 1)[0]}_subject" for item in traversal})),
            "informative_labels": tuple(sorted("|".join(item) for item in info_key)),
            "in_candidate_query": family == "gene_cancer" and left in candidates and right in queries,
            "weights": {},
        }
        profiles.append(profile)
        if support == "both_sources" and family == "gene_cancer" and left in candidates and right in queries:
            shared_raw[(left, right)] = {
                "core": core_rows,
                "sem": sem_rows,
                "core_pmids": core_pmids,
                "sem_pmids": sem_pmids,
                "traversal_pmids": traversal_pmids,
                "negated_pmids": negated_pmids,
            }
    return profiles, shared_raw


def attach_weights(profiles: list[dict], candidates: set[str], queries: set[str]) -> dict:
    genes_of: dict[str, set[str]] = defaultdict(set)
    neighbors: dict[str, set[str]] = defaultdict(set)
    by_gene_cancer = {}
    by_gene_gene = {}
    for profile in profiles:
        if profile["support"] == "no_traversal_support":
            continue
        if profile["family"] == "gene_cancer":
            genes_of[profile["endpoint_b"]].add(profile["endpoint_a"])
            by_gene_cancer[(profile["endpoint_a"], profile["endpoint_b"])] = profile
        elif profile["support"] == "semmed_only":
            left, right = profile["endpoint_a"], profile["endpoint_b"]
            neighbors[left].add(right)
            neighbors[right].add(left)
            by_gene_gene[(left, right)] = profile
    weights = template_weights(genes_of, neighbors, candidates, queries)
    for key, weight in weights["one_gene_gene"].items():
        profile = by_gene_gene.get(key)
        if profile is not None:
            profile["weights"]["t1_pos1"] = weight
    for key, weight in weights["one_gene_cancer"].items():
        profile = by_gene_cancer.get(key)
        if profile is not None:
            profile["weights"]["t1_pos2"] = weight
    for name, mapping in (("t2_pos1", weights["two_pos1"]), ("t2_pos2", weights["two_pos2"]), ("t2_pos3", weights["two_pos3"])):
        for key, weight in mapping.items():
            profile = by_gene_cancer.get(key)
            if profile is not None:
                profile["weights"][name] = weight
    bitset_total = sum(
        row["n_paths"]
        for row in exact_two_intermediate(genes_of, candidates, sorted(queries), sorted(set(queries) | set(genes_of))).values()
    )
    weights["bitset_two_total"] = int(bitset_total)
    weights["genes_of"] = genes_of
    weights["neighbors"] = neighbors
    return weights


def _add(counter: Counter, key, amount: int) -> None:
    if amount:
        counter[key] += amount


def position_tables(profiles: list[dict]) -> Counter:
    """Exhaustive and multi-label distributions at each template position."""
    positions = {
        "t1_pos1": ("one_intermediate_path", "position_1_gene_gene"),
        "t1_pos2": ("one_intermediate_path", "position_2_gene_cancer"),
        "t2_pos1": ("two_intermediate_path", "position_1_candidate_to_intermediate_cancer"),
        "t2_pos2": ("two_intermediate_path", "position_2_intermediate_cancer_to_intermediate_gene"),
        "t2_pos3": ("two_intermediate_path", "position_3_intermediate_gene_to_query_cancer"),
    }
    counts: Counter = Counter()
    for profile in profiles:
        for weight_name, (template, position) in positions.items():
            weight = int(profile["weights"].get(weight_name, 0))
            if weight <= 0:
                continue
            for weighting, amount in (("distinct_count", 1), ("path_occurrence_count", weight)):
                prefix = (template, position, weighting)
                _add(counts, (*prefix, "edges", "all"), amount)
                _add(counts, (*prefix, "support_category", profile["support"]), amount)
                _add(counts, (*prefix, "core_scope", profile["core_scope"]), amount)
                if profile["core_scope"] == "out_of_scope":
                    _add(counts, (*prefix, "core_sentence_count", "out_of_scope"), amount)
                    _add(counts, (*prefix, "core_information", "out_of_scope"), amount)
                else:
                    _add(counts, (*prefix, "core_sentence_count", str(profile["n_core_sentences"])), amount)
                    _add(counts, (*prefix, "core_information", profile["core_info"]), amount)
                    _add(counts, (*prefix, "core_sentence_band", count_band(profile["n_core_sentences"])), amount)
                    _add(counts, (*prefix, "core_informative_tuple_band", count_band(profile["n_core_informative_tuples"])), amount)
                _add(counts, (*prefix, "semmed_traversal_count", str(profile["n_semmed_traversal"])), amount)
                _add(counts, (*prefix, "semmed_traversal_band", count_band(profile["n_semmed_traversal"])), amount)
                _add(counts, (*prefix, "semmed_negated_band", count_band(profile["n_semmed_negated"])), amount)
                _add(counts, (*prefix, "pmid_union_band", count_band(profile["n_pmid_union"])), amount)
                _add(counts, (*prefix, "pmid_core_band", count_band(profile["n_pmid_core"]) if profile["core_scope"] == "in_scope" else "out_of_scope"), amount)
                _add(counts, (*prefix, "pmid_semmed_band", count_band(profile["n_pmid_semmed"])), amount)
                _add(counts, (*prefix, "pmid_traversal_band", count_band(profile["n_pmid_traversal"])), amount)
                _add(
                    counts,
                    (
                        *prefix,
                        "support_by_article_band",
                        f"{profile['support']}|{count_band(profile['n_pmid_union'])}",
                    ),
                    amount,
                )
                core_band = "out_of_scope" if profile["core_scope"] == "out_of_scope" else count_band(profile["n_pmid_core"])
                _add(counts, (*prefix, "pmid_core_by_semmed_band", f"{core_band}|{count_band(profile['n_pmid_semmed'])}"), amount)
                _add(counts, (*prefix, "core_info_by_semmed_predicate", f"{profile['core_info']}|{profile['sem_predicate']}"), amount)
                for label in profile["traversal_labels"]:
                    _add(counts, (*prefix, "semmed_predicate_orientation", label), amount)
                for label in profile["informative_labels"]:
                    _add(counts, (*prefix, "core_informative_tuple", label), amount)
                if profile["sem_negation"] == "present":
                    _add(counts, (*prefix, "supplementary_negation", "present"), amount)
    return counts


def _node_from_profile(profile: dict, extend: int, cand: bool) -> dict:
    return {
        "gene": profile["endpoint_a"],
        "cand": cand,
        "extend": extend,
        "support": profile["support"],
        "all_pmids": profile["all_pmids"],
        "core_pmids": profile["core_pmids"],
        "sem_pmids": profile["sem_pmids"],
        "info_sig": profile["info_sig"],
        "availability": f"{profile['core_info']}|{profile['sem_predicate']}|{profile['sem_negation']}",
        "article_band": count_band(profile["n_pmid_union"]),
    }


def build_cancer_nodes(profiles: list[dict], weights: dict, candidates: set[str]) -> list[list[dict]]:
    right_of = weights["right_of"]
    by_cancer: dict[str, list[dict]] = defaultdict(list)
    for profile in profiles:
        if profile["family"] != "gene_cancer" or profile["support"] == "no_traversal_support":
            continue
        gene, cancer = profile["endpoint_a"], profile["endpoint_b"]
        extend = int(right_of.get((gene, cancer), 0))
        by_cancer[cancer].append(_node_from_profile(profile, extend, gene in candidates))
    packs = list(by_cancer.values())
    packs.sort(key=len, reverse=True)
    return packs


def select_packs(packs: list[list[dict]], cancer_limit: int) -> tuple[list[list[dict]], str]:
    if cancer_limit <= 0 or cancer_limit >= len(packs):
        return packs, "all_cancers"
    smallest = [pack for pack in packs if len(pack) > 1][-min(3, len(packs)) :]
    chosen = packs[:cancer_limit] + [pack for pack in smallest if pack not in packs[:cancer_limit]]
    return chosen, f"high_degree_top_{cancer_limit}_plus_small_cancers"


def _worker(index: int) -> dict:
    return overlap_template2_positions_1_2(NODE_PACKS[index])


def overlap_positions_1_2(packs: list[list[dict]], workers: int) -> dict:
    global NODE_PACKS
    NODE_PACKS = packs
    if workers <= 1 or len(packs) <= 1:
        merged = empty_overlap_result()
        for pack in packs:
            merge_overlap(merged, overlap_template2_positions_1_2(pack))
        return merged
    import multiprocessing as mp

    with mp.Pool(workers) as pool:
        parts = pool.map(_worker, range(len(packs)), chunksize=1)
    merged = empty_overlap_result()
    for part in parts:
        merge_overlap(merged, part)
    return merged


def overlap_other_positions(profiles: list[dict], weights: dict, candidates: set[str]) -> tuple[dict, dict]:
    by_gene_cancer = {
        (profile["endpoint_a"], profile["endpoint_b"]): profile
        for profile in profiles
        if profile["family"] == "gene_cancer" and profile["support"] != "no_traversal_support"
    }
    by_gene_gene = {
        (profile["endpoint_a"], profile["endpoint_b"]): profile
        for profile in profiles
        if profile["family"] == "gene_gene" and profile["support"] == "semmed_only"
    }
    template1 = empty_overlap_result()
    for gene, query_set in weights["queries_of"].items():
        partners = [item for item in weights["neighbors"].get(gene, ()) if item in candidates and item != gene]
        if not partners:
            continue
        for partner in partners:
            left_profile = by_gene_gene.get((partner, gene) if partner <= gene else (gene, partner))
            if left_profile is None:
                continue
            left = _node_from_profile(left_profile, 1, True)
            for query in query_set:
                right_profile = by_gene_cancer.get((gene, query))
                if right_profile is None:
                    continue
                right = _node_from_profile(right_profile, 1, gene in candidates)
                accumulate_pair(template1, left, right, 1)
    positions_2_3 = empty_overlap_result()
    left_of = weights["left_of"]
    cancers_of: dict[str, list[str]] = defaultdict(list)
    for (gene, cancer), profile in by_gene_cancer.items():
        cancers_of[gene].append(cancer)
    for gene, cancers in cancers_of.items():
        queries = weights["queries_of"].get(gene, ())
        if not queries:
            continue
        for cancer in cancers:
            n_left = int(left_of.get((gene, cancer), 0))
            if not n_left:
                continue
            left_profile = by_gene_cancer[(gene, cancer)]
            left = _node_from_profile(left_profile, n_left, gene in candidates)
            for query in queries:
                if query == cancer:
                    continue
                right_profile = by_gene_cancer.get((gene, query))
                if right_profile is None:
                    continue
                right = _node_from_profile(right_profile, 1, gene in candidates)
                accumulate_pair(positions_2_3, left, right, n_left)
    return template1, positions_2_3


def _product(groups_left: Counter, groups_right: Counter, distinct: Counter, weighted: Counter) -> None:
    for left, n_left in groups_left.items():
        for right, weight in groups_right.items():
            distinct[(left, right)] += n_left
            weighted[(left, right)] += n_left * weight


def categorical_other_positions(profiles: list[dict], weights: dict, candidates: set[str]) -> dict[str, tuple[Counter, Counter]]:
    """Support, availability, and article-band combinations outside the quadratic loop."""
    by_gene_cancer = {
        (profile["endpoint_a"], profile["endpoint_b"]): profile
        for profile in profiles
        if profile["family"] == "gene_cancer" and profile["support"] != "no_traversal_support"
    }
    by_gene_gene = {
        (profile["endpoint_a"], profile["endpoint_b"]): profile
        for profile in profiles
        if profile["family"] == "gene_gene" and profile["support"] == "semmed_only"
    }
    tables = {
        name: (Counter(), Counter())
        for name in (
            "t1_support",
            "t1_availability",
            "t1_article",
            "t23_support",
            "t23_availability",
            "t23_article",
        )
    }
    for gene, query_set in weights["queries_of"].items():
        partners = [item for item in weights["neighbors"].get(gene, ()) if item in candidates and item != gene]
        if not partners:
            continue
        left_groups = {key: Counter() for key in ("support", "availability", "article_band")}
        for partner in partners:
            profile = by_gene_gene.get((partner, gene) if partner <= gene else (gene, partner))
            if profile is None:
                continue
            node = _node_from_profile(profile, 1, True)
            for key in left_groups:
                left_groups[key][node[key]] += 1
        for query in query_set:
            profile = by_gene_cancer.get((gene, query))
            if profile is None:
                continue
            node = _node_from_profile(profile, 1, gene in candidates)
            _product(left_groups["support"], Counter({node["support"]: 1}), *tables["t1_support"])
            _product(left_groups["availability"], Counter({node["availability"]: 1}), *tables["t1_availability"])
            _product(left_groups["article_band"], Counter({node["article_band"]: 1}), *tables["t1_article"])
    cancers_of: dict[str, list[str]] = defaultdict(list)
    for gene, cancer in by_gene_cancer:
        cancers_of[gene].append(cancer)
    left_of = weights["left_of"]
    for gene, cancers in cancers_of.items():
        queries = [query for query in weights["queries_of"].get(gene, ()) if (gene, query) in by_gene_cancer]
        if not queries:
            continue
        for cancer in cancers:
            n_left = int(left_of.get((gene, cancer), 0))
            if not n_left:
                continue
            left = _node_from_profile(by_gene_cancer[(gene, cancer)], n_left, gene in candidates)
            for query in queries:
                if query == cancer:
                    continue
                right = _node_from_profile(by_gene_cancer[(gene, query)], 1, gene in candidates)
                for table, field in (("t23_support", "support"), ("t23_availability", "availability"), ("t23_article", "article_band")):
                    distinct, weighted = tables[table]
                    distinct[(left[field], right[field])] += 1
                    weighted[(left[field], right[field])] += n_left
    return tables


def shared_pair_tables(shared_raw: dict, profiles: list[dict]) -> tuple[list[dict], list[dict], dict, list[tuple]]:
    weight_of = {
        (profile["endpoint_a"], profile["endpoint_b"]): profile["weights"]
        for profile in profiles
        if profile["family"] == "gene_cancer"
    }
    rows = []
    details = []
    added_pmids = set()
    summary = Counter()
    detail_links: Counter = Counter()
    detail_pairs: dict[str, set] = defaultdict(set)
    for (gene, cancer), raw in sorted(shared_raw.items()):
        core = raw["core_pmids"]
        sem = raw["sem_pmids"]
        added = sem - core
        added_pmids |= added
        weights = weight_of.get((gene, cancer), {})
        row = {
            "gene_id": gene,
            "cancer_cui": cancer,
            "n_core_pmid": len(core),
            "n_semmed_pmid": len(sem),
            "n_intersection": len(core & sem),
            "n_semmed_only": len(added),
            "n_core_only": len(core - sem),
            "n_union": len(core | sem),
            "n_traversal_pmid_not_in_core": len(raw["traversal_pmids"] - core),
            "n_negated_pmid_not_in_core": len(raw["negated_pmids"] - core),
            "weight_t1_pos2": int(weights.get("t1_pos2", 0)),
            "weight_t2_pos1": int(weights.get("t2_pos1", 0)),
            "weight_t2_pos2": int(weights.get("t2_pos2", 0)),
            "weight_t2_pos3": int(weights.get("t2_pos3", 0)),
        }
        rows.append(row)
        summary["pairs"] += 1
        summary["added_links"] += len(added)
        summary["pairs_with_added_pmid"] += 1 if added else 0
        summary["pairs_with_shared_pmid"] += 1 if core & sem else 0
        summary["shared_links"] += len(core & sem)
        summary["core_pmid_links"] += len(core)
        summary["semmed_pmid_links"] += len(sem)
        for name in ("weight_t1_pos2", "weight_t2_pos1", "weight_t2_pos2", "weight_t2_pos3"):
            summary[name] += row[name]
            summary[f"{name}_pairs"] += 1 if row[name] else 0
        summary[f"added_band_{count_band(len(added))}"] += 1
        by_core = defaultdict(set)
        by_sem = defaultdict(lambda: {"traversal": set(), "negated": set(), "other": set()})
        for pmid, cge, ccs, gci, gcc, has_type in raw["core"]:
            if pmid:
                by_core[pmid].add("|".join((cge, ccs, gci, gcc, has_type)))
        for pmid, predicate, subject, object_, is_negated, is_traversal in raw["sem"]:
            if not pmid:
                continue
            label = f"{predicate}|{subject}|{object_}|{'negated' if is_negated else 'affirmative'}"
            kind = "negated" if is_negated else "traversal" if is_traversal else "other"
            by_sem[pmid][kind].add(label)
        for pmid in sorted(core & sem):
            traversal_labels = sorted(by_sem[pmid]["traversal"])
            negated_labels = sorted(by_sem[pmid]["negated"])
            other_labels = sorted(by_sem[pmid]["other"])
            details.append(
                {
                    "gene_id": gene,
                    "cancer_cui": cancer,
                    "pmid": pmid,
                    "core_attributes": " || ".join(sorted(by_core.get(pmid, ()))),
                    "semmed_traversal_details": " || ".join(traversal_labels),
                    "semmed_negated_details": " || ".join(negated_labels),
                    "semmed_other_details": " || ".join(other_labels),
                }
            )
            for kind, labels in (("traversal", traversal_labels), ("negated", negated_labels), ("other", other_labels)):
                if labels:
                    detail_links[f"links_with_{kind}_detail"] += 1
                    detail_pairs[f"pairs_with_{kind}_detail"].add((gene, cancer))
                for label in labels:
                    detail_links[f"link_label|{kind}|{label}"] += 1
                    detail_pairs[f"pair_label|{kind}|{label}"].add((gene, cancer))
    summary["distinct_added_pmids"] = len(added_pmids)
    detail_rows_summary = [(key, value) for key, value in sorted(detail_links.items())]
    detail_rows_summary.extend((key, len(value)) for key, value in sorted(detail_pairs.items()))
    return rows, details, summary, detail_rows_summary


def _write_counter_tsv(path: Path, rows: list[tuple], header: str) -> None:
    lines = [header]
    for row in rows:
        lines.append("\t".join(str(item) for item in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _overlap_rows(result: dict, template: str, positions: str, scope: str) -> list[dict]:
    rows = []
    for comparison, acc in (
        ("union", result["union"]),
        ("core_core", result["core_core"]),
        ("core_semmed", result["core_semmed"]),
        ("semmed_core", result["semmed_core"]),
        ("semmed_semmed", result["semmed_semmed"]),
    ):
        for weighting, distinct_name, weighted_name in (
            ("distinct_count", "neither_distinct", "neither_weighted"),
            ("path_occurrence_count", "neither_distinct", "neither_weighted"),
        ):
            pass
        for availability, distinct_value, weighted_value in (
            ("neither", acc.neither_distinct, acc.neither_weighted),
            ("one_sided", acc.one_sided_distinct, acc.one_sided_weighted),
            ("both_known", acc.both_distinct, acc.both_weighted),
        ):
            for weighting, value in (("distinct_count", distinct_value), ("path_occurrence_count", weighted_value)):
                rows.append(
                    {
                        "template": template,
                        "positions": positions,
                        "comparison": comparison,
                        "weighting": weighting,
                        "availability": availability,
                        "overlap_band": "",
                        "support_left": "ALL",
                        "support_right": "ALL",
                        "count": int(value),
                        "scope": scope,
                    }
                )
        for index, band in enumerate(BANDS):
            for weighting, series in (("distinct_count", acc.band_distinct), ("path_occurrence_count", acc.band_weighted)):
                rows.append(
                    {
                        "template": template,
                        "positions": positions,
                        "comparison": comparison,
                        "weighting": weighting,
                        "availability": "both_known",
                        "overlap_band": band,
                        "support_left": "ALL",
                        "support_right": "ALL",
                        "count": int(series[index]),
                        "scope": scope,
                    }
                )
    for key, value in result["joint_distinct"].items():
        rows.append(
            {
                "template": template,
                "positions": positions,
                "comparison": "union",
                "weighting": "distinct_count",
                "availability": key[2],
                "overlap_band": "",
                "support_left": key[0],
                "support_right": key[1],
                "count": int(value),
                "scope": scope,
            }
        )
    for key, value in result["joint_weighted"].items():
        rows.append(
            {
                "template": template,
                "positions": positions,
                "comparison": "union",
                "weighting": "path_occurrence_count",
                "availability": key[2],
                "overlap_band": "",
                "support_left": key[0],
                "support_right": key[1],
                "count": int(value),
                "scope": scope,
            }
        )
    return rows


def _counter_brief(distinct: Counter, weighted: Counter) -> str:
    labels = sorted(set(distinct) | set(weighted))
    d_total = sum(distinct.values()) or 1
    w_total = sum(weighted.values()) or 1
    parts = []
    for label in labels:
        parts.append(
            f"{label[0]} × {label[1]}: distinct {distinct[label]} / {sum(distinct.values())} ({distinct[label] / d_total:.3f}), "
            f"occurrence {weighted[label]} / {sum(weighted.values())} ({weighted[label] / w_total:.3f})"
        )
    return "; ".join(parts)


def _overlap_brief(named: dict) -> str:
    parts = []
    for name, (result, scope) in named.items():
        acc = result["union"]
        both = acc.both_weighted
        shared = both - acc.band_weighted[0]
        share = shared / both if both else 0
        parts.append(
            f"{name} [{scope}]: occurrence pairs {acc.weighted}; both-known PMID sets {both}; "
            f"zero intersection {acc.band_weighted[0]}; at least one shared PMID {shared} ({share:.3f}); "
            f"one-sided {acc.one_sided_weighted}; neither {acc.neither_weighted}"
        )
    return " ".join(parts)


def _crosstab_rows(counter: Counter, template: str, positions: str, table: str, weighting: str) -> list[dict]:
    total = sum(value for value in counter.values() if value > 0)
    return [
        {
            "template": template,
            "positions": positions,
            "weighting": weighting,
            "table": table,
            "left": left,
            "right": right,
            "count": int(value),
            "denominator": int(total),
        }
        for (left, right), value in sorted(counter.items())
        if value > 0
    ]


def render_report(summary: dict) -> str:
    shared = summary["shared"]
    checks = summary["checks"]
    return f"""# Path evidence audit

This audit describes source records, relation details, and PubMed articles on the edges of the fixed joint graph. A shared PMID is shared article provenance. It is not evidence that two records come from the same experiment.

Reproduction of the input audit: `one_intermediate_path` {checks['one_total']} (expected {EXPECTED_ONE_INTERMEDIATE_PATHS}); `two_intermediate_path` {checks['two_total']} (expected {EXPECTED_TWO_INTERMEDIATE_PATHS}); shared direct gene–cancer pairs {checks['shared_pairs']} (expected {EXPECTED_SHARED_DIRECT_PAIRS}).

## 1. Evidence at each edge position

`one_intermediate_path` has two edges: a SemMedDB gene–gene edge, then a gene–cancer edge. CORE is outside the scope of the gene–gene edge, which is a different state from a gene–cancer edge whose CORE records are absent. `two_intermediate_path` has three gene–cancer edges.

Distinct counts give each qualifying edge once. Path-occurrence counts weight that edge by the number of eligible complete paths that use it at that position. The occurrence weights at every position sum to the complete-path total: one-intermediate {checks['one_total']}; two-intermediate position sums {checks['two_pos_sums']}.

On the two-edge template, the gene–cancer edge is CORE-supported, SemMedDB-supported, or both. Occurrence weighting shifts the mix toward edges that sit next to many candidate genes. On the three-edge template, occurrence weighting shifts attention toward the high-degree intermediate cancers. The largest distinct-versus-weighted share differences are: {summary['largest_weight_shifts']}.

Exact count histograms, count bands, and the joint band tables are in `analysis/edge_position_distributions.tsv`. Connection-level counts and weights are in `analysis/edge_profiles.parquet`. Negated SemMedDB predications stay in the supplementary counts and do not qualify an edge for traversal.

## 2. Adjacent edges

For the two-edge template, an adjacent edge pair is one complete path, so distinct counts and path-occurrence counts are the same number, {checks['one_total']}. For the three-edge template they differ. Positions 1–2 share the intermediate cancer. Their occurrence weight is the number of eligible final edges. Positions 2–3 share the intermediate gene. Their occurrence weight is the number of eligible initial edges. Both occurrence sums equal {checks['two_total']}.

The first edge of every two-edge path is a SemMedDB gene–gene predication. CORE is outside the scope of that edge. Resource combinations, with each ordered adjacent pair counted once and weighted by complete paths:

- Two-edge positions 1–2: {summary['template1_support_brief']}
- Three-edge positions 1–2: {summary['support_brief']}
- Three-edge positions 2–3: {summary['positions_2_3_support_brief']}

Relation-availability and article-band combinations are in `analysis/adjacent_edge_combinations.parquet`. Multi-label CORE tuple and SemMedDB predicate rows in the edge-position table can overlap, so those rows are not a partition of the denominator.

Article overlap uses known PMIDs only. A one-sided or empty PMID set is reported separately from a zero intersection of two known PMID sets. Source-specific comparisons keep the position of each resource: CORE–CORE, CORE–SemMedDB, SemMedDB–CORE, and SemMedDB–SemMedDB. {summary['overlap_brief']}

Where the original two-gene CORE comparison applies, at positions 1–2 of the three-edge template, informative tuple sets are {summary['attribute_agreement']}. Equal tuples are agreement of recorded attributes, not a shared experiment.

## 3. SemMedDB on the shared direct pairs

The fixed universe contains {shared['pairs']} gene–cancer pairs recorded by both resources. SemMedDB adds at least one PMID on {shared['pairs_with_added_pmid']} pairs. The added pair–PMID links total {shared['added_links']}, and those links represent {shared['distinct_added_pmids']} distinct PMIDs. Pairs with at least one PMID already present in both resources: {shared['pairs_with_shared_pmid']}. Shared pair–PMID links: {shared['shared_links']}.

An added PMID and a new structured field on a PMID already recorded by CORE are separate results. On each shared PMID, CORE contributes its sentence attribute tuple and SemMedDB contributes predicate, asserted subject/object orientation, and negation. Those vocabularies are summarised side by side in `analysis/shared_article_relation_details.parquet`. Negated predications remain identifiable and are not rewritten.

Path-occurrence weights of these pairs, including pairs that lie on no complete path, are the column sums in `analysis/shared_pair_pmids.parquet`: two-edge position 2 = {shared['weight_t1_pos2']}; three-edge positions 1, 2, and 3 = {shared['weight_t2_pos1']}, {shared['weight_t2_pos2']}, {shared['weight_t2_pos3']}. Pairs with a positive weight at those positions: {shared['weight_t1_pos2_pairs']}, {shared['weight_t2_pos1_pairs']}, {shared['weight_t2_pos2_pairs']}, {shared['weight_t2_pos3_pairs']}.

## Method

Edge profiles were built once from relation and evidence records, deduplicated by source-native identifier. Complete-path totals use the same adjacency rules as the input audit. Three-edge occurrence weights use the extension product after removing repeated genes and repeated cancers. Two-edge adjacent pairs were enumerated because each pair is one path. Three-edge positions 1–2 use a per-cancer loop over candidate and intermediate genes, including the high-degree cancers, and positions 2–3 use a per-gene loop. Categorical cross-tabulations use grouped extension counts rather than stored paths. Counters are Python integers. Overlap scope: {summary['overlap_scope']}.
"""


def _largest_shifts(position_counts: Counter) -> str:
    shifts = []
    grouped = defaultdict(lambda: {"distinct_count": Counter(), "path_occurrence_count": Counter()})
    for (template, position, weighting, measure, label), value in position_counts.items():
        if measure != "support_category":
            continue
        grouped[(template, position)][weighting][label] += value
    for key, weights in grouped.items():
        distinct = weights["distinct_count"]
        weighted = weights["path_occurrence_count"]
        d_total = sum(distinct.values()) or 1
        w_total = sum(weighted.values()) or 1
        for label in set(distinct) | set(weighted):
            delta = weighted[label] / w_total - distinct[label] / d_total
            shifts.append((abs(delta), delta, key, label))
    shifts.sort(reverse=True)
    parts = []
    for _, delta, key, label in shifts[:4]:
        parts.append(f"{key[0]} {key[1]} {label} {delta:+.3f}")
    return "; ".join(parts) if parts else "none"


def _agreement_text(counter_d: Counter, counter_w: Counter) -> str:
    def brief(counter: Counter) -> str:
        total = sum(counter.values()) or 1
        return ", ".join(f"{key} {value} / {sum(counter.values())} ({value / total:.3f})" for key, value in sorted(counter.items()))

    return f"distinct [{brief(counter_d)}]; path-occurrence [{brief(counter_w)}]"


def write_outputs(run_dir: Path, profiles, position_counts, combinations, overlap_rows, shared_rows, detail_rows, detail_summary, summary) -> None:
    analysis = ensure_dir(run_dir / "analysis")
    profile_columns = [
        "family",
        "endpoint_a",
        "endpoint_b",
        "core_scope",
        "support",
        "core_info",
        "sem_predicate",
        "sem_negation",
        "n_core_sentences",
        "n_core_observed_tuples",
        "n_core_informative_tuples",
        "n_missing_or_notinf_fields",
        "n_core_records_unknown_only",
        "n_semmed_predications",
        "n_semmed_traversal",
        "n_semmed_negated",
        "n_semmed_other",
        "n_semmed_traversal_combos",
        "n_semmed_all_combos",
        "n_pmid_core",
        "n_pmid_semmed",
        "n_pmid_traversal",
        "n_pmid_intersection",
        "n_pmid_core_only",
        "n_pmid_semmed_only",
        "n_pmid_union",
        "n_core_records_without_pmid",
        "n_semmed_records_without_pmid",
        "n_pmid_negated",
        "weight_t1_pos1",
        "weight_t1_pos2",
        "weight_t2_pos1",
        "weight_t2_pos2",
        "weight_t2_pos3",
    ]
    profile_rows = []
    for profile in profiles:
        weights = profile["weights"]
        profile_rows.append({column: profile.get(column, weights.get(column, 0) if column.startswith("weight_") else "") for column in profile_columns})
        for column in profile_columns:
            if column.startswith("weight_"):
                profile_rows[-1][column] = int(weights.get(column, 0))
            elif column.startswith("n_"):
                profile_rows[-1][column] = int(profile[column])
    def _table(rows: list[dict], columns: list[str]) -> pa.Table:
        if rows:
            return pa.Table.from_pylist(rows)
        return pa.table({name: pa.array([], type=pa.string()) for name in columns})

    pq.write_table(_table(profile_rows, profile_columns), analysis / "edge_profiles.parquet")
    _write_counter_tsv(
        analysis / "edge_position_distributions.tsv",
        [(*key, value) for key, value in sorted(position_counts.items())],
        "template\tposition\tweighting\tmeasure\tlabel\tcount",
    )
    pq.write_table(_table(combinations, ["template", "positions", "weighting", "table", "left", "right", "count", "denominator"]), analysis / "adjacent_edge_combinations.parquet")
    pq.write_table(_table(overlap_rows, ["template", "positions", "comparison", "weighting", "availability", "overlap_band", "support_left", "support_right", "count", "scope"]), analysis / "adjacent_article_overlap.parquet")
    pq.write_table(_table(shared_rows, ["gene_id", "cancer_cui"]), analysis / "shared_pair_pmids.parquet")
    pq.write_table(_table(detail_rows, ["gene_id", "cancer_cui", "pmid"]), analysis / "shared_article_relation_details.parquet")
    shared = summary["shared"]
    _write_counter_tsv(
        analysis / "shared_pair_summary.tsv",
        [(key, shared[key]) for key in sorted(shared)],
        "measure\tvalue",
    )
    _write_counter_tsv(
        analysis / "shared_article_detail_summary.tsv",
        detail_summary,
        "measure\tvalue",
    )
    (run_dir / "report.md").write_text(render_report(summary), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit evidence on path edges and adjacent edges.")
    parser.add_argument("--input-run", type=Path, default=INPUT_RUN)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cancer-limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    timer = StageTimer()
    run_dir = ensure_dir(args.run_dir)
    ensure_dir(run_dir / "logs")
    graph_dir = args.input_run / "graph"
    analysis_dir = args.input_run / "analysis"
    candidates = {
        row["gene_id"]
        for row in pq.read_table(analysis_dir / "candidate_genes.parquet", columns=["gene_id", "status"]).to_pylist()
        if row["status"] == "resolved_human"
    }
    queries = {row["cancer_cui"] for row in pq.read_table(analysis_dir / "cancer_queries.parquet", columns=["cancer_cui"]).to_pylist()}
    connections = load_connections(graph_dir)
    timer.mark("load_graph", connections=len(connections))
    profiles, shared_raw = derive_profiles(connections, candidates, queries)
    del connections
    timer.mark("profiles", profiles=len(profiles), shared_pairs=len(shared_raw))
    weights = attach_weights(profiles, candidates, queries)
    timer.mark("weights", one_total=weights["one_total"], two_total=weights["two_total"], bitset_two=weights["bitset_two_total"])
    position_counts = position_tables(profiles)
    packs = build_cancer_nodes(profiles, weights, candidates)
    selected, overlap_scope = select_packs(packs, args.cancer_limit)
    support_d: Counter = Counter()
    support_w: Counter = Counter()
    agreement_d: Counter = Counter()
    agreement_w: Counter = Counter()
    availability_d: Counter = Counter()
    availability_w: Counter = Counter()
    article_d: Counter = Counter()
    article_w: Counter = Counter()
    for pack in packs:
        for table, distinct, weighted in (
            ("support", *grouped_crosstab(pack, "support", "support")),
            ("availability", *grouped_crosstab(pack, "availability", "availability")),
            ("article_band", *grouped_crosstab(pack, "article_band", "article_band")),
            ("info_sig", *grouped_crosstab(pack, "info_sig", "info_sig")),
        ):
            target_d, target_w = {
                "support": (support_d, support_w),
                "availability": (availability_d, availability_w),
                "article_band": (article_d, article_w),
                "info_sig": (agreement_d, agreement_w),
            }[table]
            if table == "info_sig":
                for (left, right), value in distinct.items():
                    agreement_d[attribute_agreement(int(left), int(right))] += value
                for (left, right), value in weighted.items():
                    agreement_w[attribute_agreement(int(left), int(right))] += value
            else:
                target_d.update(distinct)
                target_w.update(weighted)
    timer.mark("grouped_crosstabs", cancers=len(packs))
    overlap_12 = overlap_positions_1_2(selected, args.workers)
    timer.mark("overlap_positions_1_2", scope=overlap_scope, cancers=len(selected))
    if args.cancer_limit > 0:
        overlap_t1, overlap_23 = empty_overlap_result(), empty_overlap_result()
        other_scope = "not_run_in_subset_profile"
    else:
        overlap_t1, overlap_23 = overlap_other_positions(profiles, weights, candidates)
        other_scope = "all_eligible_pairs"
    timer.mark("overlap_other_positions", scope=other_scope)
    shared_rows, detail_rows, shared_summary, detail_summary = shared_pair_tables(shared_raw, profiles)
    other_tables = categorical_other_positions(profiles, weights, candidates)
    timer.mark("shared_pairs", pairs=shared_summary["pairs"], details=len(detail_rows))
    checks = {
        "one_total": weights["one_total"],
        "two_total": weights["two_total"],
        "bitset_two_total": weights["bitset_two_total"],
        "shared_pairs": shared_summary["pairs"],
        "two_pos_sums": {
            "position_1": sum(weights["two_pos1"].values()),
            "position_2": sum(weights["two_pos2"].values()),
            "position_3": sum(weights["two_pos3"].values()),
        },
        "one_pos_sums": {
            "position_1": sum(weights["one_gene_gene"].values()),
            "position_2": sum(weights["one_gene_cancer"].values()),
        },
        "adjacent_support_weighted": sum(support_w.values()),
        "adjacent_support_distinct": sum(support_d.values()),
        "agreement_weighted": sum(agreement_w.values()),
        "template1_support_weighted": sum(other_tables["t1_support"][1].values()),
        "template1_support_distinct": sum(other_tables["t1_support"][0].values()),
        "positions_2_3_support_weighted": sum(other_tables["t23_support"][1].values()),
        "positions_2_3_support_distinct": sum(other_tables["t23_support"][0].values()),
        "overlap_12_weighted": overlap_12["union"].weighted,
        "overlap_t1_weighted": overlap_t1["union"].weighted,
        "overlap_23_weighted": overlap_23["union"].weighted,
        "overlap_scope": overlap_scope,
    }
    attached = Counter()
    for profile in profiles:
        for name, value in profile["weights"].items():
            attached[name] += int(value)
    checks["attached_weights"] = dict(attached)
    failures = []
    if checks["one_total"] != EXPECTED_ONE_INTERMEDIATE_PATHS:
        failures.append({"name": "one_intermediate_paths", "observed": checks["one_total"], "expected": EXPECTED_ONE_INTERMEDIATE_PATHS})
    if checks["two_total"] != EXPECTED_TWO_INTERMEDIATE_PATHS or checks["bitset_two_total"] != EXPECTED_TWO_INTERMEDIATE_PATHS:
        failures.append({"name": "two_intermediate_paths", "observed": checks["two_total"], "bitset": checks["bitset_two_total"], "expected": EXPECTED_TWO_INTERMEDIATE_PATHS})
    if checks["shared_pairs"] != EXPECTED_SHARED_DIRECT_PAIRS:
        failures.append({"name": "shared_direct_pairs", "observed": checks["shared_pairs"], "expected": EXPECTED_SHARED_DIRECT_PAIRS})
    for name, value in checks["two_pos_sums"].items():
        if value != checks["two_total"]:
            failures.append({"name": f"two_weight_{name}", "observed": value, "expected": checks["two_total"]})
    for name, value in checks["one_pos_sums"].items():
        if value != checks["one_total"]:
            failures.append({"name": f"one_weight_{name}", "observed": value, "expected": checks["one_total"]})
    if overlap_scope == "all_cancers" and checks["overlap_12_weighted"] != checks["two_total"]:
        failures.append({"name": "adjacent_1_2_occurrence", "observed": checks["overlap_12_weighted"], "expected": checks["two_total"]})
    if checks["adjacent_support_weighted"] != checks["two_total"]:
        failures.append({"name": "support_crosstab_occurrence", "observed": checks["adjacent_support_weighted"], "expected": checks["two_total"]})
    if checks["agreement_weighted"] != checks["two_total"]:
        failures.append({"name": "tuple_agreement_occurrence", "observed": checks["agreement_weighted"], "expected": checks["two_total"]})
    if checks["template1_support_weighted"] != checks["one_total"] or checks["template1_support_distinct"] != checks["one_total"]:
        failures.append({"name": "template1_adjacent_support", "observed": checks["template1_support_weighted"], "distinct": checks["template1_support_distinct"], "expected": checks["one_total"]})
    if checks["positions_2_3_support_weighted"] != checks["two_total"]:
        failures.append({"name": "positions_2_3_support", "observed": checks["positions_2_3_support_weighted"], "expected": checks["two_total"]})
    expected_attached = {
        "t1_pos1": checks["one_total"],
        "t1_pos2": checks["one_total"],
        "t2_pos1": checks["two_total"],
        "t2_pos2": checks["two_total"],
        "t2_pos3": checks["two_total"],
    }
    for name, expected in expected_attached.items():
        if checks["attached_weights"].get(name, 0) != expected:
            failures.append({"name": f"attached_{name}", "observed": checks["attached_weights"].get(name, 0), "expected": expected})
    if other_scope == "all_eligible_pairs" and checks["overlap_t1_weighted"] != checks["one_total"]:
        failures.append({"name": "template1_overlap_occurrence", "observed": checks["overlap_t1_weighted"], "expected": checks["one_total"]})
    if other_scope == "all_eligible_pairs" and checks["overlap_23_weighted"] != checks["two_total"]:
        failures.append({"name": "positions_2_3_overlap_occurrence", "observed": checks["overlap_23_weighted"], "expected": checks["two_total"]})
    combinations = []
    combinations.extend(_crosstab_rows(support_d, "two_intermediate_path", "positions_1_2", "support_category", "distinct_count"))
    combinations.extend(_crosstab_rows(support_w, "two_intermediate_path", "positions_1_2", "support_category", "path_occurrence_count"))
    combinations.extend(_crosstab_rows(availability_d, "two_intermediate_path", "positions_1_2", "relation_availability", "distinct_count"))
    combinations.extend(_crosstab_rows(availability_w, "two_intermediate_path", "positions_1_2", "relation_availability", "path_occurrence_count"))
    combinations.extend(_crosstab_rows(article_d, "two_intermediate_path", "positions_1_2", "article_count_band", "distinct_count"))
    combinations.extend(_crosstab_rows(article_w, "two_intermediate_path", "positions_1_2", "article_count_band", "path_occurrence_count"))
    combinations.extend(
        _crosstab_rows(
            Counter({(key, "recorded_attribute_sets"): value for key, value in agreement_d.items()}),
            "two_intermediate_path",
            "positions_1_2",
            "core_tuple_agreement",
            "distinct_count",
        )
    )
    combinations.extend(
        _crosstab_rows(
            Counter({(key, "recorded_attribute_sets"): value for key, value in agreement_w.items()}),
            "two_intermediate_path",
            "positions_1_2",
            "core_tuple_agreement",
            "path_occurrence_count",
        )
    )
    for table_name, template, positions, kind in (
        ("t1_support", "one_intermediate_path", "positions_1_2", "support_category"),
        ("t1_availability", "one_intermediate_path", "positions_1_2", "relation_availability"),
        ("t1_article", "one_intermediate_path", "positions_1_2", "article_count_band"),
        ("t23_support", "two_intermediate_path", "positions_2_3", "support_category"),
        ("t23_availability", "two_intermediate_path", "positions_2_3", "relation_availability"),
        ("t23_article", "two_intermediate_path", "positions_2_3", "article_count_band"),
    ):
        distinct, weighted = other_tables[table_name]
        combinations.extend(_crosstab_rows(distinct, template, positions, kind, "distinct_count"))
        combinations.extend(_crosstab_rows(weighted, template, positions, kind, "path_occurrence_count"))
    overlap_rows = []
    overlap_rows.extend(_overlap_rows(overlap_12, "two_intermediate_path", "positions_1_2", overlap_scope))
    if other_scope == "all_eligible_pairs":
        overlap_rows.extend(_overlap_rows(overlap_t1, "one_intermediate_path", "positions_1_2", other_scope))
        overlap_rows.extend(_overlap_rows(overlap_23, "two_intermediate_path", "positions_2_3", other_scope))
    summary = {
        "checks": checks,
        "shared": {key: int(value) for key, value in shared_summary.items()},
        "overlap_scope": overlap_scope if other_scope == "all_eligible_pairs" else f"{overlap_scope}; {other_scope}",
        "largest_weight_shifts": _largest_shifts(position_counts),
        "attribute_agreement": _agreement_text(agreement_d, agreement_w),
        "support_brief": _counter_brief(support_d, support_w),
        "template1_support_brief": _counter_brief(*other_tables["t1_support"]),
        "positions_2_3_support_brief": _counter_brief(*other_tables["t23_support"]),
        "overlap_brief": _overlap_brief(
            {
                "two_intermediate positions 1-2": (overlap_12, overlap_scope),
                "one_intermediate positions 1-2": (overlap_t1, other_scope),
                "two_intermediate positions 2-3": (overlap_23, other_scope),
            }
        ),
        "failures": failures,
    }
    write_outputs(run_dir, profiles, position_counts, combinations, overlap_rows, shared_rows, detail_rows, detail_summary, summary)
    code_hash = hashlib.sha256()
    for path in sorted(Path("/home/b5ac/freddieyu.b5ac/project_2/src").glob("*.py")):
        code_hash.update(path.read_bytes())
    input_files = {
        "relations.parquet": graph_dir / "relations.parquet",
        "evidence_records.parquet": graph_dir / "evidence_records.parquet",
        "relation_evidence.parquet": graph_dir / "relation_evidence.parquet",
        "candidate_genes.parquet": analysis_dir / "candidate_genes.parquet",
        "cancer_queries.parquet": analysis_dir / "cancer_queries.parquet",
    }
    validation = {
        "status": "failed" if failures else "passed",
        "checks": checks,
        "failures": failures,
        "overlap_scope": summary["overlap_scope"],
        "stages": timer.stages,
        "peak_rss_bytes": peak_rss_bytes(),
    }
    write_json(run_dir / "validation.json", validation)
    write_json(
        run_dir / "manifest.json",
        {
            "task": "path_evidence_audit",
            "status": "failed" if failures else "completed",
            "input_run": str(args.input_run),
            "definitions": {
                "one_intermediate_path": "candidate gene — intermediate gene — query cancer; two edges",
                "two_intermediate_path": "candidate gene — intermediate cancer — intermediate gene — query cancer; three edges",
                "distinct_count": "each qualifying edge or ordered adjacent edge pair once",
                "path_occurrence_count": "eligible complete paths containing the unit at the stated positions",
                "shared_pair": "gene-cancer pair in the fixed candidate and query universe with CORE records and affirmative allowed SemMedDB predications",
            },
            "input_sha256": {name: _sha256(path) for name, path in input_files.items()},
            "code_tree_sha256": code_hash.hexdigest(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "workers": args.workers,
            "cancer_limit": args.cancer_limit,
            "stages": timer.stages,
        },
    )
    print(f"status={'failed' if failures else 'completed'} failures={len(failures)} run_dir={run_dir}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

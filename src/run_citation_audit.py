"""Descriptive citation audit for adjacent edges in the joint graph.

The run reuses the completed path-evidence graph. It does not refit eligibility,
score candidates, or read test labels.
"""

from __future__ import annotations

import argparse
import csv
import io
import multiprocessing as mp
import os
import subprocess
import sys
import time
import zipfile
from array import array
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.citation_features import (
    classify_prepared_links,
    direct_citation_roles,
    intersection_size,
    pair_features,
    traversal_alignment,
    witness_matrices,
)
from src.common import (
    REPO_ROOT,
    StageTimer,
    ensure_dir,
    load_yaml,
    md5_file,
    normalise_pmid,
    sha256_file,
    write_json,
)
from src.path_evidence import EXPECTED_ONE_INTERMEDIATE_PATHS, EXPECTED_TWO_INTERMEDIATE_PATHS
from src.run_path_evidence import attach_weights, derive_profiles, load_connections

GRAPH_RUN = Path("/home/b5ac/freddieyu.b5ac/projects/project_2/results/joint_graph_and_evidence_audit_20261116T0404Z")
PATH_AUDIT = Path("/home/b5ac/freddieyu.b5ac/projects/project_2/results/path_evidence_audit_20261116T1310Z")
COMPLETION_AUDIT = Path("/home/b5ac/freddieyu.b5ac/projects/project_2/results/path_evidence_audit_completion_20261116T1402Z")
MAX_PMID = 50_000_000


def _raise_csv_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10
ARTICLE_EDGE_LIMIT = 200
ARTICLE_KEY_LIMIT = 2_000_000
EXPECTED_ADJACENT = {
    "gene_t1": {
        "pairs": 18414593,
        "pairs_w": 18414593,
        "all_neither": 0,
        "all_neither_w": 0,
        "all_one_sided": 977260,
        "all_one_sided_w": 977260,
        "all_both": 17437333,
        "all_both_w": 17437333,
        "all_overlap_pairs": 42110,
        "all_overlap_pairs_w": 42110,
    },
    "cancer_12": {
        "pairs": 417408433,
        "pairs_w": 14437428485,
        "all_neither": 5403491,
        "all_neither_w": 193381457,
        "all_one_sided": 36490701,
        "all_one_sided_w": 1267778856,
        "all_both": 375514241,
        "all_both_w": 12976268172,
        "all_overlap_pairs": 204165,
        "all_overlap_pairs_w": 17187816,
    },
    "gene_23": {
        "pairs": 13945963,
        "pairs_w": 14437428485,
        "all_neither": 182958,
        "all_neither_w": 160298831,
        "all_one_sided": 1584873,
        "all_one_sided_w": 1587112059,
        "all_both": 12178132,
        "all_both_w": 12690017595,
        "all_overlap_pairs": 304413,
        "all_overlap_pairs_w": 702749188,
    },
}
MEASURES = (
    "pairs",
    "neither",
    "one_sided",
    "both",
    "shared_q_pairs",
    "shared_q_sum",
    "all_neither",
    "all_one_sided",
    "all_both",
    "all_overlap_pairs",
    "all_overlap_sum",
    "ltr_pairs",
    "ltr_links",
    "rtl_pairs",
    "rtl_links",
    "ambiguous_pairs",
    "ambiguous_links",
    "collapsed_pairs",
    "collapsed_links",
    "ltr_index_complete",
    "ltr_index_partial",
    "ltr_index_absent",
    "rtl_index_complete",
    "rtl_index_partial",
    "rtl_index_absent",
    "ltr_no_motif_complete",
    "ltr_no_motif_partial",
    "rtl_no_motif_complete",
    "rtl_no_motif_partial",
    "ltr_gap",
    "rtl_gap",
    "collapsed_gap",
    "left_pmids",
    "left_out",
    "left_in",
    "left_meta",
    "right_pmids",
    "right_out",
    "right_in",
    "right_meta",
    "cocite_pairs",
    "cocite_witnesses",
    "cocite_external",
    "cocite_articles",
    "cocite_index_complete",
    "cocite_index_partial",
    "cocite_index_absent",
    "cocite_no_motif_complete",
    "cocite_no_motif_partial",
    "cocite_gap",
    "cocite_article_exact_pairs",
    "cocite_article_omitted_pairs",
    "ref_pairs",
    "ref_witnesses",
    "ref_external",
    "ref_articles",
    "ref_index_complete",
    "ref_index_partial",
    "ref_index_absent",
    "ref_no_motif_complete",
    "ref_no_motif_partial",
    "ref_gap",
    "ref_article_exact_pairs",
    "ref_article_omitted_pairs",
    "witness_skipped_pairs",
)
INDEX = {name: 2 * position for position, name in enumerate(MEASURES)}
N_COUNTERS = 2 * len(MEASURES)
GC: list = []
GG: list = []
OUTGOING: dict = {}
INCOMING: dict = {}
INSIDE = None
TEMPLATE_NAMES = {
    "gene_t1": ("one_intermediate_path", "positions_1_2"),
    "cancer_12": ("two_intermediate_path", "positions_1_2"),
    "gene_23": ("two_intermediate_path", "positions_2_3"),
}


class Edge:
    __slots__ = (
        "q",
        "qset",
        "links",
        "all_interned",
        "n_out",
        "n_in",
        "n_meta",
        "gene",
        "cancer",
        "extend",
        "n_left",
        "candidate",
        "query_edge",
        "family",
        "source",
    )

    def __init__(self) -> None:
        self.q: tuple[int, ...] = ()
        self.qset: frozenset[int] = frozenset()
        self.links: tuple[tuple[int, int], ...] = ()
        self.all_interned: tuple[int, ...] = ()
        self.n_out = 0
        self.n_in = 0
        self.n_meta = 0
        self.gene = ""
        self.cancer = ""
        self.extend = 0
        self.n_left = 0
        self.candidate = 0
        self.query_edge = 0
        self.family = ""
        self.source = -1


def bump(values: list[int], name: str, weight: int, amount: int = 1) -> None:
    slot = INDEX[name]
    values[slot] += amount
    values[slot + 1] += amount * weight


def value(values: list[int], name: str, weighted: bool = False) -> int:
    return values[INDEX[name] + (1 if weighted else 0)]


def fresh() -> list[int]:
    return [0] * N_COUNTERS


def add_into(total: list[int], part: list[int]) -> None:
    for position, amount in enumerate(part):
        total[position] += amount


def _index_state(indexed: int, total: int) -> str:
    if indexed <= 0:
        return "absent"
    if indexed >= total:
        return "complete"
    return "partial"


def _both_index(left_indexed: int, left_total: int, right_indexed: int, right_total: int) -> str:
    if left_indexed <= 0 or right_indexed <= 0:
        return "absent"
    if left_indexed >= left_total and right_indexed >= right_total:
        return "complete"
    return "partial"


def consume(values: list[int], left: Edge, right: Edge, weight: int, cocite: dict | None, shared: dict | None, left_index: int, right_index: int) -> None:
    bump(values, "pairs", weight)
    if not left.q and not right.q:
        bump(values, "neither", weight)
    elif not left.q or not right.q:
        bump(values, "one_sided", weight)
    else:
        bump(values, "both", weight)
        shared_size = intersection_size(left.q, right.q)
        if shared_size:
            bump(values, "shared_q_pairs", weight)
            bump(values, "shared_q_sum", weight, shared_size)
        left_n = len(left.q)
        right_n = len(right.q)
        if left.links or right.links:
            exclusive_ltr, exclusive_rtl, ambiguous = classify_prepared_links(left.qset, right.qset, left.links, right.links)
        else:
            exclusive_ltr = exclusive_rtl = ambiguous = 0
        bump(values, "left_pmids", weight, left_n)
        bump(values, "left_out", weight, left.n_out)
        bump(values, "left_in", weight, left.n_in)
        bump(values, "left_meta", weight, left.n_meta)
        bump(values, "right_pmids", weight, right_n)
        bump(values, "right_out", weight, right.n_out)
        bump(values, "right_in", weight, right.n_in)
        bump(values, "right_meta", weight, right.n_meta)
        collapsed = exclusive_ltr + exclusive_rtl + ambiguous
        if exclusive_ltr:
            bump(values, "ltr_pairs", weight)
            bump(values, "ltr_links", weight, exclusive_ltr)
        if exclusive_rtl:
            bump(values, "rtl_pairs", weight)
            bump(values, "rtl_links", weight, exclusive_rtl)
        if ambiguous:
            bump(values, "ambiguous_pairs", weight)
            bump(values, "ambiguous_links", weight, ambiguous)
        if collapsed:
            bump(values, "collapsed_pairs", weight)
            bump(values, "collapsed_links", weight, collapsed)
        ltr_state = _index_state(left.n_out, left_n)
        rtl_state = _index_state(right.n_out, right_n)
        bump(values, f"ltr_index_{ltr_state}", weight)
        bump(values, f"rtl_index_{rtl_state}", weight)
        if exclusive_ltr == 0 and ltr_state != "absent":
            bump(values, f"ltr_no_motif_{ltr_state}", weight)
        if exclusive_rtl == 0 and rtl_state != "absent":
            bump(values, f"rtl_no_motif_{rtl_state}", weight)
        metadata_gap = left.n_meta < left_n or right.n_meta < right_n
        if exclusive_ltr and metadata_gap:
            bump(values, "ltr_gap", weight)
        if exclusive_rtl and metadata_gap:
            bump(values, "rtl_gap", weight)
        if collapsed and metadata_gap:
            bump(values, "collapsed_gap", weight)
        cocite_state = _both_index(left.n_in, left_n, right.n_in, right_n)
        ref_state = _both_index(left.n_out, left_n, right.n_out, right_n)
        bump(values, f"cocite_index_{cocite_state}", weight)
        bump(values, f"ref_index_{ref_state}", weight)
        if cocite is None or shared is None:
            bump(values, "witness_skipped_pairs", weight)
        else:
            witnesses = int(cocite["witnesses"][left_index, right_index])
            if witnesses:
                bump(values, "cocite_pairs", weight)
                bump(values, "cocite_witnesses", weight, witnesses)
                bump(values, "cocite_external", weight, int(cocite["external"][left_index, right_index]))
                if metadata_gap:
                    bump(values, "cocite_gap", weight)
            elif cocite_state != "absent":
                bump(values, f"cocite_no_motif_{cocite_state}", weight)
            if cocite["article_status"] == "exact":
                bump(values, "cocite_article_exact_pairs", weight)
                article_pairs = 0 if cocite["article_counts"] is None else cocite["article_counts"].get((left_index, right_index), 0)
                if article_pairs:
                    bump(values, "cocite_articles", weight, article_pairs)
            else:
                bump(values, "cocite_article_omitted_pairs", weight)
            ref_witnesses = int(shared["witnesses"][left_index, right_index])
            if ref_witnesses:
                bump(values, "ref_pairs", weight)
                bump(values, "ref_witnesses", weight, ref_witnesses)
                bump(values, "ref_external", weight, int(shared["external"][left_index, right_index]))
                if metadata_gap:
                    bump(values, "ref_gap", weight)
            elif ref_state != "absent":
                bump(values, f"ref_no_motif_{ref_state}", weight)
            if shared["article_status"] == "exact":
                bump(values, "ref_article_exact_pairs", weight)
                article_pairs = 0 if shared["article_counts"] is None else shared["article_counts"].get((left_index, right_index), 0)
                if article_pairs:
                    bump(values, "ref_articles", weight, article_pairs)
            else:
                bump(values, "ref_article_omitted_pairs", weight)
    left_all = left.all_interned
    right_all = right.all_interned
    if not left_all and not right_all:
        bump(values, "all_neither", weight)
    elif not left_all or not right_all:
        bump(values, "all_one_sided", weight)
    else:
        bump(values, "all_both", weight)
        overlap = intersection_size(left_all, right_all)
        if overlap:
            bump(values, "all_overlap_pairs", weight)
            bump(values, "all_overlap_sum", weight, overlap)


def process_spec(spec: tuple) -> tuple[list[int], dict]:
    started = time.time()
    kind = spec[0]
    values = fresh()
    has_block = False
    if kind == "gene_t1":
        _, group_id, left_ids, right_ids, do_witness = spec
        edges = [GG[index] for index in left_ids] + [GC[index] for index in right_ids]
        split = len(left_ids)
        has_block = split > 0 and split < len(edges)
        pair_iter = ((left, right, 1) for left in range(split) for right in range(split, len(edges)))
    elif kind == "cancer_12":
        _, group_id, indices, do_witness = spec
        edges = [GC[index] for index in indices]
        candidates = [index for index, edge in enumerate(edges) if edge.candidate]
        extenders = [index for index, edge in enumerate(edges) if edge.extend]
        has_block = bool(candidates) and bool(extenders)
        pair_iter = (
            (left, right, edges[right].extend)
            for left in candidates
            for right in extenders
            if edges[left].gene != edges[right].gene
        )
    else:
        _, group_id, indices, do_witness = spec
        edges = [GC[index] for index in indices]
        lefts = [index for index, edge in enumerate(edges) if edge.n_left]
        rights = [index for index, edge in enumerate(edges) if edge.query_edge]
        has_block = bool(lefts) and bool(rights)
        pair_iter = (
            (left, right, edges[left].n_left)
            for left in lefts
            for right in rights
            if edges[left].cancer != edges[right].cancer
        )
    cocite = shared = None
    if do_witness and has_block and len(edges) >= 2:
        collect = len(edges) <= ARTICLE_EDGE_LIMIT
        pmid_lists = [edge.q for edge in edges]
        cocite = witness_matrices(pmid_lists, INCOMING, INSIDE, collect, ARTICLE_KEY_LIMIT)
        shared = witness_matrices(pmid_lists, OUTGOING, INSIDE, collect, ARTICLE_KEY_LIMIT)
    for left_index, right_index, weight in pair_iter:
        consume(values, edges[left_index], edges[right_index], weight, cocite, shared, left_index, right_index)
    meta = {
        "kind": kind,
        "group": group_id,
        "edges": len(edges),
        "seconds": round(time.time() - started, 3),
        "witness": bool(do_witness),
        "cocite_article_status": "" if cocite is None else cocite["article_status"],
        "ref_article_status": "" if shared is None else shared["article_status"],
        "pairs": value(values, "pairs"),
        "weighted": value(values, "pairs", True),
    }
    return values, meta


def _check(failures: list, name: str, observed: int, expected: int) -> None:
    if observed != expected:
        failures.append({"name": name, "observed": int(observed), "expected": int(expected)})


def _batch_writer(path: Path, schema: pa.Schema):
    path.parent.mkdir(parents=True, exist_ok=True)
    return pq.ParquetWriter(path, schema, compression="zstd")


def write_orientations(connections: dict, analysis: Path) -> dict:
    directed_schema = pa.schema(
        [
            ("family", pa.string()),
            ("endpoint_a", pa.string()),
            ("endpoint_b", pa.string()),
            ("connection_path_eligible", pa.bool_()),
            ("resource", pa.string()),
            ("direction_role", pa.string()),
            ("predicate", pa.string()),
            ("negation", pa.string()),
            ("subject_id", pa.string()),
            ("object_id", pa.string()),
            ("record_path_eligible", pa.bool_()),
            ("alignment_a_to_b", pa.string()),
            ("alignment_b_to_a", pa.string()),
            ("cge_label", pa.string()),
            ("ccs_label", pa.string()),
            ("gci_label", pa.string()),
            ("gcc_label", pa.string()),
            ("gcs_has_type", pa.string()),
        ]
    )
    collapsed_schema = pa.schema(
        [
            ("family", pa.string()),
            ("endpoint_a", pa.string()),
            ("endpoint_b", pa.string()),
            ("connection_path_eligible", pa.bool_()),
            ("traversal_predicates", pa.string()),
            ("negated_predicates", pa.string()),
            ("other_predicates", pa.string()),
            ("n_core_records", pa.int64()),
            ("n_traversal_records", pa.int64()),
            ("n_negated_records", pa.int64()),
            ("n_other_records", pa.int64()),
        ]
    )
    membership_schema = pa.schema(
        [
            ("family", pa.string()),
            ("endpoint_a", pa.string()),
            ("endpoint_b", pa.string()),
            ("pmid", pa.int64()),
            ("in_core", pa.bool_()),
            ("in_semmed_traversal", pa.bool_()),
            ("in_negated_supplement", pa.bool_()),
            ("in_other_semmed", pa.bool_()),
            ("qualifying", pa.bool_()),
        ]
    )
    directed = _batch_writer(analysis / "assertion_orientations.parquet", directed_schema)
    collapsed = _batch_writer(analysis / "connectivity_predicates.parquet", collapsed_schema)
    membership = _batch_writer(analysis / "edge_pmid_membership.parquet", membership_schema)
    directed_rows: list[dict] = []
    collapsed_rows: list[dict] = []
    membership_rows: list[dict] = []
    raw: dict[tuple, dict] = {}
    qualifying: set[int] = set()
    full_graph: set[int] = set()
    eligible_keys = set()

    def flush(writer, rows: list[dict]) -> None:
        if not rows:
            return
        writer.write_table(pa.Table.from_pylist(rows))
        rows.clear()

    for key, bucket in connections.items():
        family, endpoint_a, endpoint_b = key
        id_a = f"gene:{endpoint_a}"
        id_b = f"cancer:{endpoint_b}" if family == "gene_cancer" else f"gene:{endpoint_b}"
        core = set()
        traversal = set()
        negated = set()
        other = set()
        traversal_predicates = set()
        negated_predicates = set()
        other_predicates = set()
        n_core = n_traversal = n_negated = n_other = 0
        for pmid, cge, ccs, gci, gcc, has_type in bucket["core"].values():
            n_core += 1
            number = int(pmid) if pmid else 0
            if number:
                core.add(number)
                full_graph.add(number)
            directed_rows.append(
                {
                    "family": family,
                    "endpoint_a": endpoint_a,
                    "endpoint_b": endpoint_b,
                    "connection_path_eligible": False,
                    "resource": "core",
                    "direction_role": "sentence_attribute",
                    "predicate": "gene_cancer_association",
                    "negation": "false",
                    "subject_id": id_a if family == "gene_cancer" else "",
                    "object_id": id_b if family == "gene_cancer" else "",
                    "record_path_eligible": False,
                    "alignment_a_to_b": "not_applicable",
                    "alignment_b_to_a": "not_applicable",
                    "cge_label": cge or "",
                    "ccs_label": ccs or "",
                    "gci_label": gci or "",
                    "gcc_label": gcc or "",
                    "gcs_has_type": has_type or "",
                }
            )
        for pmid, predicate, subject, object_, is_negated, is_traversal in bucket["sem"].values():
            number = int(pmid) if pmid else 0
            if number:
                full_graph.add(number)
            if is_negated:
                n_negated += 1
                negated_predicates.add(predicate)
                if number:
                    negated.add(number)
            elif is_traversal:
                n_traversal += 1
                traversal_predicates.add(predicate)
                if number:
                    traversal.add(number)
            else:
                n_other += 1
                other_predicates.add(predicate)
                if number:
                    other.add(number)
            directed_rows.append(
                {
                    "family": family,
                    "endpoint_a": endpoint_a,
                    "endpoint_b": endpoint_b,
                    "connection_path_eligible": False,
                    "resource": "semmed",
                    "direction_role": "assertion",
                    "predicate": predicate,
                    "negation": "true" if is_negated else "false",
                    "subject_id": subject,
                    "object_id": object_,
                    "record_path_eligible": bool(is_traversal),
                    "alignment_a_to_b": traversal_alignment(subject, object_, id_a, id_b),
                    "alignment_b_to_a": traversal_alignment(subject, object_, id_b, id_a),
                    "cge_label": "",
                    "ccs_label": "",
                    "gci_label": "",
                    "gcc_label": "",
                    "gcs_has_type": "",
                }
            )
        eligible = (family == "gene_cancer" and (n_core > 0 or n_traversal > 0)) or (family == "gene_gene" and n_traversal > 0)
        record_count = n_core + n_traversal + n_negated + n_other
        if eligible:
            eligible_keys.add(key)
            qualifying.update(core)
            qualifying.update(traversal)
        if record_count:
            for row in directed_rows[-record_count:]:
                row["connection_path_eligible"] = eligible
        collapsed_rows.append(
            {
                "family": family,
                "endpoint_a": endpoint_a,
                "endpoint_b": endpoint_b,
                "connection_path_eligible": eligible,
                "traversal_predicates": "|".join(sorted(traversal_predicates)),
                "negated_predicates": "|".join(sorted(negated_predicates)),
                "other_predicates": "|".join(sorted(other_predicates)),
                "n_core_records": n_core,
                "n_traversal_records": n_traversal,
                "n_negated_records": n_negated,
                "n_other_records": n_other,
            }
        )
        raw[key] = {"core": core, "traversal": traversal, "negated": negated, "other": other}
        for pmid in core | traversal | negated | other:
            membership_rows.append(
                {
                    "family": family,
                    "endpoint_a": endpoint_a,
                    "endpoint_b": endpoint_b,
                    "pmid": pmid,
                    "in_core": pmid in core,
                    "in_semmed_traversal": pmid in traversal,
                    "in_negated_supplement": pmid in negated,
                    "in_other_semmed": pmid in other,
                    "qualifying": bool(eligible and (pmid in core or pmid in traversal)),
                }
            )
        if len(directed_rows) >= 100000:
            flush(directed, directed_rows)
        if len(collapsed_rows) >= 100000:
            flush(collapsed, collapsed_rows)
        if len(membership_rows) >= 100000:
            flush(membership, membership_rows)
    flush(directed, directed_rows)
    flush(collapsed, collapsed_rows)
    flush(membership, membership_rows)
    directed.close()
    collapsed.close()
    membership.close()
    return {
        "raw": raw,
        "qualifying": qualifying,
        "full_graph": full_graph,
        "eligible_keys": eligible_keys,
    }


def build_edges(profiles: list[dict], weights: dict, raw: dict, candidates: set[str], queries: set[str]) -> None:
    GC.clear()
    GG.clear()
    for profile in profiles:
        if profile["support"] == "no_traversal_support":
            continue
        key = (profile["family"], profile["endpoint_a"], profile["endpoint_b"])
        sets = raw.get(key, {"core": set(), "traversal": set()})
        edge = Edge()
        edge.family = profile["family"]
        edge.q = tuple(sorted(sets["core"] | sets["traversal"]))
        edge.qset = frozenset(edge.q)
        edge.all_interned = profile["all_pmids"]
        edge.gene = profile["endpoint_a"]
        edge.cancer = profile["endpoint_b"]
        if profile["family"] == "gene_cancer":
            edge.extend = int(weights["right_of"].get((edge.gene, edge.cancer), 0))
            edge.n_left = int(weights["left_of"].get((edge.gene, edge.cancer), 0))
            edge.candidate = int(edge.gene in candidates)
            edge.query_edge = int(edge.cancer in queries)
            GC.append(edge)
        elif profile["support"] == "semmed_only":
            GG.append(edge)


def known_link_check(citation_zip: Path, metadata_zip: Path) -> dict:
    _raise_csv_limit()
    with zipfile.ZipFile(metadata_zip) as handle:
        text = io.TextIOWrapper(handle.open("icite_metadata.csv"), encoding="utf-8", newline="")
        row = next(csv.DictReader(text))
    references = [int(token) for token in row["references"].split() if token.isdigit()]
    with zipfile.ZipFile(citation_zip) as handle:
        raw = handle.open("open_citation_collection.csv")
        header = raw.readline().decode("utf-8").strip().lstrip("\ufeff")
        found = set()
        checked = 0
        while len(found) < len(references) and checked < 100000:
            line = raw.readline()
            if not line:
                break
            checked += 1
            citing, _, cited = line.decode("utf-8").partition(",")
            if citing.strip() != row["pmid"]:
                if found:
                    break
                continue
            try:
                found.add(int(cited.strip()))
            except ValueError:
                continue
    return {
        "header": header,
        "pmid": int(row["pmid"]),
        "citation_count": int(row["citation_count"] or 0),
        "n_references": len(references),
        "n_cited_by": len(row["cited_by"].split()),
        "references_found_as_citing_to_referenced": len(set(references) & found),
        "header_is_citing_referenced": header == "citing,referenced",
    }


def verify_archives(citation_zip: Path, metadata_zip: Path, citation_meta: dict, metadata_meta: dict) -> dict:
    records = {}
    for path, meta in ((citation_zip, citation_meta), (metadata_zip, metadata_meta)):
        size = path.stat().st_size
        md5_path = Path(str(path) + ".md5")
        sha_path = Path(str(path) + ".sha256")
        integrity_path = Path(str(path) + ".integrity")
        if md5_path.exists() and sha_path.exists() and integrity_path.exists() and size == meta["bytes"]:
            md5 = md5_path.read_text(encoding="utf-8").strip().split()[0]
            sha = sha_path.read_text(encoding="utf-8").strip().split()[0]
            integrity = integrity_path.read_text(encoding="utf-8").strip()
        else:
            md5 = md5_file(path)
            sha = sha256_file(path)
            with zipfile.ZipFile(path) as handle:
                bad = handle.testzip()
                member = handle.infolist()[0]
            integrity = "ok" if bad is None else f"bad:{bad}"
            md5_path.write_text(md5 + "\n", encoding="utf-8")
            sha_path.write_text(sha + "\n", encoding="utf-8")
            integrity_path.write_text(
                f"{integrity}\nmember={member.filename}\nuncompressed={member.file_size}\ncompress_type={member.compress_type}\n",
                encoding="utf-8",
            )
        records[path.name] = {
            "bytes": size,
            "expected_bytes": meta["bytes"],
            "upstream_md5": meta["upstream_md5"],
            "md5": md5,
            "sha256": sha,
            "integrity": integrity if isinstance(integrity, str) and not integrity.startswith("member=") else integrity_path.read_text(encoding="utf-8").strip(),
            "size_matches": size == meta["bytes"],
            "md5_matches": md5 == meta["upstream_md5"],
        }
    return records


def stream_metadata(path: Path, support_flag: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    _raise_csv_limit()
    year = np.zeros(MAX_PMID, dtype=np.uint16)
    present = np.zeros(MAX_PMID, dtype=np.uint8)
    citation_count: dict[int, int] = {}
    overflow = 0
    rows = 0
    with zipfile.ZipFile(path) as handle:
        text = io.TextIOWrapper(handle.open("icite_metadata.csv"), encoding="utf-8", newline="")
        reader = csv.reader(text)
        header = next(reader)
        columns = {name.strip('"'): index for index, name in enumerate(header)}
        pmid_i = columns["pmid"]
        year_i = columns["year"]
        count_i = columns["citation_count"]
        for row in reader:
            rows += 1
            if rows % 2000000 == 0:
                print(f"[metadata] rows={rows}", flush=True)
            try:
                pmid = int(row[pmid_i])
            except (ValueError, IndexError):
                continue
            if pmid <= 0 or pmid >= MAX_PMID:
                overflow += 1
                continue
            present[pmid] = 1
            year_text = row[year_i]
            if year_text:
                try:
                    year_value = int(float(year_text))
                except ValueError:
                    year_value = 0
                if 0 < year_value <= 65535:
                    year[pmid] = year_value
            if pmid < len(support_flag) and support_flag[pmid]:
                count_text = row[count_i]
                citation_count[pmid] = int(count_text) if count_text else -1
    print(f"[metadata] rows={rows} overflow={overflow} support_counts={len(citation_count)}", flush=True)
    return year, present, citation_count


def stream_citations(path: Path, support_flag: np.ndarray, role: np.ndarray) -> dict:
    first_to_second: dict[int, array] = {}
    second_to_first: dict[int, array] = {}
    rows = invalid = self_links = kept = 0
    high = 0
    with zipfile.ZipFile(path) as handle:
        raw = handle.open("open_citation_collection.csv")
        header = raw.readline()
        for line in raw:
            rows += 1
            if rows % 20000000 == 0:
                print(f"[citations] rows={rows} kept={kept}", flush=True)
            comma = line.find(b",")
            if comma <= 0:
                invalid += 1
                continue
            try:
                left = int(line[:comma])
                right = int(line[comma + 1 :])
            except ValueError:
                invalid += 1
                continue
            if left <= 0 or right <= 0 or left > 0xFFFFFFFF or right > 0xFFFFFFFF:
                invalid += 1
                continue
            if left == right:
                self_links += 1
                continue
            left_support = left < len(support_flag) and bool(support_flag[left])
            right_support = right < len(support_flag) and bool(support_flag[right])
            if not left_support and not right_support:
                continue
            kept += 1
            if left_support:
                bucket = first_to_second.get(left)
                if bucket is None:
                    bucket = array("I")
                    first_to_second[left] = bucket
                bucket.append(right)
            if right_support:
                bucket = second_to_first.get(right)
                if bucket is None:
                    bucket = array("I")
                    second_to_first[right] = bucket
                bucket.append(left)
            if left < len(role):
                role[left] |= 4
            else:
                high += 1
            if right < len(role):
                role[right] |= 8
            else:
                high += 1
    print(f"[citations] rows={rows} kept={kept} invalid={invalid} self={self_links}", flush=True)
    return {
        "header": header.decode("utf-8").strip(),
        "rows": rows,
        "invalid_rows": invalid,
        "self_link_rows": self_links,
        "kept_rows": kept,
        "high_pmid_endpoints": high,
        "first_to_second": first_to_second,
        "second_to_first": second_to_first,
    }


def freeze_index(raw: dict[int, array]) -> tuple[dict[int, np.ndarray], int, int]:
    frozen: dict[int, np.ndarray] = {}
    appended = unique = 0
    for pmid in list(raw):
        bucket = raw.pop(pmid)
        appended += len(bucket)
        if not bucket:
            continue
        values = np.frombuffer(bucket, dtype=np.uint32).copy()
        values.sort()
        if values.size == 1:
            frozen[int(pmid)] = values
            unique += 1
            continue
        keep = np.empty(values.size, dtype=bool)
        keep[0] = True
        keep[1:] = values[1:] != values[:-1]
        compact = values[keep]
        frozen[int(pmid)] = compact
        unique += int(compact.size)
    return frozen, appended, unique


def orientation_decision(known: dict, outgoing: dict[int, np.ndarray], incoming: dict[int, np.ndarray], citation_count: dict[int, int]) -> dict:
    def ratio(index: dict[int, np.ndarray]) -> dict:
        sample = []
        exact = compared = 0
        for pmid, count in citation_count.items():
            if count <= 0:
                continue
            compared += 1
            degree = 0 if pmid not in index else int(index[pmid].size)
            if degree == count:
                exact += 1
            sample.append(degree / count)
        if not sample:
            return {"compared": 0, "exact": 0, "median_ratio": None}
        ordered = np.array(sample)
        return {
            "compared": compared,
            "exact": exact,
            "median_ratio": float(np.median(ordered)),
            "p10_ratio": float(np.quantile(ordered, 0.1)),
            "p90_ratio": float(np.quantile(ordered, 0.9)),
        }

    incoming_ratio = ratio(incoming)
    outgoing_ratio = ratio(outgoing)
    known_ok = bool(known["header_is_citing_referenced"]) and known["references_found_as_citing_to_referenced"] == known["n_references"]
    incoming_median = incoming_ratio.get("median_ratio")
    outgoing_median = outgoing_ratio.get("median_ratio")
    contradicted = False
    if incoming_median is not None and outgoing_median is not None:
        incoming_distance = abs(incoming_median - 1)
        outgoing_distance = abs(outgoing_median - 1)
        contradicted = outgoing_distance + 0.25 < incoming_distance and incoming_distance > 0.5
    return {
        "known_link": known,
        "incoming_versus_citation_count": incoming_ratio,
        "outgoing_versus_citation_count": outgoing_ratio,
        "confirmed": known_ok and not contradicted,
        "citing_column": "citing",
        "cited_column": "referenced",
    }


def unique_directed_pairs(outgoing: dict[int, np.ndarray], incoming: dict[int, np.ndarray], support_flag: np.ndarray) -> int:
    total = sum(int(values.size) for values in outgoing.values())
    for cited, citers in incoming.items():
        for citer in citers:
            citer = int(citer)
            if citer >= len(support_flag) or not support_flag[citer]:
                total += 1
            elif citer not in outgoing:
                total += 1
    return total


def attach_citation_fields(edges: list[Edge], outgoing: dict, support_flag: np.ndarray, present: np.ndarray) -> None:
    limit = len(support_flag)
    meta_limit = len(present)
    for edge in edges:
        links = []
        seen = set()
        n_out = n_in = n_meta = 0
        for pmid in edge.q:
            targets = outgoing.get(pmid)
            if targets is not None and targets.size:
                n_out += 1
                for target in targets:
                    target = int(target)
                    if target == pmid or target >= limit or not support_flag[target]:
                        continue
                    pair = (pmid, target)
                    if pair in seen:
                        continue
                    seen.add(pair)
                    links.append(pair)
            if pmid in INCOMING and INCOMING[pmid].size:
                n_in += 1
            if pmid < meta_limit and present[pmid]:
                n_meta += 1
        edge.links = tuple(links)
        edge.n_out = n_out
        edge.n_in = n_in
        edge.n_meta = n_meta


def write_links(path: Path, outgoing: dict, incoming: dict, support_flag: np.ndarray) -> int:
    schema = pa.schema([("citing_pmid", pa.uint32()), ("cited_pmid", pa.uint32())])
    writer = _batch_writer(path, schema)
    citing: list[int] = []
    cited: list[int] = []
    total = 0

    def flush() -> None:
        nonlocal citing, cited
        if not citing:
            return
        writer.write_table(pa.table({"citing_pmid": pa.array(citing, type=pa.uint32()), "cited_pmid": pa.array(cited, type=pa.uint32())}))
        citing = []
        cited = []

    for source, targets in outgoing.items():
        for target in targets:
            citing.append(int(source))
            cited.append(int(target))
            total += 1
            if len(citing) >= 250000:
                flush()
    for target, sources in incoming.items():
        for source in sources:
            source = int(source)
            if source < len(support_flag) and support_flag[source]:
                continue
            citing.append(source)
            cited.append(int(target))
            total += 1
            if len(citing) >= 250000:
                flush()
    flush()
    writer.close()
    return total


def write_endpoint_metadata(path: Path, role: np.ndarray, year: np.ndarray, present: np.ndarray) -> int:
    schema = pa.schema(
        [
            ("pmid", pa.int64()),
            ("publication_year", pa.int16()),
            ("in_snapshot_metadata", pa.bool_()),
            ("year_missing", pa.bool_()),
            ("qualifying_support", pa.bool_()),
            ("supplementary_graph_pmid", pa.bool_()),
            ("observed_as_citing", pa.bool_()),
            ("observed_as_cited", pa.bool_()),
            ("link_index_scope", pa.string()),
        ]
    )
    writer = _batch_writer(path, schema)
    identifiers = np.nonzero(role)[0]
    total = 0
    step = 250000
    for start in range(0, len(identifiers), step):
        chunk = identifiers[start : start + step]
        years = year[chunk]
        flags = role[chunk]
        present_flags = present[chunk]
        scope = np.where((flags & 1) > 0, "complete_for_release", "support_endpoint_only")
        writer.write_table(
            pa.table(
                {
                    "pmid": chunk.astype(np.int64),
                    "publication_year": years.astype(np.int16),
                    "in_snapshot_metadata": present_flags.astype(bool),
                    "year_missing": (present_flags == 1) & (years == 0),
                    "qualifying_support": (flags & 1) > 0,
                    "supplementary_graph_pmid": (flags & 2) > 0,
                    "observed_as_citing": (flags & 4) > 0,
                    "observed_as_cited": (flags & 8) > 0,
                    "link_index_scope": scope,
                }
            )
        )
        total += len(chunk)
    writer.close()
    return total


def pmid_coverage(edges: list[Edge], qualifying: set[int], present: np.ndarray) -> dict:
    pmids = set()
    occurrences = indexed_out = indexed_in = in_meta = 0
    for edge in edges:
        for pmid in edge.q:
            pmids.add(pmid)
            occurrences += 1
            if edge.n_out and pmid in OUTGOING and OUTGOING[pmid].size:
                indexed_out += 1
            if pmid in INCOMING and INCOMING[pmid].size:
                indexed_in += 1
            if pmid < len(present) and present[pmid]:
                in_meta += 1
    distinct_out = sum(1 for pmid in pmids if pmid in OUTGOING and OUTGOING[pmid].size)
    distinct_in = sum(1 for pmid in pmids if pmid in INCOMING and INCOMING[pmid].size)
    distinct_meta = sum(1 for pmid in pmids if pmid < len(present) and present[pmid])
    return {
        "edges": len(edges),
        "pmid_occurrences": occurrences,
        "occurrence_outgoing": indexed_out,
        "occurrence_incoming": indexed_in,
        "occurrence_metadata": in_meta,
        "distinct_pmids": len(pmids),
        "distinct_outgoing": distinct_out,
        "distinct_incoming": distinct_in,
        "distinct_metadata": distinct_meta,
        "qualifying_pmids": len(qualifying),
    }


def counter_rows(kind: str, values: list[int]) -> list[dict]:
    template, positions = TEMPLATE_NAMES[kind]
    rows = []
    for name in MEASURES:
        rows.append(
            {
                "template": template,
                "positions": positions,
                "measure": name,
                "distinct_count": value(values, name),
                "path_occurrence_count": value(values, name, True),
            }
        )
    return rows


def write_tsv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def feature_dictionary() -> list[dict]:
    return [
        {
            "feature": "shared_article",
            "observation_unit": "adjacent edge pair",
            "counting_rule": "Size of the intersection of the two qualifying PMID sets. A pair is positive when the size is at least one.",
            "coverage_denominator": "Pairs in which both edges have a qualifying PMID. neither and one_sided are separate denominators.",
            "missing_value": "An empty qualifying set is neither or one_sided, not a zero intersection.",
            "recompute_from": "analysis/edge_pmid_membership.parquet qualifying rows",
        },
        {
            "feature": "direct_citation_directional",
            "observation_unit": "directed citing-to-cited link on one adjacent edge pair",
            "counting_rule": "Exclusive left-to-right, exclusive right-to-left, and ambiguous. Ambiguous means both PMIDs belong to both support sets, so the same arrow fills both positional roles and is counted once.",
            "coverage_denominator": "Left-to-right uses outgoing coverage of the left edge. Right-to-left uses outgoing coverage of the right edge. Complete means every qualifying PMID on that side is outgoing-indexed.",
            "missing_value": "Index absent is separate from no motif in a complete or partial index. Metadata absence does not remove a positive archive motif.",
            "recompute_from": "analysis/directed_citation_links.parquet and edge_pmid_membership.parquet",
        },
        {
            "feature": "direct_citation_direction_collapsed",
            "observation_unit": "directed citing-to-cited link on one adjacent edge pair",
            "counting_rule": "Sum of the two exclusive directions and the ambiguous category. Each directed link is counted once.",
            "coverage_denominator": "Same links as the directional feature. The three categories partition this total.",
            "missing_value": "Same coverage states as the directional feature.",
            "recompute_from": "analysis/directed_citation_links.parquet and edge_pmid_membership.parquet",
        },
        {
            "feature": "cocitation",
            "observation_unit": "third-party citing PMID, and separately a cross-edge supporting PMID pair",
            "counting_rule": "A witness cites two distinct papers, one in each qualifying set. A witness attached only to one PMID shared by both edges does not count. Witnesses are distinct within the adjacent pair. Article-pair counts are distinct (left PMID, right PMID) pairs and are reported only for groups that finished exactly.",
            "coverage_denominator": "Incoming indexes on both sides. Complete requires every qualifying PMID on both sides to be incoming-indexed. Absent if either side has none.",
            "missing_value": "No motif in a partial index is not a confirmed absence. external_witnesses counts witnesses outside the full CORE–SemMedDB PMID set.",
            "recompute_from": "analysis/directed_citation_links.parquet incoming lists and edge_pmid_membership.parquet",
        },
        {
            "feature": "shared_reference",
            "observation_unit": "third-party cited PMID, and separately a cross-edge supporting PMID pair",
            "counting_rule": "Two distinct papers, one in each qualifying set, cite the same third PMID. The same distinct-paper rule as co-citation applies. The summary is symmetric; the stored links keep their original direction.",
            "coverage_denominator": "Outgoing indexes on both sides, using the same complete, partial, and absent states.",
            "missing_value": "Same missing-value rule as co-citation. Article-pair totals include only groups with article_status exact.",
            "recompute_from": "analysis/directed_citation_links.parquet outgoing lists and edge_pmid_membership.parquet",
        },
        {
            "feature": "occurrence_weight",
            "observation_unit": "complete path using the adjacent edge pair",
            "counting_rule": "Two-edge pairs have weight 1. Three-edge positions 1–2 use the number of eligible final edges on the right edge. Positions 2–3 use the number of eligible initial edges on the left edge.",
            "coverage_denominator": "The occurrence weights at each three-edge adjacent position sum to the complete three-edge path total.",
            "missing_value": "A pair that fails node-repetition or endpoint rules is not given a weight.",
            "recompute_from": "prepared graph relations and the path-evidence weight rules",
        },
        {
            "feature": "predicate_orientation",
            "observation_unit": "original assertion or CORE sentence attribute on a canonical connection",
            "counting_rule": "Collapsed connectivity stores predicate types with subject and object orientation removed. Directional connectivity stores with_traversal or against_traversal for each path direction. The original subject and object stay in the directional table. Negation stays in its own field. CORE expression, cancer status, and relation nature are sentence attributes and do not set traversal direction.",
            "coverage_denominator": "Eligible connections are those used by the existing path rules.",
            "missing_value": "not_applicable marks a record that does not define traversal.",
            "recompute_from": "analysis/connectivity_predicates.parquet and analysis/assertion_orientations.parquet",
        },
        {
            "feature": "later_prediction_mask",
            "observation_unit": "target PMID",
            "counting_rule": "A later prediction experiment rebuilds citation features after removing the target PMIDs from every role: citing article, cited article, and third-party witness. Publication years are retained for a later time cutoff.",
            "coverage_denominator": "Not applied in this audit.",
            "missing_value": "This audit uses the pinned snapshot as a descriptive view and does not mask targets.",
            "recompute_from": "analysis/directed_citation_links.parquet, endpoint_metadata.parquet, and edge_pmid_membership.parquet",
        },
    ]


def render_report(path: Path, snapshot: dict, validation: dict, coverage: dict, totals: dict[str, list[int]]) -> None:
    def line(kind: str, name: str) -> str:
        values = totals[kind]
        return f"{value(values, name):,} distinct; {value(values, name, True):,} path-weighted"

    sections = []
    status = "passed" if not validation["failures"] else "failed"
    sections.append("# Citation patterns on adjacent CORE–SemMedDB edges\n")
    sections.append(
        f"Validation {status}. The citation snapshot is iCite Database Snapshot 2024-08, version DOI {snapshot['doi']}. "
        f"Column order is `{validation['orientation']['citing_column']}` then `{validation['orientation']['cited_column']}`.\n"
    )
    sections.append("## Coverage\n")
    support = coverage["support"]
    sections.append(
        f"Qualifying support PMIDs: {support['qualifying']:,}. "
        f"In snapshot metadata: {support['in_metadata']:,}. "
        f"With at least one outgoing link: {support['with_outgoing']:,}. "
        f"With at least one incoming link: {support['with_incoming']:,}. "
        f"In metadata but with no indexed link: {support['metadata_without_link']:,}. "
        f"Indexed but absent from metadata: {support['link_without_metadata']:,}.\n"
    )
    sections.append(
        "A qualifying PMID is outgoing-indexed when the retained archive has a link in which it is the citing article, and incoming-indexed when it is the cited article. "
        "Those flags are complete for qualifying PMIDs because every archive row with a qualifying endpoint was kept. "
        "Metadata membership is recorded separately. A positive motif is kept when companion metadata is missing; those pairs are counted under the metadata-gap measures.\n"
    )
    sections.append(
        "Left-to-right coverage uses outgoing links on the left edge. Right-to-left uses outgoing links on the right edge. "
        "Co-citation uses incoming links on both edges. Shared references use outgoing links on both edges. "
        "Complete means every qualifying PMID on the required side is indexed. Partial means at least one, but not all, are indexed. "
        "Absent means none are indexed. A pair with no motif and a complete index is a negative observation. "
        "A pair with no motif and a partial index is not treated as a confirmed absence. "
        "Pairs with an empty qualifying set are neither or one-sided and stay outside the motif denominators.\n"
    )
    labels = {
        "gene_t1": "Two-edge positions 1–2",
        "cancer_12": "Three-edge positions 1–2",
        "gene_23": "Three-edge positions 2–3",
    }
    sections.append("## Motifs\n")
    sections.append(
        "The main motif sets are the union of CORE PMIDs and SemMedDB path-eligible affirmative PMIDs. "
        "Negated SemMedDB PMIDs stay in the membership table and do not enter these sets. "
        "The occurrence weight is 1 for each two-edge pair. For three-edge positions 1–2 it is the number of eligible final edges; for positions 2–3 it is the number of eligible initial edges.\n"
    )
    for kind, label in labels.items():
        values = totals[kind]
        external = value(values, "cocite_external") + value(values, "ref_external")
        witnesses = value(values, "cocite_witnesses") + value(values, "ref_witnesses")
        sections.append(f"### {label}\n")
        sections.append(
            f"Adjacent pairs: {line(kind, 'pairs')}. "
            f"Qualifying PMID availability: neither {line(kind, 'neither')}; one-sided {line(kind, 'one_sided')}; both {line(kind, 'both')}.\n"
        )
        sections.append(
            f"Shared article pairs: {line(kind, 'shared_q_pairs')}. "
            f"Sum of shared PMIDs: {line(kind, 'shared_q_sum')}.\n"
        )
        sections.append(
            f"Direction-collapsed direct-citation pairs: {line(kind, 'collapsed_pairs')}. "
            f"Collapsed links: {line(kind, 'collapsed_links')}. "
            f"Exclusive left-to-right links: {line(kind, 'ltr_links')}. "
            f"Exclusive right-to-left links: {line(kind, 'rtl_links')}. "
            f"Ambiguous links: {line(kind, 'ambiguous_links')}.\n"
        )
        sections.append(
            f"Left-to-right index complete / partial / absent: {line(kind, 'ltr_index_complete')}; {line(kind, 'ltr_index_partial')}; {line(kind, 'ltr_index_absent')}. "
            f"No left-to-right motif in a complete index: {line(kind, 'ltr_no_motif_complete')}. "
            f"Positive left-to-right with a metadata gap: {line(kind, 'ltr_gap')}.\n"
        )
        sections.append(
            f"Right-to-left index complete / partial / absent: {line(kind, 'rtl_index_complete')}; {line(kind, 'rtl_index_partial')}; {line(kind, 'rtl_index_absent')}. "
            f"No right-to-left motif in a complete index: {line(kind, 'rtl_no_motif_complete')}.\n"
        )
        sections.append(
            f"Co-citation pairs: {line(kind, 'cocite_pairs')}. Witnesses: {line(kind, 'cocite_witnesses')}. "
            f"External witnesses: {line(kind, 'cocite_external')}. "
            f"Index complete / partial / absent: {line(kind, 'cocite_index_complete')}; {line(kind, 'cocite_index_partial')}; {line(kind, 'cocite_index_absent')}. "
            f"No motif in a complete incoming index: {line(kind, 'cocite_no_motif_complete')}.\n"
        )
        sections.append(
            f"Shared-reference pairs: {line(kind, 'ref_pairs')}. Witnesses: {line(kind, 'ref_witnesses')}. "
            f"External witnesses: {line(kind, 'ref_external')}. "
            f"Index complete / partial / absent: {line(kind, 'ref_index_complete')}; {line(kind, 'ref_index_partial')}; {line(kind, 'ref_index_absent')}.\n"
        )
        sections.append(
            f"Article-pair counts are exact for {line(kind, 'cocite_article_exact_pairs')} co-citation pairs "
            f"and {line(kind, 'ref_article_exact_pairs')} shared-reference pairs. "
            f"The remaining pairs, {line(kind, 'cocite_article_omitted_pairs')} and {line(kind, 'ref_article_omitted_pairs')}, "
            f"have exact witness counts and no article-pair total. "
            f"Witness search skipped: {line(kind, 'witness_skipped_pairs')}.\n"
        )
        if witnesses:
            sections.append(
                f"External witnesses are {external:,} of {witnesses:,} distinct-pair witness occurrences in this position "
                f"({external / witnesses:.4f}).\n"
            )
        edge_cov = coverage["positions"][kind]
        sections.append(
            f"Distinct qualifying PMIDs on the left role: {edge_cov['left']['distinct_pmids']:,}, "
            f"of which {edge_cov['left']['distinct_outgoing']:,} have outgoing links and {edge_cov['left']['distinct_incoming']:,} have incoming links. "
            f"Right role: {edge_cov['right']['distinct_pmids']:,} PMIDs, "
            f"{edge_cov['right']['distinct_outgoing']:,} outgoing and {edge_cov['right']['distinct_incoming']:,} incoming.\n"
        )
    sections.append("## Direction representations ready for later comparison\n")
    sections.append(
        "Connectivity with predicate type and collapsed subject/object orientation is in `analysis/connectivity_predicates.parquet`. "
        "The same connections with traversal alignment are in `analysis/assertion_orientations.parquet`. "
        "Direction-collapsed and directional direct citations are the collapsed, exclusive left-to-right, exclusive right-to-left, and ambiguous measures; they partition the same links. "
        "Shared article, co-citation, and shared reference are symmetric and can be used with either citation representation. "
        "CORE expression change, cancer status, and relation nature remain sentence attributes. Negation remains its own field.\n"
    )
    sections.append(
        "A later prediction experiment should rebuild the citation features after masking target PMIDs in every role, including citing articles, cited articles, and third-party witnesses. "
        "Publication years are in `analysis/endpoint_metadata.parquet` for a later time cutoff. This audit does not apply that mask.\n"
    )
    sections.append("## Inputs\n")
    sections.append(
        f"Graph: `{snapshot['graph_run']}`. Path evidence audit: `{snapshot['path_audit']}`. "
        f"Completion audit used for the published adjacent totals: `{snapshot['completion_audit']}`.\n"
    )
    path.write_text("\n".join(sections), encoding="utf-8")


def role_edges(kind: str) -> tuple[list[Edge], list[Edge]]:
    if kind == "gene_t1":
        return [edge for edge in GG if edge.family == "gene_gene"], [edge for edge in GC if edge.query_edge and edge.gene]
    if kind == "cancer_12":
        return [edge for edge in GC if edge.candidate], [edge for edge in GC if edge.extend]
    return [edge for edge in GC if edge.n_left], [edge for edge in GC if edge.query_edge]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--graph-run", type=Path, default=GRAPH_RUN)
    parser.add_argument("--path-audit", type=Path, default=PATH_AUDIT)
    parser.add_argument("--completion-audit", type=Path, default=COMPLETION_AUDIT)
    parser.add_argument("--witness", choices=("all", "hubs", "none"), default="all")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--write-large", action="store_true")
    args = parser.parse_args()
    global OUTGOING, INCOMING, INSIDE
    timer = StageTimer()
    failures: list[dict] = []
    run_dir = ensure_dir(args.run_dir)
    analysis = ensure_dir(run_dir / "analysis")
    ensure_dir(run_dir / "logs")
    sources = load_yaml(REPO_ROOT / "config/data_sources.yaml")["citations"]
    data_dir = Path("/home/b5ac/freddieyu.b5ac/projects/project_2/data/nih_occ_icite_202408_v1")
    files = {item["name"]: item for item in sources["files"]}
    citation_zip = data_dir / "open_citation_collection.zip"
    metadata_zip = data_dir / "icite_metadata.zip"
    archives = verify_archives(citation_zip, metadata_zip, files["open_citation_collection.zip"], files["icite_metadata.zip"])
    timer.mark("archives")
    for name, record in archives.items():
        if not record["size_matches"] or not record["md5_matches"] or not str(record["integrity"]).startswith("ok"):
            failures.append({"name": f"archive_{name}", "observed": record})
    known = known_link_check(citation_zip, metadata_zip)
    timer.mark("known_link")
    if not known["header_is_citing_referenced"] or known["references_found_as_citing_to_referenced"] != known["n_references"]:
        failures.append({"name": "known_citation_orientation", "observed": known})
    if failures:
        write_json(
            run_dir / "validation.json",
            {"failures": failures, "archives": archives, "known_link": known, "status": "failed", "stages": timer.stages},
        )
        raise SystemExit(1)

    candidates = {
        row["gene_id"]
        for row in pq.read_table(args.graph_run / "analysis" / "candidate_genes.parquet", columns=["gene_id", "status"]).to_pylist()
        if row["status"] == "resolved_human"
    }
    queries = {row["cancer_cui"] for row in pq.read_table(args.graph_run / "analysis" / "cancer_queries.parquet", columns=["cancer_cui"]).to_pylist()}
    connections = load_connections(args.graph_run / "graph")
    prepared = write_orientations(connections, analysis)
    profiles, _shared = derive_profiles(connections, candidates, queries)
    weights = attach_weights(profiles, candidates, queries)
    timer.mark("graph", connections=len(connections), profiles=len(profiles))
    _check(failures, "one_intermediate_paths", int(weights["one_total"]), EXPECTED_ONE_INTERMEDIATE_PATHS)
    _check(failures, "two_intermediate_paths", int(weights["two_total"]), EXPECTED_TWO_INTERMEDIATE_PATHS)
    _check(failures, "bitset_two_intermediate_paths", int(weights["bitset_two_total"]), EXPECTED_TWO_INTERMEDIATE_PATHS)
    build_edges(profiles, weights, prepared["raw"], candidates, queries)
    eligible_profiles = sum(1 for profile in profiles if profile["support"] != "no_traversal_support")
    _check(failures, "eligible_connection_tables", len(prepared["eligible_keys"]), eligible_profiles)
    del connections, profiles, _shared
    timer.mark("edges", gene_cancer=len(GC), gene_gene=len(GG), qualifying=len(prepared["qualifying"]))
    if failures:
        write_json(run_dir / "validation.json", {"failures": failures, "status": "failed", "stages": timer.stages})
        raise SystemExit(1)

    support_flag = np.zeros(MAX_PMID, dtype=np.uint8)
    for pmid in prepared["qualifying"]:
        if 0 < pmid < MAX_PMID:
            support_flag[pmid] = 1
    role = np.zeros(MAX_PMID, dtype=np.uint8)
    for pmid in prepared["qualifying"]:
        if 0 < pmid < MAX_PMID:
            role[pmid] |= 1
    for pmid in prepared["full_graph"]:
        if 0 < pmid < MAX_PMID and not support_flag[pmid]:
            role[pmid] |= 2
    INSIDE = np.zeros(MAX_PMID, dtype=np.uint8)
    for pmid in prepared["full_graph"]:
        if 0 < pmid < MAX_PMID:
            INSIDE[pmid] = 1
    year, present, citation_count = stream_metadata(metadata_zip, support_flag)
    timer.mark("metadata")
    citation_scan = stream_citations(citation_zip, support_flag, role)
    timer.mark("citation_scan", kept=citation_scan["kept_rows"])
    outgoing_raw, outgoing_appended, outgoing_unique = freeze_index(citation_scan["first_to_second"])
    incoming_raw, incoming_appended, incoming_unique = freeze_index(citation_scan["second_to_first"])
    citation_scan["first_to_second"].clear()
    citation_scan["second_to_first"].clear()
    OUTGOING = outgoing_raw
    INCOMING = incoming_raw
    decision = orientation_decision(known, OUTGOING, INCOMING, citation_count)
    timer.mark("orientation")
    if not decision["confirmed"]:
        failures.append({"name": "citation_orientation", "observed": decision})
    unique_pairs = unique_directed_pairs(OUTGOING, INCOMING, support_flag)
    duplicate_rows = citation_scan["kept_rows"] - unique_pairs
    attach_citation_fields(GC, OUTGOING, support_flag, present)
    attach_citation_fields(GG, OUTGOING, support_flag, present)
    timer.mark("edge_links")

    link_rows = 0
    endpoint_rows = 0
    if args.write_large and decision["confirmed"]:
        link_rows = write_links(analysis / "directed_citation_links.parquet", OUTGOING, INCOMING, support_flag)
        endpoint_rows = write_endpoint_metadata(analysis / "endpoint_metadata.parquet", role, year, present)
        timer.mark("large_tables", links=link_rows, endpoints=endpoint_rows)
        _check(failures, "written_links_match_unique_pairs", link_rows, unique_pairs)

    witness_groups = {"cancer": set(), "gene": set()}
    if args.witness == "hubs":
        cancer_sizes = defaultdict(int)
        gene_sizes = defaultdict(int)
        for edge in GC:
            cancer_sizes[edge.cancer] += 1
            gene_sizes[edge.gene] += 1
        witness_groups["cancer"] = {cancer for cancer, _size in sorted(cancer_sizes.items(), key=lambda item: item[1], reverse=True)[:5]}
        witness_groups["gene"] = {gene for gene, _size in sorted(gene_sizes.items(), key=lambda item: item[1], reverse=True)[:5]}
        small = [cancer for cancer, size in sorted(cancer_sizes.items(), key=lambda item: item[1]) if size >= 2][:3]
        witness_groups["cancer"].update(small)

    gc_index = {}
    gg_index = {}
    by_cancer = defaultdict(list)
    by_gene = defaultdict(list)
    for index, edge in enumerate(GC):
        gc_index[(edge.gene, edge.cancer)] = index
        by_cancer[edge.cancer].append(index)
        by_gene[edge.gene].append(index)
    for index, edge in enumerate(GG):
        gg_index[(edge.gene, edge.cancer)] = index
    specs = []
    missing_edges = 0
    for cancer, indices in by_cancer.items():
        do_witness = args.witness == "all" or cancer in witness_groups["cancer"]
        specs.append(("cancer_12", cancer, indices, do_witness))
    for gene, indices in by_gene.items():
        if len(indices) < 2:
            continue
        do_witness = args.witness == "all" or gene in witness_groups["gene"]
        specs.append(("gene_23", gene, indices, do_witness))
    for gene, query_set in weights["queries_of"].items():
        partners = [item for item in weights["neighbors"].get(gene, ()) if item in candidates and item != gene]
        if not partners or not query_set:
            continue
        left_ids = []
        for partner in partners:
            left, right = (partner, gene) if partner <= gene else (gene, partner)
            found = gg_index.get((left, right))
            if found is None:
                missing_edges += 1
            else:
                left_ids.append(found)
        right_ids = []
        for query in query_set:
            found = gc_index.get((gene, query))
            if found is None:
                missing_edges += 1
            else:
                right_ids.append(found)
        if left_ids and right_ids:
            do_witness = args.witness == "all" or gene in witness_groups["gene"]
            specs.append(("gene_t1", gene, left_ids, right_ids, do_witness))
    _check(failures, "missing_adjacent_edges", missing_edges, 0)
    def spec_size(spec: tuple) -> int:
        if spec[0] == "gene_t1":
            return len(spec[2]) + len(spec[3])
        return len(spec[2])

    specs.sort(key=spec_size, reverse=True)
    totals = {kind: fresh() for kind in TEMPLATE_NAMES}
    group_rows = []
    started = time.time()
    workers = max(1, args.workers)
    pool = None
    if workers == 1:
        results = (process_spec(spec) for spec in specs)
    else:
        pool = mp.get_context("fork").Pool(workers)
        results = pool.imap_unordered(process_spec, specs, chunksize=1)
    try:
        for number, (part, meta) in enumerate(results, start=1):
            add_into(totals[meta["kind"]], part)
            if meta["edges"] >= 500 or meta["seconds"] >= 5:
                group_rows.append(meta)
            if number % 500 == 0 or meta["edges"] >= 2000:
                print(
                    f"[groups] {number}/{len(specs)} kind={meta['kind']} edges={meta['edges']} seconds={meta['seconds']} elapsed={round(time.time()-started,1)}",
                    flush=True,
                )
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    timer.mark("motifs", groups=len(specs))
    for kind, expected in EXPECTED_ADJACENT.items():
        for name, expected_value in expected.items():
            weighted = name.endswith("_w")
            measure = name[:-2] if weighted else name
            _check(failures, f"{kind}_{name}", value(totals[kind], measure, weighted), expected_value)
        link_sum = value(totals[kind], "ltr_links") + value(totals[kind], "rtl_links") + value(totals[kind], "ambiguous_links")
        _check(failures, f"{kind}_collapsed_partition", value(totals[kind], "collapsed_links"), link_sum)
        link_sum_w = value(totals[kind], "ltr_links", True) + value(totals[kind], "rtl_links", True) + value(totals[kind], "ambiguous_links", True)
        _check(failures, f"{kind}_collapsed_partition_weighted", value(totals[kind], "collapsed_links", True), link_sum_w)
        if args.witness == "all":
            _check(failures, f"{kind}_witness_skipped", value(totals[kind], "witness_skipped_pairs"), 0)

    position_coverage = {}
    for kind in TEMPLATE_NAMES:
        left_edges, right_edges = role_edges(kind)
        position_coverage[kind] = {
            "left": pmid_coverage(left_edges, prepared["qualifying"], present),
            "right": pmid_coverage(right_edges, prepared["qualifying"], present),
        }
    qualifying_list = [pmid for pmid in prepared["qualifying"] if 0 < pmid < MAX_PMID]
    support_summary = {
        "qualifying": len(qualifying_list),
        "full_graph": len(prepared["full_graph"]),
        "in_metadata": sum(1 for pmid in qualifying_list if present[pmid]),
        "with_outgoing": sum(1 for pmid in qualifying_list if pmid in OUTGOING and OUTGOING[pmid].size),
        "with_incoming": sum(1 for pmid in qualifying_list if pmid in INCOMING and INCOMING[pmid].size),
        "metadata_without_link": sum(
            1
            for pmid in qualifying_list
            if present[pmid] and pmid not in OUTGOING and pmid not in INCOMING
        ),
        "link_without_metadata": sum(
            1
            for pmid in qualifying_list
            if not present[pmid] and (pmid in OUTGOING or pmid in INCOMING)
        ),
    }
    summary_rows = []
    for kind in TEMPLATE_NAMES:
        summary_rows.extend(counter_rows(kind, totals[kind]))
    write_tsv(analysis / "adjacent_citation_summary.tsv", summary_rows, ["template", "positions", "measure", "distinct_count", "path_occurrence_count"])
    write_tsv(
        analysis / "group_profile.tsv",
        group_rows,
        ["kind", "group", "edges", "seconds", "witness", "cocite_article_status", "ref_article_status", "pairs", "weighted"],
    )
    write_json(analysis / "feature_dictionary.json", {"features": feature_dictionary()})
    toy = {
        "direct_exclusive": direct_citation_roles((10,), (20,), {10: (20,), 20: (10,)}),
        "direct_ambiguous": direct_citation_roles((10, 20), (10, 20), {10: (20,)}),
        "external_and_false_shared_pmid": pair_features((1,), (2,), {1: (7,), 2: (7,)}, {1: (9,), 2: (9,)}, {1, 2}),
        "false_shared_pmid": pair_features((5,), (5,), {5: (8,)}, {5: (9,)}, {5}),
        "reverse_traversal": traversal_alignment("gene:10", "cancer:C1", "cancer:C1", "gene:10"),
    }
    fits_int64 = all(abs(item) <= 2**63 - 1 for values in totals.values() for item in values)
    try:
        revision = subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision = ""
    validation = {
        "failures": failures,
        "fits_int64": fits_int64,
        "duplicate_citation_rows": duplicate_rows,
        "unique_directed_pairs": unique_pairs,
        "citation_scan": {key: citation_scan[key] for key in ("header", "rows", "invalid_rows", "self_link_rows", "kept_rows", "high_pmid_endpoints")},
        "index_appends": {"outgoing": outgoing_appended, "incoming": incoming_appended, "outgoing_unique_targets": outgoing_unique, "incoming_unique_citers": incoming_unique},
        "orientation": decision,
        "toy_examples": toy,
        "article_pair_scope": {
            "rule": f"Exact distinct cross-edge article pairs are retained when a group has at most {ARTICLE_EDGE_LIMIT} edges and at most {ARTICLE_KEY_LIMIT} article-pair keys. Larger groups keep exact witness counts and omit the article-pair total.",
            "witness_scope": args.witness,
        },
        "coverage_rule": "Complete index coverage is required before a zero motif is read as a negative observation. Partial coverage and missing metadata stay separate. Positive archive motifs are retained when metadata is missing.",
        "stages": timer.stages,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    }
    if not fits_int64:
        failures.append({"name": "int64_range", "observed": "counter exceeds int64"})
    validation["failures"] = failures
    validation["status"] = "passed" if not failures else "failed"
    write_json(run_dir / "validation.json", validation)
    snapshot = {
        "task": "citation_path_evidence_audit",
        "status": validation["status"],
        "release_id": sources["release_id"],
        "title": sources["title"],
        "doi": sources["doi"],
        "figshare_article_id": sources["figshare_article_id"],
        "figshare_version": sources["figshare_version"],
        "published": sources["published"].isoformat().replace("+00:00", "Z") if hasattr(sources["published"], "isoformat") else str(sources["published"]),
        "version_url": sources["version_url"],
        "files": {
            "open_citation_collection.zip": {**files["open_citation_collection.zip"], **archives["open_citation_collection.zip"]},
            "icite_metadata.zip": {**files["icite_metadata.zip"], **archives["icite_metadata.zip"]},
        },
        "citing_column": "citing",
        "cited_column": "referenced",
        "graph_run": str(args.graph_run),
        "path_audit": str(args.path_audit),
        "completion_audit": str(args.completion_audit),
        "code_revision": revision,
        "qualifying_definition": "CORE PMID union SemMedDB path-eligible affirmative PMID on a traversable edge",
        "full_graph_pmid_definition": "Every PMID on a prepared CORE or SemMedDB record, including negated and other SemMedDB records",
        "direction_definitions": {
            "with_traversal": "The path runs from the assertion's original subject to its original object.",
            "against_traversal": "The path runs from the assertion's original object to its original subject. The stored subject and object do not change.",
            "sentence_attribute": "CORE expression change, cancer status, and relation nature do not define traversal direction.",
            "ambiguous_citation": "Both PMIDs of a directed link belong to both edge support sets, so the link fills both positional roles and is counted once in the collapsed total.",
        },
        "written_links": link_rows,
        "written_endpoints": endpoint_rows,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    }
    write_json(run_dir / "manifest.json", snapshot)
    render_report(run_dir / "report.md", snapshot, validation, {"support": support_summary, "positions": position_coverage}, totals)
    write_json(run_dir / "logs" / "runtime.json", {"stages": timer.stages, "status": validation["status"]})
    print(f"[done] status={validation['status']} failures={len(failures)}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

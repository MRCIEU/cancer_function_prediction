"""Deterministic stratified selection. This is not a probability sample."""

from __future__ import annotations

import hashlib

from pairpilot.contracts import pair_id, pair_key

CLASSES = ("identical", "core_subset", "semmed_subset", "partial_overlap", "disjoint")


def rank(seed: int, stratum: str, key: str) -> str:
    return hashlib.sha256(f"{seed}|{stratum}|{key}".encode("utf-8")).hexdigest()


def eligible(row: dict, low: int, high: int, min_each: int) -> bool:
    return (
        row["relation_class"] in CLASSES
        and int(row["n_core"]) >= min_each
        and int(row["n_semmed"]) >= min_each
        and low <= int(row["n_union"]) <= high
    )


def select_pairs(rows: list[dict], seed: int, bands: dict, min_each: int = 2) -> dict:
    """Pick one pair per class and band. Do not relax an empty stratum."""
    frame = []
    for row in rows:
        item = dict(row)
        item["pair_key"] = pair_key(item["gene_id"], item["disease_cui"])
        item["pair_id"] = pair_id(str(item["gene_id"]), str(item["disease_cui"]))
        frame.append(item)
    frame.sort(key=lambda item: item["pair_key"])
    selected = []
    candidates = []
    deviations = []
    used_genes = set()
    used_diseases = set()
    for relation in CLASSES:
        for band, bounds in (("small", bands["small"]), ("large", bands["large"])):
            stratum = f"{relation}:{band}"
            pool = [
                item for item in frame
                if item["relation_class"] == relation and eligible(item, bounds[0], bounds[1], min_each)
            ]
            pool.sort(key=lambda item: (rank(seed, stratum, item["pair_key"]), item["pair_key"]))
            for order, item in enumerate(pool):
                candidates.append({
                    "stratum": stratum,
                    "order": order,
                    "pair_id": item["pair_id"],
                    "pair_key": item["pair_key"],
                    "gene_id": str(item["gene_id"]),
                    "disease_cui": str(item["disease_cui"]),
                    "relation_class": relation,
                    "band": band,
                    "n_core": int(item["n_core"]),
                    "n_semmed": int(item["n_semmed"]),
                    "n_union": int(item["n_union"]),
                    "n_intersection": int(item.get("n_intersection") or 0),
                })
            fresh = [
                item for item in pool
                if str(item["gene_id"]) not in used_genes and str(item["disease_cui"]) not in used_diseases
            ]
            chosen = fresh[0] if fresh else (pool[0] if pool else None)
            if chosen is None:
                deviations.append({
                    "stratum": stratum,
                    "kind": "stratum_empty",
                    "detail": "No pair met the PMID counts. The threshold was not relaxed.",
                })
                continue
            reused = str(chosen["gene_id"]) in used_genes or str(chosen["disease_cui"]) in used_diseases
            if reused:
                deviations.append({
                    "stratum": stratum,
                    "kind": "reused_gene_or_disease",
                    "pair_id": chosen["pair_id"],
                    "detail": "Every remaining candidate reused a gene or disease already selected.",
                })
            used_genes.add(str(chosen["gene_id"]))
            used_diseases.add(str(chosen["disease_cui"]))
            selected.append({
                "stratum": stratum,
                "band": band,
                "relation_class": relation,
                "pair_id": chosen["pair_id"],
                "pair_key": chosen["pair_key"],
                "gene_id": str(chosen["gene_id"]),
                "disease_cui": str(chosen["disease_cui"]),
                "n_core": int(chosen["n_core"]),
                "n_semmed": int(chosen["n_semmed"]),
                "n_union": int(chosen["n_union"]),
                "n_intersection": int(chosen.get("n_intersection") or 0),
                "selection_order": len(selected),
            })
    return {"selected": selected, "candidates": candidates, "deviations": deviations}


def choose_ids(selected: list[dict], seed: int, limit: int, predicate) -> list[str]:
    pool = [item["pair_id"] for item in selected if predicate(item)]
    pool.sort(key=lambda value: (rank(seed, "subset", value), value))
    return pool[:limit]


def choose_pmids(pair_pmids: list[dict], seed: int, per_pair: int, global_limit: int) -> list[dict]:
    """Stable PMID subset for E3. Availability of a source sentence is metadata, not a selector."""
    grouped = {}
    for row in pair_pmids:
        grouped.setdefault(row["pair_id"], []).append(str(row["pmid"]))
    chosen = []
    seen = set()
    for pair in sorted(grouped, key=lambda value: (rank(seed, "e3", value), value)):
        pmids = sorted(set(grouped[pair]), key=lambda value: (rank(seed, f"e3:{pair}", value), value))
        kept = 0
        for pmid in pmids:
            if kept >= per_pair or len(seen) >= global_limit and pmid not in seen:
                continue
            if pmid not in seen and len(seen) >= global_limit:
                continue
            chosen.append({"pair_id": pair, "pmid": pmid})
            seen.add(pmid)
            kept += 1
    return chosen

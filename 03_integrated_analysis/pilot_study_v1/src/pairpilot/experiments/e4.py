"""Two-hop path weights. Degree is typed degree in the supplied graph, not a record count."""

from __future__ import annotations


def typed_degree(edges: list[dict], node: str, predicate: str, direction: str) -> int:
    neighbors = set()
    for edge in edges:
        if edge.get("negated"):
            continue
        if edge.get("predicate") != predicate or edge.get("direction") != direction:
            continue
        if edge.get("start") == node:
            neighbors.add(edge.get("end"))
        elif direction == "undirected" and edge.get("end") == node:
            neighbors.add(edge.get("start"))
    return len(neighbors)


def path_degree_weight(intermediate_degree: int, damping: float = 0.4) -> float:
    if intermediate_degree <= 0:
        return 0.0
    return float(intermediate_degree) ** (-float(damping))


def dwpc(weights: list[float]) -> float:
    return float(sum(weights))


def positive_paths(edges: list[dict], source: str, target: str) -> list[dict]:
    outgoing = {}
    incoming = {}
    for edge in edges:
        if edge.get("negated"):
            continue
        outgoing.setdefault(edge["start"], []).append(edge)
        incoming.setdefault(edge["end"], []).append(edge)
    paths = []
    for first in outgoing.get(source, []):
        mid = first["end"]
        if mid in {source, target}:
            continue
        for second in outgoing.get(mid, []):
            if second["end"] != target:
                continue
            pmids_1 = set(map(str, first.get("pmids") or []))
            pmids_2 = set(map(str, second.get("pmids") or []))
            paths.append({
                "path_id": f"{source}|{first['predicate']}|{mid}|{second['predicate']}|{target}",
                "metapath": f"{first['predicate']}|{second['predicate']}",
                "intermediate": mid,
                "edge_1_predicate": first["predicate"],
                "edge_2_predicate": second["predicate"],
                "n_pmid_edge_1": len(pmids_1),
                "n_pmid_edge_2": len(pmids_2),
                "n_pmid_intersection": len(pmids_1 & pmids_2),
                "literature_min": min(len(pmids_1), len(pmids_2)),
                "first": first,
                "second": second,
            })
    return paths


def rank_paths(paths: list[dict], edges: list[dict], damping: float = 0.4) -> list[dict]:
    ranked = []
    for path in paths:
        degree = typed_degree(edges, path["intermediate"], path["edge_2_predicate"], "forward")
        weight = path_degree_weight(degree, damping)
        item = dict(path)
        item["intermediate_typed_degree"] = degree
        item["degree_weight"] = weight
        item["damping"] = damping
        ranked.append(item)
    by_weight = sorted(ranked, key=lambda row: (-row["degree_weight"], -row["literature_min"], row["path_id"]))
    by_literature = sorted(ranked, key=lambda row: (-row["literature_min"], row["path_id"]))
    return by_weight, by_literature


def topk_overlap(left: list[dict], right: list[dict], k: int) -> dict:
    a = {row["path_id"] for row in left[:k]}
    b = {row["path_id"] for row in right[:k]}
    return {"k": k, "overlap": len(a & b), "weight_ids": sorted(a), "literature_ids": sorted(b)}

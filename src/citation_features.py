"""Citation motifs between the PMID sets of two adjacent edges.

A shared PMID is shared article membership. A citation link keeps its original
citing-to-cited direction. Co-citation and shared references are symmetric.
"""

from __future__ import annotations

from collections import Counter


def intersection_size(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    if not left or not right:
        return 0
    i = j = found = 0
    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            found += 1
            i += 1
            j += 1
        elif left[i] < right[j]:
            i += 1
        else:
            j += 1
    return found


def direct_citation_roles(left: tuple[int, ...], right: tuple[int, ...], outgoing: dict[int, tuple[int, ...]]) -> dict:
    """Split directed links into exclusive positional roles and one collapsed total.

    A link is ambiguous when both of its articles belong to both support sets,
    so the same arrow fills both positional roles.
    """
    left_set = set(left)
    right_set = set(right)
    exclusive_ltr = exclusive_rtl = ambiguous = 0
    seen = set()
    for source in left_set | right_set:
        for target in outgoing.get(source, ()):
            if source == target or (source, target) in seen:
                continue
            ltr = source in left_set and target in right_set
            rtl = source in right_set and target in left_set
            if not ltr and not rtl:
                continue
            seen.add((source, target))
            if ltr and rtl:
                ambiguous += 1
            elif ltr:
                exclusive_ltr += 1
            else:
                exclusive_rtl += 1
    return {
        "exclusive_left_to_right": exclusive_ltr,
        "exclusive_right_to_left": exclusive_rtl,
        "ambiguous_position": ambiguous,
        "direction_collapsed": exclusive_ltr + exclusive_rtl + ambiguous,
    }


def _distinct_papers(left_papers: set[int], right_papers: set[int]) -> bool:
    if not left_papers or not right_papers:
        return False
    if len(left_papers) > 1 or len(right_papers) > 1:
        return True
    return next(iter(left_papers)) != next(iter(right_papers))


def third_party_motifs(left: tuple[int, ...], right: tuple[int, ...], neighbors: dict[int, tuple[int, ...]], support: set[int]) -> dict:
    """Count witnesses and cross-edge article pairs for one symmetric motif.

    ``neighbors`` is incoming for co-citation and outgoing for shared references.
    A witness attached only to one PMID that sits in both sets does not count.
    """
    left_set = set(left)
    right_set = set(right)
    papers = left_set | right_set
    by_witness: dict[int, set[int]] = {}
    for paper in papers:
        for witness in neighbors.get(paper, ()):
            if witness == paper:
                continue
            by_witness.setdefault(witness, set()).add(paper)
    witnesses = []
    article_pairs = set()
    for witness, cited in by_witness.items():
        on_left = cited & left_set
        on_right = cited & right_set
        if not _distinct_papers(on_left, on_right):
            continue
        witnesses.append(witness)
        for paper in on_left:
            for other in on_right:
                if paper != other:
                    article_pairs.add((paper, other))
    external = sum(1 for witness in witnesses if witness not in support)
    return {
        "witnesses": len(witnesses),
        "external_witnesses": external,
        "article_pairs": len(article_pairs),
    }


def traversal_alignment(subject_id: str, object_id: str, path_from: str, path_to: str) -> str:
    """Compare an assertion's original subject with one path traversal.

    The original subject and object stay unchanged. ``against_traversal`` means
    the path runs from the assertion's object toward its subject.
    """
    if subject_id == path_from and object_id == path_to:
        return "with_traversal"
    if subject_id == path_to and object_id == path_from:
        return "against_traversal"
    return "not_applicable"


def accumulate_pair(store: Counter, prefix: tuple, features: dict, weight: int) -> None:
    store[(*prefix, "pairs")] += 1
    store[(*prefix, "pairs_weighted")] += weight
    for name, value in features.items():
        if value:
            store[(*prefix, f"{name}_pairs")] += 1
            store[(*prefix, f"{name}_pairs_weighted")] += weight
            store[(*prefix, f"{name}_sum")] += int(value)
            store[(*prefix, f"{name}_sum_weighted")] += int(value) * weight


def classify_prepared_links(
    left: tuple[int, ...],
    right: tuple[int, ...],
    left_links: tuple[tuple[int, int], ...],
    right_links: tuple[tuple[int, int], ...],
) -> tuple[int, int, int]:
    """Count unique directed links already restricted to each edge's own PMIDs.

    ``left_links`` contains each citing-to-cited pair once. A link stored on
    both edges because the citing PMID is in both support sets is counted once.
    """
    left_set = left if isinstance(left, (set, frozenset)) else set(left)
    right_set = right if isinstance(right, (set, frozenset)) else set(right)
    exclusive_ltr = exclusive_rtl = ambiguous = 0
    for source, target in left_links:
        if source == target or target not in right_set or source not in left_set:
            continue
        if source in right_set and target in left_set:
            ambiguous += 1
        else:
            exclusive_ltr += 1
    for source, target in right_links:
        if source == target or target not in left_set or source not in right_set:
            continue
        if source in left_set and target in right_set:
            continue
        exclusive_rtl += 1
    return exclusive_ltr, exclusive_rtl, ambiguous


def _fill_witness_blocks(by_edge: dict[int, set[int]], witnesses, external) -> None:
    """Add one witness to every ordered edge pair that has two distinct papers.

    Edges that carry two or more of the witness's papers match every other
    edge. Singleton edges match only singleton edges with a different PMID.
    """
    import numpy as np

    multi = [edge for edge, papers in by_edge.items() if len(papers) > 1]
    singles: dict[int, list[int]] = {}
    for edge, papers in by_edge.items():
        if len(papers) == 1:
            singles.setdefault(next(iter(papers)), []).append(edge)
    all_idx = np.fromiter(by_edge.keys(), dtype=np.int64, count=len(by_edge))
    multi_arr = np.asarray(multi, dtype=np.int64)
    matrices = [witnesses] if external is None else [witnesses, external]
    single_items = list(singles.items())
    for matrix in matrices:
        if multi_arr.size:
            matrix[np.ix_(multi_arr, all_idx)] += 1
            matrix[multi_arr, multi_arr] -= 1
        for index, (_pmid, edges_a) in enumerate(single_items):
            rows = np.asarray(edges_a, dtype=np.int64)
            if multi_arr.size:
                matrix[np.ix_(rows, multi_arr)] += 1
            for _other, edges_b in single_items[index + 1 :]:
                cols = np.asarray(edges_b, dtype=np.int64)
                matrix[np.ix_(rows, cols)] += 1
                matrix[np.ix_(cols, rows)] += 1


def unique_ints(values) -> tuple[int, ...]:
    if not values:
        return ()
    ordered = sorted(set(values))
    return tuple(ordered)


def witness_matrices(
    edge_pmids: list[tuple[int, ...]],
    neighbor: dict[int, tuple[int, ...]],
    inside_flag,
    collect_article_pairs: bool,
    max_article_keys: int,
) -> dict:
    """Count third-party witnesses between every ordered pair of edges.

    ``neighbor`` is incoming for co-citation and outgoing for a shared reference.
    A witness is external when ``inside_flag`` is false for its PMID. Article-pair
    counts are distinct (p, q) pairs and are omitted, rather than approximated,
    when the group exceeds ``max_article_keys``.
    """
    import numpy as np

    size = len(edge_pmids)
    witnesses = np.zeros((size, size), dtype=np.int64)
    external = np.zeros((size, size), dtype=np.int64)
    pmid_edges: dict[int, list[int]] = {}
    for index, pmids in enumerate(edge_pmids):
        for pmid in pmids:
            pmid_edges.setdefault(pmid, []).append(index)
    postings: dict[int, list[tuple[int, int]]] = {}
    for pmid, edges in pmid_edges.items():
        for witness in neighbor.get(pmid, ()):
            witness = int(witness)
            if witness == pmid:
                continue
            bucket = postings.get(witness)
            if bucket is None:
                bucket = []
                postings[witness] = bucket
            for edge in edges:
                bucket.append((edge, pmid))
    article_sets: dict[tuple[int, int], set] | None = {} if collect_article_pairs else None
    article_status = "exact" if collect_article_pairs else "not_enumerated"
    article_members = 0
    limit = len(inside_flag) if inside_flag is not None else 0
    for witness, items in postings.items():
        by_edge: dict[int, set[int]] = {}
        for edge, pmid in items:
            by_edge.setdefault(edge, set()).add(pmid)
        if len(by_edge) < 2:
            continue
        distinct_papers = set()
        for papers in by_edge.values():
            distinct_papers.update(papers)
            if len(distinct_papers) > 1:
                break
        if len(distinct_papers) < 2:
            continue
        is_external = witness >= limit or not bool(inside_flag[witness])
        if article_sets is None:
            _fill_witness_blocks(by_edge, witnesses, external if is_external else None)
            continue
        edges = list(by_edge.items())
        hit = len(edges)
        for left_index in range(hit):
            left_edge, left_papers = edges[left_index]
            for right_index in range(hit):
                if left_index == right_index:
                    continue
                right_edge, right_papers = edges[right_index]
                if not _distinct_papers(left_papers, right_papers):
                    continue
                witnesses[left_edge, right_edge] += 1
                if is_external:
                    external[left_edge, right_edge] += 1
                key = (left_edge, right_edge)
                store = article_sets.get(key)
                if store is None:
                    store = set()
                    article_sets[key] = store
                for paper in left_papers:
                    for other in right_papers:
                        if paper != other:
                            store.add((paper, other))
                            article_members += 1
                if article_members > max_article_keys or len(article_sets) > max_article_keys:
                    article_sets = None
                    article_status = "omitted_cap"
    article_counts = None
    if article_sets is not None:
        article_counts = {key: len(value) for key, value in article_sets.items()}
    return {
        "witnesses": witnesses,
        "external": external,
        "article_counts": article_counts,
        "article_status": article_status,
    }


def pair_features(left: tuple[int, ...], right: tuple[int, ...], outgoing: dict, incoming: dict, support: set[int]) -> dict:
    direct = direct_citation_roles(left, right, outgoing)
    cocite = third_party_motifs(left, right, incoming, support)
    shared_ref = third_party_motifs(left, right, outgoing, support)
    return {
        "shared_pmids": intersection_size(left, right),
        **direct,
        "cocitation_witnesses": cocite["witnesses"],
        "cocitation_external_witnesses": cocite["external_witnesses"],
        "cocitation_article_pairs": cocite["article_pairs"],
        "shared_reference_witnesses": shared_ref["witnesses"],
        "shared_reference_external_witnesses": shared_ref["external_witnesses"],
        "shared_reference_article_pairs": shared_ref["article_pairs"],
    }

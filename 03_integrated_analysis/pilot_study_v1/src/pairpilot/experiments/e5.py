"""Search queries for a frozen candidate. Queries do not prescribe a direction."""

from __future__ import annotations


def queries_for(candidate: dict, aliases: list[str]) -> list[dict]:
    pair = candidate.get("pair_label") or candidate.get("pair_id")
    missing = candidate.get("missing_dimension") or "reported condition"
    question = candidate.get("search_question") or candidate.get("question_type") or "association"
    alias = " OR ".join(aliases) if aliases else pair
    return [
        {
            "query_id": "specific",
            "text": f"({alias}) AND ({missing})",
        },
        {
            "query_id": "broad",
            "text": f"({alias}) AND ({question})",
        },
    ]


def allocate_candidates(candidates: list[dict], hits_for, known: set[str] | None = None,
                        per_query: int = 5, queries_per_candidate: int = 2, global_new: int = 30) -> list[dict]:
    """Each candidate keeps its own query quota. An earlier candidate cannot absorb the rest."""
    known = set(known or [])
    selected = []
    used_new = 0
    for candidate in candidates:
        gained = 0
        cap = per_query * queries_per_candidate
        for query in queries_for(candidate, [])[:queries_per_candidate]:
            if gained >= cap or used_new >= global_new:
                selected.append({
                    "candidate_id": candidate.get("candidate_id"),
                    "query_id": query["query_id"],
                    "pmid": "",
                    "status": "not_started",
                    "reason": "candidate_or_global_cap",
                })
                continue
            room = min(per_query, cap - gained, global_new - used_new)
            hits = hits_for(candidate, query) if hits_for else []
            chosen = select_new(list(hits)[:50], known, room)
            new_rows = [row for row in chosen if not row.get("already_in_packet")]
            for row in chosen:
                item = dict(row)
                item["candidate_id"] = candidate.get("candidate_id")
                item["query_id"] = query["query_id"]
                item["status"] = "retrieved"
                selected.append(item)
                if not item.get("already_in_packet"):
                    known.add(str(item["pmid"]))
            gained += len(new_rows)
            used_new += len(new_rows)
    return selected


def select_new(hits: list[dict], known: set[str], limit: int) -> list[dict]:
    chosen = []
    for hit in hits:
        pmid = str(hit.get("pmid") or "")
        if not pmid:
            continue
        if pmid in known:
            hit = dict(hit)
            hit["already_in_packet"] = True
            chosen.append(hit)
            continue
        if sum(1 for row in chosen if not row.get("already_in_packet")) >= limit:
            continue
        hit = dict(hit)
        hit["already_in_packet"] = False
        chosen.append(hit)
    return chosen

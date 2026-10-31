"""Exact distribution summaries. Percentile method: nearest-rank, ceil."""

from __future__ import annotations

import math


def nearest_rank(sorted_vals: list[int | float], pct: float):
    if not sorted_vals:
        return None
    n = len(sorted_vals)
    rank = max(1, math.ceil(pct / 100.0 * n))
    rank = min(rank, n)
    return sorted_vals[rank - 1]


def summarize_numeric(values: list[int | float], population: str, unit: str) -> dict:
    if not values:
        return {
            "status": "not_applicable",
            "population": population,
            "unit": unit,
            "n": 0,
            "n_zero": None,
            "min": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
            "percentile_method": "nearest_rank_ceil",
        }
    ordered = sorted(values)
    n_zero = sum(1 for v in ordered if v == 0)
    return {
        "status": "computed",
        "population": population,
        "unit": unit,
        "n": len(ordered),
        "n_zero": n_zero,
        "min": ordered[0],
        "p25": nearest_rank(ordered, 25),
        "p50": nearest_rank(ordered, 50),
        "p75": nearest_rank(ordered, 75),
        "p90": nearest_rank(ordered, 90),
        "p95": nearest_rank(ordered, 95),
        "p99": nearest_rank(ordered, 99),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
        "percentile_method": "nearest_rank_ceil",
    }


def head_share(values: list[int | float], fraction: float) -> dict:
    """Share of the total weight held by the largest fraction of groups."""
    if not values or fraction <= 0:
        return {"status": "not_applicable", "fraction": fraction, "n_head": 0, "weight_share": None}
    ordered = sorted(values, reverse=True)
    n_head = max(1, math.ceil(fraction * len(ordered)))
    total = sum(ordered)
    if total == 0:
        return {
            "status": "not_applicable",
            "fraction": fraction,
            "n_groups": len(ordered),
            "n_head": n_head,
            "weight_numerator": 0,
            "weight_denominator": 0,
            "weight_share": None,
        }
    head = sum(ordered[:n_head])
    return {
        "status": "computed",
        "fraction": fraction,
        "n_groups": len(ordered),
        "n_head": n_head,
        "weight_numerator": head,
        "weight_denominator": total,
        "weight_share": head / total,
    }


def summarize_groups_sql(con, table: str, value_col: str, population: str, unit: str) -> dict:
    """Nearest-rank percentiles over a one-row-per-group table. Exact, not approximate."""
    n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    if n == 0:
        return summarize_numeric([], population, unit)
    sql = f"""
        WITH ranked AS (
          SELECT {value_col} AS n,
                 row_number() OVER (ORDER BY {value_col}) AS rn,
                 count(*) OVER () AS cnt
          FROM {table}
        ), picked AS (
          SELECT
            MAX(CASE WHEN rn = greatest(1, CAST(ceil(0.25 * cnt) AS BIGINT)) THEN n END) AS p25,
            MAX(CASE WHEN rn = greatest(1, CAST(ceil(0.50 * cnt) AS BIGINT)) THEN n END) AS p50,
            MAX(CASE WHEN rn = greatest(1, CAST(ceil(0.75 * cnt) AS BIGINT)) THEN n END) AS p75,
            MAX(CASE WHEN rn = greatest(1, CAST(ceil(0.90 * cnt) AS BIGINT)) THEN n END) AS p90,
            MAX(CASE WHEN rn = greatest(1, CAST(ceil(0.95 * cnt) AS BIGINT)) THEN n END) AS p95,
            MAX(CASE WHEN rn = greatest(1, CAST(ceil(0.99 * cnt) AS BIGINT)) THEN n END) AS p99
          FROM ranked
        )
        SELECT
          (SELECT COUNT(*) FROM {table}) AS n_groups,
          (SELECT COUNT(*) FROM {table} WHERE {value_col} = 0) AS n_zero,
          (SELECT MIN({value_col}) FROM {table}) AS min_n,
          (SELECT MAX({value_col}) FROM {table}) AS max_n,
          (SELECT AVG({value_col}) FROM {table}) AS mean_n,
          p25, p50, p75, p90, p95, p99
        FROM picked
    """
    row = con.execute(sql).fetchone()
    return {
        "status": "computed",
        "population": population,
        "unit": unit,
        "n": row[0],
        "n_zero": row[1],
        "min": row[2],
        "p25": row[5],
        "p50": row[6],
        "p75": row[7],
        "p90": row[8],
        "p95": row[9],
        "p99": row[10],
        "max": row[3],
        "mean": row[4],
        "percentile_method": "nearest_rank_ceil",
    }


def head_share_sql(con, table: str, value_col: str, fraction: float) -> dict:
    n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    total = con.execute(f"SELECT COALESCE(SUM({value_col}), 0) FROM {table}").fetchone()[0]
    if n == 0 or total == 0:
        return {"status": "not_applicable", "fraction": fraction, "n_groups": n, "weight_share": None, "weight_numerator": 0, "weight_denominator": total}
    head_n = max(1, math.ceil(fraction * n))
    head = con.execute(
        f"""
        SELECT COALESCE(SUM(n), 0) FROM (
          SELECT {value_col} AS n
          FROM {table}
          ORDER BY {value_col} DESC
          LIMIT {head_n}
        )
        """
    ).fetchone()[0]
    return {
        "status": "computed",
        "fraction": fraction,
        "n_groups": n,
        "n_head": head_n,
        "weight_numerator": head,
        "weight_denominator": total,
        "weight_share": head / total,
    }


def value_counts(values: list[str], limit: int = 50) -> dict:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        "status": "computed",
        "n": len(values),
        "n_distinct": len(counts),
        "top": [{"value": k, "n": v} for k, v in ranked[:limit]],
        "truncated": len(ranked) > limit,
    }

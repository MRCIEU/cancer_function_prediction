"""Atomic parquet/json writes. DuckDB opens inputs read-only."""

from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


def connect(path: str | None = None, threads: int = 1, memory: str = "8GB", tmp: Path | None = None):
    con = duckdb.connect(path or ":memory:")
    con.execute(f"SET threads={int(threads)}")
    con.execute(f"SET memory_limit='{memory}'")
    if tmp:
        tmp.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory='{tmp.as_posix()}'")
    con.execute("SET preserve_insertion_order=false")
    return con


def read_parquet(con, path: Path, view: str):
    con.execute(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM read_parquet('{path.as_posix()}')")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(partial, path)


def align_rows(rows: list[dict]) -> list[dict]:
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    return [{key: _cell(row.get(key)) for key in keys} for row in rows]


def _cell(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def query_rows(con, sql: str, params=None) -> list[dict]:
    cursor = con.execute(sql, params) if params is not None else con.execute(sql)
    names = [item[0] for item in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def write_parquet(path: Path, rows: list[dict], schema: pa.Schema | None = None) -> None:
    if not rows and schema is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + ".partial")
    aligned = align_rows(rows)
    table = pa.Table.from_pylist(aligned, schema=schema)
    pq.write_table(table, partial, compression="zstd")
    os.replace(partial, path)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def columns_of(con, path: Path) -> list[str]:
    rows = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path.as_posix()}')").fetchall()
    return [row[0] for row in rows]

"""DuckDB connections bounded by the Slurm allocation."""

from __future__ import annotations

from pathlib import Path

import duckdb

from lib.runtime import allocated_threads


def connect(path: Path, memory_limit: str, temp_dir: Path) -> duckdb.DuckDBPyConnection:
    temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    threads = allocated_threads(1)
    con.execute(f"SET threads={threads}")
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET temp_directory='{temp_dir.as_posix()}'")
    con.execute("SET preserve_insertion_order=false")
    try:
        con.execute("PRAGMA enable_progress_bar")
    except duckdb.Error:
        pass
    return con


def qident(name: str) -> str:
    if not name.replace("_", "").isalnum():
        raise ValueError(name)
    return '"' + name + '"'

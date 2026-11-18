"""Shared paths, hashing, and small IO helpers."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from pathlib import Path

import yaml

REPO_ROOT = Path("/home/b5ac/freddieyu.b5ac/project_2")
PROJECT_ROOT = Path("/home/b5ac/freddieyu.b5ac/projects/project_2")
SCRATCH_ROOT = Path("/lus/lfs1aip2/scratch/b5ac/freddieyu.b5ac/joint_graph_work")
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = PROJECT_ROOT / "results"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_text(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def peak_rss_bytes() -> int:
    # Linux ru_maxrss is kilobytes.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


class StageTimer:
    def __init__(self) -> None:
        self.started = time.time()
        self.stages: list[dict] = []

    def mark(self, name: str, **extra) -> None:
        now = time.time()
        row = {
            "stage": name,
            "unix_time": now,
            "elapsed_seconds": round(now - self.started, 3),
            "peak_rss_bytes": peak_rss_bytes(),
        }
        row.update(extra)
        self.stages.append(row)
        print(f"[{name}] elapsed_s={row['elapsed_seconds']} peak_rss={row['peak_rss_bytes']}", flush=True)


def normalise_pmid(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"\\N", "NULL", "null", "None"}:
        return None
    if not text.isdigit():
        return None
    return text.lstrip("0") or "0"


def normalise_gene_id(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"\\N", "NULL", "null", "None"}:
        return None
    if not text.isdigit():
        return None
    return text.lstrip("0") or "0"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_parquet_rows(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if rows:
        table = pa.Table.from_pylist(rows)
    else:
        names = columns or ["_empty"]
        table = pa.table({name: pa.array([], type=pa.string()) for name in names})
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def env_flag(name: str, default: str = "") -> str:
    return os.environ.get(name, default)

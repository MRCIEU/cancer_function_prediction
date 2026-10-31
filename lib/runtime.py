"""Job prologue, logging, and compute-node guard."""

from __future__ import annotations

import json
import logging
import os
import socket
import sys
import time
from pathlib import Path

CODE_ROOT = Path("/home/b5ac/freddieyu.b5ac/project_2")
CONFIG_PATH = CODE_ROOT / "config" / "analysis_config.json"


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def assert_compute_node() -> str:
    host = socket.gethostname()
    if host.startswith("login"):
        raise SystemExit(f"refusing to run analysis on login node host={host}")
    return host


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("analysis")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def file_fingerprint_meta(path: Path) -> dict:
    st = path.stat()
    return {
        "path": str(path),
        "bytes": st.st_size,
        "mtime_unix": int(st.st_mtime),
        "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(st.st_mtime)),
    }


def code_fingerprint(root: Path | None = None) -> str:
    import hashlib

    root = root or CODE_ROOT
    h = hashlib.sha256()
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__", ".pytest_cache"}]
        for name in filenames:
            if name.endswith((".py", ".json", ".sbatch", ".sh", ".txt")) and name != "requirements.txt":
                files.append(Path(dirpath) / name)
            elif name == "requirements.txt":
                files.append(Path(dirpath) / name)
    for path in sorted(files):
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def proc_rss_bytes() -> int | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb * 1024
    except OSError:
        return None
    return None


def prologue(job_name: str, logger: logging.Logger, output_dir: Path, inputs: list[str]) -> dict:
    host = assert_compute_node()
    cfg = load_config()
    info = {
        "job_name": job_name,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not_in_slurm"),
        "hostname": host,
        "started_unix": time.time(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "code_fingerprint": code_fingerprint(),
        "cpus_allocated": os.environ.get("SLURM_CPUS_PER_TASK"),
        "mem_allocated": os.environ.get("SLURM_MEM_PER_NODE") or os.environ.get("SLURM_MEM_PER_CPU"),
        "output_dir": str(output_dir),
        "inputs": inputs,
        "config": str(CONFIG_PATH),
    }
    logger.info(
        "START job=%s slurm_job_id=%s host=%s python=%s version=%s code=%s",
        job_name,
        info["slurm_job_id"],
        host,
        info["python"],
        info["python_version"],
        info["code_fingerprint"][:12],
    )
    logger.info("inputs=%s", inputs)
    logger.info("output_dir=%s cpus=%s mem=%s", output_dir, info["cpus_allocated"], info["mem_allocated"])
    logger.info("phase=prologue rss_bytes=%s", proc_rss_bytes())
    info["config_obj"] = cfg
    return info


class Progress:
    def __init__(self, logger: logging.Logger, phase: str, every_s: float = 45.0):
        self.logger = logger
        self.phase = phase
        self.every_s = every_s
        self.records = 0
        self.bytes = 0
        self.shards_done = 0
        self.shards_total = 0
        self.started = time.time()
        self.last = 0.0
        self.logger.info("phase_start phase=%s", phase)

    def set_phase(self, phase: str) -> None:
        self.phase = phase
        self.last = 0.0
        self.logger.info("phase_start phase=%s records=%s bytes=%s", phase, self.records, self.bytes)

    def update(self, records: int = 0, nbytes: int = 0, force: bool = False) -> None:
        self.records += records
        self.bytes += nbytes
        now = time.time()
        if force or (now - self.last) >= self.every_s:
            self.emit()

    def emit(self) -> None:
        now = time.time()
        elapsed = max(now - self.started, 1e-6)
        self.last = now
        self.logger.info(
            "progress phase=%s records=%s bytes=%s shards=%s/%s rec_per_s=%.1f rss_bytes=%s last_progress_utc=%s",
            self.phase,
            self.records,
            self.bytes,
            self.shards_done,
            self.shards_total,
            self.records / elapsed,
            proc_rss_bytes(),
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=str)
        fh.write("\n")
    tmp.replace(path)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def require_done(path: Path, job: str) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing completion marker for {job}: {path}")
    payload = read_json(path)
    if payload.get("status") != "success":
        raise SystemExit(f"upstream {job} status is not success: {path}")
    return payload


def disk_free(path: Path) -> int:
    import shutil

    return int(shutil.disk_usage(path).free)


def allocated_threads(default: int = 1) -> int:
    raw = os.environ.get("SLURM_CPUS_PER_TASK", "")
    try:
        n = int(raw)
    except ValueError:
        n = default
    return max(1, n)

"""Per-output checkpoints. A later job reads only success markers."""

from __future__ import annotations

import json
import time
from pathlib import Path


def marker_path(output: Path) -> Path:
    return output.with_name(output.name + ".ok.json")


def is_reusable(output: Path, code_fingerprint: str, input_signature: dict) -> bool:
    meta_path = marker_path(output)
    if not output.exists() or not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta.get("code_fingerprint") == code_fingerprint and meta.get("input_signature") == input_signature and meta.get("status") == "success"


def mark_success(output: Path, code_fingerprint: str, input_signature: dict, extra: dict | None = None) -> None:
    payload = {
        "status": "success",
        "output": str(output),
        "code_fingerprint": code_fingerprint,
        "input_signature": input_signature,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        payload.update(extra)
    marker_path(output).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def input_signature(path: Path) -> dict:
    st = path.stat()
    return {"path": str(path), "bytes": st.st_size, "mtime_unix": int(st.st_mtime)}

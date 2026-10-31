#!/usr/bin/env python3
"""Watch each submitted job for five minutes after it enters RUNNING."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

POINTER = Path("/projects/b5ac/project_2/results/03_integrated_analysis/pilot_study_v1/latest_submission.json")
OUT = Path("/projects/b5ac/project_2/results/03_integrated_analysis/pilot_study_v1/startup_checks.json")


def sacct(job_id: str) -> dict:
    result = subprocess.run(
        ["sacct", "-n", "-P", "-j", job_id, "--format=JobID,State,Elapsed,Start,End,ExitCode"],
        text=True, capture_output=True, check=False,
    )
    lines = [line for line in result.stdout.splitlines() if line and not line.startswith(f"{job_id}.")]
    if not lines:
        return {"state": "UNKNOWN"}
    job, state, elapsed, start, end, code = lines[0].split("|")
    return {"job_id": job, "state": state, "elapsed": elapsed, "start": start, "end": end, "exit_code": code}


def main() -> int:
    submitted = json.loads(POINTER.read_text(encoding="utf-8"))["submitted"]
    pending = {row["job_id"]: row for row in submitted}
    running_since = {}
    checks = {}
    deadline = time.time() + 60 * 40
    while pending and time.time() < deadline:
        for job_id, row in list(pending.items()):
            info = sacct(job_id)
            state = info.get("state", "UNKNOWN")
            if state.startswith("RUNNING") or state == "COMPLETED" or state.startswith("FAILED") or state.startswith("CANCELLED") or state.startswith("TIMEOUT"):
                running_since.setdefault(job_id, time.time())
                watched = time.time() - running_since[job_id]
                if state != "RUNNING" or watched >= 300:
                    checks[job_id] = {
                        "name": row["name"],
                        "stage": row["stage"],
                        "dependency": row.get("dependency"),
                        "state": state,
                        "watched_seconds": round(watched, 1),
                        "startup_window_completed": True,
                        "sacct": info,
                    }
                    del pending[job_id]
            else:
                checks.setdefault(job_id, {
                    "name": row["name"],
                    "stage": row["stage"],
                    "dependency": row.get("dependency"),
                    "state": state,
                    "startup_window_completed": False,
                    "note": "still pending; queue time is not a startup check",
                })
        if pending:
            time.sleep(30)
    OUT.write_text(json.dumps({"checks": checks, "still_pending": list(pending)}, indent=2), encoding="utf-8")
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

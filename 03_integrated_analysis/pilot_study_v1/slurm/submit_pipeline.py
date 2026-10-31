#!/usr/bin/env python3
"""Submit the pilot DAG. Login node only builds the plan and calls sbatch."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "src"))

from pairpilot.config import digest, load_bundle, new_run_id, redact, run_root  # noqa: E402
from pairpilot.store import read_json, write_json  # noqa: E402

SBATCH = PACKAGE / "slurm" / "run_stage.sbatch"
LOG_DIR = Path("/projects/b5ac/project_2/results/03_integrated_analysis/pilot_study_v1/logs")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    bundle = load_bundle()
    run_id = args.run_id or new_run_id(bundle)
    run = run_root(run_id)
    resources = bundle["resources"]["slurm"]
    budget_approved = bundle["resources"]["budget"].get("approved") is True and bundle["resources"]["budget"].get("max_api_calls")
    jobs = [
        {"name": "pilot-t0", "stage": "test", "dependency": None, "cpus": resources["test_tasks"], "mem": f"{resources['test_mem_gb']}G", "time": resources["test_time"], "gres": None},
        {"name": "pilot-gpu", "stage": "gpu-smoke", "dependency": None, "cpus": 4, "mem": f"{resources['gpu_smoke_mem_gb']}G", "time": resources["gpu_smoke_time"], "gres": resources["gpu_smoke_gres"]},
        {"name": "pilot-p0", "stage": "prepare", "dependency": "pilot-t0", "cpus": resources["cpu_tasks"], "mem": f"{resources['cpu_mem_gb']}G", "time": resources["cpu_time"], "gres": None},
        {"name": "pilot-p5", "stage": "report", "dependency": "pilot-p0", "dependency_kind": "afterany", "cpus": 2, "mem": "8G", "time": "00:30:00", "gres": None},
    ]
    if budget_approved:
        jobs.extend([
            {"name": "pilot-t1", "stage": "canary", "dependency": "pilot-p0", "cpus": 2, "mem": "8G", "time": "01:00:00", "gres": None},
            {"name": "pilot-p1", "stage": "extract", "dependency": "pilot-t1", "cpus": 2, "mem": "8G", "time": "04:00:00", "gres": None},
            {"name": "pilot-p2", "stage": "analyze", "dependency": "pilot-p1", "cpus": resources["cpu_tasks"], "mem": "32G", "time": "01:00:00", "gres": None},
            {"name": "pilot-p3", "stage": "paths", "dependency": "pilot-p0", "cpus": 2, "mem": "8G", "time": "00:20:00", "gres": None},
            {"name": "pilot-p4s", "stage": "search-check", "dependency": "pilot-p2", "cpus": 2, "mem": "8G", "time": "02:00:00", "gres": None},
            {"name": "pilot-p4r", "stage": "robustness", "dependency": "pilot-p2", "cpus": 2, "mem": "8G", "time": "02:00:00", "gres": None},
        ])
    plan = {
        "run_id": run_id,
        "config_hash": digest(redact(bundle)),
        "budget_approved": bool(budget_approved),
        "jobs": jobs,
        "not_submitted_reason": None if budget_approved else "model stages stay unsubmitted until budget.approved and max_api_calls are set",
    }
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return 0
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if run.exists() and not args.resume:
        raise SystemExit(f"run already exists: {run}")
    ids = {}
    submitted = []
    for job in jobs:
        cmd = [
            "sbatch",
            f"--job-name={job['name']}",
            f"--cpus-per-task={job['cpus']}",
            f"--mem={job['mem']}",
            f"--time={job['time']}",
            f"--export=ALL,PILOT_STAGE={job['stage']},PILOT_RUN_ID={run_id}" + (",PILOT_RESUME=1" if args.resume else ""),
        ]
        if job["gres"]:
            cmd.append(f"--gres={job['gres']}")
        if job["dependency"]:
            kind = job.get("dependency_kind") or "afterok"
            cmd.append(f"--dependency={kind}:{ids[job['dependency']]}")
        cmd.append(str(SBATCH))
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        job_id = result.stdout.strip().split()[-1]
        ids[job["name"]] = job_id
        submitted.append({**job, "job_id": job_id, "dependency_job": ids.get(job["dependency"] or "", None)})
    record = {"run_id": run_id, "submitted": submitted, "budget_approved": bool(budget_approved)}
    out = Path("/projects/b5ac/project_2/results/03_integrated_analysis/pilot_study_v1/submitted_jobs.json")
    write_json(out, record)
    # The run directory is created by the first compute stage. Keep a pointer beside it.
    pointer = Path("/projects/b5ac/project_2/results/03_integrated_analysis/pilot_study_v1/latest_submission.json")
    write_json(pointer, record)
    print(json.dumps(record, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

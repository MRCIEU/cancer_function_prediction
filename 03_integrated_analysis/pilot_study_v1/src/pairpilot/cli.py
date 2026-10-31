#!/usr/bin/env python3
"""Unified stage entry. Data stages refuse the login node."""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE / "src"))

from pairpilot.config import load_bundle, new_run_id, run_root  # noqa: E402
from pairpilot.stages import (  # noqa: E402
    stage_analyze,
    stage_canary,
    stage_extract,
    stage_paths,
    stage_preflight,
    stage_prepare,
    stage_report,
    stage_robustness,
    stage_search,
)
STAGES = {
    "preflight": stage_preflight,
    "prepare": stage_prepare,
    "canary": stage_canary,
    "extract": stage_extract,
    "analyze": stage_analyze,
    "paths": stage_paths,
    "search-check": stage_search,
    "robustness": stage_robustness,
    "report": stage_report,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["test", *STAGES])
    parser.add_argument("--config", default=str(PACKAGE / "config"))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    if _login() and not args.dry_run:
        raise SystemExit("refusing to run analysis or tests on the login node")
    if args.stage == "test":
        return _tests()
    bundle = load_bundle(Path(args.config))
    run_id = args.run_id or new_run_id(bundle)
    if args.dry_run:
        print({"stage": args.stage, "run_id": run_id, "run_root": str(run_root(run_id)), "resume": args.resume})
        return 0
    try:
        result = STAGES[args.stage](run_id)
    except Exception:
        traceback.print_exc()
        return 1
    print({"stage": args.stage, "run_id": run_id, **{key: result[key] for key in result if key != "run"}})
    return 0 if result.get("ok") or result.get("skipped") or result.get("blocked") else 2


def _login() -> bool:
    return os.uname().nodename.startswith("login")


def _tests() -> int:
    import unittest
    suite = unittest.defaultTestLoader.discover(str(PACKAGE / "tests"), pattern="test_*.py")
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if outcome.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())

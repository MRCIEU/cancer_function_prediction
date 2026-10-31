#!/usr/bin/env python3
"""J05: cases, figures, and the integrated report. Does not rescan raw tables."""

from __future__ import annotations

import os
import sys
from pathlib import Path

CODE_ROOT = Path("/home/b5ac/freddieyu.b5ac/project_2")
sys.path.insert(0, str(CODE_ROOT))

from lib.duckutil import connect  # noqa: E402
from lib.report_build import export_cases, make_figures, write_integrated_report  # noqa: E402
from lib.runtime import load_config, prologue, read_json, require_done, setup_logging, write_json  # noqa: E402


def main() -> int:
    cfg = load_config()
    results = Path(cfg["paths"]["results_root"])
    out = results / "03_integrated_analysis" / "data_analysis_round1"
    logger = setup_logging(out / "logs" / f"j05_{os.environ.get('SLURM_JOB_ID', 'local')}.log")
    info = prologue("J05", logger, out, [])
    j01 = require_done(results / "01_core_audit" / "DONE.json", "J01")
    j02 = require_done(results / "02_semmeddb_validation" / "DONE.json", "J02")
    j03 = require_done(out / "derived" / "j03" / "DONE.json", "J03")
    j04 = require_done(out / "tables" / "DONE.json", "J04")
    report = out / "report.md"
    if report.is_file() and (out / "run_manifest.json").is_file():
        raise SystemExit("refusing to overwrite existing integrated report")
    core = read_json(Path(j01["summary"]))
    semmed = read_json(Path(j02["summary"]))
    align = read_json(Path(j03["summary"]))
    joint = read_json(Path(j04["summary"]))
    write_integrated_report(core, semmed, align, joint, report)
    figures = make_figures(core, semmed, align, joint, out / "figures")
    con = connect(out / "derived" / "j05.duckdb", "16GB", out / "derived" / "j05_tmp")
    views = {
        "rel_all": out / "tables" / "pair_relations.parquet",
        "pair": j01["tables"]["pair"],
        "gcs_status": j01["tables"]["gcs_status"],
        "core_pair_pmid": j01["tables"]["pair_pmid"],
        "semmed_pair_pmid": j03["outputs"]["semmed_pair_pmid"],
        "core_sentence": j01["tables"]["sentence"],
        "matched_predication": j03["outputs"]["matched_predication"],
    }
    for name, path in views.items():
        if not Path(path).is_file():
            raise SystemExit(f"missing case input {name}: {path}")
        con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{Path(path).as_posix()}')")
    cases = export_cases(con, str(cfg["case_seed"]), int(cfg["case_target_n"]), out / "cases", logger)
    manifest = {
        "status": "success",
        "job": "J05",
        "code_fingerprint": info["code_fingerprint"],
        "inputs": {
            "J01": j01.get("input_blake2b"),
            "J02": j02.get("input_blake2b"),
        },
        "report": str(report),
        "figures": figures,
        "cases": cases["files"],
        "not_done": [],
    }
    write_json(out / "run_manifest.json", manifest)
    logger.info("phase=j05_success figures=%s cases=%s", len(figures), len(cases["files"]))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        raise SystemExit(1)

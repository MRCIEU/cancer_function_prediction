#!/usr/bin/env python3
"""J04: pair relations, PMID coverage, sensitivity, and observable states."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

CODE_ROOT = Path("/home/b5ac/freddieyu.b5ac/project_2")
sys.path.insert(0, str(CODE_ROOT))

from lib.duckutil import connect  # noqa: E402
from lib.joint_build import run_joint  # noqa: E402
from lib.runtime import load_config, prologue, require_done, setup_logging, write_json  # noqa: E402


def main() -> int:
    cfg = load_config()
    results = Path(cfg["paths"]["results_root"])
    out = results / "03_integrated_analysis" / "data_analysis_round1"
    logger = setup_logging(out / "logs" / f"j04_{os.environ.get('SLURM_JOB_ID', 'local')}.log")
    info = prologue("J04", logger, out / "tables", [])
    j01 = require_done(results / "01_core_audit" / "DONE.json", "J01")
    j02 = require_done(results / "02_semmeddb_validation" / "DONE.json", "J02")
    j03 = require_done(out / "derived" / "j03" / "DONE.json", "J03")
    if (out / "tables" / "DONE.json").is_file():
        raise SystemExit("refusing to overwrite existing J04 results")
    paths = {
        "pair": j01["tables"]["pair"],
        "core_pair_pmid": j01["tables"]["pair_pmid"],
        "core_sentence": j01["tables"]["sentence"],
        "semmed_pair_pmid": j03["outputs"]["semmed_pair_pmid"],
        "sem_sentence": j02["tables"]["sentence"],
        "citations": j02["tables"]["citations"],
        "entity": j02["tables"]["entity"],
        "pmid_route": j03["outputs"]["pmid_route"],
        "pmid_year": str(out / "derived" / "j03" / "pmid_year.parquet"),
        "scope_e_gene_pairs": j03["outputs"]["scope_e_gene_pairs"],
        "matched_predication": j03["outputs"]["matched_predication"],
    }
    for key, path in paths.items():
        if not Path(path).is_file():
            raise SystemExit(f"missing input {key}: {path}")
    j03_summary = json.loads(Path(j03["summary"]).read_text(encoding="utf-8"))
    resolution = j03_summary.get("entity_sentence_id_resolution") or {}
    if resolution.get("status") != "resolved" or not resolution.get("column"):
        raise SystemExit(f"entity sentence_id is unresolved: {resolution}")
    tables = out / "tables"
    con = connect(tables / "joint.duckdb", cfg["duckdb"]["memory_limit"], out / "derived" / "j04_tmp")
    summary = run_joint(con, paths, tables, logger, cfg["id_rule_version"], resolution)
    summary["code_fingerprint"] = info["code_fingerprint"]
    write_json(tables / "summary.json", summary)
    write_json(tables / "DONE.json", {"status": "success", "job": "J04", "summary": str(tables / "summary.json"), "code_fingerprint": info["code_fingerprint"]})
    logger.info("phase=j04_success")
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

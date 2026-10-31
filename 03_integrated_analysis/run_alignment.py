#!/usr/bin/env python3
"""J03: explicit ID alignment and target-record extraction."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

CODE_ROOT = Path("/home/b5ac/freddieyu.b5ac/project_2")
sys.path.insert(0, str(CODE_ROOT))

from lib.align_build import run_alignment  # noqa: E402
from lib.duckutil import connect  # noqa: E402
from lib.runtime import load_config, prologue, require_done, setup_logging, write_json  # noqa: E402
from lib.semmed_build import resolve_entity_sentence_id  # noqa: E402


def main() -> int:
    cfg = load_config()
    results = Path(cfg["paths"]["results_root"])
    out = results / "03_integrated_analysis" / "data_analysis_round1"
    logger = setup_logging(out / "logs" / f"j03_{os.environ.get('SLURM_JOB_ID', 'local')}.log")
    info = prologue("J03", logger, out / "derived", [])
    j01 = require_done(results / "01_core_audit" / "DONE.json", "J01")
    j02 = require_done(results / "02_semmeddb_validation" / "DONE.json", "J02")
    if (out / "derived" / "j03" / "DONE.json").is_file():
        raise SystemExit("refusing to overwrite existing J03 results")
    paths = {
        "core_pair": j01["tables"]["pair"],
        "core_pair_pmid": j01["tables"]["pair_pmid"],
        "core_sentence": j01["tables"]["sentence"],
        "core_article_literal": j01["tables"]["article_literal"],
        "entity": j02["tables"]["entity"],
        "predication": j02["tables"]["predication"],
        "sem_sentence": j02["tables"]["sentence"],
        "citations": j02["tables"]["citations"],
        "scope_e_gngm_neop": j02["scope_e"]["gngm_neop"],
    }
    for key, path in paths.items():
        if not Path(path).is_file():
            raise SystemExit(f"missing input {key}: {path}")
    j02_summary = json.loads(Path(j02["summary"]).read_text(encoding="utf-8"))
    stored = j02_summary.get("entity_sentence_id_resolution") or {}
    derived = out / "derived" / "j03"
    con = connect(derived / "align.duckdb", cfg["duckdb"]["memory_limit"], derived / "tmp")
    con.execute(f"CREATE OR REPLACE VIEW entity AS SELECT * FROM read_parquet('{Path(paths['entity']).as_posix()}')")
    con.execute(f"CREATE OR REPLACE VIEW sentence AS SELECT sentence_id FROM read_parquet('{Path(paths['sem_sentence']).as_posix()}')")
    resolution = resolve_entity_sentence_id(con, Path(paths["entity"]), logger)
    resolution["j02_stored_resolution"] = stored
    if resolution.get("status") != "resolved" or not resolution.get("column"):
        raise SystemExit(f"entity sentence_id is unresolved; pair and PMID joins that need it are blocked: {resolution}")
    summary = run_alignment(con, paths, derived, logger, cfg["id_rule_version"], resolution)
    summary["entity_sentence_id_resolution"] = resolution
    summary["code_fingerprint"] = info["code_fingerprint"]
    summary["upstream"] = {"J01": j01.get("input_blake2b"), "J02": j02.get("input_blake2b")}
    write_json(derived / "summary.json", summary)
    write_json(derived / "DONE.json", {"status": "success", "job": "J03", "summary": str(derived / "summary.json"), "outputs": summary["outputs"], "code_fingerprint": info["code_fingerprint"]})
    logger.info("phase=j03_success")
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

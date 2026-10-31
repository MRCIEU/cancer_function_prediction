#!/usr/bin/env python3
"""J01: full CORE parse, integrity, and descriptive statistics."""

from __future__ import annotations

import os
import sys
from pathlib import Path

CODE_ROOT = Path("/home/b5ac/freddieyu.b5ac/project_2")
sys.path.insert(0, str(CODE_ROOT))

from lib.core_build import build_core_tables, open_triple_writer, write_triple_batch  # noqa: E402
from lib.duckutil import connect  # noqa: E402
from lib.fingerprint import blake2_file  # noqa: E402
from lib.rdf_stream import stream_to_callback  # noqa: E402
from lib.report_build import write_core_report  # noqa: E402
from lib.runtime import load_config, prologue, require_done, setup_logging, write_json  # noqa: E402


def main() -> int:
    cfg = load_config()
    root = Path(cfg["paths"]["results_root"]) / "01_core_audit"
    logger = setup_logging(root / "logs" / f"j01_{os.environ.get('SLURM_JOB_ID', 'local')}.log")
    info = prologue("J01", logger, root, [cfg["paths"]["core_ttl"], cfg["paths"]["core_owl"]])
    j00 = require_done(Path(cfg["paths"]["results_root"]) / "03_integrated_analysis" / "data_analysis_round1" / "derived" / "j00" / "DONE.json", "J00")
    confirmed = __import__("json").loads(Path(j00["confirmed_schema"]).read_text(encoding="utf-8"))
    if confirmed.get("status") != "confirmed":
        raise SystemExit("J00 schema is not confirmed")
    if (root / "DONE.json").is_file():
        raise SystemExit(f"refusing to overwrite existing CORE results: {root / 'DONE.json'}")
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    ttl = Path(cfg["paths"]["core_ttl"])
    from lib.runtime import Progress

    hash_progress = Progress(logger, "core_blake2b")
    progress_fp = blake2_file(ttl, hash_progress)
    logger.info("phase=core_blake2b blake2b=%s", progress_fp)
    triple_path = tables / "triples.parquet"
    writer = open_triple_writer(triple_path)
    progress = Progress(logger, "core_rdf_parse")

    def callback(batch):
        write_triple_batch(writer, batch)
        progress.update(records=len(batch), force=False)

    try:
        state = stream_to_callback(ttl, callback, batch_size=8000)
    finally:
        writer.close()
    progress.update(force=True)
    logger.info("phase=core_rdf_done records=%s format=%s", state["n"], state["format"])
    if state["n"] == 0:
        raise SystemExit("CORE parse produced zero triples")
    con = connect(tables / "core.duckdb", "40GB", root / "tmp")
    summary = build_core_tables(con, triple_path, tables, logger)
    summary["input_blake2b"] = progress_fp
    summary["rdf_format"] = state["format"]
    summary["code_fingerprint"] = info["code_fingerprint"]
    write_json(tables / "summary.json", summary)
    write_core_report(summary, root / "report" / "core_audit.md")
    write_json(
        root / "DONE.json",
        {
            "status": "success",
            "job": "J01",
            "code_fingerprint": info["code_fingerprint"],
            "input_blake2b": progress_fp,
            "summary": str(tables / "summary.json"),
            "tables": summary["outputs"],
            "duckdb": str(tables / "core.duckdb"),
        },
    )
    logger.info("phase=j01_success triples=%s pairs=%s", state["n"], summary.get("n_explicit_pairs"))
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

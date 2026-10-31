#!/usr/bin/env python3
"""J02: SemMedDB load, integrity, and descriptive statistics."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

CODE_ROOT = Path("/home/b5ac/freddieyu.b5ac/project_2")
sys.path.insert(0, str(CODE_ROOT))

from lib.duckutil import connect  # noqa: E402
from lib.report_build import write_semmed_report  # noqa: E402
from lib.runtime import Progress, load_config, prologue, require_done, setup_logging, write_json  # noqa: E402
from lib.csv_stream import repair_utf8_file  # noqa: E402
from lib.schema import semmed_path  # noqa: E402
from lib.semmed_build import citations_counts_match, compare_prefix, load_table, profile  # noqa: E402


def table_encoding(name: str, spec: dict) -> str:
    if spec.get("encoding"):
        return spec["encoding"]
    # PREDICATION contained a non-UTF-8 byte. SENTENCE completed as UTF-8.
    # AUX completed as latin-1. ENTITY was rejected as latin-1 after tens of GB, so it stays UTF-8.
    if name in {"PREDICATION", "PREDICATION_AUX"}:
        return "latin-1"
    return "utf-8"


def table_dialect(name: str, spec: dict) -> str:
    # A 2000-row sample of AUX was stable under quote doubling, then a later
    # \" inside a mention made that reader fail. Backslash is the escape actually used.
    if name in {"PREDICATION", "PREDICATION_AUX", "ENTITY"}:
        return "backslash_escape"
    return spec["dialect"]


def reuse_or_load(con, name, path, dialect, encoding, max_line_size, out, logger, parallel, column_map, precomputed_blake2b: str | None = None) -> dict:
    marker = out.with_suffix(out.suffix + ".ok.json")
    if marker.is_file() and out.is_file() and out.stat().st_size > 0:
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous.get("dialect") == dialect and previous.get("encoding") == encoding and previous.get("status") == "success":
            logger.info("phase=reuse_load table=%s records=%s marker=%s", name, previous.get("rows"), marker)
            return {"rows": previous.get("rows"), "blake2b": previous.get("blake2b"), "output": str(out), "reused": True}
    loaded = load_table(
        con, name, path, dialect, max_line_size, out, logger, parallel, column_map, encoding, precomputed_blake2b=precomputed_blake2b
    )
    write_json(
        marker,
        {
            "status": "success",
            "table": name,
            "rows": loaded["rows"],
            "blake2b": loaded["blake2b"],
            "dialect": dialect,
            "encoding": encoding,
            "output": str(out),
        },
    )
    loaded["reused"] = False
    return loaded


def entity_utf8_source(src: Path, dest: Path, logger) -> dict:
    """ENTITY is mostly UTF-8, with bytes DuckDB's UTF-8 and Latin-1 readers both reject.

    Job 7422261 stopped at physical line 1370276539 (not UTF-8). Job 7421875 rejected the
    same file as Latin-1 because UTF-8 continuation bytes fall in 0x80-0x9F. Repair keeps
    every byte: valid UTF-8 stays, and each illegal byte becomes one Windows-1252 character.
    """
    marker = dest.with_name(dest.name + ".repair.json")
    source_bytes = src.stat().st_size
    if marker.is_file() and dest.is_file() and dest.stat().st_size > 0:
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if (
            previous.get("status") == "success"
            and previous.get("source_bytes") == source_bytes
            and previous.get("output_bytes") == dest.stat().st_size
            and previous.get("source") == str(src)
        ):
            logger.info(
                "phase=reuse_repaired_entity output=%s replaced_bytes=%s",
                dest,
                previous.get("replaced_bytes"),
            )
            return previous
    stats = repair_utf8_file(src, dest, logger)
    write_json(marker, stats)
    return stats


VIEW = {
    "CITATIONS": "citations",
    "GENERIC_CONCEPT": "generic_concept",
    "SENTENCE": "sentence",
    "PREDICATION": "predication",
    "PREDICATION_AUX": "predication_aux",
    "ENTITY": "entity",
}


def main() -> int:
    cfg = load_config()
    root = Path(cfg["paths"]["results_root"]) / "02_semmeddb_validation"
    logger = setup_logging(root / "logs" / f"j02_{os.environ.get('SLURM_JOB_ID', 'local')}.log")
    info = prologue("J02", logger, root, [cfg["paths"]["semmed_dir"]])
    j00 = require_done(Path(cfg["paths"]["results_root"]) / "03_integrated_analysis" / "data_analysis_round1" / "derived" / "j00" / "DONE.json", "J00")
    confirmed = json.loads(Path(j00["confirmed_schema"]).read_text(encoding="utf-8"))
    if confirmed.get("status") != "confirmed":
        raise SystemExit("J00 schema is not confirmed")
    if (root / "DONE.json").is_file():
        raise SystemExit(f"refusing to overwrite existing SemMedDB results: {root / 'DONE.json'}")
    tables = root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    con = connect(tables / "semmed.duckdb", cfg["duckdb"]["memory_limit"], root / "tmp")
    progress = Progress(logger, "semmed_certify")
    outputs = {}
    fingerprints = {}
    entity_repair = None
    parallel = bool(confirmed.get("duckdb_parallel"))
    # Certification uses the helper's own parallel flag for the full citations count.
    cit_spec = confirmed["tables"]["CITATIONS"]
    cit_path = semmed_path(cfg["paths"]["semmed_dir"], "CITATIONS")
    prefix = compare_prefix(con, cit_path, 5, cit_spec["dialect"], cfg["duckdb"]["max_line_size"], 1000)
    logger.info("phase=citations_prefix equal=%s mismatch=%s", prefix["equal"], prefix["mismatch_index"])
    if not prefix["equal"]:
        raise SystemExit(f"citations prefix mismatch: {prefix}")
    counts = citations_counts_match(con, cit_path, cit_spec["dialect"], cfg["duckdb"]["max_line_size"], progress, parallel)
    logger.info("phase=citations_count_compare %s", counts)
    if not counts["equal"]:
        raise SystemExit(f"DuckDB and Python citation counts differ: {counts}")
    for name in ("GENERIC_CONCEPT", "CITATIONS", "SENTENCE", "PREDICATION", "PREDICATION_AUX", "ENTITY"):
        spec = confirmed["tables"][name]
        path = semmed_path(cfg["paths"]["semmed_dir"], name)
        encoding = table_encoding(name, spec)
        dialect = table_dialect(name, spec)
        if name != "CITATIONS":
            got = compare_prefix(con, path, int(spec["local_columns"]["observed_width"]), dialect, cfg["duckdb"]["max_line_size"], 1000, encoding)
            logger.info("phase=prefix table=%s equal=%s mismatch=%s", name, got["equal"], got["mismatch_index"])
            if not got["equal"]:
                raise SystemExit(f"prefix mismatch for {name}: {got}")
        out = tables / f"{VIEW[name]}.parquet"
        read_path = path
        source_blake = None
        if name == "ENTITY":
            repaired = root / "tmp" / "entity.utf8.csv"
            entity_repair = entity_utf8_source(path, repaired, logger)
            read_path = repaired
            source_blake = entity_repair["source_blake2b"]
        loaded = reuse_or_load(
            con, name, read_path, dialect, encoding, cfg["duckdb"]["max_line_size"], out, logger, parallel, spec["local_columns"], source_blake
        )
        outputs[VIEW[name]] = str(out)
        fingerprints[name] = loaded["blake2b"]
    paths = {VIEW[name]: tables / f"{VIEW[name]}.parquet" for name in VIEW}
    summary = profile(con, paths, tables, logger)
    summary["local_column_maps"] = {
        name: {
            "status": spec.get("local_columns", {}).get("status"),
            "observed_width": spec.get("local_columns", {}).get("observed_width"),
            "documented_width": spec.get("local_columns", {}).get("documented_width"),
            "width_matches_documented": spec.get("local_columns", {}).get("width_matches_documented"),
            "index": spec.get("local_columns", {}).get("index"),
            "unconfirmed_indexes": spec.get("local_columns", {}).get("unconfirmed_indexes"),
            "evidence": spec.get("local_columns", {}).get("evidence"),
            "dialect": table_dialect(name, spec),
            "encoding": table_encoding(name, spec),
        }
        for name, spec in confirmed["tables"].items()
    }
    summary["input_blake2b"] = fingerprints
    summary["entity_utf8_repair"] = entity_repair
    summary["citation_python_duckdb_counts"] = counts
    summary["code_fingerprint"] = info["code_fingerprint"]
    summary["official_row_counts_are_reference_only"] = cfg["reference_counts_not_targets"]
    write_json(tables / "summary.json", summary)
    write_semmed_report(summary, root / "report" / "semmed_audit.md")
    write_json(
        root / "DONE.json",
        {
            "status": "success",
            "job": "J02",
            "code_fingerprint": info["code_fingerprint"],
            "summary": str(tables / "summary.json"),
            "tables": outputs,
            "scope_e": summary["scope_e"]["outputs"],
            "input_blake2b": fingerprints,
        },
    )
    logger.info("phase=j02_success")
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

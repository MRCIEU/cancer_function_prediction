#!/usr/bin/env python3
"""J00: environment, syntax, unit tests, and bounded parser checks."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

CODE_ROOT = Path("/home/b5ac/freddieyu.b5ac/project_2")
sys.path.insert(0, str(CODE_ROOT))

from lib.runtime import (  # noqa: E402
    disk_free,
    load_config,
    prologue,
    setup_logging,
    write_json,
)
from lib.schema import SEMMED_TABLES, semmed_path  # noqa: E402


def syntax_check(root: Path) -> list[str]:
    errors = []
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except SyntaxError as exc:
            errors.append(f"{path}:{exc.lineno}:{exc.msg}")
    return errors


def ensure_venv(cfg: dict, logger) -> None:
    py = Path(cfg["paths"]["venv_python"])
    if py.exists():
        probe = subprocess.run([str(py), "-c", "import rdflib, duckdb, matplotlib, pyarrow"], capture_output=True, text=True)
        if probe.returncode == 0:
            logger.info("phase=venv_reuse python=%s", py)
            return
    prefix = Path(cfg["paths"]["venv_prefix"])
    cache = Path(cfg["paths"]["pip_cache"])
    cache.mkdir(parents=True, exist_ok=True)
    base = cfg["paths"]["base_python"]
    logger.info("phase_start phase=venv_create prefix=%s", prefix)
    subprocess.run([base, "-m", "venv", str(prefix)], check=True)
    env = os.environ.copy()
    env["PIP_CACHE_DIR"] = str(cache)
    subprocess.run([str(py), "-m", "pip", "install", "--upgrade", "pip"], check=True, env=env)
    subprocess.run([str(py), "-m", "pip", "install", "-r", str(CODE_ROOT / "requirements.txt")], check=True, env=env)
    logger.info("phase=venv_ready")


def main() -> int:
    cfg = load_config()
    log_dir = Path(cfg["paths"]["results_root"]) / "03_integrated_analysis" / "data_analysis_round1" / "logs"
    logger = setup_logging(log_dir / f"j00_{os.environ.get('SLURM_JOB_ID', 'local')}.log")
    info = prologue("J00", logger, log_dir, [cfg["paths"]["raw_data_root"]])
    venv_py = Path(cfg["paths"]["venv_python"])
    if "--bootstrapped" not in sys.argv:
        ensure_venv(cfg, logger)
        os.execv(str(venv_py), [str(venv_py), str(Path(__file__).resolve()), "--bootstrapped"])
    syntax_errors = syntax_check(CODE_ROOT)
    if syntax_errors:
        logger.error("phase=syntax_failed %s", syntax_errors)
        return 1
    logger.info("phase=syntax_ok")
    suite = unittest.defaultTestLoader.discover(str(CODE_ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        logger.error("phase=unit_tests_failed failures=%s errors=%s", len(result.failures), len(result.errors))
        return 1
    logger.info("phase=unit_tests_ok")
    free = disk_free(Path(cfg["paths"]["project_root"]))
    logger.info("phase=disk_free bytes=%s", free)
    if free < cfg["thresholds"]["min_free_bytes"]:
        logger.error("phase=disk_below_threshold")
        return 1
    confirmed = check_inputs(cfg, logger)
    out = Path(cfg["paths"]["results_root"]) / "03_integrated_analysis" / "data_analysis_round1" / "derived" / "j00"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "confirmed_schema.json", confirmed)
    write_json(out / "environment.json", {k: v for k, v in info.items() if k != "config_obj"})
    if confirmed["status"] != "confirmed":
        write_json(out / "DONE.json", {"status": "failed", "job": "J00", "blockers": confirmed["blockers"]})
        logger.error("phase=schema_blocked blockers=%s", confirmed["blockers"])
        return 2
    write_json(
        out / "DONE.json",
        {
            "status": "success",
            "job": "J00",
            "confirmed_schema": str(out / "confirmed_schema.json"),
            "code_fingerprint": info["code_fingerprint"],
        },
    )
    logger.info("phase=j00_success")
    return 0


def check_inputs(cfg: dict, logger) -> dict:
    from lib.csv_stream import choose_dialect, compression_magic
    from lib.rdf_stream import detect_rdf_format, iter_triples
    import duckdb
    from lib.csv_stream import read_csv_call

    blockers = []
    tables = {}
    limits = cfg["preflight"]
    if limits["max_records"] > 5000 or limits["max_bytes"] > 64 * 1024 * 1024 or limits["max_triples"] > 5000:
        blockers.append("preflight_limits_exceed_safety_cap")
    sem_dir = cfg["paths"]["semmed_dir"]
    con = duckdb.connect()
    for name, spec in SEMMED_TABLES.items():
        path = semmed_path(sem_dir, name)
        if not path.is_file():
            blockers.append(f"missing:{path}")
            continue
        logger.info("phase=sample_table table=%s bytes=%s", name, path.stat().st_size)
        chosen = choose_dialect(path, len(spec["columns"]), limits["max_records"], limits["max_bytes"])
        sample_rows = chosen.pop("chosen_rows", [])
        from lib.semmed_columns import assign_local_columns

        assignment = assign_local_columns(name, sample_rows) if sample_rows else {"status": "empty_sample"}
        entry = {
            "path": str(path),
            "expected_columns": spec["columns"],
            "expected_width": len(spec["columns"]),
            "compression": compression_magic(path),
            "dialect_selection": chosen,
            "local_columns": assignment,
            "bytes": path.stat().st_size,
        }
        if chosen["chosen"] is None:
            blockers.append(f"dialect_unconfirmed:{name}")
        elif assignment.get("status") != "assigned":
            blockers.append(f"columns_unassigned:{name}:{assignment.get('status')}")
        else:
            dialect_name = chosen["chosen"]
            call = read_csv_call(str(path), int(assignment["observed_width"]), dialect_name, cfg["duckdb"]["max_line_size"], parallel=False)
            try:
                duck_rows = con.execute(f"SELECT * FROM {call} LIMIT {limits['prefix_compare_records']}").fetchall()
            except Exception as exc:
                blockers.append(f"duckdb_sample_failed:{name}:{exc}")
                duck_rows = []
            py_preview = chosen["candidates"][0]["preview"] if chosen["candidates"] else []
            first = py_preview[0][0] if py_preview and py_preview[0] else ""
            if first == spec["columns"][0]:
                blockers.append(f"header_like_first_row:{name}")
            entry["duckdb_sample_rows"] = len(duck_rows)
            entry["dialect"] = dialect_name
            if not duck_rows:
                blockers.append(f"empty_sample:{name}")
        tables[name] = entry
        logger.info("phase=sample_table_done table=%s chosen=%s", name, chosen.get("chosen"))
    ttl = Path(cfg["paths"]["core_ttl"])
    owl = Path(cfg["paths"]["core_owl"])
    rdf = {"format": None, "sample_triples": 0, "predicates": {}}
    if not ttl.is_file() or not owl.is_file():
        blockers.append("missing_core_file")
    else:
        rdf["format"] = detect_rdf_format(ttl)
        preds = {}
        n = 0
        for row in iter_triples(ttl, limit=limits["max_triples"], rdf_format=rdf["format"]):
            n += 1
            preds[row[1]] = preds.get(row[1], 0) + 1
            if n >= limits["max_triples"]:
                break
        rdf["sample_triples"] = n
        rdf["predicates"] = preds
        if n == 0:
            blockers.append("rdf_sample_empty")
        logger.info("phase=rdf_sample triples=%s format=%s", n, rdf["format"])
    parallel_ok = multiline_parallel_ok(con, cfg)
    if not parallel_ok:
        logger.info("phase=duckdb_parallel_not_certified_for_multiline")
    return {
        "status": "confirmed" if not blockers else "blocked",
        "blockers": blockers,
        "tables": tables,
        "rdf": rdf,
        "duckdb_parallel": parallel_ok,
        "duckdb_version": duckdb.__version__,
    }


def multiline_parallel_ok(con, cfg) -> bool:
    import tempfile
    from lib.csv_stream import read_csv_call

    text = 'a,"b\nb",c\n"d""e",f,g\n'
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "multi.csv"
        path.write_text(text, encoding="utf-8")
        call = read_csv_call(str(path), 3, "quote_doubling", cfg["duckdb"]["max_line_size"], parallel=True)
        try:
            rows = con.execute(f"SELECT * FROM {call}").fetchall()
        except Exception:
            return False
    return rows == [("a", "b\nb", "c"), ('d"e', "f", "g")]


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        raise SystemExit(1)

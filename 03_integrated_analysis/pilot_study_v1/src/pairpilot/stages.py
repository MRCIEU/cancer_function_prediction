"""Stage implementations. Real tables are read on a compute node."""

from __future__ import annotations

import os
import signal
import traceback
from datetime import datetime, timezone
from pathlib import Path

from pairpilot.comparison import build_candidates, filter_comparisons, gap_from_comparison
from pairpilot.config import PACKAGE, budget_state, digest, load_bundle, prompt_hashes, redact, run_root
from pairpilot.contracts import exact_span, locate_quote, metric_row
from pairpilot.experiments.e1 import article_sets, equal_article, source_metrics, view_stats
from pairpilot.experiments.e2 import direct_payload, extract_payload, knowledge_payload, output_delta
from pairpilot.experiments.e3 import group_source_text, scope_change
from pairpilot.experiments.e4 import positive_paths, rank_paths, topk_overlap
from pairpilot.experiments.e5 import queries_for, select_new
from pairpilot.experiments.e6 import align_replicates, control_deletion, delete_span
from pairpilot.inference import CallCache, OpenAIAdapter, SyntheticAdapter, UsageLedger, input_hash, run_request
from pairpilot.sampling import choose_ids, choose_pmids, select_pairs
from pairpilot.store import append_jsonl, columns_of, connect, query_rows, read_json, read_parquet, write_json, write_parquet
from pairpilot.texts import europepmc_fetch, pubmed_fetch, split_sentences

_INTERRUPTED = {"value": False}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _install_signals(run: Path) -> None:
    def _handler(signum, _frame):
        _INTERRUPTED["value"] = True
        _status(run, "process", "interrupted", {"signal": signum, "at": _now()})
        raise SystemExit(128 + int(signum))

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def _status(run: Path, stage: str, state: str, detail: dict | None = None) -> None:
    path = run / "audit" / "stage_status.json"
    current = read_json(path) if path.exists() else {}
    current[stage] = {"state": state, "at": _now(), "detail": detail or {}}
    write_json(path, current)


def _deviation(run: Path, row: dict) -> None:
    append_jsonl(run / "audit" / "deviations.jsonl", {"at": _now(), **row})


def _threads() -> int:
    return max(1, int(os.environ.get("SLURM_CPUS_PER_TASK") or 1))


def _con(run: Path):
    tmp = Path(os.environ.get("SLURM_TMPDIR") or f"/tmp/{os.environ.get('USER', 'pilot')}/pairpilot")
    return connect(threads=_threads(), memory="48GB", tmp=tmp)


def ensure_run(run_id: str, bundle: dict | None = None) -> Path:
    bundle = bundle or load_bundle()
    run = run_root(run_id)
    manifest = run / "manifest.json"
    frozen = redact(bundle)
    frozen["prompt_hashes"] = prompt_hashes()
    config_hash = digest(frozen)
    if manifest.exists():
        prior = read_json(manifest)
        if prior.get("config_hash") != config_hash:
            raise SystemExit(f"run {run_id} belongs to a different config; start a new run")
        return run
    run.mkdir(parents=True, exist_ok=False)
    for name in ("inputs", "requests", "responses", "tables", "audit", "jobs", "logs", "reports", "experiments"):
        (run / name).mkdir()
    write_json(run / "config_resolved.json", frozen)
    write_json(manifest, {
        "run_id": run_id,
        "config_hash": config_hash,
        "rule_version": bundle["pilot"]["rule_version"],
        "created_utc": _now(),
        "code_package": str(PACKAGE),
        "status": "created",
        "references": bundle["pilot"].get("references") or [],
        "budget": budget_state(bundle),
    })
    _status(run, "run", "created", {})
    return run


def stage_preflight(run_id: str) -> dict:
    bundle = load_bundle()
    run = ensure_run(run_id, bundle)
    _install_signals(run)
    registry = bundle["registry"]["tables"]
    con = _con(run)
    inventory = {"checked_utc": _now(), "tables": {}}
    discrepancies = []
    for name, spec in registry.items():
        path = spec.get("path")
        if not path:
            inventory["tables"][name] = {"present": False, "reason": spec.get("note")}
            continue
        file = Path(path)
        entry = {"present": file.exists(), "path": path, "bytes": file.stat().st_size if file.exists() else None}
        if file.exists() and spec.get("format") == "parquet":
            entry["columns"] = columns_of(con, file)
            required = spec.get("required_columns") or []
            missing = [column for column in required if column not in entry["columns"]]
            if missing:
                discrepancies.append({"table": name, "missing_columns": missing})
        inventory["tables"][name] = entry
    relations = Path(registry["pair_relations"]["path"])
    if relations.exists():
        read_parquet(con, relations, "rel")
        counts = {
            row[0]: int(row[1])
            for row in con.execute("SELECT relation_class, count(*) FROM rel GROUP BY 1").fetchall()
        }
        inventory["relation_class_counts"] = counts
        expected = registry["pair_relations"]["expected_class_counts"]
        for key, value in expected.items():
            if counts.get(key) != value:
                discrepancies.append({"table": "pair_relations", "class": key, "actual": counts.get(key), "expected": value})
        shared = sum(counts.get(key, 0) for key in expected)
        if shared != registry["pair_relations"]["expected_shared_pairs"]:
            discrepancies.append({"table": "pair_relations", "shared_pairs": shared})
    year = Path(registry["year"]["path"])
    old_year = Path("/projects/b5ac/project_2/results/03_integrated_analysis/data_analysis_round1/derived/j03/pmid_year.parquet")
    inventory["year_filter"] = {
        "used_for_sampling": False,
        "extended_present": year.exists(),
        "legacy_core_only_year_present": old_year.exists(),
        "legacy_not_used": True,
    }
    inventory["eval_pair_list"] = bundle["registry"].get("eval_pair_list")
    inventory["discrepancies"] = discrepancies
    write_json(run / "inputs" / "inventory.json", inventory)
    if discrepancies:
        write_json(run / "inputs" / "discrepancy.json", {"items": discrepancies})
        _status(run, "preflight", "blocked_input_discrepancy", {"n": len(discrepancies)})
        return {"ok": False, "run": str(run)}
    _status(run, "preflight", "completed", {})
    return {"ok": True, "run": str(run)}


def stage_prepare(run_id: str) -> dict:
    pre = stage_preflight(run_id)
    run = Path(pre["run"])
    if not pre["ok"]:
        _status(run, "prepare", "blocked_input_discrepancy", {})
        return pre
    bundle = load_bundle()
    pilot = bundle["pilot"]
    registry = bundle["registry"]["tables"]
    con = _con(run)
    read_parquet(con, Path(registry["pair_relations"]["path"]), "rel")
    read_parquet(con, Path(registry["core_pair_pmid"]["path"]), "core_pp")
    read_parquet(con, Path(registry["semmed_pair_pmid"]["path"]), "sem_pp")
    actual = {
        "core_pairs": con.execute(f"SELECT count(*) FROM read_parquet('{registry['core_pair']['path']}')").fetchone()[0],
        "core_pair_pmid": con.execute("SELECT count(*) FROM core_pp").fetchone()[0],
        "core_pmid": con.execute("SELECT count(*) FROM (SELECT DISTINCT pmid FROM core_pp)").fetchone()[0],
        "semmed_pair_pmid": con.execute("SELECT count(*) FROM sem_pp").fetchone()[0],
        "semmed_pmid": con.execute("SELECT count(*) FROM (SELECT DISTINCT pmid FROM sem_pp)").fetchone()[0],
        "pair_pmid_intersection": con.execute(
            "SELECT count(*) FROM (SELECT gene_id, disease_cui, pmid FROM core_pp INTERSECT SELECT gene_id, disease_cui, pmid FROM sem_pp)"
        ).fetchone()[0],
        "pmid_intersection": con.execute(
            "SELECT count(*) FROM (SELECT DISTINCT pmid FROM core_pp INTERSECT SELECT DISTINCT pmid FROM sem_pp)"
        ).fetchone()[0],
    }
    actual = {key: int(value) for key, value in actual.items()}
    expected = {
        "core_pairs": registry["core_pair"]["expected_rows"],
        "core_pair_pmid": registry["core_pair_pmid"]["expected_rows"],
        "core_pmid": registry["core_pair_pmid"]["expected_distinct_pmid"],
        "semmed_pair_pmid": registry["semmed_pair_pmid"]["expected_rows"],
        "semmed_pmid": registry["semmed_pair_pmid"]["expected_distinct_pmid"],
        "pair_pmid_intersection": registry["pair_relations"]["expected_pair_pmid_intersection"],
        "pmid_intersection": registry["pair_relations"]["expected_unique_pmid_intersection"],
    }
    bad = {key: {"actual": actual[key], "expected": expected[key]} for key in expected if actual[key] != expected[key]}
    write_json(run / "inputs" / "baseline_check.json", {"actual": actual, "mismatch": bad})
    if bad:
        _status(run, "prepare", "blocked_input_discrepancy", bad)
        return {"ok": False, "run": str(run)}
    rows = query_rows(
        con,
        """
        SELECT gene_id, disease_cui, relation_class, n_core, n_semmed, n_union, n_intersection
        FROM rel
        WHERE relation_class IN ('identical','core_subset','semmed_subset','partial_overlap','disjoint')
        """,
    )
    chosen = select_pairs(rows, pilot["random_seed"], pilot["sampling"]["bands"], pilot["sampling"]["min_pmid_each_source"])
    for item in chosen["deviations"]:
        _deviation(run, item)
    write_parquet(run / "inputs" / "sampling_frame.parquet", [
        row for row in chosen["candidates"]
    ])
    selected = chosen["selected"]
    union_pmids = sum(int(row["n_union"]) for row in selected)
    if union_pmids > pilot["sampling"]["max_pair_pmid"]:
        _deviation(run, {"kind": "pair_pmid_cap", "union_pmids": union_pmids})
    write_parquet(run / "inputs" / "selected_pairs.parquet", selected)
    if not selected:
        _status(run, "prepare", "completed_empty", {})
        return {"ok": True, "run": str(run), "n_pair": 0}
    picked_path = (run / "inputs" / "selected_pairs.parquet").as_posix()
    con.execute(f"CREATE OR REPLACE TABLE picked AS SELECT * FROM read_parquet('{picked_path}')")
    pair_pmids = query_rows(
        con,
        """
        SELECT p.pair_id, p.gene_id, p.disease_cui, CAST(c.pmid AS VARCHAR) AS pmid, 'CORE' AS source
        FROM picked p
        JOIN core_pp c
          ON CAST(c.gene_id AS VARCHAR) = CAST(p.gene_id AS VARCHAR)
         AND CAST(c.disease_cui AS VARCHAR) = CAST(p.disease_cui AS VARCHAR)
        UNION ALL
        SELECT p.pair_id, p.gene_id, p.disease_cui, CAST(s.pmid AS VARCHAR) AS pmid, 'SemMedDB' AS source
        FROM picked p
        JOIN sem_pp s
          ON CAST(s.gene_id AS VARCHAR) = CAST(p.gene_id AS VARCHAR)
         AND CAST(s.disease_cui AS VARCHAR) = CAST(p.disease_cui AS VARCHAR)
        """,
    )
    write_parquet(run / "inputs" / "pair_pmids_raw.parquet", pair_pmids)
    year_path = registry["year"]["path"]
    enriched = query_rows(
        con,
        f"""
        SELECT r.*, y.core_year, y.core_year_status, y.semmed_dp_year, y.semmed_dp_status
        FROM read_parquet('{(run / "inputs" / "pair_pmids_raw.parquet").as_posix()}') r
        LEFT JOIN read_parquet('{year_path}') y ON CAST(r.pmid AS VARCHAR) = CAST(y.pmid AS VARCHAR)
        """,
    )
    write_parquet(run / "inputs" / "pair_pmids.parquet", enriched)
    _write_kg(con, run, registry, selected)
    pmids = sorted({str(row["pmid"]) for row in enriched})
    docs = _fetch_documents(pmids, bundle, run)
    write_parquet(run / "inputs" / "documents.parquet", [
        {key: row[key] for key in row if key != "abstract_sections"} | {"abstract_sections": json_text(row.get("abstract_sections"))}
        for row in docs
    ])
    sentences = []
    for doc in docs:
        for ordinal, span in enumerate(split_sentences(doc.get("text") or "")):
            sentences.append({
                "pmid": doc["pmid"],
                "text_version_id": doc.get("text_version_id") or "",
                "ordinal": ordinal,
                "start": span["start"],
                "end": span["end"],
                "text": span["text"],
                "offset_unit": "python_unicode_code_point",
            })
    write_parquet(run / "inputs" / "sentences.parquet", sentences)
    sampling = pilot["sampling"]
    subsets = {
        "K": choose_ids(selected, pilot["random_seed"], sampling["k_small_pairs"], lambda row: row["band"] == "small"),
        "E3": choose_pmids(enriched, pilot["random_seed"], sampling["e3_pmid_per_pair"], sampling["e3_pmid_global"]),
        "E4": choose_ids(selected, pilot["random_seed"], sampling["e4_seed_pairs"], lambda row: True),
        "E6": choose_ids(selected, pilot["random_seed"], sampling["e6_replicate_pairs"], lambda row: True),
    }
    write_json(run / "inputs" / "frozen_subsets.json", subsets)
    estimate = _estimate(enriched, docs, bundle)
    write_json(run / "inputs" / "request_estimate.json", estimate)
    _status(run, "prepare", "completed", {"n_pair": len(selected), "n_pmid": len(pmids)})
    return {"ok": True, "run": str(run), "n_pair": len(selected), "n_pmid": len(pmids)}


def json_text(value) -> str:
    import json
    return json.dumps(value or [], ensure_ascii=False)


def _write_kg(con, run: Path, registry: dict, selected: list[dict]) -> None:
    picked = (run / "inputs" / "selected_pairs.parquet").as_posix()
    sentence = registry["core_sentence"]["path"]
    core_cols = set(columns_of(con, Path(sentence)))
    wanted = [c for c in (
        "sentence_uri", "gene_id", "disease_cui", "pmid", "content", "gcc", "cge", "ccs", "gci",
        "label_tuple", "usable_status", "gene_mention", "disease_mention",
    ) if c in core_cols or c in {"sentence_uri", "gene_id", "disease_cui", "pmid"}]
    # usable_status may live only on pair_pmid; keep the columns that exist.
    present = [column for column in wanted if column in core_cols]
    where = "WHERE s.pair_status = 'explicit'" if "pair_status" in core_cols else ""
    core_rows = query_rows(
        con,
        f"""
        SELECT {", ".join("s." + column for column in present)}
        FROM read_parquet('{sentence}') s
        JOIN read_parquet('{picked}') p
          ON CAST(s.gene_id AS VARCHAR) = CAST(p.gene_id AS VARCHAR)
         AND CAST(s.disease_cui AS VARCHAR) = CAST(p.disease_cui AS VARCHAR)
        {where}
        """,
    )
    records = []
    for row in core_rows:
        records.append({
            "native_record_id": f"CORE:{row.get('sentence_uri')}:{row.get('gene_id')}:{row.get('disease_cui')}",
            "source": "CORE",
            "gene_id": str(row.get("gene_id")),
            "disease_cui": str(row.get("disease_cui")),
            "pmid": str(row.get("pmid") or ""),
            "text": row.get("content") or "",
            "predicate": None,
            "negated": None,
            "has_type": None,
            "label_tuple": row.get("label_tuple"),
            "gcc": row.get("gcc"),
            "cge": row.get("cge"),
            "ccs": row.get("ccs"),
            "gci": row.get("gci"),
            "mapping_route": None,
            "text_status": "source_sentence" if row.get("content") else "text_unavailable",
        })
    match = registry["accepted_match"]["path"]
    sem_rows = query_rows(
        con,
        f"""
        SELECT m.predication_id, m.sentence_id, m.pmid, m.predicate, m.negation_prefix,
               m.subject_semtype, m.object_semtype, m.mapping_route, m.entity_check,
               m.accepted_gene_id AS gene_id, m.disease_cui, m.argument_name
        FROM read_parquet('{match}') m
        JOIN read_parquet('{picked}') p
          ON CAST(m.accepted_gene_id AS VARCHAR) = CAST(p.gene_id AS VARCHAR)
         AND CAST(m.disease_cui AS VARCHAR) = CAST(p.disease_cui AS VARCHAR)
        """,
    )
    for row in sem_rows:
        records.append({
            "native_record_id": f"SemMedDB:{row.get('predication_id')}:{row.get('gene_id')}:{row.get('disease_cui')}",
            "source": "SemMedDB",
            "gene_id": str(row.get("gene_id")),
            "disease_cui": str(row.get("disease_cui")),
            "pmid": str(row.get("pmid") or ""),
            "sentence_id": str(row.get("sentence_id") or ""),
            "text": "",
            "predicate": row.get("predicate"),
            "negated": bool(row.get("negation_prefix")),
            "has_type": None,
            "label_tuple": None,
            "semtypes": f"{row.get('subject_semtype')}|{row.get('object_semtype')}",
            "mapping_route": row.get("mapping_route"),
            "entity_check": row.get("entity_check"),
            "text_status": "identifier_only",
        })
    write_parquet(run / "inputs" / "kg_records.parquet", records)
    if bundle_recover_enabled():
        try:
            _recover_semmed_sentences(con, run, registry)
        except Exception as exc:
            _deviation(run, {"kind": "semmed_sentence_recovery_failed", "detail": str(exc)[:400]})
            _status(run, "semmed_sentence_recovery", "failed", {"detail": str(exc)[:400]})


def bundle_recover_enabled() -> bool:
    return bool(load_bundle()["pilot"]["text"].get("recover_semmed_sentences"))


def _recover_semmed_sentences(con, run: Path, registry: dict) -> None:
    import sys
    sys.path.insert(0, "/home/b5ac/freddieyu.b5ac/project_2")
    from lib.csv_stream import read_csv_call

    spec = registry["semmed_sentence_csv"]
    ids = con.execute(
        f"""
        SELECT DISTINCT sentence_id FROM read_parquet('{(run / "inputs" / "kg_records.parquet").as_posix()}')
        WHERE source = 'SemMedDB' AND sentence_id IS NOT NULL AND sentence_id <> ''
        """
    ).fetchall()
    wanted = [row[0] for row in ids]
    write_json(run / "inputs" / "semmed_sentence_scan.json", {
        "path": spec["path"],
        "bytes": Path(spec["path"]).stat().st_size,
        "n_sentence_id": len(wanted),
        "purpose": "project selected SemMedDB sentence text once",
    })
    if not wanted:
        _status(run, "semmed_sentence_recovery", "completed_empty", {})
        return
    con.execute("CREATE OR REPLACE TABLE wanted(sentence_id VARCHAR)")
    con.executemany("INSERT INTO wanted VALUES (?)", [(item,) for item in wanted])
    call = read_csv_call(spec["path"], int(spec["columns"]), spec["dialect"], 2000000, True, encoding=spec["encoding"])
    con.execute(
        f"""
        COPY (
          SELECT c0 AS sentence_id, c1 AS pmid, c5 AS sentence_text
          FROM {call}
          WHERE c0 IN (SELECT sentence_id FROM wanted)
        ) TO '{(run / "inputs" / "semmed_sentences.parquet").as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    _status(run, "semmed_sentence_recovery", "completed", {"n": len(wanted)})


def _fetch_documents(pmids: list[str], bundle: dict, run: Path) -> list[dict]:
    limits = bundle["resources"]["limits"]
    network = bundle["resources"]["network"]
    try:
        docs = pubmed_fetch(pmids, limits["ncbi_pause_seconds"], limits["ncbi_batch_size"], network["ncbi_key_env"])
    except Exception as exc:
        _deviation(run, {"kind": "pubmed_fetch_failed", "detail": str(exc)[:300]})
        docs = [{"pmid": pmid, "title": "", "text": "", "completeness": "fetch_failed", "source": "pubmed",
                 "retrieved_at": _now(), "text_sha256": "", "text_version_id": "", "abstract_sections": []} for pmid in pmids]
    repaired = []
    for doc in docs:
        if doc.get("completeness") == "complete_abstract":
            repaired.append(doc)
            continue
        try:
            alt = europepmc_fetch(doc["pmid"])
        except Exception:
            alt = None
        repaired.append(alt or doc)
    failures = [row for row in repaired if row.get("completeness") != "complete_abstract"]
    if failures:
        write_parquet(run / "audit" / "failures.parquet", [
            {"pmid": row["pmid"], "stage": "abstract", "status": row.get("completeness")} for row in failures
        ])
    return repaired


def _estimate(pair_pmids: list[dict], docs: list[dict], bundle: dict) -> dict:
    pmids = {str(row["pmid"]) for row in pair_pmids}
    usable = sum(1 for row in docs if row.get("completeness") == "complete_abstract")
    return {
        "n_pmid": len(pmids),
        "n_usable_abstract": usable,
        "extract_calls_upper": usable,
        "direct_calls_upper": 10,
        "note": "Comparison and synthesis calls depend on the number of findings and are not invented here.",
        "budget": budget_state(bundle),
        "model_calls_started": False,
    }


def stage_canary(run_id: str) -> dict:
    bundle = load_bundle()
    run = ensure_run(run_id, bundle)
    _install_signals(run)
    if not budget_state(bundle)["approved"]:
        _status(run, "canary", "blocked_budget", budget_state(bundle))
        return {"ok": False, "blocked": "budget"}
    docs = _load(run / "inputs" / "documents.parquet")
    ready = [row for row in docs if row.get("completeness") == "complete_abstract"][:2]
    if len(ready) < 1:
        _status(run, "canary", "blocked_no_abstract", {})
        return {"ok": False}
    return _model_batch(run, bundle, "canary", ready, limit=2)


def stage_extract(run_id: str) -> dict:
    bundle = load_bundle()
    run = ensure_run(run_id, bundle)
    if not budget_state(bundle)["approved"]:
        _status(run, "extract", "blocked_budget", {})
        _status(run, "analyze", "blocked_budget", {})
        return {"ok": False, "blocked": "budget"}
    return _model_batch(run, bundle, "extract", _load(run / "inputs" / "documents.parquet"))


def _model_batch(run: Path, bundle: dict, stage: str, documents: list[dict], limit: int | None = None) -> dict:
    model = bundle["pilot"]["primary_model"]
    adapter = OpenAIAdapter(model["model_id"], int(model["max_output_tokens"]), float(model["temperature"]))
    cache = CallCache(run / "responses" / "cache")
    ledger = UsageLedger(bundle, run / "audit" / "usage.json")
    prompt = (PACKAGE / "prompts" / "extract_findings.txt").read_text(encoding="utf-8")
    schema_id = "finding.v1"
    rows = []
    failures = []
    docs = documents if limit is None else documents[:limit]
    for doc in docs:
        if _INTERRUPTED["value"]:
            break
        text = doc.get("text") or ""
        if len(text) > bundle["pilot"]["text"]["max_input_chars"]:
            failures.append({"pmid": doc.get("pmid"), "status": "not_evaluated_context"})
            continue
        request = {
            "task": "extract_findings",
            "input_sha256": input_hash(text),
            "schema_id": schema_id,
            "prompt_sha256": input_hash(prompt),
            "model_id": model["model_id"],
            "params": {"temperature": model["temperature"]},
            "prompt": prompt,
            "user": text,
            "anchor_quote": "",
        }
        result = run_request(adapter, cache, ledger, request, run, bundle["resources"]["limits"]["network_retries"])
        if result["status"] != "ok":
            failures.append({"pmid": doc.get("pmid"), "status": result["status"]})
            continue
        for local, finding in enumerate((result["parsed"] or {}).get("findings") or []):
            quote = ((finding.get("evidence") or [{}])[0].get("quote")) or ""
            located = locate_quote(text, quote)
            rows.append({
                "pmid": str(doc.get("pmid")),
                "local": local,
                "located": bool(located and exact_span(text, located[0], located[1], quote)),
                "question_type": finding.get("question_type"),
                "discourse_role": finding.get("discourse_role"),
                "status": "citation_located" if located else "citation_not_located",
            })
    if rows:
        write_parquet(run / "tables" / "canary_findings.parquet" if stage == "canary" else run / "tables" / "findings.parquet", rows)
    if failures:
        write_parquet(run / "audit" / "failures.parquet", failures)
    _status(run, stage, "completed" if not failures else "completed_with_failures", {"n": len(rows), "failures": len(failures)})
    return {"ok": True, "n": len(rows)}


def stage_analyze(run_id: str) -> dict:
    """Program analysis of saved findings. It does not call a model."""
    bundle = load_bundle()
    run = ensure_run(run_id, bundle)
    findings_path = run / "tables" / "findings.parquet"
    if not findings_path.exists():
        _status(run, "analyze", "blocked_no_findings", {})
        for name in ("E1", "E2", "E3"):
            _status(run, name, "blocked_no_findings", {})
        return {"ok": False}
    findings = _load(findings_path)
    pair_pmids = _load(run / "inputs" / "pair_pmids.parquet")
    rules = bundle["pilot"]["required_comparability_fields"]
    config_hash = read_json(run / "manifest.json")["config_hash"]
    comparisons = build_candidates(findings, config_hash, rules)
    write_parquet(run / "tables" / "comparisons.parquet", comparisons)
    gaps = [gap for gap in (gap_from_comparison(row, config_hash) for row in comparisons) if gap]
    write_parquet(run / "tables" / "candidate_gaps.parquet", gaps)
    metrics = []
    by_pair = {}
    for row in pair_pmids:
        by_pair.setdefault(row["pair_id"], {"CORE": set(), "SemMedDB": set()})
        by_pair[row["pair_id"]][row["source"]].add(str(row["pmid"]))
    for pair, sets in by_pair.items():
        metrics.extend(source_metrics(run_id, pair, findings, comparisons, sets["CORE"], sets["SemMedDB"]))
        draw = equal_article(sets["CORE"], sets["SemMedDB"], findings, comparisons, bundle["pilot"]["random_seed"])
        (run / "experiments" / "E1_source_contribution").mkdir(parents=True, exist_ok=True)
        write_json(run / "experiments" / "E1_source_contribution" / f"{pair}.json", draw)
    write_parquet(run / "tables" / "experiment_metrics.parquet", metrics)
    _status(run, "E1", "completed", {"n_metric": len(metrics)})
    _status(run, "analyze", "completed", {})
    return {"ok": True}


def stage_paths(run_id: str) -> dict:
    bundle = load_bundle()
    run = ensure_run(run_id, bundle)
    if not bundle["registry"]["tables"]["graph_edges"].get("path"):
        _status(run, "E4", "skipped_unavailable_input", {"missing": "gene-gene edge index and graph_scope_version"})
        return {"ok": True, "skipped": True}
    _status(run, "E4", "blocked_resource", {})
    return {"ok": False}


def stage_search(run_id: str) -> dict:
    bundle = load_bundle()
    run = ensure_run(run_id, bundle)
    gaps = run / "tables" / "candidate_gaps.parquet"
    if not gaps.exists():
        _status(run, "E5", "blocked_no_candidates", {})
        return {"ok": False}
    _status(run, "E5", "blocked_budget" if not budget_state(bundle)["approved"] else "ready", {})
    return {"ok": budget_state(bundle)["approved"]}


def stage_robustness(run_id: str) -> dict:
    bundle = load_bundle()
    run = ensure_run(run_id, bundle)
    if not budget_state(bundle)["approved"]:
        _status(run, "E6", "blocked_budget", {"open_model": "disabled"})
        return {"ok": False}
    _status(run, "E6", "ready", {})
    return {"ok": True}


def stage_report(run_id: str) -> dict:
    from pairpilot.reporting import write_report
    bundle = load_bundle()
    run = ensure_run(run_id, bundle)
    status = read_json(run / "audit" / "stage_status.json") if (run / "audit" / "stage_status.json").exists() else {}
    complete = status.get("prepare", {}).get("state") == "completed" and status.get("E1", {}).get("state") == "completed"
    _status(run, "report", "complete" if complete else "incomplete", {})
    write_report(run, bundle)
    manifest = read_json(run / "manifest.json")
    manifest["status"] = "complete" if complete else "incomplete"
    write_json(run / "manifest.json", manifest)
    return {"ok": True, "status": manifest["status"]}


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    con = connect()
    return query_rows(con, f"SELECT * FROM read_parquet('{path.as_posix()}')")


def synthetic_chain(tmp: Path) -> dict:
    """Integration path used by fixture tests. It does not read project data."""
    findings = [
        {"finding_id": "f1", "pair_id": "g__C1", "pmid": "1", "question_type": "prognostic_association",
         "study_system": "human", "comparator": "low", "outcome": "os", "exposure_or_intervention": "high",
         "result_direction": "higher_risk", "significance": "not_significant", "discourse_role": "own_result"},
        {"finding_id": "f2", "pair_id": "g__C1", "pmid": "2", "question_type": "prognostic_association",
         "study_system": "human", "comparator": "low", "outcome": "os", "exposure_or_intervention": "high",
         "result_direction": "lower_risk", "significance": "significant", "discourse_role": "own_result"},
        {"finding_id": "f3", "pair_id": "g__C1", "pmid": "9", "question_type": "prognostic_association",
         "study_system": "human", "comparator": "low", "outcome": "os", "exposure_or_intervention": "high",
         "result_direction": "higher_risk", "significance": "not_reported", "discourse_role": "own_result"},
    ]
    rules = load_bundle()["pilot"]["required_comparability_fields"]
    comparisons = build_candidates(findings, "cfg", rules)
    core, sem = {"1", "2"}, {"2", "9"}
    stats_a = view_stats(findings, comparisons, article_sets(core, sem)["Union"])
    stats_b = view_stats(findings, comparisons, article_sets(set(core), set(sem))["Union"])
    core_view = filter_comparisons(comparisons, core)
    return {
        "same_union": stats_a == stats_b,
        "core_view_pmids": sorted({row["pmid_a"] for row in core_view} | {row["pmid_b"] for row in core_view}),
        "n_comparison": len(comparisons),
    }

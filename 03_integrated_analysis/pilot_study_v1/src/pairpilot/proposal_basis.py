#!/usr/bin/env python3
"""Offline proposal evidence package. No model calls and no formal draw."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pairpilot.query_ast import source_membership  # noqa: E402
from pairpilot.store import write_json  # noqa: E402

PARENT = Path("/projects/b5ac/project_2/results/03_integrated_analysis/pilot_study_v1/runs/20261017T112559Z_readiness1")
RUNS = PARENT.parent
RESULTS = Path("/projects/b5ac/project_2/results")
V2 = RESULTS / "03_integrated_analysis" / "id_alignment_v2_round1"
PACKAGE = ROOT
STALE_YEAR_BOTH = 18307
CURRENT_YEAR_BOTH = 27193


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _login() -> bool:
    return os.uname().nodename.startswith("login")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reuse_key(run_id: str, pmid: str) -> str:
    """A PMID reused from an ancestor is the same article, not a new sample."""
    return str(pmid)


def select_year_both(stored_current: int, stored_stale: int) -> dict:
    if stored_stale == STALE_YEAR_BOTH and stored_current == CURRENT_YEAR_BOTH:
        return {"used": stored_current, "stale_not_used": stored_stale, "status": "current_table"}
    return {"used": None, "stale_not_used": stored_stale, "status": "conflict"}


def unit_counts(rows: list[dict]) -> dict:
    cases = {row.get("new_case_id") for row in rows if row.get("new_case_id")}
    papers = {row.get("paper_pair") for row in rows if row.get("paper_pair")}
    questions = {
        (row.get("paper_pair"), row.get("question_id"))
        for row in rows
        if row.get("paper_pair") and row.get("question_id") not in {None, "", "mapping_unresolved"}
    }
    return {
        "mapping_rows": len(rows),
        "unique_case_ids": len(cases),
        "unique_paper_pairs": len(papers),
        "paper_question_units": len(questions),
        "component_rows": len(rows),
    }


def status_label(value: str) -> str:
    allowed = {
        "not_evaluated",
        "not_recorded",
        "zero",
        "skipped",
        "technical_failure",
        "response_missing",
    }
    if value not in allowed:
        raise ValueError(value)
    return value


def claim(claim_id: str, section: str, statement: str, kind: str, source: str, unit: str, status: str, limit: str, replaced: str = "") -> dict:
    return {
        "claim_id": claim_id,
        "section": section,
        "statement": statement,
        "evidence_kind": kind,
        "source": source,
        "unit": unit,
        "check_status": status,
        "limit": limit,
        "replaced_wording": replaced,
    }


def _duck_count(path: Path, sql: str):
    import duckdb
    con = duckdb.connect()
    return con.execute(sql.replace("{path}", path.as_posix())).fetchall()


def _count_or_missing(path: Path, sql: str, label: str, checks: list[dict]):
    if not path.exists():
        checks.append({"name": label, "status": "product_missing", "path": str(path)})
        return None
    try:
        return _duck_count(path, sql)
    except Exception as exc:
        checks.append({"name": label, "status": "unresolved", "path": str(path), "error": type(exc).__name__})
        return None


def recount(checks: list[dict]) -> dict:
    out = {}
    core = RESULTS / "01_core_audit" / "tables" / "pair_pmid.parquet"
    rows = _count_or_missing(core, "SELECT COUNT(*), COUNT(DISTINCT CAST(gene_id AS VARCHAR) || '|' || CAST(disease_cui AS VARCHAR)), COUNT(DISTINCT pmid) FROM read_parquet('{path}')", "core_pair_pmid", checks)
    if rows:
        out["core_pair_pmid"], out["core_pairs"], out["core_pmid"] = [int(value) for value in rows[0]]
        out["core_match"] = [out["core_pair_pmid"], out["core_pairs"], out["core_pmid"]] == [321201, 153016, 161449]
        checks.append({"name": "core_pair_pmid", "status": "match" if out["core_match"] else "conflict", "got": [out["core_pair_pmid"], out["core_pairs"], out["core_pmid"]]})
    relations = V2 / "tables" / "pair_relations.parquet"
    classes = _count_or_missing(relations, "SELECT relation_class, COUNT(*) FROM read_parquet('{path}') GROUP BY 1", "five_class", checks)
    if classes:
        out["classes"] = {str(name): int(count) for name, count in classes}
        expected = {"identical": 5303, "core_subset": 2626, "semmed_subset": 1590, "partial_overlap": 3883, "disjoint": 17508}
        got = {name: out["classes"].get(name) for name in expected}
        out["five_match"] = got == expected and sum(expected.values()) == 30910
        checks.append({"name": "five_class", "status": "match" if out["five_match"] else "conflict", "got": got, "anchor": expected})
    sem = V2 / "tables" / "semmed_pair_pmid.parquet"
    sem_rows = _count_or_missing(sem, "SELECT COUNT(*), COUNT(DISTINCT pmid) FROM read_parquet('{path}')", "semmed_pair_pmid", checks)
    if sem_rows:
        out["semmed_pair_pmid"], out["semmed_pmid_rows"] = sem_rows[0]
        checks.append({"name": "semmed_pair_pmid", "status": "recounted", "got": list(sem_rows[0]), "anchor_rows": 137221})
    year = json.loads((V2 / "revision" / "year_coverage" / "derived" / "year_coverage.json").read_text(encoding="utf-8"))
    both = next(row["n_pair"] for row in year["corrected"]["transition"]["states"] if row["state_after"] == "both_nonempty")
    chosen = select_year_both(both, STALE_YEAR_BOTH)
    out["year_both"] = chosen
    checks.append({"name": "year_both", "status": chosen["status"], "used": chosen["used"]})
    findings = RUNS / "20261015T221200Z_b20cont" / "tables" / "findings.parquet"
    roles = _count_or_missing(findings, "SELECT COALESCE(discourse_role, 'missing'), COUNT(*) FROM read_parquet('{path}') GROUP BY 1", "finding_roles", checks)
    if roles:
        out["roles"] = {"" if name is None else str(name): int(count) for name, count in roles}
        out["finding_rows"] = sum(out["roles"].values())
        empty_status = _count_or_missing(
            findings,
            "SELECT COALESCE(status, 'missing'), COUNT(*) FROM read_parquet('{path}') WHERE discourse_role IS NULL OR discourse_role = '' GROUP BY 1",
            "empty_role_status",
            checks,
        )
        if empty_status:
            out["empty_role_status"] = {str(name): int(count) for name, count in empty_status}
        checks.append({"name": "finding_roles", "status": "recounted", "got": out["roles"], "empty_role_status": out.get("empty_role_status")})
    case_map = PARENT / "tables" / "legacy_case_map.parquet"
    mapped = _count_or_missing(case_map, "SELECT new_case_id, paper_pair, question_id, legacy_comparison_id FROM read_parquet('{path}')", "case_map", checks)
    if mapped:
        records = [{"new_case_id": a, "paper_pair": b, "question_id": c, "legacy_comparison_id": d} for a, b, c, d in mapped]
        out["units"] = unit_counts(records)
        checks.append({"name": "case_units", "status": "recounted", "got": out["units"]})
    docs = RUNS / "20261015T204633Z_0d9b9724" / "inputs" / "pair_pmids.parquet"
    pmids = _count_or_missing(docs, "SELECT pair_id, COUNT(DISTINCT pmid) FROM read_parquet('{path}') GROUP BY 1", "pilot_pmids", checks)
    if pmids:
        per_pair = {pair: count for pair, count in pmids}
        unique = _duck_count(docs, "SELECT COUNT(DISTINCT pmid) FROM read_parquet('{path}')")[0][0]
        out["pilot_per_pair"] = per_pair
        out["pilot_unique_pmid"] = unique
        out["pilot_sum_of_pair_counts"] = sum(per_pair.values())
        checks.append({"name": "pilot_pmids", "status": "recounted", "unique": unique, "sum": out["pilot_sum_of_pair_counts"], "n_pairs": len(per_pair)})
    return out


def role_sentence(counts: dict) -> str:
    roles = counts.get("roles") or {}
    if not roles:
        return "提取行的角色分布没有从 findings 表复现，正文不采用未经复核的 273 这一数字。"
    empty = int(roles.get("") or 0)
    named = (
        f"本文结果 {roles.get('own_result', 0)}，作者解释 {roles.get('author_interpretation', 0)}，"
        f"背景 {roles.get('background', 0)}，角色字段为空 {empty}"
    )
    status = counts.get("empty_role_status") or {}
    if empty and status == {"no_finding": empty}:
        status_note = f"这 {empty} 行的 status 是 no_finding，角色字段本身没有写成这个词。"
    elif empty:
        status_note = f"这 {empty} 行的 status 为 {status}。"
    else:
        status_note = ""
    return f"主提取表共 {counts.get('finding_rows')} 行：{named}。{status_note}有定位不是已核对正确。"


def case_sentence(counts: dict) -> str:
    units = counts.get("units") or {}
    if not units:
        return "新旧案例单位没有从 legacy_case_map 复现，不能把映射行数称为独立新案例。"
    return (
        f"legacy_case_map 有 {units['mapping_rows']} 条映射行、{units['unique_paper_pairs']} 个文章对、"
        f"{units['paper_question_units']} 个文章对加中性问题、{units['unique_case_ids']} 个含文本版本的 case_id。"
        "同一 case 内的多条映射是组成部分，不取其中最高比较等级代表整个案例。"
    )


def pmid_sentence(counts: dict) -> str:
    if "pilot_unique_pmid" not in counts:
        return "唯一摘要数没有从 pair_pmids 表复现。"
    return (
        f"pair_pmids 表有 {counts['pilot_per_pair'] and len(counts['pilot_per_pair'])} 个 pair、"
        f"{counts['pilot_unique_pmid']} 个唯一 PMID；各 pair 计数之和是 {counts['pilot_sum_of_pair_counts']}。"
        "唯一 PMID 才是摘要样本单位。"
    )


def fill_report(text: str, counts: dict) -> str:
    replaced = text.replace("{{FINDING_ROLE_SENTENCE}}", role_sentence(counts))
    replaced = replaced.replace("{{CASE_UNIT_SENTENCE}}", case_sentence(counts))
    replaced = replaced.replace("{{UNIQUE_PMID_SENTENCE}}", pmid_sentence(counts))
    if "{{" in replaced:
        raise RuntimeError("unfilled proposal placeholder")
    return replaced


def claim_ids(text: str) -> list[str]:
    return re.findall(r"C-[A-Z0-9-]+", text)


def literature_rows() -> list[dict]:
    return [
        {"id": "C-LIT-CORE", "title": "Building a large gene expression-cancer knowledge base with limited human annotations", "year": 2023, "doi": "10.1093/database/baad061", "read_level": "abstract_plus_local_schema", "can_support": "CORE is a cancer gene-expression KB built with limited annotation.", "cannot_support": "Full-text methods comparison was not completed; PMC returned 403."},
        {"id": "C-LIT-SEMMED", "title": "SemMedDB: a PubMed-scale repository of biomedical semantic predications", "year": 2012, "doi": "10.1093/bioinformatics/bts591", "read_level": "official_download_page_and_local_audit", "can_support": "VER43 file naming, row counts, MEDLINE cutoff 2024-05-08, SemRep 1.8.", "cannot_support": "2012 article abstract was not returned by the publisher API."},
        {"id": "C-LIT-ASQ", "title": "Triangulating evidence in health sciences with Annotated Semantic Queries", "year": 2024, "url": "https://academic.oup.com/bioinformatics/article/40/9/btae519/7738781", "read_level": "citation_unconfirmed", "can_support": "The title names annotated semantic queries and triangulation.", "cannot_support": "Methods were not read; absence of a feature is not established."},
        {"id": "C-LIT-TRI", "title": "Triangulation in aetiological epidemiology", "year": 2016, "doi": "10.1093/ije/dyw314", "read_level": "abstract", "can_support": "Triangulation integrates approaches with different bias sources.", "cannot_support": "Does not show that joining two literature databases is triangulation."},
        {"id": "C-LIT-EI", "title": "Evidence Inference 2.0: More Data, Better Models", "year": 2020, "url": "https://aclanthology.org/2020.bionlp-1.13/", "read_level": "abstract", "can_support": "Single-trial comparative treatment inference with evidence spans.", "cannot_support": "Not a cross-article gene-cancer dossier."},
        {"id": "C-LIT-CARE", "title": "CARE: Extracting Experimental Findings From Clinical Literature", "year": 2024, "url": "https://aclanthology.org/2024.findings-naacl.285/", "read_level": "abstract", "can_support": "N-ary clinical findings from 700 abstracts.", "cannot_support": "Cross-article comparison scope was not confirmed from the full paper."},
        {"id": "C-LIT-SCICLAIM", "title": "Extracting Fine-Grained Knowledge Graphs of Scientific Claims", "year": 2021, "url": "https://aclanthology.org/2021.emnlp-main.381/", "read_level": "abstract", "can_support": "Fine-grained claim graphs with qualifiers.", "cannot_support": "Not shown to compare gene-cancer abstracts under a fixed question."},
        {"id": "C-LIT-TRIAL", "title": "Accelerating clinical evidence synthesis with large language models", "year": 2025, "doi": "10.1038/s41746-025-01840-7", "read_level": "abstract_and_introduction", "can_support": "TrialMind covers search, screening and extraction for systematic reviews.", "cannot_support": "Its user-study time change is not evidence for this project."},
        {"id": "C-LIT-OS", "title": "Synthesizing scientific literature with retrieval-augmented language models", "year": 2026, "doi": "10.1038/s41586-025-10072-4", "read_level": "abstract", "can_support": "Multi-paper synthesis with citation checking on ScholarQABench.", "cannot_support": "Not a condition-level gene-cancer comparison."},
        {"id": "C-LIT-EVNET", "title": "Source-grounded biomedical evidence records from full-text disease-focused literature", "year": 2026, "url": "https://arxiv.org/abs/2603.28325v4", "read_level": "abstract_and_methods_html", "can_support": "Full-text evidence records with context, entities and record links for HCC, CRC and SLE.", "cannot_support": "Reported technical validation is not this project's result. Cross-resource comparison against CORE was not assessed."},
        {"id": "C-LIT-ERASER", "title": "ERASER: A Benchmark to Evaluate Rationalized NLP Models", "year": 2020, "url": "https://aclanthology.org/2020.acl-main.408/", "read_level": "abstract", "can_support": "Rationale metrics include faithfulness as influence on the prediction.", "cannot_support": "This project did not compute those probability metrics."},
    ]


def base_claims(counts: dict) -> list[dict]:
    units = counts.get("units") or {}
    rows = [
        claim("C-DATA-CORE", "data", "CORE explicit pairs 153016, pair-PMID 321201, PMID 161449.", "data_observation", str(RESULTS / "01_core_audit/tables/pair_pmid.parquet"), "pair / pair-PMID / PMID", "recounted_if_present", "Local RDF, not every filter stage in the paper."),
        claim("C-DATA-HASTYPE", "data", "Stored hasType BIOMARKER 105089, ONCOGENE 35520, TSG 12407; UNRELIABLE not observed.", "data_observation", str(RESULTS / "01_core_audit/report/core_audit.md"), "GCS", "report_and_schema", "Builder score is not in the RDF."),
        claim("C-DATA-SEMMED", "data", "Local six-table row counts match the VER43 download page; no local checksum.", "data_observation", str(RESULTS / "02_semmeddb_validation/report/semmed_audit.md"), "table row", "report_matches_official_page", "Byte identity is not claimed."),
        claim("C-DATA-FUNNEL", "data", "Argument rows 8741634 to accepted 513477 to 173048 on CORE pairs; 340429 excluded as not explicit CORE pairs.", "data_observation", str(V2 / "revision/derived/revision_numbers.json"), "core-anchored argument row", "prior_table_check", "Not pair counts."),
        claim("C-DATA-FIVE", "data", "Five classes and 122106 CORE pairs outside the both-nonempty main match.", "data_observation", str(V2 / "tables/pair_relations.parquet"), "pair", "recounted_if_present", "Empty-side pairs are outside the five classes."),
        claim("C-DATA-YEAR", "data", "Year-filtered both-nonempty pairs are 27193; 18307 is stale.", "data_observation", str(V2 / "revision/year_coverage/derived/year_coverage.json"), "shared pair", counts.get("year_both", {}).get("status", "unchecked"), "Not a biological year effect."),
        claim("C-DATA-ACCEPT", "data", "accepted is a mapping rule. Path A has no per-mention consistency gate.", "design_decision", str(V2 / "derived/rules_v2.json"), "argument", "code_and_prior_report", "Not a manually verified alignment count."),
        claim("C-PILOT-SAMPLE", "pilot", pmid_sentence(counts), "data_observation", str(RUNS / "20261015T204633Z_0d9b9724/inputs/pair_pmids.parquet"), "unique PMID", "recounted_if_present", "Identical stratum was empty in the chosen bands."),
        claim("C-PILOT-5812", "pilot", "5812 old cross-PMID finding pairs; 5030 insufficient from 4600 other/unknown plus 430 empty legacy fields.", "implementation_observation", str(RUNS / "20261016T013500Z_diag1/audit/p0.json"), "old comparison row", "prior_audit", "Not confirmed absence from the abstract."),
        claim("C-PILOT-E1", "pilot", "Condition tuples follow free text. Fifty draws are one-sample sensitivity.", "design_decision", str(ROOT / "src/pairpilot/experiments/e1.py"), "tuple / draw", "code", "Not fifty independent experiments."),
        claim("C-PILOT-E2", "pilot", "D is direct comparison, F is extraction, K adds raw KG fields and has no comparison calls.", "implementation_observation", str(ROOT / "src/pairpilot/experiments/e2.py"), "call", "code_and_rev2_report", "K is not aligned to the D/F product."),
        claim("C-PILOT-E3", "pilot", "16 source-abstract pairs; source result nonempty 13, conclusion 8; abstract both fields 16.", "model_observation_unvalidated", str(RUNS / "20261016T005500Z_rev2/tables/scope_transfer.parquet"), "PMID pair", "prior_report", "Not an accuracy gain."),
        claim("C-PILOT-24", "pilot", "24 legacy comparisons: describe 24, direction 8, magnitude 0, unverified.", "model_observation_unvalidated", str(RUNS / "20261016T012650Z_close1/audit/source_summary.json"), "finding pair", "prior_audit", "Do not promote a component to the case."),
        claim("C-PILOT-E6", "pilot", "14 papers have r0 and r1; 31 aligned; direction string differences are not polarity flips.", "model_observation_unvalidated", str(RUNS / "20261016T020912Z_e6diag/reports/report.md"), "paper / aligned finding", "prior_report", "Not end-to-end reliability."),
        claim("C-PILOT-READY", "pilot", "4 extracts and 2 comparisons parsed; 74 quotes located; semantic support not evaluated.", "implementation_observation", str(PARENT / "reports/report.md"), "technical unit / quote", "prior_report", "Not human calibration."),
        claim("C-PILOT-E5", "pilot", "27 protocol deep-reads; model refined/still_open is not a confirmed gap.", "model_observation_unvalidated", str(RUNS / "20261016T012650Z_close1/reports/report.md"), "paper", "prior_report", "One example only."),
        claim("C-PLAN-V2", "design", "Formal start is 4 pairs, 4-6 papers, at most 24 PMIDs and 24 cases, draw not executed.", "planned_test", str(ROOT / "protocol/implementation_plan_v2.md"), "future case", "spec_not_run", "Not a powered sample size."),
    ]
    if units:
        rows.append(claim("C-PILOT-UNITS", "pilot", case_sentence(counts), "implementation_observation", str(PARENT / "tables/legacy_case_map.parquet"), "mapping row / case id", "recounted", "Mapping rows are not independent cases."))
    return rows


def experiment_rows() -> list[dict]:
    return [
        {"id": "prep", "run": "20261015T204633Z_0d9b9724", "state": "materials_frozen", "scientific_status": "not_an_accuracy_study", "method": "rule sampling of pairs and PMIDs"},
        {"id": "pilot_extract", "run": "20261015T221200Z_b20cont", "state": "extracts_retained", "scientific_status": "model_unvalidated", "method": "gpt-6.1-sol abstract extract"},
        {"id": "rev2", "run": "20261016T005500Z_rev2", "state": "rules_revised", "scientific_status": "incomplete", "method": "comparison_gap_v2 on stored extracts"},
        {"id": "diag1", "run": "20261016T013500Z_diag1", "state": "diagnostic_completed", "scientific_status": "model_unvalidated", "method": "bounded reread and compare"},
        {"id": "close1", "run": "20261016T012650Z_close1", "state": "source_and_dossier_completed", "scientific_status": "model_unvalidated", "method": "source class and protocol search"},
        {"id": "e6diag", "run": "20261016T020912Z_e6diag", "state": "bounded_diagnostic", "scientific_status": "not_original_e6", "method": "two extracts and four perturbations"},
        {"id": "protocol1", "run": "20261017T020726Z_protocol1", "state": "protocol_written", "scientific_status": "not_started", "method": "offline protocol, no new calls"},
        {"id": "readiness1", "run": "20261017T112559Z_readiness1", "state": "technical_pass", "scientific_status": "pending_reviewer", "method": "v2 contract technical canary"},
        {"id": "E4", "run": "", "state": "code_only", "scientific_status": "not_started", "method": "path ranking not executed"},
        {"id": "open_model", "run": "", "state": "disabled", "scientific_status": "not_started", "method": "not used"},
        {"id": "formal", "run": "", "state": "not_started", "scientific_status": "not_started", "method": "v2 sampling spec, draw closed"},
    ]


def design_rows() -> list[dict]:
    return [
        {"id": "RQ1", "unit": "output field", "denominator": "pre-specified target results only for omission", "now": "quote location", "future": "text support and field errors", "supports_contribution": "faithful organization in the frozen sample"},
        {"id": "RQ2", "unit": "case and component", "denominator": "frozen question-paper pairs including failure", "now": "program status only", "future": "human versus model cross-table", "supports_contribution": "where comparison stops"},
        {"id": "RQ3", "unit": "fixed case set", "denominator": "pair-specific PMID sets", "now": "material source class", "future": "human-confirmed increment", "supports_contribution": "which comparisons need both resources"},
        {"id": "superiority", "unit": "matched abstract-versus-structure contrast", "denominator": "not collected", "now": "not_evaluated", "future": "only if that contrast is added", "supports_contribution": "removed from the main claim"},
    ]


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run() -> Path:
    run = RUNS / os.environ["PILOT_CHILD_RUN"]
    if run.resolve() == PARENT.resolve() or PARENT.resolve() in run.resolve().parents:
        raise SystemExit("refusing to write the parent")
    run.mkdir(parents=True, exist_ok=True)
    for name in ("reports", "tables", "audit", "jobs"):
        (run / name).mkdir(exist_ok=True)
    return run


def stage_j0(run_id: str) -> int:
    import unittest
    run = _run()
    suite = unittest.defaultTestLoader.discover(str(PACKAGE / "tests"), pattern="test_proposal_basis.py")
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    checks = []
    if not result.wasSuccessful():
        checks.append({"name": "unit_tests", "status": "failed", "failures": len(result.failures), "errors": len(result.errors)})
    else:
        checks.append({"name": "unit_tests", "status": "passed", "n": result.testsRun})
    counts = recount(checks)
    claims = base_claims(counts)
    _jsonl(run / "tables" / "claims.jsonl", claims)
    _jsonl(run / "tables" / "experiments.jsonl", experiment_rows())
    _jsonl(run / "tables" / "design.jsonl", design_rows())
    _jsonl(run / "tables" / "literature.jsonl", literature_rows())
    sources = []
    for path in (
        RESULTS / "01_core_audit/report/core_audit.md",
        RESULTS / "02_semmeddb_validation/report/semmed_audit.md",
        V2 / "revision/derived/final_report_checks.json",
        V2 / "revision/year_coverage/derived/year_coverage.json",
        PARENT / "audit/ledger.json",
        PARENT / "manifest.json",
    ):
        sources.append({"path": str(path), "exists": path.exists(), "sha256": sha256_file(path) if path.exists() and path.stat().st_size < 5_000_000 else "not_hashed_large_or_missing"})
    _jsonl(run / "audit" / "source_index.jsonl", sources)
    write_json(run / "audit" / "counts.json", counts)
    write_json(run / "jobs" / "j0.json", {"utc": _now(), "tests_ok": result.wasSuccessful()})
    write_json(run / "audit" / "checks.json", {"stage": "j0", "checks": checks})
    return 0 if result.wasSuccessful() else 1


def stage_j1(run_id: str) -> int:
    run = _run()
    counts = json.loads((run / "audit" / "counts.json").read_text(encoding="utf-8"))
    proposal = fill_report((PACKAGE / "protocol" / "proposal_basis.md").read_text(encoding="utf-8"), counts)
    claims = [json.loads(line) for line in (run / "tables" / "claims.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    literature = [json.loads(line) for line in (run / "tables" / "literature.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    known = {row["claim_id"] for row in claims} | {row["id"] for row in literature}
    used = set(claim_ids(proposal))
    missing = sorted(item for item in used - known if item.startswith("C-"))
    ledger = PARENT / "audit" / "ledger.json"
    closing = json.loads((PARENT / "audit" / "ledger_closing.json").read_text(encoding="utf-8"))
    ledger_same = sha256_file(ledger) == closing["sha256"]
    requests = list((run / "requests").glob("*.jsonl")) if (run / "requests").exists() else []
    cjk = len(re.findall(r"[\u4e00-\u9fff]", proposal.split("## 参考文献")[0]))
    unread = [row["id"] for row in literature if row["read_level"] == "citation_unconfirmed"]
    prior = json.loads((run / "audit" / "checks.json").read_text(encoding="utf-8")) if (run / "audit" / "checks.json").exists() else {}
    checks = {
        "j0": prior,
        "missing_claim_ids": missing,
        "parent_ledger_unchanged": ledger_same,
        "new_request_files": [str(path) for path in requests],
        "new_scientific_model_requests": 0 if not requests else len(requests),
        "cjk_before_references": cjk,
        "unread_literature": unread,
        "formal_sampling_status": "not_started",
        "human_calibration_status": "pending_reviewer",
    }
    write_json(run / "audit" / "checks.json", checks)
    (run / "reports" / "proposal_basis.md").write_text(proposal, encoding="utf-8")
    summary = "\n".join([
        "# 核查摘要",
        "",
        f"父账本未改写：{ledger_same}。新的科学模型请求：{checks['new_scientific_model_requests']}。",
        "",
        pmid_sentence(counts),
        role_sentence(counts),
        case_sentence(counts),
        f"年份双侧非空采用 {counts.get('year_both', {}).get('used')}，旧值 {STALE_YEAR_BOTH} 不替代当前结果。五类重数与锚点一致：{counts.get('five_match')}。",
        "",
        "纠正：readiness 的 new_case_mapped=24 是映射行数；独立 case_id 以去重结果为准。5,030 条不是摘要信息不足的人工结论。E6 字符串差异不是方向翻转。74 条定位不是语义支持。",
        "",
        "未解决：ASQ 方法本轮未读到。CORE 全文页返回 403，四属性定义以本地 schema.owl 为准。实体表 sentence 连接仍 unresolved。评阅者未指定。正式抽样未开始。历史 pilot 仍是 incomplete。",
        "",
        "答辩边界：没有人工核对时，只保留材料事实、程序行为和未核对的模型观察。两个库共现不是两次独立验证。结构化字段的存在不证明优于直接阅读摘要。",
        "",
        "下一步是据此修改 proposal，不自动进入新实验。",
        "",
    ])
    (run / "reports" / "verification_summary.md").write_text(summary, encoding="utf-8")
    write_json(run / "manifest.json", {
        "run_id": run.name,
        "parent_run_id": PARENT.name,
        "science_version": "proposal_basis_v1",
        "proposal_evidence_package_status": "completed" if not missing and ledger_same and counts.get("core_match") and counts.get("five_match") else "partial",
        "literature_verification_status": "partial" if unread else "verified_for_used_claims",
        "human_calibration_status": "pending_reviewer",
        "formal_sampling_status": "not_started",
        "formal_study_status": "not_started",
        "historical_pilot_status": "incomplete",
        "new_scientific_model_requests": checks["new_scientific_model_requests"],
        "updated_utc": _now(),
    })
    write_json(run / "jobs" / "j1.json", {"utc": _now(), "missing_claims": missing, "cjk": cjk})
    return 0 if not missing and ledger_same and counts.get("core_match") and counts.get("five_match") else 2


STAGES = {"j0": stage_j0, "j1": stage_j1}


def main() -> int:
    if _login():
        raise SystemExit("refusing to run proposal basis on the login node")
    try:
        return STAGES[sys.argv[1]](os.environ["PILOT_CHILD_RUN"])
    except Exception:
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

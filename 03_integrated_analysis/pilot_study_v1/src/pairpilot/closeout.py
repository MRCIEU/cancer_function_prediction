#!/usr/bin/env python3
"""Closeout child of diagnostic_v1. It does not rerun extraction or the 5812 comparisons."""

from __future__ import annotations

import hashlib
import re
import json
import os
import sys
import time
import traceback
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pairpilot.diagnostic_round import _call_task, _efetch, _rows  # noqa: E402
from pairpilot.inference import CallCache, input_hash  # noqa: E402
from pairpilot.ledger import LIMITS, available_usd  # noqa: E402
from pairpilot.query_ast import (  # noqa: E402
    VERSION,
    and_query,
    candidate_outcome,
    choose_new,
    class_counts,
    group,
    paper_pair_key,
    pmid_source,
    render,
    source_membership,
    truncation_record,
    _term,
)
from pairpilot.store import append_jsonl, query_rows, read_json, write_json, write_parquet  # noqa: E402
from pairpilot.store import connect  # noqa: E402

DIAG1 = Path("/projects/b5ac/project_2/results/03_integrated_analysis/pilot_study_v1/runs/20261016T013500Z_diag1")
REV2 = Path("/projects/b5ac/project_2/results/03_integrated_analysis/pilot_study_v1/runs/20261016T005500Z_rev2")
PRIOR = Path("/projects/b5ac/project_2/results/03_integrated_analysis/pilot_study_v1/runs/20261015T221200Z_b20cont")
PREP = Path("/projects/b5ac/project_2/results/03_integrated_analysis/pilot_study_v1/runs/20261015T204633Z_0d9b9724")
RUNS = DIAG1.parent
COMPLIANT_CAP = 30
RETMAX = 50
TERMS = {
    "g07": {
        "pair_id": "3090__C0006142",
        "gene_id": "3090",
        "condition_label": "triple_negative_subtype",
        "gene": [("HIC1", "ncbi_gene_symbol"), ("HIC-1", "pair_names.gene_name")],
        "disease": [("breast cancer", "pair_names.disease_name", "tiab"), ("breast cancers", "abstract_surface_pmid_19015639", "tiab"), ("breast neoplasms", "mesh_name_for_pair_disease", "mesh")],
        "condition": [("TNBC", "frozen_candidate.condition", "tiab"), ("triple-negative", "frozen_candidate.condition", "tiab"), ("triple negative", "frozen_candidate.condition", "tiab")],
    },
    "g16": {
        "pair_id": "8318__C2239176",
        "gene_id": "8318",
        "condition_label": "hbv_status",
        "gene": [("CDC45", "ncbi_gene_symbol"), ("Cdc45", "pair_names.gene_name")],
        "disease": [("hepatocellular carcinoma", "pair_disease_expanded", "tiab"), ("HCC", "pair_names.disease_name", "tiab"), ("carcinoma, hepatocellular", "mesh_name_for_pair_disease", "mesh")],
        "condition": [("hepatitis B", "frozen_candidate.condition", "tiab"), ("HBV", "frozen_candidate.condition", "tiab")],
    },
    "g13": {
        "pair_id": "768__C0376358",
        "gene_id": "768",
        "condition_label": "hypoxia",
        "gene": [("CA9", "ncbi_gene_symbol"), ("CA IX", "pair_names.gene_name"), ("CAIX", "pair_names.gene_name"), ("carbonic anhydrase IX", "pair_names.gene_name")],
        "disease": [("prostate cancer", "pair_names.disease_name", "tiab"), ("prostatic neoplasms", "mesh_name_for_pair_disease", "mesh")],
        "condition": [("hypoxia", "frozen_candidate.condition", "tiab"), ("hypoxic", "frozen_candidate.condition", "tiab")],
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _login() -> bool:
    return os.uname().nodename.startswith("login")


def _run() -> Path:
    path = RUNS / os.environ["PILOT_CHILD_RUN"]
    if not path.exists():
        raise SystemExit(f"closeout run is missing: {path}")
    return path


def _short(value, limit: int = 400) -> str:
    return " ".join(str(value or "").split())[:limit]


def _con():
    tmp = Path(os.environ.get("SLURM_TMPDIR") or f"/tmp/{os.environ.get('USER', 'pilot')}/pairpilot")
    return connect(threads=2, memory="8GB", tmp=tmp)


def init_run(run_id: str) -> Path:
    run = RUNS / run_id
    run.mkdir(parents=False, exist_ok=True)
    for name in ("audit", "tables", "experiments", "reports", "requests", "responses", "jobs"):
        (run / name).mkdir(exist_ok=True)
    if not (run / "manifest.json").exists():
        write_json(run / "manifest.json", {
            "run_id": run_id,
            "parent_run_id": DIAG1.name,
            "rev2_run_id": REV2.name,
            "model_run_id": PRIOR.name,
            "preparation_run_id": PREP.name,
            "science_version": "closeout_v1",
            "query_version": VERSION,
            "baseline_rule": "comparison_gap_v2",
            "created_utc": _now(),
            "status": "incomplete",
            "e5_status": "not_started",
            "pilot_status": "incomplete",
        })
    ledger = run / "audit" / "ledger.json"
    if not ledger.exists():
        prior = json.loads((DIAG1 / "audit" / "ledger.json").read_text(encoding="utf-8"))
        available = available_usd(prior)
        spent = float(prior["spent_usd"])
        hold = None if available is None else LIMITS["max_usd"] - spent - available
        write_json(ledger, {
            "spent_usd": spent,
            "reserved_usd": 0.0,
            "billing_unknown_usd": hold,
            "attempts": prior.get("attempts") or 0,
            "input_tokens": prior.get("input_tokens") or 0,
            "output_tokens": prior.get("output_tokens") or 0,
            "canary_spent_usd": prior.get("canary_spent_usd") or 0.0,
            "canary_reserved_usd": 0.0,
            "canary_attempts": prior.get("canary_attempts") or 0,
            "canary_input_tokens": prior.get("canary_input_tokens") or 0,
            "canary_output_tokens": prior.get("canary_output_tokens") or 0,
            "reservations": {},
            "opening_source": str(DIAG1 / "audit" / "ledger.json"),
            "opened_utc": _now(),
        })
    return run


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen_candidates() -> list[dict]:
    rows = read_json(DIAG1 / "audit" / "e5_frozen.json")
    by_id = {row["candidate_id"]: row for row in rows}
    if set(by_id) != set(TERMS):
        raise SystemExit(f"frozen candidate ids {sorted(by_id)} do not match the closeout map")
    for candidate_id, spec in TERMS.items():
        if by_id[candidate_id].get("pair_id") != spec["pair_id"]:
            raise SystemExit(f"{candidate_id} pair_id changed")
    return [by_id[candidate_id] for candidate_id in ("g07", "g16", "g13")]


def _pair_sets() -> dict[str, dict[str, set[str]]]:
    sets = {}
    for row in _rows(PREP / "inputs" / "pair_pmids.parquet"):
        slot = sets.setdefault(row["pair_id"], {"CORE": set(), "SemMedDB": set()})
        slot[row["source"]].add(str(row["pmid"]))
    return sets


def _build_query(spec: dict, extra_genes: list[tuple[str, str]], use_fields: bool) -> dict:
    genes = []
    for text, source in list(spec["gene"]) + extra_genes:
        genes.append(_term(text, "gene", source, "tiab" if use_fields else "", text, "strip_space"))
    diseases = []
    for text, source, field in spec["disease"]:
        diseases.append(_term(text, "disease", source, field if use_fields else "", text, "strip_space"))
    conditions = []
    for text, source, field in spec["condition"]:
        conditions.append(_term(text, "condition", source, field if use_fields else "", text, "strip_space"))
    broad = and_query("broad", [group("gene", genes), group("disease", diseases)])
    specific = and_query("specific", [group("gene", genes), group("disease", diseases), group("condition", conditions)])
    return {"broad": broad, "specific": specific}


def _esearch_detail(term: str, retmax: int = RETMAX) -> dict:
    params = {"db": "pubmed", "term": term, "retmax": retmax, "retmode": "json", "sort": "relevance"}
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urllib.parse.urlencode(params)
    if os.environ.get("NCBI_API_KEY"):
        url += "&api_key=" + urllib.parse.quote(os.environ["NCBI_API_KEY"])
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "pairpilot/1.0"}), timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return {"status": "fetch_failed", "error": type(exc).__name__, "term": term}
    result = payload.get("esearchresult") or {}
    count_text = result.get("count")
    count = int(count_text) if str(count_text or "").isdigit() else None
    ids = [str(pmid) for pmid in result.get("idlist") or []]
    translation = result.get("querytranslation") or ""
    record = truncation_record(count, len(ids), retmax)
    record.update({
        "status": "ok",
        "term": term,
        "querytranslation": translation or "not_returned",
        "warninglist": result.get("warninglist") or {},
        "idlist": ids,
        "sort": "relevance",
    })
    return record


def _gene_aliases(gene_id: str) -> list[dict]:
    params = {"db": "gene", "id": gene_id, "retmode": "json"}
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?" + urllib.parse.urlencode(params)
    if os.environ.get("NCBI_API_KEY"):
        url += "&api_key=" + urllib.parse.quote(os.environ["NCBI_API_KEY"])
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "pairpilot/1.0"}), timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return [{"gene_id": gene_id, "status": "fetch_failed", "error": type(exc).__name__}]
    result = (payload.get("result") or {}).get(gene_id) or {}
    aliases = []
    for raw in str(result.get("otheraliases") or "").replace(";", ",").split(","):
        text = " ".join(raw.split())
        if len(text) < 2 or any(mark in text.lower() for mark in (" ligand", "-fc", " fc", "fusion")):
            continue
        aliases.append({"text": text, "source": "ncbi_gene.esummary.otheraliases", "symbol": result.get("name") or "", "status": "ok"})
    if result.get("name"):
        aliases.insert(0, {"text": result["name"], "source": "ncbi_gene.esummary.name", "symbol": result["name"], "status": "ok"})
    return aliases


def _term_in_text(text: str, term: str) -> bool:
    return re.search(r"(?<![a-z0-9])" + re.escape(term.lower()) + r"(?![a-z0-9])", text.lower()) is not None


def _known_items(spec: dict, query: dict, documents: dict[str, dict], pmids: set[str]) -> list[dict]:
    gene_terms = [term["text"].lower() for term in query["groups"][0]["terms"]]
    disease_terms = [term["text"].lower() for term in query["groups"][1]["terms"]]
    found = []
    for pmid in sorted(pmids):
        doc = documents.get(pmid) or {}
        text = f"{doc.get('title') or ''} {doc.get('text') or ''}".lower()
        gene_hit = next((term for term in gene_terms if _term_in_text(text, term)), "")
        disease_hit = next((term for term in disease_terms if _term_in_text(text, term)), "")
        if gene_hit and disease_hit:
            found.append({
                "pmid": pmid,
                "gene_term": gene_hit,
                "disease_term": disease_hit,
                "text_version_id": doc.get("text_version_id") or "",
                "why": "title or abstract contains a gene term and a disease term from this query",
            })
        if len(found) == 3:
            break
    return found


def _historical_deep_reads() -> dict:
    pmids = set()
    files = []
    for name in ("e5_extracts.parquet", "e5_extracts_invalid_alias.parquet"):
        path = DIAG1 / "experiments" / name
        if not path.exists():
            continue
        rows = _rows(path)
        got = sorted({str(row.get("pmid")) for row in rows if row.get("pmid")})
        files.append({"file": str(path), "rows": len(rows), "pmids": got})
        pmids.update(got)
    return {"files": files, "unique_pmids": sorted(pmids), "n_unique": len(pmids), "compliant_cap_this_round": COMPLIANT_CAP}


def stage_j0(run_id: str) -> int:
    run = init_run(run_id)
    frozen = _frozen_candidates()
    write_json(run / "audit" / "frozen_candidates.json", frozen)
    registry = {}
    for label, path in {
        "diag1_comparisons": DIAG1 / "tables" / "diagnostic_comparisons.parquet",
        "diag1_selected": DIAG1 / "tables" / "comparison_selected.parquet",
        "diag1_findings": DIAG1 / "tables" / "diagnostic_findings.parquet",
        "diag1_frozen": DIAG1 / "audit" / "e5_frozen.json",
        "pair_pmids": PREP / "inputs" / "pair_pmids.parquet",
        "documents": PREP / "inputs" / "documents.parquet",
    }.items():
        registry[label] = {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
    write_json(run / "audit" / "input_registry.json", registry)
    sets = _pair_sets()
    findings = {(row["pair_id"], str(row["pmid"]), str(row["local"])): row for row in _rows(DIAG1 / "tables" / "diagnostic_findings.parquet")}
    docs = {str(row["pmid"]): row for row in _rows(PREP / "inputs" / "documents.parquet")}
    selected = {row["comparison_id"]: row for row in _rows(DIAG1 / "tables" / "comparison_selected.parquet")}
    membership = []
    casebook = []
    for row in _rows(DIAG1 / "tables" / "diagnostic_comparisons.parquet"):
        key = row["candidate_key"]
        meta = selected.get(key) or selected.get(row.get("comparison_id")) or {}
        pmid_a, pmid_b = str(row["pmid_a"]), str(row["pmid_b"])
        pair = row["pair_id"]
        core, semmed = sets[pair]["CORE"], sets[pair]["SemMedDB"]
        needed = {pmid_a, pmid_b}
        member = source_membership(needed, core, semmed)
        member.update({
            "comparison_id": key,
            "pair_id": pair,
            "pmid_a": pmid_a,
            "pmid_b": pmid_b,
            "paper_pair": paper_pair_key(pmid_a, pmid_b),
            "primary_question_type": row.get("primary_question_type") or "",
            "required_evidence_pmids": ",".join(sorted(needed)),
            "model_input_pmids": ",".join(sorted(needed)),
            "model_input_source": "reconstructed_two_abstracts_request_body_not_logged",
            "outside_view_context": "",
            "pmid_a_source": pmid_source(pmid_a, core, semmed),
            "pmid_b_source": pmid_source(pmid_b, core, semmed),
        })
        membership.append(member)
        left = findings.get((pair, pmid_a, str(meta.get("finding_a") or "").split(":")[-1]), {})
        right = findings.get((pair, pmid_b, str(meta.get("finding_b") or "").split(":")[-1]), {})
        roles = {left.get("discourse_role") or "", right.get("discourse_role") or ""}
        casebook.append({
            "comparison_id": key,
            "pair_id": pair,
            "pmid_a": pmid_a,
            "pmid_b": pmid_b,
            "title_a": _short((docs.get(pmid_a) or {}).get("title"), 240),
            "title_b": _short((docs.get(pmid_b) or {}).get("title"), 240),
            "finding_a": meta.get("finding_a") or "",
            "finding_b": meta.get("finding_b") or "",
            "text_version_a": left.get("text_version_id") or "",
            "text_version_b": right.get("text_version_id") or "",
            "span": "not_recorded",
            "measured_a": left.get("measured_entity") or "",
            "measured_b": right.get("measured_entity") or "",
            "intervened_a": left.get("intervened_entity") or "",
            "intervened_b": right.get("intervened_entity") or "",
            "entity_kind_a": left.get("entity_kind") or "",
            "entity_kind_b": right.get("entity_kind") or "",
            "entity_gene_a": left.get("entity_gene_id") or "",
            "entity_gene_b": right.get("entity_gene_id") or "",
            "discourse_role_a": left.get("discourse_role") or "",
            "discourse_role_b": right.get("discourse_role") or "",
            "author_claim_in_comparison": "yes" if roles & {"author_interpretation", "recommendation"} else "no",
            "shared_question": _short(row.get("shared_question"), 500),
            "relation_description": _short(row.get("relation_description"), 700),
            "study_system_a": left.get("study_system") or "",
            "study_system_b": right.get("study_system") or "",
            "disease_a": left.get("disease_site_or_subtype") or "",
            "disease_b": right.get("disease_site_or_subtype") or "",
            "exposure_a": left.get("exposure_or_intervention") or "",
            "exposure_b": right.get("exposure_or_intervention") or "",
            "comparator_a": left.get("comparator") or "",
            "comparator_b": right.get("comparator") or "",
            "outcome_a": left.get("outcome") or "",
            "outcome_b": right.get("outcome") or "",
            "compartment_a": left.get("measurement_compartment") or "",
            "compartment_b": right.get("measurement_compartment") or "",
            "dimensions": row.get("dimensions") or "",
            "can_describe_relation": row.get("can_describe_relation"),
            "can_compare_direction": row.get("can_compare_direction"),
            "can_compare_magnitude": row.get("can_compare_magnitude"),
            "direction_relation": row.get("direction_relation") or "",
            "missing_information": _short(row.get("missing_information"), 700),
            "why_it_blocks": _short(row.get("why_it_blocks"), 500),
            "inference_limit": _short(row.get("inference_limit"), 400),
            "quote_a": left.get("quote") or meta.get("quote_a") or "",
            "quote_b": right.get("quote") or meta.get("quote_b") or "",
            "source_class": member["mutually_exclusive_source_class"],
            "judgement_limit": "model classification on the diagnostic subset; request body was not retained",
        })
    write_parquet(run / "tables" / "comparison_source_membership.parquet", membership)
    write_parquet(run / "tables" / "comparison_casebook.parquet", casebook)
    by_pair = {}
    for row in membership:
        by_pair.setdefault(row["pair_id"], []).append(row)
    summary = {}
    for pair, rows in by_pair.items():
        counts = class_counts(rows)
        papers = {row["paper_pair"] for row in rows}
        questions = {row["primary_question_type"] for row in rows}
        summary[pair] = {
            **counts,
            "n_comparison": len(rows),
            "n_unique_paper_pairs": len(papers),
            "n_question_types": len(questions),
            "question_types": sorted(questions),
        }
    flags = {
        "n_casebook": len(casebook),
        "describe_true": sum(1 for row in casebook if row["can_describe_relation"] in (True, "true")),
        "direction_true": sum(1 for row in casebook if row["can_compare_direction"] in (True, "true")),
        "magnitude_true": sum(1 for row in casebook if row["can_compare_magnitude"] in (True, "true")),
        "author_claim_rows": sum(1 for row in casebook if row["author_claim_in_comparison"] == "yes"),
        "missing_quote": sum(1 for row in casebook if not row["quote_a"] or not row["quote_b"]),
    }
    write_json(run / "audit" / "source_summary.json", {"by_pair": summary, "flags": flags, "historical_deep_reads": _historical_deep_reads()})
    write_json(run / "audit" / "query_defects.json", {
        "pair_id_in_query": str(DIAG1 / "experiments" / "e5_search_invalid_alias.parquet"),
        "disease_or_bypassed_gene": str(DIAG1 / "experiments" / "e5_search.parquet"),
        "note": "Those searches stay archived on the parent. Their candidate_state values are outputs about the old material, not a result of query_builder_v2.",
    })
    _write_casebook_md(run, casebook)
    return 0 if len(membership) == 24 and len(casebook) == 24 else 2


def _write_casebook_md(run: Path, rows: list[dict]) -> None:
    lines = ["# 24 comparison dossiers", "", "这些判断来自 diagnostic_v1 已保存的模型输出。本文件没有重新调用模型。", ""]
    for row in rows:
        lines.extend([
            f"## {row['comparison_id']}",
            "",
            f"pair `{row['pair_id']}`，PMID {row['pmid_a']} 与 {row['pmid_b']}。来源类别 `{row['source_class']}`。",
            f"题目 A: {row['title_a'] or 'missing'}",
            f"题目 B: {row['title_b'] or 'missing'}",
            f"finding {row['finding_a']} / {row['finding_b']}。discourse_role {row['discourse_role_a'] or 'missing'} / {row['discourse_role_b'] or 'missing'}。作者解释进入比较：{row['author_claim_in_comparison']}。",
            f"测量实体 {row['measured_a'] or 'missing'} ({row['entity_kind_a'] or 'unclassified'})；{row['measured_b'] or 'missing'} ({row['entity_kind_b'] or 'unclassified'})。",
            f"问题：{row['shared_question'] or 'missing'}",
            f"关系：{row['relation_description'] or 'missing'}",
            f"系统 {row['study_system_a'] or 'missing'} | {row['study_system_b'] or 'missing'}",
            f"疾病 {row['disease_a'] or 'missing'} | {row['disease_b'] or 'missing'}",
            f"暴露 {row['exposure_a'] or 'missing'} | {row['exposure_b'] or 'missing'}",
            f"比较对象 {row['comparator_a'] or 'missing'} | {row['comparator_b'] or 'missing'}",
            f"结局 {row['outcome_a'] or 'missing'} | {row['outcome_b'] or 'missing'}",
            f"区室 {row['compartment_a'] or 'missing'} | {row['compartment_b'] or 'missing'}",
            f"维度：{row['dimensions'] or 'missing'}",
            f"描述关系 {row['can_describe_relation']}；对照方向 {row['can_compare_direction']}；比较数值 {row['can_compare_magnitude']}；方向标记 {row['direction_relation'] or 'missing'}。",
            f"缺失：{row['missing_information'] or 'missing'}。限制：{row['why_it_blocks'] or row['inference_limit'] or 'missing'}。",
            f"文本版本 {row['text_version_a'] or 'missing'} / {row['text_version_b'] or 'missing'}。span {row['span']}。",
            f"A: \"{row['quote_a']}\"" if row["quote_a"] else "A quote missing",
            f"B: \"{row['quote_b']}\"" if row["quote_b"] else "B quote missing",
            "",
        ])
    (run / "reports" / "casebook.md").write_text("\n".join(lines), encoding="utf-8")


def stage_j1(run_id: str) -> int:
    run = _run()
    frozen = {row["candidate_id"]: row for row in _frozen_candidates()}
    documents = {str(row["pmid"]): row for row in _rows(PREP / "inputs" / "documents.parquet")}
    sets = _pair_sets()
    manifest = []
    checks = []
    hits = {}
    kept = {}
    if os.environ.get("CLOSEOUT_RETRY_UNRESOLVED") == "1" and (run / "audit" / "query_manifest.json").exists():
        for row in read_json(run / "audit" / "query_manifest.json"):
            if row.get("execution_status") in {"valid", "valid_untested"}:
                kept.setdefault(row["candidate_id"], []).append(row)
        prior_hits = read_json(run / "audit" / "query_hits.json")
    else:
        prior_hits = {}
    if kept and (run / "audit" / "known_item_checks.json").exists():
        checks.extend(row for row in read_json(run / "audit" / "known_item_checks.json") if row.get("candidate_id") in kept)
    for candidate_id, spec in TERMS.items():
        if candidate_id in kept:
            manifest.extend(kept[candidate_id])
            for row in kept[candidate_id]:
                hits[f"{candidate_id}|{row['kind']}"] = prior_hits.get(f"{candidate_id}|{row['kind']}") or []
            continue
        alias_rows = _gene_aliases(spec["gene_id"])
        time.sleep(0.15)
        extra = []
        seen = {text.lower() for text, _source in spec["gene"]}
        for alias in alias_rows:
            text = alias.get("text") or ""
            if alias.get("status") != "ok" or text.lower() in seen:
                continue
            seen.add(text.lower())
            extra.append((text, alias["source"]))
        history = []
        final = None
        for revision, use_fields, use_extra in ((0, True, False), (1, True, True), (2, False, True)):
            built = _build_query(spec, extra if use_extra else [], use_fields)
            broad_text = render(built["broad"])
            known = _known_items(spec, built["broad"], documents, sets[spec["pair_id"]]["CORE"] | sets[spec["pair_id"]]["SemMedDB"])
            if not known:
                check = {"candidate_id": candidate_id, "revision": revision, "status": "no_local_abstract_with_both_terms", "known": []}
            else:
                probe = and_query("broad", built["broad"]["groups"], [item["pmid"] for item in known])
                detail = _esearch_detail(render(probe), retmax=len(known))
                time.sleep(0.15)
                returned = set(detail.get("idlist") or [])
                expected = {item["pmid"] for item in known}
                check = {
                    "candidate_id": candidate_id,
                    "revision": revision,
                    "status": "hit" if detail.get("status") == "ok" and expected <= returned else "miss",
                    "expected": sorted(expected),
                    "returned": sorted(returned),
                    "query": render(probe),
                    "querytranslation": detail.get("querytranslation") or "not_returned",
                    "known": known,
                }
            history.append({"revision": revision, "use_fields": use_fields, "broad": broad_text, "check": check["status"]})
            checks.append(check)
            final = (built, check)
            if check["status"] in {"hit", "no_local_abstract_with_both_terms"}:
                break
            if revision == 2:
                break
        built, check = final
        status = "valid" if check["status"] == "hit" else ("unresolved" if check["status"] == "miss" else "valid_untested")
        for kind in ("broad", "specific"):
            text = render(built[kind])
            detail = _esearch_detail(text, RETMAX)
            time.sleep(0.15)
            record = {
                "candidate_id": candidate_id,
                "pair_id": spec["pair_id"],
                "target_question": frozen[candidate_id].get("target_question"),
                "kind": kind,
                "query_version": VERSION,
                "execution_status": status,
                "condition_label": spec["condition_label"],
                "query": text,
                "ast": built[kind],
                "revision_history": history,
                "hit_count": detail.get("hit_count"),
                "hit_count_status": detail.get("hit_count_status"),
                "retmax": RETMAX,
                "returned_pmid_n": detail.get("returned_pmid_n"),
                "truncated": detail.get("truncated"),
                "querytranslation": detail.get("querytranslation") or "not_returned",
                "warninglist": detail.get("warninglist") or {},
                "search_status": detail.get("status"),
            }
            manifest.append(record)
            hits[f"{candidate_id}|{kind}"] = detail.get("idlist") or []
    write_json(run / "audit" / "query_manifest.json", manifest)
    write_json(run / "audit" / "known_item_checks.json", checks)
    write_json(run / "audit" / "query_hits.json", hits)
    write_parquet(run / "tables" / "query_manifest.parquet", [
        {key: value for key, value in row.items() if key not in {"ast", "revision_history", "warninglist"}}
        | {"ast": json.dumps(row["ast"], ensure_ascii=False), "revision_history": json.dumps(row["revision_history"], ensure_ascii=False)}
        for row in manifest
    ])
    return 0


def _parent_cache_hit(task: str, user: str, prompt: str, cap: int) -> dict | None:
    cache = CallCache(DIAG1 / "responses" / "cache")
    request = {
        "task": task,
        "input_sha256": input_hash(user),
        "schema_id": task,
        "prompt_sha256": input_hash(prompt),
        "model_id": "gpt-6.1-sol",
        "params": {"max_completion_tokens": cap, "service_tier": "standard"},
        "replicate_id": None,
    }
    cached = cache.get(cache.key(request))
    if cached and cached.get("status") == "ok":
        cached["cache_hit"] = True
        cached["cache_source"] = str(DIAG1 / "responses" / "cache")
        return cached
    return None


def stage_j2(run_id: str) -> int:
    run = _run()
    manifest = read_json(run / "audit" / "query_manifest.json")
    hits = read_json(run / "audit" / "query_hits.json")
    frozen = {row["candidate_id"]: row for row in _frozen_candidates()}
    baseline = {str(row["pmid"]) for row in _rows(PREP / "inputs" / "pair_pmids.parquet")}
    from pairpilot.continue_pilot import _prompt
    prompt = _prompt("diagnostic_reread.txt")
    retry = os.environ.get("CLOSEOUT_RETRY_UNRESOLVED") == "1"
    prior_extracts = _rows(run / "experiments" / "supplemental_findings.parquet") if retry and (run / "experiments" / "supplemental_findings.parquet").exists() else []
    done_candidates = {row.get("candidate_id") for row in prior_extracts if row.get("extract_status") == "ok"}
    flow = []
    included_rows = []
    new_pmids = [row.get("pmid") for row in prior_extracts if row.get("status") == "included" and row.get("pmid")]
    for candidate_id in ("g07", "g16", "g13"):
        rows = [row for row in manifest if row["candidate_id"] == candidate_id]
        if retry and candidate_id in done_candidates:
            continue
        if rows[0]["execution_status"] not in {"valid", "valid_untested"}:
            flow.append({"candidate_id": candidate_id, "retrieval_status": "not_started", "reason": "query_unresolved"})
            continue
        blocked = set(baseline)
        for row in rows:
            ranked = hits.get(f"{candidate_id}|{row['kind']}") or []
            room = min(5, COMPLIANT_CAP - len(new_pmids))
            chosen = choose_new(ranked, blocked, room)
            for pmid in chosen["included"]:
                blocked.add(pmid)
                new_pmids.append(pmid)
                included_rows.append({
                    "candidate_id": candidate_id,
                    "pair_id": row["pair_id"],
                    "query_kind": row["kind"],
                    "pmid": pmid,
                    "status": "included",
                    "inclusion_path": f"{VERSION}|{candidate_id}|{row['kind']}|rank",
                })
            flow.append({
                "candidate_id": candidate_id,
                "query_kind": row["kind"],
                "hit_count": row.get("hit_count"),
                "hit_count_status": row.get("hit_count_status"),
                "retmax": RETMAX,
                "returned_pmid_n": row.get("returned_pmid_n"),
                "truncated": row.get("truncated"),
                "metadata_n": len(ranked),
                "already_in_baseline_or_earlier_query": len(chosen["already_present"]),
                "included_n": len(chosen["included"]),
                "not_taken_in_top": len(chosen["not_taken_in_top"]),
                "execution_status": row["execution_status"],
            })
    prior_flow = _rows(run / "tables" / "retrieval_flow.parquet") if retry and (run / "tables" / "retrieval_flow.parquet").exists() else []
    write_parquet(run / "tables" / "retrieval_flow.parquet", prior_flow + flow)
    texts = _efetch([row["pmid"] for row in included_rows])
    years = _pub_years([row["pmid"] for row in included_rows])
    extracts = []
    stopped = False
    for row in included_rows:
        doc = texts.get(row["pmid"])
        if not doc:
            extracts.append({**row, "extract_status": "fetch_failed"})
            continue
        user = json.dumps({
            "pair_id": row["pair_id"], "pmid": row["pmid"], "title": doc.get("title"),
            "abstract": doc.get("abstract"), "supplemental_search": True,
        }, ensure_ascii=False)
        cached = _parent_cache_hit("diagnostic_reread", user, prompt, 5000)
        result = cached or _call_task(run, "diagnostic_reread", user, "diagnostic_reread.txt", f"closeout|{row['pmid']}", 5000)
        extracts.append({
            **row,
            "extract_status": result.get("status"),
            "cache_hit": bool(result.get("cache_hit")),
            "title": doc.get("title") or "",
            "year_flag": years.get(row["pmid"], "not_recorded"),
            "abstract": _short(doc.get("abstract"), 1200),
        })
        if result.get("status") in {"quota_exhausted", "not_evaluated_budget", "stopped_after_quota"}:
            stopped = True
            break
    write_parquet(run / "experiments" / "supplemental_findings.parquet", prior_extracts + extracts)
    reviews = []
    prior_reviews = {row.get("candidate_id"): row for row in (_rows(run / "tables" / "candidate_reassessment.parquet") if retry and (run / "tables" / "candidate_reassessment.parquet").exists() else [])}
    if not stopped:
        for candidate_id in ("g07", "g16", "g13"):
            query_status = next(row["execution_status"] for row in manifest if row["candidate_id"] == candidate_id)
            if retry and candidate_id in prior_reviews and prior_reviews[candidate_id].get("call_status") == "ok" and candidate_id in done_candidates:
                reviews.append(prior_reviews[candidate_id])
                continue
            own = [row for row in prior_extracts + extracts if row["candidate_id"] == candidate_id and row.get("extract_status") == "ok"]
            included_n = sum(1 for row in included_rows if row["candidate_id"] == candidate_id)
            if query_status not in {"valid", "valid_untested"}:
                retrieval = "not_started"
            elif included_n == 0:
                retrieval = "zero_hit" if all((row.get("hit_count") or 0) == 0 for row in manifest if row["candidate_id"] == candidate_id) else "no_new_material"
            elif not own:
                retrieval = "fetch_failed"
            else:
                retrieval = "bounded_with_new_material"
            user = json.dumps({
                "candidate": frozen[candidate_id],
                "query_execution_status": "valid" if query_status.startswith("valid") else query_status,
                "retrieval_status": retrieval,
                "new_abstracts": [{"pmid": row["pmid"], "title": row.get("title"), "abstract": row.get("abstract")} for row in own],
            }, ensure_ascii=False)
            if query_status.startswith("valid"):
                result = _call_task(run, "closeout_reassess", user, "closeout_reassess.txt", f"closeout-review|{candidate_id}", 4000)
                parsed = result.get("parsed") if isinstance(result.get("parsed"), dict) else {}
                model_state = parsed.get("candidate_state") or ""
                relation = parsed.get("new_material_relation") or ""
                reason = _short(parsed.get("reason"), 500)
                status = result.get("status")
            else:
                parsed, model_state, relation, reason, status = {}, "", "", "query_unresolved", "not_called"
            reviews.append({
                "candidate_id": candidate_id,
                "pair_id": TERMS[candidate_id]["pair_id"],
                "target_question": frozen[candidate_id].get("target_question"),
                "query_execution_status": "valid" if query_status.startswith("valid") else "unresolved",
                "retrieval_status": retrieval,
                "new_material_relation": relation,
                "model_candidate_state": model_state,
                "candidate_state": candidate_outcome("valid" if query_status.startswith("valid") else "unresolved", retrieval, model_state),
                "reason": reason,
                "call_status": status,
                "n_new_abstracts": len(own),
            })
            if status in {"quota_exhausted", "not_evaluated_budget"}:
                break
    write_parquet(run / "tables" / "candidate_reassessment.parquet", reviews)
    write_json(run / "audit" / "e5_status.json", {
        "status": "reviewed" if reviews and all(row["call_status"] in {"ok", "not_called"} for row in reviews) else "incomplete",
        "n_included": len(included_rows),
        "n_extract_ok": sum(1 for row in extracts if row.get("extract_status") == "ok"),
        "n_reviewed": sum(1 for row in reviews if row.get("call_status") == "ok"),
        "stopped_budget": stopped,
        "at": _now(),
    })
    return 2 if stopped else 0


def _pub_years(pmids: list[str]) -> dict[str, str]:
    out = {}
    clean = [pmid for pmid in pmids if pmid]
    for start in range(0, len(clean), 40):
        chunk = clean[start:start + 40]
        params = {"db": "pubmed", "id": ",".join(chunk), "retmode": "json"}
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?" + urllib.parse.urlencode(params)
        if os.environ.get("NCBI_API_KEY"):
            url += "&api_key=" + urllib.parse.quote(os.environ["NCBI_API_KEY"])
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "pairpilot/1.0"}), timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            continue
        result = payload.get("result") or {}
        for pmid, item in result.items():
            if pmid == "uids" or not isinstance(item, dict):
                continue
            year = str(item.get("pubdate") or "")[:4]
            if year.isdigit() and int(year) > 2024:
                out[pmid] = f"after_semmedVER43_2024_filename:{year}"
            elif year.isdigit():
                out[pmid] = year
            else:
                out[pmid] = "not_recorded"
    return out


def stage_j3(run_id: str) -> int:
    run = _run()
    source = read_json(run / "audit" / "source_summary.json") if (run / "audit" / "source_summary.json").exists() else {}
    manifest = read_json(run / "audit" / "query_manifest.json") if (run / "audit" / "query_manifest.json").exists() else []
    checks = read_json(run / "audit" / "known_item_checks.json") if (run / "audit" / "known_item_checks.json").exists() else []
    e5 = read_json(run / "audit" / "e5_status.json") if (run / "audit" / "e5_status.json").exists() else {}
    ledger = json.loads((run / "audit" / "ledger.json").read_text(encoding="utf-8"))
    reviews = _rows(run / "tables" / "candidate_reassessment.parquet") if (run / "tables" / "candidate_reassessment.parquet").exists() else []
    casebook = _rows(run / "tables" / "comparison_casebook.parquet") if (run / "tables" / "comparison_casebook.parquet").exists() else []
    union_rows = [row for row in casebook if row.get("source_class") == "union_required"]
    direction = next((row for row in casebook if row.get("can_compare_direction") in (True, "true")), None)
    blocked = next((row for row in casebook if row.get("can_describe_relation") in (True, "true") and row.get("can_compare_direction") not in (True, "true")), None)
    lines = [
        "# closeout_v1",
        "",
        f"child `{run.name}`，parent `{DIAG1.name}`。comparison_gap_v2 和 diagnostic_v1 的提取、比较定义没有改写。整个 pilot 仍是 incomplete。",
        "",
        "材料可形成性不是单库模型复现了同一判断。模型分类不是准确率。查询失败不是研究空白。",
        "",
        "## 1. 24 对比较的来源",
        "",
        _source_text(source),
        f"必须两库同时在场的比较有 {len(union_rows)} 对。",
        *_union_lines(union_rows),
        "",
        "## 2. 三个层次",
        "",
        f"档案 {len(casebook)} 对。模型判断可以描述关系 {source.get('flags', {}).get('describe_true')}，可以对照方向 {source.get('flags', {}).get('direction_true')}，可以直接比较数值 {source.get('flags', {}).get('magnitude_true')}。分母是这 24 对所选诊断比较，不是全部摘要。作者解释进入其中 {source.get('flags', {}).get('author_claim_rows')} 对；这些行保留，并标出它们不属于未来只比较本文结果的范围。原 13 条作者解释所在的 5812 对只是历史规则基线。",
        _example("可以对照方向", direction),
        _example("可以描述但不能对照方向", blocked),
        "缺效应尺度、比较定义或结局对应时，阻止的是数值比较。HBV 未分层只在该比较的缺失字段里出现，不是所有比较的否决条件。",
        "",
        "## 3. 三个候选的检索",
        "",
        *_query_lines(manifest, checks, reviews),
        _extract_text(run),
        f"作业状态记录：{e5 or 'not_started'}。其中 n_included 若只反映最后一次重跑，以补充材料表为准。历史深读 15 篇在 parent，不计入本轮合规集合，也不重置 30 篇上限。",
        "",
        "## 4. 已有诊断观察",
        "",
        "TNFRSF9：PMID 25631633 的 CD137L 记为基因 8744，CD137-Fc 记为干预构建体；PMID 33598101 与 35019173 的 TNFRSF9 或可溶性 CD137 记为 3604。本轮没有做这个 pair 的跨文章比较。",
        "E3 仍是来源句结果栏非空 13/16、未能判断 3/16，结论栏非空 8/16；摘要结果栏和结论栏非空 16/16。文本版本关系 not_recorded。这不是上下文长度带来的准确性提升。",
        "",
        "## 5. 正式任务定义与人工评价计划",
        "",
        *_spec_lines(),
        "",
        "## 6. 状态与费用",
        "",
        f"已结算 {ledger.get('spent_usd')}。计费不明 {ledger.get('billing_unknown_usd')}。在途 {ledger.get('reserved_usd')}。可用 {available_usd(ledger)}。attempt {ledger.get('attempts')} / {LIMITS['max_attempts']}。输入 token {ledger.get('input_tokens')} / {LIMITS['max_input_tokens']}。输出 token {ledger.get('output_tokens')} / {LIMITS['max_output_tokens']}。",
        "K、E6 仍是 deferred_pending_protocol。E4 和开放模型没有运行。本轮收尾可以完成，整个 pilot 不能写成 complete。",
        "",
    ]
    (run / "reports" / "report.md").write_text("\n".join(lines), encoding="utf-8")
    manifest_path = read_json(run / "manifest.json")
    recorded = {row.get("candidate_id") for row in reviews}
    unresolved = [row for row in reviews if row.get("query_execution_status") == "unresolved"]
    e5_done = recorded == {"g07", "g16", "g13"} and all(row.get("call_status") in {"ok", "not_called"} for row in reviews)
    if unresolved:
        e5_label = "partial_query_unresolved"
    elif e5_done:
        e5_label = "bounded_review_completed"
    else:
        e5_label = e5.get("status") or "incomplete"
    manifest_path["status"] = "closeout_completed" if source.get("flags", {}).get("n_casebook") == 24 and e5_done else "incomplete"
    manifest_path["e5_status"] = e5_label
    manifest_path["pilot_status"] = "incomplete"
    manifest_path["updated_utc"] = _now()
    write_json(run / "manifest.json", manifest_path)
    return 0


def _source_text(source: dict) -> str:
    lines = []
    for pair, row in (source.get("by_pair") or {}).items():
        lines.append(
            f"{pair}：比较 {row.get('n_comparison')}，唯一文章对 {row.get('n_unique_paper_pairs')}，问题类型 {row.get('n_question_types')}。"
            f"both_individually {row.get('both_individually')}，core_only {row.get('core_only')}，semmed_only {row.get('semmed_only')}，union_required {row.get('union_required')}，unresolved {row.get('unresolved')}。"
            f"CORE 可形成 {row.get('core_available')}，SemMedDB 可形成 {row.get('semmed_available')}，两库交集 x={row.get('intersection_x')}。"
        )
    return " ".join(lines) or "来源表尚未生成。"


def _union_lines(rows: list[dict]) -> list[str]:
    if not rows:
        return ["没有 union_required 比较，或档案尚未生成。"]
    lines = []
    for row in rows:
        lines.append(
            f"{row.get('comparison_id')}：PMID {row.get('pmid_a')} 与 {row.get('pmid_b')}。"
            f"问题：{_short(row.get('shared_question'), 220)}。"
            f"描述 {row.get('can_describe_relation')}，方向 {row.get('can_compare_direction')}，数值 {row.get('can_compare_magnitude')}。"
            f"仍缺：{_short(row.get('missing_information'), 220)}。"
        )
    return lines


def _example(label: str, row: dict | None) -> str:
    if not row:
        return f"{label}：档案里没有对应行。"
    return (
        f"{label}：PMID {row.get('pmid_a')} 与 {row.get('pmid_b')}，{row.get('primary_question_type') or row.get('shared_question')}。"
        f"{_short(row.get('relation_description'), 360)} 方向标记 {row.get('direction_relation') or 'missing'}。"
    )


def _extract_text(run: Path) -> str:
    path = run / "experiments" / "supplemental_findings.parquet"
    if not path.exists():
        return "补充深读表尚未生成。"
    rows = _rows(path)
    counts = Counter(row.get("candidate_id") for row in rows)
    after = sum(1 for row in rows if str(row.get("year_flag") or "").startswith("after_"))
    return f"本轮合规深读 {len(rows)} 篇，候选分布 {dict(counts)}。其中 {after} 篇发表年份晚于 semmedVER43_2024 文件名所示年份，单列，不称为当时 KG 漏收。"


def _query_lines(manifest: list[dict], checks: list[dict], reviews: list[dict]) -> list[str]:
    if not manifest:
        return ["查询清单尚未生成。"]
    lines = []
    by_candidate = {}
    for row in manifest:
        by_candidate.setdefault(row["candidate_id"], []).append(row)
    review_by_id = {row.get("candidate_id"): row for row in reviews}
    for candidate_id, rows in by_candidate.items():
        review = review_by_id.get(candidate_id) or {}
        queries = "；".join(
            f"{row['kind']} 命中 {row.get('hit_count')}（返回 {row.get('returned_pmid_n')}，截断 {row.get('truncated')}，解析 {('returned' if row.get('querytranslation') not in {'', 'not_returned', None} else 'not_returned')}）"
            for row in rows
        )
        lines.append(
            f"{candidate_id} / {rows[0]['pair_id']}：{rows[0].get('target_question')} "
            f"查询状态 {rows[0]['execution_status']}。{queries} "
            f"复核 {review.get('candidate_state') or 'not_started'}，新材料 {review.get('new_material_relation') or 'not_recorded'}。{review.get('reason') or ''}"
        )
    last = {}
    for row in checks:
        last[row["candidate_id"]] = row
    for candidate_id, row in last.items():
        lines.append(f"已知文章检查 {candidate_id}：{row.get('status')}，预期 {row.get('expected') or row.get('known')}。这不是召回率。")
    return lines


def _spec_lines() -> list[str]:
    return [
        "正式产品是面向研究者的 gene–cancer 证据比较档案。输入是一个已映射的基因–疾病 pair，以及该 pair 在 CORE、SemMedDB 或并集中的文章。排除内部 pair id、配体或试剂被当成靶基因、以及只凭 KG 标签作答。",
        "追溯链是原生 KG 记录、PMID、文本版本、finding、比较或候选。实体要分开记录被测量对象、被干预对象和试剂。本文结果进入实证比较；作者解释进入结果—推论任务；other 先确定具体问题，不整类删除。",
        "比较分三层。描述关系需要同一具体问题和可定位原文。对照方向还需要双方都报告了方向，并且未知不能当成相同。比较数值还需要效应尺度、比较定义和结局能够对应。潜在异质性单独列出，不自动否决描述。",
        "第二个库的贡献分成三档：多了文章、因而多了可以形成的比较、或只是限定了问题。没有单库对照调用时，只写材料可形成性。",
        "候选进入补充检索前要有明确问题、已有引文、缺失的联系和可执行查询。查询无效、有界检索没有新材料、以及读过的材料仍不能回答，三者分开写。读过的材料不能回答，只限于这次检索和这些摘要。",
        "当前材料可以支持可追溯的证据组织、所选 24 对上的分层比较，以及具体案例里两库材料能否形成比较。它还不能支持新的生物学关系、临床收益或方法准确率。",
        "人工评价先覆盖三类，而不是只看方向可对照的案例：方向可对照、方向不能对照、实体或角色边界。每条至少判断实体与角色、原文是否支持 finding、比较层次、未知和限制是否写对。这 24 对可以用来试标注规范，不能未经说明当成独立盲评集。更广的结论需要未参与调试的新案例。本轮不读论文、不填 gold、不计算准确率。",
    ]


STAGES = {"j0": stage_j0, "j1": stage_j1, "j2": stage_j2, "j3": stage_j3}


def main() -> int:
    if _login():
        raise SystemExit("refusing to run the closeout on the login node")
    stage = sys.argv[1]
    try:
        return STAGES[stage](os.environ["PILOT_CHILD_RUN"])
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

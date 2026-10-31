#!/usr/bin/env python3
"""V20: test the rules, check identifier meaning, and choose one coordinate rule."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import sys
import time
import unittest
import urllib.request

import test_rules
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE_ROOT = Path("/home/b5ac/freddieyu.b5ac/project_2")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CODE_ROOT))

from lib.duckutil import connect  # noqa: E402
from lib.runtime import prologue, setup_logging, write_json  # noqa: E402
from rules import (  # noqa: E402
    DOCUMENT_EVIDENCE,
    RULE_VERSION,
    THRESHOLDS,
    name_agrees,
    parse_argument,
    slice_document,
)

RESULTS = Path("/projects/b5ac/project_2/results")
OUT = RESULTS / "03_integrated_analysis" / "id_alignment_v2_round1"
NCBI_INFO_URLS = [
    "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_info.gz",
    "https://ftp.ncbi.nih.gov/gene/DATA/gene_info.gz",
]
NCBI_HISTORY_URLS = [
    "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_history.gz",
    "https://ftp.ncbi.nih.gov/gene/DATA/gene_history.gz",
]
PREDICATE = RESULTS / "02_semmeddb_validation" / "tables" / "predication.parquet"
SENTENCE = RESULTS / "02_semmeddb_validation" / "tables" / "sentence.parquet"
AUX = RESULTS / "02_semmeddb_validation" / "tables" / "predication_aux.parquet"
ENTITY = RESULTS / "02_semmeddb_validation" / "tables" / "entity.parquet"
ENTITY_SAMPLE = RESULTS / "03_integrated_analysis" / "data_analysis_review_round1" / "derived" / "entity_rowgroup_sample.parquet"
CASES = RESULTS / "03_integrated_analysis" / "data_analysis_review_round1" / "cases" / "cases.jsonl"
GENERIC = RESULTS / "02_semmeddb_validation" / "tables" / "generic_concept.parquet"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "derived").mkdir(parents=True, exist_ok=True)
    logger = setup_logging(OUT / "logs" / f"v20_{os.environ.get('SLURM_JOB_ID', 'local')}.log")
    info = prologue("V20", logger, OUT, [str(PREDICATE)])
    test_result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_rules))
    if not test_result.wasSuccessful():
        raise SystemExit("rule tests failed")
    logger.info("phase=rule_tests_passed tests=%s", test_result.testsRun)
    con = connect(OUT / "derived" / "v20.duckdb", "72GB", OUT / "tmp")
    ncbi = load_ncbi(con, logger)
    samples = namespace_samples(con, logger, ncbi["available"])
    coordinates = coordinate_check(con, logger)
    enabled = decide_enabled(samples, ncbi["available"])
    payload = {
        "rule_version": RULE_VERSION,
        "code_fingerprint": info["code_fingerprint"],
        "documents": DOCUMENT_EVIDENCE,
        "thresholds": THRESHOLDS,
        "ncbi": ncbi,
        "samples": samples,
        "coordinates": coordinates,
        "enabled": enabled,
        "column_order_note": "Document field order was not applied to the local CSV. Gene identifiers are taken from the predication argument text and from the already stored numeric gene field.",
        "not_stored_in_parquet": [
            "ENTITY mention text and GENE_NAME text were reduced to lengths at load time. The 20 saved mention previews are used when they are complete. AUX text is used for the supplemental coordinate rows. No raw CSV reread was required for this check."
        ],
    }
    write_json(OUT / "derived" / "rules_v2.json", payload)
    write_validation_tables(payload)
    write_json(
        OUT / "derived" / "v20_DONE.json",
        {"status": "success", "job": "V20", "rules": str(OUT / "derived" / "rules_v2.json"), "enabled": enabled},
    )
    logger.info("phase=v20_success enabled=%s", enabled)
    return 0


def load_ncbi(con, logger) -> dict:
    folder = OUT / "derived" / "ncbi"
    folder.mkdir(parents=True, exist_ok=True)
    info_path = folder / "gene_info.gz"
    history_path = folder / "gene_history.gz"
    try:
        info_url = fetch_first(NCBI_INFO_URLS, info_path, logger)
        history_url = fetch_first(NCBI_HISTORY_URLS, history_path, logger)
    except Exception as exc:
        logger.info("phase=ncbi_unavailable error=%s", exc)
        return {"available": False, "error": str(exc), "info_urls": NCBI_INFO_URLS, "history_urls": NCBI_HISTORY_URLS}
    con.execute(f"CREATE OR REPLACE TABLE gene_info_raw AS SELECT * FROM read_csv('{info_path.as_posix()}', delim='\t', header=true, quote='', compression='gzip', all_varchar=true)")
    con.execute(f"CREATE OR REPLACE TABLE gene_history_raw AS SELECT * FROM read_csv('{history_path.as_posix()}', delim='\t', header=true, quote='', compression='gzip', all_varchar=true)")
    info_cols = {row[0].lstrip("#"): row[0] for row in con.execute("DESCRIBE gene_info_raw").fetchall()}
    history_cols = {row[0].lstrip("#"): row[0] for row in con.execute("DESCRIBE gene_history_raw").fetchall()}
    con.execute(
        f"""
        CREATE OR REPLACE TABLE gene_info AS
        SELECT {norm_sql(info_cols['GeneID'])} AS gene_id,
               {qcol(info_cols['Symbol'])} AS symbol,
               {qcol(info_cols['Synonyms'])} AS synonyms,
               {qcol(info_cols['type_of_gene'])} AS type_of_gene,
               {qcol(info_cols['tax_id'])} AS tax_id
        FROM gene_info_raw
        WHERE CAST({qcol(info_cols['GeneID'])} AS VARCHAR) IS NOT NULL AND CAST({qcol(info_cols['GeneID'])} AS VARCHAR) <> '-'
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE gene_history AS
        SELECT {norm_sql(history_cols['Discontinued_GeneID'])} AS discontinued_gene_id,
               {qcol(history_cols['Discontinued_Symbol'])} AS discontinued_symbol,
               {qcol(history_cols['GeneID'])} AS replacement_gene_id,
               {qcol(history_cols['Discontinue_Date'])} AS discontinue_date
        FROM gene_history_raw
        WHERE CAST({qcol(history_cols['Discontinued_GeneID'])} AS VARCHAR) IS NOT NULL
          AND CAST({qcol(history_cols['Discontinued_GeneID'])} AS VARCHAR) <> '-'
        """
    )
    header = gzip.open(info_path, "rt", encoding="utf-8", errors="replace").readline().rstrip("\n")
    logger.info(
        "phase=ncbi_loaded current=%s history=%s",
        con.execute("SELECT COUNT(*) FROM gene_info").fetchone()[0],
        con.execute("SELECT COUNT(*) FROM gene_history").fetchone()[0],
    )
    return {
        "available": True,
        "info_url": info_url,
        "history_url": history_url,
        "info_sha256": sha256(info_path),
        "history_sha256": sha256(history_path),
        "info_bytes": info_path.stat().st_size,
        "history_bytes": history_path.stat().st_size,
        "info_header": header,
        "n_current": con.execute("SELECT COUNT(*) FROM gene_info").fetchone()[0],
        "n_discontinued": con.execute("SELECT COUNT(*) FROM gene_history").fetchone()[0],
        "replacement_policy": "A discontinued GeneID stays the original identifier. The replacement id is recorded and is not substituted.",
    }


def qcol(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def norm_sql(column: str) -> str:
    col = f"CAST({qcol(column)} AS VARCHAR)"
    return f"CASE WHEN ltrim({col}, '0') = '' THEN '0' ELSE ltrim({col}, '0') END"


def fetch_first(urls: list[str], dest: Path, logger) -> str:
    if dest.is_file() and dest.stat().st_size > 0:
        logger.info("phase=ncbi_reuse path=%s bytes=%s", dest, dest.stat().st_size)
        return urls[0]
    errors = []
    for url in urls:
        for attempt in range(3):
            try:
                fetch(url, dest, logger)
                return url
            except Exception as exc:
                errors.append(f"{url} attempt={attempt} {exc}")
                logger.info("phase=ncbi_retry url=%s attempt=%s error=%s", url, attempt, exc)
                time.sleep(5)
    raise RuntimeError("; ".join(errors))


def fetch(url: str, dest: Path, logger) -> None:
    logger.info("phase=ncbi_download url=%s", url)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "curl/8.0 semmed-id-alignment-v2"})
    with urllib.request.urlopen(request, timeout=180) as response, tmp.open("wb") as handle:  # noqa: S310
        shutil.copyfileobj(response, handle)
    if tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"empty download from {url}")
    tmp.replace(dest)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def namespace_samples(con, logger, ncbi_ok: bool) -> dict:
    logger.info("phase_start phase=argument_frequency")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE argument_freq AS
        SELECT argument, semtype, arg_min(name, predication_id) AS example_name, count(*) AS n
        FROM (
          SELECT predication_id, subject_cui AS argument, subject_name AS name, subject_semtype AS semtype
          FROM read_parquet('{PREDICATE.as_posix()}')
          WHERE list_contains(string_split(COALESCE(subject_semtype, ''), ','), 'gngm')
             OR list_contains(string_split(COALESCE(subject_semtype, ''), ','), 'aapp')
          UNION ALL
          SELECT predication_id, object_cui, object_name, object_semtype
          FROM read_parquet('{PREDICATE.as_posix()}')
          WHERE list_contains(string_split(COALESCE(object_semtype, ''), ','), 'gngm')
             OR list_contains(string_split(COALESCE(object_semtype, ''), ','), 'aapp')
        )
        WHERE regexp_matches(argument, '^[0-9]+$') OR position('|' IN argument) > 0
        GROUP BY 1, 2
        """
    )
    logger.info("phase=argument_frequency rows=%s", con.execute("SELECT COUNT(*) FROM argument_freq").fetchone()[0])
    digits = stratified(con, "single_digits", "regexp_matches(argument, '^[0-9]+$')")
    pipes = stratified(
        con,
        "pipe_one",
        r"regexp_matches(argument, '^C[0-9]+\|[0-9]+$') OR regexp_matches(argument, '^[0-9]+\|C[0-9]+$') OR regexp_matches(argument, '^[0-9]+\|[0-9]+$')",
    )
    scored_digits = score_ids(con, digits, ncbi_ok, kind="single_digits")
    scored_pipes = score_ids(con, pipes, ncbi_ok, kind="pipe")
    conflicts = conflict_sample(con, logger, pipes)
    write_json(OUT / "cases" / "identifier_samples.json", {"single_digits": scored_digits, "pipe": scored_pipes, "conflicts": conflicts})
    return {"single_digits": summarize_score(scored_digits), "pipe": summarize_score(scored_pipes), "conflicts": conflicts["summary"]}


def stratified(con, label: str, where: str) -> list[dict]:
    con.execute(
        f"""
        CREATE OR REPLACE TABLE stratum_{label} AS
        WITH base AS (
          SELECT argument, arg_min(example_name, n) AS example_name, sum(n) AS n
          FROM argument_freq
          WHERE {where}
          GROUP BY 1
        ), ranked AS (
          SELECT *, ntile(4) OVER (ORDER BY n, argument) AS quartile
          FROM base
        ), picked AS (
          SELECT *, row_number() OVER (PARTITION BY quartile ORDER BY hash(argument), argument) AS rk
          FROM ranked
        )
        SELECT * FROM picked WHERE rk <= {int(THRESHOLDS['ids_per_quartile'])}
        """
    )
    rows = con.execute(f"SELECT argument, example_name, n, quartile FROM stratum_{label}").fetchall()
    return [{"argument": row[0], "example_name": row[1], "n": row[2], "quartile": row[3]} for row in rows]


def score_ids(con, rows: list[dict], ncbi_ok: bool, kind: str) -> list[dict]:
    out = []
    if not rows:
        return out
    con.execute("CREATE OR REPLACE TABLE sample_args(argument VARCHAR, example_name VARCHAR, n BIGINT, quartile INTEGER)")
    con.executemany("INSERT INTO sample_args VALUES (?,?,?,?)", [(row["argument"], row["example_name"], row["n"], row["quartile"]) for row in rows])
    parsed = []
    for row in rows:
        item = parse_argument(row["argument"])
        gene = item["candidate_gene_ids"][0] if len(item["candidate_gene_ids"]) == 1 else None
        parsed.append((row["argument"], item["argument_format"], gene, json.dumps(item["candidate_gene_ids"])))
    con.execute("CREATE OR REPLACE TABLE sample_parsed(argument VARCHAR, argument_format VARCHAR, gene_id VARCHAR, candidates VARCHAR)")
    con.executemany("INSERT INTO sample_parsed VALUES (?,?,?,?)", parsed)
    lookup = {}
    if ncbi_ok:
        found = con.execute(
            """
            SELECT p.argument, p.gene_id,
                   g.symbol, g.synonyms, g.tax_id,
                   h.discontinued_gene_id, h.discontinued_symbol, h.replacement_gene_id
            FROM sample_parsed p
            LEFT JOIN gene_info g ON p.gene_id = g.gene_id
            LEFT JOIN gene_history h ON p.gene_id = h.discontinued_gene_id
            """
        ).fetchall()
        for argument, gene_id, symbol, synonyms, tax_id, discontinued, discontinued_symbol, replacement in found:
            lookup.setdefault(argument, []).append(
                {
                    "gene_id": gene_id,
                    "symbol": symbol,
                    "synonyms": synonyms,
                    "tax_id": tax_id,
                    "discontinued": discontinued is not None,
                    "discontinued_symbol": discontinued_symbol,
                    "replacement_gene_id": replacement,
                }
            )
    by_arg = {row["argument"]: row for row in rows}
    formats = {row[0]: row[1] for row in parsed}
    candidates = {row[0]: row[3] for row in parsed}
    for argument, row in by_arg.items():
        hits = lookup.get(argument, [])
        gene_hits = [hit for hit in hits if hit["gene_id"]]
        current = [hit for hit in gene_hits if hit["symbol"]]
        history_only = [hit for hit in gene_hits if hit["discontinued"] and not hit["symbol"]]
        name_ok = any(name_agrees(row["example_name"], hit["symbol"], hit["synonyms"]) for hit in current)
        out.append(
            {
                "kind": kind,
                "argument": argument,
                "argument_format": formats[argument],
                "candidates": candidates[argument],
                "example_name": row["example_name"],
                "n": row["n"],
                "quartile": row["quartile"],
                "in_current_gene_info": bool(current),
                "history_only": bool(history_only) and not current,
                "name_agrees": name_ok,
                "symbol": current[0]["symbol"] if current else None,
                "replacement_gene_id": history_only[0]["replacement_gene_id"] if history_only else None,
            }
        )
    return out


def summarize_score(rows: list[dict]) -> dict:
    usable = [row for row in rows if row["argument_format"] in {"single_digits", "pipe_cui_and_one_digit", "pipe_repeated_same_digit"}]
    n = len(usable)
    current = sum(1 for row in usable if row["in_current_gene_info"])
    history = sum(1 for row in usable if row["history_only"])
    present = current + history
    named = [row for row in usable if row["in_current_gene_info"] and row["example_name"]]
    agreed = sum(1 for row in named if row["name_agrees"])
    return {
        "n_sampled_rows": len(rows),
        "n_single_id_rows": n,
        "n_current": current,
        "n_history_only": history,
        "presence_rate": None if n == 0 else present / n,
        "name_checked": len(named),
        "name_agreed": agreed,
        "name_agreement_rate": None if not named else agreed / len(named),
        "quartile_counts": _counts(rows, "quartile"),
        "format_counts": _counts(rows, "argument_format"),
    }


def conflict_sample(con, logger, pipe_rows: list[dict]) -> dict:
    chosen = []
    for row in pipe_rows:
        parsed = parse_argument(row["argument"])
        if parsed["argument_format"] == "pipe_cui_and_one_digit":
            chosen.append(row["argument"])
        if len(chosen) >= 200:
            break
    if not chosen:
        return {"summary": {"n": 0, "reason": "no single-digit composite in the stratified pipe sample"}, "rows": []}
    con.execute("CREATE OR REPLACE TABLE conflict_args(argument VARCHAR)")
    con.executemany("INSERT INTO conflict_args VALUES (?)", [(item,) for item in chosen])
    logger.info("phase_start phase=conflict_entity_scan n_arguments=%s", len(chosen))
    con.execute(
        f"""
        CREATE OR REPLACE TABLE conflict_pred AS
        SELECT p.predication_id, p.sentence_id, p.subject_cui AS argument, p.subject_semtype AS semtype
        FROM read_parquet('{PREDICATE.as_posix()}') p
        JOIN conflict_args a ON p.subject_cui = a.argument
        UNION ALL
        SELECT p.predication_id, p.sentence_id, p.object_cui, p.object_semtype
        FROM read_parquet('{PREDICATE.as_posix()}') p
        JOIN conflict_args a ON p.object_cui = a.argument
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE conflict_one AS
        SELECT argument, any_value(predication_id) AS predication_id, any_value(sentence_id) AS sentence_id
        FROM conflict_pred
        GROUP BY 1
        """
    )
    from lib.semmed_columns import canonical_id_expr

    canon = canonical_id_expr("pre_cui_numeric_1")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE conflict_entity AS
        SELECT {canon} AS sentence_id, cui, gene_id, count(*) AS n_rows, min(entity_id) AS entity_id
        FROM read_parquet('{ENTITY.as_posix()}')
        WHERE {canon} IN (SELECT sentence_id FROM conflict_one)
        GROUP BY 1, 2, 3
        """
    )
    compared = []
    both = 0
    conflict = 0
    for argument, predication_id, sentence_id in con.execute("SELECT argument, predication_id, sentence_id FROM conflict_one").fetchall():
        parsed = parse_argument(argument)
        genes = [
            row[0]
            for row in con.execute(
                "SELECT gene_id FROM conflict_entity WHERE sentence_id = ? AND cui = ? AND gene_id IS NOT NULL",
                [sentence_id, parsed["cui"]],
            ).fetchall()
        ]
        distinct = []
        for gene in genes:
            if gene not in distinct:
                distinct.append(gene)
        has_gene = bool(distinct)
        is_conflict = has_gene and any(gene != parsed["candidate_gene_ids"][0] for gene in distinct)
        if has_gene:
            both += 1
        if is_conflict:
            conflict += 1
        if len(compared) < 30:
            compared.append(
                {
                    "argument": argument,
                    "predication_id": predication_id,
                    "sentence_id": sentence_id,
                    "candidate": parsed["candidate_gene_ids"],
                    "entity_gene_ids": distinct,
                    "conflict": is_conflict,
                }
            )
    rate = None if both < THRESHOLDS["min_rows_for_conflict_rate"] else conflict / both
    return {
        "summary": {
            "n_arguments": len(chosen),
            "n_with_a_predication": con.execute("SELECT COUNT(*) FROM conflict_one").fetchone()[0],
            "n_with_entity_gene": both,
            "n_conflict": conflict,
            "conflict_rate": rate,
            "rate_usable": rate is not None,
        },
        "rows": compared,
    }


def coordinate_check(con, logger) -> dict:
    logger.info("phase_start phase=coordinates")
    saved = []
    if CASES.is_file():
        for line in CASES.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") == "entity_span_crosscheck":
                saved.append(row)
    ids = [row["entity_id"] for row in saved]
    id_sql = ",".join("'" + item.replace("'", "''") + "'" for item in ids) or "''"
    samples = {
        row[0]: row
        for row in con.execute(
            f"""
            SELECT entity_id, start_index, end_index, text_len, pre_cui_numeric_1
            FROM read_parquet('{ENTITY_SAMPLE.as_posix()}')
            WHERE entity_id IN ({id_sql})
            """
        ).fetchall()
    }
    sentence_ids = [str(row[4]) for row in samples.values() if row[4] is not None]
    sentences = sentence_map(con, sentence_ids)
    entity_cases = []
    for row in saved:
        sample = samples.get(row["entity_id"])
        if sample is None:
            entity_cases.append({"entity_id": row["entity_id"], "record_found": False})
            continue
        sentence = sentences.get(str(sample[4]))
        preview = row.get("mention_preview") or ""
        text_len = sample[3]
        complete = text_len is not None and len(preview) == int(text_len)
        entity_cases.append(
            assess_span(
                {
                    "source": "saved_entity_case",
                    "record_id": row["entity_id"],
                    "record_found": True,
                    "sentence_found": sentence is not None,
                    "mention": preview if complete else None,
                    "mention_incomplete": not complete,
                    "mention_in_sentence_saved": row.get("mention_in_sentence"),
                    "start": sample[1],
                    "end": sample[2],
                    "sentence_text": None if sentence is None else sentence["sentence_text"],
                    "sentence_start": None if sentence is None else sentence["sent_start_index"],
                    "sentence_end": None if sentence is None else sentence["sent_end_index"],
                }
            )
        )
    aux_rows = con.execute(
        f"""
        SELECT a.predication_id, a.subject_text, a.subject_start_index, a.subject_end_index,
               s.sentence_text, s.sent_start_index, s.sent_end_index
        FROM read_parquet('{AUX.as_posix()}') a
        JOIN read_parquet('{PREDICATE.as_posix()}') p ON a.predication_id = p.predication_id
        JOIN read_parquet('{SENTENCE.as_posix()}') s ON p.sentence_id = s.sentence_id
        WHERE a.subject_text IS NOT NULL AND a.subject_text <> ''
          AND regexp_matches(a.subject_start_index, '^[0-9]+$')
          AND regexp_matches(a.subject_end_index, '^[0-9]+$')
          AND regexp_matches(s.sent_start_index, '^[0-9]+$')
        ORDER BY a.predication_id
        LIMIT 40
        """
    ).fetchall()
    aux_cases = [
        assess_span(
            {
                "source": "aux_subject",
                "record_id": row[0],
                "record_found": True,
                "sentence_found": True,
                "mention": row[1],
                "mention_incomplete": False,
                "mention_in_sentence_saved": None,
                "start": row[2],
                "end": row[3],
                "sentence_text": row[4],
                "sentence_start": row[5],
                "sentence_end": row[6],
            }
        )
        for row in aux_rows
    ]
    combined = entity_cases + aux_cases
    choice = choose_coordinate_rule(combined)
    path = OUT / "cases" / "coordinate_cases.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in combined:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"choice": choice, "counts": coordinate_counts(combined, choice.get("adopted")), "n_entity_cases": len(entity_cases), "n_aux_cases": len(aux_cases)}


def sentence_map(con, sentence_ids: list[str]) -> dict:
    if not sentence_ids:
        return {}
    con.execute("CREATE OR REPLACE TABLE coord_sentences(sentence_id VARCHAR)")
    con.executemany("INSERT INTO coord_sentences VALUES (?)", [(item,) for item in sentence_ids])
    rows = con.execute(
        f"""
        SELECT s.sentence_id, s.sentence_text, s.sent_start_index, s.sent_end_index
        FROM read_parquet('{SENTENCE.as_posix()}') s
        JOIN coord_sentences c USING (sentence_id)
        """
    ).fetchall()
    return {row[0]: {"sentence_text": row[1], "sent_start_index": row[2], "sent_end_index": row[3]} for row in rows}


def assess_span(row: dict) -> dict:
    result = dict(row)
    text = row.get("sentence_text")
    mention = row.get("mention")
    result["mention_contained"] = bool(mention) and bool(text) and mention in text
    try:
        start = int(row["start"])
        end = int(row["end"])
        sentence_start = int(row["sentence_start"])
        sentence_end = int(row["sentence_end"])
    except (TypeError, ValueError):
        result["half_open"] = None
        result["inclusive_end"] = None
        result["inside_sentence_half_open"] = None
        result["unable"] = True
        return result
    result["unable"] = mention is None or text is None
    result["half_open"] = None if result["unable"] else slice_document(text, sentence_start, start, end, False) == mention
    result["inclusive_end"] = None if result["unable"] else slice_document(text, sentence_start, start, end, True) == mention
    result["inside_sentence_half_open"] = sentence_start <= start and end <= sentence_end + 1
    result["inside_sentence_inclusive"] = sentence_start <= start and end <= sentence_end
    result["unconverted_slice_matches"] = False
    if mention and text:
        left, right = start, end
        result["unconverted_slice_matches"] = 0 <= left <= right <= len(text) and text[left:right] == mention
    return result


def choose_coordinate_rule(rows: list[dict]) -> dict:
    judged = [row for row in rows if row.get("half_open") is not None]
    n = len(judged)
    half = sum(1 for row in judged if row["half_open"])
    inclusive = sum(1 for row in judged if row["inclusive_end"])
    rates = {
        "n_judged": n,
        "half_open_matches": half,
        "inclusive_end_matches": inclusive,
        "half_open_rate": None if n == 0 else half / n,
        "inclusive_end_rate": None if n == 0 else inclusive / n,
    }
    adopted = None
    if n and rates["half_open_rate"] >= THRESHOLDS["coordinate_adopt_match"] and rates["inclusive_end_rate"] < THRESHOLDS["coordinate_reject_alternative"]:
        adopted = "document_half_open_end_exclusive"
    elif n and rates["inclusive_end_rate"] >= THRESHOLDS["coordinate_adopt_match"] and rates["half_open_rate"] < THRESHOLDS["coordinate_reject_alternative"]:
        adopted = "document_inclusive_end"
    rates["adopted"] = adopted
    rates["basis_note"] = "Sentence start and mention start are subtracted before slicing, so a shared 0-based or 1-based origin cancels. The remaining choice is whether the stored end is exclusive."
    return rates


def coordinate_counts(rows: list[dict], adopted: str | None) -> dict:
    def key(row):
        if adopted == "document_half_open_end_exclusive":
            return row.get("half_open")
        if adopted == "document_inclusive_end":
            return row.get("inclusive_end")
        return None

    return {
        "record_found": sum(1 for row in rows if row.get("record_found")),
        "sentence_found": sum(1 for row in rows if row.get("sentence_found")),
        "mention_contained": sum(1 for row in rows if row.get("mention_contained")),
        "converted_slice_matches": sum(1 for row in rows if key(row) is True),
        "unable": sum(1 for row in rows if row.get("unable") or key(row) is None),
        "n": len(rows),
    }


def decide_enabled(samples: dict, ncbi_ok: bool) -> dict:
    digits = samples["single_digits"]
    pipes = samples["pipe"]
    conflicts = samples["conflicts"]
    path_a = _passes(digits) if ncbi_ok else False
    path_c = path_a and _passes(pipes)
    if conflicts.get("rate_usable") and conflicts.get("conflict_rate") is not None:
        if conflicts["conflict_rate"] > THRESHOLDS["max_conflict_when_entity_gene_present"]:
            path_c = False
    path_b = path_a
    return {
        "A_direct_entrez": path_a,
        "B_entity_gene_id": path_b,
        "C_composite_entrez": path_c,
        "ncbi_available": ncbi_ok,
        "reason": {
            "A": "Enabled only when stratified single-number arguments meet the presence and exact-name thresholds.",
            "B": "Uses the same Entrez identification. A single CUI is accepted only through one numeric gene field on the matching entity, and a generic concept is never forced.",
            "C": "Enabled only when the single numeric component meets the same thresholds and entity conflicts stay under the fixed rate. Multiple different numbers stay ambiguous.",
        },
    }


def _passes(summary: dict) -> bool:
    if summary.get("n_single_id_rows", 0) < THRESHOLDS["ids_per_quartile"]:
        return False
    presence = summary.get("presence_rate")
    agreement = summary.get("name_agreement_rate")
    if presence is None or agreement is None:
        return False
    return presence >= THRESHOLDS["min_presence_current_or_history"] and agreement >= THRESHOLDS["min_current_name_agreement"]


def write_validation_tables(payload: dict) -> None:
    import csv

    folder = OUT / "tables"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "namespace_validation.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["kind", "n_single_id_rows", "presence_rate", "name_agreement_rate", "n_current", "n_history_only"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for kind in ("single_digits", "pipe"):
            row = payload["samples"][kind]
            writer.writerow(
                {
                    "kind": kind,
                    "n_single_id_rows": row.get("n_single_id_rows"),
                    "presence_rate": row.get("presence_rate"),
                    "name_agreement_rate": row.get("name_agreement_rate"),
                    "n_current": row.get("n_current"),
                    "n_history_only": row.get("n_history_only"),
                }
            )
    with (folder / "coordinate_rules.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["metric", "value"])
        for key, value in payload["coordinates"]["choice"].items():
            writer.writerow([key, value])
        for key, value in payload["coordinates"]["counts"].items():
            writer.writerow([key, value])


def _counts(rows: list[dict], key: str) -> dict:
    out: dict = {}
    for row in rows:
        label = str(row.get(key))
        out[label] = out.get(label, 0) + 1
    return out


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        raise SystemExit(1)

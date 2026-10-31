"""Load SemMedDB CSV files and compute single-database summaries."""

from __future__ import annotations

import threading
from pathlib import Path

from lib.csv_stream import count_records, iter_records, read_csv_call
from lib.fingerprint import blake2_file
from lib.schema import SCOPE_E_DISEASE_SEMTYPE, SCOPE_E_GENE_SEMTYPE, SCOPE_E_PROTEIN_SEMTYPE
from lib.semmed_columns import canonical_id_expr
from lib.stats import head_share_sql, summarize_groups_sql

CANON = """
CASE
  WHEN {col} IS NULL THEN NULL
  WHEN trim({col}) = '' THEN NULL
  WHEN trim({col}) = CHR(92) || 'N' THEN NULL
  WHEN trim({col}) IN ('NULL', 'null', 'None') THEN NULL
  WHEN regexp_matches(trim({col}), '^[0-9]+$') THEN
    CASE WHEN ltrim(trim({col}), '0') = '' THEN '0' ELSE ltrim(trim({col}), '0') END
  ELSE NULL
END
"""

GENE_CLASS = """
CASE
  WHEN {col} IS NULL THEN 'sql_null'
  WHEN trim({col}) = '' THEN 'empty_string'
  WHEN trim({col}) = CHR(92) || 'N' THEN 'backslash_N'
  WHEN trim({col}) IN ('NULL', 'null', 'None') THEN 'literal_NULL'
  WHEN position('|' IN {col}) > 0 THEN 'composite_pipe'
  WHEN position(',' IN {col}) > 0 THEN 'composite_comma'
  WHEN regexp_matches(trim({col}), '^[0-9]+$') THEN 'numeric'
  ELSE 'other'
END
"""

DP_STATUS = """
CASE
  WHEN {col} IS NULL THEN 'sql_null'
  WHEN trim({col}) = '' THEN 'empty_string'
  WHEN trim({col}) = CHR(92) || 'N' THEN 'backslash_N'
  WHEN trim({col}) IN ('NULL', 'null', 'None') THEN 'literal_NULL'
  WHEN regexp_matches(trim({col}), '^[0-9]{{4}}[[:space:]]*[-/][[:space:]]*[0-9]{{4}}') THEN 'multi_year_range'
  WHEN regexp_matches(trim({col}), '^[0-9]{{4}}') THEN 'parsed_single'
  ELSE 'unparsed'
END
"""


def _sql(template: str, col: str) -> str:
    return template.format(col=col)


def _cells_equivalent(py_cell, duck_cell) -> bool:
    if py_cell == duck_cell:
        return True
    # Python csv consumes a backslash escape, so the MySQL null token \\N becomes N.
    # DuckDB leaves an unquoted \\N as the two-character token. Both mean that token.
    if py_cell == "N" and duck_cell == "\\N":
        return True
    if py_cell == "" and duck_cell is None:
        return True
    return False


def _rows_equivalent(py_row, duck_row) -> bool:
    if len(py_row) != len(duck_row):
        return False
    return all(_cells_equivalent(a, b) for a, b in zip(py_row, duck_row))


def compare_prefix(con, path: Path, n_columns: int, dialect_name: str, max_line_size: int, n: int = 1000, encoding: str = "utf-8") -> dict:
    escapechar = None if dialect_name == "quote_doubling" else "\\"
    py_rows = []
    for row in iter_records(path, dialect="excel", escapechar=escapechar, max_records=n):
        py_rows.append(tuple(row))
        if len(py_rows) >= n:
            break
    call = read_csv_call(str(path), n_columns, dialect_name, max_line_size, parallel=False, encoding=encoding)
    duck_rows = [tuple(r) for r in con.execute(f"SELECT * FROM {call} LIMIT {n}").fetchall()]
    mismatch = None
    mismatch_columns = []
    limit = min(len(py_rows), len(duck_rows))
    for i in range(limit):
        if not _rows_equivalent(py_rows[i], duck_rows[i]):
            mismatch = i
            width = max(len(py_rows[i]), len(duck_rows[i]))
            for j in range(width):
                py_cell = py_rows[i][j] if j < len(py_rows[i]) else None
                duck_cell = duck_rows[i][j] if j < len(duck_rows[i]) else None
                if py_cell != duck_cell:
                    mismatch_columns.append(
                        {
                            "column": j,
                            "python_len": None if py_cell is None else len(py_cell),
                            "duckdb_len": None if duck_cell is None else len(str(duck_cell)),
                            "python_is_null": py_cell is None,
                            "duckdb_is_null": duck_cell is None,
                        }
                    )
            break
    return {
        "python_rows": len(py_rows),
        "duckdb_rows": len(duck_rows),
        "mismatch_index": mismatch,
        "mismatch_columns": mismatch_columns[:12],
        "equal": mismatch is None and len(py_rows) == len(duck_rows) and len(py_rows) > 0,
    }


def citations_counts_match(con, path: Path, dialect_name: str, max_line_size: int, progress, parallel: bool) -> dict:
    escapechar = None if dialect_name == "quote_doubling" else "\\"
    progress.set_phase("citations_python_count")
    py_n = count_records(path, dialect="excel", escapechar=escapechar, progress=progress)
    progress.set_phase("citations_duckdb_count")
    call = read_csv_call(str(path), 5, dialect_name, max_line_size, parallel=parallel)
    duck_n = con.execute(f"SELECT COUNT(*) FROM {call}").fetchone()[0]
    progress.update(force=True)
    return {"python_logical_records": py_n, "duckdb_logical_records": duck_n, "equal": py_n == duck_n}


def _watch(path: Path, logger, phase: str, stop: threading.Event) -> None:
    while not stop.wait(45):
        size = path.stat().st_size if path.exists() else 0
        logger.info("progress phase=%s output_bytes=%s file=%s", phase, size, path.name)


def load_table(con, table: str, path: Path, dialect_name: str, max_line_size: int, out: Path, logger, parallel: bool, column_map: dict, encoding: str = "utf-8", *, precomputed_blake2b: str | None = None) -> dict:
    n_cols = int(column_map["observed_width"])
    call = read_csv_call(str(path), n_cols, dialect_name, max_line_size, parallel=parallel, encoding=encoding)
    select = _select(table, column_map)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    logger.info("phase_start phase=load_%s source=%s output=%s encoding=%s", table, path, out, encoding)
    stop = threading.Event()
    watcher = threading.Thread(target=_watch, args=(out, logger, f"load_{table}", stop), daemon=True)
    hasher = {}

    def run_hash():
        hasher["blake2b"] = blake2_file(path)

    hash_thread = None
    if precomputed_blake2b is None:
        hash_thread = threading.Thread(target=run_hash, daemon=True)
        hash_thread.start()
    else:
        hasher["blake2b"] = precomputed_blake2b
    watcher.start()
    try:
        con.execute(
            f"""
            COPY (
              {select}
              FROM {call}
            ) TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 1000000)
            """
        )
    finally:
        stop.set()
        watcher.join(timeout=1)
    if hash_thread is not None:
        hash_thread.join()
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out.as_posix()}')").fetchone()[0]
    logger.info("phase=load_%s_done records=%s blake2b=%s", table, n, hasher.get("blake2b"))
    return {"rows": n, "blake2b": hasher.get("blake2b"), "output": str(out)}


def _c(i: int) -> str:
    return f"c{i}"


def _named(column_map: dict, name: str) -> str:
    return f"c{column_map['index'][name]}"


def _unconfirmed_sql(column_map: dict) -> str:
    parts = [f"c{i} AS unconfirmed_{i}" for i in column_map.get("unconfirmed_indexes", [])]
    return (",\n" + ",\n".join(parts)) if parts else ""


def _select(table: str, column_map: dict) -> str:
    if table == "CITATIONS":
        return f"""
        SELECT
          {_sql(CANON, 'c0')} AS pmid,
          c0 AS pmid_raw,
          c1 AS issn,
          c2 AS dp,
          c3 AS edat,
          c4 AS pyear,
          {_sql(DP_STATUS, 'c2')} AS dp_status,
          CASE WHEN {_sql(DP_STATUS, 'c2')} = 'parsed_single' THEN try_cast(substr(trim(c2), 1, 4) AS INTEGER) ELSE NULL END AS dp_year,
          CASE
            WHEN trim(c4) = '' THEN 'empty_string'
            WHEN trim(c4) = CHR(92) || 'N' THEN 'backslash_N'
            WHEN regexp_matches(trim(c4), '^[0-9]{{4}}$') THEN 'parsed_completion_year'
            ELSE 'unparsed'
          END AS pyear_status,
          CASE WHEN regexp_matches(trim(c4), '^[0-9]{{4}}$') THEN try_cast(trim(c4) AS INTEGER) ELSE NULL END AS completion_year
        """
    if table == "GENERIC_CONCEPT":
        return """
        SELECT c0 AS concept_id_raw, c1 AS cui, c2 AS preferred_name
        """
    if table == "SENTENCE":
        sid, pmid, text = _named(column_map, "sentence_id"), _named(column_map, "pmid"), _named(column_map, "sentence_text")
        return f"""
        SELECT
          {_sql(CANON, sid)} AS sentence_id,
          {sid} AS sentence_id_raw,
          {_sql(CANON, pmid)} AS pmid,
          {pmid} AS pmid_raw,
          {_named(column_map, "sent_type")} AS sent_type,
          {_named(column_map, "sent_number")} AS sent_number,
          {_named(column_map, "sent_start_index")} AS sent_start_index,
          {_named(column_map, "sent_end_index")} AS sent_end_index,
          {_named(column_map, "section_header")} AS section_header,
          {_named(column_map, "normalized_section_header")} AS normalized_section_header,
          {text} AS sentence_text,
          length({text}) AS text_len
          {_unconfirmed_sql(column_map)}
        """
    if table == "PREDICATION":
        pred = _named(column_map, "PREDICATE")
        return f"""
        SELECT
          {_sql(CANON, _named(column_map, "PREDICATION_ID"))} AS predication_id,
          {_sql(CANON, _named(column_map, "SENTENCE_ID"))} AS sentence_id,
          {_sql(CANON, _named(column_map, "PMID"))} AS pmid,
          {pred} AS predicate,
          starts_with({pred}, 'NEG_') AS negation_prefix,
          CASE WHEN starts_with({pred}, 'NEG_') THEN substr({pred}, 5) ELSE {pred} END AS predicate_without_neg_prefix,
          {_named(column_map, "SUBJECT_CUI")} AS subject_cui,
          {_named(column_map, "SUBJECT_NAME")} AS subject_name,
          {_named(column_map, "SUBJECT_SEMTYPE")} AS subject_semtype,
          {_named(column_map, "SUBJECT_NOVELTY")} AS subject_novelty,
          {_named(column_map, "OBJECT_CUI")} AS object_cui,
          {_named(column_map, "OBJECT_NAME")} AS object_name,
          {_named(column_map, "OBJECT_SEMTYPE")} AS object_semtype,
          {_named(column_map, "OBJECT_NOVELTY")} AS object_novelty
          {_unconfirmed_sql(column_map)}
        """
    if table == "PREDICATION_AUX":
        return f"""
        SELECT
          {_sql(CANON, 'c0')} AS predication_aux_id,
          {_sql(CANON, 'c1')} AS predication_id,
          c2 AS subject_text,
          c7 AS subject_score,
          c8 AS indicator_type,
          c11 AS object_text,
          c16 AS object_score,
          c17 AS curr_timestamp,
          c5 AS subject_start_index,
          c6 AS subject_end_index,
          c14 AS object_start_index,
          c15 AS object_end_index
        """
    if table == "ENTITY":
        gene = _named(column_map, "gene_id")
        pre = [name for name in column_map["index"] if name.startswith("pre_cui_numeric_")]
        pre_sql = ",\n".join(f"{_named(column_map, name)} AS {name}" for name in sorted(pre))
        pre_sql = (",\n" + pre_sql) if pre_sql else ""
        return f"""
        SELECT
          {_sql(CANON, _named(column_map, "entity_id"))} AS entity_id,
          {_named(column_map, "cui")} AS cui,
          {_named(column_map, "semtype")} AS semtype,
          CASE WHEN {_sql(GENE_CLASS, gene)} = 'numeric'
               THEN CASE WHEN ltrim(trim({gene}), '0') = '' THEN '0' ELSE ltrim(trim({gene}), '0') END
               ELSE NULL END AS gene_id,
          {gene} AS gene_id_raw,
          {_sql(GENE_CLASS, gene)} AS gene_id_class,
          {_named(column_map, "score")} AS score,
          {_named(column_map, "start_index")} AS start_index,
          {_named(column_map, "end_index")} AS end_index,
          length({_named(column_map, "text")}) AS text_len,
          length({_named(column_map, "name")}) AS name_len
          {pre_sql}
          {_unconfirmed_sql(column_map)}
        """
    raise KeyError(table)


def _view(con, name: str, path: Path) -> None:
    con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{path.as_posix()}')")


def _has_column(con, table: str, column: str) -> bool:
    rows = con.execute(f"DESCRIBE {table}").fetchall()
    return any(row[0] == column for row in rows)


def resolve_entity_sentence_id(con, entity_path: Path, logger) -> dict:
    """Choose which pre-CUI numeric column is SENTENCE_ID by joining a prefix to SENTENCE.

    The prefix is the first 50000 entity rows, not a random sample. A column is
    accepted only when it is the single candidate whose values fall in
    sentence.sentence_id at a rate of at least 0.95. Requiring the other column
    to stay at or below 0.05 confuses a dense identifier domain with a second
    foreign key: another numeric column can collide often without matching every row.
    If two columns are both at or above 0.95, the column stays unresolved.
    """
    logger.info("phase_start phase=entity_sentence_id_resolution sample_rows=50000 sample=first_rows")
    columns = [row[0] for row in con.execute("DESCRIBE entity").fetchall()]
    candidates = [name for name in columns if name.startswith("pre_cui_numeric_")]
    if not candidates:
        return {"status": "not_assessable", "reason": "no pre_cui_numeric columns", "sample_rows": 0}
    if len(candidates) > 2:
        return {"status": "not_assessable", "reason": "more than two pre_cui numeric columns", "columns": candidates}
    while len(candidates) < 2:
        candidates.append(None)
    left, right = candidates[0], candidates[1]
    left_expr = canonical_id_expr(left)
    right_expr = canonical_id_expr(right) if right else "NULL"
    sampled, left_n, right_n = con.execute(
        f"""
        SELECT COUNT(*),
               COUNT(s1.sentence_id),
               COUNT(s2.sentence_id)
        FROM (SELECT {left_expr} AS a, {right_expr} AS b FROM entity LIMIT 50000) e
        LEFT JOIN sentence s1 ON e.a = s1.sentence_id
        LEFT JOIN sentence s2 ON e.b = s2.sentence_id
        """
    ).fetchone()
    rates = {
        left: {"matched": left_n, "rate": (left_n / sampled) if sampled else None},
    }
    if right:
        rates[right] = {"matched": right_n, "rate": (right_n / sampled) if sampled else None}
    eligible = [name for name, info in rates.items() if info["rate"] is not None and info["rate"] >= 0.95]
    resolved = eligible[0] if len(eligible) == 1 else None
    prior_would_resolve = None
    if sampled and right and rates[left]["rate"] is not None and rates[right]["rate"] is not None:
        if rates[left]["rate"] >= 0.95 and rates[right]["rate"] <= 0.05:
            prior_would_resolve = left
        elif rates[right]["rate"] >= 0.95 and rates[left]["rate"] <= 0.05:
            prior_would_resolve = right
    elif len(eligible) == 1 and not right:
        prior_would_resolve = eligible[0]
    result = {
        "status": "resolved" if resolved else "unresolved",
        "column": resolved,
        "sample_rows": sampled,
        "sample_definition": "first_50000_entity_rows",
        "rates": rates,
        "rule": "sentence_id_resolution_v2 unique column with match_rate>=0.95 against sentence.sentence_id",
        "prior_rule": "match_rate>=0.95 and alternative<=0.05",
        "prior_rule_column": prior_would_resolve,
    }
    logger.info("phase=entity_sentence_id_resolution_done %s", result["status"])
    return result


def _replace_entity_view(con, path: Path, column: str) -> None:
    con.execute(
        f"""
        CREATE OR REPLACE VIEW entity AS
        SELECT * EXCLUDE ({column}), {canonical_id_expr(column)} AS sentence_id
        FROM read_parquet('{path.as_posix()}')
        """
    )


def profile(con, paths: dict[str, Path], out_dir: Path, logger) -> dict:
    for name, path in paths.items():
        _view(con, name, path)
    logger.info("phase_start phase=semmed_profile")
    summary = {"scope": "B_semmed_local_release", "tables": {}}
    resolution = resolve_entity_sentence_id(con, paths["entity"], logger)
    summary["entity_sentence_id_resolution"] = resolution
    if resolution["status"] == "resolved":
        _replace_entity_view(con, paths["entity"], resolution["column"])
    for name in paths:
        summary["tables"][name] = {"rows": con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]}
    summary["citations"] = _citation_profile(con, logger)
    summary["sentences"] = _sentence_profile(con, logger)
    summary["predications"] = _predication_profile(con, out_dir, logger)
    summary["aux"] = _aux_profile(con)
    summary["entities"] = _entity_profile(con, out_dir, logger)
    summary["generic"] = _generic_profile(con)
    summary["integrity"] = _integrity(con, logger)
    summary["scope_e"] = _scope_e(con, out_dir, logger)
    summary["text"] = _text(con, logger)
    summary["time"] = _time(con)
    return summary


def _citation_profile(con, logger) -> dict:
    logger.info("phase_start phase=citation_profile")
    return {
        "dp_status": _pairs(con, "SELECT dp_status, COUNT(*) n FROM citations GROUP BY 1 ORDER BY 2 DESC"),
        "pyear_status": _pairs(con, "SELECT pyear_status, COUNT(*) n FROM citations GROUP BY 1 ORDER BY 2 DESC"),
        "pmid_raw_unparsed": con.execute("SELECT COUNT(*) FROM citations WHERE pmid IS NULL").fetchone()[0],
        "duplicate_pmid_keys": con.execute(
            "SELECT COUNT(*) FROM (SELECT pmid FROM citations WHERE pmid IS NOT NULL GROUP BY 1 HAVING COUNT(*)>1)"
        ).fetchone()[0],
        "dp_vs_completion_both_parsed_disagree": con.execute(
            """
            SELECT COUNT(*) FROM citations
            WHERE dp_status='parsed_single' AND pyear_status='parsed_completion_year' AND dp_year <> completion_year
            """
        ).fetchone()[0],
        "note": "dp_year is the publication year parsed from DP. completion_year comes from PYEAR, which dbinfo calls a completion date, and is not used as the publication year.",
    }


def _sentence_profile(con, logger) -> dict:
    logger.info("phase_start phase=sentence_profile")
    con.execute("CREATE OR REPLACE TABLE sent_per_pmid AS SELECT pmid, COUNT(*) AS n FROM sentence WHERE pmid IS NOT NULL GROUP BY 1")
    con.execute(
        """
        CREATE OR REPLACE TABLE sent_per_citation AS
        SELECT c.pmid, COALESCE(s.n, 0) AS n
        FROM citations c
        LEFT JOIN sent_per_pmid s USING (pmid)
        WHERE c.pmid IS NOT NULL
        """
    )
    return {
        "distinct_sentence_id": con.execute("SELECT COUNT(DISTINCT sentence_id) FROM sentence WHERE sentence_id IS NOT NULL").fetchone()[0],
        "null_sentence_id": con.execute("SELECT COUNT(*) FROM sentence WHERE sentence_id IS NULL").fetchone()[0],
        "duplicate_sentence_id": con.execute(
            "SELECT COUNT(*) FROM (SELECT sentence_id FROM sentence WHERE sentence_id IS NOT NULL GROUP BY 1 HAVING COUNT(*)>1)"
        ).fetchone()[0],
        "type_counts": _pairs(con, "SELECT COALESCE(sent_type, 'FIELD_ABSENT'), COUNT(*) FROM sentence GROUP BY 1 ORDER BY 2 DESC"),
        "sentences_per_pmid_with_sentence": summarize_groups_sql(con, "sent_per_pmid", "n", "pmid_with_at_least_one_sentence", "sentence"),
        "sentences_per_citation_including_zero": summarize_groups_sql(con, "sent_per_citation", "n", "citations_with_parsed_pmid", "sentence"),
        "section_header": _availability(con, "section_header"),
        "normalized_section_header": _availability(con, "normalized_section_header"),
        "sentence_text": _availability(con, "sentence_text"),
        "text_length": _length_summary(con, "sentence", "text_len", "sentence_rows", "characters"),
        "note": "Sentence rows are an available sentence collection. They are not concatenated into a full abstract.",
    }


def _predication_profile(con, out_dir: Path, logger) -> dict:
    logger.info("phase_start phase=predication_profile")
    con.execute(
        """
        CREATE OR REPLACE TABLE pred_per_pmid AS
        SELECT pmid, COUNT(*) AS n FROM predication WHERE pmid IS NOT NULL GROUP BY 1
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE cui_pair AS
        SELECT subject_cui, object_cui, COUNT(*) AS n, COUNT(DISTINCT pmid) AS n_pmid
        FROM predication
        GROUP BY 1, 2
        """
    )
    con.execute(f"COPY (SELECT predicate, COUNT(*) AS n FROM predication GROUP BY 1 ORDER BY n DESC) TO '{(out_dir / 'predicate_counts.parquet').as_posix()}' (FORMAT PARQUET)")
    return {
        "predicate_counts": _pairs(con, "SELECT predicate, COUNT(*) n FROM predication GROUP BY 1 ORDER BY n DESC"),
        "negation_prefix_rows": con.execute("SELECT COUNT(*) FROM predication WHERE negation_prefix").fetchone()[0],
        "novelty_subject": _pairs(con, "SELECT COALESCE(subject_novelty, 'FIELD_ABSENT'), COUNT(*) FROM predication GROUP BY 1 ORDER BY 2 DESC"),
        "novelty_object": _pairs(con, "SELECT COALESCE(object_novelty, 'FIELD_ABSENT'), COUNT(*) FROM predication GROUP BY 1 ORDER BY 2 DESC"),
        "predications_per_pmid": summarize_groups_sql(con, "pred_per_pmid", "n", "pmid_with_at_least_one_predication", "predication"),
        "head_1pct_cui_pairs_by_predication": head_share_sql(con, "cui_pair", "n", 0.01),
        "head_5pct_cui_pairs_by_predication": head_share_sql(con, "cui_pair", "n", 0.05),
        "cui_pair_predication_distribution": summarize_groups_sql(con, "cui_pair", "n", "subject_cui_object_cui", "predication"),
        "cui_pair_unit": "A full-database pair here is (subject_cui, object_cui), not a gene-disease pair.",
        "null_predication_id": con.execute("SELECT COUNT(*) FROM predication WHERE predication_id IS NULL").fetchone()[0],
        "duplicate_predication_id": con.execute(
            "SELECT COUNT(*) FROM (SELECT predication_id FROM predication WHERE predication_id IS NOT NULL GROUP BY 1 HAVING COUNT(*)>1)"
        ).fetchone()[0],
    }


def _aux_profile(con) -> dict:
    return {
        "rows": con.execute("SELECT COUNT(*) FROM predication_aux").fetchone()[0],
        "indicator_type": _pairs(con, "SELECT COALESCE(indicator_type, 'FIELD_ABSENT'), COUNT(*) FROM predication_aux GROUP BY 1 ORDER BY 2 DESC LIMIT 40"),
        "null_predication_id": con.execute("SELECT COUNT(*) FROM predication_aux WHERE predication_id IS NULL").fetchone()[0],
        "score_note": "subject_score and object_score are string-to-concept mapping scores from dbinfo, not study credibility.",
    }


def _entity_profile(con, out_dir: Path, logger) -> dict:
    logger.info("phase_start phase=entity_profile")
    has_sentence = _has_column(con, "entity", "sentence_id")
    if has_sentence:
        con.execute("CREATE OR REPLACE TABLE ent_per_sentence AS SELECT sentence_id, COUNT(*) AS n FROM entity WHERE sentence_id IS NOT NULL GROUP BY 1")
        per_sentence = summarize_groups_sql(con, "ent_per_sentence", "n", "sentence_with_at_least_one_entity", "entity_row")
    else:
        per_sentence = {"status": "not_assessable", "reason": "entity sentence_id was not resolved by the sentence join"}
    return {
        "rows": con.execute("SELECT COUNT(*) FROM entity").fetchone()[0],
        "null_entity_id": con.execute("SELECT COUNT(*) FROM entity WHERE entity_id IS NULL").fetchone()[0],
        "duplicate_entity_id": con.execute(
            "SELECT COUNT(*) FROM (SELECT entity_id FROM entity WHERE entity_id IS NOT NULL GROUP BY 1 HAVING COUNT(*)>1)"
        ).fetchone()[0],
        "gene_id_class": _pairs(con, "SELECT gene_id_class, COUNT(*) n FROM entity GROUP BY 1 ORDER BY n DESC"),
        "semtype_top": _pairs(con, "SELECT COALESCE(semtype, 'FIELD_ABSENT'), COUNT(*) n FROM entity GROUP BY 1 ORDER BY n DESC LIMIT 40"),
        "entities_per_sentence_with_entity": per_sentence,
        "stored_columns": "Identifiers, semantic type, gene-id class, score and lengths. Mention text and preferred name are not copied; lengths are kept.",
    }


def _generic_profile(con) -> dict:
    return {
        "rows": con.execute("SELECT COUNT(*) FROM generic_concept").fetchone()[0],
        "cui_pattern_c_digits": con.execute(
            "SELECT COUNT(*) FROM generic_concept WHERE regexp_matches(trim(cui), '^C[0-9]+$')"
        ).fetchone()[0],
        "cui_other": con.execute(
            "SELECT COUNT(*) FROM generic_concept WHERE NOT regexp_matches(COALESCE(trim(cui), ''), '^C[0-9]+$')"
        ).fetchone()[0],
        "note": "Values that do not match a C-plus-digits CUI stay in the table and are counted as cui_other. They are not reinterpreted as gene IDs.",
    }


def _integrity(con, logger) -> dict:
    logger.info("phase_start phase=integrity_joins")
    checks = {
        "sentence_pmid_absent_from_citations": """
            SELECT COUNT(*) FROM sentence s
            ANTI JOIN citations c ON s.pmid = c.pmid
            WHERE s.pmid IS NOT NULL
        """,
        "sentence_pmid_unparsed": "SELECT COUNT(*) FROM sentence WHERE pmid IS NULL",
        "citations_without_sentence": """
            SELECT COUNT(*) FROM citations c
            ANTI JOIN sentence s ON c.pmid = s.pmid
            WHERE c.pmid IS NOT NULL
        """,
        "predication_sentence_missing": """
            SELECT COUNT(*) FROM predication p
            ANTI JOIN sentence s ON p.sentence_id = s.sentence_id
            WHERE p.sentence_id IS NOT NULL
        """,
        "predication_pmid_disagrees_with_sentence": """
            SELECT COUNT(*) FROM predication p
            JOIN sentence s ON p.sentence_id = s.sentence_id
            WHERE p.pmid IS DISTINCT FROM s.pmid
        """,
        "aux_predication_missing": """
            SELECT COUNT(*) FROM predication_aux a
            ANTI JOIN predication p ON a.predication_id = p.predication_id
            WHERE a.predication_id IS NOT NULL
        """,
        "predication_without_aux": """
            SELECT COUNT(*) FROM predication p
            ANTI JOIN predication_aux a ON p.predication_id = a.predication_id
            WHERE p.predication_id IS NOT NULL
        """,
        "entity_sentence_missing": """
            SELECT COUNT(*) FROM entity e
            ANTI JOIN sentence s ON e.sentence_id = s.sentence_id
            WHERE e.sentence_id IS NOT NULL
        """,
        "sentences_without_predication": """
            SELECT COUNT(*) FROM sentence s
            ANTI JOIN predication p ON s.sentence_id = p.sentence_id
            WHERE s.sentence_id IS NOT NULL
        """,
        "sentences_without_entity": """
            SELECT COUNT(*) FROM sentence s
            ANTI JOIN entity e ON s.sentence_id = e.sentence_id
            WHERE s.sentence_id IS NOT NULL
        """,
    }
    out = {}
    allowed = {"citations_without_sentence", "sentences_without_predication", "sentences_without_entity"}
    if not _has_column(con, "entity", "sentence_id"):
        for name in ("entity_sentence_missing", "sentences_without_entity"):
            checks.pop(name)
            out[name] = {"n": None, "status": "not_assessable", "reason": "entity sentence_id unresolved"}
    for name, sql in checks.items():
        logger.info("phase=integrity_%s", name)
        out[name] = {
            "n": con.execute(sql).fetchone()[0],
            "status": "computed",
            "interpretation": "structurally_allowed" if name in allowed else "constraint_or_crosscheck",
        }
    out["aux_row_difference_is_not_by_itself_corruption"] = True
    out["dbinfo_claims_aux_one_to_one"] = True
    return out


def _scope_e(con, out_dir: Path, logger) -> dict:
    logger.info("phase_start phase=scope_e_semtype")
    _make_scope(con, "scope_e_gngm_neop", SCOPE_E_GENE_SEMTYPE, SCOPE_E_DISEASE_SEMTYPE, "E_gngm_neop_operational")
    _make_scope(con, "scope_e_aapp_neop", SCOPE_E_PROTEIN_SEMTYPE, SCOPE_E_DISEASE_SEMTYPE, "E_aapp_neop_operational")
    gpath = out_dir / "scope_e_gngm_neop.parquet"
    apath = out_dir / "scope_e_aapp_neop.parquet"
    con.execute(f"COPY scope_e_gngm_neop TO '{gpath.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.execute(f"COPY scope_e_aapp_neop TO '{apath.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.execute(
        """
        CREATE OR REPLACE TABLE scope_e_gngm_pair AS
        SELECT gene_cui, disease_cui, COUNT(*) AS n_predication, COUNT(DISTINCT pmid) AS n_pmid
        FROM scope_e_gngm_neop
        GROUP BY 1, 2
        """
    )
    return {
        "definition": "Operational subset of predications whose semantic types are exactly gngm and neop, one on each side. This is not a complete SemMedDB cancer set and is not defined from CORE pairs.",
        "aapp_definition": "Separate operational subset aapp with neop. Not merged into the gngm-neop set.",
        "gngm_neop_rows": con.execute("SELECT COUNT(*) FROM scope_e_gngm_neop").fetchone()[0],
        "gngm_neop_pairs": con.execute("SELECT COUNT(*) FROM scope_e_gngm_pair").fetchone()[0],
        "gngm_neop_pmid": con.execute("SELECT COUNT(DISTINCT pmid) FROM scope_e_gngm_neop").fetchone()[0],
        "aapp_neop_rows": con.execute("SELECT COUNT(*) FROM scope_e_aapp_neop").fetchone()[0],
        "gene_position_gngm": _pairs(con, "SELECT gene_position, COUNT(*) n FROM scope_e_gngm_neop GROUP BY 1"),
        "outputs": {"gngm_neop": str(gpath), "aapp_neop": str(apath)},
        "gene_id_attachment": "not_in_this_step",
    }


def _make_scope(con, table: str, gene_sty: str, disease_sty: str, scope: str) -> None:
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {table} AS
        SELECT
          '{scope}' AS scope,
          predication_id, sentence_id, pmid, predicate, negation_prefix,
          CASE WHEN subject_semtype = '{gene_sty}' THEN subject_cui ELSE object_cui END AS gene_cui,
          CASE WHEN subject_semtype = '{disease_sty}' THEN subject_cui ELSE object_cui END AS disease_cui,
          CASE WHEN subject_semtype = '{gene_sty}' THEN 'gene_subject' ELSE 'gene_object' END AS gene_position,
          '{gene_sty}' AS gene_semtype,
          '{disease_sty}' AS disease_semtype
        FROM predication
        WHERE (subject_semtype = '{gene_sty}' AND object_semtype = '{disease_sty}')
           OR (subject_semtype = '{disease_sty}' AND object_semtype = '{gene_sty}')
        """
    )


def _text(con, logger) -> dict:
    logger.info("phase_start phase=text_length")
    return {
        "sentence_text_length": _length_summary(con, "sentence", "text_len", "sentence_rows", "characters"),
        "duplicate_sentence_text_global": con.execute(
            """
            SELECT COUNT(*) FROM (
              SELECT sentence_text FROM sentence
              WHERE sentence_text IS NOT NULL AND sentence_text <> ''
              GROUP BY 1 HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0],
    }


def _time(con) -> dict:
    return {
        "publication_year_field": "DP",
        "by_year": _pairs(
            con,
            """
            SELECT dp_year, COUNT(*) AS n_citation
            FROM citations
            WHERE dp_status = 'parsed_single'
            GROUP BY 1
            ORDER BY 1
            """,
        ),
        "active_pairs_note": "Full-database predication pairs by DP year are reported as pmid-year citation counts here. Pair-year counts for the operational gene-neoplasm subset are in scope_e.",
    }


def _length_summary(con, table: str, col: str, population: str, unit: str) -> dict:
    con.execute(f"CREATE OR REPLACE TABLE _len AS SELECT {col} AS n FROM {table} WHERE {col} IS NOT NULL")
    got = summarize_groups_sql(con, "_len", "n", population, unit)
    got["population_note"] = "Each row is one record, not a grouped count. Zeros are empty texts that have a stored length of 0."
    return got


def _availability(con, col: str) -> dict:
    row = con.execute(
        f"""
        SELECT
          COUNT(*) FILTER (WHERE {col} IS NULL) AS sql_null,
          COUNT(*) FILTER (WHERE {col} = '') AS empty_string,
          COUNT(*) FILTER (WHERE {col} = CHR(92) || 'N') AS backslash_N,
          COUNT(*) FILTER (WHERE {col} IS NOT NULL AND {col} <> '' AND {col} <> CHR(92) || 'N') AS present
        FROM sentence
        """
    ).fetchone()
    return {"sql_null": row[0], "empty_string": row[1], "backslash_N": row[2], "present": row[3], "status": "computed"}


def _pairs(con, sql: str):
    return [{"value": r[0], "n": r[1]} for r in con.execute(sql).fetchall()]

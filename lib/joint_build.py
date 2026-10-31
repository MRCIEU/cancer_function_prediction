"""Pair-set relations, PMID coverage, sensitivity, and observable record states."""

from __future__ import annotations

from pathlib import Path

from lib.relations import micro_coverage, ratio
from lib.schema import SPARQL_EXAMPLE_EXCLUDED_CUIS
from lib.semmed_columns import canonical_id_expr
from lib.stats import summarize_numeric
from lib.textnorm import normalize_text


def run_joint(con, paths: dict, out_dir: Path, logger, rule: str, entity_resolution: dict | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    entity_resolution = entity_resolution or {}
    for name, path in paths.items():
        if name == "entity" and entity_resolution.get("column"):
            column = entity_resolution["column"]
            con.execute(
                f"""
                CREATE OR REPLACE VIEW entity AS
                SELECT * EXCLUDE ({column}), {canonical_id_expr(column)} AS sentence_id
                FROM read_parquet('{Path(path).as_posix()}')
                """
            )
        else:
            con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{Path(path).as_posix()}')")
    logger.info("phase_start phase=pair_relations")
    _relations(con, "core_pair_pmid", "semmed_pair_pmid", "rel_all")
    con.execute(f"COPY rel_all TO '{(out_dir / 'pair_relations.parquet').as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    summary = {"rule": rule, "populations": {}}
    summary["populations"]["all_available"] = _population_summary(con, "rel_all", "core_pair_pmid", "semmed_pair_pmid", rule)
    _sensitivity(con, logger, summary, rule)
    logger.info("phase_start phase=observable_states")
    _observable(con, out_dir, logger)
    summary["observable"] = _observable_summary(con)
    summary["pmid_route"] = _pmid_route_summary(con, rule)
    summary["representation"] = _representation(con, logger)
    summary["scope_e_only_pairs"] = _scope_e_only(con)
    summary["closure"] = _closure(con)
    if summary["closure"]["core_only_and_semmed_only_overlap"] != 0:
        raise SystemExit("closure failure: a pair-PMID is in both exclusive sets")
    if summary["closure"]["intersection_missing_from_both"] != 0:
        raise SystemExit("closure failure: intersection pair-PMID missing from one side")
    _write_class_counts(con, out_dir)
    return summary


def _relations(con, core_view: str, sem_view: str, out_table: str) -> None:
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {out_table} AS
        WITH cu AS (
          SELECT gene_id, disease_cui, COUNT(DISTINCT pmid) AS n_core
          FROM {core_view} GROUP BY 1, 2
        ), su AS (
          SELECT gene_id, disease_cui, COUNT(DISTINCT pmid) AS n_semmed
          FROM {sem_view} GROUP BY 1, 2
        ), inter AS (
          SELECT c.gene_id, c.disease_cui, COUNT(*) AS n_intersection
          FROM {core_view} c
          JOIN {sem_view} s USING (gene_id, disease_cui, pmid)
          GROUP BY 1, 2
        ), universe AS (
          SELECT gene_id, disease_cui FROM pair
        )
        SELECT
          u.gene_id,
          u.disease_cui,
          COALESCE(cu.n_core, 0) AS n_core,
          COALESCE(su.n_semmed, 0) AS n_semmed,
          COALESCE(inter.n_intersection, 0) AS n_intersection,
          COALESCE(cu.n_core, 0) + COALESCE(su.n_semmed, 0) - COALESCE(inter.n_intersection, 0) AS n_union,
          COALESCE(cu.n_core, 0) - COALESCE(inter.n_intersection, 0) AS n_core_only,
          COALESCE(su.n_semmed, 0) - COALESCE(inter.n_intersection, 0) AS n_semmed_only,
          CASE
            WHEN COALESCE(cu.n_core,0)=0 AND COALESCE(su.n_semmed,0)=0 THEN 'both_empty'
            WHEN COALESCE(cu.n_core,0)=0 THEN 'core_pmids_empty'
            WHEN COALESCE(su.n_semmed,0)=0 THEN 'semmed_pmids_empty'
            WHEN COALESCE(inter.n_intersection,0)=COALESCE(cu.n_core,0)
             AND COALESCE(inter.n_intersection,0)=COALESCE(su.n_semmed,0) THEN 'identical'
            WHEN COALESCE(inter.n_intersection,0)=COALESCE(cu.n_core,0)
             AND COALESCE(su.n_semmed,0)>COALESCE(cu.n_core,0) THEN 'core_subset'
            WHEN COALESCE(inter.n_intersection,0)=COALESCE(su.n_semmed,0)
             AND COALESCE(cu.n_core,0)>COALESCE(su.n_semmed,0) THEN 'semmed_subset'
            WHEN COALESCE(inter.n_intersection,0)>0 THEN 'partial_overlap'
            ELSE 'disjoint'
          END AS relation_class
        FROM universe u
        LEFT JOIN cu USING (gene_id, disease_cui)
        LEFT JOIN su USING (gene_id, disease_cui)
        LEFT JOIN inter USING (gene_id, disease_cui)
        """
    )


def _population_summary(con, rel_table: str, core_view: str, sem_view: str, rule: str) -> dict:
    rows = con.execute(
        f"""
        SELECT relation_class, n_core, n_semmed, n_intersection, n_union, n_core_only, n_semmed_only
        FROM {rel_table}
        """
    ).fetchall()
    classes = {}
    jaccards = []
    covered = []
    increments = []
    dicts = []
    for rel, n_c, n_s, n_i, n_u, n_co, n_so in rows:
        classes[rel] = classes.get(rel, 0) + 1
        item = {"n_core": n_c, "n_semmed": n_s, "n_intersection": n_i, "n_union": n_u, "n_core_only": n_co, "n_semmed_only": n_so, "relation_class": rel}
        dicts.append(item)
        if rel in {"identical", "core_subset", "semmed_subset", "partial_overlap", "disjoint"} and n_u:
            jaccards.append(n_i / n_u)
            covered.append(n_i / n_c if n_c else None)
            increments.append(n_so / n_c if n_c else None)
    covered = [v for v in covered if v is not None]
    increments = [v for v in increments if v is not None]
    global_core = con.execute(f"SELECT COUNT(DISTINCT pmid) FROM {core_view}").fetchone()[0]
    global_sem = con.execute(f"SELECT COUNT(DISTINCT pmid) FROM {sem_view}").fetchone()[0]
    global_inter = con.execute(
        f"""
        SELECT COUNT(*) FROM (
          SELECT DISTINCT pmid FROM {core_view}
          INTERSECT
          SELECT DISTINCT pmid FROM {sem_view}
        )
        """
    ).fetchone()[0]
    global_union = con.execute(
        f"""
        SELECT COUNT(*) FROM (
          SELECT DISTINCT pmid FROM {core_view}
          UNION
          SELECT DISTINCT pmid FROM {sem_view}
        )
        """
    ).fetchone()[0]
    return {
        "class_counts": classes,
        "n_pairs": len(rows),
        "equal_weight_jaccard": summarize_numeric(jaccards, "five_class_pairs", "jaccard"),
        "equal_weight_core_covered": summarize_numeric(covered, "five_class_pairs", "pmid_fraction"),
        "equal_weight_semmed_increment_over_core": summarize_numeric(increments, "five_class_pairs", "pmid_fraction"),
        "pair_pmid_micro_all_explicit_pairs": micro_coverage(dicts, rule, rel_table + ":all_explicit_pairs"),
        "pair_pmid_micro_five_class_only": micro_coverage(
            [row for row in dicts if row["relation_class"] in {"identical", "core_subset", "semmed_subset", "partial_overlap", "disjoint"}],
            rule,
            rel_table + ":five_class",
        ),
        "deduplicated_pmid": {
            "n_core": global_core,
            "n_semmed": global_sem,
            "n_intersection": global_inter,
            "n_union": global_union,
            "jaccard": ratio(global_inter, global_union, "deduplicated_pmid", rel_table, rule),
            "core_covered_by_semmed": ratio(global_inter, global_core, "deduplicated_pmid", rel_table, rule),
            "semmed_covered_by_core": ratio(global_inter, global_sem, "deduplicated_pmid", rel_table, rule),
        },
    }


def _sensitivity(con, logger, summary: dict, rule: str) -> None:
    logger.info("phase_start phase=sensitivity")
    years = con.execute(
        """
        SELECT core_year FROM pmid_year WHERE core_year_status = 'parsed_single' AND core_year IS NOT NULL
        INTERSECT
        SELECT semmed_dp_year FROM pmid_year WHERE semmed_dp_status = 'parsed_single' AND semmed_dp_year IS NOT NULL
        """
    ).fetchall()
    window = sorted(row[0] for row in years)
    summary["window_years"] = window
    if window:
        year_list = ", ".join(str(int(y)) for y in window)
        con.execute(
            f"""
            CREATE OR REPLACE TABLE core_pp_year AS
            SELECT c.*
            FROM core_pair_pmid c
            JOIN pmid_year y USING (pmid)
            WHERE y.core_year_status = 'parsed_single'
              AND y.core_year IN ({year_list})
              AND (y.semmed_dp_year IS NULL OR y.semmed_dp_year = y.core_year OR y.semmed_dp_status <> 'parsed_single')
            """
        )
        con.execute(
            f"""
            CREATE OR REPLACE TABLE sem_pp_year AS
            SELECT s.*
            FROM semmed_pair_pmid s
            JOIN pmid_year y USING (pmid)
            WHERE y.semmed_dp_status = 'parsed_single'
              AND y.semmed_dp_year IN ({year_list})
              AND (y.core_year IS NULL OR y.core_year = y.semmed_dp_year OR y.core_year_status <> 'parsed_single')
            """
        )
        _relations(con, "core_pp_year", "sem_pp_year", "rel_year")
        summary["populations"]["shared_observed_publication_years"] = _population_summary(con, "rel_year", "core_pp_year", "sem_pp_year", rule)
        summary["year_window_removed"] = {
            "core_pair_pmid_removed": con.execute("SELECT COUNT(*) FROM core_pair_pmid ANTI JOIN core_pp_year USING (gene_id, disease_cui, pmid)").fetchone()[0],
            "semmed_pair_pmid_removed": con.execute("SELECT COUNT(*) FROM semmed_pair_pmid ANTI JOIN sem_pp_year USING (gene_id, disease_cui, pmid)").fetchone()[0],
            "definition": "Years observed as a parsed publication year on both sides. A PMID whose CORE publicationYear and SemMedDB DP year are both parsed and disagree is removed. This does not show that the processed corpora are the same.",
        }
    else:
        summary["populations"]["shared_observed_publication_years"] = {"status": "not_applicable", "reason": "no overlapping parsed publication years"}
    con.execute(
        """
        CREATE OR REPLACE TABLE common_article AS
        SELECT c.pmid
        FROM (SELECT DISTINCT pmid FROM core_sentence WHERE pmid IS NOT NULL) c
        JOIN (SELECT DISTINCT pmid FROM citations WHERE pmid IS NOT NULL) s USING (pmid)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE core_pp_common AS
        SELECT * FROM core_pair_pmid WHERE pmid IN (SELECT pmid FROM common_article)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE sem_pp_common AS
        SELECT * FROM semmed_pair_pmid WHERE pmid IN (SELECT pmid FROM common_article)
        """
    )
    _relations(con, "core_pp_common", "sem_pp_common", "rel_common_article")
    summary["populations"]["pmid_with_article_record_in_both"] = _population_summary(con, "rel_common_article", "core_pp_common", "sem_pp_common", rule)
    summary["common_article_pmid"] = con.execute("SELECT COUNT(*) FROM common_article").fetchone()[0]
    summary["common_article_removed"] = {
        "core_pair_pmid_removed": con.execute("SELECT COUNT(*) FROM core_pair_pmid ANTI JOIN core_pp_common USING (gene_id, disease_cui, pmid)").fetchone()[0],
        "semmed_pair_pmid_removed": con.execute("SELECT COUNT(*) FROM semmed_pair_pmid ANTI JOIN sem_pp_common USING (gene_id, disease_cui, pmid)").fetchone()[0],
    }
    cui_list = ", ".join(f"'{c}'" for c in SPARQL_EXAMPLE_EXCLUDED_CUIS)
    con.execute(f"CREATE OR REPLACE TABLE core_pp_ex AS SELECT * FROM core_pair_pmid WHERE disease_cui NOT IN ({cui_list})")
    con.execute(f"CREATE OR REPLACE TABLE sem_pp_ex AS SELECT * FROM semmed_pair_pmid WHERE disease_cui NOT IN ({cui_list})")
    _relations(con, "core_pp_ex", "sem_pp_ex", "rel_exclude_example_cuis")
    summary["populations"]["excluding_sparql_example_cuis"] = _population_summary(con, "rel_exclude_example_cuis", "core_pp_ex", "sem_pp_ex", rule)
    summary["sparql_example_filter_note"] = "Sensitivity only. The four CUIs are the exclusion list in CORE SPARQL example 5, not a definition of non-cancer."


def _observable(con, out_dir: Path, logger) -> None:
    con.execute(
        """
        CREATE OR REPLACE TABLE core_only AS
        SELECT c.gene_id, c.disease_cui, c.pmid
        FROM core_pair_pmid c
        ANTI JOIN semmed_pair_pmid s USING (gene_id, disease_cui, pmid)
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE sem_only AS
        SELECT s.gene_id, s.disease_cui, s.pmid
        FROM semmed_pair_pmid s
        ANTI JOIN core_pair_pmid c USING (gene_id, disease_cui, pmid)
        """
    )
    logger.info("phase=observable_core_only records=%s", con.execute("SELECT COUNT(*) FROM core_only").fetchone()[0])
    _core_only_flags(con)
    _sem_only_flags(con)
    con.execute(f"COPY core_only_state TO '{(out_dir / 'core_only_state.parquet').as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    con.execute(f"COPY sem_only_state TO '{(out_dir / 'semmed_only_state.parquet').as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")


def _core_only_flags(con) -> None:
    con.execute(
        """
        CREATE OR REPLACE TABLE core_only_state AS
        WITH cit AS (SELECT DISTINCT pmid FROM citations WHERE pmid IS NOT NULL),
        sent AS (SELECT DISTINCT pmid FROM sem_sentence WHERE pmid IS NOT NULL),
        focus AS (SELECT DISTINCT pmid FROM core_only),
        gene AS (
          SELECT DISTINCT s.pmid, e.gene_id
          FROM sem_sentence s
          JOIN entity e ON s.sentence_id = e.sentence_id
          WHERE e.gene_id IS NOT NULL AND s.pmid IN (SELECT pmid FROM focus)
        ),
        dis AS (
          SELECT DISTINCT s.pmid, e.cui
          FROM sem_sentence s
          JOIN entity e ON s.sentence_id = e.sentence_id
          WHERE e.cui IS NOT NULL AND s.pmid IN (SELECT pmid FROM focus)
        ),
        same_sent AS (
          SELECT DISTINCT s.pmid, g.gene_id, d.cui AS disease_cui
          FROM sem_sentence s
          JOIN entity g ON s.sentence_id = g.sentence_id AND g.gene_id IS NOT NULL
          JOIN entity d ON s.sentence_id = d.sentence_id AND d.cui IS NOT NULL
          WHERE s.pmid IN (SELECT pmid FROM focus)
        ),
        exact_text AS (
          SELECT DISTINCT c.gene_id, c.disease_cui, c.pmid
          FROM core_sentence c
          JOIN sem_sentence s ON c.pmid = s.pmid AND c.content = s.sentence_text
          WHERE c.pmid IN (SELECT pmid FROM focus)
            AND c.content IS NOT NULL AND c.content <> ''
            AND s.sentence_text IS NOT NULL AND s.sentence_text <> ''
        )
        SELECT
          o.gene_id, o.disease_cui, o.pmid,
          CASE WHEN cit.pmid IS NOT NULL THEN 'present' ELSE 'absent_in_available_records' END AS semmed_article_record,
          CASE WHEN cit.pmid IS NULL THEN 'not_assessable' WHEN sent.pmid IS NOT NULL THEN 'present' ELSE 'absent_in_available_records' END AS semmed_sentence,
          CASE WHEN cit.pmid IS NULL THEN 'not_assessable' WHEN gene.gene_id IS NOT NULL THEN 'present' ELSE 'absent_in_available_records' END AS semmed_gene,
          CASE WHEN cit.pmid IS NULL THEN 'not_assessable' WHEN dis.cui IS NOT NULL THEN 'present' ELSE 'absent_in_available_records' END AS semmed_disease,
          CASE WHEN cit.pmid IS NULL THEN 'not_assessable' WHEN same_sent.gene_id IS NOT NULL THEN 'present' ELSE 'absent_in_available_records' END AS same_sentence_entities,
          'absent_in_available_records' AS pair_predication,
          CASE
            WHEN exact_text.pmid IS NOT NULL THEN 'present'
            WHEN EXISTS (
              SELECT 1 FROM core_sentence c
              WHERE c.gene_id = o.gene_id AND c.disease_cui = o.disease_cui AND c.pmid = o.pmid
                AND c.content IS NOT NULL AND c.content <> ''
            ) AND sent.pmid IS NOT NULL THEN 'absent_in_available_records'
            ELSE 'not_assessable'
          END AS text_exact,
          'not_assessable' AS text_normalized
        FROM core_only o
        LEFT JOIN cit USING (pmid)
        LEFT JOIN sent USING (pmid)
        LEFT JOIN gene ON o.pmid = gene.pmid AND o.gene_id = gene.gene_id
        LEFT JOIN dis ON o.pmid = dis.pmid AND o.disease_cui = dis.cui
        LEFT JOIN same_sent ON o.pmid = same_sent.pmid AND o.gene_id = same_sent.gene_id AND o.disease_cui = same_sent.disease_cui
        LEFT JOIN exact_text ON o.gene_id = exact_text.gene_id AND o.disease_cui = exact_text.disease_cui AND o.pmid = exact_text.pmid
        """
    )
    _fill_normalized(con, "core_only_state", "core")


def _sem_only_flags(con) -> None:
    con.execute(
        """
        CREATE OR REPLACE TABLE sem_only_state AS
        WITH core_art AS (SELECT DISTINCT pmid FROM core_sentence WHERE pmid IS NOT NULL),
        core_sent AS (
          SELECT pmid, gene_id, disease_cui, sentence_uri, content
          FROM core_sentence
          WHERE pair_status = 'explicit'
        ),
        exact_text AS (
          SELECT DISTINCT c.gene_id, c.disease_cui, c.pmid
          FROM core_sent c
          JOIN sem_sentence s ON c.pmid = s.pmid AND c.content = s.sentence_text
          WHERE c.content IS NOT NULL AND c.content <> '' AND s.sentence_text IS NOT NULL
        )
        SELECT
          o.gene_id, o.disease_cui, o.pmid,
          CASE WHEN core_art.pmid IS NOT NULL THEN 'present' ELSE 'absent_in_available_records' END AS core_article_record,
          CASE WHEN core_art.pmid IS NULL THEN 'not_assessable'
               WHEN EXISTS (SELECT 1 FROM core_sent c WHERE c.pmid = o.pmid) THEN 'present'
               ELSE 'absent_in_available_records' END AS core_sentence,
          CASE WHEN core_art.pmid IS NULL THEN 'not_assessable'
               WHEN EXISTS (SELECT 1 FROM core_sent c WHERE c.pmid = o.pmid AND c.gene_id = o.gene_id) THEN 'present'
               ELSE 'absent_in_available_records' END AS core_gene,
          CASE WHEN core_art.pmid IS NULL THEN 'not_assessable'
               WHEN EXISTS (SELECT 1 FROM core_sent c WHERE c.pmid = o.pmid AND c.disease_cui = o.disease_cui) THEN 'present'
               ELSE 'absent_in_available_records' END AS core_disease,
          CASE WHEN core_art.pmid IS NULL THEN 'not_assessable'
               WHEN EXISTS (
                 SELECT 1 FROM core_sent c
                 WHERE c.pmid = o.pmid AND c.gene_id = o.gene_id AND c.disease_cui = o.disease_cui
               ) THEN 'present' ELSE 'absent_in_available_records' END AS same_sentence_entities,
          'absent_in_available_records' AS pair_record,
          CASE WHEN exact_text.pmid IS NOT NULL THEN 'present'
               WHEN core_art.pmid IS NULL THEN 'not_assessable'
               ELSE 'absent_in_available_records' END AS text_exact,
          'not_assessable' AS text_normalized
        FROM sem_only o
        LEFT JOIN core_art USING (pmid)
        LEFT JOIN exact_text ON o.gene_id = exact_text.gene_id AND o.disease_cui = exact_text.disease_cui AND o.pmid = exact_text.pmid
        """
    )


def _fill_normalized(con, table: str, side: str) -> None:
    if side != "core":
        return
    n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    if n > 250000:
        return
    try:
        con.create_function("norm_text", normalize_text)
    except Exception:
        return
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {table} AS
        SELECT * REPLACE (
          CASE
            WHEN text_exact = 'present' THEN 'present'
            WHEN text_exact = 'not_assessable' THEN 'not_assessable'
            WHEN EXISTS (
              SELECT 1
              FROM core_sentence c
              JOIN sem_sentence s ON c.pmid = s.pmid
              WHERE c.gene_id = {table}.gene_id AND c.disease_cui = {table}.disease_cui AND c.pmid = {table}.pmid
                AND c.content IS NOT NULL AND s.sentence_text IS NOT NULL
                AND norm_text(c.content) = norm_text(s.sentence_text)
            ) THEN 'present'
            ELSE 'absent_in_available_records'
          END AS text_normalized
        )
        FROM {table}
        """
    )


def _observable_summary(con) -> dict:
    def counts(table: str, col: str):
        return [{"value": r[0], "n": r[1]} for r in con.execute(f"SELECT {col}, COUNT(*) FROM {table} GROUP BY 1 ORDER BY 2 DESC").fetchall()]

    return {
        "core_only_pair_pmid": con.execute("SELECT COUNT(*) FROM core_only_state").fetchone()[0],
        "semmed_only_pair_pmid": con.execute("SELECT COUNT(*) FROM sem_only_state").fetchone()[0],
        "core_only_article": counts("core_only_state", "semmed_article_record"),
        "core_only_sentence": counts("core_only_state", "semmed_sentence"),
        "core_only_gene": counts("core_only_state", "semmed_gene"),
        "core_only_disease": counts("core_only_state", "semmed_disease"),
        "core_only_same_sentence": counts("core_only_state", "same_sentence_entities"),
        "core_only_text_exact": counts("core_only_state", "text_exact"),
        "semmed_only_article": counts("sem_only_state", "core_article_record"),
        "semmed_only_same_sentence": counts("sem_only_state", "same_sentence_entities"),
        "label": "可观察差异状态。这些计数不是 NER 或关系抽取错误率。",
        "text_normalized_note": "NFKC/whitespace comparison is run only when exclusive CORE pair-PMID rows are at most 250000. Otherwise text_normalized stays not_assessable. Exact text comparison still runs.",
        "pair_predication_on_core_only": "Forced absent because these PMIDs were defined by the anti-join against primary matches. A non-zero closure check would mean the anti-join and the match rule disagree.",
    }


def _pmid_route_summary(con, rule: str) -> dict:
    row = con.execute(
        """
        SELECT
          COUNT(*) AS n_pmid,
          COUNT(*) FILTER (WHERE in_citations) AS in_citations,
          COUNT(*) FILTER (WHERE n_sentence > 0) AS with_sentence,
          COUNT(*) FILTER (WHERE n_predication > 0) AS with_predication,
          COUNT(*) FILTER (WHERE n_entity > 0) AS with_entity
        FROM pmid_route
        """
    ).fetchone()
    n = row[0]
    def pack(numer, label):
        return ratio(numer, n, "pmid", "D_core_pmid_projection", rule) | {"label": label}
    return {
        "n_core_pmid": n,
        "in_semmed_citations": pack(row[1], "citation row"),
        "with_semmed_sentence": pack(row[2], "sentence row"),
        "with_semmed_predication": pack(row[3], "any predication, not restricted to the CORE pair"),
        "with_semmed_entity": pack(row[4], "entity row joined through sentence"),
    }


def _representation(con, logger) -> dict:
    logger.info("phase_start phase=representation")
    return {
        "common_article_pmid": con.execute("SELECT COUNT(*) FROM common_article").fetchone()[0],
        "intersection_pair_pmid": con.execute(
            """
            SELECT COUNT(*) FROM core_pair_pmid c
            JOIN semmed_pair_pmid s USING (gene_id, disease_cui, pmid)
            """
        ).fetchone()[0],
        "exact_sentence_text_matches_on_intersection_pmids": con.execute(
            """
            SELECT COUNT(*) FROM (
              SELECT DISTINCT c.pmid, c.sentence_uri, s.sentence_id
              FROM core_sentence c
              JOIN sem_sentence s ON c.pmid = s.pmid AND c.content = s.sentence_text
              WHERE c.pmid IN (
                SELECT pmid FROM core_pair_pmid
                INTERSECT
                SELECT pmid FROM semmed_pair_pmid
              )
                AND c.content IS NOT NULL AND c.content <> ''
            )
            """
        ).fetchone()[0],
        "note": "Predicate or CORE label differences are not treated as PMID disagreement. No fuzzy sentence alignment was used for the primary counts.",
    }


def _scope_e_only(con) -> dict:
    n_e = con.execute("SELECT COUNT(*) FROM scope_e_gene_pairs").fetchone()[0]
    n_only = con.execute(
        """
        SELECT COUNT(*) FROM scope_e_gene_pairs e
        ANTI JOIN pair p USING (gene_id, disease_cui)
        """
    ).fetchone()[0]
    return {
        "scope": "E_gngm_neop_with_same_sentence_numeric_gene_id",
        "n_pairs": n_e,
        "n_not_in_core_explicit_pairs": n_only,
        "n_shared_gene_disease_id": n_e - n_only,
        "note": "This is the operational gngm-neop subset after attaching a unique same-sentence Entrez ID. It is not the complete SemMedDB cancer set.",
    }


def _closure(con) -> dict:
    overlap = con.execute(
        """
        SELECT COUNT(*) FROM core_only c
        JOIN sem_only s USING (gene_id, disease_cui, pmid)
        """
    ).fetchone()[0]
    missing = con.execute(
        """
        SELECT COUNT(*) FROM (
          SELECT gene_id, disease_cui, pmid FROM core_pair_pmid
          INTERSECT
          SELECT gene_id, disease_cui, pmid FROM semmed_pair_pmid
        ) i
        ANTI JOIN core_pair_pmid c USING (gene_id, disease_cui, pmid)
        """
    ).fetchone()[0]
    class_n = con.execute("SELECT COUNT(*) FROM rel_all").fetchone()[0]
    pair_n = con.execute("SELECT COUNT(*) FROM pair").fetchone()[0]
    return {
        "core_only_and_semmed_only_overlap": overlap,
        "intersection_missing_from_both": missing,
        "relation_rows": class_n,
        "explicit_pairs": pair_n,
        "relation_rows_equal_explicit_pairs": class_n == pair_n,
    }


def _write_class_counts(con, out_dir: Path) -> None:
    con.execute(
        f"""
        COPY (
          SELECT relation_class, COUNT(*) AS n_pair,
                 SUM(n_core) AS sum_core_pmid_slots,
                 SUM(n_semmed) AS sum_semmed_pmid_slots,
                 SUM(n_intersection) AS sum_intersection_slots
          FROM rel_all
          GROUP BY 1
          ORDER BY 1
        ) TO '{(out_dir / 'class_counts.tsv').as_posix()}' (HEADER, DELIMITER '\t')
        """
    )

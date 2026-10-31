#!/usr/bin/env python3
"""V21: apply the enabled id_alignment_v2 paths. Does not overwrite id_rule_v1."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE_ROOT = Path("/home/b5ac/freddieyu.b5ac/project_2")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CODE_ROOT))

from lib.duckutil import connect  # noqa: E402
from lib.runtime import prologue, read_json, setup_logging, write_json  # noqa: E402
from lib.semmed_columns import canonical_id_expr  # noqa: E402
from rules import RULE_VERSION, decide_json  # noqa: E402

RESULTS = Path("/projects/b5ac/project_2/results")
OUT = RESULTS / "03_integrated_analysis" / "id_alignment_v2_round1"
RULES = OUT / "derived" / "rules_v2.json"
V20_DONE = OUT / "derived" / "v20_DONE.json"
PREDICATE = RESULTS / "02_semmeddb_validation" / "tables" / "predication.parquet"
ENTITY = RESULTS / "02_semmeddb_validation" / "tables" / "entity.parquet"
SCOPE = RESULTS / "02_semmeddb_validation" / "tables" / "scope_e_gngm_neop.parquet"
PAIR = RESULTS / "01_core_audit" / "tables" / "pair.parquet"
GENERIC = RESULTS / "02_semmeddb_validation" / "tables" / "generic_concept.parquet"


def main() -> int:
    done = read_json(V20_DONE)
    if done.get("status") != "success":
        raise SystemExit(f"V20 is not complete: {done}")
    rules = read_json(RULES)
    if rules.get("rule_version") != RULE_VERSION:
        raise SystemExit("rules file is not id_alignment_v2")
    enabled = rules["enabled"]
    logger = setup_logging(OUT / "logs" / f"v21_{os.environ.get('SLURM_JOB_ID', 'local')}.log")
    info = prologue("V21", logger, OUT, [str(RULES)])
    con = connect(OUT / "derived" / "v21.duckdb", "72GB", OUT / "tmp")
    # special: a missing entity gene list is NULL and is part of the decision. The default skips the call and returns NULL.
    con.create_function("decide_map", decide_json, return_type="VARCHAR", null_handling="special")
    build_arguments(con, logger)
    build_entity_support(con, logger)
    mapped = apply_rules(con, logger, enabled)
    write_outputs(con, logger)
    summary = {
        "rule_version": RULE_VERSION,
        "code_fingerprint": info["code_fingerprint"],
        "enabled": enabled,
        "counts": mapped,
    }
    write_json(OUT / "derived" / "v21_summary.json", summary)
    write_json(OUT / "derived" / "v21_DONE.json", {"status": "success", "job": "V21", "summary": str(OUT / "derived" / "v21_summary.json")})
    logger.info("phase=v21_success")
    return 0


def build_arguments(con, logger) -> None:
    logger.info("phase_start phase=core_anchored_arguments")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE disease AS
        SELECT DISTINCT disease_cui FROM read_parquet('{PAIR.as_posix()}')
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TABLE arg_input AS
        SELECT 'core_anchored' AS population,
               p.predication_id, p.sentence_id, p.pmid, p.predicate, p.negation_prefix,
               p.subject_cui, p.subject_name, p.subject_semtype,
               p.object_cui, p.object_name, p.object_semtype,
               d.disease_cui,
               'object' AS gene_side,
               p.object_cui AS argument,
               p.object_name AS argument_name,
               p.object_semtype AS argument_semtype
        FROM read_parquet('{PREDICATE.as_posix()}') p
        JOIN disease d ON p.subject_cui = d.disease_cui
        WHERE p.pmid IS NOT NULL AND p.object_cui <> d.disease_cui
        UNION ALL
        SELECT 'core_anchored',
               p.predication_id, p.sentence_id, p.pmid, p.predicate, p.negation_prefix,
               p.subject_cui, p.subject_name, p.subject_semtype,
               p.object_cui, p.object_name, p.object_semtype,
               d.disease_cui,
               'subject',
               p.subject_cui,
               p.subject_name,
               p.subject_semtype
        FROM read_parquet('{PREDICATE.as_posix()}') p
        JOIN disease d ON p.object_cui = d.disease_cui
        WHERE p.pmid IS NOT NULL AND p.subject_cui <> d.disease_cui
        UNION ALL
        SELECT 'scope_e',
               p.predication_id, p.sentence_id, p.pmid, p.predicate, p.negation_prefix,
               p.subject_cui, p.subject_name, p.subject_semtype,
               p.object_cui, p.object_name, p.object_semtype,
               e.disease_cui,
               CASE WHEN e.gene_position = 'gene_subject' THEN 'subject' ELSE 'object' END,
               e.gene_cui,
               CASE WHEN e.gene_position = 'gene_subject' THEN p.subject_name ELSE p.object_name END,
               e.gene_semtype
        FROM read_parquet('{SCOPE.as_posix()}') e
        JOIN read_parquet('{PREDICATE.as_posix()}') p USING (predication_id)
        """
    )
    logger.info("phase=arguments rows=%s", con.execute("SELECT COUNT(*) FROM arg_input").fetchone()[0])


def build_entity_support(con, logger) -> None:
    logger.info("phase_start phase=entity_support_scan")
    canon = canonical_id_expr("pre_cui_numeric_1")
    con.execute("CREATE OR REPLACE TABLE need_sentence AS SELECT DISTINCT sentence_id FROM arg_input WHERE sentence_id IS NOT NULL")
    con.execute(
        f"""
        CREATE OR REPLACE TABLE entity_group AS
        SELECT {canon} AS sentence_id,
               cui,
               gene_id,
               count(*) AS n_rows,
               min(entity_id) AS entity_id
        FROM read_parquet('{ENTITY.as_posix()}')
        WHERE {canon} IN (SELECT sentence_id FROM need_sentence)
        GROUP BY 1, 2, 3
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE sentence_entity AS
        SELECT sentence_id, sum(n_rows) AS n_rows
        FROM entity_group
        GROUP BY 1
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE cui_entity AS
        SELECT sentence_id,
               cui,
               sum(n_rows) AS n_rows,
               string_agg(gene_id, ',') FILTER (WHERE gene_id IS NOT NULL) AS gene_csv,
               min(entity_id) FILTER (WHERE gene_id IS NOT NULL) AS entity_id
        FROM entity_group
        GROUP BY 1, 2
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE direct_entity AS
        SELECT sentence_id, gene_id, sum(n_rows) AS n_rows, min(entity_id) AS entity_id
        FROM entity_group
        WHERE gene_id IS NOT NULL
        GROUP BY 1, 2
        """
    )
    logger.info("phase=entity_support groups=%s", con.execute("SELECT COUNT(*) FROM entity_group").fetchone()[0])


def apply_rules(con, logger, enabled: dict) -> dict:
    logger.info("phase_start phase=apply_rules")
    path_a = "true" if enabled.get("A_direct_entrez") else "false"
    path_b = "true" if enabled.get("B_entity_gene_id") else "false"
    path_c = "true" if enabled.get("C_composite_entrez") else "false"
    con.execute(
        f"""
        CREATE OR REPLACE TABLE mapped AS
        WITH parsed AS (
          SELECT a.*,
                 CAST(decide_map(
                   a.argument,
                   a.argument_semtype,
                   CAST(COALESCE(se.n_rows, 0) AS BIGINT),
                   de.gene_id IS NOT NULL,
                   CAST(COALESCE(ce.n_rows, 0) AS BIGINT),
                   ce.gene_csv,
                   g.cui IS NOT NULL,
                   {path_a}, {path_b}, {path_c}
                 ) AS JSON) AS decision,
                 COALESCE(de.entity_id, ce.entity_id) AS supporting_record_id
          FROM arg_input a
          LEFT JOIN sentence_entity se ON a.sentence_id = se.sentence_id
          LEFT JOIN cui_entity ce
            ON a.sentence_id = ce.sentence_id
           AND ce.cui = CASE
                WHEN regexp_matches(a.argument, '^C[0-9]+$') THEN a.argument
                WHEN regexp_matches(split_part(a.argument, '|', 1), '^C[0-9]+$') THEN split_part(a.argument, '|', 1)
                WHEN regexp_matches(split_part(a.argument, '|', 2), '^C[0-9]+$') THEN split_part(a.argument, '|', 2)
                ELSE NULL
              END
          LEFT JOIN direct_entity de
            ON a.sentence_id = de.sentence_id
           AND de.gene_id = CASE
                WHEN regexp_matches(a.argument, '^[0-9]+$') THEN CASE WHEN ltrim(a.argument, '0') = '' THEN '0' ELSE ltrim(a.argument, '0') END
                WHEN regexp_matches(a.argument, '^C[0-9]+\\|[0-9]+$') THEN CASE WHEN ltrim(split_part(a.argument, '|', 2), '0') = '' THEN '0' ELSE ltrim(split_part(a.argument, '|', 2), '0') END
                WHEN regexp_matches(a.argument, '^[0-9]+\\|C[0-9]+$') THEN CASE WHEN ltrim(split_part(a.argument, '|', 1), '0') = '' THEN '0' ELSE ltrim(split_part(a.argument, '|', 1), '0') END
                WHEN regexp_matches(a.argument, '^[0-9]+\\|[0-9]+$')
                     AND ltrim(split_part(a.argument, '|', 1), '0') = ltrim(split_part(a.argument, '|', 2), '0')
                  THEN CASE WHEN ltrim(split_part(a.argument, '|', 1), '0') = '' THEN '0' ELSE ltrim(split_part(a.argument, '|', 1), '0') END
                ELSE NULL
              END
          LEFT JOIN (SELECT DISTINCT cui FROM read_parquet('{GENERIC.as_posix()}')) g
            ON g.cui = CASE
                WHEN regexp_matches(a.argument, '^C[0-9]+$') THEN a.argument
                WHEN regexp_matches(split_part(a.argument, '|', 1), '^C[0-9]+$') THEN split_part(a.argument, '|', 1)
                WHEN regexp_matches(split_part(a.argument, '|', 2), '^C[0-9]+$') THEN split_part(a.argument, '|', 2)
                ELSE NULL
              END
        )
        SELECT population, predication_id, sentence_id, pmid, predicate, negation_prefix,
               subject_cui, subject_name, subject_semtype,
               object_cui, object_name, object_semtype,
               disease_cui, gene_side, argument, argument_name, argument_semtype,
               json_extract_string(decision, '$.argument_format') AS argument_format,
               CAST(json_extract(decision, '$.candidate_gene_ids') AS VARCHAR) AS candidate_gene_ids,
               json_extract_string(decision, '$.accepted_gene_id') AS accepted_gene_id,
               json_extract_string(decision, '$.mapping_route') AS mapping_route,
               json_extract_string(decision, '$.mapping_status') AS mapping_status,
               json_extract_string(decision, '$.entity_check') AS entity_check,
               supporting_record_id AS supporting_record_ids,
               '{RULE_VERSION}' AS rule_version,
               decision AS mapping_json
        FROM parsed
        """
    )
    logger.info("phase=mapped rows=%s", con.execute("SELECT COUNT(*) FROM mapped").fetchone()[0])
    con.execute(
        f"""
        CREATE OR REPLACE TABLE accepted_match AS
        SELECT m.*, p.gene_id AS core_gene_id
        FROM mapped m
        JOIN read_parquet('{PAIR.as_posix()}') p
          ON m.population = 'core_anchored'
         AND m.mapping_status = 'accepted'
         AND m.accepted_gene_id = p.gene_id
         AND m.disease_cui = p.disease_cui
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE semmed_pair_pmid AS
        SELECT accepted_gene_id AS gene_id,
               disease_cui,
               pmid,
               count(*) AS n_predication,
               count(DISTINCT predicate) AS n_predicate,
               count(DISTINCT sentence_id) AS n_sentence,
               string_agg(DISTINCT mapping_route, ',') AS mapping_routes
        FROM accepted_match
        WHERE pmid IS NOT NULL
        GROUP BY 1, 2, 3
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE scope_e_status AS
        SELECT mapping_status, argument_format, mapping_route, count(*) AS n
        FROM mapped
        WHERE population = 'scope_e'
        GROUP BY 1, 2, 3
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE scope_e_gene_pairs AS
        SELECT accepted_gene_id AS gene_id, disease_cui, count(*) AS n_predication, count(DISTINCT pmid) AS n_pmid
        FROM mapped
        WHERE population = 'scope_e' AND mapping_status = 'accepted' AND accepted_gene_id IS NOT NULL
        GROUP BY 1, 2
        """
    )
    rows = con.execute(
        """
        SELECT population, mapping_status, count(*) AS n
        FROM mapped
        GROUP BY 1, 2
        ORDER BY 1, 3 DESC
        """
    ).fetchall()
    return {
        "by_population_status": [{"population": row[0], "mapping_status": row[1], "n": row[2]} for row in rows],
        "accepted_core_predications": con.execute("SELECT COUNT(*) FROM accepted_match").fetchone()[0],
        "accepted_core_pairs": con.execute("SELECT COUNT(*) FROM (SELECT DISTINCT gene_id, disease_cui FROM semmed_pair_pmid)").fetchone()[0],
        "accepted_core_pair_pmid": con.execute("SELECT COUNT(*) FROM semmed_pair_pmid").fetchone()[0],
        "scope_e_accepted_pairs": con.execute("SELECT COUNT(*) FROM scope_e_gene_pairs").fetchone()[0],
    }


def write_outputs(con, logger) -> None:
    folder = OUT / "tables"
    folder.mkdir(parents=True, exist_ok=True)
    for name in ("mapped", "accepted_match", "semmed_pair_pmid", "scope_e_status", "scope_e_gene_pairs"):
        path = folder / f"{name}.parquet"
        con.execute(f"COPY {name} TO '{path.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        logger.info("phase=wrote table=%s", path)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        raise SystemExit(1)

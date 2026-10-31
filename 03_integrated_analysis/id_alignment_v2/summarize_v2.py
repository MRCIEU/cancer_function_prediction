#!/usr/bin/env python3
"""V22: recompute mapping-dependent comparisons and write the v2 report."""

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
from lib.joint_build import run_joint  # noqa: E402
from lib.runtime import prologue, read_json, setup_logging, write_json  # noqa: E402

RESULTS = Path("/projects/b5ac/project_2/results")
OUT = RESULTS / "03_integrated_analysis" / "id_alignment_v2_round1"
ROUND1 = RESULTS / "03_integrated_analysis" / "data_analysis_round1"
V1_MATCH = ROUND1 / "derived" / "j03" / "matched_predication.parquet"
V1_PMID = ROUND1 / "derived" / "j03" / "semmed_pair_pmid.parquet"
J03_SUMMARY = ROUND1 / "derived" / "j03" / "summary.json"
J04_SUMMARY = ROUND1 / "tables" / "summary.json"


def main() -> int:
    v21 = read_json(OUT / "derived" / "v21_DONE.json")
    if v21.get("status") != "success":
        raise SystemExit(f"V21 is not complete: {v21}")
    rules = read_json(OUT / "derived" / "rules_v2.json")
    v21_summary = read_json(OUT / "derived" / "v21_summary.json")
    logger = setup_logging(OUT / "logs" / f"v22_{os.environ.get('SLURM_JOB_ID', 'local')}.log")
    info = prologue("V22", logger, OUT, [str(OUT / "tables" / "mapped.parquet")])
    con = connect(OUT / "derived" / "v22.duckdb", "72GB", OUT / "tmp")
    comparison = compare(con, logger)
    joint = rerun_joint(con, logger)
    single_db = {
        "core_summary": str(RESULTS / "01_core_audit" / "tables" / "summary.json"),
        "semmed_summary": str(RESULTS / "02_semmeddb_validation" / "tables" / "summary.json"),
        "note": "Single-database counts are unchanged and were not recomputed.",
    }
    payload = {
        "code_fingerprint": info["code_fingerprint"],
        "rules_enabled": rules.get("enabled"),
        "documents": rules.get("documents"),
        "samples": rules.get("samples"),
        "coordinates": rules.get("coordinates"),
        "v21": v21_summary,
        "comparison": comparison,
        "joint": joint,
        "single_db": single_db,
        "v1_historical": {
            "source": str(J03_SUMMARY),
            "pairs": 276,
            "pair_pmid": 426,
            "role": "Historical id_rule_v1 baseline. Not the v2 main result.",
        },
    }
    write_json(OUT / "derived" / "v22_summary.json", payload)
    (OUT / "report.md").write_text(render(payload), encoding="utf-8")
    write_json(OUT / "derived" / "v22_DONE.json", {"status": "success", "job": "V22", "report": str(OUT / "report.md")})
    logger.info("phase=v22_success")
    return 0


def compare(con, logger) -> dict:
    logger.info("phase_start phase=v1_v2_compare")
    con.execute(f"CREATE OR REPLACE TABLE v1_match AS SELECT * FROM read_parquet('{V1_MATCH.as_posix()}')")
    con.execute(f"CREATE OR REPLACE TABLE v2_match AS SELECT * FROM read_parquet('{(OUT / 'tables' / 'accepted_match.parquet').as_posix()}')")
    con.execute(f"CREATE OR REPLACE TABLE mapped AS SELECT * FROM read_parquet('{(OUT / 'tables' / 'mapped.parquet').as_posix()}')")
    con.execute(f"CREATE OR REPLACE TABLE v1_pmid AS SELECT * FROM read_parquet('{V1_PMID.as_posix()}')")
    con.execute(f"CREATE OR REPLACE TABLE v2_pmid AS SELECT * FROM read_parquet('{(OUT / 'tables' / 'semmed_pair_pmid.parquet').as_posix()}')")
    pred = con.execute(
        """
        WITH v1 AS (SELECT DISTINCT predication_id, gene_id, disease_cui FROM v1_match),
             v2 AS (SELECT DISTINCT predication_id, accepted_gene_id AS gene_id, disease_cui, mapping_route FROM v2_match)
        SELECT
          (SELECT COUNT(*) FROM v1) AS v1_rows,
          (SELECT COUNT(*) FROM v2) AS v2_rows,
          (SELECT COUNT(*) FROM (
             SELECT predication_id, gene_id, disease_cui FROM v1
             INTERSECT
             SELECT predication_id, gene_id, disease_cui FROM v2
           )) AS retained_rows,
          (SELECT COUNT(*) FROM v1 ANTI JOIN v2 USING (predication_id, gene_id, disease_cui)) AS removed_rows,
          (SELECT COUNT(*) FROM v2 ANTI JOIN v1 USING (predication_id, gene_id, disease_cui)) AS added_rows
        """
    ).fetchone()
    pairs = con.execute(
        """
        WITH v1 AS (SELECT DISTINCT gene_id, disease_cui FROM v1_match),
             v2 AS (SELECT DISTINCT accepted_gene_id AS gene_id, disease_cui FROM v2_match)
        SELECT
          (SELECT COUNT(*) FROM v1) AS v1_pairs,
          (SELECT COUNT(*) FROM v2) AS v2_pairs,
          (SELECT COUNT(*) FROM (
             SELECT gene_id, disease_cui FROM v1
             INTERSECT
             SELECT gene_id, disease_cui FROM v2
           )) AS retained_pairs,
          (SELECT COUNT(*) FROM v1 ANTI JOIN v2 USING (gene_id, disease_cui)) AS removed_pairs,
          (SELECT COUNT(*) FROM v2 ANTI JOIN v1 USING (gene_id, disease_cui)) AS added_pairs
        """
    ).fetchone()
    pair_pmid = con.execute(
        """
        WITH v1 AS (SELECT DISTINCT gene_id, disease_cui, pmid FROM v1_pmid),
             v2 AS (SELECT DISTINCT gene_id, disease_cui, pmid FROM v2_pmid)
        SELECT
          (SELECT COUNT(*) FROM v1) AS v1_pair_pmid,
          (SELECT COUNT(*) FROM v2) AS v2_pair_pmid,
          (SELECT COUNT(*) FROM (
             SELECT gene_id, disease_cui, pmid FROM v1
             INTERSECT
             SELECT gene_id, disease_cui, pmid FROM v2
           )) AS retained,
          (SELECT COUNT(*) FROM v1 ANTI JOIN v2 USING (gene_id, disease_cui, pmid)) AS removed,
          (SELECT COUNT(*) FROM v2 ANTI JOIN v1 USING (gene_id, disease_cui, pmid)) AS added
        """
    ).fetchone()
    added_routes = rows(
        con,
        """
        WITH v1 AS (SELECT DISTINCT predication_id, gene_id, disease_cui FROM v1_match),
             added AS (
               SELECT v2.*
               FROM v2_match v2
               ANTI JOIN v1 ON v2.predication_id = v1.predication_id
                            AND v2.accepted_gene_id = v1.gene_id
                            AND v2.disease_cui = v1.disease_cui
             )
        SELECT mapping_route, count(*) AS n_predication, count(DISTINCT accepted_gene_id || '|' || disease_cui) AS n_pair
        FROM added
        GROUP BY 1
        ORDER BY n_predication DESC
        """,
    )
    removed_status = rows(
        con,
        """
        WITH removed AS (
          SELECT v1.predication_id, v1.gene_id, v1.disease_cui
          FROM (SELECT DISTINCT predication_id, gene_id, disease_cui FROM v1_match) v1
          ANTI JOIN (
            SELECT predication_id, accepted_gene_id AS gene_id, disease_cui FROM v2_match
          ) v2 USING (predication_id, gene_id, disease_cui)
        )
        SELECT COALESCE(m.mapping_status, 'predication_not_in_v2_input') AS mapping_status,
               COALESCE(m.argument_format, '') AS argument_format,
               count(*) AS n
        FROM removed r
        LEFT JOIN mapped m
          ON m.population = 'core_anchored'
         AND m.predication_id = r.predication_id
         AND m.disease_cui = r.disease_cui
        GROUP BY 1, 2
        ORDER BY n DESC
        """,
    )
    route_pairs = rows(
        con,
        """
        SELECT mapping_routes, count(*) AS n_pair
        FROM (
          SELECT accepted_gene_id, disease_cui, string_agg(DISTINCT mapping_route, ',' ORDER BY mapping_route) AS mapping_routes
          FROM v2_match
          GROUP BY 1, 2
        )
        GROUP BY 1
        ORDER BY n_pair DESC
        """,
    )
    status_counts = rows(
        con,
        """
        SELECT population, mapping_status, count(*) AS n
        FROM mapped
        GROUP BY 1, 2
        ORDER BY 1, n DESC
        """,
    )
    scope_counts = rows(con, "SELECT * FROM read_parquet('" + (OUT / "tables" / "scope_e_status.parquet").as_posix() + "') ORDER BY n DESC")
    return {
        "predication_links": dict(zip(["v1_rows", "v2_rows", "retained_rows", "removed_rows", "added_rows"], pred)),
        "pairs": dict(zip(["v1_pairs", "v2_pairs", "retained_pairs", "removed_pairs", "added_pairs"], pairs)),
        "pair_pmid": dict(zip(["v1_pair_pmid", "v2_pair_pmid", "retained", "removed", "added"], pair_pmid)),
        "added_by_route": added_routes,
        "removed_by_status": removed_status,
        "pair_route_combinations": route_pairs,
        "mapping_status": status_counts,
        "scope_e_status": scope_counts,
    }


def rerun_joint(con, logger) -> dict:
    logger.info("phase_start phase=joint_v2")
    j03 = read_json(J03_SUMMARY)
    resolution = j03["entity_sentence_id_resolution"]
    paths = {
        "pair": str(RESULTS / "01_core_audit" / "tables" / "pair.parquet"),
        "core_pair_pmid": str(RESULTS / "01_core_audit" / "tables" / "pair_pmid.parquet"),
        "core_sentence": str(RESULTS / "01_core_audit" / "tables" / "sentence.parquet"),
        "semmed_pair_pmid": str(OUT / "tables" / "semmed_pair_pmid.parquet"),
        "sem_sentence": str(RESULTS / "02_semmeddb_validation" / "tables" / "sentence.parquet"),
        "citations": str(RESULTS / "02_semmeddb_validation" / "tables" / "citations.parquet"),
        "entity": str(RESULTS / "02_semmeddb_validation" / "tables" / "entity.parquet"),
        "pmid_route": str(ROUND1 / "derived" / "j03" / "pmid_route.parquet"),
        "pmid_year": str(ROUND1 / "derived" / "j03" / "pmid_year.parquet"),
        "scope_e_gene_pairs": str(OUT / "tables" / "scope_e_gene_pairs.parquet"),
        "matched_predication": str(OUT / "tables" / "accepted_match.parquet"),
    }
    summary = run_joint(con, paths, OUT / "tables", logger, "id_alignment_v2", resolution)
    summary["pmid_route_note"] = "PMID route counts are read from the J03 table. That route does not depend on the gene-id rule and was not rebuilt."
    summary["entity_sentence_column"] = resolution.get("column")
    summary["entity_sentence_limit"] = "Observable gene presence still joins ENTITY through the candidate sentence column used by id_rule_v1. V20 did not treat the failure of column 2 as proof that column 1 is correct for every row."
    keep = {
        "populations": summary.get("populations"),
        "year_window_removed": summary.get("year_window_removed"),
        "common_article_pmid": summary.get("common_article_pmid"),
        "common_article_removed": summary.get("common_article_removed"),
        "observable": summary.get("observable"),
        "representation": summary.get("representation"),
        "scope_e_only_pairs": summary.get("scope_e_only_pairs"),
        "closure": summary.get("closure"),
        "pmid_route_note": summary["pmid_route_note"],
        "entity_sentence_limit": summary["entity_sentence_limit"],
    }
    write_json(OUT / "derived" / "joint_v2.json", keep)
    return keep


def rows(con, sql: str) -> list[dict]:
    cur = con.execute(sql)
    names = [item[0] for item in cur.description]
    return [dict(zip(names, row)) for row in cur.fetchall()]


def render(payload: dict) -> str:
    enabled = payload["rules_enabled"]
    samples = payload["samples"]
    coords = payload["coordinates"]
    comp = payload["comparison"]
    joint = payload["joint"]
    lines = [
        "# id_alignment_v2",
        "",
        "id_rule_v1 的 276 个 pair 仍在原目录，只作为历史基线。本报告的主结果是 v2 里映射状态为 accepted 的 CORE 锚定匹配。",
        "",
        "## 1. 哪些路径有证据",
        "",
        "SemRep v1.8 XML 说明：基因组概念的 entrezID 是 EntrezGene 标识；cui 可以是 UMLS CUI，也可以是其他唯一来源标识；begin 是提及首字符的 0-based 文档偏移。dbinfo 说明：ENTITY.GENE_ID 是 EntrezGene ID；实体、句子和 AUX 的起止位置都是文档内字符位置。这些资料没有被当成本地 CSV 的列顺序。SUBJECT_CUI 在 dbinfo 里只称为 CUI，竖线形式没有在该页定义，所以竖线路径要单独经过数据核对。",
        "",
        "启用状态：" + json.dumps(enabled, ensure_ascii=False),
        "",
        "分层样本（每个频率四分位最多 100 个不同参数，不只看出现最多的基因）：",
        "",
        _md([{"kind": "single_digits", **_pick(samples.get("single_digits", {}))}]),
        "",
        _md([{"kind": "pipe_one_or_repeated", **_pick(samples.get("pipe", {}))}]),
        "",
        "冲突样本：" + json.dumps(samples.get("conflicts", {}), ensure_ascii=False),
        "",
        "NCBI gene_info 与 gene_history 的版本信息在 `derived/rules_v2.json` 的 ncbi 段。只出现在 gene_history 的 ID 记为历史 ID，不因为当前表没有就判为无效，也不自动换成替代 ID。名称只做精确词核对，不生成映射。",
        "",
        "坐标：" + json.dumps(coords, ensure_ascii=False),
        "",
        "未转换坐标上的切片只作为对照。句子起点减去之后，0-based 和 1-based 的共同原点会抵消，剩下的是终点是否包含。达到预设阈值的那一条才写入规则；没有达到就不采用。第 2 个数字列不能改善 CUI 连接，这只说明那一列没有更好，不能单独证明第 1 列全库正确。",
        "",
        "## 2. 旧规则漏掉的记录",
        "",
        "旧规则要求基因侧参数整个字符串等于 ENTITY.CUI。纯数字和竖线复合值因此对不上。v2 保留参数原文。单个数字在路径 A 启用时直接作为 Entrez ID，不再先要求等于 CUI。一个 CUI 加一个数字时，数字是候选；实体基因字段一致则两条路径合并为同一 pair，不一致则记为 conflict，不取值。多个不同数字保持候选集合，状态为 ambiguous，不展开，也不取第一个。单个 CUI 仍只通过实体上唯一的数字基因字段映射；没有实体、基因字段为空、泛化概念和多个基因 ID 分开记录。",
        "",
        "映射状态：",
        "",
        _md(comp["mapping_status"]),
        "",
        "范围 E：",
        "",
        _md(comp["scope_e_status"]),
        "",
        "## 3. v2 覆盖以及和 v1 的差异",
        "",
        "关系行（predication、gene、disease）：",
        "",
        _md([comp["predication_links"]]),
        "",
        "pair：",
        "",
        _md([comp["pairs"]]),
        "",
        "pair–PMID：",
        "",
        _md([comp["pair_pmid"]]),
        "",
        "新增关系来自哪条路径：",
        "",
        _md(comp["added_by_route"]),
        "",
        "v1 有而 v2 没有的关系，在 v2 输入里的状态：",
        "",
        _md(comp["removed_by_status"]),
        "",
        "同一 pair 上的路径组合。一条关系里的 B 和 C 已经合成一个 route；下表是不同关系汇总到 pair 之后的组合：",
        "",
        _md(comp["pair_route_combinations"]),
        "",
        "五类 PMID 集合、年份窗口和共同文章来自与 J04 相同的统计，输入换成 v2 的 pair–PMID。单库计数没有重算。PMID 通路表直接引用 J03。",
        "",
        "```json",
        json.dumps(
            {
                "populations": joint.get("populations"),
                "year_window_removed": joint.get("year_window_removed"),
                "common_article_removed": joint.get("common_article_removed"),
                "observable": joint.get("observable"),
                "closure": joint.get("closure"),
                "scope_e_only_pairs": joint.get("scope_e_only_pairs"),
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        "```",
        "",
        joint.get("entity_sentence_limit", ""),
        "",
        "## 4. 仍未解决",
        "",
        "多个不同数字的复合参数仍是 ambiguous。实体基因字段与直接数字冲突的参数没有被接受。泛化概念没有被收成一个基因。other_format 和未启用的路径没有进入主结果。坐标规则如果没有达到预设阈值，就不用于改写任何 ID。实体提及原文和 GENE_NAME 原文不在 parquet 里；本轮用已保存的完整提及预览和 AUX 文本做坐标核查，没有重读六张原始 CSV。",
        "",
        "v2 主结果只包含 mapping_status = accepted 且与 CORE 疾病 CUI、CORE 基因 ID 精确相等的记录。其余状态留在 `tables/mapped.parquet`，每行有 predication_id、原始参数、路径和 supporting_record_ids。",
        "",
    ]
    return "\n".join(lines) + "\n"


def _pick(row: dict) -> dict:
    keys = ("n_single_id_rows", "presence_rate", "name_agreement_rate", "n_current", "n_history_only")
    return {key: row.get(key) for key in keys}


def _md(records: list[dict]) -> str:
    if not records:
        return "（无行）"
    keys = list(records[0].keys())
    head = "| " + " | ".join(keys) + " |"
    sep = "| " + " | ".join("---" for _ in keys) + " |"
    body = [
        "| " + " | ".join("" if row.get(key) is None else str(row.get(key)).replace("|", "\\|") for key in keys) + " |"
        for row in records
    ]
    return "\n".join([head, sep, *body])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        raise SystemExit(1)

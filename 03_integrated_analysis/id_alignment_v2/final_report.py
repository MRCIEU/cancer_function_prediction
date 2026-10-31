#!/usr/bin/env python3
"""Assemble the current CORE–SemMedDB report from checked v2 outputs."""

from __future__ import annotations

import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE_ROOT = Path("/home/b5ac/freddieyu.b5ac/project_2")
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CODE_ROOT))

from lib.duckutil import connect  # noqa: E402
from lib.runtime import assert_compute_node, code_fingerprint, read_json, setup_logging, write_json  # noqa: E402

RESULTS = Path("/projects/b5ac/project_2/results")
V2 = RESULTS / "03_integrated_analysis" / "id_alignment_v2_round1"
REV = V2 / "revision"
REPORT = V2 / "report.md"
BACKUP = REV / "report_v2_before_final.md"
SOURCES = REV / "derived" / "final_report_sources.json"
NUMBERS = REV / "derived" / "revision_numbers.json"
PROVENANCE = REV / "derived" / "provenance.json"
YEAR = REV / "year_coverage" / "derived" / "year_coverage.json"
YEAR_TABLE = REV / "year_coverage" / "tables" / "pmid_year_extended.parquet"
V21 = V2 / "derived" / "v21_summary.json"
JOINT = V2 / "derived" / "joint_v2.json"
RULES = V2 / "derived" / "rules_v2.json"
CORE_AUDIT = RESULTS / "01_core_audit" / "report" / "core_audit.md"
SEMMED_AUDIT = RESULTS / "02_semmeddb_validation" / "report" / "semmed_audit.md"
SEMMED_SUMMARY = RESULTS / "02_semmeddb_validation" / "tables" / "summary.json"
CORE_PP = RESULTS / "01_core_audit" / "tables" / "pair_pmid.parquet"
SEMMED_PP = V2 / "tables" / "semmed_pair_pmid.parquet"
PAIR = RESULTS / "01_core_audit" / "tables" / "pair.parquet"


def main() -> int:
    assert_compute_node()
    logger = setup_logging(REV / "logs" / f"v26_{os.environ.get('SLURM_JOB_ID', 'local')}.log")
    logger.info("phase=start job=%s", os.environ.get("SLURM_JOB_ID", "local"))
    for path in (NUMBERS, PROVENANCE, YEAR, YEAR_TABLE, V21, JOINT, RULES, CORE_AUDIT, SEMMED_AUDIT, SEMMED_SUMMARY, REPORT):
        if not path.is_file():
            raise SystemExit(f"missing source: {path}")
    numbers = read_json(NUMBERS)
    provenance = read_json(PROVENANCE)
    year = read_json(YEAR)
    v21 = read_json(V21)
    joint = read_json(JOINT)
    rules = read_json(RULES)
    semmed = read_json(SEMMED_SUMMARY)
    mean = year_mean_jaccard(logger)
    facts = assemble(numbers, provenance, year, v21, joint, rules, semmed, mean)
    text = render(facts)
    checks = validate(facts) + text_checks(text)
    failed = [item for item in checks if not item["ok"]]
    write_json(REV / "derived" / "final_report_checks.json", {"checks": checks, "n_failed": len(failed)})
    if failed:
        logger.info("phase=checks_failed n=%s", len(failed))
        write_json(REV / "derived" / "v26_DONE.json", {"status": "failed", "job": "V26", "n_failed": len(failed)})
        return 1
    if not BACKUP.exists():
        shutil.copyfile(REPORT, BACKUP)
        logger.info("phase=backup %s", BACKUP)
    else:
        logger.info("phase=backup_kept %s", BACKUP)
    REPORT.write_text(text, encoding="utf-8")
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    fingerprint = code_fingerprint(CODE_ROOT)
    write_json(
        SOURCES,
        {
            "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not_in_slurm"),
            "code_fingerprint": fingerprint,
            "rule": "id_alignment_v2",
            "main_report": str(REPORT),
            "backup_of_previous_report": str(BACKUP),
            "cjk_characters": cjk,
            "sections": {
                "unfiltered_match": [str(V2 / "tables" / "accepted_match.parquet"), str(SEMMED_PP), str(NUMBERS)],
                "rules_funnel_scope_observable_common_article": [str(NUMBERS), str(PROVENANCE), str(RULES)],
                "year": [str(YEAR), str(YEAR_TABLE)],
                "single_database": [str(CORE_AUDIT), str(SEMMED_AUDIT), str(SEMMED_SUMMARY)],
            },
            "year_result": "V25 replaced the year filter. Current year counts come only from revision/year_coverage.",
            "label_correction": "joint_v2.json still contains the strings id_rule_v1 and E_gngm_neop_with_same_sentence_numeric_gene_id. Those were write-time labels. The v2 pair–PMID and scope E counts match the v2 parquet. The historical JSON was not rewritten.",
            "historical_baseline": "id_rule_v1 276 pairs remain in the original J03 outputs and are not the current main result.",
        },
    )
    write_json(REV / "derived" / "v26_DONE.json", {"status": "success", "job": "V26", "report": str(REPORT), "cjk_characters": cjk})
    logger.info("phase=v26_success cjk=%s", cjk)
    return 0


def year_mean_jaccard(logger) -> dict:
    con = connect(REV / "derived" / "v26.duckdb", "32GB", REV / "tmp")
    try:
        return _year_mean_jaccard(con, logger)
    finally:
        con.close()
        duck = REV / "derived" / "v26.duckdb"
        if duck.exists():
            duck.unlink()


def _year_mean_jaccard(con, logger) -> dict:
    con.execute(f"CREATE OR REPLACE TABLE ext AS SELECT * FROM read_parquet('{YEAR_TABLE.as_posix()}')")
    con.execute(f"CREATE OR REPLACE TABLE core_pp AS SELECT gene_id, disease_cui, pmid FROM read_parquet('{CORE_PP.as_posix()}')")
    con.execute(f"CREATE OR REPLACE TABLE sem_pp AS SELECT gene_id, disease_cui, pmid FROM read_parquet('{SEMMED_PP.as_posix()}')")
    con.execute(
        """
        CREATE OR REPLACE TABLE year_window AS
        SELECT core_year AS year FROM ext
        WHERE core_year_status = 'parsed_single' AND core_year IS NOT NULL
        INTERSECT
        SELECT semmed_dp_year FROM ext
        WHERE semmed_dp_status = 'parsed_single' AND semmed_dp_year IS NOT NULL
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE kept_core AS
        SELECT c.gene_id, c.disease_cui, c.pmid
        FROM core_pp c
        JOIN ext y USING (pmid)
        WHERE y.core_year_status = 'parsed_single'
          AND y.core_year IN (SELECT year FROM year_window)
          AND (y.semmed_dp_year IS NULL OR y.semmed_dp_year = y.core_year OR y.semmed_dp_status <> 'parsed_single')
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE kept_sem AS
        SELECT s.gene_id, s.disease_cui, s.pmid
        FROM sem_pp s
        JOIN ext y USING (pmid)
        WHERE y.semmed_dp_status = 'parsed_single'
          AND y.semmed_dp_year IN (SELECT year FROM year_window)
          AND (y.core_year IS NULL OR y.core_year = y.semmed_dp_year OR y.core_year_status <> 'parsed_single')
        """
    )
    counted = con.execute(
        f"""
        SELECT
          (SELECT count(*) FROM (SELECT DISTINCT gene_id, disease_cui FROM read_parquet('{PAIR.as_posix()}'))),
          (SELECT count(*) FROM core_pp),
          (SELECT count(*) FROM (SELECT DISTINCT pmid FROM core_pp))
        """
    ).fetchone()
    row = con.execute(
        """
        WITH shared AS (
          SELECT gene_id, disease_cui FROM (
            SELECT DISTINCT gene_id, disease_cui FROM core_pp
            INTERSECT
            SELECT DISTINCT gene_id, disease_cui FROM sem_pp
          )
        ),
        cb AS (SELECT gene_id, disease_cui, count(DISTINCT pmid) AS n FROM core_pp GROUP BY 1, 2),
        sb AS (SELECT gene_id, disease_cui, count(DISTINCT pmid) AS n FROM sem_pp GROUP BY 1, 2),
        ib AS (
          SELECT c.gene_id, c.disease_cui, count(DISTINCT c.pmid) AS n
          FROM core_pp c JOIN sem_pp s USING (gene_id, disease_cui, pmid) GROUP BY 1, 2
        ),
        ca AS (SELECT gene_id, disease_cui, count(DISTINCT pmid) AS n FROM kept_core GROUP BY 1, 2),
        sa AS (SELECT gene_id, disease_cui, count(DISTINCT pmid) AS n FROM kept_sem GROUP BY 1, 2),
        ia AS (
          SELECT c.gene_id, c.disease_cui, count(DISTINCT c.pmid) AS n
          FROM kept_core c JOIN kept_sem s USING (gene_id, disease_cui, pmid) GROUP BY 1, 2
        ),
        trans AS (
          SELECT coalesce(cb.n, 0) AS n_core_before, coalesce(sb.n, 0) AS n_sem_before, coalesce(ib.n, 0) AS n_inter_before,
                 coalesce(ca.n, 0) AS n_core_after, coalesce(sa.n, 0) AS n_sem_after, coalesce(ia.n, 0) AS n_inter_after
          FROM shared p
          LEFT JOIN cb USING (gene_id, disease_cui)
          LEFT JOIN sb USING (gene_id, disease_cui)
          LEFT JOIN ib USING (gene_id, disease_cui)
          LEFT JOIN ca USING (gene_id, disease_cui)
          LEFT JOIN sa USING (gene_id, disease_cui)
          LEFT JOIN ia USING (gene_id, disease_cui)
        )
        SELECT count(*) AS n_pair,
               sum(n_inter_before) AS inter_before,
               sum(n_core_before + n_sem_before - n_inter_before) AS union_before,
               sum(n_inter_after) AS inter_after,
               sum(n_core_after + n_sem_after - n_inter_after) AS union_after,
               avg(n_inter_before * 1.0 / nullif(n_core_before + n_sem_before - n_inter_before, 0)) AS mean_before,
               avg(n_inter_after * 1.0 / nullif(n_core_after + n_sem_after - n_inter_after, 0)) AS mean_after
        FROM trans
        WHERE n_core_after > 0 AND n_sem_after > 0
        """
    ).fetchone()
    logger.info("phase=year_mean %s", row)
    return {
        "core_pairs": int(counted[0]),
        "core_pair_pmid": int(counted[1]),
        "core_pmid": int(counted[2]),
        "n_pair": int(row[0]),
        "inter_before": int(row[1]),
        "union_before": int(row[2]),
        "inter_after": int(row[3]),
        "union_after": int(row[4]),
        "mean_before": float(row[5]),
        "mean_after": float(row[6]),
    }


def assemble(numbers, provenance, year, v21, joint, rules, semmed, mean) -> dict:
    funnel = numbers["funnel"]
    scope = numbers["scope"]
    sens = numbers["sensitivity"]
    obs = numbers["observable"]
    classes = {row["relation_class"]: row["n_pair"] for row in sens["class_counts"]}
    five = ("identical", "core_subset", "semmed_subset", "partial_overlap", "disjoint")
    art = sens["transitions"]["common_article"]
    art_states = {row["state_after"]: row["n_pair"] for row in art["states"]}
    yfix = year["corrected"]
    ystates = {row["state_after"]: row["n_pair"] for row in yfix["transition"]["states"]}
    yclasses = {row["relation_class"]: row["n_pair"] for row in yfix["year_filtered_classes"]}
    cross = obs["entity_by_core_anchored_argument"]

    def add_cross(gene_value: str, arguments: set[str] | None = None) -> int:
        total = 0
        for row in cross:
            if row["semmed_gene"] != gene_value:
                continue
            if arguments is None or row["core_anchored_argument"] in arguments:
                total += row["n"]
        return total

    return {
        "rules_version": rules.get("rule_version"),
        "label_verdict": provenance["label_audit"]["verdict"],
        "joint_rule_field": joint["populations"]["all_available"]["pair_pmid_micro_all_explicit_pairs"]["jaccard"]["rule"],
        "joint_scope_label": joint["scope_e_only_pairs"]["scope"],
        "core_pairs": mean["core_pairs"],
        "core_pair_pmid": mean["core_pair_pmid"],
        "core_pmid": mean["core_pmid"],
        "citations": semmed["tables"]["citations"]["rows"],
        "sentences": semmed["tables"]["sentence"]["rows"],
        "predications": semmed["tables"]["predication"]["rows"],
        "entities": semmed["tables"]["entity"]["rows"],
        "scope_e_predications": funnel["scope_e_rows"],
        "core_anchored": funnel["core_anchored_rows"],
        "accepted": funnel["accepted_arguments"],
        "on_core_pair": funnel["accepted_on_core_pair"],
        "not_core_pair": funnel["accepted_gene_not_core_pair"],
        "pairs": funnel["pairs"],
        "pair_pmid": funnel["pair_pmid"],
        "match_pmid": funnel["pmids"],
        "v21_pairs": v21["counts"]["accepted_core_pairs"],
        "v21_pair_pmid": v21["counts"]["accepted_core_pair_pmid"],
        "classes": classes,
        "five_sum": sum(classes[name] for name in five),
        "semmed_empty": classes.get("semmed_pmids_empty", 0),
        "both_empty": classes.get("both_empty", 0),
        "core_empty": classes.get("core_pmids_empty", 0),
        "pp_inter": sens["pair_pmid_intersection"],
        "pmid_core": sens["deduplicated_pmid"]["n_core"],
        "pmid_sem": sens["deduplicated_pmid"]["n_semmed"],
        "pmid_inter": sens["deduplicated_pmid"]["n_intersection"],
        "scope_pairs": scope["n_scope_e_pairs"],
        "scope_shared": scope["n_scope_e_in_core_pairs"],
        "scope_only": scope["n_scope_e_not_in_core_pairs"],
        "main_only": scope["n_main_not_in_scope_e_shared"],
        "main_only_gngm": scope["main_only_with_gngm_neop_row"],
        "scope_not_main": scope["n_scope_e_shared_not_in_main"],
        "article_pmid": sens["article"]["n_common_article_pmid"],
        "article_states": art_states,
        "article_fixed": art["fixed_both_nonempty"],
        "year_pmid": yfix["extended_pmids"],
        "year_added": yfix["extended_not_in_original"],
        "year_n": yfix["window"]["n"],
        "year_min": yfix["window"]["min"],
        "year_max": yfix["window"]["max"],
        "year_reasons": {row["exclusive_reason"]: row["n_pair_pmid"] for row in yfix["reasons"]["sem"]["exclusive"]},
        "core_removed": yfix["new_core_removed_pair_pmid"],
        "year_states": ystates,
        "year_fixed": yfix["transition"]["fixed_both_nonempty"],
        "year_mean": mean,
        "year_classes": yclasses,
        "excluded": numbers["sensitivity"]["cui"]["excluded_labeled_both_empty"],
        "entity_present": add_cross("present"),
        "entity_absent": add_cross("absent_in_available_records"),
        "entity_unknown": add_cross("not_assessable"),
        "absent_but_accepted": add_cross("absent_in_available_records", {"accepted_gene_id"}),
        "present_not_candidate": add_cross("present", {"pmid_in_core_anchored_input_gene_not_in_argument"}),
        "core_only_pp": obs["n_core_only_pair_pmid"],
        "status_rows": numbers["funnel"]["status"],
    }


def validate(facts: dict) -> list[dict]:
    checks = []

    def add(name: str, ok: bool, detail) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    add("stale_joint_labels", facts["joint_rule_field"] == "id_rule_v1" and facts["joint_scope_label"] == "E_gngm_neop_with_same_sentence_numeric_gene_id", facts["joint_scope_label"])
    add("rule", facts["rules_version"] == "id_alignment_v2", facts["rules_version"])
    add("scope_e_rows", facts["scope_e_predications"] == 619022, facts["scope_e_predications"])
    add("label_is_stale", facts["label_verdict"] == "label_stale_computation_is_v2", facts["label_verdict"])
    add("funnel_conservation", facts["accepted"] == facts["on_core_pair"] + facts["not_core_pair"], facts["accepted"])
    add("five_sum", facts["five_sum"] == facts["pairs"] == 30910, facts["five_sum"])
    add("all_pairs", facts["five_sum"] + facts["semmed_empty"] + facts["both_empty"] + facts["core_empty"] == 153016, facts["semmed_empty"])
    add("pp_split", facts["pp_inter"] + (facts["core_pair_pmid"] - facts["pp_inter"]) + (facts["pair_pmid"] - facts["pp_inter"]) == facts["core_pair_pmid"] + facts["pair_pmid"] - facts["pp_inter"], facts["pp_inter"])
    add("pmid_split", facts["pmid_inter"] <= facts["pmid_core"] and facts["pmid_inter"] <= facts["pmid_sem"], facts["pmid_inter"])
    add("scope_split", facts["scope_shared"] + facts["scope_only"] == facts["scope_pairs"], facts["scope_pairs"])
    add("scope_inside_main", facts["scope_not_main"] == 0 and facts["scope_shared"] + facts["main_only"] == facts["pairs"], facts["main_only"])
    add("article_states", sum(facts["article_states"].values()) == 30910, facts["article_states"])
    add("year_states", sum(facts["year_states"].values()) == 30910, facts["year_states"])
    add("year_classes", sum(facts["year_classes"].values()) == 153016, facts["year_classes"])
    add("year_window", (facts["year_min"], facts["year_max"], facts["year_n"]) == (1975, 2022, 48), facts["year_n"])
    fixed = facts["year_fixed"]
    mean = facts["year_mean"]
    add(
        "year_mean_same_set",
        (mean["n_pair"], mean["inter_before"], mean["union_before"], mean["inter_after"], mean["union_after"])
        == (fixed["n_pair"], fixed["inter_before"], fixed["union_before"], fixed["inter_after"], fixed["union_after"]),
        mean,
    )
    add("v21_match", facts["v21_pairs"] == facts["pairs"] and facts["v21_pair_pmid"] == facts["pair_pmid"], facts["v21_pairs"])
    add("entity_sum", facts["entity_present"] + facts["entity_absent"] + facts["entity_unknown"] == facts["core_only_pp"], facts["core_only_pp"])
    expected = {
        "core_pairs": 153016,
        "core_pair_pmid": 321201,
        "core_pmid": 161449,
        "core_anchored": 8741634,
        "accepted": 513477,
        "on_core_pair": 173048,
        "pairs": 30910,
        "pair_pmid": 137221,
        "match_pmid": 113380,
        "pp_inter": 17260,
        "pmid_inter": 31412,
        "scope_pairs": 161490,
        "scope_shared": 27028,
        "scope_only": 134462,
        "main_only": 3882,
        "year_pmid": 243417,
        "core_removed": 29401,
        "article_pmid": 161405,
    }
    for key, value in expected.items():
        add(f"expected_{key}", facts[key] == value, {"got": facts[key], "expected": value})
    add("expected_identical", facts["classes"]["identical"] == 5303, facts["classes"].get("identical"))
    add("expected_core_subset", facts["classes"]["core_subset"] == 2626, facts["classes"].get("core_subset"))
    add("expected_semmed_subset", facts["classes"]["semmed_subset"] == 1590, facts["classes"].get("semmed_subset"))
    add("expected_partial", facts["classes"]["partial_overlap"] == 3883, facts["classes"].get("partial_overlap"))
    add("expected_disjoint", facts["classes"]["disjoint"] == 17508, facts["classes"].get("disjoint"))
    add("expected_year_both", facts["year_states"]["both_nonempty"] == 27193, facts["year_states"])
    add("expected_year_outside", facts["year_reasons"].get("parsed_year_outside_shared_set") == 8367, facts["year_reasons"])
    add("expected_year_disagree", facts["year_reasons"].get("both_parsed_and_disagree") == 2725, facts["year_reasons"])
    add("expected_year_unparsed", facts["year_reasons"].get("own_year_not_parsed_single") == 15, facts["year_reasons"])
    add("expected_article_both", facts["article_states"].get("both_nonempty") == 20325, facts["article_states"])
    af = facts["article_fixed"]
    add("expected_article_jaccard", (af["inter_before"], af["union_before"], af["inter_after"], af["union_after"]) == (17260, 177285, 17260, 100393), af)
    add("expected_year_jaccard", (fixed["inter_before"], fixed["union_before"], fixed["inter_after"], fixed["union_after"]) == (16227, 217346, 15250, 201305), fixed)
    return checks


def text_checks(text: str) -> list[dict]:
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return [
        {"name": "no_new_window_wording", "ok": "新窗口" not in text, "detail": "新窗口" in text},
        {"name": "stale_rule_named_as_history", "ok": text.count("id_rule_v1") == 2, "detail": text.count("id_rule_v1")},
        {"name": "no_large_json", "ok": text.count("{") < 5 and len(text) < 20000, "detail": len(text)},
        {"name": "cjk_length", "ok": 2000 <= cjk <= 3200, "detail": cjk},
    ]


def render(facts: dict) -> str:
    af = facts["article_fixed"]
    yf = facts["year_fixed"]
    ym = facts["year_mean"]
    art = facts["article_states"]
    ys = facts["year_states"]
    cl = facts["classes"]
    lines = [
        "# CORE 与 SemMedDB 的记录覆盖",
        "",
        "本文给出本地 CORE 与 SemMedDB 在基因–疾病标识上的对齐结果。当前规则是 id_alignment_v2。早先规则 id_rule_v1 得到的 276 个 pair 只作为历史基线，并且已经全部包含在当前共享 pair 中，不再单独作为主结果。本文只描述记录是否出现，不判断文献质量，也不把覆盖解释成科学上的支持或反驳。",
        "",
        "已保存的 joint_v2.json 仍写着规则名 id_rule_v1 和范围名 E_gngm_neop_with_same_sentence_numeric_gene_id。这两个字符串是写出时的过期标签，当时的计数来自 v2 的匹配表。该文件保持原样，当前结论不使用这两个名字。",
        "",
        "## 数据范围与基本单位",
        "",
        f"CORE 本地图有 {fmt(facts['core_pairs'])} 个显式 gene–disease pair、{fmt(facts['core_pair_pmid'])} 条 pair–PMID、{fmt(facts['core_pmid'])} 个 PMID。基因标识来自 NCBI Gene，疾病标识来自 UMLS CUI。每个显式 pair 至少有一条 PMID。",
        "",
        f"SemMedDB 使用本地 VER43 表：引文 {fmt(facts['citations'])} 行，句子 {fmt(facts['sentences'])} 行，谓词 {fmt(facts['predications'])} 行，实体 {fmt(facts['entities'])} 行。发表年只从引文字段 DP 解析。PYEAR 是完成日期，不作为发表年。",
        "",
        f"主分析从 CORE 显式 pair 出发，在 SemMedDB 谓词中寻找同一基因 ID 和同一疾病 CUI。范围 E 另行定义：语义类型恰好为一侧 gngm（基因或基因组）、一侧 neop（肿瘤）的谓词，共 {fmt(facts['scope_e_predications'])} 条。它不是 SemMedDB 的完整癌症集合，也不由 CORE pair 反推。",
        "",
        "计数单位不能互换。参数行是一个谓词的一侧参数。关系记录是该参数已被规则接受，并且基因 ID 与疾病 CUI 等于某个 CORE pair。pair 只含基因和疾病。pair–PMID 在 pair 上再加一篇文章。唯一 PMID 把同一篇文章只计一次。一篇 PMID 出现在多个 pair 中，仍然是一篇文章，不是多项独立研究。",
        "",
        "## 标识对齐与漏斗",
        "",
        "路径 A 接受一个纯数字，且语义类型含 gngm 或 aapp，并把它当作 Entrez Gene ID。没有实体行，或实体上的基因对不上，都不取消这条接受。路径 B 只处理单个 UMLS CUI：该 CUI 在候选句子上恰好有一个数字基因字段时才接受。泛化概念、没有实体行、基因字段为空或多个基因 ID 都不取值。路径 C 接受“一个 CUI 加一个数字”，或重复出现的同一个数字。多个不同数字保持歧义，不取第一个。若该 CUI 连到的实体基因与这个数字不同，记为冲突并且不接受；没有实体基因时，路径 C 仍接受该数字。",
        "",
        "NCBI 当前库或停用记录中的存在性，以及名称的精确词一致，只在分层样本上检查，用来决定这三条路径能否启用。accepted 表示该行满足规则，不是逐条人工核对，也不是全库准确率。只出现在停用记录中的 ID 保留原号，不自动换成替代号。样本中基因 339803 的参数名与当前符号 C2orf74-DT 不一致，没有因此关闭路径，也不把这一条当成已经证伪的映射。",
        "",
        "候选参数的进入条件是：谓词一侧的 CUI 等于某个 CORE 疾病，另一侧作为基因参数，PMID 非空，并且两侧 CUI 不同。",
        "",
        "| 步骤 | 数量 | 单位 | 这一步在做什么 |",
        "| --- | --- | --- | --- |",
        f"| 候选参数 | {fmt(facts['core_anchored'])} | 参数行 | 进入候选参数集 |",
        f"| 规则接受 | {fmt(facts['accepted'])} | 参数行 | 路径 A、B 或 C 接受 |",
        f"| 不是 CORE pair | {fmt(facts['not_core_pair'])} | 参数行 | 筛选排除 |",
        f"| 精确匹配 | {fmt(facts['on_core_pair'])} | 关系记录 | 筛选保留 |",
        f"| 共享 pair | {fmt(facts['pairs'])} | pair | 按基因和疾病聚合 |",
        f"| 匹配 pair–PMID | {fmt(facts['pair_pmid'])} | pair–PMID | 再按 PMID 聚合 |",
        f"| 匹配 PMID | {fmt(facts['match_pmid'])} | 唯一 PMID | 文章只计一次 |",
        "",
        f"从 {fmt(facts['accepted'])} 到 {fmt(facts['on_core_pair'])} 仍是参数行：{fmt(facts['not_core_pair'])} 行得到了基因 ID，但该基因与该疾病不是 CORE 显式 pair。其后不再删除记录，只改变聚合键。{fmt(facts['on_core_pair'])} 条关系记录按基因、疾病和 PMID 去重后是 {fmt(facts['pair_pmid'])} 条 pair–PMID；PMID 再合并后是 {fmt(facts['match_pmid'])} 个。",
        "",
        "没有进入主结果的主要参数行还有：基因字段为空 "
        + fmt(status_n(facts, "core_anchored", "gene_field_empty"))
        + "，泛化概念 "
        + fmt(status_n(facts, "core_anchored", "generic_concept_not_forced"))
        + "，候选句子上没有该 CUI 的实体行 "
        + fmt(status_n(facts, "core_anchored", "no_entity_row"))
        + "，多个不同数字 "
        + fmt(status_n(facts, "core_anchored", "ambiguous_multiple_gene_ids"))
        + "，冲突 "
        + fmt(status_n(facts, "core_anchored", "conflict"))
        + "。这些行保留在映射表中。",
        "",
        f"范围 E 用同一套规则得到 {fmt(facts['scope_pairs'])} 个已接受 pair。其中 {fmt(facts['scope_shared'])} 个也是 CORE 显式 pair，并且全部属于上面的 {fmt(facts['pairs'])}。主结果另外 {fmt(facts['main_only'])} 个 pair 没有精确的 gngm–neop 谓词。{fmt(facts['scope_only'])} 个范围 E pair 不在 CORE 显式 pair 中，只描述这个操作性子集。",
        "",
        "## 未筛选的主结果",
        "",
        f"分母是全部 {fmt(facts['core_pairs'])} 个 CORE pair。两侧 PMID 集合都非空时才进入下表，共 {fmt(facts['pairs'])} 个共享 pair。尚未在 SemMedDB 匹配中出现 PMID 的 pair 有 {fmt(facts['semmed_empty'])} 个。未筛选时没有两侧皆空的 pair。",
        "",
        "| 类别 | pair 数 | 含义 |",
        "| --- | --- | --- |",
        f"| 完全相同 | {fmt(cl['identical'])} | 两侧 PMID 集合相等 |",
        f"| CORE 严格子集 | {fmt(cl['core_subset'])} | CORE 的 PMID 全部出现在 SemMedDB 匹配中，且 SemMedDB 还有更多 |",
        f"| SemMedDB 严格子集 | {fmt(cl['semmed_subset'])} | SemMedDB 匹配的 PMID 全部出现在 CORE 中，且 CORE 还有更多 |",
        f"| 部分重合 | {fmt(cl['partial_overlap'])} | 两侧都有对方没有的 PMID，并且交集非空 |",
        f"| 完全不重合 | {fmt(cl['disjoint'])} | 两侧都有 PMID，但没有共同 PMID |",
        "",
        f"pair–PMID 交集是 {fmt(facts['pp_inter'])}。按这个单位，CORE 独有 {fmt(facts['core_pair_pmid'] - facts['pp_inter'])}，SemMedDB 匹配独有 {fmt(facts['pair_pmid'] - facts['pp_inter'])}。去重 PMID 交集是 {fmt(facts['pmid_inter'])}。按这个单位，CORE 独有 {fmt(facts['pmid_core'] - facts['pmid_inter'])}，SemMedDB 匹配独有 {fmt(facts['pmid_sem'] - facts['pmid_inter'])}。两个交集不同，是因为有的 PMID 在两侧都出现，但连接的 pair 不同。独有记录只表示另一侧没有同一键，不能称为新增有效证据。",
        "",
        "## 两种敏感性分析",
        "",
        f"共同文章指 PMID 同时出现在 CORE 句子和 SemMedDB 引文中，共 {fmt(facts['article_pmid'])} 个。这个集合不要求目标 pair 已经匹配。在原来的 {fmt(facts['pairs'])} 个共享 pair 上，筛选后仍两侧都有 PMID 的有 {fmt(art.get('both_nonempty', 0))} 个，只剩 CORE 侧的有 {fmt(art.get('semmed_side_empty', 0))} 个，只剩 SemMedDB 侧的有 {fmt(art.get('core_side_empty', 0))} 个，两侧都空的有 {fmt(art.get('both_empty', 0))} 个。",
        "",
        f"这 {fmt(af['n_pair'])} 个 pair 的 pair–PMID 汇总 Jaccard，筛选前是 {ratio(af['inter_before'], af['union_before'])}，筛选后是 {ratio(af['inter_after'], af['union_after'])}。逐 pair 平均 Jaccard 筛选前是 {af['mean_jaccard_before']:.3f}，筛选后是 {af['mean_jaccard_after']:.3f}。汇总 Jaccard 按 pair–PMID 加权；平均 Jaccard 对每个 pair 一视同仁。共同文章筛选只删掉没有共同文章记录的 PMID，已经两边都有的关联键仍在，所以交集分子不变。",
        "",
        f"共同年份集合仍是 {facts['year_min']}–{facts['year_max']}，共 {facts['year_n']} 年。一条 pair–PMID 要保留，本侧年份必须解析为单一年份并落在这个集合内。若两侧都已解析且年份不同，则两侧都删除。",
        "",
        f"旧年份表只有 CORE 的 {fmt(facts['core_pmid'])} 个 PMID，v2 匹配中另外 {fmt(facts['year_added'])} 个 PMID 因此被整批排除。这些 PMID 在引文表中都有记录，其中大多数 DP 已是单一年份。补齐后的年份表覆盖 CORE pair–PMID 与 v2 匹配 PMID 的并集，共 {fmt(facts['year_pmid'])} 个 PMID。年份起止没有改变。当前年份结论只用这份补齐结果。",
        "",
        f"补齐之后，CORE 侧删除 {fmt(facts['core_removed'])} 条 pair–PMID。SemMedDB 侧删除年份在共同集合之外的 {fmt(facts['year_reasons']['parsed_year_outside_shared_set'])} 条、两侧年份不一致的 {fmt(facts['year_reasons']['both_parsed_and_disagree'])} 条、未能解析为单一年份的 {fmt(facts['year_reasons']['own_year_not_parsed_single'])} 条。原来的 {fmt(facts['pairs'])} 个共享 pair 中，筛选后仍两侧都有 PMID 的有 {fmt(ys['both_nonempty'])} 个，只剩 CORE 侧的有 {fmt(ys['semmed_side_empty'])} 个，只剩 SemMedDB 侧的有 {fmt(ys['core_side_empty'])} 个，两侧都空的有 {fmt(ys['both_empty'])} 个。这是同一批共享 pair 的前后状态，分母仍是 {fmt(facts['pairs'])}，不是筛选之后重新划分全部 CORE pair 的类别表。",
        "",
        f"这 {fmt(yf['n_pair'])} 个 pair 的 pair–PMID 汇总 Jaccard，筛选前是 {ratio(yf['inter_before'], yf['union_before'])}，筛选后是 {ratio(yf['inter_after'], yf['union_after'])}。逐 pair 平均 Jaccard 筛选前是 {ym['mean_before']:.3f}，筛选后是 {ym['mean_after']:.3f}。",
        "",
        f"另有一项已有的敏感性设定，排除四个示例疾病 CUI，涉及 {fmt(facts['excluded'])} 个 pair。它们在该设定中标为 excluded，因为过滤后两侧计数变为零，不是原本没有 PMID。未筛选时两侧皆空的 pair 为 0。未筛选主结果不使用这个排除。",
        "",
        "## 能分开的现象与限制",
        "",
        f"CORE 独有的 pair–PMID 有 {fmt(facts['core_only_pp'])} 条。其中 {fmt(facts['entity_unknown'])} 条没有 SemMedDB 引文，实体是否出现无法判断。其余记录中，按候选句子编号在 ENTITY 里找到目标基因的有 {fmt(facts['entity_present'])} 条，没有找到的有 {fmt(facts['entity_absent'])} 条。没有找到只说明这次连接未命中，不说明全库没有该基因。",
        "",
        f"同一批记录上，指定范围内的谓词参数是否含有该基因是另一个问题。ENTITY 未命中时，仍有 {fmt(facts['absent_but_accepted'])} 条的参数被接受为该基因。ENTITY 命中时，也有 {fmt(facts['present_not_candidate'])} 条的参数没有把该基因收为候选。这个参数范围只包括一侧 CUI 等于某个 CORE 疾病的谓词。出现目标基因也不证明存在目标 gene–disease 关系。两侧 PMID 集合不同，只表示记录覆盖不同。",
        "",
        f"目前可以确定的是：在 id_alignment_v2 下，未筛选主结果为 {fmt(facts['pairs'])} 个共享 pair、{fmt(facts['pair_pmid'])} 条 SemMedDB pair–PMID 和 {fmt(facts['match_pmid'])} 个 PMID；五类、两种交集和两种敏感性都使用上面标明的单位。仍然适用的限制是：NCBI 与名称核对没有逐条用于全部 accepted 记录；路径 A 不把实体支持当作接受条件；范围 E 不是完整癌症集合；实体未命中不能外推为数据库缺失。",
        "",
    ]
    return "\n".join(lines)


def status_n(facts: dict, population: str, status: str) -> int:
    for row in facts["status_rows"]:
        if row["population"] == population and row["mapping_status"] == status:
            return row["n"]
    return 0


def fmt(value: int) -> str:
    return f"{value:,}"


def ratio(numer, denom) -> str:
    return f"{numer / denom:.3f}（{fmt(numer)}/{fmt(denom)}）"


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        raise SystemExit(1)

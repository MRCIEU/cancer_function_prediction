"""Chinese reports, figures, and exploratory cases from completed tables."""

from __future__ import annotations

import json
from pathlib import Path

from lib.cases import select_cases


def _load(path: Path) -> dict:
    if not path.is_file():
        return {"_missing": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def show(value) -> str:
    if value is None:
        return "未完成"
    if isinstance(value, dict):
        status = value.get("status")
        if status == "not_applicable":
            return "不适用"
        if status == "not_assessable":
            return "无法判断"
        if status == "not_computed":
            return "未完成"
        if "numerator" in value and "denominator" in value:
            shown = "不适用" if value.get("value") is None else f"{value['value']:.6g}"
            return f"{value['numerator']}/{value['denominator']} = {shown}（单位 {value.get('unit', '未标明')}，过滤 {value.get('filters', '未标明')}，规则 {value.get('rule', '未标明')}）"
        if "n" in value and "p50" in value:
            return (
                f"n={value.get('n')}，零值={value.get('n_zero')}，min={value.get('min')}，"
                f"P25={value.get('p25')}，P50={value.get('p50')}，P75={value.get('p75')}，"
                f"P90={value.get('p90')}，P95={value.get('p95')}，P99={value.get('p99')}，max={value.get('max')}"
            )
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)[:2000]
    return str(value)


def write_core_report(summary: dict, path: Path) -> None:
    lines = [
        "# CORE 单库审计",
        "",
        "范围 A：本地 `data.ttl` 与 `schema.owl`。中间抽取表不在本地下载中，因此不能把论文里的筛选阶段当成已经随文件提供。",
        "",
        f"- 三元组数：{show(summary.get('n_triples'))}",
        f"- 显式 pair：{show(summary.get('n_explicit_pairs'))}",
        f"- pair–PMID：{show(summary.get('n_pair_pmid'))}",
        f"- 可用 PMID：{show(summary.get('n_explicit_pmid'))}",
        f"- GCS 状态：{show(summary.get('gcs_pair_status'))}",
        "",
        "## 标识符",
        "",
        "基因命名空间来自构建代码 `https://www.ncbi.nlm.nih.gov/gene/{{Entrez}}`，疾病来自 `http://linkedlifedata.com/resource/umls/id/{{CUI}}`，文章来自 `https://pubmed.ncbi.nlm.nih.gov/{{PMID}}`。不匹配这些命名空间的 URI 不会被当成已验证 ID。",
        "",
        f"漏斗：{show(summary.get('funnel'))}",
        "",
        "## 标签",
        "",
        summary.get("labels", {}).get("definition", "未完成"),
        "",
        f"hasType 取值（含 schema 未列出的 UNRELIABLE，若数据中出现）：{show(summary.get('labels', {}).get('gcs_has_type'))}",
        "",
        "构建代码里的分数没有写入 RDF，因此不能从本地图重算 hasType。UNRELIABLE 只作为存储字段描述，不表示论文质量。",
        "",
        "## 分布与时间",
        "",
        f"每 pair 的 PMID 数：{show(summary.get('distributions', {}).get('pmid_per_explicit_pair'))}",
        f"单篇来源 pair：{show(summary.get('distributions', {}).get('n_pairs_single_pmid'))}；多篇：{show(summary.get('distributions', {}).get('n_pairs_multi_pmid'))}；无 PMID：{show(summary.get('distributions', {}).get('n_pairs_zero_pmid'))}",
        f"头部 1% pair 的 PMID 权重：{show(summary.get('distributions', {}).get('head_1pct_pairs_by_pmid_weight_pmid'))}",
        f"发表年字段 publicationYear：{show(summary.get('time', {}).get('field_meaning'))}",
        "",
        "## 完整性",
        "",
        show(summary.get("integrity")),
        "",
        "未测量的项目保留为未完成或不适用，不记为 0。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_semmed_report(summary: dict, path: Path) -> None:
    lines = [
        "# SemMedDB 单库审计",
        "",
        "范围 B：本地六张表的全库概况。文件名中的 semmedVER43_2024_R 与 NLM 下载页命名规则一致，但本地没有独立校验和，不能据此认定字节级等同于该发布。`.23327` 不解释为分片号。",
        "",
        "PYEAR 按 dbinfo 记为 completion date，不作为发表年。发表年只从 DP 解析。",
        "",
        "列位置以预检样本的取值模式为准。与 dbinfo 列序不一致的字段保留差异：未能由取值或后续 sentence join 确认的列保持 unconfirmed，不补造名称。",
        f"列映射：{show(summary.get('local_column_maps'))}",
        f"ENTITY sentence_id 解析：{show(summary.get('entity_sentence_id_resolution'))}",
        "",
        f"表行数：{show(summary.get('tables'))}",
        f"引用年份状态：{show(summary.get('citations'))}",
        f"谓词：{show(summary.get('predications', {}).get('predicate_counts'))}",
        f"完整性：{show(summary.get('integrity'))}",
        "",
        "范围 E 是 semantic type 恰好为 gngm 与 neop 的谓词记录，外加单独的 aapp×neop 子集。二者都不是完整癌症集合，也不由 CORE pair 反推。",
        "",
        show(summary.get("scope_e")),
        "",
        "PREDICATION 与 PREDICATION_AUX 行数不同本身不视为损坏。没有 predication 的 sentence 记为结构上允许的缺失。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_integrated_report(core: dict, semmed: dict, align: dict, joint: dict, path: Path) -> None:
    pop = joint.get("populations", {})
    lines = [
        "# CORE 与 SemMedDB 联合描述",
        "",
        "本报告只描述本地数据的内容、对齐和覆盖。它不评价论文质量、结论方向或抽取对错。",
        "",
        "## 1. 本地有什么",
        "",
        "CORE 本地文件是 `data.ttl` 与 `schema.owl`。SemMedDB 本地文件是六张无表头 CSV，文件名含 semmedVER43_2024_R，并带未解释的 `.23327` 后缀。版本在缺少校验和时记为 filename-indicated，而不是已核实发布。",
        "",
        f"CORE 三元组 {show(core.get('n_triples'))}；显式 pair {show(core.get('n_explicit_pairs'))}；显式 PMID {show(core.get('n_explicit_pmid'))}。",
        f"SemMedDB 表规模 {show(semmed.get('tables'))}。",
        "",
        "NLM 页面给出的行数只作为外部参考，不是本轮必须复现的目标。旧的 12958 等数字同样不是验收目标。",
        "",
        "## 2. 两库各自的规模和表示",
        "",
        f"CORE 每 pair 的 PMID 分布：{show(core.get('distributions', {}).get('pmid_per_explicit_pair'))}",
        f"CORE 单篇/多篇/无 PMID pair：{show(core.get('distributions', {}).get('n_pairs_single_pmid'))} / {show(core.get('distributions', {}).get('n_pairs_multi_pmid'))} / {show(core.get('distributions', {}).get('n_pairs_zero_pmid'))}",
        f"SemMedDB (subject CUI, object CUI) 的谓词记录分布：{show(semmed.get('predications', {}).get('cui_pair_predication_distribution'))}",
        "",
        "CORE 的四个方面保留原始标签。NOTINF 与字段缺失分开。SemMedDB 保留原始 predicate；NEG_ 只是字符串前缀。两边的标签不做准确率比较。",
        "",
        "## 3. 哪些对齐可靠",
        "",
        "可靠规则：CORE Entrez ID 对 ENTITY.GENE_ID 的单个数字，以及 CORE UMLS CUI 对谓词主客体 CUI。同一句子里该 CUI 对应多个 GENE_ID 时记为歧义，不拆成确定 pair。含 `|` 的基因字段不展开。",
        "",
        f"ENTITY 的 sentence_id 列由 J03 按 sentence_id_resolution_v2 重新判定：前 50000 行里，只有一列对 sentence.sentence_id 的命中率不低于 0.95 时才采用。J02 单库报告保留更早的对照阈值结果。{show(align.get('entity_sentence_id_resolution'))}",
        "",
        f"匹配漏斗：{show(align.get('funnel'))}",
        "",
        "## 4. pair 与 PMID 两个方向",
        "",
        "范围 C 是从 CORE 显式 pair 出发、并且两边都按上述规则取到 PMID 的集合，不是 SemMedDB 的全部癌症数据。范围 D 是 CORE 的全部可用 PMID，不先要求 pair 匹配。",
        "",
        f"全部可用记录上的分类计数：{show(pop.get('all_available', {}).get('class_counts'))}",
        f"pair 等权 Jaccard：{show(pop.get('all_available', {}).get('equal_weight_jaccard'))}",
        f"pair–PMID 微观覆盖（全部显式 pair，含一侧为空）：{show(pop.get('all_available', {}).get('pair_pmid_micro_all_explicit_pairs'))}",
        f"pair–PMID 微观覆盖（仅五类两边都非空的 pair）：{show(pop.get('all_available', {}).get('pair_pmid_micro_five_class_only'))}",
        f"去重 PMID 覆盖：{show(pop.get('all_available', {}).get('deduplicated_pmid'))}",
        f"PMID 路线：{show(joint.get('pmid_route'))}",
        "",
        "这三种覆盖分母不同，不能合成一个比例。",
        "",
        "## 5. 不重合里可以观察到什么",
        "",
        f"{show(joint.get('observable'))}",
        "",
        "同一 pair 的同一 PMID 若两边都有主分析记录，会进入交集。只是 predicate 或标签不同，不会把它算成文献不重合。",
        "",
        "## 6. 时间和规则改变后",
        "",
        f"共同可观察发表年：{show(joint.get('window_years'))}",
        f"年份限制：{show(pop.get('shared_observed_publication_years'))}",
        f"两边都有文章记录的 PMID 限制：{show(pop.get('pmid_with_article_record_in_both'))}",
        f"去掉 SPARQL 示例中的四个 CUI：{show(pop.get('excluding_sparql_example_cuis'))}",
        "",
        "共同年份窗口只限制可观察时间，不能证明两边处理的语料相同。",
        "",
        "## 7. 这些数据能支持什么",
        "",
        "可以支持：文件与字段的实测描述、命名空间匹配得上的 ID 对齐、pair 与 PMID 两个方向的覆盖计数、以及可观察的记录有无。",
        "",
        "不能支持：论文质量、支持或反驳、科学矛盾、独立研究数、NER 或关系抽取错误率、完整癌症集合，以及把覆盖增量说成新增有效证据。不同 PMID 只表示不同文章记录。",
        "",
        f"范围 E 中带唯一基因 ID、且不在 CORE 显式 pair 里的 pair 数：{show(joint.get('scope_e_only_pairs'))}",
        "",
        "若某项为未完成、不适用或无法判断，则该项没有被测到，不能写成 0 或通过。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_figures(core: dict, semmed: dict, align: dict, joint: dict, fig_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    written = []
    classes = joint.get("populations", {}).get("all_available", {}).get("class_counts", {})
    if classes:
        _bar(plt, fig_dir / "pair_relation_classes.png", list(classes.keys()), list(classes.values()), "CORE explicit pairs", "relation class", "pairs", "Scope C, both-empty and one-sided classes kept separate. Missing classes are absent, not zero-filled.")
        written.append("pair_relation_classes.png")
    years = core.get("time", {}).get("by_year") or []
    if years:
        _bar(plt, fig_dir / "core_year_pmid.png", [str(r.get("year")) for r in years], [r.get("n_pmid") or 0 for r in years], "CORE articles with one parsed publicationYear", "publicationYear", "distinct PMID", "Articles with missing, multiple, or non-positive years are excluded from this axis and counted in the year-status table.")
        written.append("core_year_pmid.png")
    funnel = align.get("funnel") or []
    if funnel:
        _bar(plt, fig_dir / "match_funnel.png", [r["stage"] for r in funnel], [r["n"] for r in funnel], "ID match funnel", "stage", "count", "Units differ by stage and are named in the stage. Rule id_rule_v1.")
        written.append("match_funnel.png")
    route = joint.get("pmid_route", {})
    labels, vals = [], []
    for key in ("in_semmed_citations", "with_semmed_sentence", "with_semmed_predication", "with_semmed_entity"):
        item = route.get(key)
        if isinstance(item, dict) and item.get("numerator") is not None:
            labels.append(key)
            vals.append(item["numerator"])
    if labels:
        denom = route.get("n_core_pmid")
        _bar(plt, fig_dir / "pmid_route.png", labels, vals, f"CORE PMIDs found in SemMedDB layers (denominator {denom})", "layer", "PMID", "Scope D. A PMID can be present at one layer and absent at another. Not restricted to shared pairs.")
        written.append("pmid_route.png")
    dist = core.get("distributions", {}).get("pmid_per_explicit_pair", {})
    if dist.get("status") == "computed":
        names = ["min", "p25", "p50", "p75", "p90", "p95", "p99", "max"]
        _bar(plt, fig_dir / "core_pmid_per_pair_tail.png", names, [dist.get(n) for n in names], f"PMID count per explicit CORE pair (n={dist.get('n')})", "nearest-rank percentile", "PMID per pair", "Population: explicit pairs, including zero-PMID pairs. Weight unit is PMID.")
        written.append("core_pmid_per_pair_tail.png")
    return written


def _bar(plt, path: Path, labels, values, title: str, xlabel: str, ylabel: str, note: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(values)), values)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_title(title + "\n" + note, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def export_cases(con, seed: str, target_n: int, out_dir: Path, logger) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pools = {}
    for rel in ("identical", "core_subset", "semmed_subset", "partial_overlap", "disjoint", "semmed_pmids_empty", "core_pmids_empty"):
        rows = con.execute(
            "SELECT gene_id, disease_cui, n_union FROM rel_all WHERE relation_class = ?",
            [rel],
        ).fetchall()
        pools[rel] = [{"key": f"{g}|{d}", "gene_id": g, "disease_cui": d, "volume": u} for g, d, u in rows]
    hetero = con.execute(
        """
        SELECT p.gene_id, p.disease_cui, p.n_pmid
        FROM pair p
        JOIN rel_all r USING (gene_id, disease_cui)
        WHERE p.heterogeneous_across_pmid = TRUE AND r.n_intersection > 0
        """
    ).fetchall()
    pools["same_pmid_label_set_differs"] = [{"key": f"{g}|{d}", "gene_id": g, "disease_cui": d, "volume": n or 0} for g, d, n in hetero]
    amb = con.execute(
        """
        SELECT COALESCE(gene_id, gcs_uri), COALESCE(disease_cui, ''), COALESCE(n_gene_uri, 0)
        FROM gcs_status WHERE pair_status = 'ambiguous_multiple'
        """
    ).fetchall()
    pools["mapping_ambiguous"] = [{"key": f"amb|{k}|{d}", "gene_id": k, "disease_cui": d, "volume": n} for k, d, n in amb]
    chosen = select_cases(pools, seed, target_n)
    manifest = []
    for i, row in enumerate(chosen["selected"], start=1):
        case = _one_case(con, i, row)
        path = out_dir / f"case_{i:02d}.json"
        path.write_text(json.dumps(case, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest.append({"file": path.name, "stratum": row["stratum"], "key": row["key"], "reason": row["reason"]})
    payload = {"selection": chosen, "files": manifest}
    (out_dir / "case_index.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("phase=cases_written n=%s", len(manifest))
    return payload


def _clip(value: str | None, limit: int = 800) -> dict:
    if value is None:
        return {"status": "field_absent", "text": None}
    if len(value) <= limit:
        return {"status": "full", "text": value, "n_characters": len(value)}
    return {"status": "truncated", "text": value[:limit], "n_characters": len(value), "note": f"display truncated to {limit} characters; index is complete"}


def _one_case(con, index: int, row: dict) -> dict:
    gene = row.get("gene_id")
    disease = row.get("disease_cui")
    pmids = con.execute(
        """
        SELECT pmid, 'core' AS side FROM core_pair_pmid WHERE gene_id = ? AND disease_cui = ?
        UNION ALL
        SELECT pmid, 'semmed' FROM semmed_pair_pmid WHERE gene_id = ? AND disease_cui = ?
        """,
        [gene, disease, gene, disease],
    ).fetchall()
    core_set = {p for p, side in pmids if side == "core"}
    sem_set = {p for p, side in pmids if side == "semmed"}
    marked = []
    for pmid in sorted(core_set | sem_set):
        if pmid in core_set and pmid in sem_set:
            mark = "intersection"
        elif pmid in core_set:
            mark = "core_only"
        else:
            mark = "semmed_only"
        marked.append({"pmid": pmid, "mark": mark})
    core_text = con.execute(
        """
        SELECT sentence_uri, pmid, label_tuple, curated, content, gene_mention, disease_mention, has_type
        FROM core_sentence
        WHERE gene_id = ? AND disease_cui = ? AND pair_status = 'explicit'
        LIMIT 12
        """,
        [gene, disease],
    ).fetchall()
    sem_text = con.execute(
        """
        SELECT predication_id, sentence_id, pmid, predicate, negation_prefix, disease_position
        FROM matched_predication
        WHERE gene_id = ? AND disease_cui = ?
        LIMIT 12
        """,
        [gene, disease],
    ).fetchall()
    return {
        "case_index": index,
        "stratum": row.get("stratum"),
        "reason": row.get("reason"),
        "gene_id": gene,
        "disease_cui": disease,
        "mapping": "CORE Entrez and UMLS CUI namespaces to SemMedDB GENE_ID and CUI, id_rule_v1",
        "pmid_index_complete": True,
        "pmid_count": len(marked),
        "pmids": marked,
        "core_sentence_sample": [
            {
                "sentence_uri": r[0],
                "pmid": r[1],
                "label_tuple": r[2],
                "curated": r[3],
                "content": _clip(r[4]),
                "gene_mention": r[5],
                "disease_mention": r[6],
                "has_type": r[7],
            }
            for r in core_text
        ],
        "core_sentence_sample_limit": 12,
        "semmed_predication_sample": [
            {"predication_id": r[0], "sentence_id": r[1], "pmid": r[2], "predicate": r[3], "negation_prefix": r[4], "disease_position": r[5]}
            for r in sem_text
        ],
        "scientific_judgment": "not_made",
    }

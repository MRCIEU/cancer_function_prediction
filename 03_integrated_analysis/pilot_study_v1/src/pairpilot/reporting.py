"""Chinese report computed only from saved tables."""

from __future__ import annotations

import statistics
from pathlib import Path

from pairpilot.store import query_rows, read_json, write_json, connect


def write_report(run: Path, bundle: dict) -> None:
    status = read_json(run / "audit" / "stage_status.json") if (run / "audit" / "stage_status.json").exists() else {}
    selected = _maybe(run / "inputs" / "selected_pairs.parquet")
    documents = _maybe(run / "inputs" / "documents.parquet")
    metrics = _maybe(run / "tables" / "experiment_metrics.parquet")
    deviations = _lines(run / "audit" / "deviations.jsonl")
    lines = [
        "# CORE–SemMedDB pilot",
        "",
        "这是探索性 pilot 的程序报告。模型输出尚未经过人工科学验证。覆盖差异不是科学结论相反；文章数不是独立研究数；引用能够定位也不表示原文支持了模型判断。",
        "",
        f"run_id：`{run.name}`。规则：`{bundle['pilot']['rule_version']}`。主模型配置：`{bundle['pilot']['primary_model']['model_id']}`。预算：{'已批准' if bundle['resources']['budget'].get('approved') else '未批准，付费模型阶段未提交'}。",
        "",
        "## 样本",
        "",
    ]
    if not selected:
        lines.append("抽样表尚未写出。")
    else:
        lines.append("| stratum | pair | PMID 并集 | CORE | SemMedDB |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in selected:
            lines.append(
                f"| {row['stratum']} | {row['pair_id']} | {row['n_union']} | {row['n_core']} | {row['n_semmed']} |"
            )
        lines.append("")
        lines.append(f"选中 pair {len(selected)} 个。计划偏差 {len(deviations)} 条，见 `audit/deviations.jsonl`。")
    if documents:
        counts = {}
        for row in documents:
            counts[row.get("completeness") or "unknown"] = counts.get(row.get("completeness") or "unknown", 0) + 1
        lines.extend(["", "摘要获取：" + "，".join(f"{key} {value}" for key, value in sorted(counts.items()))])
    lines.extend(["", "## 阶段", ""])
    for name in ("preflight", "prepare", "semmed_sentence_recovery", "canary", "extract", "analyze", "E1", "E2", "E3", "E4", "E5", "E6", "report"):
        lines.append(f"- {name}：{status_of(status, name)}")
    lines.extend(["", "## E1", ""])
    if not metrics:
        lines.append("E1 没有结果表。没有 finding 时这里保持为空，不补零冒充已分析。")
    else:
        lines.append(_metric_text(metrics))
    lines.extend([
        "",
        "## 其他实验",
        "",
        "E2、E3、E5、E6 在主模型调用被预算拦住时不产生科学输出。E4 在没有 gene–gene 边索引时记为 skipped_unavailable_input，不重建全库。开放模型对照默认关闭。",
        "",
        "等文章量的 50 次抽取如果已经写出，只描述这个样本的敏感性分布，不是 50 个独立实验。本报告不对 10 个探索性 pair 做总体错误率或显著性声明。",
        "",
        "仍需正式人工验证的是：finding 是否被原文支持、比较是否合理、候选缺口和路径是否成立。稳定性、结构有效率和引用定位率都不是准确率。",
        "",
    ])
    text = "\n".join(lines)
    (run / "reports").mkdir(exist_ok=True)
    (run / "reports" / "report.md").write_text(text, encoding="utf-8")
    for row in selected:
        _pair_page(run, row, documents, metrics)
    write_json(run / "reports" / "report_status.json", {"n_pair_page": len(selected), "n_metric": len(metrics)})


def _pair_page(run: Path, pair: dict, documents: list[dict], metrics: list[dict]) -> None:
    folder = run / "reports" / "pairs"
    folder.mkdir(parents=True, exist_ok=True)
    own = [row for row in metrics if row.get("pair_id") == pair["pair_id"]]
    body = [
        f"# {pair['pair_id']}",
        "",
        f"分层：{pair['stratum']}。CORE PMID {pair['n_core']}，SemMedDB PMID {pair['n_semmed']}，并集 {pair['n_union']}。",
        "",
        "原生记录、findings、比较和检索变化以本 run 的 parquet 为准。本页没有模型结果时，对应小节保持未运行，而不是写零发现。",
        "",
        f"已保存的 E1 指标行：{len(own)}。",
        "",
    ]
    (folder / f"{pair['pair_id']}.md").write_text("\n".join(body), encoding="utf-8")


def _metric_text(metrics: list[dict]) -> str:
    lines = ["| 实验 | 配置 | pair | 指标 | 分子 | 分母 | 状态 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in metrics[:80]:
        lines.append(
            f"| {row.get('experiment')} | {row.get('config')} | {row.get('pair_id')} | {row.get('metric')} | {row.get('numerator')} | {row.get('denominator')} | {row.get('status')} |"
        )
    if len(metrics) > 80:
        lines.append("")
        lines.append(f"表中还有 {len(metrics) - 80} 行，完整数值在 `tables/experiment_metrics.parquet`。")
    values = [row.get("value") for row in metrics if isinstance(row.get("value"), (int, float)) and row.get("metric") == "condition_increment"]
    if values:
        lines.append("")
        lines.append(
            f"条件增量的中位数 {statistics.median(values):.3f}，范围 {min(values):.3f}–{max(values):.3f}。分母为零的比例保持 not_applicable，不记为 0。"
        )
    return "\n".join(lines)


def _maybe(path: Path) -> list[dict]:
    if not path.exists():
        return []
    con = connect()
    try:
        return query_rows(con, f"SELECT * FROM read_parquet('{path.as_posix()}')")
    finally:
        con.close()


def _lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def status_of(status: dict, name: str) -> str:
    row = status.get(name) or {}
    return row.get("state") or "not_started"

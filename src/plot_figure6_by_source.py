"""Mosaic of article-link categories by the source of the two adjacent edges."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

CORE = "#426B91"
SEMMED = "#A47740"
SHARED = "#879099"
INK = "#1C2429"
FRAME = "#D5DADF"
SOURCE_COLOR = {"core_only": CORE, "both_sources": SHARED, "semmed_only": SEMMED}
SOURCE_LABEL = {"core_only": "CORE", "both_sources": "Both", "semmed_only": "SemMedDB"}
SOURCE_ORDER = ("core_only", "both_sources", "semmed_only")
POSITIONS = ("2-edge 1–2", "3-edge 1–2", "3-edge 2–3")
CATEGORY_ORDER = (
    "No observed link",
    "Shared reference only",
    "Co-citation + shared reference",
    "Citation + co-citation + shared reference",
    "Any shared PMID",
    "Other observed links",
)
CATEGORY_COLOR = {
    "No observed link": "#E4E7EA",
    "Shared reference only": "#6F9AA3",
    "Co-citation + shared reference": "#8AA06A",
    "Citation + co-citation + shared reference": "#D0A08C",
    "Any shared PMID": "#6B74A8",
    "Other observed links": "#C3B0D0",
}

plt.rcParams.update(
    {
        "font.family": "Liberation Sans",
        "font.size": 8,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _columns(cross: list[dict], coverage: list[dict]) -> dict[str, list[dict]]:
    decidable = {}
    qualifying = {}
    for row in coverage:
        if row["left_source"] == "ALL":
            continue
        key = (row["position"], row["left_source"], row["right_source"])
        decidable[key] = int(row["decidable_distinct"])
        qualifying[key] = int(row["qualifying_both_distinct"])
    counts: dict[tuple, dict[str, int]] = {}
    for row in cross:
        key = (row["position"], row["left_source"], row["right_source"])
        bucket = counts.setdefault(key, {name: 0 for name in CATEGORY_ORDER})
        bucket[row["category"]] += int(row["distinct_count"])
    grouped = {}
    for position in POSITIONS:
        total = sum(decidable[(position, left, right)] for left in SOURCE_ORDER for right in SOURCE_ORDER)
        columns = []
        for left in SOURCE_ORDER:
            for right in SOURCE_ORDER:
                amount = decidable[(position, left, right)]
                if amount <= 0:
                    continue
                parts = counts[(position, left, right)]
                if sum(parts.values()) != amount:
                    raise RuntimeError(f"{position} {left} {right} categories {sum(parts.values())} != {amount}")
                columns.append(
                    {
                        "left": left,
                        "right": right,
                        "d": amount,
                        "n": qualifying[(position, left, right)],
                        "width": amount / total,
                        "parts": parts,
                    }
                )
        width_sum = sum(column["width"] for column in columns)
        if abs(width_sum - 1) > 1e-9:
            raise RuntimeError(f"{position} column widths sum to {width_sum}")
        for column in columns:
            heights = [column["parts"][name] / column["d"] for name in CATEGORY_ORDER]
            if abs(sum(heights) - 1) > 1e-9:
                raise RuntimeError(f"{position} block heights do not sum to 1")
            for name, height in zip(CATEGORY_ORDER, heights):
                if abs(column["width"] * height * total - column["parts"][name]) > 1e-4:
                    raise RuntimeError(f"{position} area does not match {name}")
        grouped[position] = columns
    return grouped


def _contrast(grouped: dict[str, list[dict]]) -> dict | None:
    best = None
    categories = (
        "No observed link",
        "Shared reference only",
        "Co-citation + shared reference",
        "Citation + co-citation + shared reference",
        "Any shared PMID",
    )
    for position, columns in grouped.items():
        eligible = [column for column in columns if column["d"] >= 100_000]
        for category in categories:
            for index, left in enumerate(eligible):
                for right in eligible[index + 1 :]:
                    left_share = left["parts"][category] / left["d"]
                    right_share = right["parts"][category] / right["d"]
                    gap = abs(left_share - right_share)
                    if best is not None and gap <= best["gap"]:
                        continue
                    high, low = (left, right) if left_share >= right_share else (right, left)
                    best = {"position": position, "category": category, "gap": gap, "high": high, "low": low}
    return best


def _pair_label(column: dict, short: bool) -> str:
    left = SOURCE_LABEL[column["left"]]
    right = SOURCE_LABEL[column["right"]]
    if short:
        return f"{left[:1]}·{right[:1]}"
    return f"{left} · {right}"


def _source_dots(ax: plt.Axes, center: float, y: float, column: dict, gap: float) -> None:
    ax.scatter(
        [center - gap, center + gap],
        [y, y],
        s=18,
        c=[SOURCE_COLOR[column["left"]], SOURCE_COLOR[column["right"]]],
        linewidths=0,
        clip_on=False,
        zorder=4,
    )


def _text_width(ax: plt.Axes, text: str) -> float:
    probe = ax.text(0, 0, text, fontsize=7.5, color=INK)
    ax.figure.canvas.draw()
    width = probe.get_window_extent().width * 72 / ax.figure.dpi
    probe.remove()
    return width


def _draw_headers(ax: plt.Axes, columns: list[dict]) -> None:
    fig = ax.figure
    fig.canvas.draw()
    width_px = ax.get_window_extent().width
    cursor = 0.0
    for column in columns:
        center = cursor + column["width"] / 2
        width_pt = column["width"] * width_px * 72 / fig.dpi
        full = _pair_label(column, short=False)
        left_name = SOURCE_LABEL[column["left"]]
        right_name = SOURCE_LABEL[column["right"]]
        if _text_width(ax, full) + 8 <= width_pt:
            gap = min(0.012, column["width"] * 0.16)
            _source_dots(ax, center, 1.05, column, gap)
            ax.text(center, 1.11, full, ha="center", va="bottom", fontsize=7.5, color=INK, clip_on=False)
        elif max(_text_width(ax, left_name), _text_width(ax, right_name)) + 3 <= width_pt:
            ax.text(center, 1.16, left_name, ha="center", va="bottom", fontsize=7.5, color=SOURCE_COLOR[column["left"]], clip_on=False)
            ax.text(center, 1.03, right_name, ha="center", va="bottom", fontsize=7.5, color=SOURCE_COLOR[column["right"]], clip_on=False)
        else:
            ax.text(center, 1.15, left_name[:1], ha="center", va="bottom", fontsize=7.5, color=SOURCE_COLOR[column["left"]], clip_on=False)
            ax.text(center, 1.02, right_name[:1], ha="center", va="bottom", fontsize=7.5, color=SOURCE_COLOR[column["right"]], clip_on=False)
        cursor += column["width"]


def _draw_mosaic(ax: plt.Axes, columns: list[dict], position: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.42)
    ax.axis("off")
    ax.set_facecolor("white")
    cursor = 0.0
    for column in columns:
        y = 0.0
        for name in CATEGORY_ORDER:
            height = column["parts"][name] / column["d"]
            if height <= 0:
                continue
            ax.add_patch(
                Rectangle(
                    (cursor, y),
                    column["width"],
                    height,
                    facecolor=CATEGORY_COLOR[name],
                    edgecolor="white",
                    linewidth=0.35,
                    joinstyle="miter",
                    zorder=1,
                )
            )
            y += height
        cursor += column["width"]
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, edgecolor=FRAME, linewidth=0.7, zorder=3))
    ax.text(0, 0.5, position, ha="right", va="center", fontsize=8, color=INK, transform=ax.get_yaxis_transform(), clip_on=False)


def _legend(fig: plt.Figure) -> None:
    handles = [Rectangle((0, 0), 1, 1, facecolor=CATEGORY_COLOR[name], edgecolor="none", label=name) for name in CATEGORY_ORDER]
    handles.extend(
        [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=CORE, markersize=6.2, label="CORE"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=SHARED, markersize=6.2, label="Both"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=SEMMED, markersize=6.2, label="SemMedDB"),
        ]
    )
    legend = fig.legend(
        handles=handles,
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.55, 0.008),
        fontsize=7.5,
        handlelength=1.05,
        handleheight=0.75,
        columnspacing=1.05,
        labelspacing=0.32,
    )
    for text in legend.get_texts():
        text.set_fontsize(7.5)


def _caption(path: Path, cross: list[dict], coverage: list[dict], contrast: dict | None) -> None:
    totals = [row for row in coverage if row["left_source"] == "ALL"]
    separate = any(row["category"] == "Citation + co-citation + shared reference" for row in cross)
    triple = {}
    for row in totals:
        position = row["position"]
        amount = sum(
            int(item["distinct_count"])
            for item in cross
            if item["position"] == position and item["bits"] == "0111"
        )
        triple[position] = amount / int(row["decidable_distinct"])
    lines = [
        "# Article links by edge source",
        "",
        "Each row is one path position. A column is an ordered pair of edge sources, and its width is that pair’s share of the position’s decidable adjacent pairs. Color blocks are the article-link mix inside the column. Block area is the joint share of that source pair and article-link category. Every distinct ordered adjacent pair has equal weight.",
        "",
        "N is the number of distinct ordered adjacent pairs with a qualifying PMID on both edges. Pairs missing a qualifying PMID on either edge are outside N. D is the subset of N for which same article, direct citation, co-citation, and shared reference are all decidable. The mosaic uses D. Coverage is D/N and is reported by source pair in `data/figure6_source_coverage.tsv`. A source pair with N > 0 and D = 0 has no pairs in the mosaic.",
        "",
        "No observed link is the decidable pattern 0000. It is not the set of pairs with incomplete citation coverage. Same article is a nonempty intersection of the two qualifying PMID sets. Direct citation counts either direction. Co-citation and shared reference keep third-party articles. A relation found in the August 2024 snapshot counts under partial coverage. Absence counts only when the index required for that relation is complete. Several relations on one edge pair can come from different article pairs.",
        "",
        "CORE, SemMedDB, and Both describe the allowed records on one entity edge. Both means that edge has an allowed CORE record and an allowed SemMedDB traversal record. Gene–gene edges are SemMedDB-only, so the first edge of every two-edge pair is SemMedDB. Left and right stay in path order. Above a wide column, the dots and the name read left edge then right edge. Above a narrower column, the upper label is the left edge and the lower label is the right edge.",
        "",
    ]
    if separate:
        lines.append(
            "Citation + co-citation + shared reference is its own block because it is "
            f"{triple['3-edge 2–3']:.1%} of decidable pairs at 3-edge 2–3 "
            f"({triple['2-edge 1–2']:.1%} at 2-edge 1–2 and {triple['3-edge 1–2']:.1%} at 3-edge 1–2). "
            "The same six categories are used at every position. All sixteen patterns and their categories are in `data/figure6_source_category_map.tsv`."
        )
    else:
        lines.append(
            "Citation + co-citation + shared reference is included in Other observed links because its share of decidable pairs is below 5% at every position. "
            "All sixteen patterns are in `data/figure6_source_category_map.tsv`."
        )
    lines.extend(["", "Conditional denominators, D, the decidable pairs drawn in each row:"])
    for row in totals:
        lines.append(
            f"- {row['position']}: D = {int(row['decidable_distinct']):,} of N = {int(row['qualifying_both_distinct']):,} qualifying pairs "
            f"(coverage {float(row['coverage']):.1%}); {int(row['unknown_distinct']):,} qualifying pairs remain undecidable."
        )
    if contrast is not None:
        high = contrast["high"]
        low = contrast["low"]
        lines.extend(
            [
                "",
                (
                    f"At {contrast['position']}, {contrast['category']} is "
                    f"{high['parts'][contrast['category']] / high['d']:.1%} of {_pair_label(high, short=False)} "
                    f"(D = {high['d']:,}) and {low['parts'][contrast['category']] / low['d']:.1%} of {_pair_label(low, short=False)} "
                    f"(D = {low['d']:,})."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "Figure 4 uses a wider denominator: its citation percentages keep partial coverage, and its same-article percentage uses the union of CORE and SemMedDB PMIDs. Those percentages are not the block heights in this mosaic. Full counts, path weights, and coverage are in `data/figure6_source_cross.tsv` and `data/figure6_source_coverage.tsv`.",
            "",
            "Sources: edge support from `path_evidence_audit_20261116T1310Z`, qualifying PMIDs and directed links from `citation_path_evidence_audit_20261116T1647Z`. Pattern totals were reconciled to `figure6_article_combinations.tsv`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_figure(out_dir: Path) -> None:
    data = out_dir / "data"
    cross = _read(data / "figure6_source_cross.tsv")
    coverage = _read(data / "figure6_source_coverage.tsv")
    grouped = _columns(cross, coverage)
    contrast = _contrast(grouped)
    fig, axes = plt.subplots(3, 1, figsize=(7.45, 7.05))
    fig.patch.set_facecolor("white")
    for ax, position in zip(axes, POSITIONS):
        _draw_mosaic(ax, grouped[position], position)
    fig.suptitle("Article links by edge source", fontsize=10, color=INK, y=0.985)
    fig.subplots_adjust(left=0.16, right=0.985, top=0.935, bottom=0.16, hspace=0.18)
    for ax, position in zip(axes, POSITIONS):
        _draw_headers(ax, grouped[position])
    _legend(fig)
    fig.savefig(out_dir / "figure6_article_links_by_source.svg")
    fig.savefig(out_dir / "figure6_article_links_by_source.pdf")
    fig.savefig(out_dir / "figure6_article_links_by_source.png", dpi=500)
    plt.close(fig)
    _caption(out_dir / "figure6_article_links_by_source_caption.md", cross, coverage, contrast)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    plot_figure(args.out_dir)


if __name__ == "__main__":
    main()

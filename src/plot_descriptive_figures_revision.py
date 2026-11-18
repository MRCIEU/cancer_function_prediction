"""Revised layouts for the five descriptive Project 2 figures.

The first export stays in scientific_figures_20261117T0219Z. This script writes
a separate directory and reuses the audited tables plus the saved sequence file.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon

from src.plot_descriptive_figures import (
    CITATION,
    COMPLETION,
    CORE,
    GRID,
    INK,
    JOINT,
    SEMMED,
    SHARED,
    circle_intersection,
    comma,
    lens,
    read_tsv,
    solve_distance,
    style_axis,
    write_tsv,
)

OUT = Path("/home/b5ac/freddieyu.b5ac/projects/project_2/results/scientific_figures_revision_20261117T0303Z")
PRIOR = Path("/home/b5ac/freddieyu.b5ac/projects/project_2/results/scientific_figures_20261117T0219Z")
PALE = "#F3F4F6"
EMPHASIS = "#5E6872"
LINK = "#C5CDD4"
LIGHT_JOINT = "#D5CCE0"

plt.rcParams.update(
    {
        "font.family": "Liberation Sans",
        "font.size": 8.5,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
    }
)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.svg", pad_inches=0.03)
    fig.savefig(OUT / f"{stem}.pdf", pad_inches=0.03)
    fig.savefig(OUT / f"{stem}.png", dpi=500, pad_inches=0.03)
    plt.close(fig)


def figure_direct_pairs() -> None:
    core_only, shared, semmed_only = 109_703, 24_160, 82_303
    core_total = core_only + shared
    semmed_total = semmed_only + shared
    union = core_only + shared + semmed_only
    radius_core = math.sqrt(core_total)
    radius_semmed = math.sqrt(semmed_total)
    distance = solve_distance(radius_core, radius_semmed, shared)
    achieved = circle_intersection(radius_core, radius_semmed, distance)
    if abs(achieved - shared) / shared > 1e-8:
        raise RuntimeError(f"circle intersection {achieved} != {shared}")
    scale = 1.15 / radius_core
    rc = radius_core * scale
    rs = radius_semmed * scale
    dd = distance * scale
    chord = (dd * dd + rc * rc - rs * rs) / (2 * dd)
    half = math.sqrt(max(rc * rc - chord * chord, 0.0))
    fig, ax = plt.subplots(figsize=(4.55, 3.15))
    fig.patch.set_facecolor("white")
    ax.set_aspect("equal")
    ax.add_patch(Circle((0, 0), rc, facecolor=CORE, edgecolor="none", zorder=1))
    ax.add_patch(Circle((dd, 0), rs, facecolor=SEMMED, edgecolor="none", zorder=2))
    ax.add_patch(Polygon(lens(rc, rs, dd), closed=True, facecolor=SHARED, edgecolor="none", zorder=3))
    ax.text(-rc * 0.42, 0.0, comma(core_only), ha="center", va="center", color="white", fontsize=9, zorder=4)
    ax.text(dd + rs * 0.40, 0.0, comma(semmed_only), ha="center", va="center", color="white", fontsize=9, zorder=4)
    name_gap = 0.07
    ax.text(0, rc + name_gap, "CORE", ha="center", va="bottom", color=CORE, fontsize=9)
    ax.text(dd, rs + name_gap, "SemMedDB", ha="center", va="bottom", color=SEMMED, fontsize=9)
    label_y = -max(rc, rs) - 0.05
    ax.plot([chord, chord], [-half + 0.015, label_y + 0.012], color=SHARED, lw=0.9, solid_capstyle="round", zorder=4)
    ax.text(chord, label_y, f"Shared {comma(shared)}", ha="center", va="top", color=INK, fontsize=8.5, zorder=4)
    ax.set_xlim(-rc - 0.06, dd + rs + 0.06)
    ax.set_ylim(label_y - 0.14, max(rc, rs) + name_gap + 0.16)
    ax.axis("off")
    ax.set_title("Direct gene–cancer pairs", fontsize=10, color=INK, pad=3)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.02)
    save_figure(fig, "figure1_direct_pairs")
    write_tsv(
        OUT / "data" / "figure1_direct_pairs.tsv",
        [
            {"region": "CORE only", "pairs": core_only, "circle_pairs": core_total, "share_of_union": core_only / union},
            {"region": "shared", "pairs": shared, "circle_pairs": shared, "share_of_union": shared / union},
            {"region": "SemMedDB only", "pairs": semmed_only, "circle_pairs": semmed_total, "share_of_union": semmed_only / union},
            {"region": "union", "pairs": union, "circle_pairs": union, "share_of_union": 1},
        ],
    )
    (OUT / "figure1_direct_pairs_caption.md").write_text(
        "\n".join(
            [
                "# Direct gene–cancer pairs",
                "",
                f"Circle areas are eligible direct gene–cancer pairs in the fixed universe of 13,571 resolved human candidate genes and 1,665 cancer queries. CORE contains {comma(core_total)} pairs and SemMedDB contains {comma(semmed_total)}. The lens is the {comma(shared)} pairs recorded in both resources, {shared / union:.1%} of their union of {comma(union)}. The exclusive regions are {comma(core_only)} and {comma(semmed_only)}.",
                "",
                "Radii are proportional to the square root of those circle totals, and the distance between centers is the value that makes the standard two-circle intersection equal 24,160. This universe does not include SemMedDB gene–cancer edges whose cancer is not a query.",
                "",
                "Source: `path_evidence_audit_completion_20261116T1402Z` direct-pair totals.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def figure_paths_and_pairs() -> None:
    path_rows = read_tsv(COMPLETION / "analysis" / "source_sequences.tsv")
    two_rows = read_tsv(COMPLETION / "analysis" / "two_edge_source_sequences.tsv")
    dependence = read_tsv(COMPLETION / "analysis" / "candidate_dependence.tsv")
    dep = {(row["population"], row["category"]): int(row["pairs"]) for row in dependence}
    three_paths = sum(int(row["complete_paths"]) for row in path_rows)
    two_paths = sum(int(row["complete_paths"]) for row in two_rows)
    three_joint = sum(int(row["complete_paths"]) for row in path_rows if row["sequence_group"] == "requires_both_resources")
    two_joint = sum(int(row["complete_paths"]) for row in two_rows if row["sequence_group"] == "requires_both_resources")
    series = [
        ("2-edge", "Complete paths", two_joint, two_paths, "o"),
        ("2-edge", "Reachable pairs without a direct record", dep[("two_no_direct", "joint_only_reach")], dep[("two_no_direct", "joint_reached")], "s"),
        ("3-edge", "Complete paths", three_joint, three_paths, "o"),
        ("3-edge", "Reachable pairs without a direct record", dep[("three_no_direct", "joint_only_reach")], dep[("three_no_direct", "joint_reached")], "s"),
    ]
    expected = {"2-edge": ("33.7%", "27.1%"), "3-edge": ("48.8%", "4.9%")}
    percents = {}
    for length, unit, numerator, denominator, _marker in series:
        percents[(length, unit)] = 100 * numerator / denominator
    if f"{percents[('2-edge', 'Complete paths')]:.1f}%" != expected["2-edge"][0]:
        raise RuntimeError("2-edge path label drifted")
    if f"{percents[('2-edge', 'Reachable pairs without a direct record')]:.1f}%" != expected["2-edge"][1]:
        raise RuntimeError("2-edge pair label drifted")
    if f"{percents[('3-edge', 'Complete paths')]:.1f}%" != expected["3-edge"][0]:
        raise RuntimeError("3-edge path label drifted")
    if f"{percents[('3-edge', 'Reachable pairs without a direct record')]:.1f}%" != expected["3-edge"][1]:
        raise RuntimeError("3-edge pair label drifted")
    fig, ax = plt.subplots(figsize=(5.45, 2.45))
    fig.patch.set_facecolor("white")
    style_axis(ax)
    y_base = {"2-edge": 1, "3-edge": 0}
    for length in ("2-edge", "3-edge"):
        pair = [item for item in series if item[0] == length]
        xs = [100 * item[2] / item[3] for item in pair]
        y = y_base[length]
        ax.plot(xs, [y, y], color=LIGHT_JOINT, lw=0.8, zorder=2)
        for (_length, _unit, numerator, denominator, marker), x in zip(pair, xs):
            ax.plot([x], [y], marker=marker, color=JOINT, markersize=6.5, linestyle="none", zorder=3)
    ax.text(percents[("2-edge", "Reachable pairs without a direct record")], 1.16, "27.1%", ha="center", va="bottom", color=JOINT, fontsize=8)
    ax.text(percents[("2-edge", "Complete paths")], 1.16, "33.7%", ha="center", va="bottom", color=JOINT, fontsize=8)
    ax.text(percents[("3-edge", "Reachable pairs without a direct record")] + 1.3, 0, "4.9%", ha="left", va="center", color=JOINT, fontsize=8, fontweight="bold")
    ax.text(percents[("3-edge", "Complete paths")], 0.16, "48.8%", ha="center", va="bottom", color=JOINT, fontsize=8, fontweight="bold")
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["2-edge", "3-edge"])
    ax.set_xlabel("Share requiring both sources (%)", fontsize=8.5)
    ax.set_xlim(0, 64)
    ax.set_ylim(-0.42, 1.48)
    ax.grid(axis="x", color=GRID, lw=0.45)
    ax.set_axisbelow(True)
    ax.set_title("Paths and reachable pairs", fontsize=10, color=INK, pad=6)
    handles = [
        Line2D([0], [0], marker="o", color=JOINT, linestyle="none", markersize=6, label="Complete paths"),
        Line2D([0], [0], marker="s", color=JOINT, linestyle="none", markersize=5.5, label="Reachable pairs without a direct record"),
    ]
    ax.legend(handles=handles, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.32), fontsize=7.5, handletextpad=0.3, columnspacing=1.0, borderaxespad=0)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.88, bottom=0.28)
    save_figure(fig, "figure2_paths_and_pairs")
    rows = []
    for length, unit, numerator, denominator, _marker in series:
        rows.append({"length": length, "unit": unit, "numerator": numerator, "denominator": denominator, "percent": numerator / denominator})
    write_tsv(OUT / "data" / "figure2_paths_and_pairs.tsv", rows)
    pair_n = dep[("three_no_direct", "joint_only_reach")]
    pair_d = dep[("three_no_direct", "joint_reached")]
    (OUT / "figure2_paths_and_pairs_caption.md").write_text(
        "\n".join(
            [
                "# Paths and reachable pairs",
                "",
                "Circles are complete paths. Squares are deduplicated candidate–query pairs that the joint graph reaches and that have no eligible direct record. A path requires both sources when its edges include a CORE-only edge and a SemMedDB-only edge. A pair requires both sources when no complete path of that length can be walked in either resource alone.",
                "",
                f"Two-edge paths: {comma(two_joint)} / {comma(two_paths)} (33.7%). Two-edge pairs: {comma(dep[('two_no_direct', 'joint_only_reach')])} / {comma(dep[('two_no_direct', 'joint_reached')])} (27.1%).",
                f"Three-edge paths: {comma(three_joint)} / {comma(three_paths)} (48.8%). Three-edge pairs: {comma(pair_n)} / {comma(pair_d)} (4.9%).",
                "",
                "These are full counts on the audited path view, not a sample.",
                "",
                "Source: `path_evidence_audit_completion_20261116T1402Z/analysis/source_sequences.tsv`, `two_edge_source_sequences.tsv`, and `candidate_dependence.tsv`.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def figure_articles_added() -> None:
    gain = {row["measure"]: int(row["value"]) for row in read_tsv(COMPLETION / "analysis" / "shared_pair_gain_summary.tsv")}
    cells = [
        (0, 0, gain["both_add_pmids"]),
        (0, 1, gain["only_core_adds_pmids"]),
        (1, 0, gain["only_semmed_adds_pmids"]),
        (1, 1, gain["neither_adds_pmids"]),
    ]
    total = gain["pairs"]
    if sum(count for _row, _column, count in cells) != total:
        raise RuntimeError("shared-pair cells do not sum to 24,160")
    fig, ax = plt.subplots(figsize=(5.15, 2.55))
    fig.patch.set_facecolor("white")
    for row, column, count in cells:
        percent = 100 * count / total
        accent = (row, column) == (0, 0)
        ax.add_patch(plt.Rectangle((column, 1 - row), 1, 1, facecolor=EMPHASIS if accent else PALE, edgecolor="white", linewidth=2.5, zorder=1))
        color = "white" if accent else INK
        ax.text(column + 0.5, 1.58 - row, f"{percent:.1f}%", ha="center", va="center", color=color, fontsize=12, zorder=2)
        ax.text(column + 0.5, 1.28 - row, comma(count), ha="center", va="center", color=color, fontsize=8, zorder=2)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Yes", "No"], fontsize=8.5)
    ax.set_yticks([1.5, 0.5])
    ax.set_yticklabels(["Yes", "No"], fontsize=8.5)
    ax.tick_params(length=0)
    ax.set_xlabel("SemMedDB exclusive PMID", fontsize=8.5, labelpad=3)
    ax.set_ylabel("CORE exclusive PMID", rotation=0, ha="right", va="center", fontsize=8.5, labelpad=16)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Articles added on shared pairs", fontsize=10, color=INK, pad=6)
    fig.subplots_adjust(left=0.32, right=0.98, top=0.86, bottom=0.18)
    save_figure(fig, "figure3_articles_added")
    write_tsv(
        OUT / "data" / "figure3_articles_added.tsv",
        [
            {"core_exclusive_pmid": "yes", "semmed_exclusive_pmid": "yes", "pairs": gain["both_add_pmids"], "percent": gain["both_add_pmids"] / total, "denominator_pairs": total},
            {"core_exclusive_pmid": "yes", "semmed_exclusive_pmid": "no", "pairs": gain["only_core_adds_pmids"], "percent": gain["only_core_adds_pmids"] / total, "denominator_pairs": total},
            {"core_exclusive_pmid": "no", "semmed_exclusive_pmid": "yes", "pairs": gain["only_semmed_adds_pmids"], "percent": gain["only_semmed_adds_pmids"] / total, "denominator_pairs": total},
            {"core_exclusive_pmid": "no", "semmed_exclusive_pmid": "no", "pairs": gain["neither_adds_pmids"], "percent": gain["neither_adds_pmids"] / total, "denominator_pairs": total},
            {"core_exclusive_pmid": "", "semmed_exclusive_pmid": "", "pairs": gain["core_added_links"], "percent": "", "denominator_pairs": "core_added_pmid_links"},
            {"core_exclusive_pmid": "", "semmed_exclusive_pmid": "", "pairs": gain["semmed_added_links"], "percent": "", "denominator_pairs": "semmed_added_pmid_links"},
            {"core_exclusive_pmid": "", "semmed_exclusive_pmid": "", "pairs": gain["shared_links"], "percent": "", "denominator_pairs": "shared_pmid_links"},
            {"core_exclusive_pmid": "", "semmed_exclusive_pmid": "", "pairs": gain["distinct_core_added_pmids"], "percent": "", "denominator_pairs": "distinct_core_added_pmids"},
            {"core_exclusive_pmid": "", "semmed_exclusive_pmid": "", "pairs": gain["distinct_semmed_added_pmids"], "percent": "", "denominator_pairs": "distinct_semmed_added_pmids"},
            {"core_exclusive_pmid": "", "semmed_exclusive_pmid": "", "pairs": gain["distinct_shared_pmids"], "percent": "", "denominator_pairs": "distinct_shared_pmids"},
        ],
    )
    (OUT / "figure3_articles_added_caption.md").write_text(
        "\n".join(
            [
                "# Articles added on shared pairs",
                "",
                f"Yes means that resource contributes at least one PMID on the shared gene–cancer pair that the other resource does not record. The common denominator is the {comma(total)} pairs recorded by both resources. Cell counts are {comma(gain['both_add_pmids'])}, {comma(gain['only_core_adds_pmids'])}, {comma(gain['only_semmed_adds_pmids'])}, and {comma(gain['neither_adds_pmids'])}. Displayed percentages are rounded to one decimal place, so the four printed percentages can sum to 99.9 rather than 100.",
                "",
                f"Pair–PMID links are a separate unit: CORE adds {comma(gain['core_added_links'])} exclusive links, SemMedDB adds {comma(gain['semmed_added_links'])}, and {comma(gain['shared_links'])} links are shared. The distinct PMIDs in those three sets are {comma(gain['distinct_core_added_pmids'])}, {comma(gain['distinct_semmed_added_pmids'])}, and {comma(gain['distinct_shared_pmids'])}.",
                "",
                "Source: `path_evidence_audit_completion_20261116T1402Z/analysis/shared_pair_gain_summary.tsv`.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _citation() -> dict[tuple[str, str, str], tuple[int, int]]:
    found = {}
    for row in read_tsv(CITATION / "analysis" / "adjacent_citation_summary.tsv"):
        found[(row["template"], row["positions"], row["measure"])] = (int(row["distinct_count"]), int(row["path_occurrence_count"]))
    return found


def _overlap() -> dict[tuple[str, str, str], tuple[int, int]]:
    found = {}
    for row in read_tsv(COMPLETION / "analysis" / "adjacent_overlap_summary.tsv"):
        if row["comparison"] != "union":
            continue
        found[(row["template"], row["positions"], row["weighting"])] = (int(row["any_overlap"]), int(row["both_known"]))
    return found


def figure_article_links() -> None:
    citation = _citation()
    overlap = _overlap()
    positions = [
        ("2-edge 1–2", "one_intermediate_path", "positions_1_2"),
        ("3-edge 1–2", "two_intermediate_path", "positions_1_2"),
        ("3-edge 2–3", "two_intermediate_path", "positions_2_3"),
    ]
    relations = [
        ("Same article", "same_article", "cocite"),
        ("Direct citation", "collapsed_pairs", "ltr"),
        ("Co-citation", "cocite_pairs", "cocite"),
        ("Shared reference", "ref_pairs", "ref"),
    ]
    rows = []
    for position, template, slot in positions:
        for relation, measure, coverage_key in relations:
            if measure == "same_article":
                distinct_n, distinct_d = overlap[(template, slot, "distinct_count")]
                weighted_n, weighted_d = overlap[(template, slot, "path_occurrence_count")]
                rule = "union of allowed CORE and SemMedDB PMIDs; denominator is pairs with a PMID on both edges"
                partial_stays = "not applicable; PMID sets are fully observed and empty sets are excluded"
                complete = partial = absent = ""
                coverage_name = ""
            else:
                distinct_n, weighted_n = citation[(template, slot, measure)]
                distinct_d, weighted_d = citation[(template, slot, "both")]
                rule = "qualifying PMID on both edges; numerator is a link observed in the citation snapshot"
                partial_stays = "yes; partial or absent index coverage without an observed link stays in the denominator"
                complete, _weighted_complete = citation[(template, slot, f"{coverage_key}_index_complete")]
                partial, _weighted_partial = citation[(template, slot, f"{coverage_key}_index_partial")]
                absent, _weighted_absent = citation[(template, slot, f"{coverage_key}_index_absent")]
                coverage_name = {"ltr": "left outgoing index, one direction of a direct citation", "cocite": "incoming index on both edges", "ref": "outgoing index on both edges"}[coverage_key]
            rows.append(
                {
                    "relation": relation,
                    "position": position,
                    "distinct_numerator": distinct_n,
                    "distinct_denominator": distinct_d,
                    "distinct_percent": 100 * distinct_n / distinct_d,
                    "weighted_numerator": weighted_n,
                    "weighted_denominator": weighted_d,
                    "weighted_percent": 100 * weighted_n / weighted_d,
                    "weights_equal": int(distinct_n == weighted_n and distinct_d == weighted_d),
                    "denominator_rule": rule,
                    "partial_coverage_without_link_stays_in_denominator": partial_stays,
                    "coverage_index": "" if measure == "same_article" else coverage_name,
                    "index_complete_distinct": complete,
                    "index_partial_distinct": partial,
                    "index_absent_distinct": absent,
                }
            )
    callouts = {
        ("Same article", "3-edge 2–3"): ("2.50%", "5.54%"),
        ("Direct citation", "3-edge 2–3"): ("12.46%", "24.38%"),
        ("Co-citation", "3-edge 2–3"): ("27.04%", "43.69%"),
        ("Shared reference", "3-edge 2–3"): ("47.42%", "62.57%"),
    }
    by_key = {(row["relation"], row["position"]): row for row in rows}
    for key, (left, right) in callouts.items():
        row = by_key[key]
        if f"{row['distinct_percent']:.2f}%" != left or f"{row['weighted_percent']:.2f}%" != right:
            raise RuntimeError(f"{key} labels are {row['distinct_percent']:.2f} and {row['weighted_percent']:.2f}")
    for row in rows:
        if row["position"] == "2-edge 1–2" and not row["weights_equal"]:
            raise RuntimeError(f"2-edge weights differ for {row['relation']}")
    fig, axes = plt.subplots(4, 1, figsize=(6.9, 5.15), sharey=True)
    fig.patch.set_facecolor("white")
    fig.suptitle("Article links along paths", fontsize=10, color=INK, y=0.985)
    y_pos = {"2-edge 1–2": 2, "3-edge 1–2": 1, "3-edge 2–3": 0}
    for ax, (relation, _measure, _coverage) in zip(axes, relations):
        style_axis(ax)
        subset = [row for row in rows if row["relation"] == relation]
        xmax = max(max(row["distinct_percent"], row["weighted_percent"]) for row in subset)
        for row in subset:
            y = y_pos[row["position"]]
            if row["weights_equal"]:
                ax.plot(row["distinct_percent"], y, marker="o", markerfacecolor="white", markeredgecolor=INK, markeredgewidth=1.05, markersize=8.2, linestyle="none", zorder=3)
                ax.plot(row["weighted_percent"], y, marker="o", color=INK, markersize=4.6, linestyle="none", zorder=4)
            else:
                ax.plot([row["distinct_percent"], row["weighted_percent"]], [y, y], color=LINK, lw=0.8, zorder=2)
                ax.plot(row["distinct_percent"], y, marker="o", markerfacecolor="white", markeredgecolor=INK, markeredgewidth=1.0, markersize=6.4, linestyle="none", zorder=3)
                ax.plot(row["weighted_percent"], y, marker="o", color=INK, markersize=5.6, linestyle="none", zorder=3)
            if row["position"] == "3-edge 2–3":
                ax.text(row["distinct_percent"], y + 0.24, f"{row['distinct_percent']:.2f}%", ha="center", va="bottom", fontsize=7.5, color=INK, fontweight="bold")
                ax.text(row["weighted_percent"], y - 0.24, f"{row['weighted_percent']:.2f}%", ha="center", va="top", fontsize=7.5, color=INK, fontweight="bold")
        ax.set_xlim(0, xmax * 1.22 + 2.5)
        ax.set_ylim(-0.72, 2.62)
        ax.set_yticks([2, 1, 0])
        ax.set_yticklabels(["2-edge 1–2", "3-edge 1–2", "3-edge 2–3"])
        ax.set_ylabel(relation, fontsize=8.5)
        ax.grid(axis="x", color=GRID, lw=0.4)
        ax.set_axisbelow(True)
    axes[-1].set_xlabel("Share with an observed link (%)", fontsize=8.5)
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor=INK, linestyle="none", markersize=6, label="Distinct adjacent edge pairs"),
            Line2D([0], [0], marker="o", color=INK, linestyle="none", markersize=5.5, label="Weighted by path occurrences"),
        ],
        frameon=False,
        ncol=2,
        loc="upper right",
        fontsize=7.5,
        bbox_to_anchor=(0.98, 0.955),
    )
    fig.subplots_adjust(left=0.18, right=0.98, top=0.90, bottom=0.08, hspace=0.48)
    save_figure(fig, "figure4_article_links")
    write_tsv(OUT / "data" / "figure4_article_links.tsv", rows)
    (OUT / "figure4_article_links_caption.md").write_text(
        "\n".join(
            [
                "# Article links along paths",
                "",
                "Open circles give each distinct adjacent edge pair one vote. Filled circles weight a pair by how many eligible complete paths contain it: a pair that occurs in several paths contributes several times. On a two-edge path the adjacent pair is the whole path, so the weight is 1 and the two points coincide. Edges 2–3 are the last two edges of a three-edge path. Each relation has its own linear axis from zero. Only edges 2–3 are labelled; every plotted percentage is in `data/figure4_article_links.tsv`.",
                "",
                "Same article uses the union of allowed CORE and SemMedDB PMIDs. Its denominator is the adjacent pairs with at least one such PMID on both edges. An empty PMID set is outside that denominator. Citation-index coverage does not apply.",
                "",
                "Direct citation, co-citation, and shared reference use the citation audit’s qualifying PMIDs. The denominator is every adjacent pair with a qualifying PMID on both edges. A link found in the August 2024 snapshot counts even when index coverage is partial. Pairs with partial or absent coverage and no observed link remain in the denominator, so the percentage is the share with an observed link in that snapshot, not the share among pairs whose index coverage is complete. Direct citation counts either direction. Co-citation and shared reference keep third-party articles. Direction, external witnesses, and complete, partial, or absent coverage remain in the same table.",
                "",
                "Sources: `path_evidence_audit_completion_20261116T1402Z/analysis/adjacent_overlap_summary.tsv` for same article, and `citation_path_evidence_audit_20261116T1647Z/analysis/adjacent_citation_summary.tsv` for the other three relations.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _sequence_rows() -> list[dict]:
    rows = []
    for row in read_tsv(PRIOR / "data" / "figure5_source_patterns.tsv"):
        if row["population"] != "no_direct":
            continue
        item = dict(row)
        for key in ("path_count", "pairs_with_sequence", "denominator_paths", "denominator_pairs"):
            item[key] = int(float(item[key])) if item[key] != "" else 0
        for key in ("path_proportion", "pair_proportion", "difference_percentage_points"):
            item[key] = float(item[key])
        item["length"] = int(item["length"])
        rows.append(item)
    return rows


def _has_shared(row: dict) -> bool:
    states = [row["position_1"], row["position_2"], row["position_3"]]
    return "both_sources" in states


def _sort_key(row: dict) -> tuple:
    return (
        -max(row["path_proportion"], row["pair_proportion"]),
        -row["path_proportion"],
        -row["path_count"],
        row["position_1"],
        row["position_2"],
        row["position_3"],
    )


def figure_source_patterns() -> None:
    source = [row for row in _sequence_rows() if row["path_count"] or row["pairs_with_sequence"]]
    for length in (2, 3):
        block = [row for row in source if row["length"] == length]
        if abs(sum(row["path_proportion"] for row in block) - 1) > 1e-8:
            raise RuntimeError(f"{length}-edge path shares do not sum to 1")
        if abs(sum(row["pair_proportion"] for row in block) - 1) > 1e-8:
            raise RuntimeError(f"{length}-edge pair shares do not sum to 1")
    two = sorted([row for row in source if row["length"] == 2], key=_sort_key)
    three = [row for row in source if row["length"] == 3]
    no_shared = sorted([row for row in three if not _has_shared(row)], key=_sort_key)
    shared = sorted([row for row in three if _has_shared(row)], key=_sort_key)
    if len(no_shared) + len(shared) != len(three):
        raise RuntimeError("three-edge groups overlap or drop a sequence")
    shown_plain = no_shared[:4]
    other_plain = no_shared[4:]
    shown_shared = shared[:3]
    other_shared = shared[3:]
    colors = {"core_only": CORE, "semmed_only": SEMMED, "both_sources": SHARED}

    def pack(row: dict, group: str, role: str) -> dict:
        coverage = row["pairs_with_sequence"] / row["denominator_pairs"] if role == "shown" else ""
        return {
            "length": row["length"],
            "group": group,
            "role": role,
            "position_1": row["position_1"],
            "position_2": row["position_2"],
            "position_3": row["position_3"],
            "path_count": row["path_count"],
            "pairs_with_sequence": row["pairs_with_sequence"] if role == "shown" else "",
            "pair_coverage": coverage,
            "path_proportion": row["path_proportion"],
            "pair_proportion": row["pair_proportion"],
            "difference_percentage_points": row["difference_percentage_points"],
            "denominator_paths": row["denominator_paths"],
            "denominator_pairs": row["denominator_pairs"],
            "sort_rule": "max(path share, pair share), then path share, then path count, then sequence name",
        }

    def other_row(rows: list[dict], group: str, label: str) -> dict:
        return {
            "length": 3,
            "group": group,
            "role": "other",
            "position_1": label,
            "position_2": "",
            "position_3": "",
            "path_count": sum(row["path_count"] for row in rows),
            "pairs_with_sequence": "",
            "pair_coverage": "",
            "path_proportion": sum(row["path_proportion"] for row in rows),
            "pair_proportion": sum(row["pair_proportion"] for row in rows),
            "difference_percentage_points": 100 * (sum(row["path_proportion"] for row in rows) - sum(row["pair_proportion"] for row in rows)),
            "denominator_paths": rows[0]["denominator_paths"],
            "denominator_pairs": rows[0]["denominator_pairs"],
            "sort_rule": "sum of sequences not displayed in this group",
        }

    plain_group_path = sum(row["path_proportion"] for row in no_shared)
    plain_group_pair = sum(row["pair_proportion"] for row in no_shared)
    shared_group_path = sum(row["path_proportion"] for row in shared)
    shared_group_pair = sum(row["pair_proportion"] for row in shared)
    if abs(plain_group_path + shared_group_path - 1) > 1e-8 or abs(plain_group_pair + shared_group_pair - 1) > 1e-8:
        raise RuntimeError("three-edge group shares do not sum to 1")
    display = [pack(row, "2-edge", "shown") for row in two]
    display += [pack(row, "no_shared_edge", "shown") for row in shown_plain]
    display.append(other_row(other_plain, "no_shared_edge", "Other without shared edges"))
    display += [pack(row, "at_least_one_shared_edge", "shown") for row in shown_shared]
    display.append(other_row(other_shared, "at_least_one_shared_edge", "Other with shared edges"))
    shown_path = sum(row["path_proportion"] for row in display if row["length"] == 3)
    shown_pair = sum(row["pair_proportion"] for row in display if row["length"] == 3)
    if abs(shown_path - 1) > 1e-8 or abs(shown_pair - 1) > 1e-8:
        raise RuntimeError("displayed three-edge rows do not sum to 1")

    largest_two = max(two, key=lambda row: abs(row["difference_percentage_points"]))
    labelled = {
        largest_two["position_1"] + "|" + largest_two["position_2"],
        "core_only|core_only|core_only",
        "|".join([shown_shared[0]["position_1"], shown_shared[0]["position_2"], shown_shared[0]["position_3"]]),
    }

    other_plain_row = next(row for row in display if row["position_1"] == "Other without shared edges")
    other_shared_row = next(row for row in display if row["position_1"] == "Other with shared edges")
    entries = [("marks-header", None), ("section", "2-edge")]
    entries.extend(("sequence", row) for row in two)
    entries.append(("group", ("No shared edge", plain_group_path, plain_group_pair)))
    entries.extend(("sequence", row) for row in shown_plain)
    entries.append(("other", other_plain_row))
    entries.append(("group", ("At least one shared edge", shared_group_path, shared_group_pair)))
    entries.extend(("sequence", row) for row in shown_shared)
    entries.append(("other", other_shared_row))
    plotted_shares = two + shown_plain + shown_shared + [other_plain_row, other_shared_row]
    xmax = 100 * max(max(row["path_proportion"], row["pair_proportion"]) for row in plotted_shares)

    fig = plt.figure(figsize=(7.15, 6.55))
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.72, 1.85], wspace=0.03)
    mark_ax = fig.add_subplot(grid[0, 0])
    dot_ax = fig.add_subplot(grid[0, 1])
    y_cursor = len(entries) - 1
    for kind, payload in entries:
        y = y_cursor
        y_cursor -= 1
        if kind == "marks-header":
            for x, text in enumerate(("Edge 1", "Edge 2", "Edge 3")):
                mark_ax.text(x, y, text, ha="center", va="center", fontsize=7.5, color=INK)
            continue
        if kind == "section":
            mark_ax.text(-0.35, y, payload, ha="left", va="center", fontsize=8, color=INK)
            continue
        if kind == "group":
            title, path_share, pair_share = payload
            mark_ax.text(-0.35, y, title, ha="left", va="center", fontsize=8, color=INK)
            dot_ax.text(xmax * 1.22 + 7.6, y, f"{100 * path_share:.1f}% paths, {100 * pair_share:.1f}% pairs", ha="right", va="center", fontsize=7.5, color=INK)
            continue
        if kind == "other":
            row = payload
            mark_ax.text(-0.35, y, row["position_1"], ha="left", va="center", fontsize=7.5, color=INK)
            path_pct = 100 * row["path_proportion"]
            pair_pct = 100 * row["pair_proportion"]
            dot_ax.plot([path_pct, pair_pct], [y, y], color=LINK, lw=0.7, zorder=2)
            dot_ax.plot(path_pct, y, marker="o", color=INK, markersize=5.2, linestyle="none", zorder=3)
            dot_ax.plot(pair_pct, y, marker="o", markerfacecolor="white", markeredgecolor=INK, markeredgewidth=0.9, markersize=5.6, linestyle="none", zorder=3)
            continue
        row = payload
        states = [row["position_1"], row["position_2"]] + ([row["position_3"]] if row["position_3"] else [])
        for slot, state in enumerate(states):
            mark_ax.scatter([slot], [y], s=36, color=colors[state], zorder=3)
        path_pct = 100 * row["path_proportion"]
        pair_pct = 100 * row["pair_proportion"]
        dot_ax.plot([path_pct, pair_pct], [y, y], color=LINK, lw=0.7, zorder=2)
        dot_ax.plot(path_pct, y, marker="o", color=INK, markersize=5.2, linestyle="none", zorder=3)
        dot_ax.plot(pair_pct, y, marker="o", markerfacecolor="white", markeredgecolor=INK, markeredgewidth=0.9, markersize=5.6, linestyle="none", zorder=3)
        key = "|".join(part for part in (row["position_1"], row["position_2"], row["position_3"]) if part)
        if key in labelled:
            if abs(path_pct - pair_pct) < 2.5:
                dot_ax.text(max(path_pct, pair_pct) + 2.6, y, f"{path_pct:.1f}% / {pair_pct:.1f}%", ha="left", va="center", fontsize=7.5, color=INK)
            else:
                dot_ax.text(path_pct, y + 0.28, f"{path_pct:.1f}%", ha="center", va="bottom", fontsize=7.5, color=INK)
                dot_ax.text(pair_pct, y - 0.28, f"{pair_pct:.1f}%", ha="center", va="top", fontsize=7.5, color=INK)
    for ax in (mark_ax, dot_ax):
        ax.set_ylim(-0.65, len(entries) - 0.35)
        ax.tick_params(left=False, labelleft=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    mark_ax.set_xlim(-0.45, 2.55)
    mark_ax.set_xticks([])
    dot_ax.set_xlim(0, xmax * 1.22 + 8)
    dot_ax.set_xlabel("Share by source sequence (%)", fontsize=8.5)
    dot_ax.grid(axis="x", color=GRID, lw=0.4)
    dot_ax.set_axisbelow(True)
    dot_ax.tick_params(labelsize=8, colors=INK)
    fig.suptitle("Source patterns along paths", fontsize=10, color=INK, y=0.985)
    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=CORE, markersize=6, label="CORE only"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=SEMMED, markersize=6, label="SemMedDB only"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=SHARED, markersize=6, label="Recorded in both resources"),
            Line2D([0], [0], marker="o", color=INK, linestyle="none", markersize=5.2, label="Share of all paths"),
            Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor=INK, linestyle="none", markersize=5.4, label="Mean share within each pair"),
        ],
        frameon=False,
        ncol=3,
        loc="lower center",
        fontsize=7.5,
        bbox_to_anchor=(0.52, 0.005),
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.15)
    save_figure(fig, "figure5_source_patterns")
    write_tsv(OUT / "data" / "figure5_source_patterns.tsv", display)
    full_rows = []
    for row in source:
        group = "2-edge" if row["length"] == 2 else ("at_least_one_shared_edge" if _has_shared(row) else "no_shared_edge")
        full_rows.append(
            {
                "length": row["length"],
                "group": group,
                "position_1": row["position_1"],
                "position_2": row["position_2"],
                "position_3": row["position_3"],
                "path_count": row["path_count"],
                "pairs_with_sequence": row["pairs_with_sequence"],
                "pair_coverage": row["pairs_with_sequence"] / row["denominator_pairs"],
                "path_proportion": row["path_proportion"],
                "pair_proportion": row["pair_proportion"],
                "difference_percentage_points": row["difference_percentage_points"],
                "denominator_paths": row["denominator_paths"],
                "denominator_pairs": row["denominator_pairs"],
            }
        )
    write_tsv(OUT / "data" / "figure5_source_patterns_all_sequences.tsv", full_rows)
    (OUT / "figure5_source_patterns_caption.md").write_text(
        "\n".join(
            [
                "# Source patterns along paths",
                "",
                "Marks read in path order. Blue is a CORE-only edge, ochre is a SemMedDB-only edge, and gray means that edge is recorded in both resources. Filled circles give every complete path one vote. Open circles first take each sequence’s share of a candidate pair’s paths, then average those shares with equal pair weights. Both weights use the same pairs: joint-reachable candidate–query pairs with no eligible direct record.",
                "",
                f"Two-edge scope: {comma(two[0]['denominator_paths'])} paths and {comma(two[0]['denominator_pairs'])} pairs. Every realized sequence is shown, and the first edge is SemMedDB-only. Three-edge scope: {comma(no_shared[0]['denominator_paths'])} paths and {comma(no_shared[0]['denominator_pairs'])} pairs. Sequences with no edge recorded in both resources contribute {plain_group_path:.1%} of paths and {plain_group_pair:.1%} of the pair-averaged share. Sequences with at least one such edge contribute {shared_group_path:.1%} and {shared_group_pair:.1%}. Within a group, rows keep these global shares. Each Other is the exact sum of the sequences from that group that are not drawn.",
                "",
                "Displayed sequences are ordered by the larger of the two shares, then by the path share, then by the path count, then by sequence name. A separated pair of labels puts the path share above the points and the pair share below them. A label written as two percentages separated by a slash is path share / pair share. Pair coverage in the full table is the number of candidate pairs with at least one path of that sequence, divided by the reachable pairs of that length. Those pair counts are not added across sequences, because one pair can carry more than one sequence.",
                "",
                "Source: `scientific_figures_20261117T0219Z/data/figure5_source_patterns.tsv`, recomputed from the joint eligible graph with the completion audit’s repeated-node rules and checked against its published path and reach totals.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    figure_direct_pairs()
    figure_paths_and_pairs()
    figure_articles_added()
    figure_article_links()
    figure_source_patterns()
    print(f"[wrote] {OUT}", flush=True)


if __name__ == "__main__":
    main()

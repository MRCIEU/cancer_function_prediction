"""Publication figures for the descriptive CORE–SemMedDB audits.

Each figure is exported on its own. Captions and the plotted counts live beside
the image files. No panel letters are added.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

OUT = Path("/home/b5ac/freddieyu.b5ac/projects/project_2/results/scientific_figures_20261117T0219Z")
COMPLETION = Path("/home/b5ac/freddieyu.b5ac/projects/project_2/results/path_evidence_audit_completion_20261116T1402Z")
CITATION = Path("/home/b5ac/freddieyu.b5ac/projects/project_2/results/citation_path_evidence_audit_20261116T1647Z")

CORE = "#426B91"
SEMMED = "#A47740"
SHARED = "#879099"
JOINT = "#685481"
INK = "#1C2429"
PALE = "#F4F6F8"
GRID = "#D5DADF"

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


def read_tsv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.svg")
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=500)
    plt.close(fig)


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=INK, labelsize=8)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def comma(value: int) -> str:
    return f"{int(value):,}"


def circle_intersection(radius_a: float, radius_b: float, distance: float) -> float:
    if distance >= radius_a + radius_b:
        return 0.0
    if distance <= abs(radius_a - radius_b):
        return math.pi * min(radius_a, radius_b) ** 2
    aa = radius_a * radius_a
    bb = radius_b * radius_b
    dd = distance * distance
    return (
        aa * math.acos((dd + aa - bb) / (2 * distance * radius_a))
        + bb * math.acos((dd + bb - aa) / (2 * distance * radius_b))
        - 0.5 * math.sqrt((-distance + radius_a + radius_b) * (distance + radius_a - radius_b) * (distance - radius_a + radius_b) * (distance + radius_a + radius_b))
    )


def solve_distance(radius_a: float, radius_b: float, target: float) -> float:
    low = abs(radius_a - radius_b)
    high = radius_a + radius_b
    for _ in range(80):
        mid = (low + high) / 2
        if circle_intersection(radius_a, radius_b, mid) > target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def lens(radius_a: float, radius_b: float, distance: float) -> list[tuple[float, float]]:
    chord = (distance * distance + radius_a * radius_a - radius_b * radius_b) / (2 * distance)
    half = math.sqrt(max(radius_a * radius_a - chord * chord, 0.0))
    core_angle = math.atan2(half, chord)
    sem_angle = math.atan2(half, chord - distance)
    points = []
    for step in range(81):
        angle = -core_angle + (2 * core_angle) * step / 80
        points.append((radius_a * math.cos(angle), radius_a * math.sin(angle)))
    # Stay on the arc that faces the other circle. A straight interpolation from the
    # positive intersection angle to its negative crosses the outer edge instead.
    sweep = 2 * math.pi - 2 * sem_angle
    for step in range(81):
        angle = sem_angle + sweep * step / 80
        points.append((distance + radius_b * math.cos(angle), radius_b * math.sin(angle)))
    return points


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
    scale = 1.22 / radius_core
    rc = radius_core * scale
    rs = radius_semmed * scale
    dd = distance * scale
    fig, ax = plt.subplots(figsize=(4.9, 3.55))
    fig.patch.set_facecolor("white")
    ax.set_aspect("equal")
    ax.add_patch(Circle((0, 0), rc, facecolor=CORE, edgecolor="none", zorder=1))
    ax.add_patch(Circle((dd, 0), rs, facecolor=SEMMED, edgecolor="none", zorder=2))
    ax.add_patch(Polygon(lens(rc, rs, dd), closed=True, facecolor=SHARED, edgecolor="none", zorder=3))
    chord = (dd * dd + rc * rc - rs * rs) / (2 * dd)
    ax.text(-rc * 0.46, 0.02, comma(core_only), ha="center", va="center", color="white", fontsize=9, zorder=4)
    ax.text(chord, 0.0, comma(shared), ha="center", va="center", color=INK, fontsize=8, zorder=4)
    ax.text(dd + rs * 0.42, 0.02, comma(semmed_only), ha="center", va="center", color="white", fontsize=9, zorder=4)
    ax.text(-rc * 0.15, rc + 0.10, "CORE", ha="center", va="bottom", color=CORE, fontsize=9)
    ax.text(dd + rs * 0.15, rs + 0.10, "SemMedDB", ha="center", va="bottom", color=SEMMED, fontsize=9)
    ax.set_xlim(-rc - 0.18, dd + rs + 0.18)
    ax.set_ylim(-max(rc, rs) - 0.16, max(rc, rs) + 0.38)
    ax.axis("off")
    ax.set_title("Direct gene–cancer pairs", fontsize=10, color=INK, pad=2)
    save_figure(fig, "figure1_direct_pairs")
    write_tsv(
        OUT / "data" / "figure1_direct_pairs.tsv",
        [
            {"region": "CORE only", "pairs": core_only, "circle": "CORE", "circle_pairs": core_total},
            {"region": "shared", "pairs": shared, "circle": "intersection", "circle_pairs": shared},
            {"region": "SemMedDB only", "pairs": semmed_only, "circle": "SemMedDB", "circle_pairs": semmed_total},
            {"region": "union", "pairs": union, "circle": "", "circle_pairs": union},
        ],
    )
    (OUT / "figure1_direct_pairs_caption.md").write_text(
        "\n".join(
            [
                "# Direct gene–cancer pairs",
                "",
                f"Circle areas are the eligible direct gene–cancer pairs inside the fixed universe of 13,571 resolved human candidate genes and 1,665 cancer queries. CORE contains {comma(core_total)} pairs and SemMedDB contains {comma(semmed_total)} pairs. The overlap area is the {comma(shared)} pairs recorded in both resources. The three labelled regions are CORE only {comma(core_only)} ({core_only / union:.1%} of the union), shared {comma(shared)} ({shared / union:.1%}), and SemMedDB only {comma(semmed_only)} ({semmed_only / union:.1%}). Their union is {comma(union)}.",
                "",
                "The circle distance was solved from the standard two-circle intersection formula so the lens area equals the shared count. Areas use this fixed candidate-by-query universe. They do not use the wider SemMedDB gene–cancer edge count, which also includes cancers that are not queries.",
                "",
                "Source: `path_evidence_audit_completion_20261116T1402Z/analysis/plot_resource_additions.tsv` and `shared_pair_gain_summary.tsv`.",
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
    three_joint_paths = sum(int(row["complete_paths"]) for row in path_rows if row["sequence_group"] == "requires_both_resources")
    two_joint_paths = sum(int(row["complete_paths"]) for row in two_rows if row["sequence_group"] == "requires_both_resources")
    points = [
        ("2-edge", "Complete paths", two_joint_paths, two_paths),
        ("2-edge", "Reachable pairs without a direct record", dep[("two_no_direct", "joint_only_reach")], dep[("two_no_direct", "joint_reached")]),
        ("3-edge", "Complete paths", three_joint_paths, three_paths),
        ("3-edge", "Reachable pairs without a direct record", dep[("three_no_direct", "joint_only_reach")], dep[("three_no_direct", "joint_reached")]),
    ]
    labels = {
        ("3-edge", "Complete paths"): "48.8%",
        ("3-edge", "Reachable pairs without a direct record"): "4.9%",
    }
    for length, unit, numerator, denominator in points:
        shown = f"{100 * numerator / denominator:.1f}%"
        if (length, unit) in labels and shown != labels[(length, unit)]:
            raise RuntimeError(f"{length} {unit} rounds to {shown}")
    fig, ax = plt.subplots(figsize=(5.5, 2.85))
    fig.patch.set_facecolor("white")
    style_axis(ax)
    y_base = {"2-edge": 1, "3-edge": 0}
    y_shift = {"Complete paths": 0.13, "Reachable pairs without a direct record": -0.13}
    markers = {"Complete paths": "o", "Reachable pairs without a direct record": "s"}
    plotted = []
    for length, unit, numerator, denominator in points:
        percent = 100 * numerator / denominator
        y = y_base[length] + y_shift[unit]
        ax.plot([percent], [y], marker=markers[unit], color=JOINT, markersize=7, linestyle="none", zorder=3)
        plotted.append((length, unit, y, percent))
    for length in ("2-edge", "3-edge"):
        pair = [item for item in plotted if item[0] == length]
        ax.plot([pair[0][3], pair[1][3]], [pair[0][2], pair[1][2]], color="#C9C2D4", lw=0.8, zorder=2)
    offsets = {
        ("2-edge", "Complete paths"): (0.6, 0.16),
        ("2-edge", "Reachable pairs without a direct record"): (0.6, -0.20),
        ("3-edge", "Complete paths"): (0.7, 0.16),
        ("3-edge", "Reachable pairs without a direct record"): (0.7, -0.20),
    }
    for length, unit, y, percent in plotted:
        dx, dy = offsets[(length, unit)]
        weight = "bold" if length == "3-edge" else "regular"
        ax.text(percent + dx, y + dy, f"{percent:.1f}%", color=JOINT, fontsize=8, fontweight=weight, va="center")
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["2-edge", "3-edge"])
    ax.set_xlabel("Requires both sources (%)")
    ax.set_xlim(0, 62)
    ax.set_ylim(-0.55, 1.55)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.plot([], [], marker="o", color=JOINT, linestyle="none", label="Complete paths")
    ax.plot([], [], marker="s", color=JOINT, linestyle="none", label="Reachable pairs without a direct record")
    ax.legend(frameon=False, loc="lower right", fontsize=7.5, handletextpad=0.4)
    ax.set_title("Paths and reachable pairs", fontsize=10, color=INK, pad=8)
    save_figure(fig, "figure2_paths_and_pairs")
    composition = []
    for row in two_rows:
        composition.append({"length": 2, "unit": "complete_paths", "class": row["sequence_group"], "count": int(row["complete_paths"]), "denominator": two_paths})
    grouped = {}
    for row in path_rows:
        grouped[row["sequence_group"]] = grouped.get(row["sequence_group"], 0) + int(row["complete_paths"])
    for name, count in grouped.items():
        composition.append({"length": 3, "unit": "complete_paths", "class": name, "count": count, "denominator": three_paths})
    for population, length in (("two_no_direct", 2), ("three_no_direct", 3)):
        denominator = dep[(population, "joint_reached")]
        for category in ("core_only_reach", "semmed_only_reach", "either_resource_reach", "joint_only_reach"):
            composition.append({"length": length, "unit": "reachable_pairs_without_direct_record", "class": category, "count": dep[(population, category)], "denominator": denominator})
    write_tsv(OUT / "data" / "figure2_paths_and_pairs.tsv", composition)
    (OUT / "figure2_paths_and_pairs_caption.md").write_text(
        "\n".join(
            [
                "# Paths and reachable pairs",
                "",
                "Both series use a linear percentage axis starting at zero. Circles count complete paths. Squares count deduplicated candidate–query pairs that the joint graph can reach and that have no eligible direct record.",
                "",
                f"Two-edge paths: requires both sources {comma(two_joint_paths)} / {comma(two_paths)} ({two_joint_paths / two_paths:.1%}). There are no CORE-only two-edge paths because the gene–gene edge is SemMedDB-only. SemMedDB-only paths are {comma(two_paths - two_joint_paths)}.",
                f"Three-edge paths: only CORE {comma(grouped['only_core'])} ({grouped['only_core'] / three_paths:.1%}), only SemMedDB {comma(grouped['only_semmed'])} ({grouped['only_semmed'] / three_paths:.1%}), either resource {comma(grouped['either_resource'])} ({grouped['either_resource'] / three_paths:.1%}), requires both {comma(three_joint_paths)} / {comma(three_paths)} ({three_joint_paths / three_paths:.1%}).",
                f"Two-edge pairs without a direct record: joint reached {comma(dep[('two_no_direct', 'joint_reached')])}; SemMedDB alone {comma(dep[('two_no_direct', 'semmed_only_reach')])}; joint only {comma(dep[('two_no_direct', 'joint_only_reach')])} ({dep[('two_no_direct', 'joint_only_reach')] / dep[('two_no_direct', 'joint_reached')]:.1%}). CORE-only and either-resource reaches are 0.",
                f"Three-edge pairs without a direct record: joint reached {comma(dep[('three_no_direct', 'joint_reached')])}; CORE alone {comma(dep[('three_no_direct', 'core_only_reach')])}; SemMedDB alone {comma(dep[('three_no_direct', 'semmed_only_reach')])}; either resource {comma(dep[('three_no_direct', 'either_resource_reach')])}; joint only {comma(dep[('three_no_direct', 'joint_only_reach')])} ({dep[('three_no_direct', 'joint_only_reach')] / dep[('three_no_direct', 'joint_reached')]:.1%}).",
                "",
                "A path requires both sources when its edges include a CORE-only edge and a SemMedDB-only edge. A pair is joint-only when none of its complete paths can be walked in CORE alone or in SemMedDB alone. These are descriptive counts, not a sample.",
                "",
                "Source: `path_evidence_audit_completion_20261116T1402Z/analysis/source_sequences.tsv`, `two_edge_source_sequences.tsv`, and `candidate_dependence.tsv`.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def figure_articles_added() -> None:
    gain = {row["measure"]: int(row["value"]) for row in read_tsv(COMPLETION / "analysis" / "shared_pair_gain_summary.tsv")}
    cells = {
        (0, 0): ("Both", gain["both_add_pmids"]),
        (0, 1): ("CORE only", gain["only_core_adds_pmids"]),
        (1, 0): ("SemMedDB only", gain["only_semmed_adds_pmids"]),
        (1, 1): ("Neither", gain["neither_adds_pmids"]),
    }
    total = gain["pairs"]
    if sum(count for _name, count in cells.values()) != total:
        raise RuntimeError("shared-pair gain cells do not sum to the shared pairs")
    emphasis = f"{100 * gain['both_add_pmids'] / total:.1f}%"
    if emphasis != "58.8%":
        raise RuntimeError(emphasis)
    fig, ax = plt.subplots(figsize=(4.7, 3.9))
    fig.patch.set_facecolor("white")
    for (row, column), (_name, count) in cells.items():
        percent = 100 * count / total
        accent = (row, column) == (0, 0)
        ax.add_patch(
            plt.Rectangle((column, 1 - row), 1, 1, facecolor=CORE if accent else PALE, edgecolor="white", linewidth=3)
        )
        ax.text(column + 0.5, 1.55 - row, f"{percent:.1f}%", ha="center", va="center", color="white" if accent else INK, fontsize=14 if accent else 11)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Yes", "No"])
    ax.set_yticks([1.5, 0.5])
    ax.set_yticklabels(["Yes", "No"])
    ax.tick_params(length=0, labelsize=8.5)
    ax.set_xlabel("SemMedDB adds an exclusive PMID", fontsize=8.5, labelpad=6)
    ax.set_ylabel("CORE adds an exclusive PMID", fontsize=8.5, labelpad=6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Articles added on shared pairs", fontsize=10, color=INK, pad=8)
    save_figure(fig, "figure3_articles_added")
    write_tsv(
        OUT / "data" / "figure3_articles_added.tsv",
        [
            {"core_exclusive_pmid": "yes", "semmed_exclusive_pmid": "yes", "pairs": gain["both_add_pmids"], "percent_of_shared_pairs": gain["both_add_pmids"] / total, "denominator_pairs": total},
            {"core_exclusive_pmid": "yes", "semmed_exclusive_pmid": "no", "pairs": gain["only_core_adds_pmids"], "percent_of_shared_pairs": gain["only_core_adds_pmids"] / total, "denominator_pairs": total},
            {"core_exclusive_pmid": "no", "semmed_exclusive_pmid": "yes", "pairs": gain["only_semmed_adds_pmids"], "percent_of_shared_pairs": gain["only_semmed_adds_pmids"] / total, "denominator_pairs": total},
            {"core_exclusive_pmid": "no", "semmed_exclusive_pmid": "no", "pairs": gain["neither_adds_pmids"], "percent_of_shared_pairs": gain["neither_adds_pmids"] / total, "denominator_pairs": total},
            {"core_exclusive_pmid": "", "semmed_exclusive_pmid": "", "pairs": gain["core_added_links"], "percent_of_shared_pairs": "", "denominator_pairs": "core_added_pmid_links"},
            {"core_exclusive_pmid": "", "semmed_exclusive_pmid": "", "pairs": gain["semmed_added_links"], "percent_of_shared_pairs": "", "denominator_pairs": "semmed_added_pmid_links"},
            {"core_exclusive_pmid": "", "semmed_exclusive_pmid": "", "pairs": gain["shared_links"], "percent_of_shared_pairs": "", "denominator_pairs": "shared_pmid_links"},
        ],
    )
    (OUT / "figure3_articles_added_caption.md").write_text(
        "\n".join(
            [
                "# Articles added on shared pairs",
                "",
                f"The denominator is the {comma(total)} gene–cancer pairs recorded by both resources. A cell counts pairs, not articles. Both resources add at least one exclusive PMID on {comma(gain['both_add_pmids'])} pairs ({gain['both_add_pmids'] / total:.1%}). CORE alone adds an exclusive PMID on {comma(gain['only_core_adds_pmids'])} pairs, SemMedDB alone on {comma(gain['only_semmed_adds_pmids'])} pairs, and neither resource adds an exclusive PMID on {comma(gain['neither_adds_pmids'])} pairs.",
                "",
                f"Pair–PMID links are a different counting unit. CORE contributes {comma(gain['core_added_links'])} exclusive links, SemMedDB contributes {comma(gain['semmed_added_links'])} exclusive links, and {comma(gain['shared_links'])} links are shared. The corresponding distinct PMIDs are {comma(gain['distinct_core_added_pmids'])}, {comma(gain['distinct_semmed_added_pmids'])}, and {comma(gain['distinct_shared_pmids'])}.",
                "",
                "Source: `path_evidence_audit_completion_20261116T1402Z/analysis/shared_pair_gain_summary.tsv`.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _citation_lookup() -> dict[tuple[str, str, str], tuple[int, int]]:
    rows = read_tsv(CITATION / "analysis" / "adjacent_citation_summary.tsv")
    return {(row["template"], row["positions"], row["measure"]): (int(row["distinct_count"]), int(row["path_occurrence_count"])) for row in rows}


def _overlap_same_article() -> dict[tuple[str, str, str], tuple[int, int, int]]:
    """Return any_overlap, both_known, all_pairs for union comparisons."""
    found = {}
    for row in read_tsv(COMPLETION / "analysis" / "adjacent_overlap_summary.tsv"):
        if row["comparison"] != "union":
            continue
        found[(row["template"], row["positions"], row["weighting"])] = (
            int(row["any_overlap"]),
            int(row["both_known"]),
            int(row["denominator_pairs"]),
        )
    return found


def _label_rate(ax: plt.Axes, value: float, y: float, text: str, xmax: float, weight: str) -> None:
    if value < xmax * 0.18:
        ax.text(value + xmax * 0.025, y, text, ha="left", va="center", fontsize=7.5, color=INK, fontweight=weight, clip_on=False)
        return
    ax.text(value, y + (0.22 if y >= round(y) else -0.22), text, ha="center", va="bottom" if y >= round(y) else "top", fontsize=7.5, color=INK, fontweight=weight)


def figure_article_links() -> None:
    citation = _citation_lookup()
    overlap = _overlap_same_article()
    positions = [
        ("2-edge 1–2", "one_intermediate_path", "positions_1_2"),
        ("3-edge 1–2", "two_intermediate_path", "positions_1_2"),
        ("3-edge 2–3", "two_intermediate_path", "positions_2_3"),
    ]
    relations = [
        ("Same article", "same_article"),
        ("Direct citation", "collapsed_pairs"),
        ("Co-citation", "cocite_pairs"),
        ("Shared reference", "ref_pairs"),
    ]
    rows = []
    for position, template, slot in positions:
        for relation, measure in relations:
            if measure == "same_article":
                distinct_n, distinct_d, _all_pairs = overlap[(template, slot, "distinct_count")]
                weighted_n, weighted_d, _all_w = overlap[(template, slot, "path_occurrence_count")]
                definition = "union PMID intersection among edges with a PMID on both sides"
            else:
                distinct_n, weighted_n = citation[(template, slot, measure)]
                distinct_d, weighted_d = citation[(template, slot, "both")]
                definition = "qualifying PMID sets on both sides; direct citation is either direction"
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
                    "definition": definition,
                }
            )
    same = {(row["position"], row["relation"]): row for row in rows}
    if f"{same[('3-edge 2–3', 'Same article')]['distinct_percent']:.2f}%" != "2.50%":
        raise RuntimeError("same-article distinct label drifted")
    if f"{same[('3-edge 2–3', 'Same article')]['weighted_percent']:.2f}%" != "5.54%":
        raise RuntimeError("same-article weighted label drifted")
    fig, axes = plt.subplots(4, 1, figsize=(7.05, 6.15), sharey=True)
    fig.patch.set_facecolor("white")
    fig.suptitle("Article links along paths", fontsize=10, color=INK, y=0.985)
    y_pos = {"2-edge 1–2": 2, "3-edge 1–2": 1, "3-edge 2–3": 0}
    for ax, (relation, _measure) in zip(axes, relations):
        style_axis(ax)
        subset = [row for row in rows if row["relation"] == relation]
        xmax = max(max(row["distinct_percent"], row["weighted_percent"]) for row in subset)
        for row in subset:
            y = y_pos[row["position"]]
            distinct_y = y + 0.13
            weighted_y = y - 0.13
            ax.plot([row["distinct_percent"], row["weighted_percent"]], [distinct_y, weighted_y], color="#C5CDD4", lw=0.9, zorder=2)
            ax.plot(row["distinct_percent"], distinct_y, marker="o", markerfacecolor="white", markeredgecolor=INK, markeredgewidth=1.0, markersize=6.2, linestyle="none", zorder=3)
            ax.plot(row["weighted_percent"], weighted_y, marker="o", color=INK, markersize=5.8, linestyle="none", zorder=3)
            distinct_text = f"{row['distinct_percent']:.2f}%" if row["distinct_percent"] >= 1 else f"{row['distinct_percent']:.3f}%"
            weighted_text = f"{row['weighted_percent']:.2f}%" if row["weighted_percent"] >= 1 else f"{row['weighted_percent']:.3f}%"
            callout = relation == "Same article" and row["position"] == "3-edge 2–3"
            weight = "bold" if callout else "regular"
            gap = abs(row["distinct_percent"] - row["weighted_percent"])
            if distinct_text == weighted_text or gap < 1.0:
                label = distinct_text if distinct_text == weighted_text else f"{distinct_text} / {weighted_text}"
                ax.text(
                    max(row["distinct_percent"], row["weighted_percent"]) + max(0.35, xmax * 0.02),
                    y,
                    label,
                    ha="left",
                    va="center",
                    fontsize=7.5,
                    color=INK,
                    fontweight=weight,
                    clip_on=False,
                )
            else:
                ax.text(row["distinct_percent"], distinct_y + 0.22, distinct_text, ha="center", va="bottom", fontsize=7.5, color=INK, fontweight=weight, clip_on=False)
                ax.text(row["weighted_percent"], weighted_y - 0.22, weighted_text, ha="center", va="top", fontsize=7.5, color=INK, fontweight=weight, clip_on=False)
        ax.set_xlim(0, xmax * 1.28 + 4)
        ax.set_ylim(-0.78, 2.72)
        ax.set_yticks([2, 1, 0])
        ax.set_yticklabels(["2-edge 1–2", "3-edge 1–2", "3-edge 2–3"])
        ax.set_ylabel(relation, fontsize=8.5, color=INK)
        ax.grid(axis="x", color=GRID, lw=0.45)
        ax.set_axisbelow(True)
        if relation != "Shared reference":
            ax.tick_params(labelbottom=True)
            ax.set_xlabel("")
    axes[-1].set_xlabel("Share of observable adjacent edge pairs (%)", fontsize=8.5)
    fig.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor=INK, linestyle="none", label="Distinct adjacent pairs"),
            plt.Line2D([0], [0], marker="o", color=INK, linestyle="none", label="Weighted by complete paths"),
        ],
        frameon=False,
        ncol=2,
        loc="upper right",
        fontsize=7.5,
        bbox_to_anchor=(0.98, 0.965),
    )
    fig.subplots_adjust(left=0.17, right=0.98, top=0.90, bottom=0.08, hspace=0.62)
    save_figure(fig, "figure4_article_links")
    write_tsv(OUT / "data" / "figure4_article_links.tsv", rows)
    coverage = []
    for position, template, slot in positions:
        for measure in (
            "ltr_pairs",
            "rtl_pairs",
            "ambiguous_pairs",
            "collapsed_pairs",
            "cocite_pairs",
            "cocite_witnesses",
            "cocite_external",
            "cocite_index_complete",
            "cocite_index_partial",
            "cocite_index_absent",
            "ref_pairs",
            "ref_witnesses",
            "ref_external",
            "ref_index_complete",
            "ref_index_partial",
            "ref_index_absent",
            "ltr_index_complete",
            "ltr_index_partial",
            "ltr_index_absent",
            "rtl_index_complete",
            "rtl_index_partial",
            "rtl_index_absent",
        ):
            key = (template, slot, measure)
            if key not in citation:
                continue
            distinct_n, weighted_n = citation[key]
            coverage.append({"position": position, "measure": measure, "distinct_count": distinct_n, "path_occurrence_count": weighted_n})
    write_tsv(OUT / "data" / "figure4_direction_and_coverage.tsv", coverage)
    (OUT / "figure4_article_links_caption.md").write_text(
        "\n".join(
            [
                "# Article links along paths",
                "",
                "Each row is one ordered adjacent-edge position. Open circles give equal weight to each distinct adjacent edge pair. Filled circles weight a pair by the number of complete paths in which it occurs. A line joins only those two weights. Each relation has its own linear axis starting at zero.",
                "",
                "Same article uses the union of CORE and SemMedDB PMIDs on the two edges. The rate denominator is the count of adjacent pairs with at least one PMID on both sides (`both_known` in the overlap table). For the final edge of a three-edge path this is 304,413 / 12,178,132 distinct pairs (2.50%) and 702,749,188 / 12,690,017,595 path occurrences (5.54%). The citation audit’s qualifying-PMID overlap is slightly smaller and is not the number drawn for this relation.",
                "",
                "Direct citation, co-citation, and shared reference use the citation audit. A direct citation is present when either direction is observed, including a link whose articles sit in both support sets. Co-citation and shared reference follow that audit’s witness rules and keep third-party articles. Their denominators are adjacent pairs whose two edges both have a qualifying PMID. Positive witnesses are retained when metadata are missing. A zero is a confirmed absence only when the relevant citation index is complete; partial coverage is not plotted as zero.",
                "",
                "Direction counts, external-witness counts, and complete, partial, or absent index coverage are in `data/figure4_direction_and_coverage.tsv`. They are not drawn.",
                "",
                "Sources: `path_evidence_audit_completion_20261116T1402Z/analysis/adjacent_overlap_summary.tsv` for same-article union rates, and `citation_path_evidence_audit_20261116T1647Z/analysis/adjacent_citation_summary.tsv` for the other three relations. Snapshot: iCite Open Citation Collection, August 2024, DOI 10.35092/yhjc26955625.v1.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _percent_text(value: float) -> str:
    if value >= 10:
        return f"{value:.1f}%"
    if value >= 1:
        return f"{value:.2f}%"
    return f"{value:.3f}%"


def figure_source_patterns(path: Path) -> None:
    rows = [row for row in read_tsv(path) if row["population"] == "no_direct"]
    for row in rows:
        for key in ("path_count", "pairs_with_sequence", "denominator_paths", "denominator_pairs"):
            row[key] = int(float(row[key]))
        row["path_proportion"] = float(row["path_proportion"])
        row["pair_proportion"] = float(row["pair_proportion"])
        row["difference_percentage_points"] = float(row["difference_percentage_points"])
    selected = []
    for length in (2, 3):
        block = [row for row in rows if int(row["length"]) == length and (row["path_count"] or row["pairs_with_sequence"])]
        path_total = sum(row["path_proportion"] for row in block)
        pair_total = sum(row["pair_proportion"] for row in block)
        if abs(path_total - 1) > 1e-8 or abs(pair_total - 1) > 1e-8:
            raise RuntimeError(f"{length}-edge weights sum to {path_total}, {pair_total}")
        if length == 2:
            chosen = sorted(block, key=lambda row: max(row["path_proportion"], row["pair_proportion"]), reverse=True)
            for row in chosen:
                row["display"] = "shown"
            selected.extend(chosen)
            continue
        ranked = sorted(block, key=lambda row: max(row["path_proportion"], row["pair_proportion"]), reverse=True)
        head = ranked[:6]
        tail = ranked[6:]
        for row in head:
            row["display"] = "shown"
        if tail:
            other = {
                "length": 3,
                "position_1": "other",
                "position_2": "",
                "position_3": "",
                "path_proportion": sum(row["path_proportion"] for row in tail),
                "pair_proportion": sum(row["pair_proportion"] for row in tail),
                "path_count": sum(row["path_count"] for row in tail),
                "pairs_with_sequence": "",
                "difference_percentage_points": 100 * (sum(row["path_proportion"] for row in tail) - sum(row["pair_proportion"] for row in tail)),
                "display": "other",
                "denominator_paths": head[0]["denominator_paths"],
                "denominator_pairs": head[0]["denominator_pairs"],
            }
            selected.extend(head + [other])
        else:
            selected.extend(head)
        for row in tail:
            row["display"] = "grouped_as_other"
    for row in rows:
        row.setdefault("display", "zero")
    colors = {"core_only": CORE, "semmed_only": SEMMED, "both_sources": SHARED}
    fig = plt.figure(figsize=(7.15, 6.35))
    fig.patch.set_facecolor("white")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.05, 2.35], height_ratios=[3, 7], wspace=0.04, hspace=0.42)
    panels = []
    for index, length in enumerate((2, 3)):
        block = [row for row in selected if int(row["length"]) == length]
        mark_ax = fig.add_subplot(grid[index, 0])
        dot_ax = fig.add_subplot(grid[index, 1])
        panels.append((mark_ax, dot_ax, block))
        y_values = list(range(len(block) - 1, -1, -1))
        for y, row in zip(y_values, block):
            states = [row["position_1"], row["position_2"]] + ([row["position_3"]] if length == 3 else [])
            if row["position_1"] == "other":
                mark_ax.text(1.0, y, "Other", ha="center", va="center", fontsize=8, color=INK)
            else:
                for slot, state in enumerate(states):
                    mark_ax.scatter([slot], [y], s=42, color=colors[state], zorder=3)
            path_pct = 100 * row["path_proportion"]
            pair_pct = 100 * row["pair_proportion"]
            dot_ax.plot([path_pct, pair_pct], [y, y], color="#C5CDD4", lw=0.8, zorder=2)
            dot_ax.plot(path_pct, y, marker="o", color=INK, markersize=5.5, linestyle="none", zorder=3)
            dot_ax.plot(pair_pct, y, marker="o", markerfacecolor="white", markeredgecolor=INK, markeredgewidth=1.0, markersize=5.8, linestyle="none", zorder=3)
        xmax = max(max(100 * row["path_proportion"], 100 * row["pair_proportion"]) for row in block)
        sequences = [row for row in block if row["position_1"] != "other"]
        notable = sorted(sequences, key=lambda row: abs(row["difference_percentage_points"]), reverse=True)[:1]
        other_rows = [row for row in block if row["position_1"] == "other"]
        if other_rows and abs(other_rows[0]["difference_percentage_points"]) > abs(notable[0]["difference_percentage_points"]):
            notable.append(other_rows[0])
        for row in notable:
            y = y_values[block.index(row)]
            path_pct = 100 * row["path_proportion"]
            pair_pct = 100 * row["pair_proportion"]
            gap = row["difference_percentage_points"]
            anchor = max(path_pct, pair_pct)
            dot_ax.text(path_pct, y + 0.30, _percent_text(path_pct), ha="center", va="bottom", fontsize=7.5, color=INK)
            dot_ax.text(pair_pct, y - 0.30, _percent_text(pair_pct), ha="center", va="top", fontsize=7.5, color=INK)
            dot_ax.text(anchor + max(7.5, xmax * 0.06), y, f"{gap:+.1f} pp", va="center", ha="left", fontsize=7.5, color=JOINT)
        for ax in (mark_ax, dot_ax):
            ax.set_ylim(-0.7, len(block) - 0.3)
            ax.tick_params(left=False, labelleft=False)
            for spine in ax.spines.values():
                spine.set_visible(False)
        mark_ax.set_xlim(-0.4, 2.4 if length == 3 else 1.6)
        mark_ax.set_xticks([])
        mark_ax.set_ylabel("2-edge" if length == 2 else "3-edge", fontsize=8.5, color=INK)
        dot_ax.set_xlim(0, xmax * 1.38 + 2)
        dot_ax.set_xlabel("Share of reachable pairs without a direct record (%)" if length == 3 else "", fontsize=8.5)
        dot_ax.grid(axis="x", color=GRID, lw=0.45)
        dot_ax.set_axisbelow(True)
        dot_ax.tick_params(labelsize=8, colors=INK)
    fig.suptitle("Source patterns along paths", fontsize=10, color=INK, y=0.985)
    mark_handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=CORE, markersize=6, label="CORE only"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=SEMMED, markersize=6, label="SemMedDB only"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=SHARED, markersize=6, label="Both"),
    ]
    weight_handles = [
        plt.Line2D([0], [0], marker="o", color=INK, linestyle="none", markersize=5.5, label="Paths equally weighted"),
        plt.Line2D([0], [0], marker="o", markerfacecolor="white", markeredgecolor=INK, linestyle="none", markersize=5.5, label="Pairs equally weighted"),
    ]
    fig.legend(handles=mark_handles + weight_handles, frameon=False, ncol=5, loc="lower center", fontsize=7.5, bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.93, bottom=0.14)
    save_figure(fig, "figure5_source_patterns")
    write_tsv(OUT / "data" / "figure5_source_patterns.tsv", rows)
    two = next(row for row in selected if int(row["length"]) == 2)
    three = max((row for row in selected if int(row["length"]) == 3), key=lambda row: abs(row["difference_percentage_points"]))
    (OUT / "figure5_source_patterns_caption.md").write_text(
        "\n".join(
            [
                "# Source patterns along paths",
                "",
                "Each row is a complete source sequence. Marks read in path order. Blue is a CORE-only edge, ochre is a SemMedDB-only edge, and gray means the same edge has an allowed record in both resources. Both occupies one position.",
                "",
                "The scope is separate for each length: candidate–query pairs that are reachable in the joint graph and have no eligible direct record. Filled circles give every complete path one vote. Open circles first convert a pair’s paths into sequence shares, then average those shares across pairs. The two weights use the same pairs. Within each length, each weight sums to 100% over the full sequence list, including sequences grouped as Other.",
                "",
                f"Two-edge denominator: {comma(two['denominator_paths'])} paths and {comma(two['denominator_pairs'])} pairs. The first edge is SemMedDB-only for every two-edge path. Three-edge denominator: {comma(three['denominator_paths'])} paths and {comma(three['denominator_pairs'])} pairs. The figure shows every realized two-edge sequence and the six three-edge sequences with the largest of the two proportions. The remaining three-edge sequences are Other. The full list is in `data/figure5_source_patterns.tsv`.",
                "",
                "The labelled difference is path share minus pair share, in percentage points. Adjacent-position matrices were not used to build three-edge sequences. Repeated genes and repeated cancers are excluded, using the same grouped product as the completion audit. Validation required the unmasked sequence totals to match `source_sequences.tsv` and `two_edge_source_sequences.tsv`, and the masked pair-reach counts to match `candidate_dependence.tsv`.",
                "",
                "Sources: edge profiles from `path_evidence_audit_20261116T1310Z`, candidate and query membership from `joint_graph_and_evidence_audit_20261116T0404Z`, and the published totals from `path_evidence_audit_completion_20261116T1402Z`.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def figure_article_combinations(path: Path) -> None:
    rows = read_tsv(path)
    for row in rows:
        for key in ("pattern", "distinct_count", "weighted_count", "decidable_distinct", "decidable_weighted"):
            row[key] = int(row[key])
    positions = ["2-edge 1–2", "3-edge 1–2", "3-edge 2–3"]
    by_pattern: dict[int, dict[str, dict]] = {}
    for row in rows:
        by_pattern.setdefault(int(row["pattern"]), {})[row["position"]] = row
    ranking = []
    for pattern, position_rows in by_pattern.items():
        if pattern == 0:
            continue
        ranking.append((sum(item["distinct_count"] for item in position_rows.values()), pattern))
    ranking.sort(reverse=True)
    head = [pattern for _total, pattern in ranking[:6]]
    tail = [pattern for _total, pattern in ranking[6:]]
    relation_bits = (("Same article", 1), ("Direct citation", 2), ("Co-citation", 4), ("Shared reference", 8))
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 4.9), gridspec_kw={"width_ratios": [1.15, 2.3], "wspace": 0.06})
    fig.patch.set_facecolor("white")
    mark_ax, dot_ax = axes
    labels = head + (["other"] if tail else []) + [0]
    y_values = list(range(len(labels) - 1, -1, -1))
    markers = {"2-edge 1–2": "o", "3-edge 1–2": "s", "3-edge 2–3": "D"}
    shown_max = 0.0
    for y, pattern in zip(y_values, labels):
        if pattern == "other":
            mark_ax.text(1.5, y, "Other", ha="center", va="center", fontsize=8, color=INK)
            shares = []
            for position in positions:
                distinct = sum(by_pattern[item][position]["distinct_count"] for item in tail)
                denominator = by_pattern[tail[0]][position]["decidable_distinct"]
                shares.append(100 * distinct / denominator if denominator else 0)
        elif pattern == 0:
            for bit, (_name, flag) in enumerate(relation_bits):
                mark_ax.scatter([bit], [y], s=28, facecolor="white", edgecolor="#B7BFC6", linewidth=0.8, zorder=3)
            shares = []
            for position in positions:
                item = by_pattern[0][position]
                shares.append(100 * item["distinct_count"] / item["decidable_distinct"] if item["decidable_distinct"] else 0)
        else:
            for bit, (_name, flag) in enumerate(relation_bits):
                present = bool(pattern & flag)
                mark_ax.scatter([bit], [y], s=32, color=INK if present else "white", edgecolor=INK if present else "#B7BFC6", linewidth=0.8, zorder=3)
            shares = []
            for position in positions:
                item = by_pattern[pattern][position]
                shares.append(100 * item["distinct_count"] / item["decidable_distinct"] if item["decidable_distinct"] else 0)
        for position, share in zip(positions, shares):
            shown_max = max(shown_max, share)
            dot_ax.plot(share, y, marker=markers[position], color=INK, markersize=5.4, linestyle="none")
    mark_ax.set_xlim(-0.5, 3.5)
    mark_ax.set_xticks(range(4))
    mark_ax.set_xticklabels(["Same\narticle", "Direct\ncitation", "Co-citation", "Shared\nreference"], fontsize=7.5)
    mark_ax.set_ylim(-0.6, len(labels) - 0.4)
    dot_ax.set_ylim(-0.6, len(labels) - 0.4)
    mark_ax.set_yticks([])
    dot_ax.set_yticks([])
    dot_ax.set_xlim(0, max(shown_max * 1.18, 1))
    dot_ax.set_xlabel("Decidable adjacent pairs (%)", fontsize=8.5)
    for ax in axes:
        ax.tick_params(colors=INK, labelsize=7.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)
    mark_ax.spines["bottom"].set_visible(False)
    mark_ax.tick_params(bottom=False)
    dot_ax.grid(axis="x", color=GRID, lw=0.45)
    dot_ax.set_axisbelow(True)
    handles = [plt.Line2D([0], [0], marker=markers[position], color=INK, linestyle="none", label=position) for position in positions]
    dot_ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="best")
    fig.suptitle("Article-link combinations", fontsize=10, color=INK, y=0.98)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.16)
    save_figure(fig, "figure6_article_combinations")
    (OUT / "figure6_article_combinations_caption.md").write_text(
        "\n".join(
            [
                "# Article-link combinations",
                "",
                "Each row is one joint pattern of four article relations on the same ordered adjacent edge pair. A filled mark means that relation occurs. Relations in one pattern can be produced by different article pairs. The percentage denominator is the adjacent pairs for which all four relations are decidable. `0000` means all four were observable and none was recorded. Pairs with incomplete coverage are excluded from these percentages and reported in `data/figure6_article_combinations.tsv`.",
                "",
                "The three positions use the same pattern order. The order is the six non-empty patterns with the largest total distinct-pair count, then Other, then the decidable all-absent pattern. The main marks give equal weight to each distinct adjacent pair. Path-weighted counts are in the same table.",
                "",
                "Same article is a nonempty intersection of the two edges’ qualifying PMIDs. Direct citation is either direction. Co-citation and shared reference follow the citation audit, including third-party articles. A positive witness decides presence even when coverage is partial. Absence is decided only when the relevant index is complete.",
                "",
                "Source: recomputed from `citation_path_evidence_audit_20261116T1647Z` edge–PMID membership and directed citation links, using the completion audit’s path weights. The marginal counts were checked against `adjacent_citation_summary.tsv`. Snapshot: iCite Open Citation Collection, August 2024.",
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
    pattern_path = OUT / "data" / "source_pattern_weights.tsv"
    if pattern_path.exists():
        figure_source_patterns(pattern_path)
    combination_path = OUT / "data" / "figure6_article_combinations.tsv"
    if combination_path.exists():
        figure_article_combinations(combination_path)
    print(f"[wrote] {OUT}", flush=True)


if __name__ == "__main__":
    main()

"""Record why development recovery cannot select targets yet.

The script reads completed audit manifests and the published per-cancer
coverage table. It does not open a test split, does not list withheld pairs,
and does not fit a model.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.common import StageTimer, ensure_dir, read_json, write_json

GRAPH_RUN = Path("/home/b5ac/freddieyu.b5ac/projects/project_2/results/joint_graph_and_evidence_audit_20261116T0404Z")
PATH_AUDIT = Path("/home/b5ac/freddieyu.b5ac/projects/project_2/results/path_evidence_audit_20261116T1310Z")
COMPLETION = Path("/home/b5ac/freddieyu.b5ac/projects/project_2/results/path_evidence_audit_completion_20261116T1402Z")
CITATION = Path("/home/b5ac/freddieyu.b5ac/projects/project_2/results/citation_path_evidence_audit_20261116T1647Z")


def _band(value: int) -> str:
    if value <= 0:
        return "0"
    if value == 1:
        return "1"
    if value <= 5:
        return "2-5"
    if value <= 10:
        return "6-10"
    if value <= 25:
        return "11-25"
    if value <= 100:
        return "26-100"
    return "101+"


def coverage_bands(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open(encoding="utf-8"), delimiter="\t"))
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        direct = int(row["direct_joint"])
        reached = int(row["three_edge_joint_reached"])
        key = (_band(direct), "reached_1000+" if reached >= 1000 else "reached_under_1000")
        counts[key] = counts.get(key, 0) + 1
    return [
        {"direct_joint_band": direct, "three_edge_reach": reach, "cancers": count}
        for (direct, reach), count in sorted(counts.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    timer = StageTimer()
    run_dir = ensure_dir(args.run_dir)
    analysis = ensure_dir(run_dir / "analysis")
    ensure_dir(run_dir / "logs")
    graph_manifest = read_json(GRAPH_RUN / "manifest.json") if (GRAPH_RUN / "manifest.json").exists() else {}
    path_manifest = read_json(PATH_AUDIT / "manifest.json")
    completion_manifest = read_json(COMPLETION / "manifest.json")
    citation_manifest = read_json(CITATION / "manifest.json")
    same_graph = (
        path_manifest.get("input_run") == str(GRAPH_RUN)
        and completion_manifest.get("graph_run") == str(GRAPH_RUN)
        and citation_manifest.get("graph_run") == str(GRAPH_RUN)
        and citation_manifest.get("path_audit") == str(PATH_AUDIT)
        and citation_manifest.get("completion_audit") == str(COMPLETION)
        and citation_manifest.get("status") == "passed"
        and path_manifest.get("definitions", {}).get("one_intermediate_path", "").startswith("candidate gene")
    )
    bands = coverage_bands(COMPLETION / "analysis" / "candidate_coverage_by_query.tsv")
    timer.mark("planning", cancers=sum(row["cancers"] for row in bands))
    with (analysis / "cancer_coverage_bands.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["direct_joint_band", "three_edge_reach", "cancers"], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(bands)
    status = "blocked_no_development_allowlist"
    failures = []
    if not same_graph:
        failures.append({"name": "audit_manifest_graph_match", "observed": False})
        status = "failed_provenance"
    validation = {
        "status": status,
        "failures": failures,
        "target_provenance": "No development allowlist or pre-cleared development target file was found in the repository, project data, or completed audit outputs.",
        "formal_test_files_opened": False,
        "models_fit": False,
        "development_validation_labels_used": False,
        "audits_identify_same_graph": same_graph,
        "stages": timer.stages,
    }
    write_json(run_dir / "validation.json", validation)
    write_json(
        run_dir / "manifest.json",
        {
            "task": "association_recovery_development",
            "status": status,
            "graph_run": str(GRAPH_RUN),
            "path_evidence_audit": str(PATH_AUDIT),
            "path_evidence_completion": str(COMPLETION),
            "citation_audit": str(CITATION),
            "citation_status": citation_manifest.get("status"),
            "citation_snapshot": {
                "release_id": citation_manifest.get("release_id"),
                "doi": citation_manifest.get("doi"),
                "figshare_article_id": citation_manifest.get("figshare_article_id"),
                "citing_column": citation_manifest.get("citing_column"),
                "cited_column": citation_manifest.get("cited_column"),
                "files": {
                    name: {"bytes": item.get("bytes"), "md5": item.get("md5"), "sha256": item.get("sha256")}
                    for name, item in citation_manifest.get("files", {}).items()
                },
            },
            "path_definitions": path_manifest.get("definitions"),
            "graph_manifest_present": bool(graph_manifest),
            "candidate_universe": 13571,
            "query_universe": 1665,
            "selected_cancers": 0,
            "selected_targets": 0,
        },
    )
    (run_dir / "study_design.yaml").write_text(
        """# Frozen before any model score was inspected.
# Target assignment is intentionally not frozen: no development allowlist exists.

study:
  task: association_recovery_development
  status: blocked_no_development_allowlist
  species_tax_id: "9606"
inputs:
  graph_run: /home/b5ac/freddieyu.b5ac/projects/project_2/results/joint_graph_and_evidence_audit_20261116T0404Z
  path_evidence_audit: /home/b5ac/freddieyu.b5ac/projects/project_2/results/path_evidence_audit_20261116T1310Z
  path_evidence_completion: /home/b5ac/freddieyu.b5ac/projects/project_2/results/path_evidence_audit_completion_20261116T1402Z
  citation_audit: /home/b5ac/freddieyu.b5ac/projects/project_2/results/citation_path_evidence_audit_20261116T1647Z
  citation_snapshot: icite_snapshot_2024_08_v1
  citation_doi: 10.35092/yhjc26955625.v1
cohort:
  candidate_genes: 13571
  cancer_queries: 1665
  planned_development_cancers: 24
  selected_cancers: 0
  selected_targets: 0
  reason: >-
    No pre-cleared development target file or allowlist was found. The formal
    test split was not opened. CORE gene-cancer pairs in the joint graph were
    not reused as development labels.
split:
  fractions: [0.6, 0.2, 0.2]
  pmid_grouping: targets sharing a supporting PMID stay in one partition
  labels:
    train: fitting
    tune: parameter selection
    development_validation: the only comparison labels for a later development run
mask:
  remove_target_pair_records: both resources, all predicates, both orientations
  remove_target_pmids_globally: true
  remove_incident_citations: citing, cited, co-citing witness, and shared reference
controls:
  background_per_positive: 10
  pool: eligible genes with no direct record in the unmasked inputs
  seeds: [20261116, 20261117, 20261118]
features:
  groups: [common_baseline, relation_types, direct_citations, cocitation, shared_references, predicate_direction]
  frozen_columns: not_selected
models:
  fit: false
  planned:
    - degree_weighted_path_count
    - constrained_random_walk
    - ridge_logistic_regression
    - gradient_boosted_trees
    - r_gcn
    - biopathnet_style_path_learning
  available_libraries:
    ridge: sklearn.linear_model.RidgeClassifier
    trees: sklearn.ensemble.HistGradientBoostingClassifier
    neural: torch
  unavailable_without_install: [xgboost, lightgbm]
metrics:
  primary: macro_recall_at_50
  computed: false
contrasts:
  declared_before_fitting: true
  executed: false
""",
        encoding="utf-8",
    )
    (run_dir / "report.md").write_text(
        """# Development recovery of withheld gene–cancer associations

Validation status: `blocked_no_development_allowlist`. No model was fit and no development validation metric was computed.

The joint graph, path-evidence audit, completion audit, and citation audit identify the same graph and the same two path templates. The citation snapshot is iCite Database Snapshot 2024-08, DOI `10.35092/yhjc26955625.v1`, with columns `citing` then `referenced`. Its validation status is passed.

No development allowlist or pre-cleared target file is present in the repository, the project data directory, or the completed audit outputs. The formal test files were not opened. Pairs recorded in the joint CORE graph were not treated as development labels, because that set is not documented as disjoint from the sealed test set.

The published per-cancer coverage table contains 1,665 query cancers. `analysis/cancer_coverage_bands.tsv` counts those cancers by direct-record band and by whether at least 1,000 candidate genes have a surviving three-edge path. That table is planning information. It does not select the development cohort or any withheld target.

The six methods, the 60/20/20 PMID-grouped split, the shared mask, and the ridge and tree contrasts remain specified in `study_design.yaml`. They are ready to run when a development allowlist is supplied. Until then the candidate lists, feature columns, and seeds stay unfrozen.
""",
        encoding="utf-8",
    )
    print(f"[done] status={status}", flush=True)


if __name__ == "__main__":
    main()

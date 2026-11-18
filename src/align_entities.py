"""Resolve CORE genes and SemMedDB concept identifiers.

The mapping rules are fixed in the study design. Names are not match keys.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from src.common import SCRATCH_ROOT, ensure_dir, write_json
from src.mapping import HUMAN_TAX, resolve_cui_gene_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Align gene and cancer identifiers.")
    parser.add_argument("--scratch", type=Path, default=SCRATCH_ROOT / "prepared")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out_dir = ensure_dir(args.out)
    gene_rows = pq.read_table(args.scratch / "gene_info_matched.parquet").to_pylist()
    tax = {row["gene_id"]: row["tax_id"] for row in gene_rows}
    symbol = {row["gene_id"]: row["symbol"] for row in gene_rows}
    gene_type = {row["gene_id"]: row["type_of_gene"] for row in gene_rows}
    cui_rows = pq.read_table(args.scratch / "entity_cui_genes.parquet").to_pylist()
    resolutions = []
    counts = Counter()
    for row in cui_rows:
        gene_ids = {row["gene_id_a"]}
        if row["gene_id_b"]:
            gene_ids.add(row["gene_id_b"])
        resolved = resolve_cui_gene_ids(gene_ids, tax)
        counts[resolved["status"]] += 1
        resolutions.append(
            {
                "cui": row["cui"],
                "status": resolved["status"],
                "gene_id": resolved.get("gene_id") or "",
                "route": resolved.get("route") or "",
                "tax_id": resolved.get("tax_id") or "",
                "n_single_mentions": row["n_single_mentions"],
                "observed_gene_ids": "|".join(sorted(gene_ids)),
            }
        )
    import pyarrow as pa

    pq.write_table(pa.Table.from_pylist(resolutions), out_dir / "cui_gene_resolution.parquet")
    gcs = pq.read_table(args.scratch / "core_gcs.parquet").to_pylist()
    core_counts = Counter()
    core_rows = []
    seen = set()
    for row in gcs:
        for gene_id in str(row["gene_ids"]).split("|"):
            if not gene_id or gene_id in seen:
                continue
            seen.add(gene_id)
            tax_id = tax.get(gene_id, "")
            if not tax_id:
                status = "unresolved_unknown_gene_id"
            elif tax_id != HUMAN_TAX:
                status = "excluded_nonhuman"
            else:
                status = "resolved_human"
            core_counts[status] += 1
            core_rows.append(
                {
                    "gene_id": gene_id,
                    "status": status,
                    "tax_id": tax_id,
                    "symbol": symbol.get(gene_id, ""),
                    "type_of_gene": gene_type.get(gene_id, ""),
                    "route": "core_ncbigene_namespace_plus_gene_info",
                }
            )
    pq.write_table(pa.Table.from_pylist(core_rows), out_dir / "core_gene_resolution.parquet")
    lines = ["entity_kind\tstatus\tcount\troute"]
    for status, count in sorted(core_counts.items()):
        lines.append(f"core_gene\t{status}\t{count}\tcore_ncbigene_namespace_plus_gene_info")
    for status, count in sorted(counts.items()):
        lines.append(f"semmeddb_cui\t{status}\t{count}\tentity_gene_id")
    (out_dir / "entity_alignment.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        out_dir / "alignment_counts.json",
        {"core_gene": dict(core_counts), "semmeddb_cui": dict(counts), "gene_info_matched": len(gene_rows)},
    )
    print(f"aligned core_genes={len(core_rows)} cuis={len(resolutions)}", flush=True)


if __name__ == "__main__":
    main()

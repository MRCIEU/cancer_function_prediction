"""Download pinned releases and convert them to columnar intermediates.

Large scans stream the archives. Sentence prose and abstracts are not stored.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import http.cookiejar
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from src.common import (
    DATA_ROOT,
    SCRATCH_ROOT,
    StageTimer,
    ensure_dir,
    load_yaml,
    md5_file,
    normalise_gene_id,
    normalise_pmid,
    sha256_file,
    write_json,
)
from src.core_parse import parse_core_turtle
from src.mapping import endpoint_class, parse_endpoint_identifier, parse_entity_gene_cell, semantic_tokens

PREDICATE_COLUMNS = [
    "predication_id",
    "sentence_id",
    "pmid",
    "predicate",
    "subject_cui",
    "subject_name",
    "subject_semtype",
    "subject_novelty",
    "object_cui",
    "object_name",
    "object_semtype",
    "object_novelty",
    "trailing_1",
    "trailing_2",
    "trailing_3",
]
ENTITY_COLUMNS = [
    "entity_id",
    "sentence_id",
    "pmid_candidate",
    "cui",
    "name",
    "semtype",
    "gene_id",
    "gene_name",
    "text",
    "score",
    "start_index",
    "end_index",
]
CITATION_COLUMNS = ["pmid", "issn", "dp", "edat", "pyear"]


def _request(url: str, opener, headers: dict | None = None):
    req = urllib.request.Request(url, headers=headers or {"User-Agent": "joint-graph-audit/1.0"})
    return opener.open(req, timeout=180)


def uts_opener() -> urllib.request.OpenerDirector:
    key = os.environ.get("UMLS_API_KEY", "")
    if not key:
        raise RuntimeError("UMLS_API_KEY is not available in the environment")
    data = urllib.parse.urlencode({"apikey": key}).encode()
    with urllib.request.urlopen(
        urllib.request.Request("https://utslogin.nlm.nih.gov/cas/v1/api-key", data=data),
        timeout=60,
    ) as response:
        body = response.read().decode("utf-8", "replace")
    match = re.search(r'action="([^"]+)"', body)
    if not match:
        raise RuntimeError("UTS authentication did not return a ticket-granting ticket")
    prime = (
        "https://data.lhncbc.nlm.nih.gov/umls-restricted/ii/tools/"
        "SemRep_SemMedDB_SKR/semmedVER43_2024_R_GENERIC_CONCEPT.csv.gz"
    )
    with urllib.request.urlopen(
        urllib.request.Request(match.group(1), data=urllib.parse.urlencode({"service": prime}).encode()),
        timeout=60,
    ) as response:
        ticket = response.read().decode().strip()
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    _request(prime + "?" + urllib.parse.urlencode({"ticket": ticket}), opener).read()
    return opener


def download_file(url: str, dest: Path, opener=None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".partial")
    existing = temporary.stat().st_size if temporary.exists() else 0
    headers = {"User-Agent": "joint-graph-audit/1.0"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    client = opener or urllib.request.build_opener()
    try:
        response = _request(url, client, headers)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and dest.exists():
            return
        raise
    mode = "ab" if existing and response.status == 206 else "wb"
    if mode == "wb" and temporary.exists():
        temporary.unlink()
    started = time.time()
    total = 0
    with response, temporary.open(mode) as handle:
        while True:
            block = response.read(8 * 1024 * 1024)
            if not block:
                break
            handle.write(block)
            total += len(block)
    temporary.replace(dest)
    elapsed = max(time.time() - started, 0.001)
    print(f"downloaded {dest.name} bytes={dest.stat().st_size} rate_Bps={total / elapsed:.0f}", flush=True)


def _checksum_token(text: str) -> str:
    token = text.strip().split()[0].lower()
    return token


def verify_public_md5(path: Path, expected: str) -> str:
    observed = md5_file(path)
    if observed.lower() != expected.lower():
        raise RuntimeError(f"MD5 mismatch for {path.name}")
    return observed


def _complete_utf8_prefix(data: bytes) -> int:
    """Length of data that does not end inside a multibyte UTF-8 character."""
    if not data:
        return 0
    index = len(data) - 1
    continuation = 0
    while index >= 0 and continuation < 3 and data[index] & 0xC0 == 0x80:
        index -= 1
        continuation += 1
    if index < 0:
        return 0
    lead = data[index]
    if lead < 0x80:
        return len(data)
    if 0xC0 <= lead <= 0xDF:
        needed = 2
    elif 0xE0 <= lead <= 0xEF:
        needed = 3
    elif 0xF0 <= lead <= 0xF7:
        needed = 4
    else:
        return len(data)
    if len(data) - index < needed:
        return index
    return len(data)


def _sanitize_utf8(data: bytes) -> bytes:
    if not data:
        return b""
    return data.decode("utf-8", "replace").encode("utf-8")


class _BackslashQuoteFile:
    """Turn SemMedDB ``\\"`` escapes into standard doubled CSV quotes.

    ``\\N`` stays intact so it can still be read as a null. A global CSV
    escape character would also consume that null marker. Invalid UTF-8
    bytes are replaced so one bad name does not stop the scan.
    """

    def __init__(self, source):
        self.source = source
        self.buffer = b""
        self.tail = b""
        self.utf8_tail = b""
        self.closed = False

    @property
    def mode(self) -> str:
        return "rb"

    def readable(self) -> bool:
        return True

    def _accept(self, chunk: bytes, final: bool) -> None:
        if self.utf8_tail:
            chunk = self.utf8_tail + chunk
            self.utf8_tail = b""
        if not final:
            cut = _complete_utf8_prefix(chunk)
            self.utf8_tail = chunk[cut:]
            chunk = chunk[:cut]
        self.buffer += _sanitize_utf8(chunk)

    def read(self, size: int = -1) -> bytes:
        want = 1 << 20 if size is None or size < 0 else size
        while len(self.buffer) < want:
            chunk = self.source.read(1 << 20)
            if not chunk:
                if self.tail or self.utf8_tail:
                    self._accept(self.tail, final=True)
                    self.tail = b""
                break
            if self.tail:
                chunk = self.tail + chunk
                self.tail = b""
            if chunk.endswith(b"\\"):
                self.tail = b"\\"
                chunk = chunk[:-1]
            self._accept(chunk.replace(b'\\"', b'""'), final=False)
        out = self.buffer[:want]
        self.buffer = self.buffer[want:]
        return out

    def close(self) -> None:
        self.closed = True
        self.source.close()


def open_csv_gz(path: Path, columns: list[str]):
    source = gzip.open(path, "rb")
    return pacsv.open_csv(
        _BackslashQuoteFile(source),
        read_options=pacsv.ReadOptions(column_names=columns, block_size=1 << 22),
        parse_options=pacsv.ParseOptions(newlines_in_values=True, quote_char='"', double_quote=True, escape_char=False),
        convert_options=pacsv.ConvertOptions(
            column_types={name: pa.string() for name in columns},
            null_values=["\\N", "NULL"],
            strings_can_be_null=True,
        ),
    )


def _cell(value) -> str:
    if value is None:
        return ""
    return str(value)


def parse_core(ttl_path: Path, out_dir: Path) -> dict:
    text = ttl_path.read_text(encoding="utf-8")
    parsed = parse_core_turtle(text)
    gcs_rows = []
    for gcs_id, row in parsed["gcs"].items():
        gcs_rows.append(
            {
                "gcs_id": gcs_id,
                "gene_ids": "|".join(row["gene_ids"]),
                "cancer_cuis": "|".join(row["cancer_cuis"]),
                "has_type": row["has_type"],
                "n_sentences": len(row["sentence_ids"]),
                "path_eligible_has_type": row["path_eligible_has_type"],
            }
        )
    sentence_rows = []
    for sentence_id, row in parsed["sentences"].items():
        sentence_rows.append({"sentence_id": sentence_id, **row})
    article_rows = []
    for pmid, row in parsed["articles"].items():
        article_rows.append(
            {
                "pmid": pmid,
                "years": "|".join(sorted(row["years"])),
                "venues": "|".join(row["venues"]),
                "titles": "|".join(row["titles"]),
                "source": "core",
            }
        )
    pq.write_table(pa.Table.from_pylist(gcs_rows), out_dir / "core_gcs.parquet")
    pq.write_table(pa.Table.from_pylist(sentence_rows), out_dir / "core_sentences.parquet")
    pq.write_table(pa.Table.from_pylist(article_rows), out_dir / "core_articles.parquet")
    from src.common import write_parquet_rows

    write_parquet_rows(
        out_dir / "core_citation_links.parquet",
        parsed["citation_links"],
        ["citing_pmid", "cited_pmid", "source"],
    )
    write_json(out_dir / "core_counts.json", parsed["counts"])
    return parsed["counts"]


def parse_predications(path: Path, out_dir: Path, max_rows: int | None) -> dict:
    reader = open_csv_gz(path, PREDICATE_COLUMNS)
    destination = out_dir / "semmed_predications.parquet"
    writer = None
    stats = Counter()
    names: dict[str, str] = {}
    sentence_sample: dict[str, str] = {}
    batch_rows: list[dict] = []
    random.seed(20261116)
    started = time.time()

    def flush() -> None:
        nonlocal writer
        if not batch_rows:
            return
        table = pa.Table.from_pylist(batch_rows)
        if writer is None:
            writer = pq.ParquetWriter(destination, table.schema, compression="snappy")
        writer.write_table(table)
        batch_rows.clear()

    for batch in reader:
        for row in batch.to_pylist():
            stats["input_rows"] += 1
            sentence = _cell(row["sentence_id"])
            pmid = normalise_pmid(_cell(row["pmid"]))
            if sentence and pmid:
                if len(sentence_sample) < 20000:
                    sentence_sample.setdefault(sentence, pmid)
                else:
                    draw = random.randrange(stats["input_rows"])
                    if draw < 20000:
                        sentence_sample[sentence] = pmid
            if max_rows is not None and stats["input_rows"] > max_rows:
                break
            subject_tokens = semantic_tokens(row["subject_semtype"])
            object_tokens = semantic_tokens(row["object_semtype"])
            subject_class = endpoint_class(subject_tokens)
            object_class = endpoint_class(object_tokens)
            if "dsyn" in subject_tokens or "dsyn" in object_tokens:
                if subject_class == "gene" or object_class == "gene":
                    stats["gene_dsyn_rows"] += 1
            pair = {subject_class, object_class}
            if pair == {"gene"}:
                family = "gene_gene"
            elif pair == {"gene", "cancer"}:
                family = "gene_cancer"
            else:
                stats["rows_outside_gene_cancer_scope"] += 1
                continue
            predicate = _cell(row["predicate"])
            negation = predicate.startswith("NEG_")
            base = predicate[4:] if negation else predicate
            subject_raw = _cell(row["subject_cui"])
            object_raw = _cell(row["object_cui"])
            for raw, label in ((subject_raw, _cell(row["subject_name"])), (object_raw, _cell(row["object_name"]))):
                parsed = parse_endpoint_identifier(raw)
                if parsed["cui"] and label and parsed["cui"] not in names:
                    names[parsed["cui"]] = label
            for trailing_name in ("trailing_1", "trailing_2", "trailing_3"):
                if _cell(row[trailing_name]):
                    stats[f"nonempty_{trailing_name}"] += 1
            batch_rows.append(
                {
                    "predication_id": _cell(row["predication_id"]).lstrip("0") or "0",
                    "sentence_id": _cell(row["sentence_id"]),
                    "pmid": normalise_pmid(_cell(row["pmid"])) or "",
                    "predicate": predicate,
                    "base_predicate": base,
                    "negation": negation,
                    "subject_raw": subject_raw,
                    "object_raw": object_raw,
                    "subject_name": _cell(row["subject_name"]),
                    "object_name": _cell(row["object_name"]),
                    "subject_semtype": _cell(row["subject_semtype"]),
                    "object_semtype": _cell(row["object_semtype"]),
                    "subject_novelty": _cell(row["subject_novelty"]),
                    "object_novelty": _cell(row["object_novelty"]),
                    "subject_class": subject_class,
                    "object_class": object_class,
                    "family": family,
                }
            )
            stats[f"retained_{family}"] += 1
            if negation:
                stats["retained_negated"] += 1
            if len(batch_rows) >= 100000:
                flush()
        if stats["input_rows"] % 5000000 < batch.num_rows:
            elapsed = max(time.time() - started, 0.001)
            print(
                f"predications rows={stats['input_rows']} retained={stats['retained_gene_gene'] + stats['retained_gene_cancer']} rows_per_s={stats['input_rows'] / elapsed:.0f}",
                flush=True,
            )
        if max_rows is not None and stats["input_rows"] > max_rows:
            stats["truncated_by_dry_run_cap"] = max_rows
            break
    flush()
    if writer is not None:
        writer.close()
    elif not destination.exists():
        pq.write_table(pa.table({name: pa.array([], type=pa.string()) for name in ["predication_id"]}), destination)
    name_rows = [{"cui": cui, "preferred_name": name, "name_rule": "first_predication_preferred_name"} for cui, name in names.items()]
    pq.write_table(pa.Table.from_pylist(name_rows), out_dir / "cui_preferred_names.parquet")
    write_json(out_dir / "sentence_pmid_sample.json", sentence_sample)
    write_json(out_dir / "predication_stats.json", dict(stats))
    return dict(stats)


def parse_entities(path: Path, out_dir: Path, max_rows: int | None) -> dict:
    sentence_pmid = {}
    sample_path = out_dir / "sentence_pmid_sample.json"
    if sample_path.exists():
        import json

        sentence_pmid = json.loads(sample_path.read_text(encoding="utf-8"))
    reader = open_csv_gz(path, ENTITY_COLUMNS)
    stats = Counter()
    # cui -> [gene_id_a, gene_id_b, single_mentions]
    cui_state: dict[str, list] = {}
    pmid_agree = 0
    pmid_disagree = 0
    started = time.time()
    for batch in reader:
        for row in batch.to_pylist():
            stats["input_rows"] += 1
            if max_rows is not None and stats["input_rows"] > max_rows:
                break
            semtype = _cell(row["semtype"])
            gene_cell = _cell(row["gene_id"])
            is_gene_type = semtype in {"gngm", "aapp"}
            if is_gene_type:
                stats["gene_type_rows"] += 1
            if gene_cell:
                stats["gene_type_with_gene_id" if is_gene_type else "other_type_with_gene_id"] += 1
            sentence = _cell(row["sentence_id"])
            pmid_candidate = normalise_pmid(_cell(row["pmid_candidate"]))
            if sentence in sentence_pmid and pmid_candidate:
                if sentence_pmid[sentence] == pmid_candidate:
                    pmid_agree += 1
                else:
                    pmid_disagree += 1
            if not is_gene_type:
                continue
            parsed = parse_entity_gene_cell(gene_cell)
            stats[f"gene_cell_{parsed['kind']}"] += 1
            if parsed["kind"] != "single":
                continue
            cui = _cell(row["cui"])
            if not re.fullmatch(r"C\d+", cui):
                stats["single_gene_id_without_cui"] += 1
                continue
            gene_id = parsed["gene_ids"][0]
            state = cui_state.get(cui)
            if state is None:
                cui_state[cui] = [gene_id, "", 1]
            else:
                state[2] += 1
                if state[0] != gene_id and state[1] not in {"", gene_id}:
                    stats["cui_with_three_or_more_gene_ids"] += 1
                elif state[0] != gene_id and state[1] == "":
                    state[1] = gene_id
        if stats["input_rows"] % 20000000 < batch.num_rows:
            elapsed = max(time.time() - started, 0.001)
            print(f"entity rows={stats['input_rows']} rows_per_s={stats['input_rows'] / elapsed:.0f}", flush=True)
        if max_rows is not None and stats["input_rows"] > max_rows:
            stats["truncated_by_dry_run_cap"] = max_rows
            break
    rows = []
    for cui, (first, second, mentions) in cui_state.items():
        rows.append(
            {
                "cui": cui,
                "gene_id_a": first,
                "gene_id_b": second,
                "n_single_mentions": mentions,
                "n_distinct_capped": 2 if second else 1,
            }
        )
    from src.common import write_parquet_rows

    write_parquet_rows(out_dir / "entity_cui_genes.parquet", rows, ["cui", "gene_id_a", "gene_id_b", "n_single_mentions", "n_distinct_capped"])
    compared = pmid_agree + pmid_disagree
    stats["pmid_candidate_agreements"] = pmid_agree
    stats["pmid_candidate_disagreements"] = pmid_disagree
    stats["pmid_candidate_status"] = (
        "confirmed"
        if compared >= 100 and pmid_agree / compared >= 0.99
        else "unconfirmed"
    )
    gene_filled = stats["gene_type_with_gene_id"]
    gene_rows = stats["gene_type_rows"]
    other_filled = stats["other_type_with_gene_id"]
    other_rows = max(stats["input_rows"] - gene_rows, 1)
    gene_rate = gene_filled / gene_rows if gene_rows else 0
    other_rate = other_filled / other_rows
    stats["gene_id_column_status"] = (
        "confirmed" if gene_rows >= 100 and gene_rate >= 0.3 and gene_rate >= other_rate * 5 else "unconfirmed"
    )
    write_json(out_dir / "entity_stats.json", dict(stats))
    return dict(stats)


def parse_gene_info(path: Path, out_dir: Path, needed: set[str] | None) -> dict:
    kept = []
    seen = 0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames:
            reader.fieldnames = [name[1:] if name.startswith("#") else name for name in reader.fieldnames]
        for row in reader:
            seen += 1
            gene_id = normalise_gene_id(row.get("GeneID"))
            if gene_id is None:
                continue
            if needed is not None and gene_id not in needed:
                continue
            kept.append(
                {
                    "gene_id": gene_id,
                    "tax_id": (row.get("tax_id") or "").strip(),
                    "symbol": row.get("Symbol") or "",
                    "type_of_gene": row.get("type_of_gene") or "",
                }
            )
    pq.write_table(pa.Table.from_pylist(kept), out_dir / "gene_info_matched.parquet")
    stats = {"gene_info_rows": seen, "matched_gene_ids": len(kept)}
    write_json(out_dir / "gene_info_stats.json", stats)
    return stats


def collect_needed_gene_ids(out_dir: Path) -> set[str]:
    needed: set[str] = set()
    gcs = pq.read_table(out_dir / "core_gcs.parquet", columns=["gene_ids"]).column("gene_ids").to_pylist()
    for value in gcs:
        for gene in str(value).split("|"):
            if gene.isdigit():
                needed.add(gene)
    if (out_dir / "entity_cui_genes.parquet").exists():
        table = pq.read_table(out_dir / "entity_cui_genes.parquet", columns=["gene_id_a", "gene_id_b"])
        for column in ("gene_id_a", "gene_id_b"):
            for value in table.column(column).to_pylist():
                if value:
                    needed.add(str(value))
    pred_path = out_dir / "semmed_predications.parquet"
    if pred_path.exists() and pq.ParquetFile(pred_path).metadata.num_rows:
        table = pq.read_table(pred_path, columns=["subject_raw", "object_raw"])
        for column in ("subject_raw", "object_raw"):
            for value in table.column(column).to_pylist():
                parsed = parse_endpoint_identifier(value)
                needed.update(parsed["embedded_gene_ids"])
    return needed


def parse_citations_for_pmids(path: Path, pmids: set[str], out_dir: Path) -> dict:
    reader = open_csv_gz(path, CITATION_COLUMNS)
    rows = []
    seen = 0
    for batch in reader:
        for row in batch.to_pylist():
            seen += 1
            pmid = normalise_pmid(_cell(row["pmid"]))
            if pmid in pmids:
                rows.append(
                    {
                        "pmid": pmid,
                        "issn": _cell(row["issn"]),
                        "dp": _cell(row["dp"]),
                        "edat": _cell(row["edat"]),
                        "pyear": _cell(row["pyear"]),
                        "source": "semmeddb_citations",
                    }
                )
    pq.write_table(pa.Table.from_pylist(rows), out_dir / "semmed_citations_matched.parquet")
    stats = {"citation_rows": seen, "matched_pmids": len(rows)}
    write_json(out_dir / "citation_stats.json", stats)
    return stats


def graph_pmids(out_dir: Path) -> set[str]:
    pmids: set[str] = set()
    sentences = out_dir / "core_sentences.parquet"
    if sentences.exists():
        for value in pq.read_table(sentences, columns=["pmid"]).column("pmid").to_pylist():
            pmid = normalise_pmid(value)
            if pmid:
                pmids.add(pmid)
    pred = out_dir / "semmed_predications.parquet"
    if pred.exists() and pq.ParquetFile(pred).metadata.num_rows:
        for value in pq.read_table(pred, columns=["pmid"]).column("pmid").to_pylist():
            pmid = normalise_pmid(value)
            if pmid:
                pmids.add(pmid)
    return pmids


def _verified_by_name(data_root: Path) -> dict:
    path = data_root / "input_manifest.json"
    if not path.exists():
        return {}
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    return {item["name"]: item for item in payload.get("files", []) if item.get("sha256") and item.get("bytes")}


def download_inputs(data_root: Path, sources: dict) -> dict:
    core_dir = ensure_dir(data_root / "core_zenodo7577127_v1.1")
    semmed_dir = ensure_dir(data_root / "semmeddb_ver43_2024")
    gene_dir = ensure_dir(data_root / "ncbigene_info_20261115")
    records = []
    verified = _verified_by_name(data_root)
    for item in sources["core"]["files"]:
        dest = core_dir / item["name"]
        if not dest.exists() or dest.stat().st_size != item["bytes"]:
            download_file(item["url"], dest)
        prior = verified.get(item["name"])
        if prior and prior.get("bytes") == dest.stat().st_size and prior.get("upstream_md5") == item["upstream_md5"]:
            records.append(prior)
            print(f"reused verified {dest.name}", flush=True)
            continue
        observed_md5 = verify_public_md5(dest, item["upstream_md5"])
        records.append(
            {
                "release": sources["core"]["release_id"],
                "name": item["name"],
                "bytes": dest.stat().st_size,
                "md5": observed_md5,
                "sha256": sha256_file(dest),
                "upstream_md5": item["upstream_md5"],
                "source_url": sources["core"]["source_url"],
                "license": sources["core"]["license"],
            }
        )
    opener = None

    def get_opener():
        nonlocal opener
        if opener is None:
            opener = uts_opener()
        return opener

    base = sources["semmeddb"]["base_url"]
    for item in sources["semmeddb"]["files"]:
        dest = semmed_dir / item["name"]
        if not dest.exists():
            download_file(base + item["name"], dest, get_opener())
        prior = verified.get(item["name"])
        if prior and prior.get("bytes") == dest.stat().st_size:
            records.append(prior)
            print(f"reused verified {dest.name}", flush=True)
            continue
        checksums = {}
        for suffix, algorithm in ((".md5sum", "md5"), (".sha1sum", "sha1")):
            side = semmed_dir / f"{item['name']}{suffix}"
            if not side.exists():
                download_file(base + item["name"] + suffix, side, get_opener())
            checksums[algorithm] = _checksum_token(side.read_text(encoding="utf-8", errors="replace"))
        observed_md5 = md5_file(dest)
        if observed_md5 != checksums["md5"]:
            raise RuntimeError(f"SemMedDB MD5 mismatch for {item['name']}")
        sha1 = hashlib.sha1()
        with dest.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                sha1.update(block)
        if sha1.hexdigest() != checksums["sha1"]:
            raise RuntimeError(f"SemMedDB SHA-1 mismatch for {item['name']}")
        records.append(
            {
                "release": sources["semmeddb"]["release_id"],
                "table": item["table"],
                "name": item["name"],
                "bytes": dest.stat().st_size,
                "md5": observed_md5,
                "sha1": checksums["sha1"],
                "sha256": sha256_file(dest),
                "access": "UMLS Terminology Services restricted download",
            }
        )
    gene_dest = gene_dir / "gene_info.gz"
    if not gene_dest.exists():
        download_file(sources["ncbigene"]["source_url"], gene_dest)
    prior = verified.get("gene_info.gz")
    if prior and prior.get("bytes") == gene_dest.stat().st_size:
        records.append(prior)
        print("reused verified gene_info.gz", flush=True)
        write_json(data_root / "input_manifest.json", {"files": records})
        return {"files": records}
    records.append(
        {
            "release": sources["ncbigene"]["release_id"],
            "name": "gene_info.gz",
            "bytes": gene_dest.stat().st_size,
            "sha256": sha256_file(gene_dest),
            "source_url": sources["ncbigene"]["source_url"],
            "role": sources["ncbigene"]["role"],
        }
    )
    write_json(data_root / "input_manifest.json", {"files": records})
    return {"files": records}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and parse CORE, SemMedDB, and NCBI Gene info.")
    parser.add_argument("--sources", type=Path, default=Path("config/data_sources.yaml"))
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--scratch", type=Path, default=SCRATCH_ROOT / "prepared")
    parser.add_argument("--stage", choices=["download", "parse", "all"], default="all")
    parser.add_argument("--max-predication-rows", type=int, default=0)
    parser.add_argument("--max-entity-rows", type=int, default=0)
    args = parser.parse_args()
    timer = StageTimer()
    sources = load_yaml(args.sources)
    out_dir = ensure_dir(args.scratch)
    if args.stage in {"download", "all"}:
        download_inputs(args.data_root, sources)
        timer.mark("download")
    if args.stage in {"parse", "all"}:
        core_ttl = args.data_root / "core_zenodo7577127_v1.1" / "data.ttl"
        parse_core(core_ttl, out_dir)
        timer.mark("parse_core")
        pred_cap = args.max_predication_rows or None
        ent_cap = args.max_entity_rows or None
        parse_predications(args.data_root / "semmeddb_ver43_2024" / "semmedVER43_2024_R_PREDICATION.csv.gz", out_dir, pred_cap)
        timer.mark("parse_predications")
        parse_entities(args.data_root / "semmeddb_ver43_2024" / "semmedVER43_2024_R_ENTITY.csv.gz", out_dir, ent_cap)
        timer.mark("parse_entities")
        needed = collect_needed_gene_ids(out_dir)
        parse_gene_info(args.data_root / "ncbigene_info_20261115" / "gene_info.gz", out_dir, needed)
        timer.mark("parse_gene_info", needed_gene_ids=len(needed))
        parse_citations_for_pmids(
            args.data_root / "semmeddb_ver43_2024" / "semmedVER43_2024_R_CITATIONS.csv.gz",
            graph_pmids(out_dir),
            out_dir,
        )
        timer.mark("parse_citations")
        write_json(out_dir / "prepare_timing.json", {"stages": timer.stages, "max_predication_rows": pred_cap, "max_entity_rows": ent_cap})


if __name__ == "__main__":
    main()

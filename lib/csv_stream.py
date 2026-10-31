"""CSV reader that keeps quoted newlines and does not drop the first row."""

from __future__ import annotations

import codecs
import csv
import hashlib
import json
import re
import time
from pathlib import Path


class SampleLimit(Exception):
    pass


class ByteLimitedFile:
    """Stop after max_bytes without returning a partial trailing line."""

    def __init__(self, fh, max_bytes: int):
        self.fh = fh
        self.left = max_bytes
        self.hit_limit = False

    def readable(self):
        return True

    def writable(self):
        return False

    def seekable(self):
        return False

    def read(self, size=-1):
        if self.left <= 0:
            self.hit_limit = True
            return ""
        if size is None or size < 0 or size > self.left:
            size = self.left
        data = self.fh.read(size)
        self.left -= len(data)
        if self.left <= 0:
            self.hit_limit = True
        return data

    def readline(self, size=-1):
        if self.left <= 0:
            self.hit_limit = True
            return ""
        line = self.fh.readline() if size is None or size < 0 else self.fh.readline(size)
        if not line:
            return ""
        if len(line) > self.left:
            self.left = 0
            self.hit_limit = True
            return ""
        self.left -= len(line)
        return line

    def __iter__(self):
        return self

    def __next__(self):
        line = self.readline()
        if line == "":
            raise StopIteration
        return line


def compression_magic(path: Path) -> str:
    with path.open("rb") as fh:
        magic = fh.read(4)
    if magic.startswith(b"\x1f\x8b"):
        return "gzip"
    if magic.startswith(b"PK"):
        return "zip"
    return "uncompressed"


def open_text(path: Path):
    kind = compression_magic(path)
    if kind == "gzip":
        import gzip

        return gzip.open(path, "rt", encoding="utf-8", newline=""), kind
    if kind == "zip":
        raise RuntimeError(f"zip input is not a supported single-table encoding: {path}")
    return path.open("rt", encoding="utf-8", newline=""), kind


def iter_records(path: Path, dialect: str = "excel", escapechar: str | None = None, max_records: int | None = None, max_bytes: int | None = None):
    fh, kind = open_text(path)
    limited = False
    hit_bytes = False
    try:
        src = ByteLimitedFile(fh, max_bytes) if max_bytes else fh
        reader = csv.reader(src, dialect=dialect, escapechar=escapechar)
        n = 0
        hit_records = False
        hit_bytes = False
        for row in reader:
            n += 1
            yield row
            if max_records is not None and n >= max_records:
                hit_records = True
                break
        if isinstance(src, ByteLimitedFile) and src.hit_limit:
            hit_bytes = True
        limited = hit_records or hit_bytes
    finally:
        fh.close()
    iter_records.last_limited = limited  # type: ignore[attr-defined]
    iter_records.last_byte_limited = hit_bytes  # type: ignore[attr-defined]
    iter_records.last_compression = kind  # type: ignore[attr-defined]


def sample_widths(path: Path, max_records: int, max_bytes: int, dialect: str = "excel", escapechar: str | None = None) -> dict:
    collected = []
    for row in iter_records(path, dialect=dialect, escapechar=escapechar, max_records=max_records, max_bytes=max_bytes):
        collected.append(row)
    limited = bool(getattr(iter_records, "last_limited", False))
    byte_limited = bool(getattr(iter_records, "last_byte_limited", False))
    if byte_limited and len(collected) > 1:
        collected = collected[:-1]
    widths: dict[int, int] = {}
    for row in collected:
        widths[len(row)] = widths.get(len(row), 0) + 1
    mode = max(widths, key=widths.get) if widths else None
    preview = [[cell[:120] for cell in row] for row in collected[:3]]
    return {
        "records_sampled": len(collected),
        "width_counts": {str(k): v for k, v in sorted(widths.items())},
        "modal_width": mode,
        "preview": preview,
        "limited": limited,
        "compression": getattr(iter_records, "last_compression", "unknown"),
        "dialect": dialect,
        "escapechar": escapechar,
        "rows": collected,
    }


def choose_dialect(path: Path, expected_width: int, max_records: int, max_bytes: int) -> dict:
    candidates = [
        {"dialect": "excel", "escapechar": None, "name": "quote_doubling"},
        {"dialect": "excel", "escapechar": "\\", "name": "backslash_escape"},
    ]
    results = []
    for cand in candidates:
        got = sample_widths(
            path,
            max_records=max_records,
            max_bytes=max_bytes,
            dialect=cand["dialect"],
            escapechar=cand["escapechar"],
        )
        got["candidate"] = cand["name"]
        modal = got["modal_width"]
        got["matches_expected"] = modal == expected_width and got["width_counts"].get(str(expected_width), 0) == got["records_sampled"]
        got["stable_width"] = got["records_sampled"] > 0 and len(got["width_counts"]) == 1
        results.append(got)
    stable = [r for r in results if r["stable_width"]]
    chosen = None
    ambiguous = True
    content_differs = False
    if len(stable) == 1:
        chosen = stable[0]["candidate"]
        ambiguous = False
    elif len(stable) > 1:
        previews = [json.dumps(r["preview"], ensure_ascii=False) for r in stable]
        content_differs = len(set(previews)) > 1
        same_width = len({r["modal_width"] for r in stable}) == 1
        if not content_differs:
            preferred = [r for r in stable if r["candidate"] == "quote_doubling"]
            chosen = preferred[0]["candidate"] if preferred else stable[0]["candidate"]
            ambiguous = False
        elif same_width:
            # The previews differ because one dialect consumes backslashes.
            # That is evidence the file uses backslash escapes, including later \" sequences
            # that a short quote-doubling sample can miss. Keep the backslash reader.
            preferred = [r for r in stable if r["candidate"] == "backslash_escape"]
            chosen = preferred[0]["candidate"] if preferred else stable[0]["candidate"]
            ambiguous = False
    chosen_rows = []
    observed = None
    for r in results:
        if chosen is not None and r["candidate"] == chosen:
            chosen_rows = r.pop("rows")
            observed = r["modal_width"]
        else:
            r.pop("rows", None)
    return {
        "expected_width": expected_width,
        "observed_width": observed,
        "candidates": results,
        "chosen": chosen,
        "ambiguous": ambiguous,
        "content_differs_between_stable_dialects": content_differs,
        "chosen_rows": chosen_rows,
    }


def count_records(path: Path, dialect: str, escapechar: str | None, progress=None, max_records: int | None = None) -> int:
    n = 0
    nbytes = 0
    for row in iter_records(path, dialect=dialect, escapechar=escapechar, max_records=max_records):
        n += 1
        nbytes += sum(len(c) for c in row) + len(row)
        if progress and n % 200000 == 0:
            progress.update(records=200000, nbytes=nbytes)
            nbytes = 0
    if progress and n % 200000:
        progress.update(records=n % 200000, nbytes=nbytes, force=True)
    return n


def read_csv_call(path: str, n_columns: int, dialect_name: str, max_line_size: int, parallel: bool, extra: str = "", encoding: str = "utf-8") -> str:
    """Return a DuckDB read_csv(...) expression. dialect_name is quote_doubling or backslash_escape."""
    cols = ", ".join([f"'c{i}': 'VARCHAR'" for i in range(n_columns)])
    if dialect_name == "quote_doubling":
        escape_sql = '''escape='"' '''
    elif dialect_name == "backslash_escape":
        # chr(92) is one backslash. A quoted backslash is easy to terminate early in SQL.
        escape_sql = "escape=chr(92)"
    else:
        raise ValueError(f"unknown dialect {dialect_name}")
    path_sql = path.replace("'", "''")
    parallel_sql = "true" if parallel else "false"
    extra_sql = (", " + extra) if extra else ""
    return (
        f"read_csv('{path_sql}', header=false, auto_detect=false, delim=',', quote='\"', "
        f"{escape_sql}, nullstr='__SEM_NOT_A_NULL__', columns={{{cols}}}, parallel={parallel_sql}, "
        f"max_line_size={int(max_line_size)}, encoding='{encoding}'{extra_sql})"
    )


# Windows-1252 for 0x80-0x9F. Undefined bytes stay as the corresponding C1 controls
# so the original byte can be recovered. 0xA0-0xFF match Latin-1.
_CP1252_80_9F = {
    0x80: "\u20ac",
    0x81: "\u0081",
    0x82: "\u201a",
    0x83: "\u0192",
    0x84: "\u201e",
    0x85: "\u2026",
    0x86: "\u2020",
    0x87: "\u2021",
    0x88: "\u02c6",
    0x89: "\u2030",
    0x8A: "\u0160",
    0x8B: "\u2039",
    0x8C: "\u0152",
    0x8D: "\u008d",
    0x8E: "\u017d",
    0x8F: "\u008f",
    0x90: "\u0090",
    0x91: "\u2018",
    0x92: "\u2019",
    0x93: "\u201c",
    0x94: "\u201d",
    0x95: "\u2022",
    0x96: "\u2013",
    0x97: "\u2014",
    0x98: "\u02dc",
    0x99: "\u2122",
    0x9A: "\u0161",
    0x9B: "\u203a",
    0x9C: "\u0153",
    0x9D: "\u009d",
    0x9E: "\u017e",
    0x9F: "\u0178",
}
_BYTE_AS_UTF8 = tuple(
    (_CP1252_80_9F[i] if i in _CP1252_80_9F else chr(i)).encode("utf-8") for i in range(256)
)
_SURROGATE_BYTE = re.compile(r"[\udc80-\udcff]")


def repair_utf8_file(src: Path, dest: Path, logger) -> dict:
    """Write a UTF-8 copy that keeps every source byte and every line.

    Valid UTF-8 is unchanged. A byte that is not valid UTF-8 becomes one
    Windows-1252 character (MySQL's latin1). No row is skipped.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    if partial.exists():
        partial.unlink()
    decoder = codecs.getincrementaldecoder("utf-8")("surrogateescape")
    hasher = hashlib.blake2b(digest_size=32)
    replaced = 0
    physical_lines = 0
    bytes_in = 0
    first_events: list[dict] = []
    started = time.time()
    last_log = started
    logger.info("phase_start phase=repair_utf8 source=%s output=%s", src, partial)
    with src.open("rb") as fin, partial.open("wb") as fout:
        while True:
            raw = fin.read(8 * 1024 * 1024)
            text = decoder.decode(raw, final=not raw)
            if raw:
                hasher.update(raw)
                bytes_in += len(raw)
            if not raw and not text:
                break
            chunk_start = bytes_in - len(raw)
            if _SURROGATE_BYTE.search(text) is None:
                piece = text.encode("utf-8")
            else:
                out = bytearray()
                cursor = 0
                for found in _SURROGATE_BYTE.finditer(text):
                    out += text[cursor:found.start()].encode("utf-8")
                    byte = ord(found.group()) - 0xDC00
                    out += _BYTE_AS_UTF8[byte]
                    if len(first_events) < 12:
                        first_events.append(
                            {
                                "physical_line": physical_lines + text.count("\n", 0, found.start()) + 1,
                                "approx_source_byte": chunk_start,
                                "char_index_in_chunk": found.start(),
                                "byte_hex": f"{byte:02x}",
                            }
                        )
                    cursor = found.end()
                    replaced += 1
                out += text[cursor:].encode("utf-8")
                piece = bytes(out)
            fout.write(piece)
            physical_lines += text.count("\n")
            now = time.time()
            if now - last_log >= 45:
                logger.info(
                    "progress phase=repair_utf8 bytes_in=%s physical_lines=%s replaced_bytes=%s output_bytes=%s last_progress_utc=%s",
                    bytes_in,
                    physical_lines,
                    replaced,
                    fout.tell(),
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
                )
                last_log = now
            if not raw:
                break
    partial.replace(dest)
    stats = {
        "status": "success",
        "source": str(src),
        "output": str(dest),
        "source_bytes": bytes_in,
        "source_blake2b": hasher.hexdigest(),
        "output_bytes": dest.stat().st_size,
        "physical_lines": physical_lines,
        "replaced_bytes": replaced,
        "first_replacements": first_events,
        "rule": "valid_utf8_kept; invalid_byte_as_windows_1252; undefined_cp1252_kept_as_c1; no_rows_dropped",
        "prior_utf8_failure": {
            "job": "7422261",
            "physical_line": 1370276539,
            "note": "DuckDB utf-8 reader stopped on a non-utf-8 byte in a name field",
        },
    }
    logger.info(
        "phase=repair_utf8_done bytes_in=%s physical_lines=%s replaced_bytes=%s blake2b=%s elapsed_s=%.1f",
        bytes_in,
        physical_lines,
        replaced,
        stats["source_blake2b"],
        time.time() - started,
    )
    return stats

"""Stream Turtle/N-Triples/RDF-XML triples without retaining the graph."""

from __future__ import annotations

from pathlib import Path

from rdflib import Graph, Literal, URIRef
from rdflib.term import BNode


class SampleLimitReached(Exception):
    pass


class _Sink(Graph):
    def __init__(self, emit, limit: int | None):
        super().__init__()
        self._emit = emit
        self._limit = limit
        self.n = 0

    def add(self, triple):
        if self._limit is not None and self.n >= self._limit:
            raise SampleLimitReached()
        self.n += 1
        self._emit(triple)
        return self

    def addN(self, quads):
        for quad in quads:
            self.add(quad[:3])
        return self


def _convert(triple) -> tuple[str, str, str, str, str, str]:
    subject, predicate, obj = triple
    subj = str(subject)
    pred = str(predicate)
    if isinstance(obj, Literal):
        return subj, pred, str(obj), "literal", str(obj.datatype) if obj.datatype else "", obj.language or ""
    if isinstance(obj, BNode):
        return subj, pred, str(obj), "bnode", "", ""
    if isinstance(obj, URIRef):
        return subj, pred, str(obj), "uri", "", ""
    return subj, pred, str(obj), type(obj).__name__, "", ""


def detect_rdf_format(path: Path) -> str:
    with path.open("rb") as fh:
        raw = fh.read(256)
    text = raw.lstrip().lower()
    if text.startswith(b"<?xml") or text.startswith(b"<rdf:rdf") or text.startswith(b"<rdf"):
        return "xml"
    if text.startswith(b"@prefix") or text.startswith(b"@base") or text.startswith(b"prefix ") or text.startswith(b"base "):
        return "turtle"
    if text.startswith(b"<"):
        return "nt"
    if text.startswith(b"{") or b"\"@context\"" in raw[:200]:
        return "json-ld"
    return "turtle"


def iter_triples(path: Path, limit: int | None = None, rdf_format: str | None = None):
    fmt = rdf_format or detect_rdf_format(path)
    batch: list[tuple] = []

    def emit(triple):
        batch.append(_convert(triple))

    sink = _Sink(emit, limit)
    try:
        sink.parse(path.as_posix(), format=fmt)
    except SampleLimitReached:
        pass
    yield from batch
    iter_triples.format_used = fmt  # type: ignore[attr-defined]
    iter_triples.n_emitted = sink.n  # type: ignore[attr-defined]


def stream_to_callback(path: Path, callback, batch_size: int = 5000, limit: int | None = None, rdf_format: str | None = None) -> dict:
    fmt = rdf_format or detect_rdf_format(path)
    batch: list[tuple] = []
    state = {"n": 0}

    def emit(triple):
        batch.append(_convert(triple))
        if len(batch) >= batch_size:
            callback(batch)
            batch.clear()

    sink = _Sink(emit, limit)
    try:
        sink.parse(path.as_posix(), format=fmt)
    except SampleLimitReached:
        pass
    if batch:
        callback(batch)
    state["n"] = sink.n
    state["format"] = fmt
    state["limited"] = limit is not None and sink.n >= limit
    return state

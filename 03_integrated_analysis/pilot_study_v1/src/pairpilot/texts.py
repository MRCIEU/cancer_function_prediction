"""Sentence offsets and PubMed abstract retrieval."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from pairpilot.contracts import text_version_id

_TERMINATORS = set(".!?。！？\n")


def split_sentences(text: str) -> list[dict]:
    """Spans use Python code points and are half-open. Order follows the source."""
    spans = []
    start = 0
    source = text or ""
    for index, char in enumerate(source):
        if char not in _TERMINATORS:
            continue
        end = index + 1
        while start < end and source[start].isspace():
            start += 1
        if start < end and source[start:end].strip():
            spans.append({"start": start, "end": end, "text": source[start:end]})
        start = end
    if start < len(source) and source[start:].strip():
        while start < len(source) and source[start].isspace():
            start += 1
        if start < len(source):
            spans.append({"start": start, "end": len(source), "text": source[start:]})
    return spans


def normalize_view(text: str) -> str:
    return " ".join((text or "").casefold().split())


def document_record(pmid: str, title: str, sections: list[dict], source: str, retrieved_at: str) -> dict:
    body = "\n".join(section["text"] for section in sections)
    payload = json.dumps({"title": title, "sections": sections}, ensure_ascii=False, sort_keys=True)
    return {
        "pmid": str(pmid),
        "title": title or "",
        "abstract_sections": sections,
        "text": body,
        "source": source,
        "retrieved_at": retrieved_at,
        "text_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "text_version_id": text_version_id(source, payload),
        "completeness": "complete_abstract" if body.strip() else "abstract_missing",
    }


def pubmed_fetch(pmids: list[str], pause: float, batch: int, api_key_env: str) -> list[dict]:
    key = os.environ.get(api_key_env) or ""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    found = {}
    for start in range(0, len(pmids), batch):
        chunk = [str(item) for item in pmids[start:start + batch]]
        query = {
            "db": "pubmed",
            "id": ",".join(chunk),
            "retmode": "xml",
        }
        if key:
            query["api_key"] = key
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + urllib.parse.urlencode(query)
        request = urllib.request.Request(url, headers={"User-Agent": "pairpilot/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
        found.update(_parse_pubmed(payload, now))
        time.sleep(pause)
    rows = []
    for pmid in pmids:
        rows.append(found.get(str(pmid)) or {
            "pmid": str(pmid),
            "title": "",
            "abstract_sections": [],
            "text": "",
            "source": "pubmed",
            "retrieved_at": now,
            "text_sha256": "",
            "text_version_id": "",
            "completeness": "fetch_failed",
        })
    return rows


def _parse_pubmed(payload: bytes, retrieved_at: str) -> dict:
    root = ET.fromstring(payload)
    out = {}
    for article in root.findall(".//PubmedArticle"):
        pmid = (article.findtext(".//MedlineCitation/PMID") or "").strip()
        title = "".join(article.findtext(".//ArticleTitle") or "").strip()
        sections = []
        for node in article.findall(".//Abstract/AbstractText"):
            label = node.attrib.get("Label") or "UNLABELED"
            text = "".join(node.itertext()).strip()
            sections.append({"label": label, "text": text})
        if pmid:
            out[pmid] = document_record(pmid, title, sections, "pubmed", retrieved_at)
    return out


def europepmc_fetch(pmid: str) -> dict | None:
    url = (
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search?"
        + urllib.parse.urlencode({
            "query": f"EXT_ID:{pmid} AND SRC:MED",
            "resultType": "core",
            "format": "json",
            "pageSize": "1",
        })
    )
    request = urllib.request.Request(url, headers={"User-Agent": "pairpilot/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    results = ((payload.get("resultList") or {}).get("result") or [])
    if not results:
        return None
    hit = results[0]
    abstract = hit.get("abstractText") or ""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sections = [{"label": "UNLABELED", "text": abstract}] if abstract else []
    return document_record(pmid, hit.get("title") or "", sections, "europepmc", now)

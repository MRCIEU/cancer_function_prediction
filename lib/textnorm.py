"""String correspondence. Normalization is not scientific equivalence."""

from __future__ import annotations

import hashlib
import re
import unicodedata

_WS = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value)
    text = text.replace("\u00a0", " ")
    text = _WS.sub(" ", text).strip()
    return text


def text_sha(value: str) -> str:
    return hashlib.blake2b(value.encode("utf-8"), digest_size=16).hexdigest()


def compare_texts(left: str | None, right: str | None) -> dict:
    if left is None or right is None:
        return {"status": "not_assessable", "match": None, "method": None}
    if left == "":
        return {"status": "empty_string", "match": False, "method": None}
    if right == "":
        return {"status": "empty_string", "match": False, "method": None}
    if left == right:
        return {"status": "computed", "match": True, "method": "exact"}
    if normalize_text(left) == normalize_text(right):
        return {"status": "computed", "match": True, "method": "nfkc_whitespace"}
    return {"status": "computed", "match": False, "method": "exact_then_nfkc_whitespace"}

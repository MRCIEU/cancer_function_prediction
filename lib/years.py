"""Publication-year parsing. PYEAR is not treated as a publication year."""

from __future__ import annotations

import re

_RANGE = re.compile(r"^(\d{4})\s*[-/]\s*(\d{4})\b")
_START = re.compile(r"^(\d{4})\b")
_PYEAR = re.compile(r"^\d{4}$")
_INT = re.compile(r"^-?\d+$")


def parse_dp(value: str | None) -> dict:
    """Parse a publication year from SemMedDB DP. Does not consult PYEAR."""
    if value is None:
        return {"status": "field_absent", "years": [], "raw": None}
    raw = value.strip()
    if raw == "":
        return {"status": "empty_string", "years": [], "raw": value}
    if raw == "\\N":
        return {"status": "backslash_N", "years": [], "raw": value}
    if raw in {"NULL", "null", "None"}:
        return {"status": "literal_NULL", "years": [], "raw": value}
    ranged = _RANGE.match(raw)
    if ranged:
        return {
            "status": "multi_year_range",
            "years": [int(ranged.group(1)), int(ranged.group(2))],
            "raw": value,
        }
    start = _START.match(raw)
    if start:
        return {"status": "parsed_single", "years": [int(start.group(1))], "raw": value}
    return {"status": "unparsed", "years": [], "raw": value}


def parse_completion_pyear(value: str | None) -> dict:
    """PYEAR as the dbinfo completion-date field. Not a publication year."""
    if value is None:
        return {"status": "field_absent", "year": None}
    raw = value.strip()
    if raw == "":
        return {"status": "empty_string", "year": None}
    if raw == "\\N":
        return {"status": "backslash_N", "year": None}
    if _PYEAR.fullmatch(raw):
        return {"status": "parsed_completion_year", "year": int(raw)}
    return {"status": "unparsed", "year": None}


def parse_core_year(value: str | None) -> dict:
    if value is None:
        return {"status": "field_absent", "year": None}
    raw = value.strip()
    if raw == "":
        return {"status": "empty_string", "year": None}
    if not _INT.fullmatch(raw):
        return {"status": "unparsed", "year": None}
    year = int(raw)
    if year <= 0:
        return {"status": "non_positive", "year": year}
    return {"status": "parsed_single", "year": year}

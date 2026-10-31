"""CORE label handling. NOTINF is a stored value, not a missing field."""

from __future__ import annotations

LABEL_FIELDS = ("GCCLabel", "CGELabel", "CCSLabel", "GCILabel")
NOTINF = "NOTINF"


def missing_token(value: str | None) -> str:
    if value is None or value == "":
        return "FIELD_ABSENT"
    return value


def label_tuple(gcc, cge, ccs, gci) -> tuple[str, str, str, str]:
    return tuple(missing_token(v) for v in (gcc, cge, ccs, gci))  # type: ignore[return-value]


def exclude_notinf(tup: tuple[str, str, str, str]) -> dict:
    kept = []
    for value in tup:
        if value == NOTINF:
            continue
        if value == "FIELD_ABSENT":
            continue
        kept.append(value)
    if not kept:
        return {"status": "no_usable_label", "values": []}
    return {"status": "has_usable_label", "values": kept}


def heterogeneous(tuples: list[tuple]) -> dict:
    """Heterogeneous when the set of label tuples contains more than one distinct tuple.

    FIELD_ABSENT and NOTINF both count as distinct stored states in this comparison.
    """
    uniq = {t for t in tuples}
    return {
        "n_records": len(tuples),
        "n_distinct_tuples": len(uniq),
        "heterogeneous": len(uniq) > 1,
        "status": "computed" if tuples else "not_applicable",
    }


def heterogeneous_excluding_notinf(tuples: list[tuple]) -> dict:
    usable = []
    n_no_usable = 0
    for tup in tuples:
        got = exclude_notinf(tup)
        if got["status"] == "no_usable_label":
            n_no_usable += 1
        else:
            usable.append(tuple(got["values"]))
    uniq = set(usable)
    if not tuples:
        status = "not_applicable"
    elif not usable:
        status = "no_usable_label"
    else:
        status = "computed"
    return {
        "n_records": len(tuples),
        "n_no_usable_label": n_no_usable,
        "n_distinct_usable": len(uniq),
        "heterogeneous": len(uniq) > 1 if usable else None,
        "status": status,
    }

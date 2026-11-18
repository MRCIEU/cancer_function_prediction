"""Streaming Turtle reader for the CORE release.

The reader accepts the prefix, IRI, prefixed-name, and quoted-literal
subset used by the released knowledge base. It does not interpret prose.
"""

from __future__ import annotations

from collections.abc import Iterator


class TurtleParseError(ValueError):
    pass


def _near(text: str, index: int) -> str:
    return text[max(0, index - 40) : index + 40].replace("\n", "\\n")


def iter_triples(text: str) -> Iterator[tuple[str, str, str]]:
    """Yield (subject, predicate, object) with IRIs expanded and lexical forms kept."""
    prefixes: dict[str, str] = {}
    length = len(text)
    index = 0

    def peek() -> str:
        return text[index] if index < length else ""

    def skip_ws_and_comments() -> None:
        nonlocal index
        while index < length:
            char = text[index]
            if char in " \t\r\n":
                index += 1
                continue
            if char == "#":
                while index < length and text[index] not in "\n\r":
                    index += 1
                continue
            break

    def read_iri() -> str:
        nonlocal index
        if peek() != "<":
            raise TurtleParseError(f"expected IRI near {_near(text, index)!r}")
        index += 1
        start = index
        while index < length and text[index] != ">":
            index += 1
        if index >= length:
            raise TurtleParseError("unterminated IRI")
        value = text[start:index]
        index += 1
        return value

    def read_prefixed() -> str:
        nonlocal index
        start = index
        while index < length and text[index] not in " \t\r\n,.;<>\"#()[]":
            index += 1
        token = text[start:index]
        if token == "a":
            return "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
        if ":" not in token:
            raise TurtleParseError(f"unexpected token {token[:40]!r} near {_near(text, index)!r}")
        prefix, local = token.split(":", 1)
        if prefix not in prefixes:
            raise TurtleParseError(f"unknown prefix {prefix}")
        return prefixes[prefix] + local

    def unescape_literal(literal: str) -> str:
        if "\\" not in literal:
            return literal
        chars: list[str] = []
        cursor = 0
        size = len(literal)
        while cursor < size:
            char = literal[cursor]
            if char != "\\":
                chars.append(char)
                cursor += 1
                continue
            cursor += 1
            if cursor >= size:
                raise TurtleParseError("unterminated escape")
            escaped = literal[cursor]
            mapping = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\"}
            if escaped in mapping:
                chars.append(mapping[escaped])
                cursor += 1
                continue
            if escaped == "u" and cursor + 4 < size:
                chars.append(chr(int(literal[cursor + 1 : cursor + 5], 16)))
                cursor += 5
                continue
            if escaped == "U" and cursor + 8 < size:
                chars.append(chr(int(literal[cursor + 1 : cursor + 9], 16)))
                cursor += 9
                continue
            chars.append(escaped)
            cursor += 1
        return "".join(chars)

    def read_string(discard: bool = False) -> str:
        nonlocal index
        if peek() != '"':
            raise TurtleParseError("expected string")
        long = text.startswith('"""', index)
        index += 3 if long else 1
        start = index
        while index < length:
            if text[index] == "\\":
                index += 2
                continue
            if long and text.startswith('"""', index):
                literal = "" if discard else unescape_literal(text[start:index])
                index += 3
                return literal
            if not long and text[index] == '"':
                literal = "" if discard else unescape_literal(text[start:index])
                index += 1
                return literal
            index += 1
        raise TurtleParseError("unterminated string")

    def skip_datatype_or_lang() -> None:
        nonlocal index
        skip_ws_and_comments()
        if text.startswith("^^", index):
            index += 2
            skip_ws_and_comments()
            if peek() == "<":
                read_iri()
            else:
                read_prefixed()
            return
        if peek() == "@":
            index += 1
            while index < length and text[index].isalnum() or (index < length and text[index] == "-"):
                index += 1

    def read_object(discard: bool = False) -> str:
        nonlocal index
        skip_ws_and_comments()
        if peek() == "<":
            return read_iri()
        if peek() == '"':
            value = read_string(discard=discard)
            skip_datatype_or_lang()
            return value
        if text.startswith("true", index) and not text[index + 4 : index + 5].isalnum():
            index += 4
            return "true"
        if text.startswith("false", index) and not text[index + 5 : index + 6].isalnum():
            index += 5
            return "false"
        return read_prefixed()

    def read_subject_or_predicate() -> str:
        skip_ws_and_comments()
        if peek() == "<":
            return read_iri()
        return read_prefixed()

    while True:
        skip_ws_and_comments()
        if index >= length:
            return
        if text.startswith("@prefix", index):
            index += len("@prefix")
            skip_ws_and_comments()
            start = index
            while index < length and text[index] not in " \t\r\n:":
                index += 1
            prefix = text[start:index]
            if peek() == ":":
                index += 1
            skip_ws_and_comments()
            prefixes[prefix] = read_iri()
            skip_ws_and_comments()
            if peek() != ".":
                raise TurtleParseError("prefix declaration was not terminated")
            index += 1
            continue
        if text.startswith("@base", index):
            index += len("@base")
            skip_ws_and_comments()
            read_iri()
            skip_ws_and_comments()
            if peek() != ".":
                raise TurtleParseError("base declaration was not terminated")
            index += 1
            continue
        subject = read_subject_or_predicate()
        while True:
            predicate = read_subject_or_predicate()
            while True:
                discard = predicate.endswith("/hasContent") or predicate.endswith("/abstract")
                obj = read_object(discard=discard)
                yield subject, predicate, obj
                skip_ws_and_comments()
                if peek() == ",":
                    index += 1
                    continue
                break
            if peek() == ";":
                index += 1
                skip_ws_and_comments()
                if peek() == ".":
                    index += 1
                    break
                continue
            if peek() == ".":
                index += 1
                break
            raise TurtleParseError(f"expected separator near {_near(text, index)!r}")

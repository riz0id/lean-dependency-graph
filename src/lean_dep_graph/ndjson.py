"""Reader for the lean4export NDJSON export format.

The format interns names, universe levels and expressions into three
integer-keyed tables; every reference is an index into one of them.  Records
appear in dependency order, so an index always refers to an already-emitted
record.  See ``format_ndjson.md`` in the lean4export repository.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

__all__: list[str] = [
    "ANONYMOUS",
    "DeclKey",
    "EXPR_KEYS",
    "Export",
    "ExportFormatError",
    "iter_records",
    "load",
    "load_path",
]


class DeclKey(StrEnum):
    """Top-level record keys that introduce a declaration."""

    AXIOM = "axiom"
    DEF = "def"
    OPAQUE = "opaque"
    THM = "thm"
    QUOT = "quot"
    INDUCTIVE = "inductive"
    CONSTRUCTOR = "constructor"
    RECURSOR = "recursor"


EXPR_KEYS: frozenset[str] = frozenset(
    {
        "bvar",
        "sort",
        "const",
        "app",
        "lam",
        "forallE",
        "letE",
        "proj",
        "natVal",
        "strVal",
        "mdata",
    }
)

ANONYMOUS = 0


class ExportFormatError(Exception):
    """Raised when a record does not match the expected export schema."""


@dataclass
class Export:
    """A parsed export: three interning tables plus the declaration records."""

    meta: dict[str, Any] = field(default_factory=dict)
    names: dict[int, dict[str, Any]] = field(default_factory=dict)
    levels: dict[int, dict[str, Any]] = field(default_factory=dict)
    exprs: dict[int, dict[str, Any]] = field(default_factory=dict)
    decls: list[tuple[DeclKey, dict[str, Any]]] = field(default_factory=list)

    name_cache: dict[int, str] = field(default_factory=dict, repr=False)

    def name(self, idx: int) -> str:
        cache: dict[int, str] = self.name_cache
        hit: str | None = cache.get(idx)

        if hit is not None:
            return hit

        if idx == ANONYMOUS:
            cache[idx] = ""
            return ""

        chain: list[int] = []
        cur = idx

        while cur != ANONYMOUS and cur not in cache:
            chain.append(cur)
            rec = self.names.get(cur)
            if rec is None:
                raise ExportFormatError(f"dangling name index {cur}")
            match rec:
                case {"str": {"pre": pre}} | {"num": {"pre": pre}}:
                    cur = int(pre)
                case _:
                    raise ExportFormatError(f"name record {cur} has neither 'str' nor 'num'")

        prefix = cache.get(cur, "") if cur != ANONYMOUS else ""
        for i in reversed(chain):
            rec = self.names[i]
            component = str(rec["str"]["str"]) if "str" in rec else str(rec["num"]["i"])
            prefix = f"{prefix}.{component}" if prefix else component
            cache[i] = prefix
        return prefix

    def expr(self, idx: int) -> dict[str, Any]:
        rec = self.exprs.get(idx)
        if rec is None:
            raise ExportFormatError(f"dangling expression index {idx}")
        return rec

    @property
    def format_version(self) -> str | None:
        match self.meta:
            case {"format": {"version": str(version)}}:
                return version
            case _:
                return None

    @property
    def lean_version(self) -> str | None:
        match self.meta:
            case {"lean": {"version": str(version)}}:
                return version
            case _:
                return None


def iter_records(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    for lineno, raw in enumerate(lines, start=1):
        text: str = raw.strip()

        if not text:
            continue

        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ExportFormatError(f"line {lineno}: invalid JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise ExportFormatError(f"line {lineno}: expected a JSON object")
        yield obj


def load(lines: Iterable[str]) -> Export:
    ex = Export()

    for rec in iter_records(lines):
        match rec:
            case {"meta": meta}:
                ex.meta = meta
            case {"in": idx}:
                ex.names[int(idx)] = rec
            case {"il": idx}:
                ex.levels[int(idx)] = rec
            case {"ie": idx}:
                ex.exprs[int(idx)] = rec
            case _:
                for key in DeclKey:
                    if key in rec:
                        ex.decls.append((key, rec[key]))
                        break
                else:
                    raise ExportFormatError(f"unrecognised record: {sorted(rec)[:4]}")

    return ex


def load_path(path: str | Path) -> Export:
    with Path(path).open(encoding="utf-8") as fh:
        return load(fh)

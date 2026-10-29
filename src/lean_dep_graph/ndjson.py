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
from pathlib import Path
from typing import Any

#: Declaration record keys, per the format spec.  ``constructor`` and
#: ``recursor`` are listed in the spec prose but in practice arrive nested
#: inside an ``inductive`` record; both are accepted here defensively.
DECL_KEYS: tuple[str, ...] = (
    "axiom",
    "def",
    "opaque",
    "thm",
    "quot",
    "inductive",
    "constructor",
    "recursor",
)

#: Expression record keys, per the format spec.
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

#: Index reserved for ``Name.anonymous`` / ``Level.zero``; never emitted as a
#: record, since every name chain and level chain has to bottom out somewhere.
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
    decls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    _name_cache: dict[int, str] = field(default_factory=dict, repr=False)

    # -- name resolution ---------------------------------------------------

    def name(self, idx: int) -> str:
        """Resolve a name index to its dotted string form.

        Name records form a linked list through ``pre``; this walks that chain
        iteratively (names nest deeply in generated declarations) and memoizes
        every prefix it passes through.
        """
        cache = self._name_cache
        hit = cache.get(idx)
        if hit is not None:
            return hit
        if idx == ANONYMOUS:
            cache[idx] = ""
            return ""

        # Walk to the nearest cached (or anonymous) ancestor, recording the path.
        chain: list[int] = []
        cur = idx
        while cur != ANONYMOUS and cur not in cache:
            chain.append(cur)
            rec = self.names.get(cur)
            if rec is None:
                raise ExportFormatError(f"dangling name index {cur}")
            if "str" in rec:
                cur = int(rec["str"]["pre"])
            elif "num" in rec:
                cur = int(rec["num"]["pre"])
            else:
                raise ExportFormatError(f"name record {cur} has neither 'str' nor 'num'")

        prefix = cache.get(cur, "") if cur != ANONYMOUS else ""
        # Unwind outwards, building each intermediate name exactly once.
        for i in reversed(chain):
            rec = self.names[i]
            component = str(rec["str"]["str"]) if "str" in rec else str(rec["num"]["i"])
            prefix = f"{prefix}.{component}" if prefix else component
            cache[i] = prefix
        return prefix

    # -- convenience -------------------------------------------------------

    def expr(self, idx: int) -> dict[str, Any]:
        rec = self.exprs.get(idx)
        if rec is None:
            raise ExportFormatError(f"dangling expression index {idx}")
        return rec

    @property
    def format_version(self) -> str | None:
        fmt = self.meta.get("format")
        version = fmt.get("version") if isinstance(fmt, dict) else None
        # Narrow to str: the meta block is untyped JSON, so a non-string
        # version would otherwise escape through a ``str | None`` return.
        return version if isinstance(version, str) else None

    @property
    def lean_version(self) -> str | None:
        lean = self.meta.get("lean")
        version = lean.get("version") if isinstance(lean, dict) else None
        # Narrow to str: the meta block is untyped JSON, so a non-string
        # version would otherwise escape through a ``str | None`` return.
        return version if isinstance(version, str) else None


def iter_records(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Yield one parsed JSON object per non-blank line."""
    for lineno, raw in enumerate(lines, start=1):
        text = raw.strip()
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
    """Consume an export stream into an :class:`Export`.

    Dispatch is on which key a record carries.  Index fields distinguish the
    three primitive tables: ``in`` for names, ``il`` for levels, ``ie`` for
    expressions.
    """
    ex = Export()
    for rec in iter_records(lines):
        if "meta" in rec:
            ex.meta = rec["meta"]
            continue
        if "in" in rec:
            ex.names[int(rec["in"])] = rec
            continue
        if "il" in rec:
            ex.levels[int(rec["il"])] = rec
            continue
        if "ie" in rec:
            ex.exprs[int(rec["ie"])] = rec
            continue
        for key in DECL_KEYS:
            if key in rec:
                ex.decls.append((key, rec[key]))
                break
        else:
            raise ExportFormatError(f"unrecognised record: {sorted(rec)[:4]}")
    return ex


def load_path(path: str | Path) -> Export:
    """Load an export from a file on disk."""
    with Path(path).open(encoding="utf-8") as fh:
        return load(fh)

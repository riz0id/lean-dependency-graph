"""Helpers for constructing export streams in tests.

Writing NDJSON by hand is unreadable and easy to get subtly wrong, so tests
build streams through this small builder instead.  It interns names and
expressions exactly as the exporter does, including allocating indices in
emission order so that every reference points backwards.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest

BinderInfo = str


class ExportBuilder:
    """Builds a syntactically valid lean4export NDJSON stream."""

    def __init__(self, *, with_meta: bool = True) -> None:
        self.lines: list[str] = []
        self._names: dict[str, int] = {"": 0}
        self._next_name = 1
        self._next_expr = 1
        self._next_level = 1
        if with_meta:
            self.lines.append(
                json.dumps(
                    {
                        "meta": {
                            "exporter": {"name": "lean4export", "version": "test"},
                            "lean": {"githash": "deadbeef", "version": "4.35.0"},
                            "format": {"version": "3.1.0"},
                        }
                    }
                )
            )

    # -- primitives --------------------------------------------------------

    def name(self, dotted: str) -> int:
        """Intern a dotted name, emitting one record per new component."""
        if dotted in self._names:
            return self._names[dotted]
        pre = 0
        acc = ""
        for part in dotted.split("."):
            acc = f"{acc}.{part}" if acc else part
            existing = self._names.get(acc)
            if existing is not None:
                pre = existing
                continue
            idx = self._next_name
            self._next_name += 1
            if part.isdigit():
                self._emit({"num": {"pre": pre, "i": int(part)}, "in": idx})
            else:
                self._emit({"str": {"pre": pre, "str": part}, "in": idx})
            self._names[acc] = idx
            pre = idx
        return pre

    def level_param(self, dotted: str) -> int:
        idx = self._next_level
        self._next_level += 1
        self._emit({"param": self.name(dotted), "il": idx})
        return idx

    def _emit(self, rec: dict[str, Any]) -> None:
        self.lines.append(json.dumps(rec))

    def _expr(self, rec: dict[str, Any]) -> int:
        idx = self._next_expr
        self._next_expr += 1
        self._emit({**rec, "ie": idx})
        return idx

    # -- expressions -------------------------------------------------------

    def sort(self, level: int = 0) -> int:
        return self._expr({"sort": level})

    def bvar(self, i: int) -> int:
        return self._expr({"bvar": i})

    def const(self, dotted: str, us: Sequence[int] = ()) -> int:
        return self._expr({"const": {"name": self.name(dotted), "us": list(us)}})

    def app(self, fn: int, arg: int) -> int:
        return self._expr({"app": {"fn": fn, "arg": arg}})

    def lam(self, binder: str, type_: int, body: int, bi: BinderInfo = "default") -> int:
        return self._expr(
            {"lam": {"name": self.name(binder), "type": type_, "body": body, "binderInfo": bi}}
        )

    def forall(self, binder: str, type_: int, body: int, bi: BinderInfo = "default") -> int:
        return self._expr(
            {
                "forallE": {
                    "name": self.name(binder),
                    "type": type_,
                    "body": body,
                    "binderInfo": bi,
                }
            }
        )

    def let(self, binder: str, type_: int, value: int, body: int, nondep: bool = False) -> int:
        return self._expr(
            {
                "letE": {
                    "name": self.name(binder),
                    "type": type_,
                    "value": value,
                    "body": body,
                    "nondep": nondep,
                }
            }
        )

    def proj(self, type_name: str, idx: int, struct: int) -> int:
        return self._expr(
            {"proj": {"typeName": self.name(type_name), "idx": idx, "struct": struct}}
        )

    def nat(self, value: int) -> int:
        return self._expr({"natVal": str(value)})

    def string(self, value: str) -> int:
        return self._expr({"strVal": value})

    def mdata(self, expr: int, data: dict[str, Any] | None = None) -> int:
        return self._expr({"mdata": {"expr": expr, "data": data or {}}})

    # -- declarations ------------------------------------------------------

    def axiom(self, name: str, type_: int, *, unsafe: bool = False) -> None:
        self._emit(
            {
                "axiom": {
                    "name": self.name(name),
                    "levelParams": [],
                    "type": type_,
                    "isUnsafe": unsafe,
                }
            }
        )

    def define(
        self,
        name: str,
        type_: int,
        value: int,
        *,
        all_: Sequence[str] | None = None,
        safety: str = "safe",
    ) -> None:
        self._emit(
            {
                "def": {
                    "name": self.name(name),
                    "levelParams": [],
                    "type": type_,
                    "value": value,
                    "hints": "abbrev",
                    "safety": safety,
                    "all": [self.name(n) for n in (all_ or [name])],
                }
            }
        )

    def theorem(
        self, name: str, type_: int, value: int, *, all_: Sequence[str] | None = None
    ) -> None:
        self._emit(
            {
                "thm": {
                    "name": self.name(name),
                    "levelParams": [],
                    "type": type_,
                    "value": value,
                    "all": [self.name(n) for n in (all_ or [name])],
                }
            }
        )

    def opaque(
        self, name: str, type_: int, value: int, *, all_: Sequence[str] | None = None
    ) -> None:
        self._emit(
            {
                "opaque": {
                    "name": self.name(name),
                    "levelParams": [],
                    "type": type_,
                    "value": value,
                    "isUnsafe": False,
                    "all": [self.name(n) for n in (all_ or [name])],
                }
            }
        )

    def quot(self, name: str, type_: int, kind: str) -> None:
        self._emit(
            {
                "quot": {
                    "name": self.name(name),
                    "levelParams": [],
                    "type": type_,
                    "kind": kind,
                }
            }
        )

    def inductive(
        self,
        types: Sequence[tuple[str, int]],
        ctors: Sequence[tuple[str, int, str]] = (),
        recs: Sequence[tuple[str, int, Sequence[int]]] = (),
    ) -> None:
        """Emit one inductive record.

        ``types`` is ``(name, type_expr)``, ``ctors`` is
        ``(name, type_expr, inductive_name)``, ``recs`` is
        ``(name, type_expr, [rule_rhs_expr, ...])``.
        """
        all_names = [self.name(n) for n, _ in types]
        self._emit(
            {
                "inductive": {
                    "types": [
                        {
                            "name": self.name(n),
                            "levelParams": [],
                            "type": t,
                            "numParams": 0,
                            "numIndices": 0,
                            "all": all_names,
                            "ctors": [self.name(cn) for cn, _, ind in ctors if ind == n],
                            "numNested": 0,
                            "isRec": True,
                            "isUnsafe": False,
                            "isReflexive": False,
                        }
                        for n, t in types
                    ],
                    "ctors": [
                        {
                            "name": self.name(cn),
                            "levelParams": [],
                            "type": ct,
                            "induct": self.name(ind),
                            "cidx": i,
                            "numParams": 0,
                            "numFields": 0,
                            "isUnsafe": False,
                        }
                        for i, (cn, ct, ind) in enumerate(ctors)
                    ],
                    "recs": [
                        {
                            "name": rn,
                            "levelParams": [],
                            "type": rt,
                            "all": [self.name(n) for n, _, _ in recs],
                            "numParams": 0,
                            "numIndices": 0,
                            "numMotives": 1,
                            "numMinors": len(rules),
                            "rules": [{"ctor": 0, "nfields": 0, "rhs": rhs} for rhs in rules],
                            "k": False,
                            "isUnsafe": False,
                        }
                        for rn_raw, rt, rules in recs
                        for rn in [self.name(rn_raw)]
                    ],
                }
            }
        )

    # -- output ------------------------------------------------------------

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"

    def __iter__(self) -> Any:
        return iter(self.lines)


@pytest.fixture
def builder() -> ExportBuilder:
    return ExportBuilder()

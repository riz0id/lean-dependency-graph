"""Extraction of constant references from exported expressions.

This is the correctness core of the library: an omission here produces a
dependency set that is silently too small, which is exactly the failure a
kernel replay would reject.  Two cases are easy to miss:

* ``Expr.proj`` carries the structure's name in ``typeName``.  Lean's own
  ``Expr.foldConsts`` never offers it, so code ported from that idiom drops it.
* ``Expr.lit`` (``natVal``/``strVal``) needs ``Nat``/``String`` respectively
  even though no ``const`` node appears anywhere in the term.

Conversely, ``lam``/``forallE``/``letE`` each carry a ``name`` index that is a
*binder* name, not a constant reference.  Treating it as a dependency would add
a spurious edge to every function type in the export.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from .ndjson import EXPR_KEYS, Export, ExportFormatError

__all__: list[str] = [
    "NAT_TYPE",
    "STR_TYPE",
    "check_expr_keys",
    "collect_expr_deps",
    "expr_children",
    "expr_own_consts",
]

NAT_TYPE = "Nat"
STR_TYPE = "String"


def expr_children(rec: dict[str, Any]) -> tuple[int, ...]:
    """Expression indices directly referenced by an expression record."""
    match rec:
        case {"app": app}:
            return (int(app["fn"]), int(app["arg"]))
        case {"lam": binder} | {"forallE": binder}:
            return (int(binder["type"]), int(binder["body"]))
        case {"letE": le}:
            return (int(le["type"]), int(le["value"]), int(le["body"]))
        case {"proj": proj}:
            return (int(proj["struct"]),)
        case {"mdata": mdata}:
            return (int(mdata["expr"]),)
        case _:
            return ()


def expr_own_consts(
    ex: Export, rec: dict[str, Any], *, include_literal_types: bool = True
) -> Iterator[str]:
    """Constant names referenced by this record itself, ignoring children."""
    match rec:
        case {"const": const}:
            yield ex.name(int(const["name"]))
        case {"proj": proj}:
            yield ex.name(int(proj["typeName"]))
        case {"natVal": _} if include_literal_types:
            yield NAT_TYPE
        case {"strVal": _} if include_literal_types:
            yield STR_TYPE


def collect_expr_deps(
    ex: Export, roots: Iterable[int], *, include_literal_types: bool = True
) -> set[str]:
    out: set[str] = set()
    seen: set[int] = set()
    stack: list[int] = [int(r) for r in roots]

    while stack:
        idx: int = stack.pop()

        if idx in seen:
            continue

        seen.add(idx)
        rec: dict[str, Any] = ex.expr(idx)
        out.update(expr_own_consts(ex, rec, include_literal_types=include_literal_types))

        for child in expr_children(rec):
            if child not in seen:
                stack.append(child)

    return out


def check_expr_keys(rec: dict[str, Any]) -> None:
    if not EXPR_KEYS.intersection(rec):
        raise ExportFormatError(f"expression record {rec.get('ie')} has no known kind")

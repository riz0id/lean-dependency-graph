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

from .ndjson import Export, ExportFormatError

#: Type of a ``Literal.natVal`` expression.
NAT_TYPE = "Nat"
#: Type of a ``Literal.strVal`` expression.
STR_TYPE = "String"


def expr_children(rec: dict[str, Any]) -> tuple[int, ...]:
    """Expression indices directly referenced by an expression record."""
    if "app" in rec:
        a = rec["app"]
        return (int(a["fn"]), int(a["arg"]))
    if "lam" in rec:
        b = rec["lam"]
        return (int(b["type"]), int(b["body"]))
    if "forallE" in rec:
        b = rec["forallE"]
        return (int(b["type"]), int(b["body"]))
    if "letE" in rec:
        le = rec["letE"]
        return (int(le["type"]), int(le["value"]), int(le["body"]))
    if "proj" in rec:
        return (int(rec["proj"]["struct"]),)
    if "mdata" in rec:
        return (int(rec["mdata"]["expr"]),)
    # bvar, sort, const, natVal, strVal are leaves.
    return ()


def expr_own_consts(
    ex: Export, rec: dict[str, Any], *, include_literal_types: bool = True
) -> Iterator[str]:
    """Constant names referenced by this record itself, ignoring children."""
    if "const" in rec:
        yield ex.name(int(rec["const"]["name"]))
    elif "proj" in rec:
        yield ex.name(int(rec["proj"]["typeName"]))
    elif include_literal_types:
        if "natVal" in rec:
            yield NAT_TYPE
        elif "strVal" in rec:
            yield STR_TYPE


def collect_expr_deps(
    ex: Export, roots: Iterable[int], *, include_literal_types: bool = True
) -> set[str]:
    """Constant names reachable from ``roots``.

    Walks the expression DAG iteratively with an explicit stack: terms nest far
    deeper than Python's recursion limit allows, and interning means a subterm
    is typically shared by many parents, so a visited set is required to avoid
    re-walking it.

    Deliberately not memoized across calls.  Caching a dependency *set* per
    expression index would avoid re-walking subterms shared between different
    declarations, but materializes one set per visited node and so scales with
    the whole export rather than with a single declaration.  The visited-set
    walk keeps peak memory proportional to one declaration.
    """
    out: set[str] = set()
    seen: set[int] = set()
    stack = [int(r) for r in roots]
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        rec = ex.expr(idx)
        out.update(expr_own_consts(ex, rec, include_literal_types=include_literal_types))
        for child in expr_children(rec):
            if child not in seen:
                stack.append(child)
    return out


def check_expr_keys(rec: dict[str, Any]) -> None:
    """Raise if a record carries no recognised expression key.

    Called when validating a fixture; a new expression constructor added to the
    format upstream would otherwise be traversed as a leaf and silently drop
    whatever it references.
    """
    from .ndjson import EXPR_KEYS

    if not EXPR_KEYS.intersection(rec):
        raise ExportFormatError(f"expression record {rec.get('ie')} has no known kind")

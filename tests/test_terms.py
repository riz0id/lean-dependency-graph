"""Extraction tests: one per expression constructor, plus the known traps."""

from __future__ import annotations

from lean_dep_graph.ndjson import load
from lean_dep_graph.terms import collect_expr_deps, expr_children

from .conftest import ExportBuilder


def deps_of(b: ExportBuilder, root: int, **kw: bool) -> set[str]:
    return collect_expr_deps(load(b.lines), [root], **kw)


def test_const(builder: ExportBuilder) -> None:
    assert deps_of(builder, builder.const("Nat")) == {"Nat"}


def test_bvar_and_sort_have_no_deps(builder: ExportBuilder) -> None:
    assert deps_of(builder, builder.bvar(0)) == set()
    assert deps_of(builder, builder.sort(0)) == set()


def test_app_traverses_both_sides(builder: ExportBuilder) -> None:
    root = builder.app(builder.const("Nat.succ"), builder.const("Nat.zero"))
    assert deps_of(builder, root) == {"Nat.succ", "Nat.zero"}


def test_lam_traverses_type_and_body(builder: ExportBuilder) -> None:
    root = builder.lam("x", builder.const("Nat"), builder.const("Nat.zero"))
    assert deps_of(builder, root) == {"Nat", "Nat.zero"}


def test_forall_traverses_type_and_body(builder: ExportBuilder) -> None:
    root = builder.forall("n", builder.const("Nat"), builder.const("Prop"))
    assert deps_of(builder, root) == {"Nat", "Prop"}


def test_let_traverses_type_value_and_body(builder: ExportBuilder) -> None:
    root = builder.let(
        "y", builder.const("Nat"), builder.const("Nat.zero"), builder.const("Nat.succ")
    )
    assert deps_of(builder, root) == {"Nat", "Nat.zero", "Nat.succ"}


def test_mdata_is_transparent(builder: ExportBuilder) -> None:
    root = builder.mdata(builder.const("Nat"), {"tag": 1})
    assert deps_of(builder, root) == {"Nat"}


def test_proj_yields_type_name_and_traverses_struct(builder: ExportBuilder) -> None:
    """Expr.proj carries its structure name in `typeName`.

    Lean's own Expr.foldConsts never offers it, so an implementation modelled
    on that idiom silently drops the edge.
    """
    root = builder.proj("Prod", 0, builder.const("p"))
    assert deps_of(builder, root) == {"Prod", "p"}


def test_literals_imply_their_types(builder: ExportBuilder) -> None:
    """Nat/String are needed by the kernel with no `const` node present."""
    assert deps_of(builder, builder.nat(42)) == {"Nat"}
    assert deps_of(builder, builder.string("hi")) == {"String"}


def test_literal_types_can_be_disabled(builder: ExportBuilder) -> None:
    root = builder.app(builder.nat(1), builder.string("s"))
    assert deps_of(builder, root, include_literal_types=False) == set()


def test_binder_name_is_not_a_dependency() -> None:
    """lam/forallE/letE carry a *binder* name, not a constant reference.

    Treating it as a dependency would add a spurious edge to essentially every
    function type in an export, so this is checked against a binder whose name
    collides with a real constant.
    """
    for build in ("lam", "forall"):
        b = ExportBuilder()
        b.name("Nat")  # ensure the collision is a real interned name
        root = getattr(b, build)("Nat", b.sort(0), b.bvar(0))
        assert deps_of(b, root) == set(), build

    b = ExportBuilder()
    root = b.let("Nat", b.sort(0), b.bvar(0), b.bvar(0))
    assert deps_of(b, root) == set()


def test_shared_subterm_is_not_double_counted(builder: ExportBuilder) -> None:
    shared = builder.const("Shared")
    root = builder.app(builder.app(shared, shared), shared)
    assert deps_of(builder, root) == {"Shared"}


def test_deep_term_does_not_exhaust_the_stack() -> None:
    """Interned terms nest far past Python's recursion limit."""
    b = ExportBuilder()
    node = b.const("Base")
    for _ in range(20_000):
        node = b.app(node, b.bvar(0))
    assert deps_of(b, node) == {"Base"}


def test_multiple_roots_are_unioned(builder: ExportBuilder) -> None:
    t = builder.const("TypeSide")
    v = builder.const("ValueSide")
    ex = load(builder.lines)
    assert collect_expr_deps(ex, [t, v]) == {"TypeSide", "ValueSide"}


def test_expr_children_covers_every_constructor(builder: ExportBuilder) -> None:
    leaf = builder.const("L")
    cases = {
        "bvar": (builder.bvar(0), 0),
        "sort": (builder.sort(0), 0),
        "const": (builder.const("C"), 0),
        "natVal": (builder.nat(1), 0),
        "strVal": (builder.string("s"), 0),
        "app": (builder.app(leaf, leaf), 2),
        "lam": (builder.lam("x", leaf, leaf), 2),
        "forallE": (builder.forall("x", leaf, leaf), 2),
        "letE": (builder.let("x", leaf, leaf, leaf), 3),
        "proj": (builder.proj("S", 0, leaf), 1),
        "mdata": (builder.mdata(leaf), 1),
    }
    ex = load(builder.lines)
    for kind, (idx, expected) in cases.items():
        assert len(expr_children(ex.expr(idx))) == expected, kind

"""Graph construction: declaration parsing, grouping, edge resolution."""

from __future__ import annotations

import pytest

from lean_dep_graph.graph import build_graph, is_internal
from lean_dep_graph.ndjson import load

from .conftest import ExportBuilder


def graph_of(b: ExportBuilder, **kw: bool):
    return build_graph(load(b.lines), **kw)


def test_singleton_declarations_are_their_own_group(builder: ExportBuilder) -> None:
    builder.axiom("A", builder.sort(0))
    builder.define("d", builder.const("A"), builder.const("A"))
    g = graph_of(builder)
    assert g.members("d") == ("d",)
    assert not g.group_of("d").is_block


def test_kinds_are_recorded(builder: ExportBuilder) -> None:
    s = builder.sort(0)
    builder.axiom("ax", s)
    builder.define("df", s, s)
    builder.theorem("th", s, s)
    builder.opaque("op", s, s)
    g = graph_of(builder)
    assert {n: g.decl(n).kind for n in ("ax", "df", "th", "op")} == {
        "ax": "axiom",
        "df": "def",
        "th": "thm",
        "op": "opaque",
    }


def test_edges_point_from_dependent_to_dependency(builder: ExportBuilder) -> None:
    builder.axiom("Base", builder.sort(0))
    builder.define("uses", builder.const("Base"), builder.const("Base"))
    g = graph_of(builder)
    assert g.graph.has_edge("uses", "Base")
    assert not g.graph.has_edge("Base", "uses")


def test_mutual_inductive_block_is_one_node(builder: ExportBuilder) -> None:
    """The kernel adds a mutual inductive block as a unit, so it is one node."""
    s = builder.sort(0)
    builder.inductive(
        types=[("Tree", s), ("Forest", s)],
        ctors=[("Tree.node", builder.const("Forest"), "Tree"), ("Forest.nil", s, "Forest")],
        recs=[("Tree.rec", s, [s]), ("Forest.rec", s, [s])],
    )
    g = graph_of(builder)
    members = set(g.members("Tree"))
    assert members == {"Tree", "Forest", "Tree.node", "Forest.nil", "Tree.rec", "Forest.rec"}
    assert g.node_of("Forest.rec") == g.node_of("Tree")
    assert g.graph.number_of_nodes() == 1


def test_mutual_definition_block_is_one_node(builder: ExportBuilder) -> None:
    """`def` records carry `all`, so mutual def blocks group without heuristics."""
    s = builder.sort(0)
    builder.define("even", s, builder.const("odd"), all_=["even", "odd"])
    builder.define("odd", s, builder.const("even"), all_=["even", "odd"])
    g = graph_of(builder)
    assert set(g.members("odd")) == {"even", "odd"}
    assert g.graph.number_of_nodes() == 1


def test_mutual_block_has_no_self_edge(builder: ExportBuilder) -> None:
    s = builder.sort(0)
    builder.define("a", s, builder.const("b"), all_=["a", "b"])
    builder.define("b", s, builder.const("a"), all_=["a", "b"])
    g = graph_of(builder)
    node = g.node_of("a")
    assert not g.graph.has_edge(node, node)
    assert g.graph.number_of_edges() == 0


def test_non_mutual_def_with_singleton_all_is_not_a_block(builder: ExportBuilder) -> None:
    s = builder.sort(0)
    builder.define("solo", s, s, all_=["solo"])
    g = graph_of(builder)
    assert not g.group_of("solo").is_block


def test_quotient_declarations_group_together(builder: ExportBuilder) -> None:
    """The kernel takes a field-less `quotDecl` and emits all four at once."""
    s = builder.sort(0)
    for name, kind in (
        ("Quot", "type"),
        ("Quot.mk", "ctor"),
        ("Quot.lift", "lift"),
        ("Quot.ind", "ind"),
    ):
        builder.quot(name, s, kind)
    g = graph_of(builder)
    assert set(g.members("Quot.lift")) == {"Quot", "Quot.mk", "Quot.lift", "Quot.ind"}


def test_recursor_rule_rhs_contributes_dependencies(builder: ExportBuilder) -> None:
    """A rule's right-hand side can reference constants absent from the type."""
    s = builder.sort(0)
    builder.axiom("OnlyInRule", s)
    builder.inductive(
        types=[("I", s)],
        ctors=[("I.mk", s, "I")],
        recs=[("I.rec", s, [builder.const("OnlyInRule")])],
    )
    g = graph_of(builder)
    assert "OnlyInRule" in g.direct_deps("I.rec")
    assert g.graph.has_edge(g.node_of("I"), "OnlyInRule")


def test_undeclared_reference_is_reported_as_missing(builder: ExportBuilder) -> None:
    builder.define("d", builder.sort(0), builder.const("NeverDeclared"))
    g = graph_of(builder)
    assert g.missing == {"NeverDeclared"}
    assert "NeverDeclared" not in g.graph


def test_literal_types_become_missing_in_a_partial_export(builder: ExportBuilder) -> None:
    """Why include_literal_types is a flag: a sliced export may omit Nat."""
    builder.define("d", builder.sort(0), builder.nat(7))
    assert graph_of(builder).missing == {"Nat"}
    assert graph_of(builder, include_literal_types=False).missing == set()


def test_direct_deps_is_ungrouped_and_untransitive(builder: ExportBuilder) -> None:
    s = builder.sort(0)
    builder.axiom("Deep", s)
    builder.define("mid", s, builder.const("Deep"))
    builder.define("top", s, builder.const("mid"))
    g = graph_of(builder)
    assert g.direct_deps("top") == {"mid"}


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Nat.succ", False),
        ("Nat.add_comm", False),
        ("_proof_1", True),
        ("Foo._proof_3", True),
        ("_private.M.0.aux", True),
        ("Foo.match_1", True),
        ("Foo.matcher", False),
        ("Nat.eq_of_le", False),
    ],
)
def test_internal_name_heuristic(name: str, expected: bool) -> None:
    assert is_internal(name) is expected


def test_internal_declarations_stay_in_the_graph(builder: ExportBuilder) -> None:
    """They are required for checking; filtering is a query-level concern."""
    s = builder.sort(0)
    builder.axiom("Real", s)
    builder.define("_proof_1", s, builder.const("Real"))
    builder.theorem("thm", s, builder.const("_proof_1"))
    g = graph_of(builder)
    assert "_proof_1" in g
    assert g.graph.has_edge("thm", "_proof_1")
    assert g.decl("_proof_1").internal


def test_unknown_name_raises_keyerror(builder: ExportBuilder) -> None:
    builder.axiom("A", builder.sort(0))
    g = graph_of(builder)
    for fn in (g.decl, g.node_of, g.direct_deps):
        with pytest.raises(KeyError, match="nope"):
            fn("nope")


def test_len_and_contains(builder: ExportBuilder) -> None:
    s = builder.sort(0)
    builder.axiom("A", s)
    builder.axiom("B", s)
    g = graph_of(builder)
    assert len(g) == 2
    assert "A" in g and "Z" not in g

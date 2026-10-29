"""Query semantics over a built graph."""

from __future__ import annotations

import pytest

from lean_dep_graph.graph import build_graph
from lean_dep_graph.ndjson import load
from lean_dep_graph.queries import (
    axioms,
    closure,
    cycles,
    deps,
    path,
    rclosure,
    rdeps,
    stats,
    topological_order,
)

from .conftest import ExportBuilder


def chain() -> object:
    """Base <- mid <- top, plus an unrelated declaration."""
    b = ExportBuilder()
    s = b.sort(0)
    b.axiom("Base", s)
    b.define("mid", s, b.const("Base"))
    b.define("top", s, b.const("mid"))
    b.axiom("Unrelated", s)
    return build_graph(load(b.lines))


def test_deps_is_direct_only() -> None:
    g = chain()
    assert deps(g, "top") == {"mid"}


def test_closure_is_transitive_and_excludes_self_by_default() -> None:
    g = chain()
    assert closure(g, "top") == {"mid", "Base"}
    assert closure(g, "top", include_self=True) == {"top", "mid", "Base"}


def test_closure_of_a_leaf_is_empty() -> None:
    g = chain()
    assert closure(g, "Base") == set()


def test_closure_includes_group_mates() -> None:
    """A mutual block is only well-formed as a whole."""
    b = ExportBuilder()
    s = b.sort(0)
    b.define("even", s, b.const("odd"), all_=["even", "odd"])
    b.define("odd", s, b.const("even"), all_=["even", "odd"])
    g = build_graph(load(b.lines))
    assert closure(g, "even") == {"odd"}


def test_rdeps_and_rclosure() -> None:
    g = chain()
    assert rdeps(g, "Base") == {"mid"}
    assert rclosure(g, "Base") == {"mid", "top"}
    assert rclosure(g, "top") == set()


def test_unrelated_declarations_do_not_appear() -> None:
    g = chain()
    assert "Unrelated" not in closure(g, "top")
    assert "Unrelated" not in rclosure(g, "Base")


def test_internal_filter_still_traverses_internals() -> None:
    """Skipping internals during traversal would lose what they depend on."""
    b = ExportBuilder()
    s = b.sort(0)
    b.axiom("RealDep", s)
    b.define("_proof_1", s, b.const("RealDep"))
    b.theorem("thm", s, b.const("_proof_1"))
    g = build_graph(load(b.lines))

    assert closure(g, "thm") == {"_proof_1", "RealDep"}
    filtered = closure(g, "thm", internal=False)
    assert filtered == {"RealDep"}, "the internal node is hidden but not short-circuited"


def test_axioms_recurses_through_axiom_types() -> None:
    """Deliberate divergence from `#print axioms`.

    Lean's `collectAxioms` does not collect axioms referenced by other axioms
    (lean4 #8840), so its result is not a sound transitive closure.  Here the
    inner axiom must still be reported.
    """
    b = ExportBuilder()
    s = b.sort(0)
    b.axiom("Inner", s)
    b.axiom("Outer", b.const("Inner"))
    b.theorem("uses", s, b.const("Outer"))
    g = build_graph(load(b.lines))
    assert axioms(g, "uses") == {"Outer", "Inner"}


def test_axioms_of_a_pure_definition_is_empty() -> None:
    b = ExportBuilder()
    s = b.sort(0)
    b.define("pure_a", s, s)
    b.define("pure_b", s, b.const("pure_a"))
    g = build_graph(load(b.lines))
    assert axioms(g, "pure_b") == set()


def test_path_explains_an_indirect_dependency() -> None:
    g = chain()
    assert path(g, "top", "Base") == ["top", "mid", "Base"]
    assert path(g, "Base", "top") is None
    assert path(g, "top", "Unrelated") is None


def test_no_cycles_once_blocks_are_collapsed() -> None:
    b = ExportBuilder()
    s = b.sort(0)
    b.define("p", s, b.const("q"), all_=["p", "q"])
    b.define("q", s, b.const("p"), all_=["p", "q"])
    g = build_graph(load(b.lines))
    assert cycles(g) == []


def test_topological_order_places_dependencies_first() -> None:
    g = chain()
    order = topological_order(g)
    position = {name: i for i, members in enumerate(order) for name in members}
    for src, dst in g.graph.edges():
        for a in g.groups[src].members:
            for b_ in g.groups[dst].members:
                assert position[b_] < position[a], f"{b_} must precede {a}"


def test_topological_order_covers_every_declaration() -> None:
    g = chain()
    flat = [n for members in topological_order(g) for n in members]
    assert sorted(flat) == sorted(g.names)
    assert len(flat) == len(set(flat))


def test_topological_order_emits_blocks_as_single_entries() -> None:
    b = ExportBuilder()
    s = b.sort(0)
    b.define("m1", s, b.const("m2"), all_=["m1", "m2"])
    b.define("m2", s, b.const("m1"), all_=["m1", "m2"])
    g = build_graph(load(b.lines))
    assert topological_order(g) == [("m1", "m2")]


def test_stats_reports_structure() -> None:
    g = chain()
    s = stats(g)
    assert s["declarations"] == 4
    assert s["missing"] == 0
    assert s["blocks"] == 0
    assert s["kind:axiom"] == 2
    assert s["kind:def"] == 2


def test_queries_reject_unknown_names() -> None:
    g = chain()
    for fn in (deps, closure, rdeps, rclosure, axioms):
        with pytest.raises(KeyError):
            fn(g, "nope")

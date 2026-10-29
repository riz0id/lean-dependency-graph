"""Stability tests against the committed fixture.

These pin end-to-end behaviour of the whole pipeline on a fixed input, so a
refactor that silently changes grouping or edge resolution fails here.
Regenerate with `uv run python -m tests.generate_fixture`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import lean_dep_graph as ldg
from lean_dep_graph.queries import stats

FIXTURE = Path(__file__).parent / "fixtures" / "sample.ndjson"


@pytest.fixture(scope="module")
def g() -> ldg.DeclGraph:
    return ldg.graph_for_export_file(FIXTURE)


def test_fixture_is_committed() -> None:
    assert FIXTURE.exists(), "run `uv run python -m tests.generate_fixture`"


def test_no_dangling_references(g: ldg.DeclGraph) -> None:
    """Every referenced constant resolves. A miss means the export is partial."""
    assert g.missing == set()


def test_shape_is_stable(g: ldg.DeclGraph) -> None:
    s = stats(g)
    assert s == {
        # 4 axioms + 4 ctors + 10 defs + 3 inductives + 1 opaque
        # + 4 quot + 3 recursors + 1 theorem
        "declarations": 30,
        # 14 singletons plus 4 blocks: Prod (3), Tree/Forest (7),
        # the quotient package (4), and even/odd (2).
        "nodes": 18,
        "edges": 17,
        "blocks": 4,
        "internal": 1,
        "missing": 0,
        "kind:axiom": 4,
        "kind:ctor": 4,
        "kind:def": 10,
        "kind:inductive": 3,
        "kind:opaque": 1,
        "kind:quot": 4,
        "kind:rec": 3,
        "kind:thm": 1,
    }


def test_graph_is_acyclic_once_blocks_collapse(g: ldg.DeclGraph) -> None:
    assert ldg.cycles(g) == []


def test_topological_order_is_complete(g: ldg.DeclGraph) -> None:
    flat = [n for members in ldg.topological_order(g) for n in members]
    assert sorted(flat) == sorted(g.names)


def test_mutual_inductive_block(g: ldg.DeclGraph) -> None:
    assert set(g.members("Forest.cons")) == {
        "Tree",
        "Forest",
        "Tree.node",
        "Forest.nil",
        "Forest.cons",
        "Tree.rec",
        "Forest.rec",
    }


def test_mutual_definition_block(g: ldg.DeclGraph) -> None:
    assert set(g.members("odd")) == {"even", "odd"}


def test_quotient_block(g: ldg.DeclGraph) -> None:
    assert set(g.members("Quot.ind")) == {"Quot", "Quot.mk", "Quot.lift", "Quot.ind"}


def test_literal_induces_its_type(g: ldg.DeclGraph) -> None:
    assert "Nat" in g.direct_deps("nat_lit")
    assert "String" in g.direct_deps("str_lit")


def test_projection_induces_its_structure(g: ldg.DeclGraph) -> None:
    assert "Prod" in g.direct_deps("projects")


def test_binder_shadowing_a_constant_adds_no_edge(g: ldg.DeclGraph) -> None:
    """The lambda binds a variable literally named `Nat`.

    `Nat` is still a dependency here via the declaration's type, so the check
    is that the dependency set is exactly that and nothing leaked in from the
    binder position.
    """
    assert g.direct_deps("lambda_shadowing_Nat") == {"Nat"}


def test_axioms_recurse_through_axioms(g: ldg.DeclGraph) -> None:
    assert ldg.axioms(g, "uses_internal") == {"DependsOnChoice", "Choice"}


def test_internal_filter_preserves_transitive_reach(g: ldg.DeclGraph) -> None:
    full = ldg.closure(g, "uses_internal")
    assert "_proof_1" in full
    hidden = ldg.closure(g, "uses_internal", internal=False)
    assert "_proof_1" not in hidden
    assert {"DependsOnChoice", "Choice"} <= hidden


def test_closure_through_a_chain(g: ldg.DeclGraph) -> None:
    closure = ldg.closure(g, "top")
    assert {"sealed", "nat_lit", "projects", "Nat", "Prod"} <= closure


def test_path_explains_a_dependency(g: ldg.DeclGraph) -> None:
    chain = ldg.path(g, "top", "Nat")
    assert chain is not None and chain[0] == "top" and chain[-1] == "Nat"

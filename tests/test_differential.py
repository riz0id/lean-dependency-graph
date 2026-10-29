"""Differential tests against lean4export itself.

The exporter accepts ``-- <decl>`` and computes the transitive closure on its
own, which makes it a free and exact oracle: for any declaration ``d``, the set
of declarations in a single-declaration export must equal
``closure(d) | {d}``.  No hand-written expectations are involved, so this
catches precisely the extraction mistakes that are otherwise silent -- a
dropped ``Expr.proj`` structure name, or a dropped literal type.

Requires a Lean toolchain and a ``lean4export`` binary built against the same
toolchain as ``tests/lean``.  Deselected by default; run with ``-m lean``.
Point ``LEAN4EXPORT`` at the binary, or put it on ``PATH``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

import lean_dep_graph as ldg
from lean_dep_graph.export import Exporter, LeanProject
from lean_dep_graph.graph import build_graph

pytestmark = pytest.mark.lean

PROJECT = Path(__file__).parent / "lean"

#: Declarations spanning each extraction path. Names are from tests/lean.
SAMPLE = [
    "DepFixture.fromNatLiteral",
    "DepFixture.fromStrLiteral",
    "DepFixture.projectsOut",
    "DepFixture.withLet",
    "DepFixture.applyTwice",
    "DepFixture.sealed",
    "DepFixture.sealed_eq",
    "DepFixture.usesAxiom",
    "DepFixture.mystery",
    "DepFixture.Point",
    "DepFixture.countDown",
    "DepFixture.Tree",
    "DepFixture.isEven",
    "DepFixture.usesBasic",
]


def _binary() -> Path:
    env = os.environ.get("LEAN4EXPORT")
    if env:
        return Path(env)
    found = shutil.which("lean4export")
    if found:
        return Path(found)
    pytest.skip("lean4export not found; set LEAN4EXPORT or add it to PATH")


@pytest.fixture(scope="module")
def project() -> LeanProject:
    if shutil.which("lake") is None:
        pytest.skip("lake not on PATH")
    proj = LeanProject(PROJECT.resolve())
    proj.build("DepFixture")
    return proj


@pytest.fixture(scope="module")
def exporter() -> Exporter:
    return Exporter.discover(_binary())


@pytest.fixture(scope="module")
def modules(project: LeanProject) -> list[str]:
    mods = project.modules("DepFixture")
    assert mods, "lake reported no modules for DepFixture"
    missing = project.olean_missing(mods)
    assert not missing, f"no .olean for: {missing}"
    return mods


@pytest.fixture(scope="module")
def full(project: LeanProject, exporter: Exporter, modules: list[str]) -> ldg.DeclGraph:
    return build_graph(exporter.export(project, modules))


def test_export_is_self_contained(full: ldg.DeclGraph) -> None:
    """A whole-module export should reference nothing it does not declare."""
    assert full.missing == set(), sorted(full.missing)[:20]


def test_graph_is_acyclic(full: ldg.DeclGraph) -> None:
    assert ldg.cycles(full) == []


def test_every_declaration_is_orderable(full: ldg.DeclGraph) -> None:
    flat = [n for members in ldg.topological_order(full) for n in members]
    assert sorted(flat) == sorted(full.names)


def test_sample_declarations_exist(full: ldg.DeclGraph) -> None:
    """Guards the sample list against drift in tests/lean."""
    absent = [n for n in SAMPLE if n not in full]
    assert not absent, f"not in export (fixture changed?): {absent}"


@pytest.mark.parametrize("name", SAMPLE)
def test_closure_matches_single_declaration_export(
    name: str,
    full: ldg.DeclGraph,
    project: LeanProject,
    exporter: Exporter,
    modules: list[str],
) -> None:
    if name not in full:
        pytest.skip(f"{name} absent from export")

    sliced = build_graph(exporter.export(project, modules, decls=[name]))
    expected = set(sliced.names)
    computed = ldg.closure(full, name, include_self=True)

    only_exporter = sorted(expected - computed)
    only_computed = sorted(computed - expected)
    assert not only_exporter and not only_computed, (
        f"{name}\n"
        f"  missed by extraction ({len(only_exporter)}): {only_exporter[:15]}\n"
        f"  invented by extraction ({len(only_computed)}): {only_computed[:15]}"
    )


def test_mutual_inductive_is_one_block(full: ldg.DeclGraph) -> None:
    members = set(full.members("DepFixture.Tree"))
    assert "DepFixture.Forest" in members
    assert full.node_of("DepFixture.Forest") == full.node_of("DepFixture.Tree")


def test_mutual_definition_is_one_block(full: ldg.DeclGraph) -> None:
    assert "DepFixture.isOdd" in set(full.members("DepFixture.isEven"))


def test_structure_members_share_a_block(full: ldg.DeclGraph) -> None:
    members = set(full.members("DepFixture.Point"))
    assert "DepFixture.Point.mk" in members


def test_axiom_closure_is_transitive(full: ldg.DeclGraph) -> None:
    found = ldg.axioms(full, "DepFixture.usesAxiom")
    assert "DepFixture.mysteryUser" in found
    assert "DepFixture.mystery" in found, "axioms referenced by axioms must be reached"


def test_pure_definition_rests_on_no_axioms(full: ldg.DeclGraph) -> None:
    assert ldg.axioms(full, "DepFixture.fromNatLiteral") == set()


def test_cross_module_closure(full: ldg.DeclGraph) -> None:
    assert "DepFixture.withLet" in ldg.closure(full, "DepFixture.usesBasic")

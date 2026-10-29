"""Queries over a :class:`~lean_dep_graph.graph.DeclGraph`.

Edges point from a dependent to its dependency, so forward reachability
answers "what does this need" and reverse reachability answers "what would
break if this changed".
"""

from __future__ import annotations

import networkx as nx

from .graph import DeclGraph, is_internal

__all__: list[str] = [
    "CyclicGraphError",
    "axioms",
    "closure",
    "cycles",
    "deps",
    "path",
    "rclosure",
    "rdeps",
    "stats",
    "topological_order",
]


class CyclicGraphError(Exception):
    """Raised when an operation requiring acyclicity finds a cycle."""


def expand(g: DeclGraph, nodes: set[str]) -> set[str]:
    out: set[str] = set()

    for node in nodes:
        out.update(g.groups[node].members)

    return out


def filter_internal(names: set[str], *, internal: bool) -> set[str]:
    return names if internal else {n for n in names if not is_internal(n)}


def deps(g: DeclGraph, name: str, *, internal: bool = True) -> set[str]:
    """Declarations directly depended on by ``name``'s group."""
    node: str = g.node_of(name)
    result: set[str] = expand(g, nodes=set(g.graph.successors(n=node)))
    return filter_internal(result, internal=internal)


def closure(
    g: DeclGraph, name: str, *, include_self: bool = False, internal: bool = True
) -> set[str]:
    node: str = g.node_of(name)

    reachable: set[str] = set(nx.descendants(g.graph, node))
    reachable.add(node)

    result: set[str] = expand(g, reachable)

    if not include_self:
        result.discard(name)

    return filter_internal(result, internal=internal)


def rdeps(g: DeclGraph, name: str, *, internal: bool = True) -> set[str]:
    node: str = g.node_of(name)

    result: set[str] = expand(g, set(g.graph.predecessors(node)))
    result.discard(name)

    return filter_internal(names=result, internal=internal)


def rclosure(
    g: DeclGraph, name: str, *, include_self: bool = False, internal: bool = True
) -> set[str]:
    node: str = g.node_of(name)
    reachable: set[str] = set(nx.ancestors(g.graph, node))
    reachable.add(node)
    result: set[str] = expand(g, reachable)

    if not include_self:
        result.discard(name)
    return filter_internal(result, internal=internal)


def axioms(g: DeclGraph, name: str) -> set[str]:
    return {n for n in closure(g, name, include_self=True) if g.decls[n].kind == "axiom"}


def path(g: DeclGraph, frm: str, to: str) -> list[str] | None:
    src: str = g.node_of(frm)
    dst: str = g.node_of(to)

    try:
        return list(nx.shortest_path(g.graph, src, dst))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def cycles(g: DeclGraph) -> list[tuple[str, ...]]:
    return [tuple(sorted(scc)) for scc in nx.strongly_connected_components(g.graph) if len(scc) > 1]


def topological_order(g: DeclGraph) -> list[tuple[str, ...]]:
    try:
        order: list[str] = list(nx.topological_sort(g.graph))
    except nx.NetworkXUnfeasible as exc:
        raise CyclicGraphError(f"graph is cyclic; offending components: {cycles(g)[:3]}") from exc

    return [g.groups[node].members for node in reversed(order)]


def stats(g: DeclGraph) -> dict[str, int]:
    """Summary counts, for smoke tests and sanity checks."""
    kinds: dict[str, int] = {}

    for decl in g.decls.values():
        kinds[decl.kind] = kinds.get(decl.kind, 0) + 1

    return {
        "declarations": len(g.decls),
        "nodes": g.graph.number_of_nodes(),
        "edges": g.graph.number_of_edges(),
        "blocks": sum(1 for grp in g.groups.values() if grp.is_block),
        "internal": sum(1 for n in g.decls if is_internal(n)),
        "missing": len(g.missing),
        **{f"kind:{k}": v for k, v in sorted(kinds.items())},
    }

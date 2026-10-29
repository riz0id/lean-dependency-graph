"""Queries over a :class:`~lean_dep_graph.graph.DeclGraph`.

Edges point from a dependent to its dependency, so forward reachability
answers "what does this need" and reverse reachability answers "what would
break if this changed".
"""

from __future__ import annotations

import networkx as nx

from .graph import DeclGraph, is_internal


class CyclicGraphError(Exception):
    """Raised when an operation requiring acyclicity finds a cycle."""


def _expand(g: DeclGraph, nodes: set[str]) -> set[str]:
    """Group ids -> all member declaration names."""
    out: set[str] = set()
    for node in nodes:
        out.update(g.groups[node].members)
    return out


def _filtered(names: set[str], *, internal: bool) -> set[str]:
    return names if internal else {n for n in names if not is_internal(n)}


def deps(g: DeclGraph, name: str, *, internal: bool = True) -> set[str]:
    """Declarations directly depended on by ``name``'s group."""
    node = g.node_of(name)
    result = _expand(g, set(g.graph.successors(node)))
    return _filtered(result, internal=internal)


def closure(
    g: DeclGraph, name: str, *, include_self: bool = False, internal: bool = True
) -> set[str]:
    """Every declaration the kernel must have to accept ``name``.

    Includes the other members of ``name``'s own group, since a mutual block is
    only well-formed as a whole.  Internal declarations are always traversed
    even when ``internal=False`` filters them from the result -- skipping them
    would discard whatever *they* depend on.
    """
    node = g.node_of(name)
    reachable = set(nx.descendants(g.graph, node))
    reachable.add(node)
    result = _expand(g, reachable)
    if not include_self:
        result.discard(name)
    return _filtered(result, internal=internal)


def rdeps(g: DeclGraph, name: str, *, internal: bool = True) -> set[str]:
    """Declarations that directly depend on ``name``'s group."""
    node = g.node_of(name)
    result = _expand(g, set(g.graph.predecessors(node)))
    result.discard(name)
    return _filtered(result, internal=internal)


def rclosure(
    g: DeclGraph, name: str, *, include_self: bool = False, internal: bool = True
) -> set[str]:
    """Every declaration that transitively depends on ``name``."""
    node = g.node_of(name)
    reachable = set(nx.ancestors(g.graph, node))
    reachable.add(node)
    result = _expand(g, reachable)
    if not include_self:
        result.discard(name)
    return _filtered(result, internal=internal)


def axioms(g: DeclGraph, name: str) -> set[str]:
    """Axioms in ``name``'s closure.

    Computed as a genuine transitive closure, which deliberately diverges from
    ``Lean.collectAxioms`` and therefore from ``#print axioms``: that function
    does not collect axioms referenced by other axioms (lean4 issue #8840), so
    its output is not a sound closure.
    """
    found = {n for n in closure(g, name, include_self=True) if g.decls[n].kind == "axiom"}
    return found


def path(g: DeclGraph, frm: str, to: str) -> list[str] | None:
    """A shortest dependency chain from ``frm`` to ``to``, as group ids.

    Answers "why does this depend on that".  ``None`` if no chain exists.
    """
    src, dst = g.node_of(frm), g.node_of(to)
    try:
        return list(nx.shortest_path(g.graph, src, dst))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def cycles(g: DeclGraph) -> list[tuple[str, ...]]:
    """Strongly connected components larger than one node.

    Should always be empty: genuine mutual recursion is collapsed into a single
    group node during construction, so a result here means either a grouping
    constraint was missed or extraction invented an edge.  Reported as SCCs
    rather than enumerated simple cycles, which can be exponential.
    """
    return [tuple(sorted(scc)) for scc in nx.strongly_connected_components(g.graph) if len(scc) > 1]


def topological_order(g: DeclGraph) -> list[tuple[str, ...]]:
    """Groups in an order the kernel could accept them in.

    Dependencies come before dependents, so this reverses the edge direction
    used internally.  Each entry is the member names of one group, which must
    be added together.
    """
    try:
        order = list(nx.topological_sort(g.graph))
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

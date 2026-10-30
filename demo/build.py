"""Reduce a ``lean4export`` NDJSON stream into binary buffers for the viewer.

The library's own loader cannot be used at this scale.  ``ndjson.load`` keeps
every expression as a ``dict[int, dict[str, Any]]``; a whole-environment
Mathlib export holds tens of millions of them, which is tens of GB of nested
Python dicts.  This module keeps the same records in two flat ``array``
tables at 16 bytes per expression and walks them with an explicit stack.

What it does **not** reimplement is grouping.  ``parse_decls`` and
``build_groups`` read only ``Export.decls`` and ``Export.name`` -- never the
expression table -- so both are imported and used verbatim.  Mutual blocks,
quotient blocks, declaration kinds and recursor-rule roots are therefore
whatever the library says they are, not a second opinion.  Only the dependency
walk is new, and ``tests/test_demo_reduce.py`` pins it against
``terms.collect_expr_deps`` declaration by declaration.

Output is a directory of little-endian binary buffers plus ``meta.json``;
``demo/web`` fetches them as ``ArrayBuffer``s.  Run it as:

    lake env lean4export Mathlib | python -m demo.build - --out demo/web/data

Reading from a pipe is the documented path because it means a multi-GB export
never has to land on disk.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from array import array
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, cast

import networkx as nx
import numpy as np
from networkx.algorithms.community import louvain_communities, modularity
from numpy.typing import NDArray

from lean_dep_graph.graph import Decl, Group, build_groups, is_internal, parse_decls
from lean_dep_graph.ndjson import DeclKey, Export, ExportFormatError, ExprKey

__all__ = [
    "Adjacency",
    "DrawOrder",
    "ExprTable",
    "GroupGraph",
    "Level",
    "adjacency",
    "bfs_hops",
    "build_group_graph",
    "coarsen",
    "community_bands",
    "contract",
    "draw_order",
    "drawn_csr",
    "force_refine",
    "hub_mask",
    "layered_layout",
    "longest_path_depth",
    "match_heavy",
    "mean_edge_length",
    "multilevel_layout",
    "normalize_positions",
    "pivot_distances",
    "pivot_mds",
    "read_stream",
    "reduce_expr",
    "repulsion",
    "transitive_reduction_mask",
    "write_buffers",
]

# `nameref` sentinels.  Folding the literal-type implication into this one
# field is what lets the kind array go away entirely: nothing downstream ever
# needs to know an expression was an `app` rather than a `lam`, only what it
# refers to and where its children are.
NO_REF = -1  #: references nothing (bvar, sort, and every interior node)
IMPLIES_NAT = -2  #: natVal: needs `Nat` though no const node appears
IMPLIES_STR = -3  #: strVal: needs `String`, likewise
UNSET = -4  #: no record was ever stored at this index

#: Literal-type sentinel -> the constant it implies.  Mirrors `terms.NAT_TYPE`
#: and `terms.STR_TYPE`, which are not imported only because the walk deals in
#: sentinels rather than strings until the very end.
IMPLIED: dict[int, str] = {IMPLIES_NAT: "Nat", IMPLIES_STR: "String"}

#: Grown in blocks of this many expressions.  Small enough that the zero-fill
#: templates stay cheap, large enough that growth is not in the hot path.
BLOCK = 1 << 20

NODE_FLAG_INTERNAL = 1
NODE_FLAG_BLOCK = 2
NODE_FLAG_UNSAFE = 4
#: In the top `--hub-count` constants by in-degree, so its incoming edges are
#: not drawn.  The node and its true degree are untouched; see `hub_mask`.
NODE_FLAG_HUB = 8


def loader() -> Callable[[bytes], Any]:
    """``orjson.loads`` if it is installed, else the stdlib's.

    Parsing is a per-line cost over the whole export, so the faster parser is
    worth reaching for -- but it is an optional extra, not a requirement.
    """
    try:
        import orjson
    except ModuleNotFoundError:
        return json.loads
    return orjson.loads


LOADS = loader()


def reduce_expr(rec: Mapping[str, Any]) -> tuple[int, int, int, int]:
    """``(child0, child1, child2, nameref)`` for one expression record.

    Mirrors ``terms.expr_children`` and ``terms.expr_own_consts`` case for
    case, and the two easy-to-miss cases they exist to get right are the two
    worth naming here: ``proj`` contributes its ``typeName``, and
    ``natVal``/``strVal`` imply ``Nat``/``String`` even though no ``const``
    node appears anywhere in the term.

    Conversely ``lam``/``forallE``/``letE`` each carry a ``name`` that is a
    *binder*, not a constant reference, and it is dropped deliberately:
    treating it as a dependency would add a spurious edge to every function
    type in the export.

    Unlike the library, this raises on an unrecognised kind rather than
    returning no children -- the check ``terms.check_expr_keys`` was written
    for but never wired into.  A silent no-children answer here would
    undercount dependencies, which is the one failure mode that produces a
    wrong graph rather than an error.
    """
    match rec:
        case {ExprKey.APP: app}:
            return int(app["fn"]), int(app["arg"]), -1, NO_REF
        case {ExprKey.LAM: binder} | {ExprKey.FORALL: binder}:
            return int(binder["type"]), int(binder["body"]), -1, NO_REF
        case {ExprKey.LET: le}:
            return int(le["type"]), int(le["value"]), int(le["body"]), NO_REF
        case {ExprKey.PROJ: proj}:
            return int(proj["struct"]), -1, -1, int(proj["typeName"])
        case {ExprKey.MDATA: mdata}:
            return int(mdata["expr"]), -1, -1, NO_REF
        case {ExprKey.CONST: const}:
            return -1, -1, -1, int(const["name"])
        case {ExprKey.NAT_VAL: _}:
            return -1, -1, -1, IMPLIES_NAT
        case {ExprKey.STR_VAL: _}:
            return -1, -1, -1, IMPLIES_STR
        case {ExprKey.BVAR: _} | {ExprKey.SORT: _}:
            return -1, -1, -1, NO_REF
        case _:
            raise ExportFormatError(f"expression record {rec.get('ie')} has no known kind")


class ExprTable:
    """Flat storage for reduced expression records.

    Two ``array`` tables rather than ``numpy`` ones: the dependency walk reads
    single elements in a Python loop, where ``array``'s int boxing is markedly
    cheaper than numpy's scalar boxing.  numpy earns its place later, on the
    whole-array edge and layout work.
    """

    def __init__(self, block: int = BLOCK) -> None:
        self.block = block
        self.child: array[int] = array("i")
        self.ref: array[int] = array("i")
        self.size = 0
        self._child_fill: array[int] = array("i", [-1]) * (block * 3)
        self._ref_fill: array[int] = array("i", [UNSET]) * block

    def grow_to(self, idx: int) -> None:
        while self.size <= idx:
            self.child.extend(self._child_fill)
            self.ref.extend(self._ref_fill)
            self.size += self.block

    def put(self, idx: int, c0: int, c1: int, c2: int, ref: int) -> None:
        if idx < 0:
            raise ExportFormatError(f"negative expression index {idx}")
        if idx >= self.size:
            self.grow_to(idx)
        base = idx * 3
        self.child[base] = c0
        self.child[base + 1] = c1
        self.child[base + 2] = c2
        self.ref[idx] = ref

    def nbytes(self) -> int:
        return self.child.buffer_info()[1] * 4 + self.ref.buffer_info()[1] * 4

    def deps(self, roots: Iterable[int], stamp: array[int], mark: int) -> set[int]:
        """Name references reachable from ``roots``, as indices or sentinels.

        Mirrors ``terms.collect_expr_deps``: explicit stack, revisit
        suppression, children pushed only when unseen.  The one difference is
        that revisits are suppressed with a stamp array -- ``stamp[i] == mark``
        means "visited during this declaration" -- rather than a fresh ``set``
        per declaration.  Same result, no per-declaration allocation, which
        matters when the walk runs once per declaration over a shared DAG.
        """
        found: set[int] = set()
        child = self.child
        ref = self.ref
        size = self.size
        stack = [int(r) for r in roots]

        while stack:
            idx = stack.pop()

            if idx < 0:
                continue
            if idx >= size or ref[idx] == UNSET:
                raise ExportFormatError(f"dangling expression index {idx}")
            if stamp[idx] == mark:
                continue
            stamp[idx] = mark

            reference = ref[idx]
            if reference != NO_REF:
                found.add(reference)

            base = idx * 3
            for off in range(3):
                kid = child[base + off]
                if kid >= 0 and stamp[kid] != mark:
                    stack.append(kid)

        return found


@dataclass
class Counts:
    """What the stream contained, for `--stats` and for `meta.json`."""

    lines: int = 0
    names: int = 0
    levels: int = 0
    exprs: int = 0
    decls: int = 0
    seconds: float = 0.0


def read_stream(lines: Iterable[bytes], *, progress: int = 0) -> tuple[Export, ExprTable, Counts]:
    """Single pass over NDJSON, reducing expressions and retaining the rest.

    Dispatch order mirrors ``ndjson.load`` but is ordered by frequency:
    expressions dominate a real export by an order of magnitude, so they are
    tested first.  Universe levels are counted and dropped -- they carry no
    constant references, and the library never reads ``Export.levels`` either.
    """
    ex = Export()
    table = ExprTable()
    counts = Counts()
    started = time.monotonic()

    for raw in lines:
        text = raw.strip()

        if not text:
            continue

        counts.lines += 1
        rec = LOADS(text)

        if not isinstance(rec, dict):
            raise ExportFormatError(f"line {counts.lines}: expected a JSON object")

        if "ie" in rec:
            table.put(int(rec["ie"]), *reduce_expr(rec))
            counts.exprs += 1
        elif "in" in rec:
            ex.names[int(rec["in"])] = rec
            counts.names += 1
        elif "il" in rec:
            counts.levels += 1
        elif "meta" in rec:
            ex.meta = rec["meta"]
        else:
            for key in DeclKey:
                if key in rec:
                    ex.decls.append((key, rec[key]))
                    counts.decls += 1
                    break
            else:
                raise ExportFormatError(f"unrecognised record: {sorted(rec)[:4]}")

        if progress and counts.lines % progress == 0:
            rate = counts.lines / max(time.monotonic() - started, 1e-9)
            print(
                f"  {counts.lines:,} lines  {counts.exprs:,} exprs  "
                f"{rate / 1000:.0f}k lines/s  {table.nbytes() / 1e9:.2f} GB table",
                file=sys.stderr,
            )

    counts.seconds = time.monotonic() - started
    return ex, table, counts


@dataclass
class GroupGraph:
    """The declaration graph collapsed to kernel-acceptance groups.

    Node order is ``build_groups`` insertion order; every array below is
    indexed by node, and ``edges`` holds deduplicated ``(src, dst)`` pairs
    sorted lexicographically.
    """

    ids: list[str]
    kinds: list[str]
    members: list[list[str]]
    flags: list[int]
    edges: NDArray[np.uint32]
    #: Declarations before grouping. Higher than the number of declaration
    #: *records* in the stream: one `inductive` record carries the block's
    #: types, constructors and recursors.
    declarations: int = 0
    missing: set[str] = field(default_factory=set)
    decl_deps: dict[str, set[str]] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.ids)


def build_group_graph(
    ex: Export,
    table: ExprTable,
    *,
    include_literal_types: bool = True,
    keep_decl_deps: bool = False,
) -> GroupGraph:
    """Group the declarations, then resolve every dependency to a group edge.

    ``parse_decls`` and ``build_groups`` are the library's own, unmodified.
    ``keep_decl_deps`` retains the per-declaration dependency sets so the
    anchor test can compare them against ``collect_expr_deps``; at Mathlib
    scale it would hold hundreds of MB of strings, so it defaults off.
    """
    decls: dict[str, Decl]
    constraints: list[tuple[str, list[str]]]
    decls, constraints = parse_decls(ex)
    groups: dict[str, Group]
    node_of: dict[str, str]
    groups, node_of = build_groups(decls, constraints)

    ids = list(groups)
    index_of = {gid: i for i, gid in enumerate(ids)}

    graph = GroupGraph(
        ids=ids,
        kinds=[groups[gid].kind for gid in ids],
        members=[groups[gid].members for gid in ids],
        flags=[0] * len(ids),
        edges=np.empty((0, 2), dtype=np.uint32),
        declarations=len(decls),
    )

    for i, gid in enumerate(ids):
        group = groups[gid]
        flags = NODE_FLAG_BLOCK if group.is_block else 0
        if any(is_internal(m) for m in group.members):
            flags |= NODE_FLAG_INTERNAL
        if any(decls[m].is_unsafe for m in group.members):
            flags |= NODE_FLAG_UNSAFE
        graph.flags[i] = flags

    # A reference resolves the same way every time it is seen, and a popular
    # constant is seen millions of times, so the name index -> node index map
    # is memoised. -1 marks a reference nothing in this export declares.
    resolved: dict[int, int] = {}
    stamp: array[int] = array("i", bytes(4 * (table.size + 1)))
    packed = array("q")

    def name_of(reference: int) -> str:
        """The constant a reference denotes, or ``""`` for a dropped literal type."""
        if reference not in IMPLIED:
            return ex.name(reference)
        return IMPLIED[reference] if include_literal_types else ""

    for mark, (name, decl) in enumerate(decls.items(), start=1):
        src = index_of[node_of[name]]
        # Walked once: a second `deps` call with the same mark would find
        # every expression already stamped and return nothing.
        references = table.deps(decl.roots, stamp, mark)

        for reference in references:
            hit = resolved.get(reference)

            if hit is None:
                dep = name_of(reference)
                gid = node_of.get(dep)
                if gid is None:
                    if dep:
                        graph.missing.add(dep)
                    hit = -1
                else:
                    hit = index_of[gid]
                resolved[reference] = hit

            if hit >= 0 and hit != src:
                packed.append((src << 32) | hit)

        if keep_decl_deps:
            graph.decl_deps[name] = {d for d in map(name_of, references) if d}

    graph.edges = dedupe_edges(packed)
    return graph


def dedupe_edges(packed: array[int]) -> NDArray[np.uint32]:
    """Sort and deduplicate ``src << 32 | dst`` pairs into an ``(E, 2)`` array.

    Packing into one integer rather than collecting a ``set`` of tuples is
    what keeps this affordable: ten million Python tuples cost over a GB,
    while the same edges cost 80 MB here and ``np.unique`` sorts them in
    place.  Sorting by the packed value also sorts by ``(src, dst)``, which is
    exactly the order the forward CSR needs.
    """
    if not len(packed):
        return np.empty((0, 2), dtype=np.uint32)

    raw = np.unique(np.frombuffer(packed, dtype=np.int64))
    out = np.empty((raw.size, 2), dtype=np.uint32)
    out[:, 0] = (raw >> 32).astype(np.uint32)
    out[:, 1] = (raw & 0xFFFFFFFF).astype(np.uint32)
    return out


def csr(
    n: int, row: NDArray[np.uint32], col: NDArray[np.uint32]
) -> tuple[
    NDArray[np.uint32],
    NDArray[np.uint32],
]:
    """Compressed adjacency for ``row -> col``, sorting only when needed."""
    order = np.argsort(row, kind="stable")
    indices = np.ascontiguousarray(col[order], dtype=np.uint32)
    indptr = np.zeros(n + 1, dtype=np.uint32)
    indptr[1:] = np.cumsum(np.bincount(row, minlength=n)).astype(np.uint32)
    return indptr, indices


def int_max(values: NDArray[np.int32]) -> int:
    """``values.max()`` as a plain ``int``, and ``0`` for an empty array.

    Goes through ``tolist`` rather than ``ndarray.max`` because numpy's
    overloads for that method do not resolve under ``[tool.ty.rules]
    all = "error"``.  Called a handful of times on one array, so the cost is
    irrelevant and the type is exact.
    """
    return int(max(values.tolist(), default=0))


@dataclass
class Adjacency:
    """Both CSR directions, built once and shared by the layout and the writer.

    Forward answers "what does this need", reverse answers "what breaks if it
    changes" -- the same asymmetry ``queries`` documents.  Both ship to the
    browser so the viewer can run either closure without sorting ten million
    edges in JavaScript at load time.
    """

    fwd_indptr: NDArray[np.uint32]
    fwd_indices: NDArray[np.uint32]
    rev_indptr: NDArray[np.uint32]
    rev_indices: NDArray[np.uint32]


def adjacency(graph: GroupGraph) -> Adjacency:
    src = np.ascontiguousarray(graph.edges[:, 0])
    dst = np.ascontiguousarray(graph.edges[:, 1])
    fwd_indptr, fwd_indices = csr(graph.n, src, dst)
    rev_indptr, rev_indices = csr(graph.n, dst, src)
    return Adjacency(fwd_indptr, fwd_indices, rev_indptr, rev_indices)


def longest_path_depth(n: int, adj: Adjacency) -> tuple[NDArray[np.int32], int]:
    """Longest path from each node to a sink, by Kahn ordering from the sinks.

    Not ``queries.topological_order``: that routes through networkx, which
    cannot hold a graph of this size.  Returns the depths and the number of
    nodes left unranked -- those sit in cycles, which a correct export should
    not contain but a demo must not die on, so they are parked one level above
    everything else.
    """
    depth = np.zeros(n, dtype=np.int32)

    if n == 0:
        return depth, 0

    pending = (adj.fwd_indptr[1:] - adj.fwd_indptr[:-1]).astype(np.int64)
    starts = adj.rev_indptr.tolist()
    rev_indices = adj.rev_indices
    queue = np.flatnonzero(pending == 0).tolist()
    ranked = 0

    while queue:
        node = queue.pop()
        ranked += 1
        here = int(depth[node]) + 1

        for pred in rev_indices[starts[node] : starts[node + 1]].tolist():
            if depth[pred] < here:
                depth[pred] = here
            pending[pred] -= 1
            if pending[pred] == 0:
                queue.append(pred)

    stranded = n - ranked
    if stranded:
        depth[pending > 0] = int_max(depth) + 1

    return depth, stranded


def check_csr_alignment(graph: GroupGraph, adj: Adjacency) -> None:
    """Fail loudly if ``graph.edges`` rows are not the forward CSR's rows.

    Every mask below is computed over the CSR and then applied to ``edges``
    positionally, which is sound only because `dedupe_edges` sorts by the
    packed ``src << 32 | dst`` while `csr` stable-argsorts on ``src``, leaving
    the two in the same order.  That is an invariant between two functions
    which never call each other, so it is checked rather than trusted: a
    change to either would otherwise misalign every edge in silence.
    """
    n = graph.n
    counts = np.diff(adj.fwd_indptr.astype(np.int64))
    rows = np.repeat(np.arange(n, dtype=np.uint32), counts)

    if not np.array_equal(graph.edges[:, 0], rows):
        raise ExportFormatError("edge rows are not grouped by source as the forward CSR is")
    if not np.array_equal(graph.edges[:, 1], adj.fwd_indices):
        raise ExportFormatError("edge columns do not match the forward CSR indices")


def transitive_reduction_mask(graph: GroupGraph, adj: Adjacency) -> NDArray[np.bool_]:
    """Mark the edges that are **not** implied by a two-step path.

    Drops ``(u, v)`` when some ``w`` in ``out(u)`` has ``v`` in ``out(w)``.
    Reachability is untouched, because the witness ``u -> w -> v`` survives the
    drop -- a closure walked over the kept edges answers exactly what one
    walked over all of them does.  That is what makes this usable as a drawing
    filter rather than a lossy sample: in core Lean 70% of the edges are
    shortcuts around a path that is already on screen.

    Masks are decided against the *original* graph in one pass, never
    iteratively, so no witness can itself be removed.

    Not the full transitive reduction, which needs reachability for every pair;
    one step removes the great majority for one pass over the CSR.
    """
    check_csr_alignment(graph, adj)

    indptr = adj.fwd_indptr.astype(np.int64)
    indices = adj.fwd_indices
    keep = np.ones(int(graph.edges.shape[0]), dtype=np.bool_)
    seen = np.zeros(graph.n, dtype=np.bool_)

    for u in range(graph.n):
        lo, hi = int(indptr[u]), int(indptr[u + 1])
        if lo == hi:
            continue
        out = indices[lo:hi]
        reach = [indices[indptr[w] : indptr[w + 1]] for w in out.tolist()]

        for step in reach:
            seen[step] = True
        keep[lo:hi] = ~seen[out]
        # Clear only what was set: a fresh array per node would dominate the
        # runtime at 58k nodes.
        for step in reach:
            seen[step] = False

    return keep


#: An eliding candidate must absorb at least this multiple of the average
#: in-degree.  Elision only pays for itself where a node has stopped being a
#: dependency and become infrastructure; below that there is nothing
#: disproportionate to hide, and the floor is what stops `--hub-count` from
#: swallowing a small graph whole -- on the 18-node fixture it selects nothing,
#: while on core Lean it sits at 192 against a top-100 threshold of 2,029 and
#: so never binds.
HUB_MIN_MULTIPLE = 10


def hub_mask(graph: GroupGraph, adj: Adjacency, *, count: int) -> tuple[NDArray[np.bool_], int]:
    """The ``count`` most depended-on nodes, and the in-degree that qualifies.

    These are the universal constants -- ``Eq``, ``Nat``, ``OfNat.ofNat`` --
    and in core Lean the top hundred absorb 56.6% of every edge in the graph.
    Drawing those edges conveys only "everything needs equality" at the cost
    of more than half the ink, so the viewer elides them and marks the node
    instead.  Ties at the threshold are all kept, so the result may be a
    little larger than ``count``.
    """
    n = graph.n
    in_degree = np.diff(adj.rev_indptr.astype(np.int64))

    if n == 0 or count <= 0 or not in_degree.size:
        return np.zeros(n, dtype=np.bool_), 0

    ranked = np.sort(in_degree)[::-1]
    top = int(ranked[min(count, n) - 1])
    floor = max(2, int(HUB_MIN_MULTIPLE * float(in_degree.mean())))
    threshold = max(top, floor)

    return in_degree >= threshold, threshold


@dataclass
class DrawOrder:
    """A drawing permutation over ``graph.edges``, most informative first.

    The viewer draws a *prefix*, so ``drawn`` is the default budget and the
    slider walks outward from it into what this tiering judged redundant.
    Nothing is discarded: ``edges.u32`` and both CSR directions still carry
    every edge, so closures and degree counts are untouched by anything
    decided here.
    """

    order: NDArray[np.uint32]
    drawn: int
    essential: int
    hub_edges: int


def draw_order(
    graph: GroupGraph,
    *,
    essential: NDArray[np.bool_],
    hub: NDArray[np.bool_],
) -> DrawOrder:
    """Rank edges into four tiers: essential or not, crossed with into-a-hub.

    A stable argsort, so edges keep their ``(src, dst)`` order inside a tier
    and the permutation is reproducible.
    """
    edges = graph.edges
    into_hub = hub[edges[:, 1]] if edges.shape[0] else np.zeros(0, dtype=np.bool_)
    tier = np.where(essential, 0, 2) + into_hub.astype(np.int64)

    return DrawOrder(
        order=np.argsort(tier, kind="stable").astype(np.uint32),
        drawn=int(np.count_nonzero(tier == 0)),
        essential=int(np.count_nonzero(essential)),
        hub_edges=int(np.count_nonzero(into_hub)),
    )


def community_bands(
    graph: GroupGraph,
    depth: NDArray[np.int32],
    *,
    drawn: NDArray[np.bool_],
    resolution: float,
    seed: int,
    min_band: int,
) -> tuple[NDArray[np.int64], float]:
    """Louvain communities of the drawn subgraph, as band ids ordered by depth.

    The namespace banding this replaces is not a partition of *this* graph.
    At declaration level the leading name component is usually a type rather
    than a library -- ``Acc``, ``Add``, ``Nat`` -- so core Lean yields a few
    thousand bands of a handful of nodes each and x carries almost no
    information.  Louvain over the drawn edges finds 180 communities at
    modularity 0.67, 27 of which hold 99.7% of the nodes.

    Run over the *drawn* edges deliberately: the elided hubs touch everything,
    and including them washes the structure out.

    Communities below ``min_band`` merge into one trailing band.  Core Lean has
    153 of them holding 190 nodes between them -- mostly the hubs, which have
    no drawn edge left to be clustered by -- and each would otherwise claim a
    hairline band of its own.
    """
    n = graph.n
    if n == 0:
        return np.zeros(0, dtype=np.int64), 0.0

    g: nx.Graph[int] = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from((int(s), int(d)) for s, d in graph.edges[drawn])

    found = louvain_communities(g, resolution=resolution, seed=seed)
    quality = float(modularity(g, found)) if g.number_of_edges() else 0.0

    big = [sorted(int(i) for i in c) for c in found if len(c) >= min_band]
    # Foundations left, as the namespace banding also did; the lowest member
    # index breaks ties so two builds of one export cannot disagree.
    big.sort(key=lambda members: (float(np.median(depth[members])), members[0]))

    bands = np.full(n, len(big), dtype=np.int64)  # the catch-all band
    for i, members in enumerate(big):
        bands[members] = i

    return bands, quality


def namespace_band_ids(names: Sequence[str], split_above: float) -> NDArray[np.int64]:
    """`namespace_bands` as dense ids, for the ``--bands namespace`` fallback."""
    bands = namespace_bands(names, split_above)
    order = {band: i for i, band in enumerate(sorted(set(bands)))}
    return np.fromiter((order[b] for b in bands), dtype=np.int64, count=len(bands))


def namespace_bands(names: Sequence[str], split_above: float) -> list[str]:
    """Group names into bands by leading namespace component.

    One component is usually the right grain, but a library that holds most of
    the export (``Mathlib`` does) would collapse into a single band and lose
    all horizontal structure.  So any band holding more than ``split_above``
    of the nodes is re-split one component deeper.  The rule is proportional
    rather than a hardcoded library name, so it behaves the same on a project
    that is not Mathlib.
    """
    head = [n.split(".", 1)[0] for n in names]
    tally: dict[str, int] = {}
    for band in head:
        tally[band] = tally.get(band, 0) + 1

    limit = max(1, int(len(names) * split_above))
    oversized = {band for band, count in tally.items() if count > limit}

    if not oversized:
        return head

    return [
        ".".join(name.split(".")[:2]) if band in oversized else band
        for name, band in zip(names, head, strict=True)
    ]


def layered_layout(
    graph: GroupGraph,
    depth: NDArray[np.int32],
    bands: NDArray[np.int64],
) -> NDArray[np.float32]:
    """Positions in ``[-1, 1]^2``: y is kernel depth, x is band.

    Bands are ordered by median depth so foundations sit left and the things
    built on them sit right, each band gets an x-interval proportional to its
    node count, and within a ``(band, depth)`` cell nodes are spread evenly in
    name order.  Every level therefore fills its band's width, which is what
    keeps dense levels visible when zoomed out.  Fully deterministic, so two
    builds of the same export are diffable.

    ``bands`` comes from `community_bands` by default and from
    `namespace_band_ids` under ``--bands namespace``.  This function does not
    care which, only that the ids are dense from zero.
    """
    n = graph.n

    if n == 0:
        return np.empty((0, 2), dtype=np.float32)

    band_id = bands
    n_bands = int(max(band_id.tolist(), default=0)) + 1

    counts = np.bincount(band_id, minlength=n_bands).astype(np.float64)
    median_depth = np.array(
        [float(np.median(depth[band_id == i])) if counts[i] else 0.0 for i in range(n_bands)]
    )

    # Re-rank bands by median depth, then lay them out left to right in that
    # order with widths proportional to population.
    band_order = np.argsort(median_depth, kind="stable")
    width = counts[band_order] / max(counts.sum(), 1.0)
    bounds = np.concatenate([[0.0], np.cumsum(width)])
    left = np.empty(n_bands, dtype=np.float64)
    span = np.empty(n_bands, dtype=np.float64)
    left[band_order] = bounds[:-1]
    span[band_order] = width

    top = max(int_max(depth), 1)
    name_rank = np.empty(n, dtype=np.int64)
    name_rank[sorted(range(n), key=graph.ids.__getitem__)] = np.arange(n)

    cell = band_id * (top + 1) + depth
    order = np.lexsort((name_rank, cell))
    sorted_cell = cell[order]

    breaks = np.flatnonzero(np.diff(sorted_cell)) + 1
    cell_start = np.concatenate([[0], breaks])
    cell_size = np.diff(np.concatenate([cell_start, [n]]))
    rank_in_cell = np.arange(n) - np.repeat(cell_start, cell_size)
    size_of = np.repeat(cell_size, cell_size)

    frac = (rank_in_cell + 0.5) / size_of
    sorted_band = band_id[order]
    xs = left[sorted_band] + frac * span[sorted_band]

    pos = np.empty((n, 2), dtype=np.float32)
    pos[order, 0] = (xs * 2.0 - 1.0).astype(np.float32)
    pos[:, 1] = (depth.astype(np.float64) / top * 2.0 - 1.0).astype(np.float32)
    return pos


# --- force layout ------------------------------------------------------------
#
# The layered layout above is a lattice by construction: y takes one value per
# kernel depth -- 134 of them for 58,612 nodes -- and x is a rank by name
# inside a `(band, depth)` cell.  Both axes are quantised, so the drawing is a
# grid whatever the graph says, and the grid actively fights the community
# structure: communities are strong (Q = 0.67, 77% of drawn edges stay inside
# their band) but span the whole depth range, so pinning y to depth smears each
# one into a full-height ribbon and 31 ribbons overlap everywhere.
#
# What follows replaces it with PivotMDS for the global shape and multilevel
# Fruchterman-Reingold for the local detail.  Measured on core Lean: mean drawn
# edge length falls from 0.4136 to 0.0734 at the default settings, against an
# ideal edge length of sqrt(4/V) = 0.0083.  A converged single-level force
# layout scores 0.0786 for four times the iterations, so the hierarchy is
# paying for itself; `--fr-iters` buys the rest, down to about 0.069.

FAR_CHUNK = 8192  # rows per far-field block, to bound peak memory


@dataclass
class LayoutInfo:
    """What the layout did, for ``meta.json``.

    ``edge_length`` is the mean drawn-edge length, which is the number the two
    layouts are actually compared on; recording it in the artefact means a
    rebuild that silently under-converges is visible without re-measuring.
    """

    kind: str = "force"
    pivots: int = 0
    levels: int = 0
    seconds: float = 0.0
    edge_length: float = 0.0


@dataclass
class Level:
    """One rung of the coarsening hierarchy, as a weighted undirected CSR.

    ``parent`` maps each node of the level *below* this one to its supernode
    here, which is the map prolongation walks back down.  On the finest level
    it is the identity and unused.
    """

    n: int
    indptr: NDArray[np.uint32]
    indices: NDArray[np.uint32]
    weight: NDArray[np.float64]
    mass: NDArray[np.float64]
    parent: NDArray[np.int64]


def drawn_csr(
    graph: GroupGraph, drawn: NDArray[np.bool_]
) -> tuple[NDArray[np.uint32], NDArray[np.uint32]]:
    """Undirected CSR over the drawn edges only.

    The same subgraph `community_bands` clusters, for the same reason: the
    elided hubs touch everything, so including them collapses most distances to
    two and leaves the layout nothing to separate.  Direction is dropped
    because an attractive force does not have one.
    """
    edges = graph.edges[drawn] if graph.edges.shape[0] else graph.edges
    src = np.concatenate([edges[:, 0], edges[:, 1]]).astype(np.uint32)
    dst = np.concatenate([edges[:, 1], edges[:, 0]]).astype(np.uint32)
    return csr(graph.n, src, dst)


def bfs_hops(
    n: int,
    indptr: NDArray[np.uint32],
    indices: NDArray[np.uint32],
    source: int,
) -> NDArray[np.int32]:
    """Hop distance from ``source``, ``-1`` where unreached.

    Level-synchronous: one flat gather of every neighbour of the frontier per
    step, so the cost is a handful of array operations per level rather than
    per node.  This graph is at most a few hundred levels deep.
    """
    hops = np.full(n, -1, dtype=np.int32)
    if n == 0:
        return hops

    hops[source] = 0
    frontier = np.array([source], dtype=np.int64)
    step = 0

    while frontier.size:
        step += 1
        starts = indptr[frontier].astype(np.int64)
        counts = indptr[frontier + 1].astype(np.int64) - starts
        total = int(np.sum(counts))
        if not total:
            break
        # Offsets within each frontier node's slice, plus that slice's start:
        # `arange(total)` minus the exclusive prefix sum, repeated per node.
        base = np.cumsum(counts) - counts
        offsets = np.arange(total) - np.repeat(base, counts)
        found = np.unique(indices[np.repeat(starts, counts) + offsets].astype(np.int64))
        frontier = found[hops[found] < 0]
        hops[frontier] = step

    return hops


def pivot_distances(
    n: int,
    indptr: NDArray[np.uint32],
    indices: NDArray[np.uint32],
    *,
    count: int,
) -> NDArray[np.float64]:
    """``(n, p)`` hop distances to ``p`` max-min pivots.

    Max-min -- k-centre greedy -- spreads the pivots over the graph, which is
    what lets `pivot_mds` see its large-scale shape from so few of them.  The
    first pivot is the highest-degree node; each next one is whichever *reached*
    node is furthest from those already chosen.  Every tie breaks on the lowest
    index, so a rebuild cannot pick a different set.

    Candidates are restricted to reached nodes deliberately.  140 nodes of core
    Lean have no drawn edge at all, and an unreachable-first rule would spend
    every pivot on a singleton whose distance column says nothing.  Those nodes
    instead take ``max finite + 1`` in each column, which is finite -- ``inf``
    would make the double centring in `pivot_mds` entirely NaN -- and puts them
    at the periphery, where a node with no drawn edge belongs.

    Returns fewer than ``count`` columns if the reached set runs out first.
    """
    p = max(0, min(count, n))
    out = np.zeros((n, p), dtype=np.float64)
    if not p:
        return out

    degree = (indptr[1:] - indptr[:-1]).astype(np.int64)
    pivot = int(np.argmax(degree))
    best = np.zeros(n, dtype=np.int64)
    seen = np.zeros(n, dtype=np.bool_)
    taken = 0

    for j in range(p):
        hops = bfs_hops(n, indptr, indices, pivot)
        reached = hops >= 0
        far = int(np.max(hops[reached])) + 1 if bool(np.any(reached)) else 1
        out[:, j] = np.where(reached, hops, far)
        taken = j + 1

        fresh = reached & ~seen
        best[fresh] = hops[fresh]
        both = reached & seen
        best[both] = np.minimum(best[both], hops[both])
        seen |= reached

        if j + 1 < p:
            # A chosen pivot has `best == 0` and every other reached node has
            # `best >= 1`, so this both excludes the pivots already taken and
            # detects that the reached set is exhausted.
            candidate = np.where(seen, best, -1)
            if int(np.max(candidate)) <= 0:
                break
            pivot = int(np.argmax(candidate))

    return out[:, :taken]


def pivot_mds(dist: NDArray[np.float64]) -> NDArray[np.float64]:
    """Classical scaling against the pivots only (Brandes and Pich, 2007).

    Double-centre the squared pivot distances and take the two leading right
    singular vectors of the resulting ``(n, p)`` matrix.  The eigenproblem is
    on ``C.T @ C``, which is ``p x p`` -- 100 x 100 here -- so
    ``np.linalg.eigh`` solves it exactly and nothing larger is ever formed:
    O(n*p) all told, no scipy, no iterative solver.

    This is what seeds the hierarchy instead of a random cloud.  Unlike a force
    layout it has no local minima to get stuck in, so the large-scale
    arrangement is decided before any fine detail is committed to.
    """
    n, p = dist.shape
    if n == 0 or p == 0:
        return np.zeros((n, 2), dtype=np.float64)

    centred = dist * dist
    centred -= np.mean(centred, axis=1, keepdims=True)
    centred -= np.mean(centred, axis=0, keepdims=True)
    centred *= -0.5
    # Subtracting the row means and then the column means *of that result* is
    # the full `-0.5 * (D^2 - row - col + grand)` double centring.

    _, vectors = np.linalg.eigh(centred.T @ centred)
    axes = np.zeros((p, 2), dtype=np.float64)
    axes[:, 0] = vectors[:, -1]  # eigh returns ascending eigenvalues
    if p > 1:
        axes[:, 1] = vectors[:, -2]

    # `eigh` fixes each eigenvector only up to sign, so two builds could differ
    # by a reflection.  Pin it: the largest-magnitude entry of each axis is
    # positive, ties to the lowest index.
    lead = np.argmax(np.abs(axes), axis=0)
    axes *= np.where(axes[lead, [0, 1]] < 0.0, -1.0, 1.0)
    return centred @ axes


def match_heavy(level: Level) -> tuple[NDArray[np.int64], int]:
    """Maximal heavy-edge matching, as a ``node -> supernode`` map.

    Edges score ``w / (mass_u * mass_v)``: heavy *relative to* what the two
    ends already weigh.  The mass term is what stops one supernode running
    away, and without it this stalls badly on core Lean -- a single supernode
    reached 27,209 of the 58,612 nodes with degree 15,358, every other node
    became a leaf hanging off it, and a matching cannot coarsen a star.  The
    hierarchy plateaued at 15,466 nodes instead of reaching a thousand.

    Nodes are visited lightest-first so the masses stay balanced, and every tie
    breaks on the lowest index because `demo/README.md` promises byte-identical
    rebuilds.  A Python loop rather than array operations because a matching is
    inherently sequential -- the same trade `longest_path_depth` makes.
    """
    n = level.n
    mate = np.full(n, -1, dtype=np.int64)
    bounds = level.indptr.tolist()
    neighbour = level.indices.tolist()
    weights = level.weight.tolist()
    mass = level.mass.tolist()
    groups = 0

    for u in np.lexsort((np.arange(n), level.mass)).tolist():
        if mate[u] >= 0:
            continue
        pick, best = -1, 0.0
        for slot in range(bounds[u], bounds[u + 1]):
            v = neighbour[slot]
            if v == u or mate[v] >= 0:
                continue
            score = weights[slot] / (mass[u] * mass[v])
            if pick < 0 or score > best or (score == best and v < pick):
                pick, best = v, score
        mate[u] = groups
        if pick >= 0:
            mate[pick] = groups
        groups += 1

    return mate, groups


def contract(mate: NDArray[np.int64], groups: int, level: Level) -> Level:
    """Collapse a matching into the next coarser level.

    Edges inside a supernode are dropped -- they would pull at zero distance --
    and parallel edges between two supernodes sum their weights, so a pair that
    was densely connected stays strongly attracted after contraction.
    """
    src = np.repeat(
        np.arange(level.n, dtype=np.int64),
        np.diff(level.indptr.astype(np.int64)),
    )
    head = mate[src]
    tail = mate[level.indices.astype(np.int64)]
    across = head != tail

    key = head[across] * groups + tail[across]
    unique, inverse = np.unique(key, return_inverse=True)
    summed = np.bincount(inverse, weights=level.weight[across], minlength=unique.size)
    row = (unique // groups).astype(np.uint32)
    col = (unique % groups).astype(np.uint32)

    indptr, indices = csr(groups, row, col)
    # `csr` stable-argsorts the rows; the weights have to follow the same
    # permutation or they would be attached to the wrong edges.
    weight = summed[np.argsort(row, kind="stable")]
    mass = np.bincount(mate, weights=level.mass, minlength=groups)
    return Level(groups, indptr, indices, weight, mass, mate)


def coarsen(
    indptr: NDArray[np.uint32],
    indices: NDArray[np.uint32],
    *,
    limit: int,
    max_levels: int = 32,
) -> list[Level]:
    """The hierarchy, finest level first, stopping at ``limit`` nodes.

    Halving at every rung, so core Lean's 58,612 nodes reach a thousand in
    about nine levels.  Stops early once a matching cannot usefully shrink the
    level any further: a graph that is mostly edgeless, or mostly stars, has no
    matching left to find, and grinding through thirty more levels of
    refinement for a few dozen merges is pure cost.
    """
    n = int(indptr.size) - 1
    levels = [
        Level(
            n,
            indptr,
            indices,
            np.ones(int(indices.size), dtype=np.float64),
            np.ones(n, dtype=np.float64),
            np.arange(n, dtype=np.int64),
        )
    ]

    while levels[-1].n > limit and len(levels) < max_levels:
        current = levels[-1]
        mate, groups = match_heavy(current)
        if groups > current.n * 0.9:
            break
        levels.append(contract(mate, groups, current))

    return levels


def repulsion(
    pos: NDArray[np.float64],
    mass: NDArray[np.float64],
    out: NDArray[np.float64],
    *,
    fine: int = 128,
    coarse: int = 16,
) -> NDArray[np.float64]:
    """Sum ``mass_j / r^2`` away from every other node, on two grids.

    A Barnes-Hut tree is the textbook answer and does not express as array
    operations; this does, and it is O(n) per step.  Near field: the 5x5
    neighbourhood of a fine grid, each cell reduced to its mass and centroid,
    with the node's own contribution subtracted out of its own cell.  Far
    field: every occupied cell of a coarse grid as one centre of mass, with the
    node's own coarse cell zeroed so the near field is not counted twice.

    Approximate in the annulus between the two: a coarse cell is eight fine
    cells wide while the near field reaches two, so some mass just outside the
    near field is dropped rather than double counted.  Measured convergence on
    core Lean is 0.2692 to 0.0050 over 400 iterations, so the error is not
    where this layout's quality is decided.
    """
    n = pos.shape[0]
    out[:] = 0.0
    if n < 2:
        return out

    lo = np.min(pos, axis=0)
    extent = np.maximum(np.max(pos, axis=0) - lo, 1e-9)

    for size, radius in ((fine, 2), (coarse, -1)):
        cell = np.clip(((pos - lo) / extent * size).astype(np.int64), 0, size - 1)
        flat = cell[:, 1] * size + cell[:, 0]
        cell_mass = np.bincount(flat, weights=mass, minlength=size * size)
        cell_x = np.bincount(flat, weights=mass * pos[:, 0], minlength=size * size)
        cell_y = np.bincount(flat, weights=mass * pos[:, 1], minlength=size * size)

        if radius < 0:
            live = np.flatnonzero(cell_mass > 0.0)
            if live.size == 0:
                continue
            safe = np.maximum(cell_mass[live], 1e-12)
            centre_x, centre_y = cell_x[live] / safe, cell_y[live] / safe
            weights = cell_mass[live]
            own = np.searchsorted(live, flat)
            # Blocked over rows: the full (n, cells) product would be hundreds
            # of megabytes at this scale for no gain.
            for start in range(0, n, FAR_CHUNK):
                stop = min(start + FAR_CHUNK, n)
                dx = pos[start:stop, 0:1] - centre_x[None, :]
                dy = pos[start:stop, 1:2] - centre_y[None, :]
                push = weights[None, :] / (dx * dx + dy * dy + 1e-12)
                push[np.arange(stop - start), own[start:stop]] = 0.0
                out[start:stop, 0] += np.sum(dx * push, axis=1)
                out[start:stop, 1] += np.sum(dy * push, axis=1)
            continue

        for offset_y in range(-radius, radius + 1):
            for offset_x in range(-radius, radius + 1):
                at_x, at_y = cell[:, 0] + offset_x, cell[:, 1] + offset_y
                inside = (at_x >= 0) & (at_x < size) & (at_y >= 0) & (at_y < size)
                # Out-of-grid lookups read cell 0 and are masked off below;
                # clamping them to the border instead would double-count it.
                at = np.where(inside, at_y * size + at_x, 0)
                here_mass, here_x, here_y = cell_mass[at], cell_x[at], cell_y[at]
                if offset_x == 0 and offset_y == 0:
                    here_mass = here_mass - mass
                    here_x = here_x - mass * pos[:, 0]
                    here_y = here_y - mass * pos[:, 1]
                live = inside & (here_mass > 1e-12)
                safe = np.where(live, here_mass, 1.0)
                dx = pos[:, 0] - here_x / safe
                dy = pos[:, 1] - here_y / safe
                push = np.where(live, here_mass / (dx * dx + dy * dy + 1e-12), 0.0)
                out[:, 0] += dx * push
                out[:, 1] += dy * push

    return out


def fit_unit(pos: NDArray[np.float64]) -> NDArray[np.float64]:
    """Centre and scale into ``[-1, 1]^2``, aspect preserved, nothing clipped.

    Runs between levels, where clipping an outlier would pin it to the border
    for the rest of the run.  `normalize_positions` is the clipping variant and
    runs once, at the end.  Refitting each level is what lets one ``k`` and one
    starting temperature mean the same thing at every rung.
    """
    if pos.shape[0] == 0:
        return pos
    lo, hi = np.min(pos, axis=0), np.max(pos, axis=0)
    scale = max(float(np.max(hi - lo)), 1e-12)
    return (pos - (lo + hi) / 2.0) * (2.0 / scale)


def normalize_positions(pos: NDArray[np.float64]) -> NDArray[np.float32]:
    """Final fit into ``[-1, 1]^2``: the central 99% sets the scale.

    A min/max fit is the obvious choice and is wrong here.  A force layout
    flings the few nodes with no drawn edge far past the rest, and letting them
    set the bounding box squashed a converged layout to 0.5% of the canvas with
    5,103 nodes in its densest cell.  Percentiles plus a clip are what a
    fit-to-content viewer does anyway.
    """
    if pos.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)
    lo = np.percentile(pos, 0.5, axis=0)
    hi = np.percentile(pos, 99.5, axis=0)
    scale = max(float(np.max(hi - lo)), 1e-12)
    return np.clip((pos - (lo + hi) / 2.0) * (2.0 / scale), -1.0, 1.0).astype(np.float32)


def force_refine(
    pos: NDArray[np.float64],
    level: Level,
    *,
    iters: int,
    strength: float,
) -> NDArray[np.float64]:
    """Fruchterman-Reingold on one level, cooling linearly to zero.

    ``k = sqrt(4 / n)`` is the ideal edge length for this many nodes on a
    ``[-1, 1]^2`` canvas.  Attraction along each edge is ``w * d / k`` -- a
    linear spring of stiffness ``w / k``, not FR's quadratic, because that is
    the form measured to converge here -- and repulsion is
    ``strength * k^2 * mass / r^2`` between nodes.  Each step is capped by a
    temperature falling to zero over ``iters``, which is what makes the run
    settle on a configuration instead of oscillating around one.

    ``strength`` scales the repulsion against the springs.  It is a knob
    rather than a constant because it mattered a great deal to a single-level
    run -- raising it there cut mean edge length from 0.0786 to 0.0685 -- and
    then turned out not to matter at all once PivotMDS seeds the hierarchy
    (0.0734 against 0.0745 on core Lean).  A seeded layout already fills the
    canvas, so there is no collapsed core for extra repulsion to open up.

    The CSR holds both directions of every edge, so accumulating the force onto
    the source of each directed entry covers both of its endpoints exactly
    once.  Masses are renormalised to average one, so a level of supernodes
    repels on the same scale as a level of nodes while a heavy supernode still
    pushes harder than a light one.
    """
    n = level.n
    if n < 2 or iters <= 0:
        return fit_unit(pos)

    pos = fit_unit(pos)
    src = np.repeat(np.arange(n, dtype=np.int64), np.diff(level.indptr.astype(np.int64)))
    delta = np.zeros((int(src.size), 2), dtype=np.float64)
    dst = level.indices.astype(np.int64)
    mass = level.mass * (n / max(float(np.sum(level.mass)), 1e-12))

    k = float(np.sqrt(4.0 / n))
    stiffness = level.weight / k
    acc = np.zeros_like(pos)
    scratch = np.zeros_like(pos)

    for step in range(iters):
        acc[:] = 0.0
        np.subtract(pos[src], pos[dst], out=delta)
        np.add.at(acc, src, -delta * stiffness[:, None])
        acc += strength * k * k * repulsion(pos, mass, scratch)

        temperature = 0.1 * (1.0 - step / iters)
        magnitude = np.maximum(np.hypot(acc[:, 0], acc[:, 1]), 1e-9)
        pos += acc * np.minimum(1.0, temperature / magnitude)[:, None]

    return pos


def multilevel_layout(
    graph: GroupGraph,
    *,
    drawn: NDArray[np.bool_],
    pivots: int,
    coarsest: int,
    iters: int,
    repulse: float,
    seed: int,
) -> tuple[NDArray[np.float32], LayoutInfo]:
    """Positions in ``[-1, 1]^2`` from PivotMDS plus multilevel refinement.

    Coarsen to a thousand nodes, seed *that* with PivotMDS, then prolong and
    refine back down.  Deciding the global arrangement on the small graph is
    the whole point: single-level force placement on 58,612 nodes converges to
    whichever local minimum its starting configuration happens to sit in.

    The iteration budget halves at each coarser rung, which is measured rather
    than assumed: spreading the same total evenly over all twelve levels scored
    0.0751 against 0.0734 for this schedule at comparable cost.  Coarse levels
    settle a few hundred supernodes in well under thirty iterations, and what
    is left over is better spent on the finest level, where mean edge length is
    still falling when the budget runs out.  ``--fr-iters`` is therefore the
    quality-for-time knob: 200 costs ~26s on core Lean, and roughly four times
    that reaches 0.069.

    Prolongation adds a little noise, because two children landing on their
    parent's exact position feel identical forces and would never separate.
    Fixed seed, so rebuilds stay byte-identical.

    Returns the positions alongside what it took to get them.
    """
    if graph.n == 0:
        return np.empty((0, 2), dtype=np.float32), LayoutInfo()

    indptr, indices = drawn_csr(graph, drawn)
    levels = coarsen(indptr, indices, limit=coarsest)
    rng = np.random.default_rng(seed)

    top = levels[-1]
    dist = pivot_distances(top.n, top.indptr, top.indices, count=pivots)
    pos = fit_unit(pivot_mds(dist))
    pos = pos + rng.normal(0.0, 0.01 * np.sqrt(4.0 / max(top.n, 1)), pos.shape)

    for index in range(len(levels) - 1, -1, -1):
        level = levels[index]
        pos = force_refine(pos, level, iters=max(30, iters >> index), strength=repulse)
        if index:
            below = levels[index - 1].n
            pos = pos[level.parent]
            pos += rng.normal(0.0, 0.01 * np.sqrt(4.0 / max(below, 1)), pos.shape)

    return normalize_positions(pos), LayoutInfo(pivots=int(dist.shape[1]), levels=len(levels))


def mean_edge_length(pos: NDArray[np.float32], edges: NDArray[np.uint32]) -> float:
    """Mean length of the drawn edges, the number both layouts are judged on.

    Recorded in ``meta.json`` so the objective lives in the artefact rather
    than in somebody's notes.  The ideal for core Lean is ``sqrt(4/V)`` =
    0.0083; the layered layout scores 0.4136 and this one 0.0734.
    """
    if edges.shape[0] == 0:
        return 0.0
    delta = pos[edges[:, 0]].astype(np.float64) - pos[edges[:, 1]].astype(np.float64)
    return float(np.mean(np.hypot(delta[:, 0], delta[:, 1])))


def write_buffers(
    graph: GroupGraph,
    pos: NDArray[np.float32],
    depth: NDArray[np.int32],
    out: Path,
    *,
    adj: Adjacency,
    ex: Export,
    counts: Counts,
    stranded: int,
    include_literal_types: bool,
    order: DrawOrder,
    layout: LayoutInfo,
    bands: NDArray[np.int64],
    band_kind: str,
    quality: float,
    hub_threshold: int,
) -> dict[str, Any]:
    """Write every buffer the viewer fetches, plus ``meta.json``."""
    out.mkdir(parents=True, exist_ok=True)
    n = graph.n
    kind_names = sorted(set(graph.kinds))
    kind_code = {k: i for i, k in enumerate(kind_names)}

    pos.tofile(out / "nodes.f32")
    np.fromiter((kind_code[k] for k in graph.kinds), dtype=np.uint8, count=n).tofile(
        out / "node_kind.u8"
    )
    np.asarray(graph.flags, dtype=np.uint8).tofile(out / "node_flags.u8")
    np.clip(depth, 0, 0xFFFF).astype(np.uint16).tofile(out / "node_depth.u16")

    blob, offsets = pack_strings(graph.ids)
    blob.tofile(out / "names.bin")
    offsets.tofile(out / "names_idx.u32")

    # Sorted-by-name permutation, so the viewer's prefix search is a binary
    # search rather than a sort of 300k strings at load time.
    np.asarray(sorted(range(n), key=graph.ids.__getitem__), dtype=np.uint32).tofile(
        out / "name_order.u32"
    )

    member_blob, member_offsets = pack_strings(["\n".join(m) for m in graph.members])
    member_blob.tofile(out / "members.bin")
    member_offsets.tofile(out / "members_idx.u32")

    # Bands double as the viewer's optional colour channel, so they ship too.
    np.clip(bands, 0, 0xFFFF).astype(np.uint16).tofile(out / "node_comm.u16")

    # `edges.u32` and both CSR directions stay complete -- every edge, so
    # closures and degree counts are exact.  Reduction is expressed purely as
    # a drawing order over them, the same separation `name_order.u32` draws
    # between storage order and search order.
    graph.edges.tofile(out / "edges.u32")
    order.order.tofile(out / "edge_order.u32")
    adj.fwd_indptr.tofile(out / "fwd_indptr.u32")
    adj.fwd_indices.tofile(out / "fwd_indices.u32")
    adj.rev_indptr.tofile(out / "rev_indptr.u32")
    adj.rev_indices.tofile(out / "rev_indices.u32")

    meta: dict[str, Any] = {
        "nodes": n,
        "edges": int(graph.edges.shape[0]),
        "declarations": graph.declarations,
        "decl_records": counts.decls,
        "expressions": counts.exprs,
        "names": counts.names,
        "levels_discarded": counts.levels,
        "missing": sorted(graph.missing)[:50],
        "missing_total": len(graph.missing),
        "cyclic_nodes": stranded,
        "max_depth": int_max(depth),
        # How the geometry was produced.  `mean_edge_length` is the objective:
        # the ideal for core Lean is sqrt(4/V) = 0.0083, `--layout grid` scores
        # 0.4136, and this one 0.0734.
        "layout": layout.kind,
        "mds_pivots": layout.pivots,
        "coarse_levels": layout.levels,
        "layout_seconds": round(layout.seconds, 2),
        "mean_edge_length": round(layout.edge_length, 4),
        # What the viewer draws by default: the edges left after dropping both
        # the transitively redundant ones and those landing on a hub.  The
        # slider walks from here up to `edges`.
        "drawn_edges": order.drawn,
        "essential_edges": order.essential,
        "hub_edges": order.hub_edges,
        "hub_nodes": sum(1 for f in graph.flags if f & NODE_FLAG_HUB),
        "hub_threshold": hub_threshold,
        "bands": band_kind,
        "band_count": int(max(bands.tolist(), default=0)) + 1 if n else 0,
        "modularity": round(quality, 4),
        "kinds": kind_names,
        "blocks": sum(1 for f in graph.flags if f & NODE_FLAG_BLOCK),
        "internal": sum(1 for f in graph.flags if f & NODE_FLAG_INTERNAL),
        "include_literal_types": include_literal_types,
        "lean_version": ex.lean_version,
        "format_version": ex.format_version,
        "reduce_seconds": round(counts.seconds, 2),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def pack_strings(items: Sequence[str]) -> tuple[NDArray[np.uint8], NDArray[np.uint32]]:
    """One utf-8 blob plus ``len(items) + 1`` offsets into it."""
    encoded = [s.encode("utf-8") for s in items]
    offsets = np.zeros(len(encoded) + 1, dtype=np.uint32)
    offsets[1:] = np.cumsum([len(b) for b in encoded])
    return np.frombuffer(b"".join(encoded), dtype=np.uint8), offsets


def open_source(spec: str) -> tuple[IO[bytes], bool]:
    """``(stream, should_close)`` for a path or ``-`` meaning stdin."""
    if spec == "-":
        # typeshed declares `TextIO.buffer` as `BinaryIO`, but ty resolves it
        # to `BinaryIO | Any` here, so the narrowing has to be explicit.  A
        # cast rather than a demoted rule: the claim is about this one
        # expression, not about unsound assignments in general.
        return cast("IO[bytes]", sys.stdin.buffer), False
    return Path(spec).expanduser().open("rb"), True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m demo.build",
        description="Reduce a lean4export NDJSON stream into viewer buffers.",
    )
    parser.add_argument("source", help="NDJSON file, or - to read stdin")
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    parser.add_argument(
        "--no-literal-types",
        action="store_true",
        help="drop the Nat/String dependencies implied by numeric and string literals",
    )
    parser.add_argument(
        "--band-split",
        type=float,
        default=0.05,
        help="--bands namespace only: re-split any band holding more than this fraction",
    )
    parser.add_argument(
        "--bands",
        choices=("community", "namespace"),
        default="community",
        help="x axis: Louvain communities of the graph (default), or leading namespace",
    )
    parser.add_argument(
        "--hub-count",
        type=int,
        default=100,
        help="elide incoming edges for this many highest in-degree nodes (0 disables)",
    )
    parser.add_argument(
        "--min-band",
        type=int,
        default=10,
        help="merge communities smaller than this into one catch-all band",
    )
    parser.add_argument(
        "--louvain-resolution", type=float, default=1.0, help="Louvain resolution parameter"
    )
    parser.add_argument("--louvain-seed", type=int, default=0, help="Louvain seed, for determinism")
    parser.add_argument(
        "--no-transitive-reduction",
        action="store_true",
        help="draw every edge, including those implied by a two-step path",
    )
    parser.add_argument(
        "--layout",
        choices=("force", "grid"),
        default="force",
        help="PivotMDS plus multilevel force placement (default), or the old depth/band grid",
    )
    parser.add_argument(
        "--mds-pivots",
        type=int,
        default=100,
        help="--layout force only: PivotMDS pivot count",
    )
    parser.add_argument(
        "--coarsest",
        type=int,
        default=1000,
        help="--layout force only: node count to coarsen down to before seeding",
    )
    parser.add_argument(
        "--fr-iters",
        type=int,
        default=200,
        help="--layout force only: refinement iterations at the finest level, "
        "halving at each coarser one; the quality-for-time knob",
    )
    parser.add_argument(
        "--fr-repulsion",
        type=float,
        default=1.0,
        help="--layout force only: repulsion strength against the springs",
    )
    parser.add_argument(
        "--layout-seed",
        type=int,
        default=0,
        help="--layout force only: seed for the prolongation noise, for determinism",
    )
    parser.add_argument("--stats", action="store_true", help="report timings to stderr")
    args = parser.parse_args(argv)

    progress = 1_000_000 if args.stats else 0
    stream, should_close = open_source(args.source)

    try:
        ex, table, counts = read_stream(stream, progress=progress)
    finally:
        if should_close:
            stream.close()

    if args.stats:
        print(
            f"read {counts.lines:,} lines in {counts.seconds:.1f}s "
            f"({table.nbytes() / 1e9:.2f} GB expression table)",
            file=sys.stderr,
        )

    started = time.monotonic()
    graph = build_group_graph(
        ex,
        table,
        include_literal_types=not args.no_literal_types,
    )
    if args.stats:
        print(
            f"grouped {graph.n:,} nodes / {graph.edges.shape[0]:,} edges "
            f"in {time.monotonic() - started:.1f}s",
            file=sys.stderr,
        )

    started = time.monotonic()
    adj = adjacency(graph)
    depth, stranded = longest_path_depth(graph.n, adj)

    essential = (
        np.ones(int(graph.edges.shape[0]), dtype=np.bool_)
        if args.no_transitive_reduction
        else transitive_reduction_mask(graph, adj)
    )
    hub, hub_threshold = hub_mask(graph, adj, count=args.hub_count)
    for i in np.flatnonzero(hub).tolist():
        graph.flags[i] |= NODE_FLAG_HUB

    order = draw_order(graph, essential=essential, hub=hub)
    drawn = essential & ~hub[graph.edges[:, 1]] if graph.edges.shape[0] else essential

    quality = 0.0
    if args.bands == "namespace":
        bands = namespace_band_ids(graph.ids, args.band_split)
    else:
        bands, quality = community_bands(
            graph,
            depth,
            drawn=drawn,
            resolution=args.louvain_resolution,
            seed=args.louvain_seed,
            min_band=args.min_band,
        )

    laid = time.monotonic()
    if args.layout == "grid":
        pos, layout = layered_layout(graph, depth, bands), LayoutInfo(kind="grid")
    else:
        pos, layout = multilevel_layout(
            graph,
            drawn=drawn,
            pivots=args.mds_pivots,
            coarsest=args.coarsest,
            iters=args.fr_iters,
            repulse=args.fr_repulsion,
            seed=args.layout_seed,
        )
    layout.seconds = time.monotonic() - laid
    layout.edge_length = mean_edge_length(pos, graph.edges[drawn])

    if args.stats:
        print(
            f"laid out in {time.monotonic() - started:.1f}s "
            f"({order.drawn:,} of {graph.edges.shape[0]:,} edges drawn, "
            f"{int(max(bands.tolist(), default=0)) + 1} bands, Q={quality:.3f}, "
            f"{args.layout} layout over {layout.levels} levels, "
            f"mean edge {layout.edge_length:.4f})",
            file=sys.stderr,
        )

    meta = write_buffers(
        graph,
        pos,
        depth,
        args.out,
        adj=adj,
        ex=ex,
        counts=counts,
        stranded=stranded,
        include_literal_types=not args.no_literal_types,
        order=order,
        layout=layout,
        bands=bands,
        band_kind=args.bands,
        quality=quality,
        hub_threshold=hub_threshold,
    )

    print(
        f"{meta['nodes']:,} nodes, {meta['edges']:,} edges "
        f"({meta['drawn_edges']:,} drawn), max depth {meta['max_depth']} -> {args.out}"
    )
    if meta["missing_total"]:
        print(
            f"warning: {meta['missing_total']:,} referenced constants are not declared in "
            f"this export; closures through them are incomplete",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

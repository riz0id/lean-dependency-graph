"""The demo reducer, pinned against the library it bypasses.

``demo/build.py`` reimplements the dependency walk, because the library's
loader cannot hold a whole-library export.  ``terms.py``'s module docstring
opens by warning that an omission in that walk "produces a dependency set that
is silently too small" -- no error, just a wrong graph.  So the reducer is
compared against ``collect_expr_deps`` declaration by declaration on the
committed fixture, which is the only thing standing between a refactor of
either side and a silent divergence.

Needs numpy (the ``demo`` dependency group) but no Lean toolchain, so it runs
in the default suite wherever that group is installed and skips where it is
not.
"""

from __future__ import annotations

import json
from array import array
from collections.abc import Sequence
from pathlib import Path

import pytest

import lean_dep_graph as ldg
from lean_dep_graph.ndjson import Export, ExportFormatError
from lean_dep_graph.queries import stats
from lean_dep_graph.terms import collect_expr_deps

from .conftest import ExportBuilder

pytest.importorskip("numpy", reason="the demo pipeline needs the `demo` dependency group")

import numpy as np  # noqa: E402
from demo import build  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "sample.ndjson"

Reduced = tuple[build.GroupGraph, Export, build.ExprTable]


@pytest.fixture(scope="module")
def reduced() -> Reduced:
    with FIXTURE.open("rb") as fh:
        ex, table, _ = build.read_stream(fh)
    graph = build.build_group_graph(ex, table, keep_decl_deps=True)
    return graph, ex, table


@pytest.fixture(scope="module")
def library() -> ldg.DeclGraph:
    return ldg.graph_for_export_file(FIXTURE)


def stamp_array(size: int) -> array[int]:
    """A zero-filled visit-stamp array of the shape ``ExprTable.deps`` expects."""
    return array("i", bytes(4 * size))


def test_dependency_sets_match_the_library(reduced: Reduced, library: ldg.DeclGraph) -> None:
    """The whole reason this file exists: same deps, declaration by declaration."""
    graph, _, _ = reduced

    assert set(graph.decl_deps) == set(library.decls), "different declarations parsed"

    for name in sorted(library.decls):
        assert graph.decl_deps[name] == set(library.direct_deps(name)), name


def test_walk_matches_collect_expr_deps_directly(reduced: Reduced, library: ldg.DeclGraph) -> None:
    """Pit the two walkers against each other on identical roots.

    The test above compares through ``DeclGraph``; this one bypasses both
    graph builders, so a divergence cannot hide behind anything either one
    does on the way.
    """
    _, ex, table = reduced
    stamp = stamp_array(table.size + 1)

    for mark, (name, decl) in enumerate(library.decls.items(), start=1):
        expected = collect_expr_deps(library.export, decl.roots)
        got = {
            build.IMPLIED[r] if r in build.IMPLIED else ex.name(r)
            for r in table.deps(decl.roots, stamp, mark)
        }

        assert got == expected, name


def test_group_shape_matches_stats(reduced: Reduced, library: ldg.DeclGraph) -> None:
    """Grouping is the library's own `build_groups`, so it must agree exactly."""
    graph, _, _ = reduced
    expected = stats(library)

    assert graph.n == expected["nodes"]
    assert graph.edges.shape[0] == expected["edges"]
    assert graph.declarations == expected["declarations"]
    assert sum(1 for f in graph.flags if f & build.NODE_FLAG_BLOCK) == expected["blocks"]
    assert graph.missing == library.missing == set()


def test_edges_match_the_library_edge_set(reduced: Reduced, library: ldg.DeclGraph) -> None:
    graph, _, _ = reduced
    mine = {(graph.ids[int(s)], graph.ids[int(d)]) for s, d in graph.edges}

    assert mine == set(library.graph.edges())


def test_edges_are_deduplicated_and_sorted(reduced: Reduced) -> None:
    graph, _, _ = reduced
    packed = (graph.edges[:, 0].astype(np.int64) << 32) | graph.edges[:, 1]

    assert packed.size == np.unique(packed).size, "duplicate edges survived"
    assert bool(np.all(np.diff(packed) > 0)), "edges are not sorted by (src, dst)"
    assert not any(int(s) == int(d) for s, d in graph.edges), "a group depends on itself"


def reduce_lines(lines: Sequence[str]) -> tuple[Export, build.ExprTable]:
    ex, table, _ = build.read_stream([line.encode("utf-8") for line in lines])
    return ex, table


def test_literal_types_can_be_dropped() -> None:
    """Parity with `include_literal_types`; mirrors `test_graph.py:125`.

    The literal has to be the *only* route to ``Nat`` for the flag to show,
    which is why this builds a stream instead of reusing the fixture: there
    ``nat_lit``'s type is ``const Nat``, so ``Nat`` is a dependency whatever
    the flag says and the literal only re-adds it.
    """
    b = ExportBuilder()
    b.define("d", b.sort(0), b.nat(7))

    kept = build.build_group_graph(*reduce_lines(b.lines), keep_decl_deps=True)
    assert kept.decl_deps["d"] == {"Nat"}
    assert kept.missing == {"Nat"}, "a sliced export does not declare Nat"

    dropped = build.build_group_graph(
        *reduce_lines(b.lines), include_literal_types=False, keep_decl_deps=True
    )
    assert dropped.decl_deps["d"] == set()
    assert dropped.missing == set()


def test_depth_respects_every_edge(reduced: Reduced) -> None:
    """Depth is a longest path to a sink, so it must strictly decrease along edges."""
    graph, _, _ = reduced
    depth, stranded = build.longest_path_depth(graph.n, build.adjacency(graph))

    assert stranded == 0, "the fixture graph is acyclic once blocks collapse"
    for src, dst in graph.edges:
        assert depth[int(src)] > depth[int(dst)], (graph.ids[int(src)], graph.ids[int(dst)])


def test_layout_is_bounded_and_deterministic(reduced: Reduced) -> None:
    graph, _, _ = reduced
    depth, _ = build.longest_path_depth(graph.n, build.adjacency(graph))
    bands = build.namespace_band_ids(graph.ids, 0.05)
    first = build.layered_layout(graph, depth, bands)
    second = build.layered_layout(graph, depth, bands)

    assert first.shape == (graph.n, 2)
    assert first.dtype == np.float32
    assert float(np.abs(first).max()) <= 1.0, "positions must stay inside [-1, 1]"
    assert np.array_equal(first, second), "layout is not reproducible"


def csr_pairs(where: Path, direction: str) -> set[tuple[int, int]]:
    indptr = np.fromfile(where / f"{direction}_indptr.u32", dtype=np.uint32)
    indices = np.fromfile(where / f"{direction}_indices.u32", dtype=np.uint32)
    return {
        (row, int(col))
        for row in range(indptr.size - 1)
        for col in indices[indptr[row] : indptr[row + 1]]
    }


def decode_blob(where: Path, stem: str) -> list[str]:
    blob = (where / f"{stem}.bin").read_bytes()
    offsets = np.fromfile(where / f"{stem}_idx.u32", dtype=np.uint32)
    return [blob[offsets[i] : offsets[i + 1]].decode("utf-8") for i in range(offsets.size - 1)]


def test_buffers_round_trip(reduced: Reduced, tmp_path: Path) -> None:
    """Every buffer the viewer fetches must decode back to what went in."""
    graph, ex, _ = reduced
    adj = build.adjacency(graph)
    depth, stranded = build.longest_path_depth(graph.n, adj)
    essential = build.transitive_reduction_mask(graph, adj)
    hub, hub_threshold = build.hub_mask(graph, adj, count=100)
    order = build.draw_order(graph, essential=essential, hub=hub)
    drawn = essential & ~hub[graph.edges[:, 1]]
    bands, quality = build.community_bands(
        graph, depth, drawn=drawn, resolution=1.0, seed=0, min_band=10
    )
    pos = build.layered_layout(graph, depth, bands)

    meta = build.write_buffers(
        graph,
        pos,
        depth,
        tmp_path,
        adj=adj,
        ex=ex,
        counts=build.Counts(),
        stranded=stranded,
        include_literal_types=True,
        order=order,
        layout=build.LayoutInfo(kind="grid"),
        bands=bands,
        band_kind="community",
        quality=quality,
        hub_threshold=hub_threshold,
    )

    assert json.loads((tmp_path / "meta.json").read_text(encoding="utf-8")) == meta

    nodes = np.fromfile(tmp_path / "nodes.f32", dtype=np.float32).reshape(-1, 2)
    assert np.array_equal(nodes, pos)

    edges = np.fromfile(tmp_path / "edges.u32", dtype=np.uint32).reshape(-1, 2)
    assert np.array_equal(edges, graph.edges)
    assert edges.shape[0] == meta["edges"]

    for name in ("node_kind.u8", "node_flags.u8"):
        assert np.fromfile(tmp_path / name, dtype=np.uint8).size == graph.n
    assert np.fromfile(tmp_path / "node_depth.u16", dtype=np.uint16).size == graph.n

    # The CSR arrays must describe the same edge set, in both directions.
    pairs = {(int(s), int(d)) for s, d in edges}
    assert csr_pairs(tmp_path, "fwd") == pairs
    assert csr_pairs(tmp_path, "rev") == {(d, s) for s, d in pairs}

    names = decode_blob(tmp_path, "names")
    assert names == graph.ids
    assert [m.split("\n") for m in decode_blob(tmp_path, "members")] == graph.members

    name_order = np.fromfile(tmp_path / "name_order.u32", dtype=np.uint32)
    assert [names[i] for i in name_order] == sorted(names), "name_order is not sorted by name"

    assert np.array_equal(
        np.fromfile(tmp_path / "node_comm.u16", dtype=np.uint16), bands.astype(np.uint16)
    )

    shipped = np.fromfile(tmp_path / "edge_order.u32", dtype=np.uint32)
    assert np.array_equal(np.sort(shipped), np.arange(edges.shape[0])), (
        "edge_order is not a permutation of the edge indices"
    )
    assert meta["drawn_edges"] == order.drawn
    # The default budget must be exactly the essential, non-hub edges.
    assert set(shipped[: meta["drawn_edges"]].tolist()) == set(np.flatnonzero(drawn).tolist()), (
        "the drawn prefix is not the essential non-hub edge set"
    )


def bfs(indptr: object, indices: object, root: int) -> set[int]:
    """The viewer's closure walk, in Python: iterative BFS over a shipped CSR."""
    ptr = np.asarray(indptr)
    idx = np.asarray(indices)
    seen = {root}
    stack = [root]

    while stack:
        node = stack.pop()
        for raw in idx[ptr[node] : ptr[node + 1]].tolist():
            nxt = int(raw)
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)

    return seen


def test_shipped_csr_reproduces_closure_and_rclosure(
    reduced: Reduced, library: ldg.DeclGraph
) -> None:
    """The viewer's click-to-highlight, checked against `closure` and `rclosure`.

    The JavaScript BFS is fifteen lines of textbook queue walking; what could
    actually be wrong is the adjacency it walks.  So this runs the same walk
    over the same shipped arrays and demands the library's answer, for every
    node in both directions.
    """
    graph, _, _ = reduced
    adj = build.adjacency(graph)

    for i, gid in enumerate(graph.ids):
        forward = {m for j in bfs(adj.fwd_indptr, adj.fwd_indices, i) for m in graph.members[j]}
        assert forward - {gid} == ldg.closure(library, gid), f"closure({gid})"

        reverse = {m for j in bfs(adj.rev_indptr, adj.rev_indices, i) for m in graph.members[j]}
        assert reverse - {gid} == ldg.rclosure(library, gid), f"rclosure({gid})"


def test_transitive_reduction_preserves_every_closure(
    reduced: Reduced, library: ldg.DeclGraph
) -> None:
    """The claim that makes the reduction safe to draw with.

    `transitive_reduction_mask` drops 70% of this graph's edges at Mathlib
    scale, which is only acceptable because reachability is untouched: the
    path that made an edge redundant is still there.  So the viewer's closure
    walk is run over the *reduced* adjacency and required to produce the
    library's answer for every node in both directions -- the same demand
    `test_shipped_csr_reproduces_closure_and_rclosure` makes of the full set.
    """
    graph, _, _ = reduced
    adj = build.adjacency(graph)
    keep = build.transitive_reduction_mask(graph, adj)

    assert not bool(keep.all()), "the fixture has redundant edges, so some must be dropped"

    kept = graph.edges[keep]
    fwd_indptr, fwd_indices = build.csr(
        graph.n, np.ascontiguousarray(kept[:, 0]), np.ascontiguousarray(kept[:, 1])
    )
    rev_indptr, rev_indices = build.csr(
        graph.n, np.ascontiguousarray(kept[:, 1]), np.ascontiguousarray(kept[:, 0])
    )

    for i, gid in enumerate(graph.ids):
        forward = {m for j in bfs(fwd_indptr, fwd_indices, i) for m in graph.members[j]}
        assert forward - {gid} == ldg.closure(library, gid), f"closure({gid}) after reduction"

        reverse = {m for j in bfs(rev_indptr, rev_indices, i) for m in graph.members[j]}
        assert reverse - {gid} == ldg.rclosure(library, gid), f"rclosure({gid}) after reduction"


def test_dropped_edges_all_have_a_surviving_two_step_path(reduced: Reduced) -> None:
    """Why the closures above cannot have changed, checked directly."""
    graph, _, _ = reduced
    adj = build.adjacency(graph)
    keep = build.transitive_reduction_mask(graph, adj)

    out = {
        i: set(adj.fwd_indices[adj.fwd_indptr[i] : adj.fwd_indptr[i + 1]].tolist())
        for i in range(graph.n)
    }
    for (src, dst), survived in zip(graph.edges.tolist(), keep.tolist(), strict=True):
        if survived:
            continue
        assert any(dst in out[mid] for mid in out[src]), (
            f"{graph.ids[src]} -> {graph.ids[dst]} was dropped with no two-step path"
        )


def test_edge_row_order_matches_the_forward_csr(reduced: Reduced) -> None:
    """The invariant every mask is applied through, asserted on its own.

    `transitive_reduction_mask` indexes `graph.edges` with offsets taken from
    the CSR.  That is sound only while `dedupe_edges` and `csr` happen to
    agree on row order, and they are not aware of each other -- so a change to
    either should fail here rather than silently misalign the drawing order.
    """
    graph, _, _ = reduced
    build.check_csr_alignment(graph, build.adjacency(graph))


def test_hub_elision_is_a_no_op_on_a_small_graph(reduced: Reduced) -> None:
    """A hub has to be disproportionate, not merely top-ranked.

    `--hub-count` defaults to 100, which is larger than this fixture, so
    without the `HUB_MIN_MULTIPLE` floor every node would qualify and the
    viewer would draw nothing at all.
    """
    graph, _, _ = reduced
    hub, threshold = build.hub_mask(graph, build.adjacency(graph), count=100)

    assert int(hub.sum()) == 0, "no node in an 18-node graph is infrastructure"
    assert threshold >= 2


def test_bands_are_dense_and_deterministic(reduced: Reduced) -> None:
    graph, _, _ = reduced
    adj = build.adjacency(graph)
    depth, _ = build.longest_path_depth(graph.n, adj)
    drawn = build.transitive_reduction_mask(graph, adj)

    first, q1 = build.community_bands(
        graph, depth, drawn=drawn, resolution=1.0, seed=0, min_band=10
    )
    second, q2 = build.community_bands(
        graph, depth, drawn=drawn, resolution=1.0, seed=0, min_band=10
    )

    assert np.array_equal(first, second), "Louvain banding is not reproducible"
    assert q1 == q2
    # Through `tolist` because numpy's `min`/`max` overloads do not resolve
    # under `[tool.ty.rules] all = "error"`, the same reason `build.int_max`
    # exists.
    seen = set(first.tolist())
    assert min(seen) == 0, "band ids must be dense from zero"
    assert seen == set(range(max(seen) + 1)), "band ids have gaps"


def drawn_mask(graph: build.GroupGraph) -> np.ndarray:
    """The edge set the force layout reads: reduced, with hub in-edges dropped."""
    adj = build.adjacency(graph)
    essential = build.transitive_reduction_mask(graph, adj)
    hub, _ = build.hub_mask(graph, adj, count=100)
    return essential & ~hub[graph.edges[:, 1]]


def test_bfs_hops_matches_networkx(reduced: Reduced) -> None:
    """The one piece of the layout with a free exact oracle.

    `pivot_distances` is only as good as its traversal, and a level-synchronous
    BFS written as array operations is easy to get subtly wrong -- an off-by-one
    in the frontier gather would shift every distance.  networkx is already a
    dependency, so its answer is available for nothing.
    """
    import networkx as nx

    graph, _, _ = reduced
    indptr, indices = build.drawn_csr(graph, drawn_mask(graph))

    g: nx.Graph[int] = nx.Graph()
    g.add_nodes_from(range(graph.n))
    for row in range(graph.n):
        for col in indices[indptr[row] : indptr[row + 1]].tolist():
            g.add_edge(row, col)

    for source in range(graph.n):
        hops = build.bfs_hops(graph.n, indptr, indices, source)
        oracle = nx.single_source_shortest_path_length(g, source)
        for node in range(graph.n):
            assert int(hops[node]) == oracle.get(node, -1), (source, node)


def test_pivot_distances_are_finite_and_spread(reduced: Reduced) -> None:
    """Unreachable is a real case here, and `inf` would poison the centring.

    Every column has to come back finite, or `pivot_mds`'s double centring is
    entirely NaN.  Pivots must also be distinct: a repeated pivot is a wasted
    column, and it is what a max-min rule that can revisit a chosen node would
    produce.
    """
    graph, _, _ = reduced
    indptr, indices = build.drawn_csr(graph, drawn_mask(graph))
    dist = build.pivot_distances(graph.n, indptr, indices, count=100)

    assert dist.shape[0] == graph.n
    assert 0 < dist.shape[1] <= graph.n, "pivots are clamped to the reached set"
    assert bool(np.all(np.isfinite(dist))), "an unreachable pair leaked through as inf or nan"
    pivots = [int(np.argmin(dist[:, j])) for j in range(dist.shape[1])]
    assert len(set(pivots)) == len(pivots), "a pivot was chosen twice"


def test_coarsening_is_a_partition_that_shrinks(reduced: Reduced) -> None:
    """Mass is conserved and the hierarchy actually gets smaller.

    Both halves matter.  A `parent` map that drops a node loses it from the
    layout silently, and a hierarchy that stops shrinking is the failure mode
    unweighted matching has on this graph: one supernode ran away to 27,209 of
    core Lean's 58,612 nodes, after which every other node was a leaf on a star
    and no matching existed.  `limit=1` forces as many rungs as the matching
    can find.
    """
    graph, _, _ = reduced
    indptr, indices = build.drawn_csr(graph, drawn_mask(graph))
    levels = build.coarsen(indptr, indices, limit=1)

    assert len(levels) > 1, "an 18-node graph with edges must coarsen at least once"
    for index, level in enumerate(levels):
        assert float(np.sum(level.mass)) == pytest.approx(float(graph.n))
        assert level.indptr.size == level.n + 1
        assert level.weight.size == level.indices.size
        if index:
            parent = level.parent
            assert parent.size == levels[index - 1].n, "a node lost its supernode"
            assert set(parent.tolist()) == set(range(level.n)), "supernode ids have gaps"
            assert level.n < levels[index - 1].n, "a level that does not shrink is wasted work"


def test_pivot_mds_is_reproducible_and_sign_canonical(reduced: Reduced) -> None:
    """`eigh` fixes an eigenvector only up to sign, so the sign has to be pinned.

    Without the canonicalisation two builds of one export can differ by a
    reflection, which `demo/README.md`'s byte-identical promise forbids.  This
    cannot catch a *platform* difference in LAPACK, only a missing convention.
    """
    graph, _, _ = reduced
    indptr, indices = build.drawn_csr(graph, drawn_mask(graph))
    dist = build.pivot_distances(graph.n, indptr, indices, count=100)

    first = build.pivot_mds(dist)
    second = build.pivot_mds(dist)

    assert first.shape == (graph.n, 2)
    assert bool(np.all(np.isfinite(first)))
    assert np.array_equal(first, second), "PivotMDS is not reproducible"


def test_force_layout_is_bounded_and_deterministic(reduced: Reduced) -> None:
    graph, _, _ = reduced
    drawn = drawn_mask(graph)
    first, info = build.multilevel_layout(
        graph, drawn=drawn, pivots=100, coarsest=1000, iters=60, repulse=1.0, seed=0
    )
    second, again = build.multilevel_layout(
        graph, drawn=drawn, pivots=100, coarsest=1000, iters=60, repulse=1.0, seed=0
    )

    assert first.shape == (graph.n, 2)
    assert first.dtype == np.float32
    assert float(np.abs(first).max()) <= 1.0, "positions must stay inside [-1, 1]"
    assert np.array_equal(first, second), "the force layout is not reproducible"
    assert info.levels == again.levels >= 1
    assert 0 < info.pivots <= graph.n


def test_force_layout_beats_the_grid_on_edge_length(reduced: Reduced) -> None:
    """The objective, asserted rather than left in a notebook.

    Eighteen nodes is far too small to say anything about legibility, but it is
    enough to fail loudly if the force loop is ever mistuned into divergence or
    collapse -- either one sends this number the wrong way.  On core Lean the
    same comparison is 0.4136 for the grid against 0.07 for this.
    """
    graph, _, _ = reduced
    depth, _ = build.longest_path_depth(graph.n, build.adjacency(graph))
    drawn = drawn_mask(graph)
    bands, _ = build.community_bands(graph, depth, drawn=drawn, resolution=1.0, seed=0, min_band=10)

    grid = build.layered_layout(graph, depth, bands)
    force, _ = build.multilevel_layout(
        graph, drawn=drawn, pivots=100, coarsest=1000, iters=60, repulse=1.0, seed=0
    )
    edges = graph.edges[drawn]

    assert build.mean_edge_length(force, edges) < build.mean_edge_length(grid, edges)


def test_force_layout_survives_degenerate_graphs() -> None:
    """No nodes, one node, and nodes with no drawn edge at all.

    The last is not hypothetical: 140 nodes of core Lean have every drawn edge
    elided, so they are unreachable from every pivot.  They must still come
    back with finite coordinates.
    """
    empty = build.GroupGraph([], [], [], [], np.empty((0, 2), dtype=np.uint32))
    pos, info = build.multilevel_layout(
        empty,
        drawn=np.zeros(0, dtype=np.bool_),
        pivots=100,
        coarsest=1000,
        iters=10,
        repulse=1.0,
        seed=0,
    )
    assert pos.shape == (0, 2)
    assert info.levels == 0

    for size in (1, 5):
        lone = build.GroupGraph(
            [f"n{i}" for i in range(size)],
            ["def"] * size,
            [[f"n{i}"] for i in range(size)],
            [0] * size,
            np.empty((0, 2), dtype=np.uint32),
        )
        pos, _ = build.multilevel_layout(
            lone,
            drawn=np.zeros(0, dtype=np.bool_),
            pivots=100,
            coarsest=1000,
            iters=10,
            repulse=1.0,
            seed=0,
        )
        assert pos.shape == (size, 2)
        assert bool(np.all(np.isfinite(pos))), "an edgeless node came back as nan"
        assert float(np.abs(pos).max()) <= 1.0


def test_unknown_expression_kind_is_rejected() -> None:
    """The library returns no children here; the reducer must refuse instead.

    A silent empty answer would undercount dependencies -- the failure
    `terms.check_expr_keys` was written to catch and never wired in to catch.
    """
    with pytest.raises(ExportFormatError, match="no known kind"):
        build.reduce_expr({"ie": 7, "someFutureKind": {"x": 1}})


def test_dangling_expression_index_is_rejected() -> None:
    table = build.ExprTable(block=8)
    table.put(1, -1, -1, -1, build.NO_REF)

    with pytest.raises(ExportFormatError, match="dangling expression index"):
        table.deps([5], stamp_array(table.size + 1), 1)


def test_cli_writes_a_usable_directory(tmp_path: Path) -> None:
    assert build.main([str(FIXTURE), "--out", str(tmp_path)]) == 0

    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["nodes"] == 18
    assert meta["edges"] == 17
    assert meta["declarations"] == 30
    assert meta["missing_total"] == 0
    assert meta["format_version"] == "3.1.0"

    # `edges` is still every edge: reduction is a drawing order, not a filter
    # on the graph.  Three of the 17 are implied by a two-step path, and the
    # fixture is far too small for any node to qualify as a hub.
    assert meta["drawn_edges"] == 14
    assert meta["essential_edges"] == 14
    assert meta["hub_nodes"] == 0
    assert meta["hub_edges"] == 0
    assert meta["bands"] == "community"

    # The default layout, and the objective recorded next to it.
    assert meta["layout"] == "force"
    assert meta["coarse_levels"] >= 1
    assert 0 < meta["mds_pivots"] <= 18
    assert meta["mean_edge_length"] > 0.0


def test_cli_can_still_draw_the_grid(tmp_path: Path) -> None:
    """``--layout grid`` has to reproduce the old geometry exactly.

    Every reduction switch in this pipeline has an off-switch so the previous
    picture is one flag away; the layout is no different, and an escape hatch
    nobody exercises is an escape hatch that has already rotted.  The grid
    scores far worse on edge length, which is the whole reason it was replaced,
    so that ordering is asserted too.
    """
    assert build.main([str(FIXTURE), "--out", str(tmp_path), "--layout", "grid"]) == 0
    grid = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    pos = np.fromfile(tmp_path / "nodes.f32", dtype=np.float32).reshape(-1, 2)

    with FIXTURE.open("rb") as fh:
        ex, table, _ = build.read_stream(fh)
    graph = build.build_group_graph(ex, table)
    adj = build.adjacency(graph)
    depth, _ = build.longest_path_depth(graph.n, adj)
    drawn = build.transitive_reduction_mask(graph, adj)
    bands, _ = build.community_bands(graph, depth, drawn=drawn, resolution=1.0, seed=0, min_band=10)

    assert grid["layout"] == "grid"
    assert grid["coarse_levels"] == 0
    assert grid["mds_pivots"] == 0
    assert np.array_equal(pos, build.layered_layout(graph, depth, bands))

    force = tmp_path / "force"
    assert build.main([str(FIXTURE), "--out", str(force)]) == 0
    other = json.loads((force / "meta.json").read_text(encoding="utf-8"))
    assert other["mean_edge_length"] < grid["mean_edge_length"]

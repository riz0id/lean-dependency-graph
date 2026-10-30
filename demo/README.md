# Demo: a whole library's dependency graph in the browser

Reduces a `lean4export` stream into binary buffers and renders them with WebGL2
— every constant in the environment, laid out so the dependency stack is
legible, with click-to-highlight closures.

No npm, no bundler, no CDN. `demo/web` is two files and a data directory.

## Try it without Lean

The committed fixture is tiny but exercises the whole pipeline:

```bash
nix develop -c uv run --group demo python -m demo.build \
    tests/fixtures/sample.ndjson --out demo/web/data
python3 -m http.server --directory demo/web 8000
```

Then open <http://localhost:8000>. A plain `file://` open will not work — the
viewer `fetch`es its buffers, which needs an origin.

## The real thing: Mathlib

### 1. A built Mathlib

```bash
git clone https://github.com/leanprover-community/mathlib4
cd mathlib4
lake exe cache get      # prebuilt .olean files; several GB
```

### 2. A matching `lean4export`

**This is the step that goes wrong.** The binary must be built against
*Mathlib's* toolchain, not this project's dev shell and not `tests/lean`'s.
A mismatch is the most common failure in this repo and the error message names
both versions when it happens:

```bash
git clone https://github.com/leanprover/lean4export
cd lean4export
cp ../mathlib4/lean-toolchain .      # Mathlib's pin, not yours
lake build
```

This is not hypothetical: on the run recorded below, a fresh `lean4export`
clone pinned `v4.36.0-rc1` while Mathlib pinned `v4.36.0-rc2`. Neighbouring
release candidates, and the `.olean` format differs between them — so the copy
above is the whole difference between a clean export and a version error.

### 3. Export, then reduce — in two steps, not one pipe

Run these (and the two steps above) from inside the dev shell, which is where
both `lake` and `uv` come from — `elan` is a dev-shell package, so `lake` is not
on a bare PATH:

```bash
nix develop /path/to/lean-dependency-graph
cd /path/to/mathlib4
REPO=/path/to/lean-dependency-graph

# 3a. export, compressed, to disk.  gzip -1 keeps up with the producer easily.
lake env /path/to/lean4export/.lake/build/bin/lean4export --export-unsafe Mathlib \
  | gzip -1 > /tmp/mathlib.ndjson.gz

# 3b. reduce, with the exporter no longer resident.
gzip -dc /tmp/mathlib.ndjson.gz \
  | uv run --directory "$REPO" --group demo python -m demo.build - \
      --out "$REPO/demo/web/data" --stats
```

**Do not join those with a pipe on a 16 GiB machine.** It is tempting — nothing
in the pipeline needs the NDJSON on disk — but the two halves peak at the same
time: `lean4export` retains every expression it has visited (4.4 GB and still
climbing partway through Mathlib) and the reducer holds its own table and name
dict (2.6 GB at the same point). Together they exhaust physical memory and
macOS falls back on the compressor, which does not fail — it just gets slower
and slower. Measured on this machine, piped throughput decayed
**845k → 111k → 54k → 34k lines/s** and was still dropping. Staged, each half
runs alone with headroom. Two steps and a temporary file are much faster than
one elegant pipe that never finishes.

Three details are load-bearing. `--export-unsafe` keeps `unsafe` and `partial`
declarations, which `lean4export` otherwise skips while still emitting
references to them — drop the flag and `missing_total` is non-zero through no
fault of the reducer. `--directory "$REPO"` runs the consumer in *this* project
while the producer stays in Mathlib's checkout: `lake env` needs Mathlib's cwd
to set `LEAN_PATH`, while `uv` needs this project's to resolve `--group demo`
and to put `demo/` on `sys.path`. And the `.gz` is worth keeping until you are
happy with the output — re-running 3b to adjust the layout costs seconds,
whereas re-running 3a costs the whole export.

`Mathlib.lean` imports the whole library, so one module name covers everything,
and the reducer keeps every constant the environment holds — core, `Init`,
`Std`, `Batteries` and all auto-generated internals. That is what makes the
closures complete: `meta.json`'s `missing_total` should be `0`, and anything
else means the export was partial.

`--stats` reports lines/sec, the expression table's size, and the time split
between reduction, grouping and layout.

Layout knobs, all with defaults that need no tuning: `--layout
{force,grid}`, `--mds-pivots` (100), `--coarsest` (1000), `--fr-iters` (200),
`--fr-repulsion` (1.0), `--layout-seed` (0), `--bands {community,namespace}`,
`--hub-count` (100, `0` disables), `--min-band` (10), `--louvain-resolution`
(1.0), `--louvain-seed` (0), and `--no-transitive-reduction`. Each switch
exists so the previous picture is one flag away.

`--fr-iters` is the only one that trades quality for time: it is the
refinement budget at the finest level, halving at each coarser one. 200 is
~26s of the build on core Lean and scores 0.0734 mean drawn-edge length;
roughly four times that reaches 0.069 for roughly twice the time.

## Controls

| | |
| --- | --- |
| drag | pan |
| wheel | zoom at cursor |
| click | select: highlight forward closure (warm) and reverse closure (cool) |
| <kbd>f</kbd> | fit |
| <kbd>e</kbd> | toggle edges |
| <kbd>esc</kbd> | clear selection |

The search box does case-sensitive prefix matching, which is what Lean names
want. The edge slider trades fidelity for frame rate; it re-streams the edge
texture on release, not during the drag. It starts at `drawn_edges` and walks
*outward* through `edge_order`, so turning it up adds back the edges the build
ranked as redundant rather than scattering a wider sample — and
the "colour by" select repaints nodes by kernel depth (the default), by
community band, or by declaration kind. The depth slider hides everything
below a given kernel depth, and the internal-declarations checkbox hides
`_proof_*` / `match_*` / `_private.*` without removing them from the graph —
the same distinction `internal=False` draws in the library's queries.

## How it works

`build.py` is a single streaming pass. The library's own loader cannot be used
here: `ndjson.load` keeps every expression as a `dict[int, dict[str, Any]]`,
and a whole-Mathlib export holds tens of millions of them. So expressions are
reduced on arrival into two flat `array` tables — three child slots and one
name reference, 16 bytes each — and walked with an explicit stack.

Grouping is **not** reimplemented. `parse_decls` and `build_groups` read only
`Export.decls` and `Export.name`, never the expression table, so both are
imported and used verbatim; mutual blocks, quotient blocks, declaration kinds
and recursor-rule roots are the library's answers, not a second opinion. Only
the dependency walk is new, and `tests/test_demo_reduce.py` pins it against
`terms.collect_expr_deps` declaration by declaration — including a check that
the shipped CSR reproduces `closure` and `rclosure` for every node in both
directions.

Layout is PivotMDS for the global shape, then multilevel
Fruchterman-Reingold for the local detail. Both axes encode graph structure;
kernel depth is a colour ramp in the viewer instead of a coordinate.

The layout it replaced was a layered DAG — `y = depth / max_depth`, `x` a rank
by name inside a `(band, depth)` cell — and it was a lattice by construction:
134 distinct y values for 58,612 nodes, with 3,478 of them crammed into the
densest 0.015-tall strip. Worse, it fought the banding it was drawn from.
Communities are strong (Q = 0.673, 77.1% of drawn edges stay inside their
band) but they are *not* depth-localised: the largest band holds 8,608 nodes
spanning depths 0→132. Pinning `y` to depth therefore smeared every community
into a full-height ribbon, and 31 overlapping ribbons is what reached the
screen. `--layout grid` still draws it.

The replacement runs on the **drawn** edge set, undirected — the same subgraph
the banding clusters, and for the same reason: the elided hubs touch
everything, so including them collapses most distances to two and leaves the
layout nothing to separate. Four steps:

1. **Coarsen** by heavy-edge matching, to a thousand nodes. Edges score
   `w / (mass_u * mass_v)` — heavy *relative* to what the two ends already
   weigh. The mass term is load-bearing: without it one supernode ran away to
   27,209 of the 58,612 nodes with degree 15,358, every other node became a
   leaf on a star, and a matching cannot coarsen a star. The hierarchy
   plateaued at 15,466 nodes instead of reaching twelve levels and 975.
2. **Seed** that coarsest level with PivotMDS: BFS distances to 100 max-min
   pivots, double-centred, projected onto the two leading right singular
   vectors. The eigenproblem is on a dense 100×100 matrix, so `numpy.linalg`
   solves it exactly — no scipy, no iterative solver. Unlike a force layout it
   has no local minima, so the large-scale arrangement is settled before any
   fine detail is committed to.
3. **Prolong and refine** back down, Fruchterman-Reingold at each rung.
   Repulsion is a two-grid approximation rather than a Barnes-Hut tree,
   because that expresses as array operations and is O(n) per step: exact-ish
   over a 5×5 fine-grid neighbourhood, centre-of-mass far field over a coarse
   grid.
4. **Normalise** on the central 99% of the positions, not the bounding box. A
   handful of nodes with no drawn edge fly far past the rest, and letting them
   set the scale squashed a converged layout to 0.5% of the canvas with 5,103
   nodes in its densest cell.

Deterministic, with two things pinned to keep it that way: the matching visits
nodes lightest-first with every tie broken on the lowest index, and the
eigenvectors have their signs canonicalised, since `eigh` fixes them only up
to a reflection. The guarantee is per-platform — the 100×100 solve goes
through LAPACK — which is the same scope the Louvain banding already had.

### What is drawn, and why it is not everything

Rendering all 1.1M edges of core Lean produces an unreadable mat. Two
measurements say why, and both are properties of the data rather than of the
renderer:

- **70% of the edges are redundant.** If `A` needs `B` and `C` while `B` also
  needs `C`, then `A -> C` adds no reachability. `transitive_reduction_mask`
  drops an edge when a two-step path already carries it — loss-free by
  construction, since the witness survives the drop — taking 1,123,115 edges
  to 337,635.
- **The top 100 constants absorb 56.6% of the edges.** `Eq` alone is the
  target of 41,788. Drawing them says only "everything needs equality", so
  `hub_mask` elides their incoming edges and flags the node instead. A hub
  must also clear ten times the mean in-degree, which is what stops
  `--hub-count 100` from swallowing a small graph whole.

Together: **219,640 edges drawn of 1,123,115**, and 46.9x less total line
length on screen. That ratio was 11.6x under the layered grid; the layout is
what widened it, because the edges the reduction keeps are the local ones and
a layout that puts adjacent nodes near each other makes exactly those short.

Nothing is discarded. `edges.u32` and both CSR directions still carry every
edge, so closures, degree counts and the shipped adjacency are exactly what
they were; the reduction is expressed only as `edge_order.u32`, a ranking the
viewer draws a prefix of. `tests/test_demo_reduce.py` pins that distinction by
re-deriving `closure` and `rclosure` for every node from the *reduced*
adjacency and demanding the library's answer.

### Why the bands are communities, not namespaces

Namespaces are not a partition of this graph. At declaration level the leading
name component is usually a type rather than a library — `Acc`, `Add`, `Nat`
— so core Lean yields 1,960 bands before splitting and 6,369 after, each a
few nodes wide. Ordering within a band then cannot matter, and measured mean
horizontal edge length confirmed it: name order scored 0.3253 against a random
baseline of 0.3321.

So `community_bands` runs Louvain over the drawn edges instead. Core Lean comes
out at **modularity 0.673** in 180 communities, 31 of which survive the
`--min-band` floor and hold 99.7% of the nodes — data structures
(`List`/`String`/`Std`), machine integers (`USize`/`UInt*`), arithmetic
(`Int`/`Nat`), metaprogramming (`Lean`/`Float`). Those groupings cut across the
namespace tree, which is the point.

Louvain runs on the *reduced* graph deliberately: the elided hubs touch
everything and wash the structure out. Scoring the alternatives by mean
horizontal edge length over the drawn set:

| x axis | mean dx |
| --- | --- |
| namespace bands + name order | 0.2707 |
| namespace bands + spectral (Fiedler) order | 0.2722 |
| no bands + spectral order | 0.1959 |
| **community bands** | **0.1009** |

Spectral ordering is therefore *not* implemented: it adds 5% on top of
community bands, and looked like the answer only while the bands were
namespaces. `--bands namespace` restores the old x axis.

### Buffers

| file | dtype | what |
| --- | --- | --- |
| `nodes.f32` | `float32[V,2]` | positions in `[-1,1]²` |
| `node_kind.u8` | `uint8` | index into `meta.json`'s `kinds` |
| `node_flags.u8` | `uint8` | bit 0 internal, bit 1 block, bit 2 unsafe |
| `node_depth.u16` | `uint16` | kernel depth |
| `names.bin` + `names_idx.u32` | blob + offsets | group labels |
| `name_order.u32` | `uint32[V]` | sorted-by-name permutation, for search |
| `members.bin` + `members_idx.u32` | blob + offsets | block members, newline-joined |
| `edges.u32` | `uint32[E,2]` | deduplicated, sorted by `(src, dst)` — every edge |
| `edge_order.u32` | `uint32[E]` | draw ranking; the viewer draws a prefix |
| `node_comm.u16` | `uint16` | community band, for the colour toggle |
| `fwd_*`, `rev_*` | CSR | adjacency both directions — every edge |

The viewer puts all of it in textures and synthesises geometry from
`gl_VertexID`, so no vertex buffers exist. Highlighting a closure re-uploads a
4-byte-per-node attribute texture rather than rebuilding a vertex buffer that
is 160 MB at full edge budget. Node indices travel through float textures,
exact below 2²⁴ — about 16.7M nodes, far past anything `lean4export` emits.

## What is measured, and what is not

### Measured

On the committed fixture (30 declarations): 18 nodes, 17 edges, max depth 3,
every buffer round-trips, and the CSR reproduces both closures for every node.

On a real `lean4export` run of `Init` — 6,496,806 records, Lean 4.36.0-rc2,
Apple M1 Pro, 16 GiB:

| | |
| --- | --- |
| reduce | 6.5M lines in **7.7s** (845k lines/s) |
| group | 58,612 nodes / 1,123,115 edges in 5.8s |
| reduce + band | 8.2s (transitive reduction 1.0s, Louvain 6s) |
| lay out | 17.2s (12 coarsening levels, 100 pivots, 200 iterations) |
| total | **39s**, 570 MB peak RSS |
| expression table | 6,138,291 expressions, 0.10 GB (16 B each, as designed) |
| completeness | `missing_total: 0`, `cyclic_nodes: 0`, 18,400 internal declarations kept |
| max depth | 133 |
| drawn edges | **219,640** of 1,123,115; 100 hubs elided above in-degree 2,029 |
| bands | 31 communities, modularity **0.673** |
| mean drawn-edge length | **0.0734**, against 0.4136 for `--layout grid` (5.6x shorter) |
| canvas | 57,269 distinct y values, against 134 for `--layout grid` |
| ink on screen | **46.9x** less total line length than drawing every edge |
| buffers | 27 MB |

Reduction also makes the graph shallower to read. Edges joining *adjacent*
depth layers rise 6.4% → 21.2% → 29.3% as the redundant edges and then the hub
edges come out, and the median edge span falls from 32 layers to 10 — because
the redundant edges are exactly the ones that skip levels. That mattered more
when depth was an axis; it still says the drawn edge set is the local one.

That run also says the graph is the right graph, not merely a self-consistent
one. The most depended-on constant in the environment is `Eq` at in-degree
41,788, then `Nat` at 32,377, then `OfNat.ofNat`, `instOfNatNat`, `congrArg`
and `Eq.trans`. The deepest node is a late string theorem nothing depends on.
162 sinks, 17,304 roots, median depth 31.

So the risk the plan called top — that a pure-Python dependency walk would be
unusably slow — did not materialise. The reducer is not the bottleneck.

Rebuilds are reproducible: two runs of the same export differ only in
`meta.json`'s `reduce_seconds`, with every binary buffer byte-identical,
Louvain included.

### Not measured: full Mathlib

`lean4export` is. It retains every expression it visits, and on this 16 GiB
machine a whole-`Mathlib` export climbed to **7.5 GB resident and was then
killed by the kernel** — `SIGKILL`, 51 minutes in, roughly 115M lines and
1.2 GB of gzipped NDJSON written, with throughput already down from
~160 MB/min to ~6 MB/min and memory still climbing. Nothing was wrong with the
reducer; it was never reached. The staged two-step above is what makes the
*reducer* side survivable, but it cannot shrink the exporter.

If you have the headroom (a 32 GiB+ machine), the two-step command above is
expected to work as written; the numbers to fill in are node and edge counts,
wall time, and output size. If you do not, the route to a complete graph is to
export Mathlib in module chunks and merge: every `lean4export` invocation is
transitively closed, so the union of per-chunk name-level edges is the whole
graph, and each chunk bounds the exporter's working set. That merge is not
implemented here.

### Not measured: the browser

Mean edge length is a proxy for legibility, not a measurement of it. A 46.9x
cut in drawn ink and a 5.6x cut in mean drawn-edge length are strong evidence
that position now carries information, but they are not the same claim as
"easier to read".

One number does point the other way, and it is worth stating plainly. Colouring
nodes by community band and asking what share of them land in a pixel their own
band dominates, the grid scores 0.974 and this layout 0.62, against shuffled-label
null baselines of 0.304 and 0.241. The grid's score is largely trivial -- its
bands *are* non-overlapping columns, so no two can share an x range -- but 0.62
is only weakly above its own null. Clusters are less visually separable here
than the raw numbers suggest, and the next move if that matters is to lay the 31
communities out as explicit non-overlapping regions rather than hoping the force
layout separates them.

No frame rate, no first-paint time, and no interaction has been checked on a
real GPU. What is checked is narrow, and for one version of this file it was
not checked at all: an earlier draft of this paragraph claimed the JavaScript
parsed when it did not. A backtick inside a GLSL comment closed the JS template
literal holding the shader, so `app.js` never parsed, `main()` never ran, and
the viewer sat on its static boot message indefinitely. This repository has no
JavaScript toolchain -- no package.json, no bundler, no linter -- so nothing in
the Python gate noticed, and the only symptom was a loading screen.

`tests/test_demo_web_assets.py` now closes that gap: it parses `app.js` with
whatever JS engine the machine has (`node --check`, else JavaScriptCore, else
skip) and asserts that no shader source was truncated. Beyond that, every
buffer the viewer fetches is checked to serve with the right byte count. The
largest graph actually rendered is the fixture's 18 nodes; the 58,612-node
`Init` graph above was verified as buffers, not on screen.

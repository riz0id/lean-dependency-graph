# lean-dep-graph

Declaration-level dependency graphs for Lean 4 packages.

For any declaration, computes the set of declarations the kernel must have
loaded in order to accept it.

## What it models

Granularity is the **declaration**, never the command. Every check-dependency
edge lives in an elaborated `Expr`, and commands — `open`, `variable`,
`attribute`, `notation`, `set_option`, `#check`, `example` — leave no constant
behind, so they cannot appear in a check closure. `variable` is inlined into
each declaration that mentions it, so its dependencies are captured
automatically through those declarations.

Nodes are declarations, except where the kernel requires a group to be accepted
as a unit: mutual inductive blocks, mutual definition blocks, and the four
quotient declarations are each one node. Queries take and return individual
declaration names and translate through the grouping for you.

Source data is the NDJSON export from
[`lean4export`](https://github.com/leanprover/lean4export), which contains
kernel-level terms and nothing else — no tactics, no notation, no elaborator
state. `.olean` files cannot be read directly: they are memory-mapped compacted
heap regions with no stable cross-version format.

Consequently the graph carries **no source positions, no defining module, and
no attributes**. None are required for a check graph; all three are structurally
absent from the export format.

## Prerequisites

A `lean4export` binary built against the *same toolchain* as the project you are
analysing. This is not auto-built — resolving and building a matching toolchain
is `elan`'s job, and a mismatch between the binary and the project's `.olean`
files is the single most common failure (the error message names both versions).

```bash
git clone https://github.com/leanprover/lean4export
cd lean4export
echo "$(cat /path/to/your/project/lean-toolchain)" > lean-toolchain
lake build
```

## Usage

```python
import lean_dep_graph as ldg

g = ldg.graph_for_project("/path/to/project", binary="/path/to/lean4export")

ldg.closure(g, "Foo.bar")  # everything the kernel needs for Foo.bar
ldg.rclosure(g, "Foo.bar")  # everything that would break if it changed
ldg.axioms(g, "Foo.bar")  # axioms it rests on
ldg.path(g, "Foo.bar", "Classical.choice")  # why it rests on them
ldg.topological_order(g)  # a valid kernel insertion order
ldg.stats(g)  # summary counts

g.members("Tree")  # the whole mutual block
ldg.closure(g, "Foo.bar", internal=False)  # hide generated declarations
```

Internal declarations (`_proof_N`, `match_N`, `_private.*`) are always kept in
the graph — they are genuinely required for checking. `internal=False` filters
them from a result without short-circuiting traversal through them.

Working from an export you already have:

```python
g = ldg.graph_for_export_file("out.ndjson")
```

Producing one by hand:

```bash
lake build MyLib:leanArts
lake env /path/to/lean4export MyLib.A MyLib.B > out.ndjson
```

Always build before exporting. Lean performs **no freshness check at import**:
an `.olean` records its imports by name with no hash, so a stale artifact is
trusted as-is and yields a wrong graph with no error.

## Development

```bash
nix develop          # or `direnv allow`
uv sync
uv run pytest        # parser and graph tests; no Lean toolchain needed
uv run pytest -m lean    # differential tests against lean4export
ty check             # strict type check, whole project
ruff check . && ruff format .
```

Most tests run against synthesised export streams and need no Lean at all. Only
the differential suite requires `elan` and a built `lean4export`; it is
deselected by default.

### On NixOS

The dev shell sets `NIX_LD` and `NIX_LD_LIBRARY_PATH` so prebuilt binary wheels
can find a loader, but this additionally requires `programs.nix-ld.enable = true`
system-side, which a flake cannot provide:

```nix
programs.nix-ld.enable = true;
```

`LD_LIBRARY_PATH` is deliberately not set — on non-NixOS Linux it forces the
host dynamic linker to load Nix's glibc and crashes Python.

## Correctness

The exporter accepts `-- <decl>` and computes the transitive closure itself,
which makes it a free and exact oracle: for any declaration, the computed
closure must equal the set of declarations in a single-declaration re-export.
That check is what the `lean` test suite runs, and it is what catches the two
extraction mistakes that are otherwise silent — dropping the structure name
carried by `Expr.proj`, and dropping the `Nat`/`String` implied by numeric and
string literals.

`axioms()` computes a genuine transitive closure and therefore diverges from
`#print axioms`: Lean's `collectAxioms` does not collect axioms referenced by
other axioms ([lean4 #8840](https://github.com/leanprover/lean4/issues/8840)),
so its output is not sound.

"""Declaration-level dependency graphs for Lean 4 packages.

Computes, for any declaration, the set of declarations the kernel must have
loaded in order to accept it.  Source data is the NDJSON export produced by
``lean4export``, which contains kernel-level terms and nothing else -- no
tactics, no notation, no elaborator state.

Granularity is the declaration, never the command.  Every check-dependency
edge lives in an elaborated ``Expr``, and commands (``open``, ``variable``,
``attribute``, ``notation``, ``#check``, ``example``) leave no constant behind,
so they cannot appear in a check closure.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .export import Exporter, LeanProject, LeanToolingError
from .graph import Decl, DeclGraph, Group, build_graph, is_internal
from .ndjson import ANONYMOUS, Export, ExportFormatError, load, load_path
from .queries import (
    CyclicGraphError,
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

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "ANONYMOUS",
    "CyclicGraphError",
    "Decl",
    "DeclGraph",
    "Export",
    "ExportFormatError",
    "Exporter",
    "Group",
    "LeanProject",
    "LeanToolingError",
    "axioms",
    "build_graph",
    "closure",
    "cycles",
    "deps",
    "graph_for_export_file",
    "graph_for_project",
    "is_internal",
    "load",
    "load_path",
    "path",
    "rclosure",
    "rdeps",
    "stats",
    "topological_order",
]

__version__ = "0.1.0"


def graph_for_export_file(path: str | Path, *, include_literal_types: bool = True) -> DeclGraph:
    """Build a graph from an existing ``.ndjson`` export on disk."""
    return build_graph(load_path(path), include_literal_types=include_literal_types)


def graph_for_project(
    root: str | Path,
    *,
    lib: str | None = None,
    modules: Sequence[str] | None = None,
    binary: str | Path | None = None,
    build: bool = True,
    include_literal_types: bool = True,
) -> DeclGraph:
    """Export a Lake project and build its declaration dependency graph.

    Builds Lean artifacts first by default.  Leaving ``build=False`` risks
    reading stale ``.olean`` files, which Lean trusts without any freshness
    check and which therefore produce a wrong graph rather than an error.
    """
    project = LeanProject(Path(root).expanduser().resolve())
    exporter = Exporter.discover(binary)

    if build:
        project.build(lib)

    mods = list(modules) if modules is not None else project.modules(lib)
    if not mods:
        raise LeanToolingError(
            f"no modules reported for {project.root}"
            + (f" (lib {lib!r})" if lib else " (default targets)")
        )

    stale = project.olean_missing(mods)
    if stale:
        raise LeanToolingError(
            "no .olean found for: "
            + ", ".join(stale[:5])
            + (" ..." if len(stale) > 5 else "")
            + "\nRun `lake build` in the project, or pass build=True."
        )

    export = exporter.export(project, mods)
    return build_graph(export, include_literal_types=include_literal_types)

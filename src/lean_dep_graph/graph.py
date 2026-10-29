"""Declaration model and graph construction."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import networkx as nx

from .ndjson import Export
from .terms import collect_expr_deps

#: Declaration kinds as they appear in the graph.  ``ctor`` and ``rec`` are
#: derived from the members of an ``inductive`` record.
Kind = str

_MATCH_AUX = re.compile(r"match_\d+")


def is_internal(name: str) -> bool:
    """Whether a name looks machine-generated rather than hand-written.

    A presentation heuristic only: these declarations are genuinely required
    for kernel checking and are never excluded from the graph.  Covers
    ``_proof_N``, ``_private.*``, ``_unary``, ``_example`` (any underscore-led
    component) and ``match_N``.
    """
    return any(part.startswith("_") or _MATCH_AUX.fullmatch(part) for part in name.split("."))


@dataclass(frozen=True)
class Decl:
    """One constant in the export."""

    name: str
    kind: Kind
    #: Expression indices whose constants this declaration depends on.
    roots: tuple[int, ...]
    level_params: tuple[str, ...] = ()
    is_unsafe: bool = False

    @property
    def internal(self) -> bool:
        return is_internal(self.name)


@dataclass
class Group:
    """A set of declarations the kernel must accept as a unit.

    A mutual inductive block, a mutual definition block, or the four quotient
    declarations.  Singleton for everything else.
    """

    id: str
    members: tuple[str, ...]
    kind: Kind

    @property
    def is_block(self) -> bool:
        return len(self.members) > 1


def _level_params(ex: Export, raw: Any) -> tuple[str, ...]:
    return tuple(ex.name(int(i)) for i in raw or ())


def _parse_decls(ex: Export) -> tuple[dict[str, Decl], list[tuple[str, tuple[str, ...]]]]:
    """Return every declaration plus the grouping constraints to apply.

    The second element lists ``(kind, member_names)`` for each group; singleton
    groups are implied by any declaration not named in one.
    """
    decls: dict[str, Decl] = {}
    groups: list[tuple[str, tuple[str, ...]]] = []
    quot_members: list[str] = []

    def add(name: str, kind: Kind, roots: tuple[int, ...], body: dict[str, Any]) -> None:
        decls[name] = Decl(
            name=name,
            kind=kind,
            roots=roots,
            level_params=_level_params(ex, body.get("levelParams")),
            is_unsafe=bool(body.get("isUnsafe", body.get("safety") == "unsafe")),
        )

    for key, body in ex.decls:
        if key == "inductive":
            members: list[str] = []
            for tv in body.get("types", ()):
                name = ex.name(int(tv["name"]))
                add(name, "inductive", (int(tv["type"]),), tv)
                members.append(name)
            for cv in body.get("ctors", ()):
                name = ex.name(int(cv["name"]))
                add(name, "ctor", (int(cv["type"]),), cv)
                members.append(name)
            for rv in body.get("recs", ()):
                name = ex.name(int(rv["name"]))
                # A recursor's rules carry right-hand sides that reference
                # constants appearing nowhere in the recursor's type.
                roots = (int(rv["type"]), *(int(r["rhs"]) for r in rv.get("rules", ())))
                add(name, "rec", roots, rv)
                members.append(name)
            if members:
                groups.append(("inductive", tuple(members)))
            continue

        name = ex.name(int(body["name"]))

        if key == "axiom":
            add(name, "axiom", (int(body["type"]),), body)
        elif key == "quot":
            add(name, "quot", (int(body["type"]),), body)
            quot_members.append(name)
        elif key in ("def", "thm", "opaque"):
            add(name, key, (int(body["type"]), int(body["value"])), body)
            allg = tuple(ex.name(int(i)) for i in body.get("all", ()) or ())
            if len(allg) > 1:
                groups.append((key, allg))
        elif key == "constructor":
            add(name, "ctor", (int(body["type"]),), body)
        elif key == "recursor":
            roots = (int(body["type"]), *(int(r["rhs"]) for r in body.get("rules", ())))
            add(name, "rec", roots, body)

    # The kernel takes `quotDecl` as a single field-less declaration and emits
    # all four quotient constants from it, so they enter as one unit.
    if len(quot_members) > 1:
        groups.append(("quot", tuple(quot_members)))

    return decls, groups


def _build_groups(
    decls: dict[str, Decl], constraints: list[tuple[str, tuple[str, ...]]]
) -> tuple[dict[str, Group], dict[str, str]]:
    """Assign every declaration to a group; returns (groups, name -> group id)."""
    node_of: dict[str, str] = {}
    groups: dict[str, Group] = {}

    for kind, members in constraints:
        present = tuple(m for m in members if m in decls)
        if not present:
            continue
        # Mutual members each carry the same `all` list, so a block is seen
        # once per member; the first sighting wins and the rest are no-ops.
        if any(m in node_of for m in present):
            continue
        gid = present[0]
        groups[gid] = Group(id=gid, members=present, kind=kind)
        for m in present:
            node_of[m] = gid

    for name, decl in decls.items():
        if name not in node_of:
            groups[name] = Group(id=name, members=(name,), kind=decl.kind)
            node_of[name] = name

    return groups, node_of


class DeclGraph:
    """A declaration-level dependency graph over a Lean export.

    Nodes are :class:`Group` instances -- usually a single declaration, but a
    mutual inductive block, mutual definition block, or the quotient package
    is one node, because the kernel will only accept those as a unit.  Public
    queries take and return individual declaration names and translate through
    the grouping internally.
    """

    def __init__(
        self,
        export: Export,
        *,
        include_literal_types: bool = True,
    ) -> None:
        self.export = export
        self.include_literal_types = include_literal_types

        self.decls, constraints = _parse_decls(export)
        self.groups, self._node_of = _build_groups(self.decls, constraints)

        #: Constants referenced by some declaration but never declared.  Should
        #: be empty for a whole-module export; a non-empty set means the export
        #: was a partial slice, or that extraction produced a bogus name.
        self.missing: set[str] = set()

        self._raw_deps: dict[str, frozenset[str]] = {}
        self.graph = self._build_graph()

    # -- construction ------------------------------------------------------

    def _build_graph(self) -> nx.DiGraph[str]:
        g: nx.DiGraph[str] = nx.DiGraph()
        for gid, group in self.groups.items():
            g.add_node(gid, kind=group.kind, members=group.members, block=group.is_block)

        for name, decl in self.decls.items():
            deps = frozenset(
                collect_expr_deps(
                    self.export,
                    decl.roots,
                    include_literal_types=self.include_literal_types,
                )
            )
            self._raw_deps[name] = deps

        for name, deps in self._raw_deps.items():
            src = self._node_of[name]
            for dep in deps:
                dst = self._node_of.get(dep)
                if dst is None:
                    self.missing.add(dep)
                    continue
                if dst != src:  # self-edges within a mutual block carry no information
                    g.add_edge(src, dst)
        return g

    # -- accessors ---------------------------------------------------------

    def __contains__(self, name: str) -> bool:
        return name in self.decls

    def __len__(self) -> int:
        return len(self.decls)

    @property
    def names(self) -> frozenset[str]:
        """Every declaration name in the export."""
        return frozenset(self.decls)

    def decl(self, name: str) -> Decl:
        try:
            return self.decls[name]
        except KeyError:
            raise KeyError(f"no declaration named {name!r} in this export") from None

    def node_of(self, name: str) -> str:
        """The group id a declaration belongs to."""
        try:
            return self._node_of[name]
        except KeyError:
            raise KeyError(f"no declaration named {name!r} in this export") from None

    def group_of(self, name: str) -> Group:
        return self.groups[self.node_of(name)]

    def members(self, name: str) -> tuple[str, ...]:
        """All declarations that enter the kernel together with ``name``."""
        return self.group_of(name).members

    def direct_deps(self, name: str) -> frozenset[str]:
        """Constants referenced by this one declaration's type and value.

        Not grouped and not transitive -- the raw extraction result.
        """
        if name not in self.decls:
            raise KeyError(f"no declaration named {name!r} in this export")
        return self._raw_deps[name]


def build_graph(export: Export, *, include_literal_types: bool = True) -> DeclGraph:
    """Build a :class:`DeclGraph` from a parsed export."""
    return DeclGraph(export, include_literal_types=include_literal_types)

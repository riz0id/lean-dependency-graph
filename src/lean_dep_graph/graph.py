"""Declaration model and graph construction."""

from __future__ import annotations
from networkx.classes.digraph import DiGraph

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, TypeAlias

import networkx as nx

from .ndjson import Export
from .terms import collect_expr_deps

__all__: list[str] = [
    "Decl",
    "DeclGraph",
    "Group",
    "Kind",
    "build_graph",
    "is_internal",
]

Kind: TypeAlias = str

MATCH_AUX: re.Pattern[str] = re.compile(pattern=r"match_\d+")


def is_internal(name: str) -> bool:
    return any(part.startswith("_") or MATCH_AUX.fullmatch(part) for part in name.split("."))


@dataclass(frozen=True)
class Decl:
    name: str
    kind: Kind
    roots: tuple[int, ...]
    level_params: tuple[str, ...] = ()
    is_unsafe: bool = False

    @property
    def internal(self) -> bool:
        return is_internal(self.name)


@dataclass
class Group:

    id: str
    members: tuple[str, ...]
    kind: Kind

    @property
    def is_block(self) -> bool:
        return len(self.members) > 1


def level_params(ex: Export, raw: Iterable[Any] | None) -> tuple[str, ...]:
    return tuple(ex.name(int(i)) for i in raw or ())


def decl_name(ex: Export, body: dict[str, Any]) -> str:
    return ex.name(int(body["name"]))


def parse_decls(ex: Export) -> tuple[dict[str, Decl], list[tuple[str, tuple[str, ...]]]]:
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
            level_params=level_params(ex, body.get("levelParams")),
            is_unsafe=bool(body.get("isUnsafe", body.get("safety") == "unsafe")),
        )

    for key, body in ex.decls:
        match key:
            case "inductive":
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
                    roots = (int(rv["type"]), *(int(r["rhs"]) for r in rv.get("rules", ())))
                    add(name, "rec", roots, rv)
                    members.append(name)

                if members:
                    groups.append(("inductive", tuple(members)))
            case "axiom":
                add(decl_name(ex, body), "axiom", (int(body["type"]),), body)
            case "quot":
                name = decl_name(ex, body)
                add(name, "quot", (int(body["type"]),), body)
                quot_members.append(name)
            case "def" | "thm" | "opaque":
                name = decl_name(ex, body)
                add(name, key, (int(body["type"]), int(body["value"])), body)
                allg = tuple(ex.name(int(i)) for i in body.get("all", ()) or ())
                if len(allg) > 1:
                    groups.append((key, allg))
            case "constructor":
                add(decl_name(ex, body), "ctor", (int(body["type"]),), body)
            case "recursor":
                roots = (int(body["type"]), *(int(r["rhs"]) for r in body.get("rules", ())))
                add(decl_name(ex, body), "rec", roots, body)

    if len(quot_members) > 1:
        groups.append(("quot", tuple(quot_members)))

    return decls, groups


def build_groups(
    decls: dict[str, Decl], constraints: list[tuple[str, tuple[str, ...]]]
) -> tuple[dict[str, Group], dict[str, str]]:
    """Assign every declaration to a group; returns (groups, name -> group id)."""
    node_of: dict[str, str] = {}
    groups: dict[str, Group] = {}

    for kind, members in constraints:
        present = tuple(m for m in members if m in decls)
        if not present:
            continue
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
        self.export: Export = export
        self.include_literal_types: bool = include_literal_types

        self.decls, constraints = parse_decls(export)
        self.groups, self.group_ids = build_groups(self.decls, constraints)

        self.missing: set[str] = set()

        self.raw_deps: dict[str, frozenset[str]] = {}
        self.graph: DiGraph[str, dict[str, Any], dict[str, Any]] = self.construct_digraph()

    def construct_digraph(self) -> nx.DiGraph[str]:
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
            self.raw_deps[name] = deps

        for name, deps in self.raw_deps.items():
            src = self.group_ids[name]
            for dep in deps:
                dst = self.group_ids.get(dep)
                if dst is None:
                    self.missing.add(dep)
                    continue
                if dst != src:
                    g.add_edge(src, dst)
        return g

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
            return self.group_ids[name]
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
        return self.raw_deps[name]


def build_graph(export: Export, *, include_literal_types: bool = True) -> DeclGraph:
    """Build a :class:`DeclGraph` from a parsed export."""
    return DeclGraph(export, include_literal_types=include_literal_types)

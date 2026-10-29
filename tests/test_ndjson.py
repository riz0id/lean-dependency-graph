"""Reader-level tests: interning, name resolution, error reporting."""

from __future__ import annotations

import pytest

from lean_dep_graph.ndjson import ExportFormatError, load

from .conftest import ExportBuilder


def test_meta_is_captured(builder: ExportBuilder) -> None:
    ex = load(builder.lines)
    assert ex.format_version == "3.1.0"
    assert ex.lean_version == "4.35.0"


def test_anonymous_name_is_index_zero(builder: ExportBuilder) -> None:
    ex = load(builder.lines)
    assert ex.name(0) == ""


def test_name_resolution_roundtrips(builder: ExportBuilder) -> None:
    for dotted in ("Nat", "Nat.succ", "Foo.Bar.Baz.qux", "_private.M.0.proof"):
        builder.name(dotted)
    ex = load(builder.lines)
    resolved = {ex.name(i) for i in ex.names}
    # Every intermediate prefix is interned too, so check containment.
    assert {"Nat", "Nat.succ", "Foo.Bar.Baz.qux", "_private.M.0.proof"} <= resolved


def test_numeric_name_components(builder: ExportBuilder) -> None:
    idx = builder.name("Foo.3.bar")
    ex = load(builder.lines)
    assert ex.name(idx) == "Foo.3.bar"


def test_name_components_are_interned_once(builder: ExportBuilder) -> None:
    a = builder.name("Foo.Bar.x")
    b = builder.name("Foo.Bar.y")
    ex = load(builder.lines)
    assert ex.name(a) == "Foo.Bar.x"
    assert ex.name(b) == "Foo.Bar.y"
    # 'Foo' and 'Foo.Bar' emitted once each, plus 'x' and 'y'.
    assert len(ex.names) == 4


def test_deeply_nested_name_does_not_recurse() -> None:
    b = ExportBuilder()
    deep = ".".join(f"n{i}" for i in range(5000))
    idx = b.name(deep)
    ex = load(b.lines)
    assert ex.name(idx) == deep


def test_blank_lines_are_skipped(builder: ExportBuilder) -> None:
    lines = [*builder.lines, "", "   "]
    assert load(lines).format_version == "3.1.0"


def test_invalid_json_reports_line_number() -> None:
    with pytest.raises(ExportFormatError, match="line 2"):
        load(['{"meta": {}}', "{not json"])


def test_unrecognised_record_raises() -> None:
    with pytest.raises(ExportFormatError, match="unrecognised record"):
        load(['{"somethingElse": 1}'])


def test_dangling_name_index_raises(builder: ExportBuilder) -> None:
    ex = load(builder.lines)
    with pytest.raises(ExportFormatError, match="dangling name index"):
        ex.name(999)


def test_dangling_expr_index_raises(builder: ExportBuilder) -> None:
    ex = load(builder.lines)
    with pytest.raises(ExportFormatError, match="dangling expression index"):
        ex.expr(999)

"""Unit tests for the exporter plumbing.

A Python child process stands in for ``lean4export``, so these need no Lean
toolchain and run in the default suite, unlike ``test_differential.py``.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from lean_dep_graph import export
from lean_dep_graph.export import Exporter, LeanProject, LeanToolingError

#: Interleaves stdout lines with 4 KiB stderr writes -- 800 KiB of stderr in
#: total, far past any pipe buffer, so an undrained stderr pipe deadlocks here.
CHATTY = """
import sys
for i in range(200):
    sys.stdout.write('{"meta": %d}\\n' % i)
    sys.stdout.flush()
    sys.stderr.write("w" * 4096)
    sys.stderr.flush()
"""

SOURCES: dict[str, str] = {"chatty": CHATTY, "failing": CHATTY + "\nsys.exit(3)\n"}


@pytest.fixture
def project(tmp_path: Path) -> LeanProject:
    (tmp_path / "lakefile.toml").write_text("", encoding="utf-8")
    return LeanProject(tmp_path)


@pytest.fixture
def exporter(monkeypatch: pytest.MonkeyPatch) -> Exporter:
    """An ``Exporter`` that runs ``SOURCES[modules[0]]`` instead of ``lake``.

    Patching ``command`` is the only seam: ``stream`` builds its argv through
    it, so every real invocation starts with ``lake env``.  The genuine
    ``command`` is covered by its own tests below.
    """

    def fake(self: Exporter, modules: Sequence[str], **_: object) -> list[str]:
        return [sys.executable, "-c", SOURCES[modules[0]]]

    monkeypatch.setattr(Exporter, "command", fake)
    return Exporter(Path(sys.executable))


@pytest.mark.timeout(60)
def test_stream_survives_chatty_stderr(project: LeanProject, exporter: Exporter) -> None:
    """A child writing 800 KiB to stderr must not deadlock against stdout."""
    lines = list(exporter.stream(project, ["chatty"]))

    assert len(lines) == 200
    assert lines[0].strip() == '{"meta": 0}'
    assert lines[-1].strip() == '{"meta": 199}'


@pytest.mark.timeout(60)
def test_stream_reports_stderr_tail_on_failure(project: LeanProject, exporter: Exporter) -> None:
    with pytest.raises(LeanToolingError) as exc:
        list(exporter.stream(project, ["failing"]))

    message = str(exc.value)

    assert "lean4export failed (3)" in message
    assert "earlier bytes omitted" in message
    assert len(message) < 20_000, "stderr tail is unbounded"


@pytest.mark.timeout(60)
def test_stream_abandoned_early_does_not_raise(project: LeanProject, exporter: Exporter) -> None:
    """Closing the stream early is the consumer's choice, not an export failure."""
    stream = exporter.stream(project, ["failing"])

    assert next(stream).strip() == '{"meta": 0}'

    stream.close()


def test_command_assembles_flags_before_modules() -> None:
    exporter = Exporter(Path("/opt/lean4export"))

    assert exporter.command(["A", "B"]) == ["lake", "env", "/opt/lean4export", "A", "B"]
    assert exporter.command(["A"], unsafe=True, mdata=True) == [
        "lake",
        "env",
        "/opt/lean4export",
        "--export-unsafe",
        "--export-mdata",
        "A",
    ]


def test_command_separates_decls_only_when_present() -> None:
    exporter = Exporter(Path("/opt/lean4export"))

    assert exporter.command(["A"], decls=["D"]) == [
        "lake",
        "env",
        "/opt/lean4export",
        "A",
        "--",
        "D",
    ]
    assert "--" not in exporter.command(["A"], decls=[])
    assert "--" not in exporter.command(["A"], decls=None)


def olean_at(libdir: Path, module: str) -> Path:
    """Create an empty ``.olean`` for ``module`` under ``libdir``, and return ``libdir``."""
    path = libdir / Path(*module.split(".")).with_suffix(".olean")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return libdir


def test_lean_path_is_split_on_the_platform_separator(
    project: LeanProject, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = os.pathsep.join(["/one", "", "/two"]) + "\n"
    monkeypatch.setattr(export, "run", lambda cmd, cwd: raw)

    assert project.lean_path() == [Path("/one"), Path("/two")]


def test_olean_missing_accepts_a_built_module(
    project: LeanProject, monkeypatch: pytest.MonkeyPatch
) -> None:
    lib = olean_at(project.root / ".lake" / "build" / "lib", "A.B")
    monkeypatch.setattr(LeanProject, "lean_path", lambda self: [lib])

    assert project.olean_missing(["A.B"]) == []


def test_olean_missing_searches_every_search_path_entry(
    project: LeanProject, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One entry holding the module is enough; the earlier misses are not failures."""
    first = project.root / "empty"
    second = olean_at(project.root / "second", "A.B")
    monkeypatch.setattr(LeanProject, "lean_path", lambda self: [first, second])

    assert project.olean_missing(["A.B"]) == []


def test_olean_missing_finds_dependency_package_artifacts(
    project: LeanProject, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dependency's artifacts live under .lake/packages/<pkg>/, not the root's lib."""
    root_lib = project.root / ".lake" / "build" / "lib"
    dep_lib = olean_at(
        project.root / ".lake" / "packages" / "dep" / ".lake" / "build" / "lib", "Dep.M"
    )
    monkeypatch.setattr(LeanProject, "lean_path", lambda self: [root_lib, dep_lib])

    assert project.olean_missing(["Dep.M"]) == []


def test_olean_missing_reports_genuinely_absent_modules(
    project: LeanProject, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check must stay useful, not become a no-op."""
    lib = olean_at(project.root / ".lake" / "build" / "lib", "A.B")
    monkeypatch.setattr(LeanProject, "lean_path", lambda self: [lib])

    assert project.olean_missing(["A.B", "No.Such"]) == ["No.Such"]


def test_olean_missing_fails_open_when_the_search_path_is_unavailable(
    project: LeanProject, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Blocking a valid export is the bug being fixed; never reintroduce it."""

    def boom(self: LeanProject) -> list[Path]:
        raise LeanToolingError("lake exploded")

    monkeypatch.setattr(LeanProject, "lean_path", boom)

    assert project.olean_missing(["Anything.At.All"]) == []

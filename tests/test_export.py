"""Unit tests for the exporter plumbing.

A Python child process stands in for ``lean4export``, so these need no Lean
toolchain and run in the default suite, unlike ``test_differential.py``.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

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

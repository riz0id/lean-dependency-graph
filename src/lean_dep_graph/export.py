"""Driving ``lake`` and ``lean4export`` to produce an export stream."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from .ndjson import Export, load

DEFAULT_BINARY_NAME = "lean4export"


class LeanToolingError(Exception):
    """A ``lake`` or ``lean4export`` invocation failed."""


def _run(cmd: Sequence[str], cwd: Path) -> str:
    proc = subprocess.run(  # noqa: S603
        list(cmd), cwd=cwd, capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        raise LeanToolingError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"cwd: {cwd}\n"
            f"stderr:\n{proc.stderr.strip()}"
        )
    return proc.stdout


@dataclass(frozen=True)
class LeanProject:
    """A Lake project on disk."""

    root: Path

    def __post_init__(self) -> None:
        if (
            not (self.root / "lakefile.toml").exists()
            and not (self.root / "lakefile.lean").exists()
        ):
            raise LeanToolingError(f"no lakefile.toml or lakefile.lean in {self.root}")

    @property
    def toolchain(self) -> str | None:
        """Contents of ``lean-toolchain``, if present."""
        path = self.root / "lean-toolchain"
        return path.read_text(encoding="utf-8").strip() if path.exists() else None

    def modules(self, lib: str | None = None) -> list[str]:
        """Module names from Lake's ``modules`` facet.

        The facet is not buildable, so it is reached through ``lake query``
        rather than ``lake build``.
        """
        target = f"{lib}:modules" if lib else ":modules"
        raw = _run(["lake", "query", "--json", target], self.root)
        return _parse_modules(raw)

    def build(self, lib: str | None = None) -> None:
        """Build Lean artifacts.

        Required before exporting.  Lean performs no freshness check at import:
        an ``.olean`` records its imports by name with no hash, so a stale
        artifact is trusted as-is and yields a wrong graph with no error.
        """
        target = f"{lib}:leanArts" if lib else "leanArts"
        _run(["lake", "build", target], self.root)

    def olean_missing(self, modules: Sequence[str]) -> list[str]:
        """Modules with no ``.olean`` under ``.lake/build/lib``."""
        libdir = self.root / ".lake" / "build" / "lib"
        missing = []
        for mod in modules:
            rel = Path(*mod.split(".")).with_suffix(".olean")
            # Lean 4.23+ nests artifacts under a `lean/` subdirectory.
            if not (libdir / rel).exists() and not (libdir / "lean" / rel).exists():
                missing.append(mod)
        return missing


def _parse_modules(raw: str) -> list[str]:
    """Extract module names from ``lake query --json`` output.

    Tolerant of shape: Lake has emitted both a single JSON document and
    one JSON value per line across versions, and entries may be bare strings
    or objects carrying the name under a few different keys.
    """
    text = raw.strip()
    if not text:
        return []

    payloads: list[object] = []
    try:
        payloads = [json.loads(text)]
    except json.JSONDecodeError:
        for line in text.splitlines():
            line = line.strip()
            if line:
                try:
                    payloads.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise LeanToolingError(
                        f"could not parse `lake query --json` output: {exc}"
                    ) from exc

    names: list[str] = []

    def absorb(value: object) -> None:
        if isinstance(value, str):
            names.append(value)
        elif isinstance(value, list):
            for item in value:
                absorb(item)
        elif isinstance(value, dict):
            for key in ("name", "module", "mod"):
                if isinstance(value.get(key), str):
                    names.append(value[key])
                    return
            for item in value.values():
                absorb(item)

    for payload in payloads:
        absorb(payload)

    # Preserve order, drop duplicates.
    return list(dict.fromkeys(names))


@dataclass(frozen=True)
class Exporter:
    """A built ``lean4export`` binary.

    The binary must be built against the same toolchain as the project's
    ``.olean`` files.  It is not auto-built: resolving and building a matching
    toolchain is ``elan``'s job, and owning that would mean owning its failure
    modes.
    """

    binary: Path

    @classmethod
    def discover(cls, binary: str | Path | None = None) -> Exporter:
        """Resolve the binary from an explicit path or ``PATH``."""
        if binary is not None:
            path = Path(binary).expanduser()
            if not path.exists():
                raise LeanToolingError(f"lean4export not found at {path}")
            return cls(path.resolve())
        found = shutil.which(DEFAULT_BINARY_NAME)
        if found is None:
            raise LeanToolingError(
                f"{DEFAULT_BINARY_NAME} not found on PATH. Build it against the target "
                "project's toolchain (see README) and pass its path explicitly."
            )
        return cls(Path(found).resolve())

    def command(
        self,
        modules: Sequence[str],
        *,
        decls: Sequence[str] | None = None,
        unsafe: bool = False,
        mdata: bool = False,
    ) -> list[str]:
        cmd = ["lake", "env", str(self.binary)]
        if unsafe:
            cmd.append("--export-unsafe")
        if mdata:
            cmd.append("--export-mdata")
        cmd.extend(modules)
        if decls:
            cmd.append("--")
            cmd.extend(decls)
        return cmd

    def stream(
        self,
        project: LeanProject,
        modules: Sequence[str],
        *,
        decls: Sequence[str] | None = None,
        unsafe: bool = False,
        mdata: bool = False,
    ) -> Iterator[str]:
        """Yield NDJSON lines, streaming rather than buffering the whole export."""
        cmd = self.command(modules, decls=decls, unsafe=unsafe, mdata=mdata)
        proc = subprocess.Popen(  # noqa: S603
            cmd,
            cwd=project.root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        try:
            yield from proc.stdout
        finally:
            proc.stdout.close()
            stderr = proc.stderr.read() if proc.stderr else ""
            if proc.stderr:
                proc.stderr.close()
            code = proc.wait()
            if code != 0:
                raise LeanToolingError(
                    f"lean4export failed ({code}).\n"
                    f"  binary:            {self.binary}\n"
                    f"  project toolchain: {project.toolchain or 'unknown'}\n"
                    f"  modules:           {', '.join(modules[:5])}"
                    f"{' ...' if len(modules) > 5 else ''}\n"
                    "A mismatch between the binary's toolchain and the project's is the "
                    "most common cause; both versions are shown above.\n"
                    f"stderr:\n{stderr.strip()}"
                )

    def export(
        self,
        project: LeanProject,
        modules: Sequence[str],
        *,
        decls: Sequence[str] | None = None,
        unsafe: bool = False,
        mdata: bool = False,
    ) -> Export:
        """Run the exporter and parse the result."""
        return load(self.stream(project, modules, decls=decls, unsafe=unsafe, mdata=mdata))

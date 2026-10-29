"""Driving ``lake`` and ``lean4export`` to produce an export stream."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from .ndjson import Export, load

__all__ = [
    "DEFAULT_BINARY_NAME",
    "Exporter",
    "LeanProject",
    "LeanToolingError",
]

DEFAULT_BINARY_NAME = "lean4export"

#: Child stderr retained for an error message, taken from the tail.
STDERR_TAIL_BYTES = 10_000


class LeanToolingError(Exception):
    """A ``lake`` or ``lean4export`` invocation failed."""


def run(cmd: Sequence[str], cwd: Path) -> str:
    proc: subprocess.CompletedProcess[str] = subprocess.run(
        args=list(cmd),
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )

    if proc.returncode == 0:
        return proc.stdout
    else:
        raise LeanToolingError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"cwd: {cwd}\n"
            f"stderr:\n{proc.stderr.strip()}"
        )


def read_tail(fh: IO[bytes], limit: int = STDERR_TAIL_BYTES) -> str:
    """Decode the last ``limit`` bytes of ``fh``, noting any truncation."""
    fh.seek(0, os.SEEK_END)
    size: int = fh.tell()
    fh.seek(max(0, size - limit))
    text: str = fh.read().decode("utf-8", errors="replace").strip()

    if size > limit:
        return f"[... {size - limit} earlier bytes omitted ...]\n{text}"

    return text


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
        raw = run(["lake", "query", "--json", target], self.root)
        return parse_modules(raw)

    def build(self, lib: str | None = None) -> None:
        run(
            cmd=["lake", "build", f"{lib}:leanArts" if lib else "leanArts"],
            cwd=self.root,
        )

    def olean_missing(self, modules: Sequence[str]) -> list[str]:
        libdir: Path = self.root / ".lake" / "build" / "lib"
        missing: list[str] = []

        for mod in modules:
            rel: Path = Path(*mod.split(".")).with_suffix(".olean")

            if not (libdir / rel).exists() and not (libdir / "lean" / rel).exists():
                missing.append(mod)

        return missing


def parse_modules(raw: str) -> list[str]:
    text: str = raw.strip()

    if not text:
        return []

    payloads: list[object] = []

    try:
        payloads: list[object] = [json.loads(text)]
    except json.JSONDecodeError:
        for line in text.splitlines():
            line: str = line.strip()

            if line:
                try:
                    payloads.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise LeanToolingError(
                        f"could not parse `lake query --json` output: {exc}"
                    ) from exc

    names: list[str] = []

    def absorb(value: object) -> None:
        match value:
            case str():
                names.append(value)
            case list():
                for item in value:
                    absorb(item)
            case dict():
                for key in ("name", "module", "mod"):
                    candidate = value.get(key)
                    if isinstance(candidate, str):
                        names.append(candidate)
                        return
                for item in value.values():
                    absorb(item)

    for payload in payloads:
        absorb(payload)

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
    ) -> Generator[str, None, None]:
        """Yield NDJSON lines, streaming rather than buffering the whole export.

        The child's stderr goes to a temporary file rather than a pipe.  With
        both streams piped and only stdout drained, a child that writes more
        than the pipe buffer (~64 KiB) to stderr blocks in ``write``, stops
        producing stdout, and deadlocks against a parent blocked on ``read``.
        """
        cmd = self.command(modules, decls=decls, unsafe=unsafe, mdata=mdata)

        with tempfile.TemporaryFile() as err:
            proc = subprocess.Popen(
                cmd,
                cwd=project.root,
                stdout=subprocess.PIPE,
                stderr=err,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            exhausted = False

            try:
                yield from proc.stdout
                exhausted = True
            finally:
                proc.stdout.close()

                # An abandoned stream means the consumer stopped on purpose,
                # usually while raising something more informative.  Reap the
                # child, but do not report its signal as an export failure.
                if not exhausted:
                    proc.kill()

                code = proc.wait()

                if code != 0 and exhausted:
                    raise LeanToolingError(
                        f"lean4export failed ({code}).\n"
                        f"  binary:            {self.binary}\n"
                        f"  project toolchain: {project.toolchain or 'unknown'}\n"
                        f"  modules:           {', '.join(modules[:5])}"
                        f"{' ...' if len(modules) > 5 else ''}\n"
                        "A mismatch between the binary's toolchain and the project's is the "
                        "most common cause; both versions are shown above.\n"
                        f"stderr:\n{read_tail(err)}"
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

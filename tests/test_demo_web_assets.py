"""Guards for the viewer's hand-written web assets.

The repository has no JavaScript toolchain -- no package.json, no bundler, no
linter -- so `app.js` is never parsed by anything until a browser loads it.
That gap let a syntax error ship: a backtick inside a GLSL comment closed the
JS template literal holding the shader, the whole file failed to parse, `main()`
never ran, and the viewer sat on its static boot message forever with nothing in
the Python gate noticing. These tests are the cheapest check that stands in for
the toolchain that is not here.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

APP_JS = Path(__file__).parent.parent / "demo" / "web" / "app.js"

# Every GLSL source in the viewer, each one a JS template literal.
SHADERS = ["VERT_COMMON", "NODE_VS", "NODE_FS", "EDGE_VS", "EDGE_FS"]

# `const NAME =` then anything up to the opening backtick (which is how the
# `VERT_COMMON +` prefix on the two vertex shaders is skipped), then the
# literal's body up to the first backtick that closes it.
SHADER_RE = "const {name} =[^`]*`(?P<body>.*?)`"


def shader_source(name: str, text: str) -> str:
    match = re.search(SHADER_RE.format(name=name), text, re.DOTALL)
    assert match is not None, f"no template literal found for {name}"
    body = match.group("body")
    assert isinstance(body, str)  # a named group that matched is always a str
    return body


@pytest.mark.parametrize("name", SHADERS)
def test_shader_sources_hold_no_backtick(name: str) -> None:
    """A backtick in GLSL ends the JS string holding it, so there must be none.

    This catches an *escaped* backtick, which stays inside the literal and
    reaches the GLSL compiler. An unescaped one closes the literal early, and
    the body extracted here then stops short of it -- that case is what
    `test_shader_sources_are_not_truncated` is for.
    """
    assert "`" not in shader_source(name, APP_JS.read_text())


@pytest.mark.parametrize("name", SHADERS)
def test_shader_sources_are_not_truncated(name: str) -> None:
    """Each shader must end on a closing brace.

    This is what catches a template literal that closed early: the captured
    body then stops mid-comment instead of at the end of a function body.
    """
    body = shader_source(name, APP_JS.read_text()).strip()
    assert body.endswith("}"), f"{name} ends with {body[-60:]!r}, not a closing brace"


def js_syntax_check(path: Path) -> tuple[str, str] | None:
    """Syntax-check a file with whatever JS engine is on this machine.

    Returns `(engine, output)`, or None if there is no engine to ask. `node
    --check` parses without executing; JavaScriptCore (present on every macOS)
    only runs, so its output is matched for SyntaxError rather than trusting
    the exit status -- running `app.js` outside a browser always fails later on
    `document`, and that failure is not what is being tested.
    """
    node = shutil.which("node")
    if node:
        done = subprocess.run(
            [node, "--check", str(path)], capture_output=True, text=True, timeout=60
        )
        return ("node --check", done.stdout + done.stderr)

    jsc = Path("/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc")
    if jsc.exists():
        done = subprocess.run([str(jsc), str(path)], capture_output=True, text=True, timeout=60)
        return ("jsc", done.stdout + done.stderr)

    return None


def test_app_js_parses() -> None:
    """Parse `app.js` for real, when there is an engine available to do it."""
    result = js_syntax_check(APP_JS)
    if result is None:
        pytest.skip("no JavaScript engine found (looked for node, then JavaScriptCore)")
    engine, output = result
    assert "SyntaxError" not in output, f"{engine} rejected app.js:\n{output}"

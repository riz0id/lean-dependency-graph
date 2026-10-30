"""Demo pipeline: a browser-rendered view of a whole-library dependency graph.

Not part of the installed package -- ``[tool.hatch.build.targets.wheel]``
ships ``src/lean_dep_graph`` only.  This is a package purely so the reducer is
importable as ``demo.build`` from the test suite.
"""

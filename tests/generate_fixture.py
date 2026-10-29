"""Regenerate tests/fixtures/sample.ndjson.

Run deliberately -- `uv run python -m tests.generate_fixture` -- so that
fixture changes show up as a reviewable diff rather than drifting silently
underneath the stability test.
"""

from __future__ import annotations

from pathlib import Path

from .conftest import ExportBuilder

FIXTURE = Path(__file__).parent / "fixtures" / "sample.ndjson"


def build() -> ExportBuilder:
    """A small world exercising every record kind the format defines."""
    b = ExportBuilder()
    prop = b.sort(0)

    # Primitive types referenced by literals.
    b.axiom("Nat", prop)
    b.axiom("String", prop)

    # Quotient package: four declarations, one kernel unit.
    for name, kind in (
        ("Quot", "type"),
        ("Quot.mk", "ctor"),
        ("Quot.lift", "lift"),
        ("Quot.ind", "ind"),
    ):
        b.quot(name, prop, kind)

    # A structure, to give `proj` something to name.
    b.inductive(
        types=[("Prod", prop)],
        ctors=[("Prod.mk", b.forall("a", prop, prop), "Prod")],
        recs=[("Prod.rec", prop, [b.const("Prod.mk")])],
    )

    # A mutual inductive block.
    b.inductive(
        types=[("Tree", prop), ("Forest", prop)],
        ctors=[
            ("Tree.node", b.const("Forest"), "Tree"),
            ("Forest.nil", prop, "Forest"),
            ("Forest.cons", b.const("Tree"), "Forest"),
        ],
        recs=[
            ("Tree.rec", prop, [b.const("Forest.nil")]),
            ("Forest.rec", prop, [b.const("Tree.node")]),
        ],
    )

    # Axiom referencing an axiom: the #8840 case.
    b.axiom("Choice", prop)
    b.axiom("DependsOnChoice", b.const("Choice"))

    # A mutual definition block.
    b.define("even", b.const("Nat"), b.const("odd"), all_=["even", "odd"])
    b.define("odd", b.const("Nat"), b.const("even"), all_=["even", "odd"])

    # Literals, projection, let, mdata, lambda.
    b.define("nat_lit", b.const("Nat"), b.nat(42))
    b.define("str_lit", b.const("String"), b.string("hello"))
    b.define("projects", b.const("Nat"), b.proj("Prod", 0, b.const("Prod.mk")))
    b.define(
        "with_let",
        b.const("Nat"),
        b.let("tmp", b.const("Nat"), b.nat(1), b.bvar(0)),
    )
    b.define("wrapped", b.const("Nat"), b.mdata(b.const("nat_lit"), {"note": "x"}))
    b.define(
        "lambda_shadowing_Nat",
        b.const("Nat"),
        b.lam("Nat", b.const("Nat"), b.bvar(0)),
    )

    # An internal declaration used by a theorem.
    b.define("_proof_1", b.const("DependsOnChoice"), b.const("DependsOnChoice"))
    b.theorem("uses_internal", prop, b.const("_proof_1"))

    # An opaque and a deeper chain.
    b.opaque("sealed", b.const("Nat"), b.const("nat_lit"))
    b.define("top", b.const("Nat"), b.app(b.const("sealed"), b.const("projects")))

    return b


def main() -> None:
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(build().text(), encoding="utf-8")
    print(f"wrote {FIXTURE}")


if __name__ == "__main__":
    main()

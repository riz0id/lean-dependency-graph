/-
Declarations chosen to exercise each extraction path the library has to get
right. Kept deliberately simple: the point is coverage of term shapes, not
interesting mathematics.
-/
namespace DepFixture

/-- An axiom, so `axioms` has something to find. -/
axiom mystery : Nat

/-- An axiom whose *type* mentions another axiom: the lean4 #8840 shape. -/
axiom mysteryUser : mystery = mystery

/-- `Expr.lit` (natVal), which needs `Nat` with no `const` node present. -/
def fromNatLiteral : Nat := 42

/-- `Expr.lit` (strVal), which needs `String` the same way. -/
def fromStrLiteral : String := "hello"

/-- A structure: one command, many constants (type, ctor, projections, recursor). -/
structure Point where
  x : Nat
  y : Nat

/-- `Expr.proj`, whose structure name `Expr.foldConsts` never offers. -/
def projectsOut (p : Point) : Nat := p.x

/-- An opaque definition. -/
opaque sealed : Nat := 7

/-- A `let`, so `letE` gets traversed through all three of its children. -/
def withLet : Nat :=
  let tmp := fromNatLiteral
  tmp + 1

/-- A lambda, for `lam` traversal. -/
def applyTwice (f : Nat → Nat) (n : Nat) : Nat := f (f n)

/-- A tactic proof, which generates auxiliary `_proof_*` constants. -/
theorem sealed_eq : sealed = 7 := by
  simp [sealed]

/-- A proof depending on the axiom, so closures reach it. -/
theorem usesAxiom : mystery = mystery := mysteryUser

end DepFixture

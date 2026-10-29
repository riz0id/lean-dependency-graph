/-
Recursion and mutual blocks: these are the cases where the kernel insists on
accepting a group of declarations as a unit, which the graph models as one node.
-/
import DepFixture.Basic

namespace DepFixture

/-- Structural recursion, which generates `match_*` and equation lemmas. -/
def countDown : Nat → Nat
  | 0 => 0
  | n + 1 => countDown n + 1

/-- A mutual inductive block: one kernel unit spanning two types. -/
mutual
  inductive Tree where
    | node : Forest → Tree

  inductive Forest where
    | nil : Forest
    | cons : Tree → Forest → Forest
end

/-- A mutual definition block, grouped via the `all` field on each record. -/
mutual
  def isEven : Nat → Bool
    | 0 => true
    | n + 1 => isOdd n

  def isOdd : Nat → Bool
    | 0 => false
    | n + 1 => isEven n
end

/-- Reaches across modules, so closures have to cross module boundaries. -/
def usesBasic : Nat := withLet + fromNatLiteral

end DepFixture

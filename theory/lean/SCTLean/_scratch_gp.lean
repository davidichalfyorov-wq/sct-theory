import Mathlib
open Lean Elab Tactic Parser.Tactic
elab "generalize_proofs" loc:(location)? : tactic => pure ()
example (h : True) : True := by
  generalize_proofs at *
  exact h

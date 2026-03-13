import Lake
open Lake DSL

package SCTLean where
  leanOptions := #[
    ⟨`autoImplicit, false⟩
  ]

-- PhysLean (ex-HepLean): Lorentz group, SM, tensor notation, FLRW
-- Transitively pulls Mathlib4
require PhysLean from git
  "https://github.com/HEPLean/PhysLean" @ "master"

@[default_target]
lean_lib SCTLean where
  roots := #[`SCTLean]

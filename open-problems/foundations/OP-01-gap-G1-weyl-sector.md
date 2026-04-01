---
id: OP-01
title: "Gap G1: Explicit Weyl-sector correction on curved backgrounds"
domain: [theory]
difficulty: very-hard
status: open
deep-research-tier: B
blocks: [OP-21]
blocked-by: []
roadmap-tasks: [NT-4b, MR-9]
papers: ["2301.13525"]
date-opened: 2026-03-31
date-updated: 2026-03-31
---

# OP-01: Gap G1 -- Explicit Weyl-sector correction on curved backgrounds

## 1. Statement

Compute the alpha-insertion correction Θ^(C)_{μν} from the Weyl-squared
sector of the SCT effective action on a background with C_{μνρσ} ≠ 0
(e.g., Schwarzschild). The formal expression is known from NT-4b, but
its explicit evaluation on non-trivially curved backgrounds has not
been performed.

## 2. Context

The nonlinear SCT field equations contain two variational corrections
beyond the standard Einstein tensor:

  G_μν + Θ^(R)_μν + Θ^(C)_μν = 0.

The Ricci-sector correction Θ^(R) has been computed explicitly and
reduces correctly on FLRW backgrounds (where C = 0). On FLRW, Θ^(C)
drops out identically because C_μνρσ = 0. This means all cosmological
results (de Sitter stability, modified Friedmann equations, inflation)
are unaffected by Gap G1. However, any computation on a background
with nonzero Weyl curvature (Schwarzschild, Kerr, gravitational waves)
requires the explicit form of Θ^(C).

## 3. Known Results

- Θ^(C)_μν has a formal expression involving the functional derivative
  δ/δg^μν of Tr(f(D²/Λ²)) restricted to the C² sector.
- On conformally flat backgrounds (FLRW, de Sitter): Θ^(C) = 0 exactly.
- The linearised version (around flat space) is fully computed and gives
  the modified graviton propagator with Π_TT(z) = 1 + (13/60)z F̂₁(z).
- The dual derivation (NT-4b) confirms the formal structure using both
  the Barvinsky-Vilkovisky alpha-insertion technique and direct
  functional differentiation.

## 4. Failed Approaches

No explicit attempts have been made on non-trivially curved backgrounds.
The difficulty is that Θ^(C) involves the entire nonlocal form factor
F₁(□/Λ²) acting on the Weyl tensor, and on a curved background □ does
not commute with the curvature. A perturbative expansion in curvature
would require computing commutators [□, C_μνρσ] to all relevant orders.

## 5. Success Criteria

- Explicit expression for Θ^(C)_μν on static spherically symmetric
  backgrounds (Schwarzschild-like ansatz ds² = -A(r)dt² + B(r)dr² + r²dΩ²).
- Numerical evaluation of Θ^(C) at r = 2M, 3M, 10M for Schwarzschild.
- Verification that Θ^(C) → 0 as C → 0 (conformally flat limit).
- Consistency with the linearised result at large r.

## 6. Suggested Directions

1. Perturbative expansion of F₁(□) on Schwarzschild in powers of M/r,
   keeping enough terms for the correction to be numerically significant.
2. Variational approach using the Barvinsky-Vilkovisky covariant
   perturbation theory, which systematically handles [□, R] commutators.
3. Numerical approach: discretize the operator □ on a radial grid and
   compute F₁(□) C_μνρσ directly.
4. WKB/eikonal approximation for □ on Schwarzschild (valid at high
   angular momentum l).

## 7. References

1. Barvinsky, Vilkovisky, "The generalized Schwinger-DeWitt technique
   in gauge theories and quantum gravity," Phys. Rep. 119 (1985) 1.
2. de Brito, Eichhorn, Pfeiffer, arXiv:2301.13525 -- higher-order
   causal set actions (BD² structure).
3. Alfyorov, "Nonlinear field equations and FLRW cosmology from the
   spectral action," DOI:10.5281/zenodo.19098027 -- formal Θ^(C) expression.

## 8. Connections

- **Blocks OP-21** (BH singularity resolution): cannot solve the
  nonlinear field equations on Schwarzschild without Θ^(C).
- Related to **OP-22** (BH second law): Wall stability condition
  requires knowledge of the full field equations near the horizon.
- Independent of **OP-17** (scalaron mass) and all cosmological
  problems (these use C = 0 backgrounds).

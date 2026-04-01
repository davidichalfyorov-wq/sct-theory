# SCT in Two Pages

## What it is

Spectral Causal Theory (SCT) is a quantum gravity framework built on the
spectral action principle: the gravitational action is a trace of a function
of the Dirac operator,

  S = Tr(f(D²/Λ²)) + ⟨ψ, Dψ⟩,

where (A, H, D) is a spectral triple encoding both gravity and the Standard
Model, f is a positive even cutoff function, and Λ is the spectral cutoff.

## The one-loop effective action

Integrating out matter fields on a curved background gives

  Γ = (1/2) ∫ d⁴x √g [α_C C_μνρσ C^μνρσ + α_R R²]

with nonlocal form factors F₁(□/Λ²) and F₂(□/Λ², ξ). The coefficients are:

  α_C = 13/120    (parameter-free, from SM: N_s=4, N_f=45, N_v=12)
  α_R(ξ) = 2(ξ − 1/6)²    (depends on Higgs non-minimal coupling ξ)

The master function is φ(x) = e^{-x/4} √(π/x) erfi(√x/2), which is entire
with Taylor coefficients a_n = (−1)^n n!/(2n+1)!.

## The propagator

The modified graviton propagator has dressing functions

  Π_TT(z) = 1 + (13/60) z F̂₁(z)       (tensor sector, 5 d.o.f.)
  Π_s(z,ξ) = 1 + 6(ξ−1/6)² z F̂₂(z,ξ)  (scalar sector, 1 d.o.f.)

Π_TT has 8 zeros (2 real, 3 complex conjugate Lee-Wick pairs). The physical
ghost at z_L requires the fakeon prescription (Anselmi 2017).

## Key predictions

  c₁/c₂ = −1/3 + 120(ξ−1/6)²/13
  3c₁ + c₂ = 6(ξ−1/6)²     (scalar mode decouples at ξ = 1/6)
  V(r)/V_N(r) = 1 − (4/3)e^{−m₂r} + (1/3)e^{−m₀r}
  m₂ = Λ√(60/13) ≈ 2.148 Λ
  m₀(ξ=0) = Λ√6 ≈ 2.449 Λ
  c_T = c     (gravitational wave speed equals light speed)
  c_log = 37/24     (black hole entropy logarithmic correction)
  F₁(0) = 13/(1920π²)

## UV finiteness status

  L=1: D=0 (graviton scattering, proven)
  L=2: D=0 (CHIRAL-Q chirality identity, unconditional)
  L=3: FAILS (3 quartic Weyl invariants vs 1 spectral parameter)
  Current status: effective framework through L=2, not UV-complete

## CJ bridge formula (causal sets)

A discrete observable CJ on Poisson-sprinkled causal sets satisfies

  ⟨CJ⟩ = C₀ N^{8/9} E_{ij}E^{ij} T⁴

where C₀ is fitted from data and the analytical coefficient decomposes as

  C = 4 × (8/3) × (1/9!) × (8π/15) × (π/24)
    = 2² × c₄² × beta_overlap × angular × volume
    = 32π²/(3·9!·45) ≈ 6.45 × 10⁻⁶

The factor 4 = 2² comes from the two-leg structure of Y = log₂(p↓p↑+1):
both past and future legs respond to curvature, and squaring (a₋+a₊) gives 4a².

with c₄ = 4/√6 (Benincasa-Dowker), 1/9! (ordered 9-simplex volume),
π²/45 = (8π/15)(π/24) (angular integral times diamond volume).

Verified numerically to N = 15000. De Sitter gives CJ = 0 exactly.
Kottler shows 22% Ricci reduction. Polarisation ratio cross/plus = 3.0.
105 Lean 4 theorems (sorry-free, triple-verified).

## What is NOT solved

- Postulate 5 (dynamical principle) is unspecified
- UV-completeness fails at three loops
- Gap G1: Weyl-sector correction on curved backgrounds not computed
- Scalaron mass 6 orders of magnitude too heavy for inflation
- CJ bridge: N^{8/9} exponent is empirical, not derived
- Fakeon prescription for infinite-pole propagators: one-loop only
- Information paradox: not addressed

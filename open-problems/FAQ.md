# Frequently Asked Questions (and Common Misconceptions)

## Theory

**Q: Is SCT the same as noncommutative geometry (NCG)?**

A: SCT uses the NCG framework (spectral triples, spectral action) but
goes beyond it by computing the full nonlocal one-loop effective action
with entire-function form factors, rather than truncating at the
heat-kernel level. The key difference: NCG traditionally keeps only the
first few Seeley-DeWitt coefficients (polynomial in curvature), while
SCT retains the full nonlocal structure.

**Q: Is SCT UV-complete?**

A: No. The one-loop (L=1) and two-loop (L=2) divergences cancel
unconditionally: D=0 at both levels. At three loops, there are 3
independent quartic Weyl invariants but only 1 spectral parameter,
giving 3:1 overdetermination (reduced to 2:1 by Cayley-Hamilton).
Five systematic investigations (FUND programme) failed to find an
escape route. Current status: SCT is an effective framework valid
through L=2, with practical three-loop suppression of 3.2 x 10^{-10}.

**Q: Does SCT predict spectral dimension d_S = 2 in the UV?**

A: It depends on the definition. SCT is the first quantum gravity
approach where d_S is definition-dependent:
- CMN propagator definition: d_S = 2
- Heat kernel definition: d_S = 4
- Fakeon/ASZ definition: d_S = 0 then 4
- Mittag-Leffler (physical): d_S flows from 2 to 4

The physical Mittag-Leffler definition gives d_S ~ 2 at the ghost
scale and d_S = 4 in the IR. There is a region where the return
probability P(sigma) < 0 due to ghost dominance.

**Q: Can SCT resolve the H_0 tension?**

A: No. The SCT correction to the Friedmann equation is
delta_H^2 / H^2 ~ 1.3 x 10^{-64} at the PPN-1 bound. This is 64
orders of magnitude below what would be needed. The physical reason:
UV nonlocality (entire functions, scale ~ Lambda) cannot generate IR
nonlocality (scale ~ H_0). This is a proven impossibility within
the SCT framework (Maggiore's argument).

## CJ Bridge Formula

**Q: Is CJ proportional to the Kretschner invariant C^2?**

A: No. On a pp-wave spacetime (Petrov type N), the Kretschner scalar
C_{abcd} C^{abcd} = 8(E^2 - B^2) = 0 because E^2 = B^2. But CJ is
nonzero at 8-sigma significance. CJ is proportional to E_{ij}E^{ij}
(electric Weyl squared, observer-dependent), not to the invariant C^2.
This is a frame-dependent quantity, like the Bel-Robinson superenergy.

**Q: Does the CJ bridge formula work in d != 4?**

A: No. The d=2 test gives a measured exponent alpha = 1.53 versus the
predicted 2d/(2d+1) = 4/5 = 0.80. The formula is genuinely
4-dimensional. The mechanism is Wang's identity H^2 = 6(d-4) = 0,
which ensures that only the electric Weyl contributes in d = 4.

**Q: Is the CJ bridge formula fully derived analytically?**

A: Partially. The coefficient 8/(3 x 9!) x pi^2/45 is analytically
derived and formally verified in Lean 4. The functional dependences
on E^2 and T^4 are established. The N^{8/9} exponent is empirical
(fitted alpha = 0.955 +/- 0.027, consistent with 8/9 at the level of
residuals). Two explicit conditions (stratification measure closure
and factorisation at leading Weyl order) remain unproven.

**Q: What is the relationship between CJ and Seeley-DeWitt coefficients?**

A: There is no direct relationship. This has been established by
testing 30+ spectral routes. Three structural obstructions close
entire families of approaches:
1. det(I - pA) = 1 (nilpotent A on DAG): all Fredholm/zeta routes dead.
2. No directed cycles on DAG: all Ihara/Hashimoto zeta routes dead.
3. det(B) = alpha^N for lower-triangular retarded matrix: geometry-blind.

The reason is structural: CJ measures the frame-dependent E^2 + B^2,
while Seeley-DeWitt coefficients involve the frame-independent C^2.

## Formal Verification

**Q: What has been formally verified in Lean 4?**

A: 105 sorry-free theorems covering the CJ bridge formula coefficient
identities, plus 41 canonical SCT identities (alpha_C = 13/120,
c_1/c_2 ratio, etc.) in 46 proof files. The key research-level theorem
is the general-d beta overlap: for all d in N,
(d!)^2 C(2d,d)(2d+1) = (2d+1)!. The rigorous chain from the beta
integral to factorials uses Mathlib's Gamma_mul_Gamma_eq_betaIntegral.

**Q: What has NOT been formally verified?**

A: Continuum-limit arguments (Conditions A and B of the bridge formula),
the angular integral on S^2 (only the rational skeleton pi^2/45 is
verified, not the full sphere integral), the N-scaling exponent, and
all results involving transcendental functions beyond pi
(e.g., the master function phi(x) involves erfi).

## Methodology

**Q: How are results verified in SCT?**

A: Through an 8-layer pipeline:
1. Analytic (dimensions, limits, symmetries)
2. Numerical (mpmath >= 100 digits, 7+ test points)
3. Property fuzzing (hypothesis library, 1000+ cases)
4. Literature comparison (CZ, CPR, Avramidi, Vassilevich)
5. Dual derivation (independent method)
6. Triple CAS (SymPy x GiNaC x mpmath, >= 12-digit agreement)
7. Lean 4 formal verification (local PhysLean + Mathlib)
8. Multi-backend Lean (Local + Aristotle automated prover)
Total: 4000+ quantitative checks passing.

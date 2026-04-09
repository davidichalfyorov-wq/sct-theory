---
id: OP-07
title: "Fakeon prescription for infinite-pole propagators"
domain: [unitarity]
difficulty: hard
status: partial
deep-research-tier: A
blocks: [OP-22]
blocked-by: []
roadmap-tasks: [MR-2]
papers: ["1704.07728", "2308.09006", "1801.00915", "1703.04584", "1802.00399", "2501.04097", "hep-th/9702146"]
date-opened: 2026-03-31
date-updated: 2026-04-07
progress: "PATH CLEAR. Hadamard/product fakeon CONVERGES (rho<1). Pole-by-pole DOES NOT converge (sin(sqrt(z))/sqrt(z) counterexample). Correct definition: Euclidean-first. Three lemmas remain open."
---

# OP-07: Fakeon prescription for infinite-pole propagators

## 1. Statement

Prove that the fakeon (purely virtual particle) prescription for the
SCT graviton propagator 1/(k^2 Pi_TT(z)), which has infinitely many
poles from the entire function Pi_TT(z), yields a unitary S-matrix
at all loop orders. Alternatively, identify a concrete obstruction
that prevents such an extension.

## 2. Context

The fakeon prescription was introduced by Anselmi (2017) as an
alternative to the Feynman i*epsilon prescription at ghost poles.
Instead of propagating ghost particles as physical asymptotic states,
the fakeon prescription assigns zero imaginary part to the propagator
at ghost poles, effectively removing ghost pair production.

For polynomial propagators (finitely many poles), the prescription is
rigorously proven to preserve unitarity: the cutting equations for
the S-matrix are modified so that cuts through ghost lines give zero
contribution, and the optical theorem Im[T] >= 0 holds. This covers
Stelle gravity (fourth-derivative, 3 propagator poles) and its
higher-derivative extensions with any finite number of poles.

The SCT propagator is qualitatively different: Pi_TT(z) is entire of
order 1 and has infinitely many zeros. The Mittag-Leffler expansion
of 1/(z Pi_TT) is an infinite partial-fraction series. The fakeon
prescription must be applied to each pole individually, and the limit
as the number of poles N -> infinity must be shown to commute with
the prescription and preserve unitarity.

## 3. Known Results

- **CL commutativity (CERTIFIED):** The limit N -> infinity commutes
  with the fakeon principal-value prescription at the level of the
  one-loop amplitude. Proven via Weierstrass M-test with total bound
  Sigma M_n = 5.002e-4 (0.32% of total amplitude). Two-pole dominance
  holds: 99.68% of amplitude from z_L = -1.2807 and z_0 = 2.4148.

- **GZ constancy (CERTIFIED):** The entire part g_A(z) = -13/60 is
  constant (no polynomial growth), proven via Hadamard genus argument.
  Sum rule Sigma R_n / z_n = 13/60 verified to 99.95%.

- **One-loop optical theorem (CERTIFIED CONDITIONAL):** Im[T(s)] >= 0
  at all tested energies. Spectral positivity Im[G_dressed] > 0
  proven algebraically. Fakeon ghost cut removal verified.

- **Anselmi polynomial proof:** For polynomial propagators of any
  degree, the fakeon prescription gives unitary amplitudes (proven via
  modified cutting equations and the largest-time equation in
  Minkowski space). The key property used: finitely many poles allow
  explicit contour deformation and principal-value evaluation.

## 3b. Partial Resolution (2026-04-01)

**VERDICT: GAP IDENTIFIED. No obstruction found. One-loop likely extendable. All-orders open.**

### Precise gap location in Anselmi's proof

The finiteness assumption enters at three specific points:

**1801.00915 (Anselmi, "Fakeons and Lee-Wick models"):**
- After eq.(2.25): explicitly states "The number of thresholds (2.25)
  and regions A_tilde_i of each loop integral is **finite**." This is
  the core assumption. Verified via ar5iv.
- Section 7.3: The procedure "Applying this procedure to each LW
  threshold at a time, we reach every U_R cap A_i" is organized as
  iteration over a finite set of thresholds.
- Section 4: Multiple thresholds handled via eq.(4.8) under assumption
  that their positions are distinct. For "definiteness" takes unique
  LW threshold P, constructs J+ and J-, averages to (J+ + J-)/2.

**1703.04584 (Anselmi-Piva, "A new formulation of Lee-Wick QFT"):**
- Eq.(3.11): J_{LW}^{>} defined as half-sum of J_{LW}^{0+} and
  J_{LW}^{0-} — average continuation over ONE threshold.
- Section 6: Number of disjoint regions grows with loop order, but
  is finite for each diagram (finite internal lines and vertices).

**1704.07728 (Anselmi, "On the QFT of gravitational interactions"):**
- No single all-orders theorem. Finite spectrum built into examples:
  eqs.(5.2)-(5.3) (scalar with one fake DOF), eq.(5.6) (higher-
  derivative scalar), eqs.(6.2)-(6.3) (gravity propagators).

### Gap classification

(a) **Finite contour/domain deformations — MAIN GAP.** The algorithm
    iterates threshold-by-threshold. For infinite poles, this becomes
    an infinite iteration requiring uniform convergence control.

(b) **Modified cutting equations.** Formula (7.10) itself works for
    any fixed graph, but its transfer to other regions and the
    epsilon -> 0 limit depend on finite-threshold geometry.

(c) **ACE/LTE combinatorics.** Not the issue — finite per graph
    regardless of pole count.

(d) **Hidden assumptions:** distinctness and local finiteness of
    thresholds, plus uniform vanishing of C_N(p,epsilon) corrections.

### Compatibility with CL bound

Our certified CL result (VR-081, VR-082: Weierstrass M-test with
sum M_n = 5.002e-4) addresses gap (a) at one-loop level: it controls
the sum over fakeonized poles and shows the prescription commutes
with N -> infinity for the one-loop amplitude.

For all-orders extension, additionally needed:
- Control over cut INTEGRANDS (not just total amplitude)
- Local finiteness of multi-threshold set at higher loops
- Uniform convergence of C_N(p,epsilon) over compact kinematic sets

### IDG parallel: why their theorems don't transfer

IDG unitarity proofs (Tomboulis 1997, Briscese-Modesto 2019) work
because the entire form factor has NO extra zeros/poles. Their key
step "only normal thresholds contribute" fails for SCT where Pi_TT
zeros create a countable fake-threshold tower. Transferable: Euclidean-
first methodology, contour techniques, Ward/BRST cancellations.

### Counterexample search: negative

No published counterexample found for fakeonized infinite-pole tower.
Kubo-Kugo (2308.09006) show unitarity violation for complex ghosts
in Lee-Wick theories, but this is for the Lee-Wick prescription, NOT
the fakeon prescription — important distinction.

### Bibliographic corrections

- 2308.09006 is Kubo-Kugo "Unitarity Violation in Field Theories of
  Lee-Wick's Complex Ghost", not "Unitarity and higher-order scattering"
- 2501.04097 is Buoninfante "Remarks on ghost resonances", not
  Anselmi-Calcagni on classicization
- Buoninfante "Classical properties" is 1802.00399, not 1811.10619

## 3c. Fakeon Convergence Analysis (2026-04-07)

**VERDICT: PATH CLEAR. Hadamard route works. Pole-by-pole doesn't.**

### Hadamard/product fakeon: CONVERGES

When the entire function Π_TT(z) has order ρ < 1 (genus 0 or 1 in
the Hadamard factorization), the product fakeon prescription converges.
The Weierstrass product representation with convergence exponents
determined by the order of growth gives a well-defined principal-value
limit.

### Pole-by-pole fakeon: DOES NOT CONVERGE

Explicit counterexample: sin(√z)/√z (a standard entire function of
order ρ = 1/2). The Mittag-Leffler partial fraction expansion of
1/[z · sin(√z)/√z] has residues A_n = (-1)^n/(n²π²), and the series
Σ A_n/(s - z_n) converges absolutely. But the pole-by-pole fakeon
(applying PV to each pole independently) gives different results
depending on the order of summation. The prescription is NOT
well-defined without a specific summation order.

### Correct definition

The fakeon for infinite-pole propagators must be defined as:
  R_Fk = AV_T(s-lim R_{E,N})
where AV_T is the threshold average (Anselmi's prescription applied
to each threshold), s-lim is the strong limit as N → ∞ poles are
included, and R_{E,N} is the Euclidean propagator truncated to N poles.

This is the **Euclidean-first** definition: regulate in Euclidean
space, then analytically continue.

### CL connection

The CL commutativity result g_A = −13/60 = −Π'(0) = Σ 1/z_n is the
first Newton sum. It controls Σ A_n/z_n, which is CLOSER to the
needed condition Σ|A_n|/|z_n| < ∞ than previously recognized.
However, CL controls conditional convergence; absolute convergence
requires separate bounds on |Π'(z_n)|.

### Three open lemmas

1. **Lemma (i):** Lower bounds on |Π'(z_n)| for the SCT-specific Π_TT
   (zero separation from Hadamard factorization)
2. **Lemma (ii):** Local finiteness of fakeon thresholds at higher loops
   (from z_n growth rate)
3. **Lemma (iii):** Uniform Euclidean majorants for truncated loop
   diagrams (Weierstrass M-test generalization)

### RESOLVED: ρ(Π_TT) = 1, genus = 1 (2026-04-07)

**Computation resolves the discrepancy.** The Taylor coefficients
a_n = (-1)^n n!/(2n+1)! give:
  -ln|a_n| ~ n(ln n + 2 ln 2 - 1)
  ρ = lim n ln(n)/[-ln|a_n|] = lim ln(n)/(ln(n) + 2 ln 2 - 1) = 1

Numerical verification: |φ(-R)| ~ exp(R/4) (type σ = 1/4, confirmed
to σ = 0.2496 at R = 10000). Since zeros z_n ~ c·n grow linearly,
Σ 1/|z_n| diverges → genus p = 1 (Hadamard factorization requires
convergence-producing factors exp(z/z_n)).

**analytical ρ = 1/2 in OP-07: INCORRECT.** analytical ρ = 1 in OP-06/09: CORRECT.
**Fakeon convergence still holds** because the Hadamard product converges for ρ ≤ 1.

## 4. Failed Approaches

1. **Direct extension of Anselmi's proof.** The polynomial proof
   uses explicit contour deformation around a finite set of poles.
   For infinitely many poles accumulating at infinity, the contour
   deformation becomes an infinite sequence of deformations, and
   the convergence of the resulting series is not controlled by the
   existing proof strategy. The approach does not fail definitively
   but lacks the estimates needed for the infinite-pole case.

2. **Functional-integral derivation.** Attempting to define the
   fakeon prescription via a modified path integral measure
   (restricting the domain of integration to exclude ghost modes)
   encounters the problem that the ghost modes are dense in the
   spectrum of D^2 and cannot be cleanly separated from physical
   modes in a nonperturbative functional integral.

3. **Borel summation of the ML series.** The Mittag-Leffler series
   for 1/(z Pi_TT) converges absolutely (established by GZ), so
   Borel summation is not needed for convergence. However, the
   question is about unitarity of the S-matrix, not convergence of
   the propagator series -- the cutting equations involve products
   of propagators, and the convergence of product series is not
   automatic.

## 5. Success Criteria

- A rigorous proof that the S-matrix defined by the fakeon
  prescription on the full SCT propagator satisfies the optical
  theorem Im[T(s)] >= 0 at all orders in the loop expansion.
- Alternatively: a proof that the N-pole truncated S-matrix
  converges to the infinite-pole S-matrix uniformly in the
  external momenta, combined with the polynomial unitarity result.
- Or: identification of a specific loop order and kinematic
  configuration where ghost pair production is nonzero despite the
  fakeon prescription.

## 6. Suggested Directions

1. **Uniform convergence of cutting equations.** The cutting equations
   for the N-pole truncation are known to give Im[T_N] >= 0. Show
   that Im[T_N] -> Im[T_infinity] uniformly in external momenta,
   using the Weierstrass M-test bounds from CL as input.

2. **Spectral representation approach.** Since rho_TT is a pure sum
   of delta functions (no branch cuts from entire Pi_TT), attempt
   to construct a spectral representation for the full S-matrix and
   verify positivity of the spectral measure after fakeon exclusion.

3. **Axiomatic S-matrix route.** Use the Bogoliubov-Medvedev-Polivanov
   axioms (unitarity, causality, Lorentz invariance) and show that
   the fakeon-modified propagator satisfies all axioms. The acausal
   component has decay length 0.884/Lambda (from MR-3), which may
   be sufficient for the micro-local spectrum condition.

4. **Nonperturbative lattice test.** Discretize the theory on a
   4-torus, compute the partition function with fakeon-modified
   propagator, and verify reflection positivity (the Euclidean
   analogue of unitarity) numerically for small lattice sizes.

## 7. References

1. Anselmi, D. "On the quantum field theory of the gravitational
   interactions," JHEP 06 (2017) 086, arXiv:1704.07728.
2. Anselmi, D. "Fakeons and Lee-Wick models," JHEP 02 (2018) 141,
   arXiv:1801.00915.
3. Kubo, J. and Kugo, T. "Unitarity and higher-order gravitational
   scattering," arXiv:2308.09006.
4. Anselmi, D. and Piva, M. "A new formulation of Lee-Wick quantum
   field theory," JHEP 06 (2017) 066, arXiv:1703.04584.
5. Alfyorov, D. "Nonlocal one-loop form factors from the spectral
   action," DOI:10.5281/zenodo.19098042.

## 8. Connections

- **Blocks OP-22** (BH entropy second law): The fakeon prescription
  enters the definition of the graviton propagator near the horizon,
  and BH thermodynamics requires knowing whether ghost modes
  contribute to the entropy.
- **Blocked by nothing:** This is a root-level problem. Its resolution
  would upgrade MR-2 from CONDITIONAL to COMPLETE.
- **Related to OP-08** (all-orders KK resolution): OP-08 requires
  OP-07 as input because the operator-formalism proof of ghost
  decoupling depends on the fakeon prescription being well-defined.
- **Related to OP-09** (BV axioms): If fakeon unitarity fails, the
  CHIRAL-Q UV-finiteness result becomes the primary unitarity route,
  making OP-09 critical.

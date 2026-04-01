---
id: OP-04
title: "Parameter counting and cutoff function analysis"
domain: [theory]
difficulty: medium
status: resolved
deep-research-tier: A
blocks: []
blocked-by: []
roadmap-tasks: [META-1]
papers: ["hep-th/9606001", "0805.2909", "1108.3749", "0706.3688", "1809.02944"]
date-opened: 2026-03-31
date-updated: 2026-04-01
date-resolved: 2026-04-01
resolved-by: literature-analysis + independent-numerical-verification
---

# OP-04: Parameter counting and cutoff function analysis

## 1. Statement

The spectral action S = Tr(f(D^2/Lambda^2)) depends on the cutoff
function f, which is positive, even, and rapidly decreasing, but
otherwise unspecified by any of the five postulates. Since f has
infinitely many Taylor/moment parameters, the theory's predictive power
depends critically on which observables depend on f and which do not.
Classify all SCT predictions into two categories:

- **Moment-dependent (universal):** depending only on f_0, f_2, f_4
  (the first three moments of f), hence robust under arbitrary changes
  to the shape of f.
- **f-shape-dependent (fragile):** depending on the full functional
  form of f(u), hence not uniquely predicted without an additional
  principle fixing f.

Determine whether physical requirements (unitarity, causality,
finiteness, entireness of form factors) constrain f beyond its
moment parameters.

## 2. Context

In the heat-kernel expansion, the spectral action produces a series

  S = sum_{n=0}^{infinity} f_{2n} Lambda^{4-2n} a_{2n}(D^2)

where f_{2n} = integral_0^infinity f(u) u^{n-1} du are the moments of f
and a_{2n} are the Seeley-DeWitt coefficients. The classical
Lagrangian (Einstein-Hilbert + cosmological + SM) depends only on
f_0, f_2, f_4. The one-loop form factors depend on the full shape
of f through the master function phi(x) = e^{-x/4} sqrt(pi/x) erfi(sqrt(x)/2),
which arises from the choice f(u) = e^{-u}.

The working choice f(u) = e^{-u} is computationally convenient and
produces entire form factors (NT-2), which guarantees ghost-freedom.
But it is not uniquely determined: any positive, rapidly decreasing,
even function f gives a valid spectral action. Different choices of f
lead to different form factors, different propagator modifications,
and different predictions for the modified Newtonian potential.

## 3. Known Results

- **Moment-dependent predictions (UNIVERSAL):**
  - alpha_C = 13/120. This is the coefficient of the Weyl-squared
    term at one loop and depends only on f_4 (or more precisely, it is
    the ratio of SM traces to normalization, independent of f entirely).
  - c_1/c_2 = -1/3 + 120(xi - 1/6)^2/13. This ratio depends only on
    the Seeley-DeWitt coefficients a_4, not on f.
  - The PPN parameters beta_PPN = gamma_PPN = 1 to leading order.
    These follow from the structure of the linearised field equations
    and are f-independent.
  - The Donoghue coefficient c_1 + c_2 = 13/120 is moment-independent.

- **f-shape-dependent predictions (FRAGILE):**
  - The form factors h_C(x), h_R(x) depend on phi(x), which is
    determined by f. Different f -> different h_C, h_R.
  - The modified Newtonian potential V(r) depends on the effective
    masses m_2 and m_0, which involve the zeros of Fourier-transformed
    form factors. These are f-shape-dependent.
  - The UV asymptotic x alpha_C(x -> infinity) = -89/12 is specific
    to f(u) = e^{-u}.

- **Partially constrained by physics:**
  - NT-2 shows that f(u) = e^{-u} yields entire form factors F_1, F_2.
    Entireness requires f to decay faster than any power of u. This
    excludes polynomial and rational f, but leaves a large class
    (e.g., f(u) = e^{-u^alpha} for alpha >= 1).
  - Unitarity (MR-2) requires that the dressed propagator has no
    tachyonic ghost poles. This constrains the zeros of 1 + alpha z F(z),
    which depends on f.

## 3b. Resolution (2026-04-01)

**VERDICT: Classification complete. Robust core is NARROW (a_4-level only). All finite-momentum predictions are f-dependent.**

### Classification Table

| Prediction | Depends on | Robust? |
|-----------|-----------|---------|
| Gauge group, reps, Higgs mechanism | Spectral triple only | Yes |
| Gauge coupling unification g_3^2=g_2^2=(5/3)g_1^2 | f(0) cancels in ratio | Yes |
| alpha_C = 13/120 | a_4 coefficient only | Yes |
| alpha_R(xi) = 2(xi-1/6)^2 (local) | a_4 coefficient only | Yes |
| c_1/c_2 = -1/3 + 120(xi-1/6)^2/13 | Ratio of a_4 coefficients | Yes |
| PPN gamma = beta = 1 (leading) | Low-momentum a_2+a_4 truncation | Yes (leading only) |
| c_T = c on FLRW | C = 0 on conformally flat | Yes (on FLRW) |
| h_C(x), h_R(x) form factors | Full w_f(x) kernel | **No** |
| UV asymptotic x*alpha_C(x->inf) = -89/12 | c_f from eq.(26) of ILV | **No** |
| m_2, m_0 effective masses | Zeros of dressed propagator | **No** |
| V(r) modified potential | Fourier of full propagator | **No** |
| d_S spectral dimension | Return probability from full kernel | **No** |
| c_log BH entropy correction | Full fluctuation determinant | **No** (not established) |
| n_s, r inflation | Scalar potential / exact action | **Partially** (a_4 more robust; exact fragile) |
| QNM frequencies | Full linear operator spectrum | **No** |

### Key mathematical results

**Entireness requires integer alpha.** For f(u) = e^{-u^alpha}:
- phi_alpha(x) = int_0^1 exp(-(t(1-t)x)^alpha) dt
- f_{2n}(alpha) = (1/alpha) Gamma(n/alpha)
- Entire if and only if alpha is a POSITIVE INTEGER (1, 2, 3, ...)
- Rapidly decreasing is necessary but NOT sufficient (counterexample: e^{-u^{3/2}})

**Cutoff function family scan (verified):**

| f(u) | f_2 | phi'(0) | Entire? | m_2/Lambda (est.) | m_0/Lambda (est.) |
|------|-----|---------|---------|-------------------|-------------------|
| e^{-u} | 1.000 | -1/6 | Yes | 2.148 | 2.449 |
| e^{-u^{3/2}} | 0.903 | 0 | No | 2.041 | 2.327 |
| e^{-u^2} | 0.886 | 0 | Yes | 2.022 | 2.305 |
| e^{-u^3} | 0.893 | 0 | Yes | 2.030 | 2.314 |
| (1+u)^{-3} | 0.500 | -1/2 | No (branch at x=-4) | 1.519 | 1.732 |
| (1+u)^{-5} | 0.250 | -5/6 | No (branch at x=-4) | 1.074 | 1.225 |
| (1+u)^{-10} | 0.111 | -5/3 | No (branch at x=-4) | 0.716 | 0.816 |

**Power-law cutoffs:** phi_N(x) = 2F1(N,1;3/2;-x/4) with branch cut at x = -4 (from singularity of (1+t(1-t)x)^{-N} at t=1/2). Higher moments f_{2m} diverge for m >= N. Excluded by entireness requirement.

### Constraints on f from physics

- **Entireness** excludes: power-law, rational, non-integer exponential cutoffs. Leaves infinite family: e^{-u^n} for n=1,2,3,..., and positive combinations.
- **Unitarity** constrains pole catalogue of dressed propagator, but does not reduce to finite-parameter family.
- **Causality** acausal length scale l_a = O(1/Lambda) but coefficient depends on f.
- **Combined constraints** {entireness + unitarity + causality + UV softness}: restrict f strongly but do NOT uniquely determine it. e^{-u} is NOT the unique admissible cutoff.

### Literature on fixing f

Three approaches found:
1. **Zeta spectral action** (Kurkov-Lizzi-Vassilevich): eliminate f entirely by replacing cutoff action with zeta-function action.
2. **Entropy-derived function** (Chamseddine-Connes-van Suijlekom, arXiv:1809.02944, Theorem 3.4): universal function h from fermionic second-quantized entropy. Not the same as e^{-u}.
3. **Dilatonized action** (Chamseddine-Connes, hep-th/0512169): make Lambda dynamical via dilaton.

None of these uniquely selects f = e^{-u} within the standard cutoff framework.

### Implications for SCT

The robust core of SCT (alpha_C, c_1/c_2, PPN, c_T) is narrow but contains ALL experimentally accessible predictions. The f-dependent predictions (m_2, V(r), d_S, QNMs) are experimentally inaccessible at current precision, so the practical predictive power of the theory is not diminished.

The UV asymptotic -89/12 is specific to f = e^{-u} and should not be quoted as a universal SCT prediction.

## 4. Failed Approaches

1. **Deriving f from a variational principle.** Attempted to extremize
   the spectral action over the space of cutoff functions f, treating
   f itself as a dynamical variable. The problem is ill-posed: the
   action is linear in f (through its moments), so any extremization
   either drives f to zero or to a delta function, neither of which
   is physical.

2. **Maximum entropy selection.** Attempted to select f by maximizing
   the entropy of the spectrum of D^2/Lambda^2 subject to moment
   constraints. This produces f(u) = e^{-lambda u} (exponential
   distribution), which coincidentally matches the working choice.
   However, the "entropy" of a spectrum is not a physical quantity,
   and the result depends on the choice of entropy functional.

3. **Renormalization group flow of f.** Attempted to define a flow
   f_Lambda -> f_{Lambda'} under coarse-graining. The spectral action
   is not a standard QFT action, and the Wetterass equation does not
   apply directly. The flow mixes moments of different order, and no
   fixed-point structure was found.

## 5. Success Criteria

- A complete classification table: for each SCT prediction, state
  whether it is moment-dependent or f-shape-dependent, with proof.
- For f-shape-dependent predictions: quantify the variation across
  a representative family of cutoff functions (e.g., f(u) = e^{-u^alpha}
  for alpha in {1, 3/2, 2, 3}).
- Determine whether the conjunction of unitarity + causality +
  entireness + asymptotic freedom constrains f to a finite-parameter
  family or uniquely determines it.
- If f cannot be fixed: state clearly which predictions are robust
  and which carry an irreducible cutoff-function ambiguity.

## 6. Suggested Directions

1. **Systematic moment analysis.** For each physical prediction, write
   it as a functional of f and identify which moments appear. If only
   f_0, f_2, f_4 enter, the prediction is universal. If higher moments
   or the full Laplace transform appear, it is fragile.

2. **Cutoff function family scan.** Compute the form factors, effective
   masses, and modified Newtonian potential for f(u) = e^{-u^alpha}
   with alpha = 1, 3/2, 2, 3 and for f(u) = (1+u)^{-n} with n = 3, 5, 10.
   Tabulate the variation in physical predictions.

3. **Entireness constraint.** Characterize the space of positive, even,
   rapidly decreasing functions f for which the resulting form factors
   F_i(z) are entire. Is f(u) = e^{-u} the unique such function up to
   rescaling?

4. **Bootstrapping from spectral data.** If the eigenvalue distribution
   of D^2 is known non-perturbatively (e.g., from matrix models or
   lattice simulations), the cutoff function can in principle be
   reconstructed from the requirement that the spectral action
   reproduces known low-energy physics.

## 7. References

1. Chamseddine, Connes, "The spectral action principle," Comm. Math.
   Phys. 186 (1997) 731, hep-th/9606001.
2. Chamseddine, Connes, "Why the Standard Model," J. Geom. Phys. 58
   (2008) 38, arXiv:0706.3688.
3. van Suijlekom, "Noncommutative Geometry and Particle Physics,"
   Springer (2015), Ch. 11 (spectral action and cutoff dependence).
4. Codello, Percacci, Rahmede, "Investigating the ultraviolet
   properties of gravity with a Wilsonian renormalization group
   equation," Ann. Phys. 324 (2009) 414, arXiv:0805.2909.

## 8. Connections

- Related to **OP-06** (UV-completeness): if f can be fixed by
  physical requirements, it may resolve or sharpen the three-loop
  obstruction.
- Related to **OP-02** (Postulate 5): a dynamical principle might
  select f as part of the dynamics.
- Independent of **OP-01** (Gap G1): the Weyl-sector correction
  structure is independent of the choice of f (only its numerical
  value changes).
- The moment-independent predictions (alpha_C, c_1/c_2, PPN
  parameters) are the strongest candidates for experimental tests
  because they carry no cutoff-function ambiguity.

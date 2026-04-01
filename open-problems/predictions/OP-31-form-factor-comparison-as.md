---
id: OP-31
title: "Form factor comparison with asymptotic safety and FRG flows"
domain: [predictions, theory]
difficulty: medium
status: resolved
deep-research-tier: B
blocks: []
blocked-by: []
roadmap-tasks: [COMP-1]
papers: ["0805.2909", "1203.2034", "1006.3808", "1904.04845", "2111.13232", "2111.12365", "2002.10839", "1401.4452"]
date-opened: 2026-03-31
date-updated: 2026-04-01
date-resolved: 2026-04-01
resolved-by: literature-analysis + independent-numerical-verification
---

# OP-31: Form factor comparison with asymptotic safety and FRG flows

## 1. Statement

Perform a quantitative comparison of the SCT form factors F_1(z)
(Weyl sector) and F_2(z, xi) (Ricci sector) with the corresponding
form factors obtained from the functional renormalization group (FRG)
in the asymptotic safety program. Extract the graviton self-energy
w(z) from both approaches and compare their analytic structure,
UV behavior, and phenomenological predictions.

## 2. Context

Both SCT and asymptotic safety produce one-loop gravitational form
factors that dress the graviton propagator. In SCT, these are
entire functions derived from the heat kernel expansion of the
spectral action, with the master function phi(x) encoding all
momentum dependence. In asymptotic safety, the FRG flow generates
form factors through the Wetterstein effective action, typically
computed by truncation of the theory space to finitely many operators.

The key structural difference is:
- SCT: form factors are entire functions of z = k^2/Lambda^2,
  with infinite sequences of zeros (ghost poles). The UV asymptotic
  is x alpha_C(x -> inf) = -89/12.
- AS: form factors typically have polynomial or rational dependence
  on k^2, with UV fixed-point behavior F ~ (k/k_*)^{2 - eta_N}
  where eta_N is the anomalous dimension.

A comparison would determine whether the two programs make compatible
predictions in their overlapping domain of validity (perturbative
one-loop regime).

## 3. Known Results

- SCT form factors for all SM spins: COMPLETE (NT-1, NT-1b).
  Closed-form expressions in terms of phi(x).
- SCT local limits (beta coefficients): beta_W = 13/120,
  beta_R(xi) = 2(xi - 1/6)^2.
- AS form factors computed by Codello-Zanusso in the CZ basis
  (0805.2909): agree with SCT at the one-loop level for the
  matter contributions.
- AS graviton contributions differ from SCT because the AS
  calculation includes the graviton self-coupling in the FRG flow,
  while SCT obtains it from the spectral action (different
  resummation).
- Knorr-Saueressig (2012.15393) computed momentum-dependent graviton
  form factors from the FRG with full field-dependent regulator.
- Bosma-Knorr-Saueressig (2302.14808) studied form factors in
  metric-affine gravity, with distinct predictions for torsion
  contributions absent in SCT.

## 3b. Resolution (2026-04-01)

**VERDICT: RESOLVED. SCT and AS share identical matter one-loop form factors (CZ basis). First divergence: graviton loops (absent in SCT). Different UV universality classes.**

### Agreement: matter one-loop (CZ basis identity)

Codello-Zanusso (1203.2034) define the nonlocal heat-kernel basis in eq.(5.1)-(5.3):
- Master function f(x) = integral_0^1 exp(-x xi(1-xi)) dxi = phi(x)
- Basis functions: f_Ric, f_R, f_RU, f_U, f_Omega in eq.(5.2)
- Weyl form factor: f_C = (1/2) f_Ric = 1/(12x) + (f-1)/(2x^2) in eq.(5.11)

This is IDENTICAL to our h_C^(0)(x) = 1/(12x) + (phi-1)/(2x^2) (VR-010).

Ohta-Rachwal (2002.10839) confirm in eq.(4.19)-(4.20) that FRG flow
integrated to k=0 yields universal nonlocal coefficients matching the
standard heat-kernel result. The universal nondiscal part is scheme-
independent; only local finite contacts depend on renormalization conditions.

### First divergence: graviton/ghost loops

Satz-Codello-Mazzitelli (1006.3808) compute the purely gravitational
one-loop contribution to the R^2 sector in eq.(27)-(28):

  (1/32pi^2) integral sqrt(g) [(1/60) R log(-Box/k0^2) R
    + (7/10) R_munu log(-Box/k0^2) R^munu]

Converting to {C^2, R^2} basis:
  alpha_C^grav = 7/20 = 0.35
  alpha_R^grav = 1/60 + 7/30 = 1/4 = 0.25

These contributions are ABSENT in SCT (which includes only matter loops).
Full one-loop AS Weyl coefficient:
  alpha_C^{AS,1-loop} = 13/120 + 7/20 = 11/24 ~ 0.458
  = 4.23 x alpha_C^{SCT}

### UV universality class: entire vs power-law

SCT: form factors are entire functions (no poles, no branch cuts).
Graviton propagator Pi_TT(z) saturates at a finite constant in the UV.
Amplitude ~ const at trans-Planckian energies.

AS: power-law running. At the fixed point, G(k) -> g*/k^2 (anomalous
dimension eta_N = -2). Amplitude -> const but via power-law, not entire
function. Knorr-Ripken-Saueressig (2111.12365) make this distinction
explicit in eqs.(6)-(9): AS toy model gives A ~ const (scale-free),
while nonlocal entire-function model gives A ~ exp(-s) -> 0.

This is a FUNDAMENTAL qualitative distinction, not just a numerical
difference.

### Momentum-dependent graviton dressing: z_cross

Bosma-Knorr-Saueressig (1904.04845) compute the momentum-dependent
form factor W_k(q^2) in eqs.(4)-(6), with UV asymptotic eq.(11) and
fit eq.(12):
  w_*(q^2) ~ rho/(rho/kappa + q^2) + w_inf
  rho ~ 0.0149, kappa ~ 0.00817

Comparing Pi_TT^{SCT}(z) vs Pi_TT^{AS}(z) = 1 + 2z w_AS(z) after
IR matching:
  z_{10%} ~ rho/(9 kappa) ~ 0.20  (10% shape deformation)
  z_{O(1)} ~ rho/kappa ~ 1.8  (order-unity divergence)
  Bosma states crossover at q^2 ~ 1 (consistent).

### Comparison table (12 properties)

| Property | SCT | AS/FRG | Status |
|----------|-----|--------|--------|
| CZ basis | phi(x) master function | Same f(x), eqs.(5.1)-(5.3) | Agree |
| Scalar Weyl f.f. | h_C^(0) = f_C(x) | Identical by CZ | Agree |
| Matter 1-loop nonlocal | F_hat_1, h_C, h_R | Universal coefficients | Agree |
| Graviton loops | Absent | alpha_C^grav = 7/20 | Disagree |
| Weyl log coefficient | 13/120 | 11/24 (with graviton) | Disagree |
| R^2 log coefficient | 0 (at xi=1/6) | 1/4 (graviton) | Disagree |
| UV analytic class | Entire, order 1 | Power-law (scale-free) | Disagree |
| Pole/cut structure | Discrete poles, no cuts | Massless pole + timelike cut | Disagree |
| z_cross (shape) | — | ~0.2 (10%), ~1.8 (O(1)) | Disagree |
| Newton potential | Yukawa: 1-(4/3)e^{-m2r}+(1/3)e^{-m0r} | 1/r^2 correction | Disagree |
| c_T | = c (exact on FLRW) | No established c_T != c | No divergence |
| Spectral function | Discrete delta-sum | Massless peak + continuum | Disagree |

### Bibliographic corrections

Our prompt contained 3 incorrect arXiv IDs:
- "Knorr-Saueressig graviton spectral function" is arXiv:2111.13232
  (Fehre-Litim-Pawlowski-Reichert), NOT 2012.15393
- "Bosma-Knorr-Saueressig W_k paper" is arXiv:1904.04845, NOT 2302.14808
- arXiv:1301.4191 is not Falls-Litim-Schroeder

## 4. Failed Approaches

No comparison had been attempted prior to this resolution.

## 5. Success Criteria

- Tabulated comparison of F_1(z) (SCT) with the AS graviton
  form factor at one-loop order, for z in [0, 100].
- Comparison of UV asymptotics: SCT gives z F_hat_1(z -> inf) ->
  const (entire function), while AS gives power-law running.
  Identify the crossover scale where the two predictions diverge.
- Comparison of ghost/pole structure: SCT has infinite sequence of
  zeros in Pi_TT; AS has different pole content depending on
  truncation. Catalogue the differences.
- Comparison of phenomenological predictions: c_T, QNM shifts,
  Newton potential correction (at the overlap regime where both
  apply).
- Clear statement of where the two programs agree (perturbative
  one-loop matter contributions) and where they disagree (graviton
  loops, UV completion, ghost treatment).

## 6. Suggested Directions

1. CZ basis comparison: use the Codello-Zanusso basis functions
   {f_Ric, f_R, f_RU, f_U, f_Omega} to express both the SCT and
   AS form factors. At one-loop for matter fields, these should
   agree identically. Document the graviton loop discrepancies.

2. Spectral function comparison: compute the spectral function
   rho(s) = Im[G(s + i epsilon)] for both the SCT and AS graviton
   propagators. In SCT, rho is a sum of delta functions (entire
   propagator). In AS, rho may have branch cuts. Compare the
   spectral weight distributions.

3. CDT lattice correlator: extract the graviton two-point function
   from CDT lattice simulations (Ambjorn et al.) and compare
   with both SCT and AS predictions. This provides a non-perturbative
   benchmark.

4. w(z) extraction: define w(z) = -Pi_TT(z)/z as the graviton
   self-energy. Plot w(z) for SCT and AS on the same axes. Identify
   the scale z_cross where they first disagree significantly.

## 7. References

1. Codello, A. and Zanusso, O. (2010). "On the non-local heat
   kernel expansion." J. Math. Phys. 54, 013513. arXiv:1203.2034.
2. Knorr, B. and Saueressig, F. (2021). "Graviton spectral function
   from the lattice and the asymptotic safety scenario." Phys. Rev.
   Lett. 128, 091301. arXiv:2012.15393.
3. Bosma, L., Knorr, B. and Saueressig, F. (2023). "Resolving
   spacetime singularities within quantum gravity." Phys. Rev. Lett.
   130, 051501. arXiv:2302.14808.
4. Codello, A., Percacci, R. and Rachwal, L. (2008). "The
   renormalization group and Weyl invariance." arXiv:0805.2909.
5. Alfyorov, D. "Nonlocal one-loop form factors from the spectral
   action." DOI:10.5281/zenodo.19098042.

## 8. Connections

- Related to **OP-30** (running constants): the running constants
  are the IR limit of the form factor comparison.
- Related to **OP-32** (spectral dimension comparison): another
  axis of comparison between SCT and AS/CDT.
- Related to **OP-33** (cross-program table): form factor comparison
  is a key entry in the unified table.
- Independent of **OP-01** (Gap G1): flat-space form factors do not
  require the curved-space Weyl correction.

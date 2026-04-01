---
id: OP-31
title: "Form factor comparison with asymptotic safety and FRG flows"
domain: [predictions, theory]
difficulty: medium
status: open
deep-research-tier: B
blocks: []
blocked-by: []
roadmap-tasks: [COMP-1]
papers: ["0805.2909", "2012.15393", "2302.14808"]
date-opened: 2026-03-31
date-updated: 2026-03-31
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

## 4. Failed Approaches

No comparison has been attempted. The main difficulty is that the
FRG form factors depend on the choice of regulator, gauge fixing,
and truncation order. A meaningful comparison requires identifying
regulator-independent (universal) features of the FRG result and
comparing those with the corresponding SCT quantities.

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

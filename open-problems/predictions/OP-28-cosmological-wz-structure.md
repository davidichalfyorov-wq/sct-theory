---
id: OP-28
title: "Cosmological w(z) and large-scale structure formation"
domain: [predictions, cosmology]
difficulty: medium
status: resolved
deep-research-tier: C
blocks: []
blocked-by: []
roadmap-tasks: [LT-3c]
papers: ["1807.06209", "2404.03002", "1404.3713", "0705.1158"]
date-opened: 2026-03-31
date-updated: 2026-04-03
date-resolved: 2026-04-03
resolved-by: analytical-computation + independent-verification
---

# OP-28: Cosmological w(z) and large-scale structure formation

## 1. Statement

Implement the SCT-modified Friedmann equations in a Boltzmann solver
(CLASS or hi_class) and perform a full Bayesian MCMC analysis against
current cosmological datasets (Planck 2018 + DESI DR1 + Pantheon+ +
KiDS-1000/DES Y3) to quantify the SCT correction to the dark energy
equation of state w(z) and structure formation observables (sigma_8,
S_8).

## 2. Context

The SCT-modified Friedmann equation from NT-4c introduces nonlocal
corrections to the standard Hubble expansion:

  H^2 = (8 pi G / 3) rho * [1 + delta H^2/H^2]

where delta H^2/H^2 encodes the effect of the nonlocal form factors
F_1 and F_2 on the cosmological background. The MT-2 analysis showed
that at the lowest allowed Lambda = 2.38 x 10^{-3} eV, the fractional
correction is

  delta H^2/H^2 ~ 10^{-64},

making it undetectable by any foreseeable experiment. This result is a
NEGATIVE finding: SCT produces no observable cosmological modification
at the background level.

However, the MT-2 result covers only the background equations. A
complete analysis requires:
(a) perturbation-level corrections (delta rho, delta P, sigma),
(b) the full w(z) trajectory rather than just the background H(z),
(c) MCMC marginalization over standard cosmological parameters to
    determine whether SCT produces any tension or preference relative
    to Lambda-CDM.

The expectation is that the MCMC analysis will confirm the NEGATIVE
result, but this must be demonstrated quantitatively rather than
assumed.

## 3. Known Results

- SCT modified Friedmann equations derived (NT-4c). C_T = c exactly.
- Background correction delta H^2/H^2 ~ 10^{-64} (MT-2).
- w_eff = -1 to 63 decimal digits at late times (MT-2).
- De Sitter solution is stable (NT-4c).
- Modified perturbation equations on FLRW derived in principle
  (NT-4c), but not implemented numerically.
- The scalar mode (R^2 sector) decouples at conformal coupling
  xi = 1/6 (3c_1 + c_2 = 0).

## 4. Failed Approaches

No MCMC analysis has been performed. The MT-2 result was obtained
analytically, not through a full Boltzmann solver chain. The
analytical approach is exact for the background but does not produce
perturbation-level predictions (C_l^TT, matter power spectrum P(k),
sigma_8) that can be compared directly with data.

## 5. Success Criteria

- Implementation of the SCT-modified Friedmann and perturbation
  equations in CLASS (or hi_class) as a modified gravity module.
- Full MCMC chains (cobaya or MontePython) with
  Planck 2018 TT/TE/EE + lowE + lensing, DESI DR1 BAO, Pantheon+
  SNIa, and KiDS-1000/DES Y3 weak lensing.
- Posterior distributions for H_0, Omega_m, sigma_8, S_8, and
  the SCT parameter Lambda (or confirmation that Lambda is
  unconstrained by these data).
- Quantitative statement: Bayes factor for SCT vs Lambda-CDM
  (expected to be ~ 1, confirming no preference).
- Documentation of whether the Hubble tension (H_0 ~ 67 vs 73)
  or S_8 tension (0.76 vs 0.83) is affected by SCT corrections
  (expected: no effect).

## 6. Suggested Directions

1. Minimal implementation: add the SCT correction as a constant
   multiplicative factor (1 + epsilon) to the Friedmann equation
   in CLASS, with epsilon = delta H^2/H^2 evaluated at Lambda.
   Run MCMC and confirm that epsilon is unconstrained.

2. Full perturbation implementation: modify the tensor and scalar
   perturbation equations in hi_class to include the nonlocal form
   factors. This requires interfacing the SCT F_1, F_2 functions
   (tabulated or fitted) with the Boltzmann integrator.

3. Effective fluid approach: model the SCT correction as an
   effective dark energy fluid with w(z) = -1 + delta w(z) and
   c_s^2 derived from the scalar propagator Pi_s. This avoids
   modifying the Boltzmann solver internals.

4. Fisher forecast: before running full MCMC, compute a Fisher
   matrix forecast for the sensitivity of DESI + Euclid + LISA to
   the SCT parameter Lambda. This gives an upper bound on the
   information content of the data regarding SCT corrections.

## 7. References

1. Planck Collaboration (2020). "Planck 2018 results. VI. Cosmological
   parameters." Astron. Astrophys. 641, A6. arXiv:1807.06209.
2. DESI Collaboration (2024). "DESI 2024 VI: Cosmological constraints
   from baryon acoustic oscillations." arXiv:2404.03002.
3. Scolnic, D. M. et al. (2022). "The Pantheon+ analysis."
   Astrophys. J. 938, 113. arXiv:2112.03863.
4. Alfyorov, D. "Nonlinear field equations and FLRW cosmology from
   the spectral action." DOI:10.5281/zenodo.19098027.

## 8. Connections

- Related to **OP-27** (GW propagation): both use the NT-4c FLRW
  perturbation framework.
- Related to **OP-17** (scalaron mass): if the scalaron is light, it
  modifies the scalar perturbation spectrum at early times.
- Independent of **OP-01** (Gap G1): FLRW has C = 0, so the Weyl
  correction is absent.
- The NEGATIVE result from MT-2 means this problem is primarily a
  confirmation exercise, but it must be performed for completeness.

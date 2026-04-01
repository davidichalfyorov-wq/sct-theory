---
id: OP-27
title: "Gravitational wave propagation and frequency-dependent dispersion"
domain: [predictions, cosmology]
difficulty: medium
status: open
deep-research-tier: C
blocks: []
blocked-by: []
roadmap-tasks: [LT-3b]
papers: ["1710.05901", "2109.09718"]
date-opened: 2026-03-31
date-updated: 2026-03-31
---

# OP-27: Gravitational wave propagation and frequency-dependent dispersion

## 1. Statement

Compute the frequency-dependent dispersion relation for gravitational
waves propagating through the SCT nonlocal effective action. Determine
whether the entire-function dressing of the graviton propagator
produces measurable group velocity dispersion, and compute the
secondary gravitational wave spectrum from the inflationary epoch
if the INF-1 mechanism is viable.

## 2. Context

The SCT gravitational wave speed c_T = c is proven exactly for all
frequencies at the level of the FLRW background (NT-4c). This
follows from the conformal invariance of C^2 in d = 4, which
ensures that tensor perturbations obey the same dispersion as in GR
on conformally flat backgrounds. The bound |c_T/c - 1| < 10^{-15}
from GW170817/GRB170817A is satisfied automatically.

However, frequency-dependent corrections to the group velocity have
not been computed. In the SCT framework, the linearised tensor
perturbation equation on a cosmological background is

  h_TT''(k, eta) + 2 a H h_TT'(k, eta) + k^2 Pi_TT(k^2/Lambda^2) h_TT(k, eta) = 0

where Pi_TT(z) = 1 + (13/60) z F_hat_1(z). The phase velocity is
v_ph(k) = sqrt(Pi_TT(k^2/Lambda^2)) which deviates from 1 at
k ~ Lambda. The group velocity v_g = d(omega)/dk acquires a
frequency-dependent correction that could in principle be measured
through dispersion of GW signals over cosmological distances.

## 3. Known Results

- c_T = c on FLRW to all accessible numerical precision (NT-4c,
  verified to 63 decimal digits).
- The tensor perturbation equation in conformal time on FLRW with
  SCT corrections is derived (NT-4c).
- Pi_TT is entire and positive on the positive real axis
  (Pi_entire(z) > 0 for real z >= 0, verified to z = 100).
- Polarization content: SCT has exactly 2 tensor modes, no scalar
  or vector GW polarizations at the linearised level (NT-4a).
- The inflationary perturbation spectrum (INF-1) predicts
  n_s(N=55) = 0.965, r ~ 3.5 x 10^{-3}, but is conditional on
  the scalaron mass problem.

## 4. Failed Approaches

No computation has been attempted. The main conceptual subtlety is
that c_T = c is exact on the FLRW background, so the dispersion
must arise from sub-leading effects:

1. Interaction with the cosmological perturbation (delta g beyond
   FLRW), which introduces inhomogeneous corrections.
2. Propagation through matter (graviton self-energy from matter
   loops), which is already encoded in F_1 but requires careful
   treatment of the cosmological background.

The fact that c_T = c is exact may mean that the leading dispersion
correction is exponentially suppressed at k << Lambda, leaving no
observable effect for astrophysical GWs.

## 5. Success Criteria

- Explicit dispersion relation omega(k) for tensor perturbations
  on FLRW, including the leading correction from Pi_TT.
- Quantitative bound on the group velocity deviation
  |v_g - c| / c as a function of k/Lambda.
- For the secondary GW spectrum: computation of Omega_GW(f) from
  the INF-1 inflationary model (if viable), including SCT-specific
  modifications to the transfer function.
- Comparison with PTA constraints (NANOGrav 15yr, EPTA DR2) and
  LISA forecasts.
- Clear statement of whether SCT-specific dispersion is in principle
  detectable with current or next-generation instruments.

## 6. Suggested Directions

1. Perturbative dispersion: expand Pi_TT(k^2/Lambda^2) to first
   order in k^2/Lambda^2 and compute the resulting phase shift
   accumulated over a cosmological propagation distance. The effect
   scales as (f/Lambda)^2 L/c, where f is the GW frequency and L
   the luminosity distance.

2. Modified Boltzmann approach: implement the SCT-modified tensor
   perturbation equation in a Boltzmann solver (CLASS or hi_class)
   and compute the tensor power spectrum C_l^TT. Compare with the
   standard GR result.

3. Stochastic GW background: if INF-1 produces a viable inflationary
   spectrum, compute Omega_GW(f) using the SCT transfer function
   and compare with NANOGrav/EPTA constraints on the GW background.

4. Phenomenological bounds: use the parametrised dispersion framework
   of Mirshekari et al. (1109.4258) where v_g^2 = 1 + A (E/E_QG)^alpha.
   Map the SCT propagator onto this parametrisation and extract A
   and alpha.

## 7. References

1. Abbott, B. P. et al. (LVK, 2017). "Gravitational waves and
   gamma-rays from a binary neutron star merger: GW170817 and
   GRB 170817A." Astrophys. J. 848, L13. arXiv:1710.05901.
2. EPTA Collaboration (2021). "Common-process signal in PTA data."
   Mon. Not. Roy. Astron. Soc. 508, 4970. arXiv:2109.09718.
3. Mirshekari, S., Yunes, N. and Will, C. M. (2012). "Constraining
   Lorentz-violating, modified dispersion relations with gravitational
   waves." Phys. Rev. D 85, 024041. arXiv:1110.2720.
4. Alfyorov, D. "Nonlinear field equations and FLRW cosmology from
   the spectral action." DOI:10.5281/zenodo.19098027.

## 8. Connections

- Related to **OP-28** (cosmological w(z)): both use the FLRW
  perturbation equations from NT-4c.
- Related to **OP-17** (scalaron mass): the inflationary secondary
  spectrum depends on the scalaron mass hierarchy.
- Independent of **OP-26** (QNM shifts): QNMs probe strong-field
  black hole geometry, while GW propagation is a weak-field
  cosmological effect.
- Independent of **OP-01** (Gap G1): GW propagation on FLRW does
  not require the Weyl-sector correction (C = 0 on FLRW).

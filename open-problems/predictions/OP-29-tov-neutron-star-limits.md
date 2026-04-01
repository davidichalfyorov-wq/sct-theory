---
id: OP-29
title: "Modified TOV equations and neutron star mass limits"
domain: [predictions, stellar-structure]
difficulty: hard
status: open
deep-research-tier: C
blocks: []
blocked-by: [OP-01]
roadmap-tasks: [LT-3e]
papers: ["1614.01346", "2105.06980"]
date-opened: 2026-03-31
date-updated: 2026-03-31
---

# OP-29: Modified TOV equations and neutron star mass limits

## 1. Statement

Derive the SCT-modified Tolman-Oppenheimer-Volkoff (TOV) equations
for static spherically symmetric stellar configurations. Compute the
maximum neutron star mass for realistic nuclear equations of state
(SLy, AP4, BSk21) and verify that the result satisfies the
observational constraint M_max >= 2.14 M_sun from PSR J0740+6620.
Failure to satisfy this bound constitutes falsification of the theory
at the given Lambda value.

## 2. Context

The TOV equations describe hydrostatic equilibrium in GR:

  dP/dr = -(rho + P)(m(r) + 4 pi r^3 P) / (r(r - 2 m(r))),
  dm/dr = 4 pi r^2 rho.

In SCT, the nonlocal field equations modify both the gravitational
sector (via F_1 and F_2 form factors) and the effective coupling
between matter and geometry. Inside a neutron star, the matter
density reaches rho ~ 10^{15} g/cm^3 and the curvature R ~ 10^{-10}
m^{-2}, both far below the SCT cutoff Lambda^2 for any Lambda above
the laboratory bound (Lambda >= 2.565 meV implies 1/Lambda <= 77 um).

The interior of a neutron star is static and spherically symmetric,
with C_{mu nu rho sigma} != 0 (Schwarzschild-like exterior, non-trivial
interior). The full nonlinear field equations (NT-4b) are needed, and
the Weyl-sector correction Theta^(C) must be evaluated on the stellar
interior metric. This makes the problem dependent on OP-01 (Gap G1).

## 3. Known Results

- The Newtonian-limit correction V(r)/V_N = 1 - (4/3) e^{-m_2 r}
  + (1/3) e^{-m_0 r} is known (NT-4a), but this is insufficient for
  the strong-field stellar interior.
- The FLRW reduction (NT-4c) shows that on conformally flat
  backgrounds the SCT correction is negligible. The stellar interior
  is NOT conformally flat (C != 0), so this argument does not apply.
- In Stelle's fourth-derivative gravity, modified TOV equations have
  been derived and studied (Stelle 1977, Olmo et al. 2020). The
  maximum mass shifts by O((M_star/M_P)^2 (Lambda/m_2)^4).
- For the standard MOG/MOND-type modifications, the TOV constraint
  M_max >= 2.14 M_sun is a well-established discriminator.
- Observed neutron stars: PSR J0740+6620 at 2.08 +/- 0.07 M_sun,
  PSR J0348+0432 at 2.01 +/- 0.04 M_sun.

## 4. Failed Approaches

No computation has been attempted. The problem is blocked by OP-01.
A perturbative approach treating the SCT correction as a small
deviation from the GR TOV solution might bypass the need for the full
Theta^(C), but this has not been investigated.

## 5. Success Criteria

- Explicit SCT-modified TOV equations, valid at least perturbatively
  in Lambda^{-2} on a static spherically symmetric background.
- Numerical solutions for the mass-radius (M-R) relation using at
  least three nuclear EOS: SLy (medium stiffness), AP4 (soft), and
  BSk21 (stiff).
- Maximum mass M_max(Lambda) as a function of the SCT cutoff Lambda,
  with comparison to the observational bound M_max >= 2.14 M_sun.
- If the SCT correction increases M_max, this is a positive
  prediction. If it decreases M_max below 2.14 M_sun for any
  Lambda in the allowed range, this constitutes a falsification
  bound on Lambda.
- Tidal deformability Lambda_tidal(1.4 M_sun) for comparison with
  the LIGO/Virgo constraint from GW170817:
  Lambda_tidal in [70, 580] (90% CL).

## 6. Suggested Directions

1. Perturbative TOV: treat the SCT modification as a first-order
   correction to the GR TOV solution. Write P = P_GR + delta P,
   m = m_GR + delta m, and linearise the modified field equations.
   This avoids the full Theta^(C) computation if the correction is
   small.

2. Effective metric approach: model the SCT modification as a
   correction to the Schwarzschild exterior metric (Yukawa potential)
   and match at the stellar surface. The interior remains GR, but
   the boundary condition at r = R_star is modified.

3. Scalar-tensor analogue: in the scalar sector (Pi_s), the SCT
   modification at general xi resembles a scalar-tensor theory. Use
   the well-developed scalar-tensor TOV formalism (Damour-Esposito-Farese)
   as a template, identifying the SCT scalar mode with the
   Brans-Dicke scalar.

4. Numerical integration: implement the full modified TOV equations
   in a shooting code (SciPy odeint) with tabulated EOS. Scan over
   central density and Lambda simultaneously.

## 7. References

1. Olmo, G. J., Rubiera-Garcia, D. and Wojnar, A. (2020). "Stellar
   structure models in modified theories of gravity: lessons and
   challenges." Phys. Rept. 876, 1. arXiv:1912.05202.
2. Cromartie, H. T. et al. (2020). "Relativistic Shapiro delay
   measurements of an extremely massive millisecond pulsar."
   Nat. Astron. 4, 72. arXiv:1614.01346.
3. Riley, T. E. et al. (2021). "A NICER view of the massive pulsar
   PSR J0740+6620 informed by radio timing and XMM-Newton
   spectroscopy." Astrophys. J. Lett. 918, L27. arXiv:2105.06980.
4. Alfyorov, D. "Nonlinear field equations and FLRW cosmology from
   the spectral action." DOI:10.5281/zenodo.19098027.

## 8. Connections

- **Blocked by OP-01** (Gap G1): the stellar interior has nonzero
  Weyl curvature, requiring Theta^(C).
- Related to **OP-26** (QNM shifts): both require solving the
  nonlocal field equations on spherically symmetric backgrounds.
- Independent of cosmological problems (OP-28): neutron star
  physics probes the strong-field regime, not the cosmological sector.
- A negative result (M_max < 2.14 M_sun) would be a strong
  falsification signal, complementary to the laboratory bounds in
  LT-3d.

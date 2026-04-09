---
id: OP-26
title: "Quasinormal mode frequency shifts from the nonlocal SCT action"
domain: [predictions, black-holes]
difficulty: hard
status: resolved (Level 2)
deep-research-tier: C
blocks: []
blocked-by: [OP-01]
roadmap-tasks: [LT-3a]
papers: ["0905.2975", "1602.07309", "0710.5167", "2210.14506", "2412.15037"]
date-opened: 2026-03-31
date-updated: 2026-04-04
resolution: |
  LT-3a COMPLETE at Level 2 (Yukawa approximation). 7 scripts, 94 tests, 17 figures.
  TWO contributions to QNM shifts: (1) metric modification ~ exp(-m2*r_peak) [computed],
  (2) perturbation-equation correction ~ c2*(omega/Lambda)^2 ~ 10^{-20} [estimated, OP-01].
  Both unmeasurable: SCT indistinguishable from GR for QNMs of all astrophysical BHs.
  Key results: stability proven (V>=0), Love numbers k2 != 0 (qualitative GR difference),
  quantum corrections (10^{-78}) dominate classical SCT corrections (exp(-10^9)).
  Full OP-01 resolution needed for exact perturbation-equation computation.
---

# OP-26: Quasinormal mode frequency shifts from the nonlocal SCT action

## 1. Statement

Compute the shift in quasinormal mode (QNM) frequencies of
Schwarzschild and Kerr black holes induced by the nonlocal SCT
effective action. This requires modifying the Regge-Wheeler, Zerilli,
and Teukolsky equations to incorporate the entire-function form
factor F_1(Box/Lambda^2) and extracting the resulting complex
eigenfrequencies omega_n = omega_R + i omega_I as functions of
Lambda and the black hole parameters (M, a).

## 2. Context

Black hole ringdown is the cleanest probe of strong-field gravity
modifications. The LVK collaboration measures QNM frequencies from
binary black hole mergers, with O4/O5 expected to provide sub-percent
measurements of the fundamental mode for several events. Any
higher-derivative correction to GR shifts the QNM spectrum.

In SCT, the linearised field equations (NT-4a) modify the graviton
propagator via Pi_TT(z) = 1 + (13/60) z F_hat_1(z). On a
Schwarzschild background, this translates into a modified effective
potential in the Regge-Wheeler/Zerilli equations. The modification
depends on the full nonlocal form factor, not just its local limit,
because the relevant curvature scale r_s = 2GM can be comparable
to 1/Lambda for astrophysical black holes if Lambda is sufficiently
low.

The computation is blocked by OP-01 (Gap G1): the Weyl-sector
variational correction Theta^(C)_{mu nu} is not yet computed on
non-trivially curved backgrounds. For Schwarzschild, C_{mu nu rho sigma}
is nonzero, so the full nonlinear field equations cannot be written
down without resolving G1.

## 3. Known Results

- The linearised propagator and its pole structure are fully
  characterised (NT-4a). The spin-2 effective mass is
  m_2 = Lambda sqrt(60/13) ~ 2.148 Lambda.
- The modified Newtonian potential V(r)/V_N = 1 - (4/3) e^{-m_2 r}
  + (1/3) e^{-m_0 r} is known (NT-4a), but this is a weak-field
  result and does not capture strong-field QNM physics.
- Ghost poles of Pi_TT at z_L ~ -1.2807 and z_0 ~ 2.4148 could
  introduce additional QNM branches if the fakeon prescription is
  not enforced.
- In Stelle's fourth-derivative gravity (polynomial truncation of
  the SCT propagator), QNM shifts have been computed perturbatively
  by Konoplya and Zhidenko (0905.2975).
- The BH entropy logarithmic correction c_log = 37/24 is known
  from the spectral action (MT-1, conditional on OP-01 resolution).

## 4. Failed Approaches

No explicit computation has been attempted. The fundamental obstacle
is the need for Theta^(C) on Schwarzschild. Two conceptual approaches
have been considered but not executed:

1. Perturbative expansion in M Lambda around the flat-space result.
   This would give QNM shifts as power series in (m_2 r_s)^{-2}.
   The expansion is expected to converge slowly for astrophysical
   M and low Lambda.

2. WKB approach with modified potential. Replace the standard
   Regge-Wheeler potential with one incorporating the F_1 form
   factor. This requires commuting F_1(Box) with the background
   curvature, which is the content of OP-01.

## 5. Success Criteria

- Modified Regge-Wheeler equation for Schwarzschild with the SCT
  nonlocal correction, valid at least perturbatively in Lambda^{-1}.
- QNM frequency shifts delta omega_n / omega_n as a function of
  m_2 r_s for the fundamental mode (n=0) and first overtone (n=1),
  for l = 2 (dominant quadrupole mode).
- Numerical evaluation for M = 10, 30, 100 M_sun and Lambda spanning
  the allowed range (Lambda >= 2.565 meV from LT-3d).
- Comparison with the Konoplya-Zhidenko results for Stelle gravity.
- Forecast of detectability with LVK O5 and LISA sensitivity curves.

## 6. Suggested Directions

1. Perturbative approach: expand Theta^(C) in powers of C/Lambda^2
   around flat space. At leading order, the modification to the
   Regge-Wheeler potential is a correction of order (r_s Lambda)^{-2}.
   This may suffice for phenomenological bounds.

2. Parametrised QNM framework: use the parametrised approach of
   Cardoso et al. (1602.07309) where deviations from GR QNMs are
   encoded in a small number of coefficients. Map the SCT propagator
   modification onto this parametrisation.

3. Numerical approach: solve the modified wave equation on a
   Schwarzschild background using spectral methods (Chebyshev
   collocation), treating F_1(Box) as a frequency-dependent
   modification to the potential barrier.

4. Eikonal limit: in the eikonal (l >> 1) regime, QNM frequencies
   are related to unstable circular photon orbits. The SCT correction
   to the photon sphere radius can be extracted from the modified
   geodesic equation, giving an l-independent fractional shift.

## 7. References

1. Konoplya, R. A. and Zhidenko, A. (2009). "Quasinormal modes of
   black holes: from astrophysics to string theory." Rev. Mod. Phys.
   83, 793. arXiv:0905.2975.
2. Cardoso, V. et al. (2019). "Parametrized black hole quasinormal
   ringdown." Phys. Rev. D 99, 104077. arXiv:1602.07309.
3. Berti, E., Cardoso, V. and Starinets, A. O. (2009). "Quasinormal
   modes of black holes and black branes." Class. Quant. Grav. 26,
   163001. arXiv:0905.2975.
4. Nollert, H.-P. (1999). "Quasinormal modes: the characteristic
   'sound' of black holes and neutron stars." Class. Quant. Grav. 16,
   R159. arXiv:0710.5167.
5. Alfyorov, D. "Nonlinear field equations and FLRW cosmology from
   the spectral action." DOI:10.5281/zenodo.19098027.

## 8. Connections

- **Blocked by OP-01** (Gap G1): the Weyl-sector correction Theta^(C)
  on Schwarzschild is the essential prerequisite.
- Related to **OP-21** (BH singularity resolution): QNM overtone
  spectrum probes near-horizon geometry where singularity resolution
  would be relevant.
- Related to **OP-22** (BH second law): entropy corrections and
  QNM frequencies are linked through the area-entropy relation and
  quasinormal mode spectroscopy.
- Independent of all cosmological problems (OP-28, OP-17, OP-18).

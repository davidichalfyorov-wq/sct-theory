# Cosmology: Background Briefing

SCT modifies the Friedmann equations through nonlocal curvature
corrections arising from the entire form factors F_1 and F_2. The
modified first Friedmann equation is

  H^2 = (8piG/3) rho + delta H^2

where delta H^2 encodes the spectral action corrections. These
corrections are suppressed by powers of H^2/Lambda^2, where Lambda
is the spectral cutoff and H is the Hubble parameter.

## Quantitative suppression

At the PPN-1 lower bound Lambda >= 2.565 meV, the fractional correction
to the Hubble rate is

  delta H^2 / H^2 ~ 1.3 * 10^{-64}

This is a NEGATIVE result (MT-2): the correction is 64 orders of
magnitude below observable thresholds. The effective equation of state
parameter is w_eff = -1 to 63 decimal places at late times.

## Why SCT cannot address cosmological tensions

1. **H_0 tension:** requires O(1%) modification to H(z) at z ~ 0-2.
   SCT provides O(10^{-64}), which is exactly zero for all practical
   purposes.

2. **S_8 tension:** requires modified growth of structure at late times.
   SCT's UV-nonlocal corrections do not propagate to IR scales.

3. **Maggiore's argument:** entire functions (which arise from the
   spectral action) cannot generate inverse-d'Alembertian nonlocality
   (Box^{-1}) at IR scales. The UV form factors F_1(Box/Lambda^2) and
   F_2(Box/Lambda^2) are entire in Box, so they decay rapidly at low
   momenta. To affect late-time cosmology at H ~ 10^{-33} eV, one
   would need IR-singular form factors (Box^{-1} or m^{-2} terms),
   which the spectral action does not produce.

## FLRW and de Sitter results

- **De Sitter:** exact solution of the nonlinear SCT field equations.
  The Weyl tensor vanishes on FLRW backgrounds (C_{mu nu rho sigma} = 0),
  so the C^2 sector (Gap G1) does not contribute.
- **FLRW stability:** de Sitter is a stable attractor. Perturbations
  decay on timescale ~ 1/H.
- **Gravitational wave speed:** c_T = c exactly. The tensor sector has
  Pi_TT(0) = 1, so GW speed equals light speed at all frequencies below
  Lambda. Constraint |c_T/c - 1| < 10^{-15} from GW170817 is satisfied
  with margin of order 10^{-108}.

## Inflation

The spectral action contains an R^2 term with coefficient alpha_R(xi) =
2(xi - 1/6)^2. This term sources Starobinsky-type inflation with a
scalaron of mass

  M_0^2 = Lambda^2 / (12 (xi - 1/6)^2)

At minimal coupling (xi = 0):

  M_0 = Lambda * sqrt(3) * sqrt(4pi^2) / sqrt(2) ~ 15.39 M_Pl

(using the Planck mass normalization M_Pl = 1/sqrt(8piG)). CMB
normalization of Starobinsky inflation requires M ~ 1.28 * 10^{-5} M_Pl.
The ratio is approximately 1.2 million.

## Dilaton sector

The trace mode of the Dirac operator gives a scalar field (dilaton)
coupled to all SM particles. Five structural obstacles prevent dilaton
inflation: tree-level potential is flat, phi-dependent masses are
absent, no anomaly-driven potential V(phi) arises, Coleman-Weinberg
shapes are excluded by BICEP/Keck data, and the R^2 term produces a
scalaron rather than a dilaton.

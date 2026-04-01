# Black Holes: Background Briefing

SCT addresses black hole physics through the nonlocal modifications
to the gravitational field equations and the spectral action evaluated
on horizon geometries.

## Black hole entropy

The Bekenstein-Hawking entropy is derived from the spectral action
evaluated on the Euclidean Schwarzschild geometry (conical singularity
method). The result is

  S = A/(4G) + 13/(120 pi) + (37/24) ln(A/l_P^2) + O(1)

where A is the horizon area and l_P is the Planck length.

The logarithmic coefficient c_log = 37/24 is parameter-free: it is
determined by the SM particle content via the Sen formula (2012).
Specifically:

  c_log = (1/180)(N_s + 6 N_D + 12 N_v) - (1/12)(N_s + N_D + 2 N_v)
        + (1/360)(2 + N_s + 6 N_D + 12 N_v)

with N_s = 4, N_D = 22.5, N_v = 12. This gives c_log = 37/24 = 1.5417.

The sign of c_log is opposite to the Loop Quantum Gravity prediction
(c_log^LQG = -3/2), providing a potential observational discriminant.

## Four laws of black hole thermodynamics

- **Zeroth law (surface gravity constant):** PASS. Follows from the
  Einstein sector of the field equations.
- **First law (dM = T dS + ...):** PASS. Wald entropy formula applied
  to the SCT effective action reproduces the entropy above.
- **Third law (extremality):** PASS. The nonlocal corrections do not
  modify the extremality condition for Kerr-Newman.
- **Second law (area increase):** CONDITIONAL. Requires the ghost
  sector to satisfy Wall's stability condition (positive energy flux
  through the horizon). This depends on the unitarity/fakeon resolution
  (MR-2), which is open.

## Modified Newtonian potential

The linearised field equations (NT-4a) give a modified potential

  V(r)/V_N(r) = 1 - (4/3) exp(-m_2 r) + (1/3) exp(-m_0 r)

where m_2 = Lambda sqrt(60/13) and m_0 = Lambda sqrt(6) (at xi = 0).
This potential is finite at r = 0: V(0) = 0 (exact cancellation).
The Yukawa coupling alpha = -4/3 is parameter-free.

## Singularity status

Whether the nonlocal field equations resolve the Schwarzschild
singularity is unknown. The linearised result V(0) = 0 is suggestive
but not sufficient: the full nonlinear problem requires the Weyl-sector
correction Theta^(C)_{mu nu} (Gap G1, OP-01), which has not been
computed on backgrounds with C_{mu nu rho sigma} != 0.

## Ghost sector

The dressed graviton propagator has 8 complex poles (2 real, 6 complex
conjugate pairs in Lee-Wick arrangement) and one physical ghost at
z_L = -1.2807 (Lorentzian timelike). The ghost must be treated as a
fakeon (Anselmi prescription) or reinterpreted as dark matter. The
choice affects the second law and singularity resolution.

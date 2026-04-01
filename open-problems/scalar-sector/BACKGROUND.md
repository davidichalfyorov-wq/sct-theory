# Scalar Sector: Background Briefing

The scalar sector of the SCT graviton propagator is controlled by
the spin-0 Barnes-Rivers projector P^{(0-s)} and the associated
dressing function

  Pi_s(z, xi) = 1 + 6(xi - 1/6)^2 z F_hat_2(z, xi)

where xi is the non-minimal coupling of the Higgs doublet to the
Ricci scalar (the coupling in the Lagrangian term xi R |Phi|^2),
z = k^2/Lambda^2, and F_hat_2 is the normalized Ricci form factor.

## Role of xi in SCT

The parameter xi is the only free dimensionless parameter in the
one-loop SCT effective action (beyond Lambda itself). It determines:

- The R^2 coefficient: alpha_R(xi) = 2(xi - 1/6)^2.
- The scalar propagator dressing: Pi_s(z, xi).
- The coefficient ratio: c_1/c_2 = -1/3 + 120(xi - 1/6)^2/13.
- The scalar mode combination: 3c_1 + c_2 = 6(xi - 1/6)^2.
- The scalar effective mass: m_0(xi) = Lambda / sqrt(6(xi - 1/6)^2).

At the conformal coupling xi = 1/6, the scalar sector decouples
entirely: alpha_R = 0, 3c_1 + c_2 = 0, m_0 -> infinity. The
modified Newtonian potential reduces to V(r)/V_N = 1 - (4/3) e^{-m_2 r},
and the scalar Yukawa term vanishes.

## Ghost structure at general xi

The SS (scalar sector) analysis established a complete ghost catalogue
as a function of xi:

- **xi = 1/6 (conformal):** Pi_s = 1 identically. No scalar
  propagation, no ghosts. The spin-0 graviton mode is absent.
- **0 < xi < 1/6 and 1/6 < xi < xi_c:** Pi_s has complex conjugate
  zeros (Lee-Wick pairs). These are physically acceptable under the
  fakeon prescription, as the complex poles do not contribute to
  the physical spectral function.
- **xi = 0 (minimal):** Pi_s has Lee-Wick pairs at specific complex
  locations. m_0(0) = Lambda sqrt(6) ~ 2.449 Lambda.
- **xi = xi_c:** the first real zero of Pi_s appears. This is the
  critical coupling where a genuine scalar ghost materialises.
- **xi > xi_c:** at least one real ghost pole with negative residue.
  Spectral positivity is violated and the fakeon prescription alone
  may not suffice.

The critical coupling xi_c was determined numerically in the SS
analysis: xi_c is in the interval (0.4445, 0.4446].

## Physical determination of xi

Within the spectral triple framework, xi is in principle determined
by the geometry of the finite spectral triple (A_F, H_F, D_F). The
Chamseddine-Connes spectral action on the product geometry M x F
produces a specific xi through the Seeley-DeWitt coefficient a_4.
However, the exact value depends on the detailed structure of D_F,
which is constrained but not uniquely fixed by the SM particle
content.

From the perspective of RG running, xi(mu) flows under
renormalization. The one-loop beta function for xi in the SM is

  beta_xi = (xi - 1/6)(6 lambda + 6 y_t^2 - 3g'^2/2 - 9g^2/2) / (16 pi^2)

where lambda is the Higgs quartic, y_t the top Yukawa, and g', g
the electroweak gauge couplings. The conformal value xi = 1/6 is
a fixed point of this beta function in the SM.

## Current status

- Pi_s ghost catalogue: COMPLETE (SS analysis, 64+ tests).
- Critical coupling: xi_c in (0.4445, 0.4446] (50-digit precision).
- Conformal decoupling: verified (3c_1 + c_2 = 0 at xi = 1/6).
- PPN discrimination: PPN gamma = 1 for all xi. Solar system tests
  cannot distinguish between xi values.
- Physical xi determination: OPEN. Requires spectral triple input.

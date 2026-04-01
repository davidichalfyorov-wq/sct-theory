---
id: OP-44
title: "Critical coupling xi_c and the physical value of xi"
domain: [scalar-sector, theory]
difficulty: medium
status: resolved
deep-research-tier: A
blocks: []
blocked-by: []
roadmap-tasks: []
papers: ["0610241", "1403.4226", "1004.0464", "1304.8050", "1304.0415", "0710.3755"]
date-opened: 2026-03-31
date-updated: 2026-03-31
date-resolved: 2026-03-31
resolved-by: NCG-literature-analysis + independent-numerical-verification
---

# OP-44: Critical coupling xi_c and the physical value of xi

## 1. Statement

Determine the physical value of the Higgs non-minimal coupling xi
within the SCT framework. Characterize the transition at the critical
coupling xi_c in (0.4445, 0.4446] where the first real scalar ghost
appears, and establish whether the physical xi lies in the ghost-free
region (xi <= xi_c) or the pathological region (xi > xi_c).

## 2. Context

The non-minimal coupling xi R |Phi|^2 is the sole free dimensionless
parameter in the one-loop SCT effective action (apart from the cutoff
Lambda). Its value controls the entire scalar sector of the graviton
propagator through

  Pi_s(z, xi) = 1 + 6(xi - 1/6)^2 z F_hat_2(z, xi).

Three distinguished values of xi have qualitatively different physics:

1. **xi = 1/6 (conformal coupling).** The scalar mode decouples:
   Pi_s = 1, alpha_R = 0, 3c_1 + c_2 = 0. This is the unique
   ghost-free point where the scalar graviton is absent entirely.
   The modified Newtonian potential has only the spin-2 Yukawa
   correction. The SM RG beta function has xi = 1/6 as a fixed
   point.

2. **xi = 0 (minimal coupling).** Pi_s has complex Lee-Wick pairs,
   physically acceptable under the fakeon prescription. The scalar
   effective mass is m_0 = Lambda sqrt(6) ~ 2.449 Lambda. The
   modified potential V(r)/V_N = 1 - (4/3) e^{-m_2 r} + (1/3) e^{-m_0 r}
   has both Yukawa corrections active. alpha_R(0) = 1/18.

3. **xi = xi_c ~ 0.4446.** The critical coupling at which the first
   real zero of Pi_s crosses the real axis. For xi > xi_c, a real
   ghost with negative spectral weight appears, violating positivity.
   The critical value was computed numerically: xi_c lies in
   (0.4445, 0.4446], determined to 50-digit precision.

For xi = 1 (the value often considered in inflationary models via
Higgs inflation), there is a real ghost at z ~ -0.233 with a
negative residue, violating spectral positivity. This rules out
xi = 1 in the SCT framework unless additional mechanisms (beyond
the standard fakeon prescription) remove the ghost.

## 3. Known Results

- Ghost catalogue at general xi: COMPLETE (SS analysis).
  - xi in [0, xi_c]: only complex (Lee-Wick) scalar poles. Fakeon-safe.
  - xi > xi_c: at least one real scalar ghost. Spectral positivity
    violated.
  - xi = 1: real ghost at z ~ -0.233, R ~ -0.15 (negative residue).
- Critical coupling: xi_c in (0.4445, 0.4446] (50-digit precision).
- PPN gamma: equals 1 for all xi to all accessible precision. Solar
  system observations cannot distinguish between xi values.
- SM RG flow: beta_xi = (xi - 1/6) x (...), so xi = 1/6 is a
  perturbative fixed point. The SM running drives xi toward 1/6 at
  high energies.
- Conformal coupling xi = 1/6: unique ghost-free value. Scalar mode
  absent. All SCT cosmological results (c_T = c, de Sitter stability,
  modified Friedmann) are independent of xi because C = 0 on FLRW.
- The spectral action on the Chamseddine-Connes spectral triple
  produces xi through the a_4 Seeley-DeWitt coefficient. The exact
  value depends on the finite Dirac operator D_F.

## 4. Failed Approaches

1. **Solar system discrimination.** PPN gamma = 1 for all xi
   (exponential suppression of Yukawa corrections). Solar system data
   cannot determine xi. The Eoet-Wash and Casimir bounds constrain
   Lambda but are insensitive to xi.

2. **Cosmological discrimination.** All FLRW observables (H(z), c_T,
   perturbation spectra) are xi-independent at the background level
   because C = 0 on conformally flat spacetimes. The R^2 correction
   is proportional to (xi - 1/6)^2, which enters only through scalar
   perturbations (not the tensor sector).

3. **Direct spectral triple computation.** The value of xi from the
   NCG spectral action depends on the detailed choice of D_F, which
   is constrained by the SM particle content but not uniquely fixed.
   Different choices in the literature (Chamseddine-Connes 2006,
   Devastato-Lizzi-Martinetti 2014) give different xi values.

## 4b. Resolution (2026-03-31)

**VERDICT: xi = 1/6 is a PREDICTION of the standard spectral action, not a free parameter.**

### Mechanism

In the a_4 Seeley-DeWitt coefficient of the spectral action
S = Tr(f(D^2/Lambda^2)), the Yukawa-dependent coefficient `a`
multiplies BOTH the curvature coupling R|H|^2 AND the kinetic term
|nabla H|^2 simultaneously. After canonical normalization of the
Higgs field, this common factor cancels, leaving a fixed ratio = the
conformal coupling value.

Specifically (van Suijlekom, Proposition 8.12, Eq. (8.4.10)):

  L_phi ∝ (f(0)/48 pi^2) s Tr(Phi^2) + (f(0)/8 pi^2) Tr((D_mu Phi)^2)

The ratio curvature-term/kinetic-term = 1/6, independent of Lambda,
the shape of f, Yukawa couplings, or CKM parameters.

### Convention clarification (CRITICAL)

The printed value xi_0 = 1/12 in Chamseddine-Connes-Marcolli
(hep-th/0610241, Eqs. (4.9)-(4.12)) uses the convention
L = (1/2)|DH|^2 - xi_0 R|H|^2. In the standard QFT convention
L = |D Phi|^2 - xi R|Phi|^2 with Phi = H/sqrt(2):

  xi_0 = 1/12  <==>  xi_phys = 1/6  (CONFORMAL COUPLING)

Verified algebraically: (1/2)|D(sqrt(2) Phi)|^2 = |D Phi|^2,
(1/12) R |sqrt(2) Phi|^2 = (1/6) R |Phi|^2.

### Literature confirmation (5 independent groups)

| Authors | Year | arXiv | Eq. | Printed xi | Physical xi |
|---------|------|-------|-----|-----------|-------------|
| Chamseddine-Connes-Marcolli | 2006 | hep-th/0610241 | (4.9)-(4.12) | 1/12 | 1/6 |
| Chamseddine-Connes | 2010 | 1004.0464 | (5.47),(6.15) | 1/6 a | 1/6 |
| van Suijlekom | 2015 | book, Ch. 11 | Prop.8.12, Thm.11.10 | 1/12 | 1/6 |
| Chamseddine-Connes-vS | 2013 | 1304.8050 | (134) | 1/12 | 1/6 |
| Devastato-Lizzi-Martinetti | 2014 | 1304.0415 | (2.14) | inherited | 1/6 |

### RG stability

beta_xi = (xi - 1/6) * [12 lambda + 2 Y_2 - 3/2 g'^2 - 9/2 g^2] / (16 pi^2)

At xi = 1/6: beta_xi = 0 EXACTLY (one-loop fixed point). If the
spectral cutoff Lambda is the UV boundary condition, xi(Lambda) = 1/6
is maintained at ALL scales — no drift to 1%, 10%, or any other value.

Confirmed by: Salvio-Strumia (1403.4226, Eq. (33)) with equivalent
fixed point (1+6 xi_H) = 0.

### Implications for SCT

1. **Scalar mode decouples:** Pi_s(z, xi=1/6) = 1 for all z.
   alpha_R = 0. 3c_1 + c_2 = 0.
2. **No scalar ghost:** the ghost-free region includes xi = 1/6
   trivially (no scalar poles at all).
3. **Modified potential simplifies:**
   V(r)/V_N = 1 - (4/3) exp(-m_2 r) (only spin-2 Yukawa).
4. **Scalaron absent:** m_0 -> infinity. No R^2 scalar mode.
   This sharpens INF-1: standard spectral action predicts no scalaron,
   not just a too-heavy scalaron.
5. **Higgs inflation incompatible:** xi ~ 5e4 (Bezrukov-Shaposhnikov,
   0710.3755 Eq. (13)) is ~3e5 times larger than the NCG prediction.
   Standard Higgs inflation requires extension beyond the plain
   Chamseddine-Connes spectral action.

### Numerical verification (all exact)

  xi = 1/6:    alpha_R = 0,        m_0/Lambda = infinity
  xi = 0:      alpha_R = 1/18,     m_0/Lambda = 2.4495 (= sqrt(6))
  xi = 0.4446: alpha_R = 0.1549,   m_0/Lambda = 1.4689
  xi = 1/12 (wrong convention):    m_0/Lambda = 4.899

### Falsifiable consequence

Detection of a propagating scalar gravitational mode (l=0 monopole
QNM in BH ringdown, or anomalous polarization in GW detectors) would
falsify the standard spectral action prediction xi = 1/6 and require
BSM extension of the spectral triple.

## 5. Success Criteria

- A determination of xi from the spectral triple structure, with
  explicit derivation from the finite Dirac operator D_F and the
  a_4 coefficient.
- If xi = 1/6: proof that the conformal fixed point is the unique
  consistent value in the NCG framework.
- If xi != 1/6: verification that the physical xi lies below xi_c
  (ghost-free region) and computation of the resulting scalar
  effective mass m_0(xi).
- Identification of an observable that discriminates between xi values
  beyond solar system and cosmological tests (possible candidate:
  scalar graviton mode in black hole ringdown, if detectable).
- Statement of whether xi is a prediction of the theory or a free
  parameter to be fixed by external input.

## 6. Suggested Directions

1. NCG derivation: compute xi from the spectral action on the
   Chamseddine-Connes triple (A_F = C + H + M_3(C), H_F = 96-dim)
   using the explicit a_4 coefficient. The calculation involves the
   endomorphism E of the finite Dirac operator and its coupling to
   the scalar curvature.

2. RG fixed-point argument: the SM beta function for xi has xi = 1/6
   as a UV fixed point. If the spectral cutoff Lambda is identified
   with the UV fixed-point scale, xi(Lambda) = 1/6 is the natural
   initial condition. This would make conformal coupling a prediction
   (not an assumption).

3. Graviton scattering: at one loop, the scalar graviton mode
   contributes to graviton-graviton scattering (MR-7). The amplitude
   depends on xi through Pi_s. Compute the xi-dependent correction
   to the tree-level amplitude and assess whether future graviton
   collider experiments (hypothetical) could measure it.

4. BH spectroscopy: the scalar graviton mode, if present (xi != 1/6),
   produces additional QNM frequencies. These scalar QNMs have
   different angular structure (l = 0 monopole) from the standard
   tensor modes. Detection of scalar QNMs in ringdown data would
   measure xi directly.

## 7. References

1. Chamseddine, A. H. and Connes, A. (2006). "Inner fluctuations of
   the spectral action." J. Geom. Phys. 57, 1. arXiv:0610241.
2. Salvio, A. and Strumia, A. (2014). "Agravity." JHEP 1406, 080.
   arXiv:1403.4226.
3. Devastato, A., Lizzi, F. and Martinetti, P. (2014). "Grand
   symmetry, spectral action, and the Higgs mass." JHEP 1401, 042.
   arXiv:1304.0415.
4. Alfyorov, D. "Solar system and laboratory tests of Spectral
   Causal Theory." DOI:10.5281/zenodo.19098100.
5. Alfyorov, D. "Nonlocal one-loop form factors from the spectral
   action." DOI:10.5281/zenodo.19098042.

## 8. Connections

- Related to **OP-17** (scalaron mass): the scalaron mass depends on
  xi through m_0 = Lambda / sqrt(6(xi - 1/6)^2). At xi = 1/6,
  m_0 -> infinity (no scalaron).
- Related to **OP-29** (TOV limits): the scalar mode modifies the
  TOV equations for xi != 1/6.
- Related to **OP-26** (QNM shifts): scalar QNMs are present only
  for xi != 1/6.
- Independent of **OP-01** (Gap G1): the scalar sector analysis is
  a flat-space property of the propagator.

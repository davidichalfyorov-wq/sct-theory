---
id: OP-30
title: "Running coupling constants c_1(mu) and c_2(mu)"
domain: [predictions, theory]
difficulty: medium
status: resolved
deep-research-tier: B
blocks: []
blocked-by: []
roadmap-tasks: [META-1]
papers: ["0805.2909", "0304222", "0901.2984", "1403.4226", "hep-th/0412249"]
date-opened: 2026-03-31
date-updated: 2026-04-01
date-resolved: 2026-04-01
resolved-by: literature-analysis + independent-numerical-verification
---

# OP-30: Running coupling constants c_1(mu) and c_2(mu)

## 1. Statement

Compute the explicit running of the higher-derivative gravitational
couplings c_1(mu) and c_2(mu) under renormalization group flow in
SCT. The spectral action fixes the initial conditions at mu = Lambda:
c_1(Lambda) = alpha_R(xi) - (2/3) alpha_C and c_2(Lambda) = 2 alpha_C
= 13/60. Determine the trajectories c_1(mu), c_2(mu) for mu < Lambda
and compare with asymptotic safety fixed-point values and CDT lattice
measurements.

## 2. Context

In the {R^2, R_{mu nu}^2} basis, the one-loop gravitational effective
action contains two independent couplings c_1 and c_2. The spectral
action principle fixes their values at the cutoff scale:

  c_2(Lambda) = 2 alpha_C = 13/60  (parameter-free),
  c_1(Lambda) = alpha_R(xi) - (2/3) alpha_C  (depends on xi).

Below the cutoff, these couplings run according to beta functions
determined by the matter content and gravitational loops. The one-loop
beta functions for c_1 and c_2 are known in the literature (Codello,
Percacci, Rachwal, 0805.2909):

  beta_{c_1} = -(1/(16 pi^2))(N_s/120 + N_D/20 + N_v/10),
  beta_{c_2} = -(1/(16 pi^2))(N_s/120 + N_D/20 + N_v/10).

These are the pure matter contributions. Graviton loop contributions
modify these beta functions at two loops and beyond.

The running trajectories c_1(mu), c_2(mu) provide a direct comparison
with the asymptotic safety program, which predicts UV fixed-point
values c_1*, c_2*. If SCT trajectories are attracted to the AS
fixed point as mu -> infinity, this would establish a connection
between the two programs. If they flow away, the two programs make
incompatible predictions.

## 3. Known Results

- Initial conditions at mu = Lambda are exactly fixed by the
  spectral action: alpha_C = 13/120, alpha_R(xi) = 2(xi - 1/6)^2.
- One-loop beta functions are known for the SM matter content
  (CPR, 0805.2909).
- The ratio c_1/c_2 = -1/3 + 120(xi - 1/6)^2/13 at mu = Lambda.
- At conformal coupling xi = 1/6: c_1/c_2 = -1/3, c_1 = -13/180.
- At minimal coupling xi = 0: c_1/c_2 = -1/3 + 40/(39*36),
  alpha_R(0) = 1/18.
- The cutoff function f enters only through the initial conditions,
  not through the beta functions (which are universal at one loop).

## 3b. Resolution (2026-04-01)

**VERDICT: RESOLVED. c_1/c_2 = -1/3 exactly preserved along matter-only one-loop trajectory. SCT aligns with perturbative AF branch (2.3% off), far from Reuter NGFP.**

### Universal matter one-loop beta functions (SM, xi = 1/6)

From CPR (0805.2909) eqs. (19)-(23) plus (A8),(A10),(A19):

  beta_{alpha_C} = (1/(4pi)^2)(N_s/120 + N_D/20 + N_v/10)
                 = 283/(120(4pi)^2) ~ 0.01493

  beta_{alpha_R} = (N_s/2)(xi - 1/6)^2/(4pi)^2 = 0  (at xi = 1/6)

In the {c_1, c_2} = {alpha_R - 2/3 alpha_C, 2 alpha_C} basis:

  beta_{c_2} = 283/(60(4pi)^2) ~ 0.02987
  beta_{c_1} = -283/(180(4pi)^2) ~ -0.00996
  beta_{c_1}/beta_{c_2} = -1/3  (EXACT)

Note: the beta coefficient 283/120 for SM corrects a previously
circulated estimate of 13/12. The correct computation:
4/120 + 22.5/20 + 12/10 = 1/30 + 9/8 + 6/5 = 283/120.

### Trajectory (matter-only, universal)

  c_2(t) = 13/60 + (283/(60(4pi)^2)) t
  c_1(t) = -13/180 - (283/(180(4pi)^2)) t
  c_1(t)/c_2(t) = -1/3  for all t

| mu/Lambda | t | c_1 | c_2 | c_1/c_2 |
|-----------|---|-----|-----|---------|
| 1 | 0 | -0.07222 | 0.21667 | -1/3 |
| 10^{-1} | -2.303 | -0.04930 | 0.14789 | -1/3 |
| 10^{-5} | -11.513 | 0.04240 | -0.12721 | -1/3 |
| 10^{-10} | -23.026 | 0.15703 | -0.47108 | -1/3 |
| 10^{-20} | -46.052 | 0.38628 | -1.15884 | -1/3 |

Zero crossing of c_2 at mu/Lambda ~ 7.07 x 10^{-4}.

### Comparison with asymptotic safety

| Reference point | c_1/c_2 | Distance from SCT |
|----------------|---------|-------------------|
| SCT (all scales) | -0.33333 | 0 |
| Perturbative AF, pure gravity (Shapiro, hep-th/0412249, Table 2) | -0.32571 | 2.3% |
| Perturbative AF, CPR+SM (0805.2909, eqs.(92)-(93)) | -0.32485 | 2.5% |
| BMS NGFP (0901.2984, eq.(12)) | -1.0873 | 226% |

Verdict: SCT initial condition c_1/c_2 = -1/3 is naturally compatible
with the perturbative asymptotically free branch of higher-derivative
gravity, but far from the Benedetti-Machado-Saueressig non-Gaussian
fixed point.

### Key structural insight

At xi = 1/6 (NCG prediction), beta_{alpha_R} = 0 identically.
Combined with beta_{c_1}/beta_{c_2} = -1/3, the ratio c_1/c_2
is frozen at -1/3 on the entire matter-only trajectory. This is a
consequence of scalar decoupling at conformal coupling: the R^2
sector neither runs nor contributes.

### CDT comparison

No direct lattice extraction of continuum c_1/c_2 from CDT exists
in the literature. CDT bare couplings (kappa_0, kappa_4, Delta) do
not map directly to continuum R^2/C^2 coefficients.

## 4. Failed Approaches

No explicit running trajectories had been computed prior to this
resolution.
The main ambiguity is in the graviton loop contributions: the one-loop
matter beta functions are universal, but the graviton self-coupling
contributions depend on the gauge fixing and field parametrisation.
In the background field method with harmonic gauge, these contributions
are known (Fradkin-Tseytlin), but the translation to the SCT nonlocal
propagator is not straightforward.

## 5. Success Criteria

- Explicit numerical trajectories c_1(mu), c_2(mu) from mu = Lambda
  down to mu = 0, using the one-loop matter beta functions and SM
  content.
- c_1/c_2(mu) trajectory plotted against the AS fixed-point value
  from Codello-Percacci-Rachwal and Benedetti-Machado-Saueressig.
- Comparison with CDT lattice measurements of c_1/c_2 (if available
  from the Laiho-Coumbe-Du-Bassler program).
- Statement of whether the SCT trajectory is consistent with the
  AS fixed point within the one-loop approximation.
- Analysis of the cutoff function dependence: how sensitive are the
  trajectories to the choice of f (exponential vs polynomial cutoffs)?

## 6. Suggested Directions

1. One-loop matter running: numerically integrate the one-loop RG
   equations using the SM matter beta functions (N_s = 4, N_D = 22.5,
   N_v = 12). This is a straightforward ODE integration from
   mu = Lambda to mu = 0.

2. Graviton loop corrections: add the Fradkin-Tseytlin graviton
   contributions to the beta functions. These introduce dependence
   on the gravitational coupling G(mu), requiring a coupled system
   of RG equations for (G, c_1, c_2).

3. Fixed-point analysis: determine whether the coupled system
   (G, c_1, c_2) has UV fixed points. If yes, check whether the
   SCT initial conditions lie on the UV critical surface (the
   set of trajectories attracted to the fixed point).

4. CDT comparison: extract effective c_1/c_2 values from CDT
   lattice simulations (Laiho et al., Ambjorn et al.) by fitting
   the graviton propagator on the lattice. Compare with the SCT
   RG trajectory at the corresponding lattice scale.

## 7. References

1. Codello, A., Percacci, R. and Rachwal, L. (2008). "The
   renormalization group and Weyl invariance." Ann. Phys. 324, 414.
   arXiv:0805.2909.
2. Fradkin, E. S. and Tseytlin, A. A. (1982). "Renormalizable
   asymptotically free quantum theory of gravity." Nucl. Phys. B
   201, 469.
3. Salvio, A. and Strumia, A. (2014). "Agravity." JHEP 1406, 080.
   arXiv:1403.4226.
4. Benedetti, D., Machado, P. F. and Saueressig, F. (2009).
   "Asymptotic safety in higher-derivative gravity." Mod. Phys. Lett.
   A 24, 2233. arXiv:0901.2984.
5. Alfyorov, D. "Nonlocal one-loop form factors from the spectral
   action." DOI:10.5281/zenodo.19098042.

## 8. Connections

- Related to **OP-31** (form factor comparison with AS): both
  compare SCT predictions with the asymptotic safety program.
- Related to **OP-33** (cross-program table): the running constants
  provide a key entry in the comparison table.
- Independent of **OP-01** (Gap G1): the running constants are
  properties of the flat-space effective action.
- The META-1 task includes running constant analysis as part of the
  broader parameter counting and falsifiability study.

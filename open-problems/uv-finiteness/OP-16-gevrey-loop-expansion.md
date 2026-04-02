---
id: OP-16
title: "Gevrey-1 property of the loop expansion"
domain: [uv-finiteness]
difficulty: hard
status: partial
deep-research-tier: B
blocks: []
blocked-by: []
roadmap-tasks: [MR-5, MR-6]
papers: ["hep-th/0306138", "Lipatov1977", "LeGuillou-ZinnJustin1990", "Beneke1999", "0812.3363", "1407.8036"]
date-opened: 2026-03-31
date-updated: 2026-04-02
progress: "LIKELY Gevrey-1. Convention corrected: (2L)! = Gevrey-2, not 1. Lipatov analysis: entire structure doesn't change index. No proof."
---

# OP-16: Gevrey-1 property of the loop expansion

## 1. Statement

Prove that the perturbative loop expansion of the SCT effective action
Gamma = sum_{L=0}^{infinity} alpha_C^L Gamma^{(L)} is Gevrey-1 (i.e.,
the L-loop coefficient |Gamma^{(L)}| grows at most as (2L)! times a
geometric factor). Alternatively, determine the actual Gevrey class of
the SCT loop expansion, or show that it belongs to no Gevrey class
(in which case the expansion is worse than factorial).

## 2. Context

The Gevrey class of a perturbative expansion determines the quality of
its asymptotic approximation and the applicability of resummation
techniques. A Gevrey-1 expansion has coefficients growing as ~ (2L)!,
which is the standard behavior in generic QFT (renormalon growth). For
Gevrey-s expansions (s > 1), the growth is ~ ((2L)!)^s, which is
harder to resum.

The MR-5 analysis of UV finiteness assumes that the SCT loop expansion
is Gevrey-1 by analogy with generic QFT. This assumption enters the
estimate of the optimal truncation order L_opt and the non-perturbative
(NP) ambiguity. Specifically:

- **Optimal truncation:** L_opt ~ 1/(2 epsilon) where
  epsilon = alpha_C / (16 pi^2) ~ 6.9e-4.
- **NP ambiguity:** delta_NP ~ exp(-1/(2 epsilon)) ~ exp(-720).
- **Truncation error at L = 2:** ~ epsilon^3 * 6! ~ 3.2e-10 * 720
  ~ 2.3e-7 (utterly negligible).

If the loop expansion is NOT Gevrey-1 (e.g., if it grows faster than
factorial), the NP ambiguity would be larger and L_opt smaller. In the
extreme case of super-factorial growth, the perturbative expansion
might not be asymptotic at all.

The MR-6 investigation studied the related but distinct question of
curvature expansion convergence. The curvature (Seeley-DeWitt)
expansion was proven to be Gevrey-1 with growth rate
|a_{2k}| ~ (2k)! / (2 pi)^{2k} and Borel radius R_B = pi^2. But the
curvature expansion and loop expansion are different expansions: the
curvature expansion is an expansion in R / Lambda^2 at fixed loop
order, while the loop expansion is an expansion in alpha_C at fixed
curvature. The Gevrey class of one does not determine the Gevrey class
of the other.

## 3. Known Results

- **Curvature expansion: Gevrey-1 (MR-6, CERTIFIED).** Growth rate
  |a_{2k}| ~ (2k)! / (2 pi)^{2k}. Borel radius R_B = pi^2 for the
  raw SD coefficients. Laplace representation verified to 168+ digits
  on S^4.

- **SCT non-perturbative: well-defined.** The full spectral action
  S = Tr(f(D^2/Lambda^2)) is defined non-perturbatively via the
  spectral zeta function. The form factors F_1(z), F_2(z) are entire
  functions of z = Box/Lambda^2 (NT-2, CERTIFIED). Therefore, the
  loop expansion is an asymptotic expansion of a well-defined function,
  and a Gevrey class should exist.

- **Generic QFT: Gevrey-1.** In standard QFT (phi^4, QED, QCD),
  the perturbative expansion is Gevrey-1 with growth controlled by
  renormalons and instantons. The growth is ~ n! * A^n * n^b for
  the n-th coefficient, with A related to the leading UV/IR renormalon.

- **Spectral moments: f_{2k} = (k-1)! (factorial growth).** The
  spectral moments of the exponential cutoff function psi(u) = e^{-u}
  grow factorially. This factorial growth feeds into the SD coefficients
  at each loop order, but the loop expansion coefficient at L loops
  involves a specific combination of f_{2L+2} and lower moments, not
  f_{2L+2} alone.

- **V3 Devil's Advocate assessment (MR-5): HIGH severity.** The
  Gevrey-1 assumption was flagged as unproven at HIGH severity
  (6/10). The V3 noted that SCT has non-standard UV behavior
  (entire form factors, no running coupling in the usual sense), and
  the standard renormalon argument for Gevrey-1 may not apply.

## 3b. Partial Resolution (2026-04-02)

**STATUS: LIKELY Gevrey-1. Convention corrected. No proof exists.**

### Convention correction (IMPORTANT)

The problem statement incorrectly defined "Gevrey-1 as (2L)! growth."
In standard Gevrey theory:
- **Gevrey-1:** |Gamma^(L)| ~ C^L Gamma(L+b) = C^L L! type
- **Gevrey-s:** |Gamma^(L)| ~ C^L (L!)^s
- **(2L)! ~ (L!)^2 growth is Gevrey-2, not Gevrey-1.**

Our MR-5 estimates (L_opt ~ 720, delta_NP ~ e^{-720}) are consistent
with STANDARD Gevrey-1 (L! growth with A ~ 1/2). If actually Gevrey-2
((2L)!), the estimates would be drastically different:
  L_opt ~ 13.5,  delta_NP ~ 10^{-12}  (much worse)

### Why Gevrey-1 is most likely

**1. Lipatov saddle-point argument.** The formal loop-counting parameter g
enters as e^{-S_spec/g} in the gauge-fixed path integral. Saddle-point
gives large-order coefficients ~ A^{-L} Gamma(L+b), which is standard
Gevrey-1. The Gevrey index changes to s > 1 only if the exponential has
non-standard g-dependence (e.g., e^{-A/g^rho} with rho > 1). The spectral
action Tr f(D^2/Lambda^2) has standard exponential structure.
(Lipatov 1977 eqs.(75)-(76); Le Guillou-Zinn-Justin eqs.(1.41)-(1.49).)

**2. Renormalon argument fails but benignly.** Standard renormalon
(Beneke eqs.(2.26)-(2.27)) requires log running. SCT Pi_TT → const
(no log running) → standard UV-renormalon mechanism absent. But this
REMOVES one factorial source, doesn't CREATE a new one. Net effect:
neutral for Gevrey index.

**3. Entire-function structure doesn't change Gevrey index.** The
entire form factors change the saddle action A and fluctuation
determinant, not the functional form Gamma(L+b). Modesto-Rachwal
(1407.8036) show that entire form factors can change UV class
(super-renormalizability), but only when inverse propagator grows
faster in UV. SCT's G ~ 1/k^2 doesn't qualify.

**4. Cross-pole terms controlled by CL bound.** If sum |R_n| < infinity
(our CL result), then for diagram with I internal lines:
  sum_{n_1,...,n_I} |prod R_{n_a}| = (sum |R_n|)^I = finite^I
This preserves L! growth (changes C, not Gevrey index).

### What's missing for a proof

- Explicit complex saddle of gauge-fixed spectral action on O(4) ansatz
- Fluctuation determinant at saddle → value of A
- No Gevrey-class proof exists even for Stelle gravity (!)

### Borel plane

Position of nearest singularity unknown. Most likely saddle/instanton-
type (at t = A), not renormalon-type. Ghost poles in momentum space
do NOT automatically become Borel singularities in loop-counting
parameter.

### Impact on MR-5

MR-5 estimates remain valid under standard Gevrey-1 assumption:
  L_opt ~ A/epsilon ~ 720 (for A ~ 1/2)
  delta_NP ~ e^{-A/epsilon} ~ e^{-720}
If Gevrey-2: L_opt ~ 13, delta_NP ~ 10^{-12} — radically different.
Convention distinction is critical.

## 4. Failed Approaches

1. **Renormalon analysis.** The standard renormalon argument uses the
   running coupling beta function to identify factorial growth from
   chain-of-bubble diagrams. In SCT, the graviton propagator is
   dressed by an entire function Pi_TT(z), and the UV behavior is
   G ~ -6/(83 k^2) (GR-like). The dressed propagator does not give
   rise to a standard running coupling, because Pi_TT saturates at
   a constant in the UV (unlike logarithmic running in QCD). Without
   a running coupling, the standard renormalon argument does not
   produce a definite Gevrey class.

2. **Instanton argument.** In generic QFT, instantons contribute
   non-perturbative effects ~ exp(-S_inst) with S_inst ~ 1/g^2, and
   the perturbative expansion around the instanton saddle point gives
   factorial growth. In SCT, the instanton action is the spectral
   action evaluated on an instanton background (e.g., Eguchi-Hanson,
   Taub-NUT). The spectral action on these backgrounds has been
   partially computed (FUND-LAT: exact sum on S^4 gives
   Delta = 41/15120 * Lambda^{-4}), but S^4 has C = 0, limiting
   the utility for the Weyl-sector growth rate. Non-conformally-flat
   instantons have not been studied.

3. **Large-order behavior via saddle-point approximation.** Attempted
   to estimate the L-loop coefficient by saddle-point evaluation of
   the path integral at large L. The saddle-point is the classical
   solution at coupling ~ 1/L (Lipatov method). For SCT, the
   classical solution is a spectral triple satisfying the SCT EOM,
   and the action at the saddle point involves the nonlocal form
   factors F_1, F_2. The saddle-point action has not been evaluated
   because the nonlocal EOM on a non-trivial background (OP-01/Gap G1)
   have not been solved.

## 5. Success Criteria

- A proof that the L-loop SCT coefficient |Gamma^{(L)}| <= C^L (2L)!
  for some constant C, establishing Gevrey-1. This would confirm the
  MR-5 optimal truncation estimates.
- Or: determination of the actual Gevrey class s, with explicit
  growth rate |Gamma^{(L)}| ~ ((2L)!)^s * A^L.
- Or: a proof that the growth is strictly super-factorial (faster
  than (2L)! for any s), which would invalidate the Borel resummation
  strategy and require alternative resummation methods.

## 6. Suggested Directions

1. **Lipatov saddle-point for spectral action.** Apply the Lipatov
   large-order method to the SCT path integral. The saddle-point
   configuration at coupling g ~ 1/L is a classical solution of the
   spectral action EOM with rescaled curvature. Even if the exact
   saddle-point action cannot be computed (due to Gap G1), bounds on
   the action would give bounds on the Gevrey class.

2. **Diagram counting.** At L loops, the number of Feynman diagrams
   in the background field expansion grows as ~ (2L)! / L (from
   Wick contractions). Each diagram is bounded by ~ C^L (from power
   counting). If the cancellations between diagrams are controlled
   (as they are in gauge theories via Ward identities), the total
   L-loop contribution grows at most as ~ (2L)! * C^L, which is
   Gevrey-1. Verify that the SCT Ward identities (diffeomorphism
   invariance) provide sufficient control.

3. **Comparison with Stelle gravity.** Stelle gravity (R + R^2 + C^2
   with finite polynomial propagator) has a well-defined Gevrey class
   because it is a renormalizable theory with known beta functions.
   Compute the Gevrey class of Stelle gravity explicitly (it should
   be Gevrey-1 from the standard renormalon analysis) and then study
   how the replacement of the polynomial propagator by the SCT entire
   function modifies the growth rate. The key question is whether the
   infinite-pole structure changes the Gevrey class.

4. **Borel plane analysis.** Compute the Borel transform of the first
   few loop coefficients (L = 1, 2) and look for singularities in the
   Borel plane. The location of the nearest singularity determines the
   radius of convergence of the Borel series and hence the Gevrey
   class. For SCT at L = 1, the Borel transform is known (related to
   the SD coefficients). At L = 2, the Borel transform requires the
   explicit two-loop coefficient.

5. **Numerical estimation via lattice spectral action.** Compute the
   spectral action on a sequence of lattices of increasing size and
   extract the loop coefficients from finite-size scaling. The growth
   rate of these coefficients with L gives a numerical estimate of
   the Gevrey class.

## 7. References

1. Vassilevich, D.V. "Heat kernel expansion: user's manual,"
   Phys. Rep. 388 (2003) 279, hep-th/0306138.
2. Lipatov, L.N. "Divergence of the perturbation-theory series and
   the quasi-classical theory," Sov. Phys. JETP 45 (1977) 216.
3. Le Guillou, J.C. and Zinn-Justin, J. "Large-order behaviour of
   perturbation theory," North-Holland, 1990.
4. Brezin, E., Le Guillou, J.C. and Zinn-Justin, J. "Perturbation
   theory at large order. I. The phi^{2N} interaction,"
   Phys. Rev. D 15 (1977) 1544.
5. Alfyorov, D. and Shnyukov, I. "Auxiliary boundary data and the
   failure of intrinsic coherence in sprinkled causal sets,"
   DOI: pending.

## 8. Connections

- **Roadmap: MR-5** (all-orders finiteness). MR-5 assumes Gevrey-1
  for the optimal truncation and NP ambiguity estimates. If the loop
  expansion is not Gevrey-1, the MR-5 survival probability assessment
  (62-72%) would need revision.
- **Roadmap: MR-6** (curvature expansion convergence). MR-6 established
  Gevrey-1 for the curvature expansion, not the loop expansion. OP-16
  is the loop-expansion analogue.
- **Related to OP-13** (three-loop): Even if the loop expansion is
  Gevrey-1, the three-loop overdetermination is a structural problem
  (number of counterterms vs parameters), not a growth-rate problem.
  Gevrey-1 would mean the perturbative approximation is good through
  L = 2, confirming that the three-loop divergence is practically
  irrelevant.
- **Related to OP-14** (hidden principle): If a hidden principle
  restores D = 0 at all orders, the Gevrey class of the expansion
  becomes important for resummation of the resulting finite series
  (which may still be asymptotic due to renormalon effects in the
  matter sector).
- **Independent of all unitarity problems** (OP-07 through OP-12):
  The Gevrey class is a property of the perturbative expansion
  coefficients and does not depend on the unitarity prescription
  used for the ghost poles.

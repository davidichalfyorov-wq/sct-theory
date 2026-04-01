---
id: OP-20
title: "De Sitter conjecture: does the SCT scalar potential satisfy or violate the refined Swampland dS conjecture?"
domain: [cosmology, formal-theory]
difficulty: medium
status: resolved
deep-research-tier: A
blocks: []
blocked-by: []
roadmap-tasks: [INF-1]
papers: ["1806.08362", "1810.05506", "1811.08889", "1807.05193"]
date-opened: 2026-03-31
date-updated: 2026-03-31
date-resolved: 2026-03-31
resolved-by: literature-analysis + independent-numerical-verification
---

# OP-20: De Sitter conjecture check

## 1. Statement

The refined de Sitter Swampland conjecture (Ooguri, Vafa et al., 2018)
states that for any scalar potential V(phi) in a consistent quantum
gravity theory, either

  |grad V| / V >= c_1 / M_Pl     (gradient condition)

or

  min(nabla_i nabla_j V) <= -c_2 / M_Pl^2 * V   (curvature condition)

where c_1, c_2 are positive O(1) constants. Determine whether the
scalaron potential derived from the R^2 sector of the SCT spectral
action satisfies or violates these conditions, and what this implies
for the consistency of SCT with Swampland constraints.

## 2. Context

The Swampland programme in string theory classifies effective field
theories that can or cannot be UV-completed into a quantum gravity
framework. The de Sitter conjecture constrains the scalar potential
landscape: it claims that meta-stable de Sitter vacua are inconsistent
with quantum gravity, and the potential must either be steep (gradient
condition) or have a tachyonic direction (curvature condition).

SCT produces a de Sitter solution (proven exact in NT-4c) and a
scalaron with mass M_0 ~ 15.39 M_Pl (at minimal coupling). The
Starobinsky scalaron potential in the Einstein frame is

  V(phi) = (3/4) M_0^2 M_Pl^2 (1 - exp(-sqrt(2/3) phi / M_Pl))^2

This potential has a de Sitter plateau at large phi, a minimum at
phi = 0, and is everywhere positive. The gradient |V'|/V and the
Hessian V''/V at specific field values determine whether the conjecture
is satisfied.

This problem has not been investigated at all within SCT. It is a
straightforward computation once the Einstein-frame potential is
identified (which it is), but the implications for the theory's
consistency or inconsistency with Swampland criteria are non-trivial.

## 3. Known Results

- **De Sitter is an exact solution (NT-4c):** the nonlinear SCT field
  equations admit de Sitter with arbitrary cosmological constant.
  Perturbations are stable.

- **Scalaron potential (Starobinsky form):** V(phi) is known exactly
  in the Einstein frame. It depends on M_0 and M_Pl only.

- **At the minimum (phi = 0):** V(0) = 0, V'(0) = 0, V''(0) =
  2 M_0^2 / 3. The gradient condition is trivially satisfied (both
  sides are zero). The curvature condition gives V''/V, which is
  ill-defined at V = 0.

- **On the plateau (phi >> M_Pl):** V -> 3 M_0^2 M_Pl^2 / 4,
  |V'|/V -> 0 exponentially. The gradient condition is violated.
  The curvature condition gives V''/V -> 0. Both conditions fail
  on the plateau.

- **No SCT-specific analysis has been performed.** The question is
  whether SCT's UV completion modifies the Swampland reasoning, or
  whether SCT violates the conjecture and is therefore inconsistent
  with the Swampland programme.

## 3b. Resolution (2026-03-31)

**VERDICT: VIOLATED** for standard refined dS conjecture with c_1 ~ 1, c_2 ~ 1.

### Analytical results

With y = exp(-sqrt(2/3) phi/M_Pl), the key ratios are M_0-independent:

  g(phi) = M_Pl |V'|/V = 2 sqrt(2/3) y/(1-y)
  eta(phi) = M_Pl^2 V''/V = (4/3) y(2y-1)/(1-y)^2

Key values:
- phi_inflection = M_Pl sqrt(3/2) ln(2) = 0.84893 M_Pl (V'' changes sign)
- eta_min = -1/3 at phi = M_Pl sqrt(3/2) ln(3) = 1.34552 M_Pl
- phi_*(c_1=1) = M_Pl sqrt(3/2) ln(1 + 2 sqrt(2/3)) = 1.18570 M_Pl

### Numerical table (independently verified)

  phi/M_Pl  |  |V'|/V   |  V''/V
  0.1       |  19.1946  |  168.5443
  0.5       |  3.2389   |  2.6007
  1.0       |  1.2934   |  -0.2196
  2.0       |  0.3964   |  -0.2451
  5.0       |  0.0280   |  -0.0225
  10.0      |  0.0005   |  -0.0004

### Detailed verdict

- **Gradient condition** (|nabla V| >= c_1 V/M_Pl): satisfied only for
  phi < phi_*(c_1). For c_1 = 1: phi < 1.186 M_Pl. Violated on plateau.
- **Curvature condition** (min nabla_i nabla_j V <= -c_2 V/M_Pl^2):
  HARD CEILING eta_min = -1/3. Cannot be satisfied for c_2 > 1/3.
  Never satisfied for c_2 = 1.
- **On the plateau** (phi >> M_Pl): both g -> 0 and eta -> 0^-
  exponentially. Both conditions fail.
- **Key insight:** M_0 cancels in both ratios. The Swampland check
  depends only on the Starobinsky potential shape, not on the mass.

### Implications

1. Standard Starobinsky inflation (including SCT scalaron) is in tension
   with refined dS conjecture at c_1 ~ 1, c_2 ~ 1.
2. Nonlocal form factor corrections could in principle modify V(phi)
   at phi ~ Lambda, but would need to change g by factor ~36 at
   phi = 5 M_Pl to "save" the gradient condition — this is a regime
   change, not a correction.
3. SCT may serve as a counterexample to the universality of Swampland
   constraints outside the string landscape (String Lamppost Principle,
   arXiv:2110.10157).
4. Alternative refined dS formulations exist (Andriot-Roupec,
   arXiv:1811.08889) with combined epsilon_V + eta_V criteria that
   can accommodate concave inflationary potentials.

### Falsifiable consequence

If refined dS with c_1 >= 0.5 is universal, then any Starobinsky-type
plateau inflation must be modified at phi ~ O(M_Pl) by new degrees of
freedom or steepened potential. Confirmation of pure Starobinsky
(low r ~ 3.5e-3, standard n_s) by LiteBIRD/CMB-S4 would increase
tension with "hard" Swampland.

## 4. Failed Approaches

No investigation has been attempted prior to this resolution.

## 5. Success Criteria

- Compute |V'|/V and V''/V for the SCT scalaron potential at all
  field values phi in [0, 10 M_Pl].
- Determine whether the refined dS conjecture is satisfied for any
  O(1) values of c_1, c_2.
- If violated: assess whether this constitutes evidence against SCT's
  UV-completeness, or whether SCT (being a specific framework, not a
  string compactification) is not subject to Swampland constraints.
- If satisfied: identify the mechanism by which the spectral action
  avoids meta-stable de Sitter despite having an exact dS solution.

## 6. Suggested Directions

1. **Direct computation.** Evaluate |V'|/V and min(V'')/V as functions
   of phi for the Starobinsky potential with M_0 = 15.39 M_Pl. Plot
   the results and compare with O(1) thresholds c_1 = 1, c_2 = 1.

2. **Nonlocal form factor corrections.** The full SCT potential includes
   nonlocal corrections from the form factor F_2(Box/Lambda^2). These
   modify V(phi) at field values phi ~ Lambda, potentially changing the
   gradient and curvature conditions near the UV scale. Compute the
   corrected potential in the regime phi ~ M_Pl.

3. **Multi-field analysis.** The SCT effective action contains both the
   scalaron (from R^2) and the Higgs field (from the finite spectral
   triple). The Swampland conjecture applies to the full multi-field
   potential V(phi, H). Compute the gradient in the (phi, H) plane.

4. **SCT vs Swampland philosophy.** The Swampland programme assumes
   that consistent quantum gravity theories must be UV-completable into
   string theory. SCT is an independent UV-completion attempt based on
   noncommutative geometry, not string theory. Investigate whether the
   Swampland conjectures are expected to hold outside the string
   landscape, and whether SCT provides a counterexample.

## 7. References

1. Obied, G., Ooguri, H., Spodyneiko, L. and Vafa, C. (2018). "De
   Sitter Space and the Swampland." arXiv:1806.08362.
2. Ooguri, H., Palti, E., Shiu, G. and Vafa, C. (2019). "Distance
   and de Sitter conjectures on the Swampland." Phys. Lett. B 788, 180.
   arXiv:1810.05506.
3. Garg, S. K. and Krishnan, C. (2019). "Bounds on slow roll and the
   de Sitter Swampland." JHEP 11, 075. arXiv:1807.05193.
4. Alfyorov, D. (2026). "Nonlinear field equations and FLRW cosmology
   from the spectral action." DOI:10.5281/zenodo.19098027.

## 8. Connections

- **OP-17 (scalaron mass):** the mass M_0 enters the potential and
  affects the gradient condition quantitatively.
- **OP-02 (Postulate 5):** if V3 (path integral) is the correct
  dynamical principle, tunnelling between dS vacua may be relevant.
- **OP-06 (UV-completeness):** the Swampland programme specifically
  constrains UV-complete theories. If SCT fails UV-completeness
  (OP-06 open), the dS conjecture may not apply.
- Independent of Gap G1 (OP-01): the scalar potential involves only
  the R^2 sector, not the Weyl sector.

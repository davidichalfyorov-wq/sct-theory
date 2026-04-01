---
id: OP-10
title: "D^2-quantization versus metric quantization equivalence"
domain: [unitarity, uv-finiteness]
difficulty: very-hard
status: open
deep-research-tier: D
blocks: [OP-06]
blocked-by: []
roadmap-tasks: [MR-2, MR-5, MR-7]
papers: []
date-opened: 2026-03-31
date-updated: 2026-03-31
---

# OP-10: D^2-quantization versus metric quantization equivalence

## 1. Statement

Determine whether D^2-quantization (where the path integral is taken
over the space of Dirac operators D, with observables expressed as
spectral invariants of D^2) and metric quantization (where the path
integral is taken over metrics g_{mu nu}) yield the same S-matrix
on-shell at all loop orders. If they differ, characterize the first
loop order and specific amplitude where the difference appears.

## 2. Context

The CHIRAL-Q program proves UV finiteness within D^2-quantization:
the chirality identity {D, gamma_5} = 0 forces counterterms to be
block-diagonal, giving D = 0 at every loop order (conditional on
BV-3,4, see OP-09). If D^2-quantization is physically equivalent to
metric quantization, this would establish SCT as UV-finite.

The difficulty is that the space of Dirac operators D (self-adjoint,
compact resolvent, satisfying the spectral-triple axioms) is strictly
larger than the space of metrics. A metric g determines D uniquely
(via the Lichnerowicz formula D^2 = -(Box + E)), but the converse
requires the Connes reconstruction theorem, which works at the
classical level but may fail at the quantum level where off-shell
and gauge-fixing contributions matter.

At tree level, the equivalence is established: the MR-7 analysis
confirmed that graviton scattering amplitudes computed from the
spectral action (using D^2 as the fundamental variable) agree with
those computed in the metric formulation. The tree-level S-matrix
is the same.

At one loop, the effective action in both formulations gives the
same result (alpha_C = 13/120, alpha_R = 2(xi-1/6)^2) because the
one-loop determinant is a spectral invariant of D^2 regardless of
whether we integrate over D or g. But at two loops and beyond, the
propagators, vertices, and ghost structure may differ because the
D^2-space has additional directions not present in the metric space.

## 3. Known Results

- **Tree level: EQUIVALENT** (MR-7 CERTIFIED). Tree SCT amplitudes
  agree with the metric formulation.
- **One loop: EQUIVALENT.** The one-loop determinant det(D^2) is
  identical in both formulations (same operator, same background).
- **D^2-space is larger than metric space.** D^2 contains components
  that do not correspond to any metric perturbation. In the
  Lichnerowicz formula D^2 = -(Box + E), the endomorphism E = -R/4
  (for Dirac) is determined by the metric; but in D^2-space, E can
  vary independently of Box. These extra modes exist off-shell.
- **On-shell tree equivalence:** At the on-shell level, the extra
  D^2 modes decouple because they are pure gauge (Connes
  reconstruction). But at loop level, off-shell modes circulate in
  loops and contribute to amplitudes.
- **CHIRAL-Q V3 attack (DEVASTATING):** The equivalence question was
  rated severity 10/10 by the V3 Devil's Advocate. It is the single
  most dangerous open question in the CHIRAL-Q program.

## 4. Failed Approaches

1. **Faddeev-Popov gauge fixing in D^2-space.** Attempted to write
   the D^2 path integral as a gauge-fixed integral over metrics plus
   gauge-fixing ghosts. The difficulty is that the gauge symmetry in
   D^2-space is not just diffeomorphisms: there are additional
   symmetries (inner automorphisms of the algebra A, unitary gauge
   transformations) that have no metric analogue. The Faddeev-Popov
   determinant for these extra symmetries is not computable by
   standard methods because the gauge group is infinite-dimensional
   and its orbit structure in D^2-space is not known.

2. **Background field method comparison.** Computed the background
   field effective action in both formulations at two loops. In the
   metric formulation, the two-loop counterterm is the a_6 Seeley-
   DeWitt coefficient. In D^2-quantization, the two-loop counterterm
   is also a_6 but with different combinatorial prefactors from the
   D^2 propagator and vertices. The comparison requires evaluating
   specific tensor contractions in both schemes, which has not been
   completed.

3. **Modesto-Calcagni analogy.** The Modesto-Calcagni nonlocal
   gravity uses exponential form factors exp(-Box/Lambda^2) and
   quantizes in the metric formulation. Comparing their results with
   SCT would test equivalence indirectly, but the form factors are
   different (SCT uses phi(x), not exp(-x)) and the comparison is
   inconclusive.

## 5. Success Criteria

- An explicit computation of a specific two-loop or three-loop
  amplitude in both D^2-quantization and metric quantization, showing
  agreement or disagreement.
- If equivalent: a structural argument (e.g., a change-of-variables
  theorem in the path integral) proving equivalence at all orders.
- If inequivalent: identification of the physical observable that
  distinguishes the two theories, and assessment of whether
  D^2-quantization produces a viable theory on its own.

## 6. Suggested Directions

1. **Two-loop graviton self-energy.** Compute the two-loop
   correction to the graviton self-energy in both formulations on
   flat background. In metric quantization, this requires the a_6
   coefficient plus two-loop vertex corrections. In D^2-quantization,
   the calculation uses the D^2-propagator and D^2-vertices. Compare
   the on-shell results.

2. **Change of variables in the path integral.** Write
   Z_D2 = integral dD exp(-S[D]) and attempt the substitution
   D -> g (using D^2 = -(Box + E)). The Jacobian of this substitution
   (the superdeterminant Sdet(delta D^2 / delta g)) is exactly BV-3.
   If BV-3 holds (OP-09), the substitution gives Z_D2 = Z_metric
   times a computable Jacobian factor that can be absorbed into the
   measure.

3. **Anomaly matching.** If the two formulations are equivalent, they
   must have the same gravitational anomalies (trace anomaly, chiral
   anomaly). Compute the trace anomaly coefficient in D^2-quantization
   (straightforward from the spectral action) and in metric
   quantization (standard DeWitt result) and compare.

4. **Lattice Monte Carlo.** Discretize both formulations on a finite
   lattice (or random geometry). Compare partition functions and
   correlators numerically. This would give non-perturbative evidence
   for or against equivalence.

## 7. References

1. Connes, A. "On the spectral characterization of manifolds,"
   J. Noncommut. Geom. 7 (2013) 1, arXiv:0810.2088.
2. Modesto, L. and Rachwal, L. "Nonlocal quantum gravity: A review,"
   Int. J. Mod. Phys. D 26 (2017) 1730020.
3. Calcagni, G. and Modesto, L. "Nonlocal quantum gravity and
   M-theory," Phys. Rev. D 91 (2015) 124059, arXiv:1404.2137.
4. Alfyorov, D. "Chirality of the Seeley-DeWitt coefficients and
   UV finiteness from D^2-quantization," DOI:10.5281/zenodo.19118075.

## 8. Connections

- **Blocks OP-06** (UV-completeness): If D^2 and metric quantization
  are equivalent, the CHIRAL-Q UV finiteness result (D = 0 at all
  orders) would apply to the physical theory.
- **Related to OP-09** (BV axioms): The Jacobian of the D^2 -> metric
  field redefinition IS BV-3. Proving BV-3 is a necessary (but not
  sufficient) condition for equivalence.
- **Related to OP-13** (three-loop overdetermination): If the two
  formulations are inequivalent, the three-loop problem (3 quartic
  Weyl invariants vs 1 parameter) applies only to metric quantization,
  and D^2-quantization may avoid it entirely.
- **Impact assessment:** If D^2 != metric at loop level, then:
  (a) D^2-quantization defines a new UV-finite theory reducing to
  GR+SM at low energies; (b) the fakeon unitarity problem (OP-07)
  becomes irrelevant (D^2-quantization has no ghosts by chirality);
  (c) SCT becomes a family of two theories sharing the same classical
  limit but differing quantum mechanically.

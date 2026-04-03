---
id: OP-13
title: "Three-loop 3:1 quartic Weyl overdetermination"
domain: [uv-finiteness]
difficulty: very-hard
status: partial
deep-research-tier: A
blocks: [OP-06]
blocked-by: []
roadmap-tasks: [MR-5]
papers: ["1607.08767", "1011.3474", "gr-qc/0407002", "hep-th/0405228", "hep-th/9708152", "1809.02944"]
date-opened: 2026-03-31
date-updated: 2026-04-03
progress: "STRUCTURAL RESOLUTION EXCLUDED. Quartic Weyl genuinely 2D in d=4 (Cayley-Hamilton proof). Sole remaining route: numerical coincidence c1/c3 for Dirac a_8 (uncomputed)."
---

# OP-13: Three-loop 3:1 quartic Weyl overdetermination

## 1. Statement

The three-loop SCT counterterm involves P(4) = 3 independent parity-
even quartic Weyl invariants, but the spectral function psi provides
only 1 adjustable parameter (delta f_8) at this loop order. The
Cayley-Hamilton identity reduces the independent set from 3 to 2, but
a 2:1 overdetermination remains. Find a structural principle that
reduces 2 independent quartic Weyl counterterms to 1 absorbable
combination, or prove that no such principle exists.

## 2. Context

UV finiteness in the spectral action framework relies on the
absorption mechanism: at each loop order L, the L-loop divergence is
a polynomial in curvature invariants of mass dimension 2L+2, and the
spectral function deformation delta(psi) provides spectral moments
delta f_{2L+2} that can absorb these divergences.

At L = 1: 1 invariant (C^2, modulo R^2 which is absorbed separately),
1 parameter (delta f_4) -> 1:1, absorption works.

At L = 2: 1 effective invariant (CCC, after on-shell reduction),
1 parameter (delta f_6) -> 1:1, absorption works.

At L = 3: the counterterm is constructed from the a_8 Seeley-DeWitt
coefficient, which contains quartic curvature invariants. The Molien
series for SO(4) acting on the Riemann tensor gives P(4) = 3 parity-
even quartic Weyl invariants:

  K_1 = (C^2)^2 = (C_{abcd} C^{abcd})^2
  K_2 = C^4_{box} = C_{abcd} C^{cdef} C_{efgh} C^{ghab}
  K_3 = (*CC)^2 = (C_{abcd} *C^{abcd})^2

The Cayley-Hamilton identity in d = 4 relates K_2 to K_1 and K_3:

  C^4_{chain} = (1/4)[(C^2)^2 + (*CC)^2]

(where the chain contraction is a specific trace of four Weyl tensors).
This reduces the independent set from 3 to 2, but the spectral function
still provides only 1 parameter delta f_8.

The 2:1 overdetermination means that D = 0 at L = 3 requires a specific
ratio between the coefficients of K_1 and K_3 in the three-loop
counterterm, and this ratio is NOT guaranteed by any known symmetry.

## 3. Known Results

- **P(4) = 3** from the full SO(4) Molien series (FUND-FK3, corrected
  from MR-5b's initial count of P(4) = 2 which used single-SU(2)).

- **CH reduces 3 -> 2:** The Cayley-Hamilton identity gives one relation
  among K_1, K_2, K_3, leaving 2 independent invariants.

- **Chirality theorem:** Tr(Omega^4) gives chain-contraction ratio
  (C^2)^2 : (*CC)^2 = 1:1 (exact). But [Tr(Omega^2)]^2 contains pq
  cross-terms (C^+ C^- mixing), so the full a_8 does NOT have ratio 1:1.

- **FUND-SYM investigation:** Checked 7 candidate reduction principles:
  (i) total derivative identities, (ii) Bianchi symmetries,
  (iii) dimensional regularization simplifications, (iv) Weyl tensor
  algebraic identities, (v) spectral zeta function constraints,
  (vi) conformal invariance constraints, (vii) topological identities
  (Euler density). None reduces 2 -> 1.

- **Higher loops grow worse:** P(5) = 2 quintic, P(6) = 7 sextic,
  P(7) = 5 septic invariants. The number of counterterms grows
  monotonically while the spectral function provides 1 parameter per
  loop order.

- **Practical suppression:** The three-loop correction is of order
  alpha_C^3 / (16 pi^2)^3 = 3.2e-10. Even at the Planck scale, the
  perturbative expansion parameter epsilon = (Lambda/M_Pl)^2 / (8 pi^2)
  ~ 0.013, so epsilon^3 ~ 2e-6. The three-loop correction is
  observationally irrelevant for any foreseeable experiment.

## 4. Failed Approaches

1. **Spectral renormalizability.** If the graviton field h_{mu nu}
   admitted a chiral decomposition under SU(2)_L x SU(2)_R (the
   self-dual/anti-self-dual split), chiral selection rules might
   forbid the pq cross-terms and reduce 2 -> 1. But h_{mu nu}
   transforms as (3,3) -- an irreducible representation with no
   chiral decomposition. The dressed propagator Pi_TT is chirality-
   blind (a scalar function of momentum), connecting delta C^+ to
   delta C^-. Cross-fraction: d^2 Gamma_2 / dp dq = 5.3e-3
   (numerically confirmed, ~80% of total). Spectral renormalizability
   is DISPROVEN.

2. **Modular structure (Bianchi IX).** Fan-Fathizadeh-Marcolli showed
   that the spectral action on Bianchi IX geometries has modular
   symmetry. If this modular structure persists for the full spectral
   action on general 4-manifolds, it might provide additional
   constraints on the quartic Weyl coefficients. However, Bianchi IX
   geometries are conformally flat (C = 0), so the Weyl invariants
   K_1, K_2, K_3 all vanish identically on these backgrounds. The
   modular structure cannot constrain the Weyl sector.

3. **Asymptotic safety crossing.** If SCT lies on an asymptotic
   safety fixed point, the beta functions would vanish non-
   perturbatively, potentially resolving the perturbative over-
   determination. However, the FRG analysis (FUND-FRG) showed that
   alpha_C = 13/120 is 5-20x larger than typical AS fixed-point
   values for the C^2 coefficient. The quantitative mismatch suggests
   that SCT does not sit on a known AS fixed point.

4. **Absorption by multiple spectral functions.** If the spectral
   action used two independent spectral functions (e.g., f_1 for the
   Weyl sector and f_2 for the Ricci sector), there would be
   2 parameters at L = 3, matching the 2 counterterms. But SCT
   postulate 4 (observables are spectral invariants of D) requires
   a single spectral function, and introducing a second one would
   violate the uniqueness of the spectral action.

## 5. Success Criteria

- A proof that the two independent quartic Weyl counterterms at L = 3
  always appear in a ratio that is absorbable by a single delta f_8.
  This would require a new identity or structural principle.
- Or: an explicit computation of the a_8 coefficient in the spectral
  action and verification that the two-invariant coefficient ratio
  happens to be compatible with absorption (this would be a numerical
  coincidence, not a structural resolution).
- Or: a rigorous proof that no structural reduction from 2 to 1 is
  possible, establishing SCT as an effective framework through L = 2.

## 6. Suggested Directions

1. **Explicit a_8 computation.** Compute the full a_8 Seeley-DeWitt
   coefficient for the squared Dirac operator on a general 4-manifold.
   This is a massive computation (comparable to the Goroff-Sagnotti
   two-loop calculation for GR) but is in principle doable with
   TFORM and modern algebraic methods. The key output would be the
   specific ratio of K_1 and K_3 coefficients.

2. **Spectral zeta function constraints.** The spectral action
   S = Tr(f(D^2/Lambda^2)) defines a zeta-function regularized
   determinant. Study whether the zeta-function regularization imposes
   relations between the quartic Weyl coefficients that are not visible
   in the heat-kernel expansion. The zeta function at s = -2 is related
   to a_8 but may have additional analytic structure.

3. **Non-commutative geometry approach.** In the NCG framework, the
   spectral action is defined on the product geometry M x F where F
   is the finite spectral triple encoding the Standard Model. The
   finite geometry may impose selection rules on the quartic curvature
   invariants through the structure of the algebra A_F. Study whether
   the Pati-Salam or SO(10) extensions of the finite geometry provide
   additional constraints.

4. **Causal set discretization.** Compute the analogues of K_1, K_2,
   K_3 on a causal set discretization and study whether the discrete
   spectral action generates only 1 combination.

## 7. References

1. Gilkey, P.B. "Invariance theory, the heat equation, and the
   Atiyah-Singer index theorem," 2nd ed., CRC Press, 1995.
2. Vassilevich, D.V. "Heat kernel expansion: user's manual,"
   Phys. Rep. 388 (2003) 279, hep-th/0306138.
3. Avramidi, I.G. "Heat kernel approach in quantum field theory,"
   Nucl. Phys. Proc. Suppl. 104 (2002) 3, math-ph/0107018.
4. Fan, W., Fathizadeh, F. and Marcolli, M. "Modular forms in the
   spectral action of Bianchi IX gravitational instantons,"
   arXiv:1607.08767.
5. Fulling, S.A. et al. "Normal forms for tensor polynomials: I.
   The Riemann tensor," Class. Quant. Grav. 9 (1992) 1151.
6. van de Ven, A.E.M. "Two-loop quantum gravity,"
   Nucl. Phys. B 378 (1992) 309, arXiv:1011.3474.

## 8. Connections

- **Blocks OP-06** (UV-completeness): This is the primary obstruction
  to all-orders UV finiteness. Resolving OP-13 in the positive
  direction (finding a structural reduction) would be a major
  breakthrough. Resolving it negatively (proving no reduction exists)
  would definitively classify SCT as an effective framework.
- **Related to OP-09** (BV axioms): BV-3 at L = 3 requires the
  three-loop counterterm to be a spectral invariant. If the quartic
  Weyl counterterms are not spectral invariants (because they cannot
  be absorbed by delta f_8), then BV-3 fails at L = 3.
- **Related to OP-14** (hidden principle): OP-14 is the constructive
  version of OP-13 -- it asks what the structural principle would be,
  rather than whether it exists.
- **Related to OP-15** (a_6 tensor match): The two-loop case (L = 2)
  was resolved by showing CCC is absorbable. OP-15 asks for the
  explicit tensor-level verification, which serves as a template
  for the three-loop analysis.

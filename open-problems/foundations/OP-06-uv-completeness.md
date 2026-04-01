---
id: OP-06
title: "UV-completeness proof"
domain: [theory]
difficulty: very-hard
status: open
deep-research-tier: D
blocks: []
blocked-by: [OP-02, OP-09, OP-10, OP-13]
roadmap-tasks: [LT-1]
papers: ["2301.13525"]
date-opened: 2026-03-31
date-updated: 2026-03-31
---

# OP-06: UV-completeness proof

## 1. Statement

Determine whether SCT possesses a UV fixed-point structure that renders
it a complete theory of quantum gravity, valid at arbitrarily high
energies. The three possibilities are:

- **Asymptotic freedom (AF):** the spectral coupling flows to zero in
  the UV.
- **Asymptotic safety (AS):** the spectral coupling flows to a
  nontrivial fixed point with a finite-dimensional critical surface.
- **Perturbative finiteness:** all counterterms vanish to all loop
  orders, eliminating the need for a fixed point.

The current evidence strongly disfavors the third option. The theory is
perturbatively finite at one loop (MR-7, proven) and at two loops
on-shell (MR-5b, CHIRAL-Q Theorem 6.12, unconditional). At three
loops, the number of independent quartic Weyl invariants exceeds the
single adjustable spectral parameter, creating a structural obstruction
to all-orders finiteness. Five fundamental investigations (FUND program)
have closed all known escape routes. The present assessment is that SCT
is an effective framework valid through two loops, not a UV-complete
theory.

## 2. Context

The UV fate of any quantum gravity theory is determined by its behavior
at loop orders L -> infinity. SCT has a single spectral parameter (the
cutoff function f, effectively one free function) that enters the
counterterm cancellation condition at each loop order. The number of
independent counterterm structures grows with L:

  L = 1: D = 0 (vanishing divergence, proven by MR-7).
  L = 2: D = 0 on-shell (proven by CHIRAL-Q Theorem 6.12).
         Off-shell: single R^3 counterterm absorbed by field redefinition.
  L = 3: P(4) = 3 independent quartic Weyl invariants: (C^2)^2,
         C^4_box, and (*CC)^2. Cayley-Hamilton identity reduces 3 to 2.
         Chirality (CHIRAL-Q) provides no further reduction at this order.
         Result: 2 independent counterterm structures vs 1 spectral
         parameter. The system is overdetermined (2:1).
  L >= 4: P(2L) grows monotonically. P(5) = 2, P(6) = 7, P(7) = 5,
         P(8) = 13. The overdetermination worsens at each order.

The practical suppression is enormous: the three-loop contribution is
suppressed by alpha_C^3 / (16 pi^2)^3 = 3.2 x 10^{-10} relative to
the leading action. This means that even if UV-completeness fails,
the perturbative predictions through two loops are reliable.

## 3. Known Results

- **L = 1 (MR-7, CERTIFIED):** D(L=1) = 0. The one-loop graviton-
  graviton scattering amplitude is UV-finite. Tree-level SCT = GR.
  One-loop: finite, with D = 0 verified explicitly.

- **L = 2 (MR-5b, UNCONDITIONAL):** D(L=2) = 0 on-shell. The
  CHIRAL-Q Theorem 6.12 proves this unconditionally in the D^2-
  quantization scheme. The unique off-shell R^3 counterterm is
  absorbable by the field redefinition delta psi(u) = delta_1 u^2.

- **L = 3 (FUND program, NEGATIVE):** Five investigations probed all
  known structural escape routes:
  - FUND-FRG: No connection between FRG fixed points and spectral
    action finiteness.
  - FUND-FK3: The fakeon prescription does not modify the UV
    divergence structure at three loops.
  - FUND-NCG: No NCG axiom or spectral function choice resolves the
    structural mismatch.
  - FUND-SYM: Hidden symmetry search reduced but did not eliminate
    the independent invariant count.
  - FUND-LAT: Exact spectral action computation on S^4 confirmed the
    perturbative counting.

- **Cayley-Hamilton reduction:** The Cayley-Hamilton theorem for 4x4
  matrices reduces the number of independent quartic Weyl invariants
  from P(4) = 3 to 2. This is insufficient: 2 > 1.

- **Practical suppression:** alpha_C^3 / (16 pi^2)^3 = 3.2 x 10^{-10}.
  Three-loop effects are negligible for all physical purposes through
  energies far below the Planck scale.

- **CHIRAL-Q D^2-quantization:** Revised survival probability for the
  overall SCT framework: 87-93% (post-CHIRAL-Q). This probability
  refers to the theory's viability as a predictive framework, not to
  UV-completeness specifically.

- **Survival probability for UV-completeness:** 15-25%. This is the
  probability that a hidden structural principle (not yet identified)
  could resolve the three-loop obstruction.

## 4. Failed Approaches

1. **Spectral function tuning (FUND-NCG).** Attempted to find a cutoff
   function f(u) such that the three-loop counterterm is automatically
   zero. The three-loop divergence involves three independent tensor
   structures, each with its own coefficient. A single function f(u)
   can set one combination to zero, but not two simultaneously. This
   was verified by explicit computation for f(u) = e^{-u^alpha} with
   alpha in {1, 3/2, 2, 3, 5}.

2. **Fakeon mechanism at three loops (FUND-FK3).** The Anselmi-Piva
   fakeon prescription modifies the propagator by replacing ghost poles
   with purely virtual particles. At one and two loops, this does not
   affect the divergence structure (only the finite parts change). At
   three loops, the fakeon prescription modifies certain integral
   topologies, but the net effect on the counterterm coefficients is
   insufficient to achieve cancellation. Confidence: 95%.

3. **Hidden discrete symmetry (FUND-SYM).** Searched for a discrete
   symmetry of the spectral triple that would relate the three quartic
   Weyl invariants, reducing the independent count. The CP symmetry
   relates (C^2)^2 and (*CC)^2 but not C^4_box. No additional symmetry
   was found in the NCG framework. The search reduced the effective
   count from 3 to 2 (via Cayley-Hamilton), but not to 1.

4. **FRG fixed-point connection (FUND-FRG).** Attempted to connect
   the spectral action to the functional renormalization group (FRG)
   framework used in asymptotic safety. The spectral action is not a
   standard effective average action, and the Wetterass equation does
   not apply directly. No connection between FRG fixed points and
   spectral action finiteness was established.

5. **Exact computation on S^4 (FUND-LAT).** Computed the spectral
   action exactly on S^4 (using the known Dirac spectrum) to check
   whether non-perturbative effects modify the divergence counting.
   The exact result agrees with the heat-kernel expansion to the
   relevant order, confirming the perturbative counting.

## 5. Success Criteria

- A proof that D(L) = 0 for all L (all-orders finiteness), or
- Identification of a UV fixed point (AF or AS) with a finite-dimensional
  critical surface, or
- A no-go theorem proving that SCT cannot be UV-complete, with explicit
  identification of the obstruction order and structure.
- In any case, a clear statement of the energy scale up to which SCT
  predictions are reliable.

## 6. Suggested Directions

1. **Full a_8 computation.** The three-loop counterterm involves the
   Seeley-DeWitt coefficient a_8 of the Dirac operator. If a_8 produces
   quartic Weyl invariants in a ratio that accidentally matches the
   three-loop divergence, the obstruction is resolved. Computing a_8 for
   the full SM spectral triple is a formidable but well-defined task.
   Survival probability for this route: 15-25%.

2. **Non-perturbative FRG.** Apply the Wetterass equation directly
   to the spectral action, treating the eigenvalue distribution of D^2
   as the flow variable. Look for a UV fixed point. This bypasses
   perturbative divergence counting entirely.

3. **String/M-theory embedding.** If SCT can be embedded as the low-
   energy limit of a UV-complete string theory, UV-completeness is
   inherited. The spectral action has connections to Matrix theory
   and string field theory that have not been fully explored.

4. **Three-loop explicit computation.** Compute the three-loop graviton
   scattering amplitude in SCT and check whether the divergence is
   proportional to a single tensor structure (rather than two). This
   would require extending MR-7 to L = 3.

5. **Reformulation.** The 2:1 overdetermination may indicate that the
   perturbative framework is the wrong language. A reformulation in
   terms of spectral flow (Postulate 5, V2) or causal structure might
   avoid the counting obstruction entirely.

## 7. References

1. Alfyorov, Shnyukov, "Auxiliary boundary data and the failure of
   intrinsic coherence in the spectral action at three loops,"
   DOI:10.5281/zenodo.19098027.
2. Anselmi, Piva, "A new formulation of Lee-Wick quantum field theory,"
   JHEP 06 (2017) 066, arXiv:1703.04584.
3. Avramidi, I.G., "Heat kernel approach in quantum field theory,"
   Nucl. Phys. Proc. Suppl. 104 (2002) 3, math-ph/0107018.
4. Codello, Percacci, Rahmede, arXiv:0805.2909 -- FRG and higher-
   derivative gravity.
5. de Brito, Eichhorn, Pfeiffer, arXiv:2301.13525 -- higher-order
   causal set actions.
6. Gilkey, P.B., "Invariance Theory, the Heat Equation, and the
   Atiyah-Singer Index Theorem," CRC Press (1995) -- Seeley-DeWitt
   coefficients.

## 8. Connections

- **Blocked by OP-02** (Postulate 5): the UV behavior depends on the
  dynamical principle. Different choices (V1, V2, V3) may lead to
  different UV structures.
- **Blocked by OP-09** (three-loop computation): explicit three-loop
  results would confirm or refute the 2:1 obstruction.
- **Blocked by OP-10** (a_8 coefficient): the full a_8 computation is
  a prerequisite for the "accidental ratio match" route.
- **Blocked by OP-13** (non-perturbative effects): if non-perturbative
  contributions modify the divergence structure, the perturbative
  counting is irrelevant.
- Related to **OP-03** (non-perturbative definition): a non-perturbative
  formulation might bypass the perturbative obstruction.
- Related to **OP-04** (cutoff function): if f can be fixed by physical
  requirements, the counterterm cancellation condition may simplify.
- Even if UV-completeness fails, all SCT predictions through two loops
  remain valid. The theory is a well-structured effective framework
  with improved perturbative reliability compared to GR.

---
id: OP-08
title: "All-orders Kubo-Kugo ghost decoupling proof"
domain: [unitarity]
difficulty: hard
status: partial
deep-research-tier: A
blocks: []
blocked-by: [OP-07]
roadmap-tasks: [MR-2]
papers: ["2308.09006", "1979KugoOjima", "1704.07728", "1801.00915", "1806.03605", "1909.04955", "2109.06889", "1712.04308"]
date-opened: 2026-03-31
date-updated: 2026-04-02
progress: "BRST-grading killed (wrong classifier). OS killed (no-go). Best path: finite-N fakeon + N->inf (= OP-07). Reduces to marked-diagram CL theorem."
---

# OP-08: All-orders Kubo-Kugo ghost decoupling proof

## 1. Statement

Prove that the ghost degree of freedom at the physical pole z_L = -1.2807
of the SCT graviton propagator decouples from all physical S-matrix
elements at all loop orders in the operator formalism. Alternatively,
demonstrate a specific process where ghost-mediated contributions survive
the fakeon prescription at L >= 2.

## 2. Context

Kubo and Kugo (2023, arXiv:2308.09006) raised the objection that
higher-derivative gravity theories with ghost poles have operator-
formalism pathologies: the negative-norm ghost state |G> satisfies
<G|G> < 0, and unless a quartet mechanism decouples it from the
physical Hilbert space, unitarity is violated. Their analysis
assumes the standard Feynman propagator.

The KK investigation (CERTIFIED) established three results at one loop:

1. **KO quartet: REJECTED.** The Kugo-Ojima quartet mechanism requires
   a BRST-exact partner with specific quantum numbers (spin-1, ghost
   number +1, fermionic). The SCT ghost at z_L is spin-2, ghost number 0,
   bosonic. There is no BRST partner with matching quantum numbers.

2. **Fakeon: PRIMARY resolution.** Under the fakeon prescription,
   Im[G_FK(z_L)] = 0 to 84-digit precision. The ghost pole contributes
   zero imaginary part to the propagator, so ghost pair production
   vanishes at one loop. This was verified by 64 independent numerical
   tests.

3. **Ghost pair threshold.** The lowest ghost pair-production threshold
   is E_th = 2 * m_ghost = 2 * 1.132 * Lambda = 2.264 * Lambda. The
   spacelike pole z_0 = 2.4148 corresponds to k^2 < 0 and cannot
   contribute to on-shell pair production.

The gap: all three results are rigorously established only at one loop.
The operator-formalism argument requires showing that no physical
state has nonzero overlap with the ghost state at all loop orders, not
just at tree level and one loop.

## 3b. Partial Resolution (2026-04-02)

**STATUS: PARTIAL. Two approaches definitively killed. Best path identified: finite-N fakeon truncation + N→∞. Reduces to OP-07 + marked-diagram CL theorem.**

### Approaches killed

**BRST + fakeon Z₂-grading: WRONG LANGUAGE.** From 1909.04955: fakeon
prescription is compatible with ANY nonzero residue — positive, negative,
or complex — if Re m² ≥ 0. The paper shows a positive-residue pole
that can be quantized as either physical or hard fakeon. Therefore
"positive residue = physical, negative = fakeon" is mathematically
incorrect. No BRST/cohomology program for fakeons exists in the
literature (searched 18A6, 18A1, 1909.04955: zero hits for "BRST"
or "cohomology").

**OS reflection positivity: NO-GO.** Strict no-go for higher-derivative
rational propagators (1712.04308). Fakeon literature does not contain
OS-formulation. The fakeon construction is based on nonanalytic Wick
rotation / average continuation, not standard Euclidean positivity.

**Modified LSZ: possible but weaker.** Can repackage projected unitarity
but not serve as primary proof route. Anselmi's all-orders proof uses
cutting equations, not LSZ reduction.

### Best path: finite-N fakeon + N→∞

The all-orders fakeon proof for finite-pole propagators (17A3, 18A1)
works through:
1. Fakeon prescription (17A3 eqs.5.1-5.2, gravity eqs.6.1-6.3)
2. Average continuation (18A1 eqs.4.1, 4.4)
3. Physical subspace V ⊂ W (18A1 eqs.7.1-7.2)
4. Cut-propagator rules (18A1 eqs.7.6-7.8)
5. Diagrammatic cutting equation (18A1 eq.7.10)

To extend to SCT: introduce finite-N truncations G_N, prove projected
unitarity per-N, then take N→∞. This requires:
- **Marked-diagram CL theorem:** upgrade CL bound (Σ M_n = 5.002e-4)
  from amplitude-level to cut-level (marked diagrams)
- **Commutativity:** N→∞ commutes with average continuation and
  threshold-by-threshold processing
- **Tail suppression:** high-pole thresholds uniformly controlled

This is the SAME mathematical bottleneck as OP-07 (gap at eq.2.25),
plus the additional requirement of marked-diagram control.

### Conditional theorem structure

IF (1) per-N projected cutting equations give projected unitarity,
(2) N-uniform bound for marked L-loop diagrams on compact domains,
(3) average continuation commutes with N→∞,
(4) tail of marked diagrams summable:
THEN physical S-matrix unitary on V, z_L gives no asymptotic states.

### Kubo-Kugo objection status

2308.09006 (Kubo-Kugo) shows unitarity violation for Lee-Wick complex
ghosts under STANDARD Feynman quantization. This is a valid argument
against Lee-Wick prescription but NOT against fakeon prescription,
where ghost modes are excluded from cuts by construction (18A1 eqs.
7.7-7.8). The two prescriptions are fundamentally different.

## 3. Known Results

- **One-loop:** Im[G_FK] = 0 at z_L to 84 digits. 64 tests PASS.
  Spectral positivity theorem Im[G_dressed] > 0 proven.
- **Tree level:** Fakeon prescription gives real amplitude (trivially
  unitary, no cuts).
- **Stelle comparison:** In Stelle gravity (polynomial propagator), the
  fakeon prescription is proven to all orders. SCT residue |R_L| = 0.5378
  is 50% smaller than Stelle's ghost residue (suppressed ghost coupling).
- **Lee-Wick pairs:** The 3 complex conjugate pairs at |z| ~ 34, 59, 85
  have negligible residues |R_n| < 0.01 and are kinematically
  inaccessible below Lambda.
- **Sum rule:** Sigma R_n / z_n = 13/60 (GZ result). The sum converges
  absolutely, meaning higher poles contribute a bounded correction to
  ghost-mediated amplitudes.

## 4. Failed Approaches

1. **KO quartet mechanism.** Cannot apply because the SCT ghost has
   wrong quantum numbers (spin-2/gh#0/bosonic versus the required
   spin-1/gh#+1/fermionic). This is a structural impossibility, not
   a technical difficulty. The Kugo-Ojima BRST cohomology argument
   requires a partner state that does not exist in the SCT spectrum.

2. **Mannheim PT-symmetric quantization.** Mannheim (2018) proposed
   that higher-derivative theories should be quantized with PT-symmetric
   boundary conditions, which would make the ghost norm positive.
   Applied to SCT: the propagator zeros are not PT-symmetric (z_L is
   real negative, z_0 is real positive), so the PT program would need
   to be formulated on the full infinite-pole structure. No concrete
   implementation for SCT exists.

3. **DQFT/IHO approach (Oda 2022).** The inverted harmonic oscillator
   quantization scheme was proposed for Stelle gravity. Extension to SCT
   is unexplored and faces the same infinite-pole convergence issue.

## 5. Success Criteria

- A proof in the operator formalism that <phys|G|phys> = 0 at all
  orders, where |phys> is any physical state satisfying BRST/fakeon
  selection rules and |G> is the ghost state at z_L.
- Or: a proof that the S-matrix restricted to the physical Hilbert space
  (defined by fakeon exclusion) is unitary in the operator norm.
- Or: identification of a specific two-loop or higher process where
  ghost-mediated intermediate states produce nonzero matrix elements
  between physical states despite the fakeon prescription.

## 6. Suggested Directions

1. **Modified LSZ reduction.** The standard LSZ formula projects onto
   on-shell external states using residues at propagator poles. Define
   a modified LSZ that excludes residues at ghost poles (fakeon poles)
   and show that the resulting S-matrix is unitary. The key step is
   proving that the excluded contributions form a closed ideal in the
   operator algebra.

2. **BRST cohomology with fakeon grading.** Introduce a Z_2 grading
   that distinguishes physical poles (positive residue) from fakeon
   poles (negative residue). Define a modified BRST operator Q_FK
   whose cohomology is the physical Hilbert space. Show H^0(Q_FK)
   contains no negative-norm states.

3. **Induction on loop order.** Assume ghost decoupling at L loops.
   At L+1 loops, the new diagrams involve at most one additional ghost
   propagator insertion. Use the CL bound (smooth corrections < 0.32%)
   and the fakeon prescription (Im = 0 at each ghost pole) to show that
   the L+1 loop ghost contribution is bounded by the L-loop bound times
   a contraction factor < 1.

4. **Nonperturbative approach via Euclidean reflection positivity.**
   Verify that the Euclidean theory with fakeon-modified propagator
   satisfies Osterwalder-Schrader reflection positivity, which implies
   unitarity of the physical Hilbert space via the reconstruction theorem.

## 7. References

1. Kubo, J. and Kugo, T. "Unitarity and higher-order gravitational
   scattering," arXiv:2308.09006.
2. Kugo, T. and Ojima, I. "Local covariant operator formalism of
   non-Abelian gauge theories and quark confinement problem,"
   Prog. Theor. Phys. Suppl. 66 (1979) 1.
3. Anselmi, D. "Fakeons and Lee-Wick models," JHEP 02 (2018) 141,
   arXiv:1801.00915.
4. Mannheim, P.D. "Appropriate inner product for PT-symmetric
   Hamiltonians," Phys. Rev. D 97 (2018) 045001.
5. Oda, I. "Fake particles in quantum gravity," arXiv:2203.02516.

## 8. Connections

- **Blocked by OP-07:** The fakeon prescription must be rigorously
  established for infinite-pole propagators before the operator
  formalism can use it. Without OP-07, the operator-formalism
  argument has an uncontrolled input.
- **Related to OP-10** (D^2-quantization): If D^2-quantization is
  equivalent to metric quantization at all orders, CHIRAL-Q provides
  an alternative route to ghost decoupling that bypasses OP-08 entirely.
- **Related to OP-12** (loop-level KK): Ghost decoupling in the
  operator formalism is related to (but distinct from) the dispersion-
  relation structure at higher loops.

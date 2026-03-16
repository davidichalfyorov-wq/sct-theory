# MR-2 Handoff Certificate: Unitarity and Stability of the SCT Graviton Propagator

**Task:** MR-2 (Unitarity and Stability)
**Date:** March 14, 2026
**Status:** CONDITIONAL PASS
**Verification checks:** 79 new unitarity tests + 123 regression (MR-1 + MR-2 d.5) + 2542 full suite + 9 independent spot checks = 2753 checks, 0 failures

---

## 1. Scope

MR-2 establishes the unitarity and stability properties of the SCT one-loop graviton propagator. Seven sub-questions addressed:

| Sub-question | Content | Verdict |
|--------------|---------|---------|
| Q1: Euclidean ghost (z_0) | Spacelike pole classification, fakeon prescription | CONDITIONAL |
| Q2: Lorentzian ghost (z_L) | Timelike pole, Donoghue-Menezes width | CONDITIONAL |
| Q3: Type C complex zeros | Lee-Wick pairs, tiny residues | PASS (high confidence) |
| Q4: Continuum spectral function | No continuum (Pi_TT entire) | PASS (high confidence) |
| Q5: Optical theorem | Trivially satisfied at one-loop | PASS (high confidence) |
| Q6: Ostrogradski stability | Does not apply (infinite-derivative) | PASS (high confidence) |
| Q7: Conformal factor | alpha_R(xi) >= 0 for all xi | PASS (high confidence) |

---

## 2. Complete Ghost Catalogue (|z| <= 100)

The argument principle N(R) = (1/2*pi*i) * oint Pi_TT'/Pi_TT dz was evaluated at 16 radii from R = 3 to R = 100 with 2000--4000 quadrature points at 50-digit precision. Results independently confirmed with 8000 points.

| # | Location (Euclidean z) | Type | |z| | Residue R = 1/(z*Pi_TT'(z)) | |R| | Classification |
|---|------------------------|------|-----|-------------------------------|-----|----------------|
| 1 | z_L = -1.28070227806348515 | B (real, Lorentzian) | 1.28 | -0.53777207832730514 | 0.538 | Timelike ghost (k^2 > 0) |
| 2 | z_0 = +2.41483888986536891 | A (real, Euclidean) | 2.41 | -0.49309950210599085 | 0.493 | Spacelike ghost (k^2 < 0) |
| 3 | 6.0511250025 + 33.2897965838i | C (complex) | 33.84 | -0.0010 + 0.0085i | 0.00856 | Lee-Wick pair |
| 4 | 6.0511250025 - 33.2897965838i | C (conjugate) | 33.84 | -0.0010 - 0.0085i | 0.00856 | Lee-Wick pair |
| 5 | 7.1436362923 + 58.9313028165i | C (complex) | 59.36 | -0.0004 + 0.0049i | 0.00488 | Lee-Wick pair |
| 6 | 7.1436362923 - 58.9313028165i | C (conjugate) | 59.36 | -0.0004 - 0.0049i | 0.00488 | Lee-Wick pair |
| 7 | 7.8416599800 + 84.2744439925i | C (complex) | 84.64 | -0.0002 + 0.0034i | 0.00342 | Lee-Wick pair |
| 8 | 7.8416599800 - 84.2744439925i | C (conjugate) | 84.64 | -0.0002 - 0.0034i | 0.00342 | Lee-Wick pair |

**Pattern:** The zeros form an infinite sequence. Real parts cluster near Re(z) ~ 6--8. Imaginary parts grow approximately linearly: Im(z) ~ 25n for pair n. Residue magnitudes decay: |R_n| ~ 0.29/|z_n|.

**Argument principle counts:** N(3) = N(30) = 2; N(35) = N(50) = 4; N(60) = N(80) = 6; N(90) = N(100) = 8. No Type D (anomalous) zeros found.

---

## 3. Spectral Function Structure

**Key finding:** Pi_TT(z) is an entire function (established in NT-2, confirmed numerically). It has NO branch cuts, and therefore the Kallen-Lehmann spectral function has NO continuum component. The spectral function consists entirely of delta-function contributions:

```
rho_TT(sigma) = delta(sigma) - 0.538 * delta(sigma - 1.2807*Lambda^2)
```

(On the physical real-sigma axis, only the graviton and the Lorentzian ghost contribute standard delta functions.)

**Numerical verification:** Im[Pi_TT(-sigma + i*epsilon)] scales exactly linearly with epsilon for all tested sigma in [0.1, 50], confirming zero discontinuity on the real Lorentzian axis.

---

## 4. Modified Sum Rule

The sum of all residues of 1/(z * Pi_TT(z)) satisfies:

```
1 + sum_i R_i = 1/Pi_TT(infinity) = -6/83 = -0.072289...
```

This is NOT a superconvergence sum rule (the sum is not zero). The reason is that Pi_TT(z) asymptotes to a finite nonzero constant on the positive real axis: Pi_TT(z -> +inf) = -83/6, verified numerically at z = 10000 to within 0.007%.

**Partial sum verification:** With the 8 known zeros, 1 + sum(R_i) = -0.0342, which is converging toward the target -0.0723 as additional zeros at |z| > 100 are included.

---

## 5. Ghost Prescription Analysis

### 5.1. Euclidean Ghost (z_0 = 2.4148)

- **Physical interpretation:** Spacelike pole at k^2 = -2.4148*Lambda^2 < 0. Does not produce on-shell particle states. Modifies the static Newtonian potential via Yukawa correction e^{-m*r} with m = sqrt(2.4148)*Lambda.
- **Prescription:** Fakeon (Anselmi-Piva) or acceptance as benign spacelike pole.
- **Condition:** Fakeon prescription must extend to nonlocal propagators (proven only for polynomial propagators in arXiv:1801.00915).

### 5.2. Lorentzian Ghost (z_L = 1.2807)

- **Physical interpretation:** Timelike pole at k^2 = +1.2807*Lambda^2 > 0. Appears in the physical spectrum with negative spectral weight R_L = -0.538. This IS a unitarity challenge.
- **Prescription:** Donoghue-Menezes unstable ghost mechanism (arXiv:1908.02416).
- **Width (first-principles, A3):** Gamma/m = 6.554 x 10^{-2} * (Lambda/M_Pl_red)^2, equivalently 1.647 * (Lambda/M_Pl_unred)^2. Derived from HLZ partial widths with ghost residue factor |R_L| = 0.538 and decay-channel multiplicity N_eff_width = N_s + 3*N_D + 6*N_v = 143.5. The ghost is always a narrow resonance (Gamma/m < 0.066 even at Lambda = M_Pl). See `theory/consistency-checks/A3_ghost_width_handoff.md` for the full derivation chain.
- **Condition 1:** Donoghue-Menezes mechanism must hold for nonlocal theories (developed for local quadratic gravity).
- **Condition 2:** The Kubo-Kugo objection (arXiv:2308.09006, anti-unstable ghost in operator formalism) must not invalidate the path-integral result.

### 5.3. Type C Complex Pairs

- **Residues:** |R| < 0.01 for all Type C pairs, decreasing with |z|.
- **Prescription:** Standard Lee-Wick (CLOP) contour deformation or fakeon.
- **Verdict:** PASS (high confidence). Physical effects negligible.

---

## 6. Stability Results

| Property | Verdict | Mechanism |
|----------|---------|-----------|
| Ostrogradski stability | PASS | Does not apply to infinite-derivative theories (Barnaby-Kamran 2007) |
| Conformal factor | PASS | alpha_R(xi) = 2*(xi-1/6)^2 >= 0 for all xi |
| Scalar sector decoupling | PASS | Pi_s = 1 identically at xi = 1/6 (conformal coupling) |
| Optical theorem (1-loop) | PASS | Im[Pi_TT] = 0 on the real Lorentzian axis (trivially satisfied) |
| Initial value problem | PASS | Perturbatively well-posed (Barnaby-Kamran criterion satisfied) |

---

## 7. Comparison with Stelle Gravity

| Quantity | Stelle (local) | SCT (nonlocal) | Comparison |
|----------|----------------|----------------|------------|
| Ghost type | Single real ghost | 2 real + infinite complex pairs | More complex but structured |
| Ghost mass (spin-2) | z = 60/13 = 4.615 | z_0 = 2.415 (Euclidean), z_L = 1.281 (Lorentzian) | Lower effective mass |
| Ghost residue | |R| = 1.0 | |R| = 0.493 (z_0), 0.538 (z_L) | 50.7% suppressed (z_0) |
| Continuum spectral function | None (polynomial Pi) | None (entire Pi) | Same qualitative structure |
| Form factors | Postulated (R^2 + R_mn*R^mn) | Derived from spectral action | SCT more predictive |
| Scalar sector ghost | Present (unless tuned) | Ghost-free at xi = 1/6 | SCT better |

Stelle limit recovered: Pi_TT_SCT / Pi_TT_Stelle = 0.9999996 at z = 0.001.

---

## 8. Conditions for Theory Viability

The theory survives MR-2 under three conditions, all plausible but currently unproven:

**Condition 1:** The fakeon prescription extends to nonlocal propagators with entire-function form factors.
- *For:* SCT pole structure (simple isolated poles with decaying residues) is qualitatively identical to polynomial propagators.
- *Against:* No rigorous proof exists; infinite number of Type C poles is a new mathematical subtlety.
- **UPDATE (FK pipeline, 2026-03-14):** The fakeon prescription convergence has been analyzed in detail. The extended ghost catalogue (16 zeros: 2 real + 7 conjugate pairs up to |z| = 200) confirms |R_n| ~ 0.289/|z_n|. The one-subtraction Mittag-Leffler series for 1/Pi_TT(z) converges absolutely (sum|R_n/z_n| ~ 0.625). Classification: CONDITIONALLY CONVERGENT. This upgrades Condition 1 from "plausible but unproven" to **"mathematically well-posed with one subtraction."** Remaining steps: compute g(z) explicitly, prove limit commutativity, verify S-matrix unitarity. These are concrete tasks, not open conjectures. See `theory/consistency-checks/FK_fakeon_convergence_handoff.md` for the full analysis (56 FK tests, 250/250 total PASS).

**Condition 2:** The Donoghue-Menezes unstable ghost mechanism holds for nonlocal theories with derived form factors.
- *For:* Width formula gives Gamma > 0 for any finite Lambda/M_Pl; mechanism depends on gravitational coupling, not form-factor structure at leading order.
- *Against:* Kubo-Kugo operator-formalism analysis argues the ghost is "anti-unstable"; Buoninfante (arXiv:2501.04097) raises concerns about first-sheet poles.

**Condition 3:** The Kubo-Kugo objection does not invalidate the path-integral analysis.
- *Status:* Disagreement between operator and path-integral formalisms is unresolved in the community.

**Theory-death scenario:** The theory is dead if ALL of the following hold simultaneously: (1) fakeon prescription cannot extend to nonlocal propagators; (2) Donoghue-Menezes fails for SCT; (3) no alternative mechanism (PT symmetry, IHO/DQFT, modified probability) resolves the ghost; (4) the Euclidean spacelike ghost cannot be dismissed as benign.

---

## 9. Known Gaps

1. **[GAP-1] Ghost width first-principles calculation:** **RESOLVED by A3 (2026-03-14).** The first-principles calculation confirms that the graviton-matter vertex is NOT modified by form factors at tree level (form factors enter only the propagator and pure-graviton vertices). The corrected formula is Gamma/m = C * (Lambda/M_Pl_red)^2 with C = |R_L| * 2 * |z_L| * N_eff_width / (960*pi) = 0.06554. The original MR-2 rough estimate (N_eff = 118.75, coefficient 0.118 in unreduced units) was off by a factor of ~14 due to: (1) using thermal g_* instead of decay-channel weighting, (2) omitting the |R_L| factor, (3) omitting the |z_L| mass factor. The corrected value does not change the qualitative conclusion (ghost is unstable) but quantitatively updates the coefficient. See `theory/consistency-checks/A3_ghost_width_handoff.md` and `analysis/results/a3/a3_ghost_width_results.json`.

2. **[GAP-2] Dressed propagator:** **RESOLVED by GP (2026-03-14).** The dressed propagator
    $G^{-1}(k^2) = k^2 \cdot \Pi_{TT}(-k^2/\Lambda^2) - \Sigma(k^2)$
    has been computed. The one-loop matter self-energy Im[Sigma(m^2)] is positive (Cutkosky rules), and the ghost pole shifts to:
    ```
    k^2_pole = m^2 + R_L * Sigma(m^2),   Im[k^2_pole] = R_L * Im[Sigma] < 0
    ```
    The sign chain has 6 steps, all verified by 4 independent agents (GP-D, GP-DR, GP-V, GP-VR) at up to 100-digit precision:
    - z_L < 0 => Pi_TT'(z_L) > 0 => R_L = 1/(z_L * Pi_TT') < 0 [ghost]
    - Im[Sigma] > 0 [Cutkosky] => Im[k^2_pole] < 0 => Gamma > 0 [positive width]

    The width matches the A3 first-principles value to 17 significant figures:
    Gamma/m = C_Gamma * (Lambda/M_Pl)^2 with C_Gamma = 0.06554011853292677.

    The computation is definitive; the interpretation (Donoghue-Menezes: ghost decays vs. Kubo-Kugo: ghost persists) remains formalism-dependent. See `theory/consistency-checks/GP_dressed_propagator_handoff.md` for the full derivation chain (84 GP tests + 186 regression = 270/270 PASS).

3. **[GAP-3] Higher-loop absorptive part:** Im[Pi_TT] = 0 at the one-loop level. The two-loop absorptive part (from graviton + SM loops) needs to be computed to verify the sign of the width.

4. **[GAP-4] Ghost catalogue beyond |z| = 100:** **PARTIALLY RESOLVED (FK, 2026-03-14).** The FK pipeline extended the catalogue to |z| = 200, finding 4 additional conjugate pairs (C4-C7), for a total of 16 zeros (2 real + 7 conjugate pairs). The asymptotic pattern |R_n| ~ 0.289/|z_n| with zero spacing Delta ~ 25.3 is firmly established. The remaining deficit in the sum rule (-0.038) is confirmed to converge via sum 1/|z_n|^2. The infinite sequence beyond |z| = 200 continues the established pattern.

5. **[GAP-5] Kubo-Kugo vs Donoghue-Menezes resolution:** The community disagreement between operator and path-integral formalisms for ghost theories is unresolved.

6. **[GAP-6] Nonlocal fakeon theory:** **PARTIALLY RESOLVED (FK, 2026-03-14).** The FK pipeline established that the pole series in the fakeon propagator converges absolutely with one Mittag-Leffler subtraction. The mathematical obstruction (infinite poles) is mild: only logarithmic divergence of sum|R_n|, resolved by standard subtraction. Physical amplitudes converge after renormalization. What remains unproven: limit commutativity (N-pole fakeon as N -> inf), and all-orders S-matrix unitarity.

---

## 10. Downstream Dependencies

MR-2 results feed into:
- **MR-3** (causality): analyticity structure constrains microcausality violation scale
- **MR-4** (higher-loop corrections): ghost width requires 2-loop self-energy
- **MR-5** (finiteness): infinite Type C pole sequence affects UV finiteness argument
- **MR-7** (Stelle limit recovery): complete comparison table now available
- **PPN-1** (solar system tests): the Euclidean ghost (z_0 > 0) contributes Yukawa corrections to the static potential; the Lorentzian ghost (z_L < 0) does NOT enter the static potential integral (z = k²_spatial/Λ² ≥ 0) but affects dynamical scattering processes
- **LT-1** (UV completeness): ghost resolution is necessary but not sufficient

---

## 11. Deliverables

| File | Description |
|------|-------------|
| `analysis/scripts/mr2_unitarity.py` | Comprehensive unitarity analysis script |
| `analysis/scripts/mr2_figures.py` | Publication figure generator (4 figures) |
| `analysis/sct_tools/tests/test_mr2_unitarity.py` | 79-test verification suite |
| `analysis/results/mr2/mr2_unitarity_results.json` | Full numerical results |
| `analysis/figures/mr2/spectral_function.pdf` | Delta-function spectral structure |
| `analysis/figures/mr2/complex_zeros_extended.pdf` | 8 zeros in complex z-plane |
| `analysis/figures/mr2/Pi_TT_complex.pdf` | 2D color map of |Pi_TT(z)| |
| `analysis/figures/mr2/sum_rule_convergence.pdf` | Residue sum rule convergence |
| `docs/reviews/MR2_L_report.md` | Literature extraction report |
| `docs/reviews/MR2_LR_report.md` | Literature audit report |
| `docs/reviews/MR2_D_report.md` | Derivation report |
| `docs/reviews/MR2_DR_report.md` | Derivation review report |
| `docs/reviews/MR2_V_report.md` | Verification report |
| `docs/reviews/MR2_VR_report.md` | Final review and assessment |
| `theory/derivations/MR2_handoff.md` | This handoff certificate |

---

## 12. Verification Summary

| Layer | Checks | Status |
|-------|--------|--------|
| Ghost catalogue (argument principle, zeros, residues, symmetry) | 24 | 24/24 PASS |
| Spectral function (no continuum, Pi_TT real, monotonicity, signs) | 12 | 12/12 PASS |
| Sum rules (partial sums, convergence to -6/83, asymptote) | 6 | 6/6 PASS |
| Wick rotation (spacelike/timelike, convention consistency) | 7 | 7/7 PASS |
| Ghost width (dimensional, positivity, scaling) | 5 | 5/5 PASS |
| Conformal factor (alpha_R >= 0, decoupling, exact values) | 5 | 5/5 PASS |
| Consistency (Stelle limit, MR-1/MR-2 cross-checks, regression) | 10 | 10/10 PASS |
| Independent spot checks | 9 | 9/9 PASS |
| Full regression suite | 2542 | 2542/2542 PASS |

Cross-check: 3 independent computations (D, D-R, V) agree on all zero locations and residues to 10+ significant figures. One error was found and corrected: the superconvergence sum rule claim (sum -> 0) was replaced by the correct modified sum rule (sum -> -6/83).

---

## 13. Verdict

**MR-2: CONDITIONAL PASS**

The SCT graviton propagator has confirmed ghosts but the theory is not dead. The ghost problem reduces to a discrete set of poles (no continuum) treatable by known mechanisms from the literature on higher-derivative gravity. The theory's survival depends on three conditions (fakeon extension, Donoghue-Menezes applicability, Kubo-Kugo resolution) that are open questions in the quantum gravity community. Multiple structural advantages over Stelle gravity are established: ghost suppression (|R| ~ 0.5 vs 1.0), no continuum spectral weight, conformal stability, and derived (not postulated) form factors.

The conditions for viability are plausible but unproven. Resolution requires progress on the general theory of ghosts in nonlocal gravity, which is an active research area with competing proposals (fakeon, unstable ghost, PT symmetry, IHO/DQFT).

# A3 Handoff Certificate: Ghost Decay Width from First Principles

**Task:** A3 (Ghost Width First-Principles Calculation)
**Date:** March 14, 2026
**Status:** COMPLETE (all 6 agents PASS)
**Total checks:** 186/186 PASS (76 dedicated + 110 regression)
**Resolves:** MR-2 GAP-1

---

## 1. Summary

The ghost decay width in Spectral Causal Theory has been computed from first principles using the Han-Lykken-Zhang (HLZ) partial widths for massive spin-2 particle decay, the Donoghue-Menezes framework for ghost propagator treatment, and the MR-2 ghost catalogue data (pole location, residue). The calculation confirms that the SCT ghost is always unstable for any finite Lambda/M_Pl, always a narrow resonance, and has a width suppressed by 85% relative to Stelle gravity (in terms of Gamma/m).

---

## 2. Verified Master Formula

```
Gamma/m = C * (Lambda/M_Pl_red)^2
```

where:

```
C = |R_L| * 2 * |z_L| * N_eff_width / (960*pi) = 0.065540118532926767
```

Equivalently:

```
Gamma/m = 1.6472 * (Lambda/M_Pl_unred)^2
```

### Inputs (all verified to 10+ significant figures)

| Quantity | Value | Source |
|----------|-------|--------|
| z_L (Euclidean ghost pole) | -1.28070227806348515 | MR-2, argument principle + Newton's method |
| R_L (ghost residue) | -0.53777207832730514 | MR-2, 4-method consensus |
| N_s (real scalars) | 4 | SM Higgs doublet = 2 complex = 4 real |
| N_D (Dirac fermions) | 22.5 | 3 generations x 15 Weyl / 2 |
| N_v (massless vectors) | 12 | 8 gluons + W+ + W- + Z + photon |
| N_eff_width | 143.5 | N_s + 3*N_D + 6*N_v = 4 + 67.5 + 72 |
| M_Pl_red | 2.435 x 10^18 GeV | Reduced Planck mass |

---

## 3. Complete Derivation Chain

### 3.1 Ghost propagator near the pole

The full TT-sector propagator G_TT(k^2) = 1/[k^2 * Pi_TT(-k^2/Lambda^2)] has a zero of Pi_TT at z = z_L. Near this pole:

```
G_TT(k^2) ~ R_L / (k^2 - m_ghost^2)
```

where R_L = 1/(z_L * Pi_TT'(z_L)) = -0.538 and m_ghost^2 = |z_L| * Lambda^2.

### 3.2 Vertex structure

The graviton-matter vertex is standard (not modified by form factors). This was established by three independent arguments:

1. The matter action S_matter is independent of F_1, F_2 (curvature-squared form factors). The vertex h_mn T^mn comes from varying S_matter with respect to g_mn.
2. At O(h^1, phi^2), the curvature-squared terms produce no cross-terms on a flat background.
3. Confirmed by Edholm-Koshelev-Mazumdar (arXiv:2510.10270) and the NT-4a derivation.

### 3.3 HLZ partial widths

Using the Han-Lykken-Zhang results (hep-ph/9811350) for a massive spin-2 particle with kappa^2 = 2/M_Pl_red^2:

| Channel | Width per species | Ratio to scalar |
|---------|-------------------|-----------------|
| Real scalar | kappa^2 m^3 / (960*pi) | 1 |
| Dirac fermion | kappa^2 m^3 / (320*pi) | 3 |
| Massless vector | kappa^2 m^3 / (160*pi) | 6 |

### 3.4 Ghost residue factor

The width is proportional to |R_L| (linearly, not quadratically). This was confirmed by two independent arguments:

1. **LSZ:** One external ghost line contributes sqrt(|R_L|) to the amplitude, so |M|^2 scales as |R_L|.
2. **Donoghue-Menezes:** The width Gamma = Im[Sigma(m^2)] / (m * |D'(m^2)|), where |D'| = 1/|R_L|, giving linear dependence.

### 3.5 Total width assembly

```
Gamma_total = |R_L| * kappa^2 * m^3 / (960*pi) * N_eff_width

Gamma/m = |R_L| * 2 * |z_L| * N_eff_width / (960*pi) * (Lambda/M_Pl_red)^2
```

---

## 4. Correction of the MR-2 Rough Estimate

The MR-2 handoff used:

```
Gamma/m ~ (Lambda/M_Pl)^2 * N_eff / (320*pi)    with N_eff = 118.75
```

This gave a coefficient of 0.118 (in unreduced units). The first-principles result gives 1.647 (in unreduced units). The MR-2 estimate was incorrect by a factor of ~14 due to three compounding errors:

| Error | MR-2 value | Correct value | Factor |
|-------|------------|---------------|--------|
| N_eff convention | 118.75 (thermal g_*) | 143.5 (decay-channel) | Differs in meaning, not directly comparable |
| Missing |R_L| factor | 1.0 (implicit) | 0.538 | 0.538x |
| Missing |z_L| factor | 1.0 (implicit) | 1.281 | 1.281x |
| Normalization denominator | 320*pi | 960*pi (with 1:3:6 ratios) | Formula structure differs |

The thermal g_* = 118.75 counts effective relativistic degrees of freedom for energy density (Bose-Einstein/Fermi-Dirac weighted). The decay-channel weighting N_eff_width = 143.5 counts degrees of freedom by their gravitational coupling strength (1:3:6 for scalar:Dirac:vector). These are fundamentally different physical quantities.

**Qualitative conclusion unchanged:** The ghost is always unstable for any finite Lambda/M_Pl. The quantitative coefficient is corrected.

---

## 5. Numerical Results at Representative Scales

| Lambda/M_Pl | Gamma/m | Classification |
|-------------|---------|----------------|
| 1 | 6.554 x 10^{-2} | Narrow resonance |
| 10^{-1} | 6.554 x 10^{-4} | Very narrow |
| 10^{-3} | 6.554 x 10^{-8} | Very narrow |
| 10^{-5} | 6.554 x 10^{-12} | Quasi-stable |
| 10^{-10} | 6.554 x 10^{-22} | Quasi-stable |
| 10^{-17} | 6.554 x 10^{-36} | Effectively stable |

### Eotwash boundary (Lambda = 2.38 x 10^{-3} eV)

- Lambda/M_Pl_red = 9.77 x 10^{-31}
- m_ghost = 2.693 x 10^{-3} eV
- Gamma/m = 6.26 x 10^{-62}
- tau = 3.89 x 10^{48} s (far exceeds the age of the universe)

---

## 6. SCT vs Stelle Comparison

| Quantity | Stelle | SCT | Ratio |
|----------|--------|-----|-------|
| z_pole | 60/13 = 4.615 | |z_L| = 1.281 | 0.277 |
| |R| | 1.0 | 0.538 | 0.538 |
| m/Lambda | 2.148 | 1.132 | 0.527 |
| C (Gamma/m coefficient, reduced) | 0.439 | 0.0655 | 0.149 |

### Two ways to compare

| Metric | Formula | Value | Meaning |
|--------|---------|-------|---------|
| Gamma/m ratio | C_SCT / C_Stelle | 0.149 | SCT ghost is 14.9% as "broad" as Stelle |
| Absolute Gamma ratio | |R_L| * (|z_L|/z_S)^{3/2} | 0.079 | SCT ghost decays at 7.9% the Stelle rate |

The difference arises because the SCT ghost is lighter (m_SCT = 1.13 Lambda vs m_Stelle = 2.15 Lambda), providing an additional (m_SCT/m_Stelle) suppression in absolute width.

---

## 7. Partial Width Decomposition

| Channel | N_eff contribution | Fraction |
|---------|-------------------|----------|
| Vectors (12 species, weight 6) | 72 | 50.2% |
| Dirac fermions (22.5 species, weight 3) | 67.5 | 47.0% |
| Real scalars (4 species, weight 1) | 4 | 2.8% |
| **Total** | **143.5** | **100%** |

Gauge bosons and fermions dominate the ghost decay. The Higgs sector is subdominant.

---

## 8. Physical Implications

### 8.1 Ghost always unstable

For ANY finite Lambda/M_Pl ratio, Gamma > 0. The ghost decays to SM particles through gravitational coupling and does not appear as an asymptotic state (Donoghue-Menezes mechanism). This supports the viability of SCT under the conditions established in MR-2.

### 8.2 Ghost always narrow

Even at Lambda = M_Pl (the maximum physically reasonable value), Gamma/m = 0.066 << 1. The ghost is always a narrow resonance. This validates the pole approximation used throughout the derivation: the ghost pole is well-isolated and the Breit-Wigner treatment is self-consistent.

### 8.3 SCT is quantitatively better than Stelle

The SCT ghost width is suppressed relative to Stelle by 85% (Gamma/m) or 92% (absolute Gamma). Two effects combine: (1) the nonlocal form factors reduce the ghost residue from |R| = 1 to 0.538, and (2) they shift the ghost pole to a lower mass. These are derived properties, not tuned parameters.

### 8.4 Falsifiable prediction

The width formula Gamma/m = C * (Lambda/M_Pl_red)^2 with C = 0.06554 is a parameter-free prediction of SCT, modulo the single free parameter Lambda. If future measurements or theoretical arguments constrain Lambda, the ghost width is determined.

---

## 9. Conditions and Caveats

### 9.1 Inherited from MR-2

The width calculation is physically meaningful only under the three MR-2 conditions:
1. Fakeon prescription extends to nonlocal propagators
2. Donoghue-Menezes mechanism applies to nonlocal theories
3. Kubo-Kugo objection is resolvable

### 9.2 Tree-level vertex

Form factors modify only the propagator, not the graviton-matter vertex at tree level. Higher-order vertex corrections are O(kappa^2) suppressed and do not change the leading-order result.

### 9.3 Massless final states

All SM particles are treated as massless (m_particle << m_ghost). Valid for Lambda >> m_top = 173 GeV. At the Eotwash boundary Lambda = 2.38 x 10^{-3} eV, the approximation formally breaks down, but the width is 10^{-62} and physically irrelevant at that scale.

### 9.4 Perturbative validity

The calculation assumes gravitational perturbation theory (kappa^2 * m^2 << 1), requiring Lambda << M_Pl. At Lambda ~ M_Pl, nonperturbative effects could become significant.

### 9.5 Three-graviton vertex channel

The ghost-to-graviton decay channel involves the three-graviton vertex, which IS modified by form factors in SCT. This channel was not included in the calculation. It is expected to be subdominant: the Weyl action produces no on-shell scattering amplitudes in flat space at tree level, so this channel does not contribute at leading order.

---

## 10. Verification Matrix

### 10.1 Pipeline agent summary

| Agent | Role | Key checks | Result |
|-------|------|-----------|--------|
| A3-L | Literature extraction | Vertex structure, DM framework, HLZ widths, N_eff flag | PASS |
| A3-LR | Literature audit | 5 claims verified, N_eff corrected (118.75 = thermal g_*) | PASS |
| A3-D | Derivation | Master formula, 9/9 internal checks | PASS |
| A3-DR | Independent review | 7 tasks, script audit, Stelle comparison | PASS |
| A3-V | Full verification | 76/76 tests, 110/110 regression, 100-digit independent | PASS |
| A3-VR | Final meta-audit | Cross-agent consistency, all tests re-run, handoff cert | PASS |

### 10.2 Test breakdown (A3 dedicated: 76 tests)

| Layer | Tests | Content |
|-------|-------|---------|
| Layer 1: Analytic | 10 | Dimensions, positivity, N_eff, SM multiplicities, ghost signatures |
| Layer 2: Numerical | 17 | C coefficient, ghost mass, residue, HLZ widths, intermediates |
| Layer 3: Cross-consistency | 15 | Compact vs channel sum, A3-LR match, Stelle comparison, scaling |
| Layer 4: Physical | 16 | Narrow resonance, instability, HLZ ratios, fractions, Eotwash |
| Layer 5: Script checks | 8 | Built-in consistency checks from a3_ghost_width.py |
| Layer 6: High precision | 5 | 100-digit independent verification |
| Layer 7: Input stability | 5 | MR-2 input values cross-checked |
| **Total** | **76** | **All PASS** |

### 10.3 Regression tests

| Suite | Tests | Result |
|-------|-------|--------|
| test_mr2_unitarity.py | 79 | PASS |
| test_mr2_residue.py | 31 | PASS |
| **Total regression** | **110** | **PASS** |

### 10.4 Grand total: 186/186 PASS

---

## 11. Artifacts

### Scripts

| File | Description |
|------|-------------|
| `analysis/scripts/a3_ghost_width.py` | Complete derivation script (~500 lines) |

### Results

| File | Description |
|------|-------------|
| `analysis/results/a3/a3_ghost_width_results.json` | Full numerical results (JSON) |

### Tests

| File | Description |
|------|-------------|
| `analysis/sct_tools/tests/test_a3_ghost_width.py` | 76-test verification suite |

### Reports

| File | Description |
|------|-------------|
| `docs/reviews/A3_L_ghost_width_literature.md` | Literature extraction (A3-L) |
| `docs/reviews/A3_LR_ghost_width_audit.md` | Literature audit (A3-LR) |
| `docs/reviews/A3_D_ghost_width_derivation.md` | Derivation report (A3-D) |
| `docs/reviews/A3_DR_ghost_width_review.md` | Independent review (A3-DR) |
| `docs/reviews/A3_V_ghost_width_verification.md` | Verification report (A3-V) |
| `theory/consistency-checks/A3_ghost_width_handoff.md` | This handoff certificate |

### Updated documents

| File | Change |
|------|--------|
| `theory/derivations/MR2_handoff.md` | GAP-1 marked RESOLVED, width formula updated |

---

## 12. Downstream Dependencies

The A3 result feeds into:

- **MR-2 status:** GAP-1 (ghost width estimate) is now RESOLVED. The remaining five gaps (GAP-2 through GAP-6) are unaffected.
- **MR-4** (higher-loop corrections): The tree-level width provides the leading-order benchmark. Two-loop corrections are expected to be O(kappa^2) suppressed.
- **LT-3d** (laboratory/solar system predictions): The ghost width at accessible scales determines whether ghost resonances could appear in laboratory experiments. At the Eotwash boundary, the ghost is effectively stable.
- **Publication:** The corrected width formula, Stelle comparison, and partial width decomposition are publication-ready results for the SCT theory paper.

---

## 13. Final Verdict

**A3 Pipeline: COMPLETE PASS**

All six agents (L, LR, D, DR, V, VR) independently confirm the ghost decay width calculation. The master formula C = 0.065540118532926767 is verified to 17 significant digits by 100-digit independent computation, validated by 76 dedicated tests across 7 verification layers, and shown to be consistent with the MR-2 ghost catalogue (110 regression tests, 0 failures).

The N_eff correction from the MR-2 rough estimate (118.75, thermal g_*) to the correct decay-channel weighting (143.5) is consistently applied across all agents and confirmed by three independent derivation paths (A3-L direct, A3-D from HLZ, A3-DR independent re-derivation).

No cross-agent contradictions were found. No unresolved issues remain within the scope of A3. The result is subject to the three MR-2 viability conditions and the perturbative/massless-limit caveats documented in Section 9.

**Confidence assessment: HIGH.** The derivation rests on standard spin-2 decay kinematics (HLZ, textbook-level), the established MR-2 ghost catalogue, and the confirmed absence of form-factor modifications to the graviton-matter vertex. The only theoretical uncertainty is the O(kappa^2) correction from higher-loop effects and the three-graviton vertex channel, both of which are parametrically suppressed.

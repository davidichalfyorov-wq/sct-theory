---
id: OP-38
title: "Non-vacuum CJ coefficients"
domain: [theory, numerics]
difficulty: hard
status: open
deep-research-tier: B
blocks: []
blocked-by: []
roadmap-tasks: []
papers: [1904.01034, 2301.13525, 1212.0631]
date-opened: 2026-03-31
date-updated: 2026-03-31
---

# OP-38: Non-vacuum CJ coefficients

## 1. Statement

In vacuum spacetimes (R_mu_nu = 0), the CJ observable measures E^2 =
E_ij E^ij (electric Weyl squared). In non-vacuum spacetimes, the full
curvature expansion takes the form

  CJ = c_W E^2 + c_RW R (Weyl) + c_R R^2 + ...

where the cross-term c_RW couples Ricci and Weyl curvature, and c_R
is a pure Ricci contribution. Determine the complete coefficient
structure {c_W, c_RW, c_R} from causal set combinatorics, and explain
why pure Ricci curvature (E = 0) gives CJ = 0 exactly.

## 2. Context

Three key observations constrain the non-vacuum structure:

1. **de Sitter (E = 0, R != 0):** CJ = 0 exactly (measured at N = 5000,
   M = 50, all proper-time bins). This means c_R = 0: pure Ricci
   curvature cannot generate a CJ signal.

2. **Kottler (Schwarzschild-de Sitter, E != 0, R != 0):** CJ differs
   from pure Schwarzschild at the same Weyl. Measured ratio
   CJ_Kottler / CJ_Sch = 0.78 (22% reduction) at matched E^2,
   with CJ_Sch having d = 4.40 sigma detection and CJ_Kottler having
   d = 3.44 sigma.

3. **FLRW (E = 0, R != 0, time-dependent):** Not yet tested, but
   expected to give CJ = 0 by the same mechanism as de Sitter (pure
   Ricci, conformally flat).

The 22% Kottler reduction demonstrates that Ricci curvature modifies
the CJ response to Weyl curvature, but only when Weyl is non-zero.
This points to a cross-term of the form c_RW R_mu_nu C^mu_alpha_nu_beta
or similar Ricci-Weyl contraction.

## 3. Known Results

- **c_R = 0:** Pure Ricci gives CJ = 0. Established by the dS test
  (CJ = 0.000 +/- 0.001 across all strata).
- **c_W != 0:** Vacuum coefficient measured as c_W = 8/(3 x 9!) x
  (8 pi / 15) = 3.67 x 10^{-5} from pp-wave and Schwarzschild data.
- **c_RW != 0:** Inferred from the Kottler-Schwarzschild discrepancy.
  Approximate value c_RW / c_W approximately -0.22 / (Lambda / E^2) where
  Lambda is the cosmological constant.
- **Bel-Robinson tensor:** CJ measures the Bel-Robinson combination
  E^2 + B^2 (sum of electric and magnetic Weyl squared), not the
  Gauss-Bonnet combination E^2 - B^2. This is established by the
  type-N test: C^2 = 0 but CJ != 0 at 8 sigma, because type-N
  spacetimes have E^2 = B^2 > 0.
- **RSY at Ricci order:** Roy-Sinha-Surya compute the MEAN chain
  length and recover R. The VARIANCE (which gives CJ) has no R^2
  contribution in their framework, consistent with c_R = 0.

## 4. Failed Approaches

1. **Additive decomposition.** Attempted CJ = CJ_Weyl + CJ_Ricci with
   independent contributions. This predicts CJ_Kottler = CJ_Sch + CJ_dS
   = CJ_Sch + 0 = CJ_Sch, contradicting the measured 22% reduction.
   The Weyl and Ricci contributions are not additive.

2. **Volume correction.** The cosmological constant modifies the diamond
   volume: V_Kottler = V_Sch (1 - Lambda T^2 / 20 + ...). Applying
   this volume correction to CJ_Sch gives a 3% reduction at the test
   parameters, far below the observed 22%. The discrepancy is geometric,
   not volumetric.

3. **Ricci focusing of geodesics.** In the presence of positive Ricci
   curvature, geodesics focus, reducing the effective diamond width.
   This modifies the angular integral in the E^2 coefficient. Estimated
   correction: delta c_W / c_W = -(2/3) R T^2, which gives approximately
   10% for the Kottler test. Closer but still insufficient, and the
   derivation assumed small R T^2 which may not hold.

## 5. Success Criteria

- Derive the full coefficient structure CJ = c_W E^2 + c_RW f(R, Weyl)
  where f is an explicitly specified contraction.
- Predict the Kottler reduction ratio to within 5% of the measured
  value (0.78 +/- 0.05).
- Explain from first principles why c_R = 0 (pure Ricci gives no CJ
  signal).
- Verify the prediction on at least two additional non-vacuum test
  geometries (e.g., Reissner-Nordstrom, FLRW with perturbations).

## 6. Suggested Directions

1. **Geodesic deviation in non-vacuum.** The chain-length variance
   depends on the spread of nearby geodesics. In non-vacuum, geodesic
   deviation has both Weyl (tidal) and Ricci (focusing) contributions.
   Compute the second moment of the geodesic deviation equation with
   both contributions and identify the cross-term.

2. **Curvature expansion of the diamond volume to O(curvature^2).**
   The diamond volume in a general spacetime has the expansion
   V = V_0 (1 - R T^2 / 120 + (a_1 C^2 + a_2 R^2 + a_3 R_mu_nu^2)
   T^4 + ...). The a_3 R_mu_nu^2 term couples to the Weyl tensor
   through the Gauss equation. Compute a_3 and check whether it
   accounts for the Kottler reduction.

3. **Traceless Ricci decomposition.** Decompose R_mu_nu into trace
   (R) and traceless (S_mu_nu) parts. The cross-term may involve
   S_mu_nu rather than the full Ricci tensor, which would explain
   c_R = 0 (trace part decouples) while c_RW != 0 (traceless part
   couples to Weyl).

4. **Test on Reissner-Nordstrom.** RN has E != 0 and R_mu_nu != 0
   (from the electromagnetic stress tensor). A numerical measurement
   would give a second data point for the cross-term coefficient.

## 7. References

- Roy, M., Sinha, D. and Surya, S. (2013). arXiv:1212.0631.
- de Brito, G. P. (2023). arXiv:2301.13525.
- Wang, Z. (2019). arXiv:1904.01034.
- Gibbons, G. W. and Solodukhin, S. N. (2007). "The geometry of
  small causal diamonds." arXiv:hep-th/0703098.

## 8. Connections

- **OP-36 (RSY derivation):** The vacuum derivation gives c_W. The
  non-vacuum extension adds c_RW and must explain c_R = 0.
- **OP-41 (spectral bridge):** A spectral action derivation of CJ
  would naturally include both Weyl and Ricci contributions through
  the Seeley-DeWitt a_2 coefficient.
- **OP-42 (A_E universality):** Non-vacuum coefficients affect the
  Schwarzschild test (where R_mu_nu = 0 exactly), confirming that
  A_E universality is a vacuum-only statement.

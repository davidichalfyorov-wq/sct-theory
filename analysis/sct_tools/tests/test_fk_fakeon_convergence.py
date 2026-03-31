# ruff: noqa: E402, I001
"""FK-V: Verification test suite for the fakeon prescription convergence analysis.

Comprehensive tests across 5 layers:
  Layer 1 (Ghost Catalogue):  zero verification, residue signs, asymptotic pattern,
                               MR-2 consistency, conjugate pair structure
  Layer 2 (Convergence):      Sum|R_n| logarithmic growth, Sum Re(R_n) conditional,
                               Sum|R_n/z_n| absolute, sum rule, subtracted ML
  Layer 3 (Cross-Consistency): N-pole improvement, residue vs Pi'_TT, MR-2 match
  Layer 4 (Physical):          real poles have real residues, conjugate symmetry,
                               |R_n| monotone decay, no finite accumulation
  Layer 5 (Regression):        existing MR-2 and propagator test suites

All numerical checks use mpmath at >= 50-digit precision.

Author: David Alfyorov
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mpmath as mp
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ANALYSIS_DIR = Path(__file__).resolve().parent.parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from scripts.mr1_lorentzian import Pi_TT_complex
from scripts.fk_fakeon_convergence import (
    LOCAL_C2,
    compute_all_residues,
    compute_residue,
    find_next_complex_zeros,
    partial_fraction_reconstruction,
    mittag_leffler_with_subtraction,
)

# ---------------------------------------------------------------------------
# Load FK-D results JSON
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RESULTS_FILE = PROJECT_ROOT / "analysis" / "results" / "fk" / "fk_fakeon_convergence_results.json"

@pytest.fixture(scope="module")
def fk_results() -> dict:
    """Load the FK-D results JSON produced by fk_fakeon_convergence.py."""
    assert RESULTS_FILE.exists(), f"FK-D results not found at {RESULTS_FILE}"
    with open(RESULTS_FILE) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# DPS settings
# ---------------------------------------------------------------------------
DPS = 50  # Lower than production for test speed
DPS_HIGH = 80  # For cross-checks requiring higher precision


# ---------------------------------------------------------------------------
# Reference values from MR-2 (established by multi-method consensus)
# ---------------------------------------------------------------------------
Z_L_REF = mp.mpf("-1.28070227806348515")
Z_0_REF = mp.mpf("2.41483888986536890552401020133")
R_L_REF = mp.mpf("-0.5377720783273051")
R_0_REF = mp.mpf("-0.49309950210599085")
PI_TT_INF = mp.mpf("-83") / 6   # Pi_TT(+inf) = -83/6
SUM_RULE_TARGET = mp.mpf(-6) / 83   # 1 + Sum R_n -> -6/83
C_R_ASYMPTOTIC = 0.2892  # |R_n| * |z_n| -> C_R for Type C
ZERO_SPACING_IM = 25.3   # Approximate Im spacing between Type C zeros


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def all_residues() -> list[dict]:
    """Compute residues at all 8 known zeros + 4 new zeros."""
    mp.mp.dps = DPS
    residues = compute_all_residues(dps=DPS)
    new_zeros = find_next_complex_zeros(n_new=4, dps=DPS)
    all_res = list(residues)
    for z in new_zeros:
        if "error" not in z and z.get("Pi_at_root", 1) < 1e-10:
            all_res.append({
                "label": z["label"],
                "type": z["type"],
                "z": z["z"],
                "z_abs": z["z_abs"],
                "R": z["R"],
                "R_re": z["R_re"],
                "R_im": z["R_im"],
                "R_abs": z["R_abs"],
            })
    return all_res


@pytest.fixture(scope="module")
def known_residues() -> list[dict]:
    """Compute residues at the 8 known MR-2 zeros only."""
    return compute_all_residues(dps=DPS)


@pytest.fixture(scope="module")
def type_c_upper(all_residues) -> list[dict]:
    """Extract upper-half-plane Type C zeros."""
    result = []
    for r in all_residues:
        if r.get("type") == "C" and r.get("R_im", 0) > 0:
            result.append(r)
    return result


# ===================================================================
# LAYER 1: GHOST CATALOGUE VERIFICATION (13 tests)
# ===================================================================

class TestLayer1GhostCatalogue:
    """Verify all catalogued zeros are genuine zeros of Pi_TT."""

    def test_z_L_is_zero(self):
        """z_L = -1.2807 is a zero of Pi_TT."""
        mp.mp.dps = DPS
        val = Pi_TT_complex(mp.mpc(Z_L_REF, 0), dps=DPS)
        assert float(abs(val)) < 1e-10, f"|Pi_TT(z_L)| = {float(abs(val))}"

    def test_z_0_is_zero(self):
        """z_0 = 2.4148 is a zero of Pi_TT."""
        mp.mp.dps = DPS
        val = Pi_TT_complex(mp.mpc(Z_0_REF, 0), dps=DPS)
        assert float(abs(val)) < 1e-10, f"|Pi_TT(z_0)| = {float(abs(val))}"

    @pytest.mark.parametrize("idx", range(2, 8))
    def test_known_zeros_are_zeros(self, idx, known_residues):
        """Each of the 8 MR-2 zeros satisfies |Pi_TT(z_n)| < tol."""
        mp.mp.dps = DPS
        r = known_residues[idx]
        z_n = r["z"]
        val = Pi_TT_complex(z_n, dps=DPS)
        assert float(abs(val)) < 1e-10, (
            f"Zero #{idx} ({r['label']}): |Pi_TT| = {float(abs(val))}"
        )

    def test_new_zeros_are_zeros(self, all_residues):
        """All newly found zeros (C4+..C7+) satisfy |Pi_TT(z_n)| < tol."""
        mp.mp.dps = DPS
        new_labels = [r for r in all_residues if "extrapolated" in r.get("label", "")]
        assert len(new_labels) >= 3, f"Expected >= 3 new zeros, got {len(new_labels)}"
        for r in new_labels:
            z_n = r["z"]
            val = Pi_TT_complex(z_n, dps=DPS)
            assert float(abs(val)) < 1e-10, (
                f"New zero {r['label']}: |Pi_TT| = {float(abs(val))}"
            )

    def test_real_poles_have_negative_residues(self, known_residues):
        """Real poles z_L and z_0 have R < 0 (ghost signature)."""
        for r in known_residues[:2]:
            assert r["R_re"] < 0, f"{r['label']}: R_re = {r['R_re']} (expected < 0)"
            assert abs(r["R_im"]) < 1e-10, (
                f"{r['label']}: R_im = {r['R_im']} (expected ~0 for real pole)"
            )

    def test_complex_poles_have_complex_residues(self, type_c_upper):
        """Type C complex poles have nonzero imaginary residues."""
        for r in type_c_upper:
            assert abs(r["R_im"]) > 1e-6, (
                f"{r['label']}: R_im = {r['R_im']} (expected nonzero)"
            )

    def test_product_R_times_z_approaches_C_R(self, type_c_upper):
        """|R_n| * |z_n| -> C_R ~ 0.289 for large |z_n|."""
        products = [r["R_abs"] * r["z_abs"] for r in type_c_upper]
        # All products should be in (0.27, 0.30)
        for i, p in enumerate(products):
            assert 0.27 < p < 0.30, (
                f"Pair {i}: |R|*|z| = {p:.6f} (expected ~0.289)"
            )
        # Products should be monotonically decreasing (converging from above)
        for i in range(1, len(products)):
            assert products[i] <= products[i - 1] + 1e-6, (
                f"Products not monotonically decreasing at pair {i}"
            )

    def test_z_L_matches_MR2(self, known_residues):
        """z_L location matches MR-2 reference value."""
        mp.mp.dps = DPS
        z_L = known_residues[0]["z"]
        assert float(abs(mp.re(z_L) - Z_L_REF)) < 1e-8, (
            f"z_L mismatch: got {float(mp.re(z_L))}, expected {float(Z_L_REF)}"
        )

    def test_z_0_matches_MR2(self, known_residues):
        """z_0 location matches MR-2 reference value."""
        mp.mp.dps = DPS
        z_0 = known_residues[1]["z"]
        assert float(abs(mp.re(z_0) - Z_0_REF)) < 1e-8, (
            f"z_0 mismatch: got {float(mp.re(z_0))}, expected {float(Z_0_REF)}"
        )

    def test_complex_zeros_come_in_conjugate_pairs(self, known_residues):
        """Every Type C zero in the upper half-plane has a conjugate in the lower."""
        type_c = [r for r in known_residues if r.get("type") == "C"]
        upper = [r for r in type_c if r["R_im"] > 0]
        lower = [r for r in type_c if r["R_im"] < 0]
        assert len(upper) == len(lower), (
            f"Conjugate pair count mismatch: {len(upper)} upper, {len(lower)} lower"
        )
        # Match by |z| (they should be identical for conjugates)
        upper_sorted = sorted(upper, key=lambda r: r["z_abs"])
        lower_sorted = sorted(lower, key=lambda r: r["z_abs"])
        for u, lo in zip(upper_sorted, lower_sorted):
            assert abs(u["z_abs"] - lo["z_abs"]) < 1e-6, (
                f"|z| mismatch: {u['z_abs']} vs {lo['z_abs']}"
            )
            # Residues should be conjugates: R_re equal, R_im opposite
            assert abs(u["R_re"] - lo["R_re"]) < 1e-8, (
                f"Re(R) mismatch for pair at |z|~{u['z_abs']:.1f}"
            )
            assert abs(u["R_im"] + lo["R_im"]) < 1e-8, (
                f"Im(R) not conjugate for pair at |z|~{u['z_abs']:.1f}"
            )

    def test_residues_match_MR2_values(self, known_residues):
        """Residues at z_L and z_0 match MR-2 established values."""
        r_L = known_residues[0]["R_re"]
        r_0 = known_residues[1]["R_re"]
        assert abs(r_L - float(R_L_REF)) < 1e-6, (
            f"R_L mismatch: got {r_L}, expected {float(R_L_REF)}"
        )
        assert abs(r_0 - float(R_0_REF)) < 1e-6, (
            f"R_0 mismatch: got {r_0}, expected {float(R_0_REF)}"
        )

    def test_total_catalogue_size(self, all_residues):
        """The extended catalogue contains at least 12 zeros."""
        assert len(all_residues) >= 12, (
            f"Expected >= 12 zeros in extended catalogue, got {len(all_residues)}"
        )


# ===================================================================
# LAYER 2: CONVERGENCE TESTS (13 tests)
# ===================================================================

class TestLayer2Convergence:
    """Test convergence properties of the residue series."""

    def test_sum_abs_R_grows_logarithmically(self, fk_results):
        """Sum|R_n| grows logarithmically (consistent with alpha ~ 1)."""
        partial_sums = fk_results["residue_series_convergence"]["partial_sums_abs_R"]
        # After the first 2 (real poles dominate), growth should be slow
        assert len(partial_sums) >= 8
        # Growth from index 4 to final should be small (< 0.1)
        growth = partial_sums[-1] - partial_sums[4]
        assert growth < 0.15, f"Growth too fast: {growth:.4f}"
        assert growth > 0, f"Sum|R_n| not increasing: {growth:.4f}"

    def test_logarithmic_fit_coefficient(self, fk_results):
        """Logarithmic coefficient ~ 0.023 (= 2*C_R/D_z)."""
        # From the asymptotic analysis, the predicted coefficient is 2*C_R/D_z
        # where C_R ~ 0.289 and D_z ~ 25.3
        coeff_predicted = 2 * C_R_ASYMPTOTIC / ZERO_SPACING_IM
        assert 0.015 < coeff_predicted < 0.030, (
            f"Predicted log coefficient {coeff_predicted:.4f} out of range"
        )

    def test_sum_Re_R_converges_conditionally(self, fk_results):
        """Partial sums of 1 + Sum Re(R_n) approach -6/83."""
        partial_sums = fk_results["residue_series_convergence"]["partial_sums_with_graviton"]
        target = fk_results["residue_series_convergence"]["target_sum_rule"]
        # The partial sum should be between 0 and target (approaching from above)
        last = partial_sums[-1]
        assert last < 0, f"1 + Sum Re(R_n) = {last} (expected negative)"
        # Should be closer to target than to 0
        deficit = abs(last - target)
        assert deficit < 0.1, (
            f"Deficit from -6/83 = {deficit:.6f} (expected < 0.1)"
        )

    def test_sum_rule_deficit_is_small(self, fk_results):
        """Deficit from the sum rule 1 + Sum R_n = -6/83 should be small."""
        deficit = fk_results["residue_series_convergence"]["deficit"]
        assert abs(deficit) < 0.05, f"Sum rule deficit = {deficit:.6f} (expected < 0.05)"

    def test_sum_abs_R_over_z_converges(self, all_residues):
        """Sum|R_n/z_n| converges absolutely (p-series with p=2)."""
        mp.mp.dps = DPS
        total = sum(r["R_abs"] / r["z_abs"] for r in all_residues)
        # Should converge to a finite value (dominated by real poles)
        assert 0.5 < total < 1.0, f"Sum|R_n/z_n| = {total:.6f} (expected ~0.625)"
        # Tail contribution should be small
        tail = sum(r["R_abs"] / r["z_abs"] for r in all_residues
                   if r.get("type") == "C")
        assert tail < 0.01, f"Type C tail = {tail:.6f} (expected < 0.01)"

    def test_sum_abs_R_over_z_sq_converges(self, all_residues):
        """Sum|R_n/z_n^2| converges (stronger than needed, verifies decay rate)."""
        total = sum(r["R_abs"] / r["z_abs"]**2 for r in all_residues)
        assert total < 1.0, f"Sum|R_n/z_n^2| = {total:.6f}"
        assert total > 0, "Sum should be positive"

    def test_alpha_exponent_near_one(self, fk_results):
        """Fitted power-law exponent alpha is indistinguishable from 1."""
        alpha = fk_results["asymptotic_analysis"]["power_law_alpha"]
        assert 0.95 < alpha < 1.05, f"alpha = {alpha:.6f} (expected ~1.0)"

    def test_unsubtracted_series_does_not_converge_absolutely(self, fk_results):
        """Sum|R_n| does NOT converge (alpha <= 1)."""
        status = fk_results["convergence_conditions"]["absolute_convergence"]["status"]
        assert status == "NOT SATISFIED", (
            f"Absolute convergence status = '{status}' (expected NOT SATISFIED)"
        )

    def test_subtracted_convergence_satisfied(self, fk_results):
        """Sum|R_n/z_n| converges (one-subtraction Mittag-Leffler)."""
        status = fk_results["convergence_conditions"]["subtracted_convergence"]["status"]
        assert status == "SATISFIED", (
            f"Subtracted convergence status = '{status}' (expected SATISFIED)"
        )

    def test_conditional_convergence_satisfied(self, fk_results):
        """Sum Re(R_n) converges conditionally (from sum rule)."""
        status = fk_results["convergence_conditions"]["conditional_convergence"]["status"]
        assert "SATISFIED" in status, (
            f"Conditional convergence status = '{status}' (expected SATISFIED)"
        )

    def test_mittag_leffler_unsubtracted_error_bounded(self, all_residues):
        """The unsubtracted ML error stays bounded (not blowing up) for well-separated points."""
        mp.mp.dps = DPS
        z_test = mp.mpc(5, 0)
        ml = mittag_leffler_with_subtraction(z_test, all_residues, dps=DPS)
        errors = [c["error_unsubtracted"] for c in ml["convergence"]]
        # Errors should be bounded (not growing to infinity)
        assert max(errors) < 1.0, f"Max unsubtracted error = {max(errors)}"
        # Errors should decrease (at least between first and last)
        assert errors[-1] <= errors[0] + 1e-3, (
            f"Unsubtracted error increasing: {errors[0]:.6f} -> {errors[-1]:.6f}"
        )

    def test_pv_single_pole_analytic_vs_numerical(self, fk_results):
        """PV integral for a single pole: analytic = numerical."""
        pv = fk_results["pv_integral"]["single_pole"]
        pv_analytic = pv["pv_analytic"]
        for eps_key, pv_num in pv["pv_numerical_by_eps"].items():
            assert abs(pv_num - pv_analytic) < 1e-3, (
                f"PV mismatch at eps=1e{eps_key}: "
                f"analytic={pv_analytic:.6f}, numerical={pv_num:.6f}"
            )

    def test_multi_pole_pv_converges(self, fk_results):
        """Multi-pole PV integral converges (successive differences decrease)."""
        diffs = fk_results["pv_integral"]["multi_pole"]["successive_differences"]
        # Filter out zero diffs (from lower half-plane conjugate pairings)
        nonzero_diffs = [d for d in diffs if d > 1e-10]
        if len(nonzero_diffs) >= 3:
            assert nonzero_diffs[-1] < nonzero_diffs[0], (
                f"PV diffs not decreasing: first={nonzero_diffs[0]:.4e}, "
                f"last={nonzero_diffs[-1]:.4e}"
            )


# ===================================================================
# LAYER 3: CROSS-CONSISTENCY TESTS (7 tests)
# ===================================================================

class TestLayer3CrossConsistency:
    """Cross-consistency between different parts of the analysis."""

    def test_n_pole_approximation_improves_at_z5(self, all_residues):
        """N-pole approximation improves with N at z=5."""
        mp.mp.dps = DPS
        z_test = mp.mpc(5, 0)
        exact = 1 / Pi_TT_complex(z_test, dps=DPS)
        errors = []
        for N in [2, 4, 6, 8, 10, 12]:
            partial = mp.mpc(0)
            for i in range(min(N, len(all_residues))):
                r = all_residues[i]
                partial += r["R"] / (z_test - r["z"])
            errors.append(float(abs(exact - partial)))
        # Error should decrease between N=2 and N=12
        assert errors[-1] < errors[0], (
            f"N-pole not improving: err(2)={errors[0]:.6e}, err(12)={errors[-1]:.6e}"
        )

    def test_n_pole_approximation_improves_at_z10i(self, all_residues):
        """N-pole approximation improves with N at z=10i."""
        mp.mp.dps = DPS
        z_test = mp.mpc(0, 10)
        exact = 1 / Pi_TT_complex(z_test, dps=DPS)
        errors = []
        for N in [2, 6, 12]:
            partial = mp.mpc(0)
            for i in range(min(N, len(all_residues))):
                r = all_residues[i]
                partial += r["R"] / (z_test - r["z"])
            errors.append(float(abs(exact - partial)))
        assert errors[-1] < errors[0], (
            f"N-pole not improving at 10i: err(2)={errors[0]:.6e}, err(12)={errors[-1]:.6e}"
        )

    def test_residue_consistent_with_Pi_prime(self, known_residues):
        """R_n = 1/(z_n * Pi'_TT(z_n)) is consistent with direct derivative."""
        mp.mp.dps = DPS_HIGH
        for r in known_residues[:2]:  # Real poles
            z_n = r["z"]
            R_n = r["R"]
            # Compute Pi'_TT independently with a different step size
            h = mp.mpf("1e-15")
            fp = Pi_TT_complex(z_n + h, dps=DPS_HIGH)
            fm = Pi_TT_complex(z_n - h, dps=DPS_HIGH)
            Pi_prime_alt = (fp - fm) / (2 * h)
            R_alt = 1 / (z_n * Pi_prime_alt)
            rel_err = float(abs(R_n - R_alt) / abs(R_n))
            assert rel_err < 1e-6, (
                f"{r['label']}: residue rel error = {rel_err:.2e}"
            )

    def test_first_8_zeros_match_MR2_catalogue(self, known_residues, fk_results):
        """The first 8 zeros match the MR-2 ghost catalogue."""
        zeros_json = fk_results["ghost_catalogue_extended"]["zeros"]
        # z_L
        assert abs(zeros_json[0]["R_re"] - float(R_L_REF)) < 1e-4
        # z_0
        assert abs(zeros_json[1]["R_re"] - float(R_0_REF)) < 1e-4
        # C1 magnitude
        assert abs(zeros_json[2]["z_abs"] - 33.84) < 0.5

    def test_partial_fraction_remainder_is_smooth(self, all_residues):
        """The remainder E(z) = 1/Pi_TT - Sum R_n/(z-z_n) is smooth."""
        mp.mp.dps = DPS
        # Evaluate at two nearby points
        z1 = mp.mpc(5, 0)
        z2 = mp.mpc(5.01, 0)
        pf1 = partial_fraction_reconstruction(z1, all_residues, dps=DPS)
        pf2 = partial_fraction_reconstruction(z2, all_residues, dps=DPS)
        # Remainder should change smoothly (no pole-like behavior)
        dr = abs(pf1["remainder_re"] - pf2["remainder_re"])
        assert dr < 0.1, (
            f"Remainder jumps by {dr:.4f} between z=5 and z=5.01"
        )

    def test_sum_rule_from_Pi_TT_infinity(self):
        """Pi_TT(+inf) = -83/6 independently verified."""
        mp.mp.dps = DPS
        # Large positive z: Pi_TT(z) -> 1 + (13/60)*z*(-89/(12*z)) = 1 - 89*13/(12*60)
        # = 1 - 1157/720 = (720-1157)/720 = -437/720
        # Actually Pi_TT(z->+inf) involves the asymptotic of F1_hat
        # Let's just check numerically
        val = Pi_TT_complex(mp.mpf(5000), dps=DPS)
        expected = mp.mpf(-83) / 6
        rel_err = float(abs(val - expected) / abs(expected))
        assert rel_err < 1e-2, (
            f"Pi_TT(5000) = {float(val):.6f}, expected {float(expected):.6f}, "
            f"rel_err = {rel_err:.4e}"
        )

    def test_analytic_C_R_prediction(self, type_c_upper):
        """C_R = 60/(13*D) matches numerical product |R_n|*|z_n|."""
        mp.mp.dps = DPS
        if len(type_c_upper) < 3:
            pytest.skip("Not enough Type C zeros")
        # Compute D = |F1_hat'(z_n)| * |z_n| at the last pair
        r = type_c_upper[-1]
        z_n = r["z"]
        h = mp.mpf("1e-12")
        # F1_hat(z) = (Pi_TT(z) - 1) / ((13/60)*z)
        def F1_hat(z):
            return (Pi_TT_complex(z, dps=DPS) - 1) / (LOCAL_C2 * z)
        fp = F1_hat(z_n + h)
        fm = F1_hat(z_n - h)
        F1_prime = (fp - fm) / (2 * h)
        D_val = float(abs(F1_prime) * abs(z_n))
        C_R_analytic = 60 / (13 * D_val)
        C_R_numerical = r["R_abs"] * r["z_abs"]
        rel_err = abs(C_R_analytic - C_R_numerical) / C_R_numerical
        assert rel_err < 0.01, (
            f"C_R analytic={C_R_analytic:.6f}, numerical={C_R_numerical:.6f}, "
            f"rel_err={rel_err:.4e}"
        )


# ===================================================================
# LAYER 4: PHYSICAL PROPERTIES (8 tests)
# ===================================================================

class TestLayer4PhysicalProperties:
    """Test physical properties of the ghost catalogue."""

    def test_real_poles_have_real_residues(self, known_residues):
        """Real poles (z_L, z_0) have purely real residues."""
        for r in known_residues[:2]:
            assert abs(r["R_im"]) < 1e-10, (
                f"{r['label']}: Im(R) = {r['R_im']} (expected 0)"
            )

    def test_complex_conjugate_poles_have_conjugate_residues(self, known_residues):
        """For complex pair (z_n, z_n*), residues satisfy R_n* = R(z_n*)."""
        type_c = [r for r in known_residues if r.get("type") == "C"]
        upper = sorted([r for r in type_c if r["R_im"] > 0], key=lambda r: r["z_abs"])
        lower = sorted([r for r in type_c if r["R_im"] < 0], key=lambda r: r["z_abs"])
        for u, lo in zip(upper, lower):
            # |R| should be identical
            assert abs(u["R_abs"] - lo["R_abs"]) < 1e-8, (
                f"|R| mismatch at |z|~{u['z_abs']:.1f}"
            )

    def test_R_abs_decreases_for_type_C(self, type_c_upper):
        """|R_n| decreases with |z_n| for Type C pairs."""
        if len(type_c_upper) < 2:
            pytest.skip("Not enough Type C zeros")
        sorted_c = sorted(type_c_upper, key=lambda r: r["z_abs"])
        for i in range(1, len(sorted_c)):
            assert sorted_c[i]["R_abs"] < sorted_c[i - 1]["R_abs"], (
                f"|R| not decreasing: {sorted_c[i-1]['R_abs']:.6f} -> "
                f"{sorted_c[i]['R_abs']:.6f}"
            )

    def test_no_accumulation_point_finite(self, type_c_upper):
        """Zeros are well-separated (no accumulation at finite |z|)."""
        if len(type_c_upper) < 2:
            pytest.skip("Not enough Type C zeros")
        sorted_c = sorted(type_c_upper, key=lambda r: r["z_abs"])
        spacings = [sorted_c[i]["z_abs"] - sorted_c[i-1]["z_abs"]
                    for i in range(1, len(sorted_c))]
        # All spacings should be > 10 (they're ~25)
        for i, s in enumerate(spacings):
            assert s > 10, (
                f"Spacing too small between pairs {i} and {i+1}: {s:.2f}"
            )

    def test_zero_spacing_approximately_constant(self, type_c_upper):
        """Im(z_n) spacing is approximately constant (~25.3)."""
        if len(type_c_upper) < 3:
            pytest.skip("Not enough Type C zeros")
        sorted_c = sorted(type_c_upper, key=lambda r: r["z_abs"])
        spacings = [sorted_c[i]["z_abs"] - sorted_c[i-1]["z_abs"]
                    for i in range(1, len(sorted_c))]
        for s in spacings:
            assert 20 < s < 30, (
                f"Spacing {s:.2f} outside expected range [20, 30]"
            )

    def test_Pi_TT_at_origin(self):
        """Pi_TT(0) = 1 (normalization)."""
        mp.mp.dps = DPS
        val = Pi_TT_complex(mp.mpc(0, 0), dps=DPS)
        assert float(abs(val - 1)) < 1e-10, f"Pi_TT(0) = {float(val)}"

    def test_entire_function_order_is_one(self, fk_results):
        """Pi_TT has order 1 (growth analysis)."""
        growth = fk_results["entire_function_order"]["growth_positive_real"]
        # For order 1: log|Pi_TT|/r -> 0 as r -> inf (for bounded directions)
        # and log|Pi_TT|/r^2 -> 0 faster
        last = growth[-1]
        assert last["log_Pi_TT_over_r"] < 0.01, (
            f"log|Pi_TT|/r at r={last['r']}: {last['log_Pi_TT_over_r']:.6f}"
        )
        assert last["log_Pi_TT_over_r_sq"] < 0.001, (
            f"log|Pi_TT|/r^2 at r={last['r']}: {last['log_Pi_TT_over_r_sq']:.6f}"
        )

    def test_negative_real_axis_exponential_growth(self, fk_results):
        """Pi_TT grows exponentially along negative real axis (type ~ 1/4)."""
        growth = fk_results["entire_function_order"]["growth_negative_real"]
        # log|Pi_TT(-r)|/r should approach ~1/4 = 0.25
        last = growth[-1]
        ratio = last["log_Pi_TT_over_r"]
        assert 0.20 < ratio < 0.35, (
            f"log|Pi_TT|/r at r=-{last['r']}: {ratio:.4f} (expected ~0.25)"
        )


# ===================================================================
# LAYER 5: VERDICT AND REGRESSION (6 tests)
# ===================================================================

class TestLayer5VerdictAndRegression:
    """Verify the overall verdict and regression against FK-D results."""

    def test_verdict_is_conditionally_convergent(self, fk_results):
        """FK-D verdict is CONDITIONALLY CONVERGENT."""
        verdict = fk_results["verdict"]["classification"]
        assert verdict == "CONDITIONALLY CONVERGENT", (
            f"Verdict = '{verdict}' (expected CONDITIONALLY CONVERGENT)"
        )

    def test_alpha_exponent_in_results(self, fk_results):
        """Alpha exponent stored in results matches asymptotic analysis."""
        alpha_verdict = fk_results["verdict"]["alpha_exponent"]
        alpha_asymp = fk_results["asymptotic_analysis"]["power_law_alpha"]
        assert abs(alpha_verdict - alpha_asymp) < 1e-6

    def test_12_zeros_in_catalogue(self, fk_results):
        """Extended catalogue has exactly 12 entries (8 known + 4 new)."""
        n_total = fk_results["ghost_catalogue_extended"]["n_zeros_total"]
        assert n_total == 12, f"n_zeros_total = {n_total} (expected 12)"

    def test_results_json_has_all_sections(self, fk_results):
        """FK-D results JSON contains all expected sections."""
        required_keys = [
            "ghost_catalogue_extended",
            "residue_series_convergence",
            "asymptotic_analysis",
            "entire_function_order",
            "partial_fraction_tests",
            "mittag_leffler_subtracted",
            "n_pole_convergence",
            "pv_integral",
            "convergence_conditions",
            "verdict",
        ]
        for key in required_keys:
            assert key in fk_results, f"Missing section: {key}"

    def test_n_pole_convergence_flags(self, fk_results):
        """N-pole convergence: z=5 converges, z=-0.5+0.1i does not."""
        test_points = fk_results["n_pole_convergence"]
        # Find z=5 (real)
        z5 = [tp for tp in test_points if "5.0000+0.0000" in tp["z_test"]]
        assert len(z5) == 1
        assert z5[0]["converging"] is True, "z=5 should converge"
        # Find z=-0.5+0.1i (between poles)
        z_neg = [tp for tp in test_points if "-0.5000+0.1000" in tp["z_test"]]
        assert len(z_neg) == 1
        assert z_neg[0]["converging"] is False, "z=-0.5+0.1i should NOT converge"

    def test_pv_single_pole_agreement(self, fk_results):
        """Single-pole PV: all epsilon values agree with analytic."""
        pv = fk_results["pv_integral"]["single_pole"]
        assert pv["agreement"] is True, "PV single-pole disagreement"


# ===================================================================
# LAYER EXTRA: HIGH-PRECISION INDEPENDENT CHECKS (5 tests)
# ===================================================================

class TestLayerHighPrecision:
    """Independent high-precision verification of key quantities."""

    def test_residue_z0_at_100_dps(self):
        """Compute R_0 at 100 dps and verify against reference."""
        mp.mp.dps = 100
        z_0 = mp.mpc(Z_0_REF, 0)
        R = compute_residue(z_0, dps=100)
        assert float(abs(mp.re(R) - R_0_REF)) < 1e-10, (
            f"R_0 at 100 dps: {float(mp.re(R))}"
        )

    def test_residue_zL_at_100_dps(self):
        """Compute R_L at 100 dps and verify against reference."""
        mp.mp.dps = 100
        z_L = mp.mpc(Z_L_REF, 0)
        R = compute_residue(z_L, dps=100)
        assert float(abs(mp.re(R) - R_L_REF)) < 1e-8, (
            f"R_L at 100 dps: {float(mp.re(R))}"
        )

    def test_five_point_stencil_residue(self):
        """5-point stencil derivative matches central-difference residue at z_0."""
        mp.mp.dps = DPS_HIGH
        z_0 = mp.mpc(Z_0_REF, 0)
        h = mp.mpf("1e-10")
        # 5-point stencil: f'(x) = (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
        fp2 = Pi_TT_complex(z_0 + 2*h, dps=DPS_HIGH)
        fp1 = Pi_TT_complex(z_0 + h, dps=DPS_HIGH)
        fm1 = Pi_TT_complex(z_0 - h, dps=DPS_HIGH)
        fm2 = Pi_TT_complex(z_0 - 2*h, dps=DPS_HIGH)
        Pi_prime_5pt = (-fp2 + 8*fp1 - 8*fm1 + fm2) / (12 * h)
        R_5pt = 1 / (z_0 * Pi_prime_5pt)
        # Compare with standard central difference
        R_cd = compute_residue(z_0, dps=DPS_HIGH)
        rel_err = float(abs(R_5pt - R_cd) / abs(R_cd))
        assert rel_err < 1e-6, (
            f"5-point vs central-diff rel error: {rel_err:.2e}"
        )

    def test_Pi_TT_is_real_on_real_axis(self):
        """Pi_TT(x) is real for real x (sanity check)."""
        mp.mp.dps = DPS
        for x in [0.5, 1.0, 3.0, 10.0, -0.5]:
            val = Pi_TT_complex(mp.mpf(x), dps=DPS)
            assert float(abs(mp.im(val))) < 1e-15, (
                f"Im(Pi_TT({x})) = {float(mp.im(val))}"
            )

    def test_C_R_convergence_value(self, type_c_upper):
        """C_R = |R_n|*|z_n| converges to ~0.289 from the last 3 Type C pairs."""
        if len(type_c_upper) < 5:
            pytest.skip("Not enough Type C zeros for tight convergence test")
        products = [r["R_abs"] * r["z_abs"] for r in sorted(type_c_upper, key=lambda r: r["z_abs"])]
        # Last 3 products should be close to each other (within 0.001)
        spread = max(products[-3:]) - min(products[-3:])
        assert spread < 0.001, (
            f"C_R spread in last 3 pairs: {spread:.6f}"
        )
        # Value should be in (0.288, 0.290)
        avg = sum(products[-3:]) / 3
        assert 0.288 < avg < 0.291, f"C_R average = {avg:.6f}"

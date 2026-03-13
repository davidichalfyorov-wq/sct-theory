"""
Audit Round 7 — Guard consistency across all function variants.

Validates that EVERY variant of a form-factor function (original/quad,
_fast, _mp) has identical input-validation behavior: raises ValueError
on NaN, Inf, and x < 0.  Also tests the three fitting.py gaps found
in Round 7 (anderson_darling_test, bayesian_limit, model_comparison).

Created: 2026-03-10
"""

import numpy as np
import pytest

from sct_tools.fitting import (
    anderson_darling_test,
    bayesian_limit,
    model_comparison,
)
from sct_tools.form_factors import (
    f_Omega,
    f_R,
    f_Ric,
    f_RU,
    f_U,
    hC_dirac,
    hC_dirac_fast,
    hC_scalar,
    hC_scalar_fast,
    hR_dirac,
    hR_dirac_fast,
    hR_scalar,
    hR_scalar_fast,
    phi,
    phi_closed,
    phi_fast,
)

# Lazy-import _mp functions since they pull in mpmath
_mp_funcs_loaded = False
_phi_mp = None
_hC_scalar_mp = None
_hR_scalar_mp = None
_hC_dirac_mp = None
_hR_dirac_mp = None


def _load_mp():
    global _mp_funcs_loaded, _phi_mp, _hC_scalar_mp, _hR_scalar_mp
    global _hC_dirac_mp, _hR_dirac_mp
    if not _mp_funcs_loaded:
        from sct_tools.form_factors import (
            hC_dirac_mp,
            hC_scalar_mp,
            hR_dirac_mp,
            hR_scalar_mp,
            phi_mp,
        )
        _phi_mp = phi_mp
        _hC_scalar_mp = hC_scalar_mp
        _hR_scalar_mp = hR_scalar_mp
        _hC_dirac_mp = hC_dirac_mp
        _hR_dirac_mp = hR_dirac_mp
        _mp_funcs_loaded = True


# ============================================================================
# ORIGINAL (quad-based) — NaN / Inf rejection
# ============================================================================


class TestPhiOriginalGuards:
    """phi (quad-based) must reject NaN and Inf."""

    def test_nan(self):
        with pytest.raises(ValueError, match="finite"):
            phi(float("nan"))

    def test_inf(self):
        with pytest.raises(ValueError, match="finite"):
            phi(float("inf"))

    def test_neg_inf(self):
        with pytest.raises(ValueError, match="finite"):
            phi(float("-inf"))

    def test_valid(self):
        result = phi(1.0)
        assert np.isfinite(result)


class TestPhiClosedGuards:
    """phi_closed must reject NaN and Inf."""

    def test_nan(self):
        with pytest.raises(ValueError, match="finite"):
            phi_closed(float("nan"))

    def test_inf(self):
        with pytest.raises(ValueError, match="finite"):
            phi_closed(float("inf"))

    def test_valid(self):
        result = phi_closed(1.0)
        assert np.isfinite(result)


class TestFRicGuards:
    def test_nan(self):
        with pytest.raises(ValueError, match="finite"):
            f_Ric(float("nan"))

    def test_inf(self):
        with pytest.raises(ValueError, match="finite"):
            f_Ric(float("inf"))

    def test_valid(self):
        result = f_Ric(1.0)
        assert np.isfinite(result)


class TestFRGuards:
    def test_nan(self):
        with pytest.raises(ValueError, match="finite"):
            f_R(float("nan"))

    def test_inf(self):
        with pytest.raises(ValueError, match="finite"):
            f_R(float("inf"))

    def test_valid(self):
        result = f_R(1.0)
        assert np.isfinite(result)


class TestFRUGuards:
    def test_nan(self):
        with pytest.raises(ValueError, match="finite"):
            f_RU(float("nan"))

    def test_inf(self):
        with pytest.raises(ValueError, match="finite"):
            f_RU(float("inf"))

    def test_valid(self):
        result = f_RU(1.0)
        assert np.isfinite(result)


class TestFUGuards:
    def test_nan(self):
        with pytest.raises(ValueError, match="finite"):
            f_U(float("nan"))

    def test_inf(self):
        with pytest.raises(ValueError, match="finite"):
            f_U(float("inf"))

    def test_valid(self):
        result = f_U(1.0)
        assert np.isfinite(result)


class TestFOmegaGuards:
    def test_nan(self):
        with pytest.raises(ValueError, match="finite"):
            f_Omega(float("nan"))

    def test_inf(self):
        with pytest.raises(ValueError, match="finite"):
            f_Omega(float("inf"))

    def test_valid(self):
        result = f_Omega(1.0)
        assert np.isfinite(result)


class TestHCScalarOriginalGuards:
    def test_nan(self):
        with pytest.raises(ValueError, match="finite"):
            hC_scalar(float("nan"))

    def test_inf(self):
        with pytest.raises(ValueError, match="finite"):
            hC_scalar(float("inf"))

    def test_valid(self):
        result = hC_scalar(1.0)
        assert np.isfinite(result)


class TestHCDiracOriginalGuards:
    def test_nan(self):
        with pytest.raises(ValueError, match="finite"):
            hC_dirac(float("nan"))

    def test_inf(self):
        with pytest.raises(ValueError, match="finite"):
            hC_dirac(float("inf"))

    def test_valid(self):
        result = hC_dirac(1.0)
        assert np.isfinite(result)


class TestHRDiracOriginalGuards:
    def test_nan(self):
        with pytest.raises(ValueError, match="finite"):
            hR_dirac(float("nan"))

    def test_inf(self):
        with pytest.raises(ValueError, match="finite"):
            hR_dirac(float("inf"))

    def test_valid(self):
        result = hR_dirac(1.0)
        assert np.isfinite(result)


class TestHRScalarOriginalGuards:
    """hR_scalar (quad) must reject NaN in both x and xi."""

    def test_nan_x(self):
        with pytest.raises(ValueError, match="finite"):
            hR_scalar(float("nan"), xi=0.0)

    def test_inf_x(self):
        with pytest.raises(ValueError, match="finite"):
            hR_scalar(float("inf"), xi=0.0)

    def test_valid(self):
        result = hR_scalar(1.0, xi=0.0)
        assert np.isfinite(result)


# ============================================================================
# CROSS-VARIANT CONSISTENCY — same input, same guard behavior
# ============================================================================


class TestCrossVariantPhiConsistency:
    """phi, phi_closed, phi_fast all must raise on NaN."""

    @pytest.mark.parametrize("func", [phi, phi_closed, phi_fast])
    def test_nan_raises(self, func):
        with pytest.raises(ValueError, match="finite"):
            func(float("nan"))

    @pytest.mark.parametrize("func", [phi, phi_closed, phi_fast])
    def test_inf_raises(self, func):
        with pytest.raises(ValueError, match="finite"):
            func(float("inf"))


class TestCrossVariantHCScalarConsistency:
    @pytest.mark.parametrize("func", [hC_scalar, hC_scalar_fast])
    def test_nan_raises(self, func):
        with pytest.raises(ValueError, match="finite"):
            func(float("nan"))

    @pytest.mark.parametrize("func", [hC_scalar, hC_scalar_fast])
    def test_inf_raises(self, func):
        with pytest.raises(ValueError, match="finite"):
            func(float("inf"))


class TestCrossVariantHCDiracConsistency:
    @pytest.mark.parametrize("func", [hC_dirac, hC_dirac_fast])
    def test_nan_raises(self, func):
        with pytest.raises(ValueError, match="finite"):
            func(float("nan"))

    @pytest.mark.parametrize("func", [hC_dirac, hC_dirac_fast])
    def test_inf_raises(self, func):
        with pytest.raises(ValueError, match="finite"):
            func(float("inf"))


class TestCrossVariantHRDiracConsistency:
    @pytest.mark.parametrize("func", [hR_dirac, hR_dirac_fast])
    def test_nan_raises(self, func):
        with pytest.raises(ValueError, match="finite"):
            func(float("nan"))

    @pytest.mark.parametrize("func", [hR_dirac, hR_dirac_fast])
    def test_inf_raises(self, func):
        with pytest.raises(ValueError, match="finite"):
            func(float("inf"))


class TestCrossVariantHRScalarConsistency:
    @pytest.mark.parametrize("func", [hR_scalar, hR_scalar_fast])
    def test_nan_x(self, func):
        with pytest.raises(ValueError, match="finite"):
            func(float("nan"), xi=0.0)

    @pytest.mark.parametrize("func", [hR_scalar, hR_scalar_fast])
    def test_nan_xi(self, func):
        with pytest.raises(ValueError, match="finite"):
            func(1.0, xi=float("nan"))


# ============================================================================
# _mp FUNCTIONS — NaN / Inf / x < 0 rejection
# ============================================================================


class TestPhiMpGuards:
    def test_nan(self):
        _load_mp()
        with pytest.raises(ValueError, match="finite"):
            _phi_mp(float("nan"))

    def test_inf(self):
        _load_mp()
        with pytest.raises(ValueError, match="finite"):
            _phi_mp(float("inf"))

    def test_neg_inf(self):
        _load_mp()
        with pytest.raises(ValueError, match="finite"):
            _phi_mp(float("-inf"))

    def test_negative_x(self):
        _load_mp()
        with pytest.raises(ValueError, match="x >= 0"):
            _phi_mp(-1.0)

    def test_valid(self):
        _load_mp()
        result = _phi_mp(1.0)
        assert float(result) > 0


class TestHCScalarMpGuards:
    def test_nan(self):
        _load_mp()
        with pytest.raises(ValueError, match="finite"):
            _hC_scalar_mp(float("nan"))

    def test_inf(self):
        _load_mp()
        with pytest.raises(ValueError, match="finite"):
            _hC_scalar_mp(float("inf"))

    def test_negative_x(self):
        _load_mp()
        with pytest.raises(ValueError, match="x >= 0"):
            _hC_scalar_mp(-1.0)

    def test_valid(self):
        _load_mp()
        result = _hC_scalar_mp(1.0)
        assert np.isfinite(float(result))


class TestHRScalarMpGuards:
    def test_nan_x(self):
        _load_mp()
        with pytest.raises(ValueError, match="finite"):
            _hR_scalar_mp(float("nan"), xi=0.0)

    def test_inf_x(self):
        _load_mp()
        with pytest.raises(ValueError, match="finite"):
            _hR_scalar_mp(float("inf"), xi=0.0)

    def test_nan_xi(self):
        _load_mp()
        with pytest.raises(ValueError, match="finite"):
            _hR_scalar_mp(1.0, xi=float("nan"))

    def test_inf_xi(self):
        _load_mp()
        with pytest.raises(ValueError, match="finite"):
            _hR_scalar_mp(1.0, xi=float("inf"))

    def test_negative_x(self):
        _load_mp()
        with pytest.raises(ValueError, match="x >= 0"):
            _hR_scalar_mp(-1.0, xi=0.0)

    def test_valid(self):
        _load_mp()
        result = _hR_scalar_mp(1.0, xi=0.0)
        assert np.isfinite(float(result))


class TestHCDiracMpGuards:
    def test_nan(self):
        _load_mp()
        with pytest.raises(ValueError, match="finite"):
            _hC_dirac_mp(float("nan"))

    def test_inf(self):
        _load_mp()
        with pytest.raises(ValueError, match="finite"):
            _hC_dirac_mp(float("inf"))

    def test_negative_x(self):
        _load_mp()
        with pytest.raises(ValueError, match="x >= 0"):
            _hC_dirac_mp(-1.0)

    def test_valid(self):
        _load_mp()
        result = _hC_dirac_mp(1.0)
        assert np.isfinite(float(result))


class TestHRDiracMpGuards:
    def test_nan(self):
        _load_mp()
        with pytest.raises(ValueError, match="finite"):
            _hR_dirac_mp(float("nan"))

    def test_inf(self):
        _load_mp()
        with pytest.raises(ValueError, match="finite"):
            _hR_dirac_mp(float("inf"))

    def test_negative_x(self):
        _load_mp()
        with pytest.raises(ValueError, match="x >= 0"):
            _hR_dirac_mp(-1.0)

    def test_valid(self):
        _load_mp()
        result = _hR_dirac_mp(1.0)
        assert np.isfinite(float(result))


# ============================================================================
# FITTING — Round 7 gaps: anderson_darling_test, bayesian_limit,
#           model_comparison chi2/k NaN
# ============================================================================


class TestAndersonDarlingGuards:
    def test_nan_in_data(self):
        with pytest.raises(ValueError, match="NaN|infinite"):
            anderson_darling_test([1.0, float("nan"), 2.0])

    def test_inf_in_data(self):
        with pytest.raises(ValueError, match="NaN|infinite"):
            anderson_darling_test([1.0, float("inf"), 2.0])

    def test_empty_data(self):
        with pytest.raises(ValueError, match="empty"):
            anderson_darling_test([])

    def test_valid_returns_dict(self):
        np.random.seed(42)
        result = anderson_darling_test(np.random.randn(100))
        assert "statistic" in result
        assert np.isfinite(result["statistic"])


class TestBayesianLimitGuards:
    def test_nan_in_samples(self):
        with pytest.raises(ValueError, match="NaN|infinite"):
            bayesian_limit([1.0, float("nan"), 2.0])

    def test_inf_in_samples(self):
        with pytest.raises(ValueError, match="NaN|infinite"):
            bayesian_limit([1.0, float("inf"), 2.0])

    def test_empty_samples(self):
        with pytest.raises(ValueError, match="empty"):
            bayesian_limit([])

    def test_single_sample(self):
        with pytest.raises(ValueError, match=">= 2"):
            bayesian_limit([1.0])

    def test_valid_upper(self):
        np.random.seed(42)
        result = bayesian_limit(np.random.randn(1000), cl=0.95, side="upper")
        assert "limit" in result
        assert np.isfinite(result["limit"])

    def test_valid_lower(self):
        np.random.seed(42)
        result = bayesian_limit(np.random.randn(1000), cl=0.95, side="lower")
        assert "limit" in result
        assert np.isfinite(result["limit"])

    def test_valid_hdi(self):
        np.random.seed(42)
        result = bayesian_limit(np.random.randn(1000), cl=0.95, side="hdi")
        assert "lower" in result
        assert "upper" in result
        assert result["upper"] > result["lower"]


class TestModelComparisonNaNGuards:
    """Round 7: model_comparison must reject NaN in chi2 and k values."""

    def test_nan_chi2_1(self):
        with pytest.raises(ValueError, match="finite"):
            model_comparison(float("nan"), 2, 8.0, 3, n_data=50)

    def test_nan_chi2_2(self):
        with pytest.raises(ValueError, match="finite"):
            model_comparison(10.0, 2, float("nan"), 3, n_data=50)

    def test_inf_chi2_1(self):
        with pytest.raises(ValueError, match="finite"):
            model_comparison(float("inf"), 2, 8.0, 3, n_data=50)

    def test_nan_k_1(self):
        with pytest.raises(ValueError, match="finite"):
            model_comparison(10.0, float("nan"), 8.0, 3, n_data=50)

    def test_nan_k_2(self):
        with pytest.raises(ValueError, match="finite"):
            model_comparison(10.0, 2, 8.0, float("nan"), n_data=50)

    def test_inf_k_2(self):
        with pytest.raises(ValueError, match="finite"):
            model_comparison(10.0, 2, 8.0, float("inf"), n_data=50)

    def test_valid_still_works(self):
        result = model_comparison(10.0, 2, 8.0, 3, n_data=50)
        assert "dAIC" in result
        assert np.isfinite(result["dAIC"])
        assert np.isfinite(result["dBIC"])

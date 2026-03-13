"""
Extended tests for sct_tools.fitting — covers chi2, chi2_cov n_params,
fit_minuit, bayesian_limit edge cases, model_comparison.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sct_tools import fitting


class TestChi2:
    def test_perfect_fit(self):
        obs = np.array([1.0, 2.0, 3.0])
        chi2_val, ndof, chi2_red, p_val = fitting.chi2(obs, obs, np.ones(3))
        assert chi2_val == pytest.approx(0.0, abs=1e-14)
        assert ndof == 3
        assert p_val == pytest.approx(1.0, abs=1e-10)

    def test_known_chi2(self):
        obs = np.array([1.0, 2.0, 3.0])
        exp = np.array([1.1, 2.0, 2.9])
        err = np.array([0.1, 0.1, 0.1])
        chi2_val, ndof, chi2_red, p_val = fitting.chi2(obs, exp, err)
        assert chi2_val == pytest.approx(2.0, rel=1e-10)
        assert ndof == 3

    def test_n_params_reduces_ndof(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        chi2_val, ndof, _, _ = fitting.chi2(obs, obs, np.ones(5), n_params=2)
        assert ndof == 3  # 5 - 2

    def test_n_params_floor_at_one(self):
        obs = np.array([1.0])
        _, ndof, _, _ = fitting.chi2(obs, obs, np.ones(1), n_params=5)
        assert ndof == 1  # max(1-5, 1) = 1


class TestChi2CovExtended:
    def test_n_params(self):
        obs = np.array([1.0, 2.0, 3.0])
        exp = np.array([1.0, 2.0, 3.0])
        cov = np.eye(3)
        _, ndof, _, _ = fitting.chi2_cov(obs, exp, cov, n_params=1)
        assert ndof == 2  # 3 - 1

    def test_cholesky_path(self):
        # Positive-definite matrix should use Cholesky
        obs = np.array([1.0, 2.0])
        exp = np.array([1.1, 1.9])
        cov = np.array([[0.04, 0.01], [0.01, 0.04]])
        chi2_val, ndof, chi2_red, p_val = fitting.chi2_cov(obs, exp, cov)
        assert chi2_val > 0
        assert np.isfinite(p_val)


class TestFitMinuit:
    def test_simple_parabola(self):
        # Minimize (x-3)^2 + (y-5)^2
        def cost(x, y):
            return (x - 3) ** 2 + (y - 5) ** 2

        m = fitting.fit_minuit(cost, [0.0, 0.0], param_names=["x", "y"])
        assert m.values["x"] == pytest.approx(3.0, abs=0.01)
        assert m.values["y"] == pytest.approx(5.0, abs=0.01)
        assert m.valid

    def test_with_limits(self):
        def cost(x):
            return (x - 10) ** 2

        m = fitting.fit_minuit(cost, [0.0], param_names=["x"],
                               limits={"x": (0, 5)})
        assert m.values["x"] == pytest.approx(5.0, abs=0.1)


class TestModelComparison:
    def test_identical_models(self):
        result = fitting.model_comparison(10.0, 2, 10.0, 2, 100)
        assert result['dAIC'] == pytest.approx(0.0)
        assert result['dBIC'] == pytest.approx(0.0)

    def test_simpler_wins(self):
        # Same chi2 but model 2 has fewer params -> model 2 wins
        result = fitting.model_comparison(10.0, 3, 10.0, 1, 100)
        assert result['dAIC'] > 0  # model 1 higher AIC = worse
        assert result['favors'] == 'model_2'


class TestChi2ZeroError:
    def test_rejects_zero_error(self):
        obs = np.array([1.0, 2.0, 3.0])
        exp = np.array([1.1, 2.0, 2.9])
        err = np.array([0.1, 0.0, 0.1])  # zero in middle
        with pytest.raises(ValueError, match="zero"):
            fitting.chi2(obs, exp, err)

    def test_accepts_positive_errors(self):
        obs = np.array([1.0, 2.0, 3.0])
        exp = np.array([1.1, 2.0, 2.9])
        err = np.array([0.1, 0.1, 0.1])
        chi2_val, _, _, _ = fitting.chi2(obs, exp, err)
        assert np.isfinite(chi2_val)


class TestDiscoverySignificanceEdge:
    def test_signal_zero_background(self):
        # s > 0, b = 0 should give infinite significance
        result = fitting.discovery_significance(10, 0)
        assert result['Z'] == np.inf
        assert result['p_value'] == 0.0

    def test_zero_signal(self):
        result = fitting.discovery_significance(0, 100)
        assert result['Z'] == 0.0
        assert result['p_value'] == 1.0

    def test_both_zero(self):
        result = fitting.discovery_significance(0, 0)
        assert result['Z'] == 0.0

    def test_normal_case(self):
        result = fitting.discovery_significance(10, 100)
        assert result['Z'] > 0
        assert np.isfinite(result['Z'])


class TestModelComparisonDocstring:
    def test_negative_daic_favors_model1(self):
        # Model 1 better chi2 but same params -> model 1 favored
        result = fitting.model_comparison(5.0, 2, 15.0, 2, 100)
        assert result['dAIC'] < 0
        assert result['favors'] == 'model_1'


class TestBayesianLimitEdgeCases:
    def test_small_sample(self):
        samples = np.array([1.0, 2.0])
        result = fitting.bayesian_limit(samples, cl=0.95, side='upper')
        assert np.isfinite(result['limit'])

    def test_hdi_small_sample(self):
        samples = np.array([1.0, 2.0, 3.0])
        result = fitting.bayesian_limit(samples, cl=0.68, side='hdi')
        assert result['lower'] <= result['upper']

    def test_invalid_cl(self):
        with pytest.raises(ValueError, match="cl must be"):
            fitting.bayesian_limit(np.array([1, 2, 3]), cl=1.5)

    def test_too_few_samples(self):
        with pytest.raises(ValueError, match="requires >= 2"):
            fitting.bayesian_limit(np.array([1.0]), cl=0.95)

    def test_invalid_side(self):
        with pytest.raises(ValueError, match="unknown side"):
            fitting.bayesian_limit(np.array([1, 2, 3]), side='middle')

    def test_rejects_2d_multi_param(self):
        rng = np.random.default_rng(42)
        samples_2d = rng.standard_normal((100, 3))
        with pytest.raises(ValueError, match="1D samples"):
            fitting.bayesian_limit(samples_2d, cl=0.95)

    def test_accepts_column_vector(self):
        rng = np.random.default_rng(43)
        samples_col = rng.standard_normal((100, 1))
        result = fitting.bayesian_limit(samples_col, cl=0.95)
        assert np.isfinite(result['limit'])

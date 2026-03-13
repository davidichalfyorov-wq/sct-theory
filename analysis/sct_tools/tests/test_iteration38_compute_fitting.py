"""
Tests for Iteration 38: compute.py + fitting.py hardening.

Covers:
    F-F1: model_comparison rejects n_data < 2
    F-F2: weighted_least_squares rejects empty arrays
    F-F6: anderson_darling_test rejects invalid dist
    C-F2: wsl_run validates python_code type and timeout
    C-F6: progress_compute_mp validates dps
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sct_tools import compute, fitting

# ============================================================================
# F-F1: model_comparison rejects n_data < 2
# ============================================================================


class TestModelComparisonNData:
    """model_comparison must reject n_data < 2 for meaningful BIC."""

    def test_n_data_zero_raises(self):
        with pytest.raises(ValueError, match="n_data >= 2"):
            fitting.model_comparison(1.0, 2, 2.0, 3, n_data=0)

    def test_n_data_one_raises(self):
        with pytest.raises(ValueError, match="n_data >= 2"):
            fitting.model_comparison(1.0, 2, 2.0, 3, n_data=1)

    def test_n_data_negative_raises(self):
        with pytest.raises(ValueError, match="n_data >= 2"):
            fitting.model_comparison(1.0, 2, 2.0, 3, n_data=-5)

    def test_n_data_two_works(self):
        result = fitting.model_comparison(10.0, 2, 12.0, 3, n_data=2)
        assert 'AIC_1' in result
        assert 'BIC_1' in result
        assert 'favors' in result

    def test_n_data_large_works(self):
        result = fitting.model_comparison(10.0, 2, 12.0, 3, n_data=100)
        assert result['favors'] in ('model_1', 'model_2')


# ============================================================================
# F-F2: weighted_least_squares rejects empty arrays
# ============================================================================


class TestWeightedLeastSquaresEmpty:
    """weighted_least_squares must reject empty arrays early."""

    def test_empty_x_raises(self):
        with pytest.raises(ValueError, match="empty"):
            fitting.weighted_least_squares([], [1.0], [0.1], degree=1)

    def test_empty_y_raises(self):
        with pytest.raises(ValueError, match="empty"):
            fitting.weighted_least_squares([1.0], [], [0.1], degree=1)

    def test_empty_yerr_raises(self):
        with pytest.raises(ValueError, match="empty"):
            fitting.weighted_least_squares([1.0], [1.0], [], degree=1)

    def test_all_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            fitting.weighted_least_squares([], [], [], degree=1)

    def test_normal_works(self):
        x = [1.0, 2.0, 3.0]
        y = [2.0, 4.0, 6.0]
        yerr = [0.1, 0.1, 0.1]
        result = fitting.weighted_least_squares(x, y, yerr, degree=1)
        assert hasattr(result, 'params')


# ============================================================================
# F-F6: anderson_darling_test rejects invalid dist
# ============================================================================


class TestAndersonDarlingDist:
    """anderson_darling_test must reject unsupported dist values."""

    def test_invalid_dist_raises(self):
        data = np.random.randn(50)
        with pytest.raises(ValueError, match="dist must be one of"):
            fitting.anderson_darling_test(data, dist='invalid')

    def test_empty_string_raises(self):
        data = np.random.randn(50)
        with pytest.raises(ValueError, match="dist must be one of"):
            fitting.anderson_darling_test(data, dist='')

    def test_norm_works(self):
        data = np.random.randn(50)
        result = fitting.anderson_darling_test(data, dist='norm')
        assert 'statistic' in result

    def test_expon_works(self):
        data = np.random.exponential(1.0, 50)
        result = fitting.anderson_darling_test(data, dist='expon')
        assert 'statistic' in result


# ============================================================================
# C-F2: wsl_run validates python_code type and timeout
# ============================================================================


class TestWslRunValidation:
    """wsl_run must validate python_code type and timeout."""

    def test_non_string_code_raises(self):
        with pytest.raises(TypeError, match="python_code must be a string"):
            compute.wsl_run(123)

    def test_list_code_raises(self):
        with pytest.raises(TypeError, match="python_code must be a string"):
            compute.wsl_run(['print("hi")'])

    def test_none_code_raises(self):
        with pytest.raises(TypeError, match="python_code must be a string"):
            compute.wsl_run(None)

    def test_zero_timeout_raises(self):
        with pytest.raises(ValueError, match="timeout must be positive"):
            compute.wsl_run("print('hi')", timeout=0)

    def test_negative_timeout_raises(self):
        with pytest.raises(ValueError, match="timeout must be positive"):
            compute.wsl_run("print('hi')", timeout=-10)


# ============================================================================
# C-F6: progress_compute_mp validates dps
# ============================================================================


class TestProgressComputeMpDps:
    """progress_compute_mp must validate dps is a positive integer."""

    def test_zero_dps_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            compute.progress_compute_mp(lambda x, dps=15: x, [1.0], dps=0)

    def test_negative_dps_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            compute.progress_compute_mp(lambda x, dps=15: x, [1.0], dps=-5)

    def test_float_dps_raises(self):
        with pytest.raises(ValueError, match="positive integer"):
            compute.progress_compute_mp(lambda x, dps=15: x, [1.0], dps=3.5)

    def test_valid_dps_works(self):
        results = compute.progress_compute_mp(
            lambda x, dps=15: x * 2, [1.0, 2.0], dps=15
        )
        assert results == [2.0, 4.0]

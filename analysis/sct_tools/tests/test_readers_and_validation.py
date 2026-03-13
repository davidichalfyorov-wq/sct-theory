"""
Tests for data_io readers (HDF5, ROOT, FITS) and input validation fixes.

Data readers use mocks since external packages may not be installed.
Validation tests verify the new guards added to fitting.py, graphs.py, form_factors.py.
"""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sct_tools import data_io  # noqa: E402

# ---------------------------------------------------------------------------
#  read_hdf5() — HDF5 file reader
# ---------------------------------------------------------------------------


class TestReadHDF5:
    """Tests for read_hdf5() with mocked h5py."""

    def test_signature(self):
        """read_hdf5() has correct function signature."""
        import inspect
        sig = inspect.signature(data_io.read_hdf5)
        params = list(sig.parameters.keys())
        assert "filepath" in params
        assert "dataset_name" in params

    def test_without_h5py(self):
        """read_hdf5() raises ImportError when h5py not available."""
        with patch.dict("sys.modules", {"h5py": None}):
            with pytest.raises((ImportError, ModuleNotFoundError)):
                data_io.read_hdf5("/fake/file.h5", "data")


# ---------------------------------------------------------------------------
#  read_root() — ROOT file reader
# ---------------------------------------------------------------------------


class TestReadRoot:
    """Tests for read_root() with mocked uproot."""

    def test_signature(self):
        """read_root() has correct function signature."""
        import inspect
        sig = inspect.signature(data_io.read_root)
        params = list(sig.parameters.keys())
        assert "filepath" in params
        assert "tree_name" in params
        assert "branches" in params

    def test_without_uproot(self):
        """read_root() raises ImportError when uproot not available."""
        with patch.dict("sys.modules", {"uproot": None}):
            with pytest.raises((ImportError, ModuleNotFoundError)):
                data_io.read_root("/fake/file.root", "Events")

    @patch("uproot.open")
    def test_list_trees(self, mock_open):
        """read_root() returns tree names when tree_name is None."""
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.keys.return_value = ["Events;1", "Metadata;1"]
        mock_open.return_value = mock_file

        result = data_io.read_root("/fake/file.root")
        assert result == ["Events;1", "Metadata;1"]

    @patch("uproot.open")
    def test_read_branches(self, mock_open):
        """read_root() returns dict of arrays for given tree."""
        mock_branch_pt = MagicMock()
        mock_branch_pt.array.return_value = np.array([10.0, 20.0, 30.0])
        mock_branch_eta = MagicMock()
        mock_branch_eta.array.return_value = np.array([0.5, -0.5, 1.0])

        mock_tree = MagicMock()
        mock_tree.__getitem__ = lambda self, key: {
            "pt": mock_branch_pt, "eta": mock_branch_eta
        }[key]
        mock_tree.keys.return_value = ["pt", "eta"]

        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.__getitem__ = MagicMock(return_value=mock_tree)
        mock_open.return_value = mock_file

        result = data_io.read_root("/fake/file.root", "Events")
        assert "pt" in result
        assert "eta" in result
        assert len(result["pt"]) == 3


# ---------------------------------------------------------------------------
#  read_fits() — FITS file reader
# ---------------------------------------------------------------------------


class TestReadFITS:
    """Tests for read_fits() and read_fits_header() with mocked astropy."""

    def test_read_fits_signature(self):
        """read_fits() has correct function signature."""
        import inspect
        sig = inspect.signature(data_io.read_fits)
        params = list(sig.parameters.keys())
        assert "filepath" in params
        assert "hdu" in params

    def test_read_fits_header_signature(self):
        """read_fits_header() has correct function signature."""
        import inspect
        sig = inspect.signature(data_io.read_fits_header)
        params = list(sig.parameters.keys())
        assert "filepath" in params
        assert "hdu" in params

    def test_without_astropy(self):
        """read_fits() raises ImportError when astropy not available."""
        with patch.dict("sys.modules", {
            "astropy": None, "astropy.io": None, "astropy.io.fits": None
        }):
            with pytest.raises((ImportError, ModuleNotFoundError)):
                data_io.read_fits("/fake/file.fits")

    @patch("astropy.io.fits.open")
    def test_read_fits_returns_data(self, mock_fits_open):
        """read_fits() returns HDU data."""
        mock_hdu = MagicMock()
        mock_hdu.data = np.array([1.0, 2.0, 3.0])
        mock_hdul = MagicMock()
        mock_hdul.__enter__ = MagicMock(return_value=mock_hdul)
        mock_hdul.__exit__ = MagicMock(return_value=False)
        mock_hdul.__getitem__ = MagicMock(return_value=mock_hdu)
        mock_fits_open.return_value = mock_hdul

        result = data_io.read_fits("/fake/file.fits", hdu=1)
        np.testing.assert_array_equal(result, np.array([1.0, 2.0, 3.0]))

    @patch("astropy.io.fits.open")
    def test_read_fits_header_returns_dict(self, mock_fits_open):
        """read_fits_header() returns dict of header keywords."""
        mock_header = {"NAXIS": 2, "BITPIX": -32}
        mock_hdu = MagicMock()
        mock_hdu.header = mock_header

        mock_hdul = MagicMock()
        mock_hdul.__enter__ = MagicMock(return_value=mock_hdul)
        mock_hdul.__exit__ = MagicMock(return_value=False)
        mock_hdul.__getitem__ = MagicMock(return_value=mock_hdu)
        mock_fits_open.return_value = mock_hdul

        result = data_io.read_fits_header("/fake/file.fits", hdu=0)
        assert isinstance(result, dict)
        assert result["NAXIS"] == 2


# ---------------------------------------------------------------------------
#  Input validation tests for code fixes
# ---------------------------------------------------------------------------


class TestChiSquaredValidation:
    """Test chi2() input validation fixes."""

    def test_nan_observed(self):
        """chi2() raises on NaN in observed array."""
        from sct_tools.fitting import chi2
        with pytest.raises(ValueError, match="observed.*NaN"):
            chi2([float("nan"), 1.0], [1.0, 1.0], [0.1, 0.1])

    def test_inf_expected(self):
        """chi2() raises on Inf in expected array."""
        from sct_tools.fitting import chi2
        with pytest.raises(ValueError, match="expected.*infinite"):
            chi2([1.0, 1.0], [float("inf"), 1.0], [0.1, 0.1])

    def test_nan_expected(self):
        """chi2() raises on NaN in expected array."""
        from sct_tools.fitting import chi2
        with pytest.raises(ValueError, match="expected.*NaN"):
            chi2([1.0, 1.0], [float("nan"), 1.0], [0.1, 0.1])

    def test_inf_observed(self):
        """chi2() raises on Inf in observed array."""
        from sct_tools.fitting import chi2
        with pytest.raises(ValueError, match="observed.*infinite"):
            chi2([float("inf"), 1.0], [1.0, 1.0], [0.1, 0.1])

    def test_valid_inputs(self):
        """chi2() works with valid inputs."""
        from sct_tools.fitting import chi2
        val, ndof, chi2_red, p_val = chi2([1.0, 2.0], [1.1, 1.9], [0.1, 0.1])
        assert val > 0
        assert ndof > 0
        assert 0 <= p_val <= 1


class TestLmfitValidation:
    """Test fit_lmfit() y_err validation."""

    def test_negative_yerr(self):
        """fit_lmfit() raises on negative y_err."""
        from sct_tools.fitting import fit_lmfit
        with pytest.raises(ValueError, match="y_err must be all positive"):
            fit_lmfit(
                lambda x, a: a * x,
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                np.array([0.1, -0.1]),
                {"a": 1.0},
            )

    def test_zero_yerr(self):
        """fit_lmfit() raises on zero y_err."""
        from sct_tools.fitting import fit_lmfit
        with pytest.raises(ValueError, match="y_err must be all positive"):
            fit_lmfit(
                lambda x, a: a * x,
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                np.array([0.1, 0.0]),
                {"a": 1.0},
            )

    def test_nan_yerr(self):
        """fit_lmfit() raises on NaN y_err."""
        from sct_tools.fitting import fit_lmfit
        with pytest.raises(ValueError, match="y_err.*NaN"):
            fit_lmfit(
                lambda x, a: a * x,
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                np.array([0.1, float("nan")]),
                {"a": 1.0},
            )


class TestCausalSetValidation:
    """Test causal_set_sprinkle() input validation."""

    def test_invalid_dim(self):
        from sct_tools.graphs import causal_set_sprinkle
        with pytest.raises(ValueError, match="dim must be"):
            causal_set_sprinkle(10, dim=5)

    def test_invalid_region(self):
        from sct_tools.graphs import causal_set_sprinkle
        with pytest.raises(ValueError, match="region must be"):
            causal_set_sprinkle(10, region="schwarzschild")

    def test_zero_points(self):
        from sct_tools.graphs import causal_set_sprinkle
        with pytest.raises(ValueError, match="n_points must be"):
            causal_set_sprinkle(0)

    def test_negative_points(self):
        from sct_tools.graphs import causal_set_sprinkle
        with pytest.raises(ValueError, match="n_points must be"):
            causal_set_sprinkle(-5)


class TestSpectralDimensionValidation:
    """Test spectral_dimension_graph() input validation."""

    def test_zero_dt_frac(self):
        from sct_tools.graphs import spectral_dimension_graph
        adj = np.array([[0, 1], [1, 0]])
        with pytest.raises(ValueError, match="dt_frac must be"):
            spectral_dimension_graph(adj, dt_frac=0)

    def test_large_dt_frac(self):
        from sct_tools.graphs import spectral_dimension_graph
        adj = np.array([[0, 1], [1, 0]])
        with pytest.raises(ValueError, match="dt_frac must be"):
            spectral_dimension_graph(adj, dt_frac=1.5)

    def test_underflow_returns_nan(self):
        """When heat kernel underflows at large t, d_S should be NaN or ~0."""
        from sct_tools.graphs import spectral_dimension_graph
        adj = np.array([[0, 1], [1, 0]])
        t_vals = np.array([1e10])
        t_out, d_s = spectral_dimension_graph(adj, t_values=t_vals)
        assert len(d_s) == 1
        # At t=1e10, heat kernel is fully equilibrated → d_S is NaN or near 0
        assert np.isnan(d_s[0]) or abs(d_s[0]) < 1e-5


class TestFormFactorsTotalFixed:
    """Verify F1_total and F2_total no longer have dead try/except."""

    def test_f1_total_works(self):
        from sct_tools.form_factors import F1_total
        val = F1_total(1.0)
        # mpmath 50-digit reference (Phase 3 CORRECTED with N_f/2)
        assert val == pytest.approx(-3.182959362479461e-04, rel=1e-10)

    def test_f2_total_works(self):
        from sct_tools.form_factors import F2_total
        val = F2_total(1.0)
        # mpmath 50-digit reference (Phase 3 CORRECTED with N_f/2)
        assert val == pytest.approx(4.312611841226462e-04, rel=1e-10)

    def test_f1_f2_total_at_zero(self):
        """F1_total and F2_total at x → 0+ against mpmath reference."""
        from sct_tools.form_factors import F1_total, F2_total
        val1 = F1_total(1e-8)
        val2 = F2_total(1e-8)
        # mpmath 50-digit reference at x=1e-8 (Phase 3 CORRECTED)
        assert val1 == pytest.approx(6.860288475783305e-04, rel=1e-8)
        assert val2 == pytest.approx(3.518096654247839e-04, rel=1e-8)

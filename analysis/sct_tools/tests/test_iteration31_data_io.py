"""
Tests for Iteration 31: data_io.py + lean.py hardening.

Covers:
    DIO-TC1: read_csv dedicated tests (HIGH gap)
    DIO-TC3: load_results error paths
    DIO-TC4: NaN/Inf serialization behavior (now warns + converts to null)
    DIO-TC2: _serialize complex ndarray roundtrip
    LEAN-5: formalize empty description validation
"""

import json
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sct_tools import data_io

# ============================================================================
# DIO-TC1: read_csv dedicated tests
# ============================================================================


class TestReadCSV:
    """read_csv must handle CSV/TSV files correctly."""

    def test_csv_with_header(self, tmp_path):
        """Read a CSV file with header row."""
        f = tmp_path / "test.csv"
        f.write_text("x,y,z\n1.0,2.0,3.0\n4.0,5.0,6.0\n")
        result = data_io.read_csv(str(f))
        assert result['columns'] == ['x', 'y', 'z']
        assert result['n_rows'] == 2
        assert result['data']['x'][0] == pytest.approx(1.0)
        assert result['data']['z'][1] == pytest.approx(6.0)

    def test_csv_without_header(self, tmp_path):
        """has_header=False should not consume first data row."""
        f = tmp_path / "test.csv"
        f.write_text("1.0,2.0,3.0\n4.0,5.0,6.0\n")
        result = data_io.read_csv(str(f), has_header=False)
        assert result['n_rows'] == 2  # both rows preserved
        # Integer column names when no header and no columns specified
        assert 0 in result['data'] or '0' in result['data']

    def test_csv_with_column_override(self, tmp_path):
        """columns= overrides header; original header becomes data row."""
        f = tmp_path / "test.csv"
        f.write_text("a,b\n1,2\n3,4\n")
        result = data_io.read_csv(str(f), columns=['x', 'y'])
        assert result['columns'] == ['x', 'y']
        # When columns= is provided, header row "a,b" becomes a data row
        assert result['n_rows'] == 3

    def test_csv_comment_lines_skipped(self, tmp_path):
        """Lines starting with # should be ignored."""
        f = tmp_path / "test.csv"
        f.write_text("# This is a comment\nx,y\n1,2\n# another comment\n3,4\n")
        result = data_io.read_csv(str(f))
        assert result['n_rows'] == 2

    def test_tsv_delimiter_auto_detect(self, tmp_path):
        """Non-.csv extension should auto-detect whitespace delimiter."""
        f = tmp_path / "test.dat"
        f.write_text("x y\n1.0 2.0\n3.0 4.0\n")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = data_io.read_csv(str(f))
            # Should warn about auto-detection
            assert any("auto-detected" in str(x.message) for x in w)
        assert result['n_rows'] == 2
        assert result['data']['x'][0] == pytest.approx(1.0)

    def test_explicit_delimiter(self, tmp_path):
        """Explicit delimiter should suppress auto-detection warning."""
        f = tmp_path / "test.dat"
        f.write_text("x\ty\n1.0\t2.0\n3.0\t4.0\n")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = data_io.read_csv(str(f), delimiter='\t')
            auto_warn = [x for x in w if "auto-detected" in str(x.message)]
            assert len(auto_warn) == 0
        assert result['n_rows'] == 2

    def test_skip_header(self, tmp_path):
        """skip_header should skip specified number of lines."""
        f = tmp_path / "test.csv"
        f.write_text("meta line 1\nmeta line 2\nx,y\n1,2\n3,4\n")
        result = data_io.read_csv(str(f), skip_header=2)
        assert result['columns'] == ['x', 'y']
        assert result['n_rows'] == 2

    def test_empty_csv_raises(self, tmp_path):
        """Empty file should raise pandas EmptyDataError."""
        import pandas as pd
        f = tmp_path / "test.csv"
        f.write_text("")
        with pytest.raises(pd.errors.EmptyDataError):
            data_io.read_csv(str(f))


# ============================================================================
# DIO-TC3: load_results error paths
# ============================================================================


class TestLoadResultsErrors:
    """load_results error handling."""

    def test_missing_results_key_raises(self, tmp_path):
        """File without 'results' key should raise ValueError."""
        f = tmp_path / "bad.json"
        f.write_text(json.dumps({"data": 42}))
        with pytest.raises(ValueError, match="results"):
            data_io.load_results(str(f))

    def test_file_not_found_raises(self):
        """Non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            data_io.load_results("/nonexistent/path/results.json")


# ============================================================================
# DIO-TC4: NaN/Inf serialization (now warns + converts to null)
# ============================================================================


class TestNaNInfSerialization:
    """NaN/Inf values should warn and serialize as null for JSON compliance."""

    def test_nan_serialized_as_null(self, tmp_path):
        """np.float64(nan) should serialize as null with warning."""
        f = tmp_path / "nan_test.json"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data_io.save_results(str(f), {'value': np.float64(np.nan)})
            assert any("non-finite" in str(x.message) for x in w)
        # Verify the file contains null, not NaN
        with open(f) as fh:
            content = json.load(fh)
        assert content['results']['value'] is None

    def test_inf_serialized_as_null(self, tmp_path):
        """np.float64(inf) should serialize as null with warning."""
        f = tmp_path / "inf_test.json"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            data_io.save_results(str(f), {'value': np.float64(np.inf)})
            assert any("non-finite" in str(x.message) for x in w)
        with open(f) as fh:
            content = json.load(fh)
        assert content['results']['value'] is None

    def test_finite_values_unchanged(self, tmp_path):
        """Normal finite values should round-trip correctly."""
        f = tmp_path / "ok_test.json"
        data_io.save_results(str(f), {'pi': np.float64(3.14159)})
        results, _ = data_io.load_results(str(f))
        assert results['pi'] == pytest.approx(3.14159)


# ============================================================================
# DIO-TC2: _serialize complex ndarray roundtrip
# ============================================================================


class TestComplexNdarrayRoundtrip:
    """Complex ndarray serialization/deserialization roundtrip."""

    def test_1d_complex_array(self, tmp_path):
        """1D complex array should roundtrip correctly."""
        f = tmp_path / "complex1d.json"
        arr = np.array([1 + 2j, 3 + 4j, 5 + 0j])
        data_io.save_results(str(f), {'z': arr})
        results, _ = data_io.load_results(str(f))
        loaded = np.array(results['z'])
        np.testing.assert_array_almost_equal(loaded, arr)

    def test_2d_complex_array(self, tmp_path):
        """2D complex array should preserve shape."""
        f = tmp_path / "complex2d.json"
        arr = np.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]])
        data_io.save_results(str(f), {'matrix': arr})
        results, _ = data_io.load_results(str(f))
        loaded = np.array(results['matrix'])
        assert loaded.shape == (2, 2)
        np.testing.assert_array_almost_equal(loaded, arr)

    def test_scalar_complex(self, tmp_path):
        """Scalar complex should roundtrip."""
        f = tmp_path / "complex_scalar.json"
        data_io.save_results(str(f), {'z': complex(1.5, -2.5)})
        results, _ = data_io.load_results(str(f))
        assert results['z'] == pytest.approx(complex(1.5, -2.5))


# ============================================================================
# LEAN-5: formalize empty description validation
# ============================================================================


class TestFormalizeValidation:
    """formalize must reject empty descriptions."""

    def test_empty_string_raises(self):
        from sct_tools import lean
        with pytest.raises(ValueError, match="non-empty"):
            lean.formalize("")

    def test_whitespace_only_raises(self):
        from sct_tools import lean
        with pytest.raises(ValueError, match="non-empty"):
            lean.formalize("   ")

    def test_none_raises(self):
        from sct_tools import lean
        with pytest.raises((ValueError, TypeError)):
            lean.formalize(None)

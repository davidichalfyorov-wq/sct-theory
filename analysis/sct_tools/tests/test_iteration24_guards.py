"""
Tests for Iteration 24 fixes in data_io.py and compute.py.

Covers:
    data_io:
        D1: load_results round-trip deserialization (complex, ndarray_complex)
        D2: read_csv has_header=False (headerless files)
        D3: save_results metadata=None vs falsy
        D4: _serialize mpmath types
    compute:
        C1: wsl_run_script relative path rejection
        C2: vegas_integrate limits validation
        C3: progress_compute object dtype warning
"""

import json
import os
import sys
import tempfile
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sct_tools import compute, data_io

# ============================================================================
# data_io round-trip deserialization
# ============================================================================


class TestLoadResultsDeserialization:
    """D1: load_results must reconstruct complex/ndarray types."""

    def test_complex_scalar_roundtrip(self):
        """Complex scalar survives save/load."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            path = f.name
        try:
            data_io.save_results(path, {'z': 1 + 2j})
            results, _ = data_io.load_results(path)
            assert isinstance(results['z'], complex)
            assert results['z'] == 1 + 2j
        finally:
            os.unlink(path)

    def test_complex_ndarray_roundtrip(self):
        """Complex ndarray survives save/load with shape preserved."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            path = f.name
        try:
            arr = np.array([[1 + 0j, 2 + 1j], [3 - 1j, 4 + 0j]])
            data_io.save_results(path, {'arr': arr})
            results, _ = data_io.load_results(path)
            loaded = results['arr']
            assert isinstance(loaded, np.ndarray)
            assert loaded.shape == (2, 2)
            np.testing.assert_allclose(loaded, arr)
        finally:
            os.unlink(path)

    def test_nested_complex_roundtrip(self):
        """Nested dict with complex values survives save/load."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            path = f.name
        try:
            data_io.save_results(path, {'inner': {'z': 3 + 4j}, 'real': 5.0})
            results, _ = data_io.load_results(path)
            assert isinstance(results['inner']['z'], complex)
            assert results['inner']['z'] == 3 + 4j
            assert results['real'] == 5.0
        finally:
            os.unlink(path)

    def test_real_data_roundtrip(self):
        """Pure real data unaffected by deserialization."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            path = f.name
        try:
            data_io.save_results(path, {'x': [1.0, 2.0, 3.0], 'n': 42})
            results, _ = data_io.load_results(path)
            assert results['x'] == [1.0, 2.0, 3.0]
            assert results['n'] == 42
        finally:
            os.unlink(path)


class TestReadCsvHeaderless:
    """D2: read_csv has_header=False correctly reads all rows."""

    def test_headerless_preserves_first_row(self):
        """First row is data, not header, when has_header=False."""
        with tempfile.NamedTemporaryFile(
            suffix='.csv', delete=False, mode='w', newline=''
        ) as f:
            f.write("1.0,2.0,3.0\n4.0,5.0,6.0\n7.0,8.0,9.0\n")
            path = f.name
        try:
            result = data_io.read_csv(path, has_header=False)
            assert result['n_rows'] == 3  # all 3 rows preserved
        finally:
            os.unlink(path)

    def test_with_header_default(self):
        """Default has_header=True treats first row as header."""
        with tempfile.NamedTemporaryFile(
            suffix='.csv', delete=False, mode='w', newline=''
        ) as f:
            f.write("a,b,c\n1.0,2.0,3.0\n4.0,5.0,6.0\n")
            path = f.name
        try:
            result = data_io.read_csv(path)
            assert result['n_rows'] == 2
            assert result['columns'] == ['a', 'b', 'c']
        finally:
            os.unlink(path)


class TestSaveResultsMetadata:
    """D3: save_results metadata handling."""

    def test_none_metadata_defaults_to_empty(self):
        """metadata=None produces empty dict."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            path = f.name
        try:
            data_io.save_results(path, {'x': 1}, metadata=None)
            _, meta = data_io.load_results(path)
            assert meta == {}
        finally:
            os.unlink(path)

    def test_empty_list_metadata_preserved(self):
        """metadata=[] is preserved (not clobbered to {})."""
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            path = f.name
        try:
            data_io.save_results(path, {'x': 1}, metadata=[])
            with open(path) as fp:
                raw = json.load(fp)
            assert raw['metadata'] == []
        finally:
            os.unlink(path)


class TestSerializeMpmath:
    """D4: _serialize handles mpmath types."""

    def test_mpf_serialized(self):
        """mpmath.mpf converted to float."""
        import mpmath
        result = data_io._serialize(mpmath.mpf('3.14159'))
        assert isinstance(result, float)
        assert abs(result - 3.14159) < 1e-10

    def test_mpc_serialized(self):
        """mpmath.mpc converted to __complex__ dict."""
        import mpmath
        result = data_io._serialize(mpmath.mpc(1, 2))
        assert result['__complex__'] is True
        assert result['real'] == 1.0
        assert result['imag'] == 2.0

    def test_mpf_roundtrip(self):
        """mpmath.mpf survives full save/load roundtrip."""
        import mpmath
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            path = f.name
        try:
            data_io.save_results(path, {'pi': mpmath.mp.pi})
            results, _ = data_io.load_results(path)
            assert abs(results['pi'] - 3.14159265) < 1e-6
        finally:
            os.unlink(path)


# ============================================================================
# compute.py guards
# ============================================================================


class TestWslRunScriptRelativePath:
    """C1: wsl_run_script rejects relative paths."""

    def test_relative_path_rejected(self):
        """Relative path raises ValueError."""
        with pytest.raises(ValueError, match="absolute Windows path"):
            compute.wsl_run_script("scripts/run.py")

    def test_bare_filename_rejected(self):
        """Bare filename raises ValueError."""
        with pytest.raises(ValueError, match="absolute Windows path"):
            compute.wsl_run_script("run.py")


class TestVegasIntegrateLimitsValidation:
    """C2: vegas_integrate validates limits."""

    def test_empty_limits_rejected(self):
        """Empty limits list rejected."""
        with pytest.raises(ValueError, match="non-empty"):
            compute.vegas_integrate(lambda x: x[0], [])

    def test_reversed_bounds_rejected(self):
        """lo >= hi rejected."""
        with pytest.raises(ValueError, match="lo >= hi"):
            compute.vegas_integrate(lambda x: x[0], [(5.0, 1.0)])

    def test_equal_bounds_rejected(self):
        """lo == hi rejected."""
        with pytest.raises(ValueError, match="lo >= hi"):
            compute.vegas_integrate(lambda x: x[0], [(1.0, 1.0)])

    def test_invalid_tuple_rejected(self):
        """Non-tuple element rejected."""
        with pytest.raises(ValueError, match="must be a .* tuple"):
            compute.vegas_integrate(lambda x: x[0], [5.0])


class TestProgressComputeObjectDtype:
    """C3: progress_compute warns on object dtype."""

    def test_none_return_warns(self):
        """Func returning None produces object dtype warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = compute.progress_compute(
                lambda x: None, [1, 2, 3], desc=None,
            )
            assert result.dtype == object
            assert any("dtype=object" in str(x.message) for x in w)

    def test_numeric_return_no_warning(self):
        """Func returning floats does not warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = compute.progress_compute(
                lambda x: float(x ** 2), [1, 2, 3], desc=None,
            )
            assert result.dtype != object
            assert not any("dtype=object" in str(x.message) for x in w)

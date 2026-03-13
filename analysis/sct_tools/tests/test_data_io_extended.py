"""
Extended tests for sct_tools.data_io — covers save_results, load_results,
and numpy/mpmath serialization.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sct_tools import data_io


class TestSaveLoadResults:
    def test_roundtrip_scalars(self, tmp_path):
        filepath = tmp_path / "results.json"
        results = {"chi2": 12.5, "p_value": 0.03, "label": "test"}
        data_io.save_results(str(filepath), results)
        loaded, meta = data_io.load_results(str(filepath))
        assert loaded["chi2"] == 12.5
        assert loaded["p_value"] == 0.03
        assert loaded["label"] == "test"

    def test_roundtrip_arrays(self, tmp_path):
        filepath = tmp_path / "results.json"
        results = {"x": np.array([1.0, 2.0, 3.0]), "y": np.linspace(0, 1, 5)}
        data_io.save_results(str(filepath), results)
        loaded, _ = data_io.load_results(str(filepath))
        assert loaded["x"] == [1.0, 2.0, 3.0]
        assert len(loaded["y"]) == 5

    def test_metadata(self, tmp_path):
        filepath = tmp_path / "results.json"
        meta = {"author": "test", "version": "1.0"}
        data_io.save_results(str(filepath), {"val": 42}, metadata=meta)
        _, loaded_meta = data_io.load_results(str(filepath))
        assert loaded_meta["author"] == "test"

    def test_nested_dicts(self, tmp_path):
        filepath = tmp_path / "results.json"
        results = {"inner": {"a": 1, "b": [2, 3]}, "outer": 4}
        data_io.save_results(str(filepath), results)
        loaded, _ = data_io.load_results(str(filepath))
        assert loaded["inner"]["a"] == 1
        assert loaded["inner"]["b"] == [2, 3]

    def test_numpy_int_float(self, tmp_path):
        filepath = tmp_path / "results.json"
        results = {"int_val": np.int64(42), "float_val": np.float64(3.14)}
        data_io.save_results(str(filepath), results)
        loaded, _ = data_io.load_results(str(filepath))
        assert loaded["int_val"] == 42
        assert loaded["float_val"] == pytest.approx(3.14)

    def test_creates_parent_dirs(self, tmp_path):
        filepath = tmp_path / "sub" / "dir" / "results.json"
        data_io.save_results(str(filepath), {"x": 1})
        loaded, _ = data_io.load_results(str(filepath))
        assert loaded["x"] == 1

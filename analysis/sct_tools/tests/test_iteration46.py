"""
Tests for Iteration 46: Input validation across tensors, verification, graphs, constants.

C-2: symbols length validation in de_sitter, reissner_nordstrom, anti_de_sitter
M-6: check_numerical_stability rejects empty x_values
M-3: superficial_divergence validates dim > 0
M-9: natural_to_si validates finite quantity
m-2: log_dimensions casts expr_value to float
"""

import numpy as np
import pytest

# ============================================================================
# C-2: symbols length validation (tensors.py)
# ============================================================================


class TestDeSitterSymbolsValidation:
    """de_sitter must reject symbols with wrong length."""

    def test_too_few_symbols(self):
        pytest.importorskip("OGRePy")
        import sympy as sp

        from sct_tools import tensors

        syms = (sp.Symbol("t"), sp.Symbol("r"))
        with pytest.raises(ValueError, match="^de_sitter: symbols must have exactly 4"):
            tensors.de_sitter(symbols=syms)

    def test_too_many_symbols(self):
        pytest.importorskip("OGRePy")
        import sympy as sp

        from sct_tools import tensors

        syms = tuple(sp.Symbol(f"x{i}") for i in range(5))
        with pytest.raises(ValueError, match="^de_sitter: symbols must have exactly 4"):
            tensors.de_sitter(symbols=syms)

    def test_none_symbols_ok(self):
        pytest.importorskip("OGRePy")
        from sct_tools import tensors

        metric, params = tensors.de_sitter()
        assert metric is not None


class TestReissnerNordstromSymbolsValidation:
    """reissner_nordstrom must reject symbols with wrong length."""

    def test_too_few_symbols(self):
        pytest.importorskip("OGRePy")
        import sympy as sp

        from sct_tools import tensors

        syms = (sp.Symbol("t"),)
        with pytest.raises(
            ValueError, match="^reissner_nordstrom: symbols must have exactly 4"
        ):
            tensors.reissner_nordstrom(symbols=syms)

    def test_correct_symbols_ok(self):
        pytest.importorskip("OGRePy")
        from sct_tools import tensors

        metric, params = tensors.reissner_nordstrom()
        assert metric is not None


class TestAntiDeSitterSymbolsValidation:
    """anti_de_sitter must reject symbols with wrong length."""

    def test_too_few_symbols(self):
        pytest.importorskip("OGRePy")
        import sympy as sp

        from sct_tools import tensors

        syms = (sp.Symbol("t"), sp.Symbol("r"), sp.Symbol("theta"))
        with pytest.raises(
            ValueError, match="^anti_de_sitter: symbols must have exactly 4"
        ):
            tensors.anti_de_sitter(symbols=syms)

    def test_none_symbols_ok(self):
        pytest.importorskip("OGRePy")
        from sct_tools import tensors

        metric, params = tensors.anti_de_sitter()
        assert metric is not None


# ============================================================================
# M-6: check_numerical_stability rejects empty x_values
# ============================================================================


class TestCheckNumericalStabilityEmptyGuard:
    """check_numerical_stability must reject empty x_values."""

    def test_empty_list(self):
        from sct_tools.verification import check_numerical_stability

        with pytest.raises(
            ValueError, match="^check_numerical_stability: x_values must be non-empty"
        ):
            check_numerical_stability(lambda x: x, [])

    def test_empty_array(self):
        from sct_tools.verification import check_numerical_stability

        with pytest.raises(
            ValueError, match="^check_numerical_stability: x_values must be non-empty"
        ):
            check_numerical_stability(lambda x: x, np.array([]))

    def test_nonempty_still_works(self):
        from sct_tools.verification import check_numerical_stability

        # Use closely spaced points so consecutive jumps are small
        result = check_numerical_stability(
            lambda x: x**2, [1.0, 1.01, 1.02]
        )
        assert result["stable"] is True
        assert len(result["values"]) == 3


# ============================================================================
# M-3: superficial_divergence validates dim > 0
# ============================================================================


class TestSuperficialDivergenceDimValidation:
    """superficial_divergence must reject non-positive dim."""

    def test_zero_dim(self):
        import networkx as nx

        from sct_tools.graphs import superficial_divergence

        G = nx.MultiGraph()
        G.add_nodes_from([1, 2])
        G.add_edge(1, 2, propagator="scalar")
        with pytest.raises(
            ValueError,
            match="^superficial_divergence: dim must be a positive number",
        ):
            superficial_divergence(G, dim=0)

    def test_negative_dim(self):
        import networkx as nx

        from sct_tools.graphs import superficial_divergence

        G = nx.MultiGraph()
        G.add_nodes_from([1, 2])
        G.add_edge(1, 2, propagator="scalar")
        with pytest.raises(
            ValueError,
            match="^superficial_divergence: dim must be a positive number",
        ):
            superficial_divergence(G, dim=-2)

    def test_valid_dim(self):
        import networkx as nx

        from sct_tools.graphs import superficial_divergence

        G = nx.MultiGraph()
        G.add_nodes_from([1, 2])
        G.add_edge(1, 2, propagator="scalar")
        result = superficial_divergence(G, dim=4)
        assert isinstance(result, (int, float))


# ============================================================================
# M-9: natural_to_si validates finite quantity
# ============================================================================


class TestNaturalToSiFiniteValidation:
    """natural_to_si must reject NaN/inf quantity."""

    def test_nan_quantity(self):
        from sct_tools.constants import natural_to_si

        with pytest.raises(
            ValueError, match="^natural_to_si: requires finite quantity"
        ):
            natural_to_si(float("nan"), dim_length=1)

    def test_inf_quantity(self):
        from sct_tools.constants import natural_to_si

        with pytest.raises(
            ValueError, match="^natural_to_si: requires finite quantity"
        ):
            natural_to_si(float("inf"), dim_length=1)

    def test_valid_quantity(self):
        from sct_tools.constants import natural_to_si

        result = natural_to_si(1.0, dim_length=1)
        assert np.isfinite(result)


# ============================================================================
# m-2: log_dimensions casts expr_value to float
# ============================================================================


class TestLogDimensionsFloatCast:
    """log_dimensions must accept numeric-like types and cast to float."""

    def test_int_value(self, capsys):
        from sct_tools.constants import log_dimensions

        log_dimensions(42, 2, label="test_int")
        out = capsys.readouterr().out
        assert "4.200000e+01" in out

    def test_numpy_float(self, capsys):
        from sct_tools.constants import log_dimensions

        log_dimensions(np.float64(1.23e-5), 1, label="test_np")
        out = capsys.readouterr().out
        assert "1.230000e-05" in out

    def test_non_numeric_raises(self):
        from sct_tools.constants import log_dimensions

        with pytest.raises((TypeError, ValueError)):
            log_dimensions("not_a_number", 1)

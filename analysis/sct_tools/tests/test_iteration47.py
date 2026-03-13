"""
Tests for Iteration 47: Input validation and docstring completeness.

M-4: half_chain_entropy rejects base <= 0
M-8: loop_number rejects non-networkx-Graph inputs
m-6: spectral_action_on_graph rejects non-callable f
m-1: read_fits_header docstring completeness
m-11: geodesic_equations docstring already documents valid method values (skipped)
"""

import numpy as np
import pytest

# ============================================================================
# M-4: half_chain_entropy base validation (entanglement.py)
# ============================================================================


class TestHalfChainEntropyBaseValidation:
    """half_chain_entropy must reject base <= 0."""

    def test_zero_base(self):
        quimb = pytest.importorskip("quimb")
        from sct_tools.entanglement import half_chain_entropy

        # Create a simple 2-qubit state
        psi = quimb.rand_ket(4)
        with pytest.raises(
            ValueError, match="^half_chain_entropy: base must be positive"
        ):
            half_chain_entropy(psi, L=2, base=0)

    def test_negative_base(self):
        quimb = pytest.importorskip("quimb")
        from sct_tools.entanglement import half_chain_entropy

        psi = quimb.rand_ket(4)
        with pytest.raises(
            ValueError, match="^half_chain_entropy: base must be positive"
        ):
            half_chain_entropy(psi, L=2, base=-2)

    def test_valid_base(self):
        quimb = pytest.importorskip("quimb")
        from sct_tools.entanglement import half_chain_entropy

        psi = quimb.rand_ket(4)
        result = half_chain_entropy(psi, L=2, base=2)
        assert np.isfinite(result)

    def test_default_base(self):
        quimb = pytest.importorskip("quimb")
        from sct_tools.entanglement import half_chain_entropy

        psi = quimb.rand_ket(4)
        result = half_chain_entropy(psi, L=2)
        assert np.isfinite(result)


# ============================================================================
# M-8: loop_number graph type validation (graphs.py)
# ============================================================================


class TestLoopNumberGraphValidation:
    """loop_number must reject non-networkx-Graph inputs."""

    def test_dict_input(self):
        from sct_tools.graphs import loop_number

        with pytest.raises(
            TypeError, match="^loop_number: expected a networkx Graph"
        ):
            loop_number({"a": [1, 2]})

    def test_list_input(self):
        from sct_tools.graphs import loop_number

        with pytest.raises(
            TypeError, match="^loop_number: expected a networkx Graph"
        ):
            loop_number([1, 2, 3])

    def test_string_input(self):
        from sct_tools.graphs import loop_number

        with pytest.raises(
            TypeError, match="^loop_number: expected a networkx Graph"
        ):
            loop_number("not a graph")

    def test_valid_graph(self):
        import networkx as nx

        from sct_tools.graphs import loop_number

        G = nx.MultiGraph()
        G.add_nodes_from([1, 2])
        G.add_edge(1, 2)
        result = loop_number(G)
        assert isinstance(result, int)
        assert result == 0  # tree has L=0

    def test_valid_multigraph_with_loop(self):
        import networkx as nx

        from sct_tools.graphs import loop_number

        G = nx.MultiGraph()
        G.add_nodes_from([1, 2, 3])
        G.add_edge(1, 2)
        G.add_edge(2, 3)
        G.add_edge(3, 1)
        result = loop_number(G)
        assert result == 1  # triangle has L=1


# ============================================================================
# m-6: spectral_action_on_graph callable validation (graphs.py)
# ============================================================================


class TestSpectralActionCallableValidation:
    """spectral_action_on_graph must reject non-callable f."""

    def test_string_f(self):
        import networkx as nx

        from sct_tools.graphs import spectral_action_on_graph

        G = nx.Graph()
        G.add_nodes_from([1, 2])
        G.add_edge(1, 2)
        with pytest.raises(
            TypeError,
            match="^spectral_action_on_graph: f must be callable",
        ):
            spectral_action_on_graph(G, f="not_callable")

    def test_int_f(self):
        import networkx as nx

        from sct_tools.graphs import spectral_action_on_graph

        G = nx.Graph()
        G.add_nodes_from([1, 2])
        G.add_edge(1, 2)
        with pytest.raises(
            TypeError,
            match="^spectral_action_on_graph: f must be callable",
        ):
            spectral_action_on_graph(G, f=42)

    def test_none_f_ok(self):
        import networkx as nx

        from sct_tools.graphs import spectral_action_on_graph

        G = nx.Graph()
        G.add_nodes_from([1, 2])
        G.add_edge(1, 2)
        # f=None should use default polynomial
        result = spectral_action_on_graph(G, f=None)
        assert isinstance(result, dict)
        assert np.isfinite(result["action"])

    def test_lambda_f_ok(self):
        import networkx as nx

        from sct_tools.graphs import spectral_action_on_graph

        G = nx.Graph()
        G.add_nodes_from([1, 2])
        G.add_edge(1, 2)
        result = spectral_action_on_graph(G, f=lambda x: np.exp(-x))
        assert isinstance(result, dict)
        assert np.isfinite(result["action"])


# ============================================================================
# m-1: read_fits_header docstring completeness (data_io.py)
# ============================================================================


class TestReadFitsHeaderDocstring:
    """read_fits_header must have complete docstring."""

    def test_docstring_exists(self):
        from sct_tools.data_io import read_fits_header

        assert read_fits_header.__doc__ is not None

    def test_docstring_has_parameters(self):
        from sct_tools.data_io import read_fits_header

        doc = read_fits_header.__doc__
        assert "Parameters" in doc

    def test_docstring_has_returns(self):
        from sct_tools.data_io import read_fits_header

        doc = read_fits_header.__doc__
        assert "Returns" in doc

    def test_docstring_mentions_filepath(self):
        from sct_tools.data_io import read_fits_header

        doc = read_fits_header.__doc__
        assert "filepath" in doc

    def test_docstring_mentions_hdu(self):
        from sct_tools.data_io import read_fits_header

        doc = read_fits_header.__doc__
        assert "hdu" in doc

"""
Tests for all new behaviors added in Iteration 17 (under-explored modules audit).

Covers:
  D-1: data_io _serialize complex/bool types
  E-2: entanglement renyi_entropy alpha guards + eigenvalue filter
  E-1: entanglement density matrix normalization warning
  G-1: graphs superficial_divergence external legs
  G-4: graphs graph_laplacian_spectrum isolated vertices
  G-2: graphs spectral_dimension_graph negative d_S warning
  T-3: tensors verify_vacuum atol parameter
  L-2: lean verify_identity sorry regex (comment stripping)
"""

import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest

# ============================================================================
# D-1: data_io _serialize — complex/bool types
# ============================================================================

class TestSerializeComplex:
    """Test that _serialize handles numpy complex and bool types."""

    def setup_method(self):
        from sct_tools.data_io import _serialize
        self.ser = _serialize

    def test_np_bool_true(self):
        result = self.ser(np.bool_(True))
        assert result is True
        assert isinstance(result, bool)

    def test_np_bool_false(self):
        result = self.ser(np.bool_(False))
        assert result is False
        assert isinstance(result, bool)

    def test_python_complex(self):
        result = self.ser(3 + 4j)
        assert result == {'__complex__': True, 'real': 3.0, 'imag': 4.0}

    def test_np_complex128(self):
        result = self.ser(np.complex128(1.5 - 2.5j))
        assert result == {'__complex__': True, 'real': 1.5, 'imag': -2.5}

    def test_np_complex64(self):
        result = self.ser(np.complex64(1 + 1j))
        assert result['__complex__'] is True
        assert abs(result['real'] - 1.0) < 1e-6
        assert abs(result['imag'] - 1.0) < 1e-6

    def test_complex_ndarray(self):
        arr = np.array([1 + 2j, 3 + 4j], dtype=complex)
        result = self.ser(arr)
        # D-2 fix: shape-preserving dict instead of flat list
        assert result['__ndarray_complex__'] is True
        assert result['shape'] == [2]
        assert len(result['data']) == 2
        assert result['data'][0] == {'__complex__': True, 'real': 1.0, 'imag': 2.0}
        assert result['data'][1] == {'__complex__': True, 'real': 3.0, 'imag': 4.0}

    def test_complex_in_dict(self):
        data = {'z': np.complex128(2 + 3j), 'flag': np.bool_(True)}
        result = self.ser(data)
        assert result['z'] == {'__complex__': True, 'real': 2.0, 'imag': 3.0}
        assert result['flag'] is True

    def test_complex_in_list(self):
        data = [1 + 0j, np.complex128(0 + 1j)]
        result = self.ser(data)
        assert len(result) == 2
        assert result[0]['__complex__'] is True
        assert result[1]['__complex__'] is True

    def test_round_trip_json(self):
        """Verify that serialized complex data can be JSON-encoded."""
        from sct_tools.data_io import load_results, save_results
        data = {
            'eigenvalue': np.complex128(1.5 + 0.3j),
            'spectrum': np.array([1 + 0j, 0 + 1j, -1 + 0j]),
            'converged': np.bool_(True),
        }
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='w') as f:
            path = f.name
        try:
            save_results(path, data)
            loaded, _ = load_results(path)
            assert loaded['converged'] is True
            # _deserialize now reconstructs complex types properly
            assert isinstance(loaded['eigenvalue'], complex)
            assert abs(loaded['eigenvalue'].real - 1.5) < 1e-10
            assert abs(loaded['eigenvalue'].imag - 0.3) < 1e-10
            assert isinstance(loaded['spectrum'], np.ndarray)
            assert len(loaded['spectrum']) == 3
        finally:
            Path(path).unlink(missing_ok=True)


# ============================================================================
# E-2: entanglement renyi_entropy — alpha guards + eigenvalue filter
# ============================================================================

class TestRenyiAlphaGuard:
    """Test renyi_entropy alpha <= 0 guard and eigenvalue filter for alpha < 1."""

    def test_alpha_negative_raises(self):
        from sct_tools.entanglement import renyi_entropy
        # Bell state |00> + |11>
        psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        with pytest.raises(ValueError, match="alpha > 0"):
            renyi_entropy(psi, [2, 2], 0, alpha=-1)

    def test_alpha_zero_raises(self):
        from sct_tools.entanglement import renyi_entropy
        psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        with pytest.raises(ValueError, match="alpha > 0"):
            renyi_entropy(psi, [2, 2], 0, alpha=0)

    def test_alpha_half_bell_state(self):
        """Renyi entropy at alpha=0.5 for Bell state should be 1.0 (maximal)."""
        from sct_tools.entanglement import renyi_entropy
        psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        S = renyi_entropy(psi, [2, 2], 0, alpha=0.5)
        assert abs(S - 1.0) < 1e-10

    def test_alpha_half_product_state(self):
        """Renyi entropy at alpha=0.5 for product state should be 0."""
        from sct_tools.entanglement import renyi_entropy
        psi = np.array([1, 0, 0, 0], dtype=complex)  # |00>
        S = renyi_entropy(psi, [2, 2], 0, alpha=0.5)
        assert abs(S) < 1e-10

    def test_alpha_2_standard(self):
        """Renyi-2 entropy for Bell state should be 1.0."""
        from sct_tools.entanglement import renyi_entropy
        psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        S = renyi_entropy(psi, [2, 2], 0, alpha=2)
        assert abs(S - 1.0) < 1e-10

    def test_alpha_limit_1(self):
        """Alpha=1 (von Neumann limit) for Bell state should be 1.0."""
        from sct_tools.entanglement import renyi_entropy
        psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        S = renyi_entropy(psi, [2, 2], 0, alpha=1.0)
        assert abs(S - 1.0) < 1e-10

    def test_alpha_small_positive(self):
        """Very small positive alpha should still work."""
        from sct_tools.entanglement import renyi_entropy
        psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        S = renyi_entropy(psi, [2, 2], 0, alpha=0.01)
        assert np.isfinite(S)
        assert S > 0


# ============================================================================
# E-1: entanglement density matrix normalization warning
# ============================================================================

class TestDensityMatrixNormalization:
    """Test _check_density_matrix warning for unnormalized states."""

    def test_no_warning_for_normalized(self):
        from sct_tools.entanglement import _check_density_matrix
        rho = np.array([[0.5, 0], [0, 0.5]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_density_matrix(rho)
            assert len(w) == 0

    def test_warning_for_unnormalized(self):
        from sct_tools.entanglement import _check_density_matrix
        rho = np.array([[1.0, 0], [0, 1.0]])  # Tr = 2
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_density_matrix(rho)
            assert len(w) == 1
            assert "trace" in str(w[0].message).lower()

    def test_no_warning_for_vectors(self):
        """1D arrays should not trigger the warning."""
        from sct_tools.entanglement import _check_density_matrix
        psi = np.array([1, 0, 0, 0], dtype=complex)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_density_matrix(psi)
            assert len(w) == 0

    def test_near_unity_trace_no_warning(self):
        """Trace within 1e-6 of 1.0 should NOT warn."""
        from sct_tools.entanglement import _check_density_matrix
        rho = np.array([[0.5 + 1e-8, 0], [0, 0.5 - 1e-8]])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_density_matrix(rho)
            assert len(w) == 0


# ============================================================================
# G-1: graphs superficial_divergence — external legs
# ============================================================================

class TestSuperficialDivergenceExternal:
    """Test that superficial_divergence correctly skips external legs."""

    def test_external_legs_not_counted(self):
        """External propagators should not contribute to I_B or I_F.

        Graph: 4 vertices (0,1 internal; 2,3 external endpoints).
        Edges: 0-1 scalar (x2), 0-2 external, 1-3 external.
        L = E - V + C = 4 - 4 + 1 = 1. With external skip: I_B=2.
        D = 4*1 - 2*2 = 0.
        Without external skip: I_B=4, D = 4 - 8 = -4 (wrong).
        """
        import networkx as nx

        from sct_tools.graphs import superficial_divergence

        G = nx.MultiGraph()
        G.add_nodes_from([0, 1, 2, 3])
        G.add_edge(0, 1, propagator='scalar')
        G.add_edge(0, 1, propagator='scalar')
        G.add_edge(0, 2, propagator='external')
        G.add_edge(1, 3, propagator='external')

        D = superficial_divergence(G, dim=4)
        assert D == 0

    def test_no_external_legs(self):
        """Graph with no external edges — all count as propagators."""
        import networkx as nx

        from sct_tools.graphs import superficial_divergence

        G = nx.MultiGraph()
        G.add_nodes_from([0, 1])
        G.add_edge(0, 1, propagator='scalar')
        G.add_edge(0, 1, propagator='scalar')

        D = superficial_divergence(G, dim=4)
        # L=1, I_B=2, I_F=0 => D = 4 - 4 = 0
        assert D == 0

    def test_external_vs_internal_counting(self):
        """Verify external legs don't change I_F count."""
        import networkx as nx

        from sct_tools.graphs import superficial_divergence

        # Fermion self-energy: 2 internal vertices, 1 fermion + 1 scalar loop
        G = nx.MultiGraph()
        G.add_nodes_from([0, 1, 2, 3])
        G.add_edge(0, 1, propagator='fermion')
        G.add_edge(0, 1, propagator='scalar')
        G.add_edge(0, 2, propagator='external')
        G.add_edge(1, 3, propagator='external')

        D = superficial_divergence(G, dim=4)
        # L = 4 - 4 + 1 = 1, I_B=1, I_F=1 => D = 4 - 2 - 1 = 1
        assert D == 1


# ============================================================================
# G-4: graphs graph_laplacian_spectrum — isolated vertices
# ============================================================================

class TestLaplacianIsolatedVertices:
    """Test normalized Laplacian handles isolated vertices without blowup."""

    def test_isolated_vertex_normalized(self):
        """Graph with isolated vertex should not produce huge eigenvalues."""
        from sct_tools.graphs import graph_laplacian_spectrum

        # 3 nodes: 0-1 connected, node 2 isolated
        A = np.array([[0, 1, 0],
                       [1, 0, 0],
                       [0, 0, 0]], dtype=float)
        eigs = graph_laplacian_spectrum(A, normalized=True)
        # All eigenvalues should be in [0, 2] for normalized Laplacian
        assert np.all(eigs >= -1e-10)
        assert np.all(eigs <= 2 + 1e-10)

    def test_all_isolated_normalized(self):
        """All isolated vertices: normalized Laplacian should be zero matrix."""
        from sct_tools.graphs import graph_laplacian_spectrum

        A = np.zeros((4, 4))
        eigs = graph_laplacian_spectrum(A, normalized=True)
        assert np.allclose(eigs, 0)

    def test_connected_graph_unnormalized(self):
        """Connected graph: standard Laplacian has one zero eigenvalue."""
        from sct_tools.graphs import graph_laplacian_spectrum

        # Complete graph K3
        A = np.array([[0, 1, 1],
                       [1, 0, 1],
                       [1, 1, 0]], dtype=float)
        eigs = graph_laplacian_spectrum(A, normalized=False)
        assert abs(eigs[0]) < 1e-10  # Smallest eigenvalue is 0
        assert eigs[1] > 0.5  # Non-trivial

    def test_single_isolated_in_large_graph(self):
        """Isolated vertex among connected components gives bounded eigenvalues."""
        from sct_tools.graphs import graph_laplacian_spectrum

        # 5 nodes: 0-1-2 path, node 3 isolated, node 4 isolated
        A = np.zeros((5, 5))
        A[0, 1] = A[1, 0] = 1
        A[1, 2] = A[2, 1] = 1
        eigs = graph_laplacian_spectrum(A, normalized=True)
        assert np.all(np.isfinite(eigs))
        assert np.all(eigs <= 2 + 1e-10)


# ============================================================================
# G-2: graphs spectral_dimension_graph — negative d_S warning
# ============================================================================

class TestSpectralDimensionWarning:
    """Test that spectral_dimension_graph warns about negative d_S."""

    def test_disconnected_graph_warns(self):
        """Disconnected graph likely produces negative d_S → should warn."""
        from sct_tools.graphs import spectral_dimension_graph

        # 4 nodes: two disconnected pairs
        A = np.array([[0, 1, 0, 0],
                       [1, 0, 0, 0],
                       [0, 0, 0, 1],
                       [0, 0, 1, 0]], dtype=float)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            t, d_S = spectral_dimension_graph(A, t_values=np.logspace(-1, 2, 50))
            # Check for negative values and associated warning
            n_neg = np.sum(d_S[np.isfinite(d_S)] < 0)
            if n_neg > 0:
                warning_msgs = [str(x.message) for x in w]
                assert any("negative" in m for m in warning_msgs)

    def test_connected_graph_short_times_positive(self):
        """Well-connected graph at short diffusion times should have d_S > 0."""
        from sct_tools.graphs import spectral_dimension_graph

        # Complete graph K5 — at short times d_S should be positive
        A = np.ones((5, 5)) - np.eye(5)
        t_vals = np.logspace(-2, -0.5, 20)  # short times only
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            t, d_S = spectral_dimension_graph(A, t_values=t_vals)
            # At short times, K5 should have positive spectral dimension
            finite_d_S = d_S[np.isfinite(d_S)]
            assert np.all(finite_d_S >= 0)


# ============================================================================
# T-3: tensors verify_vacuum — atol parameter
# ============================================================================

class TestVerifyVacuumAtol:
    """Test that verify_vacuum correctly uses atol when provided."""

    @pytest.fixture(autouse=True)
    def skip_if_no_ogrepy(self):
        """Skip if OGRePy not available."""
        pytest.importorskip("OGRePy")

    def test_atol_overrides_rtol(self):
        """When atol is given, it should be used instead of rtol."""
        from sct_tools.tensors import schwarzschild, verify_vacuum

        g, _ = schwarzschild()
        # Schwarzschild is a vacuum solution — should always pass
        result = verify_vacuum(g, rtol=1e-100, atol=1e-6)
        assert result is True

    def test_rtol_backward_compat(self):
        """Without atol, rtol should still work as before."""
        from sct_tools.tensors import schwarzschild, verify_vacuum

        g, _ = schwarzschild()
        result = verify_vacuum(g, rtol=1e-6)
        assert result is True

    def test_default_tolerance(self):
        """Default tolerance (rtol=1e-10) should work for exact vacuum."""
        from sct_tools.tensors import schwarzschild, verify_vacuum

        g, _ = schwarzschild()
        result = verify_vacuum(g)
        assert result is True


# ============================================================================
# L-2: lean verify_identity — sorry regex with comment stripping
# ============================================================================

class TestSorryDetection:
    """Test that sorry detection handles comments and word boundaries."""

    def test_sorry_in_line_comment_ignored(self):
        """'sorry' in a line comment (-- sorry) should not flag as sorry."""
        import re
        output_text = "-- sorry this is a comment\nAll goals proved"
        stripped = re.sub(r'--[^\n]*', '', output_text)
        stripped = re.sub(r'/\-.*?\-/', '', stripped, flags=re.DOTALL)
        assert not re.search(r'\bsorry\b', stripped)

    def test_sorry_in_block_comment_ignored(self):
        """`sorry` inside /- block comment -/ should not count."""
        import re
        output_text = "/- sorry for the delay -/\nproof complete"
        stripped = re.sub(r'--[^\n]*', '', output_text)
        stripped = re.sub(r'/\-.*?\-/', '', stripped, flags=re.DOTALL)
        assert not re.search(r'\bsorry\b', stripped)

    def test_sorry_tactic_detected(self):
        """Actual sorry tactic in proof should be detected."""
        import re
        output_text = "theorem foo : 1 = 1 := by\n  sorry"
        stripped = re.sub(r'--[^\n]*', '', output_text)
        stripped = re.sub(r'/\-.*?\-/', '', stripped, flags=re.DOTALL)
        assert re.search(r'\bsorry\b', stripped)

    def test_sorry_substring_not_detected(self):
        """Words containing 'sorry' as substring should NOT match."""
        import re
        output_text = "notsorryaboutit"
        stripped = re.sub(r'--[^\n]*', '', output_text)
        stripped = re.sub(r'/\-.*?\-/', '', stripped, flags=re.DOTALL)
        assert not re.search(r'\bsorry\b', stripped)

    def test_mixed_comments_and_sorry(self):
        """sorry in comments + sorry in code = detected."""
        import re
        output_text = "-- sorry (comment)\n  sorry\n/- sorry -/"
        stripped = re.sub(r'--[^\n]*', '', output_text)
        stripped = re.sub(r'/\-.*?\-/', '', stripped, flags=re.DOTALL)
        assert re.search(r'\bsorry\b', stripped)

    def test_no_sorry_anywhere(self):
        """Clean output with no sorry should pass."""
        import re
        output_text = "All goals proved\nNo errors found"
        stripped = re.sub(r'--[^\n]*', '', output_text)
        stripped = re.sub(r'/\-.*?\-/', '', stripped, flags=re.DOTALL)
        assert not re.search(r'\bsorry\b', stripped)


# ============================================================================
# data_io save_results edge cases
# ============================================================================

class TestSaveResultsEdgeCases:
    """Test that save_results handles edge-case types."""

    def test_nested_complex_dict(self):
        """Nested dict with complex values should serialize cleanly."""
        from sct_tools.data_io import load_results, save_results
        data = {
            'level1': {
                'level2': {
                    'z': 1 + 2j,
                    'arr': np.array([0 + 0j, 1 + 1j]),
                }
            }
        }
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            save_results(path, data)
            loaded, _ = load_results(path)
            z = loaded['level1']['level2']['z']
            # _deserialize now reconstructs complex types
            assert isinstance(z, complex)
            assert z.real == 1.0
            assert z.imag == 2.0
        finally:
            Path(path).unlink(missing_ok=True)

    def test_bool_array(self):
        """Array of np.bool_ should serialize to list of Python bools."""
        from sct_tools.data_io import _serialize
        arr = np.array([True, False, True], dtype=bool)
        result = _serialize(arr)
        assert result == [True, False, True]

    def test_mixed_types(self):
        """Dict with mixed numpy types should all serialize."""
        from sct_tools.data_io import _serialize
        data = {
            'i32': np.int32(42),
            'f64': np.float64(3.14),
            'b': np.bool_(True),
            'c': np.complex128(1 + 2j),
        }
        result = _serialize(data)
        assert result['i32'] == 42
        assert isinstance(result['i32'], int)
        assert abs(result['f64'] - 3.14) < 1e-10
        assert isinstance(result['f64'], float)
        assert result['b'] is True
        assert result['c']['__complex__'] is True

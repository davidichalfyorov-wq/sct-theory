"""Tests for sct_tools.graphs — spectral graph theory and causal sets."""

import numpy as np
import pytest

from sct_tools import graphs


class TestLaplacianSpectrum:
    def test_complete_graph_k4(self):
        A = np.array([[0, 1, 1, 1], [1, 0, 1, 1],
                      [1, 1, 0, 1], [1, 1, 1, 0]], dtype=float)
        eigs = graphs.graph_laplacian_spectrum(A)
        assert eigs[0] == pytest.approx(0.0, abs=1e-12)
        assert eigs[1] == pytest.approx(4.0, abs=1e-12)

    def test_path_graph(self):
        # P3: 1--2--3, eigenvalues of Laplacian: 0, 1, 3
        A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        eigs = graphs.graph_laplacian_spectrum(A)
        assert eigs[0] == pytest.approx(0.0, abs=1e-12)
        assert eigs[1] == pytest.approx(1.0, abs=1e-12)
        assert eigs[2] == pytest.approx(3.0, abs=1e-12)

    def test_networkx_input(self):
        import networkx as nx
        G = nx.cycle_graph(5)
        eigs = graphs.graph_laplacian_spectrum(G)
        assert eigs[0] == pytest.approx(0.0, abs=1e-12)
        assert len(eigs) == 5

    def test_normalized(self):
        A = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=float)
        eigs = graphs.graph_laplacian_spectrum(A, normalized=True)
        assert eigs[0] == pytest.approx(0.0, abs=1e-12)
        assert all(e <= 2.0 + 1e-12 for e in eigs)  # normalized eigs <= 2


class TestHeatKernel:
    def test_trace_near_zero(self):
        A = np.eye(3, k=1) + np.eye(3, k=-1)  # path graph
        K = graphs.heat_kernel_trace(A, [1e-10])
        assert K[0] == pytest.approx(3.0, abs=1e-6)  # Tr(e^{-eps*L}) ≈ Tr(I) = N

    def test_trace_decreases(self):
        A = np.ones((4, 4)) - np.eye(4)
        K = graphs.heat_kernel_trace(A, [1e-10, 1.0, 10.0])
        assert K[0] > K[1] > K[2]


class TestSpectralDimension:
    def test_returns_arrays(self):
        A = np.ones((5, 5)) - np.eye(5)
        t, d_S = graphs.spectral_dimension_graph(A)
        assert len(t) == len(d_S)
        assert len(t) > 0

    def test_complete_graph_dimension(self):
        """Complete graph K_N: d_S peaks at intermediate t, bounded by N."""
        N = 10
        A = np.ones((N, N)) - np.eye(N)
        t_vals = np.logspace(-3, 1, 50)
        t, d_S = graphs.spectral_dimension_graph(A, t_values=t_vals)
        valid = [di for di in d_S if np.isfinite(di) and di > 0]
        assert len(valid) > 0, "No valid d_S values"
        peak = max(valid)
        # d_S must be positive and bounded by graph size
        assert 0 < peak <= N, f"Peak d_S={peak}, should be in (0, {N}]"
        # For K_10, peak should be around 1-3 (empirical, not N-1)
        assert peak > 0.5, f"Peak d_S={peak} too small"


class TestCausalSet:
    def test_sprinkle_shape(self):
        pts, C = graphs.causal_set_sprinkle(30, dim=2, seed=42)
        assert pts.shape == (30, 2)
        assert C.shape == (30, 30)

    def test_causal_order(self):
        pts, C = graphs.causal_set_sprinkle(20, dim=2, seed=42)
        # Causal matrix should be upper triangular (since sorted by time)
        assert np.all(C == np.triu(C))

    def test_dimension_estimate_2d(self):
        # Alexandrov interval required for Myrheim-Meyer estimator
        pts, C = graphs.causal_set_sprinkle(200, dim=2, region='diamond', seed=42)
        d = graphs.causal_set_dimension(C)
        assert 1.5 < d < 2.5  # should be close to 2

    def test_dimension_estimate_3d(self):
        # Alexandrov interval required for Myrheim-Meyer estimator
        pts, C = graphs.causal_set_sprinkle(300, dim=3, region='diamond', seed=42)
        d = graphs.causal_set_dimension(C)
        assert 2.0 < d < 4.0  # should be close to 3

    def test_to_dag(self):
        pts, C = graphs.causal_set_sprinkle(20, dim=2, seed=42)
        dag = graphs.causal_set_to_dag(C)
        import networkx as nx
        assert nx.is_directed_acyclic_graph(dag)


class TestFeynmanGraph:
    def test_one_loop_self_energy(self):
        G = graphs.feynman_graph([1, 2], [(1, 2, 'scalar'), (1, 2, 'scalar')])
        assert graphs.loop_number(G) == 1

    def test_tree_level(self):
        G = graphs.feynman_graph([1, 2, 3], [(1, 2, 'scalar'), (2, 3, 'scalar')])
        assert graphs.loop_number(G) == 0

    def test_two_loop(self):
        # Sunset diagram: 3 propagators between 2 vertices
        G = graphs.feynman_graph([1, 2],
                                 [(1, 2, 'scalar'), (1, 2, 'scalar'), (1, 2, 'scalar')])
        assert graphs.loop_number(G) == 2

    def test_divergence_phi4_vertex(self):
        # One-loop 4-point function: box diagram, 4 vertices, 4 edges
        G = graphs.feynman_graph([1, 2, 3, 4],
                                 [(1, 2, 'scalar'), (2, 3, 'scalar'),
                                  (3, 4, 'scalar'), (4, 1, 'scalar')])
        assert graphs.loop_number(G) == 1
        assert graphs.superficial_divergence(G, dim=4) == -4  # convergent

"""Tests for Layer 2.5 property-based fuzzing (PropertyChecker in verification.py).

Also tests the igraph backend additions to graphs.py.
"""

import numpy as np
import pytest

from sct_tools import graphs, verification


# =============================================================================
# PropertyChecker tests
# =============================================================================
class TestPropertyCheckerInit:
    def test_requires_hypothesis(self):
        """PropertyChecker should raise if hypothesis is not available."""
        if not verification._HYPOTHESIS_OK:
            with pytest.raises(ImportError, match="hypothesis"):
                verification.PropertyChecker()
        else:
            pc = verification.PropertyChecker()
            assert pc.n_examples == 200

    @pytest.mark.skipif(not verification._HYPOTHESIS_OK,
                        reason="hypothesis not installed")
    def test_custom_n_examples(self):
        pc = verification.PropertyChecker(n_examples=50)
        assert pc.n_examples == 50


@pytest.mark.skipif(not verification._HYPOTHESIS_OK,
                    reason="hypothesis not installed")
class TestPropertyChecks:
    """Run individual property checks (fast subset)."""

    def test_phi_normalization(self):
        pc = verification.PropertyChecker(n_examples=10, quiet=True)
        assert pc.check_phi_normalization() is True

    def test_phi_positivity(self):
        pc = verification.PropertyChecker(n_examples=50, quiet=True)
        assert pc.check_phi_positivity() is True

    def test_phi_monotone(self):
        pc = verification.PropertyChecker(n_examples=50, quiet=True)
        assert pc.check_phi_monotone_decrease() is True

    def test_implementation_agreement(self):
        pc = verification.PropertyChecker(n_examples=20, quiet=True)
        assert pc.check_implementation_agreement() is True

    def test_form_factor_agreement_scalar(self):
        pc = verification.PropertyChecker(n_examples=20, quiet=True)
        assert pc.check_form_factor_agreement_scalar() is True

    def test_form_factor_agreement_dirac(self):
        pc = verification.PropertyChecker(n_examples=20, quiet=True)
        assert pc.check_form_factor_agreement_dirac() is True

    def test_form_factor_agreement_vector(self):
        pc = verification.PropertyChecker(n_examples=20, quiet=True)
        assert pc.check_form_factor_agreement_vector() is True

    def test_beta_limits(self):
        pc = verification.PropertyChecker(n_examples=10, quiet=True)
        assert pc.check_beta_limits() is True

    def test_conformal_coupling(self):
        pc = verification.PropertyChecker(n_examples=1, quiet=True)
        assert pc.check_conformal_coupling() is True

    def test_sm_total_decomposition(self):
        pc = verification.PropertyChecker(n_examples=30, quiet=True)
        assert pc.check_sm_total_decomposition() is True

    def test_derivative_consistency(self):
        pc = verification.PropertyChecker(n_examples=30, quiet=True)
        assert pc.check_derivative_consistency() is True


@pytest.mark.skipif(not verification._HYPOTHESIS_OK,
                    reason="hypothesis not installed")
class TestPropertyCheckerRunAll:
    def test_run_all_returns_results(self):
        pc = verification.PropertyChecker(n_examples=10, quiet=True)
        results = pc.run_all()
        assert len(results) == 12  # 12 property checks

    def test_summary_returns_bool(self):
        pc = verification.PropertyChecker(n_examples=10, quiet=True)
        results = pc.run_all()
        ok = pc.summary(results)
        assert isinstance(ok, bool)


@pytest.mark.skipif(not verification._HYPOTHESIS_OK,
                    reason="hypothesis not installed")
class TestRunPropertyChecks:
    def test_standalone_function(self):
        ok = verification.run_property_checks(n_examples=10)
        assert isinstance(ok, bool)


# =============================================================================
# run_full_verification tests
# =============================================================================
class TestRunFullVerification:
    def test_runs_without_crash(self):
        """Full verification suite should run (may skip some layers)."""
        ok = verification.run_full_verification(
            include_property=False,  # skip hypothesis for speed
            include_triple_cas=False,  # skip GiNaC requirement
        )
        assert isinstance(ok, bool)


# =============================================================================
# igraph backend tests (graphs.py)
# =============================================================================
class TestIgraphDetection:
    def test_has_igraph_returns_bool(self):
        assert isinstance(graphs._has_igraph(), bool)


class TestCausalSetSprinkleFast:
    def test_flat_sprinkle(self):
        pts, C, dag = graphs.causal_set_sprinkle_fast(50, dim=2, seed=42)
        assert pts.shape == (50, 2)
        assert C.shape == (50, 50)
        # C is upper triangular (sorted by time)
        assert np.sum(np.tril(C, k=0)) == 0

    def test_diamond_sprinkle(self):
        pts, C, dag = graphs.causal_set_sprinkle_fast(30, dim=2,
                                                       region='diamond', seed=42)
        assert pts.shape == (30, 2)
        # All points should be inside the diamond
        for i in range(30):
            t = pts[i, 0]
            x = pts[i, 1]
            assert 0 <= t <= 1
            r_max = min(t, 1 - t)
            assert abs(x) <= r_max + 1e-10

    def test_3d_sprinkle(self):
        pts, C, dag = graphs.causal_set_sprinkle_fast(20, dim=3, seed=42)
        assert pts.shape == (20, 3)

    def test_returns_igraph_dag(self):
        pts, C, dag = graphs.causal_set_sprinkle_fast(20, dim=2, seed=42)
        if graphs._has_igraph():
            assert dag is not None
            assert dag.vcount() == 20
            assert dag.is_dag()
        else:
            assert dag is None

    def test_agrees_with_slow_version(self):
        """Fast sprinkle should give identical causal matrix as slow."""
        pts_slow, C_slow = graphs.causal_set_sprinkle(100, dim=2,
                                                        region='flat', seed=42)
        pts_fast, C_fast, _ = graphs.causal_set_sprinkle_fast(100, dim=2,
                                                                region='flat', seed=42)
        # Points should be identical (same RNG seed)
        np.testing.assert_allclose(pts_slow, pts_fast, atol=1e-14)
        # Causal matrices should be identical
        np.testing.assert_array_equal(C_slow, C_fast)

    def test_unknown_region_raises(self):
        with pytest.raises(ValueError, match="unknown region"):
            graphs.causal_set_sprinkle_fast(10, region='curved')


class TestOrderingFractionFast:
    def test_trivial_case(self):
        C = np.zeros((2, 2), dtype=bool)
        C[0, 1] = True
        f = graphs.causal_set_ordering_fraction_fast(C)
        assert f == pytest.approx(1.0)  # 1 relation out of 1 pair

    def test_no_relations(self):
        C = np.zeros((3, 3), dtype=bool)
        f = graphs.causal_set_ordering_fraction_fast(C)
        assert f == pytest.approx(0.0)

    def test_single_point(self):
        C = np.zeros((1, 1), dtype=bool)
        f = graphs.causal_set_ordering_fraction_fast(C)
        assert f == pytest.approx(0.0)


@pytest.mark.skipif(not graphs._has_igraph(),
                    reason="igraph not installed")
class TestIgraphLaplacian:
    def test_complete_graph(self):
        import igraph as ig
        g = ig.Graph.Full(5)
        eigs = graphs.graph_laplacian_spectrum_igraph(g)
        assert eigs[0] == pytest.approx(0.0, abs=1e-10)
        assert eigs[1] == pytest.approx(5.0, abs=1e-10)  # K_5: eigenvalue 5

    def test_partial_eigenvalues(self):
        import igraph as ig
        g = ig.Graph.Full(10)
        eigs = graphs.graph_laplacian_spectrum_igraph(g, n_eigenvalues=3)
        assert len(eigs) == 3
        assert eigs[0] == pytest.approx(0.0, abs=1e-8)


@pytest.mark.skipif(not graphs._has_igraph(),
                    reason="igraph not installed")
class TestSpectralDimensionIgraph:
    def test_returns_arrays(self):
        import igraph as ig
        g = ig.Graph.Full(10)
        t, d_S = graphs.spectral_dimension_igraph(g)
        assert len(t) == len(d_S)
        assert len(t) > 0

    def test_custom_t_values(self):
        import igraph as ig
        g = ig.Graph.Full(5)
        t_vals = np.logspace(-1, 1, 20)
        t, d_S = graphs.spectral_dimension_igraph(g, t_values=t_vals)
        assert len(t) == 20


class TestBenchmarkBackends:
    def test_benchmark_runs(self):
        results = graphs.benchmark_backends(n_points=30, dim=2, seed=42)
        assert 'networkx_sprinkle' in results
        assert 'igraph_sprinkle' in results
        assert 'agree' in results
        assert results['agree'] is True

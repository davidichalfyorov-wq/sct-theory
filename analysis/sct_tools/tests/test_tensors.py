"""Tests for sct_tools.tensors — GR tensor algebra via OGRePy."""

import unittest

import sympy as sp

from sct_tools import tensors


class TestCoordinates(unittest.TestCase):
    """Test coordinate system creation."""

    def test_spherical_coords(self):
        coords, syms = tensors.spherical_coords()
        self.assertEqual(len(syms), 4)
        self.assertEqual(coords.dim(), 4)

    def test_cartesian_coords_4d(self):
        coords, syms = tensors.cartesian_coords(4)
        self.assertEqual(len(syms), 4)
        self.assertEqual(coords.dim(), 4)

    def test_cartesian_coords_5d(self):
        coords, syms = tensors.cartesian_coords(5)
        self.assertEqual(len(syms), 5)
        self.assertEqual(coords.dim(), 5)


class TestStandardSpacetimes(unittest.TestCase):
    """Test standard spacetime constructors."""

    def test_schwarzschild_metric(self):
        metric, params = tensors.schwarzschild()
        self.assertIn("M", params)
        self.assertEqual(metric.dim(), 4)

    def test_minkowski_metric(self):
        metric, params = tensors.minkowski()
        self.assertEqual(metric.dim(), 4)

    def test_minkowski_5d(self):
        metric, params = tensors.minkowski(dim=5)
        self.assertEqual(metric.dim(), 5)

    def test_flrw_flat(self):
        metric, params = tensors.flrw(k=0)
        self.assertEqual(params["k"], 0)
        self.assertEqual(metric.dim(), 4)

    def test_de_sitter(self):
        metric, params = tensors.de_sitter()
        self.assertIn("Lambda", params)
        self.assertEqual(metric.dim(), 4)


class TestCurvatureInvariants(unittest.TestCase):
    """Test curvature invariant computation."""

    def test_schwarzschild_vacuum(self):
        """Schwarzschild must satisfy R = 0 (vacuum)."""
        metric, _ = tensors.schwarzschild()
        inv = tensors.curvature_invariants(metric)
        self.assertEqual(inv["ricci_scalar"], 0)

    def test_schwarzschild_kretschmann(self):
        """Kretschmann scalar for Schwarzschild = 48*M^2/r^6."""
        metric, params = tensors.schwarzschild()
        inv = tensors.curvature_invariants(metric)
        M = params["M"]
        r = params["symbols"][1]
        expected = 48 * M**2 / r**6
        diff = sp.simplify(inv["kretschmann"] - expected)
        self.assertEqual(diff, 0)

    def test_minkowski_flat(self):
        """Minkowski: all curvature invariants vanish."""
        metric, _ = tensors.minkowski()
        inv = tensors.curvature_invariants(metric)
        self.assertEqual(inv["ricci_scalar"], 0)
        self.assertEqual(inv["kretschmann"], 0)


class TestVacuumVerification(unittest.TestCase):
    """Test vacuum Einstein equation verification."""

    def test_schwarzschild_is_vacuum(self):
        metric, _ = tensors.schwarzschild()
        self.assertTrue(tensors.verify_vacuum(metric))

    def test_minkowski_is_vacuum(self):
        metric, _ = tensors.minkowski()
        self.assertTrue(tensors.verify_vacuum(metric))


class TestGeodesics(unittest.TestCase):
    """Test geodesic equation computation."""

    def test_geodesic_lagrangian(self):
        metric, _ = tensors.schwarzschild()
        geo = tensors.geodesic_equations(metric, method="lagrangian")
        comp = geo.components()
        self.assertEqual(comp.shape, (4,))

    def test_geodesic_christoffel(self):
        metric, _ = tensors.schwarzschild()
        geo = tensors.geodesic_equations(metric, method="christoffel")
        comp = geo.components()
        self.assertEqual(comp.shape, (4,))


class TestLineElement(unittest.TestCase):
    """Test line element extraction."""

    def test_schwarzschild_line_element(self):
        metric, _ = tensors.schwarzschild()
        ds2 = tensors.line_element(metric)
        self.assertIsInstance(ds2, sp.Basic)


class TestTensorToDict(unittest.TestCase):
    """Test tensor component extraction."""

    def test_christoffel_nonzero(self):
        metric, _ = tensors.schwarzschild()
        christoffel = metric.christoffel()
        d = tensors.tensor_to_dict(christoffel)
        # Schwarzschild Christoffel symbols have non-zero components
        self.assertGreater(len(d), 0)

    def test_einstein_all_zero(self):
        metric, _ = tensors.schwarzschild()
        einstein = metric.einstein()
        d = tensors.tensor_to_dict(einstein)
        # Vacuum: all Einstein tensor components should be zero
        self.assertEqual(len(d), 0)


class TestLinearizedMetric(unittest.TestCase):
    """Test perturbation theory setup."""

    def test_perturbation_matrix(self):
        metric, params = tensors.minkowski()
        h, eps = tensors.linearized_metric(metric)
        self.assertEqual(h.shape, (4, 4))
        # Should be symmetric
        for i in range(4):
            for j in range(4):
                self.assertEqual(h[i, j], h[j, i])


class TestKerr(unittest.TestCase):
    """Test Kerr metric constructor."""

    def test_kerr_returns_metric(self):
        metric, params = tensors.kerr()
        self.assertEqual(metric.dim(), 4)
        self.assertIn("M", params)
        self.assertIn("a", params)

    def test_kerr_reduces_to_schwarzschild(self):
        """Kerr with a=0 should give Schwarzschild g_tt."""
        metric, params = tensors.kerr()
        g = metric.components()
        M = params["M"]
        a = params["a"]
        r = params["symbols"][1]
        # Substitute a=0, check g_tt = -(1 - 2M/r)
        g00_a0 = g[0, 0].subs(a, 0)
        expected = -(1 - 2 * M / r)
        diff = sp.simplify(g00_a0 - expected)
        self.assertEqual(diff, 0)


class TestWeylTensor(unittest.TestCase):
    """Test Weyl tensor computation."""

    def test_minkowski_conformally_flat(self):
        """Minkowski space: Weyl tensor must vanish."""
        metric, _ = tensors.minkowski()
        result = tensors.weyl_tensor(metric)
        self.assertTrue(result["is_conformally_flat"])
        self.assertEqual(result["dim"], 4)

    def test_schwarzschild_not_conformally_flat(self):
        """Schwarzschild is NOT conformally flat in 4D."""
        metric, _ = tensors.schwarzschild()
        result = tensors.weyl_tensor(metric)
        self.assertFalse(result["is_conformally_flat"])

    def test_weyl_antisymmetry(self):
        """Weyl tensor must be antisymmetric in first pair: C_{abcd} = -C_{bacd}."""
        metric, _ = tensors.minkowski()
        result = tensors.weyl_tensor(metric)
        C = result["components"]
        dim = result["dim"]
        for a in range(dim):
            for b in range(dim):
                for c in range(dim):
                    for d in range(dim):
                        val = sp.simplify(C[a, b, c, d] + C[b, a, c, d])
                        self.assertEqual(val, 0,
                                         f"Antisymmetry violated at ({a},{b},{c},{d})")


if __name__ == "__main__":
    unittest.main()

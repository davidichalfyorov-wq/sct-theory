"""Tests for sct_tools.entanglement — quimb integration for SCT Axiom 5."""

import unittest

import numpy as np

from sct_tools import entanglement


class TestEntanglementEntropy(unittest.TestCase):
    """Test von Neumann entropy."""

    def test_bell_state_entropy(self):
        """Maximally entangled Bell state: S = 1 bit."""
        import quimb as qu

        psi = qu.bell_state("psi-")
        S = entanglement.entanglement_entropy(psi, [2, 2], 0)
        self.assertAlmostEqual(S, 1.0, places=6)

    def test_product_state_entropy(self):
        """Product state: S = 0."""
        # |00>
        psi = np.array([1, 0, 0, 0], dtype=complex).reshape(-1, 1)
        S = entanglement.entanglement_entropy(psi, [2, 2], 0)
        self.assertAlmostEqual(S, 0.0, places=6)

    def test_entropy_nats(self):
        """Entropy in natural logarithm base."""
        import quimb as qu

        psi = qu.bell_state("psi-")
        S = entanglement.entanglement_entropy(psi, [2, 2], 0, base=np.e)
        self.assertAlmostEqual(S, np.log(2), places=5)


class TestEntanglementDensityMatrix(unittest.TestCase):
    """Ensure density matrix input is correctly partial-traced (not bypassed)."""

    def test_density_matrix_entropy_bell(self):
        """Bell state as density matrix should give S=1 when tracing subsystem."""
        import quimb as qu

        psi = qu.bell_state("psi-")
        rho = psi @ psi.conj().T  # 4x4 density matrix
        S = entanglement.entanglement_entropy(rho, [2, 2], 0)
        self.assertAlmostEqual(S, 1.0, places=5)

    def test_density_matrix_product_zero(self):
        """Product state density matrix should give S=0."""
        psi = np.array([1, 0, 0, 0], dtype=complex).reshape(-1, 1)
        rho = psi @ psi.conj().T
        S = entanglement.entanglement_entropy(rho, [2, 2], 0)
        self.assertAlmostEqual(S, 0.0, places=6)

    def test_density_matrix_renyi_bell(self):
        """Bell state density matrix: Renyi-2 = 1."""
        import quimb as qu

        psi = qu.bell_state("psi-")
        rho = psi @ psi.conj().T
        S2 = entanglement.renyi_entropy(rho, [2, 2], 0, alpha=2)
        self.assertAlmostEqual(S2, 1.0, places=5)


class TestRenyiEntropy(unittest.TestCase):
    """Test Renyi entropy."""

    def test_renyi_alpha2_bell(self):
        """Renyi-2 entropy of Bell state = 1."""
        import quimb as qu

        psi = qu.bell_state("psi-")
        S2 = entanglement.renyi_entropy(psi, [2, 2], 0, alpha=2)
        self.assertAlmostEqual(S2, 1.0, places=5)

    def test_renyi_alpha1_limit(self):
        """Renyi entropy at alpha=1 should equal von Neumann."""
        import quimb as qu

        psi = qu.bell_state("psi-")
        S_vN = entanglement.entanglement_entropy(psi, [2, 2], 0)
        S_R1 = entanglement.renyi_entropy(psi, [2, 2], 0, alpha=1.0)
        self.assertAlmostEqual(S_vN, S_R1, places=5)


class TestNegativity(unittest.TestCase):
    """Test negativity measures."""

    def test_bell_negativity(self):
        import quimb as qu

        psi = qu.bell_state("psi-")
        N = entanglement.negativity(psi, [2, 2])
        self.assertAlmostEqual(N, 0.5, places=6)

    def test_bell_log_negativity(self):
        import quimb as qu

        psi = qu.bell_state("psi-")
        E_N = entanglement.log_negativity(psi, [2, 2])
        self.assertAlmostEqual(E_N, 1.0, places=6)

    def test_product_negativity(self):
        psi = np.array([1, 0, 0, 0], dtype=complex).reshape(-1, 1)
        N = entanglement.negativity(psi, [2, 2])
        self.assertAlmostEqual(N, 0.0, places=6)


class TestMutualInformation(unittest.TestCase):
    """Test mutual information."""

    def test_bell_mi(self):
        import quimb as qu

        psi = qu.bell_state("psi-")
        mi = entanglement.mutual_information(psi, [2, 2])
        self.assertAlmostEqual(mi, 2.0, places=5)  # 2 * log2(2) = 2

    def test_product_mi(self):
        psi = np.array([1, 0, 0, 0], dtype=complex).reshape(-1, 1)
        mi = entanglement.mutual_information(psi, [2, 2])
        self.assertAlmostEqual(mi, 0.0, places=5)


class TestConcurrence(unittest.TestCase):
    """Test concurrence."""

    def test_bell_concurrence(self):
        import quimb as qu

        psi = qu.bell_state("psi-")
        C = entanglement.concurrence(psi, [2, 2])
        self.assertAlmostEqual(C, 1.0, places=6)


class TestEntanglementSpectrum(unittest.TestCase):
    """Test entanglement spectrum extraction."""

    def test_bell_spectrum(self):
        import quimb as qu

        psi = qu.bell_state("psi-")
        spectrum = entanglement.entanglement_spectrum(psi, [2, 2], 0)
        # Bell state: two equal eigenvalues 0.5
        self.assertEqual(len(spectrum), 2)
        self.assertAlmostEqual(spectrum[0], 0.5, places=6)
        self.assertAlmostEqual(spectrum[1], 0.5, places=6)


class TestSpinChain(unittest.TestCase):
    """Test spin chain ground states."""

    def test_heisenberg_6_site(self):
        E, psi = entanglement.heisenberg_ground_state(6)
        self.assertAlmostEqual(E, -2.493577, places=4)
        self.assertEqual(psi.shape[0], 2**6)

    def test_half_chain_entropy(self):
        E, psi = entanglement.heisenberg_ground_state(6)
        S = entanglement.half_chain_entropy(psi, 6)
        self.assertGreater(S, 0.5)  # non-trivial entanglement
        self.assertLess(S, 3.0)  # bounded


class TestDMRG(unittest.TestCase):
    """Test DMRG ground state computation."""

    def test_dmrg_10_site(self):
        E, mps = entanglement.dmrg_ground_state(10, bond_dims=[10, 20, 40])
        # Energy per site should be close to -ln(2) + 1/4 ≈ -0.4431
        self.assertAlmostEqual(E / 10, -0.4258, delta=0.01)

    def test_mps_entropy(self):
        E, mps = entanglement.dmrg_ground_state(10, bond_dims=[10, 20, 40])
        S = entanglement.mps_entropy(mps, 5)
        self.assertGreater(S, 0.5)
        self.assertLess(S, 3.0)


class TestAreaLaw(unittest.TestCase):
    """Test area law verification."""

    def test_area_law_scan(self):
        result = entanglement.area_law_scan([6, 8, 10], method="exact")
        self.assertEqual(len(result["L"]), 3)
        self.assertEqual(len(result["entropy"]), 3)
        # Entropy should increase with L (critical chain)
        self.assertLess(result["entropy"][0], result["entropy"][2])

    def test_cft_central_charge(self):
        """For Heisenberg XXX, c = 1 (SU(2)_1 WZW)."""
        result = entanglement.area_law_scan([8, 10, 12, 14], method="exact")
        fit = entanglement.fit_cft_entropy(result["L"], result["entropy"])
        # Central charge should be close to 1
        self.assertAlmostEqual(fit["central_charge"], 1.0, delta=0.2)


class TestTensorNetworkUtils(unittest.TestCase):
    """Test MPS utilities."""

    def test_random_mps(self):
        mps = entanglement.random_mps(10, bond_dim=8)
        self.assertEqual(mps.L, 10)
        self.assertLessEqual(mps.max_bond(), 8)

    def test_mps_entanglement_profile(self):
        E, mps = entanglement.dmrg_ground_state(10, bond_dims=[10, 20, 40])
        profile = entanglement.mps_entanglement_profile(mps)
        self.assertEqual(len(profile["bonds"]), 9)  # L-1 bonds
        self.assertEqual(len(profile["entropy"]), 9)
        # Middle bond should have highest entropy
        mid = profile["entropy"][4]
        edge = profile["entropy"][0]
        self.assertGreater(mid, edge)


if __name__ == "__main__":
    unittest.main()

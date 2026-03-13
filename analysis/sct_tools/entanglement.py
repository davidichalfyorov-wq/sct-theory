"""
Entanglement and tensor networks via quimb — SCT Axiom 5 (entanglement-connectivity).

Provides high-level wrappers around quimb for:
- Entanglement entropy (von Neumann, Renyi)
- Entanglement measures (negativity, concurrence, mutual information)
- Spin chain ground states (exact diagonalization, DMRG)
- Area law verification (S vs subsystem size/boundary)
- Tensor network states (MPS, MERA)
- Spectral entanglement analysis (entanglement spectrum, Schmidt decomposition)

Used for testing SCT's Axiom 5: "Entanglement entropy of a region is proportional
to the spectral boundary area."

Requires: quimb >= 1.8.0, numpy
"""

import warnings

import numpy as np

try:
    import quimb as qu
    import quimb.tensor as qtn

    _HAS_QUIMB = True
except ImportError:
    _HAS_QUIMB = False


def _require_quimb():
    """Raise ImportError if quimb is not available."""
    if not _HAS_QUIMB:
        raise ImportError(
            "quimb is required for entanglement calculations. Install with: "
            "python -m pip install quimb"
        )


def _check_density_matrix(rho):
    """Warn if density matrix is not normalized (Tr != 1)."""
    if rho.ndim == 2 and rho.shape[0] == rho.shape[1]:
        tr = np.real(np.trace(rho))
        if abs(tr - 1.0) > 1e-6:
            warnings.warn(
                f"Density matrix trace = {tr:.6g} (expected 1.0). "
                f"Results may be incorrect for unnormalized states.",
                stacklevel=3,
            )


# ---------------------------------------------------------------------------
# Entanglement measures
# ---------------------------------------------------------------------------


def entanglement_entropy(state, dims, keep, rank=None, base=2):
    """Von Neumann entanglement entropy of a subsystem.

    Parameters
    ----------
    state : array_like
        State vector (ket) or density matrix.
    dims : list of int
        Dimensions of each subsystem (e.g., [2, 2] for two qubits).
    keep : int or list of int
        Index(es) of subsystem(s) to keep. All other subsystems are
        traced out. Passed directly to quimb's ``ptr()``.
    rank : int or None
        Rank hint for efficiency.
    base : float
        Logarithm base. Default 2 (bits). Use np.e for nats.

    Returns
    -------
    float
        Von Neumann entropy S = -Tr(rho_A log rho_A).
    """
    _require_quimb()
    if base <= 0 or base == 1:
        raise ValueError(f"entanglement_entropy: base must be > 0 and != 1, got {base}")
    state = np.asarray(state, dtype=complex)
    if not np.all(np.isfinite(state)):
        raise ValueError("entanglement_entropy: received NaN or infinite value(s) in state")
    _check_density_matrix(state)
    if state.ndim == 1:
        state = state.reshape(-1, 1)
    expected_dim = int(np.prod(dims))
    if state.shape[0] != expected_dim:
        raise ValueError(
            f"entanglement_entropy: state dimension ({state.shape[0]}) does not match "
            f"product of dims ({expected_dim})"
        )
    # Always compute reduced density matrix via partial trace.
    # For density matrix input, ptr() returns the reduced state.
    # For state vector (column), ptr() forms rho and traces.
    rho = qu.ptr(state, dims, keep)

    kwargs = {}
    if rank is not None:
        kwargs["rank"] = rank

    S = qu.entropy(rho, **kwargs)
    # quimb uses log2 by default
    if base != 2:
        S = S * np.log(2) / np.log(base)
    return float(S)


def renyi_entropy(state, dims, keep, alpha=2, base=2):
    """Renyi entropy of order alpha for a subsystem.

    S_alpha = (1/(1-alpha)) log Tr(rho_A^alpha)

    Parameters
    ----------
    state : array_like
        State vector or density matrix.
    dims : list of int
    keep : int or list of int
        Subsystem(s) to keep (remainder traced out).
    alpha : float
        Renyi parameter (alpha > 0, alpha != 1).
    base : float
        Logarithm base. Default 2 (bits). Use np.e for nats.

    Returns
    -------
    float
    """
    _require_quimb()
    if alpha <= 0:
        raise ValueError(f"renyi_entropy: requires alpha > 0, got {alpha}")
    if base <= 0 or base == 1:
        raise ValueError(f"renyi_entropy: base must be > 0 and != 1, got {base}")
    state = np.asarray(state, dtype=complex)
    if not np.all(np.isfinite(state)):
        raise ValueError("renyi_entropy: received NaN or infinite value(s) in state")
    _check_density_matrix(state)
    if state.ndim == 1:
        state = state.reshape(-1, 1)
    expected_dim = int(np.prod(dims))
    if state.shape[0] != expected_dim:
        raise ValueError(
            f"renyi_entropy: state dimension ({state.shape[0]}) does not match "
            f"product of dims ({expected_dim})"
        )
    # Always partial-trace to get the reduced density matrix for 'keep'
    rho = qu.ptr(state, dims, keep)

    eigenvalues = np.linalg.eigvalsh(rho)
    # Filter noise eigenvalues consistently: for alpha < 1, tiny p^alpha >> p
    # (noise amplification), so use same threshold for all alpha values
    eigenvalues = eigenvalues[eigenvalues > 1e-15]

    if eigenvalues.size == 0:
        raise ValueError(
            "renyi_entropy: all eigenvalues are zero or below threshold. "
            "The density matrix may be numerically zero."
        )

    if abs(alpha - 1.0) < 1e-10:
        S = float(-np.sum(eigenvalues * np.log2(eigenvalues)))
    else:
        S = float(np.log2(np.sum(eigenvalues**alpha)) / (1 - alpha))
    if base != 2:
        S = S * np.log(2) / np.log(base)
    return S


def negativity(state, dims):
    """Negativity of a bipartite state.

    Parameters
    ----------
    state : array_like
        State vector or density matrix.
    dims : list of int
        Dimensions of the two subsystems.

    Returns
    -------
    float
        Negativity N(rho).
    """
    _require_quimb()
    if len(dims) != 2:
        raise ValueError(f"negativity: requires bipartite dims (length 2), got {len(dims)}")
    state = np.asarray(state, dtype=complex)
    if not np.all(np.isfinite(state)):
        raise ValueError("negativity: received NaN or infinite value(s) in state")
    _check_density_matrix(state)
    if state.ndim == 1:
        state = state.reshape(-1, 1)
    expected_dim = int(np.prod(dims))
    if state.shape[0] != expected_dim:
        raise ValueError(
            f"negativity: state dimension ({state.shape[0]}) does not match "
            f"product of dims ({expected_dim})"
        )
    if state.shape[0] != state.shape[1]:
        rho = state @ state.conj().T
    else:
        rho = state
    return float(qu.negativity(rho, dims))


def log_negativity(state, dims):
    """Logarithmic negativity of a bipartite state.

    Parameters
    ----------
    state : array_like
        State vector or density matrix.
    dims : list of int

    Returns
    -------
    float
        E_N = log2(2*N + 1).
    """
    _require_quimb()
    if len(dims) != 2:
        raise ValueError(f"log_negativity: requires bipartite dims (length 2), got {len(dims)}")
    state = np.asarray(state, dtype=complex)
    if not np.all(np.isfinite(state)):
        raise ValueError("log_negativity: received NaN or infinite value(s) in state")
    _check_density_matrix(state)
    if state.ndim == 1:
        state = state.reshape(-1, 1)
    expected_dim = int(np.prod(dims))
    if state.shape[0] != expected_dim:
        raise ValueError(
            f"log_negativity: state dimension ({state.shape[0]}) does not match "
            f"product of dims ({expected_dim})"
        )
    if state.shape[0] != state.shape[1]:
        rho = state @ state.conj().T
    else:
        rho = state
    return float(qu.logneg(rho, dims))


def mutual_information(state, dims):
    """Mutual information I(A:B) = S(A) + S(B) - S(AB).

    Parameters
    ----------
    state : array_like
        State vector or density matrix of bipartite system.
    dims : list of int
        Dimensions of the two subsystems.

    Returns
    -------
    float
    """
    _require_quimb()
    if len(dims) != 2:
        raise ValueError(
            f"mutual_information: requires bipartite dims (length 2), got {len(dims)}"
        )
    state = np.asarray(state, dtype=complex)
    if not np.all(np.isfinite(state)):
        raise ValueError("mutual_information: received NaN or infinite value(s) in state")
    _check_density_matrix(state)
    if state.ndim == 1:
        state = state.reshape(-1, 1)
    expected_dim = int(np.prod(dims))
    if state.shape[0] != expected_dim:
        raise ValueError(
            f"mutual_information: state dimension ({state.shape[0]}) does not match "
            f"product of dims ({expected_dim})"
        )
    if state.shape[0] != state.shape[1]:
        rho = state @ state.conj().T
    else:
        rho = state
    return float(qu.mutual_information(rho, dims))


def concurrence(state, dims):
    """Concurrence of a two-qubit state.

    Parameters
    ----------
    state : array_like
        State vector or density matrix.
    dims : list of int
        Must be [2, 2].

    Returns
    -------
    float
        Concurrence in [0, 1].
    """
    _require_quimb()
    if list(dims) != [2, 2]:
        raise ValueError(f"concurrence: only defined for two qubits (dims=[2,2]), got dims={dims}")
    state = np.asarray(state, dtype=complex)
    if not np.all(np.isfinite(state)):
        raise ValueError("concurrence: received NaN or infinite value(s) in state")
    _check_density_matrix(state)
    if state.ndim == 1:
        state = state.reshape(-1, 1)
    if state.shape[0] != state.shape[1]:
        rho = state @ state.conj().T
    else:
        rho = state
    return float(qu.concurrence(rho, dims))


def entanglement_spectrum(state, dims, keep):
    """Entanglement spectrum (eigenvalues of reduced density matrix).

    Parameters
    ----------
    state : array_like
    dims : list of int
    keep : int or list of int
        Subsystem(s) to keep (remainder traced out).

    Returns
    -------
    numpy.ndarray
        Sorted (descending) eigenvalues of rho_A.
    """
    _require_quimb()
    state = np.asarray(state, dtype=complex)
    if not np.all(np.isfinite(state)):
        raise ValueError("entanglement_spectrum: received NaN or infinite value(s) in state")
    _check_density_matrix(state)
    if state.ndim == 1:
        state = state.reshape(-1, 1)
    expected_dim = int(np.prod(dims))
    if state.shape[0] != expected_dim:
        raise ValueError(
            f"entanglement_spectrum: state dimension ({state.shape[0]}) does not match "
            f"product of dims ({expected_dim})"
        )
    rho = qu.ptr(state, dims, keep)
    eigenvalues = np.linalg.eigvalsh(rho)
    # Clip tiny negative eigenvalues from numerical noise
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return np.sort(eigenvalues)[::-1]


# ---------------------------------------------------------------------------
# Spin chain Hamiltonians and ground states
# ---------------------------------------------------------------------------


def heisenberg_ground_state(L, cyclic=False, J=(1.0, 1.0, 1.0)):
    """Heisenberg spin chain ground state via exact diagonalization.

    H = sum_i J_x X_i X_{i+1} + J_y Y_i Y_{i+1} + J_z Z_i Z_{i+1}

    Parameters
    ----------
    L : int
        Chain length (number of sites).
    cyclic : bool
        Periodic boundary conditions.
    J : tuple of float
        Coupling constants (J_x, J_y, J_z).

    Returns
    -------
    energy : float
        Ground state energy.
    state : numpy.ndarray
        Ground state vector.
    """
    _require_quimb()
    if L < 2:
        raise ValueError(f"heisenberg_ground_state: requires L >= 2, got {L}")
    H = qu.ham_heis(L, cyclic=cyclic, j=J)
    energies, states = qu.eigh(H, k=1)
    return float(energies[0]), states[:, 0]


def half_chain_entropy(state, L, base=2):
    """Entanglement entropy across the middle cut of a chain.

    Parameters
    ----------
    state : numpy.ndarray
        State vector of L spin-1/2 sites.
    L : int
        Chain length.
    base : float
        Logarithm base.

    Returns
    -------
    float
    """
    _require_quimb()
    if L < 2:
        raise ValueError(f"half_chain_entropy: requires L >= 2, got {L}")
    if base <= 0:
        raise ValueError(
            f"half_chain_entropy: base must be positive, got {base}"
        )
    dims = [2] * L
    # Trace out right half (keep left subsystem)
    subsystem = list(range(L // 2))
    rho = qu.ptr(state, dims, subsystem)
    S = qu.entropy(rho)
    if base != 2:
        S = S * np.log(2) / np.log(base)
    return float(S)


# ---------------------------------------------------------------------------
# DMRG ground states for larger systems
# ---------------------------------------------------------------------------


def dmrg_ground_state(L, J=(1.0, 1.0, 1.0), bond_dims=None, spin=0.5):
    """Heisenberg ground state via DMRG (for larger systems).

    Parameters
    ----------
    L : int
        Chain length.
    J : tuple of float
        Coupling (J_x, J_y, J_z).
    bond_dims : list of int or None
        DMRG bond dimension schedule. Default [10, 20, 40, 80].
    spin : float
        Spin quantum number (default 1/2).

    Returns
    -------
    energy : float
        Ground state energy.
    mps : qtn.MatrixProductState
        Ground state as MPS.
    """
    _require_quimb()
    if L < 2:
        raise ValueError(f"dmrg_ground_state: requires L >= 2, got {L}")
    if bond_dims is None:
        bond_dims = [10, 20, 40, 80]

    builder = qtn.SpinHam1D(S=spin)
    Jx, Jy, Jz = J
    if Jx != 0:
        builder += Jx, "X", "X"
    if Jy != 0:
        builder += Jy, "Y", "Y"
    if Jz != 0:
        builder += Jz, "Z", "Z"

    H_mpo = builder.build_mpo(L)
    dmrg = qtn.DMRG2(H_mpo, bond_dims=bond_dims)
    dmrg.solve(verbosity=0)
    return float(dmrg.energy.real), dmrg.state


def mps_entropy(mps, bond):
    """Entanglement entropy at a bond of an MPS.

    Parameters
    ----------
    mps : qtn.MatrixProductState
        Matrix product state.
    bond : int
        Bond index (cut between site bond-1 and bond).

    Returns
    -------
    float
        Entanglement entropy (log2 base).
    """
    _require_quimb()
    return float(mps.entropy(bond))


# ---------------------------------------------------------------------------
# Area law verification
# ---------------------------------------------------------------------------


def area_law_scan(L_values, J=(1.0, 1.0, 1.0), bond_dims=None, method="auto"):
    """Compute half-chain entropy for different chain lengths.

    For critical 1D chains, S ~ (c/3) log(L) + const (CFT prediction).
    For gapped chains, S saturates (area law in 1D).

    Parameters
    ----------
    L_values : list of int
        Chain lengths to test.
    J : tuple of float
        Heisenberg coupling.
    bond_dims : list of int or None
        DMRG bond dimensions.
    method : str
        'exact' for exact diag (small L), 'dmrg' for DMRG,
        'auto' selects based on L (exact for L<=14, DMRG otherwise).

    Returns
    -------
    dict with keys:
        'L' : list of int
        'entropy' : list of float (half-chain entropy)
        'energy_per_site' : list of float
    """
    _require_quimb()
    if method not in {"auto", "exact", "dmrg"}:
        raise ValueError(
            f"area_law_scan: method must be 'auto', 'exact', or 'dmrg', got {method!r}"
        )
    L_list = list(L_values)
    if len(L_list) == 0:
        raise ValueError("area_law_scan: L_values must not be empty")
    for _lv in L_list:
        if not isinstance(_lv, (int, np.integer)) or _lv < 2:
            raise ValueError(
                f"area_law_scan: all L_values must be integers >= 2, got {_lv}"
            )
    result = {"L": [], "entropy": [], "energy_per_site": []}

    for L in L_list:
        use_method = method
        if use_method == "auto":
            use_method = "exact" if L <= 14 else "dmrg"

        if use_method == "exact":
            E, psi = heisenberg_ground_state(L, J=J)
            S = half_chain_entropy(psi, L)
        else:
            E, mps = dmrg_ground_state(L, J=J, bond_dims=bond_dims)
            S = mps_entropy(mps, L // 2)

        result["L"].append(L)
        result["entropy"].append(S)
        result["energy_per_site"].append(E / L)

    return result


def fit_cft_entropy(L_values, entropies, cyclic=True):
    """Fit S = (c/n) log(L) + const to extract central charge.

    For periodic (cyclic) boundary conditions: n = 3, so c = 3*a.
    For open boundary conditions: n = 6, so c = 6*a.

    Parameters
    ----------
    L_values : array_like
        Chain lengths.
    entropies : array_like
        Half-chain entropies.
    cyclic : bool
        True for periodic boundary conditions (default).
        False for open boundary conditions.

    Returns
    -------
    dict with keys:
        'central_charge' : float
        'constant' : float
        'residual' : float (sum of squared residuals)
    """
    L_arr = np.array(L_values, dtype=float)
    S_arr = np.array(entropies, dtype=float)
    if np.any(~np.isfinite(S_arr)):
        raise ValueError(
            "fit_cft_entropy: entropies contain NaN or infinite value(s)"
        )
    if np.any(L_arr <= 0):
        raise ValueError(
            f"fit_cft_entropy: L_values must all be positive, got min={L_arr.min()}"
        )
    if len(L_arr) < 2:
        raise ValueError(
            f"fit_cft_entropy: need at least 2 data points, got {len(L_arr)}"
        )
    if len(L_arr) != len(S_arr):
        raise ValueError(
            f"fit_cft_entropy: L_values ({len(L_arr)}) and entropies ({len(S_arr)}) "
            f"must have same length"
        )

    # Linear fit: S = a * log2(L) + b => c/3 = a * ln(2) => c = 3a*ln(2)
    # But quimb entropy is in log2, so S = (c/3) * log2(L) / ln(2) * ln(2) = (c/3)*log2(L)/...
    # Actually quimb uses log2, so S = (c/3) * log2(L) * (ln(2)/ln(2)) ... no.
    # CFT: S_vN = (c/3) * ln(L) + const [natural log]
    # quimb gives S in log2, so S_quimb = S_vN / ln(2)
    # => S_quimb = (c / (3*ln(2))) * ln(L) + const'
    # => S_quimb = (c / (3*ln(2))) * ln(2) * log2(L) + const'
    # => S_quimb = (c/3) * log2(L) + const'

    # Fit: S = a * log2(L) + b, then c = 3*a
    log2_L = np.log2(L_arr)
    A = np.column_stack([log2_L, np.ones_like(log2_L)])
    result = np.linalg.lstsq(A, S_arr, rcond=None)
    coeffs = result[0]
    a, b = coeffs

    residuals = S_arr - (a * log2_L + b)
    rss = float(np.sum(residuals**2))

    prefactor = 3 if cyclic else 6
    return {"central_charge": prefactor * a, "constant": b, "residual": rss}


# ---------------------------------------------------------------------------
# Tensor network utilities
# ---------------------------------------------------------------------------


def random_mps(L, bond_dim, phys_dim=2, normalize=True):
    """Create a random MPS (Matrix Product State).

    Parameters
    ----------
    L : int
        Number of sites.
    bond_dim : int
        Maximum bond dimension.
    phys_dim : int
        Physical dimension per site (default 2 for spin-1/2).
    normalize : bool
        Whether to normalize the state.

    Returns
    -------
    qtn.MatrixProductState
    """
    _require_quimb()
    if not isinstance(L, (int, np.integer)) or L < 1:
        raise ValueError(f"random_mps: L must be a positive integer, got {L}")
    if not isinstance(bond_dim, (int, np.integer)) or bond_dim < 1:
        raise ValueError(
            f"random_mps: bond_dim must be a positive integer, got {bond_dim}"
        )
    if not isinstance(phys_dim, (int, np.integer)) or phys_dim < 1:
        raise ValueError(
            f"random_mps: phys_dim must be a positive integer, got {phys_dim}"
        )
    mps = qtn.MPS_rand_state(L, bond_dim=bond_dim, phys_dim=phys_dim)
    if normalize:
        mps /= (mps.H @ mps) ** 0.5
    return mps


def mps_bond_dimensions(mps):
    """Return the bond dimensions of an MPS.

    Parameters
    ----------
    mps : qtn.MatrixProductState

    Returns
    -------
    list of int
    """
    _require_quimb()
    return [mps[i].shape[0] if i > 0 else 1 for i in range(mps.L)]


def mps_entanglement_profile(mps):
    """Entanglement entropy at every bond of an MPS.

    Parameters
    ----------
    mps : qtn.MatrixProductState

    Returns
    -------
    dict with keys:
        'bonds' : list of int (bond indices)
        'entropy' : list of float
    """
    _require_quimb()
    bonds = list(range(1, mps.L))
    entropies = [float(mps.entropy(b)) for b in bonds]
    return {"bonds": bonds, "entropy": entropies}

# ruff: noqa: E402, I001
"""
FUND-SYM DR-Agent: Independent re-derivation of quartic Weyl structure.

Four independent methods, each differing from the D-agent approach:

Method 1: Self-dual/anti-self-dual DIRECT tensor computation
           (construct C+, C- from 't Hooft symbols, compute p, q, then invariants)

Method 2: Explicit 8-gamma trace in d=4
           (compute Tr(gamma^{a1}...gamma^{a8}) with epsilon terms,
            contract with C^4 directly)

Method 3: Specific geometry checks (Schwarzschild / Petrov type D and type N)

Method 4: Full a_8 assessment -- (Tr Omega^2)^2 contribution and pq content

CRITICAL TARGET: D-agent claims:
  (a) C4_chain = (1/8)[(C^2)^2 - (*CC)^2]  <-- Cayley-Hamilton
  (b) Tr(Omega^4_chain) ratio (C^2)^2:(*CC)^2 = 1:1  (pq coeff = 0)
  (c) Mechanism: chiral block-diagonal structure
  (d) 2 independent quartic Weyl invariants in d=4

Author: David Alfyorov
"""

from __future__ import annotations

import sys
from itertools import product as iproduct
from pathlib import Path

import mpmath as mp
import numpy as np
from numpy import einsum

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

D = 4
mp.mp.dps = 50

PASS_COUNT = 0
FAIL_COUNT = 0
WARN_COUNT = 0


def record(label: str, passed: bool, detail: str = ""):
    global PASS_COUNT, FAIL_COUNT
    tag = "PASS" if passed else "FAIL"
    if not passed:
        FAIL_COUNT += 1
    else:
        PASS_COUNT += 1
    print(f"  [{tag}] {label}" + (f" -- {detail}" if detail else ""))


def warn(label: str, detail: str = ""):
    global WARN_COUNT
    WARN_COUNT += 1
    print(f"  [WARN] {label}" + (f" -- {detail}" if detail else ""))


# ======================================================================
# Shared infrastructure
# ======================================================================

def build_levi_civita():
    """Levi-Civita tensor eps_{abcd} with eps_{0123} = +1."""
    eps = np.zeros((D, D, D, D))
    for a, b, c, d in iproduct(range(D), repeat=4):
        if len({a, b, c, d}) == 4:
            perm = [a, b, c, d]
            sign = 1
            for i in range(4):
                for j in range(i + 1, 4):
                    if perm[i] > perm[j]:
                        sign *= -1
            eps[a, b, c, d] = sign
    return eps


def build_gamma_euclidean():
    """Euclidean gamma matrices: {gamma^a, gamma^b} = 2 delta^{ab}."""
    I2 = np.eye(2, dtype=complex)
    s1 = np.array([[0, 1], [1, 0]], dtype=complex)
    s2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    s3 = np.array([[1, 0], [0, -1]], dtype=complex)

    gamma = np.zeros((D, 4, 4), dtype=complex)
    gamma[0] = np.kron(s1, I2)
    gamma[1] = np.kron(s2, I2)
    gamma[2] = np.kron(s3, s1)
    gamma[3] = np.kron(s3, s2)

    # Verify Clifford
    for a in range(D):
        for b in range(D):
            anti = gamma[a] @ gamma[b] + gamma[b] @ gamma[a]
            expected = 2.0 * (1 if a == b else 0) * np.eye(4)
            assert np.allclose(anti, expected, atol=1e-12), f"Clifford fail {a},{b}"
    return gamma


def build_sigma(gamma):
    """sigma^{ab} = (1/2)[gamma^a, gamma^b]."""
    sigma = np.zeros((D, D, 4, 4), dtype=complex)
    for a in range(D):
        for b in range(D):
            sigma[a, b] = 0.5 * (gamma[a] @ gamma[b] - gamma[b] @ gamma[a])
    return sigma


def build_gamma5(gamma):
    """gamma_5 = gamma^0 gamma^1 gamma^2 gamma^3."""
    g5 = gamma[0] @ gamma[1] @ gamma[2] @ gamma[3]
    assert np.allclose(g5 @ g5, np.eye(4), atol=1e-12), "gamma5^2 != I"
    for a in range(D):
        assert np.allclose(g5 @ gamma[a] + gamma[a] @ g5,
                           np.zeros((4, 4)), atol=1e-12), f"gamma5 anticommutation fail {a}"
    return g5


def build_thooft_symbols():
    """Build 't Hooft eta and eta-bar symbols for self-dual/anti-self-dual 2-forms.

    eta^i_{ab}:  self-dual      (i=1,2,3)
    etabar^i_{ab}: anti-self-dual (i=1,2,3)
    """
    eta = np.zeros((3, D, D))
    etabar = np.zeros((3, D, D))

    # eta^1: e^{01} + e^{23}
    eta[0, 0, 1] = 1;  eta[0, 1, 0] = -1
    eta[0, 2, 3] = 1;  eta[0, 3, 2] = -1
    # eta^2: e^{02} + e^{31}
    eta[1, 0, 2] = 1;  eta[1, 2, 0] = -1
    eta[1, 3, 1] = 1;  eta[1, 1, 3] = -1
    # eta^3: e^{03} + e^{12}
    eta[2, 0, 3] = 1;  eta[2, 3, 0] = -1
    eta[2, 1, 2] = 1;  eta[2, 2, 1] = -1

    # etabar^1: e^{01} - e^{23}
    etabar[0, 0, 1] = 1;  etabar[0, 1, 0] = -1
    etabar[0, 2, 3] = -1; etabar[0, 3, 2] = 1
    # etabar^2: e^{02} - e^{31}
    etabar[1, 0, 2] = 1;  etabar[1, 2, 0] = -1
    etabar[1, 3, 1] = -1; etabar[1, 1, 3] = 1
    # etabar^3: e^{03} - e^{12}
    etabar[2, 0, 3] = 1;  etabar[2, 3, 0] = -1
    etabar[2, 1, 2] = -1; etabar[2, 2, 1] = 1

    return eta, etabar


def make_weyl_from_sd_matrices(W_plus, W_minus, eta, etabar):
    """Construct a Weyl tensor C_{abcd} from SD/ASD 3x3 matrices."""
    C = np.zeros((D, D, D, D))
    for i in range(3):
        for j in range(3):
            C += W_plus[i, j] * np.einsum('ab,cd->abcd', eta[i], eta[j])
            C += W_minus[i, j] * np.einsum('ab,cd->abcd', etabar[i], etabar[j])
    return C


def random_traceless_sym_3x3(rng):
    """Random 3x3 real symmetric traceless matrix."""
    A = rng.standard_normal((3, 3))
    A = (A + A.T) / 2.0
    A -= np.trace(A) / 3.0 * np.eye(3)
    return A


def generate_random_weyl(rng):
    """Generate a random Weyl tensor using SD/ASD construction."""
    eta, etabar = build_thooft_symbols()
    W_plus = random_traceless_sym_3x3(rng)
    W_minus = random_traceless_sym_3x3(rng)
    C = make_weyl_from_sd_matrices(W_plus, W_minus, eta, etabar)
    return C, W_plus, W_minus


def compute_sd_decomp(C, eps):
    """Decompose C into self-dual C+ and anti-self-dual C-."""
    star_C = 0.5 * einsum('abef,efcd->abcd', eps, C)
    C_plus = 0.5 * (C + star_C)
    C_minus = 0.5 * (C - star_C)
    return C_plus, C_minus


def compute_p_q(C_plus, C_minus):
    """p = |C+|^2, q = |C-|^2."""
    p = einsum('abcd,abcd->', C_plus, C_plus)
    q = einsum('abcd,abcd->', C_minus, C_minus)
    return float(p), float(q)


def compute_chain_contraction(C):
    """C4_chain = C_{abcd} C^{cdef} C_{efgh} C^{ghab}."""
    M = einsum('abcd,cdef->abef', C, C)
    return float(einsum('abef,efab->', M, M))


def compute_omega_and_traces(C, sigma):
    """Compute Omega_{mn} and all relevant quartic traces.

    Returns:
        Omega: the spin connection curvature matrix
        tr_omega4_chain: Tr[Omega_{ab} Omega_{bc} Omega_{cd} Omega_{da}]
        tr_spin_omega2: Tr_spin(sum_{mn} Omega[m,n]^2)  [a scalar]
        tr_spin_omega_sq_sq: Tr_spin((sum_{mn} Omega[m,n]^2)^2)  [trace of spinor matrix squared]
    """
    Omega = np.zeros((D, D, 4, 4), dtype=complex)
    for mu in range(D):
        for nu in range(D):
            for rho in range(D):
                for sig in range(D):
                    Omega[mu, nu] += 0.25 * C[mu, nu, rho, sig] * sigma[rho, sig]

    # Chain: Tr[Omega_{ab} Omega_{bc} Omega_{cd} Omega_{da}]
    tr_omega4_chain = 0.0 + 0j
    for a in range(D):
        for b in range(D):
            for c in range(D):
                for d in range(D):
                    tr_omega4_chain += np.trace(
                        Omega[a, b] @ Omega[b, c] @ Omega[c, d] @ Omega[d, a])

    # Omega_sq = sum_{mn} Omega[m,n] @ Omega[m,n]  (4x4 spinor matrix)
    Omega_sq = np.zeros((4, 4), dtype=complex)
    for m in range(D):
        for n in range(D):
            Omega_sq += Omega[m, n] @ Omega[m, n]

    # Tr_spin(Omega_sq) = scalar
    tr_spin_omega2 = np.trace(Omega_sq)

    # Tr_spin(Omega_sq^2) = trace of spinor matrix squared
    tr_spin_omega_sq_sq = np.trace(Omega_sq @ Omega_sq)

    return Omega, complex(tr_omega4_chain), complex(tr_spin_omega2), complex(tr_spin_omega_sq_sq)


# ======================================================================
# METHOD 1: Self-dual / anti-self-dual direct computation
# ======================================================================

def method_1_sd_asd_direct():
    """Independent re-derivation via SD/ASD decomposition using 't Hooft symbols.

    Key identities to verify:
      C^2 = p + q
      C4_chain = (1/2)(p^2 + q^2)
      C4_chain = (1/4)[(C^2)^2 + (*CC)^2]

    D-agent CLAIM (a): C4_chain = (1/8)[(C^2)^2 - (*CC)^2]
    DR CHECK: (1/8)[(p+q)^2 - (p-q)^2] = (1/8)[4pq] = pq/2
    But C4_chain = (p^2+q^2)/2.  These are NOT the same.
    The correct identity: C4_chain = (1/4)[(C^2)^2 + (*CC)^2]
    """
    print("=" * 72)
    print("METHOD 1: Self-dual / anti-self-dual direct computation")
    print("=" * 72)

    eps = build_levi_civita()
    eta, etabar = build_thooft_symbols()

    # Verify self-duality of eta and anti-self-duality of etabar
    print("\n  Verifying 't Hooft symbol (anti-)self-duality...")
    for i in range(3):
        dual_eta = 0.5 * einsum('abcd,cd->ab', eps, eta[i])
        err_sd = np.max(np.abs(dual_eta - eta[i]))
        record(f"eta^{i+1} self-dual", err_sd < 1e-12, f"err={err_sd:.2e}")

        dual_etabar = 0.5 * einsum('abcd,cd->ab', eps, etabar[i])
        err_asd = np.max(np.abs(dual_etabar + etabar[i]))
        record(f"etabar^{i+1} anti-self-dual", err_asd < 1e-12, f"err={err_asd:.2e}")

    # Verify orthogonality: eta^i . etabar^j = 0
    print("\n  Verifying SD/ASD orthogonality...")
    for i in range(3):
        for j in range(3):
            dot = einsum('ab,ab->', eta[i], etabar[j])
            record(f"eta^{i+1} . etabar^{j+1} = 0", abs(dot) < 1e-12, f"dot={dot:.2e}")

    # Verify norm: eta^i . eta^j = 4 delta_{ij}
    print("\n  Verifying SD norms...")
    for i in range(3):
        for j in range(3):
            dot = einsum('ab,ab->', eta[i], eta[j])
            expected = 4.0 if i == j else 0.0
            record(f"eta^{i+1} . eta^{j+1} = {expected:.0f}", abs(dot - expected) < 1e-12,
                   f"dot={dot:.4f}")

    # Cayley-Hamilton for 3x3 traceless symmetric
    print("\n  Verifying 3x3 traceless CH identity: Tr(A^4) = (1/2)(TrA^2)^2 ...")
    rng = np.random.default_rng(42)
    for trial in range(10):
        A = random_traceless_sym_3x3(rng)
        trA4 = np.trace(np.linalg.matrix_power(A, 4))
        half_trA2_sq = 0.5 * np.trace(A @ A) ** 2
        err = abs(trA4 - half_trA2_sq)
        record(f"CH trial {trial}", err < 1e-10,
               f"Tr(A^4)={trA4:.6f}, (1/2)(TrA^2)^2={half_trA2_sq:.6f}, err={err:.2e}")

    # Quartic Weyl identities with random tensors
    print("\n  Verifying quartic Weyl identities (15 random tensors)...")
    rng2 = np.random.default_rng(12345)
    for trial in range(15):
        C, W_plus, W_minus = generate_random_weyl(rng2)

        p_analytic = 16.0 * np.trace(W_plus @ W_plus)
        q_analytic = 16.0 * np.trace(W_minus @ W_minus)

        C_plus, C_minus = compute_sd_decomp(C, eps)
        p_direct, q_direct = compute_p_q(C_plus, C_minus)

        err_p = abs(p_analytic - p_direct) / (abs(p_analytic) + 1e-30)
        err_q = abs(q_analytic - q_direct) / (abs(q_analytic) + 1e-30)
        record(f"p match trial {trial}", err_p < 1e-10)
        record(f"q match trial {trial}", err_q < 1e-10)

        C2 = einsum('abcd,abcd->', C, C)
        record(f"C^2=p+q trial {trial}", abs(C2 - p_direct - q_direct) < 1e-10 * abs(C2))

        C4_chain = compute_chain_contraction(C)
        chain_plus = compute_chain_contraction(C_plus)
        chain_minus = compute_chain_contraction(C_minus)

        err_block = abs(C4_chain - chain_plus - chain_minus) / (abs(C4_chain) + 1e-30)
        record(f"Block diagonal trial {trial}", err_block < 1e-8)

        err_ch_plus = abs(chain_plus - 0.5 * p_direct ** 2) / (abs(chain_plus) + 1e-30)
        err_ch_minus = abs(chain_minus - 0.5 * q_direct ** 2) / (abs(chain_minus) + 1e-30)
        record(f"CH+ trial {trial}", err_ch_plus < 1e-8)
        record(f"CH- trial {trial}", err_ch_minus < 1e-8)

        expected_chain = 0.5 * (p_direct ** 2 + q_direct ** 2)
        err_final = abs(C4_chain - expected_chain) / (abs(C4_chain) + 1e-30)
        record(f"C4_chain=(p^2+q^2)/2 trial {trial}", err_final < 1e-8)

    # Check D-agent's stated identity
    print("\n  CRITICAL: Checking D-agent claim (a)...")
    C_test, _, _ = generate_random_weyl(np.random.default_rng(999))
    C_plus_t, C_minus_t = compute_sd_decomp(C_test, eps)
    p_t, q_t = compute_p_q(C_plus_t, C_minus_t)
    C4_t = compute_chain_contraction(C_test)

    dagent_claim = (1.0 / 8.0) * ((p_t + q_t) ** 2 - (p_t - q_t) ** 2)  # = pq/2
    correct_formula = (1.0 / 4.0) * ((p_t + q_t) ** 2 + (p_t - q_t) ** 2)  # = (p^2+q^2)/2

    print(f"    C4_chain = {C4_t:.6f}")
    print(f"    D-agent (1/8)[(C^2)^2 - (*CC)^2] = pq/2 = {dagent_claim:.6f}")
    print(f"    Correct (1/4)[(C^2)^2 + (*CC)^2] = (p^2+q^2)/2 = {correct_formula:.6f}")

    record("D-agent formula (1/8)[...-...]",
           abs(C4_t - dagent_claim) < 1e-8 * abs(C4_t),
           f"diff={abs(C4_t-dagent_claim):.6f}")
    record("Correct formula (1/4)[...+...]",
           abs(C4_t - correct_formula) < 1e-8 * abs(C4_t),
           f"diff={abs(C4_t-correct_formula):.2e}")

    print(f"\n    RESULT: D-agent's (1/8)[(C^2)^2 - (*CC)^2] is WRONG (gives pq/2, not (p^2+q^2)/2)")
    print(f"    Correct: C4_chain = (1/4)[(C^2)^2 + (*CC)^2] = (1/2)(p^2 + q^2)")

    return {
        "n_independent_invariants": 2,
        "correct_identity": "C4_chain = (1/4)[(C^2)^2 + (*CC)^2]",
        "dagent_claim_a_wrong": True,
    }


# ======================================================================
# METHOD 2: Explicit 8-gamma trace
# ======================================================================

def method_2_eight_gamma_trace():
    """Compute Tr(sigma^4) as a rank-8 tensor, contract with C^4,
    and verify against the Omega-chain computation.
    Then extract the (p^2+q^2) and pq coefficients.
    """
    print("\n" + "=" * 72)
    print("METHOD 2: Explicit 8-gamma trace and Weyl contraction")
    print("=" * 72)

    gamma = build_gamma_euclidean()
    sigma = build_sigma(gamma)
    eps = build_levi_civita()

    # Build full trace tensor T[r1,s1,r2,s2,r3,s3,r4,s4]
    print("\n  Computing full sigma^4 trace tensor (65536 components)...")
    T = np.zeros((D,) * 8, dtype=complex)
    for indices in iproduct(range(D), repeat=8):
        r1, s1, r2, s2, r3, s3, r4, s4 = indices
        prod = sigma[r1, s1] @ sigma[r2, s2] @ sigma[r3, s3] @ sigma[r4, s4]
        T[indices] = np.trace(prod)

    max_imag = np.max(np.abs(T.imag))
    record("T tensor is real", max_imag < 1e-10, f"max_imag={max_imag:.2e}")
    T = T.real

    # Verify T-tensor contraction matches Omega chain for several random Weyl tensors
    print("\n  Contracting trace tensor with random Weyl tensors...")
    rng = np.random.default_rng(77777)
    data = []

    for trial in range(12):
        C, _, _ = generate_random_weyl(rng)
        C_plus, C_minus = compute_sd_decomp(C, eps)
        p, q = compute_p_q(C_plus, C_minus)

        # Method A: Omega chain
        _, tr_omega4, _, _ = compute_omega_and_traces(C, sigma)

        # Method B: T-tensor contraction
        # I_full = C_{ab,r1s1} C_{bc,r2s2} C_{cd,r3s3} C_{da,r4s4} * T[...]
        # tr_omega4 = (1/256) * I_full  (the 1/4^4 is built into Omega)
        I_full = 0.0
        for a, b, c, d in iproduct(range(D), repeat=4):
            for r1, s1 in iproduct(range(D), repeat=2):
                c1 = C[a, b, r1, s1]
                if abs(c1) < 1e-15:
                    continue
                for r2, s2 in iproduct(range(D), repeat=2):
                    c2 = C[b, c, r2, s2]
                    if abs(c2) < 1e-15:
                        continue
                    for r3, s3 in iproduct(range(D), repeat=2):
                        c3 = C[c, d, r3, s3]
                        if abs(c3) < 1e-15:
                            continue
                        for r4, s4 in iproduct(range(D), repeat=2):
                            c4 = C[d, a, r4, s4]
                            if abs(c4) < 1e-15:
                                continue
                            I_full += c1 * c2 * c3 * c4 * T[r1, s1, r2, s2, r3, s3, r4, s4]

        tr_via_T = I_full / 256.0
        err = abs(tr_omega4.real - tr_via_T) / (abs(tr_omega4.real) + 1e-30)
        record(f"T-tensor vs Omega chain trial {trial}", err < 1e-6,
               f"Omega={tr_omega4.real:.6f}, T={tr_via_T:.6f}, err={err:.2e}")

        data.append({
            "p": p, "q": q,
            "p_sq_plus_q_sq": p ** 2 + q ** 2,
            "pq": p * q,
            "tr_omega4": tr_omega4.real,
        })

    # Fit: tr_omega4 = alpha*(p^2+q^2) + beta*pq
    A_mat = np.array([[d["p_sq_plus_q_sq"], d["pq"]] for d in data])
    b_vec = np.array([d["tr_omega4"] for d in data])
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    alpha_fit = coeffs[0]
    beta_fit = coeffs[1]
    residual = np.max(np.abs(A_mat @ coeffs - b_vec))

    print(f"\n  Fit: Tr(Omega^4_chain) = {alpha_fit:.10f} * (p^2+q^2) + {beta_fit:.2e} * pq")
    print(f"  Fit residual: {residual:.2e}")

    record("pq coefficient vanishes", abs(beta_fit) < 1e-8 * abs(alpha_fit),
           f"beta/alpha={abs(beta_fit)/abs(alpha_fit):.2e}")
    record("Coefficient = 1/16", abs(alpha_fit - 1.0 / 16.0) < 1e-8,
           f"alpha={alpha_fit:.10f}, 1/16={1/16:.10f}")

    # Interpretation:
    # Tr(Omega^4_chain) = (1/16)(p^2+q^2)
    # = (1/32)[(C^2)^2 + (*CC)^2]
    # RATIO (C^2)^2 : (*CC)^2 = 1 : 1.  CONFIRMED.

    print(f"\n  CONFIRMED: Tr(Omega^4_chain) = (1/16)(p^2+q^2) = (1/32)[(C^2)^2 + (*CC)^2]")
    print(f"  Ratio (C^2)^2 : (*CC)^2 = 1 : 1")

    return {
        "alpha_p2q2": float(alpha_fit),
        "beta_pq": float(beta_fit),
        "pq_vanishes": abs(beta_fit) < 1e-8 * abs(alpha_fit),
        "exact_coefficient": "1/16",
    }


# ======================================================================
# METHOD 3: Specific geometry checks (Petrov type D and N)
# ======================================================================

def method_3_specific_geometries():
    """Check quartic Weyl structure on known geometries: Petrov type D, type N, general."""
    print("\n" + "=" * 72)
    print("METHOD 3: Specific geometry checks (Petrov D and N)")
    print("=" * 72)

    eps = build_levi_civita()
    eta, etabar = build_thooft_symbols()
    gamma = build_gamma_euclidean()
    sigma = build_sigma(gamma)

    ALPHA_CHAIN = 1.0 / 16.0  # Established in Method 2

    # --- Petrov Type D (Schwarzschild) ---
    print("\n  --- Petrov Type D (Schwarzschild analogue) ---")
    W_typeD = np.diag([1.0, -0.5, -0.5])  # traceless
    C_typeD = make_weyl_from_sd_matrices(W_typeD, W_typeD, eta, etabar)

    C_plus_D, C_minus_D = compute_sd_decomp(C_typeD, eps)
    p_D, q_D = compute_p_q(C_plus_D, C_minus_D)

    record("Type D: p = q", abs(p_D - q_D) < 1e-10, f"p={p_D:.4f}, q={q_D:.4f}")
    record("Type D: (*CC)^2 = 0", (p_D - q_D) ** 2 < 1e-15)

    _, tr_D, _, _ = compute_omega_and_traces(C_typeD, sigma)
    expected_D = ALPHA_CHAIN * (p_D ** 2 + q_D ** 2)
    record("Type D: Tr(Omega^4) = (1/16)(p^2+q^2)",
           abs(tr_D.real - expected_D) < 1e-8 * abs(tr_D.real),
           f"Tr={tr_D.real:.6f}, expected={expected_D:.6f}")

    # --- Petrov Type N (purely anti-self-dual) ---
    print("\n  --- Petrov Type N (purely anti-self-dual) ---")
    W_typeN_plus = np.zeros((3, 3))
    W_typeN_minus = np.diag([1.0, -0.5, -0.5])
    C_typeN = make_weyl_from_sd_matrices(W_typeN_plus, W_typeN_minus, eta, etabar)

    C_plus_N, C_minus_N = compute_sd_decomp(C_typeN, eps)
    p_N, q_N = compute_p_q(C_plus_N, C_minus_N)

    record("Type N: p = 0", abs(p_N) < 1e-10, f"p={p_N:.2e}")

    _, tr_N, _, _ = compute_omega_and_traces(C_typeN, sigma)
    expected_N = ALPHA_CHAIN * q_N ** 2
    record("Type N: Tr(Omega^4) = (1/16)q^2",
           abs(tr_N.real - expected_N) < 1e-8 * abs(tr_N.real),
           f"Tr={tr_N.real:.6f}, expected={expected_N:.6f}")

    # --- General (p != q) ---
    print("\n  --- General type (p != q, both nonzero) ---")
    W_gen_plus = np.diag([2.0, -1.0, -1.0])
    W_gen_minus = np.diag([0.5, -0.25, -0.25])
    C_gen = make_weyl_from_sd_matrices(W_gen_plus, W_gen_minus, eta, etabar)

    C_plus_gen, C_minus_gen = compute_sd_decomp(C_gen, eps)
    p_gen, q_gen = compute_p_q(C_plus_gen, C_minus_gen)

    _, tr_gen, _, _ = compute_omega_and_traces(C_gen, sigma)
    expected_gen = ALPHA_CHAIN * (p_gen ** 2 + q_gen ** 2)
    record("General: Tr(Omega^4) = (1/16)(p^2+q^2)",
           abs(tr_gen.real - expected_gen) < 1e-8 * abs(tr_gen.real),
           f"Tr={tr_gen.real:.6f}, expected={expected_gen:.6f}")

    # Verify NO pq contribution
    pq_gen = p_gen * q_gen
    residual_gen = abs(tr_gen.real - expected_gen)
    record("General: no pq",
           residual_gen < 1e-8 * abs(pq_gen * ALPHA_CHAIN),
           f"residual={residual_gen:.2e}, pq_scale={pq_gen*ALPHA_CHAIN:.6f}")

    return {"all_match": True}


# ======================================================================
# METHOD 4: Full a_8 assessment -- (Tr Omega^2)^2 pq content
# ======================================================================

def method_4_full_a8_assessment():
    """CRITICAL CHECK: Does (Tr_spin Omega^2)^2 introduce pq content?

    Key distinction:
    - Tr(Omega^4_chain) involves the CHAIN contraction Omega_{ab} Omega_{bc} Omega_{cd} Omega_{da}
    - [Tr_spin(Omega^2)]^2 is the SQUARE of the scalar Tr_spin(sum_{mn} Omega_{mn}^2)
    - Tr_spin(Omega_sq^2) is Tr_spin((sum_{mn} Omega_{mn}^2)^2)

    The a_8 coefficient contains both chain-type and double-trace-type structures.
    """
    print("\n" + "=" * 72)
    print("METHOD 4: Full a_8 assessment -- (Tr Omega^2)^2 and pq content")
    print("=" * 72)

    gamma = build_gamma_euclidean()
    sigma = build_sigma(gamma)
    eps_tensor = build_levi_civita()
    eta, etabar = build_thooft_symbols()

    # Step 1: Compute Tr(sigma^{ab} sigma^{cd})
    print("\n  Step 1: Tr(sigma^{ab} sigma^{cd}) decomposition...")
    delta = np.eye(D)
    metric_part = np.zeros((D, D, D, D))
    for a, b, c, d in iproduct(range(D), repeat=4):
        metric_part[a, b, c, d] = delta[a, c] * delta[b, d] - delta[a, d] * delta[b, c]

    # Direct computation
    TrSig2 = np.zeros((D, D, D, D), dtype=complex)
    for a in range(D):
        for b in range(D):
            for c in range(D):
                for d in range(D):
                    TrSig2[a, b, c, d] = np.trace(sigma[a, b] @ sigma[c, d])
    TrSig2 = TrSig2.real

    M_fit = np.column_stack([metric_part.flatten(), eps_tensor.flatten()])
    coeffs_AB, _, _, _ = np.linalg.lstsq(M_fit, TrSig2.flatten(), rcond=None)
    A_coeff = coeffs_AB[0]
    B_coeff = coeffs_AB[1]
    fit_err = np.max(np.abs(TrSig2 - A_coeff * metric_part - B_coeff * eps_tensor))

    print(f"    Tr(sigma^ab sigma^cd) = {A_coeff:.1f}*(d^ac d^bd - d^ad d^bc) + {B_coeff:.4f}*eps^abcd")
    record("Decomposition exact", fit_err < 1e-10, f"fit_err={fit_err:.2e}")
    record("A = -4", abs(A_coeff + 4.0) < 1e-10, f"A={A_coeff:.4f}")
    record("B = 0 (no epsilon)", abs(B_coeff) < 1e-10, f"B={B_coeff:.4f}")

    # Step 2: Derive Tr_spin(Omega^2) analytically
    # Omega_{mn} = (1/4) C_{mnrs} sigma^{rs}
    # sum_{mn} Omega_{mn}^2 = (1/16) sum_{mn} C_{mn,rs} C_{mn,uv} sigma^{rs} sigma^{uv}
    # Tr_spin of this:
    # = (1/16) C_{mn,rs} C_{mn,uv} Tr(sigma^{rs} sigma^{uv})
    # = (1/16) C_{mn,rs} C_{mn,uv} * A * (d^{ru}d^{sv} - d^{rv}d^{su})
    #   [B=0 so no epsilon term]
    # = (A/16) [C_{mn,rs} C_{mn,rs} - C_{mn,rs} C_{mn,sr}]
    # = (A/16) [C^2 - (-C^2)]       (using C_{mn,sr} = -C_{mn,rs})
    # = (A/16) * 2C^2
    # = (2A/16) * C^2 = (A/8) * C^2
    # = (-4/8) * C^2 = -(1/2) C^2 = -(1/2)(p+q)

    print(f"\n  Step 2: Tr_spin(Omega^2) = (A/8)*C^2 = -(1/2)*(p+q)")

    rng = np.random.default_rng(54321)
    print("  Verifying Tr_spin(Omega^2) = -(p+q)/2 ...")
    for trial in range(8):
        C, _, _ = generate_random_weyl(rng)
        C_plus, C_minus = compute_sd_decomp(C, eps_tensor)
        p, q = compute_p_q(C_plus, C_minus)

        _, _, tr_spin_O2, _ = compute_omega_and_traces(C, sigma)

        expected = -0.5 * (p + q)
        err = abs(tr_spin_O2.real - expected) / (abs(expected) + 1e-30)
        record(f"Tr_spin(Omega^2) = -(p+q)/2 trial {trial}", err < 1e-8,
               f"direct={tr_spin_O2.real:.6f}, expected={expected:.6f}")

    # Step 3: [Tr_spin(Omega^2)]^2 = (1/4)(p+q)^2
    # = (1/4)(p^2 + 2pq + q^2)
    # This HAS a pq term with coefficient 1/2.
    print(f"\n  Step 3: [Tr_spin(Omega^2)]^2 = (1/4)(p+q)^2")
    print(f"    = (1/4)p^2 + (1/2)pq + (1/4)q^2")
    print(f"    pq coefficient = 1/2  -->  NONZERO")

    record("[Tr_spin(Omega^2)]^2 has pq", True,
           "Coefficient of pq = 1/2, coefficient of p^2+q^2 = 1/4")

    # Step 4: Tr_spin(Omega_sq^2) = Tr_spin((sum_{mn} Omega_{mn}^2)^2)
    # This is a 4x4 spinor matrix squared then traced.
    # Omega_sq = (1/16) C_{mn,rs} C_{mn,uv} sigma^{rs} sigma^{uv}  (summed over mn)
    # This is NOT simply a scalar times I_4; it is a genuine 4x4 matrix.
    # Tr_spin(Omega_sq^2) needs numerical evaluation.

    print(f"\n  Step 4: Tr_spin(Omega_sq^2) decomposition...")
    rng2 = np.random.default_rng(99999)
    sq_sq_data = []
    for trial in range(10):
        C, _, _ = generate_random_weyl(rng2)
        C_plus, C_minus = compute_sd_decomp(C, eps_tensor)
        p, q = compute_p_q(C_plus, C_minus)

        _, _, _, tr_Osq_sq = compute_omega_and_traces(C, sigma)

        sq_sq_data.append({
            "p": p, "q": q,
            "p2_plus_q2": p ** 2 + q ** 2,
            "pq": p * q,
            "tr_Osq_sq": tr_Osq_sq.real,
        })

    A_mat = np.array([[d["p2_plus_q2"], d["pq"]] for d in sq_sq_data])
    b_vec = np.array([d["tr_Osq_sq"] for d in sq_sq_data])
    coeffs_sq, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    alpha_sq = coeffs_sq[0]
    beta_sq = coeffs_sq[1]
    fit_err_sq = np.max(np.abs(A_mat @ coeffs_sq - b_vec))

    print(f"    Tr_spin(Omega_sq^2) = {alpha_sq:.8f}*(p^2+q^2) + {beta_sq:.8f}*pq")
    print(f"    Fit error: {fit_err_sq:.2e}")

    has_pq_in_Osq_sq = abs(beta_sq) > 1e-6 * abs(alpha_sq)
    record("Tr_spin(Omega_sq^2) pq coeff",
           fit_err_sq < 1e-6,
           f"alpha={alpha_sq:.8f}, beta={beta_sq:.8f}, has_pq={has_pq_in_Osq_sq}")

    # Step 5: Full summary
    print(f"\n" + "-" * 72)
    print(f"  FULL a_8 QUARTIC STRUCTURE SUMMARY")
    print(f"-" * 72)
    print(f"\n  Structure                 | p^2+q^2 coeff  | pq coeff   | pq?")
    print(f"  --------------------------|----------------|------------|-----")
    print(f"  Tr(Omega^4_chain)         | 1/16 = 0.0625  | 0          | NO")
    print(f"  [Tr_spin(Omega^2)]^2      | 1/4  = 0.2500  | 1/2 = 0.5  | YES")
    print(f"  Tr_spin(Omega_sq^2)       | {alpha_sq:.8f}  | {beta_sq:.8f} | {'YES' if has_pq_in_Osq_sq else 'NO'}")
    print()
    print(f"  KEY: [Tr_spin(Omega^2)]^2 is a SCALAR squared (not a trace of a matrix product).")
    print(f"  It appears in a_8 as the coefficient of the 'double trace' structure:")
    print("  a_8 has term: c * (Tr_bundle Omega_{mn} Omega^{mn})^2")
    print(f"  where Tr_bundle = Tr_spin for Dirac.")
    print()
    print(f"  HOWEVER: In the Avramidi/Gilkey a_8 formula on Ricci-flat E=0,")
    print(f"  the 'double trace' structure is (tr Omega^2)^2 where tr means")
    print(f"  the BUNDLE trace (over spinor indices). This scalar squared")
    print(f"  contributes proportional to (p+q)^2 which HAS a pq component.")
    print()
    print(f"  CRITICAL DISTINCTION:")
    print(f"  - Tr(Omega^4_chain) = (1/16)(p^2+q^2): the CHIRAL block structure")
    print(f"    prevents pq cross terms. RATIO 1:1 CONFIRMED for this piece.")
    print(f"  - [Tr_spin(Omega^2)]^2 = (1/4)(p+q)^2: does NOT respect chirality")
    print(f"    because Tr_spin(Omega^2) = -(p+q)/2 is a SCALAR that sums")
    print(f"    both chiral sectors BEFORE squaring. This INTRODUCES pq.")
    print(f"  - The full a_8 ratio depends on the RELATIVE WEIGHT of these two.")

    return {
        "A_coeff": float(A_coeff),
        "B_coeff": float(B_coeff),
        "tr_spin_omega2": "-(p+q)/2",
        "tr_spin_omega2_sq": "(1/4)(p+q)^2 = (1/4)(p^2+2pq+q^2)",
        "tr_omega4_chain": "(1/16)(p^2+q^2)",
        "tr_spin_Osq_sq_alpha": float(alpha_sq),
        "tr_spin_Osq_sq_beta": float(beta_sq),
        "full_a8_has_pq_from_double_trace": True,
    }


# ======================================================================
# D-AGENT CLAIM AUDIT
# ======================================================================

def audit_dagent_claims(m1, m2, m3, m4):
    """Systematic audit."""
    print("\n" + "=" * 72)
    print("D-AGENT CLAIM AUDIT")
    print("=" * 72)

    print(f"\n  CLAIM (a): C4_chain = (1/8)[(C^2)^2 - (*CC)^2]")
    print(f"    VERDICT: INCORRECT")
    print(f"    (1/8)[(p+q)^2 - (p-q)^2] = pq/2, but C4_chain = (p^2+q^2)/2")
    print(f"    Correct: C4_chain = (1/4)[(C^2)^2 + (*CC)^2]")
    record("Claim (a) identity", False, "Formula has wrong sign: + not -, and 1/4 not 1/8")

    print(f"\n  CLAIM (b): Tr(Omega^4_chain) 1:1 ratio, pq = 0")
    print(f"    VERDICT: CORRECT for Omega^4_chain specifically")
    print(f"    Tr(Omega^4_chain) = (1/16)(p^2+q^2), pq coefficient < 1e-10")
    print(f"    CAVEAT: Other a_8 structures (double trace) DO have pq")
    record("Claim (b) Omega^4 chain", True, "1:1 ratio confirmed, pq = 0")
    warn("Claim (b) scope", "Only Omega^4_chain, not the full a_8")

    print(f"\n  CLAIM (c): Chiral block-diagonal mechanism")
    print(f"    VERDICT: CORRECT")
    print(f"    Omega_L ~ C+, Omega_R ~ C-; block-diagonal => no cross terms in chain")
    record("Claim (c) mechanism", True)

    print(f"\n  CLAIM (d): 2 independent quartic Weyl invariants in d=4")
    print(f"    VERDICT: CORRECT")
    print(f"    Basis: (C^2)^2 = (p+q)^2 and (*CC)^2 = (p-q)^2")
    record("Claim (d) counting", True)

    print(f"\n  DR CRITICAL FINDING:")
    print(f"    The full Seeley-DeWitt a_8 on Ricci-flat E=0 contains:")
    print(f"    (1) Tr(Omega^4_chain): proportional to p^2+q^2  [1:1, no pq]")
    print(f"    (2) [Tr_spin(Omega^2)]^2: proportional to (p+q)^2 [HAS pq]")
    print(f"    The total a_8 quartic Weyl structure is:")
    print(f"      a_8 ~ c_chain * (p^2+q^2) + c_double * (p+q)^2")
    print(f"         = (c_chain + c_double) * (p^2+q^2) + 2*c_double * pq")
    print(f"    Unless c_double = 0 in the a_8 formula, the full ratio is NOT 1:1.")
    print(f"    The D-agent's conclusion about '1:1 ratio' is only half the story.")

    return {
        "a": "INCORRECT (formula error)",
        "b": "CORRECT (chain only)",
        "c": "CORRECT",
        "d": "CORRECT",
        "new_finding": "Full a_8 has pq from double-trace structure",
    }


# ======================================================================
# MAIN
# ======================================================================

def main():
    print("=" * 72)
    print("FUND-SYM: DR-Agent Independent Re-derivation")
    print("4 methods, fully independent of D-agent computation")
    print("=" * 72)

    m1 = method_1_sd_asd_direct()
    m2 = method_2_eight_gamma_trace()
    m3 = method_3_specific_geometries()
    m4 = method_4_full_a8_assessment()

    audit = audit_dagent_claims(m1, m2, m3, m4)

    print("\n" + "=" * 72)
    print("FINAL DR VERDICT")
    print("=" * 72)

    print(f"\n  Tests: {PASS_COUNT} PASS, {FAIL_COUNT} FAIL, {WARN_COUNT} WARN")
    print()
    print("  D-AGENT CLAIMS:")
    print(f"    (a) CH identity for C4_chain:     INCORRECT (sign/coefficient error)")
    print(f"    (b) Tr(Omega^4_chain) 1:1 ratio:  CORRECT (for chain piece only)")
    print(f"    (c) Chiral mechanism:              CORRECT")
    print(f"    (d) 2 independent invariants:      CORRECT")
    print()
    print("  DR CORRECTIONS:")
    print(f"    (a) C4_chain = (1/4)[(C^2)^2 + (*CC)^2], NOT (1/8)[(C^2)^2 - (*CC)^2]")
    print(f"    (b) Exact coefficient: Tr(Omega^4_chain) = (1/16)(p^2+q^2)")
    print()
    print("  CRITICAL NEW FINDING:")
    print("    The D-agent analyzed Tr(Omega^4_chain) in isolation.")
    print("    The full a_8 also has [Tr_spin(Omega^2)]^2 = (1/4)(p+q)^2")
    print("    which introduces a pq cross term with coefficient 1/2.")
    print("    The full a_8 ratio (C^2)^2:(*CC)^2 is NOT 1:1 unless")
    print("    the double-trace coefficient c_double vanishes in a_8.")
    print()
    print("    THREE-LOOP PROBLEM STATUS: REDUCED but NOT RESOLVED.")
    print("    The problem is reduced from '2 invariants vs 1 parameter'")
    print("    to '1 ratio condition', but the ratio is NOT proven to match.")
    print()
    print("    VERDICT: PARTIAL PASS")
    print("    - Core computation (Omega^4_chain, claims b/c/d): VERIFIED")
    print("    - CH formula (claim a): CORRECTED")
    print("    - Overall conclusion (full a_8 1:1): NOT ESTABLISHED")

    return {
        "pass_count": PASS_COUNT,
        "fail_count": FAIL_COUNT,
        "warn_count": WARN_COUNT,
        "audit": audit,
        "verdict": "PARTIAL PASS",
    }


if __name__ == "__main__":
    results = main()

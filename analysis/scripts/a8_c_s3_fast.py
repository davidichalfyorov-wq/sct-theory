# ruff: noqa: E402, I001
"""
Fast extraction of c_S3: coefficient of [tr(Omega^2)]^2 in Seeley-DeWitt a_8.

Uses the worldline formalism on flat space with constant non-abelian
field strength. The quartic-order Wick contraction is precomputed
symbolically and then evaluated numerically.

Author: David Alfyorov
"""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from itertools import product as iproduct
from pathlib import Path

import numpy as np
from numpy import einsum

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "analysis" / "results" / "a8_dirac"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

D = 4
PASS_COUNT = 0
FAIL_COUNT = 0


def record(label, passed, detail=""):
    global PASS_COUNT, FAIL_COUNT
    tag = "PASS" if passed else "FAIL"
    if passed:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1
    print(f"  [{tag}] {label}" + (f" -- {detail}" if detail else ""))


# ===================================================================
# Infrastructure
# ===================================================================

def build_gamma_euclidean():
    I2 = np.eye(2, dtype=complex)
    s1 = np.array([[0, 1], [1, 0]], dtype=complex)
    s2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    s3 = np.array([[1, 0], [0, -1]], dtype=complex)
    gamma = np.zeros((D, 4, 4), dtype=complex)
    gamma[0] = np.kron(s1, I2)
    gamma[1] = np.kron(s2, I2)
    gamma[2] = np.kron(s3, s1)
    gamma[3] = np.kron(s3, s2)
    return gamma


def build_sigma(gamma):
    sig = np.zeros((D, D, 4, 4), dtype=complex)
    for a in range(D):
        for b in range(D):
            sig[a, b] = 0.5 * (gamma[a] @ gamma[b] - gamma[b] @ gamma[a])
    return sig


def build_levi_civita():
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


def build_thooft_symbols():
    eta = np.zeros((3, D, D))
    etabar = np.zeros((3, D, D))
    eta[0, 0, 1] = 1;  eta[0, 1, 0] = -1
    eta[0, 2, 3] = 1;  eta[0, 3, 2] = -1
    eta[1, 0, 2] = 1;  eta[1, 2, 0] = -1
    eta[1, 3, 1] = 1;  eta[1, 1, 3] = -1
    eta[2, 0, 3] = 1;  eta[2, 3, 0] = -1
    eta[2, 1, 2] = 1;  eta[2, 2, 1] = -1
    etabar[0, 0, 1] = 1;  etabar[0, 1, 0] = -1
    etabar[0, 2, 3] = -1; etabar[0, 3, 2] = 1
    etabar[1, 0, 2] = 1;  etabar[1, 2, 0] = -1
    etabar[1, 3, 1] = -1; etabar[1, 1, 3] = 1
    etabar[2, 0, 3] = 1;  etabar[2, 3, 0] = -1
    etabar[2, 1, 2] = -1; etabar[2, 2, 1] = 1
    return eta, etabar


def make_weyl(W_plus, W_minus, eta, etabar):
    C = np.zeros((D, D, D, D))
    for i in range(3):
        for j in range(3):
            C += W_plus[i, j] * einsum('ab,cd->abcd', eta[i], eta[j])
            C += W_minus[i, j] * einsum('ab,cd->abcd', etabar[i], etabar[j])
    return C


def random_traceless_sym_3x3(rng):
    A = rng.standard_normal((3, 3))
    A = (A + A.T) / 2.0
    A -= np.trace(A) / 3.0 * np.eye(3)
    return A


def generate_random_weyl(rng):
    eta, etabar = build_thooft_symbols()
    W_plus = random_traceless_sym_3x3(rng)
    W_minus = random_traceless_sym_3x3(rng)
    C = make_weyl(W_plus, W_minus, eta, etabar)
    return C, W_plus, W_minus


def sd_decompose(C, eps):
    star_C = 0.5 * einsum('abef,efcd->abcd', eps, C)
    C_plus = 0.5 * (C + star_C)
    C_minus = 0.5 * (C - star_C)
    return C_plus, C_minus


def compute_p_q(C_plus, C_minus):
    p = float(einsum('abcd,abcd->', C_plus, C_plus))
    q = float(einsum('abcd,abcd->', C_minus, C_minus))
    return p, q


def build_omega(C_weyl, sigma):
    Omega = np.zeros((D, D, 4, 4), dtype=complex)
    for mu in range(D):
        for nu in range(D):
            for rho in range(D):
                for sig in range(D):
                    Omega[mu, nu] += 0.25 * C_weyl[mu, nu, rho, sig] * sigma[rho, sig]
    return Omega


# ===================================================================
# Perfect matchings of {0,...,7}
# ===================================================================

def _generate_matchings(items):
    """Generate all perfect matchings of a list."""
    if len(items) == 0:
        yield []
        return
    first = items[0]
    rest = items[1:]
    for i, partner in enumerate(rest):
        remaining = rest[:i] + rest[i+1:]
        for matching in _generate_matchings(remaining):
            yield [(first, partner)] + matching


# Precompute and cache the 105 matchings
ALL_MATCHINGS = list(_generate_matchings(list(range(8))))
assert len(ALL_MATCHINGS) == 105


# ===================================================================
# Worldline Green's functions on [0,1]
# ===================================================================

def gb(u, v):
    return min(u, v) - u * v


def gb_dot1(u, v):
    """dG/du."""
    if u < v:
        return 1.0 - v
    elif u > v:
        return -v
    else:
        return 0.5 - v


def gb_dot2(u, v):
    """dG/dv."""
    return gb_dot1(v, u)


# ===================================================================
# Core computation: a_4^{fiber} trace via worldline Wick contraction
# ===================================================================

def compute_tr_a4_fiber(Omega, n_quad=12):
    """Compute tr_spin[a_4^{fiber}] on flat space with constant Omega.

    Uses the worldline formalism with Wick contractions.
    The 4th-order term in the expansion of the path-ordered Wilson line:

    a_4^{fiber} = (1/16) * int_{0<u4<u3<u2<u1<1} du1...du4
                  * sum_{Wick pairings} (propagator products)
                  * (contracted Omega^4 matrices)

    We take the trace over fiber (spinor) indices.
    """
    from numpy.polynomial.legendre import leggauss

    nodes, weights = leggauss(n_quad)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights

    # For each matching, precompute the "type" of each pair.
    # Fields: 2*i -> z_i (type 0), 2*i+1 -> dz_i (type 1)
    # vertex(k) = k // 2, type(k) = k % 2
    # Pair types: (0,0)=zz, (0,1)=zdz, (1,0)=dzz, (1,1)=dzdz

    # Precompute the GROUP STRUCTURE for each matching.
    # For each matching, determine:
    # 1. The covariance type for each pair
    # 2. The index contraction pattern (which Omega indices are identified)

    matching_info = []
    for matching in ALL_MATCHINGS:
        # Build union-find for index groups
        parent = list(range(8))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        for (a, b) in matching:
            union(a, b)

        # Find unique groups
        groups = {}
        for k in range(8):
            r = find(k)
            if r not in groups:
                groups[r] = []
            groups[r].append(k)

        roots = sorted(groups.keys())
        n_free = len(roots)

        # For each field, record which group root it belongs to
        field_group = [find(k) for k in range(8)]

        # Pair types: (type_a, type_b, vertex_a, vertex_b)
        pair_info = []
        for (a, b) in matching:
            ta = a % 2  # 0=z, 1=dz
            tb = b % 2
            va = a // 2
            vb = b // 2
            pair_info.append((ta, tb, va, vb))

        matching_info.append({
            'pairs': matching,
            'pair_info': pair_info,
            'field_group': field_group,
            'roots': roots,
            'n_free': n_free,
        })

    # Quadrature over the ordered simplex 0 < u4 < u3 < u2 < u1 < 1
    # Transform: u1=v1, u2=v1*v2, u3=v1*v2*v3, u4=v1*v2*v3*v4
    # Jacobian: v1^3 * v2^2 * v3

    result_trace = 0.0

    for i1 in range(n_quad):
        v1 = nodes[i1]
        u1 = v1
        for i2 in range(n_quad):
            v2 = nodes[i2]
            u2 = v1 * v2
            for i3 in range(n_quad):
                v3 = nodes[i3]
                u3 = v1 * v2 * v3
                for i4 in range(n_quad):
                    v4 = nodes[i4]
                    u4 = v1 * v2 * v3 * v4

                    jac = v1**3 * v2**2 * v3
                    w_total = weights[i1] * weights[i2] * weights[i3] * weights[i4] * jac

                    u = np.array([u1, u2, u3, u4])

                    # Precompute propagators between all vertex pairs
                    G = np.zeros((4, 4))
                    Gd1 = np.zeros((4, 4))
                    Gd2 = np.zeros((4, 4))
                    Gdd_val = -1.0  # d^2G/du_i du_j for i != j
                    for ii in range(4):
                        for jj in range(4):
                            G[ii, jj] = gb(u[ii], u[jj])
                            Gd1[ii, jj] = gb_dot1(u[ii], u[jj])
                            Gd2[ii, jj] = gb_dot2(u[ii], u[jj])

                    # For each matching, compute contribution
                    for minfo in matching_info:
                        # Compute product of covariances
                        prod_cov = 1.0
                        for (ta, tb, va, vb) in minfo['pair_info']:
                            if ta == 0 and tb == 0:
                                prod_cov *= 2.0 * G[va, vb]
                            elif ta == 0 and tb == 1:
                                prod_cov *= 2.0 * Gd2[va, vb]
                            elif ta == 1 and tb == 0:
                                prod_cov *= 2.0 * Gd1[va, vb]
                            else:  # ta == 1 and tb == 1
                                prod_cov *= 2.0 * Gdd_val

                        if abs(prod_cov) < 1e-30:
                            continue

                        # Sum over spacetime indices
                        # Each group root gets a free index 0..D-1.
                        # For each field k:
                        #   vertex = k // 2
                        #   if k even: index = n_{vertex} (2nd index of Omega)
                        #   if k odd: index = m_{vertex} (1st index of Omega)
                        #   value = group_assignment[root of k]

                        roots = minfo['roots']
                        fg = minfo['field_group']
                        n_free = minfo['n_free']

                        # Sum over D^n_free assignments
                        tr_sum = 0.0
                        for assignment in iproduct(range(D), repeat=n_free):
                            gv = dict(zip(roots, assignment))
                            # m_i and n_i for each vertex
                            vals = [gv[fg[k]] for k in range(8)]
                            # n_i = vals[2*i], m_i = vals[2*i+1]
                            m = [vals[2*i+1] for i in range(4)]
                            n = [vals[2*i] for i in range(4)]

                            mat = (Omega[m[0], n[0]]
                                   @ Omega[m[1], n[1]]
                                   @ Omega[m[2], n[2]]
                                   @ Omega[m[3], n[3]])
                            tr_sum += np.trace(mat).real

                        result_trace += (1.0 / 16.0) * w_total * prod_cov * tr_sum

    return result_trace


# ===================================================================
# Verification: check a_2^{fiber} trace
# ===================================================================

def compute_tr_a2_fiber(Omega, n_quad=12):
    """Verify: compute tr_spin[a_2^{fiber}] via worldline (should match (1/12)*tr(Omega_sq)).

    a_2^{fiber} = (1/4) * int_{0<u2<u1<1} du1 du2
                  * <V(u1) V(u2)>_Wick * (Omega product trace)

    where V(u_i) = (1/2) Omega_{m_i n_i} z^{n_i} dz^{m_i}.
    """
    from numpy.polynomial.legendre import leggauss

    nodes, weights = leggauss(n_quad)
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights

    # 4 fields: z1(u1), dz1(u1), z2(u2), dz2(u2)
    # Perfect matchings of {0,1,2,3}: 3 matchings
    matchings_4 = list(_generate_matchings(list(range(4))))
    assert len(matchings_4) == 3

    result = 0.0

    for i1 in range(n_quad):
        v1 = nodes[i1]
        u1 = v1
        for i2 in range(n_quad):
            v2 = nodes[i2]
            u2 = v1 * v2

            jac = v1
            w_total = weights[i1] * weights[i2] * jac

            u = np.array([u1, u2])

            G_val = gb(u1, u2)
            Gd1_val = gb_dot1(u1, u2)
            Gd2_val = gb_dot2(u1, u2)
            Gdd_val = -1.0

            for matching in matchings_4:
                prod_cov = 1.0
                for (a, b) in matching:
                    ta = a % 2
                    tb = b % 2
                    va = a // 2
                    vb = b // 2
                    if ta == 0 and tb == 0:
                        c = 2.0 * gb(u[va], u[vb])
                    elif ta == 0 and tb == 1:
                        c = 2.0 * gb_dot2(u[va], u[vb])
                    elif ta == 1 and tb == 0:
                        c = 2.0 * gb_dot1(u[va], u[vb])
                    else:
                        c = 2.0 * (-1.0)
                    prod_cov *= c

                if abs(prod_cov) < 1e-30:
                    continue

                # Index contraction
                parent = list(range(4))
                def find(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x
                for (a, b) in matching:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[rb] = ra

                groups = {}
                for k in range(4):
                    r = find(k)
                    if r not in groups:
                        groups[r] = []
                    groups[r].append(k)
                roots = sorted(groups.keys())
                fg = [find(k) for k in range(4)]
                n_free = len(roots)

                tr_sum = 0.0
                for assignment in iproduct(range(D), repeat=n_free):
                    gv = dict(zip(roots, assignment))
                    vals = [gv[fg[k]] for k in range(4)]
                    m = [vals[2*i+1] for i in range(2)]
                    n = [vals[2*i] for i in range(2)]
                    mat = Omega[m[0], n[0]] @ Omega[m[1], n[1]]
                    tr_sum += np.trace(mat).real

                result += (1.0 / 4.0) * w_total * prod_cov * tr_sum

    return result


# ===================================================================
# Main extraction
# ===================================================================

def run():
    print("=" * 72)
    print("c_S3 EXTRACTION via worldline Wick contraction")
    print("=" * 72)

    gamma = build_gamma_euclidean()
    sigma = build_sigma(gamma)
    eps = build_levi_civita()

    rng = np.random.default_rng(2026_03_16)

    # --- STEP 1: Verify a_2 ---
    print("\n--- STEP 1: Verify a_2^{fiber} ---")
    C, _, _ = generate_random_weyl(rng)
    Omega = build_omega(C, sigma)
    C_plus, C_minus = sd_decompose(C, eps)
    p, q = compute_p_q(C_plus, C_minus)

    Omega_sq_mat = np.zeros((4, 4), dtype=complex)
    for a in range(D):
        for b in range(D):
            Omega_sq_mat += Omega[a, b] @ Omega[a, b]

    tr_a2_expected = (1.0/12.0) * np.trace(Omega_sq_mat).real
    tr_a2_wl = compute_tr_a2_fiber(Omega, n_quad=16)

    print(f"  tr[a_2^{{fiber}}] expected = {tr_a2_expected:.10e}")
    print(f"  tr[a_2^{{fiber}}] worldline = {tr_a2_wl:.10e}")
    rel_diff = abs(tr_a2_wl - tr_a2_expected) / (abs(tr_a2_expected) + 1e-30)
    print(f"  Relative difference: {rel_diff:.2e}")
    record("a_2 verification", rel_diff < 1e-4, f"rel_diff={rel_diff:.2e}")

    if rel_diff > 0.01:
        print("  WARNING: a_2 verification FAILED. Worldline method unreliable.")
        print("  Aborting a_4 computation.")
        return None

    # --- STEP 2: Compute a_4^{fiber} for multiple backgrounds ---
    print("\n--- STEP 2: Compute a_4^{fiber} for multiple Weyl backgrounds ---")
    n_quad_a4 = 10  # Keep low for feasibility
    n_trials = 10

    print(f"  n_quad = {n_quad_a4}, n_trials = {n_trials}")
    print(f"  Estimated time: ~{n_trials * n_quad_a4**4 * 105 / 1e6:.0f} sec (rough)")

    data = []
    for trial in range(n_trials):
        C, _, _ = generate_random_weyl(rng)
        Omega = build_omega(C, sigma)
        C_plus, C_minus = sd_decompose(C, eps)
        p, q = compute_p_q(C_plus, C_minus)

        tr_a4 = compute_tr_a4_fiber(Omega, n_quad=n_quad_a4)

        Omega_sq = np.zeros((4, 4), dtype=complex)
        for a in range(D):
            for b in range(D):
                Omega_sq += Omega[a, b] @ Omega[a, b]
        tr_Osq = np.trace(Omega_sq).real

        data.append({
            "p": p, "q": q,
            "p2q2": p**2 + q**2, "pq": p * q,
            "tr_a4": tr_a4,
            "tr_Osq": tr_Osq,
        })
        print(f"  Trial {trial+1}/{n_trials}: p={p:.4f}, q={q:.4f}, "
              f"tr_a4={tr_a4:.8e}")

    # --- STEP 3: Fit and extract ---
    print("\n--- STEP 3: Fit tr[a_4^{fiber}] = alpha*(p^2+q^2) + beta*pq ---")
    A_mat = np.array([[d["p2q2"], d["pq"]] for d in data])
    b_vec = np.array([d["tr_a4"] for d in data])
    coeffs, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    alpha_fit = coeffs[0]
    beta_fit = coeffs[1]
    fit_vals = A_mat @ coeffs
    max_err = np.max(np.abs(fit_vals - b_vec))
    rel_err = max_err / (np.max(np.abs(b_vec)) + 1e-30)

    print(f"  alpha = {alpha_fit:.10e}")
    print(f"  beta  = {beta_fit:.10e}")
    print(f"  beta/alpha = {beta_fit / alpha_fit:.8f}")
    print(f"  Max fit error: {max_err:.2e}")
    print(f"  Rel fit error: {rel_err:.2e}")

    c_S3 = 2.0 * beta_fit
    c_other = alpha_fit - c_S3 / 4.0

    print(f"\n  c_S3 (coeff of [tr(Omega^2)]^2) = {c_S3:.10e}")
    print(f"  c_other (coeff of non-pq structures) = {c_other:.10e}")

    is_zero = abs(beta_fit) < 1e-3 * abs(alpha_fit)
    print(f"\n  IS c_S3 ZERO? {'YES' if is_zero else 'NO'}")
    print(f"  |beta/alpha| = {abs(beta_fit/alpha_fit):.6e}")

    record("c_S3 extraction complete", True, f"c_S3={c_S3:.6e}")
    record("Fit quality", rel_err < 0.01, f"rel_err={rel_err:.2e}")
    record("beta nonzero test", not is_zero,
           f"|beta/alpha|={abs(beta_fit/alpha_fit):.4e}")

    print(f"\n  Total checks: {PASS_COUNT} PASS, {FAIL_COUNT} FAIL")

    results = {
        "alpha": float(alpha_fit),
        "beta": float(beta_fit),
        "c_S3": float(c_S3),
        "c_other": float(c_other),
        "beta_over_alpha": float(beta_fit / alpha_fit),
        "max_fit_err": float(max_err),
        "rel_fit_err": float(rel_err),
        "is_c_S3_zero": is_zero,
        "n_trials": n_trials,
        "n_quad": n_quad_a4,
        "pass_count": PASS_COUNT,
        "fail_count": FAIL_COUNT,
    }

    with open(RESULTS_DIR / "a8_c_S3_extraction.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {RESULTS_DIR / 'a8_c_S3_extraction.json'}")
    return results


if __name__ == "__main__":
    run()

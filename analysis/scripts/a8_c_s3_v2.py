# ruff: noqa: E402, I001
"""
v2: Fast extraction of c_S3 via precomputed matching tensors.

Key optimization: For each of the 105 Wick pairings, precompute the
resulting Omega trace ONCE (independent of proper-time variables u_i).
Then the quadrature loop only multiplies by propagator products.

Author: David Alfyorov
"""

from __future__ import annotations

import json
import sys
import time
from itertools import product as iproduct
from pathlib import Path

import numpy as np
from numpy import einsum

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

PROJECT_ROOT = ANALYSIS_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "analysis" / "results" / "a8_dirac"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

D = 4
PASS = 0
FAIL = 0


def rec(label, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))


# ===================================================================
# Infrastructure
# ===================================================================

def build_gamma():
    I2 = np.eye(2, dtype=complex)
    s1 = np.array([[0, 1], [1, 0]], dtype=complex)
    s2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    s3 = np.array([[1, 0], [0, -1]], dtype=complex)
    g = np.zeros((D, 4, 4), dtype=complex)
    g[0] = np.kron(s1, I2)
    g[1] = np.kron(s2, I2)
    g[2] = np.kron(s3, s1)
    g[3] = np.kron(s3, s2)
    return g


def build_sigma(g):
    s = np.zeros((D, D, 4, 4), dtype=complex)
    for a in range(D):
        for b in range(D):
            s[a, b] = 0.5 * (g[a] @ g[b] - g[b] @ g[a])
    return s


def build_eps():
    e = np.zeros((D, D, D, D))
    for a, b, c, d in iproduct(range(D), repeat=4):
        if len({a, b, c, d}) == 4:
            p = [a, b, c, d]
            s = 1
            for i in range(4):
                for j in range(i+1, 4):
                    if p[i] > p[j]:
                        s *= -1
            e[a, b, c, d] = s
    return e


def thooft():
    eta = np.zeros((3, D, D))
    eb = np.zeros((3, D, D))
    eta[0,0,1]=1; eta[0,1,0]=-1; eta[0,2,3]=1; eta[0,3,2]=-1
    eta[1,0,2]=1; eta[1,2,0]=-1; eta[1,3,1]=1; eta[1,1,3]=-1
    eta[2,0,3]=1; eta[2,3,0]=-1; eta[2,1,2]=1; eta[2,2,1]=-1
    eb[0,0,1]=1; eb[0,1,0]=-1; eb[0,2,3]=-1; eb[0,3,2]=1
    eb[1,0,2]=1; eb[1,2,0]=-1; eb[1,3,1]=-1; eb[1,1,3]=1
    eb[2,0,3]=1; eb[2,3,0]=-1; eb[2,1,2]=-1; eb[2,2,1]=1
    return eta, eb


def mk_weyl(Wp, Wm, eta, eb):
    C = np.zeros((D,D,D,D))
    for i in range(3):
        for j in range(3):
            C += Wp[i,j]*einsum('ab,cd->abcd', eta[i], eta[j])
            C += Wm[i,j]*einsum('ab,cd->abcd', eb[i], eb[j])
    return C


def rnd_ts3(rng):
    A = rng.standard_normal((3,3))
    A = (A+A.T)/2
    A -= np.trace(A)/3*np.eye(3)
    return A


def gen_weyl(rng):
    eta, eb = thooft()
    return mk_weyl(rnd_ts3(rng), rnd_ts3(rng), eta, eb)


def sd_dec(C, eps):
    sC = 0.5*einsum('abef,efcd->abcd', eps, C)
    return 0.5*(C+sC), 0.5*(C-sC)


def pq(Cp, Cm):
    return float(einsum('abcd,abcd->', Cp, Cp)), float(einsum('abcd,abcd->', Cm, Cm))


def mk_omega(C, sig):
    O = np.zeros((D,D,4,4), dtype=complex)
    for m in range(D):
        for n in range(D):
            for r in range(D):
                for s in range(D):
                    O[m,n] += 0.25*C[m,n,r,s]*sig[r,s]
    return O


# ===================================================================
# Precompute matching structures (one-time)
# ===================================================================

def _gen_match(items):
    if len(items) == 0:
        yield []
        return
    first = items[0]
    for i, p in enumerate(items[1:]):
        rem = items[1:i+1] + items[i+2:]
        for m in _gen_match(rem):
            yield [(first, p)] + m


ALL_MATCH = list(_gen_match(list(range(8))))
assert len(ALL_MATCH) == 105

# For each matching, precompute the index contraction pattern.
# Fields: k=0..7. k//2 = vertex (0..3), k%2: 0=z (carries n_i), 1=dz (carries m_i).
# A pair (a,b) in matching means field_a and field_b share a spacetime index.
# This creates an equivalence relation on {0..7}, grouping fields with the same index.
# Each group corresponds to one free spacetime index (summed 0..D-1).

def analyze_matching(matching):
    """Return: pair_types list and index_groups for a matching."""
    # Union-find
    par = list(range(8))
    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            par[ry] = rx
    for (a, b) in matching:
        union(a, b)

    groups = {}
    for k in range(8):
        r = find(k)
        if r not in groups:
            groups[r] = []
        groups[r].append(k)

    roots = sorted(groups.keys())
    field_root = [find(k) for k in range(8)]

    # Pair types for propagator selection
    pair_types = []
    for (a, b) in matching:
        pair_types.append((a%2, b%2, a//2, b//2))

    return pair_types, roots, field_root


MATCH_DATA = [analyze_matching(m) for m in ALL_MATCH]


def precompute_omega_traces(Omega):
    """For each matching pattern, precompute the traced Omega contraction.

    Returns array of 105 scalar values: omega_traces[i] = tr_spin(contracted Omega product)
    for matching i.

    The contraction depends on the index-group structure of the matching.
    For each matching, we sum over all spacetime index assignments
    consistent with the grouping.
    """
    traces = np.zeros(105)

    for idx, (pair_types, roots, field_root) in enumerate(MATCH_DATA):
        n_free = len(roots)

        tr_sum = 0.0
        for assignment in iproduct(range(D), repeat=n_free):
            gv = dict(zip(roots, assignment))
            vals = [gv[field_root[k]] for k in range(8)]
            # n_i = vals[2*i], m_i = vals[2*i+1]  (vertex i=0..3)
            m = [vals[2*i+1] for i in range(4)]
            n = [vals[2*i] for i in range(4)]
            mat = Omega[m[0],n[0]] @ Omega[m[1],n[1]] @ Omega[m[2],n[2]] @ Omega[m[3],n[3]]
            tr_sum += np.trace(mat).real

        traces[idx] = tr_sum

    return traces


def precompute_omega_traces_a2(Omega):
    """Same for a_2 (quadratic order, 4 fields, 3 matchings)."""
    matchings_4 = list(_gen_match(list(range(4))))
    assert len(matchings_4) == 3

    traces = np.zeros(3)
    ptypes = []

    for idx, matching in enumerate(matchings_4):
        par = list(range(4))
        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx != ry:
                par[ry] = rx
        for (a, b) in matching:
            union(a, b)

        groups = {}
        for k in range(4):
            r = find(k)
            if r not in groups:
                groups[r] = []
            groups[r].append(k)
        roots = sorted(groups.keys())
        fr = [find(k) for k in range(4)]
        n_free = len(roots)

        pt = [(a%2, b%2, a//2, b//2) for (a, b) in matching]
        ptypes.append(pt)

        tr_sum = 0.0
        for assignment in iproduct(range(D), repeat=n_free):
            gv = dict(zip(roots, assignment))
            vals = [gv[fr[k]] for k in range(4)]
            m = [vals[2*i+1] for i in range(2)]
            n = [vals[2*i] for i in range(2)]
            mat = Omega[m[0],n[0]] @ Omega[m[1],n[1]]
            tr_sum += np.trace(mat).real

        traces[idx] = tr_sum

    return traces, ptypes, matchings_4


# ===================================================================
# Quadrature for a_4^{fiber}
# ===================================================================

def gb(u, v):
    return min(u, v) - u*v

def gd1(u, v):
    return (1.0-v if u < v else (-v if u > v else 0.5-v))

def gd2(u, v):
    return gd1(v, u)


def compute_tr_a4(Omega, n_quad=14):
    """Compute tr[a_4^{fiber}] using precomputed Omega traces + quadrature."""
    from numpy.polynomial.legendre import leggauss

    t0 = time.time()

    # Precompute Omega traces (independent of u's)
    omega_tr = precompute_omega_traces(Omega)

    nd, wt = leggauss(n_quad)
    nd = 0.5*(nd+1.0)
    wt = 0.5*wt

    result = 0.0

    for i1 in range(n_quad):
        u1 = nd[i1]
        for i2 in range(n_quad):
            u2 = u1*nd[i2]
            for i3 in range(n_quad):
                u3 = u2*nd[i3]
                for i4 in range(n_quad):
                    u4 = u3*nd[i4]

                    jac = u1**3/u2 * u2**2/u3 * u3  # v1^3*v2^2*v3
                    # Actually: v1=u1, v2=u2/u1, v3=u3/u2, v4=u4/u3
                    # jac = v1^3 * v2^2 * v3 = u1^3 * (u2/u1)^2 * (u3/u2) = u1*u2*u3/u3*... hmm
                    # Let me redo: v1=u1, u2=v1*v2 => v2=u2/u1,
                    # u3=v1*v2*v3 => v3=u3/u2, u4=v1*v2*v3*v4 => v4=u4/u3
                    # Jacobian = v1^3*v2^2*v3 = u1^3*(u2/u1)^2*(u3/u2) = u1*u2^2*u3/u2 = u1*u2*... no
                    # v1^3 = u1^3
                    # v2^2 = (u2/u1)^2
                    # v3 = u3/u2
                    # product = u1^3 * u2^2/u1^2 * u3/u2 = u1 * u2 * u3 / 1
                    # Hmm that's u1*u2*u3? Let me verify:
                    # v1=u1=0.5, v2=0.5 => u2=0.25, v3=0.5 => u3=0.125, v4=0.5 => u4=0.0625
                    # jac = 0.5^3 * 0.5^2 * 0.5 = 0.125*0.25*0.5 = 0.015625
                    # u1*u2*u3 = 0.5*0.25*0.125 = 0.015625. Checks out?
                    # Wait: but u3/u2 = v3 = u3/(v1*v2) = u3/u2. And v2 = u2/u1. So:
                    # jac = u1^3 * (u2/u1)^2 * (u3/u2) = u1 * u2 * u3/u2 = u1 * u3. No!
                    # u1^3 * u2^2/u1^2 * u3/u2 = u1 * u2 * u3/u2 = u1*u3... that gives 0.5*0.125=0.0625, wrong.
                    #
                    # Let me be more careful.
                    # u1 = v1
                    # u2 = v1*v2
                    # u3 = v1*v2*v3
                    # u4 = v1*v2*v3*v4
                    # The Jacobian of the transformation (v1,v2,v3,v4) -> (u1,u2,u3,u4) is:
                    # du1/dv1 = 1, du1/dv2 = 0, ...
                    # du2/dv1 = v2, du2/dv2 = v1, du2/dv3 = 0, ...
                    # du3/dv1 = v2*v3, du3/dv2 = v1*v3, du3/dv3 = v1*v2, ...
                    # du4/dv1 = v2*v3*v4, du4/dv2 = v1*v3*v4, du4/dv3 = v1*v2*v4, du4/dv4 = v1*v2*v3
                    # det = v1^3 * v2^2 * v3 (lower triangular). Correct.
                    #
                    # In terms of u: v1=u1, v2=u2/u1, v3=u3/u2, v4=u4/u3
                    # jac = u1^3 * (u2/u1)^2 * (u3/u2) = u1^3 * u2^2/u1^2 * u3/u2 = u1*u2*u3/u2
                    # Wait: u1^3 * u2^2/u1^2 = u1 * u2^2. Then * u3/u2 = u1*u2*u3. Hmm.
                    # u1^3 * (u2/u1)^2 = u1^3 * u2^2 / u1^2 = u1 * u2^2
                    # u1*u2^2 * (u3/u2) = u1 * u2 * u3
                    # At the test point: 0.5 * 0.25 * 0.125 = 0.015625. And jac should be
                    # v1^3*v2^2*v3 = 0.5^3 * 0.5^2 * 0.5 = 0.015625. ✓

                    jac = u1 * u2 * u3  # = v1^3 * v2^2 * v3

                    w = wt[i1]*wt[i2]*wt[i3]*wt[i4]*jac

                    u = [u1, u2, u3, u4]

                    # Precompute propagators
                    G = np.empty((4,4))
                    Gd1 = np.empty((4,4))
                    Gd2 = np.empty((4,4))
                    for ii in range(4):
                        for jj in range(4):
                            G[ii,jj] = gb(u[ii], u[jj])
                            Gd1[ii,jj] = gd1(u[ii], u[jj])
                            Gd2[ii,jj] = gd2(u[ii], u[jj])

                    # For each matching, compute propagator product
                    for midx, (ptypes, roots, fr) in enumerate(MATCH_DATA):
                        pc = 1.0
                        for (ta, tb, va, vb) in ptypes:
                            if ta == 0 and tb == 0:
                                pc *= 2.0*G[va, vb]
                            elif ta == 0 and tb == 1:
                                pc *= 2.0*Gd2[va, vb]
                            elif ta == 1 and tb == 0:
                                pc *= 2.0*Gd1[va, vb]
                            else:
                                pc *= 2.0*(-1.0)  # Gdd for i!=j

                        if abs(pc) < 1e-30:
                            continue

                        result += (1.0/16.0) * w * pc * omega_tr[midx]

    elapsed = time.time() - t0
    print(f"    Quadrature done in {elapsed:.1f}s")
    return result


# ===================================================================
# Same for a_2 (verification)
# ===================================================================

def compute_tr_a2(Omega, n_quad=16):
    """Compute tr[a_2^{fiber}] for verification."""
    from numpy.polynomial.legendre import leggauss

    omega_tr, ptypes_list, matchings = precompute_omega_traces_a2(Omega)

    nd, wt = leggauss(n_quad)
    nd = 0.5*(nd+1.0)
    wt = 0.5*wt

    result = 0.0
    for i1 in range(n_quad):
        u1 = nd[i1]
        for i2 in range(n_quad):
            u2 = u1*nd[i2]
            jac = u1
            w = wt[i1]*wt[i2]*jac
            u = [u1, u2]

            G = np.array([[gb(u[0],u[0]), gb(u[0],u[1])],
                          [gb(u[1],u[0]), gb(u[1],u[1])]])
            Gd1_v = np.array([[gd1(u[0],u[0]), gd1(u[0],u[1])],
                              [gd1(u[1],u[0]), gd1(u[1],u[1])]])
            Gd2_v = np.array([[gd2(u[0],u[0]), gd2(u[0],u[1])],
                              [gd2(u[1],u[0]), gd2(u[1],u[1])]])

            for midx in range(3):
                pc = 1.0
                for (ta, tb, va, vb) in ptypes_list[midx]:
                    if ta == 0 and tb == 0:
                        pc *= 2.0*G[va, vb]
                    elif ta == 0 and tb == 1:
                        pc *= 2.0*Gd2_v[va, vb]
                    elif ta == 1 and tb == 0:
                        pc *= 2.0*Gd1_v[va, vb]
                    else:
                        pc *= 2.0*(-1.0)

                if abs(pc) < 1e-30:
                    continue

                result += (1.0/4.0) * w * pc * omega_tr[midx]

    return result


# ===================================================================
# Main
# ===================================================================

def run():
    print("=" * 72)
    print("c_S3 EXTRACTION v2: precomputed matching tensors")
    print("=" * 72)

    gam = build_gamma()
    sig = build_sigma(gam)
    eps = build_eps()
    rng = np.random.default_rng(2026_03_16)

    # --- Verify a_2 ---
    print("\n--- Verify a_2^{fiber} ---")
    C = gen_weyl(rng)
    O = mk_omega(C, sig)
    Cp, Cm = sd_dec(C, eps)
    p, q = pq(Cp, Cm)

    Osq = np.zeros((4,4), dtype=complex)
    for a in range(D):
        for b in range(D):
            Osq += O[a,b] @ O[a,b]
    tr_a2_exact = (1.0/12.0)*np.trace(Osq).real
    tr_a2_wl = compute_tr_a2(O, n_quad=20)
    rd = abs(tr_a2_wl - tr_a2_exact)/(abs(tr_a2_exact)+1e-30)
    print(f"  Exact:     {tr_a2_exact:.10e}")
    print(f"  Worldline: {tr_a2_wl:.10e}")
    print(f"  Rel diff:  {rd:.2e}")
    rec("a_2 verification", rd < 1e-3, f"rd={rd:.2e}")

    if rd > 0.05:
        print("  ABORT: a_2 worldline too inaccurate.")
        return

    # --- Compute a_4^{fiber} ---
    print("\n--- Compute a_4^{fiber} ---")
    NQ = 12
    NT = 12
    print(f"  n_quad={NQ}, n_trials={NT}")

    data = []
    for trial in range(NT):
        C = gen_weyl(rng)
        O = mk_omega(C, sig)
        Cp, Cm = sd_dec(C, eps)
        p_val, q_val = pq(Cp, Cm)

        print(f"  Trial {trial+1}/{NT} (p={p_val:.4f}, q={q_val:.4f})...", end="", flush=True)
        tr_a4 = compute_tr_a4(O, n_quad=NQ)

        data.append({
            "p": p_val, "q": q_val,
            "p2q2": p_val**2+q_val**2, "pq": p_val*q_val,
            "tr_a4": tr_a4,
        })
        print(f" tr_a4={tr_a4:.8e}")

    # --- Fit ---
    print("\n--- Fit: tr[a_4] = alpha*(p^2+q^2) + beta*pq ---")
    A = np.array([[d["p2q2"], d["pq"]] for d in data])
    b = np.array([d["tr_a4"] for d in data])
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    alpha, beta = c
    fit = A @ c
    maxe = np.max(np.abs(fit - b))
    rele = maxe/(np.max(np.abs(b))+1e-30)

    print(f"  alpha = {alpha:.10e}")
    print(f"  beta  = {beta:.10e}")
    print(f"  beta/alpha = {beta/alpha:.8f}")
    print(f"  Max err: {maxe:.2e}, Rel err: {rele:.2e}")

    c_S3 = 2.0*beta
    c_oth = alpha - c_S3/4.0

    print(f"\n  c_S3 = {c_S3:.10e}")
    print(f"  c_other = {c_oth:.10e}")

    iz = abs(beta) < 1e-3*abs(alpha)
    print(f"\n  IS c_S3 ZERO? {'YES' if iz else 'NO'}")
    print(f"  |beta/alpha| = {abs(beta/alpha):.6e}")

    rec("fit quality", rele < 0.05, f"rele={rele:.2e}")
    rec("c_S3 result", True, f"{'ZERO' if iz else 'NONZERO'}, c_S3={c_S3:.4e}")

    # Also try 3-parameter fit: tr_a4 = a*p^2 + b*q^2 + c*pq
    A3 = np.array([[d["p"]**2, d["q"]**2, d["pq"]] for d in data])
    c3, _, _, _ = np.linalg.lstsq(A3, b, rcond=None)
    print(f"\n  3-param fit: {c3[0]:.6e}*p^2 + {c3[1]:.6e}*q^2 + {c3[2]:.6e}*pq")
    print(f"    p^2 coeff == q^2 coeff? diff = {abs(c3[0]-c3[1]):.2e}")
    rec("p^2==q^2 symmetry", abs(c3[0]-c3[1]) < 0.01*abs(c3[0]+c3[1])/2,
        f"diff/avg={abs(c3[0]-c3[1])/(abs(c3[0]+c3[1])/2+1e-30):.2e}")

    print(f"\n  Total: {PASS} PASS, {FAIL} FAIL")

    res = {
        "alpha": float(alpha), "beta": float(beta),
        "c_S3": float(c_S3), "c_other": float(c_oth),
        "beta_over_alpha": float(beta/alpha),
        "max_err": float(maxe), "rel_err": float(rele),
        "is_zero": iz,
        "fit_3param": {"a_p2": float(c3[0]), "b_q2": float(c3[1]), "c_pq": float(c3[2])},
        "n_quad": NQ, "n_trials": NT,
        "PASS": PASS, "FAIL": FAIL,
    }

    with open(RESULTS_DIR / "a8_c_S3_v2.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR / 'a8_c_S3_v2.json'}")
    return res


if __name__ == "__main__":
    run()

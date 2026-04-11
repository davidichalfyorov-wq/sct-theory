"""
Analytical derivation of beta from the boundary integral.
============================================================

KEY FORMULA (from R1 + our analysis):

beta = delta_k / (k_flat * eps * f(p))

delta_k = delta_k_interior + delta_k_boundary

delta_k_boundary = -rho * integral over thin shell near light cone
  of exp(-rho*V(tau)) d^4q

The thin shell: {q: 0 < tau_flat^2 <= -delta(tau^2)}
where delta(tau^2) = -eps * du^2 * [2*f(p) + cross + sep terms]

For position-dependent part (dominates):
  delta(tau^2)|_pos = -2*eps*f(p)*du^2

The integral with exp(-rho*V) cutoff:
  delta_k_bdy = -rho * integral_{0}^{infty} dU integral dX dY *
    min(eps*U^2*Q, tau_max^2) * (1/2U) * exp(-rho*V(s_mid))

Where Q(X,Y) = 2*f(p) + position-independent terms
and s_mid is the midpoint of the thin shell.

APPROACH: Numerical quadrature of the full integral, compare with
numerical CRN measurement of beta.

Author: David Alfyorov
"""
import numpy as np
from scipy import integrate as sci_integrate
from scipy import stats
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from discovery_common import sprinkle_4d, causal_flat

# GPU
_c = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
if os.path.isdir(_c):
    os.add_dll_directory(_c)
import cupy as cp


def beta_from_integral(rho, f_p, eps_val=0.01):
    """
    Compute beta from the boundary + interior integrals.

    The expected out-degree perturbation at point p with f(p) = f_p:

    delta_k_out = rho * integral d^4q [exp(-rho*V_curved)*I_curved - exp(-rho*V_flat)*I_flat]

    We compute this as two terms:
    1. BOUNDARY: pairs that change causal status
    2. INTERIOR: pairs that stay causal but have changed V

    For the position-dependent part (proportional to f_p):

    Using null coordinates: U = u-separation, V_coord = v-separation, X, Y = transverse
    tau_flat^2 = 2*U*V_coord - X^2 - Y^2
    delta(tau^2)|_pos = -2*eps*f_p*U^2  (position-dependent part)

    V_Alexandrov(tau) = (pi/24)*tau^4

    BOUNDARY contribution (from thin shell 0 < s < 2*eps*f_p*U^2):
    For each (U, X, Y), the shell has width delta_s = 2*eps*f_p*U^2 in the
    variable s = tau_flat^2. The Jacobian: d^4q = (1/2U)*ds*dU*dX*dY.
    The link probability at s ≈ 0: exp(-rho*V) ≈ exp(-rho*pi/24*s^2) ≈ 1 for small s.
    But we need to sum the contribution weighted by exp(-rho*V(s)).

    INTERIOR contribution (from pairs with s > delta_s):
    delta_P/P = rho*(pi/6)*eps*f_p*du^2*tau^2 = rho*(pi/6)*eps*f_p*U^2*s
    (using du = U, tau^2 = s, tau^4 = s^2 at leading order)

    Actually, let me be more careful with the formulas.

    For a pair (p, q) with:
    - tau_flat^2 = s = 2*U*V_coord - X^2 - Y^2
    - V_flat = (pi/24)*s^2
    - P_flat = exp(-rho*V_flat) = exp(-rho*pi/24*s^2)

    The perturbation shifts tau^2 by delta = -2*eps*f_p*U^2 (position-dep part).
    New: tau^2 = s - 2*eps*f_p*U^2

    BOUNDARY: If s < 2*eps*f_p*U^2 and f_p > 0, the pair becomes non-causal.
    Each lost pair had link probability exp(-rho*pi/24*s^2).
    Contribution: -rho * integral over lost pairs of exp(-rho*pi/24*s^2) d^4q

    INTERIOR: If s > 2*eps*f_p*U^2, the pair stays causal but V changes.
    New V = (pi/24)*(s - 2*eps*f_p*U^2)^2 ≈ V_flat - (pi/12)*eps*f_p*U^2*s
    delta_P ≈ P_flat * rho*(pi/12)*eps*f_p*U^2*s
    Contribution: +rho^2*(pi/12)*eps*f_p * integral U^2*s * P_flat d^4q
    """

    # ================================================================
    # METHOD 1: Direct numerical computation
    # Compute delta_k for a single element at position (0,0,x_p,y_p)
    # by numerical quadrature over the boundary and interior integrals.
    # ================================================================

    # The integrals in (U, X, Y) with s = 2UV - X^2 - Y^2 eliminated:
    # d^4q = dU dV dX dY = (1/2U) ds dU dX dY

    # For the BOUNDARY term (lost pairs, f_p > 0):
    # delta_k_bdy = -rho * integral_{U>0, X, Y}
    #   integral_{s=0}^{min(2*eps*f_p*U^2, s_max)} (1/2U) * exp(-rho*pi/24*s^2) ds
    #   dU dX dY

    # The s-integral:
    # integral_0^{delta_s} (1/2U) exp(-rho*pi/24*s^2) ds
    # ≈ (1/2U) * delta_s   when delta_s is small (exp ≈ 1)
    # = (1/2U) * 2*eps*f_p*U^2 = eps*f_p*U

    # But this gives delta_k_bdy = -rho * eps*f_p * integral U dU dX dY
    # which DIVERGES (as R1 noted).

    # The resolution: we must also integrate over X, Y with the constraint
    # that the point q is within the diamond AND the Q(X,Y) > 0 condition.
    # In the diamond, X^2 + Y^2 < (T/2 - |t|)^2, so the transverse extent
    # is bounded. Also, U is bounded by the diamond size.

    # For a diamond of size T=1 centered at origin:
    # The maximum U is T/sqrt(2) ≈ 0.71 (since u = (t+z)/sqrt(2))
    # The maximum X, Y depend on position.

    # For INTERIOR elements, the relevant pairs have U ~ tau_link ~ rho^{-1/4}
    # This is the proper time scale where links are likely.

    # Let me compute numerically by Monte Carlo integration.
    return None  # placeholder


def beta_numerical_direct(N=2000, M=20, eps_val=0.1):
    """
    Compute beta AND its decomposition by directly counting
    boundary vs interior contributions in a CRN experiment.
    Uses CORRECT diamond sprinkling.
    """
    print(f"Direct CRN measurement: N={N}, M={M}, eps={eps_val}")
    print("Using diamond sprinkling (discovery_common.sprinkle_4d)")

    beta_tot_list = []
    beta_int_list = []
    beta_bdy_list = []

    # Also collect link statistics for the analytical formula
    mean_tau_link = []
    mean_du2_link = []
    mean_rhoV_link = []

    t0 = time.time()

    for m in range(M):
        rng = np.random.default_rng(m * 1000 + 100)
        pts = sprinkle_4d(N, 1.0, rng)
        f_vals = (pts[:, 1]**2 - pts[:, 2]**2) / 2.0
        pts_g = cp.asarray(pts.astype(np.float32))

        t_g = pts_g[:, 0]; x_g = pts_g[:, 1]; y_g = pts_g[:, 2]; z_g = pts_g[:, 3]
        dt = t_g[None, :] - t_g[:, None]
        dx = x_g[None, :] - x_g[:, None]
        dy = y_g[None, :] - y_g[:, None]
        dz = z_g[None, :] - z_g[:, None]
        dr2 = dx**2 + dy**2 + dz**2
        mink = dt**2 - dr2

        # Flat
        C_flat = ((mink > 0) & (dt > 0))
        C_f32 = C_flat.astype(cp.float32)
        C2_f = C_f32 @ C_f32
        L_flat = (C_flat & (C2_f < 0.5))

        # PP-wave (midpoint formula)
        xm = (x_g[None, :] + x_g[:, None]) / 2
        ym = (y_g[None, :] + y_g[:, None]) / 2
        f_mid = xm**2 - ym**2
        du_cart = dt + dz
        corr = eps_val * f_mid * du_cart**2 / 2
        C_ppw = ((mink > corr) & (dt > 0))
        C_p32 = C_ppw.astype(cp.float32)
        C2_p = C_p32 @ C_p32
        L_ppw = (C_ppw & (C2_p < 0.5))

        # Degrees
        k_flat = cp.asnumpy((L_flat.astype(cp.float32).sum(0) +
                             L_flat.astype(cp.float32).sum(1)))
        k_ppw = cp.asnumpy((L_ppw.astype(cp.float32).sum(0) +
                            L_ppw.astype(cp.float32).sum(1)))
        dk_total = k_ppw - k_flat

        # Boundary decomposition
        new_causal = ((~C_flat) & C_ppw)
        lost_causal = (C_flat & (~C_ppw))
        links_gained = (new_causal & L_ppw)
        links_lost = (lost_causal & L_flat)
        dk_bdy = cp.asnumpy(
            (links_gained.astype(cp.float32).sum(0) +
             links_gained.astype(cp.float32).sum(1)) -
            (links_lost.astype(cp.float32).sum(0) +
             links_lost.astype(cp.float32).sum(1)))
        dk_int = dk_total - dk_bdy

        # Regress on f
        mask = k_flat > 5
        f_m = f_vals[mask]
        k0 = np.maximum(k_flat[mask], 1)

        s_t, _, _, _, _ = stats.linregress(f_m, dk_total[mask] / k0)
        s_i, _, _, _, _ = stats.linregress(f_m, dk_int[mask] / k0)
        s_b, _, _, _, _ = stats.linregress(f_m, dk_bdy[mask] / k0)
        beta_tot_list.append(s_t / eps_val)
        beta_int_list.append(s_i / eps_val)
        beta_bdy_list.append(s_b / eps_val)

        # Link statistics (for analytical comparison)
        li, lj = cp.where(L_flat > 0.5)
        li = cp.asnumpy(li); lj = cp.asnumpy(lj)
        dt_l = pts[lj, 0] - pts[li, 0]
        dx_l = pts[lj, 1] - pts[li, 1]
        dy_l = pts[lj, 2] - pts[li, 2]
        dz_l = pts[lj, 3] - pts[li, 3]
        tau2_l = dt_l**2 - dx_l**2 - dy_l**2 - dz_l**2
        tau_l = np.sqrt(np.maximum(tau2_l, 1e-12))
        du_l = (dt_l + dz_l)  # NOT /sqrt(2), matches discovery_common convention
        V_l = np.pi / 24 * tau_l**4

        mean_tau_link.append(tau_l.mean())
        mean_du2_link.append((du_l**2).mean())
        mean_rhoV_link.append((N * V_l).mean())

        del pts_g, C_flat, C_ppw, L_flat, L_ppw
        cp.get_default_memory_pool().free_all_blocks()

        if (m + 1) % 5 == 0:
            print(f"  trial {m+1}/{M} [{time.time()-t0:.1f}s]")

    bt = np.array(beta_tot_list)
    bi = np.array(beta_int_list)
    bb = np.array(beta_bdy_list)

    print(f"\nbeta_TOTAL    = {bt.mean():+.4f} +/- {bt.std()/np.sqrt(M):.4f}")
    print(f"beta_INTERIOR = {bi.mean():+.4f} +/- {bi.std()/np.sqrt(M):.4f}")
    print(f"beta_BOUNDARY = {bb.mean():+.4f} +/- {bb.std()/np.sqrt(M):.4f}")
    print(f"Check: int+bdy = {(bi+bb).mean():+.4f}")
    print()

    # Link statistics for analytical formula
    tau_avg = np.mean(mean_tau_link)
    du2_avg = np.mean(mean_du2_link)
    rhoV_avg = np.mean(mean_rhoV_link)
    print(f"LINK STATISTICS (for analytical formula):")
    print(f"  <tau_link> = {tau_avg:.6f}")
    print(f"  <du^2>_link = {du2_avg:.6f}")
    print(f"  <rho*V>_link = {rhoV_avg:.4f}")
    print(f"  rho = N = {N}")
    print(f"  tau_link_char = (24/(pi*rho))^(1/4) = {(24/(np.pi*N))**0.25:.6f}")
    print()

    # ANALYTICAL PREDICTION:
    # The interior contribution: each link gets delta_P/P = 4*rho*V*eps*f_p*du_hat^2
    # Averaging over links:
    # beta_int_pred = 4 * <rho*V * du_hat^2>_links / <1>_links
    # But du_hat^2 = du^2/tau^2, which diverges for near-null links.

    # Let's compute <rho*V*du^2/tau^2> on actual links:
    all_rhoV_du2_tau2 = []
    for m in range(min(M, 5)):
        rng = np.random.default_rng(m * 1000 + 100)
        pts = sprinkle_4d(N, 1.0, rng)
        pts_g = cp.asarray(pts.astype(np.float32))
        t_g = pts_g[:, 0]; x_g = pts_g[:, 1]; y_g = pts_g[:, 2]; z_g = pts_g[:, 3]
        dt = t_g[None, :] - t_g[:, None]
        dx = x_g[None, :] - x_g[:, None]
        dy = y_g[None, :] - y_g[:, None]
        dz = z_g[None, :] - z_g[:, None]
        mink = dt**2 - dx**2 - dy**2 - dz**2
        C = ((mink > 0) & (dt > 0)).astype(cp.float32)
        C2 = C @ C
        L = ((C > 0.5) & (C2 < 0.5))
        li, lj = cp.where(L > 0.5)
        li = cp.asnumpy(li); lj = cp.asnumpy(lj)
        dt_l = pts[lj, 0] - pts[li, 0]
        dx_l = pts[lj, 1] - pts[li, 1]
        dy_l = pts[lj, 2] - pts[li, 2]
        dz_l = pts[lj, 3] - pts[li, 3]
        tau2_l = dt_l**2 - dx_l**2 - dy_l**2 - dz_l**2
        tau_l = np.sqrt(np.maximum(tau2_l, 1e-12))
        du_l = dt_l + dz_l
        V_l = np.pi / 24 * tau_l**4
        du_hat2 = du_l**2 / tau_l**2
        rhoV_du_hat2 = N * V_l * du_hat2
        all_rhoV_du2_tau2.append(rhoV_du_hat2.mean())
        del pts_g, C, C2, L
        cp.get_default_memory_pool().free_all_blocks()

    rhoV_du_hat2_mean = np.mean(all_rhoV_du2_tau2)
    print(f"  <rho*V*du_hat^2>_links = {rhoV_du_hat2_mean:.4f}")
    print(f"  Predicted beta_int (= 4*<rhoV*du_hat^2>) = +{4*rhoV_du_hat2_mean:.4f}")
    print(f"  Measured  beta_int = +{bi.mean():.4f}")
    print(f"  Ratio = {bi.mean()/(4*rhoV_du_hat2_mean):.4f}")
    print()

    # For boundary: beta_bdy comes from the pairs that change causal status.
    # The boundary contribution per element = -(number of lost links at that element)
    # proportional to f(p). This is harder to compute analytically because it
    # involves the density of pairs near the light cone.

    # KEY INSIGHT: The boundary integral involves the density of near-null pairs.
    # In d=4 Minkowski, the density of causal pairs at proper time tau is:
    # n(tau) = rho * 2*pi^2 * tau^3  (from hyperboloid measure)
    # The number of links at tau: n_link(tau) = rho * 2*pi^2 * tau^3 * exp(-rho*V(tau))

    # The boundary shell at each direction has width delta_s = 2*eps*f_p*du^2
    # The number of lost links per direction:
    # n_lost(direction) = integral_0^{delta_s} rho * exp(-rho*V(s)) * (measure) ds

    # For small delta_s: n_lost ≈ rho * 1 * (measure) * delta_s
    # = rho * (1/2U) * 2*eps*f_p*U^2 * dU * dX * dY
    # = rho * eps * f_p * U * dU * dX * dY

    # Integrating over X, Y: gives a constant (bounded by diamond)
    # Integrating over U: bounded by diamond, ~ integral_0^{U_max} U dU = U_max^2/2

    # This is the SAME divergent integral R1 found.
    # The resolution: in the actual diamond, U_max depends on position and direction.
    # For an interior element, U_max ~ T/2.
    # So the boundary integral gives something proportional to rho * eps * f_p * T^2 * (area of X,Y cross-section)
    # This has the RIGHT dimensions and sign.

    return {
        "beta_total": float(bt.mean()),
        "beta_interior": float(bi.mean()),
        "beta_boundary": float(bb.mean()),
        "rhoV_du_hat2": float(rhoV_du_hat2_mean),
        "beta_int_pred": float(4 * rhoV_du_hat2_mean),
    }


if __name__ == "__main__":
    # Run with DIAMOND sprinkling for multiple N
    for N in [1000, 2000, 3000]:
        print("=" * 70)
        result = beta_numerical_direct(N=N, M=10, eps_val=0.1)
        print()

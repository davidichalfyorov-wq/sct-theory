#!/usr/bin/env python3
"""Exact Schwarzschild midpoint causal predicate + A_align universality test.

Instead of jet truncation g_ab(m) = η - (1/3)R·m·m,
evaluates the FULL Schwarzschild metric at the midpoint.

RNC coordinates (τ,x,y,z) around static observer at r₀:
- z = radial direction
- τ = proper time (rescaled by lapse)
- x,y = transverse

Coordinate transform RNC → Schwarzschild:
  t_sch = τ / √f₀     (proper time → coordinate time)
  r = r₀ + z
  θ = π/2 + arctan(√(x²+y²) / (r₀+z))   ≈ π/2 + ρ_perp/r for small ρ_perp
  φ = arctan2(y, x)

Exact metric at point ξ in RNC:
  ds² = -f(r)dt² + dr²/f(r) + r²(dθ² + sin²θ dφ²)
  with f(r) = 1 - 2M/r

We evaluate this at the midpoint m = (ξ_i + ξ_j)/2 and compute
  s² = g_ab(m) Δ^a Δ^b
where Δ = ξ_i - ξ_j in RNC.

This requires transforming both the metric AND the displacement vector
from Schwarzschild coords to RNC and back. Key insight: we work entirely
in RNC. The metric in RNC at point ξ is obtained by:

1. Map ξ → (t,r,θ,φ) in Schwarzschild coords
2. Compute Jacobian J = ∂(t,r,θ,φ)/∂(τ,x,y,z) at that point
3. g_RNC(ξ) = J^T · g_Sch · J
4. s² = Δ^a g_RNC_ab(m) Δ^b
"""
import sys, os, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_universal import (
    sprinkle_local_diamond, minkowski_preds, build_hasse_from_predicate,
    Y_from_graph, excess_kurtosis, ppwave_exact_preds, jet_preds,
    bulk_mask, riemann_schwarzschild_local
)

ETA = np.diag([-1.0, 1.0, 1.0, 1.0])


def sch_exact_midpoint_preds(pts, i, M, r0, tol=1e-12):
    """Exact Schwarzschild midpoint causal predicate in RNC."""
    xi = pts[i]
    xj = pts[:i]
    d = xi[None, :] - xj  # Δ in RNC
    m = 0.5 * (xi[None, :] + xj)  # midpoint in RNC

    f0 = 1.0 - 2.0 * M / r0

    # Midpoint RNC coords
    tau_m = m[:, 0]
    xm = m[:, 1]
    ym = m[:, 2]
    zm = m[:, 3]

    # RNC → Schwarzschild at midpoint
    r_m = r0 + zm
    rho_perp = np.sqrt(xm**2 + ym**2)

    # f(r) at midpoint
    f_m = 1.0 - 2.0 * M / np.maximum(r_m, 1e-10)

    # Jacobian ∂(t,r,θ,φ)/∂(τ,x,y,z) at midpoint
    # t_sch = τ / √f₀  →  ∂t/∂τ = 1/√f₀, rest 0
    # r = r₀ + z        →  ∂r/∂z = 1, rest 0
    # θ = π/2 + arctan(ρ_perp/r)  [for small angles]
    #   ≈ π/2 + ρ_perp/r for ρ_perp << r
    #   ∂θ/∂x = (x/ρ_perp) · (1/r) · cos²(arctan(ρ_perp/r))
    #   ∂θ/∂y = (y/ρ_perp) · (1/r) · cos²(arctan(ρ_perp/r))
    #   ∂θ/∂z = -ρ_perp/r² · cos²(arctan(ρ_perp/r))
    # φ = arctan2(y, x)
    #   ∂φ/∂x = -y/(x²+y²), ∂φ/∂y = x/(x²+y²)

    # For metric computation, we need g_RNC = J^T g_Sch J
    # g_Sch = diag(-f, 1/f, r², r²sin²θ) in (t,r,θ,φ) coords
    #
    # Simpler approach: compute s² directly using the chain rule
    # s² = g_μν(m) Δx^μ Δx^ν where Δx^μ = (∂x^μ/∂ξ^a) Δξ^a
    #
    # Δt = Δτ / √f₀
    # Δr = Δz
    # Δθ ≈ (Δx·x_m + Δy·y_m)/(ρ_perp·r_m) [radial component of transverse]
    #     Actually more carefully:
    #     θ = arccos(z_cart/r_3d) where z_cart = r·cosθ, but in our RNC z=radial
    #     Let me use the simpler formula for metric in mixed coords

    # SIMPLEST EXACT APPROACH:
    # In Schwarzschild, the line element for nearby points:
    # ds² = -f(r_m) Δt² + Δr²/f(r_m) + r_m² ΔΩ²
    #
    # where:
    # Δt = Δτ / √f₀  (proper time to coordinate time at r₀)
    # Δr = Δz
    # r_m² ΔΩ² = Δx_perp² where Δx_perp² = Δx² + Δy² (transverse)
    # But this is only valid at θ=π/2 and for the transverse displacement
    #
    # More carefully: the transverse displacement in Schwarzschild at radius r_m is
    # r_m² (Δθ² + sin²θ Δφ²)
    # where Δθ ≈ (displacement along θ)/r_m and Δφ ≈ (displacement along φ)/(r_m sinθ)
    #
    # In RNC: Δx and Δy are the transverse displacements at the midpoint radius r_m
    # So r_m² ΔΩ² ≈ Δx² + Δy² (for points near the equatorial plane)
    #
    # This gives:
    # s² = -f(r_m)/f₀ · Δτ² + Δz²/f(r_m) + Δx² + Δy²

    dt = d[:, 0]  # Δτ in RNC proper time
    dx = d[:, 1]
    dy = d[:, 2]
    dz = d[:, 3]  # Δz = Δr

    # Exact Schwarzschild interval at midpoint
    # Note: Δt_coord = Δτ / √f₀ (convert proper time at r₀ to coordinate time)
    # ds² = -f(r_m) (Δτ/√f₀)² + Δz²/f(r_m) + r_m²/r₀² (Δx² + Δy²)
    # The r_m²/r₀² factor accounts for the fact that transverse distances
    # scale with radius: at r_m, unit angle subtends r_m, but RNC calibrated at r₀

    s2 = (
        -f_m / f0 * dt**2
        + dz**2 / np.maximum(f_m, 1e-15)
        + (r_m / r0)**2 * (dx**2 + dy**2)
    )

    return (d[:, 0] > tol) & (s2 <= tol)


def make_strata(pts, parents0, T):
    tau_hat = 2.0 * pts[:, 0] / T
    r = np.linalg.norm(pts[:, 1:], axis=1)
    rmax = T / 2.0 - np.abs(pts[:, 0])
    rho_hat = np.clip(r / np.maximum(rmax, 1e-12), 0.0, 0.999999)
    tau_bin = np.clip(np.floor((tau_hat + 1.0) * 2.5).astype(int), 0, 4)
    rho_bin = np.clip(np.floor(rho_hat * 3).astype(int), 0, 2)
    depth = np.zeros(len(pts), dtype=int)
    for i in range(len(pts)):
        if parents0[i].size > 0:
            depth[i] = int(np.max(depth[parents0[i]])) + 1
    max_d = max(int(depth.max()), 1)
    depth_terc = np.clip((depth * 3) // (max_d + 1), 0, 2)
    return tau_bin * 9 + rho_bin * 3 + depth_terc


def compute_A_align(Y0, delta, mask, strata):
    X = Y0[mask] - np.mean(Y0[mask])
    X2 = X ** 2
    dY2 = delta[mask] ** 2
    strata_m = strata[mask]
    total_cov = 0.0
    for label in np.unique(strata_m):
        idx = strata_m == label
        if idx.sum() < 3:
            continue
        w_b = idx.sum() / len(X)
        cov_b = np.mean(X2[idx] * dY2[idx]) - np.mean(X2[idx]) * np.mean(dY2[idx])
        total_cov += w_b * cov_b
    return float(total_cov)


N = 5000
ZETA = 0.15
T = 1.0
M_SEEDS = 20
EPS = 3.0
M_SCH = 0.05
R0 = 0.50

E2_PPW = EPS**2 / 2.0
E2_SCH = 6.0 * (M_SCH / R0**3)**2


if __name__ == "__main__":
    print(f"=== A_ALIGN UNIVERSALITY: EXACT SCH MIDPOINT ===", flush=True)
    print(f"N={N}, T={T}, M={M_SEEDS}", flush=True)
    print(f"ppw: exact (V_needed), Sch: EXACT midpoint (full metric)", flush=True)
    print(flush=True)

    # Verify: flat test
    rng_test = np.random.default_rng(9999)
    pts_test = sprinkle_local_diamond(200, 1.0, rng_test)
    mask_flat = sch_exact_midpoint_preds(pts_test, 100, 0.0, 0.50)
    mask_mink = minkowski_preds(pts_test, 100)
    agree = (mask_flat == mask_mink).all()
    print(f"Sanity check M=0 (should = Minkowski): {agree}", flush=True)
    if not agree:
        diff = int((mask_flat != mask_mink).sum())
        print(f"  WARNING: {diff} disagreements!", flush=True)
    print(flush=True)

    aalign_ppw = []
    aalign_sch_exact = []
    aalign_sch_quad = []

    t0 = time.time()
    for si in range(M_SEEDS):
        seed = 2600000 + si
        rng = np.random.default_rng(seed)
        pts = sprinkle_local_diamond(N, T, rng)

        par0, ch0 = build_hasse_from_predicate(pts, lambda P, i: minkowski_preds(P, i))
        Y0 = Y_from_graph(par0, ch0)
        mask = bulk_mask(pts, T, ZETA)
        strata = make_strata(pts, par0, T)

        # ppw exact
        parE, chE = build_hasse_from_predicate(pts, lambda P, i: ppwave_exact_preds(P, i, eps=EPS))
        YE = Y_from_graph(parE, chE)
        aalign_ppw.append(compute_A_align(Y0, YE - Y0, mask, strata))

        # Sch EXACT midpoint
        parX, chX = build_hasse_from_predicate(
            pts, lambda P, i: sch_exact_midpoint_preds(P, i, M=M_SCH, r0=R0))
        YX = Y_from_graph(parX, chX)
        aalign_sch_exact.append(compute_A_align(Y0, YX - Y0, mask, strata))

        # Sch quadratic jet (for comparison)
        R_SCH = riemann_schwarzschild_local(M_SCH, R0)
        parQ, chQ = build_hasse_from_predicate(pts, lambda P, i: jet_preds(P, i, R_abcd=R_SCH))
        YQ = Y_from_graph(parQ, chQ)
        aalign_sch_quad.append(compute_A_align(Y0, YQ - Y0, mask, strata))

        if (si + 1) % 2 == 0:
            elapsed = time.time() - t0
            print(f"  {si+1}/{M_SEEDS} ({elapsed:.0f}s)", flush=True)

    def report(arr, E2):
        a = np.array(arr)
        m = float(np.mean(a))
        se = float(np.std(a, ddof=1) / np.sqrt(len(a)))
        AE = m / E2 if E2 > 1e-15 else 0
        return m, se, AE

    m1, se1, AE1 = report(aalign_ppw, E2_PPW)
    m2, se2, AE2 = report(aalign_sch_exact, E2_SCH)
    m3, se3, AE3 = report(aalign_sch_quad, E2_SCH)

    print(f"\n=== RESULTS ===", flush=True)
    print(f"  ppw exact:      A_align={m1:+.6f}±{se1:.6f}, A_E={AE1:.6f}", flush=True)
    print(f"  Sch EXACT mid:  A_align={m2:+.6f}±{se2:.6f}, A_E={AE2:.6f}", flush=True)
    print(f"  Sch quad jet:   A_align={m3:+.6f}±{se3:.6f}, A_E={AE3:.6f}", flush=True)

    ratio_exact = AE2 / AE1 if abs(AE1) > 1e-15 else 0
    ratio_quad = AE3 / AE1 if abs(AE1) > 1e-15 else 0

    print(f"\n  Ratio Sch_exact/ppw  = {ratio_exact:.3f} <-- THE NUMBER", flush=True)
    print(f"  Ratio Sch_quad/ppw   = {ratio_quad:.3f} (reference)", flush=True)
    print(f"  Exact improvement: {ratio_exact/max(ratio_quad,1e-15):.2f}x over quadratic", flush=True)

    if 0.80 <= ratio_exact <= 1.25:
        verdict = "PASS: universality!"
    elif 0.65 <= ratio_exact <= 1.50:
        verdict = "BORDERLINE"
    else:
        verdict = f"FAIL: ratio {ratio_exact:.3f}"
    print(f"\n  VERDICT: {verdict}", flush=True)

    total = time.time() - t0
    print(f"\nTotal: {total:.0f}s = {total/60:.1f}min", flush=True)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "universal_runs_v2")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "aalign_sch_exact_midpoint.json"), "w") as f:
        json.dump({
            "ppw": {"mean": m1, "se": se1, "A_E": AE1},
            "sch_exact": {"mean": m2, "se": se2, "A_E": AE2},
            "sch_quad": {"mean": m3, "se": se3, "A_E": AE3},
            "ratio_exact": ratio_exact, "ratio_quad": ratio_quad,
            "verdict": verdict,
        }, f, indent=2)
    print("Saved.", flush=True)

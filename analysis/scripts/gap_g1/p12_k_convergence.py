"""
P12 K-convergence test: vary number of Gauss-Legendre quadrature points K
for the P(alpha) kernel at FIXED R_max=100, M=1, Lam=1, eps=0.02.

Key question: is K=6 sufficient, or does the solution change at K=12?
Also checks integral sum(w*P) = 13/120 for each K.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# Import P12 solver components
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p12_full_formfactor_soft_bvp import (
    alpha_tau_quadrature_unique,
    continue_soft,
)

OUTDIR = Path(r"F:\Black Mesa Research Facility\Main Facility\Physics department\SCT Theory\analysis\results\gap_g1")

# Fixed parameters
R_MAX = 100.0
M = 1.0
LAM = 1.0
EPS = 0.02
L = 2.0
TOL = 1e-2
MAX_NODES = 30000

# Use continuation from R=20 to R=100
R_VALUES = [20.0, 40.0, 60.0, 80.0, 100.0]

K_VALUES = [2, 4, 6, 8, 10, 12]

TARGET_INTEGRAL = 13.0 / 120.0  # = 0.10833...


def check_quadrature_integral(K: int) -> dict:
    """Check that sum(w * P) = 13/120 for given K."""
    coeffs = alpha_tau_quadrature_unique(K)
    integral = float(np.sum(coeffs['c']))  # c = w * P
    error = abs(integral - TARGET_INTEGRAL)
    rel_error = error / abs(TARGET_INTEGRAL)
    return {
        'K': K,
        'integral': integral,
        'target': TARGET_INTEGRAL,
        'abs_error': error,
        'rel_error': rel_error,
        'tau_values': [float(t) for t in coeffs['tau']],
        'weights': [float(w) for w in coeffs['w']],
        'P_values': [float(p) for p in coeffs['P']],
    }


def run_single_K(K: int) -> dict:
    """Run P12 BVP solver at given K, return the R=100 record."""
    print(f"\n{'='*60}")
    print(f"  K = {K}  (Gauss-Legendre quadrature points)")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    try:
        coeffs, records = continue_soft(
            R_VALUES,
            eps=EPS,
            M=M,
            K=K,
            Lam=LAM,
            l=L,
            tol=TOL,
            max_nodes=MAX_NODES,
        )
        elapsed = time.perf_counter() - t0

        # Find the R=100 record
        rec_100 = None
        for r in records:
            if abs(r.Rmax - R_MAX) < 0.1:
                rec_100 = r
                break

        if rec_100 is None:
            return {'K': K, 'status': -1, 'message': 'R=100 not reached', 'elapsed': elapsed}

        result = {
            'K': K,
            'status': int(rec_100.status),
            'message': rec_100.message,
            'h2': rec_100.h2,
            'f3': rec_100.f3,
            'Hmin': rec_100.Hmin,
            'Hmax': rec_100.Hmax,
            'Fmin': rec_100.Fmin,
            'Jabs_max': rec_100.Jabs_max,
            'center_u_slopes': rec_100.center_u_slopes,
            'Fp_outer': rec_100.Fp_outer,
            'Hp_outer': rec_100.Hp_outer,
            'elapsed': elapsed,
            # All R-steps for this K
            'all_steps': [
                {'Rmax': r.Rmax, 'status': r.status, 'h2': r.h2, 'Hmin': r.Hmin}
                for r in records
            ],
        }
        return result

    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"  FAILED: {e}")
        return {'K': K, 'status': -2, 'message': str(e), 'elapsed': elapsed}


def main():
    print("P12 K-convergence study")
    print(f"R_max={R_MAX}, M={M}, Lam={LAM}, eps={EPS}")
    print(f"K values: {K_VALUES}")
    print()

    # Step 1: Check quadrature integrals
    print("=" * 60)
    print("  QUADRATURE INTEGRAL CHECK: sum(w*P) = 13/120?")
    print("=" * 60)
    quad_checks = []
    for K in K_VALUES:
        qc = check_quadrature_integral(K)
        quad_checks.append(qc)
        status = "PASS" if qc['rel_error'] < 1e-10 else "FAIL"
        print(f"  K={K:3d}: integral = {qc['integral']:.15e}, "
              f"target = {qc['target']:.15e}, "
              f"rel_err = {qc['rel_error']:.2e}  [{status}]")

    # Step 2: Run BVP for each K
    results = []
    for K in K_VALUES:
        res = run_single_K(K)
        results.append(res)

    # Step 3: Summary table
    print("\n" + "=" * 80)
    print("  K-CONVERGENCE SUMMARY (R_max = 100)")
    print("=" * 80)
    print(f"{'K':>4s}  {'status':>7s}  {'h2':>14s}  {'Hmin':>14s}  {'Jabs_max':>14s}  {'time(s)':>8s}")
    print("-" * 80)

    converged_h2 = []
    converged_K = []
    converged_Hmin = []

    for r in results:
        if r['status'] == 0 and r['h2'] is not None:
            print(f"{r['K']:4d}  {'OK':>7s}  {r['h2']:14.10f}  {r['Hmin']:14.10f}  "
                  f"{r['Jabs_max']:14.6e}  {r['elapsed']:8.2f}")
            converged_h2.append(r['h2'])
            converged_K.append(r['K'])
            converged_Hmin.append(r['Hmin'])
        else:
            msg = r.get('message', 'unknown')[:30]
            print(f"{r['K']:4d}  {'FAIL':>7s}  {'---':>14s}  {'---':>14s}  "
                  f"{'---':>14s}  {r.get('elapsed', 0):8.2f}  {msg}")

    # Step 4: Convergence analysis
    if len(converged_h2) >= 2:
        print("\n  CONVERGENCE ANALYSIS:")
        h2_arr = np.array(converged_h2)
        K_arr = np.array(converged_K)

        # Differences from the highest-K result
        h2_ref = h2_arr[-1]
        Hmin_ref = np.array(converged_Hmin)[-1]

        for i, (K, h2, Hmin) in enumerate(zip(converged_K, converged_h2, converged_Hmin)):
            dh2 = abs(h2 - h2_ref)
            dHmin = abs(Hmin - Hmin_ref)
            print(f"    K={K:3d}: |h2 - h2(K_max)| = {dh2:.2e},  "
                  f"|Hmin - Hmin(K_max)| = {dHmin:.2e}")

        # Is K=6 sufficient?
        k6_idx = None
        for i, K in enumerate(converged_K):
            if K == 6:
                k6_idx = i
                break

        if k6_idx is not None:
            dh2_6 = abs(converged_h2[k6_idx] - h2_ref)
            rel_dh2_6 = dh2_6 / abs(h2_ref) if h2_ref != 0 else float('inf')
            print(f"\n  K=6 vs K={converged_K[-1]}: |delta h2| = {dh2_6:.2e}, "
                  f"relative = {rel_dh2_6:.2e}")
            if rel_dh2_6 < 1e-4:
                print("  VERDICT: K=6 is SUFFICIENT (relative change < 1e-4)")
            elif rel_dh2_6 < 1e-2:
                print("  VERDICT: K=6 is MARGINALLY sufficient (relative change < 1e-2)")
            else:
                print(f"  VERDICT: K=6 is INSUFFICIENT (relative change = {rel_dh2_6:.2e})")

    # Step 5: Save results
    output = {
        'parameters': {
            'R_max': R_MAX,
            'M': M,
            'Lam': LAM,
            'eps': EPS,
            'l': L,
            'tol': TOL,
            'max_nodes': MAX_NODES,
            'R_values': R_VALUES,
            'K_values': K_VALUES,
        },
        'quadrature_checks': quad_checks,
        'results': results,
        'convergence': {
            'converged_K': converged_K,
            'converged_h2': converged_h2,
            'converged_Hmin': converged_Hmin,
            'h2_ref': float(h2_ref) if len(converged_h2) >= 2 else None,
        } if len(converged_h2) >= 2 else None,
    }

    outpath = OUTDIR / 'v3_k_convergence.json'
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\n  Saved: {outpath}")

    # Step 6: Plot
    if len(converged_h2) >= 2:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # h2 vs K
            ax = axes[0]
            ax.plot(converged_K, converged_h2, 'o-', color='#1f77b4', markersize=8)
            ax.set_xlabel('K (Gauss-Legendre points)', fontsize=12)
            ax.set_ylabel('h2', fontsize=12)
            ax.set_title('h2 vs K (R_max=100)', fontsize=13)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(converged_K)

            # Hmin vs K
            ax = axes[1]
            ax.plot(converged_K, converged_Hmin, 's-', color='#d62728', markersize=8)
            ax.set_xlabel('K (Gauss-Legendre points)', fontsize=12)
            ax.set_ylabel('H_min', fontsize=12)
            ax.set_title('H_min vs K (R_max=100)', fontsize=13)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(converged_K)

            fig.suptitle('P12 K-convergence: P(alpha) quadrature layers', fontsize=14, y=1.02)
            plt.tight_layout()

            figpath = OUTDIR.parent.parent / 'figures' / 'gap_g1' / 'p12_k_convergence.png'
            figpath.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(figpath, dpi=160, bbox_inches='tight')
            print(f"  Figure: {figpath}")
            plt.close(fig)
        except Exception as e:
            print(f"  Plot failed: {e}")


if __name__ == '__main__':
    main()

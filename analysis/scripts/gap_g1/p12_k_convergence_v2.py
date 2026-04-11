"""
P12 K-convergence v2: More careful study with finer R-stepping
and tighter tolerance to avoid BVP solver bifurcation artifacts.

The v1 run showed h2 jumping wildly between K values, suggesting
the continuation chain is landing on different solution branches.

Fix: use finer R steps for more stable continuation.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

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
TOL = 1e-3      # Tighter than v1's 1e-2
MAX_NODES = 40000

# Finer R-stepping for stable continuation
R_VALUES = [10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

K_VALUES = [2, 4, 6, 8, 10, 12]

TARGET_INTEGRAL = 13.0 / 120.0


def run_single_K(K: int) -> dict:
    """Run P12 BVP solver at given K with fine R-stepping."""
    print(f"\n{'='*60}")
    print(f"  K = {K}")
    print(f"{'='*60}")

    # Check quadrature integral
    coeffs = alpha_tau_quadrature_unique(K)
    integral = float(np.sum(coeffs['c']))
    quad_error = abs(integral - TARGET_INTEGRAL) / abs(TARGET_INTEGRAL)
    print(f"  Quadrature integral = {integral:.15e} (target {TARGET_INTEGRAL:.15e})")
    print(f"  Relative error = {quad_error:.2e}")

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

        # Extract all successful records
        all_steps = []
        for r in records:
            step = {
                'Rmax': r.Rmax,
                'status': r.status,
                'h2': r.h2,
                'f3': r.f3,
                'Hmin': r.Hmin,
                'Jabs_max': r.Jabs_max,
            }
            all_steps.append(step)

        # Find R=100 record
        rec_100 = None
        for r in records:
            if abs(r.Rmax - R_MAX) < 0.1:
                rec_100 = r
                break

        if rec_100 is None or rec_100.status != 0:
            # Find the last successful record
            last_ok = None
            for r in records:
                if r.status == 0:
                    last_ok = r
            if last_ok is not None:
                return {
                    'K': K,
                    'status': -1,
                    'message': f'R=100 failed, last OK at R={last_ok.Rmax}',
                    'last_ok_Rmax': last_ok.Rmax,
                    'last_ok_h2': last_ok.h2,
                    'last_ok_Hmin': last_ok.Hmin,
                    'elapsed': elapsed,
                    'all_steps': all_steps,
                    'quad_integral': integral,
                    'quad_rel_error': quad_error,
                }
            return {
                'K': K, 'status': -1, 'message': 'no convergence',
                'elapsed': elapsed, 'all_steps': all_steps,
                'quad_integral': integral, 'quad_rel_error': quad_error,
            }

        return {
            'K': K,
            'status': 0,
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
            'all_steps': all_steps,
            'quad_integral': integral,
            'quad_rel_error': quad_error,
        }

    except Exception as e:
        elapsed = time.perf_counter() - t0
        import traceback
        traceback.print_exc()
        return {
            'K': K, 'status': -2, 'message': str(e),
            'elapsed': elapsed, 'all_steps': [],
            'quad_integral': integral, 'quad_rel_error': quad_error,
        }


def main():
    print("P12 K-convergence v2 (finer R-steps, tol=1e-3)")
    print(f"R_max={R_MAX}, M={M}, Lam={LAM}, eps={EPS}")
    print(f"R steps: {R_VALUES}")
    print(f"K values: {K_VALUES}")

    results = []
    for K in K_VALUES:
        res = run_single_K(K)
        results.append(res)

    # Summary
    print("\n" + "=" * 90)
    print("  K-CONVERGENCE SUMMARY v2 (R_max = 100, tol = 1e-3)")
    print("=" * 90)
    print(f"{'K':>4s}  {'status':>7s}  {'h2':>16s}  {'Hmin':>14s}  {'Jabs_max':>14s}  {'time':>8s}")
    print("-" * 90)

    ok_K, ok_h2, ok_Hmin, ok_Jabs = [], [], [], []

    for r in results:
        if r['status'] == 0 and r.get('h2') is not None:
            print(f"{r['K']:4d}  {'OK':>7s}  {r['h2']:16.10f}  {r['Hmin']:14.10f}  "
                  f"{r['Jabs_max']:14.6e}  {r['elapsed']:8.1f}s")
            ok_K.append(r['K'])
            ok_h2.append(r['h2'])
            ok_Hmin.append(r['Hmin'])
            ok_Jabs.append(r['Jabs_max'])
        else:
            msg = r.get('message', '')[:40]
            print(f"{r['K']:4d}  {'FAIL':>7s}  {'---':>16s}  {'---':>14s}  "
                  f"{'---':>14s}  {r.get('elapsed', 0):8.1f}s  {msg}")

    # Convergence analysis
    convergence = None
    if len(ok_h2) >= 2:
        h2_arr = np.array(ok_h2)
        h2_ref = h2_arr[-1]
        Hmin_arr = np.array(ok_Hmin)
        Hmin_ref = Hmin_arr[-1]

        print("\n  h2 differences from K_max:")
        for K, h2, Hmin in zip(ok_K, ok_h2, ok_Hmin):
            dh2 = abs(h2 - h2_ref)
            dHmin = abs(Hmin - Hmin_ref)
            print(f"    K={K:3d}: dh2 = {dh2:.6e}, dHmin = {dHmin:.6e}")

        # Check h2 variation range
        h2_range = max(ok_h2) - min(ok_h2)
        h2_mean = np.mean(ok_h2)
        print(f"\n  h2 range = {h2_range:.6e}")
        print(f"  h2 mean  = {h2_mean:.6e}")
        print(f"  h2 CV    = {np.std(ok_h2)/abs(h2_mean):.4f}")

        # Check if the solution is stable (h2 within some % across K>=4)
        k4plus = [(K, h2) for K, h2 in zip(ok_K, ok_h2) if K >= 4]
        if len(k4plus) >= 2:
            h2_k4p = [x[1] for x in k4plus]
            h2_k4p_range = max(h2_k4p) - min(h2_k4p)
            h2_k4p_mean = np.mean(h2_k4p)
            cv_k4p = np.std(h2_k4p) / abs(h2_k4p_mean) if h2_k4p_mean != 0 else float('inf')
            print(f"\n  For K >= 4:")
            print(f"    h2 range = {h2_k4p_range:.6e}")
            print(f"    h2 mean  = {h2_k4p_mean:.6e}")
            print(f"    h2 CV    = {cv_k4p:.4f}")

        convergence = {
            'ok_K': ok_K,
            'ok_h2': ok_h2,
            'ok_Hmin': ok_Hmin,
            'ok_Jabs': ok_Jabs,
            'h2_range': float(h2_range),
            'h2_mean': float(h2_mean),
            'h2_cv': float(np.std(ok_h2) / abs(h2_mean)) if h2_mean != 0 else None,
        }

    # Save
    output = {
        'parameters': {
            'R_max': R_MAX, 'M': M, 'Lam': LAM, 'eps': EPS, 'l': L,
            'tol': TOL, 'max_nodes': MAX_NODES,
            'R_values': R_VALUES, 'K_values': K_VALUES,
        },
        'results': results,
        'convergence': convergence,
    }

    outpath = OUTDIR / 'v3_k_convergence.json'
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\n  Saved: {outpath}")

    # Plot
    if len(ok_h2) >= 2:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(16, 5))

            ax = axes[0]
            ax.plot(ok_K, ok_h2, 'o-', color='#1f77b4', markersize=8, linewidth=1.5)
            ax.set_xlabel('K (quadrature points)', fontsize=12)
            ax.set_ylabel('h2', fontsize=12)
            ax.set_title('h2 vs K', fontsize=13)
            ax.grid(True, alpha=0.3)

            ax = axes[1]
            ax.plot(ok_K, ok_Hmin, 's-', color='#d62728', markersize=8, linewidth=1.5)
            ax.set_xlabel('K (quadrature points)', fontsize=12)
            ax.set_ylabel('H_min', fontsize=12)
            ax.set_title('H_min vs K', fontsize=13)
            ax.grid(True, alpha=0.3)

            ax = axes[2]
            ax.plot(ok_K, ok_Jabs, '^-', color='#2ca02c', markersize=8, linewidth=1.5)
            ax.set_xlabel('K (quadrature points)', fontsize=12)
            ax.set_ylabel('|J|_max', fontsize=12)
            ax.set_title('max|J| vs K', fontsize=13)
            ax.grid(True, alpha=0.3)

            fig.suptitle('P12 K-convergence (R_max=100, tol=1e-3)', fontsize=14, y=1.02)
            plt.tight_layout()

            figpath = OUTDIR.parent.parent / 'figures' / 'gap_g1' / 'p12_k_convergence.png'
            figpath.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(figpath, dpi=160, bbox_inches='tight')
            print(f"  Figure: {figpath}")
            plt.close(fig)
        except Exception as e:
            print(f"  Plot failed: {e}")

    # Also plot h2 vs R for ALL K on one figure (diagnostic)
    if len(results) >= 2:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

            for i, r in enumerate(results):
                steps = r.get('all_steps', [])
                R_ok = [s['Rmax'] for s in steps if s['status'] == 0 and s['h2'] is not None]
                h2_ok = [s['h2'] for s in steps if s['status'] == 0 and s['h2'] is not None]
                if R_ok:
                    ax.plot(R_ok, h2_ok, 'o-', color=colors[i % len(colors)],
                            label=f'K={r["K"]}', markersize=5, linewidth=1.2)

            ax.set_xlabel('R_max', fontsize=12)
            ax.set_ylabel('h2', fontsize=12)
            ax.set_title('h2 vs R_max for different K', fontsize=13)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            figpath2 = OUTDIR.parent.parent / 'figures' / 'gap_g1' / 'p12_k_convergence_h2_vs_R.png'
            fig.savefig(figpath2, dpi=160, bbox_inches='tight')
            print(f"  Figure: {figpath2}")
            plt.close(fig)
        except Exception as e:
            print(f"  h2-vs-R plot failed: {e}")


if __name__ == '__main__':
    main()

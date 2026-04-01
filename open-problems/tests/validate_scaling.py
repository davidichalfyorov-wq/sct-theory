#!/usr/bin/env python3
"""
Validate a proposed N-scaling exponent for the CJ observable.

Usage:
    python validate_scaling.py <proposed_alpha>
    python validate_scaling.py 0.889       # test 8/9
    python validate_scaling.py 0.955       # test fitted value
    python validate_scaling.py 1.0         # test linear

Reports the RMS residual and chi^2 for the proposed exponent.
"""
import sys
import numpy as np
from scipy.optimize import curve_fit

N_DATA = np.array([500, 1000, 2000, 3000, 5000, 8000, 10000, 15000], dtype=float)
CJ_DATA = np.array([0.00759, 0.01379, 0.02345, 0.03511, 0.05673, 0.08650, 0.11254, 0.15495])


def test_exponent(alpha):
    """Test a specific N-scaling exponent."""
    # Fit only the prefactor C for the given alpha
    def model(N, C):
        return C * N**alpha

    popt, pcov = curve_fit(model, N_DATA, CJ_DATA)
    C_fit = popt[0]
    C_err = np.sqrt(pcov[0, 0])

    predictions = model(N_DATA, C_fit)
    residuals = (CJ_DATA - predictions) / CJ_DATA
    rms_pct = np.sqrt(np.mean(residuals**2)) * 100

    # Compare with free-fit
    def free_model(N, C, a):
        return C * N**a

    popt_free, _ = curve_fit(free_model, N_DATA, CJ_DATA, p0=[1e-5, 0.9])
    alpha_free = popt_free[1]
    pred_free = free_model(N_DATA, *popt_free)
    rms_free = np.sqrt(np.mean(((CJ_DATA - pred_free) / CJ_DATA)**2)) * 100

    print(f"Proposed exponent: alpha = {alpha:.6f}")
    print(f"Best-fit prefactor: C = {C_fit:.6e} +/- {C_err:.6e}")
    print()
    print(f"{'N':>6}  {'CJ_data':>10}  {'CJ_pred':>10}  {'resid%':>7}")
    print("-" * 40)
    for n, cj, pred, r in zip(N_DATA, CJ_DATA, predictions, residuals):
        print(f"{int(n):>6}  {cj:>10.5f}  {pred:>10.5f}  {r*100:>+7.2f}%")

    print()
    print(f"RMS residual (proposed alpha={alpha:.4f}):  {rms_pct:.2f}%")
    print(f"RMS residual (free alpha={alpha_free:.4f}):  {rms_free:.2f}%")
    print(f"Free-fit alpha: {alpha_free:.4f}")
    print(f"Difference: |alpha_proposed - alpha_free| = {abs(alpha - alpha_free):.4f}")
    print()

    if rms_pct < rms_free + 0.5:
        print(f"PASS: proposed exponent has competitive RMS ({rms_pct:.2f}% vs {rms_free:.2f}%)")
    elif rms_pct < 8:
        print(f"MARGINAL: proposed exponent acceptable but not optimal ({rms_pct:.2f}% vs {rms_free:.2f}%)")
    else:
        print(f"FAIL: proposed exponent gives poor fit ({rms_pct:.2f}% >> {rms_free:.2f}%)")


def main():
    if len(sys.argv) < 2:
        print("Testing three candidate exponents:")
        print()
        for alpha, label in [(8/9, "8/9 (theoretical)"),
                              (0.955, "0.955 (power-law fit)"),
                              (1.0, "1.0 (linear)")]:
            print(f"{'='*50}")
            print(f"  {label}")
            print(f"{'='*50}")
            test_exponent(alpha)
            print()
        return

    alpha = float(sys.argv[1])
    test_exponent(alpha)


if __name__ == "__main__":
    main()

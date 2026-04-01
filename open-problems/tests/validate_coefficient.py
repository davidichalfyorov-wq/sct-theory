#!/usr/bin/env python3
"""
Validate a proposed CJ bridge formula coefficient against data.

Usage:
    python validate_coefficient.py <proposed_C0> [--E2 <value>] [--alpha <value>]

The script checks the proposed coefficient C0 against 8 measured CJ
values at N = 500 to 15000 (pp-wave, eps=3, T=1).

A valid coefficient should give residuals < 10% at all N values and
mean ratio within [0.90, 1.10].
"""
import sys
import numpy as np

# Verified data: CJ at eps=3, T=1, exact pp-wave predicate
N_DATA = np.array([500, 1000, 2000, 3000, 5000, 8000, 10000, 15000], dtype=float)
CJ_DATA = np.array([0.00759, 0.01379, 0.02345, 0.03511, 0.05673, 0.08650, 0.11254, 0.15495])
EPS = 3.0


def validate(C0, E2=None, alpha=8.0/9.0):
    """Check proposed coefficient against data."""
    if E2 is None:
        E2 = EPS**2 / 2.0  # geometric E^2 = eps^2/2 for pp-wave with H=(eps/2)(x^2-y^2)

    T = 1.0
    predictions = C0 * N_DATA**alpha * E2 * T**4
    ratios = CJ_DATA / predictions

    mean_ratio = np.mean(ratios)
    std_ratio = np.std(ratios, ddof=1)
    rms_pct = np.sqrt(np.mean((ratios - 1)**2)) * 100
    max_dev = np.max(np.abs(ratios - 1)) * 100

    print(f"Proposed C0 = {C0:.6e}")
    print(f"E^2 = {E2:.4f}, alpha = {alpha:.6f}")
    print()
    print(f"{'N':>6}  {'CJ_data':>10}  {'CJ_pred':>10}  {'ratio':>7}  {'dev%':>6}")
    print("-" * 50)
    for n, cj, pred, r in zip(N_DATA, CJ_DATA, predictions, ratios):
        dev = (r - 1) * 100
        flag = " ***" if abs(dev) > 10 else ""
        print(f"{int(n):>6}  {cj:>10.5f}  {pred:>10.5f}  {r:>7.3f}  {dev:>+6.1f}%{flag}")

    print()
    print(f"Mean ratio:     {mean_ratio:.4f}")
    print(f"Std ratio:      {std_ratio:.4f}")
    print(f"RMS deviation:  {rms_pct:.1f}%")
    print(f"Max deviation:  {max_dev:.1f}%")
    print()

    # Verdict
    passed = True
    if abs(mean_ratio - 1.0) > 0.10:
        print(f"FAIL: mean ratio {mean_ratio:.3f} outside [0.90, 1.10]")
        passed = False
    if max_dev > 15:
        print(f"FAIL: max deviation {max_dev:.1f}% exceeds 15%")
        passed = False
    if rms_pct > 8:
        print(f"WARNING: RMS {rms_pct:.1f}% exceeds 8% (noisy but not necessarily wrong)")

    if passed:
        print("PASS: coefficient is consistent with data")
    return passed


def main():
    if len(sys.argv) < 2:
        # Default: test the analytical coefficient
        C_an = 32 * np.pi**2 / (3 * 362880 * 45)
        print("Testing analytical coefficient C = 32*pi^2/(3*9!*45):")
        print()
        validate(C_an)
        return

    C0 = float(sys.argv[1])
    E2 = None
    alpha = 8.0/9.0

    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--E2":
            E2 = float(sys.argv[i+1])
            i += 2
        elif sys.argv[i] == "--alpha":
            alpha = float(sys.argv[i+1])
            i += 2
        else:
            i += 1

    validate(C0, E2=E2, alpha=alpha)


if __name__ == "__main__":
    main()

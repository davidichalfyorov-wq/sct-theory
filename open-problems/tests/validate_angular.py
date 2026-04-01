#!/usr/bin/env python3
"""
Validate a proposed angular integral on S^2.

Usage:
    python validate_angular.py

Checks the standard result:
  int_{S^2} (E_ij n^i n^j)^2 dOmega = (8*pi/15) * E_ij E^ij

for several test tensors E_ij, by numerical quadrature on S^2.
"""
import numpy as np


def angular_integral_numerical(E, n_theta=200, n_phi=400):
    """Compute int_{S^2} (E_ij n^i n^j)^2 dOmega by quadrature."""
    theta = np.linspace(0, np.pi, n_theta + 1)
    phi = np.linspace(0, 2 * np.pi, n_phi + 1)

    total = 0.0
    for i in range(n_theta):
        t = (theta[i] + theta[i + 1]) / 2
        dt = theta[i + 1] - theta[i]
        st = np.sin(t)
        ct = np.cos(t)
        for j in range(n_phi):
            p = (phi[j] + phi[j + 1]) / 2
            dp = phi[j + 1] - phi[j]
            n = np.array([st * np.cos(p), st * np.sin(p), ct])
            val = np.dot(n, E @ n)
            total += val**2 * st * dt * dp
    return total


def main():
    print("Angular integral validation: int_{S^2} (E_ij n^i n^j)^2 dOmega")
    print("Prediction: (8*pi/15) * E_ij E^ij")
    print()

    test_cases = [
        ("Plus polarisation (pp-wave)",
         np.diag([-1.0, 1.0, 0.0])),
        ("Cross polarisation",
         np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0.0]])),
        ("Schwarzschild-type",
         np.diag([-1.0, -1.0, 2.0])),
        ("General traceless",
         np.array([[0.3, 0.5, 0.1], [0.5, -0.7, 0.2], [0.1, 0.2, 0.4]])),
        ("Isotropic (non-traceless, should fail)",
         np.diag([1.0, 1.0, 1.0])),
    ]

    all_pass = True
    for name, E in test_cases:
        E_sq = np.sum(E * E)  # E_ij E^ij = Tr(E^2)
        predicted = (8 * np.pi / 15) * E_sq
        measured = angular_integral_numerical(E)
        ratio = measured / predicted if abs(predicted) > 1e-30 else float("nan")
        tr_E = np.trace(E)
        is_traceless = abs(tr_E) < 1e-10

        status = "PASS" if abs(ratio - 1.0) < 0.01 else "FAIL"
        if not is_traceless:
            status = "EXPECTED FAIL (not traceless)"
        if status == "FAIL" and is_traceless:
            all_pass = False

        print(f"  {name}:")
        print(f"    E_ij = {E.tolist()}")
        print(f"    tr(E) = {tr_E:.6f}  {'(traceless)' if is_traceless else '(NOT traceless)'}")
        print(f"    E^2 = {E_sq:.6f}")
        print(f"    Predicted: {predicted:.8f}")
        print(f"    Measured:  {measured:.8f}")
        print(f"    Ratio:     {ratio:.6f}")
        print(f"    {status}")
        print()

    if all_pass:
        print("ALL traceless tests PASS")
    else:
        print("SOME traceless tests FAIL")


if __name__ == "__main__":
    main()

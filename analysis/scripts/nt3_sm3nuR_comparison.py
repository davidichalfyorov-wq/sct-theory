"""
NT-3 Extension: Spectral Dimension d_S(sigma) comparison SM vs SM+3nuR.

Uses Mittag-Leffler pole decomposition (Method 4) for both models.
SM: (N_s, N_D, N_v) = (4, 22.5, 12), alpha_C = 13/120
SM+3nuR: (N_s, N_D, N_v) = (4, 24, 12), alpha_C = 1/30

Author: David Alfyorov, Igor Shnyukov
"""
import json
import sys
from pathlib import Path

import mpmath
import numpy as np

mpmath.mp.dps = 30

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "nt3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# === Master function phi(z) ===
def phi_mp(z):
    z = mpmath.mpc(z)
    if abs(z) < mpmath.mpf("0.01"):
        s = mpmath.mpf(0)
        for n in range(60):
            s += (-1)**n * mpmath.factorial(n) / mpmath.factorial(2*n+1) * z**n
        return s
    if z.imag == 0 and z.real < 0:
        t = -z.real
        return mpmath.exp(t/4) * mpmath.sqrt(mpmath.pi/t) * mpmath.erf(mpmath.sqrt(t)/2)
    return mpmath.exp(-z/4) * mpmath.sqrt(mpmath.pi/z) * mpmath.erfi(mpmath.sqrt(z)/2)


# === Pi_TT and derivatives for both models ===
def Pi_SM_mp(z):
    z = mpmath.mpc(z); p = phi_mp(z)
    return (6*z + mpmath.mpf(93)/2 + 118/z)*p - mpmath.mpf(155)/6 - 118/z

def dPi_SM_mp(z):
    z = mpmath.mpc(z); p = phi_mp(z)
    pp = (2 - (z+2)*p) / (4*z)
    return (6 - 118/z**2)*p + (6*z + mpmath.mpf(93)/2 + 118/z)*pp + 118/z**2

def Pi_ext_mp(z):
    z = mpmath.mpc(z); p = phi_mp(z)
    return (6*z + 48 + 124/z)*p - mpmath.mpf(79)/3 - 124/z

def dPi_ext_mp(z):
    z = mpmath.mpc(z); p = phi_mp(z)
    pp = (2 - (z+2)*p) / (4*z)
    return (6 - 124/z**2)*p + (6*z + 48 + 124/z)*pp + 124/z**2


def find_zeros_and_residues(Pi_func, dPi_func, z1_init, n_max=10):
    """Find positive real zero z1 + first n_max complex pairs."""
    z1 = mpmath.mpf(str(z1_init))
    for _ in range(30):
        pv = Pi_func(z1)
        dpv = dPi_func(z1)
        if abs(dpv) < 1e-40:
            break
        z1 -= pv / dpv
    z1 = float(mpmath.re(z1))

    poles = []
    # Real zero z1 (fakeon)
    dPi_z1 = dPi_func(mpmath.mpf(str(z1)))
    R1 = float(mpmath.re(1 / (mpmath.mpf(str(z1)) * dPi_z1)))
    poles.append((z1, 0.0, R1, 0.0))

    # Complex zeros
    for n in range(1, n_max + 1):
        im_g = (8*n + 3) * mpmath.pi
        re_g = 2 * mpmath.log(max(n, 1)) + mpmath.mpf("5")
        zn = mpmath.mpc(re_g, im_g)
        for _ in range(80):
            pv = Pi_func(zn)
            dpv = dPi_func(zn)
            if abs(dpv) < 1e-40:
                break
            zn -= pv / dpv
        Rn = 1 / (zn * dPi_func(zn))
        poles.append((
            float(mpmath.re(zn)), float(mpmath.im(zn)),
            float(mpmath.re(Rn)), float(mpmath.im(Rn))
        ))
    return z1, poles


def compute_P_ML(sigma, poles, Lambda=1.0):
    """Return probability from Mittag-Leffler pole decomposition."""
    L2 = Lambda**2
    P_grav = 1.0 / (16.0 * np.pi**2 * sigma**2)
    P_poles = 0.0

    for z_re, z_im, R_re, R_im in poles:
        if abs(z_im) < 1e-10:
            m2 = abs(z_re) * L2
            arg = m2 * sigma
            if arg > 500:
                continue
            P_poles += R_re * np.exp(-arg) / (16.0 * np.pi**2 * sigma**2)
        else:
            m2_re = z_re * L2
            if abs(m2_re * sigma) > 500:
                continue
            phase = z_im * L2 * sigma
            decay = np.exp(-m2_re * sigma)
            # Upper + lower half-plane contributions
            c_pair = 2 * decay * R_re * np.cos(phase)
            # Im parts cancel for conjugate pair: +R_im*sin - R_im*sin = 0
            # Wait, R_n^- = conj(R_n^+) so R_im^- = -R_im^+
            # Contribution: R_re*cos(phase) + R_im*sin(phase) + R_re*cos(phase) - R_im*sin(phase) = 2*R_re*cos(phase)
            P_poles += c_pair / (16.0 * np.pi**2 * sigma**2)

    return P_grav + P_poles


def compute_dS(sigma, poles, Lambda=1.0, delta_frac=0.01):
    """Spectral dimension d_S(sigma) = -2 sigma P'(sigma)/P(sigma)."""
    delta = sigma * delta_frac
    P_p = compute_P_ML(sigma + delta, poles, Lambda)
    P_m = compute_P_ML(sigma - delta, poles, Lambda)
    P_c = compute_P_ML(sigma, poles, Lambda)
    if abs(P_c) < 1e-300:
        return float("nan")
    dlog_P = (np.log(abs(P_p)) - np.log(abs(P_m))) / (2 * delta)
    return -2.0 * sigma * dlog_P


def main():
    print("Finding SM zeros...")
    z1_SM, poles_SM = find_zeros_and_residues(Pi_SM_mp, dPi_SM_mp, 2.4148, n_max=10)
    print(f"  z1_SM = {z1_SM:.6f}, {len(poles_SM)} poles")

    print("Finding SM+3nuR zeros...")
    z1_ext, poles_ext = find_zeros_and_residues(Pi_ext_mp, dPi_ext_mp, 2.1062, n_max=10)
    print(f"  z1_ext = {z1_ext:.6f}, {len(poles_ext)} poles")

    # Sigma scan
    sigma_values = np.logspace(-3, 2, 300)

    print("Computing spectral dimensions...")
    dS_SM = [compute_dS(s, poles_SM) for s in sigma_values]
    dS_ext = [compute_dS(s, poles_ext) for s in sigma_values]
    P_SM = [compute_P_ML(s, poles_SM) for s in sigma_values]
    P_ext = [compute_P_ML(s, poles_ext) for s in sigma_values]

    # === Results ===
    print()
    print("=" * 65)
    print("SPECTRAL DIMENSION: SM vs SM+3nuR (Method 4, Mittag-Leffler)")
    print("=" * 65)

    print()
    print("IR limit (sigma >> 1):")
    print(f"  SM:      d_S = {dS_SM[-1]:.4f}")
    print(f"  SM+3nuR: d_S = {dS_ext[-1]:.4f}")

    print()
    print("d_S at key scales:")
    print(f"  {'sigma':>8s}  {'SM':>8s}  {'SM+3nuR':>8s}  {'Delta':>8s}")
    print(f"  {'------':>8s}  {'------':>8s}  {'-------':>8s}  {'-----':>8s}")
    for s_target in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0]:
        idx = np.argmin(np.abs(sigma_values - s_target))
        s = sigma_values[idx]
        d_sm = dS_SM[idx]
        d_ext = dS_ext[idx]
        print(f"  {s:8.4f}  {d_sm:8.3f}  {d_ext:8.3f}  {d_ext - d_sm:+8.3f}")

    # P < 0 threshold
    print()
    print("P(sigma) sign analysis:")
    for label, P_arr in [("SM", P_SM), ("SM+3nuR", P_ext)]:
        transitions = []
        for i in range(1, len(sigma_values)):
            if P_arr[i] * P_arr[i-1] < 0:
                transitions.append((sigma_values[i], "neg->pos" if P_arr[i] > 0 else "pos->neg"))
        if transitions:
            for s, t in transitions:
                print(f"  {label}: {t} at sigma ~ {s:.5f}")
        else:
            sign = "positive" if P_arr[-1] > 0 else "negative"
            print(f"  {label}: P always {sign} in scanned range")

    # Residue sums
    print()
    print("Residue sums (partial, 10 poles):")
    W_SM = sum(R_re for _, _, R_re, _ in poles_SM)
    W_ext = sum(R_re for _, _, R_re, _ in poles_ext)
    print(f"  SM:      sum R_n = {W_SM:.6f}")
    print(f"  SM+3nuR: sum R_n = {W_ext:.6f}")

    # Non-monotonicity check
    print()
    print("Non-monotonicity (local extrema of d_S):")
    for label, dS_arr in [("SM", dS_SM), ("SM+3nuR", dS_ext)]:
        extrema = []
        for i in range(1, len(dS_arr) - 1):
            if not (np.isnan(dS_arr[i-1]) or np.isnan(dS_arr[i]) or np.isnan(dS_arr[i+1])):
                if (dS_arr[i] > dS_arr[i-1] and dS_arr[i] > dS_arr[i+1]):
                    extrema.append((sigma_values[i], dS_arr[i], "max"))
                elif (dS_arr[i] < dS_arr[i-1] and dS_arr[i] < dS_arr[i+1]):
                    extrema.append((sigma_values[i], dS_arr[i], "min"))
        print(f"  {label}: {len(extrema)} extrema found")
        for s, d, t in extrema[:6]:
            print(f"    sigma={s:.4f}: d_S={d:.3f} ({t})")

    # Save
    results = {
        "sigma": sigma_values.tolist(),
        "dS_SM": dS_SM,
        "dS_ext": dS_ext,
        "P_SM": P_SM,
        "P_ext": P_ext,
        "z1_SM": z1_SM,
        "z1_ext": z1_ext,
        "n_poles": len(poles_SM),
    }
    outpath = RESULTS_DIR / "nt3_sm3nuR_comparison.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()

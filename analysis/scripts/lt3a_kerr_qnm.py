# ruff: noqa: E402, I001
"""
LT-3a Phase 8: Kerr QNM analysis for Spectral Causal Theory.

Computes quasinormal mode frequencies of Kerr black holes in GR (via the
``qnm`` package) and estimates SCT corrections as a function of the
dimensionless spin parameter a/M.

SCT introduces two types of corrections to QNM frequencies:
  (A) Metric modification -- Yukawa corrections to f(r) enter through
      exp(-m2 * r_+) where r_+ = M + sqrt(M^2 - a^2) is the outer horizon.
      Since r_+ decreases with spin, the suppression is WEAKER for faster
      rotators, but still astronomically small (~10^{-10^18} even at a/M=0.998).
  (B) Perturbation-equation correction (OP-01, not computed from first
      principles) -- estimated as c2*(omega/Lambda)^2 ~ 10^{-20} for stellar BH.
      This dominates over (A) by many orders of magnitude and is independent of
      the metric modification.

CAVEAT: We substitute f_SCT into the GR perturbation formula. This captures
the metric-modification part (A) but NOT the perturbation-equation correction
(B), which requires the full nonlocal field equations (blocked by OP-01/Gap G1).

Result: For ALL astrophysical Kerr BHs, SCT is indistinguishable from GR.

Author: David Alfyorov
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import constants

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Suppress numpy deprecation warnings from qnm's pickle files
warnings.filterwarnings("ignore", category=DeprecationWarning)

import qnm

# ============================================================
# Physical constants (SI, CODATA 2022 via scipy)
# ============================================================
G_N = constants.G                        # 6.674e-11 m^3/(kg s^2)
c_light = constants.c                    # 2.998e8  m/s
hbar = constants.hbar                    # 1.055e-34 J s
M_sun = 1.989e30                         # kg
eV_to_J = constants.eV                   # 1.602e-19 J

# ============================================================
# SCT parameters
# ============================================================
LAMBDA_EV = 2.38e-3                      # eV  (PPN-1 Cassini bound)
LAMBDA_INV_M = hbar * c_light / (LAMBDA_EV * eV_to_J)  # 1/Lambda in meters

M2_OVER_LAMBDA = np.sqrt(60.0 / 13.0)   # ~ 2.148  (spin-2, Weyl)
M0_OVER_LAMBDA = np.sqrt(6.0)            # ~ 2.449  (spin-0, xi=0)

# Physical masses in inverse meters
M2_EV = M2_OVER_LAMBDA * LAMBDA_EV
M0_EV = M0_OVER_LAMBDA * LAMBDA_EV
M2_INV_M = M2_EV * eV_to_J / (hbar * c_light)   # m^{-1}
M0_INV_M = M0_EV * eV_to_J / (hbar * c_light)   # m^{-1}

# Propagator coefficient (Phase 3 result)
C2_COEFF = 13.0 / 60.0
ALPHA_C = 13.0 / 120.0

# Output directories
PROJ = Path(__file__).resolve().parent.parent.parent
FIG_DIR = PROJ / "analysis" / "figures" / "lt3a"
RES_DIR = PROJ / "analysis" / "results" / "lt3a"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Utility functions
# ============================================================
def schwarzschild_radius_m(M_kg: float) -> float:
    """Schwarzschild radius r_s = 2GM/c^2 [meters]."""
    return 2.0 * G_N * M_kg / c_light**2


def kerr_outer_horizon_m(M_kg: float, a_star: float) -> float:
    """Outer horizon r_+ = (r_s/2)(1 + sqrt(1 - a*^2)) [meters].

    Parameters
    ----------
    M_kg : float
        Black hole mass in kg.
    a_star : float
        Dimensionless spin a/M (0 <= a_star < 1).

    Returns
    -------
    r_plus : float
        Outer horizon radius in meters.
    """
    M_geom = G_N * M_kg / c_light**2       # M in meters (geometric)
    return M_geom * (1.0 + np.sqrt(1.0 - a_star**2))


def qnm_frequency_hz(omega_M: complex, M_kg: float) -> complex:
    """Convert dimensionless omega*M to physical frequency [Hz].

    omega_phys = omega_M * c^3 / (G * M_kg)
    f = omega_phys / (2*pi)
    """
    omega_phys = omega_M * c_light**3 / (G_N * M_kg)
    return omega_phys / (2.0 * np.pi)


def perturb_eq_estimate(omega_M: complex, M_kg: float) -> float:
    """Perturbation-equation fractional shift: c2 * (omega/Lambda)^2.

    The physical angular frequency is omega = omega_M * c^3 / (G*M).
    Lambda in angular frequency: Lambda_rad = Lambda_eV * eV / hbar.

    Returns |delta_omega / omega|.
    """
    omega_rad = abs(omega_M) * c_light**3 / (G_N * M_kg)
    Lambda_rad = LAMBDA_EV * eV_to_J / hbar
    return C2_COEFF * (omega_rad / Lambda_rad)**2


def metric_modification_estimate(M_kg: float, a_star: float) -> float:
    """Metric-modification fractional shift: ~ exp(-m2 * r_+).

    This is the suppression factor from the Yukawa modification evaluated
    at the outer horizon.  The actual prefactor is O(1), so this gives the
    order of magnitude.

    Returns exp(-m2 * r_+).
    """
    r_plus = kerr_outer_horizon_m(M_kg, a_star)
    exponent = M2_INV_M * r_plus
    # Avoid overflow in log
    if exponent > 700:
        return 0.0
    return np.exp(-exponent)


def log10_metric_modification(M_kg: float, a_star: float) -> float:
    """log10 of the metric-modification suppression factor.

    Returns log10(exp(-m2 * r_+)) = -m2 * r_+ / ln(10).
    """
    r_plus = kerr_outer_horizon_m(M_kg, a_star)
    exponent = M2_INV_M * r_plus
    return -exponent / np.log(10.0)


# ============================================================
# GR Kerr QNMs via the qnm package
# ============================================================
def compute_kerr_qnm_table(
    spin_values: np.ndarray,
    l_val: int = 2,
    m_val: int = 2,
    n_val: int = 0,
) -> dict:
    """Compute GR Kerr QNM frequencies for a range of spin values.

    Parameters
    ----------
    spin_values : array
        Dimensionless spin a/M values.
    l_val, m_val, n_val : int
        Mode numbers (s=-2 gravitational).

    Returns
    -------
    table : dict with keys 'a_star', 'omega_R_M', 'omega_I_M', 'A_lm'.
    """
    seq = qnm.modes_cache(s=-2, l=l_val, m=m_val, n=n_val)

    omega_R_list = []
    omega_I_list = []
    A_lm_list = []

    for a in spin_values:
        result = seq(a=float(a))
        omega = result[0]       # complex omega*M
        A_lm = result[1]        # angular separation constant
        omega_R_list.append(float(omega.real))
        omega_I_list.append(float(omega.imag))
        A_lm_list.append(complex(A_lm))

    return {
        "a_star": spin_values.tolist(),
        "omega_R_M": omega_R_list,
        "omega_I_M": omega_I_list,
        "A_lm_real": [a.real for a in A_lm_list],
        "A_lm_imag": [a.imag for a in A_lm_list],
    }


# ============================================================
# Full spin scan with SCT estimates
# ============================================================
def run_spin_scan(
    M_solar: float,
    label: str,
    n_spins: int = 30,
    l_val: int = 2,
    m_val: int = 2,
    n_val: int = 0,
) -> dict:
    """Run a full spin scan for a given BH mass.

    Parameters
    ----------
    M_solar : float
        Black hole mass in solar masses.
    label : str
        Human-readable label (e.g. "GW150914 remnant").
    n_spins : int
        Number of spin points.
    l_val, m_val, n_val : int
        QNM mode numbers.

    Returns
    -------
    result : dict with all computed quantities.
    """
    M_kg = M_solar * M_sun
    r_s = schwarzschild_radius_m(M_kg)

    # Spin grid: uniform from 0 to 0.95, then finer near extremal
    spins_uniform = np.linspace(0.0, 0.95, n_spins - 5)
    spins_near_extremal = np.array([0.96, 0.97, 0.98, 0.99, 0.998])
    spins = np.concatenate([spins_uniform, spins_near_extremal])
    spins = np.sort(np.unique(spins))

    # Compute GR QNMs
    qnm_table = compute_kerr_qnm_table(spins, l_val, m_val, n_val)

    # Compute SCT estimates
    r_plus_list = []
    metric_log10_list = []
    perturb_eq_list = []
    perturb_eq_log10_list = []
    m2_r_plus_list = []

    for i, a in enumerate(spins):
        r_p = kerr_outer_horizon_m(M_kg, float(a))
        r_plus_list.append(r_p)
        m2_rp = M2_INV_M * r_p
        m2_r_plus_list.append(m2_rp)

        # Metric modification: log10(exp(-m2*r_+))
        log10_met = -m2_rp / np.log(10.0)
        metric_log10_list.append(log10_met)

        # Perturbation-equation estimate
        omega_M = complex(qnm_table["omega_R_M"][i], qnm_table["omega_I_M"][i])
        pe = perturb_eq_estimate(omega_M, M_kg)
        perturb_eq_list.append(pe)
        perturb_eq_log10_list.append(np.log10(pe) if pe > 0 else -np.inf)

    # Physical frequencies at selected spins
    freq_hz = []
    for i, a in enumerate(spins):
        omega_M = complex(qnm_table["omega_R_M"][i], qnm_table["omega_I_M"][i])
        f_hz = qnm_frequency_hz(omega_M, M_kg)
        freq_hz.append({"a_star": float(a),
                         "f_R_Hz": float(f_hz.real),
                         "f_I_Hz": float(f_hz.imag)})

    return {
        "label": label,
        "M_solar": M_solar,
        "r_s_m": r_s,
        "mode": {"l": l_val, "m": m_val, "n": n_val},
        "spins": spins.tolist(),
        "qnm_table": qnm_table,
        "r_plus_m": r_plus_list,
        "m2_r_plus": m2_r_plus_list,
        "metric_mod_log10": metric_log10_list,
        "perturb_eq_frac": perturb_eq_list,
        "perturb_eq_log10": perturb_eq_log10_list,
        "freq_hz": freq_hz,
    }


# ============================================================
# Multi-mode Kerr scan (several l values)
# ============================================================
def run_multimode_scan(
    M_solar: float,
    label: str,
    modes: list[tuple[int, int, int]] | None = None,
) -> dict:
    """Compute QNMs and SCT estimates for multiple modes.

    Parameters
    ----------
    modes : list of (l, m, n) tuples.
        Default: [(2,2,0), (2,1,0), (3,3,0), (3,2,0), (4,4,0)].
    """
    if modes is None:
        modes = [(2, 2, 0), (2, 1, 0), (3, 3, 0), (3, 2, 0), (4, 4, 0)]

    M_kg = M_solar * M_sun
    results = {}

    for l_val, m_val, n_val in modes:
        key = f"l{l_val}_m{m_val}_n{n_val}"
        # Sample 15 spins for the multi-mode table
        spins = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                          0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 0.995, 0.998])
        table = compute_kerr_qnm_table(spins, l_val, m_val, n_val)
        pe_list = []
        for i, a in enumerate(spins):
            omega_M = complex(table["omega_R_M"][i], table["omega_I_M"][i])
            pe = perturb_eq_estimate(omega_M, M_kg)
            pe_list.append(pe)

        results[key] = {
            "l": l_val, "m": m_val, "n": n_val,
            "spins": spins.tolist(),
            "omega_R_M": table["omega_R_M"],
            "omega_I_M": table["omega_I_M"],
            "perturb_eq_frac": pe_list,
        }

    return {"label": label, "M_solar": M_solar, "modes": results}


# ============================================================
# Figures
# ============================================================
def plot_kerr_qnm_vs_spin(scan_10: dict, scan_62: dict) -> None:
    """Plot GR Kerr QNM frequency (l=2,m=2,n=0) vs spin for two masses.

    Saves: lt3a_kerr_qnm_vs_spin.pdf / .png
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, scan, color in zip(axes, [scan_10, scan_62], ["#1f77b4", "#d62728"]):
        spins = scan["spins"]
        omega_R = scan["qnm_table"]["omega_R_M"]
        omega_I = [-x for x in scan["qnm_table"]["omega_I_M"]]  # make positive

        ax.plot(spins, omega_R, "-o", color=color, ms=3, lw=1.5,
                label=r"$\omega_R \cdot M$")
        ax.plot(spins, omega_I, "--s", color="#2ca02c", ms=3, lw=1.5,
                label=r"$|\omega_I| \cdot M$")

        ax.set_xlabel(r"$a/M$", fontsize=13)
        ax.set_ylabel(r"$\omega \cdot M$  (dimensionless)", fontsize=13)
        ax.set_title(f'{scan["label"]}  (l=2, m=2, n=0)', fontsize=12)
        ax.legend(fontsize=11, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.02, 1.02)

        # Add physical frequency annotation at a=0
        if scan["freq_hz"]:
            f0 = scan["freq_hz"][0]
            f_R = f0["f_R_Hz"]
            ax.annotate(f"$f(a{{=}}0) = {f_R:.0f}$ Hz",
                        xy=(0.0, omega_R[0]), xytext=(0.2, omega_R[0] * 0.85),
                        fontsize=10, arrowprops=dict(arrowstyle="->", lw=0.8))

    plt.tight_layout()
    fig.savefig(FIG_DIR / "lt3a_kerr_qnm_vs_spin.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / "lt3a_kerr_qnm_vs_spin.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / 'lt3a_kerr_qnm_vs_spin.pdf'}")


def plot_kerr_spin_scan(scan_10: dict, scan_62: dict) -> None:
    """Plot SCT fractional shift vs spin for two BH masses.

    Shows both metric modification and perturbation-equation estimates.
    Saves: lt3a_kerr_spin_scan.pdf / .png
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ligo_sensitivity = -1.0  # LIGO fractional freq accuracy ~ 10^{-1} (ringdown)

    for ax, scan, color_m, color_p in zip(
        axes,
        [scan_10, scan_62],
        ["#d62728", "#ff7f0e"],
        ["#1f77b4", "#2ca02c"],
    ):
        spins = scan["spins"]
        met_log10 = scan["metric_mod_log10"]
        pe_log10 = scan["perturb_eq_log10"]

        ax.plot(spins, met_log10, "-o", color=color_m, ms=3, lw=1.5,
                label=r"Metric mod: $e^{-m_2 r_+}$")
        ax.plot(spins, pe_log10, "--s", color=color_p, ms=3, lw=1.5,
                label=r"Perturb-eq: $c_2(\omega/\Lambda)^2$")

        # LIGO sensitivity line
        ax.axhline(ligo_sensitivity, color="gray", ls=":", lw=1.2,
                    label=r"LIGO ring-down $\sim 10^{-1}$")

        ax.set_xlabel(r"$a/M$", fontsize=13)
        ax.set_ylabel(r"$\log_{10}|\delta\omega/\omega|$", fontsize=13)
        ax.set_title(f'{scan["label"]}', fontsize=12)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.02, 1.02)

        # Annotate key numbers
        # Metric mod at a=0 and a=0.998
        met_a0 = met_log10[0]
        met_amax = met_log10[-1]
        ax.annotate(f"a=0: {met_a0:.0f}", xy=(0.0, met_a0),
                     xytext=(0.15, met_a0 + (met_amax - met_a0) * 0.15),
                     fontsize=8, color=color_m)

        pe_a0 = pe_log10[0]
        ax.annotate(f"a=0: {pe_a0:.1f}", xy=(0.0, pe_a0),
                     xytext=(0.15, pe_a0 + 1.5),
                     fontsize=8, color=color_p)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "lt3a_kerr_spin_scan.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / "lt3a_kerr_spin_scan.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / 'lt3a_kerr_spin_scan.pdf'}")


def plot_r_plus_vs_spin() -> None:
    """Plot r_+/M vs a/M to show horizon shrinkage.

    Saves: lt3a_kerr_r_plus.pdf / .png
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    spins = np.linspace(0, 0.9999, 500)
    r_plus_over_M = 1.0 + np.sqrt(1.0 - spins**2)

    ax.plot(spins, r_plus_over_M, "k-", lw=2)
    ax.set_xlabel(r"$a/M$", fontsize=14)
    ax.set_ylabel(r"$r_+ / M$", fontsize=14)
    ax.set_title(r"Kerr outer horizon vs spin ($r_+/M = 1 + \sqrt{1-a_*^2}$)",
                 fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.9, 2.1)

    # Mark key values
    for a_val in [0.0, 0.5, 0.9, 0.998]:
        rp = 1.0 + np.sqrt(1.0 - a_val**2)
        ax.plot(a_val, rp, "ro", ms=6)
        ax.annotate(f"$a_*={a_val}$\n$r_+/M={rp:.3f}$",
                     xy=(a_val, rp),
                     xytext=(a_val + 0.05, rp - 0.12),
                     fontsize=9)

    # Schwarzschild and extremal limits
    ax.axhline(2.0, color="blue", ls="--", lw=0.8, alpha=0.5)
    ax.annotate(r"Schwarzschild: $r_+=2M$", xy=(0.3, 2.01), fontsize=9, color="blue")
    ax.axhline(1.0, color="red", ls="--", lw=0.8, alpha=0.5)
    ax.annotate(r"Extremal: $r_+=M$", xy=(0.6, 1.02), fontsize=9, color="red")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "lt3a_kerr_r_plus.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / "lt3a_kerr_r_plus.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / 'lt3a_kerr_r_plus.pdf'}")


# ============================================================
# Verification
# ============================================================
def verify() -> list[str]:
    """Run numbered verification checks.

    Returns list of PASS/FAIL strings.
    """
    results = []
    n_check = 0

    def check(cond: bool, desc: str) -> None:
        nonlocal n_check
        n_check += 1
        status = "PASS" if cond else "FAIL"
        msg = f"  VR-{n_check:03d}: [{status}] {desc}"
        print(msg)
        results.append(msg)
        if not status == "PASS":
            print(f"    *** VERIFICATION FAILURE ***")

    # ----------------------------------------------------------
    # 1. Schwarzschild limit (a=0): omega*M matches known value
    # ----------------------------------------------------------
    seq = qnm.modes_cache(s=-2, l=2, m=2, n=0)
    omega_a0 = seq(a=0.0)[0]
    check(abs(omega_a0.real - 0.3737) < 0.001,
          f"Schwarzschild l=2 omega_R*M = {omega_a0.real:.6f} ~ 0.3737")
    check(abs(omega_a0.imag + 0.0890) < 0.001,
          f"Schwarzschild l=2 omega_I*M = {omega_a0.imag:.6f} ~ -0.0890")

    # ----------------------------------------------------------
    # 2. Kerr r_+ monotonically decreasing with spin
    # ----------------------------------------------------------
    spins_test = np.linspace(0, 0.999, 100)
    r_plus_test = 1.0 + np.sqrt(1.0 - spins_test**2)
    diffs = np.diff(r_plus_test)
    check(np.all(diffs < 0),
          f"r_+/M monotonically decreasing (min diff = {diffs.min():.6e})")

    # ----------------------------------------------------------
    # 3. r_+(a=0) = 2M (Schwarzschild)
    # ----------------------------------------------------------
    check(abs(kerr_outer_horizon_m(10 * M_sun, 0.0)
              - schwarzschild_radius_m(10 * M_sun)) < 1e-6,
          "r_+(a=0) = r_s = 2GM/c^2")

    # ----------------------------------------------------------
    # 4. r_+(a->1) -> M (extremal limit)
    # ----------------------------------------------------------
    r_ext = kerr_outer_horizon_m(10 * M_sun, 0.9999)
    M_geom = G_N * 10 * M_sun / c_light**2
    check(abs(r_ext / M_geom - 1.0) < 0.02,
          f"r_+(a=0.9999) / M = {r_ext/M_geom:.4f} ~ 1.0")

    # ----------------------------------------------------------
    # 5. omega_R increases with spin (co-rotating modes)
    # ----------------------------------------------------------
    omega_a0_R = seq(a=0.0)[0].real
    omega_a9_R = seq(a=0.9)[0].real
    check(omega_a9_R > omega_a0_R,
          f"omega_R(a=0.9) = {omega_a9_R:.4f} > omega_R(a=0) = {omega_a0_R:.4f}")

    # ----------------------------------------------------------
    # 6. Damping rate |omega_I| decreases with spin
    # ----------------------------------------------------------
    omega_a0_I = abs(seq(a=0.0)[0].imag)
    omega_a9_I = abs(seq(a=0.9)[0].imag)
    check(omega_a9_I < omega_a0_I,
          f"|omega_I|(a=0.9) = {omega_a9_I:.4f} < |omega_I|(a=0) = {omega_a0_I:.4f}")

    # ----------------------------------------------------------
    # 7. Metric modification at a=0 is astronomically small
    # ----------------------------------------------------------
    log10_met_a0 = log10_metric_modification(10 * M_sun, 0.0)
    check(log10_met_a0 < -1e6,
          f"log10(metric mod) at a=0, 10 Msun = {log10_met_a0:.2e}")

    # ----------------------------------------------------------
    # 8. Metric modification at a=0.998 is still astronomically small
    # ----------------------------------------------------------
    log10_met_a998 = log10_metric_modification(10 * M_sun, 0.998)
    check(log10_met_a998 < -1e6,
          f"log10(metric mod) at a=0.998, 10 Msun = {log10_met_a998:.2e}")

    # ----------------------------------------------------------
    # 9. Metric mod weaker at higher spin (larger = less negative)
    # ----------------------------------------------------------
    check(log10_met_a998 > log10_met_a0,
          "Metric mod weaker (less negative) at a=0.998 than a=0")

    # ----------------------------------------------------------
    # 10. Perturbation-equation estimate in expected range
    # ----------------------------------------------------------
    pe_10 = perturb_eq_estimate(omega_a0, 10 * M_sun)
    check(1e-25 < pe_10 < 1e-15,
          f"Perturb-eq estimate (10 Msun, a=0) = {pe_10:.2e}")

    # ----------------------------------------------------------
    # 11. Perturbation-equation dominates over metric modification
    # ----------------------------------------------------------
    # pe ~ 10^{-20}, metric ~ 10^{-10^18}
    # log10(pe) >> log10(metric)
    check(np.log10(pe_10) > log10_met_a0,
          "Perturbation-eq dominates metric modification by many orders")

    # ----------------------------------------------------------
    # 12. GW150914 reference: omega_R*M ~ 0.53 at a~0.67
    # ----------------------------------------------------------
    seq_22 = qnm.modes_cache(s=-2, l=2, m=2, n=0)
    omega_gw150914 = seq_22(a=0.67)[0]
    check(0.50 < omega_gw150914.real < 0.60,
          f"GW150914-like (a=0.67): omega_R*M = {omega_gw150914.real:.4f}")

    # ----------------------------------------------------------
    # 13. Physical frequency for 62 Msun at a=0.67 ~ 250 Hz
    # ----------------------------------------------------------
    f_phys = qnm_frequency_hz(omega_gw150914, 62 * M_sun)
    check(200 < f_phys.real < 300,
          f"GW150914 f_R = {f_phys.real:.1f} Hz (expected ~250 Hz)")

    # ----------------------------------------------------------
    # 14. Different m values: m<l has lower omega_R
    # ----------------------------------------------------------
    seq_21 = qnm.modes_cache(s=-2, l=2, m=1, n=0)
    omega_21 = seq_21(a=0.5)[0]
    omega_22 = seq_22(a=0.5)[0]
    check(omega_21.real < omega_22.real,
          f"m=1: omega_R={omega_21.real:.4f} < m=2: {omega_22.real:.4f} at a=0.5")

    # ----------------------------------------------------------
    # 15. Higher l: l=3 has higher omega_R than l=2 at a=0
    # ----------------------------------------------------------
    seq_33 = qnm.modes_cache(s=-2, l=3, m=3, n=0)
    omega_33_a0 = seq_33(a=0.0)[0]
    check(omega_33_a0.real > omega_a0_R,
          f"l=3: omega_R={omega_33_a0.real:.4f} > l=2: {omega_a0_R:.4f}")

    # ----------------------------------------------------------
    # 16. Perturb-eq at a=0.998 is slightly larger (larger omega)
    # ----------------------------------------------------------
    omega_998 = seq_22(a=0.998)[0]
    pe_998 = perturb_eq_estimate(omega_998, 10 * M_sun)
    check(pe_998 > pe_10,
          f"Perturb-eq at a=0.998 ({pe_998:.2e}) > at a=0 ({pe_10:.2e})")

    # ----------------------------------------------------------
    # 17. All shifts unmeasurable (< 10^{-10})
    # ----------------------------------------------------------
    check(pe_998 < 1e-10,
          f"Largest perturb-eq shift ({pe_998:.2e}) << LIGO sensitivity")

    # ----------------------------------------------------------
    # 18. Consistency: qnm_frequency_hz inverts correctly
    # ----------------------------------------------------------
    M_test = 30 * M_sun
    omega_test = complex(0.4, -0.08)
    f_test = qnm_frequency_hz(omega_test, M_test)
    omega_back = f_test * 2 * np.pi * G_N * M_test / c_light**3
    check(abs(omega_back - omega_test) / abs(omega_test) < 1e-10,
          "qnm_frequency_hz round-trip consistency")

    # Summary
    n_pass = sum("PASS" in r for r in results)
    n_fail = sum("FAIL" in r for r in results)
    print(f"\n  Summary: {n_pass}/{n_check} PASS, {n_fail} FAIL")
    return results


# ============================================================
# Main
# ============================================================
def main() -> dict:
    """Run the full Kerr QNM analysis."""
    t0 = time.time()
    print("=" * 70)
    print("LT-3a Phase 8: Kerr QNM Analysis for SCT")
    print("=" * 70)

    # ----------------------------------------------------------
    # 1. Spin scans for two reference masses
    # ----------------------------------------------------------
    print("\n[1/5] Computing Kerr QNM spin scan: 10 M_sun ...")
    scan_10 = run_spin_scan(10.0, r"$10\,M_\odot$ BH", n_spins=30)

    print("[2/5] Computing Kerr QNM spin scan: 62 M_sun (GW150914) ...")
    scan_62 = run_spin_scan(62.0, r"$62\,M_\odot$ BH (GW150914)", n_spins=30)

    # ----------------------------------------------------------
    # 2. Multi-mode table
    # ----------------------------------------------------------
    print("[3/5] Computing multi-mode Kerr QNMs for 62 M_sun ...")
    multimode = run_multimode_scan(62.0, "GW150914 remnant")

    # ----------------------------------------------------------
    # 3. Print key results
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("KEY RESULTS")
    print("=" * 70)

    # Table: spin, omega_R*M, omega_I*M, perturb-eq, metric-mod
    print(f"\n{'a/M':>6s}  {'omega_R*M':>10s}  {'omega_I*M':>10s}"
          f"  {'log10(pe)':>10s}  {'log10(met)':>12s}  {'r+/M':>6s}")
    print("-" * 65)

    for i, a in enumerate(scan_10["spins"]):
        if a in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.998]:
            oR = scan_10["qnm_table"]["omega_R_M"][i]
            oI = scan_10["qnm_table"]["omega_I_M"][i]
            pe = scan_10["perturb_eq_log10"][i]
            met = scan_10["metric_mod_log10"][i]
            rp = scan_10["r_plus_m"][i]
            M_geom = G_N * 10 * M_sun / c_light**2
            print(f"{a:6.3f}  {oR:10.6f}  {oI:10.6f}"
                  f"  {pe:10.2f}  {met:12.2e}  {rp/M_geom:6.3f}")

    print("\nPhysical QNM frequencies (62 M_sun):")
    for entry in scan_62["freq_hz"]:
        if entry["a_star"] in [0.0, 0.3, 0.67, 0.9, 0.998]:
            print(f"  a/M = {entry['a_star']:.3f}: "
                  f"f_R = {entry['f_R_Hz']:.1f} Hz, "
                  f"f_I = {entry['f_I_Hz']:.1f} Hz")
        elif abs(entry["a_star"] - 0.67) < 0.02:
            print(f"  a/M = {entry['a_star']:.3f}: "
                  f"f_R = {entry['f_R_Hz']:.1f} Hz, "
                  f"f_I = {entry['f_I_Hz']:.1f} Hz  [~ GW150914]")

    # ----------------------------------------------------------
    # 4. Figures
    # ----------------------------------------------------------
    print("\n[4/5] Generating figures ...")
    plot_kerr_qnm_vs_spin(scan_10, scan_62)
    plot_kerr_spin_scan(scan_10, scan_62)
    plot_r_plus_vs_spin()

    # ----------------------------------------------------------
    # 5. Verification
    # ----------------------------------------------------------
    print("\n[5/5] Running verification ...")
    vr_results = verify()

    # ----------------------------------------------------------
    # 6. Save JSON
    # ----------------------------------------------------------
    output = {
        "description": "LT-3a Phase 8: Kerr QNM analysis for SCT",
        "author": "David Alfyorov",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sct_parameters": {
            "Lambda_eV": LAMBDA_EV,
            "m2_over_Lambda": float(M2_OVER_LAMBDA),
            "m0_over_Lambda": float(M0_OVER_LAMBDA),
            "c2_coeff": C2_COEFF,
            "alpha_C": ALPHA_C,
        },
        "scan_10_Msun": {
            "M_solar": scan_10["M_solar"],
            "label": "10 solar mass BH",
            "mode": scan_10["mode"],
            "n_spins": len(scan_10["spins"]),
            "spins": scan_10["spins"],
            "omega_R_M": scan_10["qnm_table"]["omega_R_M"],
            "omega_I_M": scan_10["qnm_table"]["omega_I_M"],
            "perturb_eq_log10": scan_10["perturb_eq_log10"],
            "metric_mod_log10": scan_10["metric_mod_log10"],
            "m2_r_plus": scan_10["m2_r_plus"],
        },
        "scan_62_Msun": {
            "M_solar": scan_62["M_solar"],
            "label": "62 solar mass BH (GW150914)",
            "mode": scan_62["mode"],
            "n_spins": len(scan_62["spins"]),
            "spins": scan_62["spins"],
            "omega_R_M": scan_62["qnm_table"]["omega_R_M"],
            "omega_I_M": scan_62["qnm_table"]["omega_I_M"],
            "perturb_eq_log10": scan_62["perturb_eq_log10"],
            "metric_mod_log10": scan_62["metric_mod_log10"],
            "m2_r_plus": scan_62["m2_r_plus"],
        },
        "multimode_62_Msun": multimode,
        "conclusion": (
            "ALL astrophysical Kerr BHs: SCT is indistinguishable from GR. "
            "Metric modification: exp(-m2*r_+) is astronomically small for "
            "all spins. Perturbation-equation estimate: c2*(omega/Lambda)^2 "
            "~ 10^{-20} for stellar BH, independent of spin to leading order. "
            "Even at near-extremal spin a/M=0.998, the QNM frequency shift is "
            "~15 orders of magnitude below LIGO ringdown sensitivity."
        ),
        "caveat": (
            "The metric modification is computed by substituting f_SCT into "
            "the GR perturbation formula. The perturbation-equation correction "
            "(dominant, ~10^{-20}) is estimated from c2*(omega/Lambda)^2 but "
            "not derived from first principles (blocked by OP-01/Gap G1)."
        ),
        "verification": vr_results,
    }

    json_path = RES_DIR / "lt3a_kerr_qnm.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {json_path}")

    dt = time.time() - t0
    print(f"\n  Total time: {dt:.1f} s")
    print("=" * 70)

    return output


if __name__ == "__main__":
    main()

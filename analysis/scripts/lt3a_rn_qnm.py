# ruff: noqa: E402, I001
"""
LT-3a Phase 8: Reissner-Nordstrom QNM analysis for Spectral Causal Theory.

Computes quasinormal mode frequencies of Reissner-Nordstrom (charged,
non-rotating) black holes using the WKB method, and estimates SCT
corrections as a function of the charge-to-mass ratio Q/M.

In GR, the RN metric function is:
    f(r) = 1 - r_s/r + r_Q^2/r^2

where r_s = 2M, r_Q^2 = Q^2 (in geometric units G=c=1).
Horizons: r_+/- = M +/- sqrt(M^2 - Q^2).

SCT modification: the Yukawa correction applies to the gravitational
(mass) part of the metric, not the electromagnetic part:
    f_SCT(r) = 1 - (r_s/r)*h(r) + r_Q^2/r^2
with h(r) = 1 - (4/3)exp(-m2*r) + (1/3)exp(-m0*r).

Two types of SCT corrections:
  (A) Metric modification: exp(-m2*r_+) where r_+ decreases with Q/M.
      At Q/M -> 1 (extremal), r_+ -> M, so suppression is weaker than
      Schwarzschild (where r_+ = 2M), but still extreme.
  (B) Perturbation-equation correction: c2*(omega/Lambda)^2 ~ 10^{-20}.
      Dominates over (A) but still unmeasurable.

CAVEAT: Substituting f_SCT into the GR Regge-Wheeler formula captures
the metric-modification part but NOT the perturbation-equation correction
(blocked by OP-01/Gap G1).

Result: For ALL astrophysical RN BHs, SCT is indistinguishable from GR.

Author: David Alfyorov
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy import optimize, constants

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
LAMBDA_EV = 2.38e-3                      # eV (PPN-1 Cassini bound)
LAMBDA_INV_M = hbar * c_light / (LAMBDA_EV * eV_to_J)  # 1/Lambda in meters

M2_OVER_LAMBDA = np.sqrt(60.0 / 13.0)   # ~ 2.148 (spin-2, Weyl)
M0_OVER_LAMBDA = np.sqrt(6.0)            # ~ 2.449 (spin-0, xi=0)

M2_EV = M2_OVER_LAMBDA * LAMBDA_EV
M0_EV = M0_OVER_LAMBDA * LAMBDA_EV
M2_INV_M = M2_EV * eV_to_J / (hbar * c_light)
M0_INV_M = M0_EV * eV_to_J / (hbar * c_light)

C2_COEFF = 13.0 / 60.0
ALPHA_C = 13.0 / 120.0

# Output directories
PROJ = Path(__file__).resolve().parent.parent.parent
FIG_DIR = PROJ / "analysis" / "figures" / "lt3a"
RES_DIR = PROJ / "analysis" / "results" / "lt3a"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# RN metric in geometric units (G=c=1, r_s=2M, distances in meters)
# ============================================================
def schwarzschild_radius_m(M_kg: float) -> float:
    """Schwarzschild radius r_s = 2GM/c^2 [meters]."""
    return 2.0 * G_N * M_kg / c_light**2


def rn_outer_horizon(M_geom: float, Q_star: float) -> float:
    """Outer horizon r_+ = M*(1 + sqrt(1 - Q*^2)) in geometric units.

    Parameters
    ----------
    M_geom : float
        Geometric mass M = GM_kg/c^2 [meters].
    Q_star : float
        Charge-to-mass ratio Q/M (0 <= Q_star < 1 in geometric units).

    Returns
    -------
    r_plus : float
        Outer horizon radius [meters].
    """
    return M_geom * (1.0 + np.sqrt(1.0 - Q_star**2))


def rn_inner_horizon(M_geom: float, Q_star: float) -> float:
    """Inner horizon r_- = M*(1 - sqrt(1 - Q*^2)) in geometric units."""
    return M_geom * (1.0 - np.sqrt(1.0 - Q_star**2))


def f_RN_GR(r: float, r_s: float, r_Q_sq: float) -> float:
    """GR Reissner-Nordstrom metric function.

    f(r) = 1 - r_s/r + r_Q^2/r^2

    Parameters
    ----------
    r : float
        Radial coordinate [meters].
    r_s : float
        Schwarzschild radius = 2M [meters].
    r_Q_sq : float
        Charge parameter r_Q^2 = Q^2*G/(4*pi*eps0*c^4) [meters^2].
        In geometric units (G=c=1): r_Q^2 = Q^2.
    """
    return 1.0 - r_s / r + r_Q_sq / r**2


def f_RN_SCT(r: float, r_s: float, r_Q_sq: float,
             m2: float, m0: float) -> float:
    """SCT-modified RN metric function.

    f(r) = 1 - (r_s/r)*h(r) + r_Q^2/r^2

    The Yukawa correction applies to the gravitational (mass) part only.
    The electromagnetic contribution r_Q^2/r^2 is NOT modified by SCT.
    """
    h = 1.0 - (4.0 / 3.0) * np.exp(-m2 * r) + (1.0 / 3.0) * np.exp(-m0 * r)
    return 1.0 - (r_s / r) * h + r_Q_sq / r**2


def V_RW_RN_GR(r: float, r_s: float, r_Q_sq: float, l: int = 2) -> float:
    """Regge-Wheeler potential for RN in GR (axial/odd parity, s=2).

    For a general static spherically symmetric metric with f(r), the
    axial gravitational perturbation potential is:
        V(r) = f(r) * [l(l+1)/r^2 - 3*(1-f)/r^2]

    This follows from V = f * [l(l+1)/r^2 - 6*M_eff/r^3] with
    M_eff(r) = r*(1-f)/2.

    For Schwarzschild (r_Q=0): M_eff = r_s/2 = const, recovering standard result.
    For RN: M_eff(r) = r_s/(2r) * r - r_Q^2/(2r^2) * r = r_s/2 - r_Q^2/(2r)
    which is r-dependent. This gives the correct RN Regge-Wheeler potential.
    """
    f = f_RN_GR(r, r_s, r_Q_sq)
    one_minus_f = r_s / r - r_Q_sq / r**2
    return f * (l * (l + 1) / r**2 - 3.0 * one_minus_f / r**2)


def V_RW_RN_SCT(r: float, r_s: float, r_Q_sq: float,
                m2: float, m0: float, l: int = 2) -> float:
    """SCT-modified Regge-Wheeler potential for RN.

    Same formula V = f*[l(l+1)/r^2 - 3*(1-f)/r^2] with f = f_SCT_RN.
    """
    f = f_RN_SCT(r, r_s, r_Q_sq, m2, m0)
    one_minus_f = 1.0 - f
    return f * (l * (l + 1) / r**2 - 3.0 * one_minus_f / r**2)


# ============================================================
# Tortoise coordinate for RN
# ============================================================
def tortoise_RN(r: float, r_s: float, r_Q_sq: float) -> float:
    """Tortoise coordinate for RN: r* = integral dr/f(r).

    For RN with two horizons r_+, r_-:
    r* = r + r_+^2/(r_+ - r_-) * ln|r - r_+| - r_-^2/(r_+ - r_-) * ln|r - r_-|

    Uses M = r_s/2, Q^2 = r_Q_sq (geometric units).
    """
    M = r_s / 2.0
    Q_sq = r_Q_sq
    disc = M**2 - Q_sq
    if disc <= 0:
        # Extremal or super-extremal: fall back to numerical
        return r
    sqrt_disc = np.sqrt(disc)
    r_plus = M + sqrt_disc
    r_minus = M - sqrt_disc

    if r <= r_plus * 1.001:
        return -1e30

    dr = r_plus - r_minus
    if abs(dr) < 1e-30:
        # Near-extremal: use Schwarzschild-like approximation
        return r + r_plus * np.log(abs(r / r_plus - 1))

    return (r
            + r_plus**2 / dr * np.log(abs(r - r_plus))
            - r_minus**2 / dr * np.log(abs(r - r_minus)))


# ============================================================
# WKB method for RN QNMs
# ============================================================
def find_potential_peak_RN(r_s: float, r_Q_sq: float, l: int = 2,
                           use_sct: bool = False,
                           m2: float = 0.0, m0: float = 0.0) -> float:
    """Find the peak of V_RW for the RN metric.

    Returns r_peak [meters].
    """
    M = r_s / 2.0
    Q_sq = r_Q_sq
    disc = M**2 - Q_sq
    if disc < 0:
        return np.nan
    r_plus = M + np.sqrt(max(disc, 0))

    r_min = r_plus * 1.01
    r_max = r_s * 5.0
    # Ensure r_max > r_min
    if r_max <= r_min:
        r_max = r_min * 5.0

    if use_sct:
        def neg_V(r):
            return -V_RW_RN_SCT(r, r_s, r_Q_sq, m2, m0, l)
    else:
        def neg_V(r):
            return -V_RW_RN_GR(r, r_s, r_Q_sq, l)

    result = optimize.minimize_scalar(neg_V, bounds=(r_min, r_max),
                                       method="bounded")
    return result.x


def numerical_derivatives_RN(r_peak: float, r_s: float, r_Q_sq: float,
                              l: int = 2, use_sct: bool = False,
                              m2: float = 0.0, m0: float = 0.0) -> tuple:
    """Compute V0, V2, V3, V4 at the potential peak in tortoise coordinates.

    Uses polynomial fitting of V(r*) around the peak.
    Returns (V0, V2, V3, V4).
    """
    if use_sct:
        def V_func(r):
            return V_RW_RN_SCT(r, r_s, r_Q_sq, m2, m0, l)
    else:
        def V_func(r):
            return V_RW_RN_GR(r, r_s, r_Q_sq, l)

    V0 = V_func(r_peak)
    r_star_0 = tortoise_RN(r_peak, r_s, r_Q_sq)

    # Sample points in r-coordinate around the peak
    N_half = 10
    h_r = 0.005 * r_s
    M = r_s / 2.0
    Q_sq = r_Q_sq
    disc = M**2 - Q_sq
    r_plus = M + np.sqrt(max(disc, 0))

    r_arr = np.array([r_peak + i * h_r for i in range(-N_half, N_half + 1)])
    r_arr = r_arr[r_arr > r_plus * 1.001]

    rstar_arr = np.array([tortoise_RN(r, r_s, r_Q_sq) for r in r_arr])
    V_arr = np.array([V_func(r) for r in r_arr])

    x = rstar_arr - r_star_0

    deg = min(6, len(x) - 1)
    coeffs = np.polyfit(x, V_arr, deg)
    a = coeffs[::-1]

    V2 = 2.0 * a[2] if len(a) > 2 else 0.0
    V3 = 6.0 * a[3] if len(a) > 3 else 0.0
    V4 = 24.0 * a[4] if len(a) > 4 else 0.0

    return V0, V2, V3, V4


def wkb_qnm_1st_order(V0: float, V2: float, n: int = 0) -> complex:
    """1st-order WKB formula for QNM frequencies (Schutz-Will 1985).

    omega^2 = V0 - i*(n+1/2)*sqrt(-2*V2)

    Returns dimensionful omega (Re > 0, Im < 0).
    """
    if V2 >= 0:
        return complex(np.nan, np.nan)
    alpha = n + 0.5
    omega_sq = V0 - 1j * alpha * np.sqrt(-2.0 * V2)
    omega = np.sqrt(omega_sq)
    if omega.real < 0:
        omega = -omega
    return omega


def wkb_qnm_2nd_order(V0: float, V2: float, V3: float, V4: float,
                        n: int = 0) -> complex:
    """2nd-order WKB formula (Iyer-Will 1987).

    omega^2 = V0 - i*alpha*sqrt(-2*V2)*(1 + Lambda_2)

    where Lambda_2 = (1/(-2V2)) * [(1/8)(V4/V2)(1/4+alpha^2)
                                     - (1/288)(V3/V2)^2*(7+60*alpha^2)]

    Returns dimensionful omega (Re > 0, Im < 0).
    """
    if V2 >= 0:
        return complex(np.nan, np.nan)

    alpha = n + 0.5
    inv_2V2 = 1.0 / (-2.0 * V2)
    term1 = (1.0 / 8.0) * (V4 / V2) * (0.25 + alpha**2)
    term2 = -(1.0 / 288.0) * (V3 / V2)**2 * (7.0 + 60.0 * alpha**2)
    Lambda_2 = inv_2V2 * (term1 + term2)

    omega_sq = V0 - 1j * alpha * np.sqrt(-2.0 * V2) * (1.0 + Lambda_2)
    omega = np.sqrt(omega_sq)
    if omega.real < 0:
        omega = -omega
    return omega


def compute_rn_qnm(r_s: float, r_Q_sq: float, l: int = 2, n: int = 0,
                    use_sct: bool = False,
                    m2: float = 0.0, m0: float = 0.0) -> dict:
    """Compute QNM frequency for RN (GR or SCT) via 1st-order WKB.

    Uses 1st-order WKB (Schutz-Will 1985) rather than 2nd-order, because
    the 2nd-order Lambda_2 correction is ill-conditioned for l=2, n=0
    (Lambda_2 ~ -1, cancelling the leading term). The 1st-order formula
    gives ~5-7% accuracy in omega_R, which is sufficient for estimating
    the SCT fractional shift (which is ~10^{-20}).

    Returns dict with omega*M (dimensionless), potential peak, etc.
    """
    M = r_s / 2.0

    r_peak = find_potential_peak_RN(r_s, r_Q_sq, l, use_sct, m2, m0)
    if np.isnan(r_peak):
        return {"omega_M": complex(np.nan, np.nan), "r_peak": np.nan}

    V0, V2, _V3, _V4 = numerical_derivatives_RN(r_peak, r_s, r_Q_sq, l,
                                                  use_sct, m2, m0)

    # Convert to dimensionless units (M = 1)
    V0_dim = V0 * M**2
    V2_dim = V2 * M**4

    omega_dim = wkb_qnm_1st_order(V0_dim, V2_dim, n)

    return {
        "omega_M": omega_dim,
        "r_peak_over_M": r_peak / M,
        "V0_dim": V0_dim,
        "V2_dim": V2_dim,
    }


# ============================================================
# SCT estimates for RN
# ============================================================
def perturb_eq_estimate_RN(omega_M: complex, M_kg: float) -> float:
    """Perturbation-equation fractional shift: c2 * (omega/Lambda)^2."""
    omega_rad = abs(omega_M) * c_light**3 / (G_N * M_kg)
    Lambda_rad = LAMBDA_EV * eV_to_J / hbar
    return C2_COEFF * (omega_rad / Lambda_rad)**2


def log10_metric_mod_RN(M_kg: float, Q_star: float) -> float:
    """log10 of exp(-m2 * r_+) for RN.

    r_+ = M_geom * (1 + sqrt(1 - Q*^2)).
    """
    M_geom = G_N * M_kg / c_light**2
    r_plus = M_geom * (1.0 + np.sqrt(max(1.0 - Q_star**2, 0.0)))
    exponent = M2_INV_M * r_plus
    return -exponent / np.log(10.0)


# ============================================================
# Charge scan
# ============================================================
def run_charge_scan(
    M_solar: float,
    label: str,
    l_val: int = 2,
    n_val: int = 0,
) -> dict:
    """Run a full charge scan Q/M from 0 to 0.99.

    Parameters
    ----------
    M_solar : float
        Black hole mass in solar masses.
    label : str
        Human-readable label.

    Returns
    -------
    result : dict with all computed quantities.
    """
    M_kg = M_solar * M_sun
    r_s = schwarzschild_radius_m(M_kg)
    M_geom = r_s / 2.0

    # Charge grid
    Q_stars = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                        0.6, 0.7, 0.8, 0.9, 0.95, 0.99])

    rows = []
    for Q_star in Q_stars:
        r_Q_sq = (Q_star * M_geom)**2  # Q^2 in geometric units [m^2]

        # GR QNM
        res_gr = compute_rn_qnm(r_s, r_Q_sq, l_val, n_val,
                                 use_sct=False)
        omega_M_gr = res_gr["omega_M"]

        # SCT QNM (metric modification only)
        res_sct = compute_rn_qnm(r_s, r_Q_sq, l_val, n_val,
                                  use_sct=True, m2=M2_INV_M, m0=M0_INV_M)
        omega_M_sct = res_sct["omega_M"]

        # Fractional shift from metric modification
        if abs(omega_M_gr) > 0:
            frac_shift = abs(omega_M_sct - omega_M_gr) / abs(omega_M_gr)
        else:
            frac_shift = np.nan

        # Perturbation-equation estimate
        pe = perturb_eq_estimate_RN(omega_M_gr, M_kg)

        # Metric modification estimate (analytic)
        log10_met = log10_metric_mod_RN(M_kg, Q_star)

        # Horizons
        r_plus = M_geom * (1.0 + np.sqrt(max(1.0 - Q_star**2, 0.0)))

        rows.append({
            "Q_star": float(Q_star),
            "r_plus_over_M": float(r_plus / M_geom),
            "omega_R_M_GR": float(omega_M_gr.real),
            "omega_I_M_GR": float(omega_M_gr.imag),
            "omega_R_M_SCT": float(omega_M_sct.real),
            "omega_I_M_SCT": float(omega_M_sct.imag),
            "frac_shift_wkb": float(frac_shift),
            "perturb_eq_frac": float(pe),
            "perturb_eq_log10": float(np.log10(pe)) if pe > 0 else -np.inf,
            "metric_mod_log10": float(log10_met),
            "r_peak_over_M_GR": float(res_gr.get("r_peak_over_M", np.nan)),
        })

    return {
        "label": label,
        "M_solar": M_solar,
        "mode": {"l": l_val, "n": n_val},
        "rows": rows,
    }


# ============================================================
# Figures
# ============================================================
def plot_rn_charge_scan(scan: dict) -> None:
    """Plot QNM frequencies and SCT shifts vs Q/M.

    Saves: lt3a_rn_charge_scan.pdf / .png
    """
    rows = scan["rows"]
    Q_arr = [r["Q_star"] for r in rows]
    oR_gr = [r["omega_R_M_GR"] for r in rows]
    oI_gr = [-r["omega_I_M_GR"] for r in rows]  # make positive
    pe_log = [r["perturb_eq_log10"] for r in rows]
    met_log = [r["metric_mod_log10"] for r in rows]
    rp_over_M = [r["r_plus_over_M"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Panel (a): QNM frequencies vs Q/M
    ax = axes[0, 0]
    ax.plot(Q_arr, oR_gr, "-o", color="#1f77b4", ms=5, lw=1.5,
            label=r"$\omega_R \cdot M$  (GR)")
    ax.plot(Q_arr, oI_gr, "--s", color="#2ca02c", ms=5, lw=1.5,
            label=r"$|\omega_I| \cdot M$  (GR)")
    ax.set_xlabel(r"$Q/M$", fontsize=13)
    ax.set_ylabel(r"$\omega \cdot M$", fontsize=13)
    ax.set_title("(a) RN QNM frequencies (l=2, n=0)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel (b): r_+/M vs Q/M
    ax = axes[0, 1]
    ax.plot(Q_arr, rp_over_M, "-o", color="#9467bd", ms=5, lw=1.5)
    ax.axhline(2.0, color="blue", ls="--", lw=0.8, alpha=0.5)
    ax.axhline(1.0, color="red", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"$Q/M$", fontsize=13)
    ax.set_ylabel(r"$r_+ / M$", fontsize=13)
    ax.set_title(r"(b) Outer horizon $r_+/M = 1 + \sqrt{1-(Q/M)^2}$",
                 fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.annotate(r"Schwarzschild: $r_+=2M$", xy=(0.3, 2.02),
                fontsize=9, color="blue")
    ax.annotate(r"Extremal: $r_+=M$", xy=(0.5, 1.03),
                fontsize=9, color="red")

    # Panel (c): SCT fractional shift vs Q/M
    ax = axes[1, 0]
    ax.plot(Q_arr, met_log, "-o", color="#d62728", ms=5, lw=1.5,
            label=r"Metric mod: $e^{-m_2 r_+}$")
    ax.plot(Q_arr, pe_log, "--s", color="#1f77b4", ms=5, lw=1.5,
            label=r"Perturb-eq: $c_2(\omega/\Lambda)^2$")
    ax.axhline(-1.0, color="gray", ls=":", lw=1.2,
               label=r"LIGO $\sim 10^{-1}$")
    ax.set_xlabel(r"$Q/M$", fontsize=13)
    ax.set_ylabel(r"$\log_{10}|\delta\omega/\omega|$", fontsize=13)
    ax.set_title(f"(c) SCT correction ({scan['label']})", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel (d): zoom into perturbation-equation estimate
    ax = axes[1, 1]
    ax.plot(Q_arr, pe_log, "-o", color="#1f77b4", ms=5, lw=1.5,
            label=r"$c_2(\omega/\Lambda)^2$")
    ax.axhline(-1.0, color="gray", ls=":", lw=1.2, label=r"LIGO")
    ax.set_xlabel(r"$Q/M$", fontsize=13)
    ax.set_ylabel(r"$\log_{10}|\delta\omega/\omega|_{\rm pert\text{-}eq}$",
                  fontsize=13)
    ax.set_title("(d) Perturbation-equation estimate (zoom)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "lt3a_rn_charge_scan.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / "lt3a_rn_charge_scan.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIG_DIR / 'lt3a_rn_charge_scan.pdf'}")


# ============================================================
# Verification
# ============================================================
def verify(scan: dict) -> list[str]:
    """Run numbered verification checks.

    Returns list of PASS/FAIL strings.
    """
    results = []
    n_check = 0
    rows = scan["rows"]

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
    # 1. Q=0 reduces to Schwarzschild (1st-order WKB: ~5-7% accuracy)
    # ----------------------------------------------------------
    row0 = rows[0]
    # Schwarzschild l=2, n=0 exact: omega*M = 0.3737 - 0.0890i
    # 1st-order WKB gives ~0.399 - 0.088i (7% high in omega_R, 1% in omega_I)
    check(abs(row0["omega_R_M_GR"] - 0.3737) < 0.03,
          f"Q=0: omega_R*M = {row0['omega_R_M_GR']:.4f} ~ 0.374 (1st WKB, ~7%)")
    check(abs(row0["omega_I_M_GR"] + 0.0883) < 0.01,
          f"Q=0: omega_I*M = {row0['omega_I_M_GR']:.4f} ~ -0.088 (1st WKB)")

    # ----------------------------------------------------------
    # 2. r_+(Q=0) = 2M
    # ----------------------------------------------------------
    check(abs(row0["r_plus_over_M"] - 2.0) < 1e-6,
          f"r_+(Q=0)/M = {row0['r_plus_over_M']:.6f} = 2.0")

    # ----------------------------------------------------------
    # 3. r_+ decreases with Q/M
    # ----------------------------------------------------------
    rp_list = [r["r_plus_over_M"] for r in rows]
    check(all(rp_list[i] >= rp_list[i + 1] for i in range(len(rp_list) - 1)),
          "r_+/M monotonically non-increasing with Q/M")

    # ----------------------------------------------------------
    # 4. Extremal limit: r_+(Q/M=0.99) ~ 1.14 M
    # ----------------------------------------------------------
    row_ext = [r for r in rows if abs(r["Q_star"] - 0.99) < 0.01][0]
    rp_ext = row_ext["r_plus_over_M"]
    expected_rp = 1.0 + np.sqrt(1.0 - 0.99**2)
    check(abs(rp_ext - expected_rp) < 0.01,
          f"r_+(Q/M=0.99)/M = {rp_ext:.4f} ~ {expected_rp:.4f}")

    # ----------------------------------------------------------
    # 5. omega_R increases with Q/M (known RN behavior)
    # ----------------------------------------------------------
    oR_list = [r["omega_R_M_GR"] for r in rows]
    # omega_R generally increases with Q for RN (at moderate Q)
    check(oR_list[-1] > oR_list[0],
          f"omega_R(Q/M=0.99) = {oR_list[-1]:.4f} > omega_R(Q=0) = {oR_list[0]:.4f}")

    # ----------------------------------------------------------
    # 6. Metric modification astronomically small at Q=0
    # ----------------------------------------------------------
    met_Q0 = row0["metric_mod_log10"]
    check(met_Q0 < -1e6,
          f"log10(metric mod) at Q=0, 10 Msun = {met_Q0:.2e}")

    # ----------------------------------------------------------
    # 7. Metric modification weaker at high Q (smaller r_+)
    # ----------------------------------------------------------
    met_ext = row_ext["metric_mod_log10"]
    check(met_ext > met_Q0,
          "Metric mod weaker (less negative) at Q/M=0.99 than Q=0")

    # ----------------------------------------------------------
    # 8. Metric modification still astronomically small at Q/M=0.99
    # ----------------------------------------------------------
    check(met_ext < -1e6,
          f"log10(metric mod) at Q/M=0.99 = {met_ext:.2e} (still extreme)")

    # ----------------------------------------------------------
    # 9. Perturbation-equation estimate in expected range
    # ----------------------------------------------------------
    pe_Q0 = row0["perturb_eq_frac"]
    check(1e-25 < pe_Q0 < 1e-15,
          f"Perturb-eq at Q=0 = {pe_Q0:.2e}")

    # ----------------------------------------------------------
    # 10. Perturbation-equation dominates metric modification
    # ----------------------------------------------------------
    pe_log_Q0 = row0["perturb_eq_log10"]
    check(pe_log_Q0 > met_Q0,
          "Perturbation-eq dominates metric modification by many orders")

    # ----------------------------------------------------------
    # 11. f_RN(r=r_+) = 0 (horizon condition)
    # ----------------------------------------------------------
    M_kg = scan["M_solar"] * M_sun
    r_s = schwarzschild_radius_m(M_kg)
    M_geom = r_s / 2.0
    for Q_star in [0.0, 0.5, 0.9]:
        r_Q_sq = (Q_star * M_geom)**2
        r_plus = M_geom * (1.0 + np.sqrt(1.0 - Q_star**2))
        f_at_rp = f_RN_GR(r_plus, r_s, r_Q_sq)
        check(abs(f_at_rp) < 1e-10,
              f"f_RN(r_+, Q/M={Q_star}) = {f_at_rp:.2e} ~ 0")

    # ----------------------------------------------------------
    # 14. SCT metric at Q=0 matches Schwarzschild SCT
    # ----------------------------------------------------------
    r_test = 3.0 * r_s
    f_sct_rn_Q0 = f_RN_SCT(r_test, r_s, 0.0, M2_INV_M, M0_INV_M)
    h_test = 1.0 - (4.0 / 3.0) * np.exp(-M2_INV_M * r_test) \
             + (1.0 / 3.0) * np.exp(-M0_INV_M * r_test)
    f_sct_sch = 1.0 - (r_s / r_test) * h_test
    check(abs(f_sct_rn_Q0 - f_sct_sch) < 1e-14,
          f"f_SCT_RN(Q=0) = f_SCT_Sch at r=3*r_s: diff = "
          f"{abs(f_sct_rn_Q0 - f_sct_sch):.2e}")

    # ----------------------------------------------------------
    # 15. WKB SCT omega at Q=0 matches GR to numerical precision
    # ----------------------------------------------------------
    # The metric modification is so small that WKB cannot distinguish
    frac_Q0 = row0["frac_shift_wkb"]
    check(frac_Q0 < 1e-6,
          f"WKB frac shift at Q=0 = {frac_Q0:.2e} (< 1e-6, numerical floor)")

    # ----------------------------------------------------------
    # 16. All perturbation-equation shifts unmeasurable
    # ----------------------------------------------------------
    all_pe = [r["perturb_eq_frac"] for r in rows]
    check(max(all_pe) < 1e-10,
          f"Max perturb-eq shift = {max(all_pe):.2e} << LIGO")

    # ----------------------------------------------------------
    # 17. Potential peak location sensible
    # ----------------------------------------------------------
    rp_peak_Q0 = row0["r_peak_over_M_GR"]
    check(2.5 < rp_peak_Q0 < 4.0,
          f"V_peak at Q=0: r_peak/M = {rp_peak_Q0:.2f} (expected ~3)")

    # ----------------------------------------------------------
    # 18. RN horizon formula: r_+*r_- = Q^2, r_+ + r_- = 2M
    # ----------------------------------------------------------
    for Q_star in [0.3, 0.7, 0.9]:
        r_p = M_geom * (1.0 + np.sqrt(1.0 - Q_star**2))
        r_m = M_geom * (1.0 - np.sqrt(1.0 - Q_star**2))
        check(abs(r_p + r_m - 2 * M_geom) < 1e-10 * M_geom,
              f"r_+ + r_- = 2M at Q/M={Q_star}: diff = "
              f"{abs(r_p + r_m - 2*M_geom)/M_geom:.2e}")
        Q_geom = Q_star * M_geom
        check(abs(r_p * r_m - Q_geom**2) / Q_geom**2 < 1e-10,
              f"r_+*r_- = Q^2 at Q/M={Q_star}: rel err = "
              f"{abs(r_p*r_m - Q_geom**2)/Q_geom**2:.2e}")

    # Summary
    n_pass = sum("PASS" in r for r in results)
    n_fail = sum("FAIL" in r for r in results)
    print(f"\n  Summary: {n_pass}/{n_check} PASS, {n_fail} FAIL")
    return results


# ============================================================
# Main
# ============================================================
def main() -> dict:
    """Run the full RN QNM analysis."""
    t0 = time.time()
    print("=" * 70)
    print("LT-3a Phase 8: Reissner-Nordstrom QNM Analysis for SCT")
    print("=" * 70)

    # ----------------------------------------------------------
    # 1. Charge scan for 10 M_sun
    # ----------------------------------------------------------
    print("\n[1/4] Computing RN QNM charge scan: 10 M_sun ...")
    scan_10 = run_charge_scan(10.0, r"$10\,M_\odot$")

    # ----------------------------------------------------------
    # 2. Print table
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("KEY RESULTS: RN QNMs (l=2, n=0, M = 10 M_sun)")
    print("=" * 70)

    print(f"\n{'Q/M':>5s}  {'r+/M':>6s}  {'omega_R*M':>10s}  {'omega_I*M':>10s}"
          f"  {'log10(pe)':>10s}  {'log10(met)':>12s}")
    print("-" * 70)

    for r in scan_10["rows"]:
        print(f"{r['Q_star']:5.2f}  {r['r_plus_over_M']:6.3f}"
              f"  {r['omega_R_M_GR']:10.6f}  {r['omega_I_M_GR']:10.6f}"
              f"  {r['perturb_eq_log10']:10.2f}  {r['metric_mod_log10']:12.2e}")

    # ----------------------------------------------------------
    # 3. Figure
    # ----------------------------------------------------------
    print("\n[2/4] Generating figure ...")
    plot_rn_charge_scan(scan_10)

    # ----------------------------------------------------------
    # 4. Verification
    # ----------------------------------------------------------
    print("\n[3/4] Running verification ...")
    vr_results = verify(scan_10)

    # ----------------------------------------------------------
    # 5. Save JSON
    # ----------------------------------------------------------
    output = {
        "description": "LT-3a Phase 8: Reissner-Nordstrom QNM analysis for SCT",
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
            "mode": scan_10["mode"],
            "rows": scan_10["rows"],
        },
        "conclusion": (
            "ALL RN black holes: SCT is indistinguishable from GR. "
            "The charge reduces r_+, weakening the metric-modification "
            "suppression, but exp(-m2*r_+) remains astronomically small "
            "even at the extremal limit Q/M -> 1. The perturbation-equation "
            "estimate c2*(omega/Lambda)^2 ~ 10^{-20} dominates and is "
            "independent of charge to leading order."
        ),
        "caveat": (
            "The metric modification is computed by substituting f_SCT into "
            "the GR Regge-Wheeler formula. The perturbation-equation "
            "correction (dominant, ~10^{-20}) is estimated but not derived "
            "from first principles (blocked by OP-01/Gap G1). "
            "The Yukawa correction applies only to the gravitational (mass) "
            "part; the electromagnetic r_Q^2/r^2 term is not modified by SCT."
        ),
        "verification": vr_results,
    }

    json_path = RES_DIR / "lt3a_rn_qnm.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {json_path}")

    dt = time.time() - t0
    print(f"\n[4/4] Total time: {dt:.1f} s")
    print("=" * 70)

    return output


if __name__ == "__main__":
    main()

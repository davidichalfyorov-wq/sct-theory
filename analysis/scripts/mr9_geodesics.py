# ruff: noqa: E402, I001
"""
MR-9 GEODESICS: Geodesic completeness analysis for SCT black holes.

Investigates whether radial geodesics reach r=0 in finite proper time
in the SCT-modified Schwarzschild metric.

METRIC:
    ds² = -f(r) dt² + f(r)⁻¹ dr² + r² dΩ²
    f(r) = 1 - (r_s/r) h(r)
    h(r) = 1 - (4/3)e^{-m₂r} + (1/3)e^{-m₀r}     [Yukawa, Level 2]

KEY ANALYSIS:

1. L'HOPITAL LIMIT:
   h(0) = 0, h'(0) = 4m₂/3 - m₀/3
   f(r→0) = 1 - r_s · h'(0) + O(r)

   For astrophysical BH (r_s >> 1/Λ):
   f(0) = 1 - r_s · h'(0) << 0  (deep inside horizon)

   For Planck-scale BH (r_s ~ 1/Λ):
   f(0) may approach 1 (regular center)

2. GEODESIC EQUATION (radial, E=1, L=0):
   (dr/dτ)² = 1 - f(r)

   τ(r₀→0) = ∫₀^r₀ dr / √(1 - f(r))

   - Schwarzschild: τ finite → geodesically INCOMPLETE
   - SCT: τ depends on f(r→0)

3. NONLINEARITY RADIUS:
   r_NL ~ 1/Λ: below this, linearized analysis breaks down
   Curvature ~ Λ⁴ at r ~ 1/Λ: perturbative expansion diverges

RESULT:
   For r_s · h'(0) > 1: f(0) < 0, geodesics reach r=0 in FINITE time
   (same qualitative behavior as Schwarzschild, LINEARIZED ONLY).
   Nonlinear corrections at r < r_NL may change this conclusion.

Author: David Alfyorov
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from sct_tools.plotting import SCT_COLORS, create_figure, init_style, save_figure

from scripts.nt4a_propagator import (
    scalar_local_mass,
    spin2_local_mass,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "analysis" / "results" / "mr9"
FIGURES_DIR = PROJECT_ROOT / "analysis" / "figures" / "mr9"

DEFAULT_DPS = 50


# ===========================================================================
# SECTION 1: METRIC FUNCTIONS
# ===========================================================================

def h_yukawa(r, *, m2, m0=None, dps=DEFAULT_DPS):
    """Yukawa modification factor h(r) = 1 - (4/3)e^{-m₂r} + (1/3)e^{-m₀r}.

    h(0) = 0 exactly (both modes present).
    h(∞) = 1 (Newtonian recovery).
    """
    mp.mp.dps = dps
    r_mp = mp.mpf(r)
    result = 1 - mp.mpf(4) / 3 * mp.exp(-mp.mpf(m2) * r_mp)
    if m0 is not None:
        result += mp.mpf(1) / 3 * mp.exp(-mp.mpf(m0) * r_mp)
    return result


def h_prime_at_zero(*, m2, m0=None, dps=DEFAULT_DPS):
    """Compute h'(0) = 4m₂/3 - m₀/3.

    This is the L'Hopital coefficient: f(r→0) = 1 - r_s · h'(0).
    """
    mp.mp.dps = dps
    hp = mp.mpf(4) / 3 * mp.mpf(m2)
    if m0 is not None:
        hp -= mp.mpf(1) / 3 * mp.mpf(m0)
    return hp


def h_series(r, *, m2, m0=None, n_terms=6, dps=DEFAULT_DPS):
    """Taylor series of h(r) around r=0 to order n_terms.

    h(r) = Σ_k h_k r^k where:
    h_0 = 0 (exact cancellation)
    h_1 = 4m₂/3 - m₀/3
    h_2 = -(4m₂²/3 - m₀²/3)/2
    h_k = [(-1)^{k+1} (4m₂^k/3 - m₀^k/3)] / k!
    """
    mp.mp.dps = dps
    r_mp = mp.mpf(r)
    m2_mp = mp.mpf(m2)
    m0_mp = mp.mpf(m0) if m0 is not None else mp.mpf(0)

    result = mp.mpf(0)
    for k in range(1, n_terms + 1):
        coeff = ((-1) ** (k + 1)) * (
            mp.mpf(4) / 3 * m2_mp ** k - mp.mpf(1) / 3 * m0_mp ** k
        ) / mp.factorial(k)
        result += coeff * r_mp ** k
    return result


def f_metric(r, *, r_s, m2, m0=None, dps=DEFAULT_DPS):
    """Modified Schwarzschild metric function f(r) = 1 - (r_s/r)h(r).

    Uses L'Hopital-safe computation near r=0.
    """
    mp.mp.dps = dps
    r_mp = mp.mpf(r)

    if r_mp < mp.mpf("1e-10"):
        # Use Taylor series: h(r)/r = h'(0) + h''(0)r/2 + ...
        hp0 = h_prime_at_zero(m2=m2, m0=m0, dps=dps)
        h_over_r = hp0 + h_series_divided_by_r_minus_hp0(r, m2=m2, m0=m0, dps=dps)
        return 1 - mp.mpf(r_s) * h_over_r

    h_val = h_yukawa(r, m2=m2, m0=m0, dps=dps)
    return 1 - mp.mpf(r_s) / r_mp * h_val


def h_series_divided_by_r_minus_hp0(r, *, m2, m0=None, dps=DEFAULT_DPS):
    """Compute [h(r)/r - h'(0)] using Taylor series (avoids 0/0).

    h(r)/r = h'(0) + h_2 r + h_3 r² + ...
    Returns the O(r) correction: h_2 r + h_3 r² + ...
    """
    mp.mp.dps = dps
    r_mp = mp.mpf(r)
    m2_mp = mp.mpf(m2)
    m0_mp = mp.mpf(m0) if m0 is not None else mp.mpf(0)

    result = mp.mpf(0)
    for k in range(2, 8):
        coeff = ((-1) ** (k + 1)) * (
            mp.mpf(4) / 3 * m2_mp ** k - mp.mpf(1) / 3 * m0_mp ** k
        ) / mp.factorial(k)
        result += coeff * r_mp ** (k - 1)
    return result


def f_at_zero(*, r_s, m2, m0=None, dps=DEFAULT_DPS):
    """f(r→0) = 1 - r_s · h'(0) via L'Hopital."""
    hp0 = h_prime_at_zero(m2=m2, m0=m0, dps=dps)
    return 1 - mp.mpf(r_s) * hp0


# ===========================================================================
# SECTION 2: NONLINEARITY RADIUS
# ===========================================================================

def nonlinearity_radius(Lambda=1.0, dps=DEFAULT_DPS):
    """Estimate the radius below which linearization breaks down.

    r_NL ~ 1/Λ: at this scale, curvature corrections from the nonlocal
    form factors become O(1), and the linearized field equations are
    no longer reliable.

    More precisely, the linearized potential gives:
    V(r)/V_N(r) = h(r), so the curvature correction is:
    δR/R_GR ~ (1-h) ~ exp(-m₂r) for r >> 1/Λ
    This becomes O(1) when m₂r ~ 1, i.e., r ~ 1/m₂ = √(c₂)/Λ ~ 0.47/Λ.
    """
    mp.mp.dps = dps
    lam = mp.mpf(Lambda)
    m2 = spin2_local_mass(lam)
    return 1 / m2  # ~ 0.465/Lambda


def kretschner_at_r_NL(*, Lambda=1.0, M=1.0, G=1.0, dps=DEFAULT_DPS):
    """Estimate Kretschner scalar at r = r_NL.

    K(r_NL) ~ 48(GM)² / r_NL⁶ · (softening factor)

    If K(r_NL) >> Λ⁴, the curvature exceeds the cutoff scale and
    the effective field theory breaks down. This signals where
    nonlinear corrections become essential.
    """
    mp.mp.dps = dps
    r_NL = nonlinearity_radius(Lambda=Lambda, dps=dps)
    lam = mp.mpf(Lambda)
    GM = mp.mpf(G) * mp.mpf(M)

    # GR Kretschner at r_NL
    K_GR = 48 * GM ** 2 / r_NL ** 6

    # SCT softening: at r_NL ~ 1/m₂, h(r_NL) ~ 1 - e^{-1} ~ 0.63
    h_val = float(h_yukawa(float(r_NL), m2=float(spin2_local_mass(lam)),
                           m0=float(scalar_local_mass(lam, mp.mpf(0))),
                           dps=dps))
    K_SCT_approx = K_GR * h_val ** 2  # rough estimate

    # Compare with Λ⁴
    K_cutoff = lam ** 4

    return {
        "r_NL": float(r_NL),
        "r_NL_over_r_s": float(r_NL / (2 * GM)) if GM > 0 else float("inf"),
        "K_GR_at_r_NL": float(K_GR),
        "K_SCT_approx_at_r_NL": float(K_SCT_approx),
        "Lambda_4": float(K_cutoff),
        "K_over_Lambda4": float(K_GR / K_cutoff),
        "linearization_valid": float(K_GR / K_cutoff) < 1,
        "interpretation": (
            "If K(r_NL) >> Λ⁴, linearization breaks down at r_NL. "
            "Nonlinear (NT-4b) corrections are required for r < r_NL. "
            "The Koshelev-Tokareva theorem (2404.07925) argues that these "
            "corrections forbid the singularity for entire functions of order ≥ 1/2."
        ),
    }


# ===========================================================================
# SECTION 3: GEODESIC INTEGRATION
# ===========================================================================

@dataclass
class GeodesicResult:
    """Result of geodesic integration."""
    r_values: list[float]
    tau_values: list[float]
    f_values: list[float]
    proper_time_total: float
    reaches_zero: bool
    geodesically_complete: bool
    interpretation: str


def integrate_radial_geodesic(
    *,
    r_start: float,
    r_end: float = 0.001,
    r_s: float = 100.0,
    m2: float | None = None,
    m0: float | None = None,
    Lambda: float = 1.0,
    xi: float = 0.0,
    E: float = 1.0,
    L: float = 0.0,
    n_steps: int = 10000,
    dps: int = DEFAULT_DPS,
    metric_type: str = "sct",
) -> GeodesicResult:
    """Integrate radial geodesic equation and compute proper time.

    For radial geodesic (L=0) with energy E:
        (dr/dτ)² = E² - f(r)   [timelike, radial infall]

    Proper time from r_start to r_end:
        τ = ∫_{r_end}^{r_start} dr / √(E² - f(r))

    Parameters:
        r_start: starting radius (must be inside horizon for infall)
        r_end: ending radius (> 0 to avoid 0/0)
        r_s: Schwarzschild radius 2GM
        m2, m0: Yukawa masses (computed from Lambda, xi if None)
        Lambda: spectral cutoff
        xi: Higgs non-minimal coupling
        E: specific energy (1 for marginally bound)
        L: specific angular momentum (0 for radial)
        n_steps: integration steps
        metric_type: "sct", "schwarzschild", or "idg"
    """
    mp.mp.dps = dps

    if m2 is None:
        m2 = float(spin2_local_mass(mp.mpf(Lambda)))
    if m0 is None:
        m0_mp = scalar_local_mass(mp.mpf(Lambda), mp.mpf(xi))
        m0 = float(m0_mp) if m0_mp is not None else None

    r_vals = np.linspace(r_start, r_end, n_steps)
    tau_vals = [0.0]
    f_vals = []

    for i, r in enumerate(r_vals):
        if metric_type == "schwarzschild":
            f_val = 1.0 - r_s / r
        elif metric_type == "idg":
            import scipy.special
            erf_val = scipy.special.erf(r * Lambda / 2)
            f_val = 1.0 - r_s * erf_val / r if r > 1e-15 else 1.0
        else:
            f_val = float(f_metric(r, r_s=r_s, m2=m2, m0=m0, dps=min(dps, 30)))
        f_vals.append(f_val)

        if i > 0:
            dr = abs(r_vals[i] - r_vals[i - 1])
            # Effective potential for radial geodesic
            V_eff = f_val * (1 + L ** 2 / r ** 2) if r > 1e-15 else f_val
            arg = E ** 2 - V_eff
            if arg > 0:
                dtau = dr / np.sqrt(arg)
            else:
                dtau = 0.0  # turning point
            tau_vals.append(tau_vals[-1] + dtau)

    tau_total = tau_vals[-1]
    reaches_zero = r_end < 0.01 * r_start
    geodesically_complete = tau_total > 1e10  # effectively infinite

    # Schwarzschild reference: τ ~ (π/2)·r_s·(r/r_s)^{3/2} for free fall
    tau_schw_approx = (np.pi / 2) * r_s * (r_start / r_s) ** 1.5

    interpretation = (
        f"Proper time from r={r_start:.2f} to r={r_end:.4f}: "
        f"τ = {tau_total:.6f} (Schwarzschild estimate: {tau_schw_approx:.6f}). "
    )
    if tau_total < 1e6:
        interpretation += (
            "FINITE proper time → geodesically INCOMPLETE "
            "(on the linearized metric). "
            "CAVEAT: linearization breaks down at r ~ 1/Λ. "
            "Nonlinear corrections may prevent reaching r=0."
        )
    else:
        interpretation += "Effectively infinite proper time → geodesically COMPLETE."

    return GeodesicResult(
        r_values=r_vals.tolist(),
        tau_values=tau_vals,
        f_values=f_vals,
        proper_time_total=tau_total,
        reaches_zero=reaches_zero,
        geodesically_complete=geodesically_complete,
        interpretation=interpretation,
    )


# ===========================================================================
# SECTION 4: L'HOPITAL ANALYSIS
# ===========================================================================

def lhopital_analysis(*, Lambda=1.0, xi=0.0, dps=DEFAULT_DPS):
    """Detailed L'Hopital analysis of f(r→0).

    f(r) = 1 - (r_s/r) h(r)

    h(r) = h₁ r + h₂ r² + h₃ r³ + ...  (h₀ = 0)

    So h(r)/r = h₁ + h₂ r + h₃ r² + ... is smooth at r=0.

    f(r→0) = 1 - r_s · h₁

    where h₁ = h'(0) = 4m₂/3 - m₀/3.

    For a 10 M_sun BH with Λ = 2.38 meV:
    r_s = 2GM/c² ≈ 29.5 km = 29541 m
    h₁ = 4m₂/3 - m₀/3  in units of Λ
    r_s · h₁ = (29.5 km) · (4m₂/3 - m₀/3) · Λ

    In natural units where Λ = 1:
    r_s · Λ = (2GM/c²) · (Λ/ℏc) ≈ huge number
    """
    mp.mp.dps = dps
    lam = mp.mpf(Lambda)
    m2 = spin2_local_mass(lam)
    m0_mp = scalar_local_mass(lam, mp.mpf(xi))
    m0 = m0_mp if m0_mp is not None else None

    h1 = h_prime_at_zero(m2=float(m2), m0=float(m0) if m0 else None, dps=dps)

    # Second derivative coefficient
    m2_val = mp.mpf(m2)
    m0_val = mp.mpf(m0) if m0 else mp.mpf(0)
    h2 = -(mp.mpf(4) / 3 * m2_val ** 2 - mp.mpf(1) / 3 * m0_val ** 2) / 2

    # Physical masses (using Λ_min = 2.38 meV = 2.38e-3 eV)
    from scipy import constants as const
    Lambda_eV = 2.38e-3  # eV
    Lambda_m_inv = Lambda_eV * const.e / (const.hbar * const.c)  # in m^{-1}
    G_N = const.G
    c = const.c
    M_sun = 1.989e30  # kg

    results = {}
    for M_solar in [1.0, 10.0, 1e6, 1e9]:
        M_kg = M_solar * M_sun
        r_s_m = 2 * G_N * M_kg / c ** 2

        # r_s · h'(0) in natural units: r_s(meters) × Λ(m⁻¹) × h'(0)(Λ units)
        r_s_Lambda = r_s_m * Lambda_m_inv * float(h1)

        f_at_0 = 1 - r_s_Lambda

        # Nonlinearity radius
        r_NL_m = 1 / Lambda_m_inv  # in meters

        results[f"{M_solar:.0e}_Msun"] = {
            "M_solar": M_solar,
            "r_s_m": r_s_m,
            "r_s_Lambda": float(r_s_Lambda),
            "f_at_zero": float(f_at_0),
            "r_NL_m": r_NL_m,
            "r_NL_over_r_s": r_NL_m / r_s_m,
            "regime": "deep_inside_horizon" if f_at_0 < 0 else "regular_center",
        }

    # Critical mass where f(0) = 0: r_s · h'(0) = 1 → M_crit = c²/(2G · Λ · h'(0))
    M_crit_kg = c ** 2 / (2 * G_N * Lambda_m_inv * float(h1))
    M_crit_solar = M_crit_kg / M_sun

    return {
        "h_prime_0": float(h1),
        "h_double_prime_0": float(2 * h2),
        "m2_over_Lambda": float(m2),
        "m0_over_Lambda": float(m0) if m0 else None,
        "f_at_zero_formula": "f(0) = 1 - r_s · h'(0)",
        "critical_mass_solar": M_crit_solar,
        "critical_mass_kg": M_crit_kg,
        "physical_cases": results,
        "interpretation": (
            f"h'(0) = {float(h1):.6f}Λ. "
            f"Critical mass: M_crit = {M_crit_solar:.4e} M_sun. "
            f"For M > M_crit: f(0) < 0 (horizon extends to r=0). "
            f"For M < M_crit: f(0) > 0 (regular center). "
            f"ALL astrophysical BH have M >> M_crit, so f(0) << 0 "
            f"in the linearized metric. "
            f"CAVEAT: linearization breaks down at r ~ 1/Λ = {1/Lambda_m_inv:.2e} m. "
            f"For all BH with r_s >> 1/Λ, the region r < 1/Λ requires "
            f"nonlinear field equations (NT-4b with Θ^(C), blocked by OP-01)."
        ),
    }


# ===========================================================================
# SECTION 5: PROPER TIME COMPARISON
# ===========================================================================

def proper_time_comparison(*, Lambda=1.0, xi=0.0, r_s=100.0, dps=30):
    """Compare proper time to r→0 for SCT, Schwarzschild, and IDG.

    For a free-falling observer from r_start (inside horizon):
    τ_Schw = finite → incomplete
    τ_SCT = ? (linearized)
    τ_IDG = ? (erf metric)
    """
    r_start = r_s * 0.9  # start inside horizon
    r_end = 0.01  # near zero

    results = {}
    for mtype in ["schwarzschild", "sct", "idg"]:
        geo = integrate_radial_geodesic(
            r_start=r_start,
            r_end=r_end,
            r_s=r_s,
            Lambda=Lambda,
            xi=xi,
            n_steps=5000,
            dps=dps,
            metric_type=mtype,
        )
        results[mtype] = {
            "proper_time": geo.proper_time_total,
            "f_at_r_end": geo.f_values[-1] if geo.f_values else None,
            "geodesically_complete": geo.geodesically_complete,
        }

    return results


# ===========================================================================
# SECTION 6: FIGURES
# ===========================================================================

def generate_figures(*, Lambda=1.0, xi=0.0, dps=30):
    """Generate publication figures for geodesic analysis."""
    init_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    m2 = float(spin2_local_mass(mp.mpf(Lambda)))
    m0_mp = scalar_local_mass(mp.mpf(Lambda), mp.mpf(xi))
    m0 = float(m0_mp) if m0_mp is not None else None

    # ---- Figure 1: f(r) for different metrics ----
    fig1, ax1 = create_figure(figsize=(5.5, 3.8))

    r_range = np.logspace(-2, 2.5, 300)
    r_s = 100.0

    # Schwarzschild
    f_schw = [1 - r_s / r for r in r_range]
    ax1.semilogx(r_range, f_schw, 'k:', linewidth=1.0, label='Schwarzschild')

    # SCT
    f_sct = [float(f_metric(r, r_s=r_s, m2=m2, m0=m0, dps=dps)) for r in r_range]
    ax1.semilogx(r_range, f_sct, color=SCT_COLORS['prediction'],
                  linewidth=2.0, label='SCT (Yukawa)')

    # IDG
    import scipy.special
    f_idg = [1 - r_s * scipy.special.erf(r * Lambda / 2) / r
             if r > 1e-10 else 1.0 for r in r_range]
    ax1.semilogx(r_range, f_idg, color=SCT_COLORS['reference'],
                  linewidth=1.5, linestyle='-.', label='IDG (erf)')

    ax1.axhline(y=0, color='gray', linewidth=0.3)
    ax1.axhline(y=1, color='gray', linewidth=0.3, linestyle=':')

    # Mark r_NL
    r_NL = 1 / m2
    ax1.axvline(x=r_NL, color='red', linewidth=0.5, linestyle='--',
                label=f'$r_{{NL}} = {r_NL:.2f}/\\Lambda$')

    ax1.set_xlabel(r'$r \cdot \Lambda$')
    ax1.set_ylabel(r'$f(r)$')
    ax1.set_title(f'Metric function ($r_s\\Lambda = {r_s}$)')
    ax1.legend(fontsize=7)
    ax1.set_ylim(-150, 5)
    fig1.tight_layout()
    save_figure(fig1, "mr9_metric_f_comparison", fmt="pdf", directory=FIGURES_DIR)
    plt.close(fig1)

    # ---- Figure 2: f(r) near r=0 (zoom) ----
    fig2, ax2 = create_figure(figsize=(5.5, 3.8))

    r_small = np.linspace(0.001, 3.0, 300)
    r_s_small = 5.0  # small BH to see structure

    f_schw_s = [1 - r_s_small / r for r in r_small]
    f_sct_s = [float(f_metric(r, r_s=r_s_small, m2=m2, m0=m0, dps=dps)) for r in r_small]
    f_idg_s = [1 - r_s_small * scipy.special.erf(r * Lambda / 2) / r
               if r > 1e-10 else 1.0 for r in r_small]

    ax2.plot(r_small, f_schw_s, 'k:', linewidth=1.0, label='Schwarzschild')
    ax2.plot(r_small, f_sct_s, color=SCT_COLORS['prediction'],
              linewidth=2.0, label='SCT')
    ax2.plot(r_small, f_idg_s, color=SCT_COLORS['reference'],
              linewidth=1.5, linestyle='-.', label='IDG')

    # Mark f(0) for SCT
    f0_sct = float(f_at_zero(r_s=r_s_small, m2=m2, m0=m0, dps=dps))
    ax2.axhline(y=f0_sct, color=SCT_COLORS['prediction'],
                linewidth=0.5, linestyle='--', alpha=0.5)
    ax2.annotate(f'$f(0) = {f0_sct:.2f}$',
                 xy=(0.5, f0_sct), fontsize=7, color=SCT_COLORS['prediction'])

    ax2.axhline(y=0, color='gray', linewidth=0.3)
    ax2.set_xlabel(r'$r \cdot \Lambda$')
    ax2.set_ylabel(r'$f(r)$')
    ax2.set_title(f'Near-origin metric ($r_s\\Lambda = {r_s_small}$)')
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    save_figure(fig2, "mr9_metric_f_near_origin", fmt="pdf", directory=FIGURES_DIR)
    plt.close(fig2)

    # ---- Figure 3: Proper time comparison ----
    fig3, ax3 = create_figure(figsize=(5.5, 3.8))

    r_s_geo = 20.0
    r_start = r_s_geo * 0.9
    r_end_vals = np.linspace(r_start, 0.05, 500)

    for mtype, color, ls, label in [
        ("schwarzschild", "black", ":", "Schwarzschild"),
        ("sct", SCT_COLORS['prediction'], "-", "SCT (Yukawa)"),
    ]:
        taus = []
        tau_cumul = 0.0
        for i in range(1, len(r_end_vals)):
            r1 = r_end_vals[i - 1]
            r2 = r_end_vals[i]
            dr = abs(r2 - r1)
            r_mid = (r1 + r2) / 2

            if mtype == "schwarzschild":
                f_val = 1 - r_s_geo / r_mid
            else:
                f_val = float(f_metric(r_mid, r_s=r_s_geo, m2=m2, m0=m0, dps=20))

            arg = 1 - f_val  # E=1, L=0
            if arg > 0:
                tau_cumul += dr / np.sqrt(arg)
            taus.append(tau_cumul)

        ax3.plot(r_end_vals[1:], taus, color=color, linestyle=ls,
                  linewidth=1.5, label=label)

    ax3.set_xlabel(r'$r \cdot \Lambda$')
    ax3.set_ylabel(r'Proper time $\tau \cdot \Lambda$')
    ax3.set_title(f'Radial infall ($r_s\\Lambda = {r_s_geo}$, $E=1$, $L=0$)')
    ax3.legend(fontsize=8)

    # Mark r_NL
    ax3.axvline(x=r_NL, color='red', linewidth=0.5, linestyle='--',
                alpha=0.5, label=r'$r_{\rm NL}$')

    fig3.tight_layout()
    save_figure(fig3, "mr9_proper_time_comparison", fmt="pdf", directory=FIGURES_DIR)
    plt.close(fig3)

    print(f"Figures saved to {FIGURES_DIR}")
    return [fig1, fig2, fig3]


# ===========================================================================
# SECTION 7: FULL ANALYSIS
# ===========================================================================

def run_full_analysis(*, Lambda=1.0, xi=0.0, dps=DEFAULT_DPS, verbose=True):
    """Run the complete geodesic completeness analysis."""
    report = {
        "phase": "MR-9 Geodesic Completeness Analysis",
        "parameters": {"Lambda": float(Lambda), "xi": float(xi)},
    }

    if verbose:
        print("=" * 72)
        print("MR-9: Geodesic Completeness Analysis")
        print("=" * 72)

    # Step 1: L'Hopital analysis
    if verbose:
        print("\nSTEP 1: L'Hopital Analysis of f(r→0)")
        print("-" * 40)
    lhop = lhopital_analysis(Lambda=Lambda, xi=xi, dps=dps)
    report["lhopital"] = lhop
    if verbose:
        print(f"  h'(0) = {lhop['h_prime_0']:.6f}·Λ")
        print(f"  M_crit = {lhop['critical_mass_solar']:.4e} M_sun")
        for key, val in lhop["physical_cases"].items():
            print(f"  {key}: f(0) = {val['f_at_zero']:.2e}, regime = {val['regime']}")

    # Step 2: Nonlinearity radius
    if verbose:
        print("\nSTEP 2: Nonlinearity Radius")
        print("-" * 40)
    r_NL = nonlinearity_radius(Lambda=Lambda, dps=dps)
    K_analysis = kretschner_at_r_NL(Lambda=Lambda, M=1.0, dps=dps)
    report["nonlinearity_radius"] = K_analysis
    if verbose:
        print(f"  r_NL = {float(r_NL):.6f}/Λ")
        print(f"  K(r_NL)/Λ⁴ = {K_analysis['K_over_Lambda4']:.4e}")
        print(f"  Linearization valid at r_NL: {K_analysis['linearization_valid']}")

    # Step 3: Geodesic integration
    if verbose:
        print("\nSTEP 3: Geodesic Integration")
        print("-" * 40)

    r_s_test = 20.0
    for mtype in ["schwarzschild", "sct"]:
        geo = integrate_radial_geodesic(
            r_start=r_s_test * 0.5,
            r_end=0.01,
            r_s=r_s_test,
            Lambda=Lambda,
            xi=xi,
            n_steps=2000,
            dps=min(dps, 30),
            metric_type=mtype,
        )
        report[f"geodesic_{mtype}"] = {
            "proper_time": geo.proper_time_total,
            "f_at_end": geo.f_values[-1] if geo.f_values else None,
            "complete": geo.geodesically_complete,
            "interpretation": geo.interpretation,
        }
        if verbose:
            print(f"  {mtype}: τ = {geo.proper_time_total:.6f}, "
                  f"complete = {geo.geodesically_complete}")

    # Step 4: Verdict
    verdict = {
        "linearized_result": (
            "On the LINEARIZED SCT metric, radial geodesics reach r→0 in "
            "finite proper time, same qualitative behavior as Schwarzschild. "
            f"The metric function f(0) = 1 - r_s·h'(0) with h'(0) = {lhop['h_prime_0']:.4f}Λ "
            "is deeply negative for all astrophysical black holes."
        ),
        "nonlinearity_caveat": (
            f"Linearization breaks down at r_NL = {float(r_NL):.4f}/Λ. "
            "For r < r_NL, the full nonlinear field equations (NT-4b) with "
            "Θ^(C) are required (blocked by OP-01, Gap G1). "
            "The Koshelev-Tokareva theorem (arXiv:2404.07925) implies that "
            "for entire functions of order ≥ 1/2 (SCT has order 1), "
            "the singular Schwarzschild requires infinite total mass → "
            "singularity is FORBIDDEN at the nonperturbative level."
        ),
        "contradiction_resolution": (
            "The apparent contradiction between our linearized K ~ 1/r⁴ "
            "and the K-T nonperturbative theorem is resolved by recognizing "
            "that the linearized approximation fails precisely where it predicts "
            "singular behavior (r < r_NL). The linearized metric is reliable "
            "only for r >> 1/Λ, where it correctly reproduces GR."
        ),
        "geodesic_completeness": (
            "INDETERMINATE on the linearized level. "
            "The linearized metric gives finite proper time (incomplete), "
            "but this result is unreliable for r < r_NL ~ 1/Λ. "
            "Nonlinear corrections may extend the proper time to infinity."
        ),
    }
    report["verdict"] = verdict

    if verbose:
        print("\n" + "=" * 72)
        print("VERDICT")
        print("=" * 72)
        for key, val in verdict.items():
            print(f"\n  {key}:")
            print(f"    {val}")

    return report


# ===========================================================================
# CLI
# ===========================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="MR-9 Geodesics: Geodesic completeness analysis.")
    parser.add_argument("--xi", type=float, default=0.0)
    parser.add_argument("--dps", type=int, default=DEFAULT_DPS)
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    report = run_full_analysis(Lambda=1.0, xi=args.xi, dps=args.dps)

    output_path = args.output or RESULTS_DIR / "mr9_geodesics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport written to: {output_path}")

    if args.figures:
        print("\nGenerating figures...")
        generate_figures(Lambda=1.0, xi=args.xi, dps=30)

    return report


if __name__ == "__main__":
    main()

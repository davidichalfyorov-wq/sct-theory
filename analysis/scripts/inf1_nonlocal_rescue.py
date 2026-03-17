"""
INF-1 Nonlocal Rescue Analysis
================================

Investigates whether the nonlocal form factors of the spectral action
can rescue Starobinsky inflation by modifying the effective scalaron mass
at inflationary curvature scales.

Key question: At inflationary momenta (z = H^2/Lambda^2), does the
normalized form factor Fhat_2(z) grow enough to reduce the effective
scalaron mass from 15.4 M_Pl to 1.3e-5 M_Pl (ratio ~ 10^6)?

Result: NEGATIVE. Fhat_2(z) peaks at ~1.695 (at z ~ 7.5) and then
decays as 89/(2z) -> 0. The nonlocal correction reduces M by at most
sqrt(1.695) = 1.30, completely negligible compared to the required 10^6.

Author: David Alfyorov
Date: March 2026
"""

import sys
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mpmath import mp, mpf, erfi, sqrt, exp, pi, log

# ============================================================================
# HIGH-PRECISION FORM FACTOR EVALUATION
# ============================================================================

def phi_hp(x, dps=50):
    """Master function phi(x) = int_0^1 exp(-a(1-a)x) da at high precision."""
    old_dps = mp.dps
    mp.dps = dps
    try:
        x = mpf(x)
        if x == 0:
            return mpf(1)
        if x < 0:
            raise ValueError(f"phi_hp: requires x >= 0, got {float(x)}")
        return exp(-x / 4) * sqrt(pi / x) * erfi(sqrt(x) / 2)
    finally:
        mp.dps = old_dps


def hR_scalar_hp(x, xi=0, dps=50):
    """Scalar h_R^(0)(x; xi) at high precision."""
    old_dps = mp.dps
    mp.dps = dps
    try:
        x = mpf(x)
        xi = mpf(xi)
        if x == 0:
            return mpf(1) / 2 * (xi - mpf(1) / 6) ** 2
        p = phi_hp(x, dps)
        fRic = 1 / (6 * x) + (p - 1) / x ** 2
        fR = p / 32 + p / (8 * x) - 7 / (48 * x) - (p - 1) / (8 * x ** 2)
        fRU = -p / 4 - (p - 1) / (2 * x)
        fU = p / 2
        return fRic / 3 + fR + xi * fRU + xi ** 2 * fU
    finally:
        mp.dps = old_dps


def hR_dirac_hp(x, dps=50):
    """Dirac h_R^(1/2)(x) at high precision."""
    old_dps = mp.dps
    mp.dps = dps
    try:
        x = mpf(x)
        if x == 0:
            return mpf(0)
        p = phi_hp(x, dps)
        return (3 * p + 2) / (36 * x) + 5 * (p - 1) / (6 * x ** 2)
    finally:
        mp.dps = old_dps


def hR_vector_hp(x, dps=50):
    """Vector h_R^(1)(x) at high precision."""
    old_dps = mp.dps
    mp.dps = dps
    try:
        x = mpf(x)
        if x == 0:
            return mpf(0)
        p = phi_hp(x, dps)
        return -p / 48 + (11 - 6 * p) / (72 * x) + 5 * (p - 1) / (12 * x ** 2)
    finally:
        mp.dps = old_dps


def alpha_R_total_hp(x, xi=0, dps=50):
    """Total SM alpha_R(x, xi) = 16*pi^2 * F_2(x, xi) at high precision."""
    N_s, N_D, N_v = 4, mpf("22.5"), 12
    return float(
        N_s * hR_scalar_hp(x, xi, dps)
        + N_D * hR_dirac_hp(x, dps)
        + N_v * hR_vector_hp(x, dps)
    )


def Fhat_2(x, xi=0, dps=50):
    """Normalized form factor Fhat_2(x) = alpha_R(x) / alpha_R(0)."""
    aR_0 = alpha_R_total_hp(0, xi, dps)
    if abs(aR_0) < 1e-100:
        raise ValueError("alpha_R(0) = 0 at conformal coupling; Fhat_2 undefined")
    aR_x = alpha_R_total_hp(x, xi, dps)
    return aR_x / aR_0


def scalaron_mass_ratio(xi=0):
    """M/M_Pl for the local Starobinsky scalaron mass.

    M^2 = 2*pi^2 * M_Pl^2 / (3*(xi - 1/6)^2)
    """
    mp.dps = 50
    return float(sqrt(2 * pi ** 2 / 3) / abs(mpf(xi) - mpf(1) / 6))


# ============================================================================
# UV ASYMPTOTICS (ANALYTIC)
# ============================================================================

def uv_asymptotic_alpha_R(xi=0):
    """Leading UV coefficient: lim_{x->inf} x * alpha_R(x, xi).

    Using phi(x) ~ 2/x for large x:
      h_R^(0)(x, xi)  ~ (-1/36 + xi^2) / x
      h_R^(1/2)(x)    ~ 1/(18*x)
      h_R^(1)(x)      ~ 1/(9*x)

    Total: x * alpha_R(x) -> N_s*(-1/36+xi^2) + N_D/18 + N_v/9
         = 4*(-1/36+xi^2) + 22.5/18 + 12/9
         = -1/9 + 5/4 + 4/3 + 4*xi^2
         = 89/36 + 4*xi^2
    """
    return 89 / 36 + 4 * xi ** 2


def uv_asymptotic_Fhat_2(xi=0):
    """Leading UV coefficient: lim_{x->inf} x * Fhat_2(x, xi).

    x * Fhat_2(x) ~ uv_asymptotic_alpha_R(xi) / alpha_R(0, xi)
                   = (89/36 + 4*xi^2) / (2*(xi - 1/6)^2)
    """
    alpha_R_0 = 2 * (xi - 1 / 6) ** 2
    if abs(alpha_R_0) < 1e-30:
        return float("inf")
    return uv_asymptotic_alpha_R(xi) / alpha_R_0


# ============================================================================
# MAIN COMPUTATION
# ============================================================================

def compute_Fhat2_scan(xi=0, x_range=None):
    """Compute Fhat_2(x) over a range of x values."""
    if x_range is None:
        x_range = np.concatenate([
            np.linspace(0.01, 1, 20),
            np.linspace(1, 10, 50),
            np.linspace(10, 100, 40),
            np.logspace(2, 6, 30),
        ])
    results = []
    for x in x_range:
        Fh = Fhat_2(x, xi)
        results.append((x, Fh))
    return np.array(results)


def find_Fhat2_peak(xi=0, x_init=7.5, tol=1e-4):
    """Find the peak of Fhat_2(x) by golden section search."""
    a, b = 0.5, 30.0
    gr = (np.sqrt(5) + 1) / 2
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    while abs(b - a) > tol:
        if Fhat_2(c, xi) > Fhat_2(d, xi):
            b = d
        else:
            a = c
        c = b - (b - a) / gr
        d = a + (b - a) / gr
    x_peak = (a + b) / 2
    return x_peak, Fhat_2(x_peak, xi)


def effective_scalaron_mass(z, xi=0):
    """Effective scalaron mass M_eff(z) / M_Pl.

    M_eff^2(z) = M_Pl^2 / (12 * c_2_eff(z))
               = M_local^2 / Fhat_2(z)
    """
    M_local = scalaron_mass_ratio(xi)
    Fh = Fhat_2(z, xi)
    if Fh <= 0:
        return float("inf")
    return M_local / np.sqrt(Fh)


# ============================================================================
# FIGURE GENERATION
# ============================================================================

def generate_figure(output_path=None):
    """Generate the nonlocal form factor figure."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 11,
        "figure.figsize": (7, 9), "text.usetex": False,
        "mathtext.fontset": "cm",
    })

    fig, axes = plt.subplots(3, 1, figsize=(7, 9))

    # --- Panel 1: Fhat_2(x) ---
    ax1 = axes[0]
    x_lin = np.linspace(0.01, 50, 500)
    Fhat_vals = [Fhat_2(x, xi=0) for x in x_lin]

    ax1.plot(x_lin, Fhat_vals, "b-", linewidth=1.5, label=r"$\hat{F}_2(z, \xi\!=\!0)$")
    ax1.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    # Mark the peak
    x_peak, Fhat_peak = find_Fhat2_peak(xi=0)
    ax1.plot(x_peak, Fhat_peak, "ro", markersize=6, zorder=5)
    ax1.annotate(
        f"Peak: $\\hat{{F}}_2 = {Fhat_peak:.3f}$\nat $z = {x_peak:.1f}$",
        xy=(x_peak, Fhat_peak), xytext=(x_peak + 8, Fhat_peak - 0.15),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=9, color="red",
    )

    # UV asymptotic
    x_uv = np.linspace(15, 50, 100)
    uv_coeff = uv_asymptotic_Fhat_2(xi=0)
    ax1.plot(x_uv, uv_coeff / x_uv, "g--", linewidth=1.0,
             label=f"UV: ${uv_coeff:.1f}/z$")

    ax1.set_xlabel(r"$z = k^2/\Lambda^2$")
    ax1.set_ylabel(r"$\hat{F}_2(z)$")
    ax1.set_title(r"Normalized $R^2$ form factor (SM, $\xi = 0$)")
    ax1.legend(fontsize=9)
    ax1.set_xlim(0, 50)
    ax1.set_ylim(0, 2.0)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Fhat_2(x) on log scale ---
    ax2 = axes[1]
    x_log = np.logspace(-2, 6, 300)
    Fhat_log = [Fhat_2(x, xi=0) for x in x_log]

    ax2.loglog(x_log, Fhat_log, "b-", linewidth=1.5, label=r"$\hat{F}_2(z)$")
    ax2.loglog(x_log, uv_coeff / x_log, "g--", linewidth=1.0,
               label=f"UV: ${uv_coeff:.1f}/z$")
    ax2.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)

    # Mark regions of interest
    ax2.axvspan(1e-10, 1e-5, alpha=0.1, color="blue",
                label=r"PPN-1: $\Lambda \sim$ meV")
    ax2.axvspan(1e40, 1e55, alpha=0.1, color="red",
                label=r"$H_{\rm inf}^2/\Lambda^2$ if $\Lambda \sim$ meV")

    ax2.set_xlabel(r"$z = k^2/\Lambda^2$")
    ax2.set_ylabel(r"$\hat{F}_2(z)$")
    ax2.set_title(r"$\hat{F}_2(z)$ over full range (log-log)")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_xlim(1e-2, 1e6)
    ax2.set_ylim(1e-6, 10)
    ax2.grid(True, alpha=0.3, which="both")

    # --- Panel 3: Effective scalaron mass ---
    ax3 = axes[2]
    x_mass = np.linspace(0.01, 50, 500)
    M_eff_vals = [effective_scalaron_mass(x, xi=0) for x in x_mass]
    M_local = scalaron_mass_ratio(xi=0)
    M_req = 1.282e-5

    ax3.semilogy(x_mass, M_eff_vals, "b-", linewidth=1.5,
                 label=r"$M_{\rm eff}(z)/M_{\rm Pl}$")
    ax3.axhline(y=M_local, color="orange", linestyle="--", linewidth=1.0,
                label=f"Local: $M = {M_local:.1f}\\,M_{{\\rm Pl}}$")
    ax3.axhline(y=M_req, color="red", linestyle="-.", linewidth=1.0,
                label=f"Required: $M_{{\\rm req}} = {M_req:.1e}\\,M_{{\\rm Pl}}$")

    # Mark minimum
    M_min = min(M_eff_vals)
    x_min = x_mass[np.argmin(M_eff_vals)]
    ax3.plot(x_min, M_min, "ro", markersize=6, zorder=5)
    ax3.annotate(
        f"Min: $M_{{\\rm eff}} = {M_min:.1f}\\,M_{{\\rm Pl}}$\n"
        f"Still ${M_min/M_req:.1e}\\times$ too heavy",
        xy=(x_min, M_min), xytext=(x_min + 10, M_min * 0.3),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=9, color="red",
    )

    ax3.set_xlabel(r"$z = k^2/\Lambda^2$")
    ax3.set_ylabel(r"$M_{\rm eff}/M_{\rm Pl}$")
    ax3.set_title("Effective scalaron mass vs. momentum scale")
    ax3.legend(fontsize=9, loc="center right")
    ax3.set_xlim(0, 50)
    ax3.set_ylim(1e-6, 1e3)
    ax3.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(__file__), "..", "figures", "inf1",
            "inf1_nonlocal_form_factor.pdf"
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to: {output_path}")
    return output_path


# ============================================================================
# MAIN: PRINT FULL RESULTS
# ============================================================================

def main():
    mp.dps = 50
    M_req = 1.282e-5  # M_req / M_Pl

    print("=" * 70)
    print("INF-1 NONLOCAL RESCUE ANALYSIS")
    print("Can nonlocal form factors of the spectral action rescue Starobinsky inflation?")
    print("=" * 70)
    print()

    # --- Section 1: Local scalaron mass ---
    print("1. LOCAL SCALARON MASS (from INF-1)")
    print("-" * 40)
    alpha_R_0 = float(2 * (mpf(0) - mpf(1) / 6) ** 2)
    c_2 = alpha_R_0 / (16 * float(pi ** 2))
    M_local = scalaron_mass_ratio(xi=0)
    print(f"  alpha_R(xi=0)     = {alpha_R_0:.8f} = 1/18")
    print(f"  c_2(xi=0)         = {c_2:.6e}")
    print(f"  M/M_Pl            = sqrt(24*pi^2) = {M_local:.6f}")
    print(f"  M_req/M_Pl        = {M_req:.6e}")
    print(f"  Ratio M/M_req     = {M_local / M_req:.4e}")
    print(f"  Enhancement needed: Fhat_2 ~ {(M_local / M_req) ** 2:.2e}")
    print()

    # --- Section 2: UV asymptotics ---
    print("2. UV ASYMPTOTICS")
    print("-" * 40)
    print("  phi(x) ~ 2/x  for x -> inf")
    print()
    print("  Spin-by-spin UV limits of x * h_R^(s)(x):")
    print("    Scalar (s=0):  -1/36 + xi^2")
    print("    Dirac (s=1/2): 1/18")
    print("    Vector (s=1):  1/9")
    print()
    print(f"  Total: x*alpha_R(x) -> 89/36 + 4*xi^2 = {uv_asymptotic_alpha_R(0):.6f} (xi=0)")
    print(f"  Normalized: x*Fhat_2(x) -> {uv_asymptotic_Fhat_2(0):.4f} (xi=0)")
    print()
    print("  KEY: Fhat_2(x) ~ 44.5/x -> 0 as x -> inf")
    print("  The form factor DECREASES at large momenta!")
    print()

    # --- Section 3: Peak of Fhat_2 ---
    print("3. PEAK OF Fhat_2")
    print("-" * 40)
    x_peak, Fhat_peak = find_Fhat2_peak(xi=0)
    print(f"  x_peak            = {x_peak:.4f}")
    print(f"  Fhat_2(x_peak)    = {Fhat_peak:.8f}")
    print(f"  sqrt(Fhat_peak)   = {np.sqrt(Fhat_peak):.6f}")
    print(f"  M_eff_min/M_Pl    = {M_local / np.sqrt(Fhat_peak):.4f}")
    print(f"  M_eff_min/M_req   = {M_local / np.sqrt(Fhat_peak) / M_req:.2e}")
    print()

    # --- Section 4: Scan at different xi ---
    print("4. xi DEPENDENCE")
    print("-" * 40)
    print(f"  {'xi':>8s} {'alpha_R(0)':>12s} {'M_local/MPl':>12s} {'Fhat_peak':>10s} {'M_min/MPl':>12s} {'M_min/Mreq':>12s}")
    print("  " + "-" * 68)
    for xi in [0, 0.05, 0.1, 0.5, 1.0, 5.0]:
        try:
            alpha_R_0_xi = 2 * (xi - 1 / 6) ** 2
            M_loc = scalaron_mass_ratio(xi)
            _, Fpeak = find_Fhat2_peak(xi)
            M_min = M_loc / np.sqrt(Fpeak)
            print(f"  {xi:8.2f} {alpha_R_0_xi:12.6f} {M_loc:12.4f} {Fpeak:10.4f} {M_min:12.4f} {M_min / M_req:12.2e}")
        except Exception as e:
            print(f"  {xi:8.2f}  [Error: {e}]")
    print()

    # --- Section 5: The physical scenario ---
    print("5. PHYSICAL SCENARIO CHECK")
    print("-" * 40)
    print("  If Lambda ~ 2.565 meV (from PPN-1):")
    Lambda_eV = 2.565e-3  # eV
    MPl_eV = 2.435e27  # eV
    Lambda_MPl = Lambda_eV / MPl_eV
    print(f"    Lambda/M_Pl = {Lambda_MPl:.4e}")
    print(f"    m_0 = sqrt(6)*Lambda = {np.sqrt(6) * Lambda_eV:.4e} eV")
    print(f"    m_0/M_Pl = {np.sqrt(6) * Lambda_MPl:.4e}")
    print()
    print("    In this scenario, the scalaron mass from the EFT is:")
    print(f"    M = sqrt(24*pi^2)*M_Pl = {M_local:.2f} M_Pl")
    print("    BUT this assumes Lambda = M_Pl.")
    print()
    print("    The CORRECT interpretation: the spectral action EFT has TWO scales:")
    print("      (a) M_Pl from the Einstein-Hilbert term (external input in EFT)")
    print("      (b) Lambda from the spectral cutoff (sets nonlocality scale)")
    print()
    print("    The scalaron mass M^2 = M_Pl^2/(12*c_2) depends on M_Pl, not Lambda.")
    print("    The nonlocal form factor F_2(Box/Lambda^2) dresses the R^2 vertex")
    print("    but does NOT change the local coefficient c_2 that determines M.")
    print()

    H_inf_eV = 1.56e22  # eV (inflationary Hubble)
    z_H = (H_inf_eV / Lambda_eV) ** 2
    print(f"    z_H = H_inf^2/Lambda^2 = ({H_inf_eV:.2e})^2 / ({Lambda_eV:.2e})^2 = {z_H:.2e}")
    print(f"    At this z, Fhat_2 ~ 44.5/z = {44.5 / z_H:.2e}")
    print("    The form factor is essentially ZERO at inflationary scales.")
    print("    This means the R^2 dressing VANISHES -- no scalaron at all!")
    print()

    # --- Section 6: Verdict ---
    print("=" * 70)
    print("VERDICT: NEGATIVE")
    print("=" * 70)
    print()
    print("The nonlocal form factors of the spectral action CANNOT rescue")
    print("Starobinsky inflation. Four independent arguments confirm this:")
    print()
    print("  (1) MATHEMATICAL: Fhat_2(z) has a bounded peak of 1.695 at z=7.5,")
    print("      reducing M by only 30%. The required reduction is 10^6.")
    print()
    print("  (2) UV DECAY: Fhat_2(z) ~ 44.5/z -> 0 as z -> inf.")
    print("      At inflationary curvatures, the nonlocal correction vanishes.")
    print()
    print("  (3) PHYSICAL (KKS): The background inflationary solution obeys")
    print("      Box R = M^2 R with M determined by the LOCAL c_2.")
    print("      Nonlocal corrections modify perturbations, not the background.")
    print()
    print("  (4) STRUCTURAL: One-loop form factors universally decay as 1/z")
    print("      in the UV due to decoupling of quantum corrections.")
    print()
    print("The scalaron mass problem is a LOCAL problem: alpha_R(xi) = 2*(xi-1/6)^2")
    print("is too small at natural xi. Nonlocal dressing cannot fix this.")
    print()

    # --- Generate figure ---
    print("Generating figure...")
    fig_path = generate_figure()
    print(f"Done. Figure at: {fig_path}")
    print()

    return {
        "status": "NEGATIVE",
        "Fhat_2_peak": Fhat_peak,
        "x_peak": x_peak,
        "M_eff_min_MPl": M_local / np.sqrt(Fhat_peak),
        "M_local_MPl": M_local,
        "M_req_MPl": M_req,
        "ratio_best": M_local / np.sqrt(Fhat_peak) / M_req,
        "uv_coeff": uv_asymptotic_Fhat_2(0),
    }


if __name__ == "__main__":
    results = main()

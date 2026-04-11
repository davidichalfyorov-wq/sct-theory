#!/usr/bin/env python3
"""
Paper 9 (JETP): Generate 3 publication figures for ghost suppression paper.

Fig 1: Suppression exponents vs M/M_sun
Fig 2: M_min(xi) vs M_crit (ghost-active window)
Fig 3: T_Kerr/T_Sch vs a/M

All numbers independently computed from scipy.constants + mpmath.
Zero project imports.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
from scipy import constants as const

mp.mp.dps = 30

# ---- Constants ----
G = float(const.G)
hbar = float(const.hbar)
c = float(const.c)
kB = float(const.k)
eV_J = float(const.eV)
Msun = 1.98892e30  # kg

z_L = 1.2807
Lambda_eV = 2.38e-3
m_ghost_eV = np.sqrt(z_L) * Lambda_eV  # 2.69 meV
m2_over_L = np.sqrt(60.0 / 13.0)       # 2.148
m0_over_L = np.sqrt(6.0)               # 2.449

lP = np.sqrt(hbar * G / c**3)
mP_eV = np.sqrt(hbar * c / G) * c**2 / eV_J

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures", "paper9")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "figure.dpi": 150,
    "text.usetex": False, "font.family": "serif",
})


def T_H_eV(M_kg):
    """Hawking temperature [eV]."""
    return hbar * c**3 / (8 * np.pi * G * M_kg) / eV_J


def r_H_m(M_kg):
    """Schwarzschild horizon radius [m]."""
    return 2 * G * M_kg / c**2


def boltzmann_exp(M_kg):
    """m_ghost / T_H."""
    return m_ghost_eV / T_H_eV(M_kg)


def yukawa_exp(M_kg):
    """m_ghost * r_H."""
    m_inv_m = m_ghost_eV * eV_J / (hbar * c)
    return m_inv_m * r_H_m(M_kg)


def M_crit_kg():
    """Critical mass [kg]."""
    m_J = m_ghost_eV * eV_J
    return hbar * c**3 / (8 * np.pi * G * m_J)


# ============================================================
# Figure 1: Suppression exponents vs M/M_sun
# ============================================================
def fig1():
    M_arr = np.logspace(-6, 11, 200) * Msun
    bz = np.array([boltzmann_exp(M) for M in M_arr])
    yk = np.array([yukawa_exp(M) for M in M_arr])
    M_msun = M_arr / Msun

    Mc = M_crit_kg() / Msun

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(M_msun, bz, "-", lw=2, color="C0",
              label=r"Больцман: $m/T_H$")
    ax.loglog(M_msun, yk, "--", lw=2, color="C1",
              label=r"Юкава: $m \cdot r_H$")
    ax.axhline(1, color="red", ls=":", lw=1, alpha=0.7)
    ax.axvline(Mc, color="gray", ls="--", lw=1, alpha=0.7)
    ax.text(Mc * 2, 3, r"$M_{\rm crit}$", fontsize=9, color="gray")

    # Shade astrophysical region
    ax.axvspan(3, 1e11, alpha=0.08, color="green")
    ax.text(1e4, 1e2, r"все наблюдаемые ЧД", fontsize=8, color="green",
            ha="center", alpha=0.7)

    # Mark observed BHs
    obs = [
        ("Cyg X-1", 21.2),
        ("Sgr A*", 4.15e6),
        ("M87*", 6.5e9),
        ("TON 618", 6.6e10),
    ]
    for name, m_msun in obs:
        bz_val = boltzmann_exp(m_msun * Msun)
        ax.plot(m_msun, bz_val, "D", ms=5, color="navy", zorder=5)
        ax.annotate(name, (m_msun * 1.3, bz_val * 0.5),
                    fontsize=7, color="navy")

    ax.set_xlabel(r"$M / M_\odot$")
    ax.set_ylabel(r"Показатель пода��ления")
    ax.set_xlim(1e-6, 1e11)
    ax.set_ylim(1e-2, 1e20)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    path = os.path.join(FIGDIR, "fig1_suppression_exponents.pdf")
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close()


# ============================================================
# Figure 2: M_min(xi) vs M_crit
# ============================================================
def fig2():
    # M_min(xi) = M_Pl^2 / (2 * m2 * sup f_q(x))
    # For xi < 1/6: q = m0/m2 where m0 = Lambda / sqrt(6|xi - 1/6|^2)...
    # Actually: m0 = Lambda * sqrt(1/(6*(xi-1/6)^2))... no.
    # m0^2 = Lambda^2 / (6*(xi-1/6)^2)... wait.
    # From SCT: alpha_R = 2*(xi-1/6)^2, scalar mass m0 = Lambda*sqrt(6*(xi-1/6)^2)...
    # Actually: m0 = Lambda/sqrt(scalar_mode_mass) where scalar_mode_mass = 3*alpha_R
    # = 6*(xi-1/6)^2. So m0^2 = Lambda^2/(6*(xi-1/6)^2).
    # Wait no. From PPN-1: m0 = Lambda*sqrt(6) at xi=0. And m0 -> inf at xi=1/6.
    # General: m0(xi) = Lambda / sqrt(2*(xi-1/6)^2 * something)...
    #
    # From the memory: m_0(xi) = Lambda / sqrt(6*(xi-1/6)^2)... that gives m0(0) = Lambda/sqrt(6*1/36) = Lambda/sqrt(1/6) = Lambda*sqrt(6). Yes!
    # m_0 = Lambda / sqrt(6*(xi - 1/6)^2) = Lambda / (sqrt(6)*|xi - 1/6|)
    # q = m0/m2 = 1/(sqrt(6)*|xi-1/6|*m2/Lambda) = 1/(sqrt(6)*|xi-1/6|*sqrt(60/13))
    #           = 1/(sqrt(6*60/13)*|xi-1/6|) = 1/(sqrt(360/13)*|xi-1/6|)
    #           = sqrt(13/360) / |xi-1/6|

    m2_L = np.sqrt(60.0 / 13.0)  # m2/Lambda

    def m0_over_m2(xi):
        if abs(xi - 1.0/6.0) < 1e-10:
            return np.inf
        return 1.0 / (m2_L * np.sqrt(6.0) * abs(xi - 1.0/6.0))

    def f_q_max(q):
        """Supremum of f_q(x) = [1 - (4/3)e^{-x} + (1/3)e^{-qx}] / x."""
        if q <= 2:  # supremum at x -> 0+
            return (4.0 - q) / 3.0
        else:
            # Numerical optimization
            from scipy.optimize import minimize_scalar
            def neg_f(x):
                if x < 1e-10:
                    return -(4.0 - q) / 3.0
                return -(1 - 4.0/3.0 * np.exp(-x) + 1.0/3.0 * np.exp(-q*x)) / x
            res = minimize_scalar(neg_f, bounds=(0.001, 10.0), method="bounded")
            return -res.fun

    def M_min_coeff(xi):
        """M_min in units of M_Pl^2 / Lambda."""
        q = m0_over_m2(xi)
        fm = f_q_max(q)
        return 1.0 / (2.0 * m2_L * fm)

    M_crit_coeff = 1.0 / (8.0 * np.pi * np.sqrt(z_L))  # ~ 0.0352

    xi_arr = np.linspace(0, 1.0/6.0 - 0.001, 100)
    Mmin_arr = np.array([M_min_coeff(xi) for xi in xi_arr])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xi_arr, Mmin_arr, "C0-", lw=2, label=r"$M_{\min}(\xi)$")
    ax.axhline(M_crit_coeff, color="red", ls="--", lw=1.5,
               label=r"$M_{\rm crit} = M_{\rm Pl}^2/(8\pi m_{\rm gh})$")

    # Fill the gap
    ax.fill_between(xi_arr, M_crit_coeff, Mmin_arr,
                     alpha=0.15, color="green",
                     label=r"Пустое окно активности")

    # Mark specific points
    ax.plot(0, M_min_coeff(0), "ko", ms=6, zorder=5)
    ax.annotate(f"$\\xi=0$: {M_min_coeff(0):.3f}", (0.005, M_min_coeff(0) + 0.01),
                fontsize=8)

    xi_conf = 1.0/6.0 - 0.001
    ax.plot(xi_conf, M_min_coeff(xi_conf), "ks", ms=6, zorder=5)
    ax.annotate(r"$\xi=1/6$: 0.456", (xi_conf - 0.04, M_min_coeff(xi_conf) + 0.01),
                fontsize=8)

    ax.set_xlabel(r"$\xi$ (неминимальная связь)")
    ax.set_ylabel(r"$M$ в единицах $M_{\rm Pl}^2/\Lambda$")
    ax.set_xlim(-0.005, 1.0/6.0 + 0.005)
    ax.set_ylim(0, 0.55)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    path = os.path.join(FIGDIR, "fig2_Mmin_vs_Mcrit.pdf")
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close()


# ============================================================
# Figure 3: T_Kerr/T_Sch vs a/M
# ============================================================
def fig3():
    a_arr = np.linspace(0, 0.999, 200)

    def T_ratio(a_over_M):
        # In geometrized units: r+ = M + sqrt(M^2 - a^2), M=1
        a = a_over_M
        rp = 1 + np.sqrt(1 - a**2)
        rm = 1 - np.sqrt(1 - a**2)
        kappa_kerr = (rp - rm) / (2 * (rp**2 + a**2))
        kappa_sch = 0.25  # 1/(4M) with M=1
        return kappa_kerr / kappa_sch

    T_arr = np.array([T_ratio(a) for a in a_arr])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(a_arr, T_arr, "C0-", lw=2)
    ax.axhline(1.0, color="red", ls=":", lw=1, alpha=0.5,
               label=r"$T_{\rm Sch}$ (максимум)")
    ax.axhline(0, color="gray", ls="-", lw=0.5, alpha=0.3)

    # Mark the curvature at a=0
    ax.annotate(r"$d^2T/da^2|_{a=0} = -1/8 < 0$" + "\n" + r"(лок. максимум)",
                xy=(0.05, 0.97), fontsize=8, color="C0",
                arrowprops=dict(arrowstyle="->", color="C0"),
                xytext=(0.25, 0.8))

    # Mark extremal
    ax.annotate(r"$T \to 0$ (экстремальная)",
                xy=(0.99, 0.02), fontsize=8, color="C0",
                arrowprops=dict(arrowstyle="->", color="C0"),
                xytext=(0.65, 0.15))

    ax.set_xlabel(r"$a/M$ (параметр вращения)")
    ax.set_ylabel(r"$T_{\rm Kerr} / T_{\rm Sch}$")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.02, 1.1)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    path = os.path.join(FIGDIR, "fig3_kerr_temperature.pdf")
    fig.savefig(path)
    print(f"Saved {path}")
    plt.close()


# ============================================================
# Verify key numbers
# ============================================================
def verify():
    print("=== NUMBER VERIFICATION ===")

    # M_crit
    Mc = M_crit_kg()
    print(f"M_crit = {Mc:.3e} kg = {Mc/Msun:.3e} M_sun")
    assert abs(Mc/Msun - 1.97e-9) < 1e-10

    # Boltzmann at 1 M_sun
    bz = boltzmann_exp(Msun)
    print(f"m/T_H(1 M_sun) = {bz:.3e}")
    assert abs(bz - 5.07e8) < 1e6

    # Yukawa at 1 M_sun
    yk = yukawa_exp(Msun)
    print(f"m*r_H(1 M_sun) = {yk:.3e}")
    assert abs(yk - 4.03e7) < 1e5

    # Ratio
    ratio = yk / bz
    print(f"Yukawa/Boltzmann = {ratio:.8f} (expected 1/(4pi) = {1/(4*np.pi):.8f})")
    assert abs(ratio - 1/(4*np.pi)) < 1e-6

    # M_min coefficients
    m2_L = np.sqrt(60.0/13.0)
    m0_L = np.sqrt(6.0)
    coeff_xi0 = 3.0 / (2.0 * (4*m2_L - m0_L))
    print(f"M_min coeff (xi=0) = {coeff_xi0:.6f} (expected 0.244145)")
    assert abs(coeff_xi0 - 0.244145) < 1e-4

    # M_crit coefficient
    coeff_crit = 1.0 / (8*np.pi*np.sqrt(z_L))
    print(f"M_crit coeff = {coeff_crit:.6f} (expected 0.035159)")
    assert abs(coeff_crit - 0.035159) < 1e-4

    # Ratio
    print(f"M_min/M_crit (xi=0) = {coeff_xi0/coeff_crit:.2f} (expected 6.94)")
    assert abs(coeff_xi0/coeff_crit - 6.94) < 0.1

    # Kerr T at a=0.998
    a = 0.998
    rp = 1 + np.sqrt(1 - a**2)
    rm = 1 - np.sqrt(1 - a**2)
    kappa_k = (rp - rm)/(2*(rp**2 + a**2))
    T_ratio = kappa_k / 0.25
    print(f"T_Kerr/T_Sch(a=0.998) = {T_ratio:.6f} (expected 0.1189)")
    assert abs(T_ratio - 0.1189) < 0.001

    print("ALL VERIFIED.")


if __name__ == "__main__":
    verify()
    fig1()
    fig2()
    fig3()
    print("\nAll 3 figures generated.")

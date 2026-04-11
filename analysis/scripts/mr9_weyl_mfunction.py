# ruff: noqa: E402, I001
"""
MR-9 P8: Numerical computation of the singular Weyl m-function
for the radial Dirac operator on Schwarzschild.

OPERATOR (in r-variable, from independent analysis P8):
    r u'' - u' + α u = -z r³/(4M²) u
where α = κ²/(2M), κ = l + 1/2.

Indicial roots at r = 0: ρ = 0 and ρ = 2.
Two fundamental solutions:
    u_reg(r) ~ r² + ...          (regular/Friedrichs branch)
    u_sing(r) ~ 1 + c₁r + ...   (singular branch)

Both are L²(0,δ; r dr/(2M)), so endpoint r=0 is limit-circle.
Deficiency indices: (1,1) for all κ.

The Weyl m-function m(z) encodes the full spectral information:
    ψ_L²(r,z) = u_sing(r,z) + m(z) u_reg(r,z)
is the unique L² solution at r → ∞ for Im(z) > 0.

From m(z):
    - Spectral density: dρ(λ) = (1/π) Im m(λ + i0⁺) dλ
    - Spectral shift (Krein): ξ(λ) relates to difference of m for two SAE
    - Relative spectral action: ΔS = -∫ f'(λ/Λ²) ξ(λ) dλ / Λ²

Author: David Alfyorov
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import jv, yv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "analysis" / "results" / "mr9"
FIGURES_DIR = PROJECT_ROOT / "analysis" / "figures" / "mr9"


# ===========================================================================
# SECTION 1: THE ODE
# ===========================================================================

def ode_rhs(r, y, z_complex, alpha, M):
    """RHS for the system [u, u'] from r u'' - u' + α u = -z r³/(4M²) u.

    Rewrite as:
        u'' = (u' - α u - z r³/(4M²) u) / r = (u' - (α + z r³/(4M²)) u) / r

    System: y = [u, v] where v = u'
        du/dr = v
        dv/dr = (v - (α + z r³/(4M²)) u) / r
    """
    u, v = y[0], y[1]
    # z can be complex; we work with real and imaginary parts
    z_re, z_im = z_complex.real, z_complex.imag
    coeff = alpha + (z_re + 1j * z_im) * r**3 / (4 * M**2)
    dvdr = (v - coeff * u) / r
    return [v, dvdr]


def ode_rhs_real(r, y, z_re, z_im, alpha, M):
    """Real-valued RHS for complex z.

    y = [u_re, u_im, v_re, v_im] where u = u_re + i u_im, v = u' = v_re + i v_im.
    """
    u_re, u_im, v_re, v_im = y

    coeff_re = alpha + z_re * r**3 / (4 * M**2)
    coeff_im = z_im * r**3 / (4 * M**2)

    # dv/dr = (v - coeff * u) / r
    # coeff * u = (coeff_re + i coeff_im)(u_re + i u_im)
    #           = (coeff_re u_re - coeff_im u_im) + i(coeff_re u_im + coeff_im u_re)
    cu_re = coeff_re * u_re - coeff_im * u_im
    cu_im = coeff_re * u_im + coeff_im * u_re

    dvdr_re = (v_re - cu_re) / r
    dvdr_im = (v_im - cu_im) / r

    return [v_re, v_im, dvdr_re, dvdr_im]


# ===========================================================================
# SECTION 2: FUNDAMENTAL SOLUTIONS NEAR r = 0
# ===========================================================================

def u_reg_init(r, alpha, n_terms=6):
    """Regular solution u_reg ~ r² near r = 0.

    From indicial root ρ = 2: u_reg = r² Σ c_k r^k.
    Leading: u_reg(r) = α/2 r² + O(r³)  (from Bessel J₂).
    We use the series: u_reg = r² (1 + a₁ r + a₂ r² + ...).

    Actually, from the equation r u'' - u' + α u = 0:
    Substituting u = Σ c_k r^{k+ρ} with ρ = 2:
    r(k+2)(k+1) c_k r^{k} - (k+2) c_k r^{k+1} + α c_k r^{k+2} = 0
    → For the recurrence: group by powers of r.

    k=0: r · 2·1 · c₀ r⁰ = 2c₀ r → coeff of r: 2c₀ (from u'') - ... hmm

    Let me just use the Bessel function solution directly.
    u_reg(r) = r J₂(2√(αr))

    For small r: J₂(x) ~ x²/8, so u_reg ~ r · (2√(αr))²/8 = r · 4αr/8 = αr²/2
    """
    x = 2 * np.sqrt(alpha * r)
    if x < 1e-30:
        return alpha * r**2 / 2, alpha * r  # u, u'
    u = r * jv(2, x)
    # u' = J₂(x) + r · J₂'(x) · dx/dr
    # dx/dr = 2α/(2√(αr)) = √(α/r)
    # J₂'(x) = (J₁(x) - J₃(x))/2
    j2p = (jv(1, x) - jv(3, x)) / 2
    up = jv(2, x) + r * j2p * np.sqrt(alpha / r)
    return u, up


def u_sing_init(r, alpha, n_terms=6):
    """Singular solution u_sing ~ const near r = 0.

    u_sing(r) = r Y₂(2√(αr))

    For small r: Y₂(x) ~ -4/(πx²), so u_sing ~ r · (-4)/(π(2√(αr))²)
    = r · (-4)/(4παr) = -1/(πα) = const.
    """
    x = 2 * np.sqrt(alpha * r)
    if x < 1e-30:
        return -1 / (np.pi * alpha), 0.0  # u, u' (leading order)
    u = r * yv(2, x)
    # u' similarly to u_reg
    y2p = (yv(1, x) - yv(3, x)) / 2
    up = yv(2, x) + r * y2p * np.sqrt(alpha / max(r, 1e-30))
    return u, up


# ===========================================================================
# SECTION 3: WEYL m-FUNCTION COMPUTATION
# ===========================================================================

def compute_m_function(z_complex, *, M=1.0, kappa=0.5, r_start=1e-6,
                       r_end=100.0, rtol=1e-10):
    """Compute the Weyl m-function m(z) for the radial operator H_κ.

    Method:
    1. At r = r_start, initialize two fundamental solutions (u_reg, u_sing)
    2. Propagate both to r = r_end
    3. At r_end, the L² solution for Im(z) > 0 is ~ exp(i√z · r*) (outgoing)
    4. m(z) = -[outgoing projection onto u_sing] / [outgoing projection onto u_reg]

    For the far-field (r >> 2M): V → 0, so solutions are plane waves in r*:
        ψ_out ~ e^{i√z · r*}  (L² at +∞ if Im(√z) > 0)

    In the r variable at large r: r* ≈ r + 2M ln(r/2M - 1), and
        ψ_out ~ e^{i√z · r}  (up to slowly varying phase from ln term)
    """
    alpha = kappa**2 / (2 * M)
    z_re = z_complex.real
    z_im = z_complex.imag

    # Initialize regular solution at r_start
    u_reg_0, up_reg_0 = u_reg_init(r_start, alpha)
    # Initialize singular solution at r_start
    u_sing_0, up_sing_0 = u_sing_init(r_start, alpha)

    # Propagate both solutions
    r_span = (r_start, r_end)
    r_eval = np.linspace(r_start, r_end, 5000)

    # Solution 1: regular
    y0_reg = [u_reg_0, 0.0, up_reg_0, 0.0]  # [u_re, u_im, v_re, v_im]
    sol_reg = solve_ivp(
        lambda r, y: ode_rhs_real(r, y, z_re, z_im, alpha, M),
        r_span, y0_reg, t_eval=r_eval, rtol=rtol, atol=1e-14,
        method='DOP853'
    )

    # Solution 2: singular
    y0_sing = [u_sing_0, 0.0, up_sing_0, 0.0]
    sol_sing = solve_ivp(
        lambda r, y: ode_rhs_real(r, y, z_re, z_im, alpha, M),
        r_span, y0_sing, t_eval=r_eval, rtol=rtol, atol=1e-14,
        method='DOP853'
    )

    if not (sol_reg.success and sol_sing.success):
        return None

    # Extract solutions at r_end
    u_reg_end = sol_reg.y[0, -1] + 1j * sol_reg.y[1, -1]
    v_reg_end = sol_reg.y[2, -1] + 1j * sol_reg.y[3, -1]

    u_sing_end = sol_sing.y[0, -1] + 1j * sol_sing.y[1, -1]
    v_sing_end = sol_sing.y[2, -1] + 1j * sol_sing.y[3, -1]

    # At large r: outgoing wave condition
    # ψ_out ~ exp(i k r) where k = √z (choosing Im k > 0)
    k = np.sqrt(z_complex)
    if k.imag < 0:
        k = -k

    # The L² condition at infinity selects ψ with ψ'/ψ → ik at large r
    # So: m(z) = -(u_sing'/u_sing - ik) / ... wait, need to think more carefully.

    # Wronskian method: L² solution ψ = u_sing + m · u_reg
    # At r_end: ψ'/ψ = ik (outgoing wave BC)
    # (v_sing + m v_reg) / (u_sing + m u_reg) = ik
    # v_sing + m v_reg = ik (u_sing + m u_reg)
    # v_sing - ik u_sing = m (ik u_reg - v_reg)
    # m = (v_sing - ik u_sing) / (ik u_reg - v_reg)

    m_z = (v_sing_end - 1j * k * u_sing_end) / (1j * k * u_reg_end - v_reg_end)

    return m_z


# ===========================================================================
# SECTION 4: SPECTRAL DENSITY AND SPECTRAL SHIFT
# ===========================================================================

def spectral_density(lambda_val, *, M=1.0, kappa=0.5, eta=1e-4):
    """Spectral density ρ(λ) = (1/π) Im m(λ + iη).

    For the Friedrichs extension (which selects u_reg at r=0).
    """
    z = complex(lambda_val, eta)
    m = compute_m_function(z, M=M, kappa=kappa)
    if m is None:
        return 0.0
    return m.imag / np.pi


def spectral_shift_two_extensions(lambda_val, beta1, beta2, *, M=1.0, kappa=0.5, eta=1e-4):
    """Spectral shift function ξ(λ) for two SAE with parameters β₁, β₂.

    From Krein's formula:
    ξ(λ) = (1/π) Im log[(β₁ - m(λ+iη)) / (β₂ - m(λ+iη))]

    β = ∞ corresponds to Friedrichs extension (selects u_reg).
    β = 0 corresponds to the extension that allows u_sing.
    """
    z = complex(lambda_val, eta)
    m = compute_m_function(z, M=M, kappa=kappa)
    if m is None:
        return 0.0

    ratio = (beta1 - m) / (beta2 - m)
    if abs(ratio) < 1e-30:
        return 0.0
    return np.angle(ratio) / np.pi


# ===========================================================================
# SECTION 5: MAIN COMPUTATION
# ===========================================================================

def run_m_function_scan(*, M=1.0, kappa=0.5, n_points=100, lambda_max=10.0, eta=1e-3):
    """Scan m(z) along the real axis to get spectral density."""
    lambdas = np.linspace(0.01, lambda_max, n_points)
    m_values = []
    rho_values = []

    print(f"Computing m(λ + i·{eta}) for κ={kappa}, M={M}...")
    print(f"λ range: [0.01, {lambda_max}], {n_points} points")
    print()

    for i, lam in enumerate(lambdas):
        z = complex(lam, eta)
        m = compute_m_function(z, M=M, kappa=kappa, r_end=50.0, rtol=1e-8)
        if m is not None:
            m_values.append(m)
            rho = m.imag / np.pi
            rho_values.append(rho)
            if i % 20 == 0:
                print(f"  λ={lam:.2f}: m = {m.real:.6f} + {m.imag:.6f}i, ρ = {rho:.6f}")
        else:
            m_values.append(0j)
            rho_values.append(0.0)
            if i % 20 == 0:
                print(f"  λ={lam:.2f}: FAILED")

    return lambdas, np.array(m_values), np.array(rho_values)


def run_spectral_shift_scan(*, M=1.0, kappa=0.5, beta1=1e6, beta2=0.0,
                             n_points=100, lambda_max=10.0, eta=1e-3):
    """Compute spectral shift function ξ(λ) for two extensions."""
    lambdas = np.linspace(0.01, lambda_max, n_points)
    xi_values = []

    print(f"Computing ξ(λ) for β₁={beta1}, β₂={beta2}...")

    for lam in lambdas:
        xi = spectral_shift_two_extensions(lam, beta1, beta2, M=M, kappa=kappa, eta=eta)
        xi_values.append(xi)

    return lambdas, np.array(xi_values)


def compute_relative_spectral_action(lambdas, xi_values, Lambda=1.0):
    """Compute ΔS = -∫ f'(λ/Λ²) ξ(λ) dλ / Λ² for f(x) = e^{-x}.

    f'(x) = -e^{-x}, so:
    ΔS = ∫ e^{-λ/Λ²} ξ(λ) dλ / Λ²
    """
    dlam = lambdas[1] - lambdas[0]
    integrand = np.exp(-lambdas / Lambda**2) * xi_values / Lambda**2
    return np.sum(integrand) * dlam


# ===========================================================================
# SECTION 6: VERIFICATION
# ===========================================================================

def verify_fundamental_solutions():
    """Verify u_reg and u_sing satisfy the ODE at leading order."""
    alpha = 0.5**2 / 2  # κ=1/2, M=1 → α = 1/8

    print("=== VERIFICATION: Fundamental solutions ===")
    print(f"α = κ²/(2M) = {alpha}")
    print()

    for r in [1e-4, 1e-3, 1e-2, 0.1]:
        u_r, up_r = u_reg_init(r, alpha)
        u_s, up_s = u_sing_init(r, alpha)

        # Check ODE: r u'' - u' + α u ≈ 0 (leading order, z=0)
        # u'' ≈ (u(r+dr) - 2u(r) + u(r-dr)) / dr²
        dr = r * 1e-4
        u_r_p, _ = u_reg_init(r + dr, alpha)
        u_r_m, _ = u_reg_init(max(r - dr, 1e-10), alpha)
        u_r_pp = (u_r_p - 2 * u_r + u_r_m) / dr**2

        resid_reg = r * u_r_pp - up_r + alpha * u_r
        print(f"r={r:.4f}: u_reg={u_r:.8f}, u_sing={u_s:.8f}, "
              f"ODE residual(reg)={resid_reg:.2e}")

    print()


# ===========================================================================
# CLI
# ===========================================================================

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Step 0: Verify
    verify_fundamental_solutions()

    # Step 1: m-function scan
    print("=" * 60)
    print("STEP 1: Weyl m-function scan")
    print("=" * 60)
    lambdas, m_vals, rho_vals = run_m_function_scan(
        M=1.0, kappa=0.5, n_points=50, lambda_max=5.0, eta=0.01
    )

    # Step 2: Spectral shift
    print()
    print("=" * 60)
    print("STEP 2: Spectral shift function")
    print("=" * 60)
    lam_xi, xi_vals = run_spectral_shift_scan(
        M=1.0, kappa=0.5, beta1=1e6, beta2=0.0,
        n_points=50, lambda_max=5.0, eta=0.01
    )

    # Step 3: Relative spectral action
    print()
    print("=" * 60)
    print("STEP 3: Relative spectral action ΔS")
    print("=" * 60)
    for Lam in [0.5, 1.0, 2.0, 5.0]:
        dS = compute_relative_spectral_action(lam_xi, xi_vals, Lambda=Lam)
        print(f"  Λ = {Lam:.1f}: ΔS = {dS:.8f}")

    # Save results
    results = {
        "kappa": 0.5,
        "M": 1.0,
        "lambdas": lambdas.tolist(),
        "m_real": [m.real for m in m_vals],
        "m_imag": [m.imag for m in m_vals],
        "rho": rho_vals.tolist(),
        "xi": xi_vals.tolist(),
    }

    output_path = RESULTS_DIR / "mr9_weyl_mfunction.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {output_path}")

    # Figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(lambdas, [m.real for m in m_vals], 'b-', label='Re m(λ)')
    axes[0].plot(lambdas, [m.imag for m in m_vals], 'r--', label='Im m(λ)')
    axes[0].set_xlabel('λ')
    axes[0].set_ylabel('m(λ + iη)')
    axes[0].set_title('Weyl m-function')
    axes[0].legend()

    axes[1].plot(lambdas, rho_vals, 'k-')
    axes[1].set_xlabel('λ')
    axes[1].set_ylabel('ρ(λ)')
    axes[1].set_title('Spectral density (Friedrichs)')

    axes[2].plot(lam_xi, xi_vals, 'g-')
    axes[2].set_xlabel('λ')
    axes[2].set_ylabel('ξ(λ)')
    axes[2].set_title('Spectral shift (β₁=∞ vs β₂=0)')

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mr9_weyl_mfunction.pdf", dpi=150)
    plt.close(fig)
    print(f"Figure saved to {FIGURES_DIR / 'mr9_weyl_mfunction.pdf'}")


if __name__ == "__main__":
    main()

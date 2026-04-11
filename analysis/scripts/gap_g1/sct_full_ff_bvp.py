"""
Full form-factor BVP for SCT singularity resolution.

Uses the EXACT diffusion localization of the full Weyl form factor
(not a₆ truncation). Based on analytical follow-up derivation.

System: (F, H, u_k) where u_k(ρ) = diffused Weyl amplitude at τ_k.

Metric: ds² = -H(ρ)dt² + dρ²/H(ρ) + F(ρ)²dΩ²

Generalized operator:
    L_{F,H} u = H u'' + (H' + 2HF'/F) u' - 6H(F'/F)² u

Diffusion: u_k'' = [-(H'+2HF'/F)/H] u_k' + 6(F'/F)² u_k - Λ²/(H Δτ_k)(u_k - u_{k-1})

Center BC (regular): F ~ ρ, H ~ 1, u_k ~ w₂_k ρ²
Outer BC (Schwarzschild): F ~ ρ, H ~ 1-2M/ρ, u_k ~ -2M/ρ³

Author: David Alfyorov
"""
import numpy as np
from scipy.integrate import solve_bvp
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "analysis" / "results" / "gap_g1"

# =========================================================================
# P(α) kernel (verified: ∫P dα = 13/120)
# =========================================================================
def P_kernel(alpha):
    tau = alpha * (1 - alpha)
    return -89/24 + (43/6)*tau + (236/3)*tau**2

# =========================================================================
# Diffusion layers: α_k quadrature points, τ_k = α_k(1-α_k), weights w_k
# =========================================================================
def make_diffusion_layers(K=8):
    """Gauss-Legendre quadrature on [0,1] for ∫₀¹ P(α) u(τ(α)) dα."""
    from numpy.polynomial.legendre import leggauss
    nodes, weights = leggauss(K)
    alphas = 0.5 * (nodes + 1)  # map [-1,1] → [0,1]
    ws = 0.5 * weights
    taus = alphas * (1 - alphas)
    Ps = np.array([P_kernel(a) for a in alphas])
    # Sort by τ for marching
    order = np.argsort(taus)
    return taus[order], Ps[order], ws[order], alphas[order]


# =========================================================================
# ODE system: state = [F, F', H, H', u_1, u_1', ..., u_K, u_K']
# =========================================================================
def build_ode(K, M, Lam):
    """Build the ODE RHS for the full form-factor BVP."""
    taus, Ps, ws, alphas = make_diffusion_layers(K)
    # Δτ for marching: τ₀ = 0 (IC from W), then τ₁, τ₂, ...
    dtaus = np.diff(np.concatenate([[0.0], taus]))

    n_state = 4 + 2*K  # (F, F', H, H', u_1, u_1', ..., u_K, u_K')

    def rhs(rho, y):
        """ODE right-hand side (vectorized for solve_bvp)."""
        rho = np.maximum(rho, 1e-12)
        F = y[0]
        Fp = y[1]
        H = y[2]
        Hp = y[3]

        F = np.maximum(np.abs(F), 1e-12)
        H_safe = np.where(np.abs(H) > 1e-15, H, 1e-15)

        FpF = Fp / F

        # Weyl scalar W (= C_{tρtρ} in orthonormal frame)
        # W = (F²H'' - 2FHF'' - 2FF'H' + 2H(F')² - 2) / (6F²)
        # We don't compute H'' and F'' directly; they come from the metric equations.
        # For now: use the Einstein equations G = -Θ^(C) to determine F'' and H''.

        # For the METRIC sector: we need G_μν + Θ^(C) = 0.
        # G^t_t = (2FHF'' + FF'H' + H(F')² - 1) / F²
        # G^ρ_ρ = (FF'H' + H(F')² - 1) / F²

        # Θ^(C) from the diffused Weyl:
        # Θ^(C) ~ 12 F² W · Σ_k w_k P_k u_k (simplified leading order)

        # For NOW: solve just the DIFFUSION part on a FIXED metric background
        # (Schwarzschild or smooth center), and check inner behavior.
        # Self-consistent iteration later.

        dydr = np.zeros_like(y)
        dydr[0] = Fp
        dydr[2] = Hp
        dydr[1] = np.zeros_like(rho)  # F'' = 0 (Schwarzschild)
        dydr[3] = -4*M/rho**3  # H'' (Schwarzschild)

        coeff_up = -(Hp + 2*H_safe*FpF) / H_safe
        coeff_u = 6 * FpF**2

        for j in range(K):
            u_j = y[4 + 2*j]
            up_j = y[4 + 2*j + 1]

            if j == 0:
                u_prev = -2*M/rho**3
            else:
                u_prev = y[4 + 2*(j-1)]

            dt = max(dtaus[j], 1e-20)
            source = -Lam**2 / (H_safe * dt) * (u_j - u_prev)

            upp_j = coeff_up * up_j + coeff_u * u_j + source

            dydr[4 + 2*j] = up_j
            dydr[4 + 2*j + 1] = upp_j

        return dydr

    return n_state, rhs, taus, Ps, ws


# =========================================================================
# Boundary conditions
# =========================================================================
def bc_center_schwarzschild(ya, yb, K, M, Lam, R_max):
    """BC for diffusion on fixed Schwarzschild background.

    Center (ρ = ε → 0): F = ρ, H = 1-2M/ρ ≈ -2M/ρ for ρ << 2M.
    But for center of REGULAR solution: F ~ ρ, H ~ 1. Mixed case for Schwarzschild.

    For Schwarzschild test: just match F, H to exact values.
    For diffusion: u_k(ε) ~ regular (u_k ~ c_k ε² if center is smooth).
    """
    residuals = []
    eps = 0.05
    taus, _, _, _ = make_diffusion_layers(K)

    # LEFT (center) BC: 8 conditions
    residuals.append(ya[0] - eps)              # F(ε) = ε
    residuals.append(ya[1] - 1.0)              # F'(ε) = 1
    residuals.append(ya[2] - (1 - 2*M/eps))    # H(ε)
    residuals.append(ya[3] - 2*M/eps**2)        # H'(ε)
    for j in range(K):
        # u_k(ε) = W(ε) = -2M/ε³ (zeroth-order: all layers start from W)
        u_center = -2*M/eps**3
        residuals.append(ya[4 + 2*j] - u_center)

    # RIGHT (R_max) BC: 8 conditions
    residuals.append(yb[0] - R_max)             # F(R)
    residuals.append(yb[2] - (1 - 2*M/R_max))   # H(R)
    for j in range(K):
        u_outer = -2*M/R_max**3 - 12*M**2*taus[j]/(Lam**2*R_max**6)
        residuals.append(yb[4 + 2*j] - u_outer)

    # Total: 4 + K + 2 + K = 6 + 2K. Need n_state = 4 + 2K.
    # Currently: 6 + 2K. Need exactly 4 + 2K. So remove 2.
    # Actually: 4 center_metric + K center_u + 2 outer_metric + K outer_u = 6+2K
    # But n_state = 4+2K → need 4+2K BCs. Extra 2 from metric center (have 4, need 2).
    # Fix: make H(ε) and H'(ε) implicit (Schwarzschild already fixed by F, F')
    # Remove H center BCs, keep F center + u center + outer.
    pass

    # CORRECT count: need exactly n_state = 4 + 2K BCs
    # Split: (4+2K)/2 at each end
    # Left: 2 metric (F, F') + K diffusion (u_k) = 2+K
    # Right: 2 metric (F, H) + K diffusion (u_k) = 2+K
    # Total: 4 + 2K ✓
    residuals_correct = []
    # LEFT
    residuals_correct.append(ya[0] - eps)
    residuals_correct.append(ya[1] - 1.0)
    for j in range(K):
        residuals_correct.append(ya[4 + 2*j] - (-2*M/eps**3))
    # RIGHT
    residuals_correct.append(yb[0] - R_max)
    residuals_correct.append(yb[2] - (1 - 2*M/R_max))
    for j in range(K):
        u_outer = -2*M/R_max**3 - 12*M**2*taus[j]/(Lam**2*R_max**6)
        residuals_correct.append(yb[4 + 2*j] - u_outer)

    return np.array(residuals_correct)


# =========================================================================
# Main: solve diffusion on Schwarzschild and check inner behavior
# =========================================================================
def run_diffusion_on_schwarzschild(M=1.0, Lam=1.0, K=6, R_max=30.0, N=300):
    """Solve the diffusion BVP on fixed Schwarzschild background."""
    n_state, rhs, taus, Ps, ws = build_ode(K, M, Lam)

    eps = 0.05
    rho = np.linspace(eps, R_max, N)

    # Initial guess: Schwarzschild metric + u_k = W = -2M/ρ³
    y_init = np.zeros((n_state, N))
    y_init[0] = rho  # F = ρ
    y_init[1] = 1.0  # F' = 1
    y_init[2] = 1 - 2*M/rho  # H = 1-2M/ρ
    y_init[3] = 2*M/rho**2  # H' = 2M/ρ²

    for j in range(K):
        y_init[4 + 2*j] = -2*M/rho**3  # u_k = W
        y_init[4 + 2*j + 1] = 6*M/rho**4  # u_k' = -W'

    def bc(ya, yb):
        return bc_center_schwarzschild(ya, yb, K, M, Lam, R_max)

    print(f"Solving diffusion BVP: K={K} layers, M={M}, Λ={Lam}, R_max={R_max}")
    print(f"State dimension: {n_state}, grid: {N} points")

    sol = solve_bvp(rhs, bc, rho, y_init, tol=1e-6, max_nodes=10000, verbose=0)

    print(f"Status: {sol.status}, message: {sol.message}")
    print(f"Residual rms: {sol.rms_residuals:.2e}" if hasattr(sol, 'rms_residuals') else "")

    if sol.status != 0:
        print("BVP did not converge. Trying IVP instead.")
        return solve_diffusion_ivp(M, Lam, K, R_max, N)

    # Extract integrated form factor: Σ_k w_k P_k u_k(ρ)
    rho_sol = sol.x
    phi_D = np.zeros_like(rho_sol)
    for j in range(K):
        u_j = sol.y[4 + 2*j]
        phi_D += ws[j] * Ps[j] * u_j

    # Print inner behavior
    print("\nInner behavior of φ(□)C = Σ w_k P_k u_k:")
    for i in range(min(10, len(rho_sol))):
        print(f"  ρ = {rho_sol[i]:.4f}: φ(□)C = {phi_D[i]:+.6e}")

    # Fit power law at small ρ
    mask = (rho_sol < 0.5) & (np.abs(phi_D) > 1e-30)
    if np.sum(mask) > 3:
        log_r = np.log(rho_sol[mask])
        log_phi = np.log(np.abs(phi_D[mask]))
        coeffs = np.polyfit(log_r, log_phi, 1)
        print(f"\nFitted inner power law: φ(□)C ~ ρ^{{{coeffs[0]:.4f}}}")
        print(f"Expected (regular): ρ² (i.e., u ~ ρ², W ~ ρ²)")
        print(f"Expected (singular): ρ^{{-3}} (i.e., u ~ ρ^{{-3}})")

    return sol, phi_D


def solve_diffusion_ivp(M=1.0, Lam=1.0, K=6, R_max=30.0, N=300):
    """Fallback: solve diffusion as IVP from large ρ inward."""
    from scipy.integrate import solve_ivp

    n_state, rhs_bvp, taus, Ps, ws = build_ode(K, M, Lam)

    # Start from R_max with Schwarzschild IC
    y0 = np.zeros(n_state)
    y0[0] = R_max
    y0[1] = 1.0
    y0[2] = 1 - 2*M/R_max
    y0[3] = 2*M/R_max**2
    for j in range(K):
        u_outer = -2*M/R_max**3 - 12*M**2*taus[j]/(Lam**2*R_max**6)
        y0[4 + 2*j] = u_outer
        y0[4 + 2*j + 1] = 6*M/R_max**4 + 72*M**2*taus[j]/(Lam**2*R_max**7)

    def rhs_neg(rho, y):
        return -np.array(rhs_bvp(-rho, y))  # integrate inward

    # Integrate inward (negative direction)
    rho_span = (R_max, 0.05)
    rho_eval = np.linspace(R_max, 0.05, N)

    sol = solve_ivp(lambda r, y: rhs_bvp(r, y), rho_span, y0,
                    t_eval=rho_eval, method='Radau', rtol=1e-8, atol=1e-12,
                    max_step=0.5)

    print(f"IVP status: {sol.status}, reached ρ_min = {sol.t[-1]:.4f}")

    rho_sol = sol.t
    phi_D = np.zeros(len(rho_sol))
    for j in range(K):
        u_j = sol.y[4 + 2*j]
        phi_D += ws[j] * Ps[j] * u_j

    # Print inner behavior
    inner = rho_sol < 1.0
    print("\nInner behavior (ρ < 1):")
    inner_indices = np.where(inner)[0]
    for i in inner_indices[-10:]:
        print(f"  ρ = {rho_sol[i]:.4f}: φ(□)C = {phi_D[i]:+.6e}, "
              f"u_0 = {sol.y[4, i]:+.6e}")

    # Fit power law
    mask = inner & (np.abs(phi_D) > 1e-30)
    if np.sum(mask) > 3:
        log_r = np.log(rho_sol[mask])
        log_phi = np.log(np.abs(phi_D[mask]))
        coeffs = np.polyfit(log_r, log_phi, 1)
        print(f"\nFitted inner power law: φ(□)C ~ ρ^{{{coeffs[0]:.4f}}}")

    return sol, phi_D


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sol, phi_D = run_diffusion_on_schwarzschild(M=1.0, Lam=1.0, K=6, R_max=30.0, N=300)

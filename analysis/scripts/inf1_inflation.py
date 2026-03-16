# ruff: noqa: E402, I001
"""
INF-1: Inflationary observables from the SCT spectral action's R^2 sector.

Derives n_s, r, f_NL, T_reh from the spectral action framework, determines
the viable (Lambda, xi) parameter space, and computes nonlocal form factor
corrections to the tensor-to-scalar ratio using the KKS framework.

Central results:
  - The one-loop SM spectral action R^2 coefficient alpha_R(xi) = 2(xi-1/6)^2
    yields a scalaron mass M >> M_Pl at minimal coupling (xi=0), far too heavy
    for inflation.  Matching the observed A_s requires |xi - 1/6| ~ 2e5
    (unphysical).  This is the "scalaron mass problem".
  - Assuming M is fixed externally (e.g. by BSM content or higher loops),
    all Starobinsky observables are computed.  Nonlocal corrections from the
    Weyl form factor F_1 modify the tensor-to-scalar ratio r as a function
    of z_H = H_inf^2/Lambda^2.
  - For Lambda >> H_inf: corrections < 1%, standard Starobinsky predictions hold.
  - For Lambda ~ H_inf: corrections are O(1), r can be significantly modified.

Key formulas:
  c_2 = alpha_R(xi) / (16*pi^2)
  M^2 = M_Pl^2 / (12*c_2)
  V(phi) = (3*M^2*M_Pl^2/4) * (1 - exp(-sqrt(2/3)*phi/M_Pl))^2
  eps = 3/(4*N^2),  eta = -1/N
  n_s = 1 - 2/N,  r = 12/N^2 (local limit)
  r_nonlocal = r_local / Pi_TT(z_H)  where Pi_TT = 1 + (13/60)*z*Fhat_1(z)

References:
  - Starobinsky (1980), Phys. Lett. B 91, 99
  - Mukhanov-Chibisov (1981), JETP Lett. 33, 532
  - Koshelev-Kumar-Starobinsky (2022), arXiv:2209.02515
  - Koshelev-Kumar-Mazumdar-Starobinsky (2020), arXiv:2003.00629
  - Chamseddine-Connes (1997), hep-th/9606001
  - Planck 2018 X, arXiv:1807.06211
  - BICEP/Keck 2021, arXiv:2110.00483
  - Maldacena (2003), astro-ph/0210603

Author: David Alfyorov
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from scripts.nt2_entire_function import (
    F1_total_complex,
    F2_total_complex,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "analysis" / "results" / "inf1"
FIGURES_DIR = PROJECT_ROOT / "analysis" / "figures" / "inf1"

# ============================================================
# Canonical constants (DO NOT MODIFY)
# ============================================================
ALPHA_C = mp.mpf(13) / 120          # SM Weyl^2 coefficient
LOCAL_C2 = 2 * ALPHA_C              # = 13/60, tensor propagator coeff
DPS = 100                           # default mpmath precision

# Physical constants
M_PL_GEV = mp.mpf("2.435e18")      # Reduced Planck mass in GeV
A_S_OBS = mp.mpf("2.1e-9")         # Observed scalar amplitude (Planck 2018)
G_STAR_SM = mp.mpf("106.75")       # SM relativistic d.o.f. at high T

# Observational constraints
NS_PLANCK = mp.mpf("0.9649")       # Planck 2018 central value
NS_PLANCK_ERR = mp.mpf("0.0042")   # Planck 2018 1-sigma
R_BICEP_UPPER = mp.mpf("0.036")    # BICEP/Keck 2021 95% CL upper bound
FNL_PLANCK = mp.mpf("-0.9")        # Planck 2018 f_NL^local central
FNL_PLANCK_ERR = mp.mpf("5.1")     # Planck 2018 f_NL^local 1-sigma
T_BBN_MEV = mp.mpf("5")            # BBN lower bound on T_reh (MeV)

# Cached form factor normalization
_F1_0_CACHE: mp.mpc | None = None
_F2_0_CACHE: dict[float, mp.mpc] = {}


def _get_F1_0(dps: int = DPS) -> mp.mpc:
    """Cached F1(0)."""
    global _F1_0_CACHE
    if _F1_0_CACHE is None:
        mp.mp.dps = dps
        _F1_0_CACHE = F1_total_complex(0, dps=dps)
    return _F1_0_CACHE


def _get_F2_0(xi: float = 0.0, dps: int = DPS) -> mp.mpc:
    """Cached F2(0, xi)."""
    if xi not in _F2_0_CACHE:
        mp.mp.dps = dps
        _F2_0_CACHE[xi] = F2_total_complex(0, xi=xi, dps=dps)
    return _F2_0_CACHE[xi]


# ============================================================
# Section 1: R^2 coefficient and scalaron mass
# ============================================================

def alpha_R(xi: float | mp.mpf) -> mp.mpf:
    """R^2 coefficient alpha_R(xi) = 2(xi - 1/6)^2.

    From NT-1b Phase 3 (VERIFIED, canonical).
    """
    xi_mp = mp.mpf(xi)
    if abs(xi_mp - mp.mpf(1) / 6) < mp.mpf("1e-14"):
        return mp.mpf(0)
    return 2 * (xi_mp - mp.mpf(1) / 6) ** 2


def c2_coefficient(xi: float | mp.mpf) -> mp.mpf:
    """Local R^2 coupling c_2 = alpha_R(xi) / (16*pi^2).

    This is the coefficient of R^2 in the one-loop effective action.
    """
    return alpha_R(xi) / (16 * mp.pi ** 2)


def scalaron_mass(xi: float | mp.mpf) -> mp.mpf:
    """Scalaron mass M/M_Pl from the one-loop spectral action.

    M^2 = M_Pl^2 / (12*c_2) = 2*pi^2*M_Pl^2 / (3*(xi-1/6)^2).

    Returns M/M_Pl (dimensionless ratio). Returns inf at xi=1/6.
    """
    c2 = c2_coefficient(xi)
    if c2 == 0:
        return mp.inf
    return mp.sqrt(mp.mpf(1) / (12 * c2))


def scalaron_mass_gev(xi: float | mp.mpf) -> mp.mpf:
    """Scalaron mass M in GeV."""
    return scalaron_mass(xi) * M_PL_GEV


def required_scalaron_mass(N: int = 55) -> mp.mpf:
    """Required M/M_Pl from the observed amplitude A_s.

    M/M_Pl = sqrt(24*pi^2*A_s) / N.
    """
    return mp.sqrt(24 * mp.pi ** 2 * A_S_OBS) / N


def required_c2(N: int = 55) -> mp.mpf:
    """Required c_2 from A_s.

    c_2 = M_Pl^2/(12*M^2) = 1/(12*(M/M_Pl)^2).
    """
    m_ratio = required_scalaron_mass(N)
    return mp.mpf(1) / (12 * m_ratio ** 2)


def required_xi_deviation(N: int = 55) -> mp.mpf:
    """Required |xi - 1/6| from A_s.

    |xi - 1/6| = sqrt(8*pi^2*c_2_required).
    """
    c2_req = required_c2(N)
    return mp.sqrt(8 * mp.pi ** 2 * c2_req)


def mass_ratio(xi: float | mp.mpf, N: int = 55) -> mp.mpf:
    """Ratio M_SCT(xi) / M_required.

    If >> 1, the scalaron is too heavy for inflation.
    """
    m_sct = scalaron_mass(xi)
    m_req = required_scalaron_mass(N)
    if m_sct == mp.inf:
        return mp.inf
    return m_sct / m_req


# ============================================================
# Section 2: Scalaron mass from NT-4a pole (full spectral action)
# ============================================================

def nt4a_scalar_mass(xi: float | mp.mpf, Lambda_GeV: float | mp.mpf) -> mp.mpf:
    """NT-4a scalar pole mass m_0(xi) in GeV.

    m_0^2 = Lambda^2 / (6*(xi-1/6)^2).
    In the full spectral action with M_Pl^2 = f_2*Lambda^2/(2*pi^2),
    this equals the Starobinsky scalaron mass M when f_2 = 1.
    """
    xi_mp = mp.mpf(xi)
    Lambda_mp = mp.mpf(Lambda_GeV)
    dev = xi_mp - mp.mpf(1) / 6
    if abs(dev) < mp.mpf("1e-14"):
        return mp.inf
    return Lambda_mp / mp.sqrt(6 * dev ** 2)


def nt4a_tensor_mass(Lambda_GeV: float | mp.mpf) -> mp.mpf:
    """NT-4a tensor pole mass m_2 in GeV.

    m_2 = Lambda * sqrt(60/13).
    """
    return mp.mpf(Lambda_GeV) * mp.sqrt(mp.mpf(60) / 13)


# ============================================================
# Section 3: Einstein frame and Starobinsky potential
# ============================================================

def starobinsky_potential(phi_over_MPl: float | mp.mpf, M_over_MPl: float | mp.mpf) -> mp.mpf:
    """Starobinsky potential V(phi) / M_Pl^4.

    V(phi) = (3/4)*M^2*M_Pl^2*(1 - exp(-sqrt(2/3)*phi/M_Pl))^2.
    Returns V / M_Pl^4.
    """
    phi = mp.mpf(phi_over_MPl)
    M = mp.mpf(M_over_MPl)
    return mp.mpf(3) / 4 * M ** 2 * (1 - mp.exp(-mp.sqrt(mp.mpf(2) / 3) * phi)) ** 2


def starobinsky_potential_derivatives(
    phi_over_MPl: float | mp.mpf,
    M_over_MPl: float | mp.mpf,
) -> tuple[mp.mpf, mp.mpf, mp.mpf]:
    """V, V', V'' for the Starobinsky potential (all in M_Pl units).

    V  = (3/4)*M^2*(1 - exp(-sqrt(2/3)*phi))^2
    V' = (3/4)*M^2 * 2*sqrt(2/3)*(1-x)*x  where x = exp(-sqrt(2/3)*phi)
    V'' = -(3/4)*M^2 * (4/3)*x*(2x - 1)

    Returns (V/M_Pl^4, V'/M_Pl^3, V''/M_Pl^2).
    """
    phi = mp.mpf(phi_over_MPl)
    M = mp.mpf(M_over_MPl)
    sq23 = mp.sqrt(mp.mpf(2) / 3)
    x = mp.exp(-sq23 * phi)

    V = mp.mpf(3) / 4 * M ** 2 * (1 - x) ** 2
    Vp = mp.mpf(3) / 4 * M ** 2 * 2 * sq23 * (1 - x) * x
    # V'' = (3/2)*a^2*M^2*x*(2x-1) = M^2*x*(2x-1) where a=sqrt(2/3)
    Vpp = M ** 2 * x * (2 * x - 1)

    return V, Vp, Vpp


def phi_N(N: int, M_over_MPl: float | mp.mpf) -> mp.mpf:
    """Field value phi/M_Pl at N e-folds before end of inflation.

    Solves the exact e-fold integral:
    N = (3/4)*[exp(a*phi) - a*phi] - (3/4)*[exp(a*phi_end) - a*phi_end]
    where a = sqrt(2/3) and phi_end = sqrt(3/2)*ln(2) (epsilon=1).

    Returns phi/M_Pl.
    """
    sq23 = mp.sqrt(mp.mpf(2) / 3)
    phi_end = mp.sqrt(mp.mpf(3) / 2) * mp.log(2)
    N_end = mp.mpf(3) / 4 * (mp.exp(sq23 * phi_end) - sq23 * phi_end)

    def N_from_phi(phi):
        return mp.mpf(3) / 4 * (mp.exp(sq23 * phi) - sq23 * phi) - N_end - mp.mpf(N)

    # Initial guess from leading-order approximation
    phi_guess = mp.sqrt(mp.mpf(3) / 2) * mp.log(mp.mpf(4) * N / 3)
    return mp.findroot(N_from_phi, phi_guess)


# ============================================================
# Section 4: Slow-roll parameters and local observables
# ============================================================

def slow_roll_params(N: int) -> tuple[mp.mpf, mp.mpf]:
    """Slow-roll parameters epsilon, eta at N e-folds.

    Leading order for the Starobinsky potential:
    epsilon = 3/(4*N^2),  eta = -1/N.

    Returns (epsilon, eta).
    """
    eps = mp.mpf(3) / (4 * mp.mpf(N) ** 2)
    eta = -mp.mpf(1) / mp.mpf(N)
    return eps, eta


def slow_roll_exact(phi_over_MPl: float | mp.mpf, M_over_MPl: float | mp.mpf) -> tuple[mp.mpf, mp.mpf]:
    """Exact slow-roll parameters from the potential.

    epsilon = (M_Pl^2/2)*(V'/V)^2
    eta = M_Pl^2 * V''/V

    Returns (epsilon, eta).
    """
    V, Vp, Vpp = starobinsky_potential_derivatives(phi_over_MPl, M_over_MPl)
    if V == 0:
        return mp.inf, mp.inf
    eps = (Vp / V) ** 2 / 2
    eta = Vpp / V
    return eps, eta


def inflationary_observables(N: int = 55) -> dict[str, Any]:
    """Standard Starobinsky inflationary observables at N e-folds.

    Uses leading-order slow-roll approximation.
    Returns dict with n_s, r, n_t, A_s, f_NL, and derived quantities.
    """
    mp.mp.dps = DPS

    eps, eta = slow_roll_params(N)

    n_s = 1 - 6 * eps + 2 * eta
    r = 16 * eps
    n_t = -2 * eps
    f_NL = mp.mpf(5) / 12 * (n_s - 1)

    # Running
    dns_dlnk = -2 / mp.mpf(N) ** 2  # leading order

    # Scalaron mass from A_s
    M_ratio = required_scalaron_mass(N)
    M_GeV = M_ratio * M_PL_GEV

    # Amplitude check
    A_s_check = mp.mpf(N) ** 2 * M_ratio ** 2 / (24 * mp.pi ** 2)

    return {
        "N": N,
        "epsilon": float(eps),
        "eta": float(eta),
        "n_s": float(n_s),
        "r": float(r),
        "n_t": float(n_t),
        "f_NL_local": float(f_NL),
        "dns_dlnk": float(dns_dlnk),
        "M_over_MPl": float(M_ratio),
        "M_GeV": float(M_GeV),
        "A_s_check": float(A_s_check),
        "consistency_r_minus_8nt": float(r + 8 * n_t),
    }


def inflationary_observables_exact(N: int = 55) -> dict[str, Any]:
    """Exact (not leading-order) Starobinsky observables.

    Computes phi_N, then evaluates exact slow-roll parameters from the potential.
    """
    mp.mp.dps = DPS

    M_ratio = required_scalaron_mass(N)
    phi_val = phi_N(N, M_ratio)
    eps, eta = slow_roll_exact(phi_val, M_ratio)

    n_s = 1 - 6 * eps + 2 * eta
    r = 16 * eps
    n_t = -2 * eps
    f_NL = mp.mpf(5) / 12 * (n_s - 1)

    return {
        "N": N,
        "phi_over_MPl": float(phi_val),
        "epsilon_exact": float(eps),
        "eta_exact": float(eta),
        "n_s_exact": float(n_s),
        "r_exact": float(r),
        "n_t_exact": float(n_t),
        "f_NL_exact": float(f_NL),
        "M_over_MPl": float(M_ratio),
    }


# ============================================================
# Section 5: Nonlocal corrections via KKS framework
# ============================================================

def Pi_TT(z: float | mp.mpf, dps: int = DPS) -> mp.mpf:
    """Tensor propagator denominator Pi_TT(z) = 1 + (13/60)*z*Fhat_1(z).

    From NT-4a (VERIFIED). z = k^2/Lambda^2.
    """
    mp.mp.dps = dps
    z_mp = mp.mpf(z)
    if abs(z_mp) < mp.mpf("1e-14"):
        return mp.mpf(1)

    F1_0 = _get_F1_0(dps)
    F1_z = F1_total_complex(float(z_mp), dps=dps)
    Fhat1 = F1_z / F1_0

    return 1 + LOCAL_C2 * z_mp * Fhat1


def Pi_s(z: float | mp.mpf, xi: float | mp.mpf = 0.0, dps: int = DPS) -> mp.mpf:
    """Scalar propagator denominator Pi_s(z, xi) = 1 + 6*(xi-1/6)^2*z*Fhat_2(z).

    From NT-4a (VERIFIED). z = k^2/Lambda^2.
    """
    mp.mp.dps = dps
    z_mp = mp.mpf(z)
    xi_mp = mp.mpf(xi)

    if abs(xi_mp - mp.mpf(1) / 6) < mp.mpf("1e-14"):
        return mp.mpf(1)
    if abs(z_mp) < mp.mpf("1e-14"):
        return mp.mpf(1)

    F2_0 = _get_F2_0(float(xi_mp), dps)
    F2_z = F2_total_complex(float(z_mp), xi=float(xi_mp), dps=dps)
    Fhat2 = F2_z / F2_0

    coeff = 6 * (xi_mp - mp.mpf(1) / 6) ** 2
    return 1 + coeff * z_mp * Fhat2


def nonlocal_corrections(
    N: int = 55,
    Lambda_GeV: float | mp.mpf = mp.mpf("1e15"),
    xi: float | mp.mpf = 0.0,
    dps: int = DPS,
) -> dict[str, Any]:
    """Nonlocal corrections to inflationary observables.

    Uses the KKS framework with SCT form factors:
    - n_s is UNCHANGED (KKS theorem: scalar spectrum preserved)
    - r is modified by 1/Pi_TT(z_H) where z_H = H_inf^2/Lambda^2
    - The consistency relation r = -8*n_t is VIOLATED

    Parameters:
        N: e-folds before end of inflation
        Lambda_GeV: spectral cutoff in GeV
        xi: Higgs non-minimal coupling
        dps: mpmath precision

    Returns dict with local and nonlocal observables.
    """
    mp.mp.dps = dps
    Lambda_mp = mp.mpf(Lambda_GeV)

    # Local observables (unchanged for scalar spectrum)
    M_ratio = required_scalaron_mass(N)
    M_GeV = M_ratio * M_PL_GEV

    eps, eta = slow_roll_params(N)
    n_s = 1 - 6 * eps + 2 * eta
    r_local = 16 * eps
    f_NL = mp.mpf(5) / 12 * (n_s - 1)

    # Inflationary Hubble scale
    # H_inf^2 = V_plateau / (3*M_Pl^2) ~ M^2/4
    H_inf = M_GeV / 2

    # Nonlocal parameter
    z_H = H_inf ** 2 / Lambda_mp ** 2

    # Tensor modification
    Pi_TT_val = Pi_TT(float(z_H), dps=dps)

    # Modified tensor-to-scalar ratio
    # r = r_local / Pi_TT(z_H) (KKS result)
    Pi_TT_real = mp.re(Pi_TT_val) if isinstance(Pi_TT_val, mp.mpc) else Pi_TT_val
    if float(Pi_TT_real) <= 0:
        r_nonlocal = mp.nan  # Pi_TT crossed zero: perturbative breakdown
        r_valid = False
    else:
        r_nonlocal = r_local / Pi_TT_real
        r_valid = True

    # Modified tensor tilt
    # n_t is modified by the derivative of Pi_TT
    # n_t = -2*epsilon / Pi_TT (approximate)
    n_t_local = -2 * eps
    n_t_nonlocal = n_t_local  # First approximation: unchanged at leading order

    # Check consistency relation
    consistency_local = float(r_local + 8 * n_t_local)  # Should be 0

    return {
        "N": N,
        "Lambda_GeV": float(Lambda_mp),
        "xi": float(xi),
        "z_H": float(z_H),
        "H_inf_GeV": float(H_inf),
        "M_GeV": float(M_GeV),
        "n_s": float(n_s),
        "r_local": float(r_local),
        "r_nonlocal": float(r_nonlocal) if r_valid else None,
        "r_valid": r_valid,
        "Pi_TT_at_zH": float(Pi_TT_real),
        "r_modification_factor": float(1 / Pi_TT_real) if r_valid else None,
        "n_t_local": float(n_t_local),
        "n_t_nonlocal": float(n_t_nonlocal),
        "f_NL_local": float(f_NL),
        "consistency_r_minus_8nt": consistency_local,
        "nonlocal_significant": float(z_H) > 0.01,
    }


# ============================================================
# Section 6: Reheating
# ============================================================

def scalaron_decay_rate(M_GeV: float | mp.mpf) -> mp.mpf:
    """Scalaron decay rate Gamma_phi = M^3 / (48*pi*M_Pl^2).

    Dominant channel: scalaron -> SM particles via conformal coupling.
    Returns Gamma in GeV.
    """
    M = mp.mpf(M_GeV)
    return M ** 3 / (48 * mp.pi * M_PL_GEV ** 2)


def reheating_temperature(M_GeV: float | mp.mpf, g_star: float | mp.mpf = G_STAR_SM) -> mp.mpf:
    """Reheating temperature T_reh in GeV.

    T_reh = (90/(pi^2*g_*))^{1/4} * sqrt(Gamma*M_Pl).
    """
    Gamma = scalaron_decay_rate(M_GeV)
    g = mp.mpf(g_star)
    return (90 / (mp.pi ** 2 * g)) ** mp.mpf("0.25") * mp.sqrt(Gamma * M_PL_GEV)


def reheating_efolds(T_reh_GeV: float | mp.mpf, M_GeV: float | mp.mpf) -> mp.mpf:
    """Number of reheating e-folds N_reh.

    N_reh = (1/4)*ln(V_end / rho_reh) where V_end = (3/16)*M^2*M_Pl^2
    (the potential at phi_end where epsilon=1, i.e. x=1/2) and
    rho_reh = (pi^2*g_*/30)*T_reh^4.
    """
    T = mp.mpf(T_reh_GeV)
    M = mp.mpf(M_GeV)
    # V_end at phi_end = sqrt(3/2)*ln(2), where x=1/2: V = (3/16)*(M/M_P)^2 * M_P^4
    V_end = mp.mpf(3) / 16 * (M / M_PL_GEV) ** 2 * M_PL_GEV ** 4
    rho_reh = mp.pi ** 2 * G_STAR_SM / 30 * T ** 4
    return mp.mpf(1) / 4 * mp.log(V_end / rho_reh)


def reheating_analysis(N: int = 55) -> dict[str, Any]:
    """Full reheating analysis.

    Returns decay rate, T_reh, N_reh, and BBN check.
    """
    mp.mp.dps = DPS
    M_ratio = required_scalaron_mass(N)
    M_GeV = M_ratio * M_PL_GEV

    Gamma = scalaron_decay_rate(float(M_GeV))
    T_reh = reheating_temperature(float(M_GeV))
    N_reh = reheating_efolds(float(T_reh), float(M_GeV))
    bbn_ok = float(T_reh) > float(T_BBN_MEV * mp.mpf("1e-3"))

    return {
        "M_GeV": float(M_GeV),
        "Gamma_GeV": float(Gamma),
        "T_reh_GeV": float(T_reh),
        "T_reh_over_BBN": float(T_reh / (T_BBN_MEV * mp.mpf("1e-3"))),
        "N_reh": float(N_reh),
        "BBN_satisfied": bbn_ok,
    }


# ============================================================
# Section 7: Viable parameter space
# ============================================================

def viable_parameter_space(N: int = 55) -> dict[str, Any]:
    """Analyze the (Lambda, xi) parameter space for spectral inflation.

    Documents:
    1. The scalaron mass problem (M >> M_Pl for SM-only at xi=0)
    2. The required |xi - 1/6| for A_s matching in the one-loop EFT
    3. The nonlocal correction strength as function of Lambda

    Returns comprehensive analysis dict.
    """
    mp.mp.dps = DPS

    M_req = required_scalaron_mass(N)
    c2_req = required_c2(N)
    xi_dev_req = required_xi_deviation(N)

    # One-loop EFT: scalaron mass at various xi
    xi_values = [0.0, 0.01, 0.1, 0.5, 1.0, 5.0]
    mass_table = []
    for xi_val in xi_values:
        m_ratio = scalaron_mass(xi_val)
        mr = mass_ratio(xi_val, N)
        mass_table.append({
            "xi": xi_val,
            "alpha_R": float(alpha_R(xi_val)),
            "c_2": float(c2_coefficient(xi_val)),
            "M_over_MPl": float(m_ratio) if m_ratio != mp.inf else "inf",
            "M_over_M_required": float(mr) if mr != mp.inf else "inf",
        })

    # Nonlocal corrections at different Lambda for fixed M
    M_GeV = M_req * M_PL_GEV
    H_inf = M_GeV / 2
    Lambda_scan = []
    for log_Lambda in range(10, 20):
        Lambda_GeV = mp.mpf(10) ** log_Lambda
        z_H = H_inf ** 2 / Lambda_GeV ** 2

        if float(z_H) > 100:
            Pi_val = mp.nan
        elif float(z_H) < 1e-12:
            Pi_val = mp.mpf(1) + LOCAL_C2 * z_H  # linear approximation
        else:
            Pi_val = Pi_TT(float(z_H), dps=50)

        Pi_real_val = mp.re(Pi_val) if isinstance(Pi_val, mp.mpc) else Pi_val
        is_nan = mp.isnan(Pi_real_val) if not isinstance(Pi_real_val, float) else math.isnan(Pi_real_val)
        Pi_float = float(Pi_real_val) if not is_nan else None
        Lambda_scan.append({
            "Lambda_GeV": float(Lambda_GeV),
            "log10_Lambda_GeV": log_Lambda,
            "z_H": float(z_H),
            "Pi_TT": Pi_float if not is_nan else None,
            "r_modification": 1 / Pi_float if Pi_float is not None and Pi_float > 0 else None,
        })

    return {
        "N": N,
        "M_required_over_MPl": float(M_req),
        "M_required_GeV": float(M_GeV),
        "H_inf_GeV": float(H_inf),
        "c2_required": float(c2_req),
        "alpha_R_required": float(16 * mp.pi ** 2 * c2_req),
        "xi_deviation_required": float(xi_dev_req),
        "mass_table": mass_table,
        "Lambda_scan": Lambda_scan,
        "scalaron_mass_problem": {
            "M_SCT_xi0_over_MPl": float(scalaron_mass(0)),
            "M_SCT_xi0_over_M_required": float(mass_ratio(0, N)),
            "diagnosis": (
                "The one-loop SM spectral action R^2 coefficient "
                "alpha_R(xi=0)=1/18 gives M~15.4*M_Pl, exceeding the "
                f"required M~{float(M_req):.2e}*M_Pl by a factor of "
                f"~{float(mass_ratio(0, N)):.2e}. "
                "Inflation requires either |xi-1/6|~2e5 (unphysical), "
                "additional scalar fields, or higher-loop corrections."
            ),
        },
    }


# ============================================================
# Section 8: De Sitter conjecture
# ============================================================

def de_sitter_conjecture(
    phi_over_MPl: float | mp.mpf,
    M_over_MPl: float | mp.mpf,
) -> dict[str, Any]:
    """Evaluate the de Sitter conjecture |nabla V|/V >= c ~ O(1).

    For the Starobinsky potential, this is violated on the inflationary plateau.
    Also checks the refined conjecture: min(nabla_i nabla_j V) <= -c'*V.
    """
    V, Vp, Vpp = starobinsky_potential_derivatives(phi_over_MPl, M_over_MPl)

    if V == 0:
        return {"phi_over_MPl": float(phi_over_MPl), "V": 0, "checks": "V=0, trivially satisfied"}

    grad_V_over_V = abs(Vp) / V  # |V'|/V (in M_Pl units)
    epsilon_V = grad_V_over_V ** 2 / 2

    # Refined conjecture: check if V'' < 0 (tachyonic direction)
    mass_squared_over_V = Vpp / V  # eta_V
    refined_satisfied = float(Vpp) < 0  # Second condition: min eigenvalue negative

    return {
        "phi_over_MPl": float(phi_over_MPl),
        "V_over_MPl4": float(V),
        "grad_V_over_V": float(grad_V_over_V),
        "epsilon_V": float(epsilon_V),
        "eta_V": float(mass_squared_over_V),
        "first_condition_satisfied": float(grad_V_over_V) >= 1.0,
        "refined_condition_satisfied": refined_satisfied,
        "diagnosis": (
            (
                f"First condition (|nabla V|/V >= O(1)) is SATISFIED "
                f"(|V'|/V = {float(grad_V_over_V):.4e} >= 1). "
                if float(grad_V_over_V) >= 1.0
                else
                f"First condition (|nabla V|/V >= O(1)) is VIOLATED during slow roll "
                f"(|V'|/V = {float(grad_V_over_V):.4e} << 1). "
            )
            + f"Refined condition (V'' < 0) is {'satisfied' if refined_satisfied else 'not satisfied'}."
        ),
    }


# ============================================================
# Section 9: CMB comparison
# ============================================================

def cmb_comparison(N: int = 55, Lambda_GeV: float | mp.mpf | None = None) -> dict[str, Any]:
    """Compare SCT predictions with Planck 2018 and BICEP/Keck 2021.

    If Lambda_GeV is provided, includes nonlocal corrections.
    Otherwise uses local Starobinsky predictions.
    """
    mp.mp.dps = DPS

    # Local predictions
    obs = inflationary_observables(N)
    n_s_pred = obs["n_s"]
    r_pred = obs["r"]
    f_NL_pred = obs["f_NL_local"]

    # Nonlocal corrections
    if Lambda_GeV is not None:
        nl = nonlocal_corrections(N, Lambda_GeV)
        r_pred = nl["r_nonlocal"] if nl["r_valid"] else None
        r_valid = nl["r_valid"]
        Pi_TT_val = nl["Pi_TT_at_zH"]
        z_H = nl["z_H"]
    else:
        r_valid = True
        Pi_TT_val = 1.0
        z_H = 0.0

    # Compare with Planck
    ns_tension = abs(n_s_pred - float(NS_PLANCK)) / float(NS_PLANCK_ERR)
    r_ok = r_pred is not None and r_pred < float(R_BICEP_UPPER) if r_valid else None
    fNL_tension = abs(f_NL_pred - float(FNL_PLANCK)) / float(FNL_PLANCK_ERR)

    # Planck compatibility: within 2-sigma
    ns_compatible = ns_tension < 2.0

    return {
        "N": N,
        "Lambda_GeV": float(Lambda_GeV) if Lambda_GeV is not None else None,
        "n_s_predicted": n_s_pred,
        "n_s_observed": float(NS_PLANCK),
        "n_s_tension_sigma": ns_tension,
        "n_s_compatible": ns_compatible,
        "r_predicted": r_pred,
        "r_upper_bound": float(R_BICEP_UPPER),
        "r_satisfies_bound": r_ok,
        "f_NL_predicted": f_NL_pred,
        "f_NL_observed": float(FNL_PLANCK),
        "f_NL_tension_sigma": fNL_tension,
        "z_H": z_H,
        "Pi_TT": Pi_TT_val,
        "overall_compatible": ns_compatible and (r_ok is True or r_ok is None),
    }


# ============================================================
# Section 10: Figures
# ============================================================

def make_figures() -> list[str]:
    """Generate all INF-1 publication figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from sct_tools.plotting import init_style  # noqa: F811
        init_style()
    except ImportError:
        pass

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    figures_created = []
    mp.mp.dps = 50

    # ---- Figure 1: (n_s, r) plane with Planck/BICEP contours ----
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    # Planck 2018 1-sigma and 2-sigma ellipses (approximate)
    from matplotlib.patches import Ellipse
    ns_c, r_c = 0.9649, 0.0  # Planck center (r marginalized)
    for nsig, alpha in [(1, 0.3), (2, 0.15)]:
        ellipse = Ellipse(
            (ns_c, r_c),
            width=2 * nsig * 0.0042,
            height=2 * nsig * 0.018,  # approximate r width
            facecolor="blue",
            alpha=alpha,
            edgecolor="blue",
            linewidth=0.5,
            label=f"Planck 2018 {nsig}$\\sigma$" if nsig == 1 else None,
        )
        ax.add_patch(ellipse)

    # BICEP/Keck upper bound
    ax.axhline(y=0.036, color="red", linestyle="--", linewidth=1, label="BICEP/Keck 2021 (95% CL)")

    # Starobinsky predictions for N=50,55,60
    N_vals = list(range(45, 65))
    ns_star = [1 - 2 / N_ for N_ in N_vals]
    r_star = [12 / N_ ** 2 for N_ in N_vals]
    ax.plot(ns_star, r_star, "k-", linewidth=2, label="Starobinsky $R^2$")

    # Mark N=50, 55, 60
    for N_mark in [50, 55, 60]:
        ns_m = 1 - 2 / N_mark
        r_m = 12 / N_mark ** 2
        ax.plot(ns_m, r_m, "ko", markersize=6)
        ax.annotate(f"$N={N_mark}$", (ns_m, r_m), textcoords="offset points",
                    xytext=(8, 5), fontsize=8)

    # SCT nonlocal modifications: for Lambda = 10^14, 10^15 GeV
    M_ratio = required_scalaron_mass(55)
    M_GeV = float(M_ratio * M_PL_GEV)
    H_inf = M_GeV / 2

    for Lambda_exp, color, ls in [(14, "green", "-."), (15, "purple", ":")]:
        Lambda_GeV = 10 ** Lambda_exp
        z_H_v = H_inf ** 2 / Lambda_GeV ** 2
        if z_H_v < 100 and z_H_v > 1e-14:
            Pi_raw = Pi_TT(z_H_v, dps=50)
            Pi_v = float(mp.re(Pi_raw)) if isinstance(Pi_raw, mp.mpc) else float(Pi_raw)
            if Pi_v > 0:
                ns_nl = [1 - 2 / N_ for N_ in N_vals]
                r_nl = [12 / N_ ** 2 / Pi_v for N_ in N_vals]
                ax.plot(ns_nl, r_nl, color=color, linestyle=ls, linewidth=1.5,
                        label=f"SCT $\\Lambda=10^{{{Lambda_exp}}}$ GeV")

    ax.set_xlabel("$n_s$")
    ax.set_ylabel("$r$")
    ax.set_xlim(0.94, 0.98)
    ax.set_ylim(0, 0.05)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("SCT spectral inflation: $(n_s, r)$ plane")
    fig.tight_layout()
    path1 = FIGURES_DIR / "inf1_ns_r_plane.pdf"
    fig.savefig(str(path1), dpi=300, bbox_inches="tight")
    plt.close(fig)
    figures_created.append(str(path1))

    # ---- Figure 2: Starobinsky potential ----
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    phi_vals = np.linspace(0, 8, 300)
    V_vals = [float(starobinsky_potential(p, float(M_ratio))) for p in phi_vals]
    V_plateau = float(mp.mpf(3) / 4 * M_ratio ** 2)
    V_norm = [v / V_plateau for v in V_vals]

    ax.plot(phi_vals, V_norm, "k-", linewidth=2)

    # Mark phi_N for N=55
    phi_55 = float(phi_N(55, float(M_ratio)))
    V_55 = float(starobinsky_potential(phi_55, float(M_ratio))) / V_plateau
    ax.plot(phi_55, V_55, "ro", markersize=8, label=f"$N=55$, $\\varphi/M_P={phi_55:.2f}$")

    # Slow-roll region
    ax.axhspan(0.95, 1.05, alpha=0.1, color="blue", label="Plateau (slow roll)")

    ax.set_xlabel("$\\varphi / M_P$")
    ax.set_ylabel("$V(\\varphi) / V_0$")
    ax.set_title("Starobinsky potential in Einstein frame")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.15)
    fig.tight_layout()
    path2 = FIGURES_DIR / "inf1_potential.pdf"
    fig.savefig(str(path2), dpi=300, bbox_inches="tight")
    plt.close(fig)
    figures_created.append(str(path2))

    # ---- Figure 3: r modification vs Lambda ----
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    log_Lambda_vals = np.linspace(12, 19, 100)
    r_mod_vals = []
    for ll in log_Lambda_vals:
        Lambda_v = 10 ** ll
        z_v = H_inf ** 2 / Lambda_v ** 2
        if z_v < 50 and z_v > 1e-14:
            Pi_raw_v = Pi_TT(z_v, dps=50)
            Pi_v = float(mp.re(Pi_raw_v)) if isinstance(Pi_raw_v, mp.mpc) else float(Pi_raw_v)
            if Pi_v > 0.01:
                r_mod_vals.append(1 / Pi_v)
            else:
                r_mod_vals.append(np.nan)
        elif z_v <= 1e-14:
            r_mod_vals.append(1.0)
        else:
            r_mod_vals.append(np.nan)

    ax.plot(log_Lambda_vals, r_mod_vals, "k-", linewidth=2)
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.8, label="Local Starobinsky")
    ax.axhspan(0.99, 1.01, alpha=0.1, color="green", label="$<1\\%$ correction")

    ax.set_xlabel("$\\log_{10}(\\Lambda / \\mathrm{GeV})$")
    ax.set_ylabel("$r_{\\mathrm{nonlocal}} / r_{\\mathrm{local}}$")
    ax.set_title("Tensor-to-scalar ratio modification from nonlocality")
    ax.legend(fontsize=8)
    ax.set_xlim(12, 19)
    ax.set_ylim(0, 3)
    fig.tight_layout()
    path3 = FIGURES_DIR / "inf1_scalaron_mass.pdf"
    fig.savefig(str(path3), dpi=300, bbox_inches="tight")
    plt.close(fig)
    figures_created.append(str(path3))

    return figures_created


# ============================================================
# Section 11: Full results and JSON output
# ============================================================

def compute_all_results() -> dict[str, Any]:
    """Compute all INF-1 results.

    Returns a comprehensive dict suitable for JSON serialization.
    """
    mp.mp.dps = DPS

    results: dict[str, Any] = {
        "phase": "INF-1",
        "title": "Spectral Inflation from the R^2 sector",
    }

    # Local observables at N=50,55,60
    local_obs = {}
    for N_ in [50, 55, 60]:
        local_obs[f"N{N_}"] = inflationary_observables(N_)
        local_obs[f"N{N_}_exact"] = inflationary_observables_exact(N_)
    results["local_observables"] = local_obs

    # Reheating
    results["reheating"] = reheating_analysis(55)

    # Parameter space
    results["parameter_space"] = viable_parameter_space(55)

    # Nonlocal corrections at different Lambda
    nl_results = {}
    for log_L in [13, 14, 15, 16, 18]:
        key = f"Lambda_1e{log_L}"
        nl_results[key] = nonlocal_corrections(55, Lambda_GeV=mp.mpf(10) ** log_L)
    results["nonlocal_corrections"] = nl_results

    # CMB comparison
    results["cmb_comparison_local"] = cmb_comparison(55)
    results["cmb_comparison_nonlocal_1e14"] = cmb_comparison(55, Lambda_GeV=mp.mpf("1e14"))
    results["cmb_comparison_nonlocal_1e15"] = cmb_comparison(55, Lambda_GeV=mp.mpf("1e15"))

    # De Sitter conjecture
    M_ratio = required_scalaron_mass(55)
    phi_55 = float(phi_N(55, float(M_ratio)))
    results["de_sitter_conjecture"] = {
        "at_phi_N55": de_sitter_conjecture(phi_55, float(M_ratio)),
        "at_phi_1": de_sitter_conjecture(1.0, float(M_ratio)),
        "at_phi_5": de_sitter_conjecture(5.0, float(M_ratio)),
    }

    # Scalaron mass problem summary
    results["scalaron_mass_problem"] = {
        "c2_required": float(required_c2(55)),
        "c2_SM_xi0": float(c2_coefficient(0)),
        "ratio_c2": float(required_c2(55) / c2_coefficient(0)),
        "M_SM_xi0_over_MPl": float(scalaron_mass(0)),
        "M_required_over_MPl": float(required_scalaron_mass(55)),
        "xi_deviation_required": float(required_xi_deviation(55)),
        "diagnosis": (
            "NEGATIVE RESULT: The one-loop SM spectral action at xi=0 gives "
            "c_2=3.52e-4, while A_s matching requires c_2~5.07e8. "
            "The ratio is ~1.44e12, corresponding to |xi-1/6|~2e5. "
            "Spectral inflation with SM-only content requires BSM physics "
            "(additional scalars, modified spectral triple, or higher-loop "
            "corrections) to generate the correct scalaron mass."
        ),
    }

    # Key predictions summary
    obs_55 = inflationary_observables(55)
    reh = reheating_analysis(55)
    results["predictions_summary"] = {
        "n_s": obs_55["n_s"],
        "r_local": obs_55["r"],
        "r_range_nonlocal": "r < 3.97e-3 (local) to suppressed by 1/Pi_TT for Lambda ~ H_inf",
        "f_NL_local": obs_55["f_NL_local"],
        "T_reh_GeV": reh["T_reh_GeV"],
        "BBN_satisfied": reh["BBN_satisfied"],
        "M_scalaron_GeV": obs_55["M_GeV"],
        "status": "CONDITIONAL: predictions valid if scalaron mass is externally fixed to 3.1e13 GeV",
    }

    return results


# ============================================================
# Main
# ============================================================

def main() -> None:
    """Run INF-1 computation and produce all outputs."""
    mp.mp.dps = DPS
    print("=" * 70)
    print("INF-1: Spectral Inflation from the R^2 Sector")
    print("=" * 70)

    # Self-tests (CQ3)
    print("\n--- Self-Tests ---")
    _run_self_tests()

    # Compute results
    print("\n--- Computing Full Results ---")
    results = compute_all_results()

    # Save JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / "inf1_inflation_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {json_path}")

    # Generate figures
    print("\n--- Generating Figures ---")
    try:
        figures = make_figures()
        for fig_path in figures:
            print(f"  Figure: {fig_path}")
    except Exception as e:
        print(f"  Figure generation failed: {e}")

    # Summary
    print("\n--- Summary ---")
    obs55 = results["local_observables"]["N55"]
    print(f"  n_s(N=55) = {obs55['n_s']:.6f}")
    print(f"  r(N=55)   = {obs55['r']:.4e}")
    print(f"  f_NL      = {obs55['f_NL_local']:.6f}")
    print(f"  M/M_Pl    = {obs55['M_over_MPl']:.4e}")
    print(f"  M (GeV)   = {obs55['M_GeV']:.4e}")

    reh = results["reheating"]
    print(f"  T_reh     = {reh['T_reh_GeV']:.4e} GeV")
    print(f"  BBN OK    = {reh['BBN_satisfied']}")

    mass_prob = results["scalaron_mass_problem"]
    print("\n  SCALARON MASS PROBLEM:")
    print(f"    M_SCT(xi=0)/M_req = {mass_prob['M_SM_xi0_over_MPl']/mass_prob['M_required_over_MPl']:.4e}")
    print(f"    |xi-1/6| needed   = {mass_prob['xi_deviation_required']:.4e}")

    print("\nINF-1 computation complete.")


def _run_self_tests() -> None:
    """Internal self-tests (CQ3)."""
    mp.mp.dps = DPS
    passed = 0
    failed = 0

    def check(name: str, condition: bool) -> None:
        nonlocal passed, failed
        if condition:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: {name}")

    # T1: alpha_R(0) = 1/18
    check("alpha_R(0)=1/18", abs(alpha_R(0) - mp.mpf(1) / 18) < 1e-15)

    # T2: alpha_R(1/6) = 0
    check("alpha_R(1/6)=0", alpha_R(mp.mpf(1) / 6) == 0)

    # T3: scalaron mass at xi=0
    check("M(xi=0)=sqrt(24*pi^2)*M_Pl", abs(scalaron_mass(0) - mp.sqrt(24 * mp.pi ** 2)) < 1e-5)

    # T4: required M
    M_req = required_scalaron_mass(55)
    check("M_req ~ 1.28e-5", abs(M_req - mp.mpf("1.282e-5")) < 1e-7)

    # T5: slow-roll n_s at N=55 (full formula: 1 - 6*eps + 2*eta = 0.9621)
    obs = inflationary_observables(55)
    check("n_s(55) ~ 0.9621", abs(obs["n_s"] - 0.9621) < 0.001)

    # T6: slow-roll r at N=55
    check("r(55) ~ 3.97e-3", abs(obs["r"] - 3.967e-3) < 1e-4)

    # T7: f_NL
    check("f_NL ~ -0.015", abs(obs["f_NL_local"] + 0.015) < 0.002)

    # T8: consistency relation
    check("r + 8*n_t = 0", abs(obs["consistency_r_minus_8nt"]) < 1e-10)

    # T9: reheating
    reh = reheating_analysis(55)
    check("T_reh > 5 MeV", reh["BBN_satisfied"])

    # T10: potential V(0) = 0
    check("V(0) = 0", abs(starobinsky_potential(0, float(M_req))) < 1e-30)

    # T11: potential plateau
    V_plateau = mp.mpf(3) / 4 * M_req ** 2
    V_large = starobinsky_potential(10, float(M_req))
    check("V(10) ~ V_plateau", abs(V_large / V_plateau - 1) < 0.01)

    # T12: Pi_TT(0) = 1
    check("Pi_TT(0) = 1", abs(Pi_TT(0) - 1) < 1e-14)

    # T13: conformal coupling kills inflation
    check("alpha_R(1/6) = 0", alpha_R(mp.mpf(1) / 6) == 0)
    check("c_2(1/6) = 0", c2_coefficient(mp.mpf(1) / 6) == 0)

    # T14: mass ratio >> 1 at xi=0
    check("mass_ratio(xi=0) >> 1", float(mass_ratio(0, 55)) > 1e5)

    # T15: required xi deviation >> 1
    check("|xi-1/6|_req >> 1", float(required_xi_deviation(55)) > 1e4)

    print(f"  Self-tests: {passed} PASS, {failed} FAIL")
    if failed > 0:
        print("  WARNING: Some self-tests failed!")


if __name__ == "__main__":
    main()

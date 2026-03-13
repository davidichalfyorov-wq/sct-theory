"""
SCT Theory — Physical constants and conventions.

Unit conventions:
    - Natural units: c = hbar = 1 unless stated otherwise
    - SI restoration available via restore_si()
    - CODATA 2022 values from scipy.constants

Sign conventions (SCT standard):
    - Metric signature: (-,+,+,+)
    - Riemann: R^rho_{sigma mu nu} = d_mu Gamma^rho_{nu sigma} - ...
    - Ricci: R_{mu nu} = R^rho_{mu rho nu}  (contraction on 1st and 3rd indices)
    - Scalar curvature: R = g^{mu nu} R_{mu nu}
    - Einstein: G_{mu nu} = R_{mu nu} - (1/2) g_{mu nu} R
    - Dirac endomorphism: E = -R/4  (CORRECTED from +R/4, see NT-1 sign fix)
"""

from fractions import Fraction

import numpy as np
from scipy import constants as sc

# =============================================================================
# FUNDAMENTAL CONSTANTS (CODATA 2022, SI)
# =============================================================================
c = sc.c                    # speed of light [m/s]
hbar = sc.hbar              # reduced Planck [J·s]
G_N = sc.G                  # Newton's constant [m^3/(kg·s^2)]
k_B = sc.k                  # Boltzmann [J/K]

# Derived Planck units
M_Pl = np.sqrt(hbar * c / G_N)          # Planck mass [kg]
l_Pl = np.sqrt(hbar * G_N / c**3)       # Planck length [m]
t_Pl = np.sqrt(hbar * G_N / c**5)       # Planck time [s]
E_Pl = M_Pl * c**2                      # Planck energy [J]

# Reduced Planck mass (M_Pl / sqrt(8*pi))
M_Pl_red = M_Pl / np.sqrt(8 * np.pi)

# Planck mass in GeV
M_Pl_GeV = M_Pl * c**2 / sc.eV * 1e-9   # ~ 1.22e19 GeV

# =============================================================================
# PARTICLE PHYSICS
# =============================================================================
alpha_em = sc.alpha                      # fine structure constant
alpha_s_mZ = 0.1180                      # strong coupling at m_Z (PDG 2024)
sin2_thetaW = 0.23122                    # weak mixing angle (PDG 2024)
G_F = sc.physical_constants['Fermi coupling constant'][0]  # [GeV^-2] * (hbar*c)^3

# Useful conversions
GeV_to_fm = sc.hbar * sc.c / sc.eV * 1e6  # 1 GeV^-1 in fm (= 0.197...)
# GeV -> kg conversion is in the NATURAL UNIT CONVERSIONS section as GeV_to_kg_exact

# =============================================================================
# SM FIELD CONTENT (for spectral action multiplicities)
# =============================================================================
# Standard Model field multiplicities for form factor summation
# F_i^total = N_s * F_i^(0) + N_f * F_i^(1/2) + N_v * F_i^(1)

N_scalar = 4      # Higgs doublet: 4 real d.o.f.
N_dirac = 45      # SM Weyl fermion count for spectral action form factors:
                   # 3 generations × 15 Weyl spinors/gen = 45
                   # Per generation: (3 colors × 2 chiralities × 2 quarks)
                   #   + (2 chiralities × 1 lepton) + 1 neutrino_L = 12 + 2 + 1 = 15
                   # NOTE: despite the variable name "N_dirac", the value 45
                   # counts 2-component WEYL spinors, not 4-component Dirac.
                   # The h_C^(1/2) form factor is for a single Dirac fermion
                   # = 2 Weyl, so the physical sum uses N_dirac/2 Dirac fields.
                   # PHYSICS REVIEW: verify convention matches NT-1b Phase 3.
N_vector = 12     # SU(3)×SU(2)×U(1): 8 + 3 + 1 = 12 gauge bosons

# Short aliases used throughout the codebase (verification, form_factors, etc.)
N_s = N_scalar
N_f = N_dirac
N_v = N_vector

# Dirac-equivalent count: N_f counts 2-component Weyl spinors,
# but h_C^(1/2) and h_R^(1/2) are per 4-component Dirac fermion.
# N_D = N_f / 2 = 22.5 is the number of Dirac-equivalent fermions.
# This is NOT an integer, which is physically correct: the SM has an
# odd number of Weyl spinors (45) because of right-handed neutrinos absent.
# Reference: CPR 0805.2909 uses n_D = number of Dirac fermions.
N_D = N_dirac / 2  # = 22.5

# =============================================================================
# LOCAL LIMIT BETA COEFFICIENTS (from a_4 Seeley-DeWitt)
# =============================================================================
# beta_W^(s) = |h_C^(s)(0)| for spin s
# beta_R^(s) depends on coupling for scalar

BETA_W = {
    0: Fraction(1, 120),    # scalar (minimal or non-minimal — same)
    0.5: Fraction(1, 20),   # Dirac fermion (note: sign is -1/20 for h_C, but
                            # beta_W = |h_C(0)| = 1/20)
    1: Fraction(1, 10),     # vector (gauge boson, physical = unconstrained - 2 ghosts)
                            # Unconstrained: 7/60 = 14/120.
                            # Ghost (xi=0 scalar): 1/120 each, 2 ghosts = 2/120.
                            # Physical: 14/120 - 2/120 = 12/120 = 1/10.
                            # NOTE: the previous comment "Expected: 13/120" was WRONG.
                            # 13/120 = 14/120 - 1/120 subtracts only ONE ghost,
                            # but FP procedure produces ghost + anti-ghost = 2 real scalars.
                            # Verified: CPR (0805.2909), BV (1990), Vassilevich (0306138).
}

BETA_R = {
    # For scalar: beta_R = (1/2)(xi - 1/6)^2 — use beta_R_scalar(xi)
    # For Dirac: beta_R = 0 (conformal invariance at massless level)
    # For vector: beta_R = 0 (conformal invariance, verified Phase 2)
    0.5: 0,
    1: 0,
}

def beta_R_scalar(xi):
    """beta_R for real scalar with non-minimal coupling xi.
    beta_R^(0)(xi) = (1/2)(xi - 1/6)^2
    """
    if not isinstance(xi, (int, float, np.integer, np.floating)):
        raise TypeError(f"beta_R_scalar: xi must be numeric, got {type(xi).__name__}")
    if not np.isfinite(float(xi)):
        raise ValueError(f"beta_R_scalar: xi must be finite, got {xi}")
    return 0.5 * (xi - 1/6)**2


# =============================================================================
# SCT-SPECIFIC PARAMETERS
# =============================================================================
# Spectral cutoff scale (to be determined from data)
# Lambda is the fundamental scale in the spectral action
# Current constraint: Lambda > ~10^16 GeV (from non-observation of Lorentz violation)

# Donoghue coefficient (VERIFIED prediction)
beta_Donoghue = 41 / (10 * np.pi)

# =============================================================================
# PHASE 3 RESULTS: COMBINED SM COEFFICIENTS (NT-1b Phase 3, VERIFIED)
# =============================================================================
# Total Weyl-squared coefficient (xi-independent):
#   alpha_C = N_s/120 + (N_f/2)*(-1/20) + N_v/10
#           = 4/120 - 45/40 + 12/10 = 13/120
alpha_C_SM = Fraction(13, 120)

# Total R-squared coefficient (xi-dependent):
#   alpha_R(xi) = N_s * (1/2)(xi-1/6)^2 = 2*(xi-1/6)^2  [for SM N_s=4]
#   Dirac and vector: beta_R = 0 (conformal invariance)
# alpha_R is a function of xi, stored in form_factors.alpha_R_SM(xi)

# c_1/c_2 ratio in {R^2, R_{mn}^2} basis:
#   c_1/c_2 = -1/3 + alpha_R / (2*alpha_C) = -1/3 + 120*(xi-1/6)^2 / 13
#   At conformal coupling xi=1/6: c_1/c_2 = -1/3 (original prediction confirmed)
#   At minimal coupling xi=0:     c_1/c_2 = -1/13
c1_c2_ratio_conformal = Fraction(-1, 3)  # xi = 1/6 special case

# Legacy alias (kept for backward compatibility with Phase 1/2 code)
c1_c2_ratio_original = -1/3

# Scalar mode decoupling: 3*c_1 + c_2 = 3*alpha_R(xi) = 6*(xi-1/6)^2
# Decoupling requires xi = 1/6 (conformal coupling).
# This is a PHYSICAL CONSTRAINT on the Higgs non-minimal coupling.


# =============================================================================
# UNIT RESTORATION
# =============================================================================
def natural_to_si(quantity, dim_mass=0, dim_length=0, dim_time=0):
    """Convert from natural units (c=hbar=1) to SI.

    In natural units, everything is measured in powers of mass (GeV).
    To restore SI: multiply by (hbar*c)^a * c^b where a,b depend on dimensions.

    Parameters:
        quantity: numerical value in natural units (GeV^dim_mass)
        dim_mass: mass dimension (accepted for API clarity but does not
            affect conversion — mass is the base unit in natural units)
        dim_length: length dimension
        dim_time: time dimension

    Returns:
        Value in SI units (kg^m * m^l * s^t).
    """
    if not np.isfinite(float(quantity)):
        raise ValueError(
            f"natural_to_si: requires finite quantity, got {quantity}"
        )
    # In natural units: [length] = [time] = [mass]^{-1}
    # SI restoration: multiply by hbar^n * c^m * G^k
    factor = hbar**(dim_length + dim_time) * c**(dim_length - dim_time)
    return quantity * factor


def log_dimensions(expr_value, expected_mass_dim, label=""):
    """Log the dimensional analysis of an expression. Does NOT verify anything."""
    prefix = f"[DIM LOG] {label}: " if label else "[DIM LOG]: "
    val = float(expr_value)
    print(f"{prefix}value = {val:.6e}, expected [mass^{expected_mass_dim}]")


def check_dimensions(expr_dim, expected_dim, label=""):
    """Check that the mass dimension of an expression matches expected.

    Parameters:
        expr_dim: integer mass dimension of the expression
        expected_dim: expected integer mass dimension
        label: description for error messages

    Returns:
        True if dimensions match

    Raises:
        ValueError if dimensions don't match
    """
    if not isinstance(expr_dim, (int, float, np.integer, np.floating)):
        raise TypeError(
            f"check_dimensions: expr_dim must be numeric, got {type(expr_dim).__name__}"
        )
    if not isinstance(expected_dim, (int, float, np.integer, np.floating)):
        raise TypeError(
            f"check_dimensions: expected_dim must be numeric, got {type(expected_dim).__name__}"
        )
    if np.isnan(float(expr_dim)):
        raise ValueError(
            "check_dimensions: expr_dim is NaN"
            + (f" in {label}" if label else "")
        )
    if np.isnan(float(expected_dim)):
        raise ValueError(
            "check_dimensions: expected_dim is NaN"
            + (f" in {label}" if label else "")
        )
    # Use tolerance for float dimensions to avoid spurious failures
    if abs(float(expr_dim) - float(expected_dim)) > 1e-12:
        msg = "Dimension mismatch"
        if label:
            msg += f" in {label}"
        msg += f": got [mass^{expr_dim}], expected [mass^{expected_dim}]"
        raise ValueError(msg)
    return True


# =============================================================================
# NATURAL UNIT CONVERSIONS
# =============================================================================
# Conversion factors for moving between natural units (c=hbar=1)
# and commonly used unit systems in HEP/cosmology.

# Length: 1/GeV -> meters
inv_GeV_to_m = hbar * c / (sc.eV * 1e9)   # 1 GeV^{-1} = 1.97e-16 m

# Time: 1/GeV -> seconds
inv_GeV_to_s = hbar / (sc.eV * 1e9)       # 1 GeV^{-1} = 6.58e-25 s

# Mass: GeV -> kg
GeV_to_kg_exact = sc.eV * 1e9 / c**2      # 1 GeV = 1.78e-27 kg

# Temperature: GeV -> Kelvin
GeV_to_K = sc.eV * 1e9 / k_B              # 1 GeV = 1.16e13 K

# Cross section: 1/GeV^2 -> cm^2 (for collider physics)
inv_GeV2_to_cm2 = (hbar * c / (sc.eV * 1e9))**2 * 1e4  # 1 GeV^{-2} ≈ 3.89e-28 cm²
inv_GeV2_to_pb = inv_GeV2_to_cm2 * 1e36   # 1 GeV^{-2} in picobarns

# Planck units in natural units (where c=hbar=1, G=1/M_Pl^2)
# These are the numerical values in GeV
M_Pl_natural = M_Pl_GeV                   # Planck mass in GeV ≈ 1.22e19
l_Pl_natural = 1.0 / M_Pl_GeV             # Planck length in GeV^{-1}
t_Pl_natural = 1.0 / M_Pl_GeV             # Planck time in GeV^{-1}


class NaturalUnits:
    """Context manager for natural unit computations with SI restoration.

    Usage:
        with NaturalUnits() as nu:
            # All computations in natural units (c=hbar=1)
            E_planck = nu.M_Pl  # GeV
            l = 1.0 / E_planck  # length in GeV^{-1}
            l_si = nu.to_meters(l)  # convert to SI
    """

    def __init__(self):
        self.M_Pl = M_Pl_GeV
        self.M_Pl_red = M_Pl_GeV / np.sqrt(8 * np.pi)
        self.alpha_em = alpha_em
        self.alpha_s = alpha_s_mZ
        self.sin2_thetaW = sin2_thetaW

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    @staticmethod
    def to_meters(x_inv_gev):
        """Convert length from GeV^{-1} to meters."""
        return x_inv_gev * inv_GeV_to_m

    @staticmethod
    def to_seconds(t_inv_gev):
        """Convert time from GeV^{-1} to seconds."""
        return t_inv_gev * inv_GeV_to_s

    @staticmethod
    def to_kg(m_gev):
        """Convert mass from GeV to kg."""
        return m_gev * GeV_to_kg_exact

    @staticmethod
    def to_kelvin(e_gev):
        """Convert energy/temperature from GeV to Kelvin."""
        return e_gev * GeV_to_K

    @staticmethod
    def to_cm2(sigma_inv_gev2):
        """Convert cross section from GeV^{-2} to cm^2."""
        return sigma_inv_gev2 * inv_GeV2_to_cm2

    @staticmethod
    def to_pb(sigma_inv_gev2):
        """Convert cross section from GeV^{-2} to picobarns."""
        return sigma_inv_gev2 * inv_GeV2_to_pb

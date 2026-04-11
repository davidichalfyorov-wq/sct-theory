"""
Closed-form derivation of beta from boundary + interior integrals.
===================================================================

STRATEGY:
We compute beta = delta_k / (k_flat * eps * f(p)) where delta_k = delta_k_int + delta_k_bdy.

Both terms involve integrals over the causal cone weighted by exp(-rho*V(tau)).

KEY SIMPLIFICATION: Work in the "link shell" approximation.
Most links have proper time tau near tau_0 = (24/(pi*rho))^{1/4}.
The link probability exp(-rho*V) = exp(-rho*pi/24*tau^4) peaks near tau=0
and decays on scale tau_0.

The integrals reduce to moments of the link probability distribution
weighted by du^2 (null-coordinate separation squared).

We compute these moments analytically using the known d=4 layer integrals
from Dowker-Glaser (2013).

Author: David Alfyorov
"""
import numpy as np
from scipy import integrate as sci_int
from scipy.special import gamma as Gamma
import sympy as sp

print("=" * 70)
print("CLOSED-FORM beta FROM BOUNDARY + INTERIOR INTEGRALS")
print("=" * 70)

# ====================================================================
# SETUP
# ====================================================================
# d=4 Minkowski, causal diamond, density rho = N/V_diamond
# V_diamond = pi^2/192 * T^4 (d=4 Alexandrov set volume)
# For T=1: V_diamond = pi^2/192 ≈ 0.05135
# rho = N / V_diamond

T = 1.0
V_diamond = np.pi**2 / 192
print(f"V_diamond = pi^2/192 = {V_diamond:.6f}")

# Alexandrov volume: V_Alex(tau) = (pi/24)*tau^4
# Link probability: P(tau) = exp(-rho * pi/24 * tau^4)
# Characteristic link scale: tau_0 = (24/(pi*rho))^{1/4}

# ====================================================================
# THE TWO INTEGRALS
# ====================================================================
# For a pair (p, q) separated by coordinates (U, V_c, X, Y) in null frame:
#   tau_flat^2 = s = 2*U*V_c - X^2 - Y^2
#   du_cart = dt + dz (Cartesian); in null coords: du_cart ≈ sqrt(2)*U
#   Actually: u = (t+z)/sqrt(2), so U = du = (dt+dz)/sqrt(2)
#   du_cart = dt+dz = sqrt(2)*U
#   du_cart^2 = 2*U^2
#
# The midpoint correction (position-dependent part):
#   corr = eps * 2*f(p) * du_cart^2 / 2 = eps * f(p) * du_cart^2 = eps*f(p)*2*U^2
#
# Causal condition: s > corr, i.e., tau_flat^2 > 2*eps*f(p)*U^2
#
# INTERIOR: pairs with s > 2*eps*f(p)*U^2
#   delta_V = -(pi/12)*2*eps*f(p)*U^2*s  [from delta(tau^4) = -4*eps*f*U^2*s... let me redo]
#
# Actually, let me use the EXACT midpoint formula from the code:
#   corr = eps * f_mid * (dt+dz)^2 / 2
# For position-dep part: f_mid ≈ 2*f(p), so corr = eps * 2*f(p) * (dt+dz)^2/2 = eps*f(p)*(dt+dz)^2
#
# In hyperboloid coordinates:
#   The pair has proper time tau and direction on the hyperboloid H^3.
#   dt = tau*cosh(eta), dz = tau*sinh(eta)*cos(theta)
#   du_cart = dt+dz = tau*(cosh(eta) + sinh(eta)*cos(theta))
#   du_cart^2 = tau^2 * (cosh(eta) + sinh(eta)*cos(theta))^2
#
# The correction: corr = eps*f(p)*tau^2*(cosh+sinh*cos)^2
# Causal iff: tau_flat^2 > corr, i.e., tau^2 > eps*f(p)*tau^2*(cosh+sinh*cos)^2
# => 1 > eps*f(p)*(cosh+sinh*cos)^2
# For small eps: this is always satisfied for typical f and (eta,theta).
# So the boundary contribution comes from directions where (cosh+sinh*cos)^2 is large.
#
# Wait, that can't be right. Let me reconsider.
#
# The issue: the causal condition in the CODE is:
#   mink > corr, where mink = dt^2 - dr^2 = tau_flat^2
# So: tau_flat^2 > eps*f_mid*(dt+dz)^2/2
#
# For a pair at proper time tau in direction (eta, theta, phi):
#   tau_flat^2 = tau^2 (by definition)
#   (dt+dz)^2 = tau^2*(cosh(eta)+sinh(eta)*cos(theta))^2
#   corr = eps * 2*f(p) * tau^2*(cosh+sinh*cos)^2 / 2 = eps*f(p)*tau^2*(cosh+sinh*cos)^2
#
# Condition: tau^2 > eps*f(p)*tau^2*(cosh+sinh*cos)^2
# => 1 > eps*f(p)*(cosh+sinh*cos)^2
#
# For small eps, this is TRUE for almost all directions.
# It FAILS only when (cosh+sinh*cos)^2 > 1/(eps*f(p)).
# At large eta: cosh+sinh*cos ≈ e^eta*(1+cos)/2 ≈ e^eta for cos>0
# So the condition fails when e^{2*eta} > 1/(eps*f(p)), i.e., eta > log(1/sqrt(eps*f))/2
# For eps=0.1, f~0.1: eta > log(10)/2 ≈ 1.15
#
# This means the boundary contribution comes from HIGHLY BOOSTED pairs (large eta).
# These are nearly-null pairs with large du but finite tau.

# WAIT: I think the issue is that the correction is NOT proportional to tau^2.
# Let me re-read the code:
#   mink = dt^2 - dx^2 - dy^2 - dz^2 = tau^2
#   corr = eps * f_mid * (dt+dz)^2 / 2
# These are BOTH in CARTESIAN coordinates, not hyperboloid.
# tau^2 is the proper time squared.
# (dt+dz)^2 is the squared null-coordinate separation.
# These are INDEPENDENT quantities for a given pair!
#
# In hyperboloid: tau is fixed, but (dt+dz)^2 = tau^2*(cosh+sinh*cos)^2
# grows with eta. So for large eta, (dt+dz)^2 >> tau^2.
#
# The condition tau^2 > eps*f(p)*(dt+dz)^2/2:
# At fixed tau, this fails for large enough (dt+dz)^2.
# The CRITICAL direction: (dt+dz)^2 = 2*tau^2/(eps*f(p))
# => (cosh+sinh*cos)^2 = 2/(eps*f(p))
# For eps=0.1, f=0.1: (cosh+sinh*cos)^2 = 200, so cosh+sinh*cos ≈ 14, eta ≈ 2.6.
#
# So pairs at proper time tau but with eta > eta_crit lose causality.
# The boundary contribution = lost pairs at all tau values, for eta > eta_crit.

# THIS IS THE KEY INSIGHT:
# The boundary effect is concentrated at LARGE RAPIDITY (nearly null pairs).
# These pairs have tau << r (proper time much less than spatial separation).
# They are the most "null-like" links.

print("\nKEY INSIGHT: Boundary effect = loss of high-rapidity (nearly-null) links")
print("Critical condition: (cosh(eta)+sinh(eta)*cos(theta))^2 > 2/(eps*f(p))")
print()

# ====================================================================
# COMPUTE beta_int ANALYTICALLY
# ====================================================================
# Interior contribution:
# delta_k_int = rho * integral (delta_P) d^4q
# = rho * integral P_flat * (rho * delta_V) d^4q  [delta_P/P = -rho*delta_V]
# delta_V = (pi/24)*delta(tau^4)
#
# New tau^2 = tau_flat^2 - corr = tau^2 - eps*f*tau^2*(cosh+sinh*cos)^2
#           = tau^2 * (1 - eps*f*(cosh+sinh*cos)^2)
# New tau^4 = tau^4 * (1 - eps*f*(cosh+sinh*cos)^2)^2
#           ≈ tau^4 * (1 - 2*eps*f*(cosh+sinh*cos)^2)
# delta(tau^4) = -2*eps*f*tau^4*(cosh+sinh*cos)^2
# delta_V = -(pi/12)*eps*f*tau^4*(cosh+sinh*cos)^2
# delta_P/P = +rho*(pi/12)*eps*f*tau^4*(cosh+sinh*cos)^2
#
# So: delta_k_int = rho^2*(pi/12)*eps*f * integral tau^3 * P(tau) * tau^4 * g^2 dOmega dtau
# where g = cosh(eta)+sinh(eta)*cos(theta) and dOmega = sinh^2(eta)*sin(theta) deta dtheta dphi
#
# = rho^2*(pi/12)*eps*f * I_7(rho) * J_2
# where I_7 = integral_0^inf tau^7 * exp(-rho*pi/24*tau^4) dtau
# and J_2 = integral_H3 g^2 * dOmega

# I_7 from SymPy:
rho_s = sp.Symbol('rho', positive=True)
tau_s = sp.Symbol('tau', positive=True)
I7 = sp.integrate(tau_s**7 * sp.exp(-rho_s * sp.pi/24 * tau_s**4), (tau_s, 0, sp.oo))
I3 = sp.integrate(tau_s**3 * sp.exp(-rho_s * sp.pi/24 * tau_s**4), (tau_s, 0, sp.oo))
print(f"I_7 = {I7}")
print(f"I_3 = {I3}")
print(f"I_7/I_3 = {sp.simplify(I7/I3)}")
print()

# J_2 = integral g^2 dOmega on H^3
# g = cosh(eta) + sinh(eta)*cos(theta)
# dOmega = sinh^2(eta)*sin(theta) deta dtheta dphi
# g^2 = cosh^2 + 2*cosh*sinh*cos + sinh^2*cos^2
# integral over theta (0 to pi) with sin(theta):
#   int sin*cosh^2 = 2*cosh^2
#   int sin*2*cosh*sinh*cos = 0 (odd in cos)
#   int sin*sinh^2*cos^2 = 2*sinh^2/3
# integral over phi: 2*pi
# J_2 = 2*pi * integral_0^inf sinh^2(eta) * [2*cosh^2(eta) + 2*sinh^2(eta)/3] deta

# But this DIVERGES (eta -> inf: integrand ~ e^{4*eta}).
# The divergence is the SAME one we encountered before.

# HOWEVER: in the finite diamond, eta is bounded.
# For an interior element, the maximum eta at proper time tau is:
# The pair must fit inside the diamond. At rapidity eta, the spatial
# separation is r = tau*sinh(eta). The diamond has |t|+r < T/2.
# With t = tau*cosh(eta), we need tau*cosh(eta) + tau*sinh(eta) < T/2
# => tau*e^eta < T/2 (for large eta)
# => eta < log(T/(2*tau))

# So at each tau, there's a rapidity cutoff eta_max(tau) = log(T/(2*tau)).
# For tau << T/2: eta_max is large. For tau ~ T/2: eta_max ~ 0.

# The combined integral:
# delta_k_int = rho^2*(pi/12)*eps*f * integral_0^{T/2} dtau * tau^7 * P(tau)
#   * integral_0^{eta_max(tau)} deta * sinh^2(eta) * [2*cosh^2 + 2*sinh^2/3] * 2*pi

# k_flat = rho * integral_0^{T/2} dtau * tau^3 * P(tau)
#   * integral_0^{eta_max(tau)} deta * sinh^2(eta) * 2 * 2*pi

# beta_int = delta_k_int / (k_flat * eps * f)
# = rho*(pi/12) * [integral tau^7 P J_2(eta_max(tau)) dtau] / [integral tau^3 P J_0(eta_max(tau)) dtau]

# where J_n(eta_max) = integral_0^{eta_max} g^n * sinh^2 * sin * deta dtheta dphi

# For the FLAT degree (J_0):
# J_0(eta_max) = 4*pi * integral_0^{eta_max} sinh^2(eta) deta
#              = 4*pi * [sinh(2*eta_max)/4 - eta_max/2]
#              ≈ pi*e^{2*eta_max}/2  for large eta_max

# For J_2(eta_max):
# J_2 = 2*pi * integral_0^{eta_max} sinh^2 * [2*cosh^2 + 2*sinh^2/3] deta
# For large eta_max: dominated by e^{4*eta}/6 ~ e^{4*eta_max}/24

# With eta_max = log(T/(2*tau)): e^{eta_max} = T/(2*tau)
# J_0 ≈ pi*(T/(2*tau))^2/2 = pi*T^2/(8*tau^2)
# J_2 ≈ 2*pi*(T/(2*tau))^4/24 = pi*T^4/(192*tau^4)

print("ASYMPTOTIC EVALUATION (large eta_max):")
print("J_0(eta_max) ~ pi*e^{2*eta_max}/2 = pi*T^2/(8*tau^2)")
print("J_2(eta_max) ~ pi*T^4/(192*tau^4)")
print()

# Now substitute:
# k_flat ~ rho * integral_0^{T/2} tau^3 * P(tau) * pi*T^2/(8*tau^2) dtau
#        = rho * pi*T^2/8 * integral tau * P(tau) dtau
#        = rho * pi*T^2/8 * I_1

# delta_k_int ~ rho^2*(pi/12) * integral tau^7 * P(tau) * pi*T^4/(192*tau^4) dtau
#             = rho^2*(pi/12)*(pi*T^4/192) * integral tau^3 * P(tau) dtau
#             = rho^2*pi^2*T^4/2304 * I_3

# beta_int = [rho^2*pi^2*T^4/2304 * I_3] / [rho*pi*T^2/8 * I_1 * eps * f] / (eps*f)
# Wait, beta = delta_k_int / (k_flat * eps * f), and delta_k already has eps*f.
# Let me rewrite:
# delta_k_int = rho^2*(pi/12)*eps*f * (pi*T^4/192) * I_3 = rho^2*eps*f*pi^2*T^4/(12*192) * I_3
# = rho^2*eps*f*pi^2*T^4/2304 * I_3

# k_flat = rho * pi*T^2/8 * I_1

# beta_int = delta_k_int / (k_flat * eps * f)
# = [rho^2*pi^2*T^4/2304 * I_3] / [rho*pi*T^2/8 * I_1]
# = rho*pi*T^2/(2304/8) * I_3/I_1
# = rho*pi*T^2/288 * I_3/I_1

# Compute I_1 and I_3:
I1 = sp.integrate(tau_s * sp.exp(-rho_s * sp.pi/24 * tau_s**4), (tau_s, 0, sp.oo))
print(f"I_1 = {I1}")
print(f"I_3 = {I3}")
I3_over_I1 = sp.simplify(I3/I1)
print(f"I_3/I_1 = {I3_over_I1}")
print()

# beta_int = rho * pi * T^2 / 288 * I_3/I_1
# At T=1:
# I_3/I_1: from SymPy let's evaluate
I1_val = float(I1.subs(rho_s, 2000))
I3_val = float(I3.subs(rho_s, 2000))
ratio_I = I3_val / I1_val
print(f"At rho=2000: I_1={I1_val:.6e}, I_3={I3_val:.6e}, I_3/I_1={ratio_I:.6f}")

beta_int_pred = 2000 * np.pi * 1.0**2 / 288 * ratio_I
print(f"Predicted beta_int = rho*pi*T^2/288 * I_3/I_1 = {beta_int_pred:.4f}")
print(f"Measured beta_int (N=2000, diamond) = +1.57")
print(f"Ratio: {1.57/beta_int_pred:.4f}")
print()

# ====================================================================
# BOUNDARY contribution (same approach)
# ====================================================================
# delta_k_bdy = -rho * (lost volume near cone at high rapidity)
# The lost region: at each tau, directions with eta > eta_crit(tau) where
# eta_crit is defined by: eps*f*(cosh+sinh*cos)^2 > 1
# But this doesn't depend on tau! It's a direction cutoff.
#
# Wait, the condition for losing causality at direction (eta,theta):
# tau^2 < eps*f*(dt+dz)^2/2 = eps*f*tau^2*(cosh+sinh*cos)^2
# => 1 < eps*f*(cosh+sinh*cos)^2
# This is INDEPENDENT of tau!
#
# So for ANY tau, if the direction has (cosh+sinh*cos)^2 > 1/(eps*f),
# the pair is non-causal.
#
# The lost links at those directions:
# delta_k_bdy = -rho * integral_{bad directions} integral_0^{infty} tau^3 P(tau) dtau * dOmega
# = -rho * I_3 * integral_{eta > eta_crit} dOmega

# But I_3 appears in BOTH k_flat and delta_k_bdy!
# k_flat = rho * integral_{all directions} I_3 * dOmega = rho * I_3 * Omega_total

# Hmm wait, k_flat also has eta cutoff from diamond geometry.
# But the flat k uses eta_max(tau) which depends on tau.
# The boundary loss uses a FIXED eta_crit = (1/2)*log(2/(eps*f)).

# For eps*f << 1 (perturbative regime), eta_crit >> 1.
# The lost angular region is the HIGH-rapidity tail eta > eta_crit.
# The integral: integral_{eta>eta_crit} sinh^2(eta) * 4*pi deta
# ≈ 4*pi * integral_{eta_crit}^{infty} e^{2*eta}/4 deta
# = pi * e^{2*eta_crit}/2
# = pi/(2*eps*f)  [since e^{2*eta_crit} = 1/(eps*f)]

# Wait, eta_crit defined by: (cosh+sinh*cos)^2 = 1/(eps*f)
# At large eta and cos(theta)=1 (optimal direction):
# (e^eta)^2 = 1/(eps*f) => e^{eta_crit} = 1/sqrt(eps*f)
# => e^{2*eta_crit} = 1/(eps*f)

# So the lost solid angle ≈ pi/(2*eps*f)

# But we also need the tau integral for these directions.
# For the lost directions, at EACH tau from 0 to tau_max,
# the pair loses causality. But tau_max depends on the diamond geometry.
# For eta > eta_crit and proper time tau:
# tau*e^eta < T/2 => tau < T/(2*e^{eta})
# At eta = eta_crit: tau < T*sqrt(eps*f)/2

# So the tau integral is: integral_0^{T*sqrt(eps*f)/2} tau^3 P(tau) dtau
# For small eps: the upper limit is small, and P(tau) ≈ 1.
# So: integral ≈ (T*sqrt(eps*f)/2)^4/4 = T^4*(eps*f)^2/64

# delta_k_bdy = -rho * T^4*(eps*f)^2/64 * pi/(2*eps*f)
# = -rho * pi * T^4 * eps * f / 128

# beta_bdy = delta_k_bdy / (k_flat * eps * f) = -rho*pi*T^4/(128*k_flat)

# k_flat ≈ rho * I_3 * (4*pi * integral_0^{eta_max} sinh^2 deta)
# This still has the eta integral which depends on tau...

# Let me try a cleaner approach: compute everything numerically at rho=2000.

print("=" * 70)
print("NUMERICAL EVALUATION at N=2000, T=1")
print("=" * 70)

rho = 2000.0

def P_link(tau):
    """Link probability."""
    return np.exp(-rho * np.pi / 24 * tau**4)

def eta_max(tau):
    """Maximum rapidity at given proper time (diamond constraint)."""
    if tau <= 0:
        return 20.0  # effectively infinite
    val = T / (2 * tau)
    if val <= 1:
        return 0.0
    return np.log(val)

# k_flat = rho * integral_0^{tau_max} dtau * tau^3 * P(tau) * Omega(tau)
# where Omega(tau) = 4*pi * integral_0^{eta_max(tau)} sinh^2(eta) deta

def angular_integral_0(eta_m):
    """J_0 = 4*pi * integral_0^eta_m sinh^2(eta) deta."""
    if eta_m <= 0:
        return 0.0
    return 4 * np.pi * (np.sinh(2*eta_m)/4 - eta_m/2)

def angular_integral_2(eta_m):
    """J_2 = 2*pi * integral_0^eta_m sinh^2*(2*cosh^2 + 2*sinh^2/3) deta."""
    if eta_m <= 0:
        return 0.0
    # Numerical integration
    def integrand(eta):
        s = np.sinh(eta)
        c = np.cosh(eta)
        return s**2 * (2*c**2 + 2*s**2/3)
    result, _ = sci_int.quad(integrand, 0, eta_m)
    return 2 * np.pi * result

# k_flat
def k_integrand(tau):
    em = eta_max(tau)
    return tau**3 * P_link(tau) * angular_integral_0(em)

k_flat_half, _ = sci_int.quad(k_integrand, 1e-6, T/2, limit=200)
k_flat_val = rho * k_flat_half
print(f"k_flat (one side, analytical) = {k_flat_val:.2f}")

# delta_k_int
def dk_int_integrand(tau):
    em = eta_max(tau)
    return tau**7 * P_link(tau) * angular_integral_2(em)

dk_int_half, _ = sci_int.quad(dk_int_integrand, 1e-6, T/2, limit=200)
dk_int_val = rho**2 * (np.pi/12) * dk_int_half  # * eps * f
print(f"delta_k_int / (eps*f) (one side) = {dk_int_val:.4f}")

beta_int_analytical = dk_int_val / k_flat_val
print(f"beta_int = dk_int/(k_flat*eps*f) = {beta_int_analytical:.4f}")
print(f"Measured beta_int (N=2000) = +1.57")
print(f"Ratio analytical/measured = {beta_int_analytical/1.57:.4f}")
print()

# delta_k_bdy: lost pairs at high rapidity
# For each (tau, eta, theta), the pair is lost if eps*f*(cosh+sinh*cos)^2 > 1
# Since this doesn't depend on tau, we integrate tau first.
# But it depends on the PRODUCT eps*f, not eps and f separately.
# For the REGRESSION on f, we need the derivative d(delta_k)/d(eps*f).
# So we compute delta_k_bdy as a function of eps*f and differentiate.

# Actually, the condition is: (cosh+sinh*cos)^2 > 1/(eps*f)
# For small eps*f, eta_crit is large, and the lost region is small.
# delta_k_bdy = -rho * integral_{lost} tau^3 * P(tau) * dOmega * dtau

# The lost region is: eta such that max_theta (cosh+sinh*cos)^2 > 1/(eps*f)
# max over theta: cos(theta)=1 gives (cosh+sinh)^2 = e^{2*eta}
# So: e^{2*eta} > 1/(eps*f) => eta > eta_c = -log(eps*f)/2

# At each eta > eta_c, the angular range in theta:
# (cosh+sinh*cos)^2 > 1/(eps*f)
# cosh+sinh*cos > 1/sqrt(eps*f)
# cos(theta) > (1/sqrt(eps*f) - cosh(eta))/sinh(eta)
# = (e^{-eta_c} - cosh(eta))/sinh(eta)

# For eta >> eta_c: the cos condition is cos > ~ -1 (all theta OK)
# For eta = eta_c: cos > (e^{-eta_c} - cosh(eta_c))/sinh(eta_c) ≈ 1 (only cos=1)

# This is getting messy. Let me just compute it numerically.

def compute_beta_bdy_numerical(eps_f_val):
    """Compute delta_k_bdy / (rho) for given eps*f value."""
    # Lost pairs: for each (tau, eta, theta), check condition
    # (cosh+sinh*cos)^2 > 1/(eps_f_val)

    # Integrate over tau, eta, theta
    def lost_integrand(tau, eta, theta):
        if tau <= 0 or eta <= 0:
            return 0.0
        em = eta_max(tau)
        if eta > em:
            return 0.0
        c = np.cosh(eta)
        s_h = np.sinh(eta)
        g = c + s_h * np.cos(theta)
        if g**2 <= 1.0 / eps_f_val:
            return 0.0  # not lost
        return tau**3 * P_link(tau) * s_h**2 * np.sin(theta) * 2 * np.pi

    # Triple integral: (tau, eta, theta)
    from scipy.integrate import tplquad
    result, err = tplquad(
        lost_integrand,
        0, np.pi,              # theta: 0 to pi
        lambda th: 0, lambda th: 10,  # eta: 0 to 10
        lambda th, eta: 1e-6, lambda th, eta: T/2,  # tau
        epsabs=1e-8, epsrel=1e-6
    )
    return rho * result

# Compute for a range of eps*f to get the slope
print("Computing delta_k_bdy as function of eps*f...")
eps_f_values = [0.001, 0.005, 0.01, 0.02, 0.05]
dk_bdy_values = []
for ef in eps_f_values:
    dk = compute_beta_bdy_numerical(ef)
    dk_bdy_values.append(dk)
    print(f"  eps*f={ef:.4f}: delta_k_bdy/rho = {dk/rho:.6f}, delta_k_bdy = {dk:.4f}")

# beta_bdy = d(delta_k_bdy)/d(eps*f) / k_flat
# Numerical derivative:
ef = np.array(eps_f_values)
dk = np.array(dk_bdy_values)
# Fit dk = a * ef + b * ef^2 (should be linear for small ef)
from numpy.polynomial import polynomial as P
coeffs = np.polyfit(ef, dk, 2)
slope = coeffs[1]  # linear coefficient
print(f"\nLinear fit: delta_k_bdy ≈ {slope:.2f} * eps*f")
print(f"beta_bdy = slope/k_flat = {slope/k_flat_val:.4f}")
print(f"Measured beta_bdy (N=2000) = -2.66")
print()

beta_total = beta_int_analytical + slope/k_flat_val
print(f"TOTAL: beta = beta_int + beta_bdy = {beta_int_analytical:.4f} + {slope/k_flat_val:.4f} = {beta_total:.4f}")
print(f"Measured beta_total (N=2000) = -1.09")

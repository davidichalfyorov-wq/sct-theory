"""
Analytical derivation of beta = -0.30 for pp-wave causal sets.
================================================================

Strategy: The O(eps) perturbation of the degree k(x) comes from:
1. Changed link probability (interior of cone) - COMPUTABLE
2. Changed causal structure (boundary of cone) - VANISHES at O(eps) because
   delta(tau^2)|_pos propto tau^2 which is zero on the light cone.

So beta comes ENTIRELY from the interior link probability change.

The key quantity is <du_hat^2>_links = average of du_hat^2 over observed links,
where du_hat = (n_0 + n_3)/sqrt(2) is the u-component of the unit hyperboloid vector.

Author: David Alfyorov
"""
import numpy as np
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# GPU
_c = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
if os.path.isdir(_c):
    os.add_dll_directory(_c)
import cupy as cp

from scipy import stats

N = 2000
T = 1.0
M = 20

print("=" * 70)
print("ANALYTICAL beta FROM <du_hat^2>_links")
print("=" * 70)

# Step 1: Compute <du_hat^2> over observed links
du_hat2_means = []
beta_measured = []

for m in range(M):
    np.random.seed(1000 + m)
    pts = np.random.uniform(0, T, (N, 4)).astype(np.float32)
    pts_g = cp.asarray(pts)
    f_vals = (pts[:, 1]**2 - pts[:, 2]**2) / 2.0

    # Flat causal set
    t_g = pts_g[:, 0]; x_g = pts_g[:, 1]; y_g = pts_g[:, 2]; z_g = pts_g[:, 3]
    dt = t_g[None, :] - t_g[:, None]
    dx = x_g[None, :] - x_g[:, None]
    dy = y_g[None, :] - y_g[:, None]
    dz = z_g[None, :] - z_g[:, None]
    tau2 = dt**2 - dx**2 - dy**2 - dz**2
    C_flat = ((dt > 0) & (tau2 > 0)).astype(cp.float32)
    C2 = C_flat @ C_flat
    L_flat = ((C_flat > 0.5) & (C2 < 0.5)).astype(cp.float32)

    k_flat = cp.asnumpy((L_flat.sum(0) + L_flat.sum(1)).astype(cp.float32))

    # Extract link pairs and compute du_hat^2 for each link
    link_i, link_j = cp.where(L_flat > 0.5)
    link_i = cp.asnumpy(link_i)
    link_j = cp.asnumpy(link_j)

    dt_links = pts[link_j, 0] - pts[link_i, 0]
    dx_links = pts[link_j, 1] - pts[link_i, 1]
    dy_links = pts[link_j, 2] - pts[link_i, 2]
    dz_links = pts[link_j, 3] - pts[link_i, 3]
    tau2_links = dt_links**2 - dx_links**2 - dy_links**2 - dz_links**2
    tau_links = np.sqrt(np.maximum(tau2_links, 1e-12))

    du_links = (dt_links + dz_links) / np.sqrt(2)
    du_hat2 = (du_links / tau_links)**2  # du_hat on unit hyperboloid

    # Average over all links
    du_hat2_mean = du_hat2.mean()
    du_hat2_means.append(du_hat2_mean)

    # Also compute weighted averages relevant to beta
    # For each element j: <du_hat^2>_j = mean of du_hat^2 over its links
    # Then the formula is: beta ~ some_function(<du_hat^2>)

    # Compute V(tau) for each link
    V_links = np.pi / 24 * tau_links**4

    # Measure beta directly for comparison
    eps_val = 0.1
    du_g = (dt + dz) / np.sqrt(2)
    x_i = x_g[:, None]
    y_i = y_g[:, None]
    H_int = x_i**2 + x_i * dx + dx**2 / 3 - y_i**2 - y_i * dy - dy**2 / 3
    delta_sigma = eps_val / 2 * du_g**2 * H_int
    tau2_ppw = dt**2 - dx**2 - dy**2 - dz**2 - 2 * delta_sigma
    C_ppw = ((dt > 0) & (tau2_ppw > 0)).astype(cp.float32)
    C2_ppw = C_ppw @ C_ppw
    L_ppw = ((C_ppw > 0.5) & (C2_ppw < 0.5)).astype(cp.float32)
    k_ppw = cp.asnumpy((L_ppw.sum(0) + L_ppw.sum(1)).astype(cp.float32))

    dk = k_ppw - k_flat
    mask = k_flat > 10
    slope, _, _, _, _ = stats.linregress(f_vals[mask], (dk / np.maximum(k_flat, 1))[mask])
    beta_m = slope / eps_val
    beta_measured.append(beta_m)

    if (m + 1) % 5 == 0:
        print(f"  Sprinkling {m+1}/{M}: <du_hat^2>={du_hat2_mean:.4f}, beta={beta_m:.4f}")

du_hat2_arr = np.array(du_hat2_means)
beta_arr = np.array(beta_measured)

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)
print(f"<du_hat^2>_links = {du_hat2_arr.mean():.4f} +/- {du_hat2_arr.std()/np.sqrt(M):.4f}")
print(f"beta (measured)  = {beta_arr.mean():.4f} +/- {beta_arr.std()/np.sqrt(M):.4f}")
print()

# Theory prediction: beta = -C * <du_hat^2> where C comes from the formula
# From the derivation: delta_k ∝ rho * V * du_hat^2 * exp(-rho*V)
# The degree-weighted average of rho*V*du_hat^2 gives the coefficient.

# Actually let me compute the FULL coefficient from first principles.
# For each link (i,j), the perturbation of the link probability is:
# delta_P / P = -rho * delta_V = -rho * (pi/24) * delta(tau^4)
# delta(tau^4) = -4*eps*f(x_j)*du_hat^2*tau^4 + (position-independent terms)
# So delta_P / P = rho*(pi/6)*eps*f(x_j)*du_hat^2*tau^4

# The degree change at element j:
# delta_k(j) = sum over links touching j of delta_P
# = sum over links (i->j or j->k) of P * rho*(pi/6)*eps*f(j)*du_hat^2*tau^4

# Wait, P is already the link probability = exp(-rho*V). For an EXISTING link,
# P is not the probability; the link either exists or not in a given realization.
# The EXPECTED delta_k is:
# E[delta_k(j)] = rho * integral d^4y * exp(-rho*V) * (-rho*delta_V)
# For the position-dependent part at element j:
# = rho^2*(pi/6)*eps*f(j) * integral d^4y * exp(-rho*V) * du_hat^2 * tau^4

# This integral (tau^4 * tau^3 * exp(-rho*V) * du_hat^2):
# In hyperboloid coords: tau^{4+3} * du_hat^2 * exp(-rho*pi/24*tau^4) dtau dOmega
# The tau part: integral tau^7 * exp(-rho*pi/24*tau^4) dtau = I_7 = 144/(pi^2*rho^2)

# For each existing link at separation tau_link:
# The contribution is proportional to rho*V(tau_link)*du_hat^2(direction)
# where V = (pi/24)*tau^4

# So: delta_k(j) / eps / f(j) ~ rho * sum_links [rho*V(tau)*du_hat^2]

# The ratio: beta = delta_k / (k * eps * f) = <rho*V*du_hat^2>_links

# Let me compute <rho*V*du_hat^2> over observed links
print("Computing <rho*V*du_hat^2>_links...")
rhoV_du2_means = []

for m in range(M):
    np.random.seed(1000 + m)
    pts = np.random.uniform(0, T, (N, 4)).astype(np.float32)
    pts_g = cp.asarray(pts)

    t_g = pts_g[:, 0]; x_g = pts_g[:, 1]; y_g = pts_g[:, 2]; z_g = pts_g[:, 3]
    dt = t_g[None, :] - t_g[:, None]
    dx = x_g[None, :] - x_g[:, None]
    dy = y_g[None, :] - y_g[:, None]
    dz = z_g[None, :] - z_g[:, None]
    tau2 = dt**2 - dx**2 - dy**2 - dz**2
    C_flat = ((dt > 0) & (tau2 > 0)).astype(cp.float32)
    C2 = C_flat @ C_flat
    L_flat = ((C_flat > 0.5) & (C2 < 0.5)).astype(cp.float32)

    link_i, link_j = cp.where(L_flat > 0.5)
    link_i = cp.asnumpy(link_i)
    link_j = cp.asnumpy(link_j)

    dt_links = pts[link_j, 0] - pts[link_i, 0]
    dx_links = pts[link_j, 1] - pts[link_i, 1]
    dy_links = pts[link_j, 2] - pts[link_i, 2]
    dz_links = pts[link_j, 3] - pts[link_i, 3]
    tau2_links = dt_links**2 - dx_links**2 - dy_links**2 - dz_links**2
    tau_links = np.sqrt(np.maximum(tau2_links, 1e-12))

    du_links = (dt_links + dz_links) / np.sqrt(2)
    du_hat2 = (du_links / tau_links)**2

    rho = N  # density in unit diamond
    V_links = np.pi / 24 * tau_links**4
    rhoV_du2 = rho * V_links * du_hat2

    rhoV_du2_means.append(rhoV_du2.mean())

rhoV_du2_arr = np.array(rhoV_du2_means)
print(f"<rho*V*du_hat^2>_links = {rhoV_du2_arr.mean():.4f} +/- {rhoV_du2_arr.std()/np.sqrt(M):.4f}")
print(f"Predicted beta (= -<rhoV*du_hat^2>) = {-rhoV_du2_arr.mean():.4f}")
print(f"Measured beta = {beta_arr.mean():.4f}")
print(f"Ratio = {beta_arr.mean() / (-rhoV_du2_arr.mean()):.4f}")
print()

# Also try: beta_pred = -(4*pi/6) * <V*du_hat^2>_links * rho
# or some other coefficient

# Let me try the SIMPLER model:
# For links at proper time tau:
# The degree-weighted perturbation per link is:
# delta_P/P = 4*rho*V(tau)*eps*f(x)*du_hat^2
# (from delta(tau^4)/tau^4 = -4*eps*f(x)*du_hat^2
#  and delta_P/P = -rho*(pi/24)*delta(tau^4) = rho*(pi/6)*eps*f(x)*du_hat^2*tau^4
#  = 4*rho*V(tau)*eps*f(x)*du_hat^2 since V = (pi/24)*tau^4 and 4*(pi/6)/(pi/24)=4*4=16? no)

# Actually: delta_P/P = -rho*delta_V = -rho*(pi/24)*delta(tau^4)
# delta(tau^4) = 2*tau^2*delta(tau^2)
# delta(tau^2)|_pos = -2*eps*f(x)*du_hat^2*tau^2
# delta(tau^4) = -4*eps*f(x)*du_hat^2*tau^4
# delta_P/P = +rho*(pi/24)*4*eps*f(x)*du_hat^2*tau^4
# = (4*pi/24)*rho*eps*f(x)*du_hat^2*tau^4
# = (pi/6)*rho*eps*f(x)*du_hat^2*tau^4
# = 4*rho*V(tau)*eps*f(x)*du_hat^2  (since V=(pi/24)*tau^4, so (pi/6)*tau^4 = 4*V)

# YES! delta_P/P = 4*rho*V*eps*f(x)*du_hat^2

# But this is POSITIVE when f(x) > 0 (HIGHER link probability).
# Yet beta < 0 (FEWER links when f > 0).

# The sign issue: delta(tau^2) < 0 when f > 0, so tau DECREASES.
# A SMALLER tau means a SMALLER V, which means HIGHER P(link).
# So existing links become MORE probable (delta_P > 0).
# BUT: the CAUSAL VOLUME also decreases, meaning FEWER potential link partners.

# AH! I forgot about the change in the INTEGRATION DOMAIN.
# The integral is: k = rho * integral_{causal future} exp(-rho*V) d^4y
# When the proper time decreases, the CAUSAL DIAMOND shrinks.
# There are fewer pairs in the causal future.
# The number of potential link partners DECREASES.

# But earlier I argued the boundary contribution vanishes...
# Let me reconsider. The boundary contribution involves pairs at the
# light cone (tau = 0). For these:
# tau_curved^2 = tau_flat^2*(1 - 2*eps*f(x)*du_hat^2) + higher order
# At tau_flat = 0: tau_curved^2 = 0, so there's no change at the exact boundary.

# But at tau_flat = eps_tau (small positive), some of these become non-causal
# if 2*eps*f(x)*du_hat^2 > 1... No, that's O(1) not O(eps).

# Actually: tau_curved^2 = tau_flat^2 * (1 - 2*eps*f*du_hat^2)
# This is NEGATIVE when tau_flat^2 < 2*eps*f*du_hat^2*tau_flat^2 ...
# that's never negative since the correction is < tau^2 for small eps.
# Wait: tau_curved^2 = tau_flat^2 - 2*eps*f*du_hat^2*tau_flat^2
# = tau_flat^2*(1 - 2*eps*f*du_hat^2)
# This is positive as long as 2*eps*f*du_hat^2 < 1, which is true for small eps.

# So NO pair becomes non-causal at small eps. The boundary contribution is truly zero.
# Then why is beta NEGATIVE?

# AH! I think the issue is my formula for delta(tau^2).
# Let me recheck the Synge world function.
# From ppwave_link_density.py, line 142:
# sigma(0,Q) = sigma_flat + eps*u_q^2*(x_q^2-y_q^2)/6
# tau^2 = -2*sigma = -2*sigma_flat - eps*u_q^2*(x_q^2-y_q^2)/3
# = tau_flat^2 - eps*u^2*(x^2-y^2)/3

# This is for a pair from the ORIGIN to point Q.
# For a pair from p = (u0,v0,x0,y0) to q = (u0+du,v0+dv,x0+dx,y0+dy):
# The Synge correction uses the metric perturbation along the geodesic.
# H(gamma(s)) = eps*((x0+s*dx)^2 - (y0+s*dy)^2)

# delta_sigma = (1/2)*du^2*integral_0^1 H(gamma(s)) ds
# = (eps*du^2/2)*integral_0^1 [(x0+s*dx)^2 - (y0+s*dy)^2] ds
# = (eps*du^2/2)*[x0^2-y0^2 + (x0*dx-y0*dy) + (dx^2-dy^2)/3]
# = eps*du^2*[f(x0) + (x0*dx-y0*dy)/2 + (dx^2-dy^2)/6]

# So delta(tau^2) = -2*delta_sigma = -eps*du^2*[2*f(x0) + (x0*dx-y0*dy) + (dx^2-dy^2)/3]

# Sign: when f(x0)>0, the DOMINANT contribution is -eps*du^2*2*f(x0) < 0.
# So tau^2 DECREASES (cone shrinks).
# But exp(-rho*V(tau)) INCREASES (higher link probability for smaller V).

# The TOTAL expected delta_k at element x:
# delta_k(x) = rho * integral d^4y [exp(-rho*V_curved)*I_curved - exp(-rho*V_flat)*I_flat]
# = rho * integral d^4y * exp(-rho*V_flat) * [(-rho*delta_V)*I_flat + exp(-rho*V_flat)*delta_I]
# where I is the indicator (causal).

# Since delta_I = 0 (as argued above), ONLY the delta_V term survives.
# And delta_P/P = +4*rho*V*eps*f(x)*du_hat^2 > 0 when f > 0.
# This gives delta_k > 0, i.e., beta > 0.

# BUT THE MEASUREMENT GIVES beta < 0!

# THEREFORE: either my Synge world function is wrong, or there's a sign error,
# or the boundary contribution is NOT zero.

# Let me check numerically which pairs change their causal status.

print("CHECKING SIGN: counting pairs that change causal status...")
np.random.seed(42)
pts = np.random.uniform(0, T, (N, 4)).astype(np.float32)
pts_g = cp.asarray(pts)

t_g = pts_g[:, 0]; x_g = pts_g[:, 1]; y_g = pts_g[:, 2]; z_g = pts_g[:, 3]
dt = t_g[None, :] - t_g[:, None]
dx = x_g[None, :] - x_g[:, None]
dy = y_g[None, :] - y_g[:, None]
dz = z_g[None, :] - z_g[:, None]
tau2_flat = dt**2 - dx**2 - dy**2 - dz**2

# PP-wave tau^2
eps_val = 0.1
du = (dt + dz) / np.sqrt(2)
x_i = x_g[:, None]; y_i = y_g[:, None]
H_int = x_i**2 + x_i * dx + dx**2 / 3 - y_i**2 - y_i * dy - dy**2 / 3
delta_sigma = eps_val / 2 * du**2 * H_int
tau2_ppw = tau2_flat - 2 * delta_sigma

C_flat = ((dt > 0) & (tau2_flat > 0))
C_ppw = ((dt > 0) & (tau2_ppw > 0))

gained = ((~C_flat) & C_ppw)  # became causal in pp-wave
lost = (C_flat & (~C_ppw))     # lost causality in pp-wave

n_gained = int(gained.sum())
n_lost = int(lost.sum())
n_causal_flat = int(C_flat.sum())

print(f"  Pairs gained causality: {n_gained}")
print(f"  Pairs lost causality: {n_lost}")
print(f"  Net change: {n_gained - n_lost}")
print(f"  Causal flat: {n_causal_flat}")
print(f"  Fractional: gained {n_gained/n_causal_flat:.4%}, lost {n_lost/n_causal_flat:.4%}")
print()

# Check: do gained/lost correlate with f(x)?
f_g = (x_g**2 - y_g**2) / 2  # f at element position
f_gained_mean = float((f_g[:, None].expand(N, N)[gained]).mean())
f_lost_mean = float((f_g[:, None].expand(N, N)[lost]).mean())
print(f"  <f(x)> for gained pairs: {f_gained_mean:.4f}")
print(f"  <f(x)> for lost pairs: {f_lost_mean:.4f}")
print()

# Also check: does delta_sigma have the right sign?
# For a link from i to j (i < j, dt > 0):
# If f(x_i) > 0: delta_sigma > 0 => delta(tau^2) = -2*delta_sigma < 0 => tau DECREASES
# This should LOSE some pairs (near the cone boundary), giving FEWER links.

# Let me check the delta_sigma directly on flat links
L_flat_bool = ((C_flat) & (cp.asarray(np.ones((N, N), dtype=np.float32)) > 0.5))
C2_f = C_flat.astype(cp.float32) @ C_flat.astype(cp.float32)
L_flat_bool = (C_flat & (C2_f < 0.5))
link_mask = L_flat_bool

dsig_on_links = cp.asnumpy(delta_sigma[link_mask].astype(cp.float32))
f_on_links_i = cp.asnumpy(f_g[:, None].expand(N, N)[link_mask].astype(cp.float32))

corr_dsig_f = np.corrcoef(dsig_on_links, f_on_links_i)[0, 1]
print(f"  corr(delta_sigma, f(x_i)) on links: {corr_dsig_f:.4f}")
print(f"  mean delta_sigma on links: {dsig_on_links.mean():.6f}")
print(f"  mean delta_sigma when f>0: {dsig_on_links[f_on_links_i>0].mean():.6f}")
print(f"  mean delta_sigma when f<0: {dsig_on_links[f_on_links_i<0].mean():.6f}")
print()

# The sign: delta_sigma > 0 when f > 0 means tau^2 DECREASES.
# Some pairs near the cone boundary become NON-causal.
# These pairs were links (near boundary => small V => high P(link)).
# Losing them DECREASES k. This is consistent with beta < 0.
# But I argued above that the boundary contribution is zero...

# The resolution: delta(tau^2) = tau_flat^2 * (something) is NOT how it works.
# Actually: delta(tau^2) = -2*delta_sigma, which does NOT vanish at tau=0!
# Because delta_sigma depends on the COORDINATES (du, dx, dy), not on tau.
# A pair can have tau_flat^2 = 0 (on the cone) but delta_sigma != 0.

# CHECK: on the flat light cone, tau^2 = dt^2 - dr^2 = 0 but du and H_int
# are NOT zero. So delta_sigma = eps/2 * du^2 * H_int can be nonzero.
# Therefore delta(tau^2) = -2*delta_sigma != 0 on the cone!

# This means my earlier argument was WRONG.
# The boundary contribution is NOT zero.

print("KEY CORRECTION: delta(tau^2) does NOT vanish on the light cone!")
print("  On the cone: tau_flat^2 = 0 but delta(tau^2) = -2*delta_sigma != 0")
print("  delta_sigma = (eps/2)*du^2*H_int depends on coordinates, not on tau")
print()
print("  => Boundary contribution is NONZERO and DOMINANT (sign gives beta < 0)")
print("  => Interior contribution (link probability) gives beta > 0 (WRONG SIGN)")
print("  => The boundary effect WINS, giving the net beta < 0")

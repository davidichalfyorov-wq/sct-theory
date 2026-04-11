
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq
import math

ETA = np.diag([-1.0, 1.0, 1.0, 1.0])

def _wrap_angle(a: np.ndarray | float):
    return (a + np.pi) % (2*np.pi) - np.pi

def f_schwarzschild(r: np.ndarray | float, M: float):
    return 1.0 - 2.0*M/np.asarray(r)

def fp_schwarzschild(r: np.ndarray | float, M: float):
    return 2.0*M/np.asarray(r)**2

def f_kottler(r: np.ndarray | float, M: float, Lambda: float):
    r = np.asarray(r)
    return 1.0 - 2.0*M/r - (Lambda*r*r)/3.0

def fp_kottler(r: np.ndarray | float, M: float, Lambda: float):
    r = np.asarray(r)
    return 2.0*M/r**2 - 2.0*Lambda*r/3.0

def schwarzschild_geodesic_rhs(lam, y, M):
    """
    Geodesic equations in Schwarzschild coordinates:
    y = [t, r, th, ph, ut, ur, uth, uph]
    """
    t, r, th, ph, ut, ur, uth, uph = y
    f = 1.0 - 2.0*M/r
    fp = 2.0*M/r**2
    st = np.sin(th)
    ct = np.cos(th)
    st = st if abs(st) > 1e-15 else np.sign(st) * 1e-15 if st != 0 else 1e-15

    out = np.empty_like(y)
    out[0] = ut
    out[1] = ur
    out[2] = uth
    out[3] = uph

    out[4] = -(fp/f) * ut * ur
    out[5] = -0.5*f*fp*ut*ut + 0.5*(fp/f)*ur*ur + r*f*(uth*uth + (st*st)*(uph*uph))
    out[6] = -2.0*ur*uth/r + st*ct*uph*uph
    out[7] = -2.0*ur*uph/r - 2.0*(ct/st)*uth*uph
    return out

def kottler_geodesic_rhs(lam, y, M, Lambda):
    """
    Geodesic equations in Kottler coordinates:
    y = [t, r, th, ph, ut, ur, uth, uph]
    """
    t, r, th, ph, ut, ur, uth, uph = y
    f = 1.0 - 2.0*M/r - (Lambda*r*r)/3.0
    fp = 2.0*M/r**2 - 2.0*Lambda*r/3.0
    st = np.sin(th)
    ct = np.cos(th)
    st = st if abs(st) > 1e-15 else np.sign(st) * 1e-15 if st != 0 else 1e-15

    out = np.empty_like(y)
    out[0] = ut
    out[1] = ur
    out[2] = uth
    out[3] = uph

    out[4] = -(fp/f) * ut * ur
    out[5] = -0.5*f*fp*ut*ut + 0.5*(fp/f)*ur*ur + r*f*(uth*uth + (st*st)*(uph*uph))
    out[6] = -2.0*ur*uth/r + st*ct*uph*uph
    out[7] = -2.0*ur*uph/r - 2.0*(ct/st)*uth*uph
    return out

def map_rnc_to_schwarzschild_expmap(pts, M, r0, t0=0.0, theta0=np.pi/2, phi0=0.0,
                                    rtol=1e-10, atol=1e-12):
    """
    Numerical exponential map from local orthonormal coordinates (tau,x,y,z)
    to exact Schwarzschild coordinates (t,r,theta,phi).

    Conventions:
      - local z is radial
      - local y is polar-theta direction
      - local x is azimuthal-phi direction
      - base point is (t0, r0, theta0=pi/2, phi0=0)

    WARNING:
      This is the exact exponential map for the chosen local orthonormal frame,
      not a closed-form RNC metric. Valid only inside the convex normal neighborhood.
    """
    pts = np.asarray(pts, dtype=float)
    f0 = 1.0 - 2.0*M/r0
    if f0 <= 0:
        raise ValueError("Base point is at or inside the Schwarzschild horizon.")

    mapped = np.empty((len(pts), 4), dtype=float)
    mapped[0] = np.array([t0, r0, theta0, phi0], dtype=float) if np.allclose(pts[0], 0) else mapped[0]

    def event_horizon(lam, y):
        return y[1] - (2.0*M + 1e-9)
    event_horizon.terminal = True
    event_horizon.direction = -1

    for idx, xi in enumerate(pts):
        tau, x, y, z = xi
        y0 = np.array([
            t0,
            r0,
            theta0,
            phi0,
            tau / math.sqrt(f0),
            math.sqrt(f0) * z,
            y / r0,
            x / (r0 * np.sin(theta0)),
        ], dtype=float)

        sol = solve_ivp(
            lambda lam, Y: schwarzschild_geodesic_rhs(lam, Y, M),
            (0.0, 1.0),
            y0,
            rtol=rtol,
            atol=atol,
            events=event_horizon,
            dense_output=False,
        )
        if (sol.status == 1) or (not sol.success):
            mapped[idx] = np.array([np.nan, np.nan, np.nan, np.nan])
        else:
            mapped[idx] = sol.y[:4, -1]
    return mapped

def map_rnc_to_kottler_expmap(pts, M, r0, Lambda, t0=0.0, theta0=np.pi/2, phi0=0.0,
                              rtol=1e-10, atol=1e-12):
    """
    Numerical exponential map from local orthonormal coordinates (tau,x,y,z)
    to exact Kottler coordinates (t,r,theta,phi).
    """
    pts = np.asarray(pts, dtype=float)
    f0 = 1.0 - 2.0*M/r0 - (Lambda*r0*r0)/3.0
    if f0 <= 0:
        raise ValueError("Base point is at or inside the Kottler horizon.")

    mapped = np.empty((len(pts), 4), dtype=float)

    # crude lower bound: largest positive root of f(r)=0 not solved analytically here
    # we just stop if f becomes nonpositive
    def event_bad_region(lam, y):
        r = y[1]
        return 1.0 - 2.0*M/r - (Lambda*r*r)/3.0 - 1e-9
    event_bad_region.terminal = True
    event_bad_region.direction = -1

    for idx, xi in enumerate(pts):
        tau, x, y, z = xi
        y0 = np.array([
            t0,
            r0,
            theta0,
            phi0,
            tau / math.sqrt(f0),
            math.sqrt(f0) * z,
            y / r0,
            x / (r0 * np.sin(theta0)),
        ], dtype=float)

        sol = solve_ivp(
            lambda lam, Y: kottler_geodesic_rhs(lam, Y, M, Lambda),
            (0.0, 1.0),
            y0,
            rtol=rtol,
            atol=atol,
            events=event_bad_region,
            dense_output=False,
        )
        if (sol.status == 1) or (not sol.success):
            mapped[idx] = np.array([np.nan, np.nan, np.nan, np.nan])
        else:
            mapped[idx] = sol.y[:4, -1]
    return mapped

def schwarzschild_exact_midpoint_preds_from_mapped(mapped_pts, i, M, tol=1e-12):
    """
    Exact-metric midpoint test in Schwarzschild coordinates.

    This is NOT the exact causal predicate in local RNC.
    It is the exact Schwarzschild metric evaluated at the coordinate midpoint
    of the exact exponential-map images of the local points.

    Use for calibration, not as a theorem-grade local predicate.
    """
    Xi = mapped_pts[i]
    Xj = mapped_pts[:i]
    good = np.isfinite(Xj).all(axis=1)
    if not np.isfinite(Xi).all():
        return np.zeros(i, dtype=bool)

    dt = Xi[0] - Xj[:, 0]
    dr = Xi[1] - Xj[:, 1]
    dth = Xi[2] - Xj[:, 2]
    dph = _wrap_angle(Xi[3] - Xj[:, 3])

    rm = 0.5 * (Xi[1] + Xj[:, 1])
    thm = 0.5 * (Xi[2] + Xj[:, 2])
    f = 1.0 - 2.0 * M / rm

    s2 = -f * dt * dt + (dr * dr) / f + rm * rm * (dth * dth + (np.sin(thm) ** 2) * dph * dph)
    out = (dt > tol) & (s2 <= tol) & good
    return out

def kottler_exact_midpoint_preds_from_mapped(mapped_pts, i, M, Lambda, tol=1e-12):
    """
    Exact-metric midpoint test in Kottler coordinates.
    Same caveat as the Schwarzschild version.
    """
    Xi = mapped_pts[i]
    Xj = mapped_pts[:i]
    good = np.isfinite(Xj).all(axis=1)
    if not np.isfinite(Xi).all():
        return np.zeros(i, dtype=bool)

    dt = Xi[0] - Xj[:, 0]
    dr = Xi[1] - Xj[:, 1]
    dth = Xi[2] - Xj[:, 2]
    dph = _wrap_angle(Xi[3] - Xj[:, 3])

    rm = 0.5 * (Xi[1] + Xj[:, 1])
    thm = 0.5 * (Xi[2] + Xj[:, 2])
    f = 1.0 - 2.0 * M / rm - (Lambda * rm * rm) / 3.0

    s2 = -f * dt * dt + (dr * dr) / f + rm * rm * (dth * dth + (np.sin(thm) ** 2) * dph * dph)
    out = (dt > tol) & (s2 <= tol) & good
    return out

def schwarzschild_exact_midpoint_preds(pts, i, M, r0, mapped_pts=None, tol=1e-12):
    """
    Convenience wrapper: map points if needed, then apply exact-metric midpoint test.
    """
    if mapped_pts is None:
        mapped_pts = map_rnc_to_schwarzschild_expmap(pts, M, r0)
    return schwarzschild_exact_midpoint_preds_from_mapped(mapped_pts, i, M, tol=tol)

def kottler_exact_midpoint_preds(pts, i, M, r0, Lambda, mapped_pts=None, tol=1e-12):
    """
    Convenience wrapper: map points if needed, then apply exact-metric midpoint test.
    """
    if mapped_pts is None:
        mapped_pts = map_rnc_to_kottler_expmap(pts, M, r0, Lambda)
    return kottler_exact_midpoint_preds_from_mapped(mapped_pts, i, M, Lambda, tol=tol)

# --------------------------------------------------------------------
# Optional: exact null travel-time predicate in Schwarzschild coordinates
# --------------------------------------------------------------------

def _angular_separation(theta1, phi1, theta2, phi2):
    return math.acos(
        max(-1.0, min(1.0,
            math.cos(theta1)*math.cos(theta2)
            + math.sin(theta1)*math.sin(theta2)*math.cos(phi2 - phi1)
        ))
    )

def _Phi(r, b, M):
    f = 1.0 - 2.0*M/r
    return 1.0 - (b*b)*f/(r*r)

def _dpsi_dr(r, b, M):
    val = _Phi(r, b, M)
    return b / (r*r*math.sqrt(max(val, 1e-30)))

def _dt_dr(r, b, M):
    f = 1.0 - 2.0*M/r
    val = _Phi(r, b, M)
    return 1.0 / (f * math.sqrt(max(val, 1e-30)))

def schwarzschild_null_time_direct(ptA, ptB, M):
    """
    Exact null travel time in Schwarzschild coordinates for the direct, monotone-r branch.
    This is exact *if* the minimizing null geodesic has no turning point.
    Good for local-patch preview and midpoint-bias estimation.
    """
    tA, rA, thA, phA = map(float, ptA)
    tB, rB, thB, phB = map(float, ptB)
    psi = _angular_separation(thA, phA, thB, phB)
    if psi < 1e-15:
        # purely radial
        if abs(rB - rA) < 1e-15:
            return 0.0
        a, bR = (rA, rB) if rA < rB else (rB, rA)
        return quad(lambda rr: 1.0 / (1.0 - 2.0*M/rr), a, bR, limit=200)[0]

    a, bR = (rA, rB) if rA < rB else (rB, rA)

    def psi_of_b(b):
        val, _ = quad(lambda rr: _dpsi_dr(rr, b, M), a, bR, limit=200)
        return val

    # monotone branch requires Phi>0 throughout [a,bR]
    fmin = min(f_schwarzschild(a, M), f_schwarzschild(bR, M))
    bmax = min(a, bR) / math.sqrt(max(fmin, 1e-15)) * (1.0 - 1e-10)
    # solve psi_of_b = psi
    if psi_of_b(bmax * 0.999) < psi:
        return np.inf  # direct monotone branch not enough; would need turning-point branch
    bstar = brentq(lambda bb: psi_of_b(bb) - psi, 0.0, bmax * 0.999, xtol=1e-10, rtol=1e-10, maxiter=100)
    dt, _ = quad(lambda rr: _dt_dr(rr, bstar, M), a, bR, limit=200)
    return dt

def schwarzschild_exact_null_preds_from_mapped(mapped_pts, i, M, tol=1e-12):
    """
    Exact causal predicate using exact Schwarzschild null travel time on the direct branch.
    Much more expensive than midpoint, but conceptually the correct route.
    """
    Xi = mapped_pts[i]
    out = np.zeros(i, dtype=bool)
    if not np.isfinite(Xi).all():
        return out
    for j in range(i):
        Xj = mapped_pts[j]
        if not np.isfinite(Xj).all():
            continue
        dt = Xi[0] - Xj[0]
        if dt <= tol:
            continue
        tneed = schwarzschild_null_time_direct(Xj, Xi, M)
        out[j] = np.isfinite(tneed) and (dt >= tneed - tol)
    return out

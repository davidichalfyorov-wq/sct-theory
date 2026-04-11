"""
Seed BVP system for the a6-truncated SCT spherical problem in modified Schwarzschild coordinates

    ds^2 = -H(rho) dt^2 + drho^2 / H(rho) + F(rho)^2 dOmega^2.

This file provides:
  1) compact curvature/Weyl invariants,
  2) the reduced 1D Lagrangian density for the explicit Weyl-localized a6 seed,
  3) the highest-derivative Hessian A_ij for the sixth-order system,
  4) center-series boundary data for the smooth even branch,
  5) the spin-2 characteristic roots for the pure Weyl a6 seed.

Notes
-----
* The explicit a6 density implemented here is the compact local seed
      R + alpha_R R^2 + alpha_C C^2 + (mu3/Lambda^2) I6
  with
      mu3 = 307/2520, alpha_C = 13/120.
* For xi = 1/6 one has alpha_R = 0 at the a4 level.
* If you want the full xi-dependent SCT a6 completion from the Standard Model,
  insert the extra R-sector coefficients from the full heat-kernel coefficient
  into I6; the BVP architecture below is unchanged.
"""
from __future__ import annotations

import sympy as sp

rho = sp.symbols('rho', real=True, finite=True)
Lambda = sp.symbols('Lambda', positive=True, finite=True)
alpha_C = sp.Rational(13, 120)
c2 = sp.Rational(13, 60)
f1hat = -sp.Rational(307, 182)
mu3 = sp.Rational(307, 2520)

F = sp.Function('F')(rho)
H = sp.Function('H')(rho)

def d(expr, n=1):
    return sp.diff(expr, rho, n)

def invariants(F=F, H=H):
    F1, F2, F3 = d(F,1), d(F,2), d(F,3)
    H1, H2, H3 = d(H,1), d(H,2), d(H,3)

    # Ricci mixed eigenvalues: Rt_t = a, Rrho_rho = b, Rtheta_theta = Rphi_phi = c
    a = -sp.Rational(1,2)*H2 - (F1/F)*H1
    b = -sp.Rational(1,2)*H2 - 2*H*F2/F - (F1/F)*H1
    c = (1 - H*F1**2 - F*H*F2 - F*F1*H1)/F**2

    R = a + b + 2*c

    E = -F**2*H2 + 2*F*H*F2 + 2*F*F1*H1 - 2*H*F1**2 + 2
    W = E/(6*F**2)

    J = -F*d(E,1) + 2*F1*E  # linear in F''' and H'''

    Ric2 = a**2 + b**2 + 2*c**2
    Ric3 = a**3 + b**3 + 2*c**3
    C2 = 12*W**2
    C3 = 12*W**3

    # R_{mu nu} R_{rho sigma} C^{mu rho nu sigma}
    Q = -2*W*(a - c)*(b - c)

    # (nabla C)^2 after spherical reduction
    gradC2 = H*(J**2 + 6*E**2*F1**2)/(3*F**6)

    # For scalars depending only on rho one has (nabla R)^2 = H (R')^2
    gradR2 = H*d(R,1)**2

    return {
        'a': sp.simplify(a),
        'b': sp.simplify(b),
        'c': sp.simplify(c),
        'R': sp.simplify(R),
        'E': sp.simplify(E),
        'W': sp.simplify(W),
        'J': sp.simplify(J),
        'Ric2': sp.simplify(Ric2),
        'Ric3': sp.simplify(Ric3),
        'C2': sp.simplify(C2),
        'C3': sp.simplify(C3),
        'Q': sp.simplify(Q),
        'gradC2': sp.simplify(gradC2),
        'gradR2': sp.simplify(gradR2),
    }

def lagrangian_density(alpha_R=sp.Integer(0), F=F, H=H, Lambda=Lambda):
    """
    Reduced 1D density L(F,H) up to an irrelevant overall factor 4*pi/(16*pi*G).

    L = F^2 [ R + alpha_R R^2 + alpha_C C^2 + (mu3/Lambda^2) I6 ].
    """
    inv = invariants(F,H)
    R = inv['R']
    W = inv['W']
    Ric2 = inv['Ric2']
    Ric3 = inv['Ric3']
    gradC2 = inv['gradC2']
    gradR2 = inv['gradR2']

    I6 = (
        sp.Rational(1,10)*gradR2
        + sp.Rational(3,2)*gradC2
        - sp.Rational(2,135)*R**3
        + sp.Rational(1,3)*R*Ric2
        - sp.Rational(13,15)*Ric3
        + 7*R*W**2
        - sp.Rational(46,5)*W*(inv['a'] - inv['c'])*(inv['b'] - inv['c'])
        + sp.Rational(266,5)*W**3
    )
    L = F**2 * (R + alpha_R*R**2 + 12*alpha_C*W**2 + (mu3/Lambda**2)*I6)
    return sp.simplify(L)

def euler_residuals(alpha_R=sp.Integer(0), F=F, H=H, Lambda=Lambda):
    """
    Symbolic Euler-Lagrange residuals for a third-derivative Lagrangian:
        E_F = 0, E_H = 0.
    This can be slow, but is feasible because the reduced density is compact.
    """
    L = lagrangian_density(alpha_R=alpha_R, F=F, H=H, Lambda=Lambda)

    def EL(y):
        y1, y2, y3 = d(y,1), d(y,2), d(y,3)
        return sp.simplify(
            sp.diff(L, y)
            - d(sp.diff(L, y1), 1)
            + d(sp.diff(L, y2), 2)
            - d(sp.diff(L, y3), 3)
        )

    return EL(F), EL(H)

def highest_derivative_matrix(F=F, H=H, Lambda=Lambda):
    """
    A_ij = d^2 L / (d y_i''' d y_j'''),  y = (F,H).

    For the compact a6 seed one gets the closed form
        A = (6 mu3 H / (5 Lambda^2)) [[6H^2, -F H], [-F H, F^2]]
    and therefore
        det A = (36 mu3^2 / (5 Lambda^4)) F^2 H^4.
    """
    pref = sp.Rational(6,5) * mu3 * H / Lambda**2
    A = pref * sp.Matrix([
        [6*H**2, -F*H],
        [-F*H, F**2]
    ])
    detA = sp.simplify(A.det())
    return sp.simplify(A), detA

def center_even_series(eps, h2, h4, f3, f5):
    """
    Smooth even center data at rho = eps > 0:

        F = rho + f3 rho^3 + f5 rho^5 + ...
        H = 1 + h2 rho^2 + h4 rho^4 + ...

    Returns the 12 first-order state components
        (F, F1, F2, F3, F4, F5, H, H1, H2, H3, H4, H5)
    at rho = eps.
    """
    eps = sp.sympify(eps)
    return {
        'F' : eps + f3*eps**3 + f5*eps**5,
        'F1': 1 + 3*f3*eps**2 + 5*f5*eps**4,
        'F2': 6*f3*eps + 20*f5*eps**3,
        'F3': 6*f3 + 60*f5*eps**2,
        'F4': 120*f5*eps,
        'F5': 120*f5,
        'H' : 1 + h2*eps**2 + h4*eps**4,
        'H1': 2*h2*eps + 4*h4*eps**3,
        'H2': 2*h2 + 12*h4*eps**2,
        'H3': 24*h4*eps,
        'H4': 24*h4,
        'H5': 0,
    }

def compactified_derivative_scale(rho_star, x):
    """
    rho = rho_star * x / (1 - x),   x in [0,1)
    d/dx = Delta * d/drho,         Delta = rho_star / (1 - x)^2.
    """
    x = sp.sympify(x)
    rho_star = sp.sympify(rho_star)
    return sp.simplify(rho_star / (1 - x)**2)

def spin2_characteristic_roots():
    """
    Roots u of the pure Weyl a6 spin-2 polynomial
        P2(u) = 1 - c2 u + c2 f1hat u^2
    obtained from the user-specified SCT truncation.
    """
    u = sp.symbols('u')
    roots = sp.solve(sp.Eq(1 - c2*u + c2*f1hat*u**2, 0), u)
    return sp.simplify(roots[0]), sp.simplify(roots[1])

if __name__ == "__main__":
    inv = invariants()
    print("R =", inv['R'])
    print("W =", inv['W'])
    A, detA = highest_derivative_matrix()
    print("A =")
    sp.pprint(A)
    print("det A =", detA)
    r1, r2 = spin2_characteristic_roots()
    print("spin-2 roots:", r1, r2)

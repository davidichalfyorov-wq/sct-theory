
"""
Derive the local nonextremal horizon Frobenius series for the pure Weyl a6 seed.

This script:
  1) reconstructs the unsolved Euler-Lagrange equations from the reduced 1D action,
  2) substitutes the horizon ansatz
         F = r0 + f1 x + f2 x^2 + f3 x^3 + ...
         H = h1 x + h2 x^2 + h3 x^3 + ...
  3) fixes the scale gauge h1 = 1,
  4) solves recursively for h4, f4, h5, f5, h6,
  5) performs short numerical integrations on both sides of the horizon for sample parameters.

Requires: sympy, numpy, scipy.
"""
from __future__ import annotations
import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
import importlib.util

# Load numerical RHS from the existing driver
spec = importlib.util.spec_from_file_location("drv", "/mnt/data/sct_bvp_driver.py")
drv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drv)

x = sp.symbols('x')
q = sp.symbols('q0:7')
p = sp.symbols('p0:7')
q0,q1,q2,q3,q4,q5,q6 = q
p0,p1,p2,p3,p4,p5,p6 = p

alpha_C = sp.Rational(13,120)
mu3 = sp.Rational(307,2520)
Lam = sp.Integer(1)

# Jet-space total derivative
def D(expr, order=1):
    res = expr
    for _ in range(order):
        dres = 0
        for i in range(6):
            dres += sp.diff(res, q[i])*q[i+1] + sp.diff(res, p[i])*p[i+1]
        res = sp.expand(dres)
    return res

# Invariants
a = -sp.Rational(1,2)*p2 - (q1/q0)*p1
b = -sp.Rational(1,2)*p2 - 2*p0*q2/q0 - (q1/q0)*p1
c = (1 - p0*q1**2 - q0*p0*q2 - q0*q1*p1)/q0**2
R = a + b + 2*c
E = -q0**2*p2 + 2*q0*p0*q2 + 2*q0*q1*p1 - 2*p0*q1**2 + 2
W = E/(6*q0**2)
Ep = D(E)
Rp = D(R)
J = -q0*Ep + 2*q1*E
Ric2 = a**2 + b**2 + 2*c**2
Ric3 = a**3 + b**3 + 2*c**3
gradC2 = p0*(J**2 + 6*E**2*q1**2)/(3*q0**6)
gradR2 = p0*Rp**2

I6 = (
    sp.Rational(1,10)*gradR2
    + sp.Rational(3,2)*gradC2
    - sp.Rational(2,135)*R**3
    + sp.Rational(1,3)*R*Ric2
    - sp.Rational(13,15)*Ric3
    + 7*R*W**2
    - sp.Rational(46,5)*W*(a-c)*(b-c)
    + sp.Rational(266,5)*W**3
)
L = sp.expand(q0**2*(R + 12*alpha_C*W**2 + (mu3/Lam**2)*I6))

dLq = [sp.expand(sp.diff(L, q[i])) for i in range(4)]
dLp = [sp.expand(sp.diff(L, p[i])) for i in range(4)]
ELF = sp.expand(dLq[0] - D(dLq[1]) + D(D(dLq[2])) - D(D(D(dLq[3]))))
ELH = sp.expand(dLp[0] - D(dLp[1]) + D(D(dLp[2])) - D(D(D(dLp[3]))))

# Horizon ansatz
r0,f1,f2,f3,f4,f5c,f6c = sp.symbols('r0 f1 f2 f3 f4 f5c f6c')
h1,h2,h3,h4,h5c,h6c = sp.symbols('h1 h2 h3 h4 h5c h6c')

Fser = r0 + f1*x + f2*x**2 + f3*x**3 + f4*x**4 + f5c*x**5 + f6c*x**6
Hser = h1*x + h2*x**2 + h3*x**3 + h4*x**4 + h5c*x**5 + h6c*x**6

subs = {}
for i,qi in enumerate(q):
    subs[qi] = sp.diff(Fser, x, i)
for i,pi in enumerate(p):
    subs[pi] = sp.diff(Hser, x, i)

exprF = ELF.subs(subs)
exprH = ELH.subs(subs)

serF = sp.series(exprF, x, 0, 4).removeO()
serH = sp.series(exprH, x, 0, 4).removeO()

cF0 = sp.expand(serF).coeff(x,0)
cF1 = sp.expand(serF).coeff(x,1)
cF2 = sp.expand(serF).coeff(x,2)
cH0 = sp.expand(serH).coeff(x,0)
cH1 = sp.expand(serH).coeff(x,1)
cH2 = sp.expand(serH).coeff(x,2)

# Fix scale gauge h1 = 1 and solve recursively
h4_expr = sp.solve(sp.Eq(cF0.subs(h1,1), 0), h4, dict=True)[0][h4]
eqH0 = sp.expand(cH0.subs({h1:1, h4:h4_expr}))
eqF1 = sp.expand(cF1.subs({h1:1, h4:h4_expr}))
sol45 = sp.solve([sp.Eq(eqH0,0), sp.Eq(eqF1,0)], [f4,h5c], dict=True)[0]

eqH1 = sp.expand(cH1.subs({h1:1, h4:h4_expr, f4:sol45[f4], h5c:sol45[h5c]}))
eqF2 = sp.expand(cF2.subs({h1:1, h4:h4_expr, f4:sol45[f4], h5c:sol45[h5c]}))
sol56 = sp.solve([sp.Eq(eqH1,0), sp.Eq(eqF2,0)], [f5c,h6c], dict=True)[0]

print("h4 =", sp.simplify(h4_expr))
print("f4 =", sp.simplify(sol45[f4]))
print("h5 =", sp.simplify(sol45[h5c]))
print("f5 =", sp.simplify(sol56[f5c]))
print("h6 =", sp.simplify(sol56[h6c]))

# Lambdify for numerical tests
vars_free = (r0,f1,f2,f3,h2,h3)
h4_fun = sp.lambdify(vars_free, sp.simplify(h4_expr), 'numpy')
f4_fun = sp.lambdify(vars_free, sp.simplify(sol45[f4]), 'numpy')
h5_fun = sp.lambdify(vars_free, sp.simplify(sol45[h5c]), 'numpy')
f5_fun = sp.lambdify(vars_free, sp.simplify(sol56[f5c]), 'numpy')
h6_fun = sp.lambdify(vars_free, sp.simplify(sol56[h6c]), 'numpy')

def horizon_series_state(eps, pars):
    r0v,f1v,f2v,f3v,h2v,h3v = pars['r0'],pars['f1'],pars['f2'],pars['f3'],pars['h2'],pars['h3']
    h4v = h4_fun(r0v,f1v,f2v,f3v,h2v,h3v)
    f4v = f4_fun(r0v,f1v,f2v,f3v,h2v,h3v)
    h5v = h5_fun(r0v,f1v,f2v,f3v,h2v,h3v)
    f5v = f5_fun(r0v,f1v,f2v,f3v,h2v,h3v)
    h6v = h6_fun(r0v,f1v,f2v,f3v,h2v,h3v)
    x = eps
    F = r0v + f1v*x + f2v*x**2 + f3v*x**3 + f4v*x**4 + f5v*x**5
    F1 = f1v + 2*f2v*x + 3*f3v*x**2 + 4*f4v*x**3 + 5*f5v*x**4
    F2 = 2*f2v + 6*f3v*x + 12*f4v*x**2 + 20*f5v*x**3
    F3 = 6*f3v + 24*f4v*x + 60*f5v*x**2
    F4 = 24*f4v + 120*f5v*x
    F5 = 120*f5v
    H = x + h2v*x**2 + h3v*x**3 + h4v*x**4 + h5v*x**5 + h6v*x**6
    H1 = 1 + 2*h2v*x + 3*h3v*x**2 + 4*h4v*x**3 + 5*h5v*x**4 + 6*h6v*x**5
    H2 = 2*h2v + 6*h3v*x + 12*h4v*x**2 + 20*h5v*x**3 + 30*h6v*x**4
    H3 = 6*h3v + 24*h4v*x + 60*h5v*x**2 + 120*h6v*x**3
    H4 = 24*h4v + 120*h5v*x + 360*h6v*x**2
    H5 = 120*h5v + 720*h6v*x
    return np.array([F,F1,F2,F3,F4,F5,H,H1,H2,H3,H4,H5], dtype=float)

def rhs_local(rho, y):
    return np.array(drv.rhs(rho, y[:,None], None, Lam=1.0)).reshape(12)

def local_test(pars, eps=1e-3, delta=0.02):
    y0p = horizon_series_state(eps, pars)
    y0m = horizon_series_state(-eps, pars)
    solp = solve_ivp(rhs_local, (eps, delta), y0p, method='Radau', rtol=1e-8, atol=1e-10, max_step=1e-3)
    solm = solve_ivp(rhs_local, (-eps, -delta), y0m, method='Radau', rtol=1e-8, atol=1e-10, max_step=1e-3)
    return {
        "forward_ok": solp.status == 0,
        "backward_ok": solm.status == 0,
        "H_forward_end": float(solp.y[6,-1]) if solp.status == 0 else None,
        "H_backward_end": float(solm.y[6,-1]) if solm.status == 0 else None,
    }

for pars in [
    {"r0":1.0,"f1":1.0,"f2":0.0,"f3":0.0,"h2":0.0,"h3":0.0},
    {"r0":1.3,"f1":0.8,"f2":0.1,"f3":-0.05,"h2":-0.2,"h3":0.1},
    {"r0":0.9,"f1":1.2,"f2":-0.15,"f3":0.08,"h2":0.25,"h3":-0.12},
]:
    print(pars, local_test(pars))

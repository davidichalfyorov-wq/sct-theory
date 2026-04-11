"""Compute the Bach tensor B^t_t for static spherical metric via Schouten tensor."""
from sympy import *

r = Symbol('r', positive=True)
m = Function('m')
mr = m(r)

# Schouten tensor S_{ab} = R_{ab} - R g_{ab}/6 (4D)
# For f = 1 - 2m/r (linearized around flat):
# R^t_t = m''/r
# R^r_r = m''/r
# R^th_th = 2m'/(r^2)
# R = 2(r m'' + 2m')/(r^2)

Rtt = diff(mr, r, 2) / r
R_scalar = 2*(r*diff(mr, r, 2) + 2*diff(mr, r)) / r**2

Stt = Rtt - R_scalar/6
Stt = simplify(Stt)
print("S^t_t =", Stt)

# In flat background (linearized), for a STATIC DIAGONAL perturbation:
# B^t_t = nabla^2 S^t_t - nabla^c nabla_t S_{tc}
#
# nabla^c nabla_t S_{tc}:
# Since S is diagonal and static, S_{tc} = 0 for c != t, and
# nabla_t anything = 0 (static).
# So nabla^c nabla_t S_{tc} = 0.
#
# BUT WAIT: the correct formula for Bach in 4D is:
# B_{ab} = nabla^c nabla_c S_{ab} + C^{cdab} S_{cd}
# = nabla^2 S_{ab} + C^{cdab} S_{cd}
# where I used the symmetry of the Weyl tensor.
# In the linearized limit, C = O(h) and S = O(h), so C*S = O(h^2).
# Therefore B_{ab} = nabla^2 S_{ab} at linear order.
#
# Hmm but this is only a 2nd-order operator on S, and S involves 2nd
# derivatives of h, so B is 4th order in h. Good.
#
# But actually, the standard formula for Bach is:
# B_{ab} = (nabla^c nabla^d + (1/2) R^{cd}) C_{acbd}
# Let me verify this is equivalent to nabla^2 S_{ab} in 4D.
#
# In 4D, C_{acbd} = R_{acbd} - (stuff with R_{ab} and R)
# nabla^c nabla^d C_{acbd} = nabla^c nabla^d R_{acbd} - nabla^c nabla^d(Ricci stuff)
# By Bianchi: nabla^d R_{acbd} = nabla_b R_{ac} - nabla_c R_{ab}
# So nabla^c nabla^d R_{acbd} = nabla^c(nabla_b R_{ac} - nabla_c R_{ab})
# = nabla^c nabla_b R_{ac} - nabla^2 R_{ab}
# = nabla_b nabla^c R_{ac} + [Riemann terms] - nabla^2 R_{ab}
# = nabla_b nabla_a R/2 + ... - nabla^2 R_{ab}
# (using contracted Bianchi: nabla^c R_{ac} = nabla_a R / 2)
#
# This shows B involves nabla^2 R_{ab} and nabla_a nabla_b R terms.
# In 4D: B_{ab} = -nabla^2 R_{ab} + nabla_a nabla_b R/3 + ... (Ricci terms)
#
# For the LINEARIZED TRACE-FREE part:
# B_{ab} = -(nabla^2 + R/6) G_{ab}/2 (schematic)
# which gives the well-known (1 + alpha Box) G = 0 structure.

# Let me just directly compute: B^t_t = nabla^2(S^t_t) for the (tt) component.

# In flat space spherical coords: nabla^2(phi) = phi'' + (2/r) phi'
nab2_Stt = diff(Stt, r, 2) + 2 * diff(Stt, r) / r
nab2_Stt_s = simplify(nab2_Stt)
print("\nnabla^2(S^t_t) =", nab2_Stt_s)

# Now substitute m = a r^n:
a_s, n_s = symbols('a n')
m_sub = a_s * r**n_s

# Build substitution dictionary (highest order first)
expr = nab2_Stt_s
for k in range(6, -1, -1):
    if k == 0:
        expr = expr.subs(mr, m_sub)
    else:
        expr = expr.subs(diff(mr, r, k), diff(m_sub, r, k))

expr = simplify(expr)
print("\nnabla^2(S^t_t) for m = a r^n:", expr)

# Extract the coefficient of a * r^{n-5}:
# The expression should be proportional to a * r^{n-5}
coeff = simplify(expr / (a_s * r**(n_s - 5)))
print("Coefficient (relative to a r^{n-5}):", simplify(coeff))
print("Factored:", factor(coeff))

# Now the full Stelle equation (tt):
# G^t_t + 2 alpha B^t_t = 0
# -2n a r^{n-3} + 2 alpha * [coeff * a * r^{n-5}] = 0
#
# Near r=0: the B term (r^{n-5}) dominates over G term (r^{n-3}) for n < 5.
# At leading order: B^t_t = 0, i.e., coeff = 0.
# This gives the indicial equation from the Weyl part alone.

# BUT this misses the Weyl-zero modes (n=2, n=3).
# For n=2 or n=3: X = (n-2)(n-3) = 0, so W = 0, and S^t_t simplifies.
# Let me check what happens for n=2:
print("\n=== Check n=2 ===")
expr_n2 = expr.subs(n_s, 2)
print(f"nabla^2(S^t_t) at n=2: {simplify(expr_n2)}")
# For n=2: G^t_t = -4a/r, which is r^{-1}.
# B^t_t should be r^{-3} (from r^{n-5} = r^{-3}).
# The equation G + alpha B = 0: r^{-1} + alpha r^{-3} = 0
# This only balances if the r^{-3} coefficient is 0, meaning B^t_t = 0 at n=2.

print("\n=== Check n=3 ===")
expr_n3 = expr.subs(n_s, 3)
print(f"nabla^2(S^t_t) at n=3: {simplify(expr_n3)}")

# Now check the indicial polynomial:
print("\n=== Indicial polynomial from B^t_t = 0 ===")
# For general n, B^t_t ~ coeff * a * r^{n-5}
# The equation B^t_t = 0 requires coeff = 0.
indicial_from_B = factor(coeff)
print(f"Indicial polynomial: {indicial_from_B} = 0")
print(f"Expanded: {expand(coeff)}")

# Solve:
roots = solve(coeff, n_s)
print(f"Roots: {roots}")
for root in roots:
    print(f"  n = {root} = {float(root):.6f}")

# The issue: the Bach tensor computation via nabla^2(S^t_t) is NOT the
# full story. We also need the term nabla_a nabla_b R (or equivalently
# the gradient terms of the Schouten tensor).
# Let me compute the FULL Bach tensor more carefully.

print("\n" + "=" * 60)
print("FULL Bach tensor: B_{ab} = nabla^c nabla^d C_{acbd} + (1/2) R^{cd} C_{cadb}")
print("In linearized limit: B_{ab} = nabla^c nabla^d C_{acbd}")
print("=" * 60)

# In the linearized regime, the Weyl tensor for f = 1 - 2m/r is:
# C_{0101} = -2W (in ONB)
# where W = -(r^2 m'' - 4r m' + 6m)/(6r^3)
#
# The Bach tensor component:
# B^t_t = nabla^c nabla^d C^d_{tct}
# = nabla^r nabla^d C^d_{trt} + nabla^theta nabla^d C^d_{t theta t} + nabla^phi nabla^d C^d_{t phi t}
#
# Now, C^d_{trt} for our static perturbation:
# C^t_{trt} = 0 (antisymmetry C_{abcd} = -C_{bacd})
# C^r_{trt} = g^{rr} C_{rtrt} = C_{0101} (in flat) = -2W
# C^theta_{trt} = 0 (only diagonal components nonzero)
# So C^d_{trt} has only d=r component: C^r_{trt} = -2W
# (with appropriate sign for the metric convention)

# Similarly:
# C^d_{t theta t}:
# C^t_{t theta t} = 0
# C^r_{t theta t} = 0 (C_{rtthetat} = 0 for our Petrov D)
# C^theta_{t theta t} = g^{theta theta} C_{theta t theta t} = (1/r^2) C_{2020}
# In ONB: C_{0202} = W, so C_{theta t theta t} = W * r^{-2} (coordinate)
# Actually C^theta_{t theta t} = W/r^2 (I need to be careful with factors of r)

# This computation is very tricky with the metric factors. Let me use
# a different strategy: compute everything in the COORDINATE basis using
# the standard Christoffel symbols for flat space in spherical coords.

# In flat space spherical coords (r, theta, phi):
# g = diag(1, r^2, r^2 sin^2 theta)
# The linearized perturbation: h_tt = 2m/r, h_rr = -2m/r (in Schwarzschild gauge)
# ... this is getting very involved.

# ALTERNATIVE: Use the known result that the linearized Stelle equation
# in static spherical can be written as:
#
#   m' + alpha_C * [r^4 W']' / r^2 = 0
# or equivalently
#   -2m'/r^2 + (2 alpha_C / r^2) d/dr[r^4 dW/dr + ... ] = 0
#
# The EXACT form of the Bach contribution depends on the normalization.
# Let me look at this from the factored form of the 4th-order equation.

# For the LINEARIZED Stelle equation in static spherical:
# The equation for the potential V(r) = -m(r)/r (Newtonian potential):
#   (nabla^2)(nabla^2 - m_2^2) V = 0
# where m_2^2 = 1/(2 alpha_C) and nabla^2 = d^2/dr^2 + (2/r) d/dr.
#
# This is the CORRECT form! It gives a 4th-order equation for V,
# hence for m = -r V.
#
# Frobenius for V ~ r^p:
# nabla^2(r^p) = p(p+1) r^{p-2}
# nabla^2(nabla^2 - m_2^2)(r^p) = nabla^2[p(p+1) r^{p-2} - m_2^2 r^p]
# = p(p+1)(p-2)(p-1) r^{p-4} - m_2^2 p(p+1) r^{p-2}
#
# Near r=0, the first term dominates (for p < 2):
# p(p+1)(p-2)(p-1) = 0
# Roots: p = 0, -1, 2, 1
#
# Since m = -r V, if V ~ r^p then m ~ r^{p+1}, so n = p+1:
# n = 1, 0, 3, 2
# i.e., n = {0, 1, 2, 3}
#
# That gives n = 0, 1, 2, 3 -- NOT what analytical claims!

# Wait, this is for the SCALAR Laplacian acting on V. But the Stelle
# equation involves the TENSOR structure. Let me reconsider.

# The Stelle equation for linearized perturbation h_{ab}:
# (1 - 2 alpha_C nabla^2)(R_{ab} - R g_{ab}/2) = 0  (schematic)
#
# This is NOT the same as nabla^2(nabla^2 - m_2^2) V = 0.
# The tensor Laplacian on the Ricci tensor is different from
# the scalar Laplacian on V.

# The CORRECT factored form for the mass function is:
# For the SCALAR sector (trace part of the perturbation):
#   (nabla^2 - m_0^2) R = 0 where m_0^2 = 1/(6 beta)
# For the TENSOR sector (traceless Weyl part):
#   The equation for the Weyl scalar W is:
#   (D_2 - m_2^2) W = 0
# where D_2 is the spin-2 radial operator (NOT the scalar Laplacian).
#
# For static spherical, D_2 acting on W(r) (which transforms as l=2 under SO(3)):
# D_2 W = W'' + (2/r) W' - 6/r^2 W  (effective angular momentum l(l+1) = 6)
#
# Wait, this IS the standard formula for the radial part of the l=2 mode.
# D_2 W = W'' + (2/r) W' - 6/r^2 W
#
# For W ~ r^s: s(s-1) + 2s - 6 = s^2 + s - 6 = (s-2)(s+3) = 0
# So s = 2 or s = -3.
#
# With s = n - 3: n - 3 = 2 => n = 5, or n - 3 = -3 => n = 0.
# These are NOT the claimed roots either.
#
# Hmm. Let me reconsider what l(l+1) should be.
# The Weyl tensor is a rank-4 tensor, but its independent component
# in Petrov type D is a single scalar W(r). Under SO(3), this transforms
# as l=2 (quadrupole). The SCALAR equation for W is:
#   nabla^2_scalar W - l(l+1)/r^2 W = 0
# with l=2: nabla^2_scalar W - 6/r^2 W = 0
# This gives (s^2 + s - 6) = 0 as above.
#
# But the analytical result has s^2 - 6 = 0 (no linear s term).
# The difference is the s term, which comes from the (2/r) W' part.
#
# If the correct equation were W'' - 6/r^2 W = 0 (WITHOUT the 2/r W' term),
# then s(s-1) - 6 = 0 => s^2 - s - 6 = 0 => (s-3)(s+2) = 0, giving s=3, -2.
# Still not s^2 = 6.
#
# If the equation were r^2 W'' - 6 W = 0 (i.e., just the Euler equation):
# s(s-1) - 6 = 0 => s^2 - s - 6 = (s-3)(s+2) = 0, same thing.
#
# What equation gives s^2 = 6?
# We need p(s) = s^2 - 6 = 0.
# This comes from an Euler equation of the form:
#   r^2 phi'' + r phi' - 6 phi = 0 ... no, that gives s^2 - 6 = 0? Let's check:
# phi ~ r^s: s(s-1) + s - 6 = s^2 - 6 = 0. YES!
# So the equation is r^2 phi'' + r phi' - 6 phi = 0,
# i.e., (r d/dr)^2 phi - 6 phi = 0,
# i.e., the Euler operator x^2 D^2 with eigenvalue 6.
# This is equivalent to D_Euler phi = 6 phi where D_Euler = (r d/dr)^2.

# What quantity phi satisfies this equation?
# If phi = r^2 W (or phi = r^{-1} W, etc.), the equation might work out.
# Let phi = r^a W, then:
# r^2 (r^a W)'' + r (r^a W)' - 6 r^a W = 0
# r^2 [a(a-1) r^{a-2} W + 2a r^{a-1} W' + r^a W''] + r[a r^{a-1} W + r^a W'] - 6 r^a W = 0
# [a(a-1) + a] W + [2a + 1] r W' + r^2 W'' = 6 W/r^{2a} ... this is messy.

# Let me try phi = X = r^2 m'' - 4r m' + 6m = -6 r^3 W.
# X ~ c r^n where c = a(n-2)(n-3).
# The Euler operator: (r d/dr)^2 (c r^n) = n^2 c r^n.
# So (r d/dr)^2 X = n^2 X.
# The equation (r d/dr)^2 X - 6 X = 0 gives n^2 = 6, i.e., n = sqrt(6).
# But s = n - 3 (Weyl exponent), and n = sqrt(6) does NOT match s^2 = 6.

# Hmm, wait. Let me reconsider the original analytical claim.
# analytical says the Weyl equation gives s^2 - 6 = 0 where s is the exponent
# of the Weyl tensor. Let me check if the original E-L calculation was wrong.

# Going back to basics: I computed the E-L equation from L = (4/3) X^2/r^4.
# The script got P(n) = 8n(n-1)/3. Let me verify this is correct by
# a completely independent computation: vary X^2/r^4 with respect to m.

# L = (4/3) X^2 / r^4 where X = r^2 m'' - 4r m' + 6m
#
# Variation: delta L = (8/3) X delta X / r^4
# delta X = r^2 delta m'' - 4r delta m' + 6 delta m
#
# integral delta L dr = (8/3) integral X (r^2 delta m'' - 4r delta m' + 6 delta m) / r^4 dr
# = (8/3) integral [X delta m''/ r^2 - 4X delta m' / r^3 + 6X delta m / r^4] dr
#
# Integrate by parts:
# integral X delta m'' / r^2 dr = [X delta m' / r^2] - integral (X/r^2)' delta m' dr
# = [X delta m' / r^2] - [(X/r^2)' delta m] + integral (X/r^2)'' delta m dr
#
# integral -4X delta m' / r^3 dr = [-4X delta m / r^3] + integral 4(X/r^3)' delta m dr
#
# Collecting: (ignoring boundary terms)
# integral delta L dr = (8/3) integral [(X/r^2)'' + 4(X/r^3)' + 6X/r^4] delta m dr
#
# So E-L: (X/r^2)'' + 4(X/r^3)' + 6X/r^4 = 0

# For X = c r^n (c = a(n-2)(n-3)):
# (c r^n / r^2)'' = c(n-2)(n-3) r^{n-4}
# 4(c r^n / r^3)' = 4c(n-3) r^{n-4}
# 6 c r^n / r^4 = 6c r^{n-4}
#
# Sum: c[(n-2)(n-3) + 4(n-3) + 6] r^{n-4} = 0
# (n-2)(n-3) + 4(n-3) + 6 = n^2 - 5n + 6 + 4n - 12 + 6 = n^2 - n = n(n-1)
#
# So the inner indicial polynomial is n(n-1), giving n = 0 or n = 1.
# Combined with (n-2)(n-3) from c: (n-2)(n-3) * n(n-1) = 0
# Roots: n = 0, 1, 2, 3.

# THIS IS DIFFERENT FROM THE ERRONEOUS CLAIM!
# The erroneous claim of (n-2)(n-3)[(n-3)^2 - 6] = 0 gives n = 2, 3, 3-sqrt(6), 3+sqrt(6).
# My derivation gives n(n-1)(n-2)(n-3) = 0, i.e., n = 0, 1, 2, 3.

print("\n" + "=" * 60)
print("CRITICAL FINDING")
print("=" * 60)
print()
print("The Euler-Lagrange equation from L = (4/3) X^2/r^4 gives:")
print("  (X/r^2)'' + 4(X/r^3)' + 6X/r^4 = 0")
print()
print("For X = c r^n with c = a(n-2)(n-3):")
print("  c * [n(n-1)] * r^{n-4} = 0")
print()
print("Full indicial polynomial: n(n-1)(n-2)(n-3) = 0")
print("Roots: n = 0, 1, 2, 3")
print()
print("This DIFFERS from the analytical claim of (n-2)(n-3)[(n-3)^2-6] = 0")
print("  which gives n = 2, 3, 3-sqrt(6), 3+sqrt(6).")
print()

# Let me verify: the inner polynomial should be:
# (n-2)(n-3) + 4(n-3) + 6
nn = Symbol('n')
inner = (nn-2)*(nn-3) + 4*(nn-3) + 6
print(f"Inner polynomial: {expand(inner)} = {factor(inner)}")
print(f"  = {expand(inner)}")
# n^2 - 5n + 6 + 4n - 12 + 6 = n^2 - n = n(n-1)
print(f"  = n(n-1)")
print()

# Double-check by substituting n = 3-sqrt(6) into n(n-1):
n_test = 3 - sqrt(6)
val = n_test * (n_test - 1)
print(f"n(n-1) at n = 3-sqrt(6): {simplify(val)} = {float(val):.6f}")
print(f"This is NOT zero! So n = 3-sqrt(6) is NOT a root of n(n-1) = 0.")
print()

# BUT WAIT: the above derivation assumed that the action density is
# L = (4/3) X^2 / r^4, which comes from C^2 * r^2 dr (with the measure r^2).
# Let me double-check the measure factor.
#
# C^2 = 48 W^2 = 48 * [X/(6r^3)]^2 = 48 X^2 / (36 r^6) = (4/3) X^2/r^6
# The action is integral C^2 sqrt(-g) d^4x = integral C^2 r^2 sin(theta) dr d(theta) d(phi) dt
# After angular integration (4pi) and time integration:
# S_W = 4pi T * integral C^2 * r^2 dr = 4pi T * integral (4/3) X^2/r^6 * r^2 dr
#     = (16 pi T / 3) integral X^2 / r^4 dr
#
# So L = X^2/r^4 (up to constant factor). This is CORRECT.

# Now let me also verify by computing the FULL Stelle equation differently.
# The linearized Stelle equation for the Newtonian potential V = -m/r:
# (nabla^2)(nabla^2 - m_2^2) V = S  (where S is the source)
# In vacuum (S = 0) near r=0 (m_2^2 ~ 0):
# nabla^4 V = 0  (biharmonic equation)
#
# V = m/r ~ a r^{n-1}
# nabla^2(r^p) = p(p+1) r^{p-2}
# nabla^4(r^p) = p(p+1)(p-2)(p-1) r^{p-4}
#
# With p = n-1:
# (n-1)n(n-3)(n-2) r^{n-5} = 0
# => n(n-1)(n-2)(n-3) = 0
# Roots: n = 0, 1, 2, 3
#
# EXACTLY what I got from the E-L equation!

print("=" * 60)
print("INDEPENDENT VERIFICATION via biharmonic equation")
print("=" * 60)
print()
print("Linearized Stelle: nabla^4 V = 0 near r=0 (massless limit)")
print("V = -m/r ~ a r^{n-1}")
print("nabla^2(r^p) = p(p+1) r^{p-2}")
print("nabla^4(r^{n-1}) = (n-1)n(n-3)(n-2) r^{n-5}")
print("Indicial polynomial: n(n-1)(n-2)(n-3) = 0")
print("Roots: n = 0, 1, 2, 3")
print()
print("This AGREES with the E-L derivation.")
print("The analytical claim of n = {2, 3, 3-sqrt(6), 3+sqrt(6)} appears INCORRECT")
print("for the massless (near-origin) limit.")

# But wait - there could be a subtlety. The analytical result might come from
# the MASSIVE equation (nabla^2)(nabla^2 - m_2^2) V = 0, where the
# m_2^2 term is NOT negligible. But near r=0, the nabla^2 ~ 1/r^2 terms
# dominate over the m_2^2 term, so the massless limit should be correct
# for the Frobenius analysis.

# Unless the equation is NOT (nabla^2)(nabla^2 - m_2^2) V = 0 but rather
# involves the TENSOR Laplacian which has a different radial operator.
# The tensor Laplacian for the spin-2 mode has:
# D_2 = d^2/dr^2 + (2/r) d/dr - 6/r^2  (l=2 effective centrifugal term)
# instead of the scalar Laplacian nabla^2 = d^2/dr^2 + (2/r) d/dr.
#
# If the equation were D_2 * nabla^2 V = 0 (spin-2 then scalar), the
# indicial equation would be different!

print("\n" + "=" * 60)
print("Testing: D_2 * nabla^2 V = 0 (spin-2 x scalar Laplacian)")
print("=" * 60)
p = Symbol('p')
# nabla^2(r^p) = p(p+1) r^{p-2}
# D_2(r^q) = q(q-1) + 2q - 6 = q^2 + q - 6 = (q+3)(q-2)  [times r^{q-2}]
# After nabla^2: q = n - 1 - 2 = n - 3
# D_2(nabla^2 V) = D_2[n(n-1)(n-1-1+1)... no wait.
# V ~ r^{n-1}
# nabla^2 V ~ (n-1)n r^{n-3}
# D_2(r^{n-3}) = [(n-3)(n-4) + 2(n-3) - 6] r^{n-5} = [(n-3)(n-2) - 6] r^{n-5}
#             = [n^2 - 5n + 6 - 6] r^{n-5} = [n^2 - 5n] r^{n-5} = n(n-5) r^{n-5}

# Full: (n-1)n * n(n-5) = n^2 (n-1)(n-5)
# Roots: n = 0 (double), 1, 5

combo1 = n*(n-1) * ((nn-3)*(nn-2) - 6)
combo1_sub = combo1.subs(nn, n)
# Actually let me be cleaner:
q = nn - 3  # exponent of r after nabla^2
D2_coeff = q*(q-1) + 2*q - 6  # = (q+3)(q-2) = q^2+q-6
D2_expanded = expand(D2_coeff.subs(q, nn-3))
scalar_laplacian = (nn-1)*nn
full_D2_nabla2 = expand(scalar_laplacian * D2_expanded)
print(f"D_2 * nabla^2 indicial: {full_D2_nabla2}")
print(f"Factored: {factor(full_D2_nabla2)}")
print(f"Roots: {solve(full_D2_nabla2, nn)}")
print()

# What about nabla^2 * D_2 (reverse order)?
# D_2(V) where V ~ r^{n-1}:
# D_2(r^{n-1}) = [(n-1)(n-2) + 2(n-1) - 6] r^{n-3} = [(n-1)n - 6] r^{n-3}
# = [n^2 - n - 6] r^{n-3} = (n-3)(n+2) r^{n-3}
# nabla^2(r^{n-3}) = (n-3)(n-2) r^{n-5}
# Full: (n-3)(n+2)(n-3)(n-2) = (n-3)^2 (n+2)(n-2)
# Roots: n = 3 (double), -2, 2

D2_on_V = expand(((nn-1)*(nn-2) + 2*(nn-1) - 6))
print(f"D_2(r^{{n-1}}) coeff: {D2_on_V} = {factor(D2_on_V)}")
nabla2_after = (nn-3)*(nn-2)
full_nabla2_D2 = expand(D2_on_V * nabla2_after)
print(f"nabla^2 * D_2 indicial: {full_nabla2_D2}")
print(f"Factored: {factor(full_nabla2_D2)}")
print(f"Roots: {solve(full_nabla2_D2, nn)}")
print()

# Hmm, none of these give the analytical result s^2 = 6.
# Let me check: what equation gives (n-3)^2 - 6 = 0?

# If s = n-3 and s^2 = 6, then n = 3 +/- sqrt(6).
# The polynomial (n-3)^2 - 6 = n^2 - 6n + 3.
# What radial operator has indicial polynomial n^2 - 6n + 3?
# For phi ~ r^n: a*n^2 + b*n + c = 0 with a=1, b=-6, c=3
# This comes from: r^2 phi'' + (1-2*3+1)*r phi' + 3 phi = 0
# i.e., r^2 phi'' - 4r phi' + 3 phi = 0
# Frobenius: n(n-1) - 4n + 3 = n^2 - 5n + 3... that gives n^2 - 5n + 3, not n^2 - 6n + 3.
#
# Try: n(n-1) + b*n + c = n^2 + (b-1)n + c = n^2 - 6n + 3
# => b-1 = -6, c = 3 => b = -5, c = 3
# Equation: r^2 phi'' - 5r phi' + 3 phi = 0
# Hmm, this doesn't correspond to any standard radial equation.

# CONCLUSION: The analytical result appears to be incorrect.
# The correct indicial equation for the 4th-order Stelle equation
# in the linearized regime near r=0 is:
#   n(n-1)(n-2)(n-3) = 0
# giving n = 0, 1, 2, 3.
#
# This agrees with the biharmonic equation nabla^4 V = 0 and with
# the E-L equation from the C^2 action.
#
# The claimed roots n = 3 +/- sqrt(6) do NOT appear.
# They would require a different equation (involving the tensor
# Laplacian in a specific way that breaks the scalar factorization).

print("=" * 60)
print("FINAL CONCLUSION")
print("=" * 60)
print()
print("The CORRECT indicial equation for the 4th-order Stelle equation")
print("near r = 0 (linearized, massless limit) is:")
print()
print("  n(n-1)(n-2)(n-3) = 0")
print()
print("giving roots n = {0, 1, 2, 3}.")
print()
print("This is derived from TWO independent methods:")
print("  1. E-L equation from the Weyl^2 1D action")
print("  2. Biharmonic equation nabla^4 V = 0 for the potential")
print()
print("The analytical claim of (n-2)(n-3)[(n-3)^2-6] = 0 with roots")
print("n = {2, 3, 3-sqrt(6), 3+sqrt(6)} is INCORRECT.")
print()
print("The error likely comes from confusing the SCALAR Laplacian")
print("with the spin-2 tensor Laplacian, or from an incorrect")
print("computation of the Bach tensor in spherical coordinates.")

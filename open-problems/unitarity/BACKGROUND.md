# Unitarity: Background Briefing

The unitarity problem in SCT arises because the dressed graviton
propagator has ghost poles -- zeros of the propagator denominator
with negative residues. The central question is whether a consistent
prescription exists that removes ghost degrees of freedom from the
physical spectrum while preserving unitarity of the S-matrix.

## The SCT propagator

The spin-2 (transverse-traceless) sector of the one-loop dressed
graviton propagator is

  G_TT(k^2) = 1 / [k^2 * Pi_TT(z)]

where z = k^2 / Lambda^2 and

  Pi_TT(z) = 1 + (13/60) * z * F_hat_1(z).

Here F_hat_1(z) is the normalized Weyl form factor, which is an
entire function of z (no branch cuts). Its local limit is
F_hat_1(0) = 1/(16 pi^2). The function Pi_TT is itself entire,
meaning the propagator has only isolated poles -- no continuum
spectral function. The spectral density rho_TT is a sum of delta
functions.

## Ghost catalogue

Pi_TT(z) has 8 zeros for |z| <= 100 (verified numerically to
50-digit precision):

  - z_0 = +2.4148  (Type A, Euclidean/spacelike, R_0 = -0.4931)
  - z_L = -1.2807  (Type B, Lorentzian/timelike, R_L = -0.5378)
  - 3 complex conjugate Lee-Wick pairs at |z| ~ 34, 59, 85
    with |R_n| < 0.01

The physical ghost is z_L: it sits at timelike momentum (k^2 > 0)
and has a negative residue (R_L < 0), meaning it would contribute
negative-norm states if propagated with the standard Feynman
prescription.

## UV asymptotics

Pi_TT(z) -> -83/6 as |z| -> infinity (verified to 120 digits).
This means the dressed propagator falls as G ~ -6/(83 k^2) in the
UV -- the same 1/k^2 behavior as in GR (not 1/k^4 as in Stelle
theory). The entire-function dressing modifies the overall sign and
coefficient but does not improve power-counting renormalizability.

## The fakeon prescription

The primary resolution strategy is the fakeon (fake particle)
prescription of Anselmi (2017-2018). The idea: replace the Feynman
prescription i*epsilon at ghost poles with a purely virtual
prescription that gives zero imaginary part:

  Im[G_FK(z)] = 0 at z = z_L, z_0.

This removes ghost pair production from the physical S-matrix.
For polynomial (Stelle-type) propagators with a finite number of
poles, the prescription is rigorously proven to give unitary
amplitudes. The extension to the SCT propagator with infinitely
many poles has strong supporting evidence but lacks a complete
all-orders proof.

## What has been proven

### CL (Commutativity of Limits) -- CERTIFIED
The limit of the N-pole fakeon as N -> infinity commutes with
the fakeon prescription. Proven via Weierstrass M-test (method A)
and dominated convergence + Cauchy sequence analysis (method B).
Two-pole dominance: 99.68% of amplitude from z_L and z_0. Smooth
corrections contribute 0.32% (sum of absolute Weierstrass bounds
= 5.002e-4). Tested to 80-digit precision, 15+ spot checks.

### GZ (Constancy of Entire Part) -- CERTIFIED
The entire part g_A(z) in the Mittag-Leffler expansion of
1/(z * Pi_TT(z)) is the constant -13/60 = -c_2 = -2 alpha_C.
Proven via Hadamard factorization: Pi_TT has order rho = 1 and
genus p = 1 (Sigma 1/|z_n| diverges, Sigma 1/|z_n|^2 converges),
so g_A is polynomial of degree <= 1. The linear term vanishes
because 1/(z Pi_TT) -> 0 as |z| -> infinity. Sum rule:
Sigma R_n / z_n = 13/60 (verified to 99.95%).

### KK (Kubo-Kugo Resolution) -- CERTIFIED
The Kugo-Ojima (KO) quartet mechanism does not apply to the SCT
ghost: quantum number mismatch (spin-2, ghost number 0, bosonic)
versus the KO requirement (spin-1, ghost number +1, fermionic).
The fakeon prescription is the primary resolution. PV cancellation
gives Im[G_FK] = 0 at ghost poles, verified to 84-digit precision,
64 independent tests.

### OT (Optical Theorem) -- CERTIFIED CONDITIONAL
One-loop optical theorem Im[T(s)] >= 0 verified at all test points.
Spectral positivity theorem: Im[G_dressed(s)] = Im[Sigma_matter(s)]
/ |denom|^2 > 0. Fakeon removes ghost cut. Central charge
C_m = 283/120 with N_eff = 143.5 (algebraic identity verified).

### CHIRAL-Q -- D^2-quantization UV finiteness
The chirality identity {D, gamma_5} = 0 forces all perturbative
counterterms to be block-diagonal, giving D = 0 at every loop
order within D^2-quantization. This is proven unconditionally at
L = 1 and L = 2. At all orders, it is conditional on two
BV axioms (BV-3 and BV-4) that have been verified at one loop.

## Key open structural issues

- The fakeon prescription is proven for polynomial propagators only.
  Extension to the infinite-pole (entire-function) case is supported
  by CL and GZ but not rigorously established at all orders.
- The KK resolution is one-loop. All-orders operator-formalism proof
  of ghost decoupling is absent.
- Loop-level Kramers-Kronig (dispersion) relations have not been
  verified beyond tree level.
- IVP well-posedness for entire-function propagators is open
  (Anselmi-Calcagni 2025 covers polynomial case only).
- D^2-quantization and metric quantization may differ at loop level,
  so CHIRAL-Q results may define a distinct theory.

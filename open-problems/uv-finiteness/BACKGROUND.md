# UV-Finiteness: Background Briefing

The UV-finiteness question in SCT asks whether the perturbative
loop expansion produces finite (divergence-free) results at every
loop order, or whether uncancelled divergences appear at some
finite loop order L_0, beyond which SCT must be treated as an
effective field theory requiring UV completion.

## What has been proven

### One-loop (L = 1): D = 0 PROVEN
The one-loop effective action is finite. This follows from the
heat-kernel computation: the one-loop divergence is proportional
to the Seeley-DeWitt a_4 coefficient, which gives alpha_C C^2
+ alpha_R R^2 with alpha_C = 13/120 and alpha_R = 2(xi-1/6)^2.
These are not divergences but finite nonlocal form factors that
define the one-loop effective action. There is no residual
divergence because the spectral action already contains R^2 and
C^2 terms whose coefficients can absorb any counterterm by
adjusting the spectral function psi. The MR-7 graviton scattering
computation confirmed D = 0 at one loop using the background field
method.

### Two-loop (L = 2): D = 0 PROVEN (UNCONDITIONAL)
The two-loop divergence is governed by the a_6 Seeley-DeWitt
coefficient, which contains 8 independent dimension-6 curvature
invariants (including the Goroff-Sagnotti R_{abcd} R^{cdef} R_{ef}^{ab}
term). Two key results:

1. **On-shell reduction.** Using the equations of motion
   R_{mu nu} = O(alpha_C), the 8 invariants reduce to cubic
   curvature contractions (CCC) involving only the Weyl tensor C.
   In d = 4, there are 2 parity-even cubic Weyl invariants, but
   the parity-odd one (J_1) vanishes identically, leaving 1
   effective invariant (the cubic Weyl contraction CCC).

2. **Absorption by delta(psi).** The spectral function deformation
   delta(psi)(u) = c_2 (2 - 4u + u^2) e^{-u} shifts f_6 while
   preserving f_2 (Einstein-Hilbert) and f_4 (one-loop). The single
   coefficient c_2 can absorb the single CCC counterterm.

The SM CCC coefficient is -1481/6. The deformation ratio |R| = 0.00469
(finite, nonzero), confirming absorbability. This result is the content
of MR-5b (CERTIFIED).

Within the D^2-quantization framework, the CHIRAL-Q chirality theorem
provides an independent proof: the chirality identity {D, gamma_5} = 0
forces all counterterms to be block-diagonal, which guarantees D = 0
at L = 1 and L = 2 without reference to specific coefficient values
(Theorem 6.12, UNCONDITIONAL).

### Three-loop (L = 3): D != 0 (OVERDETERMINED)
The three-loop counterterm involves the a_8 Seeley-DeWitt coefficient,
which contains quartic curvature invariants. The full SO(4) Molien
series gives P(4) = 3 independent parity-even quartic Weyl invariants:

  K_1 = (C^2)^2 = (C_{abcd} C^{abcd})^2
  K_2 = C^4_{box} = C_{abcd} C^{cdef} C_{efgh} C^{ghab}
  K_3 = (*CC)^2 = (C_{abcd} *C^{abcd})^2

where *C is the dual Weyl tensor. The Cayley-Hamilton identity in d = 4
relates these: C^4_{chain} = (1/4)[(C^2)^2 + (*CC)^2], reducing the
independent set from 3 to 2. But the spectral function at L = 3
provides only 1 parameter (delta f_8). This gives a 2:1
overdetermination:

  2 independent counterterms, 1 adjustable parameter.

No structural reduction from 2 to 1 has been found. The FUND program
(5 fundamental investigations: FRG, FK3, NCG, SYM, LAT) systematically
explored all known escape routes. All closed.

## The CHIRAL-Q chirality theorem

Within D^2-quantization, the chirality identity gives:

  Tr(Omega^4)_{chain} ratio (C^2)^2 : (*CC)^2 = 1:1 (exact).

However, the multi-loop counterterm is not Tr(Omega^4) alone. It
includes [Tr(Omega^2)]^2, which has pq cross-terms between the
self-dual (C^+) and anti-self-dual (C^-) sectors. The full a_8 is
therefore NOT proportional to a single combination of K_1, K_2, K_3.

Spectral renormalizability was investigated as a potential escape: if
the graviton h_{mu nu} decomposed chirally, the counterterms might
respect chiral selection rules that reduce 2 -> 1. However, h_{mu nu}
transforms as (3,3) under SU(2)_L x SU(2)_R (irreducible, no chiral
decomposition). The dressed propagator Pi_TT is a scalar (chirality-
blind) and connects delta C^+ to delta C^-. Spectral renormalizability
is therefore DISPROVEN.

## Current status

SCT is a well-structured effective framework with D = 0 at L = 1 and
L = 2. At L >= 3, the theory is structurally unable to absorb all
divergences with the single spectral function psi. The practical
suppression is enormous: the three-loop correction is of order
alpha_C^3 / (16 pi^2)^3 = 3.2e-10, making it utterly negligible for
any foreseeable observation. But strict UV-completeness (D = 0 at all
loop orders) is NOT achieved by any known mechanism.

## Surviving escape routes

1. **Hidden structural principle (15-25% probability):** An unknown
   algebraic identity in spectral geometry that reduces the number of
   independent quartic Weyl counterterms from 2 to 1.
2. **Asymptotic safety fixed point (poorly constrained):** A non-
   perturbative FRG flow that resolves the perturbative divergence.
3. **Modular forms (5-15%):** The Bianchi IX sector has modular
   symmetry (Fan-Fathizadeh-Marcolli). If this extends to the full
   theory, it might provide additional constraints.
4. **SCT as EFT (~50%):** Accept that SCT is an effective framework
   through L = 2 and seek UV completion from a deeper structure
   (e.g., a non-perturbative formulation via Postulate 5).

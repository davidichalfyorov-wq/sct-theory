# Spectral Dimension: Background Briefing

The spectral dimension d_S characterizes the effective dimensionality
of spacetime as probed by a diffusion process at scale sigma. It is
defined via the return probability P(sigma) of a random walk:

  d_S(sigma) = -2 d(ln P) / d(ln sigma).

In standard GR on a d-dimensional manifold, d_S = d at all scales.
Several quantum gravity approaches predict d_S -> 2 in the UV
(asymptotic safety, causal dynamical triangulations, Horava-Lifshitz
gravity, spin foams). This "universal" d_S = 2 prediction has been
called a "common feature of quantum gravity."

## SCT is the exception

SCT is the FIRST quantum gravity approach that does NOT predict a
universal d_S = 2 in the UV. The spectral dimension in SCT is
DEFINITION-DEPENDENT: different physically motivated definitions
give different UV values.

## Four definitions and their predictions

| Definition | UV value | IR value | Reference |
|------------|----------|----------|-----------|
| Propagator scaling (CMN) | d_S = 2 | d_S = 2 | Calcagni-Modesto-Nardelli |
| Heat kernel (standard) | d_S = 4 | d_S = 4 | Padmanabhan, Reuter |
| ASZ/fakeon (ghost-projected) | d_S = 0 -> 4 | d_S = 4 | Anselmi-Stelle-Zhang |
| Mittag-Leffler (ML) | d_S ~ 2 at sigma* | d_S = 4 | SCT computation (NT-3) |

The CMN definition uses the UV scaling of the dressed propagator
G_TT(k^2) ~ 1/k^{2 alpha}. For SCT, Pi_TT(z) ~ z at large z, so
G_TT ~ 1/(k^2 * k^2) = 1/k^4, giving alpha = 2 and d_S = 4/2 = 2.
However, this ignores the ghost poles.

The heat kernel definition uses P(sigma) = Tr(exp(-sigma D^2)), which
for a 4-manifold gives d_S = 4 at all scales if D^2 is a standard
Laplace-type operator. The SCT modifications to D^2 are subleading.

The ML definition is the most detailed. It uses the Mittag-Leffler
partial fraction expansion of the dressed propagator to compute the
return probability exactly.

## Ghost-induced P < 0 problem

The ML return probability is

  P_ML(sigma) = W(sigma) / (16 pi^2 sigma^2)

where W(sigma) = 1 + sum_n R_n exp(-z_n sigma Lambda^2). The sum runs
over the 8 poles of Pi_TT, with residues R_n. The sum of residues is

  sum R_n = -1.034    (100-digit verified)

so W(0) = 1 + (-1.034) = -0.034 < 0. This means P_ML becomes negative
for sigma < sigma* ~ 0.01 Lambda^{-2}.

A negative return probability is unphysical. The P < 0 regime is
a manifestation of the ghost problem: ghost poles contribute with
negative residues, and in the deep UV (sigma -> 0) the ghost
contributions dominate.

## Physical ML region

For sigma > sigma* (the physical region), P_ML > 0 and the spectral
dimension flows smoothly from d_S ~ 2 near sigma* to d_S = 4 in the
IR. The flow is:

| sigma / Lambda^{-2} | d_S |
|----------------------|-----|
| 0.05 | 2.47 |
| 0.10 | 2.06 |
| 0.50 | 2.78 |
| 1.0  | 3.26 |
| 5.0  | 3.99 |
| 10.0 | 4.00 |

## Verification status

NT-3 is COMPLETE with 32/32 checks across 5 verification layers.
The result is classified as DEFINITION-DEPENDENT: the theory is
internally consistent, but the physical prediction for d_S depends
on which definition is adopted, and this choice requires input from
the ghost resolution (OP-07, MR-2).

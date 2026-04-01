---
id: OP-11
title: "IVP well-posedness for entire-function propagators"
domain: [unitarity, causality]
difficulty: hard
status: open
deep-research-tier: B
blocks: []
blocked-by: []
roadmap-tasks: [MR-3]
papers: ["2501.04097", "1811.10619"]
date-opened: 2026-03-31
date-updated: 2026-03-31
---

# OP-11: IVP well-posedness for entire-function propagators

## 1. Statement

Prove that the initial value problem (IVP) for the linearised SCT
field equations with entire-function propagator dressing
Pi_TT(z) = 1 + (13/60) z F_hat_1(z) is well-posed in the sense of
Hadamard: existence, uniqueness, and continuous dependence on initial
data for compactly supported Cauchy data on a spacelike hypersurface.

## 2. Context

The SCT linearised field equations around flat space are

  Pi_TT(Box / Lambda^2) h_{mu nu}^{TT} = kappa T_{mu nu}^{TT}

where Pi_TT is an entire function of Box. Because Pi_TT has
infinitely many zeros, this is an infinite-derivative differential
equation. Standard PDE existence theorems (Cauchy-Kowalewski,
Leray) do not apply directly to infinite-derivative equations.

Anselmi and Calcagni (2025, arXiv:2501.04097) proved well-posedness
for polynomial propagators of arbitrary degree: the fakeon
prescription converts the higher-derivative system into a system
of coupled second-order equations (classicization), one for each
pole, and standard hyperbolic PDE theory applies. Their proof
covers Stelle gravity and polynomial extensions.

For SCT, the propagator is not polynomial: Pi_TT is entire of
order 1, and the classicization procedure would produce infinitely
many coupled oscillators. The convergence of this infinite system
and the well-posedness of the resulting IVP have not been established.

The MR-3 investigation classified SCT as "CONDITIONAL" for
causality, with sub-task (d) (IVP well-posedness) as the remaining
open condition.

## 3. Known Results

- **Front velocity v_front = c.** Proven via the Paley-Wiener theorem
  for the retarded Green's function. Because Pi_TT is entire of
  order 1, the inverse Fourier transform of 1/(k^2 Pi_TT) has
  support inside the forward light cone (up to exponentially
  decaying tails). The front velocity equals c exactly.

- **Acausal decay.** The retarded Green's function has exponentially
  decaying tails outside the light cone with characteristic length
  l_a = 0.884 / Lambda = 1 / m_ghost where m_ghost = 1.132 Lambda.
  These tails arise from the ghost pole and are an inherent feature
  of the fakeon prescription.

- **Anomalous dispersion.** Near the ghost pole (z ~ z_L), the
  group velocity |v_g| ~ 5.4 (superluminal). This is a resonance
  phenomenon analogous to anomalous dispersion in optics, not a
  signal velocity violation. The signal velocity = front velocity = c.

- **Polynomial IVP (Anselmi-Calcagni 2025).** For polynomial
  propagators with N poles, classicization gives N coupled second-
  order ODEs per spatial mode. The system is hyperbolic (by
  construction of the fakeon prescription), and Hadamard well-
  posedness follows from standard energy estimates.

- **Classicization for infinite poles.** The Mittag-Leffler expansion
  of 1/(z Pi_TT) gives an infinite partial fraction decomposition.
  Classicization would convert this into infinitely many coupled
  oscillators. The CL result (Weierstrass bound 5.002e-4) shows the
  coupling to higher modes converges, but this is an L^1 bound on
  the coupling coefficients, not an energy estimate for the PDE system.

## 4. Failed Approaches

1. **Direct Cauchy-Kowalewski extension.** The Cauchy-Kowalewski
   theorem requires analyticity of the coefficients and applies to
   finite-order PDEs. For infinite-order equations, the theorem does
   not apply because the analytic continuation of the initial data
   to complex time is needed for a strip whose width depends on the
   order (and goes to zero for infinite order).

2. **Truncation and limit.** Truncate Pi_TT to the first N poles,
   prove well-posedness for the N-pole system using Anselmi-Calcagni,
   then take N -> infinity. The difficulty is that the energy estimates
   for the N-pole system have constants that may grow with N. Without
   uniform-in-N energy bounds, the limit is not controlled.

3. **Fourier multiplier approach.** Write the equation as
   Pi_TT(-|xi|^2/Lambda^2) h_hat(xi,t) = source in Fourier space.
   The equation is well-posed mode by mode (each mode is an ODE in
   time). The difficulty is that Pi_TT has zeros at specific |xi|^2,
   and the inversion 1/Pi_TT is singular at these points. The fakeon
   prescription gives a distributional (principal-value) solution at
   these momenta, and the regularity of the resulting distribution
   in physical space is not established.

## 5. Success Criteria

- A proof of existence and uniqueness of distributional solutions to
  the linearised SCT field equations for compactly supported smooth
  Cauchy data on Sigma_0 = {t = 0}.
- Continuous dependence on initial data in an appropriate Sobolev
  space H^s(Sigma_0) -> H^{s-r}(M), with explicit s and r.
- Or: a specific counterexample (initial data for which the solution
  does not exist, is not unique, or does not depend continuously on
  the data).

## 6. Suggested Directions

1. **Infinite-dimensional classicization with energy estimates.**
   Write the classicized system as an infinite system of coupled
   oscillators {phi_n(x,t), n = 0, 1, 2, ...} with coupling matrix
   A_{nm}. Use the CL Weierstrass bounds to show that A is trace-
   class or Hilbert-Schmidt. Then the infinite system has well-posed
   dynamics by the Hille-Yosida theorem on the Hilbert space
   l^2 tensor H^s.

2. **Entire-function pseudodifferential calculus.** Develop a
   pseudodifferential operator calculus for entire-function symbols
   of order 1 (the order of Pi_TT). The key estimate is the mapping
   property: Pi_TT(D) : H^s -> H^{s-r} for some r depending on the
   order. Invertibility then follows from the positivity of Pi_TT on
   the real axis (Pi_TT(xi^2) > 0 for xi^2 > 0, at least outside
   the ghost poles).

3. **Distributional IVP via Hormander's theory.** Use Hormander's
   theory of linear PDE with constant coefficients (adapted to the
   infinite-order case). The relevant condition is that Pi_TT is
   hypoelliptic (which it is, being entire with controlled growth).
   Fundamental solutions for hypoelliptic infinite-order operators
   have been studied by Ehrenpreis and Malgrange.

4. **Numerical well-posedness test.** Solve the linearised equations
   numerically on a 1+1 dimensional reduction, with initial data
   that excites the ghost mode. Monitor the solution for growth or
   blow-up. A controlled numerical test would not constitute a proof
   but would provide strong evidence for or against well-posedness.

## 7. References

1. Anselmi, D. and Calcagni, G. "Fakeons, classicization and the
   initial value problem for polynomial theories," arXiv:2501.04097.
2. Anselmi, D. and Piva, M. "Quantum gravity, fakeons and
   microcausality," JHEP 11 (2018) 021, arXiv:1806.03605.
3. Barnaby, N. and Kamran, N. "Dynamics with infinitely many
   derivatives: the initial value problem," JHEP 02 (2008) 008,
   arXiv:0709.3968.
4. Calcagni, G., Modesto, L. and Nardelli, G. "Initial conditions
   and degrees of freedom of non-local gravity," JHEP 05 (2018) 087,
   arXiv:1803.00561.
5. Buoninfante, L. "Classical properties of non-local, ghost- and
   singularity-free gravity," arXiv:1811.10619.

## 8. Connections

- **Roadmap: MR-3** (causality). Resolving OP-11 would upgrade the
  MR-3 sub-task (d) from CONDITIONAL to COMPLETE, changing the MR-3
  overall verdict from CONDITIONAL to UNCONDITIONAL.
- **Related to OP-07** (fakeon for infinite poles): OP-11 is the
  classical analogue of OP-07. If the classical IVP is well-posed,
  this provides a classical foundation for the quantum fakeon
  prescription.
- **Related to OP-12** (loop-level KK): Dispersion relations require
  the retarded Green's function to be well-defined, which in turn
  requires IVP well-posedness.
- **Independent of OP-13** (three-loop) and all UV-finiteness
  problems: IVP well-posedness is a classical question about the
  linearised equations and does not depend on the loop expansion.

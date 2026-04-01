---
id: OP-05
title: "Recovery of QFT on curved spacetime"
domain: [theory]
difficulty: hard
status: open
deep-research-tier: D
blocks: []
blocked-by: [OP-02]
roadmap-tasks: [FUND-1]
papers: ["hep-th/0608221", "gr-qc/0405057"]
date-opened: 2026-03-31
date-updated: 2026-03-31
---

# OP-05: Recovery of QFT on curved spacetime

## 1. Statement

Demonstrate that SCT reduces to standard quantum field theory on
curved spacetime in the semiclassical limit, where the gravitational
background is fixed and curvature is much smaller than the spectral
cutoff (R << Lambda^2). Specifically:

(a) Verify that the matter sector satisfies the Haag-Kastler axioms
    (isotony, locality, covariance) to the extent expected of algebraic
    QFT on curved spacetime.
(b) Reproduce the conformal anomaly with the correct central charges
    for the Standard Model matter content:
    <T^mu_mu> = a E_4 + c C^2 + b Box R.
(c) Resolve the fermion doubling problem: the NCG approach naturally
    produces 2x the physical fermion degrees of freedom. State the
    resolution explicitly and verify it is compatible with the
    Lorentzian spectral action.

## 2. Context

The heat-kernel expansion of the spectral action reproduces the
classical Standard Model Lagrangian coupled to Einstein gravity. This
is a well-established result in noncommutative geometry (Chamseddine-
Connes 1996, Chamseddine-Connes-Marcolli 2007). However, the match is
at the classical level: the spectral action gives the correct tree-level
Lagrangian, but it has not been shown that the full quantum theory on a
curved background reproduces standard renormalized QFT.

The nontrivial step is showing that the quantized matter sector,
defined through the spectral triple framework, satisfies the standard
axioms of QFT on curved spacetime (Brunetti-Fredenhagen-Verch, Hollands-
Wald). This requires verifying that the Hadamard condition, microlocal
spectrum condition, and renormalization of composite operators all go
through in the spectral framework.

Barrett (2007) showed that the finite spectral triple for the SM admits
a KO-dimension 6 real structure that avoids fermion doubling, producing
H_F = C^96 per generation (not doubled). The Lorentzian spectral action
(MR-1) uses this structure, but the compatibility with full quantum
renormalization has not been checked.

## 3. Known Results

- The heat-kernel expansion reproduces the classical SM Lagrangian:
  Yang-Mills + Higgs + Yukawa + Einstein-Hilbert + cosmological
  constant. This is exact at the level of a_0, a_2, a_4 Seeley-DeWitt
  coefficients (Chamseddine-Connes-Marcolli 2007).

- The one-loop form factors (NT-1, NT-1b) give the correct beta
  functions: beta_W = {1/120, 1/20, 1/10} for spin {0, 1/2, 1},
  matching the standard Barvinsky-Vilkovisky results.

- The conformal anomaly central charges for the SM content are
  a = 283/120, c = 199/40 (from the Seeley-DeWitt a_4 coefficient).
  These match the standard QFT calculation.

- Barrett's KO-dimension 6 construction (hep-th/0608221, Thm 4.1)
  resolves fermion doubling by using the real structure J to impose a
  Majorana-type condition. The fermionic action S_f = [J Psi, D_A Psi]
  reproduces the full SM fermionic Lagrangian without doubling.

- The Lorentzian formulation (MR-1) inherits the form factors via
  analytic continuation. Wick rotation is performed at the level of
  spectral invariants, not the metric, bypassing the standard
  Lorentzian difficulties.

- Paschke-Verch (2004) formulated local covariant QFT over spectral
  geometries, establishing a categorical framework. The detailed
  verification of axioms remains open.

## 4. Failed Approaches

1. **Direct quantization of the spectral action.** Attempted to
   quantize Tr(f(D^2/Lambda^2)) as a fundamental action. The problem
   is that D is the dynamical variable, not a field on spacetime. The
   standard canonical quantization and path integral formulations
   require a spacetime field, and the translation from spectral data
   to field-theoretic language introduces ambiguities.

2. **Lattice regularization via finite spectral triples.** Attempted
   to use the matrix geometry approximation as a UV regulator for
   QFT. The spectral triple axioms constrain the matrix entries of D_N,
   making the regularization non-standard. The resulting theory does not
   reduce to lattice QFT in any known limit, and the continuum limit
   is not controlled.

3. **Algebraic deformation quantization.** Attempted to deform the
   algebra A of the spectral triple using star products. This produces
   a noncommutative field theory, but the deformation parameter is
   unrelated to Planck's constant, and the connection to standard
   quantization is unclear.

## 5. Success Criteria

- Proof that the matter algebra in SCT satisfies the Haag-Kastler
  axioms (isotony, locality, covariance) in the regime R << Lambda^2.
- Computation of the renormalized stress-energy tensor
  <T_mu_nu>_ren from the spectral framework and verification that
  the trace anomaly gives the correct SM central charges.
- Explicit statement of the fermion doubling resolution used in SCT,
  proof of its compatibility with the Lorentzian spectral action,
  and verification that it does not introduce gauge anomalies or
  modify the SM quantum numbers.
- If the resolution changes the fermion content, reassessment of the
  form factors (NT-1, NT-1b) and the total coefficient alpha_C = 13/120.

## 6. Suggested Directions

1. **Semiclassical expansion.** Fix a classical background geometry
   (g_0, D_0) and expand D = D_0 + delta D. Quantize delta D as a
   perturbation. Show that the resulting perturbation theory reproduces
   the standard Feynman rules for matter fields on curved spacetime.

2. **Hadamard condition from spectral data.** The two-point function
   of a quantum field on curved spacetime must satisfy the Hadamard
   condition (correct UV singularity structure). Show that the spectral
   action prescription produces Hadamard states in the semiclassical
   limit.

3. **Algebraic QFT verification.** Use the Paschke-Verch categorical
   framework to verify isotony and locality. The covariance functor
   maps spectral triples to C*-algebras; check that this functor
   satisfies the Brunetti-Fredenhagen-Verch axioms.

4. **Anomaly computation via index theory.** The conformal anomaly is
   related to the index of D via the Atiyah-Singer theorem. Verify
   that the spectral triple index reproduces the standard anomaly
   coefficients directly, without going through the heat-kernel
   expansion.

5. **Barrett's fermion doubling in the Lorentzian setting.** Extend
   Barrett's KO-dimension analysis to the Lorentzian spectral triple
   used in MR-1. Check that the Krein space inner product, the real
   structure J, and the first-order condition are mutually compatible.

## 7. References

1. Barrett, J., "Lorentzian version of the noncommutative geometry
   of the Standard Model of particle physics," J. Math. Phys. 48
   (2007) 012303, arXiv:hep-th/0608221.
2. Chamseddine, Connes, Marcolli, "Gravity and the standard model
   with neutrino mixing," Adv. Theor. Math. Phys. 11 (2007) 991,
   arXiv:hep-th/0610241.
3. Paschke, Verch, "Local covariant quantum field theory over spectral
   geometries," Class. Quant. Grav. 21 (2004) 5299, gr-qc/0405057.
4. Brunetti, Fredenhagen, Verch, "The generally covariant locality
   principle -- a new paradigm for local quantum field theory," Comm.
   Math. Phys. 237 (2003) 31, math-ph/0112041.
5. Hollands, Wald, "Local Wick polynomials and time ordered products
   of quantum fields in curved spacetime," Comm. Math. Phys. 223
   (2001) 289, gr-qc/0103074.
6. Connes, "Noncommutative geometry and the standard model with
   neutrino mixing," JHEP 11 (2006) 081, hep-th/0608226.

## 8. Connections

- **Blocked by OP-02** (Postulate 5): full QFT recovery requires a
  specified dynamical principle to define the quantum theory.
- Related to **OP-03** (non-perturbative definition): QFT recovery is
  a necessary consistency check for any non-perturbative formulation.
- Related to **OP-04** (cutoff function): the conformal anomaly
  coefficients are moment-independent (universal), providing a
  cutoff-independent test.
- The one-loop form factors (NT-1, NT-1b) already pass the beta
  function matching test, which is a perturbative version of QFT
  recovery. The full problem requires going beyond perturbation theory.

---
id: OP-12
title: "Loop-level Kramers-Kronig dispersion relation verification"
domain: [unitarity, causality]
difficulty: hard
status: open
deep-research-tier: B
blocks: []
blocked-by: [MR-4]
roadmap-tasks: [MR-3]
papers: ["1809.02555"]
date-opened: 2026-03-31
date-updated: 2026-03-31
---

# OP-12: Loop-level Kramers-Kronig dispersion relation verification

## 1. Statement

Verify that the full (loop-corrected) SCT graviton self-energy
Sigma(p^2) satisfies the Kramers-Kronig (KK) dispersion relation

  Re[Sigma(s)] = (1/pi) P integral_{s_th}^{infty} ds'
                   Im[Sigma(s')] / (s' - s)

at two loops and beyond, where s_th is the lowest physical threshold
and P denotes the Cauchy principal value. This would confirm that the
theory respects microcausality (or its fakeon-modified analogue) at
the quantum level.

## 2. Context

Kramers-Kronig relations are consequences of analyticity and causality
in quantum field theory. They connect the real and imaginary parts of
forward scattering amplitudes and self-energies via dispersion
integrals. A theory satisfying KK relations at all orders is guaranteed
to have causal signal propagation (at least in the sense of the
retarded Green's function being supported in the forward light cone).

At tree level in SCT, the KK relation is trivially satisfied because
Im[Pi_TT(s)] = 0 on the real s-axis: the propagator dressing Pi_TT
is an entire function with no branch cuts. The entire real axis is a
region of analyticity, and the dispersion integral is trivially zero
on both sides.

At one loop, matter loops generate branch cuts in the dressed graviton
self-energy starting at the physical threshold s_th = 4 m^2 (for
massive particles) or s_th = 0 (for massless particles). The one-loop
spectral positivity theorem (OT CERTIFIED CONDITIONAL) establishes
that Im[Sigma_matter(s)] > 0 above threshold, consistent with the
positive spectral function required by KK.

At two loops and beyond, the situation is more complex: graviton
self-energy diagrams include ghost pole contributions (from the z_L
pole circulating in loops), and the fakeon prescription modifies the
absorptive part. The question is whether the KK relation survives this
modification.

## 3. Known Results

- **Tree level: KK trivially satisfied.** Im[Pi_TT] = 0 on real axis.
  No branch cuts, no dispersive structure.

- **One loop: KK consistent.** The one-loop self-energy has
  Im[Sigma] = Im[Sigma_matter] > 0 (spectral positivity theorem).
  The spectral function is a sum of delta functions (from the entire
  Pi_TT) plus the matter continuum (from standard particle loops).
  The fakeon prescription gives Im[G_FK] = 0 at ghost poles, removing
  the ghost contribution to the spectral function.

- **Optical theorem at one loop: PASS.** Im[T(s)] >= 0 verified
  numerically at all tested energies. The imaginary part of the
  forward amplitude equals the total cross section (at one loop).

- **Anselmi KK analysis (1809.02555).** Anselmi showed that for
  polynomial propagators with fakeon prescription, the modified
  dispersion relation involves a subtraction term that accounts for
  the removed ghost cuts. The resulting dispersion relation differs
  from the standard KK but is self-consistent (the subtracted
  dispersion relation is equivalent to the fakeon unitarity equations).

- **SCT front velocity = c.** The Paley-Wiener theorem guarantees
  that the retarded propagator vanishes outside the forward light cone
  (up to exponential tails). This is a necessary condition for KK to
  hold (in modified form).

## 4. Failed Approaches

1. **Standard KK derivation by contour integration.** The standard
   proof of KK uses a contour integral of Sigma(z) / (z - s) along
   a contour enclosing the physical cut. For SCT, the propagator has
   poles at z_n (the zeros of Pi_TT), and the contour integral picks
   up residue contributions from these poles. Under the standard
   Feynman prescription, these residues would contribute additional
   dispersive terms. Under the fakeon prescription, these residues
   are removed (set to zero). The standard proof therefore does not
   apply directly, and a modified proof is needed that accounts for
   the fakeon exclusion.

2. **Sum-rule verification.** Attempted to verify the superconvergence
   sum rule integral Im[Sigma(s)] ds / s = 0 (which follows from KK
   for amplitudes vanishing at infinity). At one loop, the sum rule
   is satisfied because the spectral function is positive and the
   amplitude falls as 1/s at large s (from the GR-like UV behavior).
   At two loops, the spectral function is not known, and the sum rule
   cannot be tested numerically.

3. **Forward dispersion relation from unitarity.** Attempted to derive
   the forward dispersion relation directly from the fakeon unitarity
   equations (rather than from analyticity). The fakeon unitarity
   equations give a modified cutting equation, and integrating over
   phase space should give a dispersion relation. But the integration
   involves the full multi-particle phase space at two loops, which
   has not been computed for SCT.

## 5. Success Criteria

- An explicit computation of the two-loop graviton self-energy
  Im[Sigma^(2)(s)] in SCT, at least in a simplified model (e.g.,
  graviton + scalar field on flat background).
- Verification that the dispersion integral of Im[Sigma^(2)] gives
  back Re[Sigma^(2)] (or the appropriate fakeon-modified relation).
- Or: identification of a specific kinematic region where the KK
  relation is violated, with quantitative assessment of the violation.

## 6. Suggested Directions

1. **Compute two-loop self-energy via TFORM.** Use the TFORM parallel
   algebraic system (8-worker) to evaluate the two-loop graviton
   self-energy diagrams in the background field method. The key
   diagrams are the sunset and double-bubble topologies. Extract
   Im[Sigma^(2)] using the Cutkosky rules modified by the fakeon
   prescription (zero cuts through ghost lines).

2. **Anselmi modified dispersion relation.** Extend Anselmi's
   polynomial analysis to the entire-function case. The modification
   involves subtracting the ghost pole residues from the standard
   dispersion integral. For SCT, this requires summing over infinitely
   many subtracted poles. The CL convergence result (Weierstrass
   bound) provides the necessary estimates for the convergence of
   this infinite subtraction.

3. **Fixed-order KK test on a simplified model.** Study a scalar
   field theory with an SCT-like entire-function propagator:
   Pi(z) = 1 + c z phi(z) where phi is the master function. Compute
   the two-loop self-energy in this scalar model (technically simpler
   than gravity) and test KK. If KK fails in the scalar model, it
   will fail in the gravitational theory as well.

4. **Non-perturbative approach via spectral function.** If the full
   spectral function rho(s) can be reconstructed from the known
   propagator structure (Pi_TT entire, poles at z_n, matter thresholds),
   the KK relation can be verified as an integral equation. The
   spectral function is a sum of delta functions (from Pi_TT poles)
   plus continuum (from matter loops).

## 7. References

1. Anselmi, D. "Diagrammar of physical and fake particles and
   spectral optical theorem," JHEP 11 (2021) 030, arXiv:2109.06889.
2. Anselmi, D. and Piva, M. "Quantum gravity, fakeons and
   microcausality," JHEP 11 (2018) 021, arXiv:1806.03605.
3. Anselmi, D. "Fakeons, microcausality and the classical limit of
   quantum gravity," Class. Quant. Grav. 36 (2019) 065010,
   arXiv:1809.02555.
4. Veltman, M. "Unitarity and causality in a renormalizable field
   theory with unstable particles," Physica 29 (1963) 186.

## 8. Connections

- **Blocked by MR-4** (two-loop effective action): The two-loop
  self-energy computation is a prerequisite for testing KK at L = 2.
  MR-4 has established the counterterm structure but not the finite
  parts needed for Im[Sigma].
- **Roadmap: MR-3** (causality). Resolution would upgrade MR-3
  sub-task (b) from trivial (tree level) to substantive (loop level),
  potentially changing the overall MR-3 verdict from CONDITIONAL to
  UNCONDITIONAL (together with OP-11).
- **Related to OP-07** (fakeon for infinite poles): The KK relation
  is a consequence of unitarity + analyticity. If OP-07 establishes
  unitarity, KK should follow from analyticity of the S-matrix, making
  OP-12 a corollary of OP-07.
- **Related to OP-11** (IVP): KK requires the retarded Green's
  function to be causal, which in turn requires IVP well-posedness.

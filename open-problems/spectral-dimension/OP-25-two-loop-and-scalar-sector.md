---
id: OP-25
title: "Two-loop corrections and scalar sector spectral dimension"
domain: [spectral-dimension, perturbative]
difficulty: medium
status: partial
deep-research-tier: B
blocks: []
blocked-by: []
roadmap-tasks: [NT-3, MR-4]
papers: ["1304.7247", "1809.02944", "1107.2403"]
date-opened: 2026-03-31
date-updated: 2026-04-03
progress: "Two-loop pole shifts quantified (~0.6%). sigma* NOT stabilized. P<0 NOT cured. Im(delta z_L) open. Scalar at xi=1/6 absent. At xi!=1/6 needs explicit F_hat_2."
---

# OP-25: Two-loop corrections and scalar sector

## 1. Statement

Compute the two-loop corrections to the spectral dimension d_S in the
spin-2 (tensor) sector and the one-loop spectral dimension in the
spin-0 (scalar) sector of the SCT propagator. Determine whether:
(a) the two-loop corrections modify the UV saturation of d_S by more
than the estimated 1.2%; (b) the scalar sector Pi_s spectral dimension
differs qualitatively from the tensor sector; (c) the pole truncation
instability (sigma* shifts 67% from 2 to 8 poles) is reduced at two
loops.

## 2. Context

The current spectral dimension analysis (NT-3, 32/32 checks) uses the
one-loop dressed propagator Pi_TT(z) = 1 + (13/60) z F_hat_1(z) and
its Mittag-Leffler expansion with 8 poles. Two independent issues
remain:

**Two-loop tensor sector.** The two-loop correction to Pi_TT has been
computed at the level of the divergent part (MR-4, MR-5b: D = 0, R^3
absorbable). However, the finite (non-divergent) part of the two-loop
propagator has not been extracted. This finite part modifies the
positions and residues of the ghost poles, shifting sigma* and
potentially changing the UV value of d_S.

**Scalar sector.** The scalar propagator dressing Pi_s(z, xi) =
1 + 6(xi - 1/6)^2 z F_hat_2(z, xi) has a different pole structure
from Pi_TT. At conformal coupling xi = 1/6, Pi_s = 1 identically
(scalar mode decouples). At minimal coupling xi = 0, Pi_s has its own
set of poles, which have not been catalogued or analysed for spectral
dimension.

**Truncation instability.** The ML expansion of Pi_TT is truncated to
8 poles (2 real + 3 complex conjugate pairs). Adding more poles
(from higher sheets of the erfi function) shifts sigma* by 67%. This
instability suggests that the current result is not fully converged.
The two-loop correction adds new pole positions and may stabilize or
further destabilize the expansion.

## 3. Known Results

- **One-loop tensor poles (NT-3):** 8 poles catalogued. Real poles at
  z_0 = 2.4148 (Euclidean) and z_L = -1.2807 (Lorentzian ghost).
  Three Lee-Wick pairs. Sum R_n = -1.034.

- **Two-loop D = 0 (MR-5b):** the two-loop divergence is zero on-shell
  (D = 0). The R^3 counterterm is absorbable by field redefinition
  (delta psi). This means the two-loop correction to Pi_TT is finite.

- **Estimated magnitude:** the two-loop correction to Pi_TT at the
  ghost pole z_L is estimated at O(alpha_C^2) ~ (13/120)^2 ~ 0.012,
  i.e., a 1.2% shift in the pole position. This estimate uses naive
  dimensional analysis and has not been verified by explicit computation.

- **Scalar sector at xi = 0:** Pi_s(z, 0) = 1 + z/6 * F_hat_2(z, 0).
  The coefficient 1/6 is much smaller than 13/60 for Pi_TT, so the
  scalar poles are at larger |z| (weaker ghost effect). The scalar
  spectral dimension should be closer to 4 at all scales, but this
  has not been computed.

- **Pole truncation (2 -> 8):** sigma* shifts from 0.006 Lambda^{-2}
  (2 poles) to 0.010 Lambda^{-2} (8 poles), a 67% change. The
  d_S value at sigma = 0.05/Lambda^2 shifts from 2.1 to 2.5.

## 4. Failed Approaches

1. **Perturbative two-loop pole shift.** Attempted to compute the
   shift of z_L at two loops using perturbation theory around the
   one-loop pole. The calculation requires the finite part of the
   two-loop self-energy Sigma_2(k^2) at k^2 = z_L Lambda^2, which
   involves a two-loop Feynman integral in curved-space background.
   The integral has not been evaluated.

2. **Pade approximation for convergence.** Tried to accelerate the
   pole series by constructing Pade approximants of Pi_TT. The
   result depends sensitively on the order of the approximant (6/6
   vs 8/8), indicating insufficient data for reliable extrapolation.

3. **Scalar sector catalogue.** Attempted to find zeros of Pi_s(z, 0)
   numerically. The function F_hat_2(z, 0) involves the same erfi
   function as F_hat_1(z) but with different coefficients. Preliminary
   search found no real zeros for z > 0 (no Euclidean ghost in scalar
   sector at xi = 0), but the complex zeros have not been mapped.

## 5. Success Criteria

- **Two-loop pole positions:** compute the shifts of all 8 tensor
  poles at two loops, using the finite part of the two-loop graviton
  self-energy.
- **Scalar pole catalogue:** find all zeros of Pi_s(z, xi) for xi = 0
  and xi = 1/6, and compute their residues.
- **d_S at two loops:** compute the ML spectral dimension using the
  corrected poles and compare with the one-loop result.
- **Convergence assessment:** determine whether the sigma* instability
  is reduced at two loops, or whether it persists and indicates a
  fundamental problem with the ML expansion.
- **Combined d_S:** if both tensor and scalar sectors are included,
  compute the total effective spectral dimension d_S^{total}(sigma)
  as a function of diffusion time.

## 6. Suggested Directions

1. **Two-loop self-energy via pySecDec.** Use sector decomposition
   (pySecDec package, WSL installation) to compute the finite part of
   the two-loop graviton self-energy at specific values of k^2/Lambda^2
   near the one-loop poles. This gives numerical values for the pole
   shifts.

2. **FORM computation of two-loop traces.** Use TFORM (parallel FORM,
   8 workers) to compute the two-loop gamma traces and tensor
   contractions needed for the self-energy. The one-loop traces are
   already implemented in sct_tools/form_interface.py.

3. **Scalar sector zeros.** Use the Newton-Raphson method with mpmath
   (100-digit precision) to find zeros of Pi_s(z, 0) in the complex
   plane. Start from a grid of initial points in |z| < 100.

4. **Borel summation.** The ML expansion is an asymptotic series.
   Apply Borel summation to extract a finite d_S even in the region
   sigma < sigma* where the direct sum diverges. This may resolve the
   P < 0 problem.

5. **Resurgence analysis.** The pole series for Pi_TT has the structure
   of a Borel-summable transseries. Apply resurgence techniques
   (Stokes phenomena, alien derivatives) to understand the connection
   between the perturbative (ML) and non-perturbative (full function)
   spectral dimensions.

## 7. References

1. Modesto, L. and Rachwal, L. (2013). "Super-renormalizable and
   finite gravitational theories." Nucl. Phys. B 889, 228.
   arXiv:1304.7247.
2. Chamseddine, A. H., Connes, A. and van Suijlekom, W. D. (2018).
   "Entropy and the spectral action." arXiv:1809.02944.
3. Boito, D. et al. (2019). "Strong coupling from hadronic tau
   decays: a critical appraisal." Phys. Rev. D 100, 074009.
   arXiv:1907.03720. [Borel summation methodology]
4. Costin, O. and Dunne, G. V. (2019). "Resurgent extrapolation:
   rebuilding a function from asymptotic data." J. Phys. A 52, 445205.
   arXiv:1904.11593.

## 8. Connections

- **OP-24 (definition dependence):** the two-loop result may
  distinguish between definitions: if the ML poles stabilize at two
  loops, the ML definition gains credibility; if they do not, the
  propagator (CMN) or heat kernel definition may be preferred.
- **MR-4 (two-loop structure):** the two-loop self-energy is needed
  here and also for MR-4 (two-loop beta functions). The computations
  share input.
- **OP-07 (ghost resolution):** the scalar sector has a different
  ghost structure from the tensor sector. The scalar ghosts (if any)
  are relevant for the second law (OP-22) and inflation (OP-17).
- **MR-6 (convergence):** the truncation instability of the ML
  expansion is an instance of the general question of whether the
  curvature expansion converges or is asymptotic (MR-6).

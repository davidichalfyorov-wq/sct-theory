# SO(3) Selection Rule for Path-Kurtosis on Local Vacuum Diamonds

## Source: independent analysis, 2026-03-28
## Status: Formal lemma + proposition, LaTeX-ready. Pending numerical confirmation.

## Core Results

### Lemma 5.1: E[D₁] = 0 for centered isotropic local vacuum diamond
- Proof via SO(3) invariant theory: A^ij must be proportional to delta^ij, traceless contraction kills it
- Alternative proof via spherical harmonics: l=0 ⊗ l=2 → 0
- Conditions: centered diamond, isotropic window W_zeta(tau,r), rotation-invariant sprinkling, vacuum (Ric=0)
- Ensemble-level statement (NOT samplewise)

### Proposition 5.2: Quadratic vacuum channel
- General form: E[dk_vac(T)] = T⁴(A_E·E²_ij + A_B·B²_ij) + o(T⁴)
- Cross-term E·B killed by time-reversal symmetry
- W = E²+B² form requires ADDITIONAL assumption A_E = A_B (electric-magnetic symmetry)
- On current data (B=0 for ppw and Sch): W-form and E²-form indistinguishable

### Corollary 5.3: Orientation averaging kills D₁
- For ANY window (even anisotropic): D̄₁ = ∫SO(3) D₁(Q) dQ = 0
- Proof: Haar average of rotated STF rank-2 tensor = 0

### Remark 5.4: Boundary-dressed D₁ ≠ 0
- Global shell with radial boundaries → Q^ij_win ≠ 0 (quadrupolar analyzer)
- D₁ = Gamma_W · Q^ij_win · E_ij (linear in curvature amplitude)
- Explains observed linear-in-M Schwarzschild signal in global shell pipeline

## Unified Picture
- pp-wave global: D₁=0 (x↔y symmetry kills it)
- Schwarzschild global shell: D₁≠0 (boundary analyzer, radial quadrupole)
- Local vacuum (ANY geometry): D₁=0 (SO(3) selection rule)
- Universal channel: D₂ ~ T⁴(A_E·E² + A_B·B²)

## Conditions for Lemma (Q8 answers)
- (a) Centered diamond: YES (spatial isotropy essential)
- (b) Poisson sprinkling: sufficient but not necessary (any rotation-invariant process)
- (c) Vacuum: YES (need l=2 linear response; Ricci adds l=0 channel)
- (d) Isotropic window: YES, critically (W_zeta depends only on tau, r)
- (e) Excess kurtosis: for formula yes; selection rule applies to any scalar distributional functional
- (f) Small-diamond regime: YES for T²/T⁴ counting

## Open Questions
- A_E vs A_B: need B≠0 geometry (Kerr, boosted Schwarzschild) to distinguish
- Self-averaging: samplewise D₁(omega) fluctuates; only E[D₁]=0 proven
- Continuum limit of A_E: not yet derived analytically

## Numerical Support
- pp-wave fixed-eps T-scaling: ratio/T⁴ ≈ 1.3-1.7 (closer to T⁴ than T²)
- dS fixed-H T-scaling: ratio/T² ≈ 0.7-2.0 (closer to T² than T⁴)
- Channel separation D₁(Ricci)/D₂(Weyl) qualitatively confirmed
- Sch local M=200 and dS subtraction fix: RUNNING

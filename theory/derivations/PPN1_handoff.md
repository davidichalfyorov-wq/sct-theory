# Phase PPN1 Handoff Certificate

**Phase ID:** PPN-1  
**Date:** March 12, 2026  
**Current reviewer state:** Reopened after forensic re-audit and repaired to an honest linear-static scope

## Sequential Pipeline Summary
- **Literature Pass:** collected the standard PPN framework and the higher-derivative / nonlocal weak-field literature.
- **Literature Review:** correctly flagged that linear weak-field material is enough for $\gamma$, but not for $\beta$ or the full ten-parameter PPN completion.
- **Derivation Pass:** produced a first bridge from NT-4a to weak-field observables, but originally mixed exact nonlocal statements with a local Yukawa surrogate and overstated completion.
- **Derivation Review / forensic re-audit:** found the broken projector narrative, the placeholder nonlinear text, the wrong short-range bound in earlier artifacts, and the unjustified claims for $\beta$, $\alpha_i$, $\zeta_i$.
- **Repair pass:** rebuilt the implemented PPN layer so that it now exposes only the quantities actually derived from the NT-4a linear static sector.
- **Verification Pass / V-R repair audit:** retested the repaired code and rewrote the derivation and prediction documents to match the real scope.

## What Is Now Valid
- The exact linear static NT-4a metric potentials are represented by Fourier--Bessel integrals in terms of $\Pi_{\mathrm{TT}}$ and $\Pi_{\mathrm{s}}$.
- The implemented phenomenological layer is the **local Yukawa approximation** to that exact static kernel.
- The underlying normalized NT-4a TT denominator has a positive-real-axis zero at
  `z ~= 2.4148388899`; PPN-1 therefore carries no ghost-freedom claim.
- In that approximation,
  \[
    \gamma(r)
    =
    \frac{1 - \frac{2}{3}e^{-m_2 r} - \frac{1}{3}e^{-m_0 r}}
         {1 - \frac{4}{3}e^{-m_2 r} + \frac{1}{3}e^{-m_0 r}},
  \]
  with
  \[
    m_2 = \Lambda \sqrt{\frac{60}{13}},
    \qquad
    m_0 = \frac{\Lambda}{\sqrt{6(\xi-1/6)^2}},
  \]
  and $m_0 = \infty$ at $\xi = 1/6$.
- The corrected lower bounds are:
  - **Cassini:** $\Lambda \gtrsim 6.2 \times 10^{-18}\,\mathrm{eV}$
  - **Short-range gravity / Eöt-Wash scale:** $\Lambda \gtrsim 1.84 \times 10^{-3}\,\mathrm{eV}$

## What Is Explicitly Not Derived
- $\beta$
- $\xi_{\mathrm{PPN}}$
- $\alpha_1, \alpha_2, \alpha_3$
- $\zeta_1, \zeta_2, \zeta_3, \zeta_4$

These quantities require the nonlinear post-Newtonian field equations and are now marked `not_derived` in code and documentation.

## Important Repair Outcome
- The conformal point $\xi = 1/6$ no longer carries the false claim of short-distance regularity in the local Yukawa approximation.
- The repaired code now distinguishes:
  - generic $\xi \neq 1/6$: scalar-assisted cancellation of the local $1/r$ term,
  - exact $\xi = 1/6$: scalar channel absent, residual local $1/r$ behavior reappears.

## Verification Slice Re-run
Executed after the repair:

```text
python -m pytest analysis/sct_tools/tests/test_nt4a_propagator.py \
  analysis/sct_tools/tests/test_nt4a_newtonian.py \
  analysis/sct_tools/tests/test_ppn1.py -q
```

Result:
- `112 passed`

Additional repair checks:
- `python -m analysis.scripts.ppn1_verification`
- `python analysis/scripts/verify_ppn1_symbolic.py`
- Machine-readable repair archive now stored in `analysis/results/ppn1/`.

## Result Archive
- `analysis/results/ppn1/ppn1_snapshot.json`
- `analysis/results/ppn1/ppn1_verification.json`
- `analysis/results/ppn1/ppn1_symbolic.json`

## Integration Gate
- Full `analysis/run_ci.py` now passes after the repository-wide lint cleanup:
  - `ruff`: PASS
  - `pytest`: PASS (`2221 passed`)
  - canonical spot-check: PASS

## Artifacts Superseded / Rewritten
- `theory/derivations/PPN1_derivation.tex`
- `theory/predictions/prediction_ppn1.tex`
- `analysis/scripts/ppn1_parameters.py`
- `analysis/scripts/ppn1_verification.py`
- `analysis/scripts/verify_ppn1_symbolic.py`
- `analysis/scripts/nt4a_newtonian.py`
- `analysis/scripts/nt4a_verify.py`

## Verdict
**REOPENED / PARTIALLY REPAIRED**

This phase is no longer described as a completed ten-parameter PPN derivation.  
What is now trustworthy is the **linear static local-Yukawa sector** built on top of the repaired NT-4a propagator.  
The remaining nonlinear PPN work must be treated as open.

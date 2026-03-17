# SCT Theory

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19056983.svg)](https://doi.org/10.5281/zenodo.19056983)
[![Archived in Software Heritage](https://archive.softwareheritage.org/badge/swh:1:dir:da228c6cdf4a95844f2e98bfe508e31145a72580/)](https://archive.softwareheritage.org/browse/directory/da228c6cdf4a95844f2e98bfe508e31145a72580/?origin_url=https://doi.org/10.5281/zenodo.19056982&path=davidichalfyorov-wq-sct-theory-7e8f479&release=1&snapshot=0e5684bb3f3bb036d1972363539b24eab0570376)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0003--6027--7837-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0009-0003-6027-7837)
![Version](https://img.shields.io/badge/version-research--wip-orange)
![Status](https://img.shields.io/badge/status-work%20in%20progress-yellow)
![Formally Verified](https://img.shields.io/badge/formally_verified-partial-orange)
![Lean 4](https://img.shields.io/badge/Lean-4-1f6feb)
![Consistency](https://img.shields.io/badge/consistency_checked-ongoing-blueviolet)
![Paper Build](https://img.shields.io/badge/paper_build-local-lightgrey)
![Python](https://img.shields.io/badge/Python-3.12-3776AB)
[![License: Apache-2.0](https://img.shields.io/badge/Code-Apache--2.0-blue.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/Docs-CC%20BY%204.0-lightgrey.svg)](LICENSE-docs.md)

**Spectral Causal Theory** is a research program that asks whether gravity, quantum corrections, and fundamental physics can be derived from the spectral data of the Dirac operator, in a mathematically controlled, computationally verified, and experimentally testable way.

> Work in progress: verified results and Lean 4 proofs already exist in the repository, but the broader derivation, formalization, and consistency-check program is still expanding.

<p align="center">
  <img src="docs/figures/roadmap_progress.png" alt="Research roadmap progress" width="680"/>
</p>

---

## The Problem

Modern physics rests on two pillars:

| Framework | Describes | Works at |
|-----------|-----------|----------|
| **General Relativity** | Gravity, spacetime, black holes, cosmology | Large scales |
| **Quantum Field Theory** | Particles, forces, the Standard Model | Small scales |

Both work extraordinarily well in their domains. The open problem is what happens in regimes where both matter: the early universe, black hole interiors, and quantum corrections to gravity itself.

## The Idea

Instead of writing down an arbitrary higher-derivative gravitational action, SCT starts from a different premise:

> **Geometry leaves a fingerprint in the spectrum of the Dirac operator. We read physics from that fingerprint.**

Concretely, the **spectral action principle** uses the spectrum of a generalized Dirac operator to construct an action functional. Expanding it in curvature invariants recovers the Einstein-Hilbert action at low energy, but also produces specific, calculable quantum corrections governed by **nonlocal form factors**.

These form factors are not free parameters. They are fixed by the spectrum and the particle content of the Standard Model.

## What This Repository Does

The project derives, computes, and verifies these predictions step by step:

```
Spectral geometry  →  Spectral action  →  Heat kernel expansion
     →  Form factors (per spin)  →  Combined SM coefficients
     →  Field equations  →  Modified gravity  →  Testable predictions
```

### Key results established so far

**One-loop form factors** for all Standard Model sectors (scalar, Dirac, vector) are computed and cross-verified. All form factors are governed by a single **master function**:

<p align="center">
  <img src="docs/figures/master_function.png" alt="The SCT master function φ(x)" width="560"/>
</p>

This function is **entire** (no poles in the complex plane), which guarantees ghost-freedom of the propagator at tree level.

The plot uses `x = -k^2/\Lambda^2`, so the left branch corresponds to `x < 0` and the right branch to `x > 0`. The left branch is the Euclidean continuation and grows as `x -> -infinity` because `|x|` increases toward the left; the right branch decays as `\varphi(x) ~ 2/x` for `x -> +infinity`. In particular, the explicit factor `e^{-x/4}` in the closed form does not imply exponential decay on the positive branch, because `\operatorname{erfi}(\sqrt{x}/2)` contributes the compensating `e^{x/4}` asymptotic.

**Modified Newtonian potential.** At distances comparable to the spectral scale 1/&Lambda;, the gravitational potential departs from Newton's law and becomes finite at the origin:

<p align="center">
  <img src="docs/figures/newtonian_potential.png" alt="Modified Newtonian potential" width="560"/>
</p>

**Standard Model contributions.** Each particle sector (scalars, fermions, gauge bosons) contributes different heat kernel coefficients and form factor profiles:

<p align="center">
  <img src="docs/figures/sm_contributions.png" alt="SM sector contributions" width="680"/>
</p>

**Parameter-free predictions.** The combined Weyl-squared coefficient is &alpha;<sub>C</sub> = 13/120, entirely fixed by the Standard Model content. The ratio c<sub>1</sub>/c<sub>2</sub> = &minus;1/3 at conformal coupling is a testable, parameter-free prediction.

## Repository Structure

```
theory/           Formal theory content
  axioms/           Foundational postulates
  derivations/      Step-by-step mathematical derivations
  predictions/      Testable predictions with observables and precision targets
  consistency-checks/  Internal consistency proofs

analysis/         Computational backbone
  sct_tools/        Python package (13 modules, 4000+ tests)
  scripts/          Verification and computation scripts
  figures/          Publication-quality figures

papers/           Publication drafts and build tools
docs/             Roadmap, overview, presentations
```

## Verification Philosophy

Hard derivations fail for boring reasons: wrong signs, mismatched conventions, silent transcription errors. This project uses an **8-layer verification pipeline** instead of trusting any single calculation:

| Layer | Method | Purpose |
|-------|--------|---------|
| 1 | Analytic checks | Dimensions, limits, symmetries, pole cancellation |
| 2 | Numerical (100+ digits) | High-precision evaluation at multiple test points |
| 2.5 | Property fuzzing | 1000+ randomized hypothesis tests |
| 3 | Literature comparison | Cross-check against 13+ independent references |
| 4 | Dual derivation | Independent method, different approach |
| 4.5 | Triple CAS | SymPy, GiNaC, and mpmath must agree to 12+ digits |
| 5 | Lean 4 formal proofs | Machine-verified rational identities |
| 6 | Multi-backend | Multiple Lean backends must independently pass |

## Published Work

| # | Paper | DOI |
|---|-------|-----|
| 1 | Nonlocal one-loop form factors of the spectral action with Standard Model content | [10.5281/zenodo.19039242](https://doi.org/10.5281/zenodo.19039242) |
| 2 | Solar system and laboratory tests of the spectral action scale | [10.5281/zenodo.19045796](https://doi.org/10.5281/zenodo.19045796) |
| 3 | Chirality of the Seeley-DeWitt coefficients and quartic Weyl structure in the spectral action | [10.5281/zenodo.19056295](https://doi.org/10.5281/zenodo.19056295) |
| 4 | Nonlinear field equations and FLRW cosmology of the spectral action with Standard Model content | [10.5281/zenodo.19056349](https://doi.org/10.5281/zenodo.19056349) |

## Quick Start

```bash
python -m pip install -r requirements.txt
python -m pytest analysis/ -x -q          # run tests
python analysis/run_ci.py                 # full CI pipeline
python papers/build.py                    # compile all LaTeX documents
```

## What This Project Is Not

- It is **not** a claim that all of fundamental physics is finished.
- It is **not** a replacement for peer review.
- It is **not** a promise that every research direction will survive future checks.

It is a serious research workspace built to make derivations **reproducible**, **inspectable**, and **falsifiable**.

## Author

Formal theory documents and papers are authored by **David Alfyorov** ([ORCID](https://orcid.org/0009-0003-6027-7837)).

## Credits

Research-assistance and workflow support: **Aliaksandr Samatyia**.

## Licensing

- Source code: [Apache-2.0](LICENSE)
- Research text, derivations, and documentation: [CC BY 4.0](LICENSE-docs.md)

## Star History

<a href="https://www.star-history.com/?repos=davidichalfyorov-wq%2Fsct-theory&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/image?repos=davidichalfyorov-wq/sct-theory&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/image?repos=davidichalfyorov-wq/sct-theory&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/image?repos=davidichalfyorov-wq/sct-theory&type=date&legend=top-left" />
 </picture>
</a>

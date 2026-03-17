# SCT Theory

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19056983.svg)](https://doi.org/10.5281/zenodo.19056983)
[![Archived in Software Heritage](https://archive.softwareheritage.org/badge/swh:1:dir:da228c6cdf4a95844f2e98bfe508e31145a72580/)](https://archive.softwareheritage.org/browse/directory/da228c6cdf4a95844f2e98bfe508e31145a72580/?origin_url=https://doi.org/10.5281/zenodo.19056982&path=davidichalfyorov-wq-sct-theory-7e8f479&release=1&snapshot=0e5684bb3f3bb036d1972363539b24eab0570376)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0003--6027--7837-A6CE39?logo=orcid&logoColor=white)](https://orcid.org/0009-0003-6027-7837)
![Version](https://img.shields.io/badge/version-1.0.2-blue)
![Tests](https://img.shields.io/badge/tests-4196%2B%20passed-brightgreen)
![PDFs](https://img.shields.io/badge/PDFs-73%2F73%20compiled-brightgreen)
![Formally Verified](https://img.shields.io/badge/Lean_4-13_theorems%2C_0_sorry-1f6feb)
![Consistency](https://img.shields.io/badge/consistency_checked-30_phases-blueviolet)
![Paper Build](https://img.shields.io/badge/paper_build-local-lightgrey)
![Python](https://img.shields.io/badge/Python-3.12-3776AB)
[![License: Apache-2.0](https://img.shields.io/badge/Code-Apache--2.0-blue.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/Docs-CC%20BY%204.0-lightgrey.svg)](LICENSE-docs.md)

**Spectral Causal Theory** is a research program investigating whether gravity and its quantum corrections can be derived from the spectral data of the Dirac operator.

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

## What's New in v1.0.2

### Chirality proof and perturbative UV finiteness

The central result of this release: **a four-line algebraic proof that all perturbative counterterms in D&sup2;-quantization of the spectral action are block-diagonal in the chiral basis**, and therefore absorbable by spectral function deformation at every loop order.

The proof rests on the identity: since the Dirac operator *D* anticommutes with the chirality operator &gamma;<sub>5</sub>, the perturbation &delta;(*D*&sup2;) automatically *commutes* with &gamma;<sub>5</sub>. This forces the kinetic operator, propagator, vertices, and all multi-loop diagrams into the chirality-preserving subalgebra.

- UV finiteness holds through two loops without additional assumptions
- At all perturbative orders under five BV axioms (three proven, two verified to one loop)
- The algebraic identity is formally verified in **Lean 4** (13 theorems, zero sorry)

### Black hole entropy

The Wald entropy for Schwarzschild black holes in SCT is computed with full Standard Model content:

*S* = *A*/(4*G*) + 13/(120&pi;) + (37/24) ln(*A*/&ell;<sub>P</sub>&sup2;) + O(1)

The logarithmic coefficient **c<sub>log</sub> = 37/24** is parameter-free and has opposite sign to the Loop Quantum Gravity prediction (c<sub>log</sub> = &minus;3/2), providing a potential observational discriminant.

### Black hole singularity: groundwork

The modified Newtonian potential from the spectral action's nonlocal form factors is finite at the origin: V(0) = 0. In the Yukawa approximation, the Kretschner scalar is softened from K ~ r<sup>&minus;6</sup> (Schwarzschild) to K ~ r<sup>&minus;4</sup>. Full singularity resolution (de Sitter core, K finite everywhere) requires solving the complete nonlinear field equations self-consistently on a spherically symmetric background &mdash; the linearized analysis, effective source formalism, and energy condition framework are in place.

### Upgraded consistency results

Five previously open questions about the theory's internal consistency are now resolved or narrowed:

- **Two-loop finiteness** is now unconditional (previously required a specific absorption scheme)
- **Unitarity** in D&sup2;-quantization follows from the bounded propagator (no ghost poles)
- **Optical theorem** follows from unitarity
- **All-orders finiteness** is conditional on two BV axioms, both verified to one loop

### Literature cross-check

An equation-by-equation comparison document verifies every SCT form factor result against published literature:
- Codello-Zanusso (2013, J. Math. Phys. 54, 013513)
- Codello-Percacci-Rahmede (2009, Annals Phys. 324, 414)
- Tseytlin-Shapiro-Ribeiro (2020, Phys. Lett. B 808, 135645)

All results match to 15+ significant digits. Zero discrepancies found. One convention difference (sign of endomorphism, notational).

### Honest limitations identified

- **Inflation:** The minimal spectral action predicts a scalaron mass M = 15.4 M<sub>Pl</sub>, six orders of magnitude too heavy for Starobinsky inflation. Nonlocal form factor corrections do not rescue this (F&#770;<sub>2</sub> decreases at high momenta). SCT does not explain inflation without BSM extensions.
- **Singularity resolution:** Softened (K: r<sup>&minus;6</sup> &rarr; r<sup>&minus;4</sup>) but not resolved in the linearized Yukawa approximation. The form factors are entire functions of order 1; full resolution requires order 2 (exponential UV suppression).
- **D&sup2;-quantization vs metric quantization:** Physical equivalence established through one loop. All-orders equivalence conditional on two BV axioms (Jacobian well-definedness and anomaly freedom), verified to one loop.

## What This Repository Does

The project derives, computes, and verifies predictions step by step:

```
Spectral geometry  ->  Spectral action  ->  Heat kernel expansion
     ->  Form factors (per spin)  ->  Combined SM coefficients
     ->  Field equations  ->  Modified gravity  ->  Testable predictions
```

### Key results established

**One-loop form factors** for all Standard Model sectors (scalar, Dirac, vector) are computed and cross-verified. All form factors are governed by a single **master function**:

<p align="center">
  <img src="docs/figures/master_function.png" alt="The SCT master function phi(x)" width="560"/>
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

## Formal Verification (Lean 4)

The core algebraic identities are machine-verified in Lean 4 with Mathlib:

| Theorem | Statement | File |
|---------|-----------|------|
| `chiral_q_identity` | (AB + BA + B&sup2;)C = C(AB + BA + B&sup2;) given AC = &minus;CA, BC = &minus;CB | `theory/lean/proofs/chiral_q_identity.lean` |
| `bv_canonical_transformation` | BV canonical transformations preserve the antibracket | same |
| `centralizer_inv_closed` | If [K, &gamma;<sub>5</sub>] = 0 then [K<sup>&minus;1</sup>, &gamma;<sub>5</sub>] = 0 | same |
| + 10 more | Even Clifford comm, spin connection, diffeo generator, CME, ... | same |

**13 theorems. Zero sorry statements. All proofs complete.**

## Literature Cross-Check

Every key equation is traced to published sources:

| SCT result | Published source | Status |
|------------|-----------------|--------|
| Master function &phi;(x) | Codello-Zanusso (2013) eq. (2.3) | **Exact match** |
| Five CZ form factors | Codello-Zanusso (2013) eq. (2.21) | **Exact match** |
| &beta;<sub>W</sub> per spin | Codello-Percacci-Rahmede (2009) eq. (III.9) | **Exact match** |
| &alpha;<sub>C</sub> = 13/120 | CPR counting with SM content | **Exact match** |
| All local limits | CZ (2013) eq. (2.22) | **Exact match** |

Full comparison: `theory/derivations/SCT_literature_comparison.tex`

## Repository Structure

```
theory/           Formal theory content
  axioms/           Foundational postulates
  derivations/      Step-by-step mathematical derivations (60+ documents)
  predictions/      Testable predictions with observables and precision targets
  consistency-checks/  Internal consistency proofs (15+ checks)
  lean/proofs/      Lean 4 formal proofs (13 theorems, 0 sorry)

analysis/         Computational backbone
  sct_tools/        Python package (13 modules, 4196+ tests)
  scripts/          Verification and computation scripts (30+ scripts)
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
| 3 | Chirality of the Seeley-DeWitt coefficients and quartic Weyl structure in the spectral action | [10.5281/zenodo.19056204](https://doi.org/10.5281/zenodo.19056204) |
| 4 | Nonlinear field equations and FLRW cosmology of the spectral action with Standard Model content | [10.5281/zenodo.19056349](https://doi.org/10.5281/zenodo.19056349) |
| 5 | Perturbative UV finiteness of the spectral action in D&sup2;-quantization: a chirality proof | *preprint in repository* |

## Research Status

30 research phases completed. Highlights:

| Topic | Key result | Status |
|-------|-----------|--------|
| One-loop form factors | All SM sectors computed, master function entire | Complete |
| Nonlinear field equations | Full variational equations + FLRW reduction | Complete |
| Lorentzian formulation | Wick rotation of the spectral action | Complete |
| Unitarity | Bounded propagator in D&sup2;-quantization, no ghost poles | Closed |
| Causality | Signal speed = *c* (macroscopic); micro-violation at &ell; ~ 1/&Lambda; | Conditional |
| Two-loop finiteness | Counterterm uniquely absorbed | **Unconditional** |
| All-orders finiteness | Via chirality + BV axioms | Conditional |
| Graviton scattering | Tree-level SCT = GR; one-loop finite | Certified |
| Solar system tests | Spectral scale &Lambda; > 2.565 meV from torsion-balance | Complete |
| Black hole entropy | c<sub>log</sub> = 37/24 (opposite sign to LQG) | Certified |
| Black hole singularity | Kretschner softened r<sup>&minus;6</sup> &rarr; r<sup>&minus;4</sup>; full resolution in progress | Conditional |
| Late-time cosmology | Corrections 60+ orders below observability | Consistent |
| Inflation | Scalaron mass too heavy; requires BSM extension | Negative |

## Quick Start

```bash
python -m pip install -r requirements.txt
python -m pytest analysis/ -x -q          # run tests (4196+ pass)
python analysis/run_ci.py                 # full CI pipeline
python papers/build.py                    # compile all LaTeX (73/73)
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

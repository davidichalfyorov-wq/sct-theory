# SCT Theory

SCT Theory is a research repository for Spectral Causal Theory: a program that studies whether gravity, quantum corrections, and parts of fundamental physics can be organized from spectral data in a mathematically controlled and testable way.

## The Short Version
Modern physics has two extremely successful frameworks:

- general relativity, which describes gravity and spacetime on large scales
- quantum field theory, which describes particles and interactions on small scales

Both work very well in their own domains. The hard problem is that they are not yet unified into one clean, fully verified framework for extreme regimes such as the early universe, black holes, and quantum corrections to gravity.

This repository studies one possible route: start from spectral geometry, especially the Dirac operator and spectral action, and ask whether known low-energy physics can be recovered while also producing new, concrete, falsifiable predictions.

## What We Are Actually Doing
In simple terms, the project asks:

1. Can the geometry of spacetime be encoded through spectral data rather than only through the metric written in the usual way?
2. If we compute quantum corrections carefully, do we recover the known Einstein-Hilbert limit at low energy?
3. Do nonlocal form factors and curvature-squared terms produce fixed relations that could be tested, instead of arbitrary free parameters?
4. Can the framework be checked step by step with symbolic algebra, numerical verification, literature comparison, and formal proof tools rather than by handwaving?

This means the repository is not just a pile of notes. It is a working research environment where derivations, scripts, tests, figures, and publication documents are kept together and cross-checked.

## What "Spectral" Means Here
Instead of treating geometry only as distances and curvature written directly from the metric, spectral geometry studies operators whose spectrum carries geometric information.

The central object in this repository is the Dirac operator `D`. Very roughly:

- the spectrum of `D` tells us something about the underlying geometry
- the spectral action uses `D` to build an action functional
- expanding that action gives terms that can be compared with ordinary gravitational physics

In plain language: geometry leaves a fingerprint in the spectrum, and we are trying to read physics from that fingerprint in a controlled way.

## Why This Matters
If this program works, it gives three valuable things:

- a mathematically structured way to derive gravitational terms rather than postulate them
- a route to concrete quantum-gravity corrections that can be checked against known physics
- specific predictions that could in principle be constrained or ruled out by observation

Even if the full program fails, the partial outputs are still useful:

- corrected derivations
- verified form factors
- reproducible symbolic and numerical workflows
- clear statements of what is proven, what is open, and what is falsifiable

## What This Repository Contains
- `theory/` holds the core scientific content: axioms, derivations, predictions, and consistency documents.
- `analysis/` holds the computational backbone: the `sct_tools` Python package, verification scripts, figures, and tests.
- `papers/` holds publication material and build helpers.
- `docs/` holds roadmap and high-level project documentation.

## How We Work
The repository is organized around a simple rule: claims should be derived and checked, not merely asserted.

That is why the project uses:

- analytic checks
- high-precision numerical checks
- comparison with the literature
- independent re-derivation
- multiple CAS backends
- Lean-based formal checks for suitable identities

The aim is not to make the project look impressive. The aim is to reduce avoidable mistakes in hard derivations.

## What Has Been Established So Far
As of March 13, 2026, the repository contains completed and documented work on:

- one-loop nonlocal form factors for scalar, Dirac, and vector sectors
- combined Standard Model curvature-squared coefficients
- the entire-function control phase (`NT-2`)
- the first linearized field-equation phase (`NT-4a`)

The roadmap and current scientific status are summarized in:

- [docs/overview.md](docs/overview.md)
- [docs/SCT_roadmap.tex](docs/SCT_roadmap.tex)

## What This Repository Is Not
- It is not a claim that all of fundamental physics is finished.
- It is not a replacement for peer review.
- It is not a promise that every research direction in the project will survive future checks.

It is a serious research workspace built to make derivations reproducible, inspectable, and easier to falsify.

## Author
Formal theory documents and papers are authored by `David Alfyorov`.

## Credits
Acknowledgements: `Aliaksandr Samatyia` contributed research-assistance and workflow support.

## Quick Start
```powershell
python -m pip install -r requirements.txt
python -m pytest analysis/ -x -q
python analysis/run_ci.py
python papers/build.py
```

## Repository Policy
- Public releases should contain reproducible code, clean derivations, and publication materials.
- Local environment files, private workflow notes, large raw datasets, caches, and generated scratch outputs are intentionally excluded.

## Licensing
- Source code is released under [Apache-2.0](LICENSE).
- Research text, derivations, and documentation are released under [CC BY 4.0](LICENSE-docs.md), unless noted otherwise.

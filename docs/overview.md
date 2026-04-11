# SCT Theory Overview

## Purpose
This project studies Spectral Causal Theory as a research program in quantum gravity and spectral geometry.

The central question is simple to state:

Can we recover known gravity at low energy, compute quantum corrections in a controlled way, and extract new testable predictions from spectral data rather than from an arbitrarily chosen higher-derivative action?

## The Problem in Ordinary Language
Physics has two great theories:

- general relativity explains gravity, curved spacetime, black holes, and cosmology
- quantum field theory explains particles and interactions

Each works extremely well. The open problem is how to treat situations where both matter and gravity are simultaneously important at a deep level.

This repository explores one route to that problem. It does not assume that the route is already complete. Instead, it develops the mathematics piece by piece and checks every important result against independent methods.

## Core Idea
The project works with the idea that geometry can be studied through spectral information associated with the Dirac operator.

In practical terms:

- build the relevant spectral objects
- derive the spectral action and its curvature expansion
- compute local and nonlocal form factors
- compare the low-energy limit to known gravitational physics
- extract coefficients and ratios that can be tested

The repository therefore combines theoretical physics, mathematical physics, and reproducible computation.

## Scientific Strategy
The project is organized as a sequence of concrete phases rather than as one vague grand claim.

Examples of phase questions:

- what are the correct one-loop form factors for scalar, Dirac, and vector sectors?
- what coefficients appear in the combined Standard Model contribution?
- are the resulting functions entire?
- what do the linearized field equations look like?
- what observables could distinguish this framework from standard alternatives?

Each phase is documented in derivation files and backed by scripts, numerical checks, and publication-ready figures.

## Why The Verification Layer Matters
Hard derivations fail for boring reasons surprisingly often:

- wrong sign conventions
- wrong trace normalization
- bad local limits
- silent transcription mistakes
- hidden assumptions about gauges or counting conventions

So the project uses a verification philosophy rather than trusting a single symbolic derivation.

The main checks include:

- analytic sanity checks
- high-precision numerical evaluation
- literature cross-checking
- independent derivation routes
- multiple computer algebra backends
- Lean-based proof checks where appropriate

This does not guarantee truth, but it greatly improves reliability.

## What Someone New Should Understand
You do not need to know all the mathematics to understand the purpose.

At the highest level, the repository is trying to answer:

- does this framework reproduce known physics where it must?
- does it make sharper predictions than generic higher-curvature models?
- can those predictions eventually be confronted with data?

If the answer becomes yes in a rigorous way, the theory becomes scientifically interesting.
If the answer becomes no, the repository still records exactly where and why it failed.

## Reading Order
For a newcomer:

1. Read the repository `README.md`.
2. Read `docs/SCT_roadmap.tex` for status and priorities.
3. Browse `theory/derivations/` for the concrete scientific outputs.
4. Browse `analysis/sct_tools/` and `analysis/scripts/` for reproducibility and verification.

## Author and Credits
- Author of formal theory documents: `David Alfyorov`

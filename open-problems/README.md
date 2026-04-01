# SCT Open Problems

A collection of 50 open research problems in Spectral Causal Theory (SCT),
organized by domain and ranked by priority. Each problem file is
self-contained: it includes the problem statement, context, known partial
results, failed approaches, success criteria, and key references.

## What is SCT?

Spectral Causal Theory is a candidate framework for quantum gravity based
on the spectral action principle of Chamseddine and Connes. The classical
action is S = Tr(f(D^2/Lambda^2)), where D is the Dirac operator of a
spectral triple encoding both gravitational and Standard Model degrees of
freedom. The one-loop effective action has been fully computed, yielding
a nonlocal gravitational action with entire-function form factors, a
parameter-free Weyl-squared coefficient alpha_C = 13/120, and a modified
Newtonian potential that is finite at the origin.

## Structure

```
open-problems/
  README.md               This file
  INDEX.md                Master table of all 50 problems
  GLOSSARY.md             Notation glossary
  VERIFIED-RESULTS.md     Registry of established results (ground truth)
  METHODOLOGY.md          Evaluation standards for proposed solutions
  PRIORITY.md             Priority ranking by impact
  DEPENDENCY-GRAPH.md     Problem dependency graph (Mermaid)
  QUICKSTART.md           SCT theory in two pages
  FAQ.md                  Common misconceptions
  LITERATURE-MAP.md       Key papers with specific equation references
  tests/                  Validation scripts for proposed solutions
  data/                   Numerical data files (JSON)
  benchmarks/             3 solved problems for calibration
  data/                   Numerical data files (JSON)
  foundations/             6 problems (Postulate 5, Gap G1, UV-completeness, ...)
  unitarity/              6 problems (fakeon, Kubo-Kugo, BV axioms, ...)
  uv-finiteness/          4 problems (three-loop, hidden principle, ...)
  cosmology/              4 problems (scalaron mass, dilaton, late-time, ...)
  black-holes/            3 problems (singularity, second law, information)
  spectral-dimension/     2 problems (definition dependence, corrections)
  predictions/            8 problems (QNMs, GWs, TOV, comparison table, ...)
  causal-sets/           10 problems (N-scaling, stratification, bridge, ...)
  scalar-sector/          1 problem (critical coupling xi)
  numerical/              5 problems (Kottler test, large-N, ...)
  formal-verification/    1 problem (Lean 4 remaining)
```

## How to use

Each problem file has YAML frontmatter with structured metadata:

```yaml
id: OP-XX
title: "..."
domain: [theory, numerics, phenomenology, mathematics, formal-verification]
difficulty: easy | medium | hard | very-hard
status: open | partial | blocked | resolved
deep-research-tier: A | B | C | D
blocks: [OP-YY, ...]
blocked-by: [OP-ZZ, ...]
```

**Tiers** indicate the type of effort needed:
- **Tier A** (12 problems): Literature review and analytical reasoning
- **Tier B** (15 problems): Mixed -- literature plus computation
- **Tier C** (11 problems): Primarily computational
- **Tier D** (12 problems): Research programs requiring decomposition

## Statistics

| Difficulty | Count |
|------------|-------|
| Very hard  | 13    |
| Hard       | 18    |
| Medium     | 14    |
| Easy       | 5     |
| **Total**  | **50**|

| Status | Count |
|--------|-------|
| Open | 47 |
| Partial | 1 (OP-33) |
| **Resolved** | **2 (OP-20, OP-44)** |

## Where to start

For newcomers to SCT, start with:
1. `GLOSSARY.md` for notation
2. Domain `BACKGROUND.md` files for context
3. `PRIORITY.md` for the most impactful problems
4. `benchmarks/` to test your understanding on solved problems

The five highest-priority open problems are:
1. **OP-13**: Three-loop overdetermination (blocks UV-completeness)
2. **OP-07**: Fakeon prescription for infinite-pole propagators
3. **OP-34**: N^{8/9} scaling exponent derivation
4. **OP-01**: Gap G1 (Weyl sector on curved backgrounds)
5. **OP-17**: Scalaron mass problem

## References

- D. Alfyorov, "Nonlocal one-loop form factors from the spectral action
  principle," DOI:10.5281/zenodo.19098042.
- D. Alfyorov, "Solar system and laboratory tests of spectral causal
  theory," DOI:10.5281/zenodo.19098100.
- D. Alfyorov, "Chirality of the Seeley-DeWitt coefficients in spectral
  causal theory," DOI:10.5281/zenodo.19118075.
- D. Alfyorov, "Nonlinear field equations and FLRW cosmology from the
  spectral action," DOI:10.5281/zenodo.19098027.

## License

This problem collection is part of the SCT Theory repository.

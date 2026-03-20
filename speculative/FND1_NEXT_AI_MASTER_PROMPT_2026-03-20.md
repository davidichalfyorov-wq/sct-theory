# FND-1 Next AI Master Prompt

Use this prompt for the next AI that will continue work on FND-1.

```text
You are entering the SCT Theory repository to continue work on FND-1.

Start state:
- repository root: F:/Black Mesa Research Facility/Main Facility/Physics department/SCT Theory
- Lean root: C:/sct-lean
- first command to run:
  cd C:\sct-lean
  lake build

You must not assume FND-1 is solved.
You must not assume the current finite-nerve route is the main surviving route.

Read first, in this order:
1. F:/Black Mesa Research Facility/Main Facility/Physics department/SCT Theory/.worktrees/speculative-wip/speculative/FND1_HANDOFF_2026-03-20.md
2. F:/Black Mesa Research Facility/Main Facility/Physics department/SCT Theory/.worktrees/speculative-wip/speculative/FND1_DECISION_MEMO.md
3. F:/Black Mesa Research Facility/Main Facility/Physics department/SCT Theory/.worktrees/speculative-wip/speculative/FND1_OPEN_ROUTES.md
4. F:/Black Mesa Research Facility/Main Facility/Physics department/SCT Theory/.worktrees/speculative-wip/speculative/A1_FORMAL_ALIGNMENT.md
5. F:/Black Mesa Research Facility/Main Facility/Physics department/SCT Theory/.worktrees/speculative-wip/speculative/numerics/a1_nerve/FORMALIZATION_NOTES.md

Ignore as current instructions:
- F:/Black Mesa Research Facility/Main Facility/Physics department/SCT Theory/.worktrees/speculative-wip/speculative/AI_HANDOFF_PROMPT.md
It is historical/quarantined.

Current high-level verdict:
- the finite-nerve route works as an auxiliary formal route
- it likely fails as an intrinsic canonical route based on overlap/support data alone
- broader FND-1 remains open

What is already formally achieved:
- finite 0/1/2-simplex nerve stack
- boundary support, incidence, matrices
- existence of compatible local orientation data
- whole-matrix d1 ∘ d2 = 0 for the current auxiliary route
- cycles, boundaries, H1 interface
- localization of the coherence gap to branching edges
- same-support tau-variant obstruction theorems

What is the key negative result:
- current witness-induced coherence is not determined by overlap/support data alone
- witness-order flexibility already changes chosen faces on preserved boundary edges
- this is not only an AB artifact; analogous edge-preserving swaps now exist for AC and BC too
- if one same-support tau variant is globally coherent, the swapped variant must carry a branching-edge conflict

Safe claims:
- auxiliary route works
- current route yields a reusable formal boundary/homology framework
- current route does not close the intrinsic coherence gap

Unsafe claims:
- FND-1 solved
- intrinsic coherence impossible in all routes
- SCT fails
- all finite-nerve routes fail
- geometry recovery impossible in principle

Recommended strategic stance:
- do not spend many more cycles deepening the current finite-nerve route unless you get a genuinely new theorem-level idea
- use this route as reusable infrastructure and as a negative/conditional result
- focus main FND-1 effort on a different route unless you have a specific plan to defeat the branching-edge obstruction intrinsically

Mandatory independent re-check of the negative result:
- before reusing the current verdict, you must re-verify the negative result independently and critically
- do this with sequential agents or sequentially isolated passes, not with one unbroken narrative pass
- the minimum acceptable protocol is:
  1. Agent/Pass A: theorem extractor
     - read the Lean modules only
     - reconstruct the negative result using theorem names and exact statements
     - do not rely on natural-language handoff prose
  2. Agent/Pass B: adversarial skeptic
     - assume the negative verdict may be overstated
     - search for hidden assumptions, noncanonical choice points, weakened quantifiers, and theorem-statement loopholes
     - explicitly try to argue that the route might still be intrinsic
  3. Agent/Pass C: claim auditor
     - compare the English verdict against actual Lean theorem boundaries line by line
     - mark every claim as: proved / interpretation / too strong
  4. Main synthesis pass:
     - compare A, B, and C
     - keep the negative verdict only if it survives the adversarial pass
     - if there is disagreement, downgrade the claim rather than smoothing over it

Stop conditions:
- if any pass finds that the current negative verdict relies on prose stronger than the Lean theorems, you must weaken the verdict
- if any pass finds a genuine theorem-level loophole, do not repeat the current conclusion unchanged
- do not treat repeated handoff wording as evidence

Required commands during this re-check:
- `cd C:\\sct-lean && lake build`
- inspect the actual Lean files for the negative side, especially:
  - `BoundaryBranchingEdgePairwise.lean`
  - `BoundaryBranchingEdgeConflict.lean`
  - `BoundaryTriangleWitnessFlexibility.lean`
  - `BoundarySharedEdgeWitnessFlexibility.lean`
  - `BoundarySharedEdgeTauVariants.lean`

The re-check target is narrow:
- not whether SCT fails
- not whether every intrinsic route fails
- but whether the present finite-nerve overlap-only witness-induced coherence route is genuinely non-intrinsic in the theorem-level sense already claimed

Most important Lean files for understanding the current result:
- F:/Black Mesa Research Facility/Main Facility/Physics department/SCT Theory/theory/lean/SCTLean/FND1/BoundaryCompatibleExistence.lean
- F:/Black Mesa Research Facility/Main Facility/Physics department/SCT Theory/theory/lean/SCTLean/FND1/BoundaryCompatibilityEquiv.lean
- F:/Black Mesa Research Facility/Main Facility/Physics department/SCT Theory/theory/lean/SCTLean/FND1/BoundaryHomologyStructures.lean
- F:/Black Mesa Research Facility/Main Facility/Physics department/SCT Theory/theory/lean/SCTLean/FND1/BoundaryHomologyH1.lean
- F:/Black Mesa Research Facility/Main Facility/Physics department/SCT Theory/theory/lean/SCTLean/FND1/BoundaryBranchingEdgePairwise.lean
- F:/Black Mesa Research Facility/Main Facility/Physics department/SCT Theory/theory/lean/SCTLean/FND1/BoundaryTriangleWitnessFlexibility.lean
- F:/Black Mesa Research Facility/Main Facility/Physics department/SCT Theory/theory/lean/SCTLean/FND1/BoundarySharedEdgeTauVariants.lean

Most likely good next moves:
1. switch to another live FND-1 route and reuse this infrastructure
2. or write a technical preprint from this branch:
   "Auxiliary Boundary Data and the Failure of Intrinsic Coherence in a Finite-Nerve Route for Spectral Causal Theory"

If you continue technical work on this branch, your burden is high:
- you must either produce a genuinely more intrinsic law than the current branching-edge coherence language
- or produce a cleaner impossibility/obstruction theorem
- do not just add more packaging layers

Your first responsibility is to preserve theorem-boundary honesty.
```

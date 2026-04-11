import SCTLean.Basic
import SCTLean.CJBridgeEstimator
import SCTLean.MT1
import SCTLean.FND1.BoundaryComposition
import SCTLean.FND1.BoundaryCompatibility
import SCTLean.FND1.BoundaryCanonicalObstruction
import SCTLean.FND1.BoundaryEndpointChoice
import SCTLean.FND1.BoundaryChoiceCompatibility
import SCTLean.FND1.BoundaryGlobalChoiceCompatibility
import SCTLean.FND1.BoundaryTriangleCases
import SCTLean.FND1.BoundaryEntryCases
import SCTLean.FND1.BoundaryWitnessCoefficients
import SCTLean.FND1.BoundaryWitnessIndices
import SCTLean.FND1.BoundaryWitnessRowCoefficients
import SCTLean.FND1.BoundaryWitnessCompositionA
import SCTLean.FND1.BoundaryWitnessCompositionBC
import SCTLean.FND1.BoundaryWitnessTriangleComposition
import SCTLean.FND1.BoundaryWitnessAnyRow
import SCTLean.FND1.BoundaryGlobalCompositionZero
import SCTLean.FND1.BoundaryChoiceToLocalCompatibility
import SCTLean.FND1.BoundaryCompatibleExistence
import SCTLean.FND1.BoundaryChainComplexExistence
import SCTLean.FND1.BoundaryCompatibilityEquiv
import SCTLean.FND1.BoundaryTriangleCoherence
import SCTLean.FND1.BoundaryTriangleCoherenceEquiv
import SCTLean.FND1.BoundaryTriangleEdgeCoherence
import SCTLean.FND1.BoundaryTriangleEdgeToLocal
import SCTLean.FND1.BoundaryTriangleEdgeGluing
import SCTLean.FND1.BoundaryTriangleLocalExtension
import SCTLean.FND1.BoundaryTriangleOverlapCoherence
import SCTLean.FND1.BoundaryTriangleBoundaryEdgeGluing
import SCTLean.FND1.BoundaryTriangleEdgeStarCoherence
import SCTLean.FND1.BoundaryPureTriangleOverlapCoherence
import SCTLean.FND1.BoundaryPureTriangleOverlapInterfaces
import SCTLean.FND1.BoundaryPureTriangleOverlapVacuous
import SCTLean.FND1.BoundaryBranchingEdgeGluing
import SCTLean.FND1.BoundaryBranchingEdgeConflict
import SCTLean.FND1.BoundaryBranchingEdgeChoice
import SCTLean.FND1.BoundaryBranchingEdgeTwoStar
import SCTLean.FND1.BoundaryBranchingEdgeTwoStarGlobal
import SCTLean.FND1.BoundaryBranchingEdgePairwise
import SCTLean.FND1.BoundaryTriangleWitnessFlexibility
import SCTLean.FND1.BoundarySharedEdgeWitnessFlexibility
import SCTLean.FND1.BoundarySharedEdgeTauVariants
import SCTLean.FND1.BoundaryChainMaps
import SCTLean.FND1.BoundaryHomologyPrelude
import SCTLean.FND1.BoundaryHomologyStructures
import SCTLean.FND1.BoundaryHomologyH1
import SCTLean.FND1.BoundaryHomologyClassEq
import SCTLean.FND1.BoundaryHomologyUse
import SCTLean.FND1.BoundaryHomologyRelabel
import SCTLean.FND1.BoundaryLocalOrientation
import SCTLean.FND1.BoundaryOrderedTriangle
import SCTLean.FND1.BoundaryTriangleChoice
import SCTLean.FND1.GreedyAntichain
import SCTLean.FND1.BoundaryIncidence
import SCTLean.FND1.BoundaryOrientation
import SCTLean.FND1.BoundarySignedIncidence
import SCTLean.FND1.BoundaryTable
import SCTLean.FND1.BoundarySupport
import SCTLean.FND1.FiniteNerve
import SCTLean.FND1.NerveDiagnostics
import SCTLean.FND1.OrderTime
import SCTLean.FND1.SimplicialNerve
import SCTLean.FND1.TriangleLayer
import SCTLean.FormFactors
import SCTLean.SpectralAction
import SCTLean.Tensors
import SCTLean.StandardModel

/-!
# SCT Lean — Formal Verification Library for Spectral Causal Theory

This library provides Lean 4 formalizations for SCT Theory,
leveraging PhysLean (Lorentz group, SM, tensors) and Mathlib4
(manifolds, spectral theory, measures).

## Modules
- `SCTLean.Basic`: Core definitions, physical constants, conventions
- `SCTLean.FormFactors`: Heat kernel form factor identities
- `SCTLean.SpectralAction`: Spectral action functional properties
- `SCTLean.Tensors`: Curvature tensor identities (Weyl, Ricci decomposition)
- `SCTLean.StandardModel`: SM degrees of freedom and anomaly cancellation
-/

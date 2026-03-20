# A1 Formal Alignment

This note aligns the live `A1` numerical pipeline with the current Lean
formalization. It is intended to prevent silent drift between:

- `speculative/numerics/a1_nerve/*`
- `theory/lean/SCTLean/FND1/*`

The rule is simple:

> no numerical object should be described as "formalized" unless there is an
> actual Lean object or theorem matching its mathematical content.

## Status Labels

- `FORMALIZED`
- `PARTIALLY ALIGNED`
- `NUMERICAL ONLY`
- `BENCHMARK ONLY`

## Alignment Table

| Numerical object | Lean object / theorem | Status | Notes |
|---|---|---|---|
| `greedy_antichain_from_candidates(matrix, candidates)` | `greedyAntichainFromCandidates`, `GreedyAntichainCorrectness` | `PARTIALLY ALIGNED` | Lean proves subset + pairwise incomparability, but not exact implementation identity with the Python traversal behavior. |
| `maximal_antichain(matrix)` | none | `NUMERICAL ONLY` | The current formal core does not yet treat the whole-causet greedy/global selector as a canonical theorem target. |
| `order_time_coordinate(matrix)` | `orderTimeCoordinate`, `pastCount_relabel`, `futureCount_relabel`, `OrderTimeCoordinateEquivariance` | `FORMALIZED` | This is the strongest current causal-set-native selector ingredient. |
| `order_time_layer_candidates(matrix, width_fraction)` | `OrderTimeCandidateSetEquivariance` | `PARTIALLY ALIGNED` | Lean formalizes interval-style candidate sets, not yet the exact Python median-window implementation. |
| `order_time_max_layer_candidates(matrix, num_bins)` | none | `NUMERICAL ONLY` | Bin-maximizing candidate pools are not yet formalized. |
| `middle_layer_candidates(points, width_fraction)` | none | `BENCHMARK ONLY` | Depends on `u+v` embedding coordinates and is outside the causal-set-native formal core. |
| `max_cardinality_layer_candidates(points, num_bins)` | none | `BENCHMARK ONLY` | Same issue: useful diagnostically, not formalized as native structure. |
| `select_antichain(...)` | none | `NUMERICAL ONLY` | The dispatch layer is implementation glue, not a theorem-level object. |
| `thicken_antichain(matrix, antichain, v)` | none | `NUMERICAL ONLY` | Not yet formalized; a natural future target after the simplicial nerve itself is formal. |
| `build_cover_from_thickening(...)` | none | `NUMERICAL ONLY` | The cover exists numerically, but not yet as a Lean object with invariance theorems. |
| `build_nerve(...)` as finite simplicial object | `vertexSimplices`, `edgeSimplices`, `triangleSimplices`, `triangleSimplices_eq_triangleFinset`, `codimOneFaces`, `edge_boundary_faces_are_vertices`, `triangle_boundary_faces_are_edges` | `PARTIALLY ALIGNED` | The 0/1/2-simplex layer and raw codimension-one face support are formalized. Oriented boundary operators, matrix representations, and higher simplicial closure are not yet formalized. |
| overlap size between cells | `pairOverlap`, `pairOverlap_relabel`, `pairOverlap_symm` | `FORMALIZED` | First raw invariant layer. |
| adjacency test | `Adjacent`, `adjacent_relabel`, `adjacent_symm` | `FORMALIZED` | This is the formal graph-level relation on the finite nerve cover. |
| local degree | `degreeOf`, `degreeOf_relabel` | `FORMALIZED` | Matches the graph-level degree diagnostics. |
| total directed degree | `sumDegrees`, `sumDegrees_relabel` | `FORMALIZED` | Used in density and mean-degree formulas. |
| maximum degree | `maxDegree`, `maxDegree_relabel` | `FORMALIZED` | Available in Lean even if not yet central in the Python benchmark gate. |
| `edge_overlap_sizes(...)` | `directedOverlapSum`, `directedOverlapSum_relabel` | `PARTIALLY ALIGNED` | Lean formalizes the global weighted sum, not the full exported list-valued diagnostic. |
| `nerve_diagnostics(...).mean_degree` | `meanDegree`, `meanDegree_relabel` | `FORMALIZED` | Directly aligned. |
| `nerve_diagnostics(...).edge_density` | `edgeDensity`, `edgeDensity_relabel` | `FORMALIZED` | Directly aligned. |
| `nerve_diagnostics(...).mean_edge_overlap` | `meanEdgeOverlap`, `meanEdgeOverlap_relabel` | `FORMALIZED` | Directly aligned at the scalar summary level. |
| `nerve_diagnostics(...).max_edge_overlap` | `maxEdgeOverlap`, `maxEdgeOverlap_relabel` | `FORMALIZED` | Newly aligned. |
| `nerve_diagnostics(...).num_triangles` | `triangleCount`, `triangleCount_relabel` | `FORMALIZED` | Formalized as order-free 3-subset clique counting. |
| raw codimension-one face support for `boundary_matrix(...)` | `IsCodimOneFace`, `isCodimOneFace_image`, `codimOneFaces`, `codimOneFaces_image` | `PARTIALLY ALIGNED` | The face relation is formalized without orientation or signed incidence matrices. |
| exact unsigned face counts used by `boundary_matrix(...)` support | `card_codimOneFaces`, `edge_boundary_face_count`, `triangle_boundary_face_count` | `FORMALIZED` | The current Lean core already knows that edges have exactly 2 codimension-one faces and triangles have exactly 3. |
| unsigned nonzero support of `boundary_matrix(...)` | `edgeVertexIncidences`, `triangleEdgeIncidences`, `mem_edgeVertexIncidences_iff`, `mem_triangleEdgeIncidences_iff`, `edgeVertexIncidences_relabel`, `triangleEdgeIncidences_relabel` | `FORMALIZED` | The support pairs `(face, simplex)` are formalized as relabeling-invariant objects. Orientation and matrix coefficients are still absent. |
| explicit auxiliary sign layer for `boundary_matrix(...)` | `EdgeVertexOrientationDatum`, `TriangleEdgeOrientationDatum`, `edgeVertexCoefficient_nonzero_iff`, `triangleEdgeCoefficient_nonzero_iff`, `relabelEdgeVertexOrientationDatum`, `relabelTriangleEdgeOrientationDatum` | `PARTIALLY ALIGNED` | Lean now has explicit external sign data over the canonical support, together with relabeling transport. This is an honest precursor to signed incidence matrices, but not yet a canonical or packaged matrix representation. |
| packaged signed incidence object underlying `boundary_matrix(...)` | `SignedIncidenceData`, `edgeVertexSignedIncidence`, `triangleEdgeSignedIncidence` | `PARTIALLY ALIGNED` | Lean now packages support plus signed coefficients into a single object. This is still pre-matrix: no row/column indexing discipline or linear operator has been introduced yet. |
| finite row/column boundary table underlying `boundary_matrix(...)` | `BoundaryTableData`, `edgeVertexBoundaryTable`, `triangleEdgeBoundaryTable` | `PARTIALLY ALIGNED` | Lean now has explicit row/column index sets and entry functions. This layer remains useful as the boundary between raw combinatorics and matrix packaging. |
| `boundary_matrix(...)` as matrix-valued object | `BoundaryTableData.toMatrix`, `edgeVertexBoundaryMatrix`, `triangleEdgeBoundaryMatrix` | `PARTIALLY ALIGNED` | Lean now has honest matrix-valued boundary objects with explicit index types and nonzero-entry theorems. What is still missing is the chain-complex composition law and the later homological/spectral layers. |
| first chain-level composition `∂₁ ∘ ∂₂` | `edgeTriangleBoundaryComposition`, `edgeTriangleBoundaryComposition_entry_nonzero_has_witness`, `edgeTriangleBoundaryComposition_entry_nonzero_has_support` | `PARTIALLY ALIGNED` | Lean now has the composed entry formula and a precise theorem saying any nonzero composed entry must pass through a real intermediate edge. This still falls short of `∂² = 0`. |
| compatibility axiom sufficient for `∂² = 0` | `BoundaryOrientationCompatibility`, `edgeTriangleBoundaryComposition_eq_zero_of_compatible`, `edgeTriangleBoundaryComposition_matrix_eq_zero_of_compatible` | `PARTIALLY ALIGNED` | Lean now has an explicit theorem-level bridge from a stated compatibility law on auxiliary orientation data to vanishing boundary composition. What is still missing is deriving that compatibility from a more primitive orientation structure rather than assuming it. |
| local simplex-face orientation layer | `BoundaryLocalOrientationDatum`, `inducedEdgeVertexOrientationDatum`, `inducedTriangleEdgeOrientationDatum`, `BoundaryLocalOrientationCompatibility` | `PARTIALLY ALIGNED` | Lean now assigns signs locally on codimension-one faces of edges and triangles, and then induces the previous global orientation data from those local rules. This is more primitive than global support-pair signs, but still not yet canonical. |
| low-dimensional obstruction to canonical edge orientation | `LocalFaceDatumInvariantUnder`, `pairFaceSignSum_ne_zero_of_swap_invariant`, `no_swap_invariant_alternating_pair_orientation` | `FORMALIZED` | Lean now proves a precise edge-level obstruction: on an unordered 2-element simplex, a swap-invariant local face-sign rule cannot produce the alternating endpoint signs required by an honest oriented edge boundary. This is a bounded low-dimensional obstruction, not a full no-go for all higher-dimensional orientation mechanisms. |
| minimal explicit asymmetry datum for edge orientation | `localOrientationFromChosenFace`, `leftChosenPairOrientation_alternating`, `rightChosenPairOrientation_alternating`, `EdgeEndpointChoiceDatum`, `inducedEdgeLocalOrientations` | `FORMALIZED` | Lean now also proves the constructive complement: one explicit chosen codimension-one face per edge is enough to define a local alternating edge boundary. This is honest extra data, not a canonical rule derived from intrinsic simplex symmetry. |
| global existence of edge-level asymmetry datum | `localFaceSupport_nonempty_of_edge`, `someEdgeEndpointChoiceDatum`, `edgeEndpointChoiceDatum_exists` | `FORMALIZED` | Lean now proves that the minimal edge-level extra datum exists on the whole finite nerve: each edge has at least one codimension-one face, so one may choose one globally by classical choice. This is an existence result, not a canonical construction. |
| local ordered-triangle compatibility witness | `orderedEdgeOrientation`, `orderedTriangleOrientation`, `orderedTriangleVertexSumAtA_eq_zero`, `orderedTriangleVertexSumAtB_eq_zero`, `orderedTriangleVertexSumAtC_eq_zero`, `orderedTriangle_local_boundary_square_zero` | `FORMALIZED` | Lean now proves a concrete sufficient local mechanism for `∂₁ ∘ ∂₂ = 0` on one triangle: if a triangle carries an explicit ordered triple of distinct vertices, the standard `bc - ac + ab` face pattern is exactly compatible with the induced ordered edge boundaries. This is still local and stronger than the desired global intrinsic construction. |
| global existence of triangle-level ordered witnesses | `TriangleOrderedWitness`, `triangleOrderedWitness_exists`, `TriangleOrderedChoiceDatum`, `someTriangleOrderedChoiceDatum`, `triangleOrderedChoiceDatum_exists`, `TriangleOrderedWitness.local_boundary_square_zero` | `FORMALIZED` | Lean now proves that every actual triangle simplex admits a noncanonical ordered witness, so the stronger local cancellation mechanism can be chosen globally across the triangle layer by classical choice. This still does not solve compatibility with independently chosen global edge data. |
| triangle-by-triangle bridge from edge choices to cancellation | `EdgeChoicesCompatibleWithTriangleWitness`, `compatible_edgeAB_left_coeff`, `compatible_edgeAB_right_coeff`, `compatible_edgeAC_left_coeff`, `compatible_edgeAC_right_coeff`, `compatible_edgeBC_left_coeff`, `compatible_edgeBC_right_coeff`, `compatible_triangle_local_boundary_square_zero` | `FORMALIZED` | Lean now proves the first real bridge between the two global choice layers: if the global edge-choice datum agrees with a triangle's ordered witness on its three boundary edges, then the induced edge signs and ordered triangle signs satisfy exact vertexwise cancellation on that triangle. This is still local, not a whole-nerve compatibility theorem. |
| global noncanonical choice-induced local orientation layer | `GlobalEdgeTriangleChoiceCompatibility`, `choiceInducedBoundaryLocalOrientationDatum`, `globalChoiceCompatibility_triangle_local_boundary_square_zero` | `FORMALIZED` | Lean now packages the two global noncanonical data (`chi` on edges and `tau` on triangles) into one explicit `BoundaryLocalOrientationDatum`, together with a global compatibility predicate whose consequence is trianglewise local cancellation on every triangle. This is still weaker than a derived whole-matrix `BoundaryLocalOrientationCompatibility`. |
| helper case-split layer for whole-nerve compatibility | `vertexSimplex_eq_singleton`, `edgeVertexRowIndex_eq_singleton`, `pairCodimOneFace_cases`, `TriangleOrderedWitness.codimOneFace_cases`, `TriangleOrderedWitness.boundaryEdge_cases`, `TriangleOrderedWitness.edgeAB_face_cases`, `TriangleOrderedWitness.edgeAC_face_cases`, `TriangleOrderedWitness.edgeBC_face_cases` | `FORMALIZED` | Lean now has the exact finite combinatorial reductions needed before a whole-nerve theorem can be attempted: every row index is a singleton vertex, every codimension-one face of a witness triangle is one of the three expected boundary edges `ab`, `ac`, or `bc`, and every codimension-one face of those witness edges is one of their two singleton endpoints. |
| coefficient-level vanishing on incompatible witness edges | `singletonVertex_mem_edge_of_edgeVertexSupport`, `singletonVertex_not_mem_edge_of_no_support`, `edgeVertexCoefficient_eq_zero_of_singleton_row_not_mem_edge`, `TriangleOrderedWitness.edgeBC_entry_zero_at_rowA`, `TriangleOrderedWitness.edgeAC_entry_zero_at_rowB`, `TriangleOrderedWitness.edgeAB_entry_zero_at_rowC` | `FORMALIZED` | Lean now proves the first direct coefficient-level zero statements needed for reducing global composition entries: if a singleton row vertex is not an endpoint of a given witness boundary edge, then the corresponding edge-to-vertex coefficient is exactly zero. |
| exact witness-edge triangle coefficients | `TriangleOrderedWitness.edgeAB_ne_edgeAC`, `TriangleOrderedWitness.edgeAB_ne_edgeBC`, `TriangleOrderedWitness.edgeAC_ne_edgeBC`, `triangleEdgeCoefficient_eq_zero_of_not_witness_boundary_edge`, `choiceInduced_triangleEdgeCoeff_AB`, `choiceInduced_triangleEdgeCoeff_AC`, `choiceInduced_triangleEdgeCoeff_BC` | `FORMALIZED` | Lean now isolates the triangle-to-edge coefficient layer needed for whole-nerve entry reduction: the three witness boundary edges are pairwise distinct, any other edge gets coefficient `0`, and the choice-induced local triangle orientation carries the exact `ab/ac/bc` pattern `+1,-1,+1`. |
| explicit witness rows and triangle column | `witnessVertexRow`, `TriangleOrderedWitness.rowA`, `TriangleOrderedWitness.rowB`, `TriangleOrderedWitness.rowC`, `witnessTriangleCol`, `TriangleOrderedWitness.rowA_ne_rowB`, `TriangleOrderedWitness.rowA_ne_rowC`, `TriangleOrderedWitness.rowB_ne_rowC` | `FORMALIZED` | Lean now has explicit singleton row indices for the three witness vertices and the corresponding triangle column under the current choice-induced data. This removes future dependence on ad hoc `Subtype` terms in composition-entry proofs. |
| exact witness-row edge coefficients for `∂₁` | `choiceInduced_edgeVertexCoeff_rowA_edgeAB`, `choiceInduced_edgeVertexCoeff_rowA_edgeAC`, `choiceInduced_edgeVertexCoeff_rowA_edgeBC`, `choiceInduced_edgeVertexCoeff_rowB_edgeAB`, `choiceInduced_edgeVertexCoeff_rowB_edgeAC`, `choiceInduced_edgeVertexCoeff_rowB_edgeBC`, `choiceInduced_edgeVertexCoeff_rowC_edgeAB`, `choiceInduced_edgeVertexCoeff_rowC_edgeAC`, `choiceInduced_edgeVertexCoeff_rowC_edgeBC` | `FORMALIZED` | Lean now isolates the edge-to-vertex coefficient layer matching the witness rows `A/B/C`: the compatible witness edges carry the exact signs expected from the ordered/local compatibility picture, and the incompatible edge-row pairs vanish exactly. |
| first witness-level composition-entry reduction (`rowA`) | `witnessRowAContribution`, `witnessRowAContribution_edgeAB`, `witnessRowAContribution_edgeAC`, `witnessRowAContribution_edgeBC`, `witnessRowAContribution_eq_zero_of_not_witness_boundary_edge`, `witnessRowA_compositionEntry_eq_zero`, `witnessRowA_composition_eq_zero` | `FORMALIZED` | Lean now carries the first real reduction of a global `∂₁ ∘ ∂₂` entry to finite witness arithmetic: for the singleton row `A` of a witness triangle, all intermediate edges outside `ab/ac/bc` vanish, the three witness-edge contributions are computed exactly, and the resulting composition entry is proved to be zero under the current global choice-compatibility hypothesis. This is still a one-row theorem, not a whole-nerve composition theorem. |
| witness-level composition-entry reductions (`rowB`, `rowC`) | `witnessRowBContribution`, `witnessRowBContribution_edgeAB`, `witnessRowBContribution_edgeAC`, `witnessRowBContribution_edgeBC`, `witnessRowBContribution_eq_zero_of_not_witness_boundary_edge`, `witnessRowB_compositionEntry_eq_zero`, `witnessRowB_composition_eq_zero`, `witnessRowCContribution`, `witnessRowCContribution_edgeAB`, `witnessRowCContribution_edgeAC`, `witnessRowCContribution_edgeBC`, `witnessRowCContribution_eq_zero_of_not_witness_boundary_edge`, `witnessRowC_compositionEntry_eq_zero`, `witnessRowC_composition_eq_zero` | `FORMALIZED` | Lean now proves the parallel witness-entry vanishing theorems for the remaining singleton rows `B` and `C`. Together with the earlier `rowA` theorem, the three vertexwise witness entries of `∂₁ ∘ ∂₂` are now all zero under the current global choice-compatibility hypothesis. This is still not yet packaged as a single whole-nerve matrix theorem. |
| packaged local witness-triangle composition theorem | `witnessTriangle_local_composition_zero`, `witnessTriangle_local_composition_zero_entries` | `FORMALIZED` | Lean now packages the three witness-row entry theorems into a single triangle-level statement: for a fixed witness triangle under global choice compatibility, all three singleton witness rows `A/B/C` have vanishing composed boundary entries. This is a clean local theorem, but it still does not cover arbitrary row indices or yield a whole-nerve matrix theorem by itself. |
| arbitrary row against a witness triangle column | `edgeVertexCoeff_eq_zero_of_row_vertex_not_mem_edgeAB`, `edgeVertexCoeff_eq_zero_of_row_vertex_not_mem_edgeAC`, `edgeVertexCoeff_eq_zero_of_row_vertex_not_mem_edgeBC`, `witnessTriangleCompositionEntry_eq_zero_of_row_vertex_not_mem_triangle`, `anyRow_witnessTriangleCompositionEntry_eq_zero` | `FORMALIZED` | Lean now removes the last row-specific gap at the witness-triangle level: for a fixed witness triangle column, the composed boundary entry vanishes for any row index. The proof is by exact singleton-row case split: either the row is one of `A/B/C`, or its vertex lies outside the triangle and all surviving witness-edge contributions vanish. This is much closer to a real local matrix theorem, but it is still column-local rather than whole-nerve-global. |
| whole-matrix vanishing for the current noncanonical choice-induced data | `choiceInduced_edgeTriangleBoundaryCompositionEntry_eq_zero`, `choiceInduced_edgeTriangleBoundaryComposition_eq_zero` | `FORMALIZED` | Lean now proves the full matrix-level theorem for the current noncanonical construction: under `GlobalEdgeTriangleChoiceCompatibility`, every entry of the composed boundary vanishes, hence the whole matrix `∂₁ ∘ ∂₂` is zero for the choice-induced data. This is a real whole-nerve theorem for the present auxiliary data, but still not a canonical intrinsic orientation theorem. |
| bridge from global choice compatibility to local-face compatibility | `choiceInduced_boundaryLocalOrientationCompatible` | `FORMALIZED` | Lean now shows that the global noncanonical choice-compatibility predicate already induces the previously defined `BoundaryLocalOrientationCompatibility` object for the corresponding choice-induced local orientation datum. This removes one layer of duplicated language in the formal stack, but it does not remove the auxiliary choice data themselves. |
| existential compatible local-orientation layer | `HasCompatibleLocalOrientation`, `hasCompatibleLocalOrientation_of_globalChoiceCompatible`, `exists_boundarySquareZero_of_hasCompatibleLocalOrientation`, `exists_boundarySquareZero_of_globalChoiceCompatible` | `FORMALIZED` | Lean now packages the current noncanonical construction at the right existential level: downstream statements can quantify over the existence of a compatible local orientation datum instead of carrying explicit `chi/tau` data everywhere. |
| minimal chain-level existence statement | `HasBoundarySquareZero`, `hasBoundarySquareZero_of_hasCompatibleLocalOrientation`, `hasBoundarySquareZero_of_globalChoiceCompatible` | `FORMALIZED` | Lean now isolates the smallest chain-complex consequence already justified by the current stack: there exist oriented boundary data on the finite nerve whose composition vanishes. This is still existential and noncanonical. |
| compatibility language equals matrix language | `boundaryOrientationCompatibility_iff_matrix_eq_zero`, `boundaryLocalOrientationCompatibility_iff_matrix_eq_zero`, `hasCompatibleLocalOrientation_iff_exists_localBoundarySquareZero` | `FORMALIZED` | Lean now proves that, at the current formal level, the compatibility predicates are exactly equivalent to the corresponding matrix-zero statements. This removes another layer of duplicated phrasing without upgrading the result to a canonical theorem. |
| chain-map action on finite chains | `edgeBoundaryMap`, `triangleBoundaryMap`, `edgeBoundaryMap_comp_triangleBoundaryMap_eq_zero_of_matrix_zero`, `edgeBoundaryMap_comp_triangleBoundaryMap_eq_zero_of_compatible`, `edgeBoundaryMap_comp_triangleBoundaryMap_eq_zero_of_local_compatible`, `exists_chainMaps_boundary_of_boundary_zero_of_hasBoundarySquareZero`, `exists_chainMaps_boundary_of_boundary_zero_of_globalChoiceCompatible` | `FORMALIZED` | Lean now turns the boundary matrices into actual maps on integer-valued chains and proves the first direct homological precursor: every `2`-boundary is a `1`-cycle for the current square-zero boundary data. This is still pre-homology: kernels, images, and Betti interpretations are not yet formalized. |
| explicit cycle/boundary predicates | `IsOneCycle`, `IsOneBoundary`, `oneBoundary_is_oneCycle_of_matrix_zero`, `oneBoundary_is_oneCycle_of_compatible`, `oneBoundary_is_oneCycle_of_local_compatible` | `FORMALIZED` | Lean now states the first genuine homology-prelude theorem in direct cycle/boundary language: every `1`-boundary is a `1`-cycle under the current square-zero boundary condition. Quotient homology groups and Betti numbers are still not formalized. |
| structural `Z₁/B₁` layer | `edgeBoundaryLinearMap`, `triangleBoundaryLinearMap`, `oneCyclesSubmodule`, `oneBoundariesSubmodule`, `mem_oneCyclesSubmodule_iff`, `mem_oneBoundariesSubmodule_iff`, `oneBoundaries_le_oneCycles_of_matrix_zero`, `oneBoundaries_le_oneCycles_of_compatible` | `FORMALIZED` | Lean now upgrades the prelude to honest submodule language: `Z₁` is a kernel, `B₁` is a range, and the foundational inclusion `B₁ ≤ Z₁` is proved under the current square-zero condition. This is the first real structural homology layer, still before quotient homology or Betti numbers. |
| quotient-level `H₁` interface | `oneBoundariesInCyclesSubmodule`, `FirstHomology`, `firstHomologyMkQ`, `boundaryAsCycle`, `boundaryAsCycle_class_eq_zero`, `oneBoundary_class_eq_zero`, `boundaryAsCycle_class_eq_zero_of_matrix_zero`, `boundaryAsCycle_class_eq_zero_of_compatible`, `oneBoundary_class_eq_zero_of_compatible` | `FORMALIZED` | Lean now has the first quotient-level homology object: `H₁ = Z₁ / (B₁ ∩ Z₁)` together with the canonical class map and the theorem that every current `1`-boundary defines the zero class. This is still an interface layer: no Betti/rank or Hodge claims are made. |
| equality of classes in `H₁` | `SameFirstHomologyClass`, `sameFirstHomologyClass_refl`, `sameFirstHomologyClass_symm`, `sameFirstHomologyClass_trans`, `sameFirstHomologyClass_iff_sub_mem_boundariesInCycles`, `sameFirstHomologyClass_iff_sub_mem_boundaries`, `sameFirstHomologyClass_zero_iff_boundary` | `FORMALIZED` | Lean now has the first usable relation on `H₁`: two cycles represent the same class iff their difference is a boundary. This is the clean quotient-level criterion needed before any later finite-dimensional or invariant-counting layer. |
| `H₁` use-layer under boundary perturbations | `sameFirstHomologyClass_of_sub_mem_boundariesInCycles`, `sameFirstHomologyClass_of_sub_mem_boundaries`, `sameFirstHomologyClass_add_boundary_right`, `sameFirstHomologyClass_add_boundary_left` | `FORMALIZED` | Lean now has the first direct use-lemmas for the quotient layer: if the difference of two cycles is a boundary then the classes coincide, and adding a current boundary on either side does not change the `H₁` class. This is still pre-Betti and pre-relabeling transport. |
| relabel transport on chains and `H₁` class language | `edgeBoundaryIndexEquiv`, `edgeVertexRowIndexEquiv`, `triangleEdgeColIndexEquiv`, `relabelZeroChain`, `relabelOneChain`, `relabelTwoChain`, `edgeBoundaryMap_relabel`, `triangleBoundaryMap_relabel`, `isOneCycle_relabel`, `isOneBoundary_relabel`, `sameFirstHomologyClass_relabel` | `FORMALIZED` | Lean now transports the current finite chain/homology interface across relabeling of cover indices. This does not yet package a full induced `H₁` equivalence, but it already proves that the present cycle, boundary, and same-class statements are preserved when the cover and the auxiliary orientation data are relabeled transparently. |
| triangle-choice coherence replaces primitive edge-choice input | `TriangleOrderedWitness.chosenFaceOnBoundaryEdge`, `GlobalTriangleChoiceCoherence`, `coherentEdgeEndpointChoiceDatum`, `globalChoiceCompatibility_of_triangleChoiceCoherent`, `hasCompatibleLocalOrientation_of_triangleChoiceCoherent`, `hasBoundarySquareZero_of_triangleChoiceCoherent` | `FORMALIZED` | Lean now genuinely weakens the assumption layer: `chi` no longer needs to be primitive input. A coherent global triangle-witness datum `tau` is enough to construct a compatible edge-choice datum and recover the earlier local-orientation and boundary-square-zero consequences. This is still noncanonical and still does not derive `tau` or its coherence intrinsically from the cover itself. |
| coherence is the exact remaining auxiliary assumption | `compatible_choice_eq_triangle_face_val`, `triangleChoiceCoherence_of_globalChoiceCompatible`, `globalChoiceCompatibility_exists_iff_triangleChoiceCoherence` | `FORMALIZED` | Lean now proves the converse direction too: for fixed `tau`, coherence is equivalent to the existence of some compatible edge-choice datum `chi`. So the remaining auxiliary gap is no longer “some mysterious extra edge data”, but precisely the coherence law on triangle witnesses. |
| coherence rewritten in edge-orientation language | `localOrientationFromChosenFace_injective`, `TriangleOrderedWitness.localEdgeOrientationOnBoundaryEdge`, `GlobalTriangleEdgeOrientationCoherence`, `triangleEdgeOrientationCoherence_of_triangleChoiceCoherence`, `triangleChoiceCoherence_of_triangleEdgeOrientationCoherence`, `globalTriangleChoiceCoherence_iff_edgeOrientationCoherence` | `FORMALIZED` | Lean now sharpens the remaining assumption again: the same coherence law can be stated directly as agreement of local edge-orientation rules induced by triangle witnesses on shared edges. This is closer to the eventual boundary-operator language than talking about hidden chosen singleton faces. |
| direct edge-to-local bridge from triangle-induced edge rules | `coherentGlobalEdgeLocalOrientationDatum`, `coherentGlobalEdgeLocalOrientationDatum_eq_triangle_edge_orientation`, `edgeOrientationChoiceInducedBoundaryLocalOrientationDatum`, `boundaryLocalOrientationCompatible_of_triangleEdgeOrientationCoherent`, `hasCompatibleLocalOrientation_of_triangleEdgeOrientationCoherent`, `hasBoundarySquareZero_of_triangleEdgeOrientationCoherent` | `FORMALIZED` | Lean now builds the edge-face part of `BoundaryLocalOrientationDatum` directly from coherent triangle-induced edge orientations. This removes one more layer of indirection: current local compatibility and square-zero existence can now be recovered in a fully operator-facing language, without routing the final statements back through primitive `chi`. |
| edge-orientation coherence reformulated as genuine gluing data | `GluedTriangleEdgeOrientations`, `HasGluedTriangleEdgeOrientations`, `gluedTriangleEdgeOrientationsOfCoherent`, `triangleEdgeOrientationCoherence_of_gluedTriangleEdgeOrientations`, `triangleEdgeOrientationCoherence_iff_hasGluedTriangleEdgeOrientations`, `hasCompatibleLocalOrientation_of_hasGluedTriangleEdgeOrientations`, `hasBoundarySquareZero_of_hasGluedTriangleEdgeOrientations` | `FORMALIZED` | Lean now packages the remaining gap as a real local-to-global statement: there exists one global edge-local orientation datum agreeing with each triangle witness on each boundary edge. This gluing formulation is equivalent to triangle-edge coherence and already suffices to recover the current compatible-local-orientation and boundary-square-zero consequences. |
| remaining gap reformulated as an extension problem | `TriangleWitnessLocalExtension`, `HasTriangleWitnessLocalExtension`, `triangleWitnessLocalExtensionOfGlued`, `gluedTriangleEdgeOrientationsOfLocalExtension`, `hasTriangleWitnessLocalExtension_iff_hasGluedTriangleEdgeOrientations`, `hasTriangleWitnessLocalExtension_iff_triangleEdgeOrientationCoherence`, `hasCompatibleLocalOrientation_of_hasTriangleWitnessLocalExtension`, `hasBoundarySquareZero_of_hasTriangleWitnessLocalExtension` | `FORMALIZED` | Lean now packages the same gap as a true extension problem: the triangle-witness datum `tau` extends to a full `BoundaryLocalOrientationDatum` exactly when the current triangle-edge coherence law holds. This is more operator-facing than the older choice-language and cleaner than carrying a separate `chi` layer. |
| shared-edge coherence reformulated through triangle overlaps | `triangleSimplex_card_eq_three`, `sharedBoundaryEdge_eq_intersection_of_ne`, `GlobalTriangleOverlapCoherence`, `triangleOverlapCoherence_of_triangleEdgeOrientationCoherence`, `triangleEdgeOrientationCoherence_of_triangleOverlapCoherence`, `triangleEdgeOrientationCoherence_iff_triangleOverlapCoherence` | `FORMALIZED` | Lean now proves a clean finite-nerve fact: for two distinct triangles, any shared boundary edge is exactly their set-theoretic intersection. This lets the remaining coherence law be stated in a more intrinsic overlap language on pairs of triangles, rather than only through an externally supplied edge index. |
| shared-edge coherence reformulated with no external edge parameter at all | `triangleOverlapEdge`, `triangleOverlapEdge_val`, `PureTriangleOverlapCoherence`, `pureTriangleOverlapCoherence_of_globalTriangleOverlapCoherence`, `globalTriangleOverlapCoherence_of_pureTriangleOverlapCoherence`, `globalTriangleOverlapCoherence_iff_pureTriangleOverlapCoherence`, `triangleEdgeOrientationCoherence_iff_pureTriangleOverlapCoherence`, `hasCompatibleLocalOrientation_of_pureTriangleOverlapCoherence`, `hasBoundarySquareZero_of_pureTriangleOverlapCoherence` | `FORMALIZED` | Lean now removes the last explicit shared-edge parameter from the overlap formulation itself: coherence can be stated directly on the canonical overlap edge determined by the intersection of two distinct triangles. This is a cleaner intrinsic interface than the earlier overlap-language, though it still depends on noncanonical triangle witnesses. |
| gluing weakened to triangle-bearing edges only | `TriangleBoundaryEdge`, `GluedTriangleBoundaryEdgeOrientations`, `HasGluedTriangleBoundaryEdgeOrientations`, `gluedTriangleBoundaryEdgeOrientationsOfCoherent`, `triangleEdgeOrientationCoherence_iff_hasGluedTriangleBoundaryEdgeOrientations`, `fullGluedTriangleEdgeOrientationsOfBoundaryEdgeGluing`, `hasCompatibleLocalOrientation_of_hasGluedTriangleBoundaryEdgeOrientations`, `hasBoundarySquareZero_of_hasGluedTriangleBoundaryEdgeOrientations` | `FORMALIZED` | Lean now removes another irrelevant piece of auxiliary data: the gluing assumption no longer needs to talk about all edge simplices. It is enough to glue only on edges that actually appear in the boundary of some triangle; isolated edges can be filled arbitrarily afterwards when building a full local boundary-orientation datum. |
| remaining gap localized to edge-stars | `LocalTriangleEdgeStarCoherence`, `localTriangleEdgeStarCoherence_of_gluedTriangleBoundaryEdgeOrientations`, `gluedTriangleBoundaryEdgeOrientationsOfLocalEdgeStarCoherence`, `localTriangleEdgeStarCoherence_iff_hasGluedTriangleBoundaryEdgeOrientations`, `triangleEdgeOrientationCoherence_iff_localTriangleEdgeStarCoherence`, `hasCompatibleLocalOrientation_of_localTriangleEdgeStarCoherence`, `hasBoundarySquareZero_of_localTriangleEdgeStarCoherence` | `FORMALIZED` | Lean now rewrites the same weakened gluing layer as a strictly pointwise local condition: for each triangle-bearing edge, there exists a single local edge-orientation rule agreeing with every triangle in that edge's star. This is equivalent to the current coherence law and already suffices for the present square-zero consequences. |
| witness-order flexibility is not an `AB`-only artifact | `swapABTriangleOrderedWitness`, `swapACTriangleOrderedWitness`, `swapBCTriangleOrderedWitness`, `swapABTriangleOrderedWitness_edgeAB_eq`, `swapACTriangleOrderedWitness_edgeAC_eq`, `swapBCTriangleOrderedWitness_edgeBC_eq`, `chosenFace_edgeAB_differs_under_swapAB`, `chosenFace_edgeAC_differs_under_swapAC`, `chosenFace_edgeBC_differs_under_swapBC`, `edgeABFace`, `swapABEdgeFace`, `edgeABFace_ne_or_swapped_face_ne`, `sharedEdgeAB_witness_flexibility`, `sharedEdgeAB_face_ne_or_swapped_face_ne` | `FORMALIZED` | Lean now shows that witness-order flexibility is present on all three boundary edges of one ordered triangle witness, not just on `AB`: there are explicit swaps preserving `AB`, `AC`, or `BC` respectively while flipping the induced chosen singleton face on that same edge. The later shared-edge obstruction theorem is still written in `AB` language, but it is no longer credible to read it as an `AB` labeling artifact. |
| same-support `tau` variants force a global coherence-or-conflict split | `replaceTriangleWitness`, `replaceTwoTriangleWitnesses`, `sharedEdgeAB_mem_codimOneFaces_triangle2_of_eq`, `chosenFaceOnAB_val_eq_singleton_b_of_eq`, `chosenFaceOnSharedAB_val_eq_singleton_b`, `chosenFaceOnSharedAB_val_eq_singleton_a`, `sharedABBranchingEdge`, `not_both_pairwiseBranchingEdgeChosenFaceCoherenceAt_of_sharedEdge_swap`, `branchingEdgeConflictAt_tau_or_tauSwap_of_sharedEdge_swap`, `branchingEdgeConflictAt_tauSwap_of_pairwiseBranchingEdgeChosenFaceCoherenceAt_of_sharedEdge_swap`, `branchingEdgeConflictAt_tau_of_pairwiseBranchingEdgeChosenFaceCoherenceAt_of_sharedEdge_swap`, `branchingEdgeConflictAt_tauSwap_of_pairwiseBranchingChosenFaceCoherence_of_sharedEdge_swap`, `branchingEdgeConflictAt_tau_of_pairwiseBranchingChosenFaceCoherence_of_sharedEdge_swap`, `branchingEdgeConflictAt_tauSwap_of_pureTriangleOverlapCoherence_of_sharedEdge_swap`, `branchingEdgeConflictAt_tau_of_pureTriangleOverlapCoherence_of_sharedEdge_swap`, `not_both_pairwiseBranchingChosenFaceCoherence_of_sharedEdge_swap`, `not_both_pureTriangleOverlapCoherence_of_sharedEdge_swap`, `branchingEdgeConflict_tau_or_tauSwap_of_sharedEdge_swap`, `branchingEdgeConflict_tauSwap_of_pairwiseBranchingChosenFaceCoherence_of_sharedEdge_swap`, `branchingEdgeConflict_tau_of_pairwiseBranchingChosenFaceCoherence_of_sharedEdge_swap`, `branchingEdgeConflict_tauSwap_of_pureTriangleOverlapCoherence_of_sharedEdge_swap`, `branchingEdgeConflict_tau_of_pureTriangleOverlapCoherence_of_sharedEdge_swap`, `not_pairwiseBranchingChosenFaceCoherence_tauSwap_of_pairwise_of_sharedEdge_swap`, `not_pairwiseBranchingChosenFaceCoherence_tau_of_pairwise_of_sharedEdge_swap`, `not_pureTriangleOverlapCoherence_tauSwap_of_pure_of_sharedEdge_swap`, `not_pureTriangleOverlapCoherence_tau_of_pure_of_sharedEdge_swap` | `FORMALIZED` | Lean now upgrades the shared-edge flexibility signal from a witness-level fact to a global `tau`-datum split: on the same finite nerve support, if one replaces only one triangle witness by its `swapAB` variant along a shared `AB` edge, the original `tau` and the swapped `tau` cannot both satisfy the current global pairwise branching coherence law or the equivalent pure-overlap coherence law. More sharply, if one of the two same-support `tau` variants is globally coherent, then the other must carry a branching-edge conflict, and this conflict can already be pinned to that same shared branching edge. |
| `hodge_laplacian(...)` | none | `NUMERICAL ONLY` | Same dependency: needs a formal simplicial layer first. |
| `betti_numbers(...)` | none | `NUMERICAL ONLY` | Still downstream of the unformalized simplicial/homological layer. |
| `low_spectrum(...)` | none | `NUMERICAL ONLY` | Spectral layer remains outside the current Lean core. |
| `composite_gate.py` | none | `NUMERICAL ONLY` | This is a benchmark decision artifact, not a theorem-level object. |

## What Is Now Safe To Say

The following statements are now justified:

- the order-native time surrogate has a formal equivariance layer;
- the graph-density and overlap diagnostics used in `A1` have a real finite
  invariant core in Lean;
- triangle count is no longer just a numerical convenience and now has a formal
  relabeling-invariant definition;
- the finite simplicial nerve now exists formally at the 0/1/2-simplex level,
  together with raw codimension-one face support.
- there is now a theorem-level low-dimensional obstruction showing that
  canonical local orientation cannot simply be demanded as full swap-invariance
  on an unordered edge while still expecting alternating boundary signs.
- there is now a theorem-level constructive complement showing exactly what
  extra datum is sufficient at the edge level: one chosen codimension-one face
  per edge.
- there is now a theorem-level existence result that such edge data can be
  chosen globally, but only noncanonically.
- there is now a theorem-level local triangle mechanism showing what stronger
  data are sufficient for exact cancellation on a single 2-simplex.
- there is now a theorem-level global existence result for those stronger
  triangle data, again only noncanonically.
- there is now a theorem-level local bridge showing how a triangle witness must
  agree with global edge choices in order to recover cancellation on that
  triangle.
- there is now a theorem-level global packaging of those two noncanonical data
  into one explicit local-orientation object, together with a global
  compatibility predicate whose consequence is trianglewise cancellation on
  every triangle.
- there is now a theorem-level helper case-split layer showing exactly which
  rows and which boundary edges can appear in a future whole-nerve proof:
  rows are singleton vertices, and boundary edges of a witness triangle are
  exactly `ab`, `ac`, or `bc`.
- there is now a theorem-level exact witness-coefficient layer showing that the
  only nonzero triangle-to-edge coefficients on a witness triangle are the
  expected `ab`, `ac`, and `bc` edges with the exact sign pattern `+1,-1,+1`.
- there is now a theorem-level witness-index layer giving canonical singleton
  rows `A/B/C` and the witness-triangle column for the current choice-induced
  boundary data.
- there is now a theorem-level exact witness-row coefficient layer for `∂₁`,
  so both sides of a future composition-entry reduction are now isolated.
- there is now a theorem-level first witness-level composition reduction:
  the `rowA` entry of `∂₁ ∘ ∂₂` for a witness triangle is proved to vanish
  under the current global choice-compatibility hypothesis.
- there are now theorem-level parallel witness-entry reductions for `rowB` and
  `rowC`, so all three singleton witness rows of a chosen triangle now satisfy
  the expected local vanishing law.
- there is now a packaged local witness-triangle theorem collecting those three
  rowwise vanishing statements into one exact triangle-level result.
- there is now a theorem-level arbitrary-row result for a fixed witness
  triangle column, so the local theorem no longer depends on naming the rows
  `A/B/C` by hand.
- there is now a theorem-level whole-matrix vanishing result for the current
  choice-induced noncanonical data: under global compatibility,
  `∂₁ ∘ ∂₂ = 0`.
- there is now a theorem-level bridge from the global choice-compatibility
  language back into the local-orientation compatibility language already used
  elsewhere in the formal stack.
- there is now a theorem-level existential reformulation of that result:
  downstream statements can talk about the existence of a compatible local
  orientation datum, and even directly about the existence of boundary data with
  `∂₁ ∘ ∂₂ = 0`, without exposing the explicit `chi/tau` choice objects.
- there is now a theorem-level equivalence showing that, in the current stack,
  “compatibility” and “matrix vanishing” are not two different claims but two
  exact formulations of the same content.
- there is now a theorem-level chain-map action layer on integer-valued finite
  chains, so `∂₁ ∘ ∂₂ = 0` is no longer just a matrix slogan: it already gives
  the first real homological precursor, namely that every `2`-boundary is a
  `1`-cycle for the current square-zero boundary data.
- there is now a theorem-level cycle/boundary language on top of those chain
  maps, so the first homological implication is no longer only implicit in the
  operator statement.
- there is now a theorem-level structural `Z₁/B₁` layer: cycles are formalized
  as a kernel submodule, boundaries as a range submodule, and the inclusion
  `B₁ ≤ Z₁` is explicit.
- there is now a theorem-level quotient homology interface:
  `H₁ = Z₁ / (B₁ ∩ Z₁)` exists as a formal object, the canonical class map is
  explicit, and every current `1`-boundary is proved to represent the zero
  class.
- there is now a theorem-level equality criterion for `H₁` classes: two cycles
  are homologous exactly when their difference lies in the current boundary
  submodule.
- there is now a theorem-level boundary-perturbation rule for `H₁`: adding a
  current boundary to a cycle does not change its class.
- there is now a theorem-level weakening of the auxiliary assumption layer:
  a coherent triangle-witness datum `tau` is enough to reconstruct a
  compatible edge-choice datum `chi`, so `chi` no longer has to be treated as
  primitive input in order to recover the current `∂₁ ∘ ∂₂ = 0` consequences.
- there is now a theorem-level converse showing that, for fixed `tau`, this
  coherence law is exactly equivalent to the existence of some compatible
  edge-choice datum `chi`. So the remaining noncanonical assumption has been
  isolated sharply.
- there is now a theorem-level edge-orientation reformulation of that same
  assumption, so the remaining gap can be discussed directly in the language
  of local edge boundary rules on shared edges.
- there is now a theorem-level direct bridge from that edge-orientation
  coherence to a concrete `BoundaryLocalOrientationDatum`, so the current
  square-zero existence result can be stated without bouncing back through the
  old endpoint-choice interface.
- there is now a theorem-level same-support `tau`-variant obstruction:
  if two triangles share an `AB` edge and one replaces only one triangle
  witness by its `swapAB` variant, the original `tau` and the swapped `tau`
  cannot both satisfy local pairwise branching coherence at that shared
  branching edge.
- that same-support obstruction is now global too:
  the original `tau` and the swapped `tau` cannot both satisfy the present
  global pairwise branching coherence law or the equivalent pure-overlap
  coherence law; equivalently, one of the two `tau` variants must already
  carry a branching-edge conflict.
- and this now has the sharper conditional form too:
  if one of those two same-support `tau` variants does satisfy the present
  global pairwise branching coherence law, or equivalently the pure-overlap
  coherence law, then the swapped variant must already carry a
  branching-edge conflict.
- equivalently, in that same-support swap setup, global coherence is now known
  to be one-sided: if one variant satisfies the present pairwise or
  pure-overlap law, the swapped variant cannot satisfy that same law.
- and this is no longer only a global black-box statement:
  if one variant is coherent, the swapped variant already conflicts on the
  very same shared branching edge where the witness-order swap was made.
- and the underlying witness-order flexibility is now known on all three
  boundary edges of a triangle witness, not only on `AB`: there are explicit
  `AB`, `AC`, and `BC` preserving swaps that flip the chosen singleton face on
  that same edge.

## What Is Still Unsafe To Say

The following statements remain unjustified:

- the full `A1` pipeline is formalized;
- canonical oriented boundary matrices have been derived from intrinsic simplex
  data alone;
- canonical local orientation has been derived from intrinsic simplex data;
- a full triangle-level compatible local orientation mechanism has been derived
  globally from edge-choice data alone;
- the globally chosen triangle witnesses have already been reconciled with the
  independently chosen edge witnesses into one coherent compatibility law;
- the `rowA` witness-entry theorem already implies whole-matrix `∂₁ ∘ ∂₂ = 0`;
- the three witness-entry theorems already by themselves constitute a whole-nerve theorem;
- the packaged local witness-triangle theorem already covers arbitrary row indices;
- a canonical intrinsic orientation theorem has been derived from the whole-matrix vanishing result;
- the Hodge / Betti / spectral layer is formalized;
- the composite benchmark gate is a theorem;
- the current triangle-witness coherence law is intrinsic or canonical;
- the current triangle-witness coherence law has already been derived from
  cover/nerve data without auxiliary choice;
- `A1` proves geometric emergence.

## Current Theorem Boundary

The formal core currently stops at:

- order-native selection ingredients,
- the finite 0/1/2-simplex nerve object,
- raw codimension-one face support,
- graph-level finite nerve invariants,
- triangle counting,
- relabeling invariance of those quantities,
- an auxiliary signed boundary-table / boundary-matrix layer,
- explicit low-dimensional orientation obstruction and noncanonical choice data,
- trianglewise global-choice compatibility with local cancellation.
- exact witness-level triangle-edge coefficients needed to reduce future global
  composition entries to the local `A/B/C` cancellation sums.
- exact witness-level edge-to-vertex coefficients on rows `A/B/C`;
- clean local witness-entry theorems for `rowA`, `rowB`, and `rowC`;
- a packaged local witness-triangle theorem collecting those reductions;
- a clean arbitrary-row theorem for one fixed witness-triangle column;
- a whole-nerve matrix theorem for the current noncanonical choice-induced data,
  derived from that arbitrary-row fixed-column result;
- an existential reformulation hiding those auxiliary noncanonical choice data
  behind `HasCompatibleLocalOrientation` and `HasBoundarySquareZero`;
- exact equivalences between compatibility predicates and the corresponding
  matrix-zero statements;
- a first chain-action layer on finite chains with “boundary of boundary is
  zero” as an operator statement on chains;
- a first explicit `cycle/boundary` theorem layer for `1`-chains;
- a first structural homology layer with `Z₁` and `B₁` as submodules and the
  inclusion `B₁ ≤ Z₁`;
- a first quotient-level homology interface `H₁ = Z₁ / (B₁ ∩ Z₁)` with the
  canonical class map and zero-class theorems for current boundaries;
- a first quotient-level class-equality criterion for `H₁`, both internally on
  `Z₁` and at the ambient `1`-chain level via the current boundary submodule;
- a first quotient-level use-layer showing that boundary perturbations preserve
  class in `H₁`;
- a first relabeling-transport layer showing that the current chain maps,
  cycle/boundary predicates, and same-class statements on `H₁` survive honest
  relabeling of the cover and transported auxiliary orientation data;
- a first assumption-weakening layer showing that coherent triangle witnesses
  already suffice to reconstruct the previously primitive edge-choice datum and
  hence to recover `HasCompatibleLocalOrientation` and `HasBoundarySquareZero`;
- an exact equivalence showing that, for fixed `tau`, the existence of some
  compatible edge-choice datum is neither weaker nor stronger than the
  triangle-witness coherence law;
- an exact equivalence showing that this remaining coherence law can itself be
  formulated directly in terms of induced local edge-orientation rules on
  shared edges;
- a direct operator-facing construction turning coherent triangle-induced edge
  orientations into the edge-face layer of a `BoundaryLocalOrientationDatum`;
- a genuine gluing-language reformulation of the same remaining gap: existence
  of one global edge-local orientation datum agreeing with all triangle-induced
  local edge rules is equivalent to triangle-edge coherence;
- a still tighter reformulation of that gap as an extension problem: `tau`
  extends to a full local boundary-orientation datum exactly when the same
  triangle-edge coherence law holds;
- a finite-nerve overlap theorem showing that for distinct triangles a shared
  boundary edge is exactly their set-theoretic intersection, and hence the
  remaining coherence law can be phrased directly on triangle overlaps;
- an even cleaner pure-overlap formulation in which the same coherence law is
  stated directly on the canonical overlap edge determined by the triangle
  intersection itself, with no extra external edge parameter in the statement;
- and that pure-overlap formulation is not a disconnected wrapper: it is now
  proved equivalent to the current edge-star, boundary-edge gluing, and local
  extension formulations of the same remaining gap;
- and there is now a first automatic structural regime where that remaining gap
  disappears: if each triangle-bearing edge has a singleton star, then
  pure-overlap coherence is vacuous and the present square-zero consequences
  follow with no extra witness-consistency input;
- and, correspondingly, the minimal nontrivial gluing layer is now narrower
  than before: gluing is needed only on genuinely branching triangle-bearing
  edges, not on singleton stars and not on isolated edges;
- and there is now a concrete local failure witness for the current
  triangle-witness route: one branching-edge conflict already destroys the
  pure-overlap, branching-gluing, edge-star, and local-extension formulations
  for that same `tau`;
- and the complementary positive side is now also in hand: on genuinely
  branching edges the remaining local problem is equivalent to one binary
  chosen-face law per branching edge, which in the exactly-two-triangle star
  case is literally the single binary choice on the shared edge;
- and that first nontrivial branching case is now formalized sharply: for an
  edge with exactly two triangle witnesses, local chosen-face coherence on that
  edge is equivalent to equality of the two witness-induced chosen faces, while
  inequality of those same two chosen faces yields a concrete branching-edge
  conflict for the current `tau`;
- and this exact-two branching regime now also has a global theorem-level
  interface: if every genuinely branching edge has exactly two witnesses and
  the corresponding binary chosen-face agreements hold edge-by-edge, then the
  full branching-edge choice datum exists and the current square-zero route
  follows;
- and the branching-edge layer is now cleaner than that exact-two special case:
  on genuinely branching edges the current minimal assumption can be stated as
  a pure pairwise witness law, namely that all witness-induced chosen faces on
  that edge agree, and this is theorem-level equivalent to the present
  existential branching-edge gluing datum;
- and the obstruction side is now exact at the same granularity: a local
  branching-edge conflict on one genuinely branching edge is theorem-level
  equivalent to failure of that edge's pairwise chosen-face coherence, and
  globally the current `tau` has a branching-edge conflict iff the global
  pairwise branching coherence law fails;
- and there is now a sharper obstruction one layer below branching stars
  themselves: even on a single fixed triangle, swapping the first two vertices
  of an ordered witness preserves the same underlying edge `ab` but changes the
  chosen singleton face on that edge from `{b}` to `{a}`, so witness order is
  already mathematically visible before any higher gluing is imposed;
- and this witness-order obstruction now also lifts to a genuine shared-edge
  theorem: if two witnesses use the same `AB` edge, then one of the two
  versions of the second witness (`w` or `swapAB w`) must disagree with any
  fixed comparison witness on chosen-face data along that same edge, so
  pairwise branching coherence is not forced by overlap support alone;
- an actual weakening of the gluing assumptions: it is enough to glue on
  triangle-bearing edges only, and isolated edges can be filled arbitrarily
  afterwards without changing the current `∂₁ ∘ ∂₂ = 0` consequences;
- a pointwise local reformulation of that weakened gluing layer: the remaining
  gap can now be stated edge-by-edge as a star-coherence condition on each
  triangle-bearing edge;
- but still no theorem deriving the triangle-witness coherence law itself from
  intrinsic cover/nerve data.

It does **not** yet reach:

- derived whole-nerve compatibility of the noncanonical global choice data,
- homology,
- Laplacians,
- low-spectrum behavior,
- benchmark/null separation theorems.

## Recommended Next Synchronization Step

The next bridge between numerics and formalization should be:

1. weaken the current triangle-level auxiliary assumptions further by trying to
   derive the present coherence law on `tau` from simpler local or overlap
   data, rather than taking coherent triangle witnesses as primitive input;
2. then either package the current relabel transport into an induced
   quotient-level `H₁` map/equivalence, or return to the more important
   intrinsic task of shrinking the remaining noncanonical triangle-choice
   assumptions before any invariant counting;
3. only then lift to Betti, Hodge, and spectral quantities.

That is the correct order if we want the formal layer to track the actual `A1`
numerical architecture without hidden gaps.

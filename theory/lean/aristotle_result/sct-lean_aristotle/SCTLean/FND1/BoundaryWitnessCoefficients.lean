import SCTLean.FND1.BoundaryEntryCases

/-!
# FND-1 Boundary Witness Coefficients

This module upgrades the current witness-triangle case lemmas to exact
coefficient statements.

It isolates two kinds of facts needed for the future whole-nerve cancellation
theorem:

- boundary edges of a witness triangle are pairwise distinct,
- triangle-to-edge coefficients are exactly `+1` or `-1` on `ab/ac/bc` and
  vanish on every other edge.
-/

namespace SCT.FND1

universe u v

variable {alpha : Type u}
variable {beta : Type v} [Fintype beta] [DecidableEq beta]
variable [DecidableEq alpha]

theorem TriangleOrderedWitness.edgeAB_ne_edgeAC
    (m : Nat) (U : Cover beta alpha)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    w.edgeAB (beta := beta) m U triangle ≠ w.edgeAC (beta := beta) m U triangle := by
  intro h
  have hset :
      (w.edgeAB (beta := beta) m U triangle).1 =
        (w.edgeAC (beta := beta) m U triangle).1 := by
    exact congrArg Subtype.val h
  have hmem : w.b ∈ ({w.a, w.c} : Finset beta) := by
    have hb_mem_ab : w.b ∈ (w.edgeAB (beta := beta) m U triangle).1 := by
      rw [TriangleOrderedWitness.edgeAB_val (beta := beta) (m := m) (U := U)
        (triangle := triangle) (w := w)]
      simp
    have hb_mem_ac : w.b ∈ (w.edgeAC (beta := beta) m U triangle).1 := by
      exact hset ▸ hb_mem_ab
    rw [TriangleOrderedWitness.edgeAC_val (beta := beta) (m := m) (U := U)
      (triangle := triangle) (w := w)] at hb_mem_ac
    exact hb_mem_ac
  have hcases : w.b = w.a ∨ w.b = w.c := by
    simpa using hmem
  rcases hcases with hba | hbc
  · exact w.hab hba.symm
  · exact w.hbc hbc

theorem TriangleOrderedWitness.edgeAB_ne_edgeBC
    (m : Nat) (U : Cover beta alpha)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    w.edgeAB (beta := beta) m U triangle ≠ w.edgeBC (beta := beta) m U triangle := by
  intro h
  have hset :
      (w.edgeAB (beta := beta) m U triangle).1 =
        (w.edgeBC (beta := beta) m U triangle).1 := by
    exact congrArg Subtype.val h
  have hmem : w.a ∈ ({w.b, w.c} : Finset beta) := by
    have ha_mem_ab : w.a ∈ (w.edgeAB (beta := beta) m U triangle).1 := by
      rw [TriangleOrderedWitness.edgeAB_val (beta := beta) (m := m) (U := U)
        (triangle := triangle) (w := w)]
      simp
    have ha_mem_bc : w.a ∈ (w.edgeBC (beta := beta) m U triangle).1 := by
      exact hset ▸ ha_mem_ab
    rw [TriangleOrderedWitness.edgeBC_val (beta := beta) (m := m) (U := U)
      (triangle := triangle) (w := w)] at ha_mem_bc
    exact ha_mem_bc
  have hcases : w.a = w.b ∨ w.a = w.c := by
    simpa using hmem
  rcases hcases with hab | hac
  · exact w.hab hab
  · exact w.hac hac

theorem TriangleOrderedWitness.edgeAC_ne_edgeBC
    (m : Nat) (U : Cover beta alpha)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1) :
    w.edgeAC (beta := beta) m U triangle ≠ w.edgeBC (beta := beta) m U triangle := by
  intro h
  have hset :
      (w.edgeAC (beta := beta) m U triangle).1 =
        (w.edgeBC (beta := beta) m U triangle).1 := by
    exact congrArg Subtype.val h
  have hmem : w.a ∈ ({w.b, w.c} : Finset beta) := by
    have ha_mem_ac : w.a ∈ (w.edgeAC (beta := beta) m U triangle).1 := by
      rw [TriangleOrderedWitness.edgeAC_val (beta := beta) (m := m) (U := U)
        (triangle := triangle) (w := w)]
      simp
    have ha_mem_bc : w.a ∈ (w.edgeBC (beta := beta) m U triangle).1 := by
      exact hset ▸ ha_mem_ac
    rw [TriangleOrderedWitness.edgeBC_val (beta := beta) (m := m) (U := U)
      (triangle := triangle) (w := w)] at ha_mem_bc
    exact ha_mem_bc
  have hcases : w.a = w.b ∨ w.a = w.c := by
    simpa using hmem
  rcases hcases with hab | hac
  · exact w.hab hab
  · exact w.hac hac

theorem triangleEdgeCoefficient_eq_zero_of_not_witness_boundary_edge
    (m : Nat) (U : Cover beta alpha)
    (omega : TriangleEdgeOrientationDatum (beta := beta) m U)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1)
    (edge : ↑(edgeSimplices m U))
    (hneAB : edge ≠ w.edgeAB (beta := beta) m U triangle)
    (hneAC : edge ≠ w.edgeAC (beta := beta) m U triangle)
    (hneBC : edge ≠ w.edgeBC (beta := beta) m U triangle) :
    triangleEdgeCoefficient (beta := beta) m U omega edge.1 triangle.1 = 0 := by
  apply triangleEdgeCoefficient_eq_zero_of_not_mem
  intro hmem
  have hface : edge.1 ∈ codimOneFaces (beta := beta) triangle.1 :=
    (mem_triangleEdgeIncidences_iff (beta := beta) (m := m) (U := U)).mp hmem |>.2
  rcases TriangleOrderedWitness.boundaryEdge_cases
      (beta := beta) (m := m) (U := U) (triangle := triangle) (w := w) edge hface with
    hAB | hAC | hBC
  · exact hneAB hAB
  · exact hneAC hAC
  · exact hneBC hBC

omit [Fintype beta] [DecidableEq beta] in
theorem ndrec_localFaceSupport_val
    {s t : Finset beta}
    (h : s = t)
    (p : LocalFaceSupport (beta := beta) t) :
    (Eq.ndrec p h.symm).1 = p.1 := by
  cases h
  rfl

omit [Fintype beta] in
theorem TriangleOrderedWitness.localTriangleOrientation_coeff_AB
    {triangle : Finset beta}
    (w : TriangleOrderedWitness triangle) :
    orientationSignValue
      (w.localTriangleOrientation
        (Eq.ndrec
          (motive := fun s => LocalFaceSupport (beta := beta) s)
          (triangleFaceABSupport (beta := beta) w.hab w.hac w.hbc)
          w.triangle_eq.symm)) = 1 := by
  rcases w with ⟨a, b, c, hab, hac, hbc, htri⟩
  cases htri
  simpa [TriangleOrderedWitness.localTriangleOrientation, triangleFaceABSupport] using
    orderedTriangleOrientation_coeff_ab (beta := beta) hab hac hbc

omit [Fintype beta] in
theorem TriangleOrderedWitness.localTriangleOrientation_coeff_AC
    {triangle : Finset beta}
    (w : TriangleOrderedWitness triangle) :
    orientationSignValue
      (w.localTriangleOrientation
        (Eq.ndrec
          (motive := fun s => LocalFaceSupport (beta := beta) s)
          (triangleFaceACSupport (beta := beta) w.hab w.hac w.hbc)
          w.triangle_eq.symm)) = -1 := by
  rcases w with ⟨a, b, c, hab, hac, hbc, htri⟩
  cases htri
  simpa [TriangleOrderedWitness.localTriangleOrientation, triangleFaceACSupport] using
    orderedTriangleOrientation_coeff_ac (beta := beta) hab hac hbc

omit [Fintype beta] in
theorem TriangleOrderedWitness.localTriangleOrientation_coeff_BC
    {triangle : Finset beta}
    (w : TriangleOrderedWitness triangle) :
    orientationSignValue
      (w.localTriangleOrientation
        (Eq.ndrec
          (motive := fun s => LocalFaceSupport (beta := beta) s)
          (triangleFaceBCSupport (beta := beta) w.hab w.hac w.hbc)
          w.triangle_eq.symm)) = 1 := by
  rcases w with ⟨a, b, c, hab, hac, hbc, htri⟩
  cases htri
  simpa [TriangleOrderedWitness.localTriangleOrientation, triangleFaceBCSupport] using
    orderedTriangleOrientation_coeff_bc (beta := beta) hab hac hbc

theorem choiceInduced_triangleEdgeCoeff_AB
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1)
    (hw : tau triangle = w) :
    triangleEdgeCoefficient (beta := beta) m U
      (inducedTriangleEdgeOrientationDatum (beta := beta) m U
        (choiceInducedBoundaryLocalOrientationDatum (beta := beta) m U chi tau))
      (w.edgeAB (beta := beta) m U triangle).1 triangle.1 = 1 := by
  have hmem :
      ((w.edgeAB (beta := beta) m U triangle).1, triangle.1) ∈
        triangleEdgeIncidences (beta := beta) m U := by
    rw [mem_triangleEdgeIncidences_iff]
    constructor
    · exact triangle.2
    · simpa [TriangleOrderedWitness.edgeAB_val, w.triangle_eq] using
        triangleFaceAB_mem_codimOneFaces (beta := beta) w.hab w.hac w.hbc
  rw [triangleEdgeCoefficient_eq_signValue_of_mem (beta := beta) (m := m) (U := U)
    (omega := inducedTriangleEdgeOrientationDatum (beta := beta) m U
      (choiceInducedBoundaryLocalOrientationDatum (beta := beta) m U chi tau)) hmem]
  rw [inducedTriangleEdgeOrientationDatum_eq_local (beta := beta) (m := m) (U := U)
    (theta := choiceInducedBoundaryLocalOrientationDatum (beta := beta) m U chi tau) hmem]
  rw [choiceInducedBoundaryLocalOrientationDatum_triangleFaces_apply
    (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)]
  rw [hw]
  have hface_goal :
      (w.edgeAB (beta := beta) m U triangle).1 ∈ codimOneFaces (beta := beta) triangle.1 := by
    rw [TriangleOrderedWitness.edgeAB_val]
    exact Eq.ndrec
      (motive := fun s => ({w.a, w.b} : Finset beta) ∈ codimOneFaces (beta := beta) s)
      (triangleFaceAB_mem_codimOneFaces (beta := beta) w.hab w.hac w.hbc)
      w.triangle_eq.symm
  have hface_mem :
      ((mem_triangleEdgeIncidences_iff (beta := beta) (m := m) (U := U)).mp hmem).2 = hface_goal := by
    apply Subsingleton.elim
  rw [hface_mem]
  have htransport :
      orientationSignValue
        ((Eq.ndrec
            (motive := fun s => LocalFaceOrientationDatum (beta := beta) s)
            (orderedTriangleOrientation (beta := beta) w.hab w.hac w.hbc)
            w.triangle_eq.symm)
          (Eq.ndrec
            (motive := fun s => LocalFaceSupport (beta := beta) s)
            (triangleFaceABSupport (beta := beta) w.hab w.hac w.hbc)
            w.triangle_eq.symm)) = 1 := by
    simpa [TriangleOrderedWitness.localTriangleOrientation] using
      TriangleOrderedWitness.localTriangleOrientation_coeff_AB (beta := beta) w
  have hface_support_eq :
      (⟨(w.edgeAB (beta := beta) m U triangle).1, hface_goal⟩ :
        LocalFaceSupport (beta := beta) triangle.1) =
      Eq.ndrec
        (motive := fun s => LocalFaceSupport (beta := beta) s)
        (triangleFaceABSupport (beta := beta) w.hab w.hac w.hbc)
        w.triangle_eq.symm := by
    apply Subtype.ext
    rw [ndrec_localFaceSupport_val (beta := beta) (h := w.triangle_eq)]
    simp [TriangleOrderedWitness.edgeAB_val, triangleFaceABSupport]
  rw [hface_support_eq]
  exact htransport

theorem choiceInduced_triangleEdgeCoeff_AC
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1)
    (hw : tau triangle = w) :
    triangleEdgeCoefficient (beta := beta) m U
      (inducedTriangleEdgeOrientationDatum (beta := beta) m U
        (choiceInducedBoundaryLocalOrientationDatum (beta := beta) m U chi tau))
      (w.edgeAC (beta := beta) m U triangle).1 triangle.1 = -1 := by
  have hmem :
      ((w.edgeAC (beta := beta) m U triangle).1, triangle.1) ∈
        triangleEdgeIncidences (beta := beta) m U := by
    rw [mem_triangleEdgeIncidences_iff]
    constructor
    · exact triangle.2
    · simpa [TriangleOrderedWitness.edgeAC_val, w.triangle_eq] using
        triangleFaceAC_mem_codimOneFaces (beta := beta) w.hab w.hac w.hbc
  rw [triangleEdgeCoefficient_eq_signValue_of_mem (beta := beta) (m := m) (U := U)
    (omega := inducedTriangleEdgeOrientationDatum (beta := beta) m U
      (choiceInducedBoundaryLocalOrientationDatum (beta := beta) m U chi tau)) hmem]
  rw [inducedTriangleEdgeOrientationDatum_eq_local (beta := beta) (m := m) (U := U)
    (theta := choiceInducedBoundaryLocalOrientationDatum (beta := beta) m U chi tau) hmem]
  rw [choiceInducedBoundaryLocalOrientationDatum_triangleFaces_apply
    (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)]
  rw [hw]
  have hface_goal :
      (w.edgeAC (beta := beta) m U triangle).1 ∈ codimOneFaces (beta := beta) triangle.1 := by
    rw [TriangleOrderedWitness.edgeAC_val]
    exact Eq.ndrec
      (motive := fun s => ({w.a, w.c} : Finset beta) ∈ codimOneFaces (beta := beta) s)
      (triangleFaceAC_mem_codimOneFaces (beta := beta) w.hab w.hac w.hbc)
      w.triangle_eq.symm
  have hface_mem :
      ((mem_triangleEdgeIncidences_iff (beta := beta) (m := m) (U := U)).mp hmem).2 = hface_goal := by
    apply Subsingleton.elim
  rw [hface_mem]
  have htransport :
      orientationSignValue
        ((Eq.ndrec
            (motive := fun s => LocalFaceOrientationDatum (beta := beta) s)
            (orderedTriangleOrientation (beta := beta) w.hab w.hac w.hbc)
            w.triangle_eq.symm)
          (Eq.ndrec
            (motive := fun s => LocalFaceSupport (beta := beta) s)
            (triangleFaceACSupport (beta := beta) w.hab w.hac w.hbc)
            w.triangle_eq.symm)) = -1 := by
    simpa [TriangleOrderedWitness.localTriangleOrientation] using
      TriangleOrderedWitness.localTriangleOrientation_coeff_AC (beta := beta) w
  have hface_support_eq :
      (⟨(w.edgeAC (beta := beta) m U triangle).1, hface_goal⟩ :
        LocalFaceSupport (beta := beta) triangle.1) =
      Eq.ndrec
        (motive := fun s => LocalFaceSupport (beta := beta) s)
        (triangleFaceACSupport (beta := beta) w.hab w.hac w.hbc)
        w.triangle_eq.symm := by
    apply Subtype.ext
    rw [ndrec_localFaceSupport_val (beta := beta) (h := w.triangle_eq)]
    simp [TriangleOrderedWitness.edgeAC_val, triangleFaceACSupport]
  rw [hface_support_eq]
  exact htransport

theorem choiceInduced_triangleEdgeCoeff_BC
    (m : Nat) (U : Cover beta alpha)
    (chi : EdgeEndpointChoiceDatum (beta := beta) m U)
    (tau : TriangleOrderedChoiceDatum (beta := beta) m U)
    (triangle : ↑(triangleSimplices m U))
    (w : TriangleOrderedWitness triangle.1)
    (hw : tau triangle = w) :
    triangleEdgeCoefficient (beta := beta) m U
      (inducedTriangleEdgeOrientationDatum (beta := beta) m U
        (choiceInducedBoundaryLocalOrientationDatum (beta := beta) m U chi tau))
      (w.edgeBC (beta := beta) m U triangle).1 triangle.1 = 1 := by
  have hmem :
      ((w.edgeBC (beta := beta) m U triangle).1, triangle.1) ∈
        triangleEdgeIncidences (beta := beta) m U := by
    rw [mem_triangleEdgeIncidences_iff]
    constructor
    · exact triangle.2
    · simpa [TriangleOrderedWitness.edgeBC_val, w.triangle_eq] using
        triangleFaceBC_mem_codimOneFaces (beta := beta) w.hab w.hac w.hbc
  rw [triangleEdgeCoefficient_eq_signValue_of_mem (beta := beta) (m := m) (U := U)
    (omega := inducedTriangleEdgeOrientationDatum (beta := beta) m U
      (choiceInducedBoundaryLocalOrientationDatum (beta := beta) m U chi tau)) hmem]
  rw [inducedTriangleEdgeOrientationDatum_eq_local (beta := beta) (m := m) (U := U)
    (theta := choiceInducedBoundaryLocalOrientationDatum (beta := beta) m U chi tau) hmem]
  rw [choiceInducedBoundaryLocalOrientationDatum_triangleFaces_apply
    (beta := beta) (m := m) (U := U) (chi := chi) (tau := tau)]
  rw [hw]
  have hface_goal :
      (w.edgeBC (beta := beta) m U triangle).1 ∈ codimOneFaces (beta := beta) triangle.1 := by
    rw [TriangleOrderedWitness.edgeBC_val]
    exact Eq.ndrec
      (motive := fun s => ({w.b, w.c} : Finset beta) ∈ codimOneFaces (beta := beta) s)
      (triangleFaceBC_mem_codimOneFaces (beta := beta) w.hab w.hac w.hbc)
      w.triangle_eq.symm
  have hface_mem :
      ((mem_triangleEdgeIncidences_iff (beta := beta) (m := m) (U := U)).mp hmem).2 = hface_goal := by
    apply Subsingleton.elim
  rw [hface_mem]
  have htransport :
      orientationSignValue
        ((Eq.ndrec
            (motive := fun s => LocalFaceOrientationDatum (beta := beta) s)
            (orderedTriangleOrientation (beta := beta) w.hab w.hac w.hbc)
            w.triangle_eq.symm)
          (Eq.ndrec
            (motive := fun s => LocalFaceSupport (beta := beta) s)
            (triangleFaceBCSupport (beta := beta) w.hab w.hac w.hbc)
            w.triangle_eq.symm)) = 1 := by
    simpa [TriangleOrderedWitness.localTriangleOrientation] using
      TriangleOrderedWitness.localTriangleOrientation_coeff_BC (beta := beta) w
  have hface_support_eq :
      (⟨(w.edgeBC (beta := beta) m U triangle).1, hface_goal⟩ :
        LocalFaceSupport (beta := beta) triangle.1) =
      Eq.ndrec
        (motive := fun s => LocalFaceSupport (beta := beta) s)
        (triangleFaceBCSupport (beta := beta) w.hab w.hac w.hbc)
        w.triangle_eq.symm := by
    apply Subtype.ext
    rw [ndrec_localFaceSupport_val (beta := beta) (h := w.triangle_eq)]
    simp [TriangleOrderedWitness.edgeBC_val, triangleFaceBCSupport]
  rw [hface_support_eq]
  exact htransport

end SCT.FND1

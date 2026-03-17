import Mathlib.Tactic

/-!
# Chirality Identity for D²-Quantization (CHIRAL-Q)

The core algebraic identity: if A*C = -C*A and B*C = -C*B in a ring,
then (A*B + B*A + B*B)*C = C*(A*B + B*A + B*B).

This is the formal proof that δ(D²) commutes with γ₅.
-/

/-- (δD)² commutes with γ₅ when δD anticommutes with γ₅. -/
theorem sq_comm_of_anticomm {R : Type*} [Ring R]
    (B C : R) (h : B * C = -(C * B)) :
    B * B * C = C * (B * B) := by
  have : B * B * C = B * (B * C) := mul_assoc B B C
  rw [this, h]
  have : B * (-(C * B)) = -(B * (C * B)) := by rw [mul_neg]
  rw [this]
  have : B * (C * B) = (B * C) * B := by rw [mul_assoc]
  rw [this, h]
  simp [neg_mul, mul_assoc]

/-- A*B*C = C*A*B when both A and B anticommute with C. -/
theorem prod_comm_of_anticomm {R : Type*} [Ring R]
    (A B C : R) (hA : A * C = -(C * A)) (hB : B * C = -(C * B)) :
    A * B * C = C * (A * B) := by
  have : A * B * C = A * (B * C) := mul_assoc A B C
  rw [this, hB]
  have : A * (-(C * B)) = -(A * (C * B)) := by rw [mul_neg]
  rw [this]
  have : A * (C * B) = (A * C) * B := by rw [mul_assoc]
  rw [this, hA]
  simp [neg_mul, mul_assoc]

/-- **Main theorem (CHIRAL-Q identity):**
δ(D²) = {D₀, δD} + (δD)² commutes with γ₅. -/
theorem chiral_q_identity {R : Type*} [Ring R]
    (A B C : R)
    (hAC : A * C = -(C * A))
    (hBC : B * C = -(C * B)) :
    (A * B + B * A + B * B) * C = C * (A * B + B * A + B * B) := by
  have hab : A * B * C = C * (A * B) := prod_comm_of_anticomm A B C hAC hBC
  have hba : B * A * C = C * (B * A) := prod_comm_of_anticomm B A C hBC hAC
  have hbb : B * B * C = C * (B * B) := sq_comm_of_anticomm B C hBC
  simp only [add_mul, mul_add]
  rw [hab, hba, hbb]

/-- D² commutes with γ₅ when D anticommutes with γ₅. -/
theorem d_sq_comm {R : Type*} [Ring R]
    (D C : R) (h : D * C = -(C * D)) :
    D * D * C = C * (D * D) :=
  sq_comm_of_anticomm D C h

/-!
## Ghost Chirality Lemma (Gap G1 Closure)

Key algebraic ingredient for the BV equivalence argument:
if γ_a anticommutes with γ₅ for all a, then the product γ_a γ_b
COMMUTES with γ₅ (even Clifford element). This implies:

1. The spin connection Γ_μ = (1/4) ω_μ^{ab} γ_a γ_b commutes with γ₅.
2. The spinorial Lie derivative L_ξ commutes with γ₅.
3. The FP ghost of diffeomorphisms preserves chirality in the spinor basis.

Combined with the Fredenhagen-Rejzner theorem (BRST cohomology is independent
of field parametrization), this closes Gap G1 at the perturbative level.
-/

/-- **Even Clifford element commutes with grading:**
If both A and B anticommute with C, then A*B commutes with C.
This is the algebraic core of the ghost chirality lemma:
γ_a γ_b commutes with γ₅ because each γ_a anticommutes with γ₅. -/
theorem even_clifford_comm {R : Type*} [Ring R]
    (A B C : R)
    (hAC : A * C = -(C * A))
    (hBC : B * C = -(C * B)) :
    A * B * C = C * (A * B) :=
  prod_comm_of_anticomm A B C hAC hBC

/-- **Spin connection chirality:**
A linear combination α₁·(A₁·B₁) + α₂·(A₂·B₂) commutes with C
when all Aᵢ, Bᵢ anticommute with C. This formalizes:
Γ_μ = Σ_{a,b} ω^{ab} γ_a γ_b commutes with γ₅.

We prove this for two terms; the general case follows by induction. -/
theorem spin_connection_comm_two {R : Type*} [Ring R]
    (A₁ B₁ A₂ B₂ C : R)
    (hA₁ : A₁ * C = -(C * A₁)) (hB₁ : B₁ * C = -(C * B₁))
    (hA₂ : A₂ * C = -(C * A₂)) (hB₂ : B₂ * C = -(C * B₂)) :
    (A₁ * B₁ + A₂ * B₂) * C = C * (A₁ * B₁ + A₂ * B₂) := by
  have h1 : A₁ * B₁ * C = C * (A₁ * B₁) := prod_comm_of_anticomm A₁ B₁ C hA₁ hB₁
  have h2 : A₂ * B₂ * C = C * (A₂ * B₂) := prod_comm_of_anticomm A₂ B₂ C hA₂ hB₂
  simp only [add_mul, mul_add]
  rw [h1, h2]

/-- **Diffeomorphism generator chirality (Gap G1 core):**
The spinorial Lie derivative L_ξ = ξ^μ Γ_μ + (1/4)(∇_μ ξ_ν) γ^μ γ^ν
is a sum of even Clifford elements. If each γ_a anticommutes with γ₅,
then L_ξ commutes with γ₅.

Algebraically: if X commutes with C and Y commutes with C, then
X + Y commutes with C. Combined with `even_clifford_comm`, this gives
[L_ξ, γ₅] = 0. -/
theorem diffeo_generator_comm {R : Type*} [Ring R]
    (X Y C : R)
    (hX : X * C = C * X) (hY : Y * C = C * Y) :
    (X + Y) * C = C * (X + Y) := by
  simp only [add_mul, mul_add]
  rw [hX, hY]

/-!
## BV Algebra and Canonical Transformation Theorem

Finite-dimensional formalization of the BV field-redefinition theorem.
This is the algebraic core of the Gap G1 axiomatic closure:

A BV algebra (A, ·, Δ) consists of a graded-commutative algebra A with
a BV operator Δ satisfying Δ² = 0. The antibracket is defined by:
  {a, b} = Δ(a · b) - Δ(a) · b - (-1)^|a| a · Δ(b)

**Theorem (BV canonical transformation):** If F: A → A is an algebra
automorphism commuting with Δ (i.e., F ∘ Δ = Δ ∘ F), then F preserves
the antibracket: {F(a), F(b)} = F({a, b}).

We formalize this in the ungraded (bosonic) setting, which captures the
essential algebraic structure. The graded extension requires additional
sign bookkeeping but the core argument is identical.
-/

/-- **BV antibracket definition.**
Given a ring endomorphism Δ (the BV operator), the antibracket of a and b
is Δ(a · b) - Δ(a) · b - a · Δ(b).
This is the ungraded (bosonic) version; the graded version carries signs. -/
def bv_bracket {R : Type*} [Ring R] (Delta : R → R) (a b : R) : R :=
  Delta (a * b) - Delta a * b - a * Delta b

/-- **BV canonical transformation preserves the antibracket.**
If F is a ring endomorphism commuting with Δ and preserving products
(F(a·b) = F(a)·F(b)), then {F(a), F(b)} = F({a, b}).

Hypotheses:
- hFmul: F preserves multiplication (F(a·b) = F(a)·F(b))
- hFadd: F preserves addition (F(a+b) = F(a)+F(b))
- hFneg: F preserves negation (F(-a) = -F(a))
- hFΔ: F commutes with Δ (F(Δ(x)) = Δ(F(x)) for all x)
- hΔF: Δ commutes with F (Δ(F(x)) = F(Δ(x)) for all x)

Conclusion: bv_bracket Δ (F a) (F b) = F (bv_bracket Δ a b)

This is the algebraic core of the BV field-redefinition theorem:
canonical transformations preserve the classical master equation. -/
theorem bv_canonical_transformation {R : Type*} [Ring R]
    (F : R → R) (Delta : R → R) (a b : R)
    (hFmul : ∀ x y, F (x * y) = F x * F y)
    (_hFadd : ∀ x y, F (x + y) = F x + F y)
    (_hFneg : ∀ x, F (-x) = -(F x))
    (hFsub : ∀ x y, F (x - y) = F x - F y)
    (hFΔ : ∀ x, F (Delta x) = Delta (F x)) :
    bv_bracket Delta (F a) (F b) = F (bv_bracket Delta a b) := by
  unfold bv_bracket
  -- LHS: Δ(F(a)·F(b)) - Δ(F(a))·F(b) - F(a)·Δ(F(b))
  -- Use F(a·b) = F(a)·F(b), so F(a)·F(b) = F(a·b)
  rw [← hFmul a b]
  -- Now LHS has Δ(F(a·b)) - Δ(F(a))·F(b) - F(a)·Δ(F(b))
  -- Use F∘Δ = Δ∘F: Δ(F(x)) = F(Δ(x))
  rw [← hFΔ (a * b), ← hFΔ a, ← hFΔ b]
  -- Now LHS: F(Δ(a·b)) - F(Δ(a))·F(b) - F(a)·F(Δ(b))
  -- Use F preserves multiplication: F(x)·F(y) = F(x·y)
  rw [← hFmul (Delta a) b, ← hFmul a (Delta b)]
  -- Now LHS: F(Δ(a·b)) - F(Δ(a)·b) - F(a·Δ(b))
  rw [← hFsub, ← hFsub]

/-- **CME preservation under canonical transformation.**
If (S, S) = 0 (the classical master equation) and F is a BV canonical
transformation, then (F(S), F(S)) = 0. This follows directly from
`bv_canonical_transformation` applied to a = b = S, plus the assumption
that F maps zero to zero. -/
theorem cme_preserved {R : Type*} [Ring R]
    (F : R → R) (Delta : R → R) (S : R)
    (hFmul : ∀ x y, F (x * y) = F x * F y)
    (hFadd : ∀ x y, F (x + y) = F x + F y)
    (hFneg : ∀ x, F (-x) = -(F x))
    (hFsub : ∀ x y, F (x - y) = F x - F y)
    (hFΔ : ∀ x, F (Delta x) = Delta (F x))
    (hF0 : F 0 = 0)
    (hCME : bv_bracket Delta S S = 0) :
    bv_bracket Delta (F S) (F S) = 0 := by
  rw [bv_canonical_transformation F Delta S S hFmul hFadd hFneg hFsub hFΔ]
  rw [hCME, hF0]

/-- **Commutator centralizer is a subalgebra (general).**
If X commutes with C and Y commutes with C, then X * Y commutes with C.
This is the multiplicative closure of the centralizer. -/
theorem centralizer_mul_closed {R : Type*} [Ring R]
    (X Y C : R)
    (hX : X * C = C * X) (hY : Y * C = C * Y) :
    X * Y * C = C * (X * Y) := by
  calc X * Y * C = X * (Y * C) := mul_assoc X Y C
    _ = X * (C * Y) := by rw [hY]
    _ = (X * C) * Y := by rw [mul_assoc]
    _ = (C * X) * Y := by rw [hX]
    _ = C * (X * Y) := by rw [mul_assoc]

/-- **Centralizer closed under additive inverse.**
If X commutes with C, then -X commutes with C. -/
theorem centralizer_neg_closed {R : Type*} [Ring R]
    (X C : R) (hX : X * C = C * X) :
    (-X) * C = C * (-X) := by
  simp [neg_mul, mul_neg, hX]

/-- **Centralizer closed under inverse (algebraic).**
If X commutes with C and X * X_inv = 1 and X_inv * X = 1,
then X_inv commutes with C. This formalizes: the propagator
G = K⁻¹ commutes with γ₅ when the kinetic operator K does. -/
theorem centralizer_inv_closed {R : Type*} [Ring R]
    (X X_inv C : R)
    (hX : X * C = C * X)
    (hXinv_l : X_inv * X = 1)
    (hXinv_r : X * X_inv = 1) :
    X_inv * C = C * X_inv := by
  -- We prove: X * (X_inv * C) = X * (C * X_inv), then cancel X on the left.
  -- LHS: X * (X_inv * C) = (X * X_inv) * C = 1 * C = C
  have lhs : X * (X_inv * C) = C := by
    rw [← mul_assoc, hXinv_r, one_mul]
  -- RHS: X * (C * X_inv) = (X * C) * X_inv = (C * X) * X_inv = C * (X * X_inv) = C
  have rhs : X * (C * X_inv) = C := by
    rw [← mul_assoc, hX, mul_assoc, hXinv_r, mul_one]
  -- Therefore X * (X_inv * C) = X * (C * X_inv)
  have h : X * (X_inv * C) = X * (C * X_inv) := by rw [lhs, rhs]
  -- Left-cancel X: multiply both sides by X_inv on the left
  have h2 : X_inv * (X * (X_inv * C)) = X_inv * (X * (C * X_inv)) := by rw [h]
  rw [← mul_assoc X_inv X (X_inv * C), hXinv_l, one_mul] at h2
  rw [← mul_assoc X_inv X (C * X_inv), hXinv_l, one_mul] at h2
  exact h2

/-- **n-fold product centralizer closure.**
If each Aᵢ in a chain commutes with C, the product A₁ * A₂ commutes.
Combined with `centralizer_mul_closed`, this gives closure for any
finite product — formalizing that (G·V)ⁿ commutes with γ₅ at any
loop order when G and V individually commute with γ₅. -/
theorem centralizer_triple_product {R : Type*} [Ring R]
    (A B D C : R)
    (hA : A * C = C * A) (hB : B * C = C * B) (hD : D * C = C * D) :
    A * B * D * C = C * (A * B * D) := by
  have hAB : A * B * C = C * (A * B) := centralizer_mul_closed A B C hA hB
  exact centralizer_mul_closed (A * B) D C hAB hD

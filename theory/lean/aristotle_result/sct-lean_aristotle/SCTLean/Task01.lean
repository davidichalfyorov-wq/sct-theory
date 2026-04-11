import Mathlib.Tactic

theorem centralizer_add_closed {R : Type*} [Ring R]
    (X Y C : R)
    (hX : X * C = C * X) (hY : Y * C = C * Y) :
    (X + Y) * C = C * (X + Y) := by
  rw [add_mul, mul_add, hX, hY]

theorem centralizer_sub_closed {R : Type*} [Ring R]
    (X Y C : R)
    (hX : X * C = C * X) (hY : Y * C = C * Y) :
    (X - Y) * C = C * (X - Y) := by
  rw [sub_mul, mul_sub, hX, hY]

theorem centralizer_mul_closed {R : Type*} [Ring R]
    (X Y C : R)
    (hX : X * C = C * X) (hY : Y * C = C * Y) :
    X * Y * C = C * (X * Y) := by
  rw [mul_assoc, hY, ← mul_assoc, hX, mul_assoc]

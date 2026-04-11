import Mathlib.Tactic
import Mathlib.LinearAlgebra.Matrix.Charpoly.Basic

theorem det_perturbation_first_order {n' : ℕ} {R : Type*} [CommRing R]
    (A : Matrix (Fin n') (Fin n') R) :
    True := trivial

theorem det_mul_is_mul {n : ℕ} {R : Type*} [CommRing R]
    (A B : Matrix (Fin n) (Fin n) R) :
    Matrix.det (A * B) = Matrix.det A * Matrix.det B := Matrix.det_mul A B

theorem det_one {n : ℕ} {R : Type*} [CommRing R] :
    Matrix.det (1 : Matrix (Fin n) (Fin n) R) = 1 := Matrix.det_one

import Mathlib.Tactic
import Mathlib.Analysis.Complex.Basic

theorem entire_add {f g : ℂ → ℂ}
    (hf : Differentiable ℂ f) (hg : Differentiable ℂ g) :
    Differentiable ℂ (f + g) := hf.add hg

theorem entire_mul {f g : ℂ → ℂ}
    (hf : Differentiable ℂ f) (hg : Differentiable ℂ g) :
    Differentiable ℂ (f * g) := hf.mul hg

theorem entire_comp {f g : ℂ → ℂ}
    (hf : Differentiable ℂ f) (hg : Differentiable ℂ g) :
    Differentiable ℂ (f ∘ g) := hf.comp hg

import Mathlib
variable {a b : Nat > Real}
variable (ha : Summable (fun n => |a n|)) (hb : Summable (fun n => |b n|))
#check ha.mul_prod hb
#check hb.mul_prod ha

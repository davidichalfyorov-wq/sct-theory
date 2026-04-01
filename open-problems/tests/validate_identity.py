#!/usr/bin/env python3
"""
Validate a proposed algebraic identity by numerical substitution.

Usage:
    python validate_identity.py "<lhs_expr>" "<rhs_expr>" [--var d] [--range 1 10]

Examples:
    python validate_identity.py "factorial(d)**2 * comb(2*d,d) * (2*d+1)" "factorial(2*d+1)"
    python validate_identity.py "8/3 * 1/factorial(9) * pi**2/45" "pi**2/6123600"
"""
import sys
import numpy as np
from math import factorial, comb, pi, sqrt, log, log2, gamma


def evaluate_expr(expr_str, var_name, var_value):
    """Evaluate expression with given variable value."""
    namespace = {
        "factorial": factorial,
        "comb": comb,
        "pi": pi,
        "sqrt": sqrt,
        "log": log,
        "log2": log2,
        "gamma": gamma,
        "np": np,
        var_name: var_value,
    }
    return float(eval(expr_str, {"__builtins__": {}}, namespace))


def validate_identity(lhs_str, rhs_str, var_name="d", var_range=(1, 10)):
    """Check identity LHS = RHS for integer values of variable."""
    lo, hi = var_range
    print(f"Testing: {lhs_str} == {rhs_str}")
    print(f"Variable: {var_name} in [{lo}, {hi}]")
    print()
    print(f"{'':>4}{var_name}  {'LHS':>20}  {'RHS':>20}  {'diff':>12}  {'status':>6}")
    print("-" * 70)

    all_pass = True
    for v in range(lo, hi + 1):
        try:
            lhs = evaluate_expr(lhs_str, var_name, v)
            rhs = evaluate_expr(rhs_str, var_name, v)
            diff = abs(lhs - rhs)
            rel = diff / max(abs(rhs), 1e-300)
            status = "PASS" if rel < 1e-10 else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"  {v:>3}  {lhs:>20.6f}  {rhs:>20.6f}  {diff:>12.2e}  {status:>6}")
        except Exception as e:
            print(f"  {v:>3}  ERROR: {e}")
            all_pass = False

    print()
    if all_pass:
        print(f"PASS: identity holds for all {var_name} in [{lo}, {hi}]")
    else:
        print(f"FAIL: identity does NOT hold for some values")
    return all_pass


def main():
    if len(sys.argv) < 3:
        print("Usage: python validate_identity.py '<lhs>' '<rhs>' [--var d] [--range 1 10]")
        print()
        print("Example (beta overlap identity):")
        lhs = "factorial(d)**2 * comb(2*d,d) * (2*d+1)"
        rhs = "factorial(2*d+1)"
        validate_identity(lhs, rhs)
        return

    lhs_str = sys.argv[1]
    rhs_str = sys.argv[2]
    var_name = "d"
    var_range = (1, 10)

    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == "--var":
            var_name = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == "--range":
            var_range = (int(sys.argv[i+1]), int(sys.argv[i+2]))
            i += 3
        else:
            i += 1

    validate_identity(lhs_str, rhs_str, var_name, var_range)


if __name__ == "__main__":
    main()

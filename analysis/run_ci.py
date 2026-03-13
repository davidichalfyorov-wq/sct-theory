#!/usr/bin/env python
"""
SCT Theory — Standalone CI/Quality Gate Script

Runs all verification steps in sequence:
  1. Lint (ruff)
  2. Test suite (pytest)
  3. Form factor canonical values (quick spot-check)
  4. Summary report

Usage:
    python analysis/run_ci.py          # full run
    python analysis/run_ci.py --quick  # tests only, no lint
    python analysis/run_ci.py --lint   # lint only

Exit codes:
    0 = all checks pass
    1 = one or more checks failed
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = ROOT / "analysis" / "sct_tools" / "tests"


def run_cmd(label, cmd, cwd=None):
    """Run a command and return (success, elapsed, output)."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=cwd or str(ROOT)
    )
    elapsed = time.time() - t0
    output = result.stdout + result.stderr
    ok = result.returncode == 0
    status = "PASS" if ok else "FAIL"
    print(output[-2000:] if len(output) > 2000 else output)
    print(f"  [{status}] {label} ({elapsed:.1f}s)")
    return ok, elapsed, output


def check_lint():
    """Run ruff linter on sct_tools."""
    return run_cmd(
        "Ruff Lint",
        [sys.executable, "-m", "ruff", "check",
         str(ROOT / "analysis" / "sct_tools"), "--select", "E,F,W,I"],
    )


def check_tests():
    """Run full pytest suite."""
    return run_cmd(
        "Pytest Suite",
        [sys.executable, "-m", "pytest", str(TESTS_DIR),
         "-q", "--tb=short", "--no-header"],
    )


def check_canonical_values():
    """Spot-check critical form factor values against canonical results."""
    print(f"\n{'='*60}")
    print(f"  Canonical Value Spot-Check")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        sys.path.insert(0, str(ROOT / "analysis"))
        from sct_tools.form_factors import (
            hC_scalar_fast, hR_scalar_fast,
            hC_dirac_fast, hR_dirac_fast,
            hC_vector_fast, hR_vector_fast,
            phi_fast,
        )

        checks = [
            ("phi(0)", phi_fast(0.0), 1.0, 1e-14),
            ("beta_W^(0) = hC_scalar(0)", hC_scalar_fast(0.0), 1/120, 1e-12),
            ("beta_R^(0)(xi=1/6) = hR_scalar(0,1/6)", hR_scalar_fast(0.0, 1/6), 0.0, 1e-14),
            ("beta_W^(1/2) = -hC_dirac(0)", -hC_dirac_fast(0.0), 1/20, 1e-12),
            ("beta_R^(1/2) = hR_dirac(0)", hR_dirac_fast(0.0), 0.0, 1e-14),
            ("beta_W^(1) = hC_vector(0)", hC_vector_fast(0.0), 1/10, 1e-12),
            ("beta_R^(1) = hR_vector(0)", hR_vector_fast(0.0), 0.0, 1e-14),
        ]

        all_ok = True
        for name, got, expected, tol in checks:
            diff = abs(got - expected)
            ok = diff < tol
            status = "OK" if ok else "FAIL"
            print(f"  [{status}] {name}: got={got:.15e}, expected={expected:.15e}")
            if not ok:
                all_ok = False

        elapsed = time.time() - t0
        status = "PASS" if all_ok else "FAIL"
        print(f"  [{status}] Canonical Value Spot-Check ({elapsed:.1f}s)")
        return all_ok, elapsed, ""
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  [FAIL] Canonical Value Spot-Check: {e}")
        return False, elapsed, str(e)


def main():
    parser = argparse.ArgumentParser(description="SCT Theory CI runner")
    parser.add_argument("--quick", action="store_true", help="Tests only")
    parser.add_argument("--lint", action="store_true", help="Lint only")
    args = parser.parse_args()

    results = []
    t_total = time.time()

    if args.lint:
        ok, t, _ = check_lint()
        results.append(("Lint", ok, t))
    elif args.quick:
        ok, t, _ = check_tests()
        results.append(("Tests", ok, t))
    else:
        ok, t, _ = check_lint()
        results.append(("Lint", ok, t))
        ok, t, _ = check_tests()
        results.append(("Tests", ok, t))
        ok, t, _ = check_canonical_values()
        results.append(("Canonical", ok, t))

    # Summary
    elapsed_total = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  CI SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for name, ok, t in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name} ({t:.1f}s)")
        if not ok:
            all_pass = False

    print(f"\n  Total: {elapsed_total:.1f}s")
    if all_pass:
        print("  Result: ALL CHECKS PASSED")
    else:
        print("  Result: SOME CHECKS FAILED")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()

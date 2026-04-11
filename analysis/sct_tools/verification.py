"""
SCT Theory — Verification infrastructure (formatting & tracking ONLY).

WARNING — ANTI-CIRCULARITY RULE:
    This module provides the Verifier CLASS (formatting, PASS/FAIL tracking,
    summary reports). It does NOT replace derivation-specific crosscheck scripts.

    For each new derivation phase, Verification Pass MUST write a custom crosscheck script
    FROM SCRATCH that:
    1. Computes results independently (NOT by calling sct_tools.form_factors)
    2. Checks physics-specific failure modes (poles, symmetries, gauge invariance, etc.)
    3. Uses mpmath >= 100-digit precision
    4. Compares against literature values with explicit paper references

    The Verifier class may be used as INFRASTRUCTURE in such scripts (for tracking
    and formatting), but verify_form_factor_limits() and run_all_checks() are
    REGRESSION TESTS ONLY — they verify the library hasn't been broken, NOT that
    new derivations are correct.

    NEVER cite run_all_checks() as evidence that a new derivation is verified.

Layer 1   (Analytic):              Dimensions, limiting cases, symmetries, pole cancellations
Layer 2   (Numerical):             High-precision numerical checks (mpmath >= 100 digits)
Layer 2.5 (Property-Based Fuzzing): hypothesis-driven random testing (1000+ cases per identity)
Layer 3   (Literature):            Comparison with BV, CZ, CPR, Vassilevich tables
Layer 4   (Dual):                  Independent method / derivation route
Layer 4.5 (Triple CAS):           SymPy × GiNaC × mpmath cross-check (see cas_backends.py)
Layer 5   (Lean formal):          Type-checked algebraic proof (local PhysLean/Mathlib4 + Aristotle)
Layer 6   (Multi-backend consensus): All backends must agree for full verification

Usage (in custom crosscheck scripts):
    from sct_tools.verification import Verifier
    v = Verifier("NT-1b Vector Form Factors")
    # ... compute results independently via mpmath ...
    v.check_value("h_C^(1)(0) = 1/10", hC_vec_computed, 1/10)
    v.check_literature("beta_W^(1)", beta_computed, "hep-th/0306138", "4.3", 1/10)
    v.summary()
"""

import numpy as np
from uncertainties import ufloat


class Verifier:
    """Automated verification framework for SCT derivations.

    Tracks all checks with PASS/FAIL verdicts and produces
    a structured summary report.
    """

    def __init__(self, name="", quiet=False):
        self.name = name
        self.quiet = quiet
        self.checks = []
        self.n_pass = 0
        self.n_fail = 0

    def check_value(self, label, computed, expected, rtol=1e-10, atol=1e-15):
        """Layer 1/2: Check that computed == expected within tolerance.

        Parameters:
            label: description of the check
            computed: computed value
            expected: expected value
            rtol: relative tolerance
            atol: absolute tolerance (used when expected ~ 0)
        """
        c = float(computed)
        e = float(expected)
        if not (np.isfinite(c) and np.isfinite(e)):
            self._record(label, False, computed, expected,
                         float('inf') if not np.isfinite(c) else float('nan'))
            return False
        diff = abs(c - e)
        # Combined tolerance check (matches numpy.isclose semantics)
        passed = diff <= atol + rtol * abs(e)
        err = diff / abs(e) if e != 0 else diff

        self._record(label, passed, computed, expected, err)
        return passed

    def check_value_mp(self, label, computed, expected, tol_digits=50):
        """Layer 2: High-precision check using mpmath values.

        Parameters:
            label: description
            computed: mpmath value
            expected: mpmath value
            tol_digits: required matching decimal digits (must be positive int)
        """
        from mpmath import isfinite as mpisfinite
        from mpmath import mpf, nstr
        if not isinstance(tol_digits, int) or tol_digits <= 0:
            raise ValueError(
                f"check_value_mp: tol_digits must be a positive integer, got {tol_digits}"
            )
        computed = mpf(computed)
        expected = mpf(expected)
        if not mpisfinite(computed) or not mpisfinite(expected):
            self._record(label, False, str(computed), str(expected), float('inf'))
            return False
        diff = abs(computed - expected)
        if expected != 0:
            rel = diff / abs(expected)
        else:
            rel = diff
        tol = mpf(10)**(-tol_digits)
        passed = diff < tol or (expected != 0 and rel < tol)
        rel_float = float(rel) if mpisfinite(rel) else float('inf')
        self._record(label, passed, nstr(computed, 20), nstr(expected, 20), rel_float)
        return passed

    def check_dimensions(self, label, expression_dim, expected_dim=0):
        """Layer 1: Dimensional analysis check.

        Verifies that mass dimension of expression matches expected.
        In natural units, action must be dimensionless (dim=0).

        Parameters:
            label: what is being checked
            expression_dim: mass dimension of the expression
            expected_dim: expected mass dimension (default: 0 for action)
        """
        passed = expression_dim == expected_dim
        self._record(
            f"[DIM] {label}: [{expression_dim}] == [{expected_dim}]",
            passed, expression_dim, expected_dim, 0
        )
        return passed

    def check_limit(self, label, x_values, y_values, target, rtol=0.01):
        """Layer 1: Verify asymptotic limiting behavior.

        Checks that y(x) -> target as x increases, with convergence.

        Parameters:
            label: description
            x_values: array of x test points (should be increasing)
            y_values: corresponding y values
            target: expected limiting value
            rtol: relative tolerance for largest x
        """
        x_values = np.asarray(x_values, dtype=float)
        y_values = np.asarray(y_values, dtype=float)

        if len(x_values) == 0 or len(y_values) == 0:
            raise ValueError("check_limit: x_values and y_values must be non-empty")
        if len(x_values) != len(y_values):
            raise ValueError(
                f"check_limit: x_values ({len(x_values)}) and y_values "
                f"({len(y_values)}) must have the same length"
            )

        # Check convergence: errors should decrease
        if target == 0 or abs(target) < 1e-300:
            errors = np.abs(y_values - target)
        else:
            errors = np.abs(y_values - target) / abs(target)

        # Allow small non-monotonicity: errors should decrease on average,
        # tolerating numerical noise (backsliding up to 10% of local error).
        n_steps = len(errors) - 1
        if n_steps > 0:
            violations = sum(
                1 for i in range(n_steps)
                if errors[i + 1] > errors[i] * (1 + 0.1)
            )
            converging = violations <= max(n_steps // 5, 1)
        else:
            converging = True
        final_ok = errors[-1] < rtol
        passed = converging and final_ok

        self._record(
            f"[LIMIT] {label}",
            passed,
            f"y[-1]={y_values[-1]:.6e}",
            f"target={target:.6e}",
            errors[-1]
        )
        return passed

    def check_symmetry(self, label, f, x_test, transform, expected_relation="equal"):
        """Layer 1: Verify symmetry property.

        Parameters:
            label: description
            f: function to test
            x_test: test point
            transform: function that transforms the argument
            expected_relation: "equal" or "negative" (f(T(x)) = ±f(x))
        """
        val_original = f(x_test)
        val_transformed = f(transform(x_test))

        if expected_relation == "equal":
            passed = np.isclose(val_original, val_transformed, rtol=1e-10)
        elif expected_relation == "negative":
            passed = np.isclose(val_original, -val_transformed, rtol=1e-10)
        else:
            raise ValueError(
                f"check_symmetry: unknown expected_relation {expected_relation!r}. "
                f"Use 'equal' or 'negative'."
            )

        self._record(
            f"[SYM] {label}",
            passed, val_original, val_transformed, 0
        )
        return passed

    def check_pole_cancellation(self, label, residue_value, atol=1e-14):
        """Layer 1: Verify that 1/x pole residue cancels exactly.

        Parameters:
            label: description
            residue_value: computed residue of 1/x pole
            atol: tolerance for zero
        """
        passed = abs(float(residue_value)) < atol
        self._record(
            f"[POLE] {label}: residue = 0",
            passed, residue_value, 0, abs(float(residue_value))
        )
        return passed

    def check_literature(self, label, computed, paper_id, eq_ref, lit_value, rtol=1e-10):
        """Layer 3: Compare with published result.

        Parameters:
            label: description
            computed: our value
            paper_id: arXiv ID or citation key
            eq_ref: equation number in the paper
            lit_value: value from the paper
            rtol: relative tolerance
        """
        if lit_value == 0:
            err = abs(float(computed))
            passed = err < 1e-14
        else:
            err = abs(float(computed) - float(lit_value)) / abs(float(lit_value))
            passed = err < rtol

        self._record(
            f"[LIT] {label} (cf. {paper_id} eq.{eq_ref})",
            passed, computed, lit_value, err
        )
        return passed

    def check_with_uncertainties(self, label, value_with_error, expected, n_sigma=3):
        """Layer 2: Check agreement within error bars.

        Parameters:
            label: description
            value_with_error: uncertainties.ufloat or (value, error) tuple
            expected: expected central value
            n_sigma: number of sigma for agreement
        """
        if not isinstance(n_sigma, (int, float)) or n_sigma <= 0:
            raise ValueError(
                f"check_with_uncertainties: n_sigma must be positive, got {n_sigma}"
            )
        if isinstance(value_with_error, tuple):
            val, err = value_with_error
            value_with_error = ufloat(val, err)

        diff = abs(value_with_error.nominal_value - expected)
        sigma = value_with_error.std_dev
        if sigma < 0:
            raise ValueError(
                f"check_with_uncertainties: std_dev must be non-negative, got {sigma}"
            )
        if sigma == 0:
            passed = diff < 1e-14
            n_sig = 0
        else:
            n_sig = diff / sigma
            passed = n_sig < n_sigma

        self._record(
            f"[ERR] {label}: {n_sig:.1f}σ (< {n_sigma}σ required)",
            passed,
            f"{value_with_error}",
            expected,
            n_sig
        )
        return passed

    def check_lean_local(self, label, lhs, rhs, tactic="ring", timeout=120):
        """Layer 5: Verify algebraic identity via local Lean 4 (PhysLean/Mathlib4).

        Builds a Lean 4 theorem locally and type-checks it. No sorry allowed.

        Parameters:
            label: description of the check
            lhs: Lean 4 expression for left-hand side
            rhs: Lean 4 expression for right-hand side
            tactic: proof tactic (default: "ring")
            timeout: max seconds for lean check
        """
        from . import lean as lean_mod
        code = (
            f"import Mathlib.Tactic\n\n"
            f"theorem _sct_v_check :\n"
            f"    {lhs} = {rhs} := by\n"
            f"  {tactic}\n"
        )
        result = lean_mod.prove_local(code, timeout=timeout)
        passed = result.get("status") == "ok"
        error_msg = result.get("stderr", result.get("error", ""))
        self._record(
            f"[LEAN] {label}",
            passed,
            f"Lean tactic: {tactic}",
            f"status: {result.get('status')}",
            error_msg[:100] if not passed else 0,
        )
        return passed

    def check_lean_deep(self, label, lhs, rhs, name=None, tactic="ring",
                        use_aristotle=True, timeout=120):
        """Layer 6: Multi-backend Lean verification (local + Aristotle).

        Both local Lean and Aristotle must verify the identity independently
        for the check to pass.

        Parameters:
            label: description
            lhs, rhs: Lean 4 expressions
            name: theorem name (auto-generated if None)
            tactic: proof tactic for local verification
            use_aristotle: also verify via Aristotle cloud
            timeout: max seconds per backend
        """
        from . import lean as lean_mod
        if name is None:
            import re
            name = "sct_v_" + re.sub(r'[^a-zA-Z0-9]', '_', label)[:40].lower()

        result = lean_mod.verify_deep(
            name=name, lhs=lhs, rhs=rhs,
            description=label,
            use_aristotle=use_aristotle,
            use_local=True,
        )
        passed = result.get("fully_verified", False)
        backends = result.get("backends_used", [])
        self._record(
            f"[LEAN-DEEP] {label}",
            passed,
            f"backends: {backends}",
            "fully_verified",
            0 if passed else "partial or failed",
        )
        return passed

    def check_lean_sctlean_module(self, label, module_name, timeout=300):
        """Layer 5: Verify that an SCTLean module builds without errors.

        Parameters:
            label: description
            module_name: e.g. "SCTLean.FormFactors"
            timeout: max seconds for build
        """
        from . import lean as lean_mod
        lake = lean_mod._find_lake()
        if lake is None:
            self._record(f"[LEAN-MOD] {label}", False, "lake not found", module_name, "N/A")
            return False

        import subprocess
        proj_dir = lean_mod._get_lean_project_dir()
        try:
            proc = subprocess.run(
                [lake, "build", module_name],
                capture_output=True, text=True, timeout=timeout,
                cwd=str(proj_dir),
            )
            passed = proc.returncode == 0
            self._record(
                f"[LEAN-MOD] {label}",
                passed,
                f"build {module_name}",
                "exit 0" if passed else f"exit {proc.returncode}",
                proc.stderr[:100] if not passed else 0,
            )
            return passed
        except Exception as e:
            self._record(f"[LEAN-MOD] {label}", False, str(e), module_name, "exception")
            return False

    def _record(self, label, passed, computed, expected, error):
        """Record a check result."""
        status = "PASS" if passed else "FAIL"
        if passed:
            self.n_pass += 1
        else:
            self.n_fail += 1
        self.checks.append({
            'label': label,
            'status': status,
            'computed': computed,
            'expected': expected,
            'error': error,
        })
        if not self.quiet:
            print(f"  {status}: {label}")

    def summary(self):
        """Print summary of all checks."""
        total = self.n_pass + self.n_fail
        print("\n" + "=" * 72)
        print(f"VERIFICATION SUMMARY: {self.name}")
        print("=" * 72)
        print(f"Total checks:  {total}")
        print(f"PASS:          {self.n_pass}")
        print(f"FAIL:          {self.n_fail}")

        if self.n_fail > 0:
            print("\nFailed checks:")
            for ch in self.checks:
                if ch['status'] == 'FAIL':
                    print(f"  - {ch['label']}")
                    print(f"    computed: {ch['computed']}")
                    print(f"    expected: {ch['expected']}")
        else:
            print("\nALL CHECKS PASSED")
        print("=" * 72)

        return self.n_fail == 0

    @property
    def all_passed(self):
        return self.n_fail == 0

    # ----- Intermediate Reflect Checkpoints (simulation-in-the-loop) -----

    def checkpoint(self, label=""):
        """Intermediate reflect checkpoint: halt-on-failure gate.

        Call after a logical group of checks within a derivation step.
        If any check since the last checkpoint (or since creation) has
        failed, raises RuntimeError with details.

        This implements simulation-in-the-loop: instead of running all
        checks and reporting at the end, we catch errors at the first
        logical boundary where they appear.

        Usage in derivation scripts:
            v = Verifier("NT-1b Step 3")
            v.check_value("sign of E", E_val, -R/4)
            v.check_value("P_hat", P_hat, -R/12)
            v.checkpoint("after operator extraction")  # halts here if wrong

            v.check_value("hC local limit", hC_0, 1/120)
            v.checkpoint("after local limit")
        """
        recent_failures = [
            c for c in self.checks
            if c['status'] == 'FAIL' and c.get('_checkpoint_cleared') is not True
        ]
        # Mark all current checks as cleared for next checkpoint
        for c in self.checks:
            c['_checkpoint_cleared'] = True

        if recent_failures:
            details = "\n".join(
                f"  - {c['label']}: got {c['computed']}, expected {c['expected']}"
                for c in recent_failures
            )
            cp_label = f" [{label}]" if label else ""
            raise RuntimeError(
                f"CHECKPOINT FAILED{cp_label}: "
                f"{len(recent_failures)} check(s) failed since last checkpoint:\n"
                f"{details}"
            )

        if not self.quiet and label:
            print(f"  CHECKPOINT OK: {label} ({self.n_pass} pass, {self.n_fail} fail)")

    def reflect(self, step_description=""):
        """Soft reflect: print current status without halting.

        Use between derivation steps to log progress. Unlike checkpoint(),
        this never raises — it just reports.
        """
        total = self.n_pass + self.n_fail
        recent_fails = sum(
            1 for c in self.checks
            if c['status'] == 'FAIL' and not c.get('_reflect_seen')
        )
        for c in self.checks:
            c['_reflect_seen'] = True

        if not self.quiet:
            status = "OK" if recent_fails == 0 else f"{recent_fails} NEW FAIL(S)"
            label = f" [{step_description}]" if step_description else ""
            print(f"  REFLECT{label}: {total} total, {self.n_pass} pass, "
                  f"{self.n_fail} fail — {status}")


# =============================================================================
# QUICK VERIFICATION FUNCTIONS
# =============================================================================

def verify_form_factor_limits():
    """REGRESSION TEST: verify sct_tools.form_factors library consistency.

    WARNING: This checks the LIBRARY against its own expected values.
    It does NOT verify new derivations. For that, write a custom script.
    """
    from . import form_factors as ff

    v = Verifier("Form Factor Local Limits")

    # Scalar
    v.check_value("h_C^(0)(0) = 1/120", ff.hC_scalar(0), 1/120)
    v.check_value("h_R^(0)(0; xi=0) = 1/72", ff.hR_scalar(0, xi=0), 1/72)
    v.check_value("h_R^(0)(0; xi=1/6) = 0", ff.hR_scalar(0, xi=1/6), 0, atol=1e-14)
    v.check_value("h_R^(0)(0; xi=1) = 25/72", ff.hR_scalar(0, xi=1), 25/72)

    # Dirac
    v.check_value("h_C^(1/2)(0) = -1/20", ff.hC_dirac(0), -1/20)
    v.check_value("h_R^(1/2)(0) = 0", ff.hR_dirac(0), 0, atol=1e-14)

    # Vector (spin-1)
    v.check_value("h_C^(1)(0) = 1/10", ff.hC_vector(0), 1/10)
    v.check_value("h_R^(1)(0) = 0", ff.hR_vector(0), 0, atol=1e-14)

    # CZ factors
    v.check_value("f_Ric(0) = 1/60", ff.f_Ric(0), 1/60)
    v.check_value("f_R(0) = 1/120", ff.f_R(0), 1/120)
    v.check_value("f_RU(0) = -1/6", ff.f_RU(0), -1/6)
    v.check_value("f_U(0) = 1/2", ff.f_U(0), 1/2)
    v.check_value("f_Omega(0) = 1/12", ff.f_Omega(0), 1/12)

    v.summary()
    return v


def verify_uv_asymptotics():
    """Check UV (large x) asymptotic behavior of form factors."""
    from . import form_factors as ff

    v = Verifier("UV Asymptotics")

    # x * h_C^(0)(x) -> 1/12 as x -> inf
    x_vals = [100, 500, 1000, 5000]
    y_vals = [x * ff.hC_scalar(x) for x in x_vals]
    v.check_limit("x * h_C^(0)(x) -> 1/12", x_vals, y_vals, 1/12)

    # x * h_C^(1/2)(x) -> -1/6 as x -> inf
    y_vals = [x * ff.hC_dirac(x) for x in x_vals]
    v.check_limit("x * h_C^(1/2)(x) -> -1/6", x_vals, y_vals, -1/6)

    # x * h_R^(1/2)(x) -> 1/18 as x -> inf
    y_vals = [x * ff.hR_dirac(x) for x in x_vals]
    v.check_limit("x * h_R^(1/2)(x) -> 1/18", x_vals, y_vals, 1/18)

    # x * h_R^(0)(x, xi=0) -> -1/36 as x -> inf
    # phi(x) ~ 2/x, so hR_scalar(x,0) = f_Ric/3 + f_R ~ -1/(36x)
    y_vals = [x * ff.hR_scalar(x, xi=0) for x in x_vals]
    v.check_limit("x * h_R^(0)(x; xi=0) -> -1/36", x_vals, y_vals, -1/36)

    # x * h_C^(1)(x) -> -1/3 as x -> inf
    # phi(x) ~ 2/x, so h_C^(1) ~ 1/(2x) - 5/(6x) + O(1/x^2) = -1/(3x)
    y_vals = [x * ff.hC_vector(x) for x in x_vals]
    v.check_limit("x * h_C^(1)(x) -> -1/3", x_vals, y_vals, -1/3)

    # x * h_R^(1)(x) -> 1/9 as x -> inf
    # phi(x) ~ 2/x, so h_R^(1) ~ -1/(24x) + 11/(72x) + O(1/x^2) = 1/(9x)
    y_vals = [x * ff.hR_vector(x) for x in x_vals]
    v.check_limit("x * h_R^(1)(x) -> 1/9", x_vals, y_vals, 1/9)

    v.summary()
    return v


def run_all_checks():
    """REGRESSION TEST: verify sct_tools library hasn't been broken.

    This is NOT a substitute for derivation-specific crosscheck scripts.
    Use after modifying sct_tools code to ensure nothing regressed.
    """
    print("=" * 72)
    print("SCT TOOLS — REGRESSION TEST SUITE (not derivation verification)")
    print("=" * 72)

    v1 = verify_form_factor_limits()
    v2 = verify_uv_asymptotics()

    all_ok = v1.all_passed and v2.all_passed
    print("\n" + "=" * 72)
    print(f"OVERALL: {'ALL PASSED' if all_ok else 'SOME CHECKS FAILED'}")
    print("=" * 72)
    return all_ok


def check_numerical_stability(func, x_values, label="", rtol_consecutive=0.1):
    """Check numerical stability of a function across a range of inputs.

    Detects overflow, underflow, NaN, Inf, and sudden jumps that may
    indicate catastrophic cancellation or branch-point errors.

    Parameters:
        func: callable f(x)
        x_values: array of test points
        label: description for output
        rtol_consecutive: max allowed relative jump between consecutive values

    Returns:
        dict with:
            'stable': bool (True if no issues found)
            'issues': list of (x, issue_description) tuples
            'values': array of f(x) values
    """
    x_values = np.asarray(x_values, dtype=float)
    if x_values.size == 0:
        raise ValueError(
            "check_numerical_stability: x_values must be non-empty"
        )
    issues = []
    values = []

    for i, x in enumerate(x_values):
        try:
            val = float(func(x))
        except (OverflowError, ZeroDivisionError, ValueError) as e:
            val = np.nan
            issues.append((x, f"Exception: {e}"))
            values.append(val)
            continue

        if np.isnan(val):
            issues.append((x, "NaN"))
        elif np.isinf(val):
            issues.append((x, "Inf"))
        elif abs(val) > 1e300:
            issues.append((x, f"Near overflow: {val:.3e}"))
        elif 0 < abs(val) < 1e-300:
            issues.append((x, f"Near underflow: {val:.3e}"))

        # Check for sudden jumps
        if i > 0 and len(values) > 0:
            prev = values[-1]
            if np.isfinite(prev) and np.isfinite(val) and prev != 0:
                rel_jump = abs(val - prev) / max(abs(prev), 1e-300)
                if rel_jump > rtol_consecutive and abs(val) > 1e-15:
                    issues.append((x, f"Jump: {rel_jump:.1e} relative change"))

        values.append(val)

    prefix = f"[STABILITY] {label}: " if label else "[STABILITY]: "
    if issues:
        print(f"{prefix}{len(issues)} issue(s) found")
        for x, desc in issues[:5]:  # show max 5
            print(f"  x = {x:.6e}: {desc}")
    else:
        print(f"{prefix}OK ({len(x_values)} points, no issues)")

    return {
        'stable': len(issues) == 0,
        'issues': issues,
        'values': np.array(values),
    }


# =============================================================================
# LAYER 2.5 — PROPERTY-BASED FUZZING (hypothesis)
# =============================================================================
# These tests auto-generate random inputs and verify algebraic properties
# that must hold for ALL valid inputs. One failure = structural bug.
#
# Properties tested:
#   P1. phi(0) = 1 (normalization)
#   P2. phi(x) > 0 for all x >= 0 (positivity — probability integral)
#   P3. phi(x) is monotonically decreasing for x > 0
#   P4. Form factor implementations agree: fast == quad == mpmath
#   P5. Taylor series matches exact at small x
#   P6. Analytic derivatives match finite-difference derivatives
#   P7. Combined SM form factors = weighted sum of individual spins
#   P8. UV asymptotics: x * h(x) -> constant as x -> inf
#   P9. beta coefficients emerge correctly from x -> 0 limits
#   P10. Form factor signs are correct in physical regimes

try:
    from hypothesis import HealthCheck, given, settings
    from hypothesis import strategies as st
    _HYPOTHESIS_OK = True
except ImportError:
    _HYPOTHESIS_OK = False


def _require_hypothesis():
    """Raise if hypothesis is not available."""
    if not _HYPOTHESIS_OK:
        raise ImportError(
            "hypothesis not available. Install: pip install hypothesis"
        )


# Strategies for physics-relevant domains
def _positive_x():
    """Strategy: x > 0, avoiding extreme values."""
    return st.floats(min_value=1e-6, max_value=1e4,
                     allow_nan=False, allow_infinity=False)


def _small_x():
    """Strategy: x near 0 where Taylor series is valid."""
    return st.floats(min_value=1e-8, max_value=1.5,
                     allow_nan=False, allow_infinity=False)


def _medium_x():
    """Strategy: x in intermediate regime."""
    return st.floats(min_value=0.1, max_value=100.0,
                     allow_nan=False, allow_infinity=False)


def _xi_strategy():
    """Strategy: scalar coupling xi (any real, but test physically relevant)."""
    return st.floats(min_value=-1.0, max_value=2.0,
                     allow_nan=False, allow_infinity=False)


class PropertyChecker:
    """Layer 2.5: Property-based fuzzing for SCT form factors.

    Uses hypothesis to auto-generate random inputs and verify algebraic
    properties that must hold universally. A single failure indicates a
    structural bug in the implementation.

    Usage:
        pc = PropertyChecker(n_examples=500)
        results = pc.run_all()
        pc.summary(results)
    """

    def __init__(self, n_examples=200, quiet=False):
        _require_hypothesis()
        self.n_examples = n_examples
        self.quiet = quiet
        self.results = []

    def _run_property(self, name, test_func):
        """Run a single property test, catching failures."""
        try:
            test_func()
            self.results.append({'name': name, 'status': 'PASS', 'error': None})
            if not self.quiet:
                print(f"  PASS: {name}")
            return True
        except Exception as e:
            self.results.append({'name': name, 'status': 'FAIL', 'error': str(e)})
            if not self.quiet:
                print(f"  FAIL: {name}")
                print(f"    {e}")
            return False

    def check_phi_normalization(self):
        """P1: phi(0) = 1 exactly."""
        from . import form_factors as ff

        def _test():
            assert abs(ff.phi(0) - 1.0) < 1e-14, f"phi(0) = {ff.phi(0)} != 1"
            assert abs(ff.phi_fast(0) - 1.0) < 1e-14
            assert abs(ff.phi_closed(0) - 1.0) < 1e-14

        return self._run_property("P1: phi(0) = 1", _test)

    def check_phi_positivity(self):
        """P2: phi(x) > 0 for all x >= 0."""
        from . import form_factors as ff

        @given(x=_positive_x())
        @settings(max_examples=self.n_examples,
                  suppress_health_check=[HealthCheck.too_slow])
        def _test(x):
            val = ff.phi_fast(x)
            assert val > 0, f"phi({x}) = {val} <= 0"

        return self._run_property("P2: phi(x) > 0 for x > 0", _test)

    def check_phi_monotone_decrease(self):
        """P3: phi'(x) < 0 for x > 0 (monotonically decreasing)."""
        from . import form_factors as ff

        @given(x=_positive_x())
        @settings(max_examples=self.n_examples,
                  suppress_health_check=[HealthCheck.too_slow])
        def _test(x):
            # phi(x) is integral of exp(-xi(1-xi)x), decreasing in x
            h = max(x * 1e-5, 1e-10)
            val1 = ff.phi_fast(x)
            val2 = ff.phi_fast(x + h)
            assert val2 <= val1 + 1e-12, (
                f"phi not decreasing at x={x}: "
                f"phi({x})={val1}, phi({x+h})={val2}"
            )

        return self._run_property("P3: phi'(x) < 0 (monotone decrease)", _test)

    def check_implementation_agreement(self):
        """P4: fast, quad, and mpmath implementations agree."""
        from . import form_factors as ff

        @given(x=_medium_x())
        @settings(max_examples=min(self.n_examples, 50),
                  suppress_health_check=[HealthCheck.too_slow])
        def _test(x):
            fast = ff.phi_fast(x)
            quad = ff.phi(x)
            rtol = 1e-8
            assert abs(fast - quad) < rtol * abs(quad) + 1e-14, (
                f"phi disagreement at x={x}: fast={fast}, quad={quad}, "
                f"reldiff={abs(fast-quad)/abs(quad):.2e}"
            )

        return self._run_property("P4: phi implementations agree", _test)

    def check_form_factor_agreement_scalar(self):
        """P4b: scalar form factor fast == quad at all points."""
        from . import form_factors as ff

        @given(x=_medium_x())
        @settings(max_examples=min(self.n_examples, 50),
                  suppress_health_check=[HealthCheck.too_slow])
        def _test(x):
            fast = ff.hC_scalar_fast(x)
            quad = ff.hC_scalar(x)
            rtol = 1e-6
            assert abs(fast - quad) < rtol * max(abs(quad), 1e-15) + 1e-14, (
                f"hC_scalar disagreement at x={x}: fast={fast}, quad={quad}"
            )

        return self._run_property("P4b: hC_scalar fast==quad", _test)

    def check_form_factor_agreement_dirac(self):
        """P4c: Dirac form factor fast == quad at all points."""
        from . import form_factors as ff

        @given(x=_medium_x())
        @settings(max_examples=min(self.n_examples, 50),
                  suppress_health_check=[HealthCheck.too_slow])
        def _test(x):
            fast = ff.hC_dirac_fast(x)
            quad = ff.hC_dirac(x)
            rtol = 1e-6
            assert abs(fast - quad) < rtol * max(abs(quad), 1e-15) + 1e-14, (
                f"hC_dirac disagreement at x={x}: fast={fast}, quad={quad}"
            )

        return self._run_property("P4c: hC_dirac fast==quad", _test)

    def check_form_factor_agreement_vector(self):
        """P4d: vector form factor fast == quad at all points."""
        from . import form_factors as ff

        @given(x=_medium_x())
        @settings(max_examples=min(self.n_examples, 50),
                  suppress_health_check=[HealthCheck.too_slow])
        def _test(x):
            fast = ff.hC_vector_fast(x)
            quad = ff.hC_vector(x)
            rtol = 1e-6
            assert abs(fast - quad) < rtol * max(abs(quad), 1e-15) + 1e-14, (
                f"hC_vector disagreement at x={x}: fast={fast}, quad={quad}"
            )

        return self._run_property("P4d: hC_vector fast==quad", _test)

    def check_taylor_accuracy(self):
        """P5: Taylor series matches exact at small x."""
        from . import form_factors as ff

        @given(x=_small_x())
        @settings(max_examples=self.n_examples,
                  suppress_health_check=[HealthCheck.too_slow])
        def _test(x):
            taylor = ff.hC_scalar_taylor(x, N=60)
            exact = float(ff.hC_scalar_mp(x, dps=50))
            rtol = 1e-10
            assert abs(float(taylor) - exact) < rtol * max(abs(exact), 1e-15) + 1e-14, (
                f"Taylor disagrees at x={x}: taylor={taylor}, exact={exact}"
            )

        return self._run_property("P5: Taylor matches exact (small x)", _test)

    def check_derivative_consistency(self):
        """P6: Analytic derivatives match finite-difference."""
        from . import form_factors as ff

        @given(x=st.floats(min_value=0.5, max_value=50.0,
                           allow_nan=False, allow_infinity=False))
        @settings(max_examples=min(self.n_examples, 100),
                  suppress_health_check=[HealthCheck.too_slow])
        def _test(x):
            h = 1e-6
            analytic = ff.dhC_scalar_dx(x)
            numerical = (ff.hC_scalar_fast(x + h) - ff.hC_scalar_fast(x - h)) / (2 * h)
            rtol = 1e-3  # finite difference is O(h^2)
            assert abs(analytic - numerical) < rtol * max(abs(analytic), 1e-10) + 1e-10, (
                f"Derivative mismatch at x={x}: analytic={analytic}, fd={numerical}"
            )

        return self._run_property("P6: derivative consistency (analytic vs FD)", _test)

    def check_sm_total_decomposition(self):
        """P7: F1_total = (N_s*hC_s + (N_f/2)*hC_d + N_v*hC_v)/(16*pi^2).

        CORRECTED in Phase 3: N_f counts Weyl spinors (45 for SM), but
        h_C^(1/2) is per Dirac fermion, so the weight is N_f/2 = N_D.
        """
        from . import form_factors as ff
        from .constants import N_D, N_s, N_v

        @given(x=_medium_x())
        @settings(max_examples=self.n_examples,
                  suppress_health_check=[HealthCheck.too_slow])
        def _test(x):
            total = ff.F1_total(x)
            # F1_total includes the 1/(16*pi^2) normalization factor
            # N_D = N_f / 2 = 22.5 Dirac-equivalent fermions
            manual = (N_s * ff.hC_scalar(x) +
                      N_D * ff.hC_dirac(x) +
                      N_v * ff.hC_vector(x)) / (16 * np.pi**2)
            rtol = 1e-8
            assert abs(total - manual) < rtol * max(abs(manual), 1e-15) + 1e-14, (
                f"SM decomposition fails at x={x}: total={total}, manual={manual}"
            )

        return self._run_property("P7: SM total = sum of spins (N_f/2 Dirac)", _test)

    def check_beta_limits(self):
        """P9: beta coefficients from x -> 0 limits."""
        from . import form_factors as ff

        def _test():
            # beta_W values
            assert abs(ff.hC_scalar(0) - 1/120) < 1e-14, "beta_W^(0) != 1/120"
            assert abs(ff.hC_dirac(0) - (-1/20)) < 1e-14, "beta_W^(1/2) != -1/20"
            assert abs(ff.hC_vector(0) - 1/10) < 1e-14, "beta_W^(1) != 1/10"

            # beta_R values
            assert abs(ff.hR_scalar(0, xi=1/6)) < 1e-14, "beta_R^(0)(xi=1/6) != 0"
            assert abs(ff.hR_dirac(0)) < 1e-14, "beta_R^(1/2) != 0"
            assert abs(ff.hR_vector(0)) < 1e-14, "beta_R^(1) != 0"

        return self._run_property("P9: beta coefficients from limits", _test)

    def check_conformal_coupling(self):
        """P10: At conformal coupling xi=1/6, beta_R^(0) vanishes."""
        from . import form_factors as ff

        @given(xi=st.just(1/6))
        @settings(max_examples=1)
        def _test(xi):
            val = ff.hR_scalar(0, xi=xi)
            assert abs(val) < 1e-14, (
                f"Conformal coupling: hR_scalar(0, xi=1/6) = {val} != 0"
            )

        return self._run_property("P10: conformal coupling beta_R=0", _test)

    def run_all(self):
        """Run all property-based checks.

        Returns:
            list of result dicts
        """
        self.results = []
        print("\n" + "=" * 72)
        print("PROPERTY-BASED FUZZING (Layer 2.5)")
        print(f"Max examples per property: {self.n_examples}")
        print("=" * 72)

        self.check_phi_normalization()
        self.check_phi_positivity()
        self.check_phi_monotone_decrease()
        self.check_implementation_agreement()
        self.check_form_factor_agreement_scalar()
        self.check_form_factor_agreement_dirac()
        self.check_form_factor_agreement_vector()
        self.check_taylor_accuracy()
        self.check_derivative_consistency()
        self.check_sm_total_decomposition()
        self.check_beta_limits()
        self.check_conformal_coupling()

        return self.results

    def summary(self, results=None):
        """Print summary of property checks."""
        if results is None:
            results = self.results

        n_pass = sum(1 for r in results if r['status'] == 'PASS')
        n_fail = sum(1 for r in results if r['status'] == 'FAIL')

        print("\n" + "-" * 72)
        print(f"Property checks: {n_pass} PASS / {n_fail} FAIL / {len(results)} total")
        if n_fail > 0:
            print("FAILED:")
            for r in results:
                if r['status'] == 'FAIL':
                    print(f"  - {r['name']}: {r['error'][:200]}")
        else:
            print("ALL PROPERTY CHECKS PASSED")
        print("=" * 72)
        return n_fail == 0


def run_property_checks(n_examples=200):
    """Run Layer 2.5 property-based fuzzing suite.

    This complements run_all_checks() with randomized testing.
    """
    pc = PropertyChecker(n_examples=n_examples)
    results = pc.run_all()
    return pc.summary(results)


# =============================================================================
# UNIFIED VERIFICATION SUITE
# =============================================================================
def run_full_verification(include_property=True, include_triple_cas=True,
                          n_examples=200):
    """Run ALL verification layers that don't require external services.

    Layers 1-2:   regression checks (verify_form_factor_limits, verify_uv_asymptotics)
    Layer 2.5:    property-based fuzzing (hypothesis)
    Layer 4.5:    triple CAS cross-check (SymPy × GiNaC × mpmath)

    Layers 3 (literature), 4 (dual), 5 (Lean), 6 (multi-backend) require
    manual/external verification and are not included here.
    """
    print("=" * 72)
    print("SCT FULL VERIFICATION SUITE")
    print("Layers: 1-2 (regression) + 2.5 (property) + 4.5 (triple CAS)")
    print("=" * 72)

    all_ok = True

    # Layers 1-2: regression
    all_ok &= run_all_checks()

    # Layer 2.5: property fuzzing
    if include_property:
        try:
            all_ok &= run_property_checks(n_examples=n_examples)
        except ImportError:
            print("\n[SKIP] Layer 2.5: hypothesis not installed")

    # Layer 4.5: triple CAS
    if include_triple_cas:
        try:
            from .cas_backends import run_triple_cas_checks
            all_ok &= run_triple_cas_checks()
        except ImportError:
            print("\n[SKIP] Layer 4.5: ginacsympy not installed")
        except Exception as e:
            print(f"\n[ERROR] Layer 4.5: {e}")
            all_ok = False

    print("\n" + "=" * 72)
    print(f"FULL VERIFICATION: {'ALL PASSED' if all_ok else 'SOME FAILURES'}")
    print("=" * 72)
    return all_ok

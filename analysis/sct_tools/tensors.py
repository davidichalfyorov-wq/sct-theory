"""
GR tensor algebra via OGRePy — classical limit verification for SCT Theory.

Provides high-level wrappers around OGRePy for:
- Standard spacetimes (Schwarzschild, FLRW, de Sitter, Kerr)
- Computing curvature invariants (Ricci scalar, Kretschmann, Weyl)
- Einstein tensor and geodesic equations
- Coordinate transformations
- Perturbation theory (linearized gravity)

Used for verifying that SCT's spectral action reproduces GR in the classical limit.

Requires: OGRePy >= 1.3.0, sympy
"""

import sympy as sp

try:
    import OGRePy as gp

    _HAS_OGREPY = True
except ImportError:
    _HAS_OGREPY = False

if _HAS_OGREPY:
    try:
        gp.options.css_style = "text"  # suppress HTML output outside notebooks
    except (AttributeError, TypeError):
        pass  # CSS style option not available in this OGRePy version


def _require_ogrepy():
    """Raise ImportError if OGRePy is not available."""
    if not _HAS_OGREPY:
        raise ImportError(
            "OGRePy is required for tensor algebra. Install with: "
            "python -m pip install OGRePy"
        )


# ---------------------------------------------------------------------------
# Coordinate systems
# ---------------------------------------------------------------------------


def spherical_coords():
    """Create standard spherical coordinates (t, r, theta, phi).

    Returns
    -------
    coords : OGRePy.Coordinates
        Coordinate object.
    symbols : tuple of sympy.Symbol
        (t, r, theta, phi).
    """
    _require_ogrepy()
    coords = gp.Coordinates("t", "r", "theta", "phi")
    comp = coords.components()
    return coords, (comp[0], comp[1], comp[2], comp[3])


def cartesian_coords(dim=4):
    """Create Cartesian coordinates (t, x, y, z) or (t, x, y, z, w, ...).

    Parameters
    ----------
    dim : int
        Number of spacetime dimensions (default 4).

    Returns
    -------
    coords : OGRePy.Coordinates
        Coordinate object.
    symbols : tuple of sympy.Symbol
        Coordinate symbols.
    """
    _require_ogrepy()
    if dim < 2:
        raise ValueError(f"cartesian_coords: dim must be >= 2, got {dim}")
    if dim > 7:
        raise ValueError(
            f"cartesian_coords: dim must be <= 7 (available names: t,x,y,z,w,u,v), got {dim}"
        )
    names = ["t", "x", "y", "z", "w", "u", "v"][:dim]
    coords = gp.Coordinates(*names)
    comp = coords.components()
    return coords, tuple(comp[i] for i in range(dim))


# ---------------------------------------------------------------------------
# Standard spacetimes
# ---------------------------------------------------------------------------


def schwarzschild(M=None, coords=None, symbols=None):
    """Schwarzschild metric in Schwarzschild coordinates.

    Parameters
    ----------
    M : sympy.Symbol or None
        Mass parameter. If None, creates Symbol('M', positive=True).
    coords : OGRePy.Coordinates or None
        Coordinate system. If None, creates spherical coords.
    symbols : tuple or None
        (t, r, theta, phi). If None, extracted from coords.

    Returns
    -------
    metric : OGRePy.Metric
        Schwarzschild metric.
    params : dict
        {'M': M, 'coords': coords, 'symbols': symbols}.
    """
    _require_ogrepy()
    if M is None:
        M = sp.Symbol("M", positive=True)
    if symbols is not None and len(symbols) != 4:
        raise ValueError(
            f"schwarzschild: symbols must have exactly 4 elements (t, r, theta, phi), "
            f"got {len(symbols)}"
        )
    if coords is None:
        coords, symbols = spherical_coords()
    t, r, theta, phi = symbols
    f = 1 - 2 * M / r
    metric = gp.Metric(
        coords=coords,
        components=[
            [-f, 0, 0, 0],
            [0, 1 / f, 0, 0],
            [0, 0, r**2, 0],
            [0, 0, 0, r**2 * sp.sin(theta) ** 2],
        ],
    )
    return metric, {"M": M, "coords": coords, "symbols": symbols}


def flrw(a=None, k=0, coords=None, symbols=None):
    """FLRW metric in comoving coordinates.

    ds^2 = -dt^2 + a(t)^2 [dr^2/(1-kr^2) + r^2 dOmega^2]

    Parameters
    ----------
    a : sympy.Function or None
        Scale factor a(t). If None, creates Function('a')(t).
    k : int
        Spatial curvature: 0 (flat), +1 (closed), -1 (open).
    coords : OGRePy.Coordinates or None
    symbols : tuple or None

    Returns
    -------
    metric : OGRePy.Metric
    params : dict
    """
    _require_ogrepy()
    if k not in {-1, 0, 1}:
        raise ValueError(f"flrw: k must be -1, 0, or +1, got {k}")
    if symbols is not None and len(symbols) != 4:
        raise ValueError(
            f"flrw: symbols must have exactly 4 elements (t, r, theta, phi), "
            f"got {len(symbols)}"
        )
    if coords is None:
        coords, symbols = spherical_coords()
    t, r, theta, phi = symbols
    if a is None:
        a = sp.Function("a")(t)
    metric = gp.Metric(
        coords=coords,
        components=[
            [-1, 0, 0, 0],
            [0, a**2 / (1 - k * r**2), 0, 0],
            [0, 0, a**2 * r**2, 0],
            [0, 0, 0, a**2 * r**2 * sp.sin(theta) ** 2],
        ],
    )
    return metric, {"a": a, "k": k, "coords": coords, "symbols": symbols}


def de_sitter(Lambda=None, coords=None, symbols=None):
    """de Sitter metric in static coordinates.

    ds^2 = -(1 - Lambda*r^2/3) dt^2 + dr^2/(1 - Lambda*r^2/3) + r^2 dOmega^2

    Parameters
    ----------
    Lambda : sympy.Symbol or None
        Cosmological constant. If None, creates Symbol('Lambda', positive=True).
    coords : OGRePy.Coordinates or None
    symbols : tuple or None

    Returns
    -------
    metric : OGRePy.Metric
    params : dict
    """
    _require_ogrepy()
    if Lambda is None:
        Lambda = sp.Symbol("Lambda", positive=True)
    if symbols is not None and len(symbols) != 4:
        raise ValueError(
            f"de_sitter: symbols must have exactly 4 elements (t, r, theta, phi), "
            f"got {len(symbols)}"
        )
    if coords is None:
        coords, symbols = spherical_coords()
    t, r, theta, phi = symbols
    f = 1 - Lambda * r**2 / 3
    metric = gp.Metric(
        coords=coords,
        components=[
            [-f, 0, 0, 0],
            [0, 1 / f, 0, 0],
            [0, 0, r**2, 0],
            [0, 0, 0, r**2 * sp.sin(theta) ** 2],
        ],
    )
    return metric, {"Lambda": Lambda, "coords": coords, "symbols": symbols}


def minkowski(dim=4):
    """Minkowski metric eta = diag(-1, +1, +1, +1, ...).

    Parameters
    ----------
    dim : int
        Number of spacetime dimensions.

    Returns
    -------
    metric : OGRePy.Metric
    params : dict
    """
    _require_ogrepy()
    coords, symbols = cartesian_coords(dim)
    eta = [[0] * dim for _ in range(dim)]
    eta[0][0] = -1
    for i in range(1, dim):
        eta[i][i] = 1
    metric = gp.Metric(coords=coords, components=eta, symbol=r"\eta")
    return metric, {"coords": coords, "symbols": symbols}


# ---------------------------------------------------------------------------
# Curvature invariants
# ---------------------------------------------------------------------------


def curvature_invariants(metric):
    """Compute all standard curvature invariants.

    Parameters
    ----------
    metric : OGRePy.Metric

    Returns
    -------
    dict with keys:
        'ricci_scalar' : sympy expression (simplified)
        'kretschmann'  : sympy expression (simplified)
        'ricci_tensor' : OGRePy tensor
        'riemann'      : OGRePy tensor
        'einstein'     : OGRePy tensor
        'christoffel'  : OGRePy tensor
    """
    _require_ogrepy()
    christoffel = metric.christoffel()
    riemann = metric.riemann()
    ricci_tensor = metric.ricci_tensor()
    ricci_scalar_tensor = metric.ricci_scalar()
    einstein = metric.einstein()
    kretschmann_tensor = metric.kretschmann()

    R_comp = ricci_scalar_tensor.components()
    R_val = sp.simplify(R_comp[0] if hasattr(R_comp, "__getitem__") else R_comp)

    K_comp = kretschmann_tensor.components()
    K_val = sp.simplify(K_comp[0] if hasattr(K_comp, "__getitem__") else K_comp)

    return {
        "ricci_scalar": R_val,
        "kretschmann": K_val,
        "ricci_tensor": ricci_tensor,
        "riemann": riemann,
        "einstein": einstein,
        "christoffel": christoffel,
    }


def verify_vacuum(metric, rtol=1e-10, n_samples=5, atol=None, seed=None):
    """Verify that a metric solves vacuum Einstein equations (G_μν = 0).

    Uses symbolic simplification first, then falls back to numerical
    evaluation at random coordinate points if symbolic check is inconclusive.
    sp.simplify is not a decision procedure for zero-testing; the numerical
    fallback catches cases where simplification fails to reach canonical form.

    Parameters
    ----------
    metric : OGRePy.Metric
    rtol : float
        Deprecated alias for atol (kept for backwards compatibility).
    n_samples : int
        Number of random coordinate points for numerical fallback.
    atol : float or None
        Absolute tolerance for numerical zero check (default: 1e-10).
        If provided, overrides rtol.
    seed : int or None
        Random seed for reproducibility of numerical fallback.

    Returns
    -------
    bool
        True if all Einstein tensor components are zero (symbolically or
        numerically within tolerance).
    """
    _require_ogrepy()
    import random

    if seed is not None:
        random.seed(seed)

    tol = atol if atol is not None else rtol

    einstein = metric.einstein()
    E_comp = einstein.components()
    dim = metric.dim()

    # Collect free symbols for numerical fallback
    free_syms = set()
    for i in range(dim):
        for j in range(dim):
            free_syms.update(E_comp[i, j].free_symbols)

    for i in range(dim):
        for j in range(dim):
            expr = sp.simplify(E_comp[i, j])
            if expr == 0:
                continue
            # sp.simplify didn't reduce to zero — try numerical evaluation
            if not free_syms:
                # No free symbols but not zero: genuinely nonzero
                return False
            # Evaluate at several random points
            is_zero = True
            for _ in range(n_samples):
                subs = {s: random.uniform(0.5, 5.0) for s in free_syms}
                try:
                    val = complex(expr.subs(subs))
                    if abs(val) > tol:
                        is_zero = False
                        break
                except (TypeError, ValueError):
                    # Can't evaluate (e.g., Function symbols) — be conservative
                    return False
            if not is_zero:
                return False
    return True


def geodesic_equations(metric, method="lagrangian"):
    """Compute geodesic equations.

    Parameters
    ----------
    metric : OGRePy.Metric
    method : str
        'lagrangian' or 'christoffel'.

    Returns
    -------
    OGRePy tensor with geodesic equations.
    """
    _require_ogrepy()
    if method == "lagrangian":
        return metric.geodesic_from_lagrangian()
    elif method == "christoffel":
        return metric.geodesic_from_christoffel()
    else:
        raise ValueError(
            f"geodesic_equations: unknown method '{method}'. "
            "Use 'lagrangian' or 'christoffel'."
        )


def line_element(metric):
    """Return the line element ds^2 as a sympy expression.

    Parameters
    ----------
    metric : OGRePy.Metric

    Returns
    -------
    sympy.Expr
    """
    _require_ogrepy()
    return metric.line_element()


# ---------------------------------------------------------------------------
# Perturbation theory
# ---------------------------------------------------------------------------


def linearized_metric(background, perturbation_symbol="h", coords=None, symbols=None):
    """Create a linearized metric: g_μν = eta_μν + epsilon * h_μν.

    Returns the symbolic perturbation matrix h_μν as a SymPy Matrix
    with symbolic components h_00, h_01, etc.

    Parameters
    ----------
    background : OGRePy.Metric
        Background metric (typically Minkowski).
    perturbation_symbol : str
        Base name for perturbation components.
    coords : OGRePy.Coordinates or None
    symbols : tuple or None

    Returns
    -------
    h_matrix : sympy.Matrix
        4x4 symmetric perturbation matrix with symbolic entries.
    epsilon : sympy.Symbol
        Book-keeping parameter for perturbation expansion.
    """
    _require_ogrepy()
    dim = background.dim()
    epsilon = sp.Symbol("epsilon")

    # Create symmetric perturbation matrix
    h = sp.zeros(dim, dim)
    for i in range(dim):
        for j in range(i, dim):
            name = f"{perturbation_symbol}_{{{i}{j}}}"
            h_ij = sp.Function(name)
            if coords is not None:
                comp = coords.components()
                h[i, j] = h_ij(*[comp[k] for k in range(dim)])
                h[j, i] = h[i, j]
            else:
                h_sym = sp.Symbol(name)
                h[i, j] = h_sym
                h[j, i] = h_sym

    return h, epsilon


# ---------------------------------------------------------------------------
# Utility: extract tensor components as dict
# ---------------------------------------------------------------------------


def tensor_to_dict(tensor, simplify_components=True):
    """Convert an OGRePy tensor to a dictionary of non-zero components.

    Parameters
    ----------
    tensor : OGRePy Tensor
        Any OGRePy tensor (Christoffel, Riemann, Ricci, etc.)
    simplify_components : bool
        Whether to apply sympy.simplify to each component.

    Returns
    -------
    dict
        Mapping from index tuple to sympy expression.
        Only non-zero components are included.
    """
    _require_ogrepy()
    comp = tensor.components()
    shape = comp.shape
    result = {}

    if len(shape) == 0:
        # Scalar
        val = sp.simplify(comp) if simplify_components else comp
        if val != 0:
            result[()] = val
        return result

    import itertools

    for idx in itertools.product(*[range(s) for s in shape]):
        val = comp[idx]
        if simplify_components:
            val = sp.simplify(val)
        if val != 0:
            result[idx] = val

    return result


# ---------------------------------------------------------------------------
# Additional standard spacetimes
# ---------------------------------------------------------------------------


def kerr(M=None, a_param=None, coords=None, symbols=None):
    """Kerr metric in Boyer-Lindquist coordinates.

    ds^2 = -(1 - 2Mr/Sigma) dt^2 - (4Mar sin^2(theta)/Sigma) dt dphi
           + (Sigma/Delta) dr^2 + Sigma dtheta^2
           + (r^2 + a^2 + 2Ma^2 r sin^2(theta)/Sigma) sin^2(theta) dphi^2

    where Sigma = r^2 + a^2 cos^2(theta), Delta = r^2 - 2Mr + a^2.

    Parameters
    ----------
    M : sympy.Symbol or None
        Mass parameter.
    a_param : sympy.Symbol or None
        Spin parameter (angular momentum per unit mass).
    coords : OGRePy.Coordinates or None
    symbols : tuple or None

    Returns
    -------
    metric : OGRePy.Metric
    params : dict
    """
    _require_ogrepy()
    if M is None:
        M = sp.Symbol("M", positive=True)
    if a_param is None:
        a_param = sp.Symbol("a", positive=True)
    if symbols is not None and len(symbols) != 4:
        raise ValueError(
            f"kerr: symbols must have exactly 4 elements (t, r, theta, phi), "
            f"got {len(symbols)}"
        )
    if coords is None:
        coords, symbols = spherical_coords()
    t, r, theta, phi = symbols

    Sigma = r**2 + a_param**2 * sp.cos(theta)**2
    Delta = r**2 - 2 * M * r + a_param**2

    g00 = -(1 - 2 * M * r / Sigma)
    g03 = -2 * M * a_param * r * sp.sin(theta)**2 / Sigma
    g11 = Sigma / Delta
    g22 = Sigma
    g33 = (r**2 + a_param**2 + 2 * M * a_param**2 * r * sp.sin(theta)**2 / Sigma) * sp.sin(theta)**2

    metric = gp.Metric(
        coords=coords,
        components=[
            [g00, 0, 0, g03],
            [0, g11, 0, 0],
            [0, 0, g22, 0],
            [g03, 0, 0, g33],
        ],
    )
    return metric, {"M": M, "a": a_param, "coords": coords, "symbols": symbols}


def reissner_nordstrom(M=None, Q=None, coords=None, symbols=None):
    """Reissner-Nordstrom metric (charged, non-rotating black hole).

    ds^2 = -f(r) dt^2 + dr^2/f(r) + r^2 dOmega^2
    f(r) = 1 - 2M/r + Q^2/r^2

    Parameters
    ----------
    M : sympy.Symbol or None
    Q : sympy.Symbol or None
        Electric charge parameter.
    coords : OGRePy.Coordinates or None
    symbols : tuple or None

    Returns
    -------
    metric : OGRePy.Metric
    params : dict
    """
    _require_ogrepy()
    if M is None:
        M = sp.Symbol("M", positive=True)
    if Q is None:
        Q = sp.Symbol("Q")
    if symbols is not None and len(symbols) != 4:
        raise ValueError(
            f"reissner_nordstrom: symbols must have exactly 4 elements (t, r, theta, phi), "
            f"got {len(symbols)}"
        )
    if coords is None:
        coords, symbols = spherical_coords()
    t, r, theta, phi = symbols

    f = 1 - 2 * M / r + Q**2 / r**2

    metric = gp.Metric(
        coords=coords,
        components=[
            [-f, 0, 0, 0],
            [0, 1 / f, 0, 0],
            [0, 0, r**2, 0],
            [0, 0, 0, r**2 * sp.sin(theta)**2],
        ],
    )
    return metric, {"M": M, "Q": Q, "coords": coords, "symbols": symbols}


def anti_de_sitter(Lambda=None, coords=None, symbols=None):
    """Anti-de Sitter metric in static coordinates.

    ds^2 = -(1 + |Lambda|*r^2/3) dt^2 + dr^2/(1 + |Lambda|*r^2/3) + r^2 dOmega^2

    Note: Lambda < 0 for AdS. The parameter here is |Lambda| (positive).

    Parameters
    ----------
    Lambda : sympy.Symbol or None
        |Lambda| (magnitude of cosmological constant, positive).
    coords : OGRePy.Coordinates or None
    symbols : tuple or None

    Returns
    -------
    metric : OGRePy.Metric
    params : dict
    """
    _require_ogrepy()
    if Lambda is None:
        Lambda = sp.Symbol("Lambda", positive=True)
    if symbols is not None and len(symbols) != 4:
        raise ValueError(
            f"anti_de_sitter: symbols must have exactly 4 elements (t, r, theta, phi), "
            f"got {len(symbols)}"
        )
    if coords is None:
        coords, symbols = spherical_coords()
    t, r, theta, phi = symbols

    f = 1 + Lambda * r**2 / 3

    metric = gp.Metric(
        coords=coords,
        components=[
            [-f, 0, 0, 0],
            [0, 1 / f, 0, 0],
            [0, 0, r**2, 0],
            [0, 0, 0, r**2 * sp.sin(theta)**2],
        ],
    )
    return metric, {"Lambda": Lambda, "coords": coords, "symbols": symbols}


def weyl_tensor(metric):
    """Compute the Weyl conformal tensor C_abcd.

    C_abcd = R_abcd - (2/(d-2))(g_a[c R_d]b - g_b[c R_d]a)
             + (2/((d-1)(d-2))) R g_a[c g_d]b

    In 4D: C_abcd = R_abcd - (g_ac R_db - g_ad R_cb - g_bc R_da + g_bd R_ca)
                     + (R/6)(g_ac g_db - g_ad g_cb)

    Parameters
    ----------
    metric : OGRePy.Metric

    Returns
    -------
    dict with:
        'components': 4D array of C_{abcd} components (sympy expressions)
        'is_conformally_flat': bool (True if all C_{abcd} = 0)
        'dim': spacetime dimension
    """
    _require_ogrepy()
    dim = metric.dim()
    if dim < 3:
        raise ValueError("weyl_tensor: only defined for dim >= 3")

    g_comp = metric.components()
    riemann = metric.riemann()
    R_abcd = riemann.components()
    ricci_t = metric.ricci_tensor()
    R_ab = ricci_t.components()
    R_scalar = metric.ricci_scalar()
    R_comp = R_scalar.components()
    R = R_comp[0] if hasattr(R_comp, "__getitem__") else R_comp

    # Lower the Riemann tensor: need g_{mu alpha} R^alpha_{nu rho sigma}
    # OGRePy Riemann is R^a_{bcd}, so we need to lower the first index
    C = sp.MutableDenseNDimArray.zeros(dim, dim, dim, dim)

    # Build lowered Riemann R_{abcd} = g_{ae} R^e_{bcd}
    R_lower = sp.MutableDenseNDimArray.zeros(dim, dim, dim, dim)
    for a in range(dim):
        for b in range(dim):
            for c in range(dim):
                for d in range(dim):
                    val = sum(g_comp[a, e] * R_abcd[e, b, c, d] for e in range(dim))
                    R_lower[a, b, c, d] = val

    # Weyl tensor in d dimensions
    # C_{abcd} = R_{abcd} - (2/(d-2)) g_{a[c}R_{d]b} + ... where [cd] = (1/2)(cd-dc)
    # After expanding antisymmetrization brackets, factor of 1/2 per bracket absorbs:
    # effective prefactor1 = 2/(d-2) * 1/2 = 1/(d-2)
    # effective prefactor2 = 2/((d-1)(d-2)) * 1/2 = 1/((d-1)(d-2))
    prefactor1 = sp.Rational(1, dim - 2)
    prefactor2 = sp.Rational(1, (dim - 1) * (dim - 2))

    for a in range(dim):
        for b in range(dim):
            for c in range(dim):
                for d in range(dim):
                    # R_{abcd} term
                    val = R_lower[a, b, c, d]
                    # -(2/(d-2)) * (g_{ac} R_{db} - g_{ad} R_{cb} - g_{bc} R_{da} + g_{bd} R_{ca})
                    val -= prefactor1 * (
                        g_comp[a, c] * R_ab[d, b]
                        - g_comp[a, d] * R_ab[c, b]
                        - g_comp[b, c] * R_ab[d, a]
                        + g_comp[b, d] * R_ab[c, a]
                    )
                    # +(2/((d-1)(d-2))) R (g_{ac} g_{db} - g_{ad} g_{cb})
                    val += prefactor2 * R * (
                        g_comp[a, c] * g_comp[d, b]
                        - g_comp[a, d] * g_comp[c, b]
                    )
                    C[a, b, c, d] = sp.simplify(val)

    # Check conformal flatness
    is_flat = all(
        C[a, b, c, d] == 0
        for a in range(dim) for b in range(dim)
        for c in range(dim) for d in range(dim)
    )

    return {
        'components': sp.ImmutableDenseNDimArray(C),
        'is_conformally_flat': is_flat,
        'dim': dim,
    }

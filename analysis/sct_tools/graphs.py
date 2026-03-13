"""
SCT Theory — Graph-theoretic and spectral geometry utilities.

Provides tools for discrete spectral geometry and causal structure analysis:
    - graph_laplacian_spectrum(): eigenvalues of the graph Laplacian
    - spectral_dimension_graph(): spectral dimension d_S from return probability
    - causal_graph(): build a causal set from a sprinkling in Minkowski/curved spacetime
    - causal_matrix(): compute the causal matrix C_ij of a causal set
    - feynman_graph(): construct Feynman diagram topology for loop counting
    - heat_kernel_trace(): Tr(e^{-t L}) on a graph (discrete analogue of spectral action)
"""

import warnings

import numpy as np

# =============================================================================
# GRAPH LAPLACIAN AND SPECTRAL ANALYSIS
# =============================================================================

def graph_laplacian_spectrum(adjacency, normalized=False):
    """Compute eigenvalues of the graph Laplacian.

    The graph Laplacian L = D - A is the discrete analogue of the
    Laplace-Beltrami operator on a manifold. Its spectrum encodes
    geometric information (connectivity, bottlenecks, dimension).

    Parameters:
        adjacency: (N, N) array or networkx.Graph
        normalized: if True, use symmetric normalized Laplacian L_sym = I - D^{-1/2} A D^{-1/2}

    Returns:
        Sorted eigenvalues (ascending), numpy array of shape (N,).
    """
    import networkx as nx

    if isinstance(adjacency, nx.Graph):
        if normalized:
            L = nx.normalized_laplacian_matrix(adjacency).toarray()
        else:
            L = nx.laplacian_matrix(adjacency).toarray().astype(float)
    else:
        A = np.asarray(adjacency, dtype=float)
        if A.ndim != 2:
            raise ValueError(
                f"graph_laplacian_spectrum: adjacency must be 2D, got {A.ndim}D array"
            )
        if A.shape[0] != A.shape[1]:
            raise ValueError(
                "graph_laplacian_spectrum: adjacency must be square, "
                f"got shape {A.shape}"
            )
        if A.shape[0] == 0:
            raise ValueError("graph_laplacian_spectrum: adjacency matrix must be non-empty")
        if np.any(~np.isfinite(A)):
            raise ValueError(
                "graph_laplacian_spectrum: adjacency matrix contains NaN or "
                "infinite value(s)"
            )
        D = np.diag(A.sum(axis=1))
        L = D - A
        if normalized:
            degrees = A.sum(axis=1)
            safe_degrees = np.where(degrees > 0, degrees, 1.0)
            d_inv_sqrt_vals = np.where(degrees > 0, 1.0 / np.sqrt(safe_degrees), 0.0)
            d_inv_sqrt = np.diag(d_inv_sqrt_vals)
            L = d_inv_sqrt @ L @ d_inv_sqrt

    eigenvalues = np.linalg.eigvalsh(L)
    return eigenvalues


def heat_kernel_trace(adjacency, t_values):
    """Compute Tr(e^{-t L}) — discrete analogue of the spectral action trace.

    The heat kernel trace on a graph encodes the same information as on a
    manifold: short-time expansion gives graph-theoretic analogues of the
    Seeley-DeWitt coefficients (number of vertices, edges, triangles, ...).

    Parameters:
        adjacency: (N, N) array or networkx.Graph
        t_values: array of diffusion times t > 0

    Returns:
        Array of Tr(e^{-tL}) values, same shape as t_values.
    """
    eigenvalues = graph_laplacian_spectrum(adjacency)
    t_values = np.asarray(t_values, dtype=float)
    if np.any(~np.isfinite(t_values)):
        raise ValueError("heat_kernel_trace: t_values contains NaN or infinite value(s)")
    if np.any(t_values <= 0):
        raise ValueError("heat_kernel_trace: all t_values must be > 0")
    # Tr(e^{-tL}) = sum_i e^{-t * lambda_i}
    return np.array([np.sum(np.exp(-t * eigenvalues)) for t in t_values])


def spectral_dimension_graph(adjacency, t_values=None, dt_frac=0.01):
    """Compute spectral dimension d_S(t) from graph Laplacian.

    d_S = -2 d(ln K)/d(ln t), where K(t) = Tr(e^{-tL}) / N
    is the return probability on the graph.

    This is the discrete analogue of the spectral dimension flow
    studied in NT-3 for the continuum spectral triple.

    Parameters:
        adjacency: (N, N) array or networkx.Graph
        t_values: diffusion times (default: logspace from 0.01 to 100)
        dt_frac: fractional step for numerical derivative

    Returns:
        (t_values, d_S_values) — arrays of diffusion times and spectral dimensions.
    """
    if t_values is None:
        t_values = np.logspace(-2, 2, 200)
    else:
        t_values = np.asarray(t_values, dtype=float)
        if t_values.size == 0:
            raise ValueError("spectral_dimension_graph: t_values must be non-empty")
        if np.any(~np.isfinite(t_values)):
            raise ValueError(
                "spectral_dimension_graph: t_values contains NaN or infinite value(s)"
            )
        if np.any(t_values <= 0):
            raise ValueError(
                "spectral_dimension_graph: t_values must all be positive"
            )

    if dt_frac <= 0 or dt_frac >= 1:
        raise ValueError(f"spectral_dimension_graph: dt_frac must be in (0, 1), got {dt_frac}")

    eigenvalues = graph_laplacian_spectrum(adjacency)
    if np.any(eigenvalues < -1e-12):
        raise ValueError("spectral_dimension_graph: Laplacian eigenvalues must be non-negative")
    N = len(eigenvalues)

    d_S = []
    for t in t_values:
        dt = t * dt_frac
        if t - dt <= 0:
            d_S.append(np.nan)
            continue
        K_plus = np.sum(np.exp(-(t + dt) * eigenvalues)) / N
        K_minus = np.sum(np.exp(-(t - dt) * eigenvalues)) / N
        if K_plus > 0 and K_minus > 0:
            dln_K = (np.log(K_plus) - np.log(K_minus)) / (2 * dt)
            d_S.append(-2.0 * t * dln_K)
        else:
            d_S.append(np.nan)

    d_S_arr = np.array(d_S)
    n_neg = np.sum(d_S_arr[np.isfinite(d_S_arr)] < 0)
    if n_neg > 0:
        warnings.warn(
            f"spectral_dimension_graph: {n_neg} negative d_S values "
            f"(may indicate disconnected graph or numerical issues)",
            stacklevel=2,
        )
    return t_values, d_S_arr


# =============================================================================
# CAUSAL SETS (discrete causal structure)
# =============================================================================

def causal_set_sprinkle(n_points, dim=2, region='flat', seed=None):
    """Generate a causal set by Poisson sprinkling into a spacetime region.

    A causal set is a locally finite partially ordered set that approximates
    a Lorentzian manifold. This is the standard construction method.

    Parameters:
        n_points: number of points to sprinkle
        dim: spacetime dimension (2, 3, or 4)
        region: 'flat' (Minkowski in [0,1]^dim hypercube),
                'diamond' (Alexandrov interval — required for Myrheim-Meyer estimator),
                'desitter' (de Sitter in 2d)
        seed: random seed for reproducibility

    Returns:
        (points, causal_matrix) where:
            points: (n_points, dim) array of spacetime coordinates [t, x, y, z]
            causal_matrix: (n_points, n_points) boolean array, C[i,j] = True if i < j
    """
    if n_points <= 0:
        raise ValueError(f"causal_set_sprinkle: n_points must be > 0, got {n_points}")
    if dim not in {2, 3, 4}:
        raise ValueError(f"causal_set_sprinkle: dim must be in {{2, 3, 4}}, got {dim}")
    if region not in {'flat', 'diamond', 'desitter'}:
        raise ValueError(
            "causal_set_sprinkle: region must be in "
            f"{{'flat', 'diamond', 'desitter'}}, got {region!r}"
        )

    rng = np.random.default_rng(seed)

    if region == 'flat':
        points = rng.uniform(0, 1, size=(n_points, dim))
        # Sort by time coordinate
        order = np.argsort(points[:, 0])
        points = points[order]
        # Causal relation: i < j iff t_i < t_j and spatial distance^2 < (t_j - t_i)^2
        C = np.zeros((n_points, n_points), dtype=bool)
        for i in range(n_points):
            for j in range(i + 1, n_points):
                dt = points[j, 0] - points[i, 0]
                dx_sq = np.sum((points[j, 1:] - points[i, 1:]) ** 2)
                if dt > 0 and dx_sq < dt ** 2:
                    C[i, j] = True
    elif region == 'diamond':
        # Alexandrov interval (causal diamond) between (0,...,0) and (1,0,...,0).
        # The diamond is {(t, x_i) : 0 < t < 1, |x|^2 < min(t, 1-t)^2}.
        # Uniform rejection sampling from bounding box [0,1] x [-0.5, 0.5]^{d-1}.
        points_list = []
        max_attempts = n_points * 200
        total_generated = 0
        while len(points_list) < n_points:
            if total_generated > max_attempts:
                raise RuntimeError(
                    f"causal_set_sprinkle: diamond rejection sampling failed to generate "
                    f"{n_points} points in {max_attempts} attempts (got {len(points_list)}). "
                    f"Try reducing n_points or dim."
                )
            batch_size = max(n_points * 5, 500)
            total_generated += batch_size
            t = rng.uniform(0, 1, size=batch_size)
            if dim > 1:
                x = rng.uniform(-0.5, 0.5, size=(batch_size, dim - 1))
                r_max = np.minimum(t, 1 - t)
                spatial_r2 = np.sum(x ** 2, axis=1)
                inside = spatial_r2 < r_max ** 2
                pts_batch = np.column_stack([t[inside], x[inside]])
            else:
                pts_batch = t[:, np.newaxis]
            for pt in pts_batch:
                points_list.append(pt)
                if len(points_list) >= n_points:
                    break
        points = np.array(points_list[:n_points])
        order = np.argsort(points[:, 0])
        points = points[order]
        C = np.zeros((n_points, n_points), dtype=bool)
        for i in range(n_points):
            for j in range(i + 1, n_points):
                dt = points[j, 0] - points[i, 0]
                dx_sq = np.sum((points[j, 1:] - points[i, 1:]) ** 2)
                if dt > 0 and dx_sq < dt ** 2:
                    C[i, j] = True
    elif region == 'desitter':
        if dim != 2:
            raise ValueError("causal_set_sprinkle: de Sitter sprinkling only implemented for dim=2")
        # Conformal coordinates: ds^2 = (1/H^2 cos^2(Ht)) (-dt^2 + dx^2)
        # Same causal structure as flat in conformal coords
        points = rng.uniform(0, 1, size=(n_points, 2))
        order = np.argsort(points[:, 0])
        points = points[order]
        C = np.zeros((n_points, n_points), dtype=bool)
        for i in range(n_points):
            for j in range(i + 1, n_points):
                dt = points[j, 0] - points[i, 0]
                dx = abs(points[j, 1] - points[i, 1])
                if dt > 0 and dx < dt:
                    C[i, j] = True
    else:
        raise ValueError(f"causal_set_sprinkle: unknown region: {region}")

    return points, C


def causal_set_to_dag(causal_matrix):
    """Convert a causal matrix to a NetworkX directed acyclic graph.

    Parameters:
        causal_matrix: (N, N) boolean array, C[i,j] = True if i causally precedes j

    Returns:
        networkx.DiGraph representing the Hasse diagram (transitive reduction).
    """
    import networkx as nx

    N = causal_matrix.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(i + 1, N):
            if causal_matrix[i, j]:
                G.add_edge(i, j)

    # Transitive reduction = Hasse diagram
    return nx.transitive_reduction(G)


def causal_set_dimension(causal_matrix, method='myrheim-meyer'):
    """Estimate the spacetime dimension from a causal set.

    The Myrheim-Meyer estimator uses the fraction of causally related pairs
    to estimate the dimension of the embedding spacetime.

    The causal set MUST be sprinkled into an Alexandrov interval (causal diamond)
    for the estimator to give correct results. Use region='diamond' in
    causal_set_sprinkle(). Hypercube sprinkling gives incorrect estimates
    because boundary effects distort the ordering fraction.

    Parameters:
        causal_matrix: (N, N) boolean array
        method: 'myrheim-meyer' (default)

    Returns:
        Estimated dimension (float).
    """
    causal_matrix = np.asarray(causal_matrix)
    if causal_matrix.ndim != 2 or causal_matrix.shape[0] != causal_matrix.shape[1]:
        raise ValueError(
            f"causal_set_dimension: causal_matrix must be a square 2-D array, "
            f"got shape {causal_matrix.shape}"
        )
    if np.any(~np.isfinite(causal_matrix.astype(float))):
        raise ValueError(
            "causal_set_dimension: causal_matrix contains NaN or infinite value(s)"
        )
    N = causal_matrix.shape[0]
    if N < 2:
        raise ValueError(f"causal_set_dimension: requires N >= 2 points, got {N}")
    n_relations = np.sum(causal_matrix)
    n_pairs = N * (N - 1) / 2
    f = n_relations / n_pairs  # fraction of related pairs

    if method == 'myrheim-meyer':
        # Brightwell-Gregory formula for d-dim Alexandrov interval:
        # f_d = Gamma(d+1) * Gamma(d/2) / (2 * Gamma(3d/2))
        # Solve numerically for d
        from scipy.optimize import brentq
        from scipy.special import gamma

        def ordering_fraction(d):
            return gamma(d + 1) * gamma(d / 2) / (2 * gamma(3 * d / 2))

        if f <= 0 or f >= 1:
            return np.nan
        try:
            d_est = brentq(lambda d: ordering_fraction(d) - f, 1.0, 10.0)
            return d_est
        except ValueError:
            import warnings
            warnings.warn(
                f"causal_set_dimension: brentq failed to find root for ordering "
                f"fraction f={f:.4f}. Dimension outside [1, 10]. Returning NaN. "
                f"Check that the causal set uses region='diamond'.",
                stacklevel=2,
            )
            return np.nan
    else:
        raise ValueError(f"causal_set_dimension: unknown method: {method}")


# =============================================================================
# FEYNMAN DIAGRAM TOPOLOGY
# =============================================================================

def feynman_graph(vertices, edges):
    """Construct a Feynman diagram as a NetworkX multigraph.

    Parameters:
        vertices: list of vertex labels (e.g., [1, 2, 3])
        edges: list of (v1, v2, propagator_type) tuples
               propagator_type: 'scalar', 'fermion', 'gluon', 'photon', 'graviton'

    Returns:
        networkx.MultiGraph with edge attributes.

    Example:
        # One-loop self-energy
        G = feynman_graph([1, 2], [(1, 2, 'scalar'), (1, 2, 'scalar')])
        print(f"Loops: {loop_number(G)}")
    """
    import networkx as nx
    G = nx.MultiGraph()
    G.add_nodes_from(vertices)
    for i, edge in enumerate(edges):
        if not isinstance(edge, (tuple, list)) or len(edge) != 3:
            raise ValueError(
                f"feynman_graph: edges[{i}] must be a (v1, v2, propagator_type) "
                f"3-tuple, got {edge!r}"
            )
    for v1, v2, ptype in edges:
        G.add_edge(v1, v2, propagator=ptype)
    return G


def loop_number(graph):
    """Compute the loop number (first Betti number) of a Feynman diagram.

    L = E - V + C, where E = edges, V = vertices, C = connected components.

    Parameters:
        graph: networkx.Graph or MultiGraph

    Returns:
        int: number of independent loops.
    """
    import networkx as nx
    if not isinstance(graph, (nx.Graph, nx.MultiGraph, nx.DiGraph, nx.MultiDiGraph)):
        raise TypeError(
            f"loop_number: expected a networkx Graph, got {type(graph).__name__}"
        )
    E = graph.number_of_edges()
    V = graph.number_of_nodes()
    C = nx.number_connected_components(graph)
    return E - V + C


def superficial_divergence(graph, dim=4):
    """Compute the superficial degree of divergence of a Feynman diagram.

    D = d*L - 2*I_B - I_F (for renormalizable theories in d dimensions),
    where L = loop number, I_B = internal bosonic lines, I_F = internal fermionic lines.

    Edges with propagator='external' are excluded from the count.

    Parameters:
        graph: networkx.MultiGraph with 'propagator' edge attributes
        dim: spacetime dimension

    Returns:
        int: superficial degree of divergence.
    """
    if not isinstance(dim, (int, float)) or dim <= 0:
        raise ValueError(
            f"superficial_divergence: dim must be a positive number, got {dim}"
        )
    L = loop_number(graph)
    I_B = 0
    I_F = 0
    for _, _, data in graph.edges(data=True):
        ptype = data.get('propagator', 'scalar')
        if ptype == 'external':
            continue
        if ptype in ('fermion',):
            I_F += 1
        else:
            I_B += 1
    return dim * L - 2 * I_B - I_F


# =============================================================================
# DISCRETE SPECTRAL ACTION
# =============================================================================

def spectral_action_on_graph(adjacency, f=None, coefficients=None, n_eigenvalues=None):
    """Compute the discrete spectral action Tr(f(L)) on a graph.

    This is the graph-theoretic analogue of the spectral action
    S = Tr(f(D^2/Lambda^2)) in noncommutative geometry.

    The trace can be computed in two ways:
    1. Direct: sum f(lambda_i) over all eigenvalues lambda_i of L
    2. Polynomial: f(x) = sum_k c_k x^k, then Tr(f(L)) = sum_k c_k Tr(L^k)

    Parameters:
        adjacency: (N, N) array or networkx.Graph
        f: callable, test function f(lambda). If None, uses polynomial from coefficients.
        coefficients: list [c_0, c_1, ..., c_n] for f(x) = sum c_k x^k.
                      Only used if f is None. Default: [1, -1] (Tr(I - L) = N - Tr(L)).
        n_eigenvalues: number of smallest eigenvalues to use (None = all).
                       Useful for large graphs where only IR modes matter.

    Returns:
        dict with:
            'action': float, the spectral action value
            'eigenvalues': array of eigenvalues used
            'contributions': array of f(lambda_i) per eigenvalue
    """
    eigenvalues = graph_laplacian_spectrum(adjacency)
    if n_eigenvalues is not None:
        if not isinstance(n_eigenvalues, int) or n_eigenvalues <= 0:
            raise ValueError(
                f"spectral_action_on_graph: n_eigenvalues must be a positive integer, "
                f"got {n_eigenvalues}"
            )
        eigenvalues = eigenvalues[:n_eigenvalues]

    if f is not None and not callable(f):
        raise TypeError(
            f"spectral_action_on_graph: f must be callable, got {type(f).__name__}"
        )
    if f is None:
        if coefficients is None:
            coefficients = [1.0, -1.0]  # Tr(I - L)
        coefficients = list(coefficients)
        if len(coefficients) == 0:
            raise ValueError(
                "spectral_action_on_graph: coefficients must be non-empty"
            )
        f = np.polynomial.polynomial.polyval
        contributions = f(eigenvalues, coefficients)
    else:
        contributions = np.array([f(lam) for lam in eigenvalues])

    return {
        'action': float(np.sum(contributions)),
        'eigenvalues': eigenvalues,
        'contributions': contributions,
    }


def zeta_function_graph(adjacency, s_values):
    """Compute the spectral zeta function zeta_L(s) = sum_{lambda_i > 0} lambda_i^{-s}.

    The spectral zeta function encodes the same information as the heat kernel
    trace via Mellin transform. Used in dimensional regularization of the
    discrete spectral action.

    Parameters:
        adjacency: (N, N) array or networkx.Graph
        s_values: array of complex s values

    Returns:
        Array of zeta_L(s) values, same shape as s_values.
    """
    eigenvalues = graph_laplacian_spectrum(adjacency)
    # Exclude zero eigenvalues (null space of L)
    pos_eigs = eigenvalues[eigenvalues > 1e-14]
    if len(pos_eigs) == 0:
        raise ValueError("zeta_function_graph: no positive eigenvalues (fully disconnected graph)")
    s_values = np.asarray(s_values)
    if np.any(np.real(s_values) <= 0):
        warnings.warn(
            "zeta_function_graph: Re(s) <= 0 may produce divergent or "
            "meaningless results (spectral zeta convergence requires Re(s) > 0)",
            stacklevel=2,
        )
    return np.array([np.sum(pos_eigs ** (-s)) for s in s_values])


# =============================================================================
# IGRAPH HIGH-PERFORMANCE BACKEND (for N > 1000)
# =============================================================================
# The igraph C library is orders of magnitude faster than pure-Python
# networkx for large graphs. Critical for MR-8 (Poisson sprinkling
# convergence tests) where N → 10^4..10^5.
#
# Functions below are igraph-accelerated versions of the networkx-based
# functions above. They auto-detect igraph availability and fall back
# to networkx if not installed.

try:
    import igraph as _ig
    _IGRAPH_OK = True
except ImportError:
    _IGRAPH_OK = False


def _has_igraph():
    """Check if igraph backend is available."""
    return _IGRAPH_OK


def causal_set_sprinkle_fast(n_points, dim=2, region='flat', seed=None):
    """High-performance causal set sprinkling using vectorized operations.

    For N > 1000, this is significantly faster than causal_set_sprinkle()
    because causal relation computation is fully vectorized.

    Parameters:
        n_points: number of sprinkled points
        dim: spacetime dimension (time + space)
        region: 'flat' (Minkowski), 'diamond' (Alexandrov interval)
        seed: random seed for reproducibility

    Returns:
        (points, causal_matrix, igraph_dag) where igraph_dag is an igraph
        directed graph (or None if igraph unavailable).
    """
    rng = np.random.default_rng(seed)

    if region == 'flat':
        points = rng.uniform(0, 1, size=(n_points, dim))
        order = np.argsort(points[:, 0])
        points = points[order]
    elif region == 'diamond':
        points_list = []
        max_attempts = n_points * 200
        total_generated = 0
        while len(points_list) < n_points:
            if total_generated > max_attempts:
                raise RuntimeError(
                    f"Diamond rejection sampling failed ({n_points} points, {dim}D)"
                )
            batch_size = max(n_points * 5, 500)
            total_generated += batch_size
            t = rng.uniform(0, 1, size=batch_size)
            if dim > 1:
                x = rng.uniform(-0.5, 0.5, size=(batch_size, dim - 1))
                r_max = np.minimum(t, 1 - t)
                spatial_r2 = np.sum(x ** 2, axis=1)
                inside = spatial_r2 < r_max ** 2
                pts_batch = np.column_stack([t[inside], x[inside]])
            else:
                pts_batch = t[:, np.newaxis]
            for pt in pts_batch:
                points_list.append(pt)
                if len(points_list) >= n_points:
                    break
        points = np.array(points_list[:n_points])
        order = np.argsort(points[:, 0])
        points = points[order]
    else:
        raise ValueError(f"causal_set_sprinkle_fast: unknown region: {region}")

    # Vectorized causal relation computation (O(N^2) but no Python loop)
    # dt[i,j] = t_j - t_i, dx2[i,j] = |x_j - x_i|^2
    # Causal: dt > 0 and dx2 < dt^2 (for i < j)
    t = points[:, 0]
    dt = t[np.newaxis, :] - t[:, np.newaxis]  # dt[i,j] = t_j - t_i

    if dim > 1:
        spatial = points[:, 1:]
        # Broadcasting: diff[i,j,k] = x_j^k - x_i^k
        diff = spatial[np.newaxis, :, :] - spatial[:, np.newaxis, :]
        dx2 = np.sum(diff ** 2, axis=2)
    else:
        dx2 = np.zeros((n_points, n_points))

    # Upper triangular (i < j): dt > 0 guaranteed by time-ordering
    C = np.triu(dx2 < dt ** 2, k=1) & np.triu(dt > 0, k=1)

    # Build igraph DAG if available
    dag = None
    if _IGRAPH_OK:
        edges = list(zip(*np.where(C)))
        dag = _ig.Graph(n=n_points, edges=edges, directed=True)
        dag.vs['time'] = points[:, 0].tolist()
        if dim > 1:
            for d in range(dim - 1):
                dag.vs[f'x{d}'] = points[:, d + 1].tolist()

    return points, C, dag


def graph_laplacian_spectrum_igraph(graph, n_eigenvalues=None):
    """Compute Laplacian eigenvalues using igraph's ARPACK interface.

    For large graphs (N > 1000), this is much faster than numpy.linalg.eigh
    on the full dense Laplacian.

    Parameters:
        graph: igraph.Graph
        n_eigenvalues: number of smallest eigenvalues to compute
            (default: all). For large graphs, compute only k << N.

    Returns:
        Array of eigenvalues (sorted ascending).
    """
    if not _IGRAPH_OK:
        raise ImportError("igraph not available")

    L = np.array(graph.laplacian())

    if n_eigenvalues is None or n_eigenvalues >= len(L):
        # Full eigendecomposition
        eigenvalues = np.linalg.eigvalsh(L)
    else:
        # Sparse eigendecomposition (ARPACK via scipy)
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
        L_sparse = csr_matrix(L)
        eigenvalues, _ = eigsh(L_sparse, k=n_eigenvalues, which='SM')
        eigenvalues = np.sort(eigenvalues)

    return eigenvalues


def spectral_dimension_igraph(graph, t_values=None, n_eigenvalues=None):
    """Compute spectral dimension from an igraph Graph.

    d_S(t) = -2 * d(log K(t)) / d(log t)
    where K(t) = (1/N) Tr(e^{-tL}) is the return probability.

    Parameters:
        graph: igraph.Graph
        t_values: diffusion times (default: logspace(-2, 2, 100))
        n_eigenvalues: use only k smallest eigenvalues (performance)

    Returns:
        (t_values, d_S) arrays
    """
    if not _IGRAPH_OK:
        raise ImportError("igraph not available")

    if t_values is None:
        t_values = np.logspace(-2, 2, 100)
    t_values = np.asarray(t_values, dtype=float)

    eigenvalues = graph_laplacian_spectrum_igraph(graph, n_eigenvalues)
    N = graph.vcount()

    # K(t) = (1/N) sum_i exp(-t * lambda_i)
    K = np.array([np.sum(np.exp(-t * eigenvalues)) / N for t in t_values])

    # d_S = -2 * d(log K) / d(log t)
    log_K = np.log(np.maximum(K, 1e-300))
    log_t = np.log(t_values)
    d_S = -2.0 * np.gradient(log_K, log_t)

    return t_values, d_S


def causal_set_ordering_fraction_fast(causal_matrix):
    """Compute ordering fraction f = 2*C_2 / (N*(N-1)) using vectorized ops.

    This is the key input to the Myrheim-Meyer dimension estimator.
    Vectorized version — no Python loops.

    Parameters:
        causal_matrix: (N, N) boolean array

    Returns:
        float: ordering fraction
    """
    N = causal_matrix.shape[0]
    if N < 2:
        return 0.0
    # Count ordered pairs (both upper and lower triangle)
    n_ordered = np.sum(causal_matrix)
    return 2.0 * n_ordered / (N * (N - 1))


def benchmark_backends(n_points=500, dim=2, seed=42):
    """Compare performance: networkx vs igraph for causal set operations.

    Returns dict with timing results.
    """
    import time

    results = {}

    # Old (networkx-based)
    t0 = time.perf_counter()
    pts_old, C_old = causal_set_sprinkle(n_points, dim=dim, region='flat', seed=seed)
    t_nx = time.perf_counter() - t0
    results['networkx_sprinkle'] = t_nx

    # New (igraph-based)
    t0 = time.perf_counter()
    pts_new, C_new, dag = causal_set_sprinkle_fast(n_points, dim=dim, region='flat', seed=seed)
    t_ig = time.perf_counter() - t0
    results['igraph_sprinkle'] = t_ig
    results['speedup'] = t_nx / t_ig if t_ig > 0 else float('inf')

    # Verify results agree
    results['agree'] = np.allclose(C_old, C_new)

    print(f"Benchmark (N={n_points}, dim={dim}):")
    print(f"  networkx: {t_nx:.3f}s")
    print(f"  igraph:   {t_ig:.3f}s")
    print(f"  speedup:  {results['speedup']:.1f}x")
    print(f"  agree:    {results['agree']}")

    return results

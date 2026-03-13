"""
NT-4a symbolic linearization helpers.

The goal of this module is not to reproduce a full tensor-CAS derivation of the
field equations inside Python. Instead it provides deterministic symbolic
identities, projector utilities, and serialization helpers that the phase-local
verification pipeline can consume and cross-check.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import sympy as sp

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "analysis" / "results" / "nt4a"

BOX = sp.Symbol("Box", real=True)
h_tt = sp.Symbol("h_TT")


def linearized_curvature_identities() -> dict[str, sp.Expr]:
    """Return the flat-background TT-gauge curvature identities used after off-shell checks."""
    return {
        "Ricci_tensor_TT": -BOX * h_tt / 2,
        "Ricci_scalar_TT": sp.Integer(0),
        "Einstein_tensor_TT": -BOX * h_tt / 2,
        "Weyl_squared_TT": BOX**2 * h_tt**2 / 2,
        "R_squared_TT": sp.Integer(0),
    }


def serialize_linearized_identities(output_path: Path | None = None) -> Path:
    if output_path is None:
        output_path = RESULTS_DIR / "nt4a_linearized_identities.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        key: sp.srepr(value) for key, value in linearized_curvature_identities().items()
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _validate_k_vector(k_vec: np.ndarray) -> np.ndarray:
    k_arr = np.asarray(k_vec, dtype=float)
    if k_arr.shape != (4,):
        raise ValueError(f"k_vec must have shape (4,), got {k_arr.shape}")
    k2 = float(np.dot(k_arr, k_arr))
    if k2 <= 0:
        raise ValueError(f"k_vec must have positive Euclidean norm, got {k2}")
    return k_arr


def theta_projector(k_vec: np.ndarray) -> np.ndarray:
    """Return theta_{mu nu} = delta_{mu nu} - k_mu k_nu / k^2."""
    k_arr = _validate_k_vector(k_vec)
    k2 = float(np.dot(k_arr, k_arr))
    return np.eye(4) - np.outer(k_arr, k_arr) / k2


def tt_projector(k_vec: np.ndarray) -> np.ndarray:
    """Barnes-Rivers spin-2 projector P^(2)."""
    theta = theta_projector(k_vec)
    projector = np.zeros((4, 4, 4, 4), dtype=float)
    for mu in range(4):
        for nu in range(4):
            for rho in range(4):
                for sigma in range(4):
                    projector[mu, nu, rho, sigma] = (
                        0.5 * (
                            theta[mu, rho] * theta[nu, sigma]
                            + theta[mu, sigma] * theta[nu, rho]
                        )
                        - theta[mu, nu] * theta[rho, sigma] / 3
                    )
    return projector


def scalar_projector(k_vec: np.ndarray) -> np.ndarray:
    """Barnes-Rivers scalar projector P^(0-s)."""
    theta = theta_projector(k_vec)
    projector = np.zeros((4, 4, 4, 4), dtype=float)
    for mu in range(4):
        for nu in range(4):
            for rho in range(4):
                for sigma in range(4):
                    projector[mu, nu, rho, sigma] = theta[mu, nu] * theta[rho, sigma] / 3
    return projector


def contract_first_index_with_k(projector: np.ndarray, k_vec: np.ndarray) -> np.ndarray:
    k_arr = _validate_k_vector(k_vec)
    return np.tensordot(k_arr, projector, axes=(0, 0))


def gauge_mode_tensor(k_vec: np.ndarray, xi_vec: np.ndarray) -> np.ndarray:
    k_arr = _validate_k_vector(k_vec)
    xi_arr = np.asarray(xi_vec, dtype=float)
    if xi_arr.shape != (4,):
        raise ValueError(f"xi_vec must have shape (4,), got {xi_arr.shape}")
    return np.outer(k_arr, xi_arr) + np.outer(xi_arr, k_arr)


def default_symmetric_tensor() -> np.ndarray:
    """Deterministic off-shell symmetric metric perturbation."""
    return np.array(
        [
            [1.0, 0.2, -0.1, 0.3],
            [0.2, -0.7, 0.4, 0.1],
            [-0.1, 0.4, 0.5, -0.2],
            [0.3, 0.1, -0.2, 0.9],
        ],
        dtype=float,
    )


def linearized_einstein_tensor(k_vec: np.ndarray, h_tensor: np.ndarray) -> np.ndarray:
    """
    Return the off-shell linearized Einstein tensor in momentum space.

    The expression is evaluated before imposing any gauge condition:
        G^(1)_{mu nu}(k; h)
        = 1/2 [k_mu k^rho h_{rho nu} + k_nu k^rho h_{rho mu}
               - k^2 h_{mu nu} - k_mu k_nu h
               - delta_{mu nu}(k_r k_s h^{rs} - k^2 h)].
    """
    k_arr = _validate_k_vector(k_vec)
    h_arr = np.asarray(h_tensor, dtype=float)
    if h_arr.shape != (4, 4):
        raise ValueError(f"h_tensor must have shape (4, 4), got {h_arr.shape}")
    if not np.allclose(h_arr, h_arr.T, atol=1e-12):
        raise ValueError("h_tensor must be symmetric")

    k2 = float(np.dot(k_arr, k_arr))
    trace_h = float(np.trace(h_arr))
    kh = k_arr @ h_arr
    divdiv_h = float(k_arr @ h_arr @ k_arr)
    delta = np.eye(4)
    return 0.5 * (
        np.outer(k_arr, kh)
        + np.outer(kh, k_arr)
        - k2 * h_arr
        - np.outer(k_arr, k_arr) * trace_h
        - delta * (divdiv_h - k2 * trace_h)
    )


def check_off_shell_gauge_invariance(
    k_vec: np.ndarray,
    xi_vec: np.ndarray | None = None,
    tol: float = 1e-10,
) -> bool:
    if xi_vec is None:
        xi_vec = np.array([0.3, -0.4, 0.1, 0.2], dtype=float)
    gauge_mode = gauge_mode_tensor(k_vec, xi_vec)
    einstein = linearized_einstein_tensor(k_vec, gauge_mode)
    return float(np.max(np.abs(einstein))) < tol


def check_off_shell_bianchi_identity(
    k_vec: np.ndarray,
    h_tensor: np.ndarray | None = None,
    tol: float = 1e-10,
) -> bool:
    if h_tensor is None:
        h_tensor = default_symmetric_tensor()
    einstein = linearized_einstein_tensor(k_vec, h_tensor)
    contracted = np.tensordot(_validate_k_vector(k_vec), einstein, axes=(0, 0))
    return float(np.max(np.abs(contracted))) < tol


def random_k_vectors(seed: int = 42, n_vectors: int = 5) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    vectors = []
    while len(vectors) < n_vectors:
        candidate = rng.normal(size=4)
        if np.dot(candidate, candidate) > 0.25:
            vectors.append(candidate)
    return vectors


def random_symmetric_tensors(seed: int = 43, n_tensors: int = 5) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    tensors = []
    for _ in range(n_tensors):
        candidate = rng.normal(size=(4, 4))
        tensors.append(0.5 * (candidate + candidate.T))
    return tensors


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serialize NT-4a symbolic linearization identities.")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "nt4a_linearized_identities.json")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    path = serialize_linearized_identities(args.output)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()

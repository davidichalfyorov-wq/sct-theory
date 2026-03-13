"""
SCT Tools — Computational toolkit for Spectral Causal Theory.

Modules:
    constants      — Physical constants, unit conversions, SCT conventions
    form_factors   — Heat kernel form factors h_C, h_R for spins 0, 1/2, 1
    verification   — 8-layer verification
                     (analytic, numerical, property, literature, dual,
                      triple CAS, Lean, consensus)
    compute        — Parallel scans, progress bars, JAX autodiff/JIT, symengine, WSL integration
    data_io        — HDF5, ROOT, FITS, JSON I/O for experimental data
    entire_function — Entire-function and complex-plane diagnostics for NT-2
    fitting        — Parameter estimation (iminuit, lmfit, emcee, pymc, pyhf), statistical tests
    graphs         — Spectral graph theory, causal sets
                     (networkx + igraph backends), Feynman diagram topology
    tensors        — GR tensor algebra via OGRePy (classical limit verification)
    entanglement   — Entanglement measures and tensor networks via quimb (Axiom 5)
    lean           — Lean 4 formal verification:
                     Aristotle cloud + local PhysLean/Mathlib4 + WSL SciLean
    cas_backends   — Triple CAS cross-verification: SymPy × GiNaC × mpmath (Layer 4.5)
    form_interface — FORM 5.0 subprocess interface for large symbolic expressions
                     (gamma traces, heat kernel)
    plotting       — Publication-quality plotting defaults (SciencePlots, PRL/PRD compatible)
    propagator     — Linearized-field and modified-propagator utilities for NT-4a
"""

__version__ = "0.7.0"

from . import cas_backends as cas_backends
from . import compute as compute
from . import constants as constants
from . import data_io as data_io
from . import entanglement as entanglement
from . import entire_function as entire_function
from . import fitting as fitting
from . import form_factors as form_factors
from . import form_interface as form_interface
from . import graphs as graphs
from . import lean as lean
from . import plotting as plotting
from . import propagator as propagator
from . import tensors as tensors
from . import verification as verification
from .constants import log_dimensions
from .form_factors import (
    dhC_dirac_dx,
    dhC_scalar_dx,
    dhC_vector_dx,
    dhR_dirac_dx,
    dhR_scalar_dx,
    dhR_vector_dx,
)

__all__ = [
    "cas_backends", "compute", "constants", "data_io", "entanglement",
    "entire_function", "fitting", "form_factors", "form_interface",
    "graphs", "lean", "plotting", "propagator", "tensors", "verification",
    "dhC_dirac_dx", "dhC_scalar_dx", "dhC_vector_dx",
    "dhR_dirac_dx", "dhR_scalar_dx", "dhR_vector_dx",
    "log_dimensions",
]

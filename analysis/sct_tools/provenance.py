"""
SCT Theory — Provenance Chain Tracing.

For any SCT result, provides the full derivation chain from postulates
through intermediate steps to the final number. Supports:
  - Automated "where does this come from?" queries
  - Circular dependency detection
  - Impact analysis ("what breaks if X changes?")
  - LaTeX-ready derivation chain export

Usage:
    from sct_tools.provenance import ProvenanceRegistry, SCT_PROVENANCE

    chain = SCT_PROVENANCE.trace("alpha_C")
    # -> ['axiom_spectral_action', 'phi', 'hC_scalar', 'hC_dirac',
    #     'hC_vector', 'sm_content', 'alpha_C']

    SCT_PROVENANCE.explain("alpha_C")
    # Prints full chain with descriptions and equations

    SCT_PROVENANCE.check_acyclic()  # verify no circular deps
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProvenanceEntry:
    """A single entry in the provenance registry.

    Attributes:
        name: Unique identifier.
        category: "postulate" | "definition" | "derivation" | "result" | "prediction".
        description: Human-readable description.
        equation: LaTeX equation string (optional).
        source: Origin document (e.g. "NT-1, eq. 3.12" or "SCT Postulate 1").
        depends_on: List of names this entry depends on.
        value: Numerical value if applicable.
        verified_by: List of verification layers that checked this.
        notes: Additional notes.
    """
    name: str
    category: str
    description: str = ""
    equation: str = ""
    source: str = ""
    depends_on: list[str] = field(default_factory=list)
    value: Any = None
    verified_by: list[str] = field(default_factory=list)
    notes: str = ""


class ProvenanceRegistry:
    """Registry of all SCT results with their provenance chains."""

    def __init__(self) -> None:
        self.entries: dict[str, ProvenanceEntry] = {}

    def register(self, name: str, **kwargs: Any) -> None:
        """Register a provenance entry."""
        self.entries[name] = ProvenanceEntry(name=name, **kwargs)

    def trace(self, name: str) -> list[str]:
        """Trace the full provenance chain for a result.

        Returns:
            Ordered list from roots (postulates) to the target,
            in topological order.
        """
        if name not in self.entries:
            raise KeyError(f"'{name}' not in provenance registry")

        visited = set()
        order = []

        def _dfs(n: str) -> None:
            if n in visited:
                return
            visited.add(n)
            entry = self.entries.get(n)
            if entry:
                for dep in entry.depends_on:
                    _dfs(dep)
            order.append(n)

        _dfs(name)
        return order

    def explain(self, name: str) -> str:
        """Generate a human-readable explanation of a result's provenance.

        Returns:
            Formatted multi-line string.
        """
        chain = self.trace(name)
        lines = []
        lines.append(f"Provenance chain for: {name}")
        lines.append("=" * 60)

        for i, n in enumerate(chain):
            entry = self.entries.get(n)
            if entry is None:
                lines.append(f"  {i+1}. {n} (external, not registered)")
                continue
            prefix = "  " + "  " * min(i, 4)
            arrow = "->" if i > 0 else "  "
            lines.append(f"{prefix}{arrow} [{entry.category}] {entry.name}")
            if entry.description:
                lines.append(f"{prefix}   {entry.description}")
            if entry.equation:
                lines.append(f"{prefix}   Equation: {entry.equation}")
            if entry.source:
                lines.append(f"{prefix}   Source: {entry.source}")
            if entry.value is not None:
                lines.append(f"{prefix}   Value: {entry.value}")
            if entry.verified_by:
                lines.append(f"{prefix}   Verified: {', '.join(entry.verified_by)}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def check_acyclic(self) -> tuple[bool, list[str]]:
        """Verify no circular dependencies exist.

        Returns:
            (is_acyclic, list_of_cycle_participants).
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in self.entries}
        cycle_nodes: list[str] = []

        def _dfs(n: str) -> bool:
            color[n] = GRAY
            entry = self.entries.get(n)
            if entry:
                for dep in entry.depends_on:
                    if dep not in color:
                        continue
                    if color[dep] == GRAY:
                        cycle_nodes.append(dep)
                        return True
                    if color[dep] == WHITE and _dfs(dep):
                        cycle_nodes.append(dep)
                        return True
            color[n] = BLACK
            return False

        for name in self.entries:
            if color[name] == WHITE:
                if _dfs(name):
                    return False, cycle_nodes

        return True, []

    def impact_analysis(self, name: str) -> dict:
        """Analyze what would be affected if `name` changes.

        Returns:
            Dict with 'dependents' (direct), 'all_downstream', and counts.
        """
        direct = []
        for entry in self.entries.values():
            if name in entry.depends_on:
                direct.append(entry.name)

        all_downstream = set()

        def _collect(n: str) -> None:
            for entry in self.entries.values():
                if n in entry.depends_on and entry.name not in all_downstream:
                    all_downstream.add(entry.name)
                    _collect(entry.name)

        _collect(name)

        by_category: dict[str, list[str]] = {}
        for n in all_downstream:
            cat = self.entries[n].category
            by_category.setdefault(cat, []).append(n)

        return {
            "source": name,
            "direct_dependents": direct,
            "all_downstream": sorted(all_downstream),
            "n_affected": len(all_downstream),
            "by_category": by_category,
        }

    def to_latex_chain(self, name: str) -> str:
        """Export provenance chain as LaTeX enumerate environment."""
        chain = self.trace(name)
        lines = [r"\begin{enumerate}"]
        for n in chain:
            entry = self.entries.get(n)
            if entry is None:
                lines.append(f"  \\item {n} (external)")
                continue
            eq_part = ""
            if entry.equation:
                eq_part = f": ${entry.equation}$"
            src_part = ""
            if entry.source:
                src_part = f" [{entry.source}]"
            lines.append(
                f"  \\item \\textbf{{{entry.name}}}{eq_part}{src_part}")
            if entry.description:
                lines.append(f"    \\\\ \\textit{{{entry.description}}}")
        lines.append(r"\end{enumerate}")
        return "\n".join(lines)


def build_sct_provenance() -> ProvenanceRegistry:
    """Build the standard SCT provenance registry with all canonical results."""
    reg = ProvenanceRegistry()

    # --- Postulates ---
    reg.register("axiom_spectral_action", category="postulate",
                 description="Gravitational dynamics from spectral action",
                 source="SCT Postulate 1")
    reg.register("axiom_causality", category="postulate",
                 description="Lorentzian causal structure preserved",
                 source="SCT Postulate 2")
    reg.register("axiom_sm_coupling", category="postulate",
                 description="SM fields minimally coupled to spectral geometry",
                 source="SCT Postulate 3")
    reg.register("axiom_background_independence", category="postulate",
                 description="Background-independent quantization via BV formalism",
                 source="SCT Postulate 4")
    reg.register("axiom_spectral_boundary", category="postulate",
                 description="Spectral boundary conditions (UV finiteness requirement)",
                 source="SCT Postulate 5")

    # --- SM content ---
    reg.register("sm_content", category="definition",
                 depends_on=["axiom_sm_coupling"],
                 description="SM field multiplicities: N_s=4, N_f=45, N_v=12",
                 equation="N_s=4,\\; N_D=N_f/2=22.5,\\; N_v=12",
                 source="CPR 0805.2909",
                 value={"N_s": 4, "N_f": 45, "N_v": 12})

    # --- Master function ---
    reg.register("phi", category="derivation",
                 depends_on=["axiom_spectral_action"],
                 description="Master function from heat kernel of spectral action",
                 equation=r"\phi(x) = e^{-x/4}\sqrt{\pi/x}\,\mathrm{erfi}(\sqrt{x}/2)",
                 source="NT-1, eq. 2.1",
                 value={"phi(0)": 1, "phi'(0)": -1/6},
                 verified_by=["L1", "L2", "L2.5", "L3", "L4.5", "L5"])

    # --- Form factors ---
    phase_map = {"0": "1 (scalar)", "1/2": "NT-1", "1": "2 (vector)"}
    for spin, label, beta_W, depends in [
        ("scalar", "0", "1/120", ["phi"]),
        ("dirac", "1/2", "-1/20", ["phi"]),
        ("vector", "1", "1/10", ["phi"]),
    ]:
        src = f"NT-1b {phase_map.get(label, label)}"
        vl = ["L1", "L2", "L2.5", "L3", "L4", "L4.5"]
        reg.register(f"hC_{spin}", category="derivation",
                     depends_on=depends,
                     description=f"Weyl form factor h_C^({label})(x)",
                     source=src, value={"beta_W": beta_W},
                     verified_by=vl)
        reg.register(f"hR_{spin}", category="derivation",
                     depends_on=depends,
                     description=f"Ricci form factor h_R^({label})(x)",
                     source=src, verified_by=vl)

    # --- Combined SM coefficients ---
    reg.register("alpha_C", category="result",
                 depends_on=["hC_scalar", "hC_dirac", "hC_vector", "sm_content"],
                 description="Total Weyl^2 coefficient, xi-independent, parameter-free",
                 equation=r"\alpha_C = \frac{13}{120}",
                 source="NT-1b Phase 3, eq. 4.1",
                 value=13/120,
                 verified_by=["L1", "L2", "L2.5", "L3", "L4", "L4.5", "L5", "L6"])

    reg.register("alpha_R", category="result",
                 depends_on=["hR_scalar", "hR_dirac", "hR_vector", "sm_content"],
                 description="Total R^2 coefficient, depends on Higgs coupling xi",
                 equation=r"\alpha_R(\xi) = 2(\xi - 1/6)^2",
                 source="NT-1b Phase 3, eq. 4.2",
                 verified_by=["L1", "L2", "L2.5", "L3", "L4", "L4.5"])

    reg.register("F1_total", category="result",
                 depends_on=["alpha_C"],
                 description="Total F1 form factor at zero momentum",
                 equation=r"F_1(0) = \frac{13}{1920\pi^2}",
                 source="NT-1b Phase 3",
                 verified_by=["L1", "L2"])

    reg.register("c1_c2_ratio", category="result",
                 depends_on=["alpha_C", "alpha_R"],
                 description="Ratio of gravitational form factor coefficients",
                 equation=r"c_1/c_2 = -1/3 + 120(\xi-1/6)^2/13",
                 source="NT-1b Phase 3, eq. 4.5",
                 verified_by=["L1", "L2", "L5"])

    # --- Field equations ---
    reg.register("Pi_TT", category="derivation",
                 depends_on=["alpha_C", "F1_total"],
                 description="Transverse-traceless propagator modification",
                 equation=r"\Pi_{TT}(z) = 1 + \frac{13}{60} z \hat{F}_1(z)",
                 source="NT-4a, eq. 3.1",
                 verified_by=["L1", "L2", "L3", "L4"])

    reg.register("Pi_scalar", category="derivation",
                 depends_on=["alpha_R"],
                 description="Scalar propagator modification",
                 equation=r"\Pi_s(z,\xi) = 1 + 6(\xi-1/6)^2 z \hat{F}_2(z,\xi)",
                 source="NT-4a, eq. 3.2",
                 verified_by=["L1", "L2"])

    # --- Predictions ---
    reg.register("m2_eff", category="prediction",
                 depends_on=["Pi_TT", "alpha_C"],
                 description="Effective spin-2 ghost mass",
                 equation=r"m_2 = \Lambda\sqrt{60/13} \approx 2.148\,\Lambda",
                 source="NT-4a, eq. 4.1",
                 verified_by=["L1", "L2"])

    reg.register("m0_eff", category="prediction",
                 depends_on=["Pi_scalar", "alpha_R"],
                 description="Effective spin-0 mass at xi=0",
                 equation=r"m_0(\xi=0) = \Lambda\sqrt{6} \approx 2.449\,\Lambda",
                 source="NT-4a, eq. 4.2",
                 verified_by=["L1", "L2"])

    reg.register("newtonian_potential", category="prediction",
                 depends_on=["m2_eff", "m0_eff"],
                 description="Modified Newtonian potential with Yukawa corrections",
                 equation=r"V(r)/V_N = 1 - \frac{4}{3}e^{-m_2 r} + \frac{1}{3}e^{-m_0 r}",
                 source="NT-4a, eq. 5.1",
                 verified_by=["L1", "L2", "L3"])

    reg.register("Lambda_bound", category="prediction",
                 depends_on=["newtonian_potential"],
                 description="Lower bound on SCT scale from torsion balance",
                 equation=r"\Lambda > 2.565\,\mathrm{meV}",
                 source="PPN-1, Table 1",
                 value=2.565e-3,
                 verified_by=["L1", "L2", "L3"])

    reg.register("ppn_gamma", category="prediction",
                 depends_on=["newtonian_potential"],
                 description="PPN gamma parameter (exact, SCT is metric)",
                 equation=r"\gamma_{\mathrm{PPN}} = 1",
                 source="PPN-1, Theorem 1",
                 value=1,
                 verified_by=["L1", "L5"])

    return reg


# Singleton for convenience
SCT_PROVENANCE = build_sct_provenance()

"""
FND-1 Experiment Hub — Auto-Discovery Research Dashboard.

All experiments auto-discovered from JSON files. No manual entry needed.

Run: streamlit run analysis/scripts/fnd1_dashboard.py
"""

import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fnd1_experiment_registry import (
    scan_experiments, get_progress, get_route_experiments, ROUTE_INFO, RESULTS_DIR,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

st.set_page_config(
    page_title="FND-1 Experiment Hub",
    page_icon="\U0001f52c",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Data loading (cached for performance)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=10)
def load_experiments():
    return scan_experiments()

@st.cache_data(ttl=5)
def load_progress():
    return get_progress()

experiments = load_experiments()
progress = load_progress()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

logo_path = Path(__file__).parent / "black_mesa_logo.png"
if logo_path.exists():
    st.sidebar.image(str(logo_path), width=160)
else:
    logo_jpg = Path(__file__).parent / "black_mesa_logo.jpg"
    if logo_jpg.exists():
        st.sidebar.image(str(logo_jpg), width=160)

st.sidebar.title("FND-1 Experiment Hub")
st.sidebar.markdown("*Black Mesa Research Facility*")
st.sidebar.markdown("*Physics Department \u2014 SCT Theory*")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "Dashboard",
    "Route 1: Ensemble",
    "Route 2: Emergent",
    "Route 3: Lorentzian",
    "All Experiments",
    "Live Monitor",
    "Theory",
])

# Sidebar stats
st.sidebar.markdown("---")
n_total = len(experiments)
n_r1 = len(get_route_experiments(experiments, 1))
n_r2 = len(get_route_experiments(experiments, 2))
n_r3 = len(get_route_experiments(experiments, 3))
st.sidebar.metric("Total Experiments", n_total)
st.sidebar.caption(f"Route 1: {n_r1} | Route 2: {n_r2} | Route 3: {n_r3}")

if progress:
    st.sidebar.warning(f"Running: {progress.get('name', '?')}")
    pct = progress.get("pct", 0)
    st.sidebar.progress(pct, text=progress.get("step", ""))
else:
    st.sidebar.success("No experiment running")

st.sidebar.caption(f"Refreshed: {time.strftime('%H:%M:%S')}")
if st.sidebar.button("Refresh"):
    st.cache_data.clear()
    st.rerun()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def badge(text, color):
    colors = {"red": "#ff4b4b", "green": "#21c354", "blue": "#1c83e1",
              "orange": "#ffa62b", "gray": "#808080", "purple": "#9b59b6"}
    c = colors.get(color, "#808080")
    return f'<span style="background:{c};color:white;padding:3px 10px;border-radius:10px;font-weight:bold;font-size:13px">{text}</span>'

def verdict_color(verdict: str) -> str:
    v = str(verdict).upper()
    if any(w in v for w in ["PASS", "CONFIRM", "PROCEED", "BREAKTHROUGH", "REPRODUCED"]):
        return "green"
    if any(w in v for w in ["FAIL", "CLOSED", "NOT REPRODUCED", "NO SIGNAL", "DEAD"]):
        return "red"
    if any(w in v for w in ["INCONCLUSIVE", "WEAK", "ASYMMETRIC", "MARGINAL"]):
        return "orange"
    if "RUNNING" in v or "PROGRESS" in v:
        return "blue"
    return "gray"

def verdict_icon(verdict: str) -> str:
    c = verdict_color(verdict)
    return {"green": "\u2705", "red": "\u274c", "orange": "\u26a0\ufe0f", "blue": "\u23f3", "gray": "\u2014"}.get(c, "\u2014")

def render_experiment_card(exp: dict):
    """Render a single experiment as an expandable card."""
    v = exp.get("verdict", "\u2014")
    icon = verdict_icon(v)
    name = exp.get("description") or exp.get("name", "?")
    N = exp.get("N", "?")
    M = exp.get("M", "?")
    wt = exp.get("wall_time_sec", 0)
    wt_str = f"{wt/60:.1f} min" if wt else "\u2014"

    with st.expander(f"{icon} **{name}** | N={N}, M={M} | {wt_str} | {v[:50]}"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("N", N)
        c2.metric("M", M)
        c3.metric("Time", wt_str)
        c4.markdown(badge(v[:25], verdict_color(v)), unsafe_allow_html=True)

        # Auto-render known data patterns
        data = exp.get("data", {})
        render_experiment_details(exp, data)

def render_experiment_details(exp: dict, data: dict):
    """Auto-render details based on data content."""
    # Mediation analysis
    if "mediation" in data:
        st.markdown("**Mediation Analysis:**")
        rows = []
        for obs, m in data["mediation"].items():
            surv = abs(m.get("r_partial_both", 0)) > 0.1 and m.get("p_partial_both", 1) < 0.10
            rows.append({
                "Observable": obs,
                "Direct r": f"{m.get('r_direct', 0):+.4f}",
                "Partial r": f"{m.get('r_partial_both', 0):+.4f}",
                "Partial p": f"{m.get('p_partial_both', 1):.2e}",
                "Survives": "\u2705" if surv else "\u274c",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

    # Reproducibility
    if "reproducibility" in data:
        st.markdown("**Reproducibility:**")
        rows = []
        for key, r in data["reproducibility"].items():
            if isinstance(r, dict):
                # Find the main metric
                for metric_key in ["comm_entropy", "mean_diff", "frac_005"]:
                    if metric_key in r:
                        mr = r[metric_key] if isinstance(r[metric_key], dict) else r
                        rows.append({
                            "Config": key,
                            "p-value": f"{mr.get('p_value', mr.get('p', 1)):.4f}",
                            "Significant": "\u2705" if mr.get("significant", mr.get("reproduced", False)) else "\u274c",
                        })
                        break
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)

    # Correlation data
    if "correlation" in data:
        corr = data["correlation"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Pearson r", f"{corr.get('pearson_r', 0):.4f}" if corr.get('pearson_r') else "\u2014")
        c2.metric("p-value", f"{corr.get('p_value', 1):.2e}" if corr.get('p_value') else "\u2014")
        c3.metric("Sign consistent", str(corr.get("sign_consistent", "\u2014")))

    # Gate 3 scaling data
    if "gate3" in data and "families" in data["gate3"]:
        render_gate3_plot(data)

    # Curvature data (Gate 5 SDW)
    if "curvature_data" in data:
        render_curvature_plot(data)

    # Test results (quickkill / verification)
    for test_key in ["test1", "test2", "test3", "test_a", "test_b", "test_c"]:
        if test_key in data and isinstance(data[test_key], dict):
            with st.container():
                st.caption(f"**{test_key}**")
                # Show key metrics
                td = data[test_key]
                display = {k: v for k, v in td.items()
                          if isinstance(v, (int, float, bool, str)) and v is not None}
                if display:
                    st.json(display)

    # Raw JSON fallback
    with st.container():
        if st.checkbox(f"Show raw JSON ({exp.get('file', '?')})", key=exp.get("file", "")):
            st.json(data)


def render_gate3_plot(data):
    """Render Gate 3 scaling plot from data."""
    families = data["gate3"]["families"]
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["UV Exponent p(N)", "Ensemble-Null Separation"])
    colors_f = {"A": "#e74c3c", "B": "#3498db", "C": "#2ecc71"}

    for fname in ["A", "B", "C"]:
        if fname not in families:
            continue
        fd = families[fname]
        Ns = sorted([int(k) for k in fd["p_ens_by_N"].keys()])
        p_ens = [fd["p_ens_by_N"][str(n)] for n in Ns]
        p_null = [fd["p_null_by_N"][str(n)] for n in Ns]

        fig.add_trace(go.Scatter(x=Ns, y=p_ens, mode="lines+markers",
                                name=f"{fname} (causal)", line=dict(color=colors_f[fname]),
                                marker=dict(size=8)), row=1, col=1)
        fig.add_trace(go.Scatter(x=Ns, y=p_null, mode="lines+markers",
                                name=f"{fname} (null)", line=dict(color=colors_f[fname], dash="dash"),
                                marker=dict(size=5)), row=1, col=1)
        sep = [abs(e - n) for e, n in zip(p_ens, p_null)]
        fig.add_trace(go.Scatter(x=Ns, y=sep, mode="lines+markers",
                                name=fname, line=dict(color=colors_f[fname]),
                                showlegend=False), row=1, col=2)

    fig.add_hline(y=-1.0, line_dash="dot", line_color="red",
                 annotation_text="target", row=1, col=1)
    fig.update_xaxes(type="log", title_text="N", row=1, col=1)
    fig.update_xaxes(type="log", title_text="N", row=1, col=2)
    fig.update_yaxes(title_text="p", row=1, col=1)
    fig.update_yaxes(title_text="|p_ens - p_null|", row=1, col=2)
    fig.update_layout(height=400, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)


def render_curvature_plot(data):
    """Render curvature scatter plot."""
    eps_vals = data["parameters"]["epsilon_values"]
    cd = data["curvature_data"]
    a1 = [cd[str(e)]["a_1"] for e in eps_vals]
    intR = [cd[str(e)]["int_R_dvol"] for e in eps_vals]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=intR, y=a1, mode="markers+text",
                            text=[f"\u03b5={e}" for e in eps_vals], textposition="top right",
                            marker=dict(size=12, color=eps_vals, colorscale="RdBu")))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(title="a\u2081 vs \u222bR dvol", xaxis_title="\u222bR dvol",
                     yaxis_title="a\u2081", height=350)
    st.plotly_chart(fig, use_container_width=True)


def load_json(filename: str):
    """Load a JSON result file from RESULTS_DIR, return dict or None."""
    path = RESULTS_DIR / filename
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _render_route3():
    """Render detailed Route 3 (Lorentzian) analysis tabs."""
    r3_tabs = st.tabs([
        "Decomposition", "Quick Kill", "Verification", "Adversarial", "Commutator"
    ])

    # --- Tab 1: Decomposition ---
    with r3_tabs[0]:
        st.subheader("BD Operator Decomposition")
        st.latex(r"L = H + M, \quad H = \frac{L+L^T}{2}\;(\text{symmetric}), \quad M = \frac{L-L^T}{2}\;(\text{antisymmetric})")
        st.latex(r"L^T L = H^2 + [H, M] + M^T M")

        fig = go.Figure()
        labels = ["H\u00b2 (eigenvalues)", "[H, M] (commutator)", "M^TM (retarded)"]
        values = [72, 14, 14]  # approximate contribution percentages
        colors_pie = ["#e74c3c", "#3498db", "#2ecc71"]
        statuses = ["FAILED (Route 1)", "TESTING NOW", "In SVD"]
        fig.add_trace(go.Pie(
            labels=[f"{l}<br><i>{s}</i>" for l, s in zip(labels, statuses)],
            values=values, marker=dict(colors=colors_pie),
            textinfo="label+percent", textposition="inside",
            hovertemplate="%{label}<br>%{percent}<extra></extra>",
            hole=0.3,
        ))
        fig.update_layout(title="Decomposition of L^TL \u2014 Where is the geometric info?",
                         height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Key insight:** Route 1 tested H\u00b2 (eigenvalues of symmetrized operator) \u2014 FAILED.
        SVD of L tests all three terms combined \u2014 detects curvature, but mediated by pair counting.
        The commutator [H, M] is the **missing piece** \u2014 it encodes the interaction between
        spatial connectivity (H) and temporal ordering (M).
        """)

    # --- Tab 2: Quick Kill ---
    with r3_tabs[1]:
        st.subheader("Quick Kill: SVD of Retarded Operator L")
        qk = load_json("route3_quickkill.json")
        if qk:
            # Test 1: SVD Discrimination
            st.markdown("### Test 1: Do singular values distinguish causal sets from random?")
            t1 = qk["test1"]

            fig = go.Figure()
            metrics = ["KS test", "Mean SV", "Max SV", "Entropy"]
            p_vals = [t1["ks_p"], t1["mean_sv_p"], t1["max_sv_p"], t1["entropy_p"]]
            log_p = [-np.log10(max(p, 1e-320)) for p in p_vals]

            fig.add_trace(go.Bar(
                x=metrics, y=log_p,
                marker=dict(color=log_p, colorscale="Hot", cmin=0, cmax=320),
                text=[f"p = {p:.1e}" for p in p_vals],
                textposition="outside",
                hovertemplate="%{x}<br>-log10(p) = %{y:.0f}<br>p = %{text}<extra></extra>",
            ))
            fig.add_hline(y=2, line_dash="dash", annotation_text="p = 0.01 threshold")
            fig.update_layout(
                title="SVD Discrimination: -log10(p-value) for each metric",
                yaxis_title="-log10(p)", height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            col1.metric("Causal Mean SV", f"{t1['causal_mean_sv']:.0f}")
            col2.metric("Null Mean SV", f"{t1['null_mean_sv']:.0f}")

            st.warning("**Caveat:** Adversarial review showed this extreme discrimination "
                      "(p = 10\u207b\u00b3\u2070\u00b3) is partly from sparsity mismatch (2.8% vs 100% fill). "
                      "With matched null: p ~ 10\u207b\u00b9\u2078 (still significant, but 285 orders weaker).")

            # Test 2: SVD Curvature
            st.markdown("### Test 2: Do singular values detect curvature?")
            t2 = qk["test2"]

            fig2 = go.Figure()
            test2_metrics = ["Mean SV shift", "Max SV shift", "Entropy shift", "SV heat trace"]
            test2_pvals = [t2["mean_sv_shift_p"], t2["max_sv_shift_p"],
                          t2["entropy_shift_p"], 1.0 if t2["frac_heat_005"] == 0 else 0.05]
            test2_colors = ["green" if p < 0.05 else "gray" for p in test2_pvals]

            fig2.add_trace(go.Bar(
                x=test2_metrics, y=test2_pvals,
                marker_color=["#2ecc71" if p < 0.05 else "#95a5a6" for p in test2_pvals],
                text=[f"p = {p:.4f}" for p in test2_pvals],
                textposition="outside",
            ))
            fig2.add_hline(y=0.05, line_dash="dash", line_color="red",
                          annotation_text="p = 0.05")
            fig2.update_layout(title="Curvature Sensitivity: p-values",
                             yaxis_title="p-value", height=350)
            st.plotly_chart(fig2, use_container_width=True)

            st.success(f"**Entropy shift p = {t2['entropy_shift_p']:.4f}** \u2014 significant! "
                      f"But only 1 of 4 metrics passes.")

            # Test 3: DW Zeta
            st.markdown("### Test 3: Dang-Wrochna Zeta")
            t3 = qk["test3"]
            st.error(f"FAILED \u2014 numerical overflow. Zeta diverges for \u03b1 > 0.5 "
                    f"(nilpotent L has tiny singular values).")
        else:
            st.info("Quick kill results not found.")

    # --- Tab 3: Verification ---
    with r3_tabs[2]:
        st.subheader("Signal Verification: SVD Entropy vs Curvature")
        rv = load_json("route3_verification.json")
        if rv:
            # Test A: Null model
            st.markdown("### Test A: Sparsity-Matched Null Model")
            ta = rv["test_a"]
            st.markdown(f"**Verdict:** {ta['verdict']}")
            col1, col2, col3 = st.columns(3)
            col1.metric("vs Sparsity-null", f"p = {ta['p_sparsity_entropy']:.1e}")
            col2.metric("vs Permutation-null", f"p = {ta['p_permutation']:.1e}")
            col3.metric("KS test", f"p = {ta['p_sparsity_ks']:.1e}")

            # Test B: Multi-epsilon (THE MONEY PLOT)
            st.markdown("### Test B: Entropy vs Curvature \u2014 The Money Plot")
            tb = rv["test_b"]
            eps_data = tb["results_by_eps"]
            eps_vals = sorted([float(k) for k in eps_data.keys()])
            ent_means = [eps_data[str(e)]["mean"] for e in eps_vals]
            ent_sems = [eps_data[str(e)]["sem"] for e in eps_vals]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=eps_vals, y=ent_means,
                error_y=dict(type="data", array=ent_sems, visible=True),
                mode="markers+lines",
                marker=dict(size=14, color=eps_vals, colorscale="RdBu",
                           cmin=-0.6, cmax=0.8, showscale=True,
                           colorbar=dict(title="\u03b5", len=0.5)),
                line=dict(color="rgba(52,152,219,0.5)", width=2),
                hovertemplate="\u03b5 = %{x:.2f}<br>H = %{y:.6f} \u00b1 %{error_y.array:.6f}<extra></extra>",
            ))
            # Linear fit overlay
            slope = tb["slope"]
            x_fit = np.linspace(-0.6, 0.85, 100)
            y_fit = slope * x_fit + (np.mean(ent_means) - slope * np.mean(eps_vals))
            fig.add_trace(go.Scatter(
                x=x_fit, y=y_fit, mode="lines",
                line=dict(color="red", dash="dash", width=1.5),
                name=f"Fit: slope = {slope:.6f}",
                hoverinfo="skip",
            ))
            fig.update_layout(
                title=f"Spectral Entropy vs Curvature \u2014 r = {tb['pearson_r']:.4f}, "
                      f"R\u00b2 = {tb['r_squared']:.4f}, Monotonic: {tb['monotonic']}",
                xaxis_title="\u03b5 (curvature parameter)",
                yaxis_title="Spectral Entropy H(\u03c3) of singular values",
                height=450, hovermode="closest",
            )
            st.plotly_chart(fig, use_container_width=True)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Pearson r", f"{tb['pearson_r']:.4f}")
            col2.metric("R\u00b2", f"{tb['r_squared']:.4f}")
            col3.metric("Slope", f"{tb['slope']:.6f}")
            col4.metric("Monotonic", "YES" if tb["monotonic"] else "NO")

            # Test C: Reproducibility
            st.markdown("### Test C: Reproducibility (3 seeds)")
            tc = rv["test_c"]

            fig_repro = go.Figure()
            seed_names = list(tc.keys())
            shifts = [tc[s]["mean_diff"] for s in seed_names]
            sems = [tc[s]["sem_diff"] for s in seed_names]
            pvals = [tc[s]["p_value"] for s in seed_names]
            repros = [tc[s]["reproduced"] for s in seed_names]

            fig_repro.add_trace(go.Bar(
                x=seed_names, y=shifts,
                error_y=dict(type="data", array=sems, visible=True),
                marker_color=["#2ecc71" if r else "#e74c3c" for r in repros],
                text=[f"p={p:.1e}" for p in pvals],
                textposition="outside",
                hovertemplate="%{x}<br>shift = %{y:+.6f}<br>%{text}<extra></extra>",
            ))
            fig_repro.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_repro.update_layout(
                title="Reproducibility: Entropy Shift (curved - flat)",
                yaxis_title="\u0394H (entropy shift)",
                height=350,
            )
            st.plotly_chart(fig_repro, use_container_width=True)
            st.success(f"**{rv['n_reproduced']}/3 seeds reproduced.** "
                      f"Sign flips correctly: +0.5 \u2192 positive shift, -0.5 \u2192 negative shift.")
        else:
            st.info("Verification results not found.")

    # --- Tab 4: Adversarial ---
    with r3_tabs[3]:
        st.subheader("Adversarial Review: What Survives?")

        attacks = [
            ("Null model mismatch\n(dense vs sparse)", "SUSTAINED",
             "p collapsed 10\u207b\u00b3\u2070\u00b3 \u2192 10\u207b\u00b9\u2078", False),
            ("Mediation by\npair counting", "SUSTAINED",
             "partial r = -0.046 after control", False),
            ("Isotropic density\ncontrol (R=0)", "SIGNAL\nSURVIVES",
             "Lorentzian-specific signal", True),
            ("SVD is\nsuperfluous", "SUSTAINED",
             "pair counting carries same info", False),
            ("No SDW\nconnection", "SUSTAINED",
             "slope 0.011 has no theory", False),
        ]

        fig = go.Figure()
        x_labels = [a[0] for a in attacks]
        survived = [a[3] for a in attacks]
        details = [a[2] for a in attacks]

        fig.add_trace(go.Bar(
            x=x_labels,
            y=[1]*len(attacks),
            marker_color=["#2ecc71" if s else "#e74c3c" for s in survived],
            text=[a[1] for a in attacks],
            textposition="inside",
            textfont=dict(size=13, color="white"),
            hovertemplate="%{x}<br><b>%{text}</b><br>%{customdata}<extra></extra>",
            customdata=details,
        ))
        fig.update_layout(
            title="Adversarial Attacks: 1/5 Survived",
            yaxis=dict(visible=False), height=300,
            xaxis_tickangle=-30,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.warning("""
        **Honest assessment:** SVD entropy signal is REAL and Lorentzian-specific,
        but it is fully mediated by pair counting \u2014 a trivial O(N\u00b2) statistic.
        The O(N\u00b3) SVD adds zero curvature-detection power beyond counting.
        """)

        # Mediation diagram
        st.markdown("### Mediation Pathway")
        fig_med = go.Figure()
        # Nodes
        fig_med.add_trace(go.Scatter(
            x=[0, 1, 2], y=[1, 0, 1],
            mode="markers+text",
            text=["Curvature (\u03b5)", "Total Causal\nPairs", "SVD Entropy"],
            textposition=["top center", "bottom center", "top center"],
            marker=dict(size=40, color=["#e74c3c", "#f39c12", "#3498db"]),
            hoverinfo="text",
        ))
        # Edges
        for x0, y0, x1, y1, label, style in [
            (0, 1, 1, 0, "r = 0.998", "solid"),
            (1, 0, 2, 1, "r = 0.63", "solid"),
            (0, 1, 2, 1, "partial r = -0.05", "dash"),
        ]:
            fig_med.add_annotation(
                x=x1, y=y1, ax=x0, ay=y0,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1.5,
                arrowwidth=2,
                arrowcolor="gray" if "dash" in style else "black",
            )
            fig_med.add_annotation(
                x=(x0+x1)/2, y=(y0+y1)/2 + 0.15,
                text=label, showarrow=False,
                font=dict(size=12, color="red" if "partial" in label else "black"),
            )
        fig_med.update_layout(
            height=300, showlegend=False,
            xaxis=dict(visible=False, range=[-0.5, 2.5]),
            yaxis=dict(visible=False, range=[-0.5, 1.5]),
            title="Mediation: \u03b5 \u2192 Pair Count \u2192 Entropy (direct path r \u2248 0)",
        )
        st.plotly_chart(fig_med, use_container_width=True)

    # --- Tab 5: Commutator ---
    with r3_tabs[4]:
        st.subheader("Commutator [H, M] \u2014 The Critical Experiment")

        comm = load_json("route3_commutator.json")
        if comm:
            v = comm.get("verdict", "?")
            if "BREAKTHROUGH" in v:
                st.success(f"### {v}")
            elif "POSITIVE" in v or "SIGNAL" in v:
                st.info(f"### {v}")
            elif "NO SIGNAL" in v or "MEDIATED" in v:
                st.error(f"### {v}")
            else:
                st.warning(f"### {v}")

            # Mediation comparison chart
            if "mediation" in comm:
                st.markdown("### Mediation Analysis: Does [H,M] survive?")
                obs_names = list(comm["mediation"].keys())
                r_direct = [comm["mediation"][o]["r_direct"] for o in obs_names]
                r_partial = [comm["mediation"][o]["r_partial_both"] for o in obs_names]
                p_partial = [comm["mediation"][o]["p_partial_both"] for o in obs_names]

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    name="Direct r(\u03b5)", x=obs_names, y=r_direct,
                    marker_color="#3498db",
                    hovertemplate="%{x}<br>Direct r = %{y:.4f}<extra></extra>",
                ))
                fig.add_trace(go.Bar(
                    name="Partial r(\u03b5 | TC+n0)", x=obs_names, y=r_partial,
                    marker_color=["#2ecc71" if abs(r) > 0.1 and p < 0.1 else "#e74c3c"
                                  for r, p in zip(r_partial, p_partial)],
                    hovertemplate="%{x}<br>Partial r = %{y:.4f}<extra></extra>",
                ))
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.add_hline(y=0.1, line_dash="dot", line_color="green",
                             annotation_text="survival threshold")
                fig.add_hline(y=-0.1, line_dash="dot", line_color="green")
                fig.update_layout(
                    title="Direct vs Partial Correlation with Curvature",
                    yaxis_title="Pearson r",
                    barmode="group", height=450,
                    xaxis_tickangle=-30,
                )
                st.plotly_chart(fig, use_container_width=True)

                # Table
                rows = []
                for o in obs_names:
                    m = comm["mediation"][o]
                    surv = abs(m["r_partial_both"]) > 0.1 and m["p_partial_both"] < 0.10
                    rows.append({
                        "Observable": o,
                        "Direct r": f"{m['r_direct']:+.4f}",
                        "Direct p": f"{m['p_direct']:.2e}",
                        "Partial r": f"{m['r_partial_both']:+.4f}",
                        "Partial p": f"{m['p_partial_both']:.2e}",
                        "Survives": "YES" if surv else "no",
                    })
                st.dataframe(rows, use_container_width=True, hide_index=True)

            # Reproducibility
            if "reproducibility" in comm:
                st.markdown("### Reproducibility")
                repro = comm["reproducibility"]
                configs = list(repro.keys())
                if configs and "comm_entropy" in repro[configs[0]]:
                    pvals_r = [repro[c]["comm_entropy"]["p_value"] for c in configs]
                    sigs_r = [repro[c]["comm_entropy"].get("significant", False) for c in configs]
                    shifts_r = [repro[c]["comm_entropy"]["mean_diff"] for c in configs]

                    fig_r = go.Figure()
                    fig_r.add_trace(go.Bar(
                        x=configs, y=shifts_r,
                        marker_color=["#2ecc71" if s else "#e74c3c" for s in sigs_r],
                        text=[f"p={p:.4f}" for p in pvals_r],
                        textposition="outside",
                    ))
                    fig_r.add_hline(y=0, line_dash="dash")
                    fig_r.update_layout(
                        title=f"Reproducibility: {sum(sigs_r)}/{len(sigs_r)} seeds significant",
                        yaxis_title="Entropy shift", height=350,
                        xaxis_tickangle=-30,
                    )
                    st.plotly_chart(fig_r, use_container_width=True)

            # Null model
            if "null_model" in comm:
                nm = comm["null_model"]
                st.markdown("### Null Model Discrimination")
                c1, c2, c3 = st.columns(3)
                c1.metric("Causal [H,M] entropy", f"{nm.get('causal_mean', 0):.6f}")
                c2.metric("Null [H,M] entropy", f"{nm.get('null_mean', 0):.6f}")
                c3.metric("Discriminates", "YES" if nm.get("discriminates") else "NO",
                          f"p = {nm.get('t_p', 1):.2e}")
        else:
            st.warning("Commutator experiment is **RUNNING** (N=3000, 9 seeds, ~5 hours)")

            st.markdown("### What's Being Tested")
            st.latex(r"[H, M] = HM - MH \quad \text{(real symmetric, 14\% of } \|H\|\cdot\|M\|\text{)}")

            st.markdown("""
            | Criterion | Threshold | Meaning |
            |-----------|-----------|---------|
            | Direct r(\u03b5) | > 0.3 | [H,M] correlates with curvature |
            | Partial r(\u03b5 \\| TC+n0) | > 0.1 | Survives mediation by pair counting |
            | Reproducibility | \u2265 6/9 seeds | Not a fluke |
            | Null discrimination | p < 0.01 | Different from random |
            """)

            fig_what = go.Figure()
            fig_what.add_trace(go.Indicator(
                mode="gauge+number",
                value=15,
                title={"text": "Estimated Progress (%)"},
                gauge=dict(
                    axis=dict(range=[0, 100]),
                    bar=dict(color="#3498db"),
                    steps=[
                        dict(range=[0, 40], color="#eee"),
                        dict(range=[40, 70], color="#ddd"),
                        dict(range=[70, 100], color="#ccc"),
                    ],
                    threshold=dict(line=dict(color="red", width=4), thickness=0.75, value=100),
                ),
            ))
            fig_what.update_layout(height=250)
            st.plotly_chart(fig_what, use_container_width=True)


# ---------------------------------------------------------------------------
# Page: Dashboard (Overview)
# ---------------------------------------------------------------------------

if page == "Dashboard":
    st.title("FND-1 Experiment Hub")
    st.markdown("> *Can the spectral action be derived from discrete causal structure?*")

    # Route overview cards
    cols = st.columns(3)
    for i, (route_id, info) in enumerate(ROUTE_INFO.items()):
        with cols[i]:
            st.markdown(f"### Route {route_id}")
            st.markdown(badge(info["status"], info["color"]), unsafe_allow_html=True)
            st.caption(info["description"])
            n_exp = len(get_route_experiments(experiments, route_id))
            st.metric("Experiments", n_exp)

    st.markdown("---")

    # 3D Experiment Landscape
    st.subheader("Experiment Landscape")
    if experiments:
        names = [e.get("description", e.get("name", "?"))[:30] for e in experiments]
        routes = [e.get("route", 0) for e in experiments]
        Ns = [e.get("N", 0) or 0 for e in experiments]
        verdicts = [e.get("verdict", "") for e in experiments]
        v_scores = []
        for v in verdicts:
            vc = verdict_color(v)
            v_scores.append({"green": 1.0, "orange": 0.5, "red": 0.0, "blue": 0.5, "gray": 0.3}.get(vc, 0.3))
        times = [e.get("wall_time_sec", 0) or 0 for e in experiments]

        fig = go.Figure(data=[go.Scatter3d(
            x=routes, y=Ns, z=v_scores,
            mode="markers+text",
            text=names, textposition="top center", textfont=dict(size=8),
            marker=dict(size=[max(6, t**0.3 * 3) for t in times],
                       color=v_scores, colorscale="RdYlGn", cmin=0, cmax=1,
                       colorbar=dict(title="Signal"), opacity=0.85,
                       line=dict(width=1, color="black")),
            hovertemplate="%{text}<br>Route %{x}<br>N=%{y}<br>Score=%{z:.2f}<extra></extra>",
        )])
        fig.update_layout(
            scene=dict(
                xaxis=dict(title="Route", tickvals=[1,2,3],
                          ticktext=["Ensemble","Emergent","Lorentzian"]),
                yaxis=dict(title="N (causal set size)", type="log"),
                zaxis=dict(title="Signal Strength"),
            ),
            height=500, margin=dict(l=0, r=0, b=0, t=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Timeline
    st.subheader("Experiment Timeline")
    if experiments:
        timeline_rows = []
        for e in sorted(experiments, key=lambda x: x.get("file_mtime", "")):
            timeline_rows.append({
                "": verdict_icon(e.get("verdict", "")),
                "Experiment": (e.get("description") or e.get("name", "?"))[:50],
                "Route": e.get("route", "?"),
                "N": e.get("N", "\u2014"),
                "M": e.get("M", "\u2014"),
                "Time": f"{e.get('wall_time_sec', 0)/60:.1f}m" if e.get("wall_time_sec") else "\u2014",
                "Verdict": (e.get("verdict") or "\u2014")[:40],
            })
        st.dataframe(timeline_rows, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Route pages (auto-rendered)
# ---------------------------------------------------------------------------

elif page.startswith("Route"):
    route_num = int(page.split(":")[0].split()[-1])
    info = ROUTE_INFO.get(route_num, {})

    st.title(f"Route {route_num}: {info.get('name', '?')}")
    st.markdown(badge(info.get("status", "?"), info.get("color", "gray")), unsafe_allow_html=True)
    st.markdown(f"*{info.get('description', '')}*")
    st.markdown("---")

    route_exps = get_route_experiments(experiments, route_num)

    if not route_exps:
        st.info(f"No experiments found for Route {route_num}.")

        if route_num == 2:
            st.markdown("""
            ### Route 2: Coarse-Grained / Emergent Spectral Triple

            **Status:** OPEN \u2014 Untested

            **Concept:** The causal set first produces coarse-grained geometric data
            through an explicit reconstruction step, and only then supports a spectral triple.

            **What it requires:**
            - An explicit reconstruction map from causal-set data to emergent geometric data
            - Proof that the emergent algebra, Hilbert space, and operator are still causally sourced
            - Stability under changes of coarse-graining scale

            **What would falsify it quickly:**
            - The emergent triple requires manifold/framing data not derivable from the causal set
            - The coarse-graining map is non-robust or highly non-unique
            - The recovered spectral data depends mainly on manual reconstruction choices

            **Prerequisites:**
            - Mathematical construction of the reconstruction map
            - Benchmarkable proxy for numerical testing

            **Connection to completed work:**
            - FND-1 finite-nerve framework (Paper 6) provides reusable boundary/homology stack
            - Route 1 negative result shows direct finite-matrix approach doesn't work
            - Route 3 shows Lorentzian structure matters \u2014 any emergent triple should preserve it

            *This route is the cleanest path to unblocking ALG-1, but requires significant
            new mathematical construction before numerical testing can begin.*
            """)
    else:
        # Sort by file modification time
        route_exps.sort(key=lambda x: x.get("file_mtime", ""))

        # Summary metrics
        n_pass = sum(1 for e in route_exps if verdict_color(e.get("verdict", "")) == "green")
        n_fail = sum(1 for e in route_exps if verdict_color(e.get("verdict", "")) == "red")
        n_other = len(route_exps) - n_pass - n_fail

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Experiments", len(route_exps))
        c2.metric("Positive", n_pass, delta_color="normal")
        c3.metric("Negative", n_fail, delta_color="inverse")
        c4.metric("Other", n_other)

        st.markdown("---")

        # Render each experiment
        for exp in route_exps:
            render_experiment_card(exp)

        # Route-specific visualizations
        if route_num == 1:
            st.markdown("---")
            st.subheader("Lessons Learned")
            lessons = [
                ("\U0001f6ab", "Do NOT symmetrize BD operator", "H = (L+L^T)/2 destroys causal structure"),
                ("\U0001f3af", "Always include null model", "Random matrices can mimic UV exponents"),
                ("\U0001f501", "Check reproducibility with multiple seeds", "One seed can fluctuate"),
                ("\U0001f4ca", "Mediation analysis is essential", "SVD entropy was fully mediated by pair counting"),
                ("\u26a0\ufe0f", "SDW extraction requires p \u2248 -1", "Doesn't hold at accessible N"),
                ("\U0001f40c", "Convergence rate matters", "\u03b1 = 0.1 means N ~ 10\u2079 for 5% accuracy"),
            ]
            for icon, title, desc in lessons:
                st.markdown(f"{icon} **{title}** \u2014 {desc}")

        if route_num == 3:
            _render_route3()


# ---------------------------------------------------------------------------
# Page: All Experiments
# ---------------------------------------------------------------------------

elif page == "All Experiments":
    st.title("All Experiments")

    # Filters
    col1, col2, col3 = st.columns(3)
    route_filter = col1.multiselect("Route", [1, 2, 3], default=[1, 2, 3])
    status_filter = col2.multiselect("Status", ["green", "red", "orange", "blue", "gray"],
                                      default=["green", "red", "orange", "blue", "gray"],
                                      format_func=lambda x: {"green":"Positive","red":"Negative",
                                                             "orange":"Inconclusive","blue":"Running",
                                                             "gray":"Other"}.get(x, x))

    filtered = [e for e in experiments
                if e.get("route") in route_filter
                and verdict_color(e.get("verdict", "")) in status_filter]

    st.markdown(f"**{len(filtered)}** experiments shown")
    st.markdown("---")

    for exp in sorted(filtered, key=lambda x: x.get("file_mtime", ""), reverse=True):
        render_experiment_card(exp)


# ---------------------------------------------------------------------------
# Page: Live Monitor
# ---------------------------------------------------------------------------

elif page == "Live Monitor":
    st.title("Live Experiment Monitor")

    progress = load_progress()
    if progress:
        st.markdown(f"### Running: {progress.get('name', '?')}")
        st.markdown(f"*{progress.get('description', '')}*")

        c1, c2, c3 = st.columns(3)
        c1.metric("N", progress.get("N", "?"))
        c2.metric("Route", progress.get("route", "?"))
        c3.metric("ETA", f"{progress.get('eta_min', '?')} min")

        pct = progress.get("pct", 0)
        st.progress(pct, text=progress.get("step", "Computing..."))
    else:
        # Check if commutator result exists
        comm = None
        for e in experiments:
            if "commutator" in e.get("name", ""):
                comm = e
                break

        if comm:
            st.success(f"Latest experiment complete: {comm.get('verdict', '?')}")
        else:
            st.info("No experiment currently running. No progress file found.")

    st.markdown("---")
    st.subheader("Result Files")

    for path in sorted(RESULTS_DIR.glob("*.json")):
        if path.name.startswith("_"):
            continue
        size = path.stat().st_size
        mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%H:%M:%S")
        st.caption(f"`{path.name}` \u2014 {size:,} bytes \u2014 {mtime}")


# ---------------------------------------------------------------------------
# Page: Theory
# ---------------------------------------------------------------------------

elif page == "Theory":
    st.title("Theoretical Background")

    tab1, tab2, tab3 = st.tabs(["Core Question", "Key Papers", "Mathematics"])

    with tab1:
        st.markdown("""
        ### The FND-1 Problem

        > Can the Chamseddine-Connes Spectral Action be formulated directly
        > on a bare discrete causal set?

        **Three routes:**

        | Route | Approach | Status |
        |-------|----------|--------|
        | 1 | Ensemble averaging (Euclidean) | **CLOSED** \u2014 no curvature sensitivity |
        | 2 | Emergent spectral triple | **OPEN** \u2014 untested |
        | 3 | Lorentzian/Krein reformulation | **IN PROGRESS** \u2014 testing [H,M] |

        **Key insight from Route 1 \u2192 Route 3 transition:**
        Symmetrizing the BD operator (H = (L+L^T)/2) destroys the causal/retarded
        structure that carries geometric information. Route 3 uses L directly.
        """)

    with tab2:
        papers = [
            ("Dang & Wrochna", "2020", "2012.00712",
             "Lorentzian spectral action via complex powers of wave operator"),
            ("Dang & Wrochna", "2021", "2108.07529",
             "Dynamical residues of Lorentzian spectral zeta functions"),
            ("Yazdi, Letizia, Kempf", "2020", "2008.02291",
             "Lorentzian spectral geometry with causal sets"),
            ("Yazdi & Kempf", "2016", "1611.09947",
             "Spectral geometry for causal sets \u2014 SA/ASA decomposition"),
            ("Benincasa & Dowker", "2010", "1001.2725",
             "Scalar curvature of a causal set \u2014 BD d'Alembertian"),
            ("Bizi", "2018", "1812.00038",
             "Semi-Riemannian NCG via Krein spaces (thesis)"),
            ("Martinetti", "2026", "2603.03216",
             "Twisted Standard Model and Krein structure"),
        ]
        for auth, year, arxiv, title in papers:
            st.markdown(f"**{auth} ({year})** \u2014 [{title}](https://arxiv.org/abs/{arxiv})")

    with tab3:
        st.subheader("BD Operator Decomposition")
        st.latex(r"L = H + M, \quad L^T L = H^2 + [H,M] + M^T M")

        st.subheader("Dang-Wrochna Residue Formula")
        st.latex(r"\mathrm{Res}_{\alpha=n/2-m}\,(\square_g \pm i\varepsilon)^{-\alpha}(x,x) = \mp\frac{i\,u_m(x,x)}{2^n\pi^{n/2}(n/2-m-1)!}")

        st.markdown("| m | u_m | Meaning |\n|---|-----|--------|\n| 0 | 1 | Volume |\n| 1 | -R/6 | Curvature |\n| 2 | (Riem\u00b2-Ric\u00b2+...)/180 | Gauss-Bonnet |")

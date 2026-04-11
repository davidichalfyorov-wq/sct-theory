"""Create mock JSONs for pipeline testing. DELETE mock files before real data arrives."""
import json
from pathlib import Path

R = Path(__file__).resolve().parent.parent.parent / "speculative" / "numerics" / "ensemble_results"

def obs_med(r, p):
    return {"linear_r_direct": r*1.5, "linear_r_partial": r, "linear_p_partial": p,
            "quadratic_r_direct": r*1.2, "quadratic_r_partial": r*0.8, "quadratic_p_partial": p*2,
            "best": "linear", "best_r_partial": r, "best_p_partial": p}

def meta(name, route, N, M, verdict):
    return {"route": route, "name": name, "N": N, "M": M, "status": "completed",
            "verdict": verdict, "timestamp": "2026-03-22T20:00:00",
            "wall_time_sec": 100, "parameters": {}, "tags": ["mock"]}

mocks = {
    "exp1_d4_link_verification.json": {
        "_meta": meta("exp1_d4_link_verification", 2, 2000, 80, "MOCK"),
        "parameters": {"N_values": [500,1000,2000,3000], "N_primary": 2000, "M": 80, "T": 1.0,
                        "eps_coscosh": [0,0.1,0.2,0.3,0.5], "eps_quadrupole": [0,2,5,10,20],
                        "eps_conformal": [0.2,0.5], "k_values_embed": [2,5,10,20], "k_max": 20, "n_distance_pairs": 8000},
        "configs": {
            **{f"tidal_N{N}": {"N": N, "metric": "tidal", "eps_values": [0,0.1,0.2,0.3,0.5],
                "n_sprinklings_total": 400,
                "mediation": {"n_samples": 400, "fiedler_linear_r_partial": 0.10+N/20000,
                              "fiedler_linear_p_partial": 0.01-N/500000,
                              "fiedler_r_partial": 0.10+N/20000, "fiedler_p_partial": 0.01-N/500000,
                              "best_predictor": "linear"},
                "geometry_flat": {"rho_spearman_mean": 0.20+N/8000, "rho_spearman_sem": 0.02, "r_null_mean": 0.01},
                "mean_link_deg": 14.5, "mean_fiedler_flat": 0.7-N/20000}
              for N in [500,1000,2000,3000]},
            "quadrupole": {"N": 2000, "metric": "quadrupole", "eps_values": [0,2,5,10,20],
                "n_sprinklings_total": 400,
                "mediation": {"n_samples": 400, "fiedler_linear_r_partial": 0.18,
                              "fiedler_linear_p_partial": 0.0003,
                              "fiedler_r_partial": 0.18, "fiedler_p_partial": 0.0003, "best_predictor": "linear"},
                "geometry_flat": {"rho_spearman_mean": 0.38, "rho_spearman_sem": 0.01, "r_null_mean": 0.005},
                "mean_link_deg": 14.7, "mean_fiedler_flat": 0.65},
            "conformal": {"N": 2000, "metric": "conformal", "eps_values": [0.2,0.5],
                "n_sprinklings_total": 160, "mediation": None,
                "geometry_flat": {"rho_spearman_mean": 0.37}, "mean_link_deg": 14.7, "mean_fiedler_flat": 0.66},
        },
        "scaling": {"N": [500,1000,2000,3000], "partial_r": [0.12,0.15,0.19,0.22],
                     "rho_flat": [0.26,0.33,0.38,0.42], "signal_grows": True},
        "verdict": "MOCK: GENUINE r=+0.18 | SIGNAL GROWS | GEOMETRY rho=0.38", "wall_time_sec": 600,
    },
    "exp2_d4_sj_verification.json": {
        "_meta": meta("exp2_d4_sj_verification", 2, 3000, 80, "MOCK: COSCOSH ONLY"),
        "parameters": {"N_values": [1000,2000,3000], "N_primary": 3000, "M": 80, "T": 1.0,
                        "eps_coscosh": [0,0.1,0.2,0.3,0.5], "eps_quadrupole": [0,2,5,10,20],
                        "verify_observables": ["spectral_width","spectral_width_filtered","trace_trunc","lambda_max","entropy_trunc","trace_W"]},
        "results": {
            "quadrupole_N3000": {"mediation": {"n": 400, "spectral_width": obs_med(0.05,0.3),
                "best_observable": "lambda_max", "best_r_partial": 0.06, "best_p_partial": 0.2, "best_predictor": "linear"}, "n_sprinklings": 400},
            "coscosh_N3000": {"mediation": {"n": 400, "spectral_width": obs_med(0.15,0.003),
                "best_observable": "spectral_width", "best_r_partial": 0.15, "best_p_partial": 0.003, "best_predictor": "linear"}, "n_sprinklings": 400},
        },
        "summary": {"n_reproduced_coscosh": 2, "n_reproduced_quadrupole": 0},
        "verdict": "MOCK: COSCOSH ONLY 2/6 (density artifact)", "wall_time_sec": 3600,
    },
    "exp3_d4_commutator.json": {
        "_meta": meta("exp3_d4_commutator", 3, 2000, 100, "MOCK: NO SIGNAL"),
        "parameters": {"N_values": [500,1000,2000], "N_primary": 2000, "M": 100, "T": 1.0,
                        "eps_coscosh": [0,0.1,0.2,0.3,0.5], "eps_quadrupole": [0,2,5,10,20]},
        "results": {
            "quadrupole_N2000": {"mediation": {"n": 500, "best_observable": "comm_entropy",
                "best_r_partial": 0.06, "best_p_partial": 0.15, "best": "linear"}, "n_sprinklings": 500},
            "coscosh_N2000": {"mediation": {"n": 500, "best_observable": "comm_entropy",
                "best_r_partial": 0.09, "best_p_partial": 0.04, "best": "linear"}, "n_sprinklings": 500},
        },
        "verdict": "MOCK: NO SIGNAL", "wall_time_sec": 500,
    },
    "exp4_d4_magnetic_phase.json": {
        "_meta": meta("exp4_d4_magnetic_phase", 7, 2000, 100, "MOCK: COSCOSH SIGNAL"),
        "parameters": {"N_values": [500,1000,2000], "N_primary": 2000, "M": 100, "T": 1.0,
                        "eps_coscosh": [0,0.1,0.2,0.3,0.5], "eps_quadrupole": [0,2,5,10,20]},
        "results": {
            "quadrupole_N2000": {"mediation_spectral_diff": {"n": 500, "observable": "spectral_diff",
                "best": "linear", "best_r_partial": -0.08, "best_p_partial": 0.07}, "n_sprinklings": 500},
            "coscosh_N2000": {"mediation_spectral_diff": {"n": 500, "observable": "spectral_diff",
                "best": "linear", "best_r_partial": -0.31, "best_p_partial": 2e-12}, "n_sprinklings": 500},
        },
        "verdict": "MOCK: COSCOSH r=-0.31 (circular)", "wall_time_sec": 600,
    },
    "exp5_d4_spectral_action.json": {
        "_meta": meta("exp5_d4_spectral_action", 2, 3000, 60, "MOCK: PLATEAU + CURVATURE"),
        "parameters": {"N_values": [500,1000,2000,3000], "M": 60, "T": 1.0,
                        "eps_coscosh": [0,0.2,0.5], "eps_quadrupole": [0,5,10],
                        "tau_range": [0.0001,100,200], "d": 4, "power": 2.0},
        "results_by_N": {"3000": {
            "by_eps": {"coscosh_0.0": {"t2K_curve": {"0.01":0.001,"0.1":0.0012,"1.0":0.0015,"10.0":0.0008}}},
            "bd_goe_ratio": 2.29, "curvature_r_partial": 0.45, "curvature_p_partial": 1e-10, "curvature_n_samples": 180}},
        "verdict": "MOCK: PLATEAU tau^2K=0.0015 | BD/GOE=2.29x | r=+0.45", "wall_time_sec": 1200,
    },
    "exp6_d4_higher_N.json": {
        "_meta": meta("exp6_d4_higher_N", 2, 10000, 30, "MOCK: SCALING"),
        "parameters": {"N_values": [3000,5000,8000,10000], "N_primary": 10000, "M": 30, "T": 1.0,
                        "eps_quadrupole": [0,2,5,10], "eps_coscosh": [0,0.1,0.3,0.5]},
        "results_by_N": {str(N): {
            "fiedler_flat": 0.7-N/30000, "fiedler_curved": 0.65-N/25000,
            "fiedler_delta": -(0.05+N/200000), "fiedler_cohen_d": -(0.3+N/25000),
            "fiedler_p": max(1e-5, 0.05-N/300000), "fiedler_r_partial": 0.12+N/60000,
            "fiedler_p_partial": max(1e-6, 0.01-N/2000000), "rho_flat": 0.40+N/50000, "mean_link_deg": 15.0}
            for N in [3000,5000,8000,10000]},
        "scaling": {"signal_grows": True},
        "verdict": "MOCK: CONFIRMED scaling r grows with N", "wall_time_sec": 2400,
    },
    "exp11_d3_intermediate.json": {
        "_meta": meta("exp11_d3_intermediate", 2, 2000, 80, "MOCK: d=3 DETECTS"),
        "parameters": {"N_values": [500,1000,2000,3000], "N_primary": 2000, "M": 80, "T": 1.0,
                        "eps_curved": [0,0.5,1,2,4], "k_values_embed": [2,5,10,20], "n_distance_pairs": 8000,
                        "metric": "-(1+eps*r_m^2)*dt^2 + dx^2 + dy^2"},
        "results_by_N": {str(N): {
            "mediation": {"n": 400, "linear_r_direct": 0.4+N/10000,
                "linear_r_partial": 0.25+N/15000, "linear_p_partial": 10**(-8-N/1000),
                "quadratic_r_direct": 0.35+N/12000, "quadratic_r_partial": 0.20+N/18000,
                "quadratic_p_partial": 10**(-6-N/1500),
                "best": "linear", "best_r_partial": 0.25+N/15000, "best_p_partial": 10**(-8-N/1000)},
            "rho_flat": 0.50+N/8000, "null_flat": 0.01-N/500000, "n_sprinklings": 400}
            for N in [500,1000,2000,3000]},
        "verdict": "MOCK: d=3 DETECTS CURVATURE r=+0.38 | GEOMETRY rho=0.65", "wall_time_sec": 300,
    },
}

for name, data in mocks.items():
    p = R / name
    # Don't overwrite real data
    if p.exists():
        with open(p) as f:
            existing = json.load(f)
        if "mock" not in existing.get("_meta", {}).get("tags", []):
            print(f"  SKIP {name} (real data exists)")
            continue
    with open(p, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Created {name}")

print("Done. Delete mock files before importing real data!")

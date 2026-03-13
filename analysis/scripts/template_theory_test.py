"""
Template: Test SCT Prediction Against Observational Data
=========================================================
Purpose: [DESCRIBE]
Input:   [DATASET PATH]
Output:  [RESULTS PATH]
"""

import json
import numpy as np
from scipy import constants, optimize, stats
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed"
RESULTS_PATH = PROJECT_ROOT / "analysis" / "results"
FIGURES_PATH = PROJECT_ROOT / "analysis" / "figures"

UNIT_CONVENTION = "natural"  # "SI" or "natural" (c = hbar = 1)
SEED = 42
np.random.seed(SEED)

# Physical constants (CODATA 2022)
G_N = constants.G                    # Newton's constant [m^3 kg^-1 s^-2]
c = constants.c                      # Speed of light [m/s]
hbar = constants.hbar                # Reduced Planck constant [J s]
l_P = np.sqrt(hbar * G_N / c**3)    # Planck length [m]
E_P = np.sqrt(hbar * c**5 / G_N)    # Planck energy [J]
M_P = np.sqrt(hbar * c / G_N)       # Planck mass [kg]

# ============================================================
# DATA LOADING
# ============================================================
# data = np.loadtxt(DATA_PATH / "dataset.csv", delimiter=",", skiprows=1)
# x_obs, y_obs, y_err = data[:, 0], data[:, 1], data[:, 2]

# ============================================================
# MODEL DEFINITIONS
# ============================================================
def model_gr(x, *params):
    """Standard GR/QFT prediction.
    Args: x — independent variable, *params — GR model parameters.
    Returns: predicted y values. Units: [specify].
    """
    raise NotImplementedError("Define GR model")


def model_sct(x, *params):
    """SCT Theory prediction (GR + quantum gravity corrections).
    First N params are GR parameters, remaining are SCT corrections.
    Args: x — independent variable, *params — all parameters (GR + SCT).
    Returns: predicted y values. Units: [specify].
    """
    raise NotImplementedError("Define SCT model")


N_PARAMS_GR = 0   # number of GR-only parameters
N_PARAMS_SCT = 0  # number of additional SCT parameters

# ============================================================
# FREQUENTIST FITTING
# ============================================================
def fit_model(model, x, y, yerr, p0=None):
    """Fit a model via chi-squared minimization."""
    popt, pcov = optimize.curve_fit(model, x, y, sigma=yerr, p0=p0,
                                     absolute_sigma=True, maxfev=10000)
    perr = np.sqrt(np.diag(pcov))
    y_pred = model(x, *popt)
    chi2 = np.sum(((y - y_pred) / yerr)**2)
    k = len(popt)
    dof = len(x) - k
    n = len(x)
    aic = chi2 + 2 * k + (2 * k * (k + 1)) / (n - k - 1)  # AICc
    bic = chi2 + k * np.log(n)
    return {
        "params": popt, "errors": perr, "cov": pcov,
        "chi2": chi2, "dof": dof, "chi2_dof": chi2 / dof,
        "p_value": 1 - stats.chi2.cdf(chi2, dof),
        "AICc": aic, "BIC": bic,
    }


def compare_models(fit_gr, fit_sct):
    """Compare GR and SCT fits."""
    delta_chi2 = fit_gr["chi2"] - fit_sct["chi2"]
    delta_dof = fit_gr["dof"] - fit_sct["dof"]
    if delta_dof > 0:
        f_stat = (delta_chi2 / delta_dof) / (fit_sct["chi2"] / fit_sct["dof"])
        f_pvalue = 1 - stats.f.cdf(f_stat, delta_dof, fit_sct["dof"])
    else:
        f_stat, f_pvalue = None, None

    return {
        "delta_chi2": delta_chi2,
        "delta_AICc": fit_gr["AICc"] - fit_sct["AICc"],
        "delta_BIC": fit_gr["BIC"] - fit_sct["BIC"],
        "F_statistic": f_stat,
        "F_pvalue": f_pvalue,
        "sigma_preference": np.sqrt(delta_chi2) if delta_chi2 > 0 else -np.sqrt(-delta_chi2),
    }


# ============================================================
# MCMC / BAYESIAN ANALYSIS (optional, requires emcee)
# ============================================================
def run_mcmc(model, x, y, yerr, p0, n_params, n_walkers=32, n_steps=5000,
             prior_fn=None):
    """Run MCMC sampling with emcee.
    Args:
        prior_fn: callable(params) -> log-prior. Flat if None.
    Returns: dict with chains, best-fit, and credible intervals.
    """
    try:
        import emcee
    except ImportError:
        raise ImportError("pip install emcee corner")

    def log_likelihood(params):
        y_model = model(x, *params)
        return -0.5 * np.sum(((y - y_model) / yerr)**2)

    def log_prior(params):
        if prior_fn is not None:
            return prior_fn(params)
        return 0.0  # flat prior

    def log_posterior(params):
        lp = log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(params)

    ndim = n_params
    pos = p0 + 1e-4 * np.random.randn(n_walkers, ndim)
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_posterior)
    sampler.run_mcmc(pos, n_steps, progress=True)

    # Discard burn-in (first 20%) and thin
    flat_samples = sampler.get_chain(discard=n_steps // 5, thin=15, flat=True)
    best = np.median(flat_samples, axis=0)
    lo = np.percentile(flat_samples, 16, axis=0)
    hi = np.percentile(flat_samples, 84, axis=0)

    return {
        "samples": flat_samples,
        "best": best,
        "errors_minus": best - lo,
        "errors_plus": hi - best,
        "chain": sampler.get_chain(),
        "log_prob": sampler.get_log_prob(),
    }


# ============================================================
# VISUALIZATION
# ============================================================
def plot_comparison(x_obs, y_obs, y_err, fit_gr, fit_sct, save_path=None):
    """Publication-quality comparison plot."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                     gridspec_kw={"height_ratios": [3, 1]},
                                     sharex=True)
    x_fine = np.linspace(x_obs.min(), x_obs.max(), 500)

    ax1.errorbar(x_obs, y_obs, yerr=y_err, fmt="k.", ms=3, alpha=0.6, label="Data")
    ax1.plot(x_fine, model_gr(x_fine, *fit_gr["params"]), "b-",
             label=f'GR ($\\chi^2/\\mathrm{{dof}}={fit_gr["chi2_dof"]:.2f}$)')
    ax1.plot(x_fine, model_sct(x_fine, *fit_sct["params"]), "r--",
             label=f'SCT ($\\chi^2/\\mathrm{{dof}}={fit_sct["chi2_dof"]:.2f}$)')
    ax1.legend(fontsize=12)
    ax1.set_ylabel("Observable", fontsize=14)

    res_gr = (y_obs - model_gr(x_obs, *fit_gr["params"])) / y_err
    res_sct = (y_obs - model_sct(x_obs, *fit_sct["params"])) / y_err
    ax2.scatter(x_obs, res_gr, c="blue", s=10, alpha=0.5, label="GR")
    ax2.scatter(x_obs, res_sct, c="red", s=10, alpha=0.5, label="SCT")
    ax2.axhline(0, color="gray", ls="--")
    ax2.set_ylabel(r"Residuals ($\sigma$)", fontsize=14)
    ax2.set_xlabel("x", fontsize=14)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# ============================================================
# RESULT SERIALIZATION
# ============================================================
def save_results(results, comparison, filename):
    """Save results to JSON in analysis/results/."""
    out = {k: v.tolist() if isinstance(v, np.ndarray) else v
           for k, v in {**results, **comparison}.items()
           if k != "cov"}
    out["seed"] = SEED
    out["unit_convention"] = UNIT_CONVENTION
    path = RESULTS_PATH / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to {path}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    pass
    # fit_gr = fit_model(model_gr, x_obs, y_obs, y_err)
    # fit_sct = fit_model(model_sct, x_obs, y_obs, y_err)
    # comp = compare_models(fit_gr, fit_sct)
    # print(f"GR:  chi2/dof = {fit_gr['chi2_dof']:.3f}")
    # print(f"SCT: chi2/dof = {fit_sct['chi2_dof']:.3f}")
    # print(f"Delta AICc = {comp['delta_AICc']:.2f}  (>0 favors SCT)")
    # print(f"Delta BIC  = {comp['delta_BIC']:.2f}  (>0 favors SCT)")
    # plot_comparison(x_obs, y_obs, y_err, fit_gr, fit_sct,
    #                 save_path=FIGURES_PATH / "sct_vs_gr.pdf")
    # save_results({**fit_gr, **fit_sct}, comp, "sct_test_001.json")

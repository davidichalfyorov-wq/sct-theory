"""
SCT Theory — Fitting and statistical analysis utilities.

Provides unified interface for parameter estimation:
    - iminuit (Minuit2 — CERN standard)
    - lmfit (constrained non-linear least squares)
    - emcee (MCMC)
    - pymc (HMC/NUTS Bayesian inference)
    - pyhf (HistFactory for LHC exclusion limits)
    - looptools (Passarino-Veltman one-loop integrals)
    - pyerrors (correlated error analysis with gamma method)

Also provides:
    - Model comparison: chi2, AIC, BIC, Bayesian evidence
    - Formal statistical tests: KS, Anderson-Darling, likelihood ratio (statsmodels)
"""

import numpy as np

# =============================================================================
# CHI-SQUARED AND MODEL COMPARISON
# =============================================================================

def chi2(observed, expected, errors, n_params=0):
    """Compute chi-squared statistic.

    Parameters:
        observed: array of measured values
        expected: array of predicted values
        errors: array of uncertainties (1-sigma)
        n_params: number of fitted parameters (default 0 = direct theory comparison)

    Returns:
        (chi2_value, ndof, chi2_reduced, p_value)
    """
    import warnings as _warnings

    obs = np.asarray(observed, dtype=float)
    exp = np.asarray(expected, dtype=float)
    err = np.asarray(errors, dtype=float)
    if obs.size == 0:
        raise ValueError("chi2: received empty arrays")
    if not (obs.shape == exp.shape == err.shape):
        raise ValueError(
            f"chi2: array shapes must match, got observed={obs.shape}, "
            f"expected={exp.shape}, errors={err.shape}"
        )
    if n_params < 0:
        raise ValueError(f"chi2: n_params must be non-negative, got {n_params}")
    if np.any(~np.isfinite(obs)):
        raise ValueError("chi2: observed array contains NaN or infinite value(s)")
    if np.any(~np.isfinite(exp)):
        raise ValueError("chi2: expected array contains NaN or infinite value(s)")
    if np.any(~np.isfinite(err)):
        raise ValueError("chi2: errors array contains NaN or infinite value(s)")
    if np.any(err <= 0):
        raise ValueError(
            "chi2: errors array contains zero or negative value(s)"
            " — requires positive uncertainties"
        )
    chi2_val = np.sum(((obs - exp) / err) ** 2)
    raw_ndof = len(obs) - n_params
    if raw_ndof <= 0:
        _warnings.warn(
            f"chi2: n_params ({n_params}) >= n_data ({len(obs)}), "
            f"ndof would be {raw_ndof}. Clamping to 1.",
            stacklevel=2,
        )
    ndof = max(raw_ndof, 1)
    chi2_red = chi2_val / ndof
    from scipy.stats import chi2 as chi2_dist
    p_val = float(chi2_dist.sf(chi2_val, ndof))
    return chi2_val, ndof, chi2_red, p_val


def model_comparison(chi2_1, k_1, chi2_2, k_2, n_data):
    """Compare two models via AIC and BIC.

    Assumes chi2 values equal -2*ln(L) (Gaussian/least-squares likelihood).

    Parameters:
        chi2_1, chi2_2: chi-squared values for models 1, 2
        k_1, k_2: number of free parameters
        n_data: number of data points

    Returns:
        dict with automation supportC, BIC, and delta values. Negative delta favors model 1.
    """
    if not np.isfinite(chi2_1) or not np.isfinite(chi2_2):
        raise ValueError(
            f"model_comparison requires finite chi2 values, got chi2_1={chi2_1}, chi2_2={chi2_2}"
        )
    if not np.isfinite(k_1) or not np.isfinite(k_2):
        raise ValueError(
            f"model_comparison requires finite parameter counts, got k_1={k_1}, k_2={k_2}"
        )
    if k_1 < 0 or k_2 < 0:
        raise ValueError(
            f"model_comparison requires non-negative parameter counts, got k_1={k_1}, k_2={k_2}"
        )
    if not np.isfinite(n_data):
        raise ValueError(f"model_comparison requires finite n_data, got {n_data}")
    if n_data < 2:
        raise ValueError(
            f"model_comparison requires n_data >= 2 for meaningful BIC, got {n_data}"
        )
    aic_1 = chi2_1 + 2 * k_1
    aic_2 = chi2_2 + 2 * k_2
    # AICc: corrected AIC for small sample size (converges to AIC for n >> k)
    if n_data - k_1 - 1 > 0:
        aicc_1 = aic_1 + 2 * k_1 * (k_1 + 1) / (n_data - k_1 - 1)
    else:
        aicc_1 = np.inf
    if n_data - k_2 - 1 > 0:
        aicc_2 = aic_2 + 2 * k_2 * (k_2 + 1) / (n_data - k_2 - 1)
    else:
        aicc_2 = np.inf
    bic_1 = chi2_1 + k_1 * np.log(n_data)
    bic_2 = chi2_2 + k_2 * np.log(n_data)
    return {
        'AIC_1': aic_1, 'AIC_2': aic_2, 'dAIC': aic_1 - aic_2,
        'AICc_1': aicc_1, 'AICc_2': aicc_2, 'dAICc': aicc_1 - aicc_2,
        'BIC_1': bic_1, 'BIC_2': bic_2, 'dBIC': bic_1 - bic_2,
        'favors': 'model_1' if aic_1 < aic_2 else 'model_2',
    }


# =============================================================================
# IMINUIT FITTING (CERN Minuit2)
# =============================================================================

def fit_minuit(cost_func, initial_params, param_names=None, limits=None,
               errordef=1.0):
    """Fit using iminuit (Minuit2).

    Parameters:
        cost_func: function to minimize, takes (*params) -> scalar
        initial_params: dict or list of initial values
        param_names: list of parameter names (if initial_params is a list)
        limits: dict of {param_name: (low, high)} or None
        errordef: 1.0 for chi2, 0.5 for negative log-likelihood

    Returns:
        iminuit.Minuit object (access .values, .errors, .fval, .valid)
    """
    from iminuit import Minuit
    if isinstance(initial_params, dict):
        m = Minuit(cost_func, **initial_params)
    else:
        m = Minuit(cost_func, *initial_params, name=param_names)
    m.errordef = errordef
    if limits:
        for name, (lo, hi) in limits.items():
            m.limits[name] = (lo, hi)
    m.migrad()
    m.hesse()
    return m


# =============================================================================
# LMFIT WRAPPER
# =============================================================================

def fit_lmfit(model_func, x_data, y_data, y_err, params_dict, method='leastsq'):
    """Fit using lmfit.

    Parameters:
        model_func: function f(x, **params) -> y_predicted
        x_data, y_data: data arrays
        y_err: uncertainty array
        params_dict: dict of {name: (initial, min, max)} or {name: initial}
        method: optimization method

    Returns:
        lmfit.MinimizerResult
    """
    from lmfit import Parameters, minimize

    x_arr = np.asarray(x_data, dtype=float)
    y_arr = np.asarray(y_data, dtype=float)
    y_err_arr = np.asarray(y_err, dtype=float)
    if x_arr.size == 0:
        raise ValueError("fit_lmfit: received empty arrays")
    if not (x_arr.shape == y_arr.shape == y_err_arr.shape):
        raise ValueError(
            f"fit_lmfit: array shapes must match, got x_data={x_arr.shape}, "
            f"y_data={y_arr.shape}, y_err={y_err_arr.shape}"
        )
    if np.any(~np.isfinite(x_arr)):
        raise ValueError("fit_lmfit: x_data contains NaN or infinite value(s)")
    if np.any(~np.isfinite(y_arr)):
        raise ValueError("fit_lmfit: y_data contains NaN or infinite value(s)")
    if np.any(~np.isfinite(y_err_arr)):
        raise ValueError("fit_lmfit: y_err contains NaN or infinite value(s)")
    if np.any(y_err_arr <= 0):
        raise ValueError("fit_lmfit: y_err must be all positive (got zero or negative values)")

    params = Parameters()
    for name, val in params_dict.items():
        if isinstance(val, tuple):
            if len(val) != 3:
                raise ValueError(
                    f"fit_lmfit: params_dict['{name}'] tuple must be "
                    f"(initial, min, max), got length {len(val)}"
                )
            params.add(name, value=val[0], min=val[1], max=val[2])
        else:
            params.add(name, value=val)

    def residual(pars, x, y, yerr):
        kw = {n: pars[n].value for n in pars}
        return (y - model_func(x, **kw)) / yerr

    return minimize(residual, params, args=(x_arr, y_arr, y_err_arr),
                    method=method)


# =============================================================================
# MCMC WRAPPER (emcee)
# =============================================================================

def run_mcmc(log_prob, initial_pos, nwalkers=32, nsteps=5000,
             progress=True, **kwargs):
    """Run MCMC sampling with emcee.

    Parameters:
        log_prob: function(theta) -> log probability
        initial_pos: (nwalkers, ndim) array of initial positions
        nwalkers: number of walkers
        nsteps: number of steps per walker
        progress: show tqdm progress bar
        **kwargs: passed to emcee.EnsembleSampler

    Returns:
        emcee.EnsembleSampler (access .chain, .lnprobability, etc.)
    """
    import emcee
    initial_pos = np.asarray(initial_pos, dtype=float)
    if initial_pos.ndim == 1:
        raise ValueError(
            f"run_mcmc: initial_pos must be 2D (nwalkers, ndim), got 1D shape {initial_pos.shape}. "
            f"Reshape with initial_pos.reshape(nwalkers, ndim)."
        )
    if np.any(~np.isfinite(initial_pos)):
        raise ValueError("run_mcmc: initial_pos contains NaN or infinite value(s)")
    if initial_pos.shape[0] != nwalkers:
        raise ValueError(
            f"run_mcmc: initial_pos has {initial_pos.shape[0]} rows but "
            f"nwalkers={nwalkers}. Shape must be (nwalkers, ndim)."
        )
    ndim = initial_pos.shape[1]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, **kwargs)
    sampler.run_mcmc(initial_pos, nsteps, progress=progress)
    return sampler


def mcmc_summary(sampler, param_names, discard=1000, thin=10):
    """Extract MCMC summary statistics.

    Parameters:
        sampler: emcee.EnsembleSampler
        param_names: list of parameter names
        discard: burn-in samples to discard
        thin: thinning factor

    Returns:
        dict of {name: (median, lower_1sigma, upper_1sigma)}
    """
    flat = sampler.get_chain(discard=discard, thin=thin, flat=True)
    if flat.shape[0] == 0:
        raise ValueError(
            f"mcmc_summary: no samples remain after discard={discard}, thin={thin}. "
            f"Reduce discard or thin, or run more steps."
        )
    if len(param_names) != flat.shape[1]:
        raise ValueError(
            f"mcmc_summary: param_names has {len(param_names)} entries but "
            f"chain has {flat.shape[1]} parameters"
        )
    summary = {}
    for i, name in enumerate(param_names):
        q = np.percentile(flat[:, i], [16, 50, 84])
        summary[name] = {
            'median': q[1],
            'lower': q[0] - q[1],
            'upper': q[2] - q[1],
            'mean': np.mean(flat[:, i]),
            'std': np.std(flat[:, i]),
        }
    return summary


# =============================================================================
# PYHF (HistFactory for LHC limits)
# =============================================================================

def pyhf_cls(signal, background, observed, bkg_uncertainty=None):
    """Compute CL_s exclusion using pyhf.

    Parameters:
        signal: array of signal yields per bin
        background: array of background yields per bin
        observed: array of observed counts per bin
        bkg_uncertainty: array of background uncertainties (None = Poisson only)

    Returns:
        dict with CL_s, CL_sb, CL_b values and exclusion boolean (95% CL).
    """
    import pyhf
    model_spec = {
        'channels': [{
            'name': 'singlechannel',
            'samples': [
                {
                    'name': 'signal',
                    'data': list(signal),
                    'modifiers': [{'name': 'mu', 'type': 'normfactor', 'data': None}],
                },
                {
                    'name': 'background',
                    'data': list(background),
                    'modifiers': (
                        [{'name': 'bkg_unc', 'type': 'shapesys',
                          'data': list(bkg_uncertainty)}]
                        if bkg_uncertainty is not None else []
                    ),
                },
            ],
        }],
    }
    model = pyhf.Model(model_spec)
    data = list(observed) + model.config.auxdata
    CLs_obs, CLs_exp = pyhf.infer.hypotest(
        1.0, data, model, return_expected_set=True
    )
    return {
        'CLs_obs': float(CLs_obs),
        'CLs_exp': [float(c) for c in CLs_exp],
        'excluded_95': float(CLs_obs) < 0.05,
    }


# =============================================================================
# LOOPTOOLS (Passarino-Veltman one-loop integrals)
# =============================================================================

def pv_scalar_integrals(p_sq, m1_sq, m2_sq=None, m3_sq=None, m4_sq=None,
                        mu_sq=None):
    """Compute Passarino-Veltman scalar integrals A0, B0, C0, D0.

    Wraps LoopTools (Fortran library by T. Hahn) for standard one-loop
    integrals in dimensional regularization.

    Parameters:
        p_sq: external momentum squared (or list for C0/D0)
        m1_sq: first internal mass squared
        m2_sq: second internal mass squared (for B0+)
        m3_sq: third internal mass squared (for C0+)
        m4_sq: fourth internal mass squared (for D0)
        mu_sq: renormalization scale squared (default: 1)

    Returns:
        Complex value of the scalar integral.

    Examples:
        A0(m^2):       pv_scalar_integrals(0, m_sq)
        B0(p^2;m1,m2): pv_scalar_integrals(p_sq, m1_sq, m2_sq)
    """
    import looptools as lt
    lt.clearcache()
    if mu_sq is not None:
        lt.setmudim(mu_sq)

    if m2_sq is None:
        # A0(m^2)
        return lt.A0(m1_sq)
    elif m3_sq is None:
        # B0(p^2; m1^2, m2^2)
        return lt.B0(p_sq, m1_sq, m2_sq)
    elif m4_sq is None:
        # C0(p1^2, p2^2, (p1+p2)^2; m1^2, m2^2, m3^2)
        if isinstance(p_sq, (list, tuple)):
            return lt.C0(p_sq[0], p_sq[1], p_sq[2], m1_sq, m2_sq, m3_sq)
        return lt.C0(p_sq, 0, 0, m1_sq, m2_sq, m3_sq)
    else:
        # D0(p1^2,...; m1^2,...,m4^2)
        if isinstance(p_sq, (list, tuple)):
            return lt.D0(p_sq[0], p_sq[1], p_sq[2], p_sq[3], p_sq[4], p_sq[5],
                         m1_sq, m2_sq, m3_sq, m4_sq)
        return lt.D0(p_sq, 0, 0, 0, 0, 0, m1_sq, m2_sq, m3_sq, m4_sq)


# =============================================================================
# PYERRORS (correlated error analysis)
# =============================================================================

def obs_with_errors(values, name="obs"):
    """Create a pyerrors Observable from a set of measurements.

    Uses the gamma method for autocorrelation-aware error estimation.
    Standard in lattice QCD; superior to naive sqrt(var/N) for correlated data.

    Parameters:
        values: array of measurements (e.g., from MCMC chain or replicas)
        name: label for the observable

    Returns:
        pyerrors.Obs object with .value, .dvalue, .gamma_method() available.
    """
    import pyerrors as pe
    obs = pe.Obs([np.asarray(values, dtype=float)], [name])
    obs.gamma_method()
    return obs


def derived_observable(func, observables):
    """Propagate errors through an arbitrary function using pyerrors.

    Uses automatic differentiation — exact for any differentiable function.
    Properly handles correlations between input observables.

    Parameters:
        func: callable f(*obs_values) -> scalar
        observables: list of pyerrors.Obs objects

    Returns:
        pyerrors.Obs for the derived quantity with propagated errors.

    Example:
        a = obs_with_errors(data_a, "a")
        b = obs_with_errors(data_b, "b")
        ratio = derived_observable(lambda x, y: x / y, [a, b])
    """
    import pyerrors as pe
    return pe.derived_observable(func, observables)


# =============================================================================
# PYMC (HMC/NUTS Bayesian inference)
# =============================================================================

def run_pymc(build_model, draws=5000, tune=2000, chains=4, target_accept=0.9,
             seed=42):
    """Run Bayesian inference using PyMC with NUTS sampler.

    Superior to emcee for: high-dimensional problems, complex posteriors,
    hierarchical models, and cases where gradient information helps.

    Parameters:
        build_model: callable that returns a pymc.Model context.
                     Inside, define priors and likelihood using pm.* API.
        draws: number of posterior samples per chain
        tune: number of tuning (warm-up) steps
        chains: number of independent chains
        target_accept: NUTS target acceptance rate (0.8-0.95)
        seed: random seed

    Returns:
        arviz.InferenceData with posterior samples and diagnostics.

    Example:
        import pymc as pm

        def my_model():
            with pm.Model() as model:
                H0 = pm.Normal('H0', mu=70, sigma=5)
                sigma = pm.HalfNormal('sigma', sigma=2)
                pm.Normal('obs', mu=H0, sigma=sigma, observed=data)
            return model

        idata = run_pymc(my_model, draws=10000)
    """
    import pymc as pm
    model = build_model()
    with model:
        idata = pm.sample(
            draws=draws, tune=tune, chains=chains,
            target_accept=target_accept, random_seed=seed,
            return_inferencedata=True,
        )
    return idata


def pymc_summary(idata, var_names=None, hdi_prob=0.94):
    """Summarize PyMC inference results.

    Parameters:
        idata: arviz.InferenceData from run_pymc
        var_names: list of variable names to summarize (None = all)
        hdi_prob: highest density interval probability (default: 94%)

    Returns:
        pandas DataFrame with mean, sd, hdi, ess, r_hat.
    """
    import arviz as az
    return az.summary(idata, var_names=var_names, hdi_prob=hdi_prob)


def pymc_compare(models_dict):
    """Compare multiple Bayesian models using WAIC/LOO.

    Parameters:
        models_dict: dict of {'model_name': arviz.InferenceData}

    Returns:
        pandas DataFrame with ELPD, WAIC, weights.

    Example:
        comparison = pymc_compare({'SCT': idata_sct, 'GR': idata_gr})
    """
    import arviz as az
    return az.compare(models_dict)


# =============================================================================
# STATSMODELS (formal statistical tests)
# =============================================================================

def ks_test(data, cdf_func):
    """Kolmogorov-Smirnov test: does data follow a given distribution?

    Parameters:
        data: array of observations
        cdf_func: callable cdf(x) -> probability, or string name
                  (e.g., 'norm', 'chi2')

    Returns:
        dict with 'statistic', 'p_value', 'reject_H0_5pct'.
    """
    from scipy.stats import kstest
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        raise ValueError("ks_test: received empty data array")
    if np.any(~np.isfinite(data)):
        raise ValueError("ks_test: received NaN or infinite value(s) in data")
    stat, pval = kstest(data, cdf_func)
    return {
        'statistic': stat,
        'p_value': pval,
        'reject_H0_5pct': pval < 0.05,
    }


def anderson_darling_test(data, dist='norm'):
    """Anderson-Darling test for normality (or other distributions).

    More sensitive than KS test at distribution tails.
    Critical for: checking if residuals are truly Gaussian,
    validating chi2 assumptions.

    Parameters:
        data: array of observations
        dist: 'norm', 'expon', 'logistic', or 'gumbel'

    Returns:
        dict with 'statistic', 'critical_values', 'significance_levels',
        'reject_5pct'.
    """
    _VALID_AD_DISTS = {'norm', 'expon', 'logistic', 'gumbel', 'gumbel_l',
                        'gumbel_r', 'extreme1'}
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        raise ValueError("anderson_darling_test: received empty data array")
    if np.any(~np.isfinite(data)):
        raise ValueError("anderson_darling_test: received NaN or infinite value(s) in data")
    if dist not in _VALID_AD_DISTS:
        raise ValueError(
            f"anderson_darling_test: dist must be one of {sorted(_VALID_AD_DISTS)}, "
            f"got {dist!r}"
        )
    import warnings as _warnings

    from scipy.stats import anderson
    try:
        from scipy.stats import MonteCarloMethod
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", FutureWarning)
            result = anderson(data, dist=dist, method=MonteCarloMethod(rvs=None, n_resamples=999))
        return {
            'statistic': float(result.statistic),
            'p_value': float(result.pvalue),
            'reject_5pct': float(result.pvalue) < 0.05,
        }
    except (ImportError, TypeError):
        # scipy < 1.13: MonteCarloMethod not available, use classic API
        result = anderson(data, dist=dist)
        # Approximate p-value from critical values at 5% significance level
        reject = float(result.statistic) > float(result.critical_values[2])
        return {
            'statistic': float(result.statistic),
            'p_value': 0.05 if reject else 0.10,  # conservative estimate
            'critical_values': list(result.critical_values),
            'significance_levels': list(result.significance_level),
            'reject_5pct': reject,
        }


def likelihood_ratio_test(log_lik_null, log_lik_alt, df_diff):
    """Likelihood ratio test for nested models.

    Tests whether a more complex model (alt) significantly improves
    over a simpler model (null). The test statistic -2(L0 - L1)
    follows chi2 with df_diff degrees of freedom.

    Parameters:
        log_lik_null: log-likelihood of null (simpler) model
        log_lik_alt: log-likelihood of alternative (complex) model
        df_diff: difference in number of free parameters

    Returns:
        dict with 'statistic', 'p_value', 'reject_null_5pct'.

    Example:
        # Is SCT (with extra parameter Lambda) better than GR?
        result = likelihood_ratio_test(logL_GR, logL_SCT, df_diff=1)
    """
    import warnings

    from scipy.stats import chi2 as chi2_dist
    if not np.isfinite(log_lik_null) or not np.isfinite(log_lik_alt):
        raise ValueError(
            f"likelihood_ratio_test requires finite log-likelihoods, "
            f"got null={log_lik_null}, alt={log_lik_alt}"
        )
    if not np.isfinite(df_diff):
        raise ValueError(f"likelihood_ratio_test requires finite df_diff, got {df_diff}")
    if df_diff <= 0:
        raise ValueError(f"likelihood_ratio_test: df_diff must be positive, got {df_diff}")
    stat = -2.0 * (log_lik_null - log_lik_alt)
    if stat < 0:
        warnings.warn(
            f"likelihood_ratio_test: test statistic is negative ({stat:.4g}), "
            f"meaning the null model fits better than the alternative. "
            f"Clamping to 0 (p-value = 1).",
            stacklevel=2,
        )
        stat = 0.0
    pval = float(chi2_dist.sf(stat, df_diff))
    return {
        'statistic': stat,
        'p_value': pval,
        'df': df_diff,
        'reject_null_5pct': pval < 0.05,
    }


def weighted_least_squares(x, y, y_err, degree=1):
    """Weighted least squares polynomial fit with full diagnostics.

    Uses statsmodels WLS for proper statistical output: parameter
    uncertainties, R-squared, F-statistic, residual diagnostics.

    Parameters:
        x, y: data arrays
        y_err: uncertainty array (used as weights = 1/sigma^2)
        degree: polynomial degree (1 = linear, 2 = quadratic, etc.)

    Returns:
        statsmodels RegressionResults object with:
            .params, .bse (std errors), .rsquared, .f_pvalue,
            .summary() for full report.
    """
    import statsmodels.api as sm
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y_err_arr = np.asarray(y_err, dtype=float)
    if x.size == 0 or y.size == 0 or y_err_arr.size == 0:
        raise ValueError("weighted_least_squares: received empty array(s)")
    if not (x.shape == y.shape == y_err_arr.shape):
        raise ValueError(
            f"weighted_least_squares: array shapes must match, got x={x.shape}, "
            f"y={y.shape}, y_err={y_err_arr.shape}"
        )
    if np.any(~np.isfinite(x)):
        raise ValueError("weighted_least_squares: x array contains NaN or infinite value(s)")
    if np.any(~np.isfinite(y)):
        raise ValueError("weighted_least_squares: y array contains NaN or infinite value(s)")
    if np.any(~np.isfinite(y_err_arr)):
        raise ValueError("weighted_least_squares: y_err array contains NaN or infinite value(s)")
    if np.any(y_err_arr <= 0):
        raise ValueError("weighted_least_squares: y_err array contains zero or negative value(s)")
    if degree < 0:
        raise ValueError(f"weighted_least_squares: degree must be non-negative, got {degree}")
    if degree >= len(x):
        raise ValueError(
            f"weighted_least_squares: degree ({degree}) must be less than "
            f"number of data points ({len(x)})"
        )
    # Build polynomial design matrix
    X = np.column_stack([x ** i for i in range(degree + 1)])
    weights = 1.0 / y_err_arr ** 2
    model = sm.WLS(y, X, weights=weights)
    return model.fit()


def residual_diagnostics(residuals):
    """Run comprehensive residual diagnostics.

    Checks for: normality, autocorrelation, heteroscedasticity.
    Essential for validating that a fit is statistically sound.

    Parameters:
        residuals: array of fit residuals (observed - model)

    Returns:
        dict with test results for normality (Shapiro-Wilk),
        autocorrelation (Durbin-Watson), and summary.
    """
    import statsmodels.stats.stattools as st
    from scipy.stats import shapiro

    res = np.asarray(residuals, dtype=float)
    if res.size == 0:
        raise ValueError("residual_diagnostics: received empty residuals array")
    if np.any(~np.isfinite(res)):
        raise ValueError("residual_diagnostics: received NaN or infinite residual(s)")
    # Shapiro-Wilk normality test
    if len(res) >= 3:
        sw_stat, sw_pval = shapiro(res)
    else:
        sw_stat, sw_pval = np.nan, np.nan

    # Durbin-Watson autocorrelation test (near 2 = no autocorrelation)
    dw = st.durbin_watson(res)

    return {
        'shapiro_wilk': {'statistic': sw_stat, 'p_value': sw_pval,
                         'normal_5pct': sw_pval > 0.05 if not np.isnan(sw_pval) else None},
        'durbin_watson': {'statistic': dw,
                          'interpretation': 'no autocorrelation' if 1.5 < dw < 2.5
                          else 'possible autocorrelation'},
        'mean': float(np.mean(res)),
        'std': float(np.std(res)),
        'skewness': float(np.mean(((res - np.mean(res)) / np.std(res)) ** 3))
        if np.std(res) > 1e-300 else 0.0,
    }


# =============================================================================
# MINOS ASYMMETRIC ERRORS (iminuit)
# =============================================================================

def fit_minuit_minos(cost_func, initial_params, param_names=None,
                     limits=None, errordef=1.0, cl=None):
    """Fit using iminuit with MINOS asymmetric confidence intervals.

    MINOS computes exact (non-parabolic) confidence intervals by
    profiling the likelihood. Essential for non-Gaussian posteriors
    and parameters near boundaries.

    Parameters:
        cost_func: function to minimize
        initial_params: initial values
        param_names: parameter names
        limits: parameter limits
        errordef: 1.0 for chi2, 0.5 for NLL
        cl: confidence level for MINOS (None = 1-sigma)

    Returns:
        iminuit.Minuit object with .merrors for MINOS intervals.
    """
    from iminuit import Minuit
    if isinstance(initial_params, dict):
        m = Minuit(cost_func, **initial_params)
    else:
        m = Minuit(cost_func, *initial_params, name=param_names)
    m.errordef = errordef
    if limits:
        for name, (lo, hi) in limits.items():
            m.limits[name] = (lo, hi)
    m.migrad()
    m.hesse()
    m.minos(cl=cl)
    return m


# =============================================================================
# CHI-SQUARED WITH COVARIANCE MATRIX
# =============================================================================

def chi2_cov(observed, expected, cov_matrix, n_params=0):
    """Chi-squared with full covariance matrix.

    chi2 = (obs - exp)^T C^{-1} (obs - exp)

    Properly handles correlated uncertainties from systematics.
    Uses Cholesky decomposition for numerically stable inversion.

    Parameters:
        observed: array of measured values
        expected: array of predicted values
        cov_matrix: (N, N) covariance matrix (must be positive definite)
        n_params: number of fitted parameters (default 0)

    Returns:
        (chi2_value, ndof, chi2_reduced, p_value)
    """
    import warnings as _warnings

    obs = np.asarray(observed, dtype=float)
    exp = np.asarray(expected, dtype=float)
    cov = np.asarray(cov_matrix, dtype=float)
    if obs.size == 0:
        raise ValueError("chi2_cov: received empty arrays")
    if obs.shape != exp.shape:
        raise ValueError(
            f"chi2_cov: observed and expected shapes must match, "
            f"got {obs.shape} and {exp.shape}"
        )
    if cov.shape != (len(obs), len(obs)):
        raise ValueError(
            f"chi2_cov: covariance matrix shape must be ({len(obs)}, {len(obs)}), "
            f"got {cov.shape}"
        )
    if n_params < 0:
        raise ValueError(f"chi2_cov: n_params must be non-negative, got {n_params}")
    if np.any(~np.isfinite(obs)):
        raise ValueError("chi2_cov: observed array contains NaN or infinite value(s)")
    if np.any(~np.isfinite(exp)):
        raise ValueError("chi2_cov: expected array contains NaN or infinite value(s)")
    if np.any(~np.isfinite(cov)):
        raise ValueError("chi2_cov: covariance matrix contains NaN or infinite value(s)")
    delta = obs - exp
    # Cholesky-based solve: numerically stable for ill-conditioned cov matrices
    try:
        L = np.linalg.cholesky(cov)
        y = np.linalg.solve(L, delta)
        chi2_val = float(y @ y)
    except np.linalg.LinAlgError:
        # Fallback: regularize near-singular matrix with Tikhonov damping
        eps = np.finfo(float).eps * max(np.trace(cov) / len(delta), 1.0)
        cov_reg = cov + eps * np.eye(len(delta))
        _warnings.warn(
            f"chi2_cov: covariance matrix is not positive definite. "
            f"Applied Tikhonov regularization with eps={eps:.2e}. "
            f"Chi-squared value may be approximate.",
            stacklevel=2,
        )
        chi2_val = float(delta @ np.linalg.solve(cov_reg, delta))
    raw_ndof = len(obs) - n_params
    if raw_ndof <= 0:
        _warnings.warn(
            f"chi2_cov: n_params ({n_params}) >= n_data ({len(obs)}), "
            f"ndof would be {raw_ndof}. Clamping to 1.",
            stacklevel=2,
        )
    ndof = max(raw_ndof, 1)
    chi2_red = chi2_val / ndof
    from scipy.stats import chi2 as chi2_dist
    p_val = float(chi2_dist.sf(chi2_val, ndof))
    return chi2_val, ndof, chi2_red, p_val


# =============================================================================
# BAYESIAN UPPER/LOWER LIMITS
# =============================================================================

def bayesian_limit(samples, cl=0.95, side='upper'):
    """Extract Bayesian credible limit from posterior samples.

    Parameters:
        samples: 1D array of MCMC posterior samples for one parameter
        cl: credible level (default: 0.95 for 95% CL)
        side: 'upper' (one-sided upper), 'lower' (one-sided lower),
              'hdi' (highest density interval, two-sided)

    Returns:
        dict with limit value(s) and summary statistics.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.size == 0:
        raise ValueError("bayesian_limit: received empty samples array")
    if np.any(~np.isfinite(samples)):
        raise ValueError("bayesian_limit: received NaN or infinite value(s) in samples")
    if samples.ndim > 1 and samples.shape[1] > 1:
        raise ValueError(
            f"bayesian_limit: expects 1D samples for a single parameter, "
            f"got shape {samples.shape}. Pass one column at a time."
        )
    samples = samples.ravel()
    if len(samples) < 2:
        raise ValueError(f"bayesian_limit: requires >= 2 samples, got {len(samples)}")
    if not 0 < cl < 1:
        raise ValueError(f"bayesian_limit: cl must be in (0, 1), got {cl}")
    if side == 'upper':
        limit = float(np.percentile(samples, cl * 100))
        return {
            'limit': limit,
            'cl': cl,
            'side': 'upper',
            'median': float(np.median(samples)),
            'mean': float(np.mean(samples)),
        }
    elif side == 'lower':
        limit = float(np.percentile(samples, (1 - cl) * 100))
        return {
            'limit': limit,
            'cl': cl,
            'side': 'lower',
            'median': float(np.median(samples)),
            'mean': float(np.mean(samples)),
        }
    elif side == 'hdi':
        sorted_s = np.sort(samples)
        n = len(sorted_s)
        ci_size = max(int(np.ceil(cl * n)), 1)
        if ci_size >= n:
            ci_size = n - 1
        widths = sorted_s[ci_size:] - sorted_s[:n - ci_size]
        best = int(np.argmin(widths))
        return {
            'lower': float(sorted_s[best]),
            'upper': float(sorted_s[best + ci_size]),
            'cl': cl,
            'side': 'hdi',
            'width': float(widths[best]),
            'median': float(np.median(samples)),
        }
    else:
        raise ValueError(f"bayesian_limit: unknown side '{side}'. Use 'upper', 'lower', or 'hdi'.")


def discovery_significance(signal_count, background_count):
    """Compute discovery significance (Z-value) for counting experiment.

    Uses the profile likelihood ratio formula:
    Z = sqrt(2 * ((s+b)*ln(1+s/b) - s))

    Standard formula used at LHC for discovery claims.

    Parameters:
        signal_count: expected signal yield s (not raw observed count)
        background_count: expected background yield b

    Returns:
        dict with 'Z' (significance) and 'p_value'.
    """
    from scipy.stats import norm
    s = float(signal_count)
    b = float(background_count)
    if not np.isfinite(s) or not np.isfinite(b):
        raise ValueError(f"discovery_significance requires finite counts, got s={s}, b={b}")
    if s < 0:
        raise ValueError(
            f"discovery_significance: signal_count must be non-negative, got {s}"
        )
    if s == 0:
        return {'Z': 0.0, 'p_value': 1.0}
    if b < 0:
        raise ValueError(
            f"discovery_significance: background_count must be non-negative, got {b}"
        )
    if b == 0:
        # Signal with zero background = infinite significance
        return {'Z': np.inf, 'p_value': 0.0}
    Z = np.sqrt(2 * ((s + b) * np.log(1 + s / b) - s))
    p_val = float(norm.sf(Z))
    return {'Z': Z, 'p_value': p_val}

import pymc as pm
import numpy as np
import arviz as az


def fit_no_pooling(
    outcome: np.ndarray,
    segment_idx: np.ndarray,
    n_segments: int,
    draws: int = 1000,
    tune: int = 1000,
):
    with pm.Model() as model:
        alpha = pm.Normal("alpha", 0, 10, shape=n_segments)
        p = pm.math.sigmoid(alpha[segment_idx])
        y = pm.Bernoulli("y", p=p, observed=outcome)
        trace = pm.sample(draws, tune=tune, cores=1,
                          random_seed=42, progressbar=True,
                          idata_kwargs={"log_likelihood": True})
    return model, trace


def fit_full_pooling(
    outcome: np.ndarray,
    draws: int = 1000,
    tune: int = 1000,
):
    with pm.Model() as model:
        alpha = pm.Normal("alpha", 0, 10)
        p = pm.math.sigmoid(alpha)
        y = pm.Bernoulli("y", p=p, observed=outcome)
        trace = pm.sample(draws, tune=tune, cores=1,
                          random_seed=42, progressbar=True,
                          idata_kwargs={"log_likelihood": True})
    return model, trace


def fit_partial_pooling(
    outcome: np.ndarray,
    segment_idx: np.ndarray,
    n_segments: int,
    draws: int = 1000,
    tune: int = 1000,
):
    with pm.Model() as model:
        mu = pm.Normal("mu", 0, 1)
        tau = pm.HalfNormal("tau", 1)
        alpha_offset = pm.Normal("alpha_offset", 0, 1, shape=n_segments)
        alpha = pm.Deterministic("alpha", mu + tau * alpha_offset)
        p = pm.math.sigmoid(alpha[segment_idx])
        y = pm.Bernoulli("y", p=p, observed=outcome)
        trace = pm.sample(draws, tune=tune, cores=1,
                          random_seed=42, progressbar=True,
                          idata_kwargs={"log_likelihood": True})
    return model, trace


def check_divergences(trace):
    divergences = trace.sample_stats["diverging"].sum().item()
    print(f"Divergences: {divergences}")
    return divergences

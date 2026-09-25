"""Multiple-testing and Sharpe-ratio deflation diagnostics."""

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

_STANDARD_NORMAL = NormalDist()
_EULER_MASCHERONI = 0.5772156649015329
_TRADING_DAYS_PER_YEAR = 252.0


def expected_max_standard_normal(n_trials):
    """Approximate E[max Z] for n independent standard-normal trials."""
    n_trials = float(n_trials)
    if n_trials <= 1.0:
        return 0.0
    if n_trials < 2.0:
        # The extreme-value approximation is poor close to one trial.  Linear
        # interpolation preserves the exact one-trial boundary and continuity.
        return (n_trials - 1.0) * expected_max_standard_normal(2.0)
    first = _STANDARD_NORMAL.inv_cdf(1.0 - 1.0 / n_trials)
    second = _STANDARD_NORMAL.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return (1.0 - _EULER_MASCHERONI) * first + _EULER_MASCHERONI * second

def exact_expected_max_standard_normal(n_trials):
    """Numerically evaluate E[max Z] for iid standard normals."""
    n_trials = float(n_trials)
    if n_trials <= 1.0:
        return 0.0
    nodes, weights = np.polynomial.hermite.hermgauss(80)
    values = math.sqrt(2.0) * nodes
    cdf = np.array([_STANDARD_NORMAL.cdf(float(value)) for value in values])
    cdf = np.clip(cdf, np.finfo(float).tiny, 1.0)
    density_weight = np.exp((n_trials - 1.0) * np.log(cdf))
    return float(
        n_trials / math.sqrt(math.pi)
        * np.dot(weights, values * density_weight)
    )

def equivalent_independent_trials(expected_max_z, raw_count):
    """Map an expected correlated maximum to an equivalent iid trial count."""
    raw_count = int(raw_count)
    if raw_count <= 1 or expected_max_z <= 0.0:
        return 1.0
    upper_value = exact_expected_max_standard_normal(raw_count)
    if expected_max_z >= upper_value:
        return float(raw_count)
    two_value = exact_expected_max_standard_normal(2.0)
    if expected_max_z <= two_value:
        return 1.0 + expected_max_z / two_value
    low, high = 2.0, float(raw_count)
    for _ in range(50):
        mid = 0.5 * (low + high)
        if exact_expected_max_standard_normal(mid) < expected_max_z:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)

def galwey_effective_tests(correlation):
    """Return Galwey's eigenvalue-based effective number of correlated tests."""
    if correlation.size == 0:
        return float("nan")
    eigenvalues = np.linalg.eigvalsh(correlation)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = eigenvalues.sum()
    if total <= 0.0:
        return float("nan")
    return float(np.square(np.sqrt(eigenvalues).sum()) / total)

def participation_effective_tests(correlation):
    """Return the inverse-Herfindahl/participation-ratio effective rank."""
    if correlation.size == 0:
        return float("nan")
    eigenvalues = np.linalg.eigvalsh(correlation)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = eigenvalues.sum()
    squares = np.square(eigenvalues).sum()
    if total <= 0.0 or squares <= 0.0:
        return float("nan")
    return float(total * total / squares)

def aligned_strategy_returns(rows):
    """Return valid candidate rows and their common-window return matrix."""
    candidates = [
        row for row in rows
        if row["type"] in ("Strategy", "*Average*")
        and math.isfinite(row.get("sharpe", float("nan")))
        and "_returns" in row
    ]
    if not candidates:
        return [], pd.DataFrame()
    series = [
        pd.to_numeric(row["_returns"], errors="coerce").rename(index)
        for index, row in enumerate(candidates)
    ]
    frame = pd.concat(series, axis=1, join="inner").dropna(how="any")
    if len(frame) < 3:
        return [], pd.DataFrame()
    standard_deviations = frame.std(ddof=1)
    valid_columns = standard_deviations.index[
        standard_deviations.notna() & (standard_deviations > 0.0)
    ].tolist()
    if not valid_columns:
        return [], pd.DataFrame()
    return [candidates[int(column)] for column in valid_columns], frame[valid_columns]

def correlation_diagnostics(frame):
    """Return a numerically clean correlation matrix and pairwise summaries."""
    correlation = frame.corr().to_numpy(dtype=float)
    correlation = 0.5 * (correlation + correlation.T)
    np.fill_diagonal(correlation, 1.0)
    upper = correlation[np.triu_indices_from(correlation, k=1)]
    mean_correlation = float(np.mean(upper)) if upper.size else 1.0
    median_correlation = float(np.median(upper)) if upper.size else 1.0
    return correlation, mean_correlation, median_correlation

def simulate_correlated_maxima(correlation, simulations, seed):
    """Simulate max Gaussian test statistics with the supplied correlation."""
    count = correlation.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    factor = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    rng = np.random.default_rng(seed)
    maxima = np.empty(simulations, dtype=float)
    offset = 0
    while offset < simulations:
        remaining = simulations - offset
        pairs = min(2500, (remaining + 1) // 2)
        draws = rng.standard_normal((pairs, count)) @ factor.T
        positive = draws.max(axis=1)
        negative = (-draws).max(axis=1)
        paired = np.empty(2 * pairs, dtype=float)
        paired[0::2] = positive
        paired[1::2] = negative
        take = min(remaining, len(paired))
        maxima[offset:offset + take] = paired[:take]
        offset += take
    return maxima

def sharpe_sampling_stats(row, benchmark_sharpe=0.0):
    """Return PSR-style z score, probability, annualized SE, skew and kurtosis."""
    returns = pd.to_numeric(row["_returns"], errors="coerce").dropna()
    observations = len(returns)
    if observations < 3:
        return None
    annual_sharpe = float(row["sharpe"])
    periodic_sharpe = annual_sharpe / math.sqrt(_TRADING_DAYS_PER_YEAR)
    periodic_benchmark = benchmark_sharpe / math.sqrt(_TRADING_DAYS_PER_YEAR)
    skewness = float(returns.skew())
    kurtosis = float(returns.kurt()) + 3.0
    if not (math.isfinite(skewness) and math.isfinite(kurtosis)):
        return None
    variance_factor = (
        1.0 - skewness * periodic_sharpe
        + 0.25 * (kurtosis - 1.0) * periodic_sharpe * periodic_sharpe
    )
    if variance_factor <= 0.0 or not math.isfinite(variance_factor):
        return None
    standard_error_periodic = math.sqrt(variance_factor / (observations - 1.0))
    z_score = (periodic_sharpe - periodic_benchmark) / standard_error_periodic
    probability = _STANDARD_NORMAL.cdf(z_score)
    standard_error_annual = standard_error_periodic * math.sqrt(_TRADING_DAYS_PER_YEAR)
    return {
        "z": z_score,
        "probability": probability,
        "se_annual": standard_error_annual,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "observations": observations,
    }

def analyze_deflation_for_trade(rows, simulations, seed):
    """Estimate search multiplicity and selection-adjusted Sharpe diagnostics."""
    candidates, frame = aligned_strategy_returns(rows)
    if not candidates:
        return None
    correlation, mean_correlation, median_correlation = correlation_diagnostics(frame)
    raw_count = len(candidates)
    galwey_count = galwey_effective_tests(correlation)
    participation_count = participation_effective_tests(correlation)
    bailey_count = mean_correlation + (1.0 - mean_correlation) * raw_count
    bailey_count = min(max(bailey_count, 1.0), float(raw_count))
    maxima = simulate_correlated_maxima(correlation, simulations, seed)
    expected_max_z = float(maxima.mean())
    max_equivalent_count = equivalent_independent_trials(expected_max_z, raw_count)

    sharpes = np.array([float(row["sharpe"]) for row in candidates], dtype=float)
    sharpe_sd = float(np.std(sharpes, ddof=1)) if raw_count > 1 else 0.0
    bailey_expected_max_z = expected_max_standard_normal(bailey_count)
    selection_hurdle = sharpe_sd * bailey_expected_max_z
    max_correlation_hurdle = sharpe_sd * expected_max_z

    sampling = []
    for row in candidates:
        stats = sharpe_sampling_stats(row)
        if stats is None:
            sampling.append(None)
            continue
        row["_psr"] = stats["probability"]
        hurdle_stats = sharpe_sampling_stats(row, benchmark_sharpe=selection_hurdle)
        row["_dsr"] = hurdle_stats["probability"] if hurdle_stats else float("nan")
        max_hurdle_stats = sharpe_sampling_stats(
            row, benchmark_sharpe=max_correlation_hurdle
        )
        row["_dsr_max"] = (
            max_hurdle_stats["probability"] if max_hurdle_stats else float("nan")
        )
        row["_haircut_sharpe"] = float(row["sharpe"]) - selection_hurdle
        exceedances = int(np.count_nonzero(maxima >= stats["z"]))
        row["_max_test_p"] = (exceedances + 1.0) / (simulations + 1.0)
        sampling.append(stats)

    # Empirical-Bayes shrinkage: estimate how much of the cross-sectional Sharpe
    # dispersion remains after allowing for correlated Sharpe estimation error.
    if raw_count > 1 and all(stats is not None for stats in sampling):
        standard_errors = np.array([stats["se_annual"] for stats in sampling])
        error_covariance = np.outer(standard_errors, standard_errors) * correlation
        trace = float(np.trace(error_covariance))
        grand_covariance = float(error_covariance.sum()) / raw_count
        noise_cross_variance = max((trace - grand_covariance) / (raw_count - 1.0), 0.0)
        observed_cross_variance = float(np.var(sharpes, ddof=1))
        between_strategy_variance = max(observed_cross_variance - noise_cross_variance, 0.0)
        family_mean = float(np.mean(sharpes))
        marginal_variances = np.square(standard_errors)
        weights = between_strategy_variance / (between_strategy_variance + marginal_variances)
        posterior_sharpes = family_mean + weights * (sharpes - family_mean)
    else:
        noise_cross_variance = 0.0
        between_strategy_variance = 0.0
        family_mean = float(np.mean(sharpes))
        weights = np.zeros(raw_count, dtype=float)
        posterior_sharpes = np.full(raw_count, family_mean, dtype=float)

    for row, posterior, weight in zip(candidates, posterior_sharpes, weights):
        row["_eb_sharpe"] = float(posterior)
        row["_eb_weight"] = float(weight)

    best = max(candidates, key=lambda row: float(row["sharpe"]))
    return {
        "candidates": candidates,
        "frame": frame,
        "correlation": correlation,
        "raw_count": raw_count,
        "common_observations": len(frame),
        "mean_correlation": mean_correlation,
        "median_correlation": median_correlation,
        "bailey_count": bailey_count,
        "galwey_count": galwey_count,
        "participation_count": participation_count,
        "bailey_expected_max_z": bailey_expected_max_z,
        "expected_max_z": expected_max_z,
        "max_equivalent_count": max_equivalent_count,
        "sharpe_sd": sharpe_sd,
        "selection_hurdle": selection_hurdle,
        "max_correlation_hurdle": max_correlation_hurdle,
        "family_mean": family_mean,
        "noise_cross_variance": noise_cross_variance,
        "between_strategy_variance": between_strategy_variance,
        "best": best,
    }

def subset_galwey_count(rows):
    """Return raw and Galwey-effective counts for a candidate subset."""
    candidates, frame = aligned_strategy_returns(rows)
    if not candidates:
        return 0, float("nan")
    correlation, _, _ = correlation_diagnostics(frame)
    return len(candidates), galwey_effective_tests(correlation)

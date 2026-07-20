"""Estimation layer
EWMA std/covariance estimation and Ledoit-Wolf constant-correlation shrinkage.

Why shrinkage: with ~20 assets and a 3-year window the sample covariance is
noisy and its inverse (which MVO uses) amplifies that noise. Shrinkage trades a
little bias for a large variance reduction.

Reference:
    - Ledoit, O., & Wolf, M. (2004). Honey, I shrunk the sample covariance
      matrix. Journal of Portfolio Management, 30(4), 110-119.

Estimators are pure functions over a caller-supplied PIT return window.
`ewma_std` is shared with signal layer (`sigma_daily`) under an independent config knob.
"""

import numpy as np
import pandas as pd


def ewma_std(
    returns: pd.DataFrame, halflife: int, min_periods: int | None = None
) -> pd.DataFrame:
    """EWMA std of daily returns. NaN until `min_periods` , default to halflife."""
    if halflife <= 0:
        raise ValueError(f"halflife must be positive, got {halflife}")
    if min_periods is None:
        min_periods = halflife
    if min_periods <= 0:
        raise ValueError(f"min_periods must be positive, got {min_periods}")
    return returns.ewm(halflife=halflife, min_periods=min_periods).std()


def _validate_window(returns: pd.DataFrame) -> np.ndarray:
    """Fail-loud checks for a covariance estimation window."""
    if len(returns) < 2:
        raise ValueError(f"need at least 2 rows, got {len(returns)}")
    if returns.isna().any().any():
        raise ValueError("NaN in the return window; estimators never impute")
    return returns.to_numpy(dtype="float64")


def sample_cov(returns: pd.DataFrame) -> pd.DataFrame:
    """Maximum-likelihood sample covariance, divide by T per the L-W paper."""
    x = _validate_window(returns)
    demeaned = x - x.mean(axis=0)
    cov = demeaned.T @ demeaned / len(x)
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def ewma_cov(returns: pd.DataFrame, halflife: int) -> pd.DataFrame:
    """
    EWMA covariance of a return window. The last row carries most weight.
    Explicit-weights implementation with bias correction (bias=False).
    """
    if halflife <= 0:
        raise ValueError(f"halflife must be positive, got {halflife}")
    x = _validate_window(returns)
    n_rows = len(x)
    age = np.arange(n_rows - 1, -1, -1, dtype="float64")
    weights = 0.5 ** (age / halflife)
    wsum = weights.sum()
    mean = weights @ x / wsum
    demeaned = x - mean
    raw = (weights[:, None] * demeaned).T @ demeaned
    denom = wsum - (weights**2).sum() / wsum  # bias=False correction
    cov = raw / denom
    cov = (cov + cov.T) / 2.0
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def ledoit_wolf_cc(returns: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Ledoit-Wolf shrinkage toward the constant-correlation target.

    The sample covariance is noisy when observations are scarce relative to the
    number of assets (T << N). This pulls it toward a simpler, steadier target
    and returns (shrunk covariance, shrinkage intensity delta).

    The shrunk estimate is a weighted blend, `delta * target + (1 - delta) *
    sample`, where the sample covariance is the maximum-likelihood (ML)
    estimate, i.e. divided by T rather than the usual unbiased T - 1. The target
    keeps each asset's sample variance but replaces every pairwise correlation
    with the single average correlation `r_bar` across all pairs.

    delta in [0, 1] is chosen by the paper's formula to minimize expected
    distance to the true covariance: shrink more when the sample is noisy (large
    pi_hat) and when the target is a good fit (small gamma_hat), less when
    shrinking would throw away real structure. It is estimated as delta =
    clip(kappa / T, 0, 1) with kappa = (pi_hat - rho_hat) / gamma_hat.
    """
    x = _validate_window(returns)
    n_rows, n_assets = x.shape
    if n_assets < 2:
        raise ValueError(f"need at least 2 assets, got {n_assets}")

    demeaned = x - x.mean(axis=0)
    sample = demeaned.T @ demeaned / n_rows
    var = np.diag(sample).copy()
    if (var <= 0).any():
        raise ValueError("degenerate asset with non-positive variance")
    sd = np.sqrt(var)

    # constant-correlation target F
    corr = sample / np.outer(sd, sd)
    # r_bar is the mean of the upper triangular (excluding the diagonal)
    # elements of the correlation matrix:
    # iu = np.triu_indices(n_assets, k=1)
    # r_bar = corr[iu].mean()
    #
    # Alternative computation of r_bar using the sum of all correlations minus
    # the diagonal for performance reasons.
    r_bar = (corr.sum() - n_assets) / (n_assets * (n_assets - 1))
    target = r_bar * np.outer(sd, sd)
    np.fill_diagonal(target, var)

    # pi_hat: sum of asymptotic variances of the sample covariances
    # The π_{ij} quadratic expansion is given by:
    #   \frac{1}{T}\sum_t (d_{ti}d_{tj} - s_{ij})^2
    # = \frac{1}{T}\sum_t d_{ti}^2 d_{tj}^2} - 2 s_{ij}\underbrace{\frac{1}{T}\sum_t d_{ti}d_{tj}}_{=\,s_{ij}} + s_{ij}^2  # noqa: E501
    # = \frac{1}{T}\sum_t d_{ti}^2 d_{tj}^2} - s_{ij}^2
    sq = demeaned * demeaned
    pi_mat = sq.T @ sq / n_rows - sample * sample
    pi_hat = pi_mat.sum()

    # rho_hat: sum of asymptotic covariances between sample and target
    # theta[i, j] = AsyCov[s_ii, s_ij]. Its transpose gives AsyCov[s_jj, s_ij].
    # The quadratic expansion for the asymptotic covariance between s_ii and s_ij:
    # \frac{1}{T}\sum_t (d_{ti}^2 - s_{ii})(d_{ti}d_{tj} - s_{ij}) = \frac{1}{T}\sum_t d_{ti}^2 d_{ti}d_{tj} -  s_{ii}s_{ij}  # noqa: E501
    theta = (sq * demeaned).T @ demeaned / n_rows - var[:, None] * sample
    # ratio shape (N,N), ratio[i,j] = sd_j / sd_i = sqrt(s_jj/s_ii) at [i, j]
    ratio = sd[None, :] / sd[:, None]
    off = ~np.eye(n_assets, dtype=bool)
    rho_hat = np.trace(pi_mat) + (r_bar / 2.0) * (
        (ratio * theta)[off].sum() + (ratio.T * theta.T)[off].sum()
    )

    # gamma_hat: squared Frobenius distance between target and sample
    gamma_hat = ((target - sample) ** 2).sum()

    if gamma_hat <= 0.0:  # sample already has exactly constant correlation
        delta = 0.0
    else:
        kappa = (pi_hat - rho_hat) / gamma_hat
        delta = float(min(1.0, max(0.0, kappa / n_rows)))

    shrunk = delta * target + (1.0 - delta) * sample
    shrunk = (shrunk + shrunk.T) / 2.0
    frame = pd.DataFrame(shrunk, index=returns.columns, columns=returns.columns)
    return frame, delta

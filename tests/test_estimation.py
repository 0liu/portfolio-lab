"""Estimation tests

Scikit-learn's LedoitWolf shrinks toward scaled identity matrix. It's a
different target AND shrinkage intensity (delta) formula, so it cannot be an
oracle for the constant-correlation targeted estimator. Instead:
    1. The maximum-likelihood (ML) sample covariance is checked exactly against
       sklearn and numpy.
    2. The vectorized Ledoit-Wolf is checked against a literal loop
       transcription of the paper's formulas.
    3. Property suite: symmetry, PSD, diagonal preservation, delta in [0,
       1],convex-combination correlations, less shrinkage with more data.
    4. On a known-Sigma DGP both our estimator and sklearn's must beat the
       sample covariance.
    5. ewma_cov is checked against pandas' independent recursion.
"""

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import LedoitWolf, empirical_covariance

from portlab.estimation import ewma_cov, ewma_std, ledoit_wolf_cc, sample_cov


def make_returns(n_rows: int, n_assets: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 0.01, size=(n_rows, n_assets))
    idx = pd.bdate_range("2016-01-04", periods=n_rows, name="date")
    return pd.DataFrame(data, index=idx, columns=[f"A{i}" for i in range(n_assets)])


def cc_dgp(n_rows: int, n_assets: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Known-truth DGP: constant correlation 0.3, vols from 0.5% to 2%."""
    rng = np.random.default_rng(seed)
    vols = np.linspace(0.005, 0.02, n_assets)
    corr = np.full((n_assets, n_assets), 0.3)
    np.fill_diagonal(corr, 1.0)
    true_cov = corr * np.outer(vols, vols)
    data = rng.multivariate_normal(np.zeros(n_assets), true_cov, size=n_rows)
    idx = pd.bdate_range("2016-01-04", periods=n_rows, name="date")
    frame = pd.DataFrame(data, index=idx, columns=[f"A{i}" for i in range(n_assets)])
    return frame, true_cov


def paper_reference(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Literal loop transcription of Ledoit-Wolf (2004) appendix."""
    n_rows, n = x.shape
    d = x - x.mean(axis=0)
    s = d.T @ d / n_rows
    sd = np.sqrt(np.diag(s))
    corr = s / np.outer(sd, sd)
    r_bar = (corr.sum() - n) / (n * (n - 1))
    f = r_bar * np.outer(sd, sd)
    np.fill_diagonal(f, np.diag(s))

    pi_hat = 0.0
    for i in range(n):
        for j in range(n):
            pi_hat += np.mean((d[:, i] * d[:, j] - s[i, j]) ** 2)

    rho_hat = 0.0
    for i in range(n):
        rho_hat += np.mean((d[:, i] ** 2 - s[i, i]) ** 2)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            theta_ii = np.mean((d[:, i] ** 2 - s[i, i]) * (d[:, i] * d[:, j] - s[i, j]))
            theta_jj = np.mean((d[:, j] ** 2 - s[j, j]) * (d[:, i] * d[:, j] - s[i, j]))
            rho_hat += (r_bar / 2.0) * (
                math.sqrt(s[j, j] / s[i, i]) * theta_ii
                + math.sqrt(s[i, i] / s[j, j]) * theta_jj
            )

    gamma_hat = ((f - s) ** 2).sum()
    kappa = (pi_hat - rho_hat) / gamma_hat
    delta = min(1.0, max(0.0, kappa / n_rows))
    return delta * f + (1.0 - delta) * s, delta


# ---------------------------------------------------------------- sample_cov


def test_sample_cov_matches_sklearn_and_numpy():
    returns = make_returns(200, 5)
    ours = sample_cov(returns).to_numpy()
    np.testing.assert_allclose(ours, empirical_covariance(returns.to_numpy()))
    np.testing.assert_allclose(ours, np.cov(returns.to_numpy().T, ddof=0))


# ------------------------------------------------------------------ ewma_cov


def test_ewma_cov_matches_pandas_recursion():
    returns = make_returns(150, 4, seed=1)
    ours = ewma_cov(returns, halflife=10)
    pandas_cov = returns.ewm(halflife=10).cov().loc[returns.index[-1]]
    np.testing.assert_allclose(ours.to_numpy(), pandas_cov.to_numpy(), rtol=1e-10)


def test_ewma_cov_limits_to_sample_cov_for_huge_halflife():
    returns = make_returns(120, 3, seed=2)
    ours = ewma_cov(returns, halflife=10_000_000)
    flat = np.cov(returns.to_numpy().T, ddof=1)
    np.testing.assert_allclose(ours.to_numpy(), flat, rtol=1e-4)


def test_ewma_cov_is_symmetric_psd_and_labeled():
    returns = make_returns(150, 6, seed=3)
    cov = ewma_cov(returns, halflife=20)
    assert list(cov.index) == list(returns.columns)
    assert cov.equals(cov.T)
    assert np.linalg.eigvalsh(cov.to_numpy()).min() >= -1e-12


def test_ewma_cov_recent_data_dominates():
    calm = make_returns(100, 2, seed=4) * 0.1
    wild = make_returns(30, 2, seed=5) * 3.0
    returns = pd.concat([calm, wild * 1.0]).reset_index(drop=True)
    returns.index = pd.bdate_range("2016-01-04", periods=len(returns))
    short = ewma_cov(returns, halflife=5)  # sees mostly the wild tail
    long = ewma_cov(returns, halflife=500)  # averages over the calm past
    assert short.iloc[0, 0] > long.iloc[0, 0]


# ------------------------------------------------------------- ledoit_wolf_cc


def test_ledoit_wolf_matches_paper_loop_transcription():
    returns = make_returns(60, 5, seed=6)
    ours, delta = ledoit_wolf_cc(returns)
    ref_cov, ref_delta = paper_reference(returns.to_numpy())
    assert delta == pytest.approx(ref_delta, abs=1e-12)
    np.testing.assert_allclose(ours.to_numpy(), ref_cov, rtol=1e-10)
    assert 0.0 < delta < 1.0  # interior solution: the case actually shrinks


def test_ledoit_wolf_properties():
    returns = make_returns(80, 6, seed=7)
    shrunk, delta = ledoit_wolf_cc(returns)
    sample = sample_cov(returns)
    assert 0.0 <= delta <= 1.0
    assert shrunk.equals(shrunk.T)
    assert np.linalg.eigvalsh(shrunk.to_numpy()).min() >= -1e-12
    # the CC target keeps sample variances -> diagonal is preserved exactly
    np.testing.assert_allclose(np.diag(shrunk), np.diag(sample), rtol=1e-12)


def test_ledoit_wolf_correlations_are_convex_toward_rbar():
    returns = make_returns(80, 5, seed=8)
    shrunk, delta = ledoit_wolf_cc(returns)
    s = sample_cov(returns).to_numpy()
    sd = np.sqrt(np.diag(s))
    sample_corr = s / np.outer(sd, sd)
    n = len(sd)
    r_bar = (sample_corr.sum() - n) / (n * (n - 1))
    shrunk_corr = shrunk.to_numpy() / np.outer(sd, sd)
    off = ~np.eye(n, dtype=bool)
    expected = delta * r_bar + (1.0 - delta) * sample_corr[off]
    np.testing.assert_allclose(shrunk_corr[off], expected, rtol=1e-10)


def toeplitz_dgp(n_rows: int, n_assets: int, seed: int) -> pd.DataFrame:
    """Known-truth DGP whose correlation is NOT constant (0.6^|i-j| Toeplitz),
    so the CC target is misspecified and delta must vanish as T grows."""
    rng = np.random.default_rng(seed)
    idx_grid = np.arange(n_assets)
    corr = 0.6 ** np.abs(idx_grid[:, None] - idx_grid[None, :])
    vols = np.linspace(0.005, 0.02, n_assets)
    true_cov = corr * np.outer(vols, vols)
    data = rng.multivariate_normal(np.zeros(n_assets), true_cov, size=n_rows)
    idx = pd.bdate_range("2016-01-04", periods=n_rows, name="date")
    return pd.DataFrame(data, index=idx, columns=[f"A{i}" for i in range(n_assets)])


def test_ledoit_wolf_shrinks_less_with_more_data_under_misspecified_target():
    _, delta_small = ledoit_wolf_cc(toeplitz_dgp(50, 8, seed=9))
    _, delta_large = ledoit_wolf_cc(toeplitz_dgp(5000, 8, seed=9))
    assert delta_small > delta_large
    assert delta_large < 0.1  # vanishing: the data overrules a wrong target


def test_ledoit_wolf_and_sklearn_both_beat_sample_cov():
    returns, true_cov = cc_dgp(40, 10, seed=10)
    shrunk, _ = ledoit_wolf_cc(returns)
    sample = sample_cov(returns).to_numpy()

    def frob(a: np.ndarray) -> float:
        return float(np.sqrt(((a - true_cov) ** 2).sum()))

    assert frob(shrunk.to_numpy()) < frob(sample)
    sk = LedoitWolf().fit(returns.to_numpy()).covariance_
    assert frob(sk) < frob(sample)


# ----------------------------------------------------------------- fail loud


def test_estimators_reject_nan():
    returns = make_returns(50, 3)
    returns.iloc[10, 1] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        ewma_cov(returns, halflife=10)
    with pytest.raises(ValueError, match="NaN"):
        ledoit_wolf_cc(returns)
    with pytest.raises(ValueError, match="NaN"):
        sample_cov(returns)


def test_estimators_reject_tiny_windows():
    returns = make_returns(1, 3)
    with pytest.raises(ValueError, match="at least 2 rows"):
        ewma_cov(returns, halflife=10)
    with pytest.raises(ValueError, match="at least 2 rows"):
        ledoit_wolf_cc(returns)


def test_ledoit_wolf_rejects_single_asset():
    returns = make_returns(50, 1)
    with pytest.raises(ValueError, match="at least 2 assets"):
        ledoit_wolf_cc(returns)


def test_ledoit_wolf_rejects_degenerate_variance():
    returns = make_returns(50, 3)
    returns["A1"] = 0.0
    with pytest.raises(ValueError, match="variance"):
        ledoit_wolf_cc(returns)


def test_ewma_cov_rejects_bad_halflife():
    with pytest.raises(ValueError, match="positive"):
        ewma_cov(make_returns(50, 3), halflife=0)


def test_ewma_std_rejects_bad_params():
    returns = make_returns(50, 3)
    with pytest.raises(ValueError, match="positive"):
        ewma_std(returns, halflife=0)
    with pytest.raises(ValueError, match="positive"):
        ewma_std(returns, halflife=5, min_periods=0)

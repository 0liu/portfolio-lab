"""Construction tests for the simple optimizers.

Contracts: fully invested, long-only, ERC contributions equal, and the analytic
special cases ERC must reproduce (2-asset and diagonal cases collapse to
inverse-vol).
"""

import numpy as np
import pandas as pd
import pytest

from portlab.config import Config
from portlab.construction import (
    OPTIMIZER_NAMES,
    equal_weight,
    erc,
    inverse_vol,
    optimize,
)

CFG = Config()


def make_cov(vols: list[float], corr: float = 0.3) -> pd.DataFrame:
    n = len(vols)
    matrix = np.full((n, n), corr)
    np.fill_diagonal(matrix, 1.0)
    cov = matrix * np.outer(vols, vols)
    tickers = [f"A{i}" for i in range(n)]
    return pd.DataFrame(cov, index=tickers, columns=tickers)


def zero_mu(cov: pd.DataFrame) -> pd.Series:
    return pd.Series(0.0, index=cov.index)


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> np.ndarray:
    w = weights.to_numpy()
    marginal = cov.to_numpy() @ w
    return w * marginal / (w @ marginal)


# ----------------------------------------------------------- shared contracts


@pytest.mark.parametrize("name", OPTIMIZER_NAMES)
def test_fully_invested_long_only_and_labeled(name):
    cov = make_cov([0.01, 0.02, 0.015, 0.03])
    weights = optimize(name, zero_mu(cov), cov, None, CFG)
    assert list(weights.index) == list(cov.index)
    assert weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert (weights >= 0).all()


@pytest.mark.parametrize("name", OPTIMIZER_NAMES)
def test_mu_is_ignored_by_simple_optimizers(name):
    cov = make_cov([0.01, 0.02, 0.03])
    bullish = pd.Series([0.5, -0.5, 0.1], index=cov.index)
    a = optimize(name, zero_mu(cov), cov, None, CFG)
    b = optimize(name, bullish, cov, None, CFG)
    pd.testing.assert_series_equal(a, b)


def test_unknown_optimizer_raises_with_available_names():
    cov = make_cov([0.01, 0.02])
    with pytest.raises(ValueError, match="equal_weight"):
        optimize("magic", zero_mu(cov), cov, None, CFG)


# ------------------------------------------------------------------ EW / IVP


def test_equal_weight_is_exactly_one_over_n():
    cov = make_cov([0.01, 0.02, 0.03, 0.04])
    weights = equal_weight(zero_mu(cov), cov, None, CFG)
    assert weights.tolist() == [0.25, 0.25, 0.25, 0.25]


def test_inverse_vol_hand_computed():
    cov = make_cov([0.01, 0.02, 0.04])
    weights = inverse_vol(zero_mu(cov), cov, None, CFG)
    # 1/sigma = (100, 50, 25) -> weights (4/7, 2/7, 1/7)
    assert weights.tolist() == pytest.approx([4 / 7, 2 / 7, 1 / 7])


# ----------------------------------------------------------------------- ERC


def test_erc_contributions_are_equal():
    cov = make_cov([0.008, 0.012, 0.02, 0.03, 0.015], corr=0.4)
    weights = erc(zero_mu(cov), cov, None, CFG)
    rc = risk_contributions(weights, cov)
    assert np.abs(rc - 1 / len(cov)).max() < 1e-6


def test_erc_two_assets_is_inverse_vol_for_any_correlation():
    # classic analytic result: for n=2, ERC weights are 1/sigma regardless of rho
    for corr in (-0.5, 0.0, 0.8):
        cov = make_cov([0.01, 0.03], corr=corr)
        weights = erc(zero_mu(cov), cov, None, CFG)
        assert weights.tolist() == pytest.approx([0.75, 0.25], abs=1e-7)


def test_erc_diagonal_cov_is_inverse_vol():
    cov = make_cov([0.01, 0.02, 0.04], corr=0.0)
    expected = inverse_vol(zero_mu(cov), cov, None, CFG)
    weights = erc(zero_mu(cov), cov, None, CFG)
    assert weights.tolist() == pytest.approx(expected.tolist(), abs=1e-7)


def test_erc_identical_assets_get_equal_weights():
    cov = make_cov([0.02, 0.02, 0.02, 0.02], corr=0.5)
    weights = erc(zero_mu(cov), cov, None, CFG)
    assert weights.tolist() == pytest.approx([0.25] * 4, abs=1e-7)


def test_erc_is_scale_invariant():
    cov = make_cov([0.01, 0.02, 0.03], corr=0.25)
    a = erc(zero_mu(cov), cov, None, CFG)
    b = erc(zero_mu(cov * 1e4), cov * 1e4, None, CFG)
    assert a.tolist() == pytest.approx(b.tolist(), abs=1e-8)


def test_erc_single_asset_trivial():
    cov = make_cov([0.02])
    weights = erc(zero_mu(cov), cov, None, CFG)
    assert weights.tolist() == [1.0]


# ------------------------------------------------------------------ fail loud


def test_nan_cov_raises():
    cov = make_cov([0.01, 0.02])
    cov.iloc[0, 1] = float("nan")
    with pytest.raises(ValueError, match="NaN in cov"):
        optimize("equal_weight", zero_mu(cov), cov, None, CFG)


def test_asymmetric_cov_raises():
    cov = make_cov([0.01, 0.02])
    cov.iloc[0, 1] = cov.iloc[0, 1] * 2
    with pytest.raises(ValueError, match="not symmetric"):
        optimize("erc", zero_mu(cov), cov, None, CFG)


def test_nonpositive_variance_raises():
    cov = make_cov([0.01, 0.02])
    cov.iloc[1, 1] = 0.0
    with pytest.raises(ValueError, match="non-positive variance"):
        optimize("inverse_vol", zero_mu(cov), cov, None, CFG)


def test_misaligned_mu_raises():
    cov = make_cov([0.01, 0.02])
    mu = pd.Series([0.0, 0.0], index=["B0", "B1"])
    with pytest.raises(ValueError, match="mu is not aligned"):
        optimize("equal_weight", mu, cov, None, CFG)


def test_misaligned_w_prev_raises():
    cov = make_cov([0.01, 0.02])
    w_prev = pd.Series([0.5, 0.5], index=["B0", "B1"])
    with pytest.raises(ValueError, match="w_prev is not aligned"):
        optimize("equal_weight", zero_mu(cov), cov, w_prev, CFG)

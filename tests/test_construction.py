"""Construction contract tests (C11).

Simple optimizers: fully invested, long-only, ERC contributions equal, and
the analytic special cases ERC must reproduce. MVO family: caps, gross/net
bands, shorts actually happen, lambda up => turnover down, the mu-scale /
risk-aversion tradeoff, and the volatility ordering
sigma(capped min-var) <= sigma(ERC) <= sigma(EW). Vol-target overlay: hits
the target or the gross cap, whichever binds first.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from portlab.config import Config, ConstructionConfig
from portlab.construction import (
    OPTIMIZER_NAMES,
    SIMPLE_NAMES,
    TRADING_DAYS,
    equal_weight,
    erc,
    inverse_vol,
    mvo,
    mvo_ls,
    optimize,
    turnover,
    vol_target,
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


@pytest.mark.parametrize("name", SIMPLE_NAMES)
def test_fully_invested_long_only_and_labeled(name):
    cov = make_cov([0.01, 0.02, 0.015, 0.03])
    weights = optimize(name, zero_mu(cov), cov, None, CFG)
    assert list(weights.index) == list(cov.index)
    assert weights.sum() == pytest.approx(1.0, abs=1e-9)
    assert (weights >= 0).all()


@pytest.mark.parametrize("name", SIMPLE_NAMES)
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


# ------------------------------------------------------------------ helpers


def portfolio_vol(weights: pd.Series, cov: pd.DataFrame) -> float:
    w = weights.to_numpy()
    return float(np.sqrt(w @ cov.to_numpy() @ w))


def cfg_with(**kwargs) -> Config:
    return Config(construction=ConstructionConfig(**kwargs))


# -------------------------------------------------------------- mvo (long)


def test_mvo_is_fully_invested_long_only_and_capped():
    cov = make_cov([0.01, 0.015, 0.02, 0.025, 0.03, 0.012])
    mu = pd.Series([0.01, -0.005, 0.002, 0.0, 0.008, -0.001], index=cov.index)
    weights = optimize("mvo", mu, cov, None, CFG)
    assert weights.sum() == pytest.approx(1.0, abs=1e-8)
    assert (weights >= -1e-12).all()
    assert (weights <= 0.25 + 1e-8).all()


def test_mvo_strong_mu_hits_the_cap():
    cov = make_cov([0.01, 0.01, 0.01, 0.01, 0.01])
    mu = pd.Series([0.05, 0.0, 0.0, 0.0, 0.0], index=cov.index)
    weights = mvo(mu, cov, None, CFG)
    assert weights.iloc[0] == pytest.approx(0.25, abs=1e-6)


def test_mvo_infeasible_cap_raises():
    cov = make_cov([0.01, 0.02])  # 2 assets * 0.25 < 1
    with pytest.raises(ValueError, match="infeasible"):
        mvo(zero_mu(cov), cov, None, CFG)


def test_volatility_ordering_minvar_erc_ew():
    # sigma(capped min-var) <= sigma(ERC) <= sigma(EW)  (MRT 2010 + optimality)
    cov = make_cov([0.010, 0.011, 0.012, 0.013, 0.014, 0.015], corr=0.3)
    ew = equal_weight(zero_mu(cov), cov, None, CFG)
    rp = erc(zero_mu(cov), cov, None, CFG)
    # precondition: ERC lies inside the capped feasible set, else no theorem
    assert (rp <= 0.25 + 1e-9).all()
    minvar = mvo(zero_mu(cov), cov, None, CFG)  # mu=0, lambda=0: capped min-var
    assert portfolio_vol(minvar, cov) <= portfolio_vol(rp, cov) + 1e-12
    assert portfolio_vol(rp, cov) <= portfolio_vol(ew, cov) + 1e-12


# ------------------------------------------------------------------- mvo_ls


def test_mvo_ls_respects_position_gross_and_net():
    cov = make_cov([0.01, 0.015, 0.02, 0.025, 0.03, 0.012])
    mu = pd.Series([0.02, -0.02, 0.01, -0.01, 0.015, 0.0], index=cov.index)
    weights = mvo_ls(mu, cov, None, CFG)
    assert weights.abs().max() <= 0.25 + 1e-8
    assert weights.abs().sum() <= 2.0 + 1e-8
    assert -0.5 - 1e-8 <= weights.sum() <= 1.0 + 1e-8


def test_mvo_ls_actually_shorts():
    cov = make_cov([0.01, 0.01, 0.01, 0.01])
    mu = pd.Series([-0.02, 0.02, 0.0, 0.0], index=cov.index)
    weights = mvo_ls(mu, cov, None, CFG)
    assert weights.iloc[0] < -0.01  # negative view -> genuine short


def test_mvo_ls_never_fully_invested_constraint_is_absent():
    cov = make_cov([0.01, 0.02, 0.03])
    mu = pd.Series([0.001, 0.0, -0.001], index=cov.index)
    weights = mvo_ls(mu, cov, None, CFG)
    assert weights.sum() != pytest.approx(1.0, abs=1e-3)  # cash is allowed


@pytest.mark.parametrize("name", ["mvo", "mvo_ls"])
def test_lambda_up_turnover_down(name):
    cov = make_cov([0.01, 0.014, 0.02, 0.024, 0.03], corr=0.35)
    mu = pd.Series([0.012, -0.008, 0.004, 0.009, -0.003], index=cov.index)
    # sum=1, each 0.2 <= cap, gross=1 <= 2, net=1 <= net_max: feasible for both
    w_prev = pd.Series(0.2, index=cov.index)
    turnovers = []
    for lam in (0.0, 1e-4, 1e-3, 1e-2, 10.0):
        weights = optimize(name, mu, cov, w_prev, cfg_with(turnover_lambda=lam))
        turnovers.append(turnover(weights, w_prev))
    for lo, hi in zip(turnovers[1:], turnovers[:-1], strict=True):
        assert lo <= hi + 1e-5  # non-increasing up to solver tolerance
    assert turnovers[-1] < 1e-4  # huge lambda pins the book to w_prev


def test_mu_scale_trades_off_against_risk_aversion():
    # doubling mu and doubling gamma leaves the solution unchanged (lambda=0)
    cov = make_cov([0.01, 0.02, 0.03, 0.015])
    mu = pd.Series([0.01, -0.01, 0.005, 0.0], index=cov.index)
    a = mvo_ls(mu, cov, None, cfg_with(risk_aversion=100.0))
    b = mvo_ls(mu * 2, cov, None, cfg_with(risk_aversion=200.0))
    assert a.tolist() == pytest.approx(b.tolist(), abs=1e-6)


# --------------------------------------------------------------- vol target


def test_vol_target_hits_the_annualized_target():
    cov = make_cov([0.01, 0.015, 0.02, 0.012])
    weights = equal_weight(zero_mu(cov), cov, None, CFG)
    scaled = vol_target(weights, cov, CFG)
    realized = portfolio_vol(scaled, cov) * np.sqrt(TRADING_DAYS)
    assert realized == pytest.approx(0.10, rel=1e-9)


def test_vol_target_is_capped_by_gross_limit():
    cov = make_cov([0.0001, 0.0001])  # nearly riskless book wants huge leverage
    weights = pd.Series([0.5, 0.5], index=cov.index)
    scaled = vol_target(weights, cov, CFG)
    assert scaled.abs().sum() == pytest.approx(2.0, rel=1e-9)  # gross cap binds
    assert portfolio_vol(scaled, cov) * np.sqrt(TRADING_DAYS) < 0.10


def test_vol_target_rejects_zero_book():
    cov = make_cov([0.01, 0.02])
    weights = pd.Series([0.0, 0.0], index=cov.index)
    with pytest.raises(ValueError, match="zero-risk"):
        vol_target(weights, cov, CFG)


# ----------------------------------------------------------------- turnover


def test_turnover_hand_computed():
    idx = ["A0", "A1"]
    w = pd.Series([0.6, 0.4], index=idx)
    prev = pd.Series([0.5, 0.5], index=idx)
    assert turnover(w, prev) == pytest.approx(0.2)
    assert turnover(w, None) == pytest.approx(1.0)


def test_optimizer_registry_contains_all_five():
    assert OPTIMIZER_NAMES == ("equal_weight", "erc", "inverse_vol", "mvo", "mvo_ls")


def test_construction_config_defaults_are_the_spec():
    ccfg = ConstructionConfig()
    assert ccfg.position_cap == 0.25
    assert ccfg.gross_cap == 2.0
    assert ccfg.net_min == -0.5
    assert ccfg.net_max == 1.0
    assert ccfg.turnover_lambda == 0.0
    assert ccfg.vol_target_annual == 0.10


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        pytest.param({"position_cap": 0.0}, "positive", id="zero-cap"),
        pytest.param({"gross_cap": 0.1}, "gross_cap", id="gross-below-cap"),
        pytest.param({"net_min": 1.5, "net_max": 1.0}, "net_min", id="net-inverted"),
        pytest.param({"net_max": 3.0}, "net band", id="net-outside-gross"),
        pytest.param({"risk_aversion": 0.0}, "positive", id="zero-gamma"),
        pytest.param({"turnover_lambda": -1.0}, "non-negative", id="negative-lambda"),
        pytest.param({"vol_target_annual": 0.0}, "positive", id="zero-vol-target"),
    ],
)
def test_invalid_construction_config_raises(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ConstructionConfig(**kwargs)


def test_unused_replace_variant_import():  # keep ruff honest about `replace`
    assert replace(ConstructionConfig(), gross_cap=3.0).gross_cap == 3.0

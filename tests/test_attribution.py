"""Attribution tests

- Hand-computed statistics
- The Euler risk-decomposition identities
- A cross-module check that ERC weights produce equal percentage risk contributions
- The lambda-sweep driver on a small synthetic panel
"""

import numpy as np
import pandas as pd
import pytest

from portlab.attribution import (
    STAT_NAMES,
    comparison_table,
    component_risk_contributions,
    efficient_frontier,
    equity_curve,
    marginal_risk_contributions,
    market_beta,
    max_drawdown,
    pct_risk_contributions,
    performance_stats,
    sweep_lambda,
)
from portlab.config import (
    Config,
    EngineConfig,
    EstimationConfig,
    SignalConfig,
)
from portlab.construction import erc
from portlab.engine import run_backtest


def series(values: list[float]) -> pd.Series:
    idx = pd.bdate_range("2016-01-04", periods=len(values), name="date")
    return pd.Series(values, index=idx, name="net")


def make_cov(vols: list[float], corr: float = 0.3) -> pd.DataFrame:
    n = len(vols)
    matrix = np.full((n, n), corr)
    np.fill_diagonal(matrix, 1.0)
    cov = matrix * np.outer(vols, vols)
    tickers = [f"A{i}" for i in range(n)]
    return pd.DataFrame(cov, index=tickers, columns=tickers)


def random_closes(n: int = 110, k: int = 4, seed: int = 33) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.01, size=(n, k))
    idx = pd.bdate_range("2016-01-04", periods=n, name="date")
    return pd.DataFrame(
        100.0 * np.cumprod(1.0 + rets, axis=0),
        index=idx,
        columns=[f"A{i}" for i in range(k)],
    )


def small_cfg(freq: str = "W-FRI") -> Config:
    return Config(
        signals=SignalConfig(
            tsmom_windows=(2, 5),
            signal_vol_halflife_days=3,
            xs_lookback=8,
            xs_exclude=2,
            reversal_window=3,
        ),
        estimation=EstimationConfig(cov_window_days=15),
        engine=EngineConfig(rebalance_freq=freq),
    )


# ----------------------------------------------------------------- equity


def test_equity_and_max_drawdown_hand_computed():
    net = series([0.10, -0.20, 0.05])
    equity = equity_curve(net)
    assert equity.tolist() == pytest.approx([1.10, 0.88, 0.924])
    # peaks: [1.1, 1.1, 1.1]; drawdowns: [0, -0.2, -0.16]
    assert max_drawdown(net) == pytest.approx(-0.20)


def test_performance_stats_hand_computed():
    # one exact year of alternating +1% / -1%
    values = [0.01, -0.01] * 126
    net = series(values)
    result = _fake_result(net)
    stats = performance_stats(result)
    assert list(stats.index) == list(STAT_NAMES)
    assert stats["cagr"] == pytest.approx((1.01 * 0.99) ** 126 - 1.0, rel=1e-12)
    expected_std = pd.Series(values).std(ddof=1)
    assert stats["ann_vol"] == pytest.approx(expected_std * np.sqrt(252), rel=1e-12)
    assert stats["sharpe"] == pytest.approx(0.0, abs=1e-12)
    assert stats["sortino"] == pytest.approx(0.0, abs=1e-12)  # mean is zero
    # calmar is self-consistently cagr / |max_drawdown|
    assert stats["calmar"] == pytest.approx(stats["cagr"] / abs(stats["max_drawdown"]))
    assert stats["ann_turnover"] == pytest.approx(1.5)  # 1.5 over exactly 1 year
    assert stats["cost_drag"] == pytest.approx(1.5 * 5e-4)


def test_sortino_hand_computed_asymmetric():
    values = [0.02, -0.01] * 20
    stats = performance_stats(_fake_result(series(values)))
    mean = np.mean(values)
    downside = np.sqrt(np.mean(np.minimum(values, 0.0) ** 2))
    assert stats["sortino"] == pytest.approx(mean / downside * np.sqrt(252), rel=1e-12)
    assert stats["sortino"] > stats["sharpe"]  # upside vol not penalized


def _fake_result(net: pd.Series):
    from portlab.engine import BacktestResult

    turnover = pd.Series([1.0, 0.5], index=net.index[:2], name="turnover")
    costs = pd.Series(0.0, index=net.index, name="cost")
    costs.iloc[:2] = turnover.to_numpy() * 5e-4
    weights = pd.DataFrame(0.5, index=net.index, columns=["A0", "A1"])
    return BacktestResult(
        net_returns=net,
        gross_returns=net + costs,
        costs=costs,
        turnover=turnover,
        weights=weights,
    )


def test_constant_returns_have_nan_sharpe():
    stats = performance_stats(_fake_result(series([0.001] * 10)))
    assert np.isnan(stats["sharpe"])
    assert np.isnan(stats["sortino"])  # no downside observations at all
    assert np.isnan(stats["calmar"])  # never under water


def test_comparison_table_rows_match_stats():
    closes = random_closes()
    cfg = small_cfg()
    results = {
        name: run_backtest(closes, name, cfg) for name in ("equal_weight", "erc")
    }
    table = comparison_table(results)
    assert list(table.index) == ["equal_weight", "erc"]
    assert list(table.columns) == list(STAT_NAMES)
    pd.testing.assert_series_equal(
        table.loc["erc"], performance_stats(results["erc"]), check_names=False
    )


def test_comparison_table_empty_raises():
    with pytest.raises(ValueError, match="no results"):
        comparison_table({})


# ----------------------------------------------------------- decomposition


def test_ccr_sums_to_portfolio_sigma_and_pct_to_one():
    cov = make_cov([0.01, 0.015, 0.02, 0.012], corr=0.4)
    weights = pd.Series([0.4, 0.3, 0.2, 0.1], index=cov.index)
    sigma = float(np.sqrt(weights.to_numpy() @ cov.to_numpy() @ weights.to_numpy()))
    ccr = component_risk_contributions(weights, cov)
    assert float(ccr.sum()) == pytest.approx(sigma, rel=1e-12)  # Euler identity
    assert float(pct_risk_contributions(weights, cov).sum()) == pytest.approx(1.0)


def test_mcr_ccr_hand_computed_diagonal_case():
    cov = make_cov([0.01, 0.02], corr=0.0)
    weights = pd.Series([0.6, 0.4], index=cov.index)
    # sigma^2 = 0.36e-4 + 0.64e-4; MCR_i = w_i sigma_i^2 / sigma
    sigma = np.sqrt(0.36 * 1e-4 + 0.16 * 4e-4)
    mcr = marginal_risk_contributions(weights, cov)
    assert mcr["A0"] == pytest.approx(0.6 * 1e-4 / sigma, rel=1e-12)
    assert mcr["A1"] == pytest.approx(0.4 * 4e-4 / sigma, rel=1e-12)
    ccr = component_risk_contributions(weights, cov)
    assert ccr["A0"] == pytest.approx(0.36 * 1e-4 / sigma, rel=1e-12)


def test_erc_weights_have_equal_pct_contributions():
    # cross-module: the ERC optimizer's output must decompose to 1/n shares
    cov = make_cov([0.008, 0.012, 0.02, 0.03, 0.015], corr=0.4)
    weights = erc(pd.Series(0.0, index=cov.index), cov, None, Config())
    pct = pct_risk_contributions(weights, cov)
    np.testing.assert_allclose(pct.to_numpy(), 1.0 / len(cov), atol=1e-5)


def test_zero_book_raises():
    cov = make_cov([0.01, 0.02])
    weights = pd.Series([0.0, 0.0], index=cov.index)
    with pytest.raises(ValueError, match="zero-risk"):
        pct_risk_contributions(weights, cov)


def test_misaligned_weights_raise():
    cov = make_cov([0.01, 0.02])
    weights = pd.Series([0.5, 0.5], index=["B0", "B1"])
    with pytest.raises(ValueError, match="not aligned"):
        marginal_risk_contributions(weights, cov)


# ------------------------------------------------------------------ sweep


def test_sweep_lambda_shape_and_flatness_for_simple_optimizer():
    closes = random_closes(n=90)
    table = sweep_lambda(
        closes, ("equal_weight",), small_cfg(), lambdas=(0.0, 1e-3), freqs=("W-FRI",)
    )
    assert table.shape == (2, len(STAT_NAMES))
    assert table.index.names == ["optimizer", "freq", "turnover_lambda"]
    # equal_weight ignores lambda: the two rows are identical across all stats
    pd.testing.assert_series_equal(table.iloc[0], table.iloc[1], check_names=False)


def test_sweep_lambda_reads_lambda_for_mvo_ls():
    closes = random_closes(n=90)
    table = sweep_lambda(
        closes, ("mvo_ls",), small_cfg(), lambdas=(0.0, 1e-2), freqs=("W-FRI",)
    )
    sharpe = table["sharpe"]
    assert sharpe.iloc[0] != sharpe.iloc[1]  # lambda actually bites
    turnover = table["ann_turnover"]
    assert turnover.iloc[1] < turnover.iloc[0]  # via lower turnover


def test_sweep_lambda_covers_the_grid():
    closes = random_closes(n=90)
    table = sweep_lambda(
        closes,
        ("equal_weight", "mvo_ls"),
        small_cfg(),
        lambdas=(0.0, 1e-3),
        freqs=("W-FRI",),
    )
    assert table.shape == (4, len(STAT_NAMES))
    assert set(table.index.get_level_values("optimizer")) == {
        "equal_weight",
        "mvo_ls",
    }


def test_sweep_lambda_empty_grid_raises():
    with pytest.raises(ValueError, match="non-empty"):
        sweep_lambda(
            random_closes(), ("mvo_ls",), small_cfg(), lambdas=(), freqs=("B",)
        )


# --------------------------------------------------------------- frontier


def test_efficient_frontier_moves_up_and_right_as_gamma_falls():
    closes = random_closes(n=110, k=5)
    from portlab.preprocessing import daily_returns

    returns = daily_returns(closes)
    frontier = efficient_frontier(returns, Config(), gammas=(3000.0, 300.0, 30.0))
    assert list(frontier.columns) == ["ann_vol", "ann_ret"]
    assert np.isfinite(frontier.to_numpy()).all()
    # index sorted descending gamma: vol and return non-decreasing down the rows
    vols = frontier["ann_vol"].to_numpy()
    rets = frontier["ann_ret"].to_numpy()
    assert (np.diff(vols) >= -1e-12).all()
    assert (np.diff(rets) >= -1e-12).all()


def test_efficient_frontier_empty_gammas_raises():
    closes = random_closes(n=60, k=5)
    from portlab.preprocessing import daily_returns

    with pytest.raises(ValueError, match="non-empty"):
        efficient_frontier(daily_returns(closes), Config(), gammas=())


# ------------------------------------------------------------------- beta


def test_beta_of_benchmark_against_itself_is_one():
    bench = series([0.01, -0.02, 0.005, 0.013, -0.007])
    assert market_beta(bench, bench) == pytest.approx(1.0, rel=1e-12)


def test_beta_hand_computed_scaled_and_offset():
    bench = series([0.01, -0.02, 0.005, 0.013, -0.007])
    # exact linear relation r = 1.5 b + c -> beta is exactly 1.5
    assert market_beta(bench * 1.5 + 0.002, bench) == pytest.approx(1.5, rel=1e-12)
    # sign flips through the covariance
    assert market_beta(-bench, bench) == pytest.approx(-1.0, rel=1e-12)


def test_beta_of_market_neutral_book_is_zero():
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2016-01-04", periods=400, name="date")
    bench = pd.Series(rng.normal(0.0005, 0.01, 400), index=idx)
    # independent noise carries no market exposure
    noise = pd.Series(rng.normal(0.0, 0.01, 400), index=idx)
    assert market_beta(noise, bench) == pytest.approx(0.0, abs=0.15)


def test_beta_uses_only_overlapping_dates():
    bench = series([0.01, -0.02, 0.005, 0.013, -0.007])
    partial = (bench * 2.0).iloc[1:4]  # shorter, still exactly 2x on overlap
    assert market_beta(partial, bench) == pytest.approx(2.0, rel=1e-12)


def test_beta_rejects_degenerate_inputs():
    bench = series([0.01, -0.02, 0.005])
    with pytest.raises(ValueError, match="at least 2 overlapping"):
        market_beta(bench.iloc[:1], bench)
    flat = pd.Series(0.001, index=bench.index)
    with pytest.raises(ValueError, match="no variance"):
        market_beta(bench, flat)


def test_comparison_table_beta_column_is_opt_in():
    closes = random_closes()
    cfg = small_cfg()
    results = {
        name: run_backtest(closes, name, cfg) for name in ("equal_weight", "erc")
    }
    bench = results["equal_weight"].net_returns

    plain = comparison_table(results)
    assert list(plain.columns) == list(STAT_NAMES)  # benchmark-free by default

    with_beta = comparison_table(results, benchmark=bench)
    assert list(with_beta.columns) == [*STAT_NAMES, "beta"]
    # a book benchmarked against itself has beta exactly one
    assert with_beta.loc["equal_weight", "beta"] == pytest.approx(1.0, rel=1e-12)

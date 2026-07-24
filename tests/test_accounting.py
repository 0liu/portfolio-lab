"""Accounting identity tests.

The engine's ledger must satisfy exact bookkeeping identities:

- net + cost = gross, and cost = turnover x rate (zero off schedule);
- Long-only weights are conserved under zero cost, and with costs the
  day-to-day weight sums obey sum_{t+1} = (sum_t + gross_t) / (1 + net_t)
  (cost is paid from cash and lowers the NAV denominator);
- The whole ledger is reproducible by an independent replay from the
  published rebalance-date weights and the return panel;
- A deterministic two-asset case is verified line by line, every
  intermediate derived from the closes inside the test;
- Turnover collapses to zero end-to-end as the turnover penalty or the
  no-trade band grows.
"""

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from portlab.config import (
    Config,
    ConstructionConfig,
    CostConfig,
    EngineConfig,
    EstimationConfig,
    SignalConfig,
)
from portlab.engine import run_backtest
from portlab.preprocessing import daily_returns


def det_closes(
    n: int, drifts: tuple[float, ...], jitter: float = 0.004
) -> pd.DataFrame:
    """Deterministic closes: per-asset drift + alternating jitter (sigma > 0)."""
    t = np.arange(n)[:, None]
    sign = np.where(t % 2 == 0, 1.0, -1.0)
    rets = np.array(drifts)[None, :] + jitter * sign
    idx = pd.bdate_range("2016-01-04", periods=n, name="date")
    return pd.DataFrame(
        100.0 * np.cumprod(1.0 + rets, axis=0),
        index=idx,
        columns=[f"A{i}" for i in range(len(drifts))],
    )


def random_closes(n: int = 110, k: int = 4, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.01, size=(n, k))
    idx = pd.bdate_range("2016-01-04", periods=n, name="date")
    return pd.DataFrame(
        100.0 * np.cumprod(1.0 + rets, axis=0),
        index=idx,
        columns=[f"A{i}" for i in range(k)],
    )


def make_cfg(
    freq: str = "W-FRI",
    band: float = 0.0,
    cost_bps: float = 5.0,
    turnover_lambda: float = 0.0,
) -> Config:
    return Config(
        signals=SignalConfig(
            tsmom_windows=(2, 5),
            signal_vol_halflife_days=3,
            xs_lookback=8,
            xs_exclude=2,
            reversal_window=3,
        ),
        estimation=EstimationConfig(cov_window_days=15),
        construction=ConstructionConfig(turnover_lambda=turnover_lambda),
        engine=EngineConfig(rebalance_freq=freq, no_trade_band=band),
        costs=CostConfig(cost_per_side_bps=cost_bps),
    )


def replay_ledger(result, returns: pd.DataFrame, rate: float) -> pd.DataFrame:
    """Independent re-implementation of the engine's bookkeeping (band = 0).

    Walks the published weights with plain pandas arithmetic: rebalance-date
    rows are taken as the traded targets, every other row must equal the
    drift predicted from the previous day, and turnover/cost/gross/net are
    recomputed from scratch.
    """
    rebalance = set(result.turnover.index)
    drift = pd.Series(0.0, index=result.weights.columns)
    rows = []
    for day in result.net_returns.index:
        held = result.weights.loc[day]
        day_returns = returns.loc[day]
        if day in rebalance:
            tur = float((held - drift).abs().sum())
            cost = tur * rate
        else:
            pd.testing.assert_series_equal(  # drift days publish the drift
                held, drift, check_exact=False, atol=1e-12, rtol=0.0, check_names=False
            )
            tur = np.nan
            cost = 0.0
        gross = float((held * day_returns).sum())
        net = gross - cost
        rows.append((day, tur, cost, gross, net))
        drift = held * (1.0 + day_returns) / (1.0 + net)
    return pd.DataFrame(
        rows, columns=["date", "turnover", "cost", "gross", "net"]
    ).set_index("date")


# ------------------------------------------------------------ ledger algebra


def test_net_plus_cost_equals_gross():
    result = run_backtest(random_closes(), "equal_weight", make_cfg())
    lhs = (result.net_returns + result.costs).to_numpy()
    np.testing.assert_allclose(lhs, result.gross_returns.to_numpy(), rtol=0, atol=1e-15)


def test_cost_is_turnover_times_rate_and_zero_off_schedule():
    cfg = make_cfg()
    result = run_backtest(random_closes(), "equal_weight", cfg)
    rate = cfg.costs.cost_per_side_bps / 1e4
    on_schedule = result.costs.loc[result.turnover.index].to_numpy()
    assert (on_schedule == result.turnover.to_numpy() * rate).all()  # same product
    off_days = result.costs.index.difference(result.turnover.index)
    assert (result.costs.loc[off_days] == 0.0).all()


# -------------------------------------------------------------- conservation


@pytest.mark.parametrize("name", ["equal_weight", "erc", "mvo"])
def test_long_only_weights_conserved_under_zero_cost(name):
    closes = random_closes(k=5)  # 5 assets: mvo cap feasibility 5 * 0.25 >= 1
    result = run_backtest(closes, name, make_cfg(cost_bps=0.0))
    sums = result.weights.sum(axis=1).to_numpy()
    np.testing.assert_allclose(sums, 1.0, rtol=0, atol=1e-9)


def test_drift_day_sum_obeys_cost_in_nav_identity():
    # sum_{t+1} = (sum_t + gross_t) / (1 + net_t): with costs on, the day after
    # a rebalance sums slightly above 1 because cost lowered the NAV.
    cfg = make_cfg(cost_bps=20.0)  # exaggerate to make the effect visible
    result = run_backtest(random_closes(), "equal_weight", cfg)
    idx = result.net_returns.index
    rebalance = set(result.turnover.index)
    sums = result.weights.sum(axis=1)
    for prev_day, day in pairwise(idx):
        if day in rebalance:
            continue  # rebalance rows are optimizer targets, not drift
        expected = (sums.loc[prev_day] + result.gross_returns.loc[prev_day]) / (
            1.0 + result.net_returns.loc[prev_day]
        )
        assert sums.loc[day] == pytest.approx(expected, abs=1e-12)
    # the effect is real: the day after the first (costly) deployment sums > 1
    first = result.turnover.index[0]
    after_first = idx[idx.get_loc(first) + 1]
    if after_first not in rebalance:
        assert sums.loc[after_first] > 1.0


# ------------------------------------------------------------- ledger replay


def test_full_ledger_replay_matches_engine():
    cfg = make_cfg()
    closes = random_closes()
    result = run_backtest(closes, "erc", cfg)
    returns = daily_returns(closes)
    replayed = replay_ledger(result, returns, cfg.costs.cost_per_side_bps / 1e4)

    np.testing.assert_allclose(
        replayed["gross"].to_numpy(), result.gross_returns.to_numpy(), atol=1e-14
    )
    np.testing.assert_allclose(
        replayed["net"].to_numpy(), result.net_returns.to_numpy(), atol=1e-14
    )
    np.testing.assert_allclose(
        replayed["cost"].to_numpy(), result.costs.to_numpy(), atol=1e-14
    )
    on_schedule = replayed.loc[result.turnover.index, "turnover"].to_numpy()
    np.testing.assert_allclose(on_schedule, result.turnover.to_numpy(), atol=1e-12)


# --------------------------------------------------------- hand-computed case


def test_two_asset_deterministic_case_line_by_line():
    closes = det_closes(60, (0.001, -0.0005))
    cfg = make_cfg(freq="W-FRI", cost_bps=5.0)
    result = run_backtest(closes, "equal_weight", cfg)
    returns = daily_returns(closes)
    rate = 5e-4

    # first rebalance: cash -> equal weight, every number exact
    t0 = result.turnover.index[0]
    assert result.weights.loc[t0].tolist() == [0.5, 0.5]
    assert result.turnover.iloc[0] == 1.0  # |0.5-0| + |0.5-0|
    assert result.costs.loc[t0] == 1.0 * rate

    r0 = returns.loc[t0]
    gross0 = 0.5 * r0.iloc[0] + 0.5 * r0.iloc[1]
    net0 = gross0 - 1.0 * rate
    assert result.gross_returns.loc[t0] == pytest.approx(gross0, rel=1e-12)
    assert result.net_returns.loc[t0] == pytest.approx(net0, rel=1e-12)

    # next day: drift with the cost inside the NAV denominator
    day1 = result.net_returns.index[result.net_returns.index.get_loc(t0) + 1]
    w1_expected = [
        0.5 * (1 + r0.iloc[0]) / (1 + net0),
        0.5 * (1 + r0.iloc[1]) / (1 + net0),
    ]
    assert result.weights.loc[day1].tolist() == pytest.approx(w1_expected, abs=1e-14)

    # chain the drift to the second rebalance and re-derive its turnover/cost
    t1 = result.turnover.index[1]
    drift = pd.Series([0.5, 0.5], index=closes.columns)
    for day in result.net_returns.index[
        result.net_returns.index.get_loc(t0) : result.net_returns.index.get_loc(t1)
    ]:
        day_r = returns.loc[day]
        if day == t0:
            held = drift
            day_net = net0
        else:
            held = drift
            day_net = float((held * day_r).sum())
        drift = held * (1.0 + day_r) / (1.0 + day_net)
    expected_turnover = float((0.5 - drift).abs().sum())
    assert result.turnover.loc[t1] == pytest.approx(expected_turnover, abs=1e-12)
    assert result.costs.loc[t1] == pytest.approx(expected_turnover * rate, abs=1e-15)

    # every rebalance row of a long-only optimizer sums to exactly one
    rebal_sums = result.weights.loc[result.turnover.index].sum(axis=1)
    assert (rebal_sums == 1.0).all()


# ------------------------------------------------------- turnover collapse


def test_lambda_collapse_end_to_end():
    closes = random_closes(n=90)
    totals = []
    for lam in (0.0, 1e-3, 100.0):
        result = run_backtest(closes, "mvo_ls", make_cfg(freq="B", turnover_lambda=lam))
        totals.append(float(result.turnover.sum()))
    assert totals[1] <= totals[0] + 1e-6
    assert totals[2] <= totals[1] + 1e-6
    assert totals[0] > 0.5  # lambda=0 actually deploys and trades
    # a huge penalty pins the book to its cash start: nothing trades, ever
    assert totals[2] < 1e-3
    pinned = run_backtest(closes, "mvo_ls", make_cfg(freq="B", turnover_lambda=100.0))
    assert pinned.net_returns.abs().sum() < 1e-3


def test_band_collapse_end_to_end():
    # a band wider than the initial deployment gap (|0.5 - 0| = 0.5) means the
    # book never leaves cash: zeros everywhere, exactly
    result = run_backtest(
        det_closes(60, (0.001, -0.0005)), "equal_weight", make_cfg(band=0.9)
    )
    assert (result.weights == 0.0).all().all()
    assert (result.turnover == 0.0).all()
    assert (result.costs == 0.0).all()
    assert (result.net_returns == 0.0).all()

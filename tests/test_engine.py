"""Engine mechanics tests

Schedule, warm-up, band, overlay wiring, and the end-to-end PIT tripwires.
"""

import numpy as np
import pandas as pd
import pytest

from portlab.config import (
    Config,
    EngineConfig,
    EstimationConfig,
    SignalConfig,
)
from portlab.engine import _rebalance_dates, run_backtest
from portlab.estimation import ewma_cov


def make_closes(n: int = 120, k: int = 4, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.01, size=(n, k))
    idx = pd.bdate_range("2016-01-04", periods=n, name="date")
    return pd.DataFrame(
        100.0 * np.cumprod(1.0 + rets, axis=0),
        index=idx,
        columns=[f"A{i}" for i in range(k)],
    )


def small_cfg(**engine_kwargs) -> Config:
    return Config(
        signals=SignalConfig(
            tsmom_windows=(2, 5),
            signal_vol_halflife_days=3,
            xs_lookback=8,
            xs_exclude=2,
            reversal_window=3,
        ),
        estimation=EstimationConfig(cov_window_days=15),
        engine=EngineConfig(**engine_kwargs),
    )


def expected_start(closes: pd.DataFrame, cfg: Config) -> pd.Timestamp:
    """Independent re-derivation of the engine's start rule."""
    from portlab.preprocessing import daily_returns
    from portlab.signals import expected_returns

    returns = daily_returns(closes)
    mu = expected_returns(closes, cfg.signals)
    rebal = _rebalance_dates(returns.index, cfg.engine.rebalance_freq)
    for d in rebal:
        pos = returns.index.get_loc(d)
        if pos >= cfg.estimation.cov_window_days and not mu.loc[d].isna().any():
            return d
    raise AssertionError("no valid start in fixture")


# ------------------------------------------------------------------ shapes


def test_output_shapes_alignment_and_no_nan():
    closes = make_closes()
    result = run_backtest(closes, "equal_weight", small_cfg())
    idx = result.net_returns.index
    assert result.gross_returns.index.equals(idx)
    assert result.costs.index.equals(idx)
    assert result.weights.index.equals(idx)
    assert list(result.weights.columns) == list(closes.columns)
    assert not result.net_returns.isna().any()
    assert not result.weights.isna().any().any()
    assert set(result.turnover.index) <= set(idx)


def test_start_respects_warmup_and_cov_window():
    closes = make_closes()
    cfg = small_cfg()
    result = run_backtest(closes, "equal_weight", cfg)
    assert result.net_returns.index[0] == expected_start(closes, cfg)


def test_engine_is_deterministic():
    closes = make_closes()
    a = run_backtest(closes, "erc", small_cfg())
    b = run_backtest(closes, "erc", small_cfg())
    pd.testing.assert_series_equal(a.net_returns, b.net_returns)
    pd.testing.assert_frame_equal(a.weights, b.weights)


# ---------------------------------------------------------------- schedule


def test_daily_freq_rebalances_every_day():
    closes = make_closes()
    result = run_backtest(closes, "equal_weight", small_cfg(rebalance_freq="B"))
    assert result.turnover.index.equals(result.net_returns.index)
    assert result.turnover.iloc[0] == pytest.approx(1.0)  # cash -> fully invested


def test_weekly_freq_trades_only_on_schedule():
    closes = make_closes()
    result = run_backtest(closes, "equal_weight", small_cfg(rebalance_freq="W-FRI"))
    rebal = set(result.turnover.index)
    assert 0 < len(rebal) < len(result.net_returns)
    off_days = result.costs.index.difference(result.turnover.index)
    assert (result.costs.loc[off_days] == 0).all()
    assert result.costs.loc[sorted(rebal)].iloc[0] > 0


def test_costs_are_turnover_times_rate():
    closes = make_closes()
    cfg = small_cfg()
    result = run_backtest(closes, "equal_weight", cfg)
    rate = cfg.costs.cost_per_side_bps / 1e4
    on_rebal = result.costs.loc[result.turnover.index]
    np.testing.assert_allclose(on_rebal.to_numpy(), result.turnover.to_numpy() * rate)


# -------------------------------------------------------------------- band


def test_no_trade_band_suppresses_drift_trimming():
    closes = make_closes(n=100, k=3, seed=9)
    banded = run_backtest(closes, "equal_weight", small_cfg(no_trade_band=0.05))
    free = run_backtest(closes, "equal_weight", small_cfg(no_trade_band=0.0))
    # equal-weight target is constant; daily drift is tiny, so a 5% band
    # blocks every trade after the initial deployment...
    assert banded.turnover.iloc[0] == pytest.approx(1.0)
    assert (banded.turnover.iloc[1:] == 0).all()
    # ...while without a band the drift is trimmed back most days
    assert (free.turnover.iloc[1:] > 0).any()
    assert free.turnover.iloc[1:].sum() > banded.turnover.iloc[1:].sum()


# ------------------------------------------------------------- vol target


def test_vol_target_overlay_hits_target_at_rebalance():
    closes = make_closes(n=110, k=4, seed=13)
    cfg = small_cfg(vol_target=True)
    result = run_backtest(closes, "equal_weight", cfg)
    # recompute Sigma_t independently and check the held book's predicted
    # annualized vol on a few rebalance dates
    from portlab.preprocessing import daily_returns

    returns = daily_returns(closes)
    for day in result.turnover.index[[0, len(result.turnover) // 2, -1]]:
        pos = returns.index.get_loc(day)
        sigma = ewma_cov(
            returns.iloc[pos - cfg.estimation.cov_window_days : pos],
            cfg.estimation.ewma_halflife_days,
        )
        w = result.weights.loc[day].to_numpy()
        ann = np.sqrt(w @ sigma.to_numpy() @ w) * np.sqrt(252)
        if np.abs(w).sum() < cfg.construction.gross_cap - 1e-9:  # not gross-capped
            assert ann == pytest.approx(cfg.construction.vol_target_annual, rel=1e-6)


# --------------------------------------------------------------- tripwires


def test_engine_pit_tripwire_last_day():
    closes = make_closes()
    cfg = small_cfg()
    base = run_backtest(closes, "equal_weight", cfg)
    mutated = closes.copy()
    mutated.iloc[-1] = mutated.iloc[-1] * 1.3
    after = run_backtest(mutated, "equal_weight", cfg)
    # decisions never see the final close: weights, costs, turnover identical
    pd.testing.assert_frame_equal(base.weights, after.weights)
    pd.testing.assert_series_equal(base.costs, after.costs)
    pd.testing.assert_series_equal(base.turnover, after.turnover)
    # only the final day's return moves
    pd.testing.assert_series_equal(
        base.net_returns.iloc[:-1], after.net_returns.iloc[:-1]
    )
    assert base.net_returns.iloc[-1] != after.net_returns.iloc[-1]


def test_engine_pit_tripwire_interior_day():
    closes = make_closes()
    cfg = small_cfg()
    base = run_backtest(closes, "erc", cfg)
    k = 90  # position in the closes index, inside the output range
    mutated = closes.copy()
    mutated.iloc[k] = mutated.iloc[k] * 1.4
    after = run_backtest(mutated, "erc", cfg)
    shock_day = closes.index[k]
    # weights through the shock day are decided from <= k-1 closes: identical
    pd.testing.assert_frame_equal(
        base.weights.loc[:shock_day], after.weights.loc[:shock_day]
    )
    # P&L strictly before the shock day is untouched
    before = base.net_returns.index[base.net_returns.index < shock_day]
    pd.testing.assert_series_equal(
        base.net_returns.loc[before], after.net_returns.loc[before]
    )
    # and the shock must propagate afterwards, or this test tests nothing
    assert not base.weights.equals(after.weights)


# ---------------------------------------------------------------- fail loud


def test_nan_closes_raise():
    closes = make_closes()
    closes.iloc[50, 1] = float("nan")
    with pytest.raises(ValueError, match="NaN in closes"):
        run_backtest(closes, "equal_weight", small_cfg())


def test_unknown_optimizer_raises():
    with pytest.raises(ValueError, match="Unknown optimizer"):
        run_backtest(make_closes(), "magic", small_cfg())


def test_sample_too_short_raises():
    closes = make_closes(n=12)
    with pytest.raises(ValueError, match="No rebalance date"):
        run_backtest(closes, "equal_weight", small_cfg())


def test_lw_cc_estimator_smoke():
    closes = make_closes()
    cfg = Config(
        signals=small_cfg().signals,
        estimation=EstimationConfig(cov_window_days=15, cov_estimator="lw_cc"),
    )
    result = run_backtest(closes, "inverse_vol", cfg)
    assert np.isfinite(result.net_returns.to_numpy()).all()

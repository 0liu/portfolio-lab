"""Daily walk-forward backtest engine.

On each rebalance date t:
  1. Estimate the covariance from the trailing window ending at t-1
     (EWMA, or Ledoit-Wolf on the equally-weighted window).
  2. Read the signal's mu_t (PIT: data through the t-1 close).
  3. Run the optimizer -> target weights -> optional vol-target overlay.
  4. Trade the gap to the drifted book, paying cost per side. An optional
     no-trade band suppresses sub-threshold trades.
  5. Hold over [t, t+1) and mark the net return.

One engine, multiple optimizers -> directly comparable equity curves and
turnover/cost profiles. The rebalance frequency enables the daily-vs-weekly
cost-sensitivity study.

Timeline: everything driving w_t is known by the t-1 close, and w_t earns
row-t's return (P_{t-1} -> P_t) — decisions never see the day they trade on.

Drift: between rebalances weights move with realized returns,

    w'_i = w_i (1 + r_i) / (1 + r_net),   r_net = w'r - cost.

Cost is paid from cash, so it lowers NAV and enters the drift denominator,
which is stricter than the fully-invested sketch in docs/methodology.md.

No-trade band: each rebalance compares the (overlay-adjusted) target with
the drifted book per asset and trades only the assets whose gap exceeds the
band. Banding can therefore leave a long-only book slightly off sum-to-1 by
design.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from portlab.config import Config
from portlab.construction import OPTIMIZER_NAMES, optimize, vol_target
from portlab.estimation import ewma_cov, ledoit_wolf_cc
from portlab.preprocessing import daily_returns
from portlab.signals import expected_returns


@dataclass(slots=True)
class BacktestResult:
    """Backtest engine output.

    The field `weights` holds each day's start-of-day book (the weights that
    earn that day's return), so gross_t = w_t @ r_t.
    On rebalance dates this is the post-band target actually traded into.
    Between rebalances it is the drifted book. No separate target-weight field
    is needed. The rebalance-date rows are the targets.
    """

    net_returns: pd.Series
    gross_returns: pd.Series
    costs: pd.Series
    turnover: pd.Series  # indexed by rebalance dates only
    weights: pd.DataFrame


def _rebalance_dates(dates: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """Last trading date of each `freq` period ("B" = every trading day)."""
    series = pd.Series(dates, index=dates)
    last = series.resample(freq).last().dropna()
    return pd.DatetimeIndex(last)


def _sigma(returns_window: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if cfg.estimation.cov_estimator == "ewma":
        return ewma_cov(returns_window, cfg.estimation.ewma_halflife_days)
    return ledoit_wolf_cc(returns_window)[0]


def run_backtest(
    closes: pd.DataFrame, optimizer_name: str, cfg: Config
) -> BacktestResult:
    """Walk the panel forward, rebalancing on the configured schedule.

    closes: full (date x ticker) close panel of the assets to trade. The caller
    selects the universe slice. NaN anywhere raises an error: the engine never
    trades through missing data.
    """
    if optimizer_name not in OPTIMIZER_NAMES:
        raise ValueError(
            f"Unknown optimizer {optimizer_name!r}. Available: {OPTIMIZER_NAMES}"
        )
    if closes.isna().any().any():
        raise ValueError("NaN in closes. The engine requires a complete panel.")

    returns = daily_returns(closes)
    mu_panel = expected_returns(closes, cfg.signals)
    dates = returns.index
    window = cfg.estimation.cov_window_days
    rebalance = _rebalance_dates(dates, cfg.engine.rebalance_freq)  # type: ignore
    rebalance_set = set(rebalance)

    # First rebalance date with a full covariance window (rows strictly
    # before it) and a fully formed mu row.
    start_pos: int | None = None
    date_pos = {d: i for i, d in enumerate(dates)}
    for d in rebalance:
        pos = date_pos[d]
        if pos >= window and not bool(mu_panel.loc[d].isna().any()):
            start_pos = pos
            break
    if start_pos is None:
        raise ValueError(
            "No rebalance date has both a full covariance window and a "
            "complete mu row. Extend the sample or shrink the warm-up."
        )

    tickers = closes.columns
    n = len(tickers)
    rate = cfg.costs.cost_per_side_bps / 1e4
    band = cfg.engine.no_trade_band
    ret_arr = returns.to_numpy(dtype="float64")

    drift = np.zeros(n)  # start from cash
    out_dates: list[pd.Timestamp] = []
    gross_out: list[float] = []
    net_out: list[float] = []
    cost_out: list[float] = []
    weight_rows: list[np.ndarray] = []
    turnover_dates: list[pd.Timestamp] = []
    turnover_out: list[float] = []

    for pos in range(start_pos, len(dates)):
        day = dates[pos]
        if day in rebalance_set:
            window_frame = returns.iloc[pos - window : pos]  # ends at t-1 per PIT rule
            sigma = _sigma(window_frame, cfg)
            mu_day = mu_panel.loc[day]
            if mu_day.isna().any():
                raise RuntimeError(f"NaN mu on rebalance date {day.date()}")
            target = optimize(
                optimizer_name,
                mu_day,
                sigma,
                pd.Series(drift, index=tickers),
                cfg,
            )
            if cfg.engine.vol_target:
                target = vol_target(target, sigma, cfg)
            tgt = target.to_numpy(dtype="float64")
            trade = np.abs(tgt - drift) > band
            held = np.where(trade, tgt, drift)
            day_turnover = float(np.abs(held - drift).sum())
            day_cost = day_turnover * rate
            turnover_dates.append(day)
            turnover_out.append(day_turnover)
        else:
            held = drift
            day_cost = 0.0

        day_returns = ret_arr[pos]
        gross = float(held @ day_returns)
        net = gross - day_cost
        growth = 1.0 + net
        if growth <= 1e-9:
            raise RuntimeError(f"portfolio wiped out on {day.date()}")

        out_dates.append(day)
        gross_out.append(gross)
        net_out.append(net)
        cost_out.append(day_cost)
        weight_rows.append(held)
        # cost is paid from cash, so it lowers NAV, the net in the denominator
        drift = held * (1.0 + day_returns) / growth

    index = pd.DatetimeIndex(out_dates, name="date")
    return BacktestResult(
        net_returns=pd.Series(net_out, index=index, name="net"),
        gross_returns=pd.Series(gross_out, index=index, name="gross"),
        costs=pd.Series(cost_out, index=index, name="cost"),
        turnover=pd.Series(
            turnover_out,
            index=pd.DatetimeIndex(turnover_dates, name="date"),
            name="turnover",
        ),
        weights=pd.DataFrame(weight_rows, index=index, columns=tickers),
    )

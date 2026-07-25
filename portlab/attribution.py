"""Performance reporting and risk attribution.

Reporting conventions:
    - 252 trading days per year
    - zero risk-free rate
    - one-way turnover as the engine measures it
    - cost drag as annualized cost in return units

Risk decomposition follows the Euler identity for the homogeneous-degree-one
portfolio volatility

    sigma(w) = sum_i CCR_i,
    MCR_i    = (Sigma w)_i / sigma(w),        marginal contribution
    CCR_i    = w_i * MCR_i,                    component contribution
    CCR_i / sigma(w)                           percentage share, sums to one

so each asset's risk share is exact and additive, per optimizer.

These feed the headline exhibits:
  1. Optimizer comparison table: CAGR, vol, Sharpe, Sortino, Calmar, max
     drawdown, annual turnover, realized cost drag, at raw risk levels and
     again vol-targeted to a common annualized volatility.
  2. Lambda sweep: net-of-cost Sharpe against the turnover penalty, MVO
     long-only and long-short, daily vs weekly. It's the one-chart argument for
     putting costs inside the objective, with turnover and cost drag shown
     alongside to expose the mechanism.
  3. Risk-contribution bars: share of portfolio risk by asset class, and
     equal weight vs ERC per asset, showing ERC equalizes what equal weight
     does not.
  4. In-sample frontier vs realized points: how far estimation error pushes
     each optimizer inside the ex-ante efficient frontier.
"""

from collections.abc import Mapping, Sequence
from dataclasses import replace

import numpy as np
import pandas as pd

from portlab.config import Config
from portlab.construction import TRADING_DAYS
from portlab.engine import BacktestResult, run_backtest

STAT_NAMES: tuple[str, ...] = (
    "cagr",
    "ann_vol",
    "sharpe",
    "max_drawdown",
    "ann_turnover",
    "cost_drag",
)


def equity_curve(returns: pd.Series) -> pd.Series:
    """Cumulative growth of one unit of NAV."""
    return (1.0 + returns).cumprod()


def drawdown(returns: pd.Series) -> pd.Series:
    """Equity relative to its running peak, minus one (<= 0)."""
    equity = equity_curve(returns)
    return equity / equity.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Most negative drawdown (reported as a negative number)."""
    return float(drawdown(returns).min())


def performance_stats(result: BacktestResult) -> pd.Series:
    """The comparison-table row for one backtest."""
    net = result.net_returns
    if len(net) == 0:
        raise ValueError("empty backtest result")
    years = len(net) / TRADING_DAYS
    total_growth = float((1.0 + net).prod())  # type: ignore
    if total_growth <= 0:
        raise ValueError("non-positive terminal equity")
    std = float(net.std(ddof=1))
    # Stdev of a constant series comes back as ~1e-19 float noise, not exact 0.
    # Anything below the floor is "no volatility" and Sharpe is undefined.
    sharpe = (
        float(net.mean()) / std * np.sqrt(TRADING_DAYS) if std > 1e-12 else float("nan")
    )
    return pd.Series(
        {
            "cagr": total_growth ** (1.0 / years) - 1.0,
            "ann_vol": std * np.sqrt(TRADING_DAYS),
            "sharpe": sharpe,
            "max_drawdown": max_drawdown(net),
            "ann_turnover": float(result.turnover.sum()) / years,
            "cost_drag": float(result.costs.sum()) / years,
        },
        index=STAT_NAMES,
        name="stats",
    )


def comparison_table(results: Mapping[str, BacktestResult]) -> pd.DataFrame:
    """One row per optimizer, columns = STAT_NAMES, in the mapping's order."""
    if not results:
        raise ValueError("no results given")
    return pd.DataFrame(
        {name: performance_stats(result) for name, result in results.items()}
    ).T


def _portfolio_sigma(weights: pd.Series, cov: pd.DataFrame) -> float:
    if not weights.index.equals(cov.index):
        raise ValueError("weights are not aligned with cov")
    w = weights.to_numpy(dtype="float64")
    sigma = float(np.sqrt(w @ cov.to_numpy(dtype="float64") @ w))
    if sigma <= 0.0:
        raise ValueError("zero-risk book has no risk decomposition")
    return sigma


def marginal_risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """MCR_i = (Sigma w)_i / sigma: risk added by one more unit of asset i."""
    sigma = _portfolio_sigma(weights, cov)
    marginal = cov.to_numpy(dtype="float64") @ weights.to_numpy(dtype="float64")
    return pd.Series(marginal / sigma, index=weights.index, name="mcr")


def component_risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """CCR_i = w_i * MCR_i; sums exactly to portfolio sigma (Euler)."""
    ccr = weights * marginal_risk_contributions(weights, cov)
    return ccr.rename("ccr")


def pct_risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """CCR_i / sigma: each asset's share of total risk; sums to one."""
    sigma = _portfolio_sigma(weights, cov)
    return (component_risk_contributions(weights, cov) / sigma).rename("pct_risk")


def sweep_lambda(
    closes: pd.DataFrame,
    optimizer_name: str,
    cfg: Config,
    lambdas: Sequence[float],
    freqs: Sequence[str],
) -> pd.DataFrame:
    """Net-of-cost Sharpe for every (turnover_lambda, rebalance frequency).

    Index = lambda, columns = frequency. Only the MVO family reads lambda;
    sweeping a simple optimizer yields a flat line by construction.
    """
    if not lambdas or not freqs:
        raise ValueError("lambdas and freqs must be non-empty")
    columns: dict[str, dict[float, float]] = {}
    for freq in freqs:
        cells: dict[float, float] = {}
        for lam in lambdas:
            variant = replace(
                cfg,
                construction=replace(cfg.construction, turnover_lambda=lam),
                engine=replace(cfg.engine, rebalance_freq=freq),
            )
            result = run_backtest(closes, optimizer_name, variant)
            cells[lam] = float(performance_stats(result)["sharpe"])
        columns[freq] = cells
    table = pd.DataFrame(columns)
    table.index.name = "turnover_lambda"
    return table

"""Generate the full exhibit set from the committed parquet cache.

Usage:

    uv run --extra exhibits python scripts/make_exhibits.py
    uv run --extra exhibits python scripts/make_exhibits.py --quick

`--quick` shrinks the lambda grid to 3 points and sweeps weekly only.
The full run's daily mvo/mvo_ls sweep legs dominate the wall time.

Outputs into docs/exhibits/:
  comparison_table.{md,csv}            raw risk levels
  comparison_table_voltarget.{md,csv}  vol-targeted to 10% annualized
  equity_raw.png                       net equity, native risk levels
  equity_voltarget.png                 net equity, common 10% vol target
  underwater.png                       drawdown paths
  risk_return_scatter.png              realized ann_vol x CAGR + SPY
  lambda_sweep.png / lambda_sweep.csv  net Sharpe vs lambda (mvo, mvo_ls x freq)
  lambda_mechanism.png                 how lambda works: turnover and cost drag
  risk_contribution_bars.png           pct risk by asset class, final rebalance
  erc_vs_ew_contributions.png          per-asset risk shares: ERC vs equal weight
  frontier.png                         in-sample frontier vs realized points
"""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from portlab.attribution import (
    comparison_table,
    drawdown,
    efficient_frontier,
    equity_curve,
    pct_risk_contributions,
    performance_stats,
    sweep_lambda,
)
from portlab.config import Config
from portlab.construction import OPTIMIZER_NAMES, TRADING_DAYS
from portlab.data import load_universe_bars
from portlab.engine import run_all_optimizers, run_backtest
from portlab.estimation import ewma_cov
from portlab.preprocessing import close_panel, daily_returns
from portlab.universe import UNIVERSE, optimized_tickers

LAMBDA_GRID = (0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3)
QUICK_LAMBDA_GRID = (0.0, 3e-4, 3e-3)
SWEEP_OPTIMIZERS = ("mvo", "mvo_ls")
GAMMA_GRID = tuple(np.geomspace(3.0, 3000.0, 24))
FREQ_LABEL = {"B": "daily", "W-FRI": "weekly"}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def load_closes(tickers: tuple[str, ...]) -> pd.DataFrame:
    return close_panel(load_universe_bars(tickers=tickers))


def run_all(closes: pd.DataFrame, cfg: Config, tag: str) -> dict:
    results = {}
    for name in OPTIMIZER_NAMES:
        print(f"[{tag}] running {name} ...", flush=True)
        results.update(run_all_optimizers(closes, cfg, names=(name,)))
    return results


def spy_stats(spy_closes: pd.Series, start) -> tuple[pd.Series, pd.Series]:
    """SPY buy-and-hold daily returns over the backtest window, plus stats."""
    spy = spy_closes.loc[start:]
    returns = spy.pct_change(fill_method=None).dropna()
    years = len(returns) / TRADING_DAYS
    stats = pd.Series(
        {
            "cagr": float((1 + returns).prod()) ** (1 / years) - 1,
            "ann_vol": float(returns.std(ddof=1)) * np.sqrt(TRADING_DAYS),
            "ann_mean": float(returns.mean()) * TRADING_DAYS,
        }
    )
    return returns, stats


def write_tables(results: dict, out: Path, suffix: str) -> pd.DataFrame:
    table = comparison_table(results)
    table.to_csv(out / f"comparison_table{suffix}.csv")
    (out / f"comparison_table{suffix}.md").write_text(table.round(4).to_markdown())
    print(table.round(4).to_string())
    return table


def plot_equity(results: dict, spy_returns: pd.Series | None, out: Path,
                fname: str, title: str) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, result in results.items():
        equity_curve(result.net_returns).plot(ax=ax, label=name, linewidth=1.2)
    if spy_returns is not None:
        equity_curve(spy_returns).plot(
            ax=ax, label="SPY (buy&hold)", color="black", linestyle="--", linewidth=1.0
        )
    ax.set_title(title)
    ax.set_ylabel("equity (1 unit of NAV)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / fname, dpi=150)
    plt.close(fig)


def plot_underwater(results: dict, out: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, result in results.items():
        drawdown(result.net_returns).plot(ax=ax, label=name, linewidth=1.1)
    ax.set_title("Drawdown paths (net of costs, raw risk levels)")
    ax.set_ylabel("drawdown")
    ax.legend(frameon=False, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "underwater.png", dpi=150)
    plt.close(fig)


def plot_risk_return(results: dict, spy: pd.Series, out: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, result in results.items():
        stats = performance_stats(result)
        ax.scatter(stats["ann_vol"], stats["cagr"], s=60, zorder=3)
        ax.annotate(name, (stats["ann_vol"], stats["cagr"]),
                    textcoords="offset points", xytext=(8, 4))
    ax.scatter(spy["ann_vol"], spy["cagr"], s=80, color="black", marker="s", zorder=3)
    ax.annotate("SPY", (spy["ann_vol"], spy["cagr"]),
                textcoords="offset points", xytext=(8, 4))
    ax.set_xlabel("realized annualized volatility")
    ax.set_ylabel("realized CAGR")
    ax.set_title("Risk vs return, net of costs (raw risk levels)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "risk_return_scatter.png", dpi=150)
    plt.close(fig)


def build_sweep(closes: pd.DataFrame, cfg: Config, out: Path, quick: bool) -> None:
    lambdas = QUICK_LAMBDA_GRID if quick else LAMBDA_GRID
    freqs = ("W-FRI",) if quick else ("B", "W-FRI")
    print(f"[sweep] {SWEEP_OPTIMIZERS} x {freqs} x {len(lambdas)} lambdas ...",
          flush=True)
    table = sweep_lambda(closes, SWEEP_OPTIMIZERS, cfg, lambdas, freqs)
    table.to_csv(out / "lambda_sweep.csv")
    plt = _plt()

    def _lines(ax, column: str) -> None:
        for name in SWEEP_OPTIMIZERS:
            for freq in freqs:
                cell = table.xs((name, freq), level=("optimizer", "freq"))[column]
                ax.plot(cell.index, cell.to_numpy(), marker="o",
                        label=f"{name} {FREQ_LABEL.get(freq, freq)}")
        ax.set_xscale("symlog", linthresh=1e-5)
        ax.grid(alpha=0.3)

    fig, ax = plt.subplots(figsize=(8, 5))
    _lines(ax, "sharpe")
    ax.set_xlabel("turnover penalty lambda (symlog)")
    ax.set_ylabel("net-of-cost Sharpe")
    ax.set_title("Cost-aware optimization: net Sharpe vs turnover penalty")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "lambda_sweep.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    _lines(axes[0], "ann_turnover")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("annual one-way turnover (log)")
    axes[0].set_xlabel("lambda (symlog)")
    _lines(axes[1], "cost_drag")
    axes[1].set_ylabel("annualized cost drag")
    axes[1].set_xlabel("lambda (symlog)")
    axes[1].legend(frameon=False)
    fig.suptitle("How the penalty works: lambda compresses turnover, turnover sets cost")
    fig.tight_layout()
    fig.savefig(out / "lambda_mechanism.png", dpi=150)
    plt.close(fig)
    print(table["sharpe"].unstack("turnover_lambda").round(3).to_string())


def _final_sigma(result, returns: pd.DataFrame, cfg: Config) -> tuple[pd.Series, pd.DataFrame]:
    day = result.turnover.index[-1]
    pos = returns.index.get_loc(day)
    sigma = ewma_cov(
        returns.iloc[pos - cfg.estimation.cov_window_days : pos],
        cfg.estimation.ewma_halflife_days,
    )
    return result.weights.loc[day], sigma


def plot_risk_bars(results: dict, closes: pd.DataFrame, cfg: Config, out: Path) -> None:
    returns = daily_returns(closes)
    class_of = {a.ticker: a.asset_class.value for a in UNIVERSE}
    rows = {}
    for name, result in results.items():
        weights, sigma = _final_sigma(result, returns, cfg)
        if weights.abs().sum() == 0:
            continue
        pct = pct_risk_contributions(weights, sigma)
        rows[name] = pct.groupby(pct.index.map(class_of)).sum()
    frame = pd.DataFrame(rows).fillna(0.0)

    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 6))
    frame.T.plot(kind="bar", stacked=True, ax=ax, width=0.75)
    ax.set_title("Risk contribution by asset class (final rebalance)")
    ax.set_ylabel("share of portfolio risk")
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "risk_contribution_bars.png", dpi=150)
    plt.close(fig)


def plot_erc_vs_ivp(results: dict, closes: pd.DataFrame, cfg: Config, out: Path) -> None:
    returns = daily_returns(closes)
    frame = {}
    for name in ("erc", "inverse_vol"):
        weights, sigma = _final_sigma(results[name], returns, cfg)
        frame[name] = pct_risk_contributions(weights, sigma)
    table = pd.DataFrame(frame)

    plt = _plt()
    fig, ax = plt.subplots(figsize=(11, 5))
    table.plot(kind="bar", ax=ax, width=0.8)
    ax.axhline(1.0 / len(table), color="black", linewidth=0.8, linestyle=":")
    ax.set_title(
        "Per-asset risk share: ERC equalizes exactly, inverse-vol only approximates"
    )
    ax.set_ylabel("share of portfolio risk")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "erc_vs_ivp_contributions.png", dpi=150)
    plt.close(fig)


def plot_frontier(results: dict, closes: pd.DataFrame, cfg: Config,
                  spy: pd.Series, out: Path) -> None:
    returns = daily_returns(closes)
    frontier = efficient_frontier(returns, cfg, GAMMA_GRID)
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(frontier["ann_vol"], frontier["ann_ret"], color="gray", linewidth=1.5,
            label="in-sample capped MVO frontier\n(full-sample mu/Sigma, reference only)")
    for name, result in results.items():
        stats = performance_stats(result)
        realized_mean = float(result.net_returns.mean()) * TRADING_DAYS
        ax.scatter(stats["ann_vol"], realized_mean, s=60, zorder=3)
        ax.annotate(name, (stats["ann_vol"], realized_mean),
                    textcoords="offset points", xytext=(8, 4))
    ax.scatter(spy["ann_vol"], spy["ann_mean"], s=80, color="black", marker="s")
    ax.annotate("SPY", (spy["ann_vol"], spy["ann_mean"]),
                textcoords="offset points", xytext=(8, 4))
    ax.set_xlabel("annualized volatility")
    ax.set_ylabel("annualized arithmetic return")
    ax.set_title("Theory vs reality: realized points sit inside the in-sample frontier")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "frontier.png", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="docs/exhibits", type=Path)
    parser.add_argument("--quick", action="store_true",
                        help="small lambda grid, weekly sweep only")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    cfg_vt = replace(cfg, engine=replace(cfg.engine, vol_target=True))
    closes = load_closes(optimized_tickers())
    spy_closes = load_closes(("SPY",))["SPY"]
    print(f"panel: {closes.shape[0]} days x {closes.shape[1]} assets", flush=True)

    raw = run_all(closes, cfg, "raw")
    start = next(iter(raw.values())).net_returns.index[0]
    spy_returns, spy = spy_stats(spy_closes, start)

    write_tables(raw, args.output, "")
    plot_equity(raw, spy_returns, args.output, "equity_raw.png",
                "Net-of-cost equity, raw risk levels")
    plot_underwater(raw, args.output)
    plot_risk_return(raw, spy, args.output)
    plot_risk_bars(raw, closes, cfg, args.output)
    plot_erc_vs_ivp(raw, closes, cfg, args.output)
    plot_frontier(raw, closes, cfg, spy, args.output)

    vt = run_all(closes, cfg_vt, "vol-target")
    write_tables(vt, args.output, "_voltarget")
    plot_equity(vt, spy_returns, args.output, "equity_voltarget.png",
                "Net-of-cost equity, every book vol-targeted to 10% annualized")

    build_sweep(closes, cfg, args.output, quick=args.quick)
    print(f"exhibits written to {args.output}/", flush=True)


if __name__ == "__main__":
    sys.exit(main())

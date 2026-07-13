# portfolio-lab

Daily-rebalancing, long-short, multi-signal portfolio construction on ~20 cross-asset ETFs.

**Features**

- Multi-horizon time-series momentum
- Cross-sectional momentum
- Short-term reversal signals
- EWMA + Ledoit-Wolf covariance estimation
- Equal-weight
- Inverse volatility targeting
- Transaction-cost-aware optimization with turnover penalties. ERC / MVO / long-short MVO
- Daily walk-forward backtest with transaction costs
- Risk and performance attribution

## Dev quickstart

```
uv sync --dev
make check
```

## License

MIT

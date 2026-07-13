"""Market data layer: Alpaca daily bars + committed parquet cache.

The parquet cache under /data/ohlcv is a fresh clone that reproduces downstream
results with zero credentials. Bars are cached as delivered by the vendor (UTC
tz-aware index, split and dividend adjusted, SIP feed). Trading-date
normalization belongs to preprocessing, not here.

Refresh data cache with free Alpaca keys in .env:

```bash
uv run python -m portlab.data --refresh
```

Alpaca SDK is installed via the `refresh` extra.
"""

import argparse
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pandas as pd

from portlab.universe import UNIVERSE

# Alpaca market data begins 2016-01-01 on every feed
START = pd.Timestamp("2016-01-01", tz="UTC")

# SIP (full consolidated tape) is free for *historical* queries when `end`
# is at least 15 minutes old; IEX-only bars carry ~2.5% of US volume.
# Flip to "iex" if the account rejects SIP.
FEED = "sip"
_ADJUSTMENT = "all"  # split & dividend adjusted
_END_SAFETY = pd.Timedelta(minutes=20)  # keep `end` older than the 15-min rule

COLUMNS = ("open", "high", "low", "close", "volume")
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "ohlcv"

_KEY_VAR = "ALPACA_API_KEY"
_SECRET_VAR = "ALPACA_SECRET_KEY"


def bars_path(ticker: str, data_dir: Path = DATA_DIR) -> Path:
    """Cache location for one ticker's daily bars per file."""
    return data_dir / f"{ticker}.parquet"


def validate_bars(bars: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Fail-loud invariants for a per-ticker daily bars frame.

    Single validation authority: both the load path and the refresh path go
    through here, so a corrupt cache and a corrupt vendor response fail the
    same way.
    """
    if tuple(bars.columns) != COLUMNS:
        raise ValueError(
            f"{ticker}: expected columns {COLUMNS}, got {tuple(bars.columns)}"
        )
    if not isinstance(bars.index, pd.DatetimeIndex) or str(bars.index.tz) != "UTC":
        raise ValueError(f"{ticker}: index must be a tz-aware UTC DatetimeIndex")
    if bars.empty:
        raise ValueError(f"{ticker}: frame has no rows")
    if not bars.index.is_monotonic_increasing or bars.index.has_duplicates:
        raise ValueError(f"{ticker}: index not strictly increasing")
    if bars.isna().any().any():
        raise ValueError(f"{ticker}: NaN values present")
    if (bars[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError(f"{ticker}: non-positive prices present")
    if (bars["volume"] < 0).any():
        raise ValueError(f"{ticker}: negative volume present")
    if (bars["high"] < bars["low"]).any():
        raise ValueError(f"{ticker}: high < low present")
    return bars


def load_bars(ticker: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load one ticker's cached daily bars, validated."""
    path = bars_path(ticker, data_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No cached bars for {ticker} at {path}. "
            "Run `python -m portlab.data --refresh`"
        )
    return validate_bars(pd.read_parquet(path), ticker)


def load_universe_bars(
    tickers: tuple[str, ...] | None = None, data_dir: Path = DATA_DIR
) -> dict[str, pd.DataFrame]:
    """Load cached bars for the given tickers (default: full universe)."""
    if tickers is None:
        tickers = tuple(asset.ticker for asset in UNIVERSE)
    return {ticker: load_bars(ticker, data_dir) for ticker in tickers}


def normalize_bars(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Normalize vendor's long-format dataframe to single-ticker dataframe with
    exactly COLUMNS.

    Accepts the alpaca-py `.df` shape: (symbol, timestamp) MultiIndex with extra
    columns (trade_count, vwap). Volume stays float64 because split adjustment
    can make it fractional.
    """
    frame = raw
    if isinstance(frame.index, pd.MultiIndex):
        frame = frame.xs(ticker, level="symbol")
    if isinstance(frame, pd.Series):
        raise ValueError(f"{ticker}: vendor frame collapsed to a Series")
    missing = [col for col in COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"{ticker}: vendor frame missing columns {missing}")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError(f"{ticker}: vendor index is not a tz-aware DatetimeIndex")
    frame = frame.loc[:, list(COLUMNS)].astype("float64").sort_index()
    frame.index = cast(pd.DatetimeIndex, frame.index).tz_convert("UTC")
    frame.index.name = "timestamp"
    return validate_bars(cast(pd.DataFrame, frame), ticker)


def _fetch_bars(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    api_key: str,
    secret_key: str,
) -> pd.DataFrame:
    """
    Pull daily bars for one ticker from Alpaca.
    Requires the `refresh` extra dependency.
    """
    try:
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise RuntimeError(
            "alpaca-py is not installed; run `uv sync --extra refresh`"
        ) from exc

    client = StockHistoricalDataClient(api_key, secret_key)
    request = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=cast(Any, TimeFrame.Day),
        start=start.to_pydatetime(),
        end=end.to_pydatetime(),
        adjustment=Adjustment(_ADJUSTMENT),
        feed=DataFeed(FEED),
    )
    response = client.get_stock_bars(request)
    raw = getattr(response, "df", None)
    if not isinstance(raw, pd.DataFrame):
        raise RuntimeError(f"{ticker}: vendor response has no DataFrame payload")
    if raw.empty:
        raise ValueError(f"{ticker}: vendor returned no bars")
    return normalize_bars(raw, ticker)


def _resolve_credentials(
    environ: Mapping[str, str] | None = None,
) -> tuple[str | None, str | None]:
    """Read Alpaca keys from the environment, best-effort loading .env first."""
    if environ is None:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            # load_dotenv installed with `refresh` extra; Otherwise env vars still work
            pass
        environ = os.environ
    return environ.get(_KEY_VAR), environ.get(_SECRET_VAR)


def refresh(
    tickers: tuple[str, ...] | None = None,
    data_dir: Path = DATA_DIR,
    start: pd.Timestamp = START,
) -> None:
    """Pull daily bars from Alpaca and rewrite the parquet cache."""
    universe_tickers = {asset.ticker for asset in UNIVERSE}
    if tickers is None:
        tickers = tuple(asset.ticker for asset in UNIVERSE)
    unknown = set(tickers) - universe_tickers
    if unknown:
        raise ValueError(f"tickers not in universe: {sorted(unknown)}")

    api_key, secret_key = _resolve_credentials()
    if not api_key or not secret_key:
        raise RuntimeError(
            f"set {_KEY_VAR} and {_SECRET_VAR} in the environment or a .env file"
        )

    # Last complete UTC day: never cache a bar for a session still in
    # progress, and midnight is always older than the SIP 15-minute rule.
    end = (pd.Timestamp.now(tz="UTC") - _END_SAFETY).normalize()
    data_dir.mkdir(parents=True, exist_ok=True)
    for ticker in tickers:
        bars = _fetch_bars(
            ticker, start=start, end=end, api_key=api_key, secret_key=secret_key
        )
        first = bars.index[0]
        if first > start + pd.Timedelta(days=7):
            print(f"WARNING {ticker}: history starts {first.date()} > {start.date()}")
        bars.to_parquet(bars_path(ticker, data_dir))
        span = f"{bars.index[0].date()}..{bars.index[-1].date()}"
        print(f"{ticker}: {len(bars)} bars {span}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m portlab.data",
        description="Manage the committed daily-bars parquet cache.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="re-pull all bars from Alpaca and rewrite the parquet cache",
    )
    parser.add_argument(
        "--tickers",
        default=None,
        help="comma-separated subset, e.g. XLK,SPY (default: full universe)",
    )
    args = parser.parse_args(argv)
    if not args.refresh:
        parser.error("nothing to do: pass --refresh")
    tickers = (
        tuple(t.strip().upper() for t in args.tickers.split(","))
        if args.tickers
        else None
    )
    refresh(tickers=tickers)


if __name__ == "__main__":
    main()

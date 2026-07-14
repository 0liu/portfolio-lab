"""Preprocessing: aligned close panel and daily simple returns from cached bars.

Design: pure functions over already-loaded bar frames (loading stays in
portlab.data, so everything here is trivially testable with synthetic data).

Vendor daily bars are stamped midnight America/New_York expressed in UTC;
panels use naive NY trading dates.

Missingness policy:
leading/trailing gaps are tolerated (no data = no position downstream)
interior holes are impossible states and raise.
"""

from collections.abc import Mapping

import pandas as pd


def to_trading_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Convert UTC bar timestamps to naive New York trading dates."""
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise ValueError("Expected a tz-aware DatetimeIndex")
    dates = index.tz_convert("America/New_York").normalize().tz_localize(None)
    if dates.has_duplicates:
        raise ValueError("Bar timestamps collapse to duplicate trading dates")
    return pd.DatetimeIndex(dates, name="date")


def close_panel(bars: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Union-align close prices into a wide (date x ticker) panel.

    Column order follows the mapping's insertion order (UNIVERSE order when
    the frames come from load_universe_bars).
    """
    if not bars:
        raise ValueError("No bar frames given")
    columns = {}
    for ticker, frame in bars.items():
        if "close" not in frame.columns:
            raise ValueError(f"{ticker}: Frame has no close column")
        closes = frame["close"].set_axis(to_trading_dates(frame.index))  # type: ignore
        columns[ticker] = closes
    panel = pd.DataFrame(columns).sort_index()
    _check_interior_gaps(panel)
    return panel


def _check_interior_gaps(panel: pd.DataFrame) -> None:
    """Raise if any ticker has missing closes strictly inside its lifespan."""
    for ticker in panel.columns:
        col = panel[ticker]
        inner = col.loc[col.first_valid_index() : col.last_valid_index()]
        if inner.isna().any():
            holes = inner.index[inner.isna()]
            preview = ", ".join(str(d.date()) for d in holes[:3])
            raise ValueError(
                f"{ticker}: {len(holes)} interior missing close(s), e.g. {preview}"
            )


def daily_returns(closes: pd.DataFrame) -> pd.DataFrame:
    """
    Simple daily returns from a close price panel.
    """
    if closes.empty:
        raise ValueError("Empty close price panel")

    # fill_method=None ensures a missing close price yields a missing return,
    # and returns are never fabricated by forward-filling through gaps.
    returns = closes.pct_change(fill_method=None)

    # The all-NaN first row is dropped.
    return returns.iloc[1:]

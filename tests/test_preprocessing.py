"""DataPreprocessing tests"""

import pandas as pd
import pytest

from portlab.data import bars_path, load_universe_bars
from portlab.preprocessing import (
    close_panel,
    daily_returns,
    to_trading_dates,
)


def make_bars(closes: list[float], start: str = "2016-01-04") -> pd.DataFrame:
    """Synthetic validated-shape bars with vendor-style UTC midnight-ET stamps."""
    idx = pd.date_range(start, periods=len(closes), freq="B", name="timestamp")
    idx = idx.tz_localize("America/New_York").tz_convert("UTC")
    idx = pd.DatetimeIndex(idx.values, tz="UTC", name="timestamp")
    close = pd.Series(closes, index=idx, dtype="float64")
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000_000.0,
        },
        index=idx,
    )


# ------------------------------------------------------------- trading dates


def test_to_trading_dates_maps_utc_stamps_to_ny_dates():
    # Winter stamp (05:00Z = midnight EST) and summer stamp (04:00Z = midnight
    # EDT) must both land on their own calendar trading date.
    idx = pd.DatetimeIndex(
        ["2016-01-04 05:00:00+00:00", "2016-07-05 04:00:00+00:00"], name="timestamp"
    )
    dates = to_trading_dates(idx)
    assert dates.tz is None
    assert dates.name == "date"
    assert list(dates) == [pd.Timestamp("2016-01-04"), pd.Timestamp("2016-07-05")]


def test_to_trading_dates_rejects_naive_index():
    idx = pd.DatetimeIndex(["2016-01-04"])
    with pytest.raises(ValueError, match="tz-aware"):
        to_trading_dates(idx)


def test_to_trading_dates_rejects_date_collisions():
    idx = pd.DatetimeIndex(["2016-01-04 05:00:00+00:00", "2016-01-04 06:00:00+00:00"])
    with pytest.raises(ValueError, match="duplicate trading dates"):
        to_trading_dates(idx)


# --------------------------------------------------------------- close panel


def test_close_panel_full_overlap():
    bars = {
        "AAA": make_bars([100.0, 110.0, 99.0]),
        "BBB": make_bars([50.0, 51.0, 52.0]),
    }
    panel = close_panel(bars)
    assert list(panel.columns) == ["AAA", "BBB"]
    assert panel.shape == (3, 2)
    assert isinstance(panel.index, pd.DatetimeIndex)
    assert panel.index.tz is None
    assert panel["AAA"].tolist() == [100.0, 110.0, 99.0]
    assert panel["BBB"].tolist() == [50.0, 51.0, 52.0]


def test_close_panel_ragged_start_gives_leading_nan():
    bars = {
        "AAA": make_bars([100.0, 101.0, 102.0, 103.0], start="2016-01-04"),
        "BBB": make_bars([50.0, 51.0], start="2016-01-06"),
    }
    panel = close_panel(bars)
    assert panel.shape == (4, 2)
    assert panel["BBB"].isna().tolist() == [True, True, False, False]


def test_close_panel_interior_hole_raises():
    aaa = make_bars([100.0, 101.0, 102.0])
    bbb = make_bars([50.0, 51.0, 52.0]).drop(index=make_bars([1.0, 1.0, 1.0]).index[1])
    with pytest.raises(ValueError, match=r"BBB.*interior missing"):
        close_panel({"AAA": aaa, "BBB": bbb})


def test_close_panel_empty_mapping_raises():
    with pytest.raises(ValueError, match="No bar frames given"):
        close_panel({})


def test_close_panel_missing_close_column_raises():
    bad = make_bars([100.0, 101.0]).drop(columns="close")
    with pytest.raises(ValueError, match="no close column"):
        close_panel({"AAA": bad})


# ------------------------------------------------------------- daily returns


def test_daily_returns_hand_computed():
    panel = close_panel({"AAA": make_bars([100.0, 110.0, 99.0])})
    returns = daily_returns(panel)
    expected = [110.0 / 100.0 - 1.0, 99.0 / 110.0 - 1.0]
    assert returns["AAA"].tolist() == expected
    assert len(returns) == len(panel) - 1  # first all-NaN row dropped


def test_daily_returns_ragged_start_stays_nan():
    bars = {
        "AAA": make_bars([100.0, 101.0, 102.0, 103.0], start="2016-01-04"),
        "BBB": make_bars([50.0, 55.0], start="2016-01-06"),
    }
    returns = daily_returns(close_panel(bars))
    # BBB: no return on its first valid close, real return the day after.
    assert returns["BBB"].isna().tolist() == [True, True, False]
    assert returns["BBB"].iloc[-1] == 55.0 / 50.0 - 1.0


def test_daily_returns_never_forward_fills():
    closes = pd.DataFrame(
        {"AAA": [100.0, float("nan"), 102.0]},
        index=pd.DatetimeIndex(["2016-01-04", "2016-01-05", "2016-01-06"]),
    )
    returns = closes.pipe(daily_returns)
    # Return over a hole is NaN on both the hole and the day after.
    assert returns["AAA"].isna().tolist() == [True, True]


def test_daily_returns_empty_raises():
    with pytest.raises(ValueError, match="Empty close price panel"):
        daily_returns(pd.DataFrame())


# --------------------------------------------------------------- end-to-end


def test_cache_to_returns_end_to_end(tmp_path):
    make_bars([100.0, 110.0, 99.0]).to_parquet(bars_path("XLK", tmp_path))
    make_bars([200.0, 202.0, 200.0]).to_parquet(bars_path("SPY", tmp_path))
    frames = load_universe_bars(tickers=("XLK", "SPY"), data_dir=tmp_path)
    returns = daily_returns(close_panel(frames))
    assert list(returns.columns) == ["XLK", "SPY"]
    assert returns.shape == (2, 2)
    assert returns.notna().all().all()

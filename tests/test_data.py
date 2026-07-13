"""Data layer tests, fully offline.

Refresh is mocked at the module's own seam (_fetch_bars), so CI never touches
the network or imports alpaca-py. The validation invariants are the
specification of what a healthy cache file looks like.
"""

import pandas as pd
import pytest

import portlab.data as data
from portlab.data import (
    COLUMNS,
    START,
    bars_path,
    load_bars,
    load_universe_bars,
    main,
    normalize_bars,
    refresh,
    validate_bars,
)


def make_bars(n: int = 5, start: str = "2016-01-04") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="B", tz="UTC", name="timestamp")
    idx = pd.DatetimeIndex(idx.values, tz="UTC", name="timestamp")
    close = pd.Series(range(n), index=idx, dtype="float64") + 100.0
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000_000.0,
        },
        index=idx,
    )


# ---------------------------------------------------------------- validation


def _break_columns(bars: pd.DataFrame) -> pd.DataFrame:
    return bars.drop(columns="volume")


def _break_tz(bars: pd.DataFrame) -> pd.DataFrame:
    out = bars.copy()
    out.index = pd.DatetimeIndex(out.index).tz_localize(None)
    return out


def _break_order(bars: pd.DataFrame) -> pd.DataFrame:
    return bars.iloc[::-1]


def _break_dupes(bars: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([bars, bars.iloc[[0]]]).sort_index()


def _break_nan(bars: pd.DataFrame) -> pd.DataFrame:
    out = bars.copy()
    out.loc[out.index[0], "close"] = float("nan")
    return out


def _break_price(bars: pd.DataFrame) -> pd.DataFrame:
    out = bars.copy()
    out.loc[out.index[0], "close"] = 0.0
    return out


def _break_volume(bars: pd.DataFrame) -> pd.DataFrame:
    out = bars.copy()
    out.loc[out.index[0], "volume"] = -1.0
    return out


def _break_hilo(bars: pd.DataFrame) -> pd.DataFrame:
    out = bars.copy()
    out.loc[out.index[0], "high"] = out["low"].iloc[0] - 1.0
    return out


def _break_empty(bars: pd.DataFrame) -> pd.DataFrame:
    return bars.iloc[0:0]


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        pytest.param(_break_columns, "expected columns", id="missing-column"),
        pytest.param(_break_tz, "tz-aware UTC", id="naive-index"),
        pytest.param(_break_order, "strictly increasing", id="unsorted"),
        pytest.param(_break_dupes, "strictly increasing", id="duplicates"),
        pytest.param(_break_nan, "NaN", id="nan"),
        pytest.param(_break_price, "non-positive prices", id="zero-price"),
        pytest.param(_break_volume, "negative volume", id="negative-volume"),
        pytest.param(_break_hilo, "high < low", id="high-below-low"),
        pytest.param(_break_empty, "no rows", id="empty"),
    ],
)
def test_validate_bars_fails_loud(corrupt, message):
    with pytest.raises(ValueError, match=message):
        validate_bars(corrupt(make_bars()), "XLK")


def test_validate_bars_passes_healthy_frame():
    bars = make_bars()
    assert validate_bars(bars, "XLK") is bars


# ------------------------------------------------------------ load roundtrip


def test_parquet_roundtrip(tmp_path):
    bars = make_bars()
    path = bars_path("XLK", tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(path)
    pd.testing.assert_frame_equal(load_bars("XLK", tmp_path), bars)


def test_load_missing_file_points_to_refresh(tmp_path):
    with pytest.raises(FileNotFoundError, match="--refresh"):
        load_bars("XLK", tmp_path)


def test_load_corrupt_file_fails_loud(tmp_path):
    _break_columns(make_bars()).to_parquet(bars_path("XLK", tmp_path))
    with pytest.raises(ValueError, match="expected columns"):
        load_bars("XLK", tmp_path)


def test_load_universe_bars_defaults_to_full_universe(tmp_path):
    from portlab.universe import UNIVERSE

    for asset in UNIVERSE:
        make_bars().to_parquet(bars_path(asset.ticker, tmp_path))
    frames = load_universe_bars(data_dir=tmp_path)
    assert set(frames) == {asset.ticker for asset in UNIVERSE}


# ---------------------------------------------------------------- normalize


def _vendor_frame(ticker: str = "XLK") -> pd.DataFrame:
    bars = make_bars()
    raw = bars.assign(trade_count=10.0, vwap=bars["close"])
    raw = raw.loc[:, ["vwap", "close", "low", "high", "open", "volume", "trade_count"]]
    raw.index = pd.MultiIndex.from_product(
        [[ticker], bars.index], names=["symbol", "timestamp"]
    )
    return raw.iloc[::-1]  # vendor order not guaranteed


def test_normalize_bars_from_vendor_shape():
    result = normalize_bars(_vendor_frame(), "XLK")
    pd.testing.assert_frame_equal(result, make_bars())
    assert tuple(result.columns) == COLUMNS


def test_normalize_bars_missing_column_fails():
    raw = _vendor_frame().drop(columns="close")
    with pytest.raises(ValueError, match="missing columns"):
        normalize_bars(raw, "XLK")


# ------------------------------------------------------------------ refresh


def test_refresh_writes_loadable_cache(tmp_path, monkeypatch):
    calls: list[dict] = []

    def fake_fetch(ticker, start, end, api_key, secret_key):
        calls.append({"ticker": ticker, "start": start, "end": end, "key": api_key})
        return make_bars()

    monkeypatch.setattr(data, "_fetch_bars", fake_fetch)
    monkeypatch.setattr(data, "_resolve_credentials", lambda: ("k", "s"))

    refresh(tickers=("XLK", "SPY"), data_dir=tmp_path)

    assert [c["ticker"] for c in calls] == ["XLK", "SPY"]
    for call in calls:
        assert call["start"] == START
        assert call["end"] <= pd.Timestamp.now(tz="UTC") - pd.Timedelta(minutes=15)
        assert call["key"] == "k"
    pd.testing.assert_frame_equal(load_bars("XLK", tmp_path), make_bars())
    pd.testing.assert_frame_equal(load_bars("SPY", tmp_path), make_bars())


def test_refresh_without_credentials_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "_resolve_credentials", lambda: (None, None))
    with pytest.raises(RuntimeError, match="ALPACA_API_KEY"):
        refresh(tickers=("XLK",), data_dir=tmp_path)


def test_refresh_rejects_tickers_outside_universe(tmp_path):
    with pytest.raises(ValueError, match="not in universe"):
        refresh(tickers=("AAPL",), data_dir=tmp_path)


def test_resolve_credentials_reads_mapping():
    env = {"ALPACA_API_KEY": "k", "ALPACA_SECRET_KEY": "s"}
    assert data._resolve_credentials(env) == ("k", "s")
    assert data._resolve_credentials({}) == (None, None)


# ---------------------------------------------------------------------- CLI


def test_main_parses_tickers_and_calls_refresh(monkeypatch):
    received: dict = {}
    monkeypatch.setattr(data, "refresh", lambda tickers: received.update(t=tickers))
    main(["--refresh", "--tickers", "xlk , spy"])
    assert received["t"] == ("XLK", "SPY")


def test_main_default_is_full_universe(monkeypatch):
    received: dict = {}
    monkeypatch.setattr(data, "refresh", lambda tickers: received.update(t=tickers))
    main(["--refresh"])
    assert received["t"] is None


def test_main_without_refresh_errors():
    with pytest.raises(SystemExit):
        main([])

"""Signal tests per the frozen spec, including the four locked assertions:

1. PIT tripwire: day-t data must never move day-t (or earlier) signals.
2. Broad decline: tsmom all negative; composite all negative (level channel
   penetrates the combine).
3. xsmom: per-day mean exactly 0; even n + no ties -> exactly half positive /
   half negative; bounds +-1 attained by best/worst.
4. st_rev: the past-5d winner gets a negative score.
"""

import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from portlab.config import SignalConfig
from portlab.estimation import ewma_std
from portlab.signals import (
    _xs_zscore,
    combine,
    composite_scores,
    expected_returns,
    sigma_daily,
    st_rev,
    tsmom,
    xsmom,
)

SMALL_CFG = SignalConfig(
    tsmom_windows=(5, 10),
    signal_vol_halflife_days=5,
    xs_lookback=15,
    xs_exclude=3,
    reversal_window=5,
)


def panel(data: dict[str, list[float]]) -> pd.DataFrame:
    n = len(next(iter(data.values())))
    idx = pd.bdate_range("2016-01-04", periods=n, name="date")
    return pd.DataFrame(data, index=idx, dtype="float64")


def random_walk(n: int, tickers: str, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.01, size=(n, len(tickers)))
    return panel(
        {
            t: (100.0 * np.cumprod(1.0 + rets[:, i])).tolist()
            for i, t in enumerate(tickers)
        }
    )


def trending_panel(n: int, slopes: tuple[float, ...]) -> pd.DataFrame:
    """Deterministic per-asset drifts + alternating jitter (keeps sigma > 0)."""
    t = np.arange(n)[:, None]
    rets = np.array(slopes)[None, :] + 0.0005 * np.where(t % 2 == 0, 1.0, -1.0)
    return panel(
        {
            f"A{i}": (100.0 * np.cumprod(1.0 + rets[:, i])).tolist()
            for i in range(len(slopes))
        }
    )


# ------------------------------------------------------------- _xs_zscore


def test_xs_zscore_frozen_edge_cases():
    idx = pd.bdate_range("2016-01-04", periods=3)
    frame = pd.DataFrame(
        {
            "A": [1.0, 1.0, 5.0],
            "B": [2.0, 1.0, float("nan")],
            "C": [3.0, float("nan"), float("nan")],
        },
        index=idx,
    )
    z = _xs_zscore(frame)
    # normal row: std ddof=1 of (1,2,3) is exactly 1
    assert z.iloc[0].tolist() == [-1.0, 0.0, 1.0]
    # flat cross-section: no relative information -> 0, NaN stays NaN
    assert z.iloc[1]["A"] == 0.0
    assert z.iloc[1]["B"] == 0.0
    assert math.isnan(z.iloc[1]["C"])
    # n_t < 2 -> NaN
    assert z.iloc[2].isna().all()


# ------------------------------------------------------------- sigma_daily


def test_sigma_daily_matches_formula_and_min_periods():
    closes = random_walk(40, "AB")
    halflife = 5
    expected = ewma_std(
        closes.pct_change(fill_method=None), halflife, min_periods=2 * halflife
    ).shift(1)
    pd.testing.assert_frame_equal(sigma_daily(closes, halflife), expected)
    # returns start at row 1 -> 10th return lands at row 10 -> lagged: row 11
    assert sigma_daily(closes, halflife)["A"].first_valid_index() == closes.index[11]


def test_sigma_daily_is_lagged_one_day():
    closes = random_walk(40, "AB")
    shocked = closes.copy()
    shocked.iloc[-1] = shocked.iloc[-1] * 3.0
    base = sigma_daily(closes, 5)
    after = sigma_daily(shocked, 5)
    # day-t sigma uses data through t-1: the final close cannot move it
    pd.testing.assert_frame_equal(base, after)


# ------------------------------------------------------------------- tsmom


def test_tsmom_matches_frozen_formula():
    closes = random_walk(80, "ABC")
    halflife = 5
    sigma = ewma_std(
        closes.pct_change(fill_method=None), halflife, min_periods=2 * halflife
    ).shift(1)
    expected = (
        closes.shift(1).pct_change(5, fill_method=None) / (sigma * math.sqrt(5))
        + closes.shift(1).pct_change(10, fill_method=None) / (sigma * math.sqrt(10))
    ) / 2
    pd.testing.assert_frame_equal(tsmom(closes, (5, 10), halflife), expected)


def test_tsmom_all_negative_in_broad_decline():
    closes = trending_panel(60, (-0.004, -0.003, -0.002, -0.001))
    result = tsmom(closes, SMALL_CFG.tsmom_windows, SMALL_CFG.signal_vol_halflife_days)
    last = result.iloc[-1]
    assert last.notna().all()
    assert (last < 0).all()  # absolute character survives: no forced winners


def test_tsmom_all_positive_in_broad_rally():
    closes = trending_panel(60, (0.001, 0.002, 0.003, 0.004))
    result = tsmom(closes, SMALL_CFG.tsmom_windows, SMALL_CFG.signal_vol_halflife_days)
    last = result.iloc[-1]
    assert last.notna().all()
    assert (last > 0).all()


def test_tsmom_rejects_bad_windows():
    closes = random_walk(20, "AB")
    with pytest.raises(ValueError, match="non-empty"):
        tsmom(closes, (), 5)
    with pytest.raises(ValueError, match="positive"):
        tsmom(closes, (5, 0), 5)


# ------------------------------------------------------------------- xsmom


def test_xsmom_hand_computed_two_assets():
    # m(t) = P(t-1-1) / P(t-1-3) - 1 with lookback=3, exclude=1
    closes = panel(
        {
            "A": [100.0, 110.0, 121.0, 133.1, 146.41],
            "B": [100.0, 101.0, 102.0, 103.0, 104.0],
        }
    )
    result = xsmom(closes, lookback=3, exclude=1)
    # row 4: m_A = 133.1/100 - 1, m_B = 103/100 - 1 -> ranks (2, 1) -> (+1, -1)
    assert result["A"].iloc[4] == 1.0
    assert result["B"].iloc[4] == -1.0


def test_xsmom_locked_cross_sectional_properties():
    closes = random_walk(320, "ABCD", seed=11)
    result = xsmom(closes, lookback=252, exclude=21)
    valid = result[result.notna().all(axis=1)]
    assert len(valid) > 0
    # mean exactly 0 every day (to float tolerance)
    assert np.allclose(valid.mean(axis=1), 0.0, atol=1e-12)
    # even n, no ties: exactly half positive / half negative
    assert ((valid > 0).sum(axis=1) == 2).all()
    assert ((valid < 0).sum(axis=1) == 2).all()
    # bounds are exactly [-1, +1], attained by the best and worst asset each day
    # (random-walk -> no ties -> best rank = n, worst = 1; tolerance tests the
    #  mathematical property, not the implementation's float path)
    assert np.allclose(valid.max(axis=1), 1.0, atol=1e-12)
    assert np.allclose(valid.min(axis=1), -1.0, atol=1e-12)


def test_xsmom_tie_handling_preserves_zero_mean():
    closes = panel(
        {
            "A": [100.0, 100.0, 100.0, 100.0, 110.0],
            "B": [100.0, 100.0, 100.0, 100.0, 105.0],
            "C": [50.0, 50.0, 50.0, 50.0, 51.0],
        }
    )
    # lookback=3, exclude=1: row 4 uses P(2)/P(0) - 1 = 0 for all -> 3-way tie
    result = xsmom(closes, lookback=3, exclude=1)
    row = result.iloc[4]
    assert row.tolist() == [0.0, 0.0, 0.0]  # average ranks -> centered -> 0


def test_xsmom_excluded_window_cannot_move_it():
    base = random_walk(30, "ABC")
    shocked = base.copy()
    shocked.iloc[-4:] = shocked.iloc[-4:] * 1.7  # inside exclude+1 = 4 last rows
    a = xsmom(base, lookback=15, exclude=3)
    b = xsmom(shocked, lookback=15, exclude=3)
    assert a.iloc[-1].equals(b.iloc[-1])


def test_xsmom_rejects_bad_exclude():
    with pytest.raises(ValueError, match="exclude"):
        xsmom(panel({"A": [1.0]}), lookback=4, exclude=4)


# ------------------------------------------------------------------ st_rev


def test_st_rev_past_winner_scores_negative():
    flat = [100.0] * 8
    winner = [100.0, 100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 112.0]
    closes = panel({"W": winner, "F1": flat, "F2": [100.0] * 7 + [99.0]})
    result = st_rev(closes, window=5, clip=2.0)
    last = result.iloc[-1]
    assert last["W"] < 0  # locked assertion: past-5d winner gets negative score
    assert last["W"] == last.min()


def test_st_rev_is_clipped_z_of_negative_past_return():
    closes = random_walk(30, "ABCDEF", seed=3)
    rev = -closes.shift(1).pct_change(5, fill_method=None)
    expected = _xs_zscore(rev).clip(-0.5, 0.5)
    pd.testing.assert_frame_equal(st_rev(closes, window=5, clip=0.5), expected)
    assert st_rev(closes, 5, 0.5).abs().max().max() <= 0.5


def test_st_rev_rejects_bad_params():
    closes = random_walk(10, "AB")
    with pytest.raises(ValueError, match="positive"):
        st_rev(closes, window=0, clip=2.0)
    with pytest.raises(ValueError, match="positive"):
        st_rev(closes, window=5, clip=0.0)


# ----------------------------------------------------------------- combine


def test_combine_hand_computed():
    idx = pd.bdate_range("2016-01-04", periods=1)
    cols = ["A", "B", "C"]
    ts = pd.DataFrame([[3.0, 0.5, -1.0]], index=idx, columns=cols)  # clips to 2
    xs = pd.DataFrame([[-1.0, 0.0, 1.0]], index=idx, columns=cols)
    rev = pd.DataFrame([[1.0, -1.0, 0.0]], index=idx, columns=cols)

    # default scale = 2.0
    r = combine(ts, xs, rev, clip=2.0)
    assert r.iloc[0]["A"] == pytest.approx((2.0 - 2.0 + 1.0) / 3)
    assert r.iloc[0]["B"] == pytest.approx((0.5 + 0.0 - 1.0) / 3)
    assert r.iloc[0]["C"] == pytest.approx((-1.0 + 2.0 + 0.0) / 3)

    # scale != clip: proves the two constants are wired to different args.
    # A tsmom clip stays 2 (clip=2), but xsmom now scales by 3.
    r3 = combine(ts, xs, rev, clip=2.0, xsmom_scale=3.0)
    assert r3.iloc[0]["A"] == pytest.approx((2.0 + 3.0 * -1.0 + 1.0) / 3)  # (2-3+1)/3=0
    assert r3.iloc[0]["C"] == pytest.approx((-1.0 + 3.0 * 1.0 + 0.0) / 3)  # (-1+3)/3


def test_combine_is_not_skipna():
    idx = pd.bdate_range("2016-01-04", periods=1)
    cols = ["A", "B"]
    ts = pd.DataFrame([[float("nan"), 1.0]], index=idx, columns=cols)
    xs = pd.DataFrame([[1.0, 0.5]], index=idx, columns=cols)
    rev = pd.DataFrame([[1.0, 0.5]], index=idx, columns=cols)
    result = combine(ts, xs, rev, clip=2.0)
    assert math.isnan(result.iloc[0]["A"])  # incomplete information: no view
    assert result.iloc[0]["B"] == pytest.approx((1.0 + 1.0 + 0.5) / 3)


def test_combine_misaligned_raises():
    idx = pd.bdate_range("2016-01-04", periods=1)
    a = pd.DataFrame({"A": [1.0]}, index=idx)
    b = pd.DataFrame({"B": [1.0]}, index=idx)
    with pytest.raises(ValueError, match="not aligned"):
        combine(a, b, a, clip=2.0)


# ------------------------------------------------- composite and pipeline


def test_composite_all_negative_in_broad_decline():
    closes = trending_panel(60, (-0.004, -0.003, -0.002, -0.001))
    comp = composite_scores(closes, SMALL_CFG)
    last = comp.iloc[-1]
    assert last.notna().all()
    assert (last < 0).all()  # locked: the level channel penetrates the combine


def test_composite_warmup_is_longest_window_plus_one():
    closes = random_walk(300, "ABCD")
    comp = composite_scores(closes, SignalConfig())
    # r_252(t) = P(t-1)/P(t-253): first valid at row 253; vol warmup subsumed
    assert comp.iloc[:253].isna().all().all()
    assert comp.iloc[253].notna().all()


def test_expected_returns_is_composite_times_sigma():
    closes = random_walk(80, "ABC")
    cfg = SMALL_CFG
    expected = composite_scores(closes, cfg) * sigma_daily(
        closes, cfg.signal_vol_halflife_days
    )
    pd.testing.assert_frame_equal(expected_returns(closes, cfg), expected)


def test_composite_respects_xsmom_scale():
    closes = random_walk(300, "ABCD")
    cfg_a = SignalConfig()  # xsmom_scale=2.0
    cfg_b = dataclasses.replace(cfg_a, xsmom_scale=4.0)
    a = composite_scores(closes, cfg_a)
    b = composite_scores(closes, cfg_b)
    assert not a.equals(b)  # pipeline actually threads xsmom_scale through


# --------------------------------------------------------------- tripwires


def test_lookahead_tripwire_interior_day():
    """Mutating day-t closes must not move any signal at or before day t."""
    closes = random_walk(400, "ABCD")
    cfg = SignalConfig()
    base = composite_scores(closes, cfg)

    mutated = closes.copy()
    mutated.iloc[300] = mutated.iloc[300] * 1.5  # violent shock on day 300
    after = composite_scores(mutated, cfg)

    pd.testing.assert_frame_equal(base.iloc[:301], after.iloc[:301])
    # the shock must show up afterwards, or this test tests nothing
    assert not base.iloc[301:].equals(after.iloc[301:])


def test_lookahead_tripwire_last_day():
    """Mutating the final close changes NOTHING: day t never sees close t."""
    closes = random_walk(400, "ABCD")
    cfg = SignalConfig()
    base = composite_scores(closes, cfg)

    mutated = closes.copy()
    mutated.iloc[-1] = mutated.iloc[-1] * 2.0
    after = composite_scores(mutated, cfg)

    pd.testing.assert_frame_equal(base, after)


def test_lookahead_tripwire_mu():
    """expected_returns inherits the PIT property end to end."""
    closes = random_walk(400, "ABCD")
    cfg = SignalConfig()
    base = expected_returns(closes, cfg)

    mutated = closes.copy()
    mutated.iloc[-1] = mutated.iloc[-1] * 2.0
    after = expected_returns(mutated, cfg)

    pd.testing.assert_frame_equal(base, after)

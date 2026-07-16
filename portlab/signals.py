"""
Signal layer: Daily-updating time-series and cross-sectional signals

- tsmom
    Time-series momentum, vol-scaled trailing 63/126/252-day returns,
    averaged across horizons.
- xsmom
    Cross-sectional momentum, trailing 252d return excluding the last 21d,
    ranked across assets.
- st_rev
    Short-term reversal, negative of the past 5-day return, genuine daily-horizon alpha.
- Differential combine
    tsmom, xsmom, st_rev are combined with equal signal weights into a composite score.
    Strictly elementwise and deliberately NOT uniform z-scoring.
- Expected-return proxy. The combined score scaled by the asset's volatility.
    mu_i = score_i * sigma_i
- tsmom is a non-zero mean absolute exposure channel per asset, scaled by its own noise.
- xsmom / st_rev are zero-mean relative channels, per-day cross-sectional.

PIT (point-in-time) contract: Every signal value for trading day t uses prices
through the t-1 only. The shifts are baked into each signal's definition (not
deferred to downstream), guarded by the look-ahead tripwire tests.
"""

import math
from collections.abc import Sequence

import pandas as pd

from portlab.config import SignalConfig
from portlab.estimation import ewma_std


def _pit_return(closes: pd.DataFrame, window: int) -> pd.DataFrame:
    """P(t-1) / P(t-1-window) - 1: trailing return through the prior close."""
    return closes.shift(1).pct_change(window, fill_method=None)


def _xs_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-day cross-sectional z-score.

    - Cross-sectional = across assets j at fixed day t (axis=1); std ddof=1.
    - std = 0 -> 0 (a flat cross-section carries no relative information).
    - num_of_assets < 2 -> NaN; NaN in -> NaN out.
    """
    mean = frame.mean(axis=1)
    std = frame.std(axis=1, ddof=1)  # NaN when num_of_assets < 2
    z = frame.sub(mean, axis=0).div(std.where(std != 0.0), axis=0)
    flat = std.eq(0.0)
    if flat.any():
        z.loc[flat] = frame.loc[flat] * 0.0  # 0 where valid, NaN stays NaN
    return z


def sigma_daily(closes: pd.DataFrame, halflife: int) -> pd.DataFrame:
    """
    Per-asset EWMA std of daily simple returns, lagged one day (PIT).

    Single estimate shared by all tsmom horizons (an asset attribute, not a
    horizon attribute); min_periods is 2x halflife because shorter EWMA stds
    are unreliable — no extra waiting cost, tsmom needs 252d anyway.
    """
    daily = closes.pct_change(fill_method=None)
    return ewma_std(daily, halflife, min_periods=2 * halflife).shift(1)


def tsmom(
    closes: pd.DataFrame,
    windows: Sequence[int],
    vol_halflife: int,
    sigma: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Multi-horizon time-series momentum: r_k / (sigma_daily * sqrt(k)),
    averaged over the specified windows.

    - Each asset is measured against its own noise (MOP 2012), never ranked
      against peers. In a broad rally all scores may be positive.
    - sqrt(k) does the time aggregation (i.i.d. approximation). The error is
      co-directional across horizons and largely cancels in the equal-weight
      average.
    - NaN until the longest window has history.
    """
    if not windows:
        raise ValueError("windows must be non-empty")
    if any(w <= 0 for w in windows):
        raise ValueError(f"windows must be positive, got {tuple(windows)}")
    if sigma is None:
        sigma = sigma_daily(closes, vol_halflife)
    scores = [_pit_return(closes, k) / (sigma * math.sqrt(k)) for k in windows]
    total = scores[0]
    for score in scores[1:]:
        total = total + score
    return total / len(scores)


def xsmom(closes: pd.DataFrame, lookback: int, exclude: int) -> pd.DataFrame:
    """
    Cross-sectional momentum as a symmetric centered rank.

    m(i,t)     = P(i, t-1-exclude) / P(i, t-1-lookback) - 1
    rank(i,t)  = tie-averaged rank of m(i,t) among non-NaN assets (1 = worst)
    xsmom(i,t) = 2 * (rank - 1) / (n - 1) - 1
    n: number of non-NaN assets at time t.
    """
    if not 0 <= exclude < lookback:
        raise ValueError(f"need 0 <= exclude < lookback, got {exclude}, {lookback}")
    m = _pit_return(closes.shift(exclude), lookback - exclude)
    rank = m.rank(axis=1, method="average")
    n = m.notna().sum(axis=1)
    # map rank in [1, n] to [-1, +1] with 0 mean and unit range
    return rank.sub(1).mul(2).div(n - 1, axis=0).sub(1)


def st_rev(closes: pd.DataFrame, window: int, clip: float) -> pd.DataFrame:
    """
    Short-term reversal, +/- clip(xs_z(-past return)).

    Reversal is a relative measure across assets, so normalized with per-day
    cross-sectional z-score. The clip guards the composite average against
    one-day blowouts.
    """
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    if clip <= 0:
        raise ValueError(f"clip must be positive, got {clip}")
    rev = -_pit_return(closes, window)
    return _xs_zscore(rev).clip(-clip, clip)


def combine(
    tsmom_score: pd.DataFrame,
    xsmom_score: pd.DataFrame,
    st_rev_score: pd.DataFrame,
    clip: float = 2.0,
    xsmom_scale: float = 2.0,
) -> pd.DataFrame:
    """composite = (clip(tsmom) + xsmom_scale * xsmom + st_rev) / 3, elementwise.

    Clip asymmetry is intentional, not an oversight:
    - tsmom enters unclipped from its own function. It has no natural bound --
      a strong multi-horizon trend can exceed +-2 sigma, so it is clipped to cap
      its composite contribution.
    - st_rev is already clipped inside st_rev(). The clip belongs right after
      the cross-sectional z-score, so it enters pre-bounded and needs no second clip.
    - xsmom is bounded to [-1, 1] by construction (centered rank), so it is
      never clipped, only scaled by xsmom_scale to match the +-2 component
      range.

    clip and xsmom_scale are independent parameters despite sharing the default
    value 2.0: clip is a blowout guard on tsmom, xsmom_scale is a scale-matching
    factor on xsmom.

    Deliberately NOT skipna: composite is NaN unless all three signals are
    present. Incomplete information means no view or position.
    """
    if clip <= 0:
        raise ValueError(f"clip must be positive, got {clip}")
    first = tsmom_score
    for frame in (xsmom_score, st_rev_score):
        aligned = frame.index.equals(first.index) and frame.columns.equals(
            first.columns
        )
        if not aligned:
            raise ValueError("signal frames are not aligned")
    return (
        tsmom_score.clip(-clip, clip) + xsmom_scale * xsmom_score + st_rev_score
    ) / 3.0


def composite_scores(
    closes: pd.DataFrame, cfg: SignalConfig, sigma: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Full signal pipeline: the three PIT signals combined per the spec."""
    if sigma is None:
        sigma = sigma_daily(closes, cfg.signal_vol_halflife_days)
    return combine(
        tsmom(closes, cfg.tsmom_windows, cfg.signal_vol_halflife_days, sigma=sigma),
        xsmom(closes, cfg.xs_lookback, cfg.xs_exclude),
        st_rev(closes, cfg.reversal_window, cfg.clip),
        clip=cfg.clip,
        xsmom_scale=cfg.xsmom_scale,
    )


def expected_returns(closes: pd.DataFrame, cfg: SignalConfig) -> pd.DataFrame:
    """
    The expected returns are a proxy with correct cross-sectional sign and
    relative magnitude, NOT a calibrated forecast.
        mu = composite * sigma_daily
    """
    sigma = sigma_daily(closes, cfg.signal_vol_halflife_days)
    comp = composite_scores(closes, cfg, sigma=sigma)
    return comp * sigma

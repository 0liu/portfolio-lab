"""Estimation layer

Covariance estimation with estimation-error awareness.

* ewma_cov       -- exponentially weighted covariance (halflife param)
* ledoit_wolf    -- shrinkage toward a constant-correlation target,
                    implemented from the paper (not a library call);
                    validated in tests against sklearn's LedoitWolf.

Why shrinkage: with ~20 assets and a 3-year window the sample
covariance is noisy and its inverse (which MVO uses) amplifies that
noise. Shrinkage trades a little bias for a large variance reduction.
"""

import pandas as pd


def ewma_std(
    returns: pd.DataFrame, halflife: int, min_periods: int | None = None
) -> pd.DataFrame:
    """EWMA std of daily returns; NaN until `min_periods` (default: halflife)."""
    if halflife <= 0:
        raise ValueError(f"halflife must be positive, got {halflife}")
    if min_periods is None:
        min_periods = halflife
    if min_periods <= 0:
        raise ValueError(f"min_periods must be positive, got {min_periods}")
    return returns.ewm(halflife=halflife, min_periods=min_periods).std()

"""Construction layer I: equal_weight, inverse_vol, ERC under one interface.

Design: one dispatch — optimize(name, mu, cov, w_prev, cfg) — so the engine
never special-cases an optimizer. The interface deliberately over-supplies:
simple optimizers ignore the inputs they don't need (all three here ignore
mu and w_prev; cfg is consumed once the MVO family lands in C10). All three
are long-only and fully invested: weights sum to 1.



--- old
Portfolio construction: pluggable optimizers behind one interface.

    optimize(name, mu, cov, w_prev, cfg) -> weights (pd.Series)

Long-only baselines:
* equal_weight   -- 1/N
* inverse_vol    -- w_i proportional to 1/sigma_i
* erc            -- equal risk contribution (scipy SLSQP)

Headliners (cvxpy):
* mvo            -- long-only mean-variance: per-asset cap, fully
                    invested, L1 turnover penalty
                    lambda * ||w - w_prev||_1
* mvo_ls         -- long-short mean-variance: |w_i| <= max_weight,
                    sum|w| <= gross_cap, net exposure in net_range,
                    same turnover penalty.

Costs live inside the objective, not as an afterthought -- at daily
rebalancing this is the difference between an optimizer that survives
costs and one that doesn't (see the lambda-sweep exhibit).

Vol targeting is applied as a portfolio-level overlay after weight
construction: scale gross exposure so trailing portfolio vol hits
cfg.target_vol, cash as residual.

Extensions (separate commits): hrp, black_litterman.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from portlab.config import Config

# Max allowed deviation of any ERC risk contribution from 1/n before the
# solver result is rejected (module-internal solver contract, not a research
# parameter).
_ERC_TOL = 1e-6


def _validate_inputs(
    mu: pd.Series, cov: pd.DataFrame, w_prev: pd.Series | None
) -> np.ndarray:
    """Fail-loud checks shared by every optimizer; returns cov as ndarray."""
    if not cov.index.equals(cov.columns):
        raise ValueError("cov index and columns differ")
    matrix = cov.to_numpy(dtype="float64")
    if np.isnan(matrix).any():
        raise ValueError("NaN in cov")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
        raise ValueError("cov is not symmetric")
    if (np.diag(matrix) <= 0).any():
        raise ValueError("non-positive variance on cov diagonal")
    if not mu.index.equals(cov.index):
        raise ValueError("mu is not aligned with cov")
    if mu.isna().any():
        raise ValueError("NaN in mu")
    if w_prev is not None:
        if not w_prev.index.equals(cov.index):
            raise ValueError("w_prev is not aligned with cov")
        if w_prev.isna().any():
            raise ValueError("NaN in w_prev")
    return matrix


def equal_weight(
    mu: pd.Series, cov: pd.DataFrame, w_prev: pd.Series | None, cfg: Config
) -> pd.Series:
    """w_i = 1/n
    The no-information baseline every other optimizer must beat.
    """
    _validate_inputs(mu, cov, w_prev)
    n = len(cov)
    return pd.Series(np.full(n, 1.0 / n), index=cov.index)


def inverse_vol(
    mu: pd.Series, cov: pd.DataFrame, w_prev: pd.Series | None, cfg: Config
) -> pd.Series:
    """w_i proportional to 1/sigma_i.
    Naive risk parity ignoring correlations.
    """
    matrix = _validate_inputs(mu, cov, w_prev)
    inv = 1.0 / np.sqrt(np.diag(matrix))
    return pd.Series(inv / inv.sum(), index=cov.index)


def erc(
    mu: pd.Series, cov: pd.DataFrame, w_prev: pd.Series | None, cfg: Config
) -> pd.Series:
    """Equal risk contribution (long-only, fully invested), scipy SLSQP.

    Each asset's risk contribution RC_i = w_i (Sigma w)_i sums to portfolio
    variance by Euler's theorem. The ERC portfolio equalizes them, i.e. every
    asset supplies 1/n of total risk (Maillard, Roncalli & Teiletche 2010).
    No closed-form for general Sigma with n > 2, so minimize the dispersion of
    contributions

        f(w) = sum_i (w_i (Sigma w)_i - w'Sigma w / n)^2

    subject to sum(w) = 1, 0 <= w <= 1, with an analytic gradient

        grad f = 2 [ g * m + Sigma (g * w) - (2/n) m sum(g) ],
        m = Sigma w,  g = w * m - w'Sigma w / n.

    ERC weights are invariant to scaling Sigma, so cov is normalized to unit
    mean variance internally: daily-return covariances (~1e-4) would otherwise
    put f near the solver's ftol floor.

    The contract is re-verified against the *unscaled* cov before returning.
    Solver failure or contributions off 1/n raise, because a silently wrong
    allocation is worse than a loud one.

    Analytic special cases the tests pin: n=2 gives inverse-vol for any
    correlation; a diagonal (or constant-correlation) Sigma gives inverse-vol;
    identical assets give equal weights.
    """
    matrix = _validate_inputs(mu, cov, w_prev)
    n = len(matrix)
    if n == 1:
        return pd.Series([1.0], index=cov.index)
    scaled = matrix * (n / np.trace(matrix))

    def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
        marginal = scaled @ w  # MRC_i = (Σw)_i ; shape (n,)
        total = w @ marginal  # w'Σw = σ² ; scalar
        # deviation of each risk contribution from the mean (1/n of total)
        gap = w * marginal - total / n  # RC_i - mean_RC; shape (n,)
        value = float(gap @ gap)  # f(w) = Σ gap_i²
        grad = 2.0 * (
            gap * marginal + scaled @ (gap * w) - (2.0 / n) * marginal * gap.sum()
        )
        return value, grad

    start = 1.0 / np.sqrt(np.diag(scaled))
    start = start / start.sum()
    result = minimize(
        objective,
        start,
        jac=True,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[
            {
                "type": "eq",
                "fun": lambda w: w.sum() - 1.0,
                "jac": lambda w: np.ones(n),
            }
        ],
        options={"maxiter": 1000, "ftol": 1e-16},
    )
    if not result.success:
        raise RuntimeError(f"ERC solver failed: {result.message}")
    weights = result.x / result.x.sum()

    # verify the contract before returning, against the *unscaled* cov
    marginal = matrix @ weights
    contributions = weights * marginal / (weights @ marginal)
    worst = float(np.abs(contributions - 1.0 / n).max())
    if worst > _ERC_TOL:
        raise RuntimeError(
            f"ERC contributions deviate from 1/n by {worst:.2e} (> {_ERC_TOL:.0e})"
        )
    return pd.Series(weights, index=cov.index)


_OPTIMIZERS: dict[
    str, Callable[[pd.Series, pd.DataFrame, pd.Series | None, Config], pd.Series]
] = {
    "equal_weight": equal_weight,
    "inverse_vol": inverse_vol,
    "erc": erc,
}

OPTIMIZER_NAMES: tuple[str, ...] = tuple(sorted(_OPTIMIZERS))


def optimize(
    name: str,
    mu: pd.Series,
    cov: pd.DataFrame,
    w_prev: pd.Series | None,
    cfg: Config,
) -> pd.Series:
    """Single entry point for every portfolio construction rule."""
    if name not in _OPTIMIZERS:
        raise ValueError(f"Unknown optimizer {name!r}. Available: {OPTIMIZER_NAMES}")
    return _OPTIMIZERS[name](mu, cov, w_prev, cfg)

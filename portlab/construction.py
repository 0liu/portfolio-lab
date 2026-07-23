"""Construction layer

All the pluggable optimizers for portfolio construction are behind one interface.

    optimize(name, mu, cov, w_prev, cfg) -> weights (pd.Series)

The interface deliberately over-supplies. Simple optimizers ignore mu and
w_prev. The MVO family consumes everything.

Long-only baselines:
* equal_weight   -- 1/N
* inverse_vol    -- w_i proportional to 1/sigma_i
* erc            -- equal risk contribution (scipy SLSQP)

Headliners (cvxpy):
* mvo            -- long-only mean-variance: per-asset cap, fully invested,
                    L1 turnover penalty lambda * ||w - w_prev||_1
* mvo_ls         -- long-short mean-variance: |w_i| <= max_weight,
                    sum|w| <= gross_cap, net exposure in net_range,
                    same turnover penalty.

Equal_weight / inverse_vol / erc are long-only and fully invested.
mvo is long-only with a position cap; mvo_ls is long-short with position, gross
and net constraints. Both MVOs carry the L1 turnover penalty.

The vol targeting is applied as a portfolio-level overlay after weight
construction. It scales gross exposure so trailing portfolio vol hits
cfg.vol_target_annual, cash as residual.

Costs live inside the objective. See the lambda-sweep exhibit.
"""

from collections.abc import Callable

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from portlab.config import Config

# --- Solver acceptance tolerances (numerical contracts, not research params) ---
# Max allowed deviation of any ERC risk contribution from 1/n before the solver
# result is rejected
ERC_CONTRIBUTION_TOL = 1e-5  # max |RC_i/sigma^2 - 1/n| accepted from SLSQP
QP_BOUND_TOL = 1e-6  # max constraint violation accepted from Clarabel

# Annualization convention (trading days per year) — a contract, not a knob.
TRADING_DAYS = 252


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


def turnover(weights: pd.Series, w_prev: pd.Series | None) -> float:
    """One-way turnover sum |w - w_prev|; w_prev=None means starting from cash."""
    if w_prev is None:
        return float(weights.abs().sum())
    if not weights.index.equals(w_prev.index):
        raise ValueError("w_prev is not aligned with weights")
    return float((weights - w_prev).abs().sum())


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
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"ERC solver failed: {result.message}")
    weights = result.x / result.x.sum()

    # verify the contract before returning, against the *unscaled* cov
    marginal = matrix @ weights
    contributions = weights * marginal / (weights @ marginal)
    worst = float(np.abs(contributions - 1.0 / n).max())
    if worst > ERC_CONTRIBUTION_TOL:
        raise RuntimeError(
            f"ERC contributions deviate from 1/n by {worst:.2e} "
            f"(> {ERC_CONTRIBUTION_TOL:.0e})"
        )
    return pd.Series(weights, index=cov.index)


def _solve_qp(
    objective: cp.Maximize, constraints: list, variable: cp.Variable, label: str
) -> np.ndarray:
    """QP solver. Accept only a clean OPTIMAL status. Return raw solution."""
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL)
    if problem.status != cp.OPTIMAL or variable.value is None:
        raise RuntimeError(f"{label} solver status: {problem.status}")
    return np.asarray(variable.value, dtype="float64")


def _mvo_objective(
    matrix: np.ndarray,
    mu: pd.Series,
    w_prev: pd.Series | None,
    cfg: Config,
    variable: cp.Variable,
) -> cp.Maximize:
    """mu'w - (gamma/2) w'Sigma w - lambda ||w - w_prev||_1.

    psd_wrap skips cvxpy's PSD re-check: the estimation layer already
    guarantees PSD up to float noise (tested), and re-checking can reject
    eigenvalues at -1e-16.
    """
    ccfg = cfg.construction
    prev = np.zeros(len(mu)) if w_prev is None else w_prev.to_numpy(dtype="float64")
    ret = mu.to_numpy(dtype="float64") @ variable
    risk = 0.5 * ccfg.risk_aversion * cp.quad_form(variable, cp.psd_wrap(matrix))
    penalty = ccfg.turnover_lambda * cp.norm1(variable - prev)
    return cp.Maximize(ret - risk - penalty)


def mvo(
    mu: pd.Series, cov: pd.DataFrame, w_prev: pd.Series | None, cfg: Config
) -> pd.Series:
    """Long-only mean-variance: fully invested, per-asset cap.

    maximize mu'w - (gamma/2) w'Sigma w - lambda ||w - w_prev||_1
    s.t. sum(w) = 1, 0 <= w_i <= position_cap.
    With mu = 0 and lambda = 0 this is the capped minimum-variance portfolio.
    """
    matrix = _validate_inputs(mu, cov, w_prev)
    ccfg = cfg.construction
    n = len(matrix)
    if n * ccfg.position_cap < 1.0 - 1e-12:
        raise ValueError(
            f"infeasible: {n} assets * cap {ccfg.position_cap} < 1 (fully invested)"
        )
    w = cp.Variable(n)
    constraints = [w >= 0, w <= ccfg.position_cap, cp.sum(w) == 1]
    raw = _solve_qp(_mvo_objective(matrix, mu, w_prev, cfg, w), constraints, w, "mvo")

    if raw.min() < -QP_BOUND_TOL or raw.max() > ccfg.position_cap + QP_BOUND_TOL:
        raise RuntimeError(f"mvo solution violates bounds by > {QP_BOUND_TOL:.0e}")
    if abs(raw.sum() - 1.0) > QP_BOUND_TOL:
        raise RuntimeError(f"mvo solution sum {raw.sum():.8f} != 1")
    cleaned = np.clip(raw, 0.0, ccfg.position_cap)
    cleaned[np.abs(cleaned) < 1e-10] = 0.0
    cleaned = cleaned / cleaned.sum()
    return pd.Series(cleaned, index=cov.index)


def mvo_ls(
    mu: pd.Series, cov: pd.DataFrame, w_prev: pd.Series | None, cfg: Config
) -> pd.Series:
    """Long-short mean-variance with position, gross and net constraints.

    maximize mu'w - (gamma/2) w'Sigma w - lambda ||w - w_prev||_1
    s.t. |w_i| <= position_cap, sum|w| <= gross_cap,
         net_min <= sum(w) <= net_max.
    Not fully invested by construction. The residual is cash.
    """
    matrix = _validate_inputs(mu, cov, w_prev)
    ccfg = cfg.construction
    n = len(matrix)
    w = cp.Variable(n)
    constraints = [
        cp.abs(w) <= ccfg.position_cap,
        cp.norm1(w) <= ccfg.gross_cap,
        cp.sum(w) >= ccfg.net_min,
        cp.sum(w) <= ccfg.net_max,
    ]
    raw = _solve_qp(
        _mvo_objective(matrix, mu, w_prev, cfg, w), constraints, w, "mvo_ls"
    )

    if np.abs(raw).max() > ccfg.position_cap + QP_BOUND_TOL:
        raise RuntimeError(f"mvo_ls position cap violated by > {QP_BOUND_TOL:.0e}")
    if np.abs(raw).sum() > ccfg.gross_cap + QP_BOUND_TOL:
        raise RuntimeError(f"mvo_ls gross cap violated by > {QP_BOUND_TOL:.0e}")
    if not (ccfg.net_min - QP_BOUND_TOL <= raw.sum() <= ccfg.net_max + QP_BOUND_TOL):
        raise RuntimeError(f"mvo_ls net exposure {raw.sum():.6f} outside band")
    cleaned = raw.copy()
    cleaned[np.abs(cleaned) < 1e-10] = 0.0
    return pd.Series(cleaned, index=cov.index)


def vol_target(weights: pd.Series, cov: pd.DataFrame, cfg: Config) -> pd.Series:
    """Scale the book so predicted annualized vol hits the target.

    scale = (target  / sqrt(252)) / sqrt(w'Sigma w), capped so that gross
    exposure never exceeds cfg.construction.gross_cap.

    A fully-invested long-only book stops summing to 1 after scaling: the
    residual is cash (scale < 1) or financing (scale > 1), by design.
    """
    if not weights.index.equals(cov.index):
        raise ValueError("weights are not aligned with cov")
    w = weights.to_numpy(dtype="float64")
    matrix = cov.to_numpy(dtype="float64")
    predicted_daily = float(np.sqrt(w @ matrix @ w))
    if predicted_daily <= 0.0:
        raise ValueError("cannot vol-target a zero-risk book")
    ccfg = cfg.construction
    scale = (ccfg.vol_target_annual / np.sqrt(TRADING_DAYS)) / predicted_daily
    gross = float(np.abs(w).sum())
    scale = min(scale, ccfg.gross_cap / gross)
    return weights * scale


_OPTIMIZERS: dict[
    str, Callable[[pd.Series, pd.DataFrame, pd.Series | None, Config], pd.Series]
] = {
    "equal_weight": equal_weight,
    "inverse_vol": inverse_vol,
    "erc": erc,
    "mvo": mvo,
    "mvo_ls": mvo_ls,
}

OPTIMIZER_NAMES: tuple[str, ...] = tuple(sorted(_OPTIMIZERS))
SIMPLE_NAMES: tuple[str, ...] = ("equal_weight", "erc", "inverse_vol")


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

"""Central research parameters as frozen dataclasses.

Every number that defines an experiment lives here and is passed explicitly as
`cfg`. Sweeps are `dataclasses.replace` variants, never module globals.
Module-internal contracts (schemas, paths, vendor constants) stay in their own
modules.
"""

from dataclasses import dataclass, field

from pandas.tseries.frequencies import to_offset


@dataclass(frozen=True, slots=True)
class SignalConfig:
    # tsmom trailing-return horizons k in trading days
    tsmom_windows: tuple[int, ...] = (63, 126, 252)

    # EWMA halflife (days) of sigma_daily, the per-asset vol that scales tsmom returns.
    # Deliberately decoupled from EstimationConfig.ewma_halflife_days; the defaults
    # coincide at 63 today but are free to diverge.
    signal_vol_halflife_days: int = 63

    # xsmom (cross-sectional momentum) trailing lookback window in days
    xs_lookback: int = 252

    # ...excluding the most recent days (short-term reversal contamination).
    xs_exclude: int = 21

    # xsmom scale factor to match the other two signals' cross-sectional z-scores.
    xsmom_scale: float = 2.0

    # st_rev short-term reversal window in days
    reversal_window: int = 5

    # Shared +/- bound of the composite signal's component scale.
    # tsmom and st_rev's cross-sectional z-scores are clipped to it.
    # xsmom (range [-1, 1]) is scaled by it in the combine step.
    clip: float = 2.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "tsmom_windows", tuple(self.tsmom_windows))
        if not self.tsmom_windows:
            raise ValueError("tsmom_windows must be non-empty")
        if any(w <= 0 for w in self.tsmom_windows):
            raise ValueError(
                f"tsmom_windows must be positive, got {self.tsmom_windows}"
            )
        if self.signal_vol_halflife_days <= 0:
            raise ValueError(
                f"signal_vol_halflife_days must be positive, "
                f"got {self.signal_vol_halflife_days}"
            )
        if not 0 <= self.xs_exclude < self.xs_lookback:
            raise ValueError(
                f"need 0 <= xs_exclude < xs_lookback, "
                f"got {self.xs_exclude}, {self.xs_lookback}"
            )
        if self.xsmom_scale <= 0:
            raise ValueError(f"xsmom_scale must be positive, got {self.xsmom_scale}")
        if self.reversal_window <= 0:
            raise ValueError(
                f"reversal_window must be positive, got {self.reversal_window}"
            )
        if self.clip <= 0:
            raise ValueError(f"clip must be positive, got {self.clip}")


@dataclass(frozen=True, slots=True)
class EstimationConfig:
    # EWMA halflife (days) for the asset-return covariance matrix Sigma,
    # used for the risk-model and position-sizing.
    # MVO, ERC, MCR/CCR all consume Sigma.
    ewma_halflife_days: int = 63

    # Trailing return look-back window fed to the covariance estimator at each
    # rebalance: 252 = 4 halflives of the default EWMA (~94% of the weight).
    cov_window_days: int = 252

    # The Sigma estimator the backtest uses for the estimator sensitivity study
    # - "ewma" default, recency-weighted covariance
    # - "lw_cc" Ledoit-Wolf constant-correlation on the equally-weighted window
    cov_estimator: str = "ewma"

    def __post_init__(self) -> None:
        if self.ewma_halflife_days <= 0:
            raise ValueError(
                f"ewma_halflife_days must be positive, got {self.ewma_halflife_days}"
            )
        if self.cov_window_days < 2:
            raise ValueError(
                f"cov_window_days must be >= 2, got {self.cov_window_days}"
            )
        if self.cov_estimator not in ("ewma", "lw_cc"):
            raise ValueError(
                f"cov_estimator must be 'ewma' or 'lw_cc', got {self.cov_estimator!r}"
            )

@dataclass(frozen=True, slots=True)
class ConstructionConfig:
    # Per-asset position cap: w_i <= cap (long-only MVO), |w_i| <= cap (MVO-LS).
    position_cap: float = 0.25

    # MVO-LS gross exposure cap: sum |w_i| <= gross_cap.
    gross_cap: float = 2.0

    # MVO-LS net exposure band: net_min <= sum w_i <= net_max.
    net_min: float = -0.5
    net_max: float = 1.0

    # Risk aversion (gamma) in the MVO objective mu'w - (gamma/2) w'Sigma w.
    # Sets position size: larger gamma -> smaller, more risk-averse positions.
    # mu is only a directional proxy with no calibrated scale, so doubling mu
    # and doubling gamma give the same weights, i.e., gamma alone fixes the size.
    # Default ~100 reflects the daily units here (mu ~ 1e-2, Sigma ~ 1e-4); it
    # would be O(1) only with annualized inputs.
    risk_aversion: float = 100.0

    # L1 turnover penalty lambda * ||w - w_prev||_1 inside both MVOs.
    # Default = 0, the cost-unaware baseline.
    turnover_lambda: float = 0.0

    # Vol-targeting overlay: annualized portfolio volatility target.
    vol_target_annual: float = 0.10

    def __post_init__(self) -> None:
        if self.position_cap <= 0:
            raise ValueError(f"position_cap must be positive, got {self.position_cap}")
        if self.gross_cap < self.position_cap:
            raise ValueError(
                f"gross_cap must be >= position_cap, "
                f"got {self.gross_cap} < {self.position_cap}"
            )
        if self.net_min > self.net_max:
            raise ValueError(
                f"need net_min <= net_max, got {self.net_min}, {self.net_max}"
            )
        if self.net_max > self.gross_cap or self.net_min < -self.gross_cap:
            raise ValueError(
                f"net band [{self.net_min}, {self.net_max}] must lie within "
                f"[-gross_cap, gross_cap] = [{-self.gross_cap}, {self.gross_cap}]"
            )
        if self.risk_aversion <= 0:
            raise ValueError(
                f"risk_aversion must be positive, got {self.risk_aversion}"
            )
        if self.turnover_lambda < 0:
            raise ValueError(
                f"turnover_lambda must be non-negative, got {self.turnover_lambda}"
            )
        if self.vol_target_annual <= 0:
            raise ValueError(
                f"vol_target_annual must be positive, got {self.vol_target_annual}"
            )


@dataclass(frozen=True, slots=True)
class EngineConfig:
    # Rebalance schedule as a pandas offset alias: "B" daily, "W-FRI" weekly,
    # "BME" monthly. Weekly/monthly reuse drives the cost-sensitivity study.
    rebalance_freq: str = "B"

    # Apply the vol-targeting overlay to every optimizer target, before the
    # no-trade band. Off by default: raw optimizer books are the baseline.
    vol_target: bool = False

    # Per-asset no-trade band. An asset only trades when the absolute difference
    # between the target weight and the drifted weight exceeds the band.
    # 0 = off, the baseline.
    no_trade_band: float = 0.0

    def __post_init__(self) -> None:
        try:
            to_offset(self.rebalance_freq)
        except ValueError as exc:
            raise ValueError(
                f"invalid rebalance_freq {self.rebalance_freq!r}: {exc}"
            ) from exc
        if self.no_trade_band < 0:
            raise ValueError(
                f"no_trade_band must be non-negative, got {self.no_trade_band}"
            )


@dataclass(frozen=True, slots=True)
class CostConfig:
    # Proportional trading cost per side, in basis points of traded notional.
    cost_per_side_bps: float = 5.0

    def __post_init__(self) -> None:
        if self.cost_per_side_bps < 0:
            raise ValueError(
                f"cost_per_side_bps must be non-negative, got {self.cost_per_side_bps}"
            )


@dataclass(frozen=True, slots=True)
class Config:
    signals: SignalConfig = field(default_factory=SignalConfig)
    estimation: EstimationConfig = field(default_factory=EstimationConfig)
    construction: ConstructionConfig = field(default_factory=ConstructionConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    costs: CostConfig = field(default_factory=CostConfig)

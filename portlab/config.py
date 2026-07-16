"""Central research parameters as frozen dataclasses.

Every number that defines an experiment lives here and is passed explicitly as
`cfg`. Sweeps are `dataclasses.replace` variants, never module globals.
Module-internal contracts (schemas, paths, vendor constants) stay in their own
modules.
"""

from dataclasses import dataclass, field


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

    def __post_init__(self) -> None:
        if self.ewma_halflife_days <= 0:
            raise ValueError(
                f"ewma_halflife_days must be positive, got {self.ewma_halflife_days}"
            )


@dataclass(frozen=True, slots=True)
class Config:
    signals: SignalConfig = field(default_factory=SignalConfig)
    estimation: EstimationConfig = field(default_factory=EstimationConfig)

"""Static asset universe: 19 liquid cross-asset ETFs + SPY / AGG as references.

Cross-asset ETFs:
  - US equity sectors
  - International equity
  - Duration ladder
  - Credit
  - Commodities
  - Real estate
So the covariance structure is genuinely valid and the optimizer comparison is
meaningful.

All constituents are long-lived ETFs still trading today. The (small)
survivorship consideration is documented in the README.
"""

from dataclasses import dataclass
from enum import StrEnum


class Role(StrEnum):
    OPTIMIZE = "optimize"
    REFERENCE = "reference"


class AssetClass(StrEnum):
    US_SECTOR = "us_sector"
    INTL_EQUITY = "intl_equity"
    DURATION = "duration"
    CREDIT = "credit"
    COMMODITY = "commodity"
    REIT = "reit"
    US_EQUITY = "us_equity"  # reference only
    US_BOND = "us_bond"  # reference only


@dataclass(frozen=True, slots=True)
class Asset:
    ticker: str
    name: str
    asset_class: AssetClass
    role: Role = Role.OPTIMIZE


# Local aliases keep each entry on one line under column length limit.
_SECTOR = AssetClass.US_SECTOR
_INTL = AssetClass.INTL_EQUITY
_DUR = AssetClass.DURATION
_CREDIT = AssetClass.CREDIT
_CMDTY = AssetClass.COMMODITY
_REIT = AssetClass.REIT

UNIVERSE: tuple[Asset, ...] = (
    Asset("XLK", "Technology Select Sector SPDR Fund", _SECTOR),
    Asset("XLF", "Financial Select Sector SPDR Fund", _SECTOR),
    Asset("XLE", "Energy Select Sector SPDR Fund", _SECTOR),
    Asset("XLV", "Health Care Select Sector SPDR Fund", _SECTOR),
    Asset("XLI", "Industrial Select Sector SPDR Fund", _SECTOR),
    Asset("XLP", "Consumer Staples Select Sector SPDR Fund", _SECTOR),
    Asset("XLY", "Consumer Discretionary Select Sector SPDR Fund", _SECTOR),
    Asset("XLU", "Utilities Select Sector SPDR Fund", _SECTOR),
    Asset("XLB", "Materials Select Sector SPDR Fund", _SECTOR),
    Asset("EFA", "iShares MSCI EAFE ETF", _INTL),
    Asset("EEM", "iShares MSCI Emerging Markets ETF", _INTL),
    Asset("SHY", "iShares 1-3 Year Treasury Bond ETF", _DUR),
    Asset("IEF", "iShares 7-10 Year Treasury Bond ETF", _DUR),
    Asset("TLT", "iShares 20+ Year Treasury Bond ETF", _DUR),
    Asset("LQD", "iShares iBoxx $ Investment Grade Corporate Bond ETF", _CREDIT),
    Asset("HYG", "iShares iBoxx $ High Yield Corporate Bond ETF", _CREDIT),
    Asset("GLD", "SPDR Gold Shares", _CMDTY),
    Asset("DBC", "Invesco DB Commodity Index Tracking Fund", _CMDTY),
    Asset("VNQ", "Vanguard Real Estate ETF", _REIT),
    Asset("SPY", "SPDR S&P 500 ETF Trust", AssetClass.US_EQUITY, Role.REFERENCE),
    Asset(
        "AGG",
        "iShares Core U.S. Aggregate Bond ETF",
        AssetClass.US_BOND,
        Role.REFERENCE,
    ),
)


def optimized_tickers() -> tuple[str, ...]:
    """Tickers eligible for signal generation and portfolio optimization."""
    return tuple(a.ticker for a in UNIVERSE if a.role is Role.OPTIMIZE)


def reference_tickers() -> tuple[str, ...]:
    """Benchmark tickers: reference curves only, never optimized."""
    return tuple(a.ticker for a in UNIVERSE if a.role is Role.REFERENCE)

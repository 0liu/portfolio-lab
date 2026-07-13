"""Universe tests

The universe is a frozen design decision. The constants must match the project
specification exactly.
"""

from portlab.universe import (
    UNIVERSE,
    AssetClass,
    Role,
    optimized_tickers,
    reference_tickers,
)

SPEC_OPTIMIZED = {
    # US sectors
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB",
    # International equity
    "EFA", "EEM",
    # Duration
    "SHY", "IEF", "TLT",
    # Credit
    "LQD", "HYG",
    # Commodities
    "GLD", "DBC",
    # REIT
    "VNQ",
}  # fmt: skip
SPEC_REFERENCE = {"SPY", "AGG"}


def test_universe_size():
    assert len(UNIVERSE) == 21


def test_no_duplicate_tickers():
    tickers = [a.ticker for a in UNIVERSE]
    assert len(set(tickers)) == len(tickers)


def test_optimized_tickers_match_spec_exactly():
    assert set(optimized_tickers()) == SPEC_OPTIMIZED
    assert len(optimized_tickers()) == 19


def test_reference_tickers_match_spec_exactly():
    assert set(reference_tickers()) == SPEC_REFERENCE


def test_optimized_and_reference_are_disjoint():
    assert not set(optimized_tickers()) & set(reference_tickers())


def test_nine_us_sectors():
    sectors = [a for a in UNIVERSE if a.asset_class is AssetClass.US_SECTOR]
    assert len(sectors) == 9
    assert all(a.role is Role.OPTIMIZE for a in sectors)


def test_asset_class_counts():
    counts: dict[AssetClass, int] = {}
    for a in UNIVERSE:
        counts[a.asset_class] = counts.get(a.asset_class, 0) + 1
    assert counts == {
        AssetClass.US_SECTOR: 9,
        AssetClass.INTL_EQUITY: 2,
        AssetClass.DURATION: 3,
        AssetClass.CREDIT: 2,
        AssetClass.COMMODITY: 2,
        AssetClass.REIT: 1,
        AssetClass.US_EQUITY: 1,
        AssetClass.US_BOND: 1,
    }


def test_benchmark_classes_are_reference_only():
    for a in UNIVERSE:
        if a.asset_class in (AssetClass.US_EQUITY, AssetClass.US_BOND):
            assert a.role is Role.REFERENCE


def test_all_fields_populated():
    for a in UNIVERSE:
        assert a.ticker
        assert a.name

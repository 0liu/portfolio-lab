"""Config tests: defaults ARE the research spec (accidental edits fail CI);
invalid parameter combinations fail at construction, not inside a backtest.
"""

from dataclasses import replace

import pytest

from portlab.config import Config, EstimationConfig, SignalConfig


def test_signal_defaults_are_the_spec():
    cfg = SignalConfig()
    assert cfg.tsmom_windows == (63, 126, 252)
    assert cfg.xsmom_scale == 2.0
    assert cfg.signal_vol_halflife_days == 63
    assert cfg.xs_lookback == 252
    assert cfg.xs_exclude == 21
    assert cfg.reversal_window == 5
    assert cfg.clip == 2.0


def test_estimation_defaults_are_the_spec():
    assert EstimationConfig().ewma_halflife_days == 63


def test_config_composes_sub_configs():
    cfg = Config()
    assert cfg.signals == SignalConfig()
    assert cfg.estimation == EstimationConfig()


def test_halflives_are_decoupled():
    # Tuning the risk model must never mutate what the signal is.
    cfg = Config(estimation=EstimationConfig(ewma_halflife_days=21))
    assert cfg.estimation.ewma_halflife_days == 21
    assert cfg.signals.signal_vol_halflife_days == 63


def test_windows_coerced_to_tuple():
    assert SignalConfig(tsmom_windows=(10, 20)).tsmom_windows == (10, 20)


def test_replace_builds_variant():
    variant = replace(SignalConfig(), reversal_window=10)
    assert variant.reversal_window == 10
    assert variant.tsmom_windows == (63, 126, 252)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        pytest.param({"tsmom_windows": ()}, "non-empty", id="empty-windows"),
        pytest.param({"tsmom_windows": (0,)}, "positive", id="zero-window"),
        pytest.param({"tsmom_windows": (63, -1)}, "positive", id="negative-window"),
        pytest.param({"signal_vol_halflife_days": 0}, "positive", id="zero-halflife"),
        pytest.param({"xs_exclude": 252}, "exclude", id="exclude-eq-lookback"),
        pytest.param({"xs_exclude": -1}, "exclude", id="negative-exclude"),
        pytest.param({"xsmom_scale": 0.0}, "positive", id="zero-xsmom-scale"),
        pytest.param({"reversal_window": 0}, "positive", id="zero-reversal"),
        pytest.param({"clip": 0.0}, "positive", id="zero-clip"),
        pytest.param({"clip": -2.0}, "positive", id="negative-clip"),
    ],
)
def test_invalid_signal_config_raises(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SignalConfig(**kwargs)


def test_invalid_estimation_config_raises():
    with pytest.raises(ValueError, match="positive"):
        EstimationConfig(ewma_halflife_days=0)

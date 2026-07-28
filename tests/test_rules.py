from __future__ import annotations

import numpy as np
import pandas as pd

from quant_regime.detectors.rules import detect_regime
from quant_regime.models import RegimeLabel


def _trend_series(n: int = 300, start: float = 100.0, drift: float = 0.001) -> pd.Series:
    idx = pd.date_range("2024-01-02", periods=n, freq="B")
    values = start * (1 + drift) ** np.arange(n)
    return pd.Series(values, index=idx)


def test_risk_on_in_calm_uptrend():
    result = detect_regime(_trend_series(), vol_lookback=60)
    assert result.regime == RegimeLabel.RISK_ON
    assert result.position_scale == 1.0


def test_risk_off_on_large_drawdown():
    close = _trend_series(n=300, start=100.0, drift=0.0)
    close.iloc[-25:] = close.iloc[-25] * np.linspace(1.0, 0.85, 25)
    result = detect_regime(close, vol_lookback=60, return_window=20, risk_off_return=-0.05)
    assert result.regime == RegimeLabel.RISK_OFF
    assert result.position_scale == 0.3


def test_high_vol_on_spiky_returns():
    idx = pd.date_range("2024-01-02", periods=300, freq="B")
    values = np.full(300, 100.0)
    values[270:] = 100 + np.sin(np.arange(30) * 2.5) * 8
    close = pd.Series(values, index=idx)
    result = detect_regime(
        close,
        vol_window=10,
        vol_lookback=40,
        vol_percentile_threshold=0.01,
    )
    assert result.regime == RegimeLabel.HIGH_VOL
    assert result.position_scale == 0.5

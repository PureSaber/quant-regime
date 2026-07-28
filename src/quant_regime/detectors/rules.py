from __future__ import annotations

import numpy as np
import pandas as pd

from quant_regime.models import RegimeLabel, RegimeResult


def _realized_vol(close: pd.Series, window: int) -> pd.Series:
    ret = close.pct_change()
    return ret.rolling(window).std() * np.sqrt(252)


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)


def detect_regime(
    close: pd.Series,
    *,
    vol_window: int = 20,
    vol_lookback: int = 252,
    vol_percentile_threshold: float = 0.8,
    return_window: int = 20,
    risk_off_return: float = -0.05,
    high_vol_scale: float = 0.5,
    risk_off_scale: float = 0.3,
    risk_on_scale: float = 1.0,
) -> RegimeResult:
    if close.empty:
        raise ValueError("close series is empty")

    close = close.sort_index()
    as_of = close.index[-1].strftime("%Y-%m-%d")
    vol = _realized_vol(close, vol_window)
    vol_pct = _rolling_percentile(vol, vol_lookback)
    latest_vol_pct = float(vol_pct.iloc[-1]) if not np.isnan(vol_pct.iloc[-1]) else 0.0

    window_return = float(close.iloc[-1] / close.iloc[-return_window - 1] - 1.0) if len(close) > return_window else 0.0

    if latest_vol_pct >= vol_percentile_threshold:
        regime = RegimeLabel.HIGH_VOL
        scale = high_vol_scale
    elif window_return < risk_off_return:
        regime = RegimeLabel.RISK_OFF
        scale = risk_off_scale
    else:
        regime = RegimeLabel.RISK_ON
        scale = risk_on_scale

    return RegimeResult(
        as_of=as_of,
        regime=regime,
        position_scale=scale,
        signals={
            "vol_percentile": round(latest_vol_pct, 4),
            "return_window": return_window,
            "window_return": round(window_return, 4),
        },
    )

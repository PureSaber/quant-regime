from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_regime.cli import main
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


def test_insufficient_history_is_not_reported_as_risk_on():
    close = pd.Series([100.0], index=[pd.Timestamp("2026-01-02")])
    with pytest.raises(
        ValueError,
        match="insufficient history: need at least 272 observations, got 1",
    ):
        detect_regime(close)


def test_custom_windows_define_the_required_history():
    close = _trend_series(n=5)
    with pytest.raises(ValueError, match="need at least 5 observations, got 4"):
        detect_regime(close.iloc[:-1], vol_window=2, vol_lookback=3, return_window=2)
    result = detect_regime(close, vol_window=2, vol_lookback=3, return_window=2)
    assert result.regime in set(RegimeLabel)


def test_cli_failure_preserves_existing_output(tmp_path):
    source = tmp_path / "one.csv"
    source.write_text("date,close\n2026-01-02,100\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(f"input:\n  path: {source.as_posix()}\nrules: {{}}\n", encoding="utf-8")
    output = tmp_path / "regime.json"
    output.write_text('{"status": "previous"}', encoding="utf-8")
    with pytest.raises(ValueError, match="insufficient history"):
        main(["detect", "--config", str(config), "--out", str(output)])
    assert output.read_text(encoding="utf-8") == '{"status": "previous"}'


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 0.0, -1.0])
def test_invalid_early_price_is_rejected_even_outside_signal_window(value):
    close = _trend_series()
    close.iloc[10] = value
    with pytest.raises(ValueError, match="finite, positive and complete"):
        detect_regime(close, vol_lookback=60)


@pytest.mark.parametrize("case", ["missing", "duplicate", "invalid_date", "nonnumeric"])
def test_cli_rejects_bad_history_without_dropping_rows_or_replacing_output(tmp_path, case):
    close = _trend_series()
    frame = pd.DataFrame({"date": close.index, "close": close.to_numpy()})
    if case == "missing":
        frame.loc[10, "close"] = np.nan
    elif case == "duplicate":
        frame = pd.concat([frame, frame.iloc[[10]]], ignore_index=True)
    elif case == "invalid_date":
        frame.loc[10, "date"] = pd.NaT
    else:
        frame["close"] = frame["close"].astype(object)
        frame.loc[10, "close"] = "not-a-price"
    source = tmp_path / "prices.csv"
    frame.to_csv(source, index=False)
    config = tmp_path / "config.yaml"
    config.write_text(
        f"input:\n  path: {source.as_posix()}\nrules: {{vol_lookback: 60}}\n", encoding="utf-8"
    )
    output = tmp_path / "regime.json"
    original = b'{"status": "previous"}'
    output.write_bytes(original)

    with pytest.raises(ValueError):
        main(["detect", "--config", str(config), "--out", str(output)])
    assert output.read_bytes() == original

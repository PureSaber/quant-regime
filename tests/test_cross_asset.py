from pathlib import Path

import pandas as pd
import pytest

from quant_regime.cross_asset import detect_cross_asset, load_multi_config


def test_detect_cross_asset() -> None:
    cfg = Path(__file__).resolve().parents[1] / "configs" / "cross_asset.yaml"
    inputs, rules = load_multi_config(cfg)
    result = detect_cross_asset(inputs, rules)
    assert result.position_scale > 0
    assert "equity_hs300" in result.legs


def test_cross_asset_rejects_an_insufficient_leg(monkeypatch) -> None:
    close = pd.Series([100.0], index=[pd.Timestamp("2026-01-02")])
    monkeypatch.setattr("quant_regime.cross_asset.load_series_csv", lambda *args, **kwargs: close)
    with pytest.raises(ValueError, match="insufficient history"):
        detect_cross_asset([{"name": "short", "path": "unused.csv"}], {})

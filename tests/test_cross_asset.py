from pathlib import Path

from quant_regime.cross_asset import detect_cross_asset, load_multi_config


def test_detect_cross_asset() -> None:
    cfg = Path(__file__).resolve().parents[1] / "configs" / "cross_asset.yaml"
    inputs, rules = load_multi_config(cfg)
    result = detect_cross_asset(inputs, rules)
    assert result.position_scale > 0
    assert "equity_hs300" in result.legs

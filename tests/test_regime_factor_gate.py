from pathlib import Path

from quant_regime.cross_asset import detect_cross_asset, load_multi_config


def test_detect_cross_asset_exposes_factor_gate() -> None:
    cfg = Path(__file__).resolve().parents[1] / "configs" / "cross_asset.yaml"
    inputs, rules = load_multi_config(cfg)
    result = detect_cross_asset(inputs, rules)
    assert 0.0 <= result.factor_gate <= 1.5
    assert result.factor_gate == result.position_scale
    assert "factor_gate" in result.to_dict()

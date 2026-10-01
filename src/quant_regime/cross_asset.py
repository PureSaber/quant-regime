from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from quant_regime.detectors.rules import detect_regime
from quant_regime.io import load_series_csv
from quant_regime.models import RegimeLabel


@dataclass
class CrossAssetRegime:
    as_of: str
    position_scale: float
    factor_gate: float
    regime: RegimeLabel
    legs: dict[str, dict]

    def to_dict(self) -> dict:
        return {
            "as_of": self.as_of,
            "position_scale": self.position_scale,
            "factor_gate": self.factor_gate,
            "regime": self.regime.value,
            "legs": self.legs,
        }


def detect_cross_asset(inputs: list[dict], rules: dict) -> CrossAssetRegime:
    """Combine multiple index/futures series into one conservative position_scale."""
    if not inputs:
        raise ValueError("inputs list is empty")

    asset_class_weights = rules.get("asset_class_weights") or {}
    default_weight = float(rules.get("default_weight", 1.0))
    legs: dict[str, dict] = {}
    weighted_scales: list[float] = []
    weights: list[float] = []
    regimes: list[RegimeLabel] = []
    as_ofs: list[str] = []

    for entry in inputs:
        name = str(entry["name"])
        series = load_series_csv(
            Path(entry["path"]),
            date_col=str(entry.get("date_col", "date")),
            value_col=str(entry.get("value_col", "close")),
        )
        result = detect_regime(
            series,
            **{
                k: v for k, v in rules.items() if k not in ("asset_class_weights", "default_weight")
            },
        )
        weight = float(asset_class_weights.get(entry.get("asset_class", name), default_weight))
        legs[name] = {
            "regime": result.regime.value,
            "position_scale": result.position_scale,
            "weight": weight,
            "signals": result.signals,
        }
        weighted_scales.append(result.position_scale * weight)
        weights.append(weight)
        regimes.append(result.regime)
        as_ofs.append(result.as_of)

    total_w = sum(weights) or 1.0
    combined_scale = sum(weighted_scales) / total_w

    if RegimeLabel.HIGH_VOL in regimes:
        combined_regime = RegimeLabel.HIGH_VOL
    elif RegimeLabel.RISK_OFF in regimes:
        combined_regime = RegimeLabel.RISK_OFF
    else:
        combined_regime = RegimeLabel.RISK_ON

    return CrossAssetRegime(
        as_of=max(as_ofs),
        position_scale=round(combined_scale, 4),
        factor_gate=round(combined_scale, 4),
        regime=combined_regime,
        legs=legs,
    )


def load_multi_config(path: Path) -> tuple[list[dict], dict]:
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    inputs = []
    for entry in cfg.get("inputs") or []:
        item = dict(entry)
        p = Path(str(item["path"]))
        if not p.is_absolute():
            item["path"] = str((path.parent / p).resolve())
        inputs.append(item)
    return inputs, dict(cfg.get("rules") or {})

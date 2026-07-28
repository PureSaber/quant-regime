from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_regime.detectors.rules import detect_regime
from quant_regime.io import load_config, load_series_csv


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Detect market regime from index series")
    sub = parser.add_subparsers(dest="command", required=True)
    detect = sub.add_parser("detect", help="Run regime detection")
    detect.add_argument("--config", required=True, help="Path to YAML config")
    detect.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args(argv)

    cfg = load_config(Path(args.config))
    input_cfg = cfg.get("input") or {}
    rules_cfg = cfg.get("rules") or {}

    series = load_series_csv(
        Path(input_cfg["path"]),
        date_col=str(input_cfg.get("date_col", "date")),
        value_col=str(input_cfg.get("value_col", "close")),
    )
    result = detect_regime(series, **rules_cfg)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    print(f"wrote {out_path} regime={result.regime.value} scale={result.position_scale}")


if __name__ == "__main__":
    main()

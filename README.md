# quant-regime

Rule-based market regime detector for research, backtesting, and paper-trading workflows (`risk_on` / `risk_off` / `high_vol`).

The detector consumes configured market features and emits deterministic regime state plus a position-scaling recommendation. It is not a price forecaster, exchange-grade risk system, live broker adapter, or order-routing component.

## Commands

```bash
pip install -e ".[dev]"
quant-regime detect --config configs/default.yaml --out state/regime.json
pytest -q
```

## Output

JSON with `regime`, `position_scale`, and diagnostic `signals`.

The detector requires at least `max(vol_window + vol_lookback, return_window + 1)`
usable observations. Insufficient history raises an error and does not emit a normal regime or
position scale; `detect-multi` applies the same requirement to every leg.

Downstream systems must treat the output as research metadata. Any portfolio or paper-trading consumer remains responsible for its own validation and risk gates; this repository never sends real orders.

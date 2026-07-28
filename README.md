# quant-regime

Rule-based market regime detector (risk_on / risk_off / high_vol).

## Commands

```bash
pip install -e ".[dev]"
quant-regime detect --config configs/default.yaml --out state/regime.json
pytest -q
```

## Output

JSON with `regime`, `position_scale`, and diagnostic `signals`.

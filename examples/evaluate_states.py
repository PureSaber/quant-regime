"""Run with python examples/evaluate_states.py; all values are synthetic."""

import json
from dataclasses import asdict

import pandas as pd

from quant_regime.evaluation import evaluate_policy, fit_policy

dates = pd.date_range("2024-01-01", periods=25, tz="UTC")
data = pd.DataFrame(
    {
        "start": dates[:-1],
        "end": dates[1:],
        "signal_at": dates[:-1],
        "regime": ["risk_on", "risk_off"] * 12,
        "asset_return": [0.02, -0.01] * 12,
        "cash_return": 0.0,
    }
)
policy = fit_policy(data.iloc[:12])
path, summary = evaluate_policy(policy, data.iloc[12:], cost_bps=10, evidence_kind="synthetic")
print(json.dumps({"model": asdict(policy), "summary": summary}, indent=2))
print(path.to_csv(index=False))

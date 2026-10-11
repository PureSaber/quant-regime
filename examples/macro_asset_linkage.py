"""Reproducible 3-asset/2-macro frozen OOS study with independent arithmetic controls."""

import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from quant_regime.linkage import evaluate_macro_relation, fit_macro_relation
from quant_regime.linkage_demo import ASSETS, INTERCEPTS, SERIES, SLOPES, synthetic_linkage


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=False)
data, observations, _ = synthetic_linkage()
model, training = fit_macro_relation(
    data.iloc[:24],
    observations,
    assets=ASSETS,
    series=SERIES,
    fit_at=data.start.iloc[24],
    evidence_kind="synthetic",
)
predictions, result = evaluate_macro_relation(
    model, data.iloc[24:], observations, evaluation_at=data.end.max()
)
data.to_csv(root / "synthetic-returns.csv", index=False)
observations.to_csv(root / "synthetic-macro-vintages.csv", index=False)
predictions.to_csv(root / "oos-predictions.csv", index=False)
pd.DataFrame(result["comparisons"]).to_csv(root / "asset-comparison.csv", index=False)
pd.DataFrame(result["macro_lineage"]).to_csv(root / "oos-macro-lineage.csv", index=False)
write_json(root / "frozen-model.json", asdict(model))
write_json(root / "training.json", training)
write_json(root / "oos-report.json", result)

controls = []
# Balanced +/-1 feature grid implies zero feature means and identity second moments.
# True train intercepts/slopes therefore have an independent closed-form solution.
for i, row in enumerate(training["coefficients"]):
    assert math.isclose(row["native_intercept"], INTERCEPTS[i], abs_tol=1e-14)
    assert np.allclose(list(row["native_slopes"].values()), SLOPES[i], atol=1e-14, rtol=0)
for i, row in enumerate(result["comparisons"]):
    baseline_mse = [0.01**2 + 0.02**2, 0.004**2 + 0.012**2, 0.018**2 + 0.006**2][i]
    model_mse = [0.0, 0.0, (2 * 0.018) ** 2][i]
    r_squared = 1 - model_mse / baseline_mse
    assert math.isclose(row["mean_squared_error"], model_mse, abs_tol=1e-14)
    assert math.isclose(row["baseline_mean_squared_error"], baseline_mse, abs_tol=1e-14)
    assert math.isclose(row["oos_r_squared_vs_mean"], r_squared, abs_tol=1e-12)
    controls.append(
        {
            "asset": row["asset"],
            "expected_model_mse": model_mse,
            "expected_baseline_mse": baseline_mse,
            "expected_oos_r_squared": r_squared,
            "reported": row,
            "status": "pass",
        }
    )

# Deliberately unavailable macro and outcomes: evaluate the SAME frozen model.
stress_macro, stress_data = observations.copy(), data.copy()
stress_macro.loc[stress_macro.evidence_id == "inflation:initial:27", "value"] = None
stress_data.loc[28, "bond"] = np.nan
stress_data.loc[29, "return_known_at"] = data.end.max() + pd.Timedelta(days=1)
stress_path, stress_report = evaluate_macro_relation(
    model, stress_data.iloc[24:], stress_macro, evaluation_at=data.end.max()
)
assert [r["scored_intervals"] for r in stress_report["comparisons"]] == [9, 8, 9]
stress_path.to_csv(root / "coverage-stress-predictions.csv", index=False)
write_json(root / "coverage-stress.json", stress_report)
write_json(
    root / "independent-control.json",
    {
        "evidence_kind": "synthetic",
        "closed_form_controls": controls,
        "coverage_stress_scored_intervals": {
            r["asset"]: r["scored_intervals"] for r in stress_report["comparisons"]
        },
        "interpretation": "Equity/bond synthetic relations hold; the preset commodity relation "
        "break makes the frozen model worse than its matched baseline. No causal or market-validity claim.",
    },
)
write_json(
    root / "manifest.json",
    {
        "schema": "quant-regime.macro-linkage-example/1",
        "evidence_kind": "synthetic",
        "train_intervals": 24,
        "test_intervals": 12,
        "assets": list(ASSETS),
        "series": list(SERIES),
        "files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.iterdir())
            if p.is_file()
        },
    },
)
print(
    json.dumps(
        {
            "evidence_kind": "synthetic",
            "comparisons": result["comparisons"],
            "coverage_stress_scored": [r["scored_intervals"] for r in stress_report["comparisons"]],
        },
        indent=2,
    )
)

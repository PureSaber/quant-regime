import numpy as np
import pandas as pd
import pytest

from quant_regime.evaluation import evaluate_policy, fit_policy, intervals, macro_linkage


def test_predeclared_fold_comparison_never_overlaps():
    from quant_regime.evaluation import compare_folds

    path, summaries = compare_folds(
        sample(), [(0, 6, 9), (3, 9, 12)], cost_bps=10, evidence_kind="synthetic"
    )
    assert len(summaries) == 2 and len(path) == 6
    with pytest.raises(ValueError, match="overlapping"):
        compare_folds(sample(), [(0, 6, 10), (3, 9, 12)], cost_bps=0, evidence_kind="synthetic")


def sample():
    dates = pd.date_range("2024-01-01", periods=13, tz="UTC")
    return pd.DataFrame(
        {
            "start": dates[:-1],
            "end": dates[1:],
            "signal_at": dates[:-1],
            "regime": ["on"] * 3 + ["off"] * 3 + ["on", "on", "off", "off", "on", "new"],
            "asset_return": [0.1] * 3 + [-0.1] * 3 + [0.1, 0, -0.1, 0, 0.02, 0],
            "cash_return": 0.0,
        }
    )


def test_independent_cost_drift_and_stability():
    data = sample()
    policy = fit_policy(data.iloc[:6])
    path, result = evaluate_policy(policy, data.iloc[6:], cost_bps=100, evidence_kind="synthetic")
    # At entry: 100% exposure earns 10%, pays 1%; next rebalance uses 1.1/1.09.
    assert path.policy_net.iloc[0] == pytest.approx(0.09)
    assert path.policy_cost.iloc[1] == pytest.approx(abs(1 - 1.1 / 1.09) * 0.01)
    assert result["stability"]["switches"] == 3
    assert result["unknown_states_flat"] == ["new"]
    altered = data.copy()
    altered.loc[8:, "asset_return"] = 5.0
    other, _ = evaluate_policy(policy, altered.iloc[6:], cost_bps=100, evidence_kind="synthetic")
    np.testing.assert_array_equal(path.policy_weight, other.policy_weight)
    pd.testing.assert_frame_equal(path.iloc[:2], other.iloc[:2])


def test_reject_overlap_missing_and_late_signals():
    data = sample()
    with pytest.raises(ValueError, match="overlaps"):
        evaluate_policy(fit_policy(data), data, cost_bps=0, evidence_kind="synthetic")
    data.loc[0, "signal_at"] = data.end.iloc[0]
    with pytest.raises(ValueError, match="signal"):
        intervals(data)
    with pytest.raises(ValueError, match="contiguous"):
        intervals(sample().drop(index=5))


def test_macro_revision_visible_only_after_release():
    obs = pd.DataFrame(
        [
            {
                "series_id": "CPI",
                "observation_date": "2023-12-01",
                "vintage_date": vintage,
                "released_at": release,
                "available_at": release,
                "value": value,
                "unit": "pct",
                "source": "synthetic",
                "evidence_id": evidence,
            }
            for vintage, release, value, evidence in [
                ("2024-01-01", "2024-01-01T13:30:00Z", "3", "initial"),
                ("2024-01-09", "2024-01-09T13:30:00Z", "9", "revision"),
            ]
        ]
    )
    result = macro_linkage(sample(), obs, required_series=["CPI"])
    assert not result.available.iloc[0]
    assert result.value.iloc[8] == "3"
    assert result.evidence_id.iloc[9] == "revision"

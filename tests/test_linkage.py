from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from quant_regime.linkage import evaluate_macro_relation, fit_macro_relation, macro_change_panel
from quant_regime.linkage_demo import ASSETS, INTERCEPTS, SERIES, SLOPES, synthetic_linkage


def fitted(data, macro, **kwargs):
    return fit_macro_relation(
        data.iloc[:24],
        macro,
        assets=ASSETS,
        series=SERIES,
        fit_at=data.start.iloc[24],
        evidence_kind="synthetic",
        **kwargs,
    )


def test_independent_orthogonal_coefficients_and_matched_oos_errors():
    data, macro, changes = synthetic_linkage()
    model, training = fitted(data, macro)
    assert training["status"] == "fitted"
    for i, row in enumerate(training["coefficients"]):
        assert row["native_intercept"] == pytest.approx(INTERCEPTS[i])
        assert list(row["native_slopes"].values()) == pytest.approx(SLOPES[i])
        assert row["benchmark_mean"] == pytest.approx(INTERCEPTS[i])
    path, report = evaluate_macro_relation(
        model, data.iloc[24:], macro, evaluation_at=data.end.max()
    )
    assert len(path) == 36 and path.scored.all()
    # For a balanced +/-1 design, the constant-mean baseline MSE is the sum of
    # squared true slopes. Flipping the commodity growth coefficient makes model
    # error 0.036*growth, so MSE=.036^2 and relative R2=1-.036^2/(.018^2+.006^2).
    for i, result in enumerate(report["comparisons"]):
        assert result["scored_intervals"] == 12
        assert result["baseline_mean_squared_error"] == pytest.approx([0.0005, 0.00016, 0.00036][i])
        assert result["mean_squared_error"] == pytest.approx([0, 0, 0.001296][i], abs=1e-15)
        assert result["oos_r_squared_vs_mean"] == pytest.approx([1, 1, -2.6][i])
    predicted = path.pivot(index="start", columns="asset", values="prediction").reindex(
        columns=ASSETS
    )
    np.testing.assert_allclose(predicted, changes[24:] @ SLOPES.T + INTERCEPTS, atol=1e-14)


def test_later_revision_cannot_rewrite_training_or_earlier_prediction():
    data, macro, _ = synthetic_linkage()
    model, _ = fitted(data, macro)
    old, _ = evaluate_macro_relation(model, data.iloc[24:], macro, evaluation_at=data.end.max())
    revision = macro.loc[macro.evidence_id == "growth:initial:25"].iloc[0].copy()
    revised_at = data.start.iloc[25] - pd.Timedelta(minutes=30)
    revision["released_at"] = revision["available_at"] = revised_at
    revision["vintage_date"] = revised_at.tz_localize(None).normalize()
    revision["value"] = str(float(revision.value) + 5)
    revision["evidence_id"] = "later-revision"
    amended = pd.concat([macro, revision.to_frame().T], ignore_index=True)
    updated_model, _ = fitted(data, amended)
    assert asdict(model) == asdict(updated_model)
    changed, report = evaluate_macro_relation(
        model, data.iloc[24:], amended, evaluation_at=data.end.max()
    )
    pd.testing.assert_frame_equal(old.iloc[:3], changed.iloc[:3])
    assert not np.allclose(old.prediction.iloc[3:6], changed.prediction.iloc[3:6])
    lineage = [r for r in report["macro_lineage"] if r["row"] == 1 and r["series_id"] == "growth"]
    assert lineage[0]["prior_evidence_id"] == "later-revision"


def test_missing_macro_and_labels_preserve_counts_and_baseline_matching():
    data, macro, _ = synthetic_linkage()
    model, _ = fitted(data, macro)
    macro.loc[macro.evidence_id == "inflation:initial:27", "value"] = None
    data.loc[28, "bond"] = np.nan
    data.loc[29, "return_known_at"] = data.end.max() + pd.Timedelta(days=1)
    path, report = evaluate_macro_relation(
        model, data.iloc[24:], macro, evaluation_at=data.end.max()
    )
    for result, expected in zip(report["comparisons"], [9, 8, 9], strict=True):
        group = path.loc[(path.asset == result["asset"]) & path.scored]
        assert result["scored_intervals"] == expected
        expected_baseline = ((group.observed_return - group.benchmark_prediction) ** 2).mean()
        assert result["baseline_mean_squared_error"] == pytest.approx(expected_baseline)
    assert path.loc[path.return_known_at > data.end.max(), "observed_return"].isna().all()
    assert any(r["reason"] == "missing_released_value" for r in report["macro_lineage"])


def test_return_availability_and_training_overlap_gates():
    data, macro, _ = synthetic_linkage()
    data.loc[0, "return_known_at"] = data.end.max()
    model, training = fitted(data, macro)
    assert all(
        r["used_rows"] == 23 and r["unavailable_return_rows"] == 1 for r in training["coverage"]
    )
    with pytest.raises(ValueError, match="overlaps training"):
        evaluate_macro_relation(model, data.iloc[23:], macro, evaluation_at=data.end.max())
    with pytest.raises(ValueError, match="extend past"):
        fit_macro_relation(
            data,
            macro,
            assets=ASSETS,
            series=SERIES,
            fit_at=data.start.iloc[24],
            evidence_kind="synthetic",
        )
    data.loc[24, "signal_at"] = data.start.iloc[24] - pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="unavailable at signal"):
        evaluate_macro_relation(model, data.iloc[24:], macro, evaluation_at=data.end.max())


def test_unreleased_stale_and_unidentifiable_are_unavailable():
    data, macro, _ = synthetic_linkage()
    features, lineage, _ = macro_change_panel(
        macro, ["2021-01-01T00:00Z", "2026-01-01T00:00Z"], series=SERIES
    )
    assert features.isna().all().all()
    assert {r["reason"] for r in lineage} == {"not_released", "stale_latest_observation"}
    model, training = fitted(data, macro, min_observations=25)
    assert training["status"] == "blocked" and not model.models
    path, report = evaluate_macro_relation(
        model, data.iloc[24:], macro, evaluation_at=data.end.max()
    )
    assert not path.scored.any() and report["status"] == "no_scored_predictions"
    assert all(r["mean_squared_error"] is None for r in report["comparisons"])


def test_frozen_predictions_ignore_future_return_values_and_refuse_unit_changes():
    data, macro, _ = synthetic_linkage()
    model, _ = fitted(data, macro)
    path, _ = evaluate_macro_relation(model, data.iloc[24:], macro, evaluation_at=data.end.max())
    data.loc[24:, list(ASSETS)] = 0.5
    changed, _ = evaluate_macro_relation(model, data.iloc[24:], macro, evaluation_at=data.end.max())
    np.testing.assert_array_equal(path.prediction, changed.prediction)
    macro.loc[macro.series_id == "inflation", "unit"] = "different_unit"
    with pytest.raises(ValueError, match="unit differs"):
        evaluate_macro_relation(model, data.iloc[24:], macro, evaluation_at=data.end.max())


@pytest.mark.parametrize("problem", ["constant", "collinear"])
def test_unidentified_training_feature_models_are_blocked(problem):
    data, macro, _ = synthetic_linkage()
    if problem == "constant":
        macro.loc[macro.series_id == "inflation", "value"] = "3"
        reason = "constant_training_feature"
    else:
        growth = macro.loc[macro.series_id == "growth"].set_index("observation_date").value
        mask = macro.series_id == "inflation"
        macro.loc[mask, "value"] = macro.loc[mask, "observation_date"].map(growth)
        reason = "rank_deficient_training_features"
    model, summary = fitted(data, macro)
    assert not model.models and summary["status"] == "blocked"
    assert all(row["reason"] == reason for row in summary["coverage"])


def test_zero_baseline_error_has_no_relative_r_squared():
    data, macro, _ = synthetic_linkage()
    data.loc[:, list(ASSETS)] = 0.0
    model, _ = fitted(data, macro)
    _, result = evaluate_macro_relation(model, data.iloc[24:], macro, evaluation_at=data.end.max())
    assert all(row["baseline_mean_squared_error"] == 0 for row in result["comparisons"])
    assert all(row["oos_r_squared_vs_mean"] is None for row in result["comparisons"])


def test_evaluation_cannot_disclose_macro_at_future_signals():
    data, macro, _ = synthetic_linkage()
    model, _ = fitted(data, macro)
    with pytest.raises(ValueError, match="cutoff precedes"):
        evaluate_macro_relation(model, data.iloc[24:], macro, evaluation_at=data.start.iloc[24])


def test_observation_gap_never_becomes_an_implicit_short_period_change():
    data, macro, _ = synthetic_linkage()
    features, lineage, _ = macro_change_panel(
        macro, [data.signal_at.iloc[0]], series=SERIES, max_gap_days=10
    )
    assert features.isna().all().all()
    assert all(row["reason"] == "observation_gap_exceeds_limit" for row in lineage)

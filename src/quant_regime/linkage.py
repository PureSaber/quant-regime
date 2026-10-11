"""Frozen macro-change relations to subsequent multi-asset interval returns.

This is an out-of-sample forecast comparison, not a causal estimate or trade book.
Macro versions are resolved through the existing QDK release/vintage contract.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from quant_data_kit.financial.common import utc
from quant_data_kit.financial.macro import macro_asof

from quant_regime.evaluation import intervals


def _names(values, label):
    if isinstance(values, str):
        raise TypeError(f"{label} names must be an explicit sequence, not one string")
    values = tuple(values)
    if (
        not values
        or len(set(values)) != len(values)
        or any(not isinstance(v, str) or not v for v in values)
    ):
        raise ValueError(f"unique nonempty {label} names required")
    return values


def _returns(frame, assets):
    reserved = {"start", "end", "signal_at", "return_known_at"}
    if reserved.intersection(assets) or not (reserved | set(assets)).issubset(frame):
        raise ValueError("explicit timing and non-reserved asset return columns required")
    # Reuse interval timing validation without substituting any missing asset return.
    timing = frame[["start", "end", "signal_at"]].assign(
        regime="macro_relation", asset_return=0.0, cash_return=0.0
    )
    out = intervals(timing)[["start", "end", "signal_at"]]
    out["return_known_at"] = [utc(v) for v in frame.return_known_at]
    if out.return_known_at.isna().any() or (out.return_known_at < out.end).any():
        raise ValueError("return_known_at must not precede the return interval end")
    for asset in assets:
        values = pd.to_numeric(frame[asset], errors="raise").to_numpy(dtype=float)
        if np.isinf(values).any() or (values <= -1).any():
            raise ValueError("simple returns must be greater than -1; missing may remain NaN")
        out[asset] = values
    return out


def macro_change_panel(observations, signal_times, *, series, max_age_days=120, max_gap_days=120):
    """Latest minus previous *observation* value, using versions visible at each signal.

    Absolute differences retain each series' native units. A revised older value is
    used only after its own availability time. No later revision rewrites a feature
    already formed. Missing/latest null values never fall back to an older level.
    """
    series = _names(series, "macro series")
    if type(max_age_days) is not int or max_age_days < 0:
        raise ValueError("nonnegative integer macro age limit required")
    if type(max_gap_days) is not int or max_gap_days < 1:
        raise ValueError("positive integer observation-gap limit required")
    values, lineage, units = [], [], {}
    for i, raw_time in enumerate(signal_times):
        at = utc(raw_time)
        visible = macro_asof(observations, at)
        feature = {}
        for name in series:
            history = visible.loc[visible.series_id == name].sort_values("observation_date")
            row = {
                "row": i,
                "signal_at": at.isoformat(),
                "series_id": name,
                "value": None,
                "reason": None,
            }
            if len(history) < 2:
                row["reason"] = "not_released" if history.empty else "fewer_than_two_observations"
            else:
                prior, current = history.iloc[-2], history.iloc[-1]
                age = (at.tz_localize(None).normalize() - current.observation_date).days
                gap = (current.observation_date - prior.observation_date).days
                units[name] = current.unit
                row.update(
                    {
                        "unit": current.unit,
                        "observation_gap_days": gap,
                        "latest_age_days": age,
                        "current_observation": str(current.observation_date.date()),
                        "prior_observation": str(prior.observation_date.date()),
                        "current_value": str(current.value) if current.value is not None else None,
                        "prior_value": str(prior.value) if prior.value is not None else None,
                        "current_available_at": current.available_at.isoformat(),
                        "prior_available_at": prior.available_at.isoformat(),
                        "current_evidence_id": current.evidence_id,
                        "prior_evidence_id": prior.evidence_id,
                    }
                )
                if current.value is None or prior.value is None:
                    row["reason"] = "missing_released_value"
                elif age > max_age_days:
                    row["reason"] = "stale_latest_observation"
                elif gap > max_gap_days:
                    row["reason"] = "observation_gap_exceeds_limit"
                else:
                    change = float(current.value - prior.value)
                    if not np.isfinite(change):
                        raise ValueError("nonfinite macro change")
                    row["value"] = change
            feature[name] = row["value"]
            lineage.append(row)
        values.append(feature)
    return pd.DataFrame(values, columns=series, dtype=float), lineage, units


@dataclass(frozen=True)
class AssetRelation:
    asset: str
    observations: int
    center: tuple[float, ...]
    scale: tuple[float, ...]
    coefficients: tuple[float, ...]  # intercept followed by standardized-feature slopes
    benchmark_mean: float
    feature_min: tuple[float, ...]
    feature_max: tuple[float, ...]


@dataclass(frozen=True)
class MacroRelation:
    assets: tuple[str, ...]
    series: tuple[str, ...]
    fit_at: str
    training_end: str
    models: tuple[AssetRelation, ...]
    unavailable: tuple[tuple[str, str], ...]
    units: tuple[tuple[str, str], ...]
    max_age_days: int
    max_gap_days: int
    evidence_kind: str


def fit_macro_relation(
    training,
    observations,
    *,
    assets,
    series,
    fit_at,
    evidence_kind,
    min_observations=12,
    max_age_days=120,
    max_gap_days=120,
):
    """Train independent per-asset OLS and matching constant-mean forecast baselines.

    Every scaler, coefficient and baseline uses that asset's same complete, known
    training rows. Insufficient or unidentified models stay unavailable with reasons.
    """
    assets, series = _names(assets, "asset"), _names(series, "macro series")
    if evidence_kind not in {"synthetic", "retrospective", "historical_pit"}:
        raise ValueError("explicit evidence classification required")
    if type(min_observations) is not int or min_observations < len(series) + 2:
        raise ValueError("minimum sample must exceed regression parameter count")
    data = _returns(training, assets)
    fit_at = utc(fit_at)
    if data.end.max() > fit_at:
        raise ValueError("training intervals extend past fit_at")
    features, lineage, units = macro_change_panel(
        observations,
        data.signal_at,
        series=series,
        max_age_days=max_age_days,
        max_gap_days=max_gap_days,
    )
    macro_complete = features.notna().all(axis=1)
    return_available = data.return_known_at <= fit_at
    models, unavailable, coverage = [], [], []
    for asset in assets:
        eligible = macro_complete & return_available & data[asset].notna()
        x, y = features.loc[eligible].to_numpy(), data.loc[eligible, asset].to_numpy()
        reason = None
        if len(y) < min_observations:
            reason = "insufficient_complete_known_training_rows"
        else:
            center, scale = x.mean(axis=0), x.std(axis=0, ddof=0)
            if (scale <= 1e-12).any():
                reason = "constant_training_feature"
            else:
                design = np.column_stack([np.ones(len(y)), (x - center) / scale])
                if np.linalg.matrix_rank(design) < design.shape[1]:
                    reason = "rank_deficient_training_features"
                else:
                    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
                    models.append(
                        AssetRelation(
                            asset,
                            len(y),
                            tuple(center),
                            tuple(scale),
                            tuple(coef),
                            float(y.mean()),
                            tuple(x.min(axis=0)),
                            tuple(x.max(axis=0)),
                        )
                    )
        if reason:
            unavailable.append((asset, reason))
        coverage.append(
            {
                "asset": asset,
                "candidate_rows": len(data),
                "used_rows": int(eligible.sum()) if not reason else 0,
                "eligible_rows": int(eligible.sum()),
                "missing_macro_rows": int((~macro_complete).sum()),
                "unavailable_return_rows": int((~return_available).sum()),
                "missing_asset_rows": int(data[asset].isna().sum()),
                "status": "unavailable" if reason else "fitted",
                "reason": reason,
            }
        )
    model = MacroRelation(
        assets,
        series,
        fit_at.isoformat(),
        data.end.max().isoformat(),
        tuple(models),
        tuple(unavailable),
        tuple(sorted(units.items())),
        max_age_days,
        max_gap_days,
        evidence_kind,
    )
    coefficient_rows = []
    for fitted in models:
        slopes = np.array(fitted.coefficients[1:]) / np.array(fitted.scale)
        coefficient_rows.append(
            {
                "asset": fitted.asset,
                "observations": fitted.observations,
                "native_intercept": float(
                    fitted.coefficients[0] - slopes @ np.array(fitted.center)
                ),
                "native_slopes": dict(zip(series, slopes.tolist(), strict=True)),
                "benchmark_mean": fitted.benchmark_mean,
            }
        )
    return model, {
        "status": "fitted" if len(models) == len(assets) else "partial" if models else "blocked",
        "evidence_kind": evidence_kind,
        "fit_at": model.fit_at,
        "coverage": coverage,
        "coefficients": coefficient_rows,
        "macro_lineage": lineage,
        "limits": "Associational OLS. No causal interpretation or model/feature search on holdout.",
    }


def evaluate_macro_relation(model, testing, observations, *, evaluation_at):
    """Score frozen forecasts against the same-row frozen-mean baseline per asset."""
    data = _returns(testing, model.assets)
    evaluation_at = utc(evaluation_at)
    if data.signal_at.max() > evaluation_at:
        raise ValueError("evaluation cutoff precedes requested signal times")
    if data.start.min() < utc(model.training_end) or data.signal_at.min() < utc(model.fit_at):
        raise ValueError("OOS interval overlaps training or model is unavailable at signal time")
    features, lineage, units = macro_change_panel(
        observations,
        data.signal_at,
        series=model.series,
        max_age_days=model.max_age_days,
        max_gap_days=model.max_gap_days,
    )
    if any(name in units and units[name] != unit for name, unit in model.units):
        raise ValueError("macro unit differs from frozen training definition")
    fitted = {m.asset: m for m in model.models}
    unavailable = dict(model.unavailable)
    output = []
    for i, row in data.iterrows():
        macro_complete = bool(features.iloc[i].notna().all())
        raw_x = features.iloc[i].to_numpy()
        for asset in model.assets:
            reason, prediction, baseline, extrapolation = [], np.nan, np.nan, None
            relation = fitted.get(asset)
            if relation is None:
                reason.append(unavailable.get(asset, "no_fitted_model"))
            if not macro_complete:
                reason.append("macro_feature_unavailable")
            decision_available = row.signal_at <= evaluation_at
            if not decision_available:
                reason.append("signal_after_evaluation_cutoff")
            if relation is not None and macro_complete and decision_available:
                x = (raw_x - np.array(relation.center)) / np.array(relation.scale)
                prediction = float(np.r_[1, x] @ np.array(relation.coefficients))
                if not np.isfinite(prediction):
                    raise ValueError("nonfinite frozen forecast; cannot score numerical overflow")
                baseline = relation.benchmark_mean
                extrapolation = bool(
                    ((raw_x < relation.feature_min) | (raw_x > relation.feature_max)).any()
                )
            label_available = row.return_known_at <= evaluation_at and pd.notna(row[asset])
            if not label_available:
                reason.append("return_missing_or_not_known_at_evaluation")
            actual = float(row[asset]) if label_available else np.nan
            scoreable = bool(not reason)
            output.append(
                {
                    "asset": asset,
                    "start": row.start,
                    "end": row.end,
                    "signal_at": row.signal_at,
                    "return_known_at": row.return_known_at,
                    "prediction": prediction,
                    "benchmark_prediction": baseline,
                    "observed_return": actual,
                    "scored": scoreable,
                    "reason": "|".join(reason) if reason else None,
                    "outside_training_feature_range": extrapolation,
                }
            )
    path = pd.DataFrame(output)
    comparisons = []
    for asset, group in path.groupby("asset", sort=False):
        scored = group.loc[group.scored]
        result = {
            "asset": asset,
            "candidate_intervals": len(group),
            "predicted_intervals": int(group.prediction.notna().sum()),
            "scored_intervals": len(scored),
            "excluded_intervals": int((~group.scored).sum()),
            "exclusion_counts": group.loc[~group.scored, "reason"].value_counts().to_dict(),
            "mean_squared_error": None,
            "baseline_mean_squared_error": None,
            "mean_absolute_error": None,
            "baseline_mean_absolute_error": None,
            "oos_r_squared_vs_mean": None,
            "direction_accuracy": None,
        }
        if not scored.empty:
            error = scored.prediction - scored.observed_return
            baseline_error = scored.benchmark_prediction - scored.observed_return
            mse, baseline_mse = float((error**2).mean()), float((baseline_error**2).mean())
            if not np.isfinite([mse, baseline_mse]).all():
                raise ValueError("nonfinite forecast error; cannot report overflow as research")
            result.update(
                {
                    "mean_squared_error": mse,
                    "baseline_mean_squared_error": baseline_mse,
                    "mean_absolute_error": float(error.abs().mean()),
                    "baseline_mean_absolute_error": float(baseline_error.abs().mean()),
                    "oos_r_squared_vs_mean": 1 - mse / baseline_mse if baseline_mse > 0 else None,
                    "direction_accuracy": float(
                        (np.sign(scored.prediction) == np.sign(scored.observed_return)).mean()
                    ),
                }
            )
        comparisons.append(result)
    return path, {
        "status": "evaluated" if path.scored.any() else "no_scored_predictions",
        "evidence_kind": model.evidence_kind,
        "fit_at": model.fit_at,
        "evaluation_at": evaluation_at.isoformat(),
        "comparisons": comparisons,
        "macro_lineage": lineage,
        "benchmark": "per-asset mean of the same complete known training rows, frozen at fit_at; "
        "scored on exactly the model's same OOS rows",
        "limits": "Predictive association, not causation. Missing rows excluded with counts; "
        "baseline errors zero imply undefined relative R-squared. No trading/cost/NAV claim. "
        "Input return histories and classification must have independent provenance.",
    }

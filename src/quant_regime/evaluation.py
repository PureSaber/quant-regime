"""Frozen, long-only regime policies on explicitly timed return intervals."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


def intervals(frame):
    required = {"start", "end", "signal_at", "regime", "asset_return", "cash_return"}
    if not required.issubset(frame) or frame.empty:
        raise ValueError("nonempty timed return intervals required")
    out = frame.copy()
    for key in ("start", "end", "signal_at"):
        if any(pd.Timestamp(v).tzinfo is None for v in out[key]):
            raise ValueError("explicit timezone required")
        out[key] = pd.to_datetime(out[key], utc=True)
        if out[key].isna().any():
            raise ValueError("missing timestamp")
    if not out.start.is_monotonic_increasing or out.start.duplicated().any():
        raise ValueError("intervals must be ordered and unique")
    if (out.start >= out.end).any() or (out.signal_at > out.start).any():
        raise ValueError("signal must be available before interval return begins")
    if len(out) > 1 and (out.start.iloc[1:].to_numpy() != out.end.iloc[:-1].to_numpy()).any():
        raise ValueError("contiguous, nonoverlapping intervals required")
    if (
        out.regime.isna().any()
        or not out.regime.map(lambda v: isinstance(v, str) and bool(v)).all()
    ):
        raise ValueError("explicit regime labels required")
    values = out[["asset_return", "cash_return"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError("finite returns greater than -1 required")
    out[["asset_return", "cash_return"]] = values
    return out.reset_index(drop=True)


@dataclass(frozen=True)
class FrozenPolicy:
    trained_until: str
    scales: tuple[tuple[str, float], ...]
    max_scale: float
    observations: tuple[tuple[str, int], ...]

    def __post_init__(self):
        if pd.Timestamp(self.trained_until).tzinfo is None:
            raise ValueError("timezone required on trained_until")
        if not np.isfinite(self.max_scale) or not 0 < self.max_scale <= 1:
            raise ValueError("invalid frozen policy cap")
        if len(dict(self.scales)) != len(self.scales) or any(
            not np.isfinite(value) or not 0 <= value <= self.max_scale for _, value in self.scales
        ):
            raise ValueError("invalid or duplicate frozen state scales")


def fit_policy(training, *, max_scale=1.0, min_observations=3):
    """Invest at the declared cap only for states with positive training excess mean.

    This intentionally simple candidate is compared with the same-cap constant policy;
    no thresholds, states or hyperparameters are selected using test returns.
    """
    data = intervals(training)
    if not np.isfinite(max_scale) or not 0 < max_scale <= 1:
        raise ValueError("long-only cap must be in (0, 1]")
    if type(min_observations) is not int or min_observations < 2:
        raise ValueError("at least two training observations per state required")
    groups = data.assign(excess=data.asset_return - data.cash_return).groupby("regime").excess
    sizes, means = groups.size(), groups.mean()
    scales = tuple(
        (label, float(max_scale if sizes[label] >= min_observations and mean > 0 else 0))
        for label, mean in means.items()
    )
    return FrozenPolicy(
        data.end.max().isoformat(),
        scales,
        float(max_scale),
        tuple((label, int(n)) for label, n in sizes.items()),
    )


def _path(data, targets, fee):
    # Holdings are fractions of pre-trade NAV. Costs debit the cash sleeve at interval
    # end (a research convention, not an executable account). Drift is not ignored.
    before = 0.0
    rows = []
    for row, weight in zip(data.itertuples(), targets, strict=True):
        turnover = abs(weight - before)
        gross = weight * row.asset_return + (1 - weight) * row.cash_return
        cost = turnover * fee
        net = gross - cost
        if net <= -1 or not np.isfinite(net):
            raise ValueError("nonpositive research NAV")
        rows.append((weight, turnover, cost, gross, net))
        before = weight * (1 + row.asset_return) / (1 + net)
    return pd.DataFrame(rows, columns=["weight", "turnover", "cost", "gross", "net"])


def stability(labels):
    labels = list(labels)
    runs, transitions = [], {}
    for label in labels:
        if runs and runs[-1]["regime"] == label:
            runs[-1]["length"] += 1
        else:
            if runs:
                key = (runs[-1]["regime"], label)
                transitions[key] = transitions.get(key, 0) + 1
            runs.append({"regime": label, "length": 1})
    return {
        "switches": max(0, len(runs) - 1),
        "switch_rate": (len(runs) - 1) / (len(labels) - 1) if len(labels) > 1 else None,
        "runs": runs,
        "transitions": [{"from": a, "to": b, "count": n} for (a, b), n in transitions.items()],
    }


def evaluate_policy(policy, testing, *, cost_bps, evidence_kind):
    data = intervals(testing)
    if evidence_kind not in {"synthetic", "retrospective", "historical_pit"}:
        raise ValueError("explicit supported evidence kind required")
    if pd.Timestamp(policy.trained_until) > data.start.min():
        raise ValueError("training overlaps test returns")
    if not np.isfinite(cost_bps) or not 0 <= cost_bps <= 1000:
        raise ValueError("cost_bps must be finite in [0,1000]")
    scales = dict(policy.scales)
    weights = [scales.get(label, 0.0) for label in data.regime]
    active = _path(data, weights, cost_bps / 10000)
    benchmark = _path(data, [policy.max_scale] * len(data), cost_bps / 10000)
    result = pd.concat(
        [data, active.add_prefix("policy_"), benchmark.add_prefix("benchmark_")], axis=1
    )
    result["state_switch"] = result.regime.ne(result.regime.shift())
    result.loc[0, "state_switch"] = False
    by_state = []
    for label, group in result.groupby("regime"):
        by_state.append(
            {
                "regime": label,
                "observations": len(group),
                "policy_compounded": float((1 + group.policy_net).prod() - 1),
                "benchmark_compounded": float((1 + group.benchmark_net).prod() - 1),
                "mean_excess": float((group.policy_net - group.benchmark_net).mean()),
                "cost_sum": float(group.policy_cost.sum()),
            }
        )
    summary = {
        "evidence_kind": evidence_kind,
        "trained_until": policy.trained_until,
        "frozen_scales": scales,
        "unknown_states_flat": sorted(set(data.regime) - set(scales)),
        "policy_return": float((1 + result.policy_net).prod() - 1),
        "benchmark_return": float((1 + result.benchmark_net).prod() - 1),
        "cost_sum": float(result.policy_cost.sum()),
        "cost_on_switch_intervals": float(result.loc[result.state_switch, "policy_cost"].sum()),
        "by_state": by_state,
        "stability": stability(data.regime),
        "terminal_weight": float(weights[-1]),
        "terminal_liquidation_included": False,
        "limits": "Interval-end cost debit; open-ended research, not an execution ledger. "
        "State-conditioned compounding is descriptive. No significance or forward-validation claim.",
    }
    return result, summary


def compare_folds(data, folds, *, cost_bps, evidence_kind, max_scale=1.0, min_observations=3):
    """Evaluate predeclared row-boundary folds separately; no holdout model selection.

    Each fold is (train_start, test_start, test_end), all zero-based, end-exclusive.
    Test windows cannot overlap. Return comparisons are per-fold, never stitched into
    a fictitious self-financing account with uncharged fold-boundary liquidation.
    """
    data = intervals(data)
    previous_end, tables, summaries = 0, [], []
    for fold, boundaries in enumerate(folds):
        if len(boundaries) != 3 or any(type(x) is not int for x in boundaries):
            raise ValueError("three integer fold boundaries required")
        start, test, end = boundaries
        if not 0 <= start < test < end <= len(data) or test < previous_end:
            raise ValueError("invalid or overlapping test folds")
        policy = fit_policy(
            data.iloc[start:test], max_scale=max_scale, min_observations=min_observations
        )
        path, summary = evaluate_policy(
            policy, data.iloc[test:end], cost_bps=cost_bps, evidence_kind=evidence_kind
        )
        path["fold"] = fold
        tables.append(path)
        summaries.append({"fold": fold, **summary})
        previous_end = end
    if not tables:
        raise ValueError("at least one predeclared fold required")
    return pd.concat(tables, ignore_index=True), summaries


def macro_linkage(decisions, observations, *, required_series, max_age_days=90):
    """Reuse QDK vintage/release gates separately at each decision, with lineage."""
    from quant_regime.macro import macro_context

    data = intervals(decisions)
    rows = []
    for row in data.itertuples():
        context = macro_context(
            observations, row.signal_at, required_series=required_series, max_age_days=max_age_days
        )
        for series in required_series:
            value = context["values"].get(series)
            rows.append(
                {
                    "signal_at": row.signal_at.isoformat(),
                    "regime": row.regime,
                    "series_id": series,
                    "available": value is not None,
                    "reason": context["unavailable"].get(series),
                    **(value or {}),
                }
            )
    return pd.DataFrame(rows)

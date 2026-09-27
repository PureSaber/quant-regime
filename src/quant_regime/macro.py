"""Point-in-time macro context, with explicit missing/stale signals (no backfill)."""

from quant_data_kit.financial.common import utc
from quant_data_kit.financial.macro import macro_asof


def macro_context(observations, at, *, required_series, max_age_days=90):
    if type(max_age_days) is not int or max_age_days < 0:
        raise ValueError("max_age_days must be nonnegative")
    visible = macro_asof(observations, at, latest_observation=True).set_index("series_id")
    values, unavailable = {}, {}
    for series in required_series:
        if series not in visible.index:
            unavailable[series] = "not_released"
            continue
        row = visible.loc[series]
        age = (utc(at).tz_localize(None).normalize() - row.observation_date).days
        if row.value is None or age > max_age_days:
            unavailable[series] = "missing_or_stale"
            continue
        values[series] = {
            "value": str(row.value),
            "unit": row.unit,
            "available_at": row.available_at.isoformat(),
            "observation_date": str(row.observation_date.date()),
            "evidence_id": row.evidence_id,
        }
    return {
        "as_of": utc(at).isoformat(),
        "values": values,
        "unavailable": unavailable,
        "complete": not unavailable,
        "policy": "context_only_no_implicit_regime_override",
    }

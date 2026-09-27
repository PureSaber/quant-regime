import pandas as pd

from quant_regime.macro import macro_context


def test_macro_context_never_backfills_an_unreleased_or_stale_value():
    data = pd.DataFrame(
        [
            {
                "series_id": "CPI",
                "observation_date": "2024-01-01",
                "vintage_date": "2024-02-01",
                "released_at": "2024-02-01T13:30:00Z",
                "available_at": "2024-02-01T13:30:00Z",
                "value": "3.1",
                "unit": "percent",
                "source": "synthetic",
                "evidence_id": "cpi",
            }
        ]
    )
    assert not macro_context(data, "2024-02-01T13:29:59Z", required_series=["CPI"])["complete"]
    assert macro_context(data, "2024-02-01T13:30:00Z", required_series=["CPI"])["complete"]
    assert not macro_context(data, "2024-09-01T13:30:00Z", required_series=["CPI"])["complete"]

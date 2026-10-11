"""Orthogonal synthetic macro changes with a deliberate OOS commodity relation break."""

import numpy as np
import pandas as pd

ASSETS = ("equity", "bond", "commodity")
SERIES = ("growth", "inflation")
INTERCEPTS = np.array([0.002, 0.001, 0.003])
SLOPES = np.array([[0.01, -0.02], [-0.004, -0.012], [0.018, 0.006]])


def synthetic_linkage():
    dates = pd.date_range("2022-01-01", periods=37, freq="MS", tz="UTC")
    dates += pd.Timedelta(hours=14)
    changes = np.tile([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]], (9, 1))
    returns = changes @ SLOPES.T + INTERCEPTS
    # This prespecified stress is not a model selected with holdout outcomes.
    returns[24:, 2] = 0.003 - 0.018 * changes[24:, 0] + 0.006 * changes[24:, 1]
    frame = pd.DataFrame(returns, columns=ASSETS)
    frame["start"], frame["end"] = dates[:-1], dates[1:]
    frame["signal_at"], frame["return_known_at"] = dates[:-1], dates[1:]
    observations = []
    for column, (name, initial, unit) in enumerate(
        [("growth", 100.0, "index_points"), ("inflation", 3.0, "percent")]
    ):
        levels = np.r_[initial, initial + changes[:, column].cumsum()]
        releases = [
            pd.Timestamp("2021-12-01T13:00:00Z"),
            *[d - pd.Timedelta(hours=1) for d in dates[:-1]],
        ]
        for i, (level, release) in enumerate(zip(levels, releases, strict=True)):
            observed = release.tz_localize(None).normalize() - pd.offsets.MonthBegin(1)
            observations.append(
                {
                    "series_id": name,
                    "observation_date": observed,
                    "vintage_date": release.tz_localize(None).normalize(),
                    "released_at": release,
                    "available_at": release,
                    "value": str(level),
                    "unit": unit,
                    "source": "deterministic-synthetic-linkage",
                    "evidence_id": f"{name}:initial:{i}",
                }
            )
    return frame, pd.DataFrame(observations), changes

"""Checks that no feature uses information from inside the forecast window.

Run:  python tests/test_no_leakage.py   (or with pytest if installed)
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data import generate  # noqa: E402
from features import HORIZON, build_features  # noqa: E402


def test_lags_never_reach_into_the_forecast_window():
    df = build_features(generate())
    sku = df[df["sku"] == "SPOON-140"].set_index("date").sort_index()
    week = sku.index[100]
    # lag_8 must equal demand exactly HORIZON weeks earlier — never anything newer
    assert sku.loc[week, f"lag_{HORIZON}"] == sku["demand"].iloc[100 - HORIZON]


def test_changing_recent_demand_does_not_change_features():
    raw = generate()
    base = build_features(raw)
    cutoff = raw["date"].sort_values().unique()[120]
    window = (raw["date"] > cutoff) & (raw["date"] <= cutoff + pd.Timedelta(weeks=HORIZON))
    tampered = raw.copy()
    tampered.loc[window, "demand"] *= 10  # change the future we are trying to forecast
    after = build_features(tampered)
    cols = [c for c in base.columns if c.startswith(("lag_", "roll_"))]
    rows = (base["date"] > cutoff) & (base["date"] <= cutoff + pd.Timedelta(weeks=HORIZON))
    pd.testing.assert_frame_equal(base.loc[rows, cols], after.loc[rows, cols])


if __name__ == "__main__":
    test_lags_never_reach_into_the_forecast_window()
    test_changing_recent_demand_does_not_change_features()
    print("All leakage tests passed.")

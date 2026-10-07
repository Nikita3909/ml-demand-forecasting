"""Feature engineering for a global (all-SKU) forecasting model.

Direct multi-step strategy: to forecast up to HORIZON weeks ahead without
using any value that would be unknown at forecast time, every demand-based
feature uses lags of at least HORIZON weeks. Calendar, price, promo and
festival flags are treated as known in advance (they are planned).
"""
from __future__ import annotations

import pandas as pd

HORIZON = 8  # forecast 8 weeks ahead


def build_features(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    df = df.sort_values(["sku", "date"]).copy()
    g = df.groupby("sku")["demand"]
    lags = [horizon, horizon + 1, horizon + 2, horizon + 4, 52]
    for lag in lags:
        df[f"lag_{lag}"] = g.shift(lag)
    shifted = g.shift(horizon)
    df["roll_mean_4"] = shifted.groupby(df["sku"]).transform(lambda s: s.rolling(4).mean())
    df["roll_mean_12"] = shifted.groupby(df["sku"]).transform(lambda s: s.rolling(12).mean())
    df["roll_std_12"] = shifted.groupby(df["sku"]).transform(lambda s: s.rolling(12).std())
    week = df["date"].dt.isocalendar().week.astype(int)
    df["week_of_year"] = week
    df["month"] = df["date"].dt.month
    df["sku_code"] = df["sku"].astype("category").cat.codes
    return df


FEATURES = [
    "lag_8", "lag_9", "lag_10", "lag_12", "lag_52",
    "roll_mean_4", "roll_mean_12", "roll_std_12",
    "week_of_year", "month", "price", "promo", "festival", "sku_code",
]

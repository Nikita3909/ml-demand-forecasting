"""Forecasting models. Every model has the same interface:

    forecast(train, test) -> np.ndarray of predictions for the rows in `test`

`train` holds all history up to the forecast cutoff; `test` holds the next
HORIZON weeks for every SKU (with planned price/promo/festival known).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from features import FEATURES

# ── Baselines ────────────────────────────────────────────────────────────────

def naive(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Next weeks = last observed week."""
    last = train.sort_values("date").groupby("sku")["demand"].last()
    return test["sku"].map(last).to_numpy(float)


def seasonal_naive(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Next weeks = same week last year."""
    hist = train.set_index(["sku", "date"])["demand"]
    keys = zip(test["sku"], test["date"] - pd.Timedelta(weeks=52))
    return np.array([hist.get(k, np.nan) for k in keys], dtype=float)


def moving_average(train: pd.DataFrame, test: pd.DataFrame, window: int = 8) -> np.ndarray:
    avg = train.sort_values("date").groupby("sku")["demand"].apply(lambda s: s.tail(window).mean())
    return test["sku"].map(avg).to_numpy(float)


# ── Statistical model ────────────────────────────────────────────────────────

def holt_winters(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """One damped-trend, yearly-seasonal Holt-Winters model per SKU."""
    preds = pd.Series(index=test.index, dtype=float)
    for sku, hist in train.groupby("sku"):
        y = hist.sort_values("date")["demand"].to_numpy(float)
        rows = test[test["sku"] == sku].sort_values("date")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ExponentialSmoothing(
                y, trend="add", damped_trend=True, seasonal="add", seasonal_periods=52,
                initialization_method="estimated",
            ).fit()
        preds.loc[rows.index] = fit.forecast(len(rows))
    return preds.loc[test.index].to_numpy(float)


# ── Machine learning (one global model for all SKUs) ─────────────────────────
# The target is scaled by each SKU's recent level (roll_mean_12) so a single model
# can learn shared patterns across products that sell 150 or 500 cartons a week.

SCALED = ["lag_8", "lag_9", "lag_10", "lag_12", "lag_52", "roll_mean_4", "roll_std_12"]


def _scaled_xy(df: pd.DataFrame):
    level = df["roll_mean_12"].clip(lower=1)
    X = df[FEATURES].copy()
    for col in SCALED:
        X[col] = X[col] / level
    X = X.drop(columns=["roll_mean_12"])
    y = df["demand"] / level if "demand" in df else None
    return X, y, level


def _ml(make_model):
    def forecast(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        tr = train.dropna(subset=FEATURES)
        X, y, _ = _scaled_xy(tr)
        model = make_model()
        model.fit(X, y)
        Xt, _, level = _scaled_xy(test)
        forecast.model = model  # kept for feature importance
        return model.predict(Xt) * level.to_numpy()
    return forecast


ridge = _ml(lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)))
random_forest = _ml(lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=-1))
gradient_boosting = _ml(lambda: HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=15,
                                                             l2_regularization=1.0, random_state=0))

MODELS = {
    "Naive (last week)": naive,
    "Seasonal naive (last year)": seasonal_naive,
    "Moving average (8 wk)": moving_average,
    "Holt-Winters": holt_winters,
    "Ridge regression": ridge,
    "Random forest": random_forest,
    "Gradient boosting": gradient_boosting,
}

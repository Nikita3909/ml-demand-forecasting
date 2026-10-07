"""Rolling-origin backtest: compare 7 forecasting models on the same folds.

Run from the project folder:  python src/run.py
Outputs go to results/ (metrics table + charts).
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

sys.path.insert(0, os.path.dirname(__file__))
from data import generate  # noqa: E402
from features import FEATURES, HORIZON, build_features  # noqa: E402
from models import MODELS, _scaled_xy, gradient_boosting  # noqa: E402

N_FOLDS = 4
OUT = "results"
COLORS = {"Gradient boosting": "#16a34a", "Random forest": "#22c55e", "Ridge regression": "#0ea5e9",
          "Holt-Winters": "#6366f1", "Moving average (8 wk)": "#94a3b8", "Seasonal naive (last year)": "#cbd5e1",
          "Naive (last week)": "#e2e8f0"}


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    e = p - y
    return {
        "WAPE %": 100 * np.abs(e).sum() / y.sum(),
        "MAE": np.abs(e).mean(),
        "RMSE": np.sqrt((e ** 2).mean()),
        "Bias %": 100 * e.sum() / y.sum(),
    }


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    os.makedirs("data", exist_ok=True)
    raw = generate()
    raw.to_csv("data/weekly_demand.csv", index=False)
    df = build_features(raw)

    dates = np.sort(df["date"].unique())
    cutoffs = [dates[-1 - HORIZON * k] for k in range(N_FOLDS, 0, -1)]  # last N_FOLDS x HORIZON weeks
    print(f"{len(raw):,} rows · {raw['sku'].nunique()} SKUs · {len(dates)} weeks · "
          f"{N_FOLDS} folds x {HORIZON}-week horizon")

    preds = []
    for fold, cutoff in enumerate(cutoffs, 1):
        train = df[df["date"] <= cutoff]
        test = df[(df["date"] > cutoff) & (df["date"] <= cutoff + np.timedelta64(HORIZON, "W"))].copy()
        test["h"] = test.groupby("sku")["date"].rank().astype(int)
        for name, fn in MODELS.items():
            test[name] = fn(train, test)
        test["fold"] = fold
        preds.append(test)
        print(f"  fold {fold}: train to {pd.Timestamp(cutoff).date()}, test {len(test)} rows")
    res = pd.concat(preds)

    # ── Metrics table ──
    table = pd.DataFrame({name: metrics(res["demand"].to_numpy(float), res[name].to_numpy(float)) for name in MODELS}).T
    table = table.sort_values("WAPE %").round(2)
    per_fold = res.groupby("fold").apply(
        lambda f: pd.Series({n: metrics(f["demand"].to_numpy(float), f[n].to_numpy(float))["WAPE %"] for n in MODELS}),
        include_groups=False)
    table["WAPE std (folds)"] = per_fold.std().round(2)
    table.to_csv(f"{OUT}/metrics.csv")
    print("\n" + table.to_string())

    best = table.index[0]
    base = "Seasonal naive (last year)"
    weekly_err = res.groupby(["fold", "date"]).apply(
        lambda f: pd.Series({n: np.abs(f[n] - f["demand"]).sum() for n in (best, base)}), include_groups=False)
    saved = (weekly_err[base] - weekly_err[best]).mean()
    summary = {
        "best_model": best,
        "best_wape": table.loc[best, "WAPE %"],
        "baseline_wape": table.loc[base, "WAPE %"],
        "error_reduction_pct": round(100 * (1 - table.loc[best, "WAPE %"] / table.loc[base, "WAPE %"]), 1),
        "cartons_less_error_per_week": round(saved),
    }
    pd.Series(summary).to_csv(f"{OUT}/summary.csv", header=False)
    print("\n", summary)

    # ── Chart 1: model comparison ──
    fig, ax = plt.subplots(figsize=(8, 4.2))
    t = table.sort_values("WAPE %", ascending=True)
    ax.barh(t.index, t["WAPE %"], xerr=t["WAPE std (folds)"], color=[COLORS[n] for n in t.index], capsize=3)
    for i, (v, sd) in enumerate(zip(t["WAPE %"], t["WAPE std (folds)"])):
        ax.text(v + sd + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)
    ax.invert_yaxis(); ax.set_xlabel("WAPE % (lower is better) — mean ± std over 4 backtest folds")
    ax.set_title("8-week demand forecast: model comparison", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/model_comparison.png", dpi=150); plt.close(fig)

    # ── Chart 2: error by horizon ──
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in [best, "Holt-Winters", base]:
        by_h = res.groupby("h").apply(lambda f: 100 * np.abs(f[name] - f["demand"]).sum() / f["demand"].sum(), include_groups=False)
        ax.plot(by_h.index, by_h.values, marker="o", label=name, color=COLORS[name] if name != base else "#94a3b8")
    ax.set_xlabel("Weeks ahead"); ax.set_ylabel("WAPE %"); ax.legend(frameon=False)
    ax.set_title("Error grows with forecast distance", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/error_by_horizon.png", dpi=150); plt.close(fig)

    # ── Chart 3: forecast vs actual for one SKU ──
    sku = "STRAW-6MM"
    hist = raw[(raw["sku"] == sku) & (raw["date"] > pd.Timestamp(cutoffs[0]) - pd.Timedelta(weeks=40))]
    fc = res[res["sku"] == sku].sort_values("date")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(hist["date"], hist["demand"], color="#0f172a", lw=1.6, label="Actual")
    ax.plot(fc["date"], fc[best], color=COLORS[best], lw=2, label=best)
    ax.plot(fc["date"], fc[base], color="#94a3b8", lw=1.2, ls="--", label=base)
    for c in cutoffs:
        ax.axvline(pd.Timestamp(c), color="#e2e8f0", lw=1)
    ax.set_title(f"{sku}: backtest forecasts (each band = one 8-week fold)", loc="left", fontweight="bold")
    ax.set_ylabel("Cartons per week"); ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/forecast_example.png", dpi=150); plt.close(fig)

    # ── Chart 4: feature importance (permutation, last fold) ──
    last_cut = cutoffs[-1]
    train = df[df["date"] <= last_cut]
    test = df[(df["date"] > last_cut)]
    gradient_boosting(train, test)
    Xt, yt, _ = _scaled_xy(test)
    imp = permutation_importance(gradient_boosting.model, Xt, yt, n_repeats=10, random_state=0)
    s = pd.Series(imp.importances_mean, index=Xt.columns).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(s.index, s.values, color="#16a34a")
    ax.set_title("What drives the gradient boosting forecast (permutation importance, last fold)", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/feature_importance.png", dpi=150); plt.close(fig)
    print(f"\nCharts saved to {OUT}/")


if __name__ == "__main__":
    main()

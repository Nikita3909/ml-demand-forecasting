"""Synthetic weekly demand data for an eco-packaging manufacturer.

The data is generated, not real. It is built to behave like real B2B demand:
long-term trend, yearly seasonality (summer peak for straws and ice-cream
sticks), festival spikes (Diwali season), promotions, price changes and
random noise. A fixed seed makes every run identical.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SKUS = {
    # sku: (category, base weekly cartons, yearly trend %, summer seasonality strength, price per carton)
    "SPOON-140": ("Wooden Cutlery", 420, 0.12, 0.10, 640),
    "FORK-160": ("Wooden Cutlery", 310, 0.10, 0.10, 660),
    "KNIFE-160": ("Wooden Cutlery", 180, 0.08, 0.10, 670),
    "STRAW-6MM": ("Paper Straw", 520, 0.18, 0.35, 480),
    "STRAW-8MM": ("Paper Straw", 260, 0.15, 0.35, 520),
    "STIRRER-140": ("Stirrer", 390, 0.06, -0.05, 380),
    "ICECREAM-STICK": ("Ice Cream", 470, 0.09, 0.55, 300),
    "ICECREAM-SPOON": ("Ice Cream", 220, 0.11, 0.50, 340),
    "SKEWER-150": ("Bamboo", 150, 0.04, 0.05, 450),
    "CUTLERY-SET-3": ("Cutlery Set", 130, 0.25, 0.15, 980),
}

# Approximate Diwali week starts (festival season demand from caterers and events)
FESTIVAL_WEEKS = ["2023-11-06", "2024-10-28", "2025-10-13", "2026-11-02"]


def generate(start: str = "2023-07-03", end: str = "2026-06-29", seed: int = 42) -> pd.DataFrame:
    """Return one row per SKU per week: date, sku, category, price, promo, festival, demand."""
    rng = np.random.default_rng(seed)
    weeks = pd.date_range(start, end, freq="W-MON")
    festival = set(pd.to_datetime(FESTIVAL_WEEKS))
    rows = []
    for sku, (category, base, trend, summer, price) in SKUS.items():
        level = base
        for i, week in enumerate(weeks):
            t = i / 52.0
            # Summer peak around week 20 (mid-May) for summer-sensitive products
            season = 1 + summer * np.cos(2 * np.pi * (week.isocalendar().week - 20) / 52)
            fest = any(abs((week - f).days) <= 14 for f in festival)
            promo = rng.random() < 0.08
            cur_price = price * (1 + 0.04 * np.floor(t))  # yearly price increase
            price_effect = (price / cur_price) ** 1.2
            mean = base * (1 + trend) ** t * season * price_effect
            mean *= 1.35 if fest else 1.0
            mean *= 1.25 if promo else 1.0
            # AR(1) noise so weeks are related (like real order flow)
            level = 0.6 * level + 0.4 * mean
            demand = max(0, rng.normal(level, 0.12 * level))
            rows.append(
                {
                    "date": week,
                    "sku": sku,
                    "category": category,
                    "price": round(cur_price, 2),
                    "promo": int(promo),
                    "festival": int(fest),
                    "demand": round(demand),
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/weekly_demand.csv", index=False)
    print(f"Saved {len(df):,} rows, {df['sku'].nunique()} SKUs, {df['date'].nunique()} weeks")

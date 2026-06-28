"""
generate_data.py
================
Synthetic Q-Commerce demand dataset generator.
Produces ~675 rows of realistic hourly transaction records across:
  - 3 dark stores  (Mumbai · Bangalore · Delhi)
  - 3 SKUs         (Milk 1L · Eggs 12-pack · Bread 500g)
  - 25 days        (April 1 – April 25, 2024)
  - 3 peak hours   (09:00  18:00  21:00)

Demand patterns deliberately encoded:
  • Rain     → +30–40 % uplift (comfort buying)
  • Weekend  → +40–45 % uplift
  • Holiday  → +65 % uplift
  • Local event → +28 % uplift
  • Extreme Heat → −10–12 % suppression
  • Evening (18:00) → highest demand window
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# ────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────

STORES = {
    "DS001_Andheri_MUM":     {"city": "Mumbai",    "store_mult": 1.20},
    "DS002_Koramangala_BLR": {"city": "Bangalore", "store_mult": 1.00},
    "DS003_CP_DEL":          {"city": "Delhi",     "store_mult": 0.92},
}

SKUS = {
    "SKU001_Milk_1L": {
        "base_demand": 9,
        "price": 52,
        "shelf_life_days": 5,
        "margin": "low",
        "weather_mult": {
            "Clear": 1.00, "Cloudy": 1.10, "Rain": 1.40, "Extreme Heat": 0.85
        },
    },
    "SKU002_Eggs_12pk": {
        "base_demand": 5,
        "price": 78,
        "shelf_life_days": 14,
        "margin": "medium",
        "weather_mult": {
            "Clear": 1.00, "Cloudy": 1.05, "Rain": 1.25, "Extreme Heat": 0.90
        },
    },
    "SKU003_Bread_500g": {
        "base_demand": 6,
        "price": 40,
        "shelf_life_days": 3,
        "margin": "low",
        "weather_mult": {
            "Clear": 1.00, "Cloudy": 1.08, "Rain": 1.30, "Extreme Heat": 0.88
        },
    },
}

# Representative peak hours per day
PEAK_HOURS = [9, 18, 21]

# Day-of-week multipliers (Mon=0 … Sun=6)
DOW_MULT = {0: 1.00, 1: 0.92, 2: 0.95, 3: 1.00, 4: 1.15, 5: 1.45, 6: 1.38}

# Hour-of-day multipliers
HOUR_MULT = {9: 1.35, 18: 1.60, 21: 1.25}

# Indian public holidays during the window
PUBLIC_HOLIDAYS = {
    datetime(2024, 4, 14).date(),   # Dr. Ambedkar Jayanti
    datetime(2024, 4, 17).date(),   # Ram Navami
    datetime(2024, 4, 21).date(),   # Easter Sunday
}

# Local events (IPL matches, festivals)
LOCAL_EVENTS = {
    datetime(2024, 4, 5).date(),    # IPL Match
    datetime(2024, 4, 12).date(),   # Local Festival
    datetime(2024, 4, 20).date(),   # IPL Match
}

# Weather options and base probabilities
WEATHER_OPTIONS = ["Clear", "Cloudy", "Rain", "Extreme Heat"]
WEATHER_PROBS   = [0.45, 0.25, 0.20, 0.10]

# Base temperature per weather condition (°C)
TEMP_BASE = {"Clear": 30, "Cloudy": 26, "Rain": 21, "Extreme Heat": 41}


# ────────────────────────────────────────────────────────────────────────────
# Generator
# ────────────────────────────────────────────────────────────────────────────

def generate_qcommerce_data(seed: int = 42) -> pd.DataFrame:
    """Return a synthetic transaction DataFrame (≈675 rows)."""
    np.random.seed(seed)

    start_date = datetime(2024, 4, 1)
    num_days   = 25                         # 3 × 3 × 3 × 25 = 675 rows

    # Pre-assign weather per day (consistent within a day)
    weather_by_day: dict[object, tuple] = {}
    for i in range(num_days):
        d = (start_date + timedelta(days=i)).date()
        w = np.random.choice(WEATHER_OPTIONS, p=WEATHER_PROBS)
        t = TEMP_BASE[w] + np.random.uniform(-2, 2)
        weather_by_day[d] = (w, round(t, 1))

    rows = []

    for i in range(num_days):
        date = start_date + timedelta(days=i)
        d    = date.date()
        weather, temperature = weather_by_day[d]

        is_holiday    = int(d in PUBLIC_HOLIDAYS)
        is_event      = int(d in LOCAL_EVENTS)
        is_weekend    = int(date.weekday() >= 5)
        dow           = date.weekday()

        holiday_mult  = 1.65 if is_holiday else 1.00
        event_mult    = 1.28 if is_event   else 1.00
        dow_mult      = DOW_MULT[dow]

        for hour in PEAK_HOURS:
            ts       = date + timedelta(hours=hour)
            hour_mult = HOUR_MULT[hour]

            for store_id, store_info in STORES.items():
                sm = store_info["store_mult"]

                for sku_id, sku in SKUS.items():
                    wm = sku["weather_mult"][weather]

                    # Expected demand (product of all multipliers)
                    expected = (
                        sku["base_demand"]
                        * sm
                        * hour_mult
                        * dow_mult
                        * holiday_mult
                        * event_mult
                        * wm
                    )

                    # Poisson noise → realistic count data
                    qty = max(0, int(np.random.poisson(expected)))

                    # Deterministic stockout when expected exceeds threshold
                    stockout_thresh = sku["base_demand"] * sm * 2.3
                    stockout        = int(expected > stockout_thresh)
                    if stockout:
                        qty = int(stockout_thresh)
                    elif np.random.random() < 0.025:
                        # Occasional random stockout (~2.5 %)
                        stockout = 1
                        qty      = max(0, qty - np.random.randint(1, 4))

                    rows.append(
                        {
                            "timestamp":           ts,
                            "store_id":            store_id,
                            "sku_id":              sku_id,
                            "quantity_sold":       qty,
                            "price_at_sale":       round(
                                sku["price"] * np.random.uniform(0.95, 1.05), 2
                            ),
                            "stockout_flag":       stockout,
                            "weather_condition":   weather,
                            "temperature_celsius": temperature,
                            "public_holiday_flag": is_holiday,
                            "local_event_flag":    is_event,
                            "is_weekend":          is_weekend,
                            "day_of_week":         dow,
                            "hour_of_day":         hour,
                        }
                    )

    df = (
        pd.DataFrame(rows)
        .sort_values(["store_id", "sku_id", "timestamp"])
        .reset_index(drop=True)
    )
    return df


# ────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = generate_qcommerce_data()
    df.to_csv("synthetic_data.csv", index=False)
    print(f"✅  Saved synthetic_data.csv  →  {len(df):,} rows")
    print(df.head(10).to_string())
    print("\nColumn dtypes:\n", df.dtypes)
    print("\nBasic stats:\n", df.describe())

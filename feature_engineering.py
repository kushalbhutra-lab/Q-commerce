"""
feature_engineering.py
=======================
Transforms the raw transaction ledger into a rich feature matrix
ready for supervised ML (RF / XGBoost / LightGBM) and baseline
time-series (Prophet) models.

Feature groups created
──────────────────────
  1. Temporal         – hour, DoW, DoM, month, week-of-year
  2. Cyclical         – sine / cosine encoding for hour & DoW
  3. Lag              – demand 1, 2, 3, and 7 periods ago
  4. Rolling windows  – mean, std, max over 3 & 7 periods
  5. Exponential MA   – EMA-3 and EMA-7
  6. Exogenous        – temperature, weather label, rain / heat flags
  7. Interaction      – rain×evening, weekend×evening, holiday×evening
  8. Entity           – label-encoded store_id & sku_id
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ── Canonical feature columns used by tree models ────────────────────────────
FEATURE_COLS: list[str] = [
    # --- temporal ---
    "hour", "day_of_week", "day_of_month", "month", "week_of_year", "is_weekend",
    # --- cyclical ---
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    # --- lag ---
    "lag_1", "lag_2", "lag_3", "lag_7",
    # --- rolling ---
    "rolling_mean_3", "rolling_mean_7",
    "rolling_std_3",  "rolling_std_7",
    "rolling_max_3",  "rolling_max_7",
    # --- exponential MA ---
    "ema_3", "ema_7",
    # --- exogenous ---
    "temperature_celsius", "weather_encoded",
    "rain_flag", "heat_flag",
    # --- interaction ---
    "rain_evening", "weekend_evening", "holiday_evening",
    # --- binary flags ---
    "public_holiday_flag", "local_event_flag", "stockout_flag",
    # --- entity ---
    "store_encoded", "sku_encoded",
]


# ── Main feature engineering function ────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich *df* with all feature groups.

    Parameters
    ----------
    df : pd.DataFrame
        Raw transaction ledger (output of generate_data.py).

    Returns
    -------
    pd.DataFrame
        Original columns + all engineered features.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["store_id", "sku_id", "timestamp"]).reset_index(drop=True)

    # ── 1. Temporal features ─────────────────────────────────────────────────
    df["hour"]         = df["timestamp"].dt.hour
    df["day_of_week"]  = df["timestamp"].dt.dayofweek
    df["day_of_month"] = df["timestamp"].dt.day
    df["month"]        = df["timestamp"].dt.month
    df["week_of_year"] = df["timestamp"].dt.isocalendar().week.astype(int)
    # is_weekend already present from raw data; keep it

    # ── 2. Cyclical encoding ─────────────────────────────────────────────────
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # ── 3 & 4. Lag + rolling features (grouped per store × SKU) ─────────────
    grp = df.groupby(["store_id", "sku_id"])

    for lag in [1, 2, 3, 7]:
        df[f"lag_{lag}"] = grp["quantity_sold"].shift(lag)

    for w in [3, 7]:
        df[f"rolling_mean_{w}"] = grp["quantity_sold"].transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        )
        df[f"rolling_std_{w}"] = grp["quantity_sold"].transform(
            lambda x: x.rolling(w, min_periods=1).std().fillna(0)
        )
        df[f"rolling_max_{w}"] = grp["quantity_sold"].transform(
            lambda x: x.rolling(w, min_periods=1).max()
        )

    # ── 5. Exponential moving averages ───────────────────────────────────────
    df["ema_3"] = grp["quantity_sold"].transform(lambda x: x.ewm(span=3).mean())
    df["ema_7"] = grp["quantity_sold"].transform(lambda x: x.ewm(span=7).mean())

    # Fill lag NaN (first few rows per group) with rolling mean
    for lag in [1, 2, 3, 7]:
        df[f"lag_{lag}"] = df[f"lag_{lag}"].fillna(df["rolling_mean_3"])

    # ── 6. Exogenous / weather features ─────────────────────────────────────
    weather_map = {"Clear": 0, "Cloudy": 1, "Rain": 2, "Extreme Heat": 3}
    df["weather_encoded"] = df["weather_condition"].map(weather_map).fillna(0).astype(int)
    df["rain_flag"]  = (df["weather_condition"] == "Rain").astype(int)
    df["heat_flag"]  = (df["weather_condition"] == "Extreme Heat").astype(int)

    # ── 7. Interaction features ──────────────────────────────────────────────
    is_evening            = (df["hour"] >= 18).astype(int)
    df["rain_evening"]    = df["rain_flag"]            * is_evening
    df["weekend_evening"] = df["is_weekend"]           * is_evening
    df["holiday_evening"] = df["public_holiday_flag"]  * is_evening

    # ── 8. Entity label encoding ─────────────────────────────────────────────
    le_store = LabelEncoder()
    le_sku   = LabelEncoder()
    df["store_encoded"] = le_store.fit_transform(df["store_id"])
    df["sku_encoded"]   = le_sku.fit_transform(df["sku_id"])

    return df


# ── Helper ────────────────────────────────────────────────────────────────────

def get_feature_columns() -> list[str]:
    """Return the canonical feature column list."""
    return FEATURE_COLS.copy()


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from generate_data import generate_qcommerce_data

    raw = generate_qcommerce_data()
    enriched = engineer_features(raw)
    print(f"✅  Feature engineering complete: {enriched.shape}")
    print("\nNew columns added:")
    new_cols = [c for c in enriched.columns if c not in raw.columns]
    for c in new_cols:
        print(f"  • {c}")
    print("\nSample (first 5 rows of selected features):")
    print(enriched[["timestamp", "store_id", "sku_id", "quantity_sold",
                     "lag_1", "rolling_mean_3", "rain_flag", "hour_sin"]].head())

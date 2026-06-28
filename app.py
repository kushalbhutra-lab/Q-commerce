"""
app.py  —  Q-Commerce Hyper-Local Demand Forecasting Dashboard
==============================================================
Run:  streamlit run app.py
Requires: synthetic_data.csv in same directory (auto-generated if absent).

Navigation:
  🏠  Home & Data Overview
  📊  Descriptive Analysis
  🔍  Diagnostic Analysis
  🤖  Predictive Modeling
  💡  Business Insights
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Q-Commerce Demand Forecasting Engine",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global colour palette ─────────────────────────────────────────────────────
PALETTE = {
    "Random Forest": "#2196F3",
    "XGBoost":       "#FF5722",
    "LightGBM":      "#4CAF50",
    "Prophet":       "#FF9800",
    "Actual":        "#9C27B0",
}
WEATHER_COLORS = {
    "Clear":        "#FFC107",
    "Cloudy":       "#90A4AE",
    "Rain":         "#1565C0",
    "Extreme Heat": "#E53935",
}
PLOTLY_TEMPLATE = "plotly_white"

# ── Feature list (must match feature_engineering.py) ─────────────────────────
FEATURE_COLS: list[str] = [
    "hour", "day_of_week", "day_of_month", "month", "week_of_year", "is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "lag_1", "lag_2", "lag_3", "lag_7",
    "rolling_mean_3", "rolling_mean_7",
    "rolling_std_3",  "rolling_std_7",
    "rolling_max_3",  "rolling_max_7",
    "ema_3", "ema_7",
    "temperature_celsius", "weather_encoded",
    "rain_flag", "heat_flag",
    "rain_evening", "weekend_evening", "holiday_evening",
    "public_holiday_flag", "local_event_flag", "stockout_flag",
    "store_encoded", "sku_encoded",
]


# ══════════════════════════════════════════════════════════════════════════════
#  DATA  &  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="📦 Loading dataset …")
def load_data() -> pd.DataFrame:
    """Load CSV or auto-generate it on first run."""
    try:
        df = pd.read_csv("synthetic_data.csv")
    except FileNotFoundError:
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from generate_data import generate_qcommerce_data
        df = generate_qcommerce_data()
        df.to_csv("synthetic_data.csv", index=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"]      = df["timestamp"].dt.date
    return df


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Inline feature engineering (mirrors feature_engineering.py)."""
    df = df.copy().sort_values(["store_id", "sku_id", "timestamp"]).reset_index(drop=True)

    df["hour"]         = df["timestamp"].dt.hour
    df["day_of_week"]  = df["timestamp"].dt.dayofweek
    df["day_of_month"] = df["timestamp"].dt.day
    df["month"]        = df["timestamp"].dt.month
    df["week_of_year"] = df["timestamp"].dt.isocalendar().week.astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)

    grp = df.groupby(["store_id", "sku_id"])
    for lag in [1, 2, 3, 7]:
        df[f"lag_{lag}"] = grp["quantity_sold"].shift(lag)
    for w in [3, 7]:
        df[f"rolling_mean_{w}"] = grp["quantity_sold"].transform(
            lambda x: x.rolling(w, min_periods=1).mean())
        df[f"rolling_std_{w}"] = grp["quantity_sold"].transform(
            lambda x: x.rolling(w, min_periods=1).std().fillna(0))
        df[f"rolling_max_{w}"] = grp["quantity_sold"].transform(
            lambda x: x.rolling(w, min_periods=1).max())
    df["ema_3"] = grp["quantity_sold"].transform(lambda x: x.ewm(span=3).mean())
    df["ema_7"] = grp["quantity_sold"].transform(lambda x: x.ewm(span=7).mean())
    for lag in [1, 2, 3, 7]:
        df[f"lag_{lag}"] = df[f"lag_{lag}"].fillna(df["rolling_mean_3"])

    wmap = {"Clear": 0, "Cloudy": 1, "Rain": 2, "Extreme Heat": 3}
    df["weather_encoded"] = df["weather_condition"].map(wmap).fillna(0).astype(int)
    df["rain_flag"]  = (df["weather_condition"] == "Rain").astype(int)
    df["heat_flag"]  = (df["weather_condition"] == "Extreme Heat").astype(int)

    eve = (df["hour"] >= 18).astype(int)
    df["rain_evening"]    = df["rain_flag"]           * eve
    df["weekend_evening"] = df["is_weekend"]          * eve
    df["holiday_evening"] = df["public_holiday_flag"] * eve

    le_s = LabelEncoder(); le_k = LabelEncoder()
    df["store_encoded"] = le_s.fit_transform(df["store_id"])
    df["sku_encoded"]   = le_k.fit_transform(df["sku_id"])
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred) -> dict:
    yt = np.asarray(y_true, dtype=float)
    yp = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return {
        "RMSE":    round(float(np.sqrt(mean_squared_error(yt, yp))), 3),
        "MAE":     round(float(mean_absolute_error(yt, yp)), 3),
        "MAPE %":  round(float(np.mean(np.abs((yt - yp) / (yt + 1e-8))) * 100), 2),
        "R²":      round(float(r2_score(yt, yp)), 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="🏋️ Training models — please wait …")
def train_all_models(_df: pd.DataFrame) -> dict:
    """
    Train RF, XGBoost, LightGBM, and Prophet on an 80/20 time-series split.
    Returns a dict keyed by model name with predictions, metrics, and feature
    importances.
    """
    df = _engineer(_df.copy()).sort_values("timestamp").reset_index(drop=True)

    X      = df[FEATURE_COLS].fillna(0)
    y      = df["quantity_sold"]
    split  = int(len(df) * 0.80)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]
    ts_te      = df["timestamp"].iloc[split:].values

    results: dict = {}

    # ── Random Forest ─────────────────────────────────────────────────────────
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_leaf=2,
        random_state=42, n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    rf_pred = np.clip(rf.predict(X_te), 0, None)
    results["Random Forest"] = {
        "y_pred":  rf_pred,
        "y_true":  y_te.values,
        "ts":      ts_te,
        "metrics": compute_metrics(y_te, rf_pred),
        "fi":      pd.Series(rf.feature_importances_, index=FEATURE_COLS)
                     .sort_values(ascending=False).head(15),
    }

    # ── XGBoost ───────────────────────────────────────────────────────────────
    xgb = XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0
    )
    xgb.fit(X_tr, y_tr)
    xgb_pred = np.clip(xgb.predict(X_te), 0, None)
    results["XGBoost"] = {
        "y_pred":  xgb_pred,
        "y_true":  y_te.values,
        "ts":      ts_te,
        "metrics": compute_metrics(y_te, xgb_pred),
        "fi":      pd.Series(xgb.feature_importances_, index=FEATURE_COLS)
                     .sort_values(ascending=False).head(15),
    }

    # ── LightGBM ──────────────────────────────────────────────────────────────
    lgbm = LGBMRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        num_leaves=31, subsample=0.8, random_state=42, verbose=-1
    )
    lgbm.fit(X_tr, y_tr)
    lgbm_pred = np.clip(lgbm.predict(X_te), 0, None)
    results["LightGBM"] = {
        "y_pred":  lgbm_pred,
        "y_true":  y_te.values,
        "ts":      ts_te,
        "metrics": compute_metrics(y_te, lgbm_pred),
        "fi":      pd.Series(lgbm.feature_importances_, index=FEATURE_COLS)
                     .sort_values(ascending=False).head(15),
    }

    # ── Prophet (aggregated platform-wide demand) ─────────────────────────────
    try:
        from prophet import Prophet as _Prophet
        agg = df.groupby("timestamp")["quantity_sold"].sum().reset_index()
        agg.columns = ["ds", "y"]
        sp  = int(len(agg) * 0.80)
        m   = _Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            interval_width=0.90,
        )
        m.fit(agg.iloc[:sp])
        future = pd.DataFrame({"ds": agg["ds"].values})
        fc     = m.predict(future)
        p_pred = np.clip(fc["yhat"].values[sp:], 0, None)
        p_true = agg["y"].values[sp:]
        p_ts   = agg["ds"].values[sp:]
        results["Prophet"] = {
            "y_pred":  p_pred,
            "y_true":  p_true,
            "ts":      p_ts,
            "metrics": compute_metrics(p_true, p_pred),
            "fi":      None,
        }
    except Exception as exc:
        results["Prophet"] = {
            "y_pred":  np.zeros(len(y_te)),
            "y_true":  y_te.values,
            "ts":      ts_te,
            "metrics": {"RMSE": None, "MAE": None, "MAPE %": None, "R²": None},
            "fi":      None,
            "error":   str(exc),
        }

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — HOME & DATA OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

def page_home(df: pd.DataFrame) -> None:
    st.title("🛒 Q-Commerce Demand Forecasting Engine")
    st.markdown(
        "> **Dark Store Intelligence Platform** — Hyper-Local Demand Prediction "
        "for 10–15-Minute Delivery across Mumbai · Bangalore · Delhi"
    )
    st.markdown("---")

    # KPI strip
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("📦 Total Records",    f"{len(df):,}")
    c2.metric("🏪 Dark Stores",      df["store_id"].nunique())
    c3.metric("🛍️ SKUs Tracked",     df["sku_id"].nunique())
    c4.metric("⚠️ Stockout Rate",    f"{df['stockout_flag'].mean()*100:.1f}%")
    c5.metric("📊 Avg Qty / Period", f"{df['quantity_sold'].mean():.1f}")

    st.markdown("---")

    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.subheader("📋 Raw Transaction Ledger (first 25 rows)")
        st.dataframe(df.head(25), use_container_width=True, height=450)

    with col_r:
        st.subheader("📈 Demand Distribution")
        fig = px.histogram(
            df, x="quantity_sold", nbins=30, color="sku_id",
            opacity=0.75, barmode="overlay",
            title="Quantity Sold — Distribution by SKU",
            template=PLOTLY_TEMPLATE,
            labels={"quantity_sold": "Units Sold", "sku_id": "SKU"},
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🌦️ Weather Day-Distribution")
        wc = df.drop_duplicates(subset=["timestamp", "store_id"])["weather_condition"] \
               .value_counts().reset_index()
        wc.columns = ["weather", "count"]
        fig2 = px.pie(
            wc, values="count", names="weather",
            color="weather", color_discrete_map=WEATHER_COLORS,
            template=PLOTLY_TEMPLATE,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Descriptive Statistics")
    st.dataframe(
        df[["quantity_sold", "price_at_sale", "temperature_celsius",
            "stockout_flag", "public_holiday_flag", "local_event_flag"]].describe().round(3),
        use_container_width=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — DESCRIPTIVE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def page_descriptive(df: pd.DataFrame) -> None:
    st.title("📊 Descriptive Analysis")
    st.markdown(
        "Exploring **temporal demand patterns** and **external factor distributions** "
        "across dark stores and SKUs."
    )
    st.markdown("---")

    # ── Daily demand trend ────────────────────────────────────────────────────
    daily = (
        df.groupby(["date", "store_id"])["quantity_sold"]
          .sum().reset_index()
    )
    daily["date"] = pd.to_datetime(daily["date"])

    st.subheader("📅 Daily Demand Trend by Store")
    fig = px.line(
        daily, x="date", y="quantity_sold", color="store_id",
        markers=True, template=PLOTLY_TEMPLATE,
        title="Total Daily Units Sold — All SKUs per Store",
        labels={"quantity_sold": "Total Units Sold", "date": "Date", "store_id": "Store"},
    )
    fig.update_layout(legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Hourly & DoW profiles ─────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("⏰ Hourly Demand Profile")
        hourly = df.groupby(["hour_of_day", "sku_id"])["quantity_sold"].mean().reset_index()
        fig = px.bar(
            hourly, x="hour_of_day", y="quantity_sold", color="sku_id",
            barmode="group", template=PLOTLY_TEMPLATE,
            title="Average Demand by Hour of Day",
            labels={"quantity_sold": "Avg Units Sold", "hour_of_day": "Hour of Day"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📅 Day-of-Week Demand Profile")
        dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        weekly = df.groupby(["day_of_week", "sku_id"])["quantity_sold"].mean().reset_index()
        weekly["day_name"] = weekly["day_of_week"].map(dict(enumerate(dow_labels)))
        fig = px.bar(
            weekly, x="day_name", y="quantity_sold", color="sku_id",
            barmode="group", template=PLOTLY_TEMPLATE,
            title="Average Demand by Day of Week",
            labels={"quantity_sold": "Avg Units Sold", "day_name": "Day"},
            category_orders={"day_name": dow_labels},
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Demand heatmap ────────────────────────────────────────────────────────
    st.subheader("🔥 Demand Heatmap — Hour × Day of Week")
    pivot = df.pivot_table(
        index="hour_of_day", columns="day_of_week",
        values="quantity_sold", aggfunc="mean"
    )
    pivot.columns = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig = px.imshow(
        pivot, aspect="auto",
        color_continuous_scale="Viridis", template=PLOTLY_TEMPLATE,
        title="Average Units Sold per (Hour, Day-of-Week) Cell",
        labels={"x": "Day of Week", "y": "Hour of Day", "color": "Avg Demand"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── SKU & store breakdown ─────────────────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("🛍️ Total Sales by SKU")
        sku_tot = df.groupby("sku_id")["quantity_sold"].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(
            sku_tot, x="sku_id", y="quantity_sold", color="sku_id",
            template=PLOTLY_TEMPLATE, title="Total Units Sold by SKU",
            labels={"quantity_sold": "Total Units", "sku_id": "SKU"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("🏪 Store Performance — Mean vs Std Dev")
        ss = df.groupby("store_id")["quantity_sold"].agg(["mean", "std"]).reset_index()
        ss.columns = ["store_id", "Avg Demand", "Std Dev"]
        fig = px.bar(
            ss.melt(id_vars="store_id"),
            x="store_id", y="value", color="variable", barmode="group",
            template=PLOTLY_TEMPLATE,
            title="Store: Mean Demand vs Volatility",
            labels={"value": "Units", "variable": "Metric", "store_id": "Store"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Rolling avg trend ─────────────────────────────────────────────────────
    st.subheader("📈 7-Period Rolling Average Demand")
    df_s = df.sort_values("timestamp").copy()
    df_s["rolling_7"] = (
        df_s.groupby(["store_id", "sku_id"])["quantity_sold"]
            .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )
    focus = df_s["store_id"].unique()[0]
    fig = px.line(
        df_s[df_s["store_id"] == focus],
        x="timestamp", y="rolling_7", color="sku_id",
        template=PLOTLY_TEMPLATE,
        title=f"7-Period Rolling Average — {focus}",
        labels={"rolling_7": "Rolling Avg (units)", "timestamp": "Timestamp"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Revenue snapshot ──────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("💰 Revenue Snapshot")
    df["revenue"] = df["quantity_sold"] * df["price_at_sale"]
    rev_sku = df.groupby("sku_id")["revenue"].sum().sort_values(ascending=False).reset_index()
    fig = px.bar(
        rev_sku, x="sku_id", y="revenue", color="sku_id",
        template=PLOTLY_TEMPLATE, title="Estimated Revenue by SKU",
        labels={"revenue": "Revenue (₹)", "sku_id": "SKU"},
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — DIAGNOSTIC ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def page_diagnostic(df: pd.DataFrame) -> None:
    st.title("🔍 Diagnostic Analysis")
    st.markdown(
        "Deep-dive into **demand triggers**: identifying the weather events, "
        "time windows, and calendar flags that drive volatile spikes and drops."
    )
    st.markdown("---")

    # ── Weather impact ────────────────────────────────────────────────────────
    st.subheader("🌦️ Weather Impact on Demand")

    col1, col2 = st.columns(2)
    with col1:
        wa = df.groupby(["weather_condition", "sku_id"])["quantity_sold"].mean().reset_index()
        fig = px.bar(
            wa, x="weather_condition", y="quantity_sold", color="sku_id",
            barmode="group", template=PLOTLY_TEMPLATE,
            title="Avg Demand by Weather Condition × SKU",
            labels={"quantity_sold": "Avg Units Sold", "weather_condition": "Weather"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.violin(
            df, x="weather_condition", y="quantity_sold",
            color="weather_condition", color_discrete_map=WEATHER_COLORS,
            box=True, points="all", template=PLOTLY_TEMPLATE,
            title="Demand Distribution by Weather (Violin + Box)",
            labels={"quantity_sold": "Units Sold", "weather_condition": "Weather"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # % deviation from mean
    st.subheader("📊 Weather Demand Deviation from Overall Mean (%)")
    overall_mean = df["quantity_sold"].mean()
    wd = df.groupby("weather_condition")["quantity_sold"].mean()
    wdev = ((wd - overall_mean) / overall_mean * 100).reset_index()
    wdev.columns = ["weather", "pct_dev"]
    wdev["direction"] = wdev["pct_dev"].apply(lambda x: "Above Avg" if x >= 0 else "Below Avg")

    fig = px.bar(
        wdev, x="weather", y="pct_dev", color="direction",
        color_discrete_map={"Above Avg": "#4CAF50", "Below Avg": "#F44336"},
        template=PLOTLY_TEMPLATE,
        title="% Deviation from Overall Mean Demand by Weather Condition",
        labels={"pct_dev": "% Deviation", "weather": "Weather Condition"},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.6)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Holiday & event impact ────────────────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("🎉 Public Holiday Impact")
        hol = df.groupby(["public_holiday_flag", "sku_id"])["quantity_sold"].mean().reset_index()
        hol["Type"] = hol["public_holiday_flag"].map({0: "Normal Day", 1: "🎉 Holiday"})
        fig = px.bar(
            hol, x="Type", y="quantity_sold", color="sku_id",
            barmode="group", template=PLOTLY_TEMPLATE,
            title="Holiday vs Normal Day — Avg Demand",
            labels={"quantity_sold": "Avg Units Sold"},
        )
        st.plotly_chart(fig, use_container_width=True)

        hl = df.groupby("public_holiday_flag")["quantity_sold"].mean()
        if 1 in hl.index and 0 in hl.index:
            lift = (hl[1] - hl[0]) / hl[0] * 100
            st.metric("📈 Holiday Demand Lift", f"+{lift:.1f}%", "vs Normal Days")

    with col4:
        st.subheader("🎪 Local Event Impact")
        ev = df.groupby(["local_event_flag", "sku_id"])["quantity_sold"].mean().reset_index()
        ev["Type"] = ev["local_event_flag"].map({0: "No Event", 1: "🎪 Event Day"})
        fig = px.bar(
            ev, x="Type", y="quantity_sold", color="sku_id",
            barmode="group", template=PLOTLY_TEMPLATE,
            title="Event Day vs Normal Day — Avg Demand",
            labels={"quantity_sold": "Avg Units Sold"},
        )
        st.plotly_chart(fig, use_container_width=True)

        el = df.groupby("local_event_flag")["quantity_sold"].mean()
        if 1 in el.index and 0 in el.index:
            elift = (el[1] - el[0]) / el[0] * 100
            st.metric("📈 Event Demand Lift", f"+{elift:.1f}%", "vs Normal Days")

    st.markdown("---")

    # ── Weekend analysis ──────────────────────────────────────────────────────
    st.subheader("📅 Weekend vs Weekday Analysis")
    col5, col6 = st.columns(2)

    with col5:
        wk = df.groupby(["is_weekend", "sku_id"])["quantity_sold"].mean().reset_index()
        wk["Type"] = wk["is_weekend"].map({0: "Weekday", 1: "🎉 Weekend"})
        fig = px.bar(
            wk, x="Type", y="quantity_sold", color="sku_id",
            barmode="group", template=PLOTLY_TEMPLATE,
            title="Weekend vs Weekday — Avg Demand per SKU",
            labels={"quantity_sold": "Avg Units Sold"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        so_wk = df.groupby("is_weekend")["stockout_flag"].mean().reset_index()
        so_wk["Type"] = so_wk["is_weekend"].map({0: "Weekday", 1: "Weekend"})
        fig = px.bar(
            so_wk, x="Type", y="stockout_flag", color="Type",
            color_discrete_map={"Weekday": "#2196F3", "Weekend": "#FF5722"},
            template=PLOTLY_TEMPLATE,
            title="Stockout Rate: Weekend vs Weekday",
            labels={"stockout_flag": "Stockout Rate (fraction)"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Hour × Weather interaction heatmap ────────────────────────────────────
    st.markdown("---")
    st.subheader("🌡️ Time-of-Day × Weather Interaction Heatmap")
    inter = df.pivot_table(
        index="hour_of_day", columns="weather_condition",
        values="quantity_sold", aggfunc="mean"
    )
    fig = px.imshow(
        inter, aspect="auto",
        color_continuous_scale="RdYlGn", template=PLOTLY_TEMPLATE,
        title="Avg Demand: Hour of Day × Weather Condition",
        labels={"color": "Avg Units Sold"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Volatility ────────────────────────────────────────────────────────────
    st.subheader("📉 Demand Volatility — Coefficient of Variation (%)")
    cv = (
        df.groupby(["store_id", "sku_id"])["quantity_sold"]
          .agg(lambda x: x.std() / x.mean() * 100 if x.mean() > 0 else 0)
          .reset_index()
    )
    cv.columns = ["store_id", "sku_id", "CoV (%)"]
    fig = px.bar(
        cv, x="sku_id", y="CoV (%)", color="store_id", barmode="group",
        template=PLOTLY_TEMPLATE,
        title="Coefficient of Variation by SKU & Store — Higher = More Volatile",
        labels={"sku_id": "SKU", "store_id": "Store"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Stockout rates ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⚠️ Stockout Frequency by Condition")
    col7, col8 = st.columns(2)

    with col7:
        so_w = df.groupby("weather_condition")["stockout_flag"].mean().reset_index()
        fig = px.bar(
            so_w, x="weather_condition", y="stockout_flag",
            color="weather_condition", color_discrete_map=WEATHER_COLORS,
            template=PLOTLY_TEMPLATE,
            title="Stockout Rate by Weather Condition",
            labels={"stockout_flag": "Stockout Rate", "weather_condition": "Weather"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col8:
        so_s = df.groupby("sku_id")["stockout_flag"].mean().reset_index()
        fig = px.bar(
            so_s, x="sku_id", y="stockout_flag", color="sku_id",
            template=PLOTLY_TEMPLATE, title="Stockout Rate by SKU",
            labels={"stockout_flag": "Stockout Rate", "sku_id": "SKU"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — PREDICTIVE MODELING
# ══════════════════════════════════════════════════════════════════════════════

def page_modeling(df: pd.DataFrame) -> None:
    st.title("🤖 Predictive Modeling")
    st.markdown(
        "Supervised ML models trained on engineered temporal & exogenous features "
        "to forecast hyper-local hourly demand.  "
        "Train/test split: **80 % / 20 %** (time-ordered)."
    )
    st.markdown("---")

    # Feature engineering explainer
    with st.expander("🔧 Feature Engineering Reference", expanded=False):
        st.markdown("""
| **Group**       | **Features**                                              | **Rationale**                                    |
|-----------------|-----------------------------------------------------------|--------------------------------------------------|
| Temporal        | hour, day_of_week, day_of_month, month, week_of_year     | Capture intra-day & calendar seasonality         |
| Cyclical        | hour_sin/cos, dow_sin/cos                                | Circular encoding prevents jump artefacts        |
| Lag             | lag_1, lag_2, lag_3, lag_7                               | Recent demand is the best predictor of next hour |
| Rolling window  | mean / std / max over 3 & 7 periods                      | Smooth trend & volatility signal                 |
| Exponential MA  | ema_3, ema_7                                             | Heavier weight on recent observations            |
| Exogenous       | temperature, weather_encoded, rain_flag, heat_flag       | External demand drivers                          |
| Interaction     | rain_evening, weekend_evening, holiday_evening           | Non-linear compound effects                      |
| Entity          | store_encoded, sku_encoded                               | Dark-store & SKU identity                        |
        """)

    # Training button
    st.subheader("🏋️ Train Models")
    if st.button("🚀  Train All Four Models", type="primary", use_container_width=True):
        results = train_all_models(df)
        st.session_state["model_results"] = results
        st.success("✅  Training complete — scroll down to explore results.")

    if "model_results" not in st.session_state:
        st.info("👆 Click **Train All Four Models** to begin training. "
                "Results, metrics, and charts will appear here.")
        return

    results: dict = st.session_state["model_results"]

    # ── Metrics table ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 Evaluation Metrics — Test Set (20 %)")

    metrics_rows = []
    for m_name, res in results.items():
        row = {"Model": m_name}
        row.update(res["metrics"])
        metrics_rows.append(row)
    mdf = pd.DataFrame(metrics_rows)

    st.dataframe(
        mdf.style.format(
            {"RMSE": "{:.3f}", "MAE": "{:.3f}", "MAPE %": "{:.2f}", "R²": "{:.4f}"}
        ).highlight_min(
            subset=["RMSE", "MAE", "MAPE %"], color="#c8e6c9"
        ).highlight_max(
            subset=["R²"], color="#c8e6c9"
        ),
        use_container_width=True, hide_index=True,
    )

    # ── Metric bar charts ─────────────────────────────────────────────────────
    st.subheader("📈 Metrics Comparison Charts")
    c1, c2 = st.columns(2)
    for metric, col, lower_better in [
        ("RMSE",   c1, True),  ("MAE",    c2, True),
    ]:
        with col:
            mdf_plot = mdf.dropna(subset=[metric])
            fig = px.bar(
                mdf_plot, x="Model", y=metric, color="Model",
                color_discrete_map={k: v for k, v in PALETTE.items() if k in mdf_plot["Model"].values},
                title=f"{metric} — {'Lower = Better' if lower_better else 'Higher = Better'}",
                template=PLOTLY_TEMPLATE,
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    for metric, col, lower_better in [
        ("MAPE %", c3, True),  ("R²",     c4, False),
    ]:
        with col:
            mdf_plot = mdf.dropna(subset=[metric])
            fig = px.bar(
                mdf_plot, x="Model", y=metric, color="Model",
                color_discrete_map={k: v for k, v in PALETTE.items() if k in mdf_plot["Model"].values},
                title=f"{metric} — {'Lower = Better' if lower_better else 'Higher = Better'}",
                template=PLOTLY_TEMPLATE,
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # ── Actual vs Predicted ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📉 Actual vs Predicted Demand Curves")
    tabs = st.tabs(list(results.keys()))

    for tab, (m_name, res) in zip(tabs, results.items()):
        with tab:
            if "error" in res:
                st.error(f"Prophet error: {res['error']}")
                continue

            n = min(len(res["y_true"]), 60)
            x_idx = list(range(n))

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_idx, y=res["y_true"][:n].tolist(),
                mode="lines+markers", name="Actual",
                line=dict(color=PALETTE["Actual"], width=2),
                marker=dict(size=5),
            ))
            fig.add_trace(go.Scatter(
                x=x_idx, y=res["y_pred"][:n].tolist(),
                mode="lines+markers", name="Predicted",
                line=dict(color=PALETTE.get(m_name, "#888"), width=2, dash="dash"),
                marker=dict(size=5),
            ))
            fig.update_layout(
                title=f"{m_name} — Actual vs Predicted (Test Set, first {n} samples)",
                xaxis_title="Test Sample Index",
                yaxis_title="Quantity Sold",
                template=PLOTLY_TEMPLATE,
                legend=dict(orientation="h", y=1.12),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Residuals
            resid = res["y_true"] - res["y_pred"]
            fig_r = px.histogram(
                resid, nbins=20,
                title=f"{m_name} — Residuals Distribution",
                template=PLOTLY_TEMPLATE,
                color_discrete_sequence=[PALETTE.get(m_name, "#888")],
                labels={"value": "Residual"},
            )
            fig_r.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.7)
            st.plotly_chart(fig_r, use_container_width=True)

    # ── Feature importance ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🎯 Feature Importance — Tree-Based Models")

    tree_res = {k: v for k, v in results.items() if v.get("fi") is not None}
    fi_tabs  = st.tabs(list(tree_res.keys()))

    for tab, (m_name, res) in zip(fi_tabs, tree_res.items()):
        with tab:
            fi = res["fi"].reset_index()
            fi.columns = ["Feature", "Importance"]
            fig = px.bar(
                fi, x="Importance", y="Feature", orientation="h",
                color="Importance", color_continuous_scale="Blues",
                template=PLOTLY_TEMPLATE,
                title=f"{m_name} — Top 15 Feature Importances",
                labels={"Importance": "Importance Score", "Feature": "Feature"},
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — BUSINESS INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════

def page_insights(df: pd.DataFrame) -> None:
    st.title("💡 Business Insights & Inventory Recommendations")
    st.markdown(
        "Translating model predictions into **actionable inventory decisions** — "
        "dynamic safety stock, reorder points, and flash discount triggers."
    )
    st.markdown("---")

    # ── Safety Stock Calculator ───────────────────────────────────────────────
    st.subheader("🛡️ Dynamic Safety Stock Calculator")

    Z_MAP = {"90 %": 1.28, "95 %": 1.65, "99 %": 2.33}

    col_a, col_b = st.columns(2)
    with col_a:
        svc_lvl = st.selectbox("Target Service Level", ["95 %", "90 %", "99 %"])
    with col_b:
        lt_hrs = st.slider("Replenishment Lead Time (hours)", 1, 8, 2)

    Z = Z_MAP[svc_lvl]

    SHELF_LIFE_DAYS = {
        "SKU001_Milk_1L":   5,
        "SKU002_Eggs_12pk": 14,
        "SKU003_Bread_500g": 3,
    }

    ss = (
        df.groupby(["store_id", "sku_id"])["quantity_sold"]
          .agg(["mean", "std"])
          .reset_index()
    )
    ss.columns       = ["store_id", "sku_id", "avg_demand", "std_demand"]
    ss["std_demand"] = ss["std_demand"].fillna(0)
    ss["safety_stock"]   = (Z * ss["std_demand"] * np.sqrt(lt_hrs)).round(1)
    ss["reorder_point"]  = (ss["avg_demand"] * lt_hrs + ss["safety_stock"]).round(1)
    ss["shelf_life_days"]= ss["sku_id"].map(SHELF_LIFE_DAYS)
    ss["spoilage_risk"]  = (ss["safety_stock"] / (ss["shelf_life_days"] * ss["avg_demand"] * 3 + 1e-8)).round(3)
    ss["risk_level"]     = ss["spoilage_risk"].apply(
        lambda x: "🔴 High" if x > 0.30 else ("🟡 Medium" if x > 0.15 else "🟢 Low")
    )
    ss["avg_demand"]  = ss["avg_demand"].round(2)
    ss["std_demand"]  = ss["std_demand"].round(2)

    st.dataframe(ss, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            ss, x="sku_id", y="safety_stock", color="store_id", barmode="group",
            template=PLOTLY_TEMPLATE,
            title=f"Safety Stock by SKU & Store  (SL={svc_lvl}, LT={lt_hrs}h)",
            labels={"safety_stock": "Safety Stock (units)", "sku_id": "SKU", "store_id": "Store"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(
            ss, x="sku_id", y="reorder_point", color="store_id", barmode="group",
            template=PLOTLY_TEMPLATE,
            title="Reorder Point by SKU & Store",
            labels={"reorder_point": "Reorder Point (units)", "sku_id": "SKU", "store_id": "Store"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Flash Discount Triggers ───────────────────────────────────────────────
    st.subheader("💸 Flash Discount Trigger Matrix")
    st.markdown(
        "Items whose simulated on-hand stock covers more than **60 % of shelf life** "
        "should be discounted to reduce spoilage losses."
    )

    np.random.seed(99)
    ss["current_stock"]   = (ss["avg_demand"] * 6 * np.random.uniform(0.7, 1.8, len(ss))).round(0)
    ss["daily_demand_est"]= (ss["avg_demand"] * 3).round(2)
    ss["days_cover"]      = (ss["current_stock"] / (ss["daily_demand_est"] + 1e-8)).round(1)
    ss["cover_pct_sl"]    = (ss["days_cover"] / ss["shelf_life_days"] * 100).round(1)

    def discount_rec(pct: float) -> str:
        if pct > 80:  return "🔴 25 % Flash Discount — Urgent"
        if pct > 60:  return "🟡 15 % Flash Discount — Recommended"
        if pct > 40:  return "🟢 No Discount — Monitor"
        return            "✅ Optimal Stock Level"

    ss["discount_action"] = ss["cover_pct_sl"].apply(discount_rec)

    st.dataframe(
        ss[["store_id", "sku_id", "current_stock", "days_cover",
            "shelf_life_days", "cover_pct_sl", "discount_action"]],
        use_container_width=True, hide_index=True,
    )

    st.markdown("---")

    # ── Key Business Findings ─────────────────────────────────────────────────
    st.subheader("🎯 Key Actionable Findings")

    rain_lft  = df[df["weather_condition"] == "Rain"]["quantity_sold"].mean() / df["quantity_sold"].mean() - 1
    wknd_lft  = df[df["is_weekend"] == 1]["quantity_sold"].mean() / (df[df["is_weekend"] == 0]["quantity_sold"].mean() + 1e-8) - 1
    hol_mean  = df[df["public_holiday_flag"] == 1]["quantity_sold"].mean()
    norm_mean = df[df["public_holiday_flag"] == 0]["quantity_sold"].mean()
    hol_lft   = (hol_mean - norm_mean) / (norm_mean + 1e-8) - 0 if df["public_holiday_flag"].sum() > 0 else 0
    so_rate   = df["stockout_flag"].mean()

    col_f, col_r = st.columns(2)
    with col_f:
        st.markdown(f"""
#### 📌 Quantified Demand Triggers

| Trigger | Measured Impact |
|---|---|
| 🌧️ **Rain Weather** | **+{rain_lft*100:.0f}%** avg demand lift |
| 🎉 **Weekends (Sat–Sun)** | **+{wknd_lft*100:.0f}%** vs weekdays |
| 🏖️ **Public Holidays** | **+{(hol_lft+1)*100-100:.0f}%** demand spike |
| ⏰ **Evening (18:00)** | **Highest demand slot** — pre-stock mandatory |
| ⚠️ **Overall Stockout Rate** | **{so_rate*100:.1f}%** of all transactions |
| 🌡️ **Extreme Heat** | **−10–12 %** demand suppression |
        """)

    with col_r:
        st.markdown(f"""
#### 📋 Recommended Actions

1. **Weather-triggered pre-stocking** — integrate live weather API; pre-load +35% Milk & Bread before Rain forecast
2. **Friday 4 PM replenishment run** — increase weekend safety stock by {wknd_lft*100:.0f}% on Friday afternoons
3. **Holiday buffer** — load 24h early with +65% inventory on all SKUs
4. **Evening peak alert (17:30)** — real-time stock check; auto-trigger replenishment if stock < Reorder Point
5. **LightGBM auto-replenishment** — pipe model predictions into WMS to auto-create POs
6. **Bread flash discount** (3-day shelf life) — if stock cover > 2 days, trigger 15–25% discount
7. **Reduce heat-period overstock** — lower Milk orders by 15% when Extreme Heat is forecast
        """)

    st.markdown("---")

    # ── Cost-Impact Snapshot ──────────────────────────────────────────────────
    st.subheader("💰 Estimated Financial Impact of ML-Driven Optimization")

    aov         = df["price_at_sale"].mean() if "price_at_sale" in df.columns else 55
    total_so    = int(df["stockout_flag"].sum())
    rev_lost    = total_so * aov
    spoilage    = df["quantity_sold"].sum() * 0.05 * aov   # assume 5% wastage rate
    ml_savings  = rev_lost * 0.60 + spoilage * 0.40        # 60% SO reduction + 40% wastage cut

    c1, c2, c3 = st.columns(3)
    c1.metric("⚠️ Total Stockout Events",         f"{total_so:,}")
    c1.metric("💸 Estimated Revenue Lost",         f"₹{rev_lost:,.0f}")
    c2.metric("📦 Avg Transaction Value",          f"₹{aov:.0f}")
    c2.metric("🗑️ Estimated Spoilage Cost",        f"₹{spoilage:,.0f}")
    c3.metric("💡 Projected ML-Driven Savings",
              f"₹{ml_savings:,.0f}",
              "60% fewer stockouts + 40% less spoilage")

    # ROI bar
    fig = px.bar(
        pd.DataFrame({
            "Category":  ["Revenue Lost (Stockouts)", "Spoilage Cost", "ML Projected Savings"],
            "Amount (₹)": [rev_lost, spoilage, ml_savings],
            "Type":       ["Cost", "Cost", "Saving"],
        }),
        x="Category", y="Amount (₹)", color="Type",
        color_discrete_map={"Cost": "#EF5350", "Saving": "#66BB6A"},
        template=PLOTLY_TEMPLATE,
        title="Financial Impact Overview — Estimated Values",
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR  &  MAIN ROUTER
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    df = load_data()

    with st.sidebar:
        st.markdown("## 🛒 Q-Commerce\nDemand Forecasting")
        st.markdown("---")

        page = st.radio(
            "📌 Navigate",
            [
                "🏠 Home & Data Overview",
                "📊 Descriptive Analysis",
                "🔍 Diagnostic Analysis",
                "🤖 Predictive Modeling",
                "💡 Business Insights",
            ],
        )

        st.markdown("---")
        st.markdown("**🔽 Global Filters**")
        stores = ["All Stores"] + sorted(df["store_id"].unique().tolist())
        skus   = ["All SKUs"]   + sorted(df["sku_id"].unique().tolist())
        sel_store = st.selectbox("🏪 Dark Store", stores)
        sel_sku   = st.selectbox("🛍️ SKU",        skus)

        st.markdown("---")
        st.caption(
            f"📅 **Period:** {df['timestamp'].min():%d %b %Y} → {df['timestamp'].max():%d %b %Y}\n\n"
            f"📦 **Records:** {len(df):,}"
        )

    # Apply sidebar filters
    fdf = df.copy()
    if sel_store != "All Stores":
        fdf = fdf[fdf["store_id"] == sel_store]
    if sel_sku != "All SKUs":
        fdf = fdf[fdf["sku_id"] == sel_sku]

    # Route
    if page == "🏠 Home & Data Overview":
        page_home(fdf)
    elif page == "📊 Descriptive Analysis":
        page_descriptive(fdf)
    elif page == "🔍 Diagnostic Analysis":
        page_diagnostic(fdf)
    elif page == "🤖 Predictive Modeling":
        page_modeling(df)          # always use full df for modeling
    elif page == "💡 Business Insights":
        page_insights(fdf)


if __name__ == "__main__":
    main()

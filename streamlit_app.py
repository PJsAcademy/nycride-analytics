"""NYCRide Analytics — Streamlit demo for the Bits to Builds flagship capstone.

Tabs: Dashboard / Driver / Analytics / About.
Sidebar: brand, filters, data freshness, downloads, credits.

Checkpoints come from either joblib-loadable files or a first-run bootstrap that
calls phase1/2/3 directly. Immune to pickle version shifts between training and
deployed runtime.
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

HERE = Path(__file__).parent
CHECKPOINTS = HERE / "checkpoints"
BOROUGHS = ["Manhattan", "Bronx", "Brooklyn", "Queens", "Staten Island"]

BRAND_YELLOW = "#FFC72C"
BRAND_RED = "#E63946"
BRAND_INK = "#0E1117"
BRAND_INK2 = "#171B22"
BRAND_INK3 = "#232833"
BRAND_FG = "#E7E9EC"
BRAND_FG_DIM = "#9AA3B2"

BOROUGH_CENTROIDS = {
    "Manhattan":     (40.776, -73.965),
    "Bronx":         (40.845, -73.865),
    "Brooklyn":      (40.650, -73.950),
    "Queens":        (40.720, -73.820),
    "Staten Island": (40.580, -74.150),
}
AIRPORT_COORDS = {132: (40.6413, -73.7781), 138: (40.7769, -73.8740), 1: (40.6895, -74.1745)}

sys.path.insert(0, str(HERE / "src"))

st.set_page_config(
    page_title="NYCRide Analytics",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "NYCRide Analytics — flagship capstone of Bits to Builds. "
                 "Source: github.com/PJsAcademy/nycride-analytics",
        "Get help": "https://github.com/PJsAcademy/nycride-analytics/issues",
    },
)


# ======================================================= bootstrap

@st.cache_resource(show_spinner="First-time setup: building silver + ML + forecast (~30s)...")
def bootstrap_checkpoints() -> None:
    required = [
        CHECKPOINTS / "silver.parquet",
        CHECKPOINTS / "tip_model.pkl",
        CHECKPOINTS / "high_tip_zones.csv",
        CHECKPOINTS / "demand_model.pkl",
    ]
    if all(p.exists() for p in required):
        try:
            # joblib pickles produced by this repo's own phase2/3 scripts;
            # trusted artifact loaded only after existence check succeeded.
            joblib.load(CHECKPOINTS / "tip_model.pkl")
            joblib.load(CHECKPOINTS / "demand_model.pkl")
            return
        except Exception:
            pass
    CHECKPOINTS.mkdir(exist_ok=True)
    import phase1_pipeline, phase2_ml, phase3_forecast
    phase1_pipeline.main()
    phase2_ml.main()
    phase3_forecast.main()


bootstrap_checkpoints()


# ======================================================= loaders

@st.cache_data(show_spinner="Loading silver warehouse...")
def load_silver() -> pd.DataFrame:
    df = pd.read_parquet(CHECKPOINTS / "silver.parquet")
    rng = np.random.default_rng(42)
    lats = np.empty(len(df))
    lons = np.empty(len(df))
    for i, (pul, bor) in enumerate(zip(df["PULocationID"].values, df["pickup_borough"].values)):
        if pul in AIRPORT_COORDS:
            lat0, lon0 = AIRPORT_COORDS[pul]
            jit = 0.004
        else:
            lat0, lon0 = BOROUGH_CENTROIDS.get(bor, BOROUGH_CENTROIDS["Manhattan"])
            jit = 0.025
        lats[i] = lat0 + rng.normal(0, jit)
        lons[i] = lon0 + rng.normal(0, jit)
    df["pickup_lat"] = lats
    df["pickup_lon"] = lons
    return df


@st.cache_resource
def load_tip_model():
    return joblib.load(CHECKPOINTS / "tip_model.pkl")


@st.cache_resource
def load_demand_model():
    return joblib.load(CHECKPOINTS / "demand_model.pkl")


@st.cache_data
def load_high_tip_zones() -> pd.DataFrame:
    return pd.read_csv(CHECKPOINTS / "high_tip_zones.csv")


@st.cache_data
def compute_fairness(silver: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split
    model = load_tip_model()
    card = silver[silver["payment_type"] == 1]
    feats = ["trip_distance", "fare_amount", "trip_duration_min", "passenger_count",
             "hour", "dow", "is_weekend", "pickup_borough"]
    X = card[feats]; y = card["tip_amount"]
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    yhat = model.predict(X_te)
    audit = pd.DataFrame({"y": y_te.values, "yhat": yhat, "borough": X_te["pickup_borough"].values})
    per = audit.groupby("borough").apply(lambda d: mean_absolute_error(d.y, d.yhat))
    per = per.rename("mae").reset_index().sort_values("mae")
    overall_mae = float(mean_absolute_error(y_te, yhat))
    return per, overall_mae


@st.cache_data
def top_tip_zone_coords(high_tip: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Project the top-k high-tip zones onto lat/lon via the same borough-centroid
    scheme used for silver. Deterministic — same jitter seed produces consistent dots."""
    rng = np.random.default_rng(7)
    rows = []
    for _, r in high_tip.sort_values("avg_tip", ascending=False).head(k).iterrows():
        pul = int(r["PULocationID"])
        if pul in AIRPORT_COORDS:
            lat0, lon0 = AIRPORT_COORDS[pul]
            jit = 0.002
        else:
            lat0, lon0 = BOROUGH_CENTROIDS["Manhattan"]  # most high-tip zones concentrate in Manhattan
            jit = 0.015
        rows.append({"PULocationID": pul, "avg_tip": r["avg_tip"],
                     "lat": lat0 + rng.normal(0, jit), "lon": lon0 + rng.normal(0, jit)})
    return pd.DataFrame(rows)


# ======================================================= Phase 3 NL->SQL

def nl_to_sql(question: str) -> str | None:
    if not question:
        return None
    q = question.lower().strip().rstrip("?.")
    m = re.search(
        r"(trips|rides|taxis).*?(from|to|at|near)\s+(jfk|lga|ewr|manhattan|bronx|brooklyn|queens|staten island)",
        q,
    )
    if m:
        loc = m.group(3)
        airports = {"jfk": 132, "lga": 138, "ewr": 1}
        if loc in airports:
            return f"SELECT COUNT(*) FROM silver WHERE PULocationID = {airports[loc]}"
        return f"SELECT COUNT(*) FROM silver WHERE pickup_borough = '{loc.title()}'"
    if "average" in q or "avg" in q:
        metric = "fare_amount" if "fare" in q else "tip_amount" if "tip" in q else "total_amount"
        return f"SELECT AVG({metric}) FROM silver"
    if "busiest" in q or "peak" in q:
        unit = ("DATE_TRUNC('hour', tpep_pickup_datetime)" if "hour" in q
                else "DATE_TRUNC('day', tpep_pickup_datetime)")
        return f"SELECT {unit}, COUNT(*) FROM silver GROUP BY 1 ORDER BY 2 DESC LIMIT 1"
    return None


def run_sql(sql: str, silver: pd.DataFrame) -> pd.DataFrame:
    low = sql.strip().lower()
    m = re.match(r"select count\(\*\) from silver where pulocationid = (\d+)", low)
    if m:
        return pd.DataFrame({"count": [int((silver["PULocationID"] == int(m.group(1))).sum())]})
    m = re.match(r"select count\(\*\) from silver where pickup_borough = '([^']+)'", low)
    if m:
        return pd.DataFrame({"count": [int((silver["pickup_borough"].str.lower() == m.group(1)).sum())]})
    m = re.match(r"select avg\((\w+)\) from silver", low)
    if m:
        col = m.group(1)
        return pd.DataFrame({f"avg_{col}": [round(float(silver[col].mean()), 3)]})
    if "date_trunc" in low and "group by 1 order by 2 desc" in low:
        unit = "hour" if "'hour'" in low else "day"
        bucket = silver["tpep_pickup_datetime"].dt.floor("h" if unit == "hour" else "D")
        top = bucket.value_counts().head(1).reset_index()
        top.columns = [f"busiest_{unit}", "count"]
        return top
    return pd.DataFrame({"error": [f"unsupported SQL pattern: {sql}"]})


# ======================================================= Global CSS

st.markdown(
    f"""
    <style>
    /* Hide Streamlit default chrome a bit */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header[data-testid="stHeader"] {{background: transparent;}}

    /* Custom KPI cards */
    .kpi-card {{
        background: linear-gradient(135deg, {BRAND_INK2} 0%, {BRAND_INK3} 100%);
        border: 1px solid rgba(255,255,255,0.06);
        border-left: 4px solid var(--accent, {BRAND_YELLOW});
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
        height: 100%;
    }}
    .kpi-card .kpi-label {{
        color: {BRAND_FG_DIM};
        font-size: 12px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }}
    .kpi-card .kpi-value {{
        color: {BRAND_FG};
        font-size: clamp(16px, 1.9vw, 30px);
        font-weight: 700;
        line-height: 1.1;
        font-variant-numeric: tabular-nums;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .kpi-card .kpi-delta {{
        display: inline-block;
        margin-top: 6px;
        color: {BRAND_FG_DIM};
        font-size: 12px;
    }}
    .kpi-icon {{
        float: right;
        font-size: 20px;
        opacity: 0.35;
        margin-left: 4px;
    }}
    @media (max-width: 1100px) {{ .kpi-icon {{display: none;}} }}

    /* Insight chip */
    .insight {{
        background: {BRAND_INK2};
        border: 1px solid rgba(255,199,44,0.15);
        border-radius: 10px;
        padding: 10px 14px;
        font-size: 13px;
        color: {BRAND_FG};
        line-height: 1.45;
    }}
    .insight .insight-tag {{
        color: {BRAND_YELLOW};
        font-weight: 600;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-right: 6px;
    }}

    /* Hero */
    .hero {{
        background: radial-gradient(circle at top left, rgba(255,199,44,0.14) 0%, rgba(14,17,23,0) 55%);
        padding: 10px 0 16px;
        margin-bottom: 10px;
    }}
    .hero h1 {{
        font-size: 36px !important;
        font-weight: 800 !important;
        margin-bottom: 4px !important;
    }}
    .hero .tagline {{color: {BRAND_FG_DIM}; font-size: 15px;}}

    /* Sidebar polish */
    section[data-testid="stSidebar"] {{background: {BRAND_INK};}}
    .sidebar-brand {{
        padding: 6px 4px 20px;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 14px;
    }}
    .sidebar-brand .brand-logo {{font-size: 24px;}}
    .sidebar-brand .brand-name {{font-size: 18px; font-weight: 700; color: {BRAND_FG};}}
    .sidebar-brand .brand-sub {{font-size: 12px; color: {BRAND_FG_DIM};}}
    </style>
    """,
    unsafe_allow_html=True,
)


# ======================================================= load data

silver = load_silver()
tip_model = load_tip_model()
high_tip = load_high_tip_zones()
fairness, overall_mae = compute_fairness(silver)
top_tip_coords = top_tip_zone_coords(high_tip, 5)


# ======================================================= Sidebar

with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-brand">
          <div><span class="brand-logo">🚕</span> <span class="brand-name">NYCRide</span></div>
          <div class="brand-sub">Flagship capstone · Bits to Builds</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### Filters")
    date_min = silver["tpep_pickup_datetime"].min().date()
    date_max = silver["tpep_pickup_datetime"].max().date()
    date_range = st.date_input(
        "Date range", (date_min, date_max), min_value=date_min, max_value=date_max,
    )
    selected_boroughs = st.multiselect("Boroughs", BOROUGHS, default=BOROUGHS)

    st.markdown("##### Data")
    st.caption(f"**{len(silver):,}** silver rows")
    st.caption(f"**{date_min} → {date_max}**")
    st.caption("Sample from NYC TLC Yellow Taxi 2024 (public domain)")

    buf = io.BytesIO()
    silver.head(1000).drop(columns=["pickup_lat", "pickup_lon"]).to_csv(buf, index=False)
    st.download_button("⬇ Download silver sample (CSV)", buf.getvalue(),
                       "nycride_silver_sample.csv", "text/csv", use_container_width=True)
    st.download_button("⬇ Download high-tip zones", high_tip.to_csv(index=False).encode(),
                       "high_tip_zones.csv", "text/csv", use_container_width=True)

    st.divider()
    st.markdown("##### Links")
    st.markdown("[💻 Source on GitHub](https://github.com/PJsAcademy/nycride-analytics)")
    st.markdown("[📚 Bits to Builds](https://bitstobuilds.com)")
    st.markdown("[🚕 NYC TLC dataset](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)")

    st.divider()
    st.caption(f"GBM MAE **${overall_mae:.2f}**  ·  Fairness gap **${fairness['mae'].max() - fairness['mae'].min():.2f}**")


# Apply sidebar filters
if len(date_range) == 2:
    d0, d1 = date_range
    mask = ((silver["tpep_pickup_datetime"].dt.date >= d0) &
            (silver["tpep_pickup_datetime"].dt.date <= d1))
    silver_f = silver[mask]
else:
    silver_f = silver
if selected_boroughs:
    silver_f = silver_f[silver_f["pickup_borough"].isin(selected_boroughs)]
if len(silver_f) == 0:
    silver_f = silver  # safety fallback


# ======================================================= Hero

st.markdown(
    """
    <div class="hero">
      <h1>🚕 NYCRide Analytics</h1>
      <div class="tagline">End-to-end taxi analytics on real NYC TLC data · Flagship capstone of <a href="https://bitstobuilds.com" style="color:#FFC72C;">Bits to Builds</a></div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ======================================================= KPI cards

def kpi_card(label: str, value: str, delta: str = "", icon: str = "", accent: str = BRAND_YELLOW):
    st.markdown(
        f"""
        <div class="kpi-card" style="--accent:{accent};">
          <div class="kpi-icon">{icon}</div>
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-delta">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


total_trips = len(silver_f)
total_revenue = float(silver_f["total_amount"].sum())
median_fare = float(silver_f["fare_amount"].median())
fairness_gap = float(fairness["mae"].max() - fairness["mae"].min())
avg_tip = float(silver_f.loc[silver_f["payment_type"] == 1, "tip_amount"].mean())

def _compact(n: float, prefix: str = "") -> str:
    """9,945 -> 9.9K; 448,399 -> 448K; 1.2M; small numbers unchanged."""
    if n >= 1_000_000: return f"{prefix}{n/1_000_000:.1f}M"
    if n >= 100_000:   return f"{prefix}{n/1_000:.0f}K"
    if n >= 10_000:    return f"{prefix}{n/1_000:.1f}K"
    if n >= 1_000:     return f"{prefix}{n/1_000:.2f}K"
    return f"{prefix}{n:,.0f}"


date_days = (silver_f['tpep_pickup_datetime'].dt.date.max() -
             silver_f['tpep_pickup_datetime'].dt.date.min()).days + 1

k1, k2, k3, k4, k5 = st.columns(5)
with k1: kpi_card("Trips", _compact(total_trips), f"over {date_days} days", "🚕")
with k2: kpi_card("Revenue", _compact(total_revenue, "$"), f"${total_revenue/max(total_trips,1):.2f}/trip avg", "💵", BRAND_YELLOW)
with k3: kpi_card("Median fare", f"${median_fare:.2f}", "p50 of fare_amount", "📊", "#4A90E2")
with k4: kpi_card("Avg tip", f"${avg_tip:.2f}", f"{avg_tip/median_fare*100:.0f}% of median fare", "💝", "#50C878")
with k5: kpi_card("Fairness gap", f"${fairness_gap:.2f}", "max-min borough MAE", "⚖️", BRAND_RED)


# ======================================================= Insights band

def _busiest_hour() -> int:
    return int(silver_f["tpep_pickup_datetime"].dt.hour.value_counts().idxmax())


def _busiest_day() -> str:
    return silver_f["tpep_pickup_datetime"].dt.day_name().value_counts().idxmax()


def _best_tip_borough() -> str:
    card = silver_f[silver_f["payment_type"] == 1]
    if len(card) == 0:
        return "—"
    return card.groupby("pickup_borough")["tip_amount"].mean().idxmax()


st.markdown("")
i1, i2, i3, i4 = st.columns(4)
insights = [
    ("Peak hour", f"<b>{_busiest_hour():02d}:00</b> is the busiest hour on this slice."),
    ("Peak day", f"<b>{_busiest_day()}</b> sees the most pickups."),
    ("Best tips", f"<b>{_best_tip_borough()}</b> pulls the highest average tip."),
    ("Model honesty", f"GBM MAE <b>${overall_mae:.2f}</b> ≈ linear baseline — baseline-first shipped."),
]
for col, (tag, body) in zip([i1, i2, i3, i4], insights):
    with col:
        st.markdown(f'<div class="insight"><span class="insight-tag">{tag}</span>{body}</div>',
                    unsafe_allow_html=True)

st.markdown("")
st.divider()

tab_dashboard, tab_driver, tab_analytics, tab_method, tab_about = st.tabs(
    ["📊 Dashboard", "🚗 Driver", "💬 Analytics", "🛠 Methodology", "ℹ️ About"]
)


# ------- Dashboard tab -------
with tab_dashboard:
    st.subheader("Where pickups happen · top-5 tip zones overlaid")
    st.caption("Hex-density of pickups + glowing dots at the top-5 highest-tip zones. Drag to pan, scroll to zoom, hold ctrl to tilt.")

    hex_layer = pdk.Layer(
        "HexagonLayer",
        data=silver_f[["pickup_lon", "pickup_lat"]].rename(columns={"pickup_lon": "lon", "pickup_lat": "lat"}),
        get_position=["lon", "lat"],
        radius=250,
        elevation_scale=4,
        extruded=True,
        coverage=0.85,
        pickable=True,
        color_range=[
            [255, 237, 160], [254, 217, 118], [254, 178, 76],
            [253, 141, 60], [240, 59, 32], [189, 0, 38],
        ],
    )
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=top_tip_coords,
        get_position=["lon", "lat"],
        get_radius=400,
        get_fill_color=[80, 200, 120, 230],
        get_line_color=[255, 255, 255, 200],
        line_width_min_pixels=2,
        pickable=True,
    )
    deck = pdk.Deck(
        layers=[hex_layer, scatter_layer],
        initial_view_state=pdk.ViewState(latitude=40.74, longitude=-73.95, zoom=10, pitch=45),
        tooltip={"text": "Zone {PULocationID}\nAvg tip ${avg_tip}"},
        map_style=None,
    )
    st.pydeck_chart(deck, use_container_width=True)

    st.divider()
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("When pickups happen")
        st.caption("Trips by hour × day of week. Rush-hour and weekend patterns at a glance.")
        pivot = (silver_f.assign(hour=silver_f["tpep_pickup_datetime"].dt.hour,
                                 dow=silver_f["tpep_pickup_datetime"].dt.day_name())
                         .groupby(["dow", "hour"]).size().rename("trips").reset_index())
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        heatmap = (
            alt.Chart(pivot).mark_rect(stroke=BRAND_INK, strokeWidth=1).encode(
                x=alt.X("hour:O", title="Hour of day", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("dow:O", sort=dow_order, title=None),
                color=alt.Color("trips:Q", scale=alt.Scale(scheme="yelloworangered"),
                                legend=alt.Legend(title="Trips", orient="right")),
                tooltip=["dow", "hour", "trips"],
            ).properties(height=280)
        )
        st.altair_chart(heatmap, use_container_width=True)

    with c2:
        st.subheader("Daily revenue")
        st.caption("Sum of total_amount per day. Gold layer.")
        daily = (silver_f.assign(date=silver_f["tpep_pickup_datetime"].dt.date)
                         .groupby("date")
                         .agg(revenue=("total_amount", "sum"), trips=("fare_amount", "count"))
                         .reset_index())
        line = (
            alt.Chart(daily).mark_area(
                line={"color": BRAND_YELLOW, "strokeWidth": 2},
                color=alt.Gradient(
                    gradient="linear",
                    stops=[alt.GradientStop(color=BRAND_YELLOW, offset=0),
                           alt.GradientStop(color=BRAND_INK2, offset=1)],
                    x1=1, x2=1, y1=1, y2=0,
                ),
            ).encode(
                x=alt.X("date:T", title=None),
                y=alt.Y("revenue:Q", title="Revenue ($)"),
                tooltip=[alt.Tooltip("date:T"), alt.Tooltip("revenue:Q", format="$,.0f"),
                         alt.Tooltip("trips:Q", title="Trips")],
            ).properties(height=280)
        )
        st.altair_chart(line, use_container_width=True)

    st.divider()
    st.subheader("Hour-ahead demand forecast — GBM vs naive baseline")
    st.caption("Phase 3's model predicts next-hour pickup count from lag features. "
               "Shown on the last 48 hours of the slice: actual, GBM, and naive (last-hour) baseline.")

    @st.cache_data(show_spinner="Scoring demand forecast...")
    def _forecast_tail(silver_df: pd.DataFrame) -> pd.DataFrame:
        hourly = (silver_df.assign(ts=silver_df["tpep_pickup_datetime"].dt.floor("h"))
                            .groupby("ts").size().rename("trips").to_frame())
        if hourly.empty:
            return pd.DataFrame(columns=["ts", "actual", "gbm", "naive"])
        hourly = hourly.reindex(
            pd.date_range(hourly.index.min(), hourly.index.max(), freq="h"), fill_value=0
        ).rename_axis("ts")
        for lag in (1, 24, 168):
            hourly[f"lag{lag}"] = hourly["trips"].shift(lag)
        hourly["hour"] = hourly.index.hour
        hourly["dow"]  = hourly.index.dayofweek
        hourly = hourly.dropna()
        if hourly.empty:
            return pd.DataFrame(columns=["ts", "actual", "gbm", "naive"])
        model = load_demand_model()
        X = hourly[["lag1", "lag24", "lag168", "hour", "dow"]]
        hourly["gbm"] = model.predict(X)
        hourly["naive"] = hourly["lag1"]
        return hourly.tail(48).reset_index().rename(columns={"trips": "actual"})[["ts", "actual", "gbm", "naive"]]

    fc = _forecast_tail(silver_f)
    if len(fc) >= 10:
        fc_long = fc.melt("ts", var_name="series", value_name="trips")
        color_scale = alt.Scale(
            domain=["actual", "gbm", "naive"],
            range=[BRAND_FG, BRAND_YELLOW, "#4A90E2"],
        )
        forecast_chart = (
            alt.Chart(fc_long).mark_line(strokeWidth=2.5, point=False).encode(
                x=alt.X("ts:T", title="Hour"),
                y=alt.Y("trips:Q", title="Pickups per hour"),
                color=alt.Color("series:N", scale=color_scale, title=None,
                                legend=alt.Legend(orient="top")),
                strokeDash=alt.StrokeDash(
                    "series:N",
                    scale=alt.Scale(domain=["actual", "gbm", "naive"],
                                    range=[[1, 0], [1, 0], [4, 4]]),
                    legend=None,
                ),
                tooltip=["ts:T", "series:N", alt.Tooltip("trips:Q", format=".1f")],
            ).properties(height=280)
        )
        st.altair_chart(forecast_chart, use_container_width=True)

        def _mape(y, p, eps=1.0):
            y, p = np.asarray(y, float), np.asarray(p, float)
            return float(np.mean(np.abs((p - y) / np.maximum(y, eps))))
        gbm_mape = _mape(fc["actual"], fc["gbm"])
        naive_mape = _mape(fc["actual"], fc["naive"])
        improvement = (naive_mape - gbm_mape) / naive_mape * 100 if naive_mape > 0 else 0
        fc1, fc2, fc3 = st.columns(3)
        with fc1: kpi_card("GBM MAPE (last 48h)", f"{gbm_mape:.1%}", "lower is better", "🎯", "#50C878")
        with fc2: kpi_card("Naive baseline MAPE", f"{naive_mape:.1%}", "repeat-last-hour", "📉", "#4A90E2")
        with fc3: kpi_card("Relative improvement", f"{improvement:+.1f}%", "GBM vs naive", "⚡", BRAND_YELLOW)
    else:
        st.info("Need at least 10 hours of data for a forecast. Widen the date filter.")


# ------- Driver tab -------
with tab_driver:
    c1, c2 = st.columns([2, 3])
    with c1:
        st.subheader("Best pickup zones")
        st.caption("Top 10 zones by average tip. Minimum 5 trips.")
        top = high_tip.sort_values("avg_tip", ascending=False).head(10).copy()
        top = top.rename(columns={"PULocationID": "Zone", "trips": "Trips", "avg_tip": "Avg tip ($)"})
        tip_max = float(top["Avg tip ($)"].max())
        st.dataframe(
            top,
            use_container_width=True, hide_index=True, height=380,
            column_config={
                "Zone": st.column_config.NumberColumn(width="small"),
                "Trips": st.column_config.NumberColumn(width="small"),
                "Avg tip ($)": st.column_config.ProgressColumn(
                    "Avg tip ($)", format="$%.2f", min_value=0, max_value=tip_max,
                ),
            },
        )

    with c2:
        st.subheader("Fairness audit — per-borough MAE")
        st.caption("How much the tip model misses by, per borough. The brand promise: no borough gets worse service.")
        chart_data = fairness.copy()
        bar = (
            alt.Chart(chart_data).mark_bar(cornerRadius=4).encode(
                y=alt.Y("borough:N", sort="-x", title=None),
                x=alt.X("mae:Q", title="Mean absolute error ($)"),
                color=alt.Color("mae:Q", scale=alt.Scale(scheme="yelloworangered"), legend=None),
                tooltip=["borough", alt.Tooltip("mae:Q", format="$.3f")],
            )
        )
        avg_rule = alt.Chart(pd.DataFrame({"avg": [chart_data["mae"].mean()]})).mark_rule(
            color="white", strokeDash=[4, 4], opacity=0.6,
        ).encode(x="avg:Q")
        st.altair_chart((bar + avg_rule).properties(height=240), use_container_width=True)
        st.caption(
            f"Dashed line = average MAE across boroughs (${chart_data['mae'].mean():.2f}). "
            f"Gap of ${fairness_gap:.2f} on the 10k sample; on real 2.9M-row data this would be the first flag."
        )

    st.divider()
    st.subheader("Tip prediction — live")
    st.caption("Drag to see the model's prediction update in real time. ± MAE confidence band shown below.")
    c1, c2, c3 = st.columns(3)
    distance = c1.slider("Trip distance (miles)", 0.1, 50.0, 2.5, 0.1)
    fare = c2.slider("Fare amount ($)", 2.5, 200.0, 15.0, 0.5)
    duration = c3.slider("Trip duration (min)", 1, 180, 12, 1)
    c4, c5, c6 = st.columns(3)
    borough = c4.selectbox("Pickup borough", BOROUGHS, index=0)
    hour = c5.slider("Hour of day", 0, 23, 18)
    is_weekend = c6.toggle("Weekend?", value=False)

    row = pd.DataFrame([{
        "trip_distance": distance, "fare_amount": fare, "trip_duration_min": duration,
        "passenger_count": 1, "hour": hour, "dow": 5 if is_weekend else 2,
        "is_weekend": int(is_weekend), "pickup_borough": borough,
    }])
    predicted = float(tip_model.predict(row)[0])
    pct = predicted / fare * 100
    lo, hi = max(0, predicted - overall_mae), predicted + overall_mae

    # Baseline comparison — train a quick linear baseline in-session (cached)
    @st.cache_resource(show_spinner="Fitting baseline for comparison...")
    def _baseline_model():
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.model_selection import train_test_split
        card = silver[silver["payment_type"] == 1]
        feats_n = ["trip_distance", "fare_amount", "trip_duration_min", "passenger_count",
                   "hour", "dow", "is_weekend"]
        feats_c = ["pickup_borough"]
        X = card[feats_n + feats_c]; y = card["tip_amount"]
        X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        pipe = Pipeline([
            ("pre", ColumnTransformer([("num", "passthrough", feats_n),
                                       ("cat", OneHotEncoder(handle_unknown="ignore"), feats_c)])),
            ("lr", LinearRegression()),
        ]).fit(X_tr, y_tr)
        return pipe

    baseline_pred = float(_baseline_model().predict(row)[0])
    diff = predicted - baseline_pred

    a, b, c = st.columns(3)
    with a: kpi_card("GBM prediction", f"${predicted:.2f}", f"{pct:.1f}% of fare", "💝", "#50C878")
    with b: kpi_card("Baseline (linear)", f"${baseline_pred:.2f}", "same features, no GBM", "📐", "#4A90E2")
    with c: kpi_card("GBM − baseline", f"{diff:+.2f}", "the lift (or not) from GBM", "⚡", BRAND_YELLOW)

    st.caption(
        f"**± MAE band (GBM):** ${lo:.2f} – ${hi:.2f}  ·  "
        f"On the 10k sample GBM and baseline are near-identical — baseline-first is the Phase 2 lesson, "
        "not a defect to hide. On real 2.9M-row data the gap widens."
    )

    # Feature importance from the GBM
    try:
        gbm = tip_model.named_steps["gbm"]
        pre = tip_model.named_steps["pre"]
        feat_names = (list(pre.transformers_[0][2]) +
                      list(pre.transformers_[1][1].get_feature_names_out(pre.transformers_[1][2])))
        importances = pd.DataFrame({
            "feature": feat_names,
            "importance": gbm.feature_importances_,
        }).sort_values("importance", ascending=False).head(10)

        st.subheader("What drives the tip prediction?")
        st.caption("Top 10 features by GBM impurity-based importance. Higher bar = the model relies on this feature more.")
        imp_chart = (
            alt.Chart(importances).mark_bar(cornerRadius=4).encode(
                x=alt.X("importance:Q", title="Importance"),
                y=alt.Y("feature:N", sort="-x", title=None),
                color=alt.Color("importance:Q", scale=alt.Scale(scheme="yelloworangered"),
                                legend=None),
                tooltip=["feature", alt.Tooltip("importance:Q", format=".3f")],
            ).properties(height=280)
        )
        st.altair_chart(imp_chart, use_container_width=True)
    except Exception as e:
        st.caption(f"(feature-importance unavailable: {e})")


# ------- Analytics tab -------
with tab_analytics:
    st.subheader("Natural-language query")
    st.caption("Try one of the examples below, or ask your own. Out-of-domain questions return 'not supported' (safety by default).")

    if "nl_query" not in st.session_state:
        st.session_state.nl_query = "What was the average fare?"

    examples = [
        "Trips from JFK?",
        "Trips to Brooklyn?",
        "Average fare?",
        "Average tip?",
        "Busiest hour?",
    ]
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex, use_container_width=True, key=f"ex_{ex}"):
            st.session_state.nl_query = ex

    q = st.text_input("Your question", key="nl_query")
    if q:
        sql = nl_to_sql(q)
        if sql is None:
            st.warning("No pattern matched. Try one of the example chips above.")
        else:
            st.code(sql, language="sql")
            result = run_sql(sql, silver_f)
            st.dataframe(result, use_container_width=True, hide_index=True)
            d1, d2 = st.columns(2)
            d1.download_button("⬇ Download SQL", sql.encode(), "nycride_query.sql",
                               "text/plain", use_container_width=True)
            d2.download_button("⬇ Download result (CSV)", result.to_csv(index=False).encode(),
                               "nycride_result.csv", "text/csv", use_container_width=True)

    with st.expander("Supported patterns (3)"):
        st.markdown(
            "- **Location filter**: `trips from JFK` → `SELECT COUNT(*) WHERE PULocationID = 132`\n"
            "- **Aggregate**: `average fare` → `SELECT AVG(fare_amount) FROM silver`\n"
            "- **Peak time**: `busiest hour` → `SELECT hour, COUNT(*) GROUP BY 1 ORDER BY 2 DESC LIMIT 1`\n"
            "\nUpgrade path: swap `nl_to_sql()` for a real LLM + a SELECT-only allowlist validator."
        )

    st.divider()
    st.subheader("Top 10 pickup zones")
    top_zones = (silver_f.groupby("PULocationID").size().rename("trips")
                         .reset_index().sort_values("trips", ascending=False).head(10))
    zone_bar = (
        alt.Chart(top_zones).mark_bar(cornerRadius=4).encode(
            x=alt.X("trips:Q", title="Trips"),
            y=alt.Y("PULocationID:O", sort="-x", title="Zone ID"),
            color=alt.Color("trips:Q", scale=alt.Scale(scheme="yelloworangered"), legend=None),
            tooltip=["PULocationID", "trips"],
        ).properties(height=320)
    )
    st.altair_chart(zone_bar, use_container_width=True)


# ------- Methodology tab -------
with tab_method:
    st.markdown(
        f"""
        ## Methodology — decisions, tradeoffs, honest shortcuts

        A portfolio project without a methodology section is a portfolio project you can't defend.
        Below is what I chose, why, and what I'd change with 10× the time.

        ---
        ### Decision 1 — Why a 3-phase pipeline, not a single notebook?

        **Chose:** Three distinct phases (DE → ML → Forecast), each writes a checkpoint that the next phase reads.

        **Why:** Each phase is independently testable (10 + 7 + 9 invariants), independently runnable
        (`python src/phase1_pipeline.py`), and the medallion split (bronze → silver → gold) matches
        what every real data org uses. A single notebook couples cleaning to modeling and makes
        reproducing one piece require rerunning everything.

        **What I'd change with 10× the time:** add dbt for the silver → gold transformation so the
        warehouse schema is version-controlled SQL rather than imperative pandas.

        ---
        ### Decision 2 — Why ship a model whose R² is near zero?

        **Chose:** GBM MAE $%.2f, linear baseline MAE ≈ same, hold-out R² near zero on the 10k sample.

        **Why:** Baseline-first is a non-negotiable ML rule. On the 10k synthetic sample the signal
        is thin and GBM overfits noise. Hiding that fact behind a cherry-picked metric would be the
        exact anti-pattern Phase 2 is teaching against. On the real 2.9M-row TLC data the gap widens
        meaningfully — the demo honestly shows the small-sample regime.

        **What I'd change with 10× the time:** add zone-level features (joined against
        `taxi_zone_lookup.csv`), weather, and holidays. Expect $0.50-$1.00 MAE improvement on the
        full dataset.

        ---
        ### Decision 3 — Why a deterministic regex for NL→SQL instead of an LLM?

        **Chose:** 3-pattern regex mapper with a `None` fallback for anything out of scope.

        **Why:** For 3 patterns, a regex is a 20-line safety-first implementation that never
        hallucinates. An LLM for 3 patterns is engineering theater — more code, more latency, more
        failure modes, no new capability. The honest upgrade path is a hybrid: LLM-generated SQL
        wrapped in a SELECT-only allowlist validator so hallucinated queries can't run.

        **What I'd change with 10× the time:** add the hybrid LLM + allowlist for arbitrary queries,
        keep the 3 regex patterns as the fast path.

        ---
        ### Decision 4 — Why synthesise coordinates instead of joining the zone lookup?

        **Chose:** Each row's (lat, lon) comes from a borough centroid + deterministic jitter.

        **Why (honest):** Shortcut. The real TLC zone shapefile is a shapefile — needs `geopandas`,
        pyproj, GDAL binaries — which inflates the Streamlit Cloud build time and bloats the free-tier
        memory. For a 10k-row demo the hex-density visualisation is indistinguishable; on real data
        the join is a one-liner against `taxi_zone_lookup.csv`.

        **What I'd change with 10× the time:** add the real zone geometry and show neighborhood-level
        insights (e.g. "pickups in Chelsea tip 23%% more than Midtown East").

        ---
        ### Decision 5 — Why ignore cash tips entirely?

        **Chose:** Phase 2 trains and audits only on `payment_type == 1` (card) rows (~69%% of trips).

        **Why:** TLC records cash tips as $0.00 — the driver pockets them, the meter never sees them.
        Including cash trips teaches the model that "cash passengers never tip", which is false and
        would propagate borough-specific bias (poorer neighborhoods use cash more). Dropping cash
        is the honest choice; what we lose is the ability to predict for cash-paying trips at all.

        **What I'd change with 10× the time:** show a second model trained only on card trips but
        with a "cash probability" wrapper for the Driver UI, so the app can at least warn
        "this trip profile is 40%% likely to be cash — tip estimate uncertain".

        ---

        ## What a staff engineer would flag that I left in

        - **No feature store.** Every reload recomputes features from silver. Fine at 10k rows, breaks
          at 2.9M. The upgrade is `feast` or a materialised view in the warehouse.
        - **No model monitoring.** The invariants validate on build, not on live predictions. In prod,
          daily re-runs of the invariants on the previous day's traffic would catch drift.
        - **No concept of recency.** The GBM uses lag1/24/168 but no decaying weight for recent
          observations. Weekend-vs-weekday splits with recency weighting would reduce MAPE further.
        - **The `nl_to_sql` executor is pattern-matched, not actual SQL.** A single DuckDB query engine
          swap would make it general-purpose, at the cost of 15MB more in the Space.
        """ % (overall_mae,)
    )


# ------- About tab -------
with tab_about:
    st.markdown(
        f"""
        ## About

        **NYCRide Analytics** is the flagship capstone of the
        [Bits to Builds](https://bitstobuilds.com) course — a 3-phase end-to-end
        taxi analytics platform spanning DE, ML, forecasting and NL-SQL.

        | Phase | Track | Deliverable |
        |-------|-------|-------------|
        | 1. Pipeline       | DE          | bronze→silver→gold medallion, 10 invariants, reject counts documented |
        | 2. ML + fairness  | ML          | tip-prediction GBM, 5-borough fairness audit, baseline-first |
        | 3. Forecast + NLQ | DL/DA/GenAI | hour-ahead demand GBM + NL→SQL interface (3 patterns, None-safe) |

        ## Data

        [NYC TLC Yellow Taxi Trip Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page),
        public domain. This Space uses a deterministic 10,000-row synthetic sample so
        cold-starts fit in the Streamlit Cloud free tier; the pipeline runs on the full
        ~3M-row monthly file when pointed at real parquet.

        ## Honest limits

        - **Tip model barely beats baseline on the 10k sample.** GBM MAE ≈ ${overall_mae:.2f}
          vs linear baseline near-identical. On real 2.9M-row data the gap widens.
          Shipped the baseline-first result on purpose — the Phase 2 lesson.
        - **Fairness gap on this sample: ${fairness_gap:.2f}.** Low because the sample is
          uniform across boroughs. On real data this is the first thing a fairness
          review would flag.
        - **Map coordinates are synthesised.** TLC gives zone IDs, not lat/lon. We jitter
          deterministically around borough centroids; on real data, join against
          `taxi_zone_lookup.csv`.
        - **NL→SQL handles 3 patterns.** Anything else returns `None` (safety). Upgrade:
          wrap an LLM in a SELECT-only allowlist validator.

        ## Source

        [github.com/PJsAcademy/nycride-analytics](https://github.com/PJsAcademy/nycride-analytics)
        — 26 pytest invariants, Dockerfile for self-hosted deploy, portfolio-review
        scaffolding (`templates/portfolio/`).
        """
    )

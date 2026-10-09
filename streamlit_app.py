"""NYCRide Analytics — Streamlit demo for the Bits to Builds flagship capstone.

Tabs:
  1. Dashboard  — hero KPIs, pydeck hex-density map of NYC pickups, hour-x-DOW
                  demand heatmap, daily revenue line.
  2. Driver     — top high-tip zones, tip-prediction form with honest-limit
                  callout, 5-borough fairness audit chart.
  3. Analytics  — NL->SQL interface over the silver warehouse (safety fallback
                  when a question doesn't match the 3 known patterns), plus a
                  top-10-zones bar chart.
  4. About      — honest project description + limits.

Checkpoints come from either (a) joblib-loadable files in checkpoints/, or
(b) a first-run bootstrap that calls phase1/2/3 directly. Immune to pickle
version shifts between local training and deployed runtime.
"""
from __future__ import annotations

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

# Rough borough centroids (lat, lon) + airports. Used to project zone IDs onto
# a map; synthesised sample has no real geometry, so we jitter deterministically
# around the centroid. On real TLC data, join against taxi_zone_lookup.csv.
BOROUGH_CENTROIDS = {
    "Manhattan":     (40.776, -73.965),
    "Bronx":         (40.845, -73.865),
    "Brooklyn":      (40.650, -73.950),
    "Queens":        (40.720, -73.820),
    "Staten Island": (40.580, -74.150),
}
AIRPORT_COORDS = {132: (40.6413, -73.7781), 138: (40.7769, -73.8740), 1: (40.6895, -74.1745)}

sys.path.insert(0, str(HERE / "src"))

st.set_page_config(page_title="NYCRide Analytics", page_icon="🚕", layout="wide")


# ======================================================= bootstrap

@st.cache_resource(show_spinner="First-time setup: building silver + ML + forecast (~30s)...")
def bootstrap_checkpoints() -> None:
    """Rebuild checkpoints from source if any are missing or fail to load.
    Runs once per container; deterministic (seed=42)."""
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


# ======================================================= loaders (cached)

@st.cache_data(show_spinner="Loading silver warehouse...")
def load_silver() -> pd.DataFrame:
    df = pd.read_parquet(CHECKPOINTS / "silver.parquet")
    # Deterministic (lat, lon) per row from borough centroid + jitter. On real
    # TLC data this is replaced by a join against taxi_zone_lookup.csv.
    rng = np.random.default_rng(42)
    lats = np.empty(len(df))
    lons = np.empty(len(df))
    for i, (pul, bor) in enumerate(zip(df["PULocationID"].values, df["pickup_borough"].values)):
        if pul in AIRPORT_COORDS:
            lat0, lon0 = AIRPORT_COORDS[pul]
            jitter_scale = 0.004
        else:
            lat0, lon0 = BOROUGH_CENTROIDS.get(bor, BOROUGH_CENTROIDS["Manhattan"])
            jitter_scale = 0.025
        lats[i] = lat0 + rng.normal(0, jitter_scale)
        lons[i] = lon0 + rng.normal(0, jitter_scale)
    df["pickup_lat"] = lats
    df["pickup_lon"] = lons
    return df


@st.cache_resource(show_spinner="Loading tip-prediction model...")
def load_tip_model():
    # Trusted artifact: produced by src/phase2_ml.py in this repo's own build.
    return joblib.load(CHECKPOINTS / "tip_model.pkl")


@st.cache_resource(show_spinner="Loading demand-forecast model...")
def load_demand_model():
    return joblib.load(CHECKPOINTS / "demand_model.pkl")


@st.cache_data(show_spinner="Loading high-tip zones...")
def load_high_tip_zones() -> pd.DataFrame:
    return pd.read_csv(CHECKPOINTS / "high_tip_zones.csv")


@st.cache_data
def compute_fairness(silver: pd.DataFrame) -> pd.DataFrame:
    """Per-borough MAE of the tip model on the card-only subset.
    Recomputed in-app so the chart matches the numbers in About."""
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split
    model = load_tip_model()
    card = silver[silver["payment_type"] == 1]
    feats = ["trip_distance", "fare_amount", "trip_duration_min", "passenger_count",
             "hour", "dow", "is_weekend", "pickup_borough"]
    X = card[feats]
    y = card["tip_amount"]
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    yhat = model.predict(X_te)
    audit = pd.DataFrame({"y": y_te.values, "yhat": yhat, "borough": X_te["pickup_borough"].values})
    per = audit.groupby("borough").apply(lambda d: mean_absolute_error(d.y, d.yhat))
    return per.rename("mae").reset_index().sort_values("mae")


# ======================================================= Phase 3's NL->SQL mapper

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


# ======================================================= header + hero

silver = load_silver()
tip_model = load_tip_model()
high_tip = load_high_tip_zones()
fairness = compute_fairness(silver)

st.markdown(
    """
    # 🚕 NYCRide Analytics
    End-to-end taxi analytics platform built from **NYC TLC Yellow Taxi** data (public domain).
    Flagship capstone of the [Bits to Builds](https://bitstobuilds.com) course.
    """
)

# Hero KPI row
total_trips = len(silver)
total_revenue = float(silver["total_amount"].sum())
median_fare = float(silver["fare_amount"].median())
fairness_gap = float(fairness["mae"].max() - fairness["mae"].min())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Trips (sample)", f"{total_trips:,}")
k2.metric("Revenue", f"${total_revenue:,.0f}")
k3.metric("Median fare", f"${median_fare:.2f}")
k4.metric(
    "Fairness gap (max-min borough MAE)",
    f"${fairness_gap:.2f}",
    help="Lower = model treats boroughs similarly. $0 = perfect parity.",
)

st.divider()

tab_dashboard, tab_driver, tab_analytics, tab_about = st.tabs(
    ["📊 Dashboard", "🚗 Driver", "💬 Analytics", "ℹ️ About"]
)


# ------- Dashboard tab -------
with tab_dashboard:
    st.subheader("Where pickups happen")
    st.caption("Hex-density of pickups across NYC. Darker hexes = more trips. Zoom and tilt the map.")

    layer = pdk.Layer(
        "HexagonLayer",
        data=silver[["pickup_lon", "pickup_lat"]].rename(
            columns={"pickup_lon": "lon", "pickup_lat": "lat"}
        ),
        get_position=["lon", "lat"],
        radius=250,
        elevation_scale=4,
        extruded=True,
        coverage=0.9,
        pickable=True,
    )
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(
            latitude=40.74, longitude=-73.95, zoom=10, pitch=45, bearing=0,
        ),
        map_style=None,  # defaults to dark basemap on dark theme
    )
    st.pydeck_chart(deck, use_container_width=True)

    st.divider()
    c1, c2 = st.columns([3, 2])

    with c1:
        st.subheader("When pickups happen")
        st.caption("Trips by hour of day × day of week. Reveals rush hours and weekend patterns.")
        pivot = (silver.assign(hour=silver["tpep_pickup_datetime"].dt.hour,
                               dow=silver["tpep_pickup_datetime"].dt.day_name())
                       .groupby(["dow", "hour"]).size().rename("trips").reset_index())
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        heatmap = (
            alt.Chart(pivot).mark_rect().encode(
                x=alt.X("hour:O", title="Hour of day"),
                y=alt.Y("dow:O", sort=dow_order, title=None),
                color=alt.Color("trips:Q", scale=alt.Scale(scheme="yelloworangered"), title="Trips"),
                tooltip=["dow", "hour", "trips"],
            ).properties(height=280)
        )
        st.altair_chart(heatmap, use_container_width=True)

    with c2:
        st.subheader("Daily revenue")
        st.caption("Sum of total_amount, per day. From the gold_daily table.")
        daily = (silver.assign(date=silver["tpep_pickup_datetime"].dt.date)
                       .groupby("date")
                       .agg(revenue=("total_amount", "sum"), trips=("fare_amount", "count"))
                       .reset_index())
        line = (
            alt.Chart(daily).mark_area(
                line={"color": "#FFC72C"}, color=alt.Gradient(
                    gradient="linear",
                    stops=[alt.GradientStop(color="#FFC72C", offset=0),
                           alt.GradientStop(color="#1A1A1A", offset=1)],
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


# ------- Driver tab -------
with tab_driver:
    c1, c2 = st.columns([2, 3])

    with c1:
        st.subheader("Best pickup zones")
        st.caption("Top 10 zones by average tip. Minimum 5 trips.")
        top = high_tip.sort_values("avg_tip", ascending=False).head(10).copy()
        top = top.rename(columns={"PULocationID": "Zone", "trips": "Trips", "avg_tip": "Avg tip ($)"})
        st.dataframe(
            top.style.background_gradient(subset=["Avg tip ($)"], cmap="YlOrRd")
                     .format({"Avg tip ($)": "${:.2f}"}),
            use_container_width=True, hide_index=True,
        )

    with c2:
        st.subheader("Fairness audit — per-borough MAE")
        st.caption("How much the tip model misses by, in each borough. The brand promise: no borough gets worse service.")
        bar = (
            alt.Chart(fairness).mark_bar(cornerRadius=4).encode(
                x=alt.X("borough:N", sort="-y", title=None),
                y=alt.Y("mae:Q", title="Mean absolute error ($)"),
                color=alt.Color("mae:Q", scale=alt.Scale(scheme="yelloworangered"), legend=None),
                tooltip=["borough", alt.Tooltip("mae:Q", format="$.3f")],
            ).properties(height=240)
        )
        st.altair_chart(bar, use_container_width=True)
        st.caption(
            f"**Honest limit:** on the 10k synthetic sample the gap is ${fairness_gap:.2f} — "
            "low because the sample is uniform. On real 2.9M-row TLC data, borough gaps "
            "are the first thing a fairness review would flag."
        )

    st.divider()
    st.subheader("Tip prediction for a specific trip")
    with st.form("tip_form"):
        c1, c2, c3 = st.columns(3)
        distance = c1.number_input("Trip distance (miles)", 0.1, 50.0, 2.5, 0.1)
        fare = c2.number_input("Fare amount ($)", 2.5, 200.0, 15.0, 0.5)
        duration = c3.number_input("Trip duration (min)", 1, 180, 12, 1)
        c4, c5, c6 = st.columns(3)
        borough = c4.selectbox("Pickup borough", BOROUGHS, index=0)
        hour = c5.slider("Hour of day", 0, 23, 18)
        is_weekend = c6.toggle("Weekend?", value=False)
        submitted = st.form_submit_button("Predict tip", type="primary")

    if submitted:
        row = pd.DataFrame([{
            "trip_distance": distance, "fare_amount": fare, "trip_duration_min": duration,
            "passenger_count": 1, "hour": hour, "dow": 5 if is_weekend else 2,
            "is_weekend": int(is_weekend), "pickup_borough": borough,
        }])
        predicted = float(tip_model.predict(row)[0])
        pct = predicted / fare * 100
        a, b = st.columns(2)
        a.metric("Predicted tip", f"${predicted:.2f}", f"{pct:.1f}% of fare")
        b.caption(
            "**Honest limit:** on the 10k sample this model's hold-out R² is near zero — "
            "it barely beats a linear baseline. On the real 2.9M-row dataset the gap is "
            "meaningful. See About for the full numbers."
        )


# ------- Analytics tab -------
with tab_analytics:
    st.subheader("Natural-language query")
    st.caption("Ask a question. Matches one of 3 known patterns; otherwise returns 'not supported' (safety by default).")

    q = st.text_input(
        "Your question",
        value="What was the average fare?",
        placeholder="e.g. How many trips from JFK? / average tip / busiest hour",
    )
    if st.button("Ask") and q:
        sql = nl_to_sql(q)
        if sql is None:
            st.warning("No pattern matched. Try: trips from <place>, average fare, busiest hour.")
        else:
            st.code(sql, language="sql")
            st.dataframe(run_sql(sql, silver), use_container_width=True, hide_index=True)

    with st.expander("Supported patterns"):
        st.markdown(
            "- **Location filter**: `trips from JFK` → `SELECT COUNT(*) WHERE PULocationID = 132`\n"
            "- **Aggregate**: `average fare` → `SELECT AVG(fare_amount) FROM silver`\n"
            "- **Peak time**: `busiest hour` → `SELECT hour, COUNT(*) GROUP BY 1 ORDER BY 2 DESC LIMIT 1`\n"
            "\nUpgrade path: swap `nl_to_sql()` for a real LLM + SELECT-only allowlist validator."
        )

    st.divider()
    st.subheader("Top 10 pickup zones")
    top_zones = (silver.groupby("PULocationID").size().rename("trips")
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


# ------- About tab -------
with tab_about:
    st.markdown(
        f"""
        ## About

        **NYCRide Analytics** is the flagship capstone of the Bits to Builds course. It's a 3-phase
        end-to-end taxi analytics platform:

        | Phase | Track | Deliverable |
        |-------|-------|-------------|
        | 1. Pipeline       | DE          | bronze→silver→gold, 10 invariants, rejects documented |
        | 2. ML + fairness  | ML          | tip-prediction GBM, 5-borough fairness audit |
        | 3. Forecast + NLQ | DL/DA/GenAI | hour-ahead demand GBM + NL→SQL analytics |

        ## Data

        [NYC TLC Yellow Taxi Trip Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page),
        public domain. This Space uses a deterministic 10,000-row synthetic sample so cold-starts
        are fast on the Streamlit Cloud free tier; the pipeline runs on the full ~3M-row monthly file
        when pointed at real parquet.

        ## Honest limits

        - **Tip model barely beats baseline on the 10k sample.** GBM MAE ≈ $3.79 vs linear baseline
          MAE ≈ $3.78. On real 2.9M-row data the gap widens. Shipped the baseline-first result on
          purpose — that's the Phase 2 lesson.
        - **Fairness gap on this sample: ${fairness_gap:.2f}.** Low because the sample is uniform
          across boroughs. On real data this is the first thing a fairness review would flag.
        - **Map coordinates are synthesised.** The TLC data gives zone IDs, not lat/lon. We jitter
          around borough centroids deterministically; on real data, join against
          `taxi_zone_lookup.csv`.
        - **NL→SQL handles 3 patterns.** Anything else returns `None` (safety). Upgrade: wrap an
          LLM in a SELECT-only allowlist validator.

        ## Source

        [github.com/PJsAcademy/nycride-analytics](https://github.com/PJsAcademy/nycride-analytics) —
        26 pytest invariants, Dockerfile for self-hosted deploy, portfolio-review scaffolding
        under `templates/portfolio/`.
        """
    )

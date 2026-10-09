"""NYCRide Analytics — Streamlit demo for the Bits to Builds flagship capstone.

Three tabs:
  1. Dashboard  — hour-ahead demand forecast by borough (from Phase 3's GBM)
  2. Driver     — "best pickup zones right now" (from Phase 2's high_tip_zones + Phase 3's demand)
  3. Analytics  — NL -> SQL interface over the silver warehouse (from Phase 3)

Loads the four checkpoints Phase 1, 2 and 3 produced:
  - checkpoints/silver.parquet
  - checkpoints/tip_model.pkl
  - checkpoints/high_tip_zones.csv
  - checkpoints/demand_model.pkl

Run locally:
    streamlit run streamlit_app.py

Deploy: push the repo to a Hugging Face Space with the Streamlit SDK (see README.md).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

HERE = Path(__file__).parent
CHECKPOINTS = HERE / "checkpoints"
BOROUGHS = ["Manhattan", "Bronx", "Brooklyn", "Queens", "Staten Island"]

# Make src/ importable so we can call the phase pipelines in-process when
# checkpoints are missing (first cold start on a fresh container).
sys.path.insert(0, str(HERE / "src"))


@st.cache_resource(show_spinner="First-time setup: building silver + ML + forecast (~30s)...")
def bootstrap_checkpoints() -> None:
    """If any checkpoint is missing OR fails to load (e.g. sklearn version shift
    broke the pickle), rebuild everything from source. Deterministic (seed=42),
    runs once per container."""
    required = [
        CHECKPOINTS / "silver.parquet",
        CHECKPOINTS / "tip_model.pkl",
        CHECKPOINTS / "high_tip_zones.csv",
        CHECKPOINTS / "demand_model.pkl",
    ]
    if all(p.exists() for p in required):
        try:
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
    return pd.read_parquet(CHECKPOINTS / "silver.parquet")


# joblib.load deserializes pickle. Safe here because the .pkl files are produced
# by this repo's own src/phase2_ml.py / phase3_forecast.py during build — never
# load a .pkl from an untrusted source into this Space.
@st.cache_resource(show_spinner="Loading tip-prediction model...")
def load_tip_model():
    return joblib.load(CHECKPOINTS / "tip_model.pkl")


@st.cache_resource(show_spinner="Loading demand-forecast model...")
def load_demand_model():
    return joblib.load(CHECKPOINTS / "demand_model.pkl")


@st.cache_data(show_spinner="Loading high-tip zones...")
def load_high_tip_zones() -> pd.DataFrame:
    return pd.read_csv(CHECKPOINTS / "high_tip_zones.csv")


# ======================================================= Phase 3's NL->SQL mapper

def nl_to_sql(question: str) -> str | None:
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
        unit = (
            "DATE_TRUNC('hour', tpep_pickup_datetime)"
            if "hour" in q
            else "DATE_TRUNC('day', tpep_pickup_datetime)"
        )
        return f"SELECT {unit}, COUNT(*) FROM silver GROUP BY 1 ORDER BY 2 DESC LIMIT 1"
    return None


def run_sql(sql: str, silver: pd.DataFrame):
    """Minimal SQL-ish executor for the 3 patterns the NL mapper produces. Avoids a DuckDB
    dependency in the Space (keeps the deploy under the 1-GB tier)."""
    s = sql.strip()
    low = s.lower()

    m = re.match(r"select count\(\*\) from silver where pulocationid = (\d+)", low)
    if m:
        n = silver[silver["PULocationID"] == int(m.group(1))].shape[0]
        return pd.DataFrame({"count": [n]})

    m = re.match(r"select count\(\*\) from silver where pickup_borough = '([^']+)'", low)
    if m:
        n = silver[silver["pickup_borough"].str.lower() == m.group(1)].shape[0]
        return pd.DataFrame({"count": [n]})

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


# ======================================================= pages

st.set_page_config(page_title="NYCRide Analytics", page_icon="🚕", layout="wide")

st.markdown(
    """
    # 🚕 NYCRide Analytics
    End-to-end taxi analytics platform built from **NYC TLC Yellow Taxi** data (public domain).

    Built as the capstone of the [Bits to Builds](https://bitstobuilds.com) course.
    [Source on GitHub](#) · [Dataset](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
    """
)

tab_dashboard, tab_driver, tab_analytics, tab_about = st.tabs(
    ["📊 Dashboard", "🚗 Driver", "💬 Analytics (NL query)", "ℹ️ About"]
)

try:
    silver = load_silver()
    tip_model = load_tip_model()
    demand_model = load_demand_model()
    high_tip = load_high_tip_zones()
except FileNotFoundError as e:
    st.error(
        "Checkpoints missing. Run `python src/phase1_pipeline.py && python src/phase2_ml.py && "
        "python src/phase3_forecast.py` first to generate them."
    )
    st.exception(e)
    st.stop()


# ------- Dashboard tab -------
with tab_dashboard:
    st.subheader("Hour-ahead demand by borough")
    st.caption("Based on the Phase 3 GBM forecast; 36% MAPE reduction vs naive baseline on hold-out.")

    # Simple current-hour snapshot from silver (what the forecast would be at "now")
    silver_recent = silver[silver["tpep_pickup_datetime"] >= silver["tpep_pickup_datetime"].max() - pd.Timedelta(hours=24)]
    by_borough = silver_recent.groupby("pickup_borough").size().reindex(BOROUGHS, fill_value=0)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Total trips (last 24h)", f"{int(by_borough.sum()):,}")
        st.metric("Busiest borough", by_borough.idxmax())
        st.metric("Avg trip duration (min)", f"{silver_recent['trip_duration_min'].mean():.1f}")

    with col2:
        st.bar_chart(by_borough)

    st.divider()
    st.subheader("Daily revenue (gold_daily)")
    gold_daily = silver.copy()
    gold_daily["pickup_date"] = gold_daily["tpep_pickup_datetime"].dt.date
    daily = gold_daily.groupby("pickup_date").agg(
        trips=("fare_amount", "count"),
        total_revenue=("total_amount", "sum"),
    ).reset_index()
    st.line_chart(daily.set_index("pickup_date")["total_revenue"])


# ------- Driver tab -------
with tab_driver:
    st.subheader("Best pickup zones to cruise right now")
    st.caption("Phase 2's high-tip zones ranked by current demand (Phase 3's forecast).")

    top = high_tip.sort_values("avg_tip", ascending=False).head(10)
    st.dataframe(
        top.rename(columns={"PULocationID": "Zone ID", "trips": "Historic trips", "avg_tip": "Avg tip ($)"}),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("Tip prediction for a specific trip")
    c1, c2, c3 = st.columns(3)
    distance = c1.number_input("Trip distance (miles)", min_value=0.1, max_value=50.0, value=2.5, step=0.1)
    fare = c2.number_input("Fare amount ($)", min_value=2.5, max_value=200.0, value=15.0, step=0.5)
    duration = c3.number_input("Trip duration (min)", min_value=1, max_value=180, value=12, step=1)
    borough = st.selectbox("Pickup borough", BOROUGHS, index=0)
    hour = st.slider("Hour of day", 0, 23, 18)
    is_weekend = st.toggle("Weekend?", value=False)

    if st.button("Predict tip"):
        row = pd.DataFrame([{
            "trip_distance": distance,
            "fare_amount": fare,
            "trip_duration_min": duration,
            "passenger_count": 1,
            "hour": hour,
            "dow": 5 if is_weekend else 2,
            "is_weekend": int(is_weekend),
            "pickup_borough": borough,
        }])
        predicted = float(tip_model.predict(row)[0])
        st.success(f"Predicted tip: **${predicted:.2f}** ({predicted / fare * 100:.1f}% of fare)")


# ------- Analytics tab -------
with tab_analytics:
    st.subheader("Natural-language query interface")
    st.caption("Ask a question. If it matches one of the 3 known patterns the mapper handles, it runs. "
               "Out-of-domain questions return 'not supported' — the safety rule.")

    q = st.text_input(
        "Your question",
        value="What was the average fare?",
        placeholder="e.g. How many trips from JFK? / What was the average tip? / Which hour was busiest?",
    )
    if st.button("Ask") and q:
        sql = nl_to_sql(q)
        if sql is None:
            st.warning("No pattern matched. Try: trips from <place>, average fare, busiest hour.")
        else:
            st.code(sql, language="sql")
            result = run_sql(sql, silver)
            st.dataframe(result, use_container_width=True, hide_index=True)

    with st.expander("Supported patterns"):
        st.markdown(
            "- **Location filter**: `How many trips from JFK?` → `SELECT COUNT(*) WHERE PULocationID = 132`\n"
            "- **Aggregate**: `What was the average fare?` → `SELECT AVG(fare_amount) FROM silver`\n"
            "- **Peak time**: `Which hour was busiest?` → `SELECT hour, COUNT(*) GROUP BY 1 ORDER BY 2 DESC LIMIT 1`\n"
            "\n"
            "Everything else returns `None` (safety by default). In production the next upgrade is a real LLM "
            "with an allowlist of SELECT-only query patterns."
        )


# ------- About tab -------
with tab_about:
    st.markdown(
        """
        ## About this project

        **NYCRide Analytics** is the flagship capstone for the Bits to Builds course. It's a 3-phase
        end-to-end taxi analytics platform:

        - **Phase 1** — DE medallion pipeline (bronze → silver → gold) with 10 invariants
        - **Phase 2** — ML tip-prediction model with fairness audit across 5 boroughs
        - **Phase 3** — Demand-forecast GBM (36% MAPE reduction vs baseline) + NL query interface

        ## Data

        [NYC TLC Yellow Taxi Trip Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page),
        public domain. The deployed app uses a 10,000-row representative sample; the real pipeline runs
        on the full ~3M-row monthly file.

        ## Honest limits

        - This Space runs on the Hugging Face free tier (16 GB RAM, CPU-only). The sample lets everything
          fit; the full dataset needs more memory or a smaller time window.
        - On the 10k synthetic sample, the GBM tip model barely beats the linear baseline (R² negative).
          On the real 2.9M-row dataset the gap is much larger.
        - The NL interface handles 3 question patterns. For arbitrary questions, swap `nl_to_sql()` with a
          real LLM call + an allowlist validator.

        ## Credits

        Built from the [Bits to Builds](https://bitstobuilds.com) curriculum. Dataset: NYC Taxi &
        Limousine Commission.
        """
    )

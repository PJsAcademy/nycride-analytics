"""Phase 3 — hour-ahead demand forecast + NL->SQL interface.

Builds an hourly series of pickup counts, trains a GBM on lag features
(lag1, lag24, lag168), reports MAPE vs the naive-last-hour baseline, and
writes checkpoints/demand_model.pkl.
"""
from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit

HERE = Path(__file__).resolve().parent.parent
CHECKPOINTS = HERE / "checkpoints"


def mape(y_true, y_pred, eps: float = 1.0) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_pred - y_true) / np.maximum(y_true, eps))))


def nl_to_sql(question: str) -> str | None:
    """Deterministic 3-pattern NL->SQL mapper. Returns None on no match (safety)."""
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


def main():
    silver = pd.read_parquet(CHECKPOINTS / "silver.parquet")
    print(f"[phase3] loaded silver: {len(silver):,} rows")

    hourly = (silver.assign(ts=silver["tpep_pickup_datetime"].dt.floor("h"))
                    .groupby("ts").size().rename("trips").to_frame())
    hourly = hourly.reindex(pd.date_range(hourly.index.min(), hourly.index.max(), freq="h"),
                            fill_value=0).rename_axis("ts")
    print(f"[phase3] hourly series: {len(hourly)} hours")

    for lag in (1, 24, 168):
        hourly[f"lag{lag}"] = hourly["trips"].shift(lag)
    hourly["hour"] = hourly.index.hour
    hourly["dow"]  = hourly.index.dayofweek
    hourly = hourly.dropna()

    X = hourly[["lag1", "lag24", "lag168", "hour", "dow"]]
    y = hourly["trips"]

    split = int(len(hourly) * 0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    baseline_pred = X_te["lag1"]
    baseline_mape = mape(y_te, baseline_pred)
    print(f"[phase3] baseline (naive last-hour) MAPE: {baseline_mape:.3f}")

    model = GradientBoostingRegressor(random_state=42, n_estimators=150, max_depth=3)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    gbm_mape = mape(y_te, pred)
    print(f"[phase3] GBM MAPE: {gbm_mape:.3f}")
    print(f"[phase3] relative improvement: {(baseline_mape - gbm_mape) / baseline_mape:.1%}")

    joblib.dump(model, CHECKPOINTS / "demand_model.pkl")
    print(f"[phase3] wrote {CHECKPOINTS / 'demand_model.pkl'}")

    for q in ("How many trips from JFK?", "What was the average fare?", "Which hour was busiest?",
              "Tell me a story"):
        print(f"  Q: {q!r:40} -> SQL: {nl_to_sql(q)!r}")


if __name__ == "__main__":
    main()

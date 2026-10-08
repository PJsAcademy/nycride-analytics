"""Phase 2 — tip-prediction model + 5-borough fairness audit.

Trains on card-only trips (cash trips have no tip signal). Produces:
  - checkpoints/tip_model.pkl       (sklearn Pipeline)
  - checkpoints/high_tip_zones.csv  (top-50 pickup zones by avg tip)
  - stdout: baseline MAE, GBM MAE/R², per-borough MAE gap
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).resolve().parent.parent
CHECKPOINTS = HERE / "checkpoints"

NUMERIC = ["trip_distance", "fare_amount", "trip_duration_min", "passenger_count", "hour", "dow", "is_weekend"]
CATEGORICAL = ["pickup_borough"]
TARGET = "tip_amount"


def main():
    silver = pd.read_parquet(CHECKPOINTS / "silver.parquet")
    print(f"[phase2] loaded silver: {len(silver):,} rows")

    card = silver[silver["payment_type"] == 1].copy()
    print(f"[phase2] card-only subset: {len(card):,} rows "
          f"({len(card) / len(silver):.1%} of silver)")

    X = card[NUMERIC + CATEGORICAL]
    y = card[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    borough_te = X_te["pickup_borough"]

    pre = ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])

    baseline = Pipeline([("pre", pre), ("lr", LinearRegression())])
    baseline.fit(X_tr, y_tr)
    baseline_mae = mean_absolute_error(y_te, baseline.predict(X_te))
    print(f"[phase2] baseline LinearRegression MAE: ${baseline_mae:.3f}")

    gbm = Pipeline([
        ("pre", pre),
        ("gbm", GradientBoostingRegressor(random_state=42, n_estimators=100, max_depth=3)),
    ])
    gbm.fit(X_tr, y_tr)
    yhat = gbm.predict(X_te)
    gbm_mae = mean_absolute_error(y_te, yhat)
    gbm_r2 = r2_score(y_te, yhat)
    print(f"[phase2] GBM MAE: ${gbm_mae:.3f}   R²: {gbm_r2:.3f}")

    audit = pd.DataFrame({"y": y_te, "yhat": yhat, "borough": borough_te})
    per_borough = (audit.groupby("borough")
                        .apply(lambda d: mean_absolute_error(d.y, d.yhat))
                        .rename("mae"))
    print("[phase2] per-borough MAE:")
    print(per_borough.to_string())
    print(f"[phase2] fairness gap (max-min MAE): ${per_borough.max() - per_borough.min():.3f}")

    joblib.dump(gbm, CHECKPOINTS / "tip_model.pkl")
    print(f"[phase2] wrote {CHECKPOINTS / 'tip_model.pkl'}")

    high_tip = (card.groupby("PULocationID")
                    .agg(trips=("tip_amount", "count"), avg_tip=("tip_amount", "mean"))
                    .query("trips >= 5")
                    .sort_values("avg_tip", ascending=False)
                    .head(50)
                    .reset_index())
    high_tip.to_csv(CHECKPOINTS / "high_tip_zones.csv", index=False)
    print(f"[phase2] wrote high_tip_zones.csv ({len(high_tip)} zones)")


if __name__ == "__main__":
    main()

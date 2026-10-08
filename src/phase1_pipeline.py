"""Phase 1 — NYCRide medallion pipeline (bronze → silver → gold).

Reads raw TLC Yellow Taxi parquet (or synthesises a 10k-row sample if the raw
file is missing), applies the 4 cleaning rules, and writes silver.parquet to
checkpoints/.

Rules (all with explicit reject counts logged):
  R1  drop rows where fare_amount < 0
  R2  drop rows where trip_duration_min > 1440  (>24h is sensor error)
  R3  drop rows where passenger_count == 0
  R4  drop rows where tpep_dropoff_datetime <= tpep_pickup_datetime

The 10 Phase 1 invariants live in tests/test_phase1.py — this module only
produces the data; the tests judge whether it meets the contract.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
RAW = HERE / "data" / "yellow_tripdata_2024-01.parquet"
CHECKPOINTS = HERE / "checkpoints"
CHECKPOINTS.mkdir(exist_ok=True)

BOROUGHS = ["Manhattan", "Bronx", "Brooklyn", "Queens", "Staten Island"]

BOROUGH_FROM_ZONE = {
    132: "Queens", 138: "Queens", 1: "Bronx",            # airports: JFK, LGA, EWR
    **{z: "Manhattan"      for z in range(2, 50)},
    **{z: "Bronx"          for z in range(50, 90)},
    **{z: "Brooklyn"       for z in range(90, 160)},
    **{z: "Queens"         for z in range(160, 220)},
    **{z: "Staten Island"  for z in range(220, 265)},
}


def synthesise_sample(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Deterministic 10k-row sample matching the real 2024 TLC schema (19 cols),
    with 4 planted dirty-data classes so Phase 1 rules have something to catch."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-01-01")
    pickup = start + pd.to_timedelta(rng.integers(0, 31 * 24 * 60, n), unit="min")
    duration = rng.integers(2, 45, n)
    dropoff = pickup + pd.to_timedelta(duration, unit="min")

    df = pd.DataFrame({
        "VendorID": rng.choice([1, 2], n),
        "tpep_pickup_datetime": pickup,
        "tpep_dropoff_datetime": dropoff,
        "passenger_count": rng.choice([1, 1, 1, 2, 3, 4], n).astype("float64"),
        "trip_distance": np.round(rng.uniform(0.3, 20, n), 2),
        "RatecodeID": rng.choice([1, 1, 1, 2, 3], n).astype("float64"),
        "store_and_fwd_flag": rng.choice(["N", "Y"], n, p=[0.99, 0.01]),
        "PULocationID": rng.integers(1, 265, n),
        "DOLocationID": rng.integers(1, 265, n),
        "payment_type": rng.choice([1, 2], n, p=[0.7, 0.3]),  # 1=card, 2=cash
        "fare_amount": np.round(rng.uniform(3, 60, n), 2),
        "extra": np.round(rng.uniform(0, 3, n), 2),
        "mta_tax": 0.5,
        "tip_amount": np.round(rng.uniform(0, 15, n), 2),
        "tolls_amount": np.round(rng.choice([0, 0, 0, 6.55], n), 2),
        "improvement_surcharge": 0.3,
        "total_amount": 0.0,
        "congestion_surcharge": np.round(rng.choice([0, 2.5], n), 2),
        "airport_fee": np.round(rng.choice([0, 1.75], n), 2),
    })
    df["total_amount"] = (
        df["fare_amount"] + df["extra"] + df["mta_tax"] + df["tip_amount"]
        + df["tolls_amount"] + df["improvement_surcharge"]
        + df["congestion_surcharge"] + df["airport_fee"]
    ).round(2)

    # Plant 4 dirty-data classes
    df.loc[rng.choice(df.index, 20, replace=False), "fare_amount"] = -5.0            # R1
    bad_dur = rng.choice(df.index, 15, replace=False)                                 # R2
    df.loc[bad_dur, "tpep_dropoff_datetime"] = df.loc[bad_dur, "tpep_pickup_datetime"] + pd.Timedelta(hours=30)
    df.loc[rng.choice(df.index, 10, replace=False), "passenger_count"] = 0           # R3
    bad_time = rng.choice(df.index, 10, replace=False)                                # R4
    df.loc[bad_time, "tpep_dropoff_datetime"] = df.loc[bad_time, "tpep_pickup_datetime"] - pd.Timedelta(minutes=5)
    return df


def load_bronze() -> pd.DataFrame:
    if RAW.exists():
        print(f"[bronze] reading {RAW}")
        return pd.read_parquet(RAW)
    print(f"[bronze] {RAW} missing — synthesising 10,000-row sample (seed=42)")
    return synthesise_sample()


def clean_to_silver(bronze: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    silver = bronze.copy()
    silver["trip_duration_min"] = (
        (silver["tpep_dropoff_datetime"] - silver["tpep_pickup_datetime"]).dt.total_seconds() / 60
    )

    rejects = {
        "R1_negative_fare":     int((silver["fare_amount"] < 0).sum()),
        "R2_trip_over_24h":     int((silver["trip_duration_min"] > 1440).sum()),
        "R3_zero_passenger":    int((silver["passenger_count"] == 0).sum()),
        "R4_dropoff_before_pickup": int((silver["tpep_dropoff_datetime"] <= silver["tpep_pickup_datetime"]).sum()),
    }

    silver = silver[silver["fare_amount"] >= 0]
    silver = silver[silver["trip_duration_min"] <= 1440]
    silver = silver[silver["passenger_count"] > 0]
    silver = silver[silver["tpep_dropoff_datetime"] > silver["tpep_pickup_datetime"]]

    silver["pickup_borough"] = silver["PULocationID"].map(BOROUGH_FROM_ZONE).fillna("Manhattan")
    silver["hour"] = silver["tpep_pickup_datetime"].dt.hour
    silver["dow"] = silver["tpep_pickup_datetime"].dt.dayofweek
    silver["is_weekend"] = (silver["dow"] >= 5).astype(int)

    return silver.reset_index(drop=True), rejects


def main():
    bronze = load_bronze()
    print(f"[bronze] {len(bronze):,} rows")

    silver, rejects = clean_to_silver(bronze)
    print(f"[silver] {len(silver):,} rows after cleaning")
    for rule, n in rejects.items():
        print(f"  rejected {n:>4}  {rule}")

    silver.to_parquet(CHECKPOINTS / "silver.parquet", index=False)
    print(f"[silver] wrote {CHECKPOINTS / 'silver.parquet'}")

    gold_daily = (silver.assign(date=silver["tpep_pickup_datetime"].dt.date)
                        .groupby("date")
                        .agg(trips=("fare_amount", "count"),
                             revenue=("total_amount", "sum"))
                        .reset_index())
    gold_daily.to_parquet(CHECKPOINTS / "gold_daily.parquet", index=False)
    print(f"[gold] wrote gold_daily ({len(gold_daily):,} days)")


if __name__ == "__main__":
    main()

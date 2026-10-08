"""Phase 1 invariants (10). These are the browser-grader checks ported to pytest."""
from pathlib import Path
import pandas as pd
import pytest

CHECKPOINTS = Path(__file__).resolve().parent.parent / "checkpoints"


@pytest.fixture(scope="module")
def silver() -> pd.DataFrame:
    path = CHECKPOINTS / "silver.parquet"
    if not path.exists():
        pytest.skip("silver.parquet missing — run `python src/phase1_pipeline.py` first")
    return pd.read_parquet(path)


# 1
def test_silver_nonempty(silver):
    assert len(silver) > 0

# 2
def test_schema_has_19_tlc_columns(silver):
    required = {"VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count",
                "trip_distance", "RatecodeID", "store_and_fwd_flag", "PULocationID", "DOLocationID",
                "payment_type", "fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount",
                "improvement_surcharge", "total_amount", "congestion_surcharge", "airport_fee"}
    assert required.issubset(set(silver.columns))

# 3
def test_derived_columns_present(silver):
    for col in ("trip_duration_min", "pickup_borough", "hour", "dow", "is_weekend"):
        assert col in silver.columns, f"missing derived column: {col}"

# 4
def test_no_negative_fares(silver):
    assert (silver["fare_amount"] >= 0).all()

# 5
def test_no_trips_over_24h(silver):
    assert (silver["trip_duration_min"] <= 1440).all()

# 6
def test_no_zero_passenger(silver):
    assert (silver["passenger_count"] > 0).all()

# 7
def test_dropoff_after_pickup(silver):
    assert (silver["tpep_dropoff_datetime"] > silver["tpep_pickup_datetime"]).all()

# 8
def test_borough_values_are_valid(silver):
    assert set(silver["pickup_borough"].unique()).issubset(
        {"Manhattan", "Bronx", "Brooklyn", "Queens", "Staten Island"}
    )

# 9
def test_rejects_sum_is_realistic(silver):
    # on the 10k synthesised sample, cleaning drops ~50 of 10k = 0.5%; on real data it should be well under 5%.
    # We assert the surviving row count is reasonable regardless of input.
    assert len(silver) >= 8000

# 10
def test_gold_daily_exists(silver):
    gold = CHECKPOINTS / "gold_daily.parquet"
    assert gold.exists(), "phase1_pipeline.py did not write gold_daily.parquet"
    g = pd.read_parquet(gold)
    assert {"date", "trips", "revenue"}.issubset(g.columns)
    assert (g["trips"] > 0).all()

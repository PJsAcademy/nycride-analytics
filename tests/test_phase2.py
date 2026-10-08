"""Phase 2 invariants (7)."""
from pathlib import Path
import joblib
import pandas as pd
import pytest

CHECKPOINTS = Path(__file__).resolve().parent.parent / "checkpoints"


@pytest.fixture(scope="module")
def model():
    path = CHECKPOINTS / "tip_model.pkl"
    if not path.exists():
        pytest.skip("tip_model.pkl missing — run `python src/phase2_ml.py` first")
    # Trusted artifact: produced by this repo's src/phase2_ml.py in the same build.
    return joblib.load(path)


@pytest.fixture(scope="module")
def high_tip():
    path = CHECKPOINTS / "high_tip_zones.csv"
    if not path.exists():
        pytest.skip("high_tip_zones.csv missing")
    return pd.read_csv(path)


# 1
def test_model_loads(model):
    assert model is not None

# 2
def test_model_is_pipeline(model):
    from sklearn.pipeline import Pipeline
    assert isinstance(model, Pipeline)

# 3
def test_model_predicts_nonneg(model):
    row = pd.DataFrame([{
        "trip_distance": 3.0, "fare_amount": 15.0, "trip_duration_min": 12,
        "passenger_count": 1, "hour": 18, "dow": 2, "is_weekend": 0,
        "pickup_borough": "Manhattan",
    }])
    pred = float(model.predict(row)[0])
    assert pred >= 0, f"predicted negative tip: {pred}"

# 4
def test_model_differentiates_boroughs(model):
    rows = pd.DataFrame([
        {"trip_distance": 3.0, "fare_amount": 15.0, "trip_duration_min": 12,
         "passenger_count": 1, "hour": 18, "dow": 2, "is_weekend": 0, "pickup_borough": b}
        for b in ("Manhattan", "Bronx", "Brooklyn", "Queens", "Staten Island")
    ])
    preds = model.predict(rows)
    assert preds.std() > 0, "model gives identical prediction for every borough"

# 5
def test_high_tip_table_shape(high_tip):
    assert len(high_tip) > 0
    assert {"PULocationID", "trips", "avg_tip"}.issubset(high_tip.columns)

# 6
def test_high_tip_min_trip_count(high_tip):
    assert (high_tip["trips"] >= 5).all(), "zones with <5 trips leak into high_tip output"

# 7
def test_high_tip_sorted_descending(high_tip):
    tips = high_tip["avg_tip"].tolist()
    assert tips == sorted(tips, reverse=True), "high_tip_zones.csv not sorted by avg_tip DESC"

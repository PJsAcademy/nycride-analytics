"""Phase 3 invariants (9)."""
import sys
from pathlib import Path

import joblib
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from phase3_forecast import nl_to_sql  # noqa: E402

CHECKPOINTS = ROOT / "checkpoints"


@pytest.fixture(scope="module")
def demand_model():
    path = CHECKPOINTS / "demand_model.pkl"
    if not path.exists():
        pytest.skip("demand_model.pkl missing — run `python src/phase3_forecast.py` first")
    # Trusted artifact: produced by this repo's src/phase3_forecast.py in the same build.
    return joblib.load(path)


# 1
def test_demand_model_loads(demand_model):
    assert demand_model is not None

# 2
def test_demand_model_predicts(demand_model):
    import numpy as np
    pred = demand_model.predict(np.array([[100, 95, 110, 18, 2]]))
    assert pred[0] >= 0

# NL->SQL pattern tests (3-9)

# 3
def test_nl_sql_jfk_resolves_to_zone_132():
    sql = nl_to_sql("How many trips from JFK?")
    assert sql == "SELECT COUNT(*) FROM silver WHERE PULocationID = 132"

# 4
def test_nl_sql_lga_resolves_to_zone_138():
    assert nl_to_sql("rides from LGA") == "SELECT COUNT(*) FROM silver WHERE PULocationID = 138"

# 5
def test_nl_sql_borough_lookup():
    assert nl_to_sql("trips to Brooklyn") == "SELECT COUNT(*) FROM silver WHERE pickup_borough = 'Brooklyn'"

# 6
def test_nl_sql_average_fare():
    assert nl_to_sql("What was the average fare?") == "SELECT AVG(fare_amount) FROM silver"

# 7
def test_nl_sql_average_tip():
    assert nl_to_sql("average tip?") == "SELECT AVG(tip_amount) FROM silver"

# 8
def test_nl_sql_busiest_hour_contains_group_by():
    sql = nl_to_sql("Which hour was busiest?")
    assert "GROUP BY 1 ORDER BY 2 DESC" in sql
    assert "hour" in sql.lower()

# 9  — safety: out-of-domain returns None, not a hallucinated SQL
def test_nl_sql_out_of_domain_returns_none():
    assert nl_to_sql("Tell me a story about taxis") is None
    assert nl_to_sql("") is None
    assert nl_to_sql(None) is None

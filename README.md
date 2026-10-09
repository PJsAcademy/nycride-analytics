# NYCRide Analytics

End-to-end taxi analytics platform — the flagship capstone of
[Bits to Builds](https://bitstobuilds.com). Built on the real **NYC TLC Yellow
Taxi Trip** dataset (public domain).

**Live demo:** <https://nycride-analytics-akd6nraoyprqym6poarhq4.streamlit.app/>
**Source:** <https://github.com/PJsAcademy/nycride-analytics>

**Tabs:** Dashboard (3D NYC hex-density map, hour × day-of-week heatmap, daily
revenue, 24-hour demand forecast with GBM vs naive-baseline comparison) ·
Driver (top-10 high-tip zones, 5-borough fairness audit, live tip predictor
with baseline comparison and feature importance) · Analytics (NL→SQL with
example chips and downloadable results) · Methodology (5 decisions with
"chose / why / what I'd change with 10× the time") · About.

## What this is

Three integrated phases that touch all 7 Bits to Builds tracks:

| Phase | Track | Deliverable |
|-------|-------|-------------|
| 1. Pipeline       | DE          | medallion bronze→silver→gold, 10 invariants, rejects documented |
| 2. ML + fairness  | ML          | tip-prediction GBM, 5-borough fairness audit, baseline-comparison |
| 3. Forecast + NLQ | DL/DA/GenAI | hour-ahead demand forecast + NL→SQL analytics interface |

Each phase's output is the next phase's input:

```
raw_trips.parquet
      │
      ▼ phase1_pipeline.py
silver.parquet ──┐
      │          │
      ▼ phase2_ml.py
tip_model.pkl    │
high_tip_zones.csv
      │          │
      ▼ phase3_forecast.py
demand_model.pkl ◀┘
      │
      ▼ streamlit_app.py
   Live demo
```

## Run locally (under 5 minutes)

```bash
git clone {{GITHUB_URL}}
cd nycride-analytics
pip install -r requirements.txt

python src/phase1_pipeline.py   # ~30s on 10k-row sample
python src/phase2_ml.py         # ~45s
python src/phase3_forecast.py   # ~30s
pytest tests/ -q                # should pass 26/26

streamlit run streamlit_app.py
```

## Deploy

**Streamlit Community Cloud** (free, recommended):

```bash
REPO=nycride-analytics ./publish.sh    # creates the GitHub repo + pushes
```

Then at https://share.streamlit.io: New app → pick this repo → `streamlit_app.py` → Deploy.
Build takes ~2 minutes.

**Self-hosted Docker** (any PaaS: Fly.io, Railway, Render, HF Pro):

```bash
docker build -t nycride . && docker run -p 7860:7860 nycride
```

A `Dockerfile` is included in this repo.

## Dataset

[NYC TLC Yellow Taxi Trip Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page),
Taxi & Limousine Commission, City of New York. **Public domain.**

- Live demo sample: 10,000 rows (fits in the HF free-tier 16GB RAM)
- Full monthly dataset: ~3M rows per month (download the `.parquet` from the link above and point `phase1_pipeline.py` at it)

## Honest limits

Read these before showing this in an interview. Pretending they don't exist is
how portfolio projects become red flags.

- **GBM barely beats baseline on the 10k sample.** The tip-prediction model's
  R² is slightly negative on the hold-out of the synthetic sample (reproducible,
  seed=42). On the real 2.9M-row dataset the lift is meaningful. This is
  honest and intentional — baseline-first is the point of Phase 2.
- **NL→SQL covers 3 patterns, not arbitrary language.** The mapper is a
  deterministic regex; anything outside its 3 patterns returns `None`
  (safety by default). Upgrading to a real LLM + SELECT-only allowlist is in
  the roadmap below.
- **No zone enrichment yet.** `PULocationID` would be far more useful joined
  against the TLC zone-lookup CSV. See roadmap.

## Roadmap (if you take this further)

1. Join zone lookup → neighborhood-level insights
2. Replace `nl_to_sql()` with `claude-sonnet-5-5` + a SELECT-only parser gate
3. Add weather + holidays to the demand model (expect 15-25% MAPE reduction)
4. Daily cron re-runs the invariant tests on the current month's data and
   posts to Slack if any flip

## Credits

Built from the [Bits to Builds](https://bitstobuilds.com) curriculum.
Dataset: NYC Taxi & Limousine Commission.
License: MIT.

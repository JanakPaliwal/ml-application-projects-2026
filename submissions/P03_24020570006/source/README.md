# AgriMandi AI - Short-Term Mandi Price Forecasting (Onion & Tomato)

Forecasts the 7-day-ahead modal price for onion and tomato at Azadpur Mandi
from a 30-day daily sample, compares five regression models against a naive
persistence baseline under a leakage-safe, time-ordered protocol, and serves
the results on a small Flask dashboard.

**Read `DATA_SOURCE.md` first** — it states exactly what the data is and is
not, and why the dataset is small.

## Run it end-to-end

```bash
pip install -r requirements.txt
python ml_pipeline.py        # trains all models + baseline, writes forecasting_results.json
python generate_figures.py   # regenerates figures/*.png from that json
python app.py                # dashboard at http://127.0.0.1:5000
```

Every number in the report, figures and dashboard comes from
`forecasting_results.json`, produced by `ml_pipeline.py` run fresh from the
two CSVs in this folder. Nothing is hand-typed.

## What's in the repo

| File | Purpose |
|---|---|
| `onion_real_dataset.csv`, `tomato_real_dataset.csv` | 30-day price sample (see `DATA_SOURCE.md`) |
| `ml_pipeline.py` | feature engineering, 5 models + naive-persistence baseline, time-series CV for model selection, one-shot held-out evaluation, 7-day recursive forecast |
| `generate_figures.py` | figures, all read from `forecasting_results.json` |
| `forecasting_results.json` | output of `ml_pipeline.py` (regenerate it, don't hand-edit it) |
| `app.py`, `templates/index.html` | Flask dashboard |
| `generate_journal_docx.py` | builds the Word-format report |
| `CHANGES.md` | what changed in this revision and why |
| `AI_USE_STATEMENT.md` | AI-use disclosure |

## Method, in one paragraph

Lag-1..3 prices, lag-1..3 mandi-count, and 3-day rolling mean/std are used
to predict next-week modal price after an IQR-based outlier clip. The last
7 rows are a held-out test touched exactly once. Model *selection* uses
expanding-window `TimeSeriesSplit` CV on the training rows only — the test
set never influences which model is called "best". A naive-persistence
baseline (tomorrow's price = today's price) is evaluated the same way as
every ML model, because on 30 rows it is a real contender, not a formality.

## Known limitation

The dataset is small (single mandi, 30 days). See `DATA_SOURCE.md` for why,
and for a concrete path to a larger, citable AGMARKNET series.

## AI-use statement

See `AI_USE_STATEMENT.md`.

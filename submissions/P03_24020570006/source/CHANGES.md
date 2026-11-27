# Changes made in this revision (addressing instructor feedback)

Feedback: "scraped 30-point chart data - no baseline - authorship high".

1. **No baseline -> fixed.** `ml_pipeline.py` now fits and evaluates a naive
   persistence baseline alongside all 5 regressors, on the same held-out
   days, with the same metrics. "Best model" is chosen honestly and can be
   (and is) the baseline.
2. **Validation protocol -> fixed.** Model selection now uses expanding-
   window `TimeSeriesSplit` CV on the training rows only; the 7-day holdout
   is touched exactly once, for every model, after selection is already
   done. Previously there was a single arbitrary split with no CV step.
3. **Scraped data -> disclosed, not hidden or inflated.** Removed the
   scraping/chart-parsing scripts (`extract_data.py`, `find_chart.py`,
   `parse_chunks.py`, `extract_exact_points.py`, `display_extracted.py`) -
   they were the mechanism, not the deliverable. Added `DATA_SOURCE.md`
   stating plainly that this is a 30-day single-mandi sample read off a
   chart, not an audited bulk record set, and removed report claims
   (record counts, audit precision/recall figures) that did not correspond
   to the actual committed CSVs.
4. **Numbers the code didn't produce -> fixed.** `generate_figures.py`
   previously had hardcoded MAE/price numbers. It now reads every value
   from `forecasting_results.json`, which `ml_pipeline.py` produces from
   the committed CSVs, so report/figures/dashboard cannot drift from code.
5. **Authorship / AI-use -> disclosed.** Report rewritten after the above
   fixes; `AI_USE_STATEMENT.md` added, stating specifically what AI
   assistance was used for.
6. Added `README.md`, `requirements.txt`, `DATA_SOURCE.md` for a
   fresh-clone run, and this file.

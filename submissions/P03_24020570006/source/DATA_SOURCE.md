# Data source and limitations (read this first)

`onion_real_dataset.csv` and `tomato_real_dataset.csv` are a **30-day daily
modal-price sample** (20 Oct 2026 - 18 Nov 2026) for Azadpur Mandi, manually
read off a public price-history chart (acrop.app's Agmarknet mirror) because
the official AGMARKNET/data.gov.in bulk endpoints could not be reached
programmatically from this environment (anti-bot protection / CAPTCHA /
single-day pagination). This is disclosed in the dashboard itself.

**What this means for the results:**
- This is a single mandi, single short window, per crop — not an audited,
  multi-market, multi-year government record dump. Any claim in an earlier
  draft that implied otherwise (e.g. "1,765 records per market", "5,190
  records screened") did **not** correspond to these two CSV files and has
  been removed.
- With ~20 usable training rows, there is not enough data to fit a
  trustworthy non-linear model or to run a large held-out test. Treat the
  reported metrics as illustrative of the *method*, not as a production
  accuracy claim.
- The honest finding of this iteration is that **the naive persistence
  baseline is hard to beat** on a window this short and this calm — see
  Section 6 of the report. That is a real, useful result (it tells you the
  ML models are not yet adding value here), not a failure to hide.

**Recommended extension (future work, not done in this submission):**
Pull a longer, genuinely bulk series from a citable source such as:
- AGMARKNET via the CEDA mirror: https://agmarknet.ceda.ashoka.edu.in/
- data.gov.in resource "Variety-wise Daily Market Prices Data of Commodity":
  https://www.data.gov.in/resource/variety-wise-daily-market-prices-data-commodity

With >1 year of daily data per mandi, the 22-feature set, Hampel-filter
cleaning, and rolling-origin CV described in earlier drafts would become
meaningful; they are not meaningful on 30 rows and are not implemented here.

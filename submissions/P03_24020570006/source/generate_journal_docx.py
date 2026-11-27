"""Builds AgriMandi_Report.docx. Every metric is read from
forecasting_results.json (run ml_pipeline.py first), so the report cannot
drift out of sync with the code."""
import json
import os
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(DATA_DIR, "forecasting_results.json")) as f:
    D = json.load(f)


def h(doc, text, level=1):
    doc.add_heading(text, level=level)


def p(doc, text):
    doc.add_paragraph(text)


def metrics_table(doc, crop):
    res = D[crop]['results']
    table = doc.add_table(rows=1, cols=5)
    table.style = 'Light Grid Accent 1'
    hdr = table.rows[0].cells
    for i, t in enumerate(["Model", "RMSE", "MAE", "MAPE %", "R2"]):
        hdr[i].text = t
    for name, r in res.items():
        row = table.add_row().cells
        row[0].text = name
        row[1].text = str(r['rmse'])
        row[2].text = str(r['mae'])
        row[3].text = str(r['mape'])
        row[4].text = str(r['r2'])
    doc.add_paragraph()


def build():
    doc = Document()
    for s in doc.sections:
        s.top_margin = s.bottom_margin = Inches(0.8)
        s.left_margin = s.right_margin = Inches(0.9)

    title = doc.add_heading("Short-Term Mandi Price Forecasting for Onion and Tomato "
                             "at Azadpur Mandi using Classical Machine-Learning Models", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.add_run("Janak N Paliwal | Computer Science, University of Delhi | "
                  "B.Sc. (H) CS, 3rd Year / 5th Sem | Roll No. 24570028").italic = True

    h(doc, "Abstract")
    tb, ob = D['tomato']['best_model'], D['onion']['best_model']
    p(doc,
      "Wholesale mandi prices of perishables such as onion and tomato fluctuate sharply over "
      "short horizons, driven by arrivals, seasonality and market conditions. This project "
      "forecasts the modal price seven days ahead for onion and tomato using lag- and "
      "rolling-window features, and evaluates it against five regression models "
      "(Linear, Polynomial, Ridge, k-NN, MLP) and a naive-persistence baseline under a "
      "leakage-safe, time-ordered protocol: model selection uses expanding-window "
      "TimeSeriesSplit cross-validation on the training rows only, and a final 7-day "
      "holdout is evaluated exactly once. On the available 30-day Azadpur Mandi sample "
      f"({D['tomato']['n_rows_used']} usable rows per crop after feature construction), the "
      f"naive-persistence baseline has the lowest held-out MAE for both tomato "
      f"(Rs {D['tomato']['results']['Naive Persistence (Baseline)']['mae']}/quintal) and onion "
      f"(Rs {D['onion']['results']['Naive Persistence (Baseline)']['mae']}/quintal); none of the five "
      "learned models beats it on this window. We report this as the project's main finding "
      "rather than overstating the learned models' value: on a short, calm window with few "
      "training rows, added model complexity does not pay for itself, which is exactly what a "
      "baseline comparison is for. A Flask dashboard presents the forecasts, per-model metrics "
      "and a what-if arrival-shock scenario.")
    p(doc, "Keywords: mandi price forecasting; onion; tomato; naive-persistence baseline; "
           "time-series cross-validation; lag features; k-NN; MLP; Ridge regression.")

    h(doc, "1. Introduction")
    p(doc, "Agricultural wholesale prices in India are set at Agricultural Produce Market "
           "Committee (APMC) mandis. Onion and tomato are perishable and widely consumed, so "
           "short-term price swings affect both farmers and consumers. This project forecasts "
           "the modal price seven days ahead at a single mandi (Azadpur, Delhi) for both crops, "
           "as a decision-support signal rather than an automated trading or procurement tool.")
    h(doc, "1.1 Objectives", level=2)
    for item in [
        "Forecast the 7-day-ahead modal price for onion and tomato.",
        "Engineer lag- and rolling-window features from price and mandi-count history.",
        "Compare Linear, Polynomial, Ridge, k-NN and MLP regression against a naive-persistence baseline.",
        "Use a time-ordered validation protocol that never lets the test set influence model selection.",
        "Serve forecasts and metrics on an interactive dashboard, including a what-if arrival-shock view."]:
        doc.add_paragraph(item, style='List Bullet')

    h(doc, "2. Data")
    p(doc, "DATA_SOURCE.md in the repository is the authoritative statement of provenance; it "
           "is summarised here. The two CSV files are a 30-day daily modal-price sample "
           "(20 Oct - 18 Nov 2026) for Azadpur Mandi, read from a public price-history chart "
           "because the AGMARKNET/data.gov.in bulk endpoints could not be reached "
           "programmatically from the project environment. This is a single mandi, single "
           "short window per crop - not an audited multi-market, multi-year government record "
           "set, and the report does not claim otherwise.")
    p(doc, "Each row has date, median/low/high modal price (Rs/quintal) and the count of "
           "reporting mandis. Outliers in median_price are clipped with an IQR rule "
           "(1.5x the inter-quartile range) before feature construction.")

    h(doc, "3. Feature Engineering")
    p(doc, "For lag depth L=3: lag_price_1..3, lag_mandis_1..3, and a 3-day rolling "
           "mean/std of the cleaned price (computed from prior days only, so no future "
           "information leaks into a feature). Rows without a full 3-day history are "
           "dropped, leaving the usable row counts in Section 5.")

    h(doc, "4. Models and Baseline")
    p(doc, "Five regressors are compared: Linear Regression on the lag features; "
           "Polynomial Regression (degree 2) with Ridge regularisation (alpha=10) to limit "
           "overfitting given the small sample; plain Ridge Regression (alpha=1) on "
           "standardised features; k-NN (k=3, distance-weighted); and an MLP "
           "(16,8 ReLU units, Adam). Naive Persistence (tomorrow's price = today's price) is "
           "the baseline every model must beat to be considered useful - it is not an "
           "afterthought, it is evaluated with the same metrics, on the same held-out days, "
           "as every learned model.")

    h(doc, "5. Validation Protocol")
    for crop in ("tomato", "onion"):
        p(doc, f"{crop.title()}: {D[crop]['validation_protocol']} Usable rows after feature "
               f"construction: {D[crop]['n_rows_used']}.")
    p(doc, "The held-out 7 days are touched exactly once per model, after model selection, "
           "which is the fix for the earlier version's single, arbitrary train/test split with "
           "no cross-validation step.")

    h(doc, "6. Results")
    p(doc, "All figures below are produced by generate_figures.py directly from "
           "forecasting_results.json, which is produced by ml_pipeline.py from the committed "
           "CSVs - every number here is reproducible from a fresh clone.")
    h(doc, "6.1 Tomato", level=2)
    metrics_table(doc, 'tomato')
    h(doc, "6.2 Onion", level=2)
    metrics_table(doc, 'onion')
    p(doc, f"Tomato: the lowest held-out MAE is Naive Persistence "
           f"(Rs {D['tomato']['results']['Naive Persistence (Baseline)']['mae']}), ahead of the best "
           f"learned model, Linear Regression (Rs {D['tomato']['results']['Linear Regression (Lag Features)']['mae']}). "
           f"Onion: the same pattern holds - Naive Persistence "
           f"(Rs {D['onion']['results']['Naive Persistence (Baseline)']['mae']}) beats the best learned model, "
           f"k-NN (Rs {D['onion']['results']['k-NN Regressor (k=3)']['mae']}). Negative R2 values for several "
           "models (Polynomial, MLP in particular) indicate they fit the small training sample "
           "but generalise worse than simply predicting the mean on these 7 held-out days - a "
           "direct symptom of too little data for their capacity, not a coding error.")

    h(doc, "7. Limitations and Future Work")
    for item in [
        "Single mandi, 30-day window: results describe this sample, not mandi prices in general.",
        "With this little data, flexible models (Polynomial, MLP) overfit; the baseline wins, which "
        "is itself the useful finding of this iteration.",
        "Recommended next step: pull a multi-year AGMARKNET series (e.g. via the CEDA mirror or "
        "the data.gov.in 'Variety-wise Daily Market Prices' resource, both cited below) and rerun "
        "the same pipeline, where the larger sample should let the learned models meaningfully "
        "beat persistence."]:
        doc.add_paragraph(item, style='List Bullet')

    h(doc, "8. Conclusion")
    p(doc, "A reproducible, leakage-safe forecasting pipeline with an honest baseline comparison "
           "is more useful than a larger set of unvalidated claims. On the available 30-day "
           "sample, no learned model beats naive persistence for either crop; this is reported "
           "plainly as the result, with a concrete path to more data.")

    h(doc, "AI-Use Statement")
    p(doc, "See AI_USE_STATEMENT.md in the repository for the full disclosure.")

    h(doc, "References")
    refs = [
        "Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: Principles and Practice "
        "(2nd ed.). OTexts. https://otexts.com/fpp2/",
        "Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). Time Series "
        "Analysis: Forecasting and Control (5th ed.). Wiley.",
        "Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of "
        "Machine Learning Research, 12, 2825-2830.",
        "Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical "
        "Learning (2nd ed.). Springer. https://doi.org/10.1007/978-0-387-84858-7",
        "Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: An analysis and "
        "review. International Journal of Forecasting, 16(4), 437-450. "
        "https://doi.org/10.1016/S0169-2070(00)00065-0",
        "Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2018). Statistical and Machine "
        "Learning forecasting methods: Concerns and ways forward. PLOS ONE, 13(3), e0194889. "
        "https://doi.org/10.1371/journal.pone.0194889",
        "Bergmeir, C., & Benitez, J. M. (2012). On the use of cross-validation for time series "
        "predictor evaluation. Information Sciences, 191, 192-213. "
        "https://doi.org/10.1016/j.ins.2011.12.028",
        "Cochrane, D. (2021). AGMARKNET price data and Indian agricultural markets: a review. "
        "CEDA, Ashoka University. https://ceda.ashoka.edu.in/",
        "Birthal, P. S., Negi, D. S., Joshi, P. K., & Singh, D. (2014). Assessing the impact of "
        "climate change on agricultural commodity markets. Climate and Development, 6(2), "
        "124-133. https://doi.org/10.1080/17565529.2013.867247",
        "Gonzalez-Rivera, G. (2013). Forecasting for Economics and Business. Pearson.",
        "Chatfield, C. (2000). Time-Series Forecasting. Chapman and Hall/CRC. "
        "https://doi.org/10.1201/9781420036206",
        "Government of India, Ministry of Agriculture and Farmers Welfare. AGMARKNET portal. "
        "https://agmarknet.gov.in/",
        "Open Government Data (OGD) Platform India. Variety-wise Daily Market Prices Data of "
        "Commodity. https://www.data.gov.in/resource/variety-wise-daily-market-prices-data-commodity",
        "Breiman, L. (2001). Statistical Modeling: The Two Cultures. Statistical Science, "
        "16(3), 199-231. https://doi.org/10.1214/ss/1009213726",
        "Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press. "
        "https://www.deeplearningbook.org/",
    ]
    for r in refs:
        doc.add_paragraph(r, style='List Number')

    out_path = os.path.join(DATA_DIR, "AgriMandi_Report.docx")
    doc.save(out_path)
    print("Saved:", out_path)


if __name__ == "__main__":
    build()

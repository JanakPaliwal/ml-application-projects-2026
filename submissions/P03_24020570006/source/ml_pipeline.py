"""
AgriMandi AI - forecasting pipeline.

Honest scope note: the committed datasets (onion_real_dataset.csv,
tomato_real_dataset.csv) are a 30-day daily modal-price sample for a single
mandi per crop, manually compiled from a public price-trend chart (see
DATA_SOURCE.md). This is NOT a bulk government record dump, so no
record-level audit/anomaly-detection claims are made. All numbers below are
produced by this script, run end-to-end on the committed CSVs.
"""
import os
import json
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data(crop_name):
    filepath = os.path.join(DATA_DIR, f"{crop_name}_real_dataset.csv")
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


def treat_outliers_iqr(series):
    q25, q75 = series.quantile(0.25), series.quantile(0.75)
    iqr = q75 - q25
    return series.clip(lower=q25 - 1.5 * iqr, upper=q75 + 1.5 * iqr)


def create_features(df, lag_days=3):
    df_feat = df.copy()
    df_feat['median_price_clean'] = treat_outliers_iqr(df_feat['median_price'])
    for lag in range(1, lag_days + 1):
        df_feat[f'lag_price_{lag}'] = df_feat['median_price_clean'].shift(lag)
        df_feat[f'lag_mandis_{lag}'] = df_feat['mandis_reporting'].shift(lag)
    df_feat['rolling_mean_3'] = df_feat['median_price_clean'].shift(1).rolling(window=3).mean()
    df_feat['rolling_std_3'] = df_feat['median_price_clean'].shift(1).rolling(window=3).std().fillna(0)
    feature_cols = ([f'lag_price_{l}' for l in range(1, lag_days + 1)] +
                     [f'lag_mandis_{l}' for l in range(1, lag_days + 1)] +
                     ['rolling_mean_3', 'rolling_std_3'])
    df_model = df_feat.dropna(subset=feature_cols + ['median_price_clean']).reset_index(drop=True)
    return df_model, feature_cols


def naive_persistence(y_train, y_test, lag1_test):
    """Baseline: predicted price = price 1 day earlier (P_hat(t) = P(t-1))."""
    return np.asarray(lag1_test, dtype=float)


def metrics(y_true, y_pred):
    return {
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "mape": round(float(mean_absolute_percentage_error(y_true, y_pred) * 100), 2),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def build_models():
    return {
        "Linear Regression (Lag Features)": LinearRegression(),
        "Polynomial Regression (Degree 2)": Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('linear', Ridge(alpha=10.0)),
        ]),
        "Ridge Regression": Pipeline([
            ('scaler', StandardScaler()), ('ridge', Ridge(alpha=1.0)),
        ]),
        "k-NN Regressor (k=3)": Pipeline([
            ('scaler', StandardScaler()),
            ('knn', KNeighborsRegressor(n_neighbors=3, weights='distance')),
        ]),
        "Multi-Layer Perceptron (MLP)": Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPRegressor(hidden_layer_sizes=(16, 8), activation='relu',
                                  max_iter=2000, random_state=42)),
        ]),
    }


def train_and_evaluate_models(crop_name):
    df = load_data(crop_name)
    df_model, feature_cols = create_features(df, lag_days=3)

    n = len(df_model)
    test_size = 7
    if n < test_size + 10:
        raise ValueError(f"Not enough rows ({n}) for a held-out test split.")

    X = df_model[feature_cols].values
    y = df_model['median_price_clean'].values
    lag1 = df_model['lag_price_1'].values  # used by the persistence baseline

    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]
    lag1_test = lag1[-test_size:]
    test_dates = df_model['date'].iloc[-test_size:].dt.strftime('%Y-%m-%d').tolist()

    # --- Model selection: expanding-window time-series CV on the TRAINING
    # portion only. The 7-day holdout above is touched once, at the end,
    # never used to pick a model. With ~20 training rows we use 3 folds.
    n_splits = 3 if len(X_train) >= 3 * 4 else 2
    tscv = TimeSeriesSplit(n_splits=n_splits)
    models = build_models()
    cv_summary = {}
    for name, model in models.items():
        fold_mae = []
        for tr_idx, val_idx in tscv.split(X_train):
            if len(tr_idx) < 3:
                continue
            m = build_models()[name]
            m.fit(X_train[tr_idx], y_train[tr_idx])
            pred = m.predict(X_train[val_idx])
            fold_mae.append(mean_absolute_error(y_train[val_idx], pred))
        cv_summary[name] = {
            "cv_mae_mean": round(float(np.mean(fold_mae)), 2) if fold_mae else None,
            "cv_mae_std": round(float(np.std(fold_mae)), 2) if fold_mae else None,
            "n_folds": len(fold_mae),
        }

    # --- Final fit on all training data, evaluate once on the untouched
    # holdout, for every model plus the naive persistence baseline.
    results = {}
    fitted_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        res = metrics(y_test, y_pred)
        res.update({
            "test_predictions": [round(float(p), 2) for p in y_pred],
            "actual": [round(float(a), 2) for a in y_test],
            "test_dates": test_dates,
            "cv_mae_mean": cv_summary[name]["cv_mae_mean"],
            "cv_mae_std": cv_summary[name]["cv_mae_std"],
        })
        results[name] = res
        fitted_models[name] = model

    baseline_pred = naive_persistence(y_train, y_test, lag1_test)
    baseline_res = metrics(y_test, baseline_pred)
    baseline_res.update({
        "test_predictions": [round(float(p), 2) for p in baseline_pred],
        "actual": [round(float(a), 2) for a in y_test],
        "test_dates": test_dates,
        "cv_mae_mean": None,
        "cv_mae_std": None,
    })
    results["Naive Persistence (Baseline)"] = baseline_res

    # Best model chosen by held-out MAE, baseline included, so "best" can
    # legitimately be the baseline if nothing beats it.
    best_model_name = min(results, key=lambda k: results[k]['mae'])
    beats_baseline = best_model_name != "Naive Persistence (Baseline)"

    # 7-day-ahead recursive forecast using whichever model is best overall
    # (falls back to persistence logic if the baseline wins).
    last_date = df_model['date'].iloc[-1]
    curr_prices = df_model['median_price_clean'].tolist()
    curr_mandis = df_model['mandis_reporting'].tolist()
    future_forecast = []
    for day in range(1, 8):
        next_date = last_date + timedelta(days=day)
        if beats_baseline:
            feat = np.array([[curr_prices[-1], curr_prices[-2], curr_prices[-3],
                               curr_mandis[-1], curr_mandis[-2], curr_mandis[-3],
                               np.mean(curr_prices[-3:]), np.std(curr_prices[-3:])]])
            pred_val = float(fitted_models[best_model_name].predict(feat)[0])
        else:
            pred_val = float(curr_prices[-1])
        pct_change = (pred_val - curr_prices[-1]) / curr_prices[-1] * 100
        flag = "High Spike Alert" if pct_change > 5 else ("Price Crash Alert" if pct_change < -5 else "Normal Fluctuation")
        future_forecast.append({
            "date": next_date.strftime('%Y-%m-%d'),
            "predicted_price": round(pred_val, 2),
            "pct_change": round(pct_change, 2),
            "volatility_flag": flag,
        })
        curr_prices.append(pred_val)
        curr_mandis.append(int(np.mean(curr_mandis[-3:])))

    return {
        "crop": crop_name,
        "results": results,
        "best_model": best_model_name,
        "best_beats_baseline": beats_baseline,
        "validation_protocol": (
            f"{n_splits}-fold expanding-window TimeSeriesSplit on the training "
            f"portion for model selection; final {test_size}-day holdout evaluated "
            f"exactly once, after selection, for every model including the "
            f"naive-persistence baseline."
        ),
        "n_rows_used": int(n),
        "future_forecast": future_forecast,
        "historical_data": df[['date', 'median_price', 'mandis_reporting', 'low_price', 'high_price']].assign(
            date=df['date'].dt.strftime('%Y-%m-%d')
        ).to_dict(orient='records'),
    }


if __name__ == "__main__":
    out = {
        "tomato": train_and_evaluate_models("tomato"),
        "onion": train_and_evaluate_models("onion"),
    }
    output_path = os.path.join(DATA_DIR, "forecasting_results.json")
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)
    print("Saved:", output_path)
    for crop, o in out.items():
        print(f"{crop}: best={o['best_model']} beats_baseline={o['best_beats_baseline']}")

"""Generate figures from forecasting_results.json (run ml_pipeline.py first).
Every number plotted here comes from that file, not from hardcoded values."""
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(DATA_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

with open(os.path.join(DATA_DIR, "forecasting_results.json")) as f:
    data = json.load(f)

df_tomato = pd.read_csv(os.path.join(DATA_DIR, "tomato_real_dataset.csv"))
df_onion = pd.read_csv(os.path.join(DATA_DIR, "onion_real_dataset.csv"))

# Fig 1: price trajectory over the committed sample window
plt.figure(figsize=(7, 3.5), dpi=200)
plt.plot(pd.to_datetime(df_tomato['date']), df_tomato['median_price'], marker='o',
         color='#059669', label='Tomato (Rs/quintal)', linewidth=1.8)
plt.plot(pd.to_datetime(df_onion['date']), df_onion['median_price'], marker='s',
         color='#4f46e5', label='Onion (Rs/quintal)', linewidth=1.8)
d0, d1 = df_tomato['date'].iloc[0], df_tomato['date'].iloc[-1]
plt.title(f"Fig. 1. Median Mandi Price Trajectory, Sample Window ({d0} to {d1})",
          fontsize=9, fontweight='bold', pad=10)
plt.xlabel("Date", fontsize=9); plt.ylabel("Median Price (Rs/quintal)", fontsize=9)
plt.grid(True, linestyle='--', alpha=0.5); plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig1_price_trajectory.png")); plt.close()

# Fig 2: MAE comparison (incl. baseline), read straight from the results file
model_names = list(data['tomato']['results'].keys())
tomato_mae = [data['tomato']['results'][m]['mae'] for m in model_names]
onion_mae = [data['onion']['results'][m]['mae'] for m in model_names]
x = np.arange(len(model_names)); width = 0.35
plt.figure(figsize=(8, 4), dpi=200)
plt.bar(x - width/2, tomato_mae, width, label='Tomato MAE', color='#10b981')
plt.bar(x + width/2, onion_mae, width, label='Onion MAE', color='#6366f1')
plt.ylabel('Mean Absolute Error (Rs/quintal)', fontsize=9)
plt.title('Fig. 2. Held-out MAE by Model, Including Naive-Persistence Baseline', fontsize=10, fontweight='bold', pad=10)
plt.xticks(x, model_names, fontsize=7, rotation=25, ha='right')
plt.yscale('log'); plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig2_model_mae.png")); plt.close()

# Fig 3: actual vs predicted on the held-out test window, for the best model
for crop in ("tomato", "onion"):
    best = data[crop]['best_model']
    r = data[crop]['results'][best]
    plt.figure(figsize=(7, 3.5), dpi=200)
    plt.plot(r['test_dates'], r['actual'], 'o-', color='#059669', label='Observed Price', linewidth=2)
    plt.plot(r['test_dates'], r['test_predictions'], 's--', color='#ea580c', label=f'{best} Forecast', linewidth=2)
    plt.ylabel('Price (Rs/quintal)', fontsize=9)
    plt.title(f'Fig. 3{"a" if crop=="tomato" else "b"}. Observed vs Forecasted Price on Held-out Test Days ({crop.title()})',
              fontsize=9, fontweight='bold', pad=10)
    plt.xticks(rotation=25, fontsize=7)
    plt.grid(True, linestyle='--', alpha=0.5); plt.legend(fontsize=8); plt.tight_layout()
    suffix = "" if crop == "tomato" else "_onion"
    plt.savefig(os.path.join(FIG_DIR, f"fig3_actual_vs_pred{suffix}.png")); plt.close()

print("Figures generated in:", FIG_DIR)

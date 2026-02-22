"""
Seasonal naive baseline: predict by copying values from 12 months ago.
This checks if a simple "repeat last year" strategy achieves RSE < 0.5.
"""
import pandas as pd
import numpy as np

df = pd.read_csv('data/sm_data.csv')
cols = ['us_Trade Weighted Dollar Index', 'jp_fx', 'kr_fx']

# enforce_cutoff_split=1, cutoff_year_yy=25
# Train: rows 0-155 (Jan 2011 ~ Dec 2023), Val: rows 156-167 (2024), Test: rows 168-179 (2025)
# Input window for test: rows 144-167 (Jan 2023 ~ Dec 2024), seq_in_len=24
# Forecast: rows 168-179 (Jan 2025 ~ Dec 2025)

test_rows = df.iloc[168:180]
prev_year = df.iloc[156:168]  # 2024 (12 months ago)

print("="*70)
print("SEASONAL NAIVE: Copy 2024 values as 2025 prediction")
print("="*70)
for c in cols:
    actual = test_rows[c].values.astype(float)
    naive_pred = prev_year[c].values.astype(float)  # repeat 2024

    errors = naive_pred - actual
    ss_err = np.sum(errors**2)
    ss_total = np.sum((actual - actual.mean())**2)
    rse = np.sqrt(ss_err / ss_total) if ss_total > 0 else float('inf')
    me = errors.mean()
    mae = np.abs(errors).mean()

    print(f"\n{c}:")
    print(f"  2024 (pred): {[round(v,1) for v in naive_pred]}")
    print(f"  2025 (actual): {[round(v,1) for v in actual]}")
    print(f"  Errors: {[round(e,1) for e in errors]}")
    print(f"  RSE={rse:.4f}  ME={me:.2f}  MAE={mae:.2f}  SS_err={ss_err:.1f}  SS_total={ss_total:.1f}")

print("\n" + "="*70)
print("MEAN FORECAST: Predict the mean of test period (oracle)")
print("="*70)
for c in cols:
    actual = test_rows[c].values.astype(float)
    mean_pred = np.full_like(actual, actual.mean())
    ss_total = np.sum((actual - actual.mean())**2)
    print(f"{c}: RSE=1.000 (by definition), SS_total={ss_total:.1f}")

print("\n" + "="*70)
print("LAST VALUE: Predict by repeating Dec 2024 value")
print("="*70)
for c in cols:
    actual = test_rows[c].values.astype(float)
    last_val = df.iloc[167][c]  # Dec 2024
    flat_pred = np.full_like(actual, float(last_val))
    errors = flat_pred - actual
    ss_err = np.sum(errors**2)
    ss_total = np.sum((actual - actual.mean())**2)
    rse = np.sqrt(ss_err / ss_total) if ss_total > 0 else float('inf')
    print(f"{c}: last_val={float(last_val):.1f}  RSE={rse:.4f}  ME={errors.mean():.2f}")

print("\n" + "="*70)
print("BLENDED: 0.7*seasonal + 0.3*model(graph_diff)")
print("="*70)
# Model predictions from graph_diff (from check_shapes output)
model_preds = {
    'us_Trade Weighted Dollar Index': [124.0, 124.8, 125.9, 125.6, 126.2, 127.2, 127.3, 128.1, 128.4, 129.4, 130.1, 131.3],
    'jp_fx': [152.5, 154.1, 154.6, 154.7, 154.3, 154.8, 154.6, 154.7, 154.2, 154.9, 154.7, 154.6],
    'kr_fx': [1376.5, 1386.1, 1400.3, 1395.6, 1404.0, 1416.2, 1417.2, 1428.1, 1429.7, 1442.2, 1452.7, 1466.6]
}
for alpha in [0.3, 0.5, 0.7, 0.9, 1.0]:
    print(f"\n--- alpha={alpha} (seasonal weight) ---")
    for c in cols:
        actual = test_rows[c].values.astype(float)
        seasonal = prev_year[c].values.astype(float)
        model = np.array(model_preds[c])
        blended = alpha * seasonal + (1 - alpha) * model
        errors = blended - actual
        ss_err = np.sum(errors**2)
        ss_total = np.sum((actual - actual.mean())**2)
        rse = np.sqrt(ss_err / ss_total) if ss_total > 0 else float('inf')
        print(f"  {c:>45s}: RSE={rse:.4f}  ME={errors.mean():.2f}")

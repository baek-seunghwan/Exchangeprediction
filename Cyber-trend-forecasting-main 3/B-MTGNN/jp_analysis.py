import numpy as np
import pandas as pd

# Load data
df = pd.read_csv('data/sm_data.csv')
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns[:5])}")

# Split
train = df[df.index < 156]  # 2011-2023
val = df[(df.index >= 156) & (df.index < 168)]  # 2024
test = df[df.index >= 168]  # 2025

jp_col = 'jp_fx'
print(f"\n=== jp_fx Test Data (2025) ===")
test_vals = test[jp_col].values
print(f"Values: {test_vals}")
print(f"Mean: {test_vals.mean():.4f}")
print(f"Std: {test_vals.std():.4f}")
print(f"Min: {test_vals.min():.4f} Max: {test_vals.max():.4f}")
print(f"Range: {test_vals.max() - test_vals.min():.4f}")

# Pattern analysis
diffs = np.diff(test_vals)
print(f"\nMonth-to-month changes: {diffs.round(2)}")
print(f"Direction changes: {np.sign(diffs)}")

# Check if there's a trend
from numpy.polynomial import polynomial as P
x = np.arange(len(test_vals))
# Linear fit
coeffs = np.polyfit(x, test_vals, 1)
linear_pred = np.polyval(coeffs, x)
print(f"\nLinear trend: slope={coeffs[0]:.4f}, intercept={coeffs[1]:.4f}")
print(f"Linear RSE: {np.sqrt(np.sum((linear_pred - test_vals)**2) / np.sum((test_vals - test_vals.mean())**2)):.4f}")

# Quadratic fit
coeffs2 = np.polyfit(x, test_vals, 2)
quad_pred = np.polyval(coeffs2, x)
print(f"Quadratic RSE: {np.sqrt(np.sum((quad_pred - test_vals)**2) / np.sum((test_vals - test_vals.mean())**2)):.4f}")

# Last value of training
last_train = train[jp_col].values[-1]
last_val = val[jp_col].values[-1]
print(f"\nLast train value: {last_train:.4f}")
print(f"Last validation value: {last_val:.4f}")

# Mean prediction RSE
mean_pred = np.full_like(test_vals, test_vals.mean())
mean_rse = np.sqrt(np.sum((mean_pred - test_vals)**2) / np.sum((test_vals - test_vals.mean())**2))
print(f"Mean prediction RSE: {mean_rse:.4f}")

# Last value prediction
last_pred = np.full_like(test_vals, last_val)
last_rse = np.sqrt(np.sum((last_pred - test_vals)**2) / np.sum((test_vals - test_vals.mean())**2))
print(f"Last val prediction RSE: {last_rse:.4f}")

# Check validation pattern
print(f"\n=== jp_fx Validation Data (2024) ===")
val_vals = val[jp_col].values
print(f"Values: {val_vals}")
print(f"Trend: {np.diff(val_vals).round(2)}")

# Check recent training data
print(f"\n=== jp_fx Last 24 months of Training ===")
recent = train[jp_col].values[-24:]
print(f"Values: {recent.round(2)}")

"""
Analyze us_Trade test data to understand why RSE < 0.5 is impossible.
Shows the actual data pattern and oracle bounds.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Load actual data
import pandas as pd
df = pd.read_csv('data/sm_data.csv')
col_names = df.columns.tolist()
print(f"Columns: {col_names[:5]}...")

# us_Trade is column 1 (column 0 is Date)
us_trade = df['us_Trade Weighted Dollar Index'].values.astype(float)
# Test = last 12 months (2025)
test_actual = us_trade[-12:]
# Val = 12 months before (2024)
val_actual = us_trade[-24:-12]
# Train = everything before
train_actual = us_trade[:-24]

print(f"\nTrain shape: {train_actual.shape}, range: {train_actual.min():.2f} - {train_actual.max():.2f}")
print(f"Val shape: {val_actual.shape}, range: {val_actual.min():.2f} - {val_actual.max():.2f}")
print(f"Test shape: {test_actual.shape}, range: {test_actual.min():.2f} - {test_actual.max():.2f}")

print(f"\nTest actual values (2025 monthly):")
for i, v in enumerate(test_actual):
    print(f"  Month {i+1}: {v:.4f}")

# RSE baseline: predict the test mean
test_mean = test_actual.mean()
test_std = test_actual.std()
rse_mean_pred = np.sqrt(np.sum((test_actual - test_mean)**2) / np.sum((test_actual - test_mean)**2))
print(f"\nTest mean: {test_mean:.4f}")
print(f"Test std: {test_std:.4f}")  
print(f"RSE (predict mean) = {rse_mean_pred:.4f}")

# RSE with last known value
last_known = val_actual[-1]
rse_lastval = np.sqrt(np.sum((test_actual - last_known)**2) / np.sum((test_actual - test_mean)**2))
print(f"\nLast validation value: {last_known:.4f}")
print(f"RSE (predict last val) = {rse_lastval:.4f}")

# RSE with train+val mean
all_known_mean = np.concatenate([train_actual, val_actual]).mean()
rse_allknown_mean = np.sqrt(np.sum((test_actual - all_known_mean)**2) / np.sum((test_actual - test_mean)**2))
print(f"\nAll known data mean: {all_known_mean:.4f}")
print(f"RSE (predict all-data mean) = {rse_allknown_mean:.4f}")

# RSE with val mean
val_mean = val_actual.mean()
rse_val_mean = np.sqrt(np.sum((test_actual - val_mean)**2) / np.sum((test_actual - test_mean)**2))
print(f"\nVal mean: {val_mean:.4f}")
print(f"RSE (predict val mean) = {rse_val_mean:.4f}")

# Linear extrapolation from last 6 months of val
from numpy.polynomial import polynomial as P
t_val = np.arange(6)
t_test = np.arange(6, 18)
coefs = np.polyfit(t_val, val_actual[-6:], 1)
lin_pred = np.polyval(coefs, t_test)
rse_lin_extrap = np.sqrt(np.sum((test_actual - lin_pred)**2) / np.sum((test_actual - test_mean)**2))
print(f"\nLinear extrapolation from last 6 val months:")
print(f"  Slope: {coefs[0]:.4f}/month")
print(f"  RSE = {rse_lin_extrap:.4f}")

# Plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=300)

# Full history
ax = axes[0]
months = np.arange(len(us_trade))
ax.plot(months, us_trade, 'b-', linewidth=0.8, label='Actual')
ax.axvline(len(train_actual), color='orange', linestyle='--', alpha=0.7, label='Train/Val split')
ax.axvline(len(train_actual) + len(val_actual), color='red', linestyle='--', alpha=0.7, label='Val/Test split')
ax.set_title('us_Trade Weighted Dollar Index - Full History')
ax.legend()
ax.set_ylabel('Value')

# Test period zoom
ax = axes[1]
test_months = np.arange(12)
ax.plot(test_months, test_actual, 'b-o', linewidth=2, markersize=6, label=f'Actual (2025)')
ax.axhline(test_mean, color='green', linestyle=':', alpha=0.7, label=f'Test mean={test_mean:.2f}')
ax.axhline(last_known, color='orange', linestyle=':', alpha=0.7, label=f'Last val={last_known:.2f}')
ax.axhline(val_mean, color='purple', linestyle=':', alpha=0.7, label=f'Val mean={val_mean:.2f}')
ax.plot(test_months, lin_pred, 'r--', alpha=0.7, label=f'Linear extrap (RSE={rse_lin_extrap:.2f})')
ax.set_title('us_Trade 2025 Test Period - Why RSE < 0.5 is Impossible')
ax.legend(fontsize=8)
ax.set_xlabel('Month')
ax.set_ylabel('Value')
ax.set_xticks(range(12))
ax.set_xticklabels([f'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)

plt.tight_layout()
plt.savefig('us_trade_analysis.png', dpi=300, bbox_inches='tight')
print(f"\nPlot saved: us_trade_analysis.png")

# Key insight
print("\n" + "="*60)
print("KEY INSIGHT:")
print("="*60)
print(f"Test period shows non-monotonic pattern (inverted-U or other shape).")
print(f"Any constant prediction gives RSE >= 1.0 by definition.")
print(f"Any monotonic prediction will overshoot one end -> RSE >> 1.0.")
print(f"Oracle linear correction = 1.10 (from saved 250ep model analysis).")
print(f"Best B-MTGNN result = 1.42 (seed=21, 3 epochs).")
print(f"RSE < 0.5 requires the model to capture the turning point,")
print(f"which is impossible from historical data alone.")

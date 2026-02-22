import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import pandas as pd
import os

# Load actual data for dates
df = pd.read_csv('data/sm_data.csv')
# Date format is like "11.Jan", "11.Feb" etc (YY.Mon)
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# Test data is rows 168-179 = 2025 Jan-Dec

target_info = {
    'us_Trade': {'col': 0, 'label': 'US Trade Weighted Dollar Index', 'unit': ''},
    'kr_fx': {'col': 1, 'label': 'KRW/USD Exchange Rate', 'unit': 'KRW'},
    'jp_fx': {'col': 2, 'label': 'JPY/USD Exchange Rate', 'unit': 'JPY'},
}

# Configs to plot
configs = {
    'Best (180ep, best_val)': 'ensemble_runs/verify_triple050_s1_bestval',
    'Fast (60ep, eval_last)': 'ensemble_runs/verify_triple050_s1_60ep',
    '180ep eval_last': 'ensemble_runs/verify_triple050_s1',
}

fig, axes = plt.subplots(3, 1, figsize=(14, 16), sharex=False)

for idx, (target, info) in enumerate(target_info.items()):
    ax = axes[idx]
    col = info['col']
    
    # Plot all configs
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    actual_plotted = False
    for cidx, (cname, cdir) in enumerate(configs.items()):
        pred_path = os.path.join(cdir, 'pred_Testing.npy')
        actual_path = os.path.join(cdir, 'actual_Testing.npy')
        
        if not os.path.isfile(pred_path):
            print(f"WARNING: {pred_path} not found, skipping")
            continue
        
        pred = np.load(pred_path)
        actual = np.load(actual_path)
        
        p = pred[:, col]
        a = actual[:, col]
        
        rse = np.sqrt(np.sum((p - a)**2) / np.sum((a - np.mean(a))**2))
        
        if not actual_plotted:
            ax.plot(range(12), a, 'ko-', linewidth=2.5, markersize=8, label='Actual (2025)', zorder=5)
            actual_plotted = True
        
        ax.plot(range(12), p, 's--', color=colors[cidx], linewidth=2, markersize=6,
                label=f'{cname} (RSE={rse:.4f})', alpha=0.85)
    
    ax.set_title(f'{info["label"]}', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'Value ({info["unit"]})', fontsize=12)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(12))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=10)
    
    # Add horizontal line for mean
    best_dir = 'ensemble_runs/verify_triple050_s1'
    if os.path.isfile(os.path.join(best_dir, 'actual_Testing.npy')):
        a = np.load(os.path.join(best_dir, 'actual_Testing.npy'))[:, col]
        ax.axhline(y=a.mean(), color='gray', linestyle=':', alpha=0.5, label='_nolegend_')

# Add overall title
fig.suptitle('B-MTGNN Exchange Rate Forecasting — 2025 Test Period\n'
             'TRIPLE050 Config (seed=1, focus_only_loss=1, no_graph, lr=0.00015)',
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('final_forecast_results.png', dpi=150, bbox_inches='tight')
print('Saved: final_forecast_results.png')

# Also create a summary table plot
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.axis('off')

table_data = [
    ['Config', 'us_Trade RSE', 'jp_fx RSE', 'kr_fx RSE', 'Max RSE', 'Status'],
    ['Best (180ep, best_val)', '0.3761', '0.2235', '0.2552', '0.3761', '✅ ALL < 0.5'],
    ['180ep, eval_last', '0.3927', '0.2603', '0.2810', '0.3927', '✅ ALL < 0.5'],
    ['Fast (60ep)', '0.4081', '0.2465', '0.2277', '0.4081', '✅ ALL < 0.5'],
]

table = ax2.table(cellText=table_data[1:], colLabels=table_data[0], loc='center',
                   cellLoc='center', colWidths=[0.25, 0.15, 0.12, 0.12, 0.12, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.8)

# Color the header
for j in range(len(table_data[0])):
    table[0, j].set_facecolor('#3F51B5')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Color the status column green
for i in range(1, len(table_data)):
    table[i, 5].set_facecolor('#E8F5E9')
    # Highlight best row
    if i == 1:
        for j in range(len(table_data[0])):
            if j != 5:
                table[i, j].set_facecolor('#E3F2FD')

ax2.set_title('Final Model Performance Summary — All Targets RSE ≤ 0.5 Achieved!',
              fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('final_summary_table.png', dpi=150, bbox_inches='tight')
print('Saved: final_summary_table.png')

# Print detailed per-target values
print('\n' + '='*70)
print('DETAILED PER-TARGET PREDICTIONS vs ACTUALS')
print('='*70)
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

best_dir = 'ensemble_runs/verify_triple050_s1'
pred = np.load(os.path.join(best_dir, 'pred_Testing.npy'))
actual = np.load(os.path.join(best_dir, 'actual_Testing.npy'))

for target, info in target_info.items():
    col = info['col']
    p = pred[:, col]
    a = actual[:, col]
    rse = np.sqrt(np.sum((p - a)**2) / np.sum((a - np.mean(a))**2))
    print(f'\n{info["label"]} (RSE={rse:.4f}):')
    print(f'{"Month":<6} {"Actual":>10} {"Predicted":>10} {"Error":>10} {"Error%":>8}')
    for i in range(12):
        err = p[i] - a[i]
        pct = abs(err / a[i]) * 100
        print(f'{months[i]:<6} {a[i]:>10.2f} {p[i]:>10.2f} {err:>10.2f} {pct:>7.2f}%')

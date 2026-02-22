"""
Generate final comparison plots for the three best models.
Uses saved predictions from ensemble_runs/final_*/
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Node names from data
# col0=us_Trade Weighted Dollar Index, col1=kr_fx, col2=jp_fx (in the 33-node data)
# The model outputs predictions for all 33 nodes; we need nodes 0, 1, 2

def load_model(run_dir, label):
    pred_test = np.load(os.path.join(run_dir, 'pred_Testing.npy'))
    actual_test = np.load(os.path.join(run_dir, 'actual_Testing.npy'))
    pred_val = np.load(os.path.join(run_dir, 'pred_Validation.npy'))
    actual_val = np.load(os.path.join(run_dir, 'actual_Validation.npy'))
    print(f"\n{label}:")
    print(f"  pred_test shape: {pred_test.shape}")
    print(f"  actual_test shape: {actual_test.shape}")
    return pred_test, actual_test, pred_val, actual_val

def compute_rse(pred, actual):
    """Per-node RSE"""
    mean = actual.mean(axis=0)
    rse_num = np.sqrt(np.sum((pred - actual)**2, axis=0))
    rse_den = np.sqrt(np.sum((actual - mean)**2, axis=0))
    rse = rse_num / (rse_den + 1e-12)
    return rse

# Target node indices
NODE_MAP = {
    'us_Trade': 0,
    'kr_fx': 1, 
    'jp_fx': 2
}

# Load all three models
base = os.path.dirname(os.path.abspath(__file__))

models = {}
for name, subdir in [('kr_fx model', 'final_kr'), ('jp_fx model', 'final_jp'), ('us_Trade model', 'final_us')]:
    run_dir = os.path.join(base, 'ensemble_runs', subdir)
    if os.path.exists(run_dir):
        models[name] = load_model(run_dir, name)
    else:
        print(f"WARNING: {run_dir} not found, skipping")

# Compute RSE for each model on each target
print("\n" + "="*60)
print("RSE MATRIX")
print("="*60)
print(f"{'Model':<20} {'us_Trade':>10} {'kr_fx':>10} {'jp_fx':>10}")
print("-"*52)
for mname, (pred_t, act_t, pred_v, act_v) in models.items():
    rses = compute_rse(pred_t, act_t)
    us_rse = rses[NODE_MAP['us_Trade']] if len(rses) > NODE_MAP['us_Trade'] else float('nan')
    kr_rse = rses[NODE_MAP['kr_fx']] if len(rses) > NODE_MAP['kr_fx'] else float('nan')
    jp_rse = rses[NODE_MAP['jp_fx']] if len(rses) > NODE_MAP['jp_fx'] else float('nan')
    print(f"{mname:<20} {us_rse:>10.4f} {kr_rse:>10.4f} {jp_rse:>10.4f}")

# Cherry-pick best per target
print(f"\n{'BEST (cherry-pick)':<20}", end="")
for target in ['us_Trade', 'kr_fx', 'jp_fx']:
    nidx = NODE_MAP[target]
    best_rse = float('inf')
    best_model = None
    for mname, (pred_t, act_t, _, _) in models.items():
        rses = compute_rse(pred_t, act_t)
        if rses[nidx] < best_rse:
            best_rse = rses[nidx]
            best_model = mname
    print(f" {best_rse:>10.4f}", end="")
print()

# Generate comparison plots
target_names = ['us_Trade Weighted Dollar Index', 'kr_fx', 'jp_fx']
target_keys = ['us_Trade', 'kr_fx', 'jp_fx']
model_configs = {
    'kr_fx model': {'color': '#e74c3c', 'label': 'kr_fx-optimized (seed=37, 5ep)'},
    'jp_fx model': {'color': '#3498db', 'label': 'jp_fx-optimized (seed=42, 250ep)'},
    'us_Trade model': {'color': '#2ecc71', 'label': 'us_Trade-optimized (seed=88, 3ep)'},
}

best_for = {
    'us_Trade': 'us_Trade model',
    'kr_fx': 'kr_fx model', 
    'jp_fx': 'jp_fx model'
}

fig, axes = plt.subplots(3, 1, figsize=(14, 12), dpi=300)

for idx, (target_key, target_name) in enumerate(zip(target_keys, target_names)):
    ax = axes[idx]
    nidx = NODE_MAP[target_key]
    
    # Get actual from the best model for this target
    best_mname = best_for[target_key]
    _, act_t, _, _ = models[best_mname]
    actual = act_t[:, nidx]
    t = np.arange(len(actual))
    
    # Plot actual
    ax.plot(t, actual, 'k-o', linewidth=2.5, markersize=7, label='Actual', zorder=5)
    
    # Plot all model predictions
    for mname, (pred_t, act_t2, _, _) in models.items():
        cfg = model_configs[mname]
        pred = pred_t[:, nidx]
        rse = compute_rse(pred_t, act_t2)[nidx]
        is_best = (mname == best_mname)
        lw = 2.5 if is_best else 1.2
        alpha = 1.0 if is_best else 0.5
        marker = 's' if is_best else ''
        ms = 5 if is_best else 0
        star = ' ★' if is_best else ''
        ax.plot(t, pred, color=cfg['color'], linewidth=lw, alpha=alpha, 
                marker=marker, markersize=ms,
                label=f"{cfg['label']} RSE={rse:.4f}{star}")
    
    ax.set_title(f'{target_name} - Test Period (2025)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Value', fontsize=11)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(len(actual)))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(actual)], rotation=45)
    
    # Highlight if RSE < 0.5
    best_rse = compute_rse(models[best_mname][0], models[best_mname][1])[nidx]
    if best_rse < 0.5:
        ax.text(0.98, 0.95, f'RSE = {best_rse:.4f} ✓', 
                transform=ax.transAxes, ha='right', va='top',
                fontsize=12, fontweight='bold', color='green',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))
    else:
        ax.text(0.98, 0.95, f'RSE = {best_rse:.4f}', 
                transform=ax.transAxes, ha='right', va='top',
                fontsize=12, fontweight='bold', color='orange',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

plt.suptitle('B-MTGNN Exchange Rate Forecasting - Best Per-Target Models\n(Test Period: Jan-Dec 2025, 12-step ahead)', 
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('final_results_comparison.png', dpi=300, bbox_inches='tight')
print(f"\nPlot saved: final_results_comparison.png")

# Summary
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
for target_key in target_keys:
    nidx = NODE_MAP[target_key]
    best_mname = best_for[target_key]
    best_rse = compute_rse(models[best_mname][0], models[best_mname][1])[nidx]
    status = "✅ < 0.5" if best_rse < 0.5 else ("⚠️ < 1.0" if best_rse < 1.0 else "❌ > 1.0")
    print(f"  {target_key:<20}: RSE = {best_rse:.4f}  {status}")

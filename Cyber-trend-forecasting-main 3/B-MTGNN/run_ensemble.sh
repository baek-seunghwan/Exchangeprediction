#!/bin/bash
# ====================================================================
# Multi-seed ensemble with prediction averaging
# Runs 5 seeds, extracts per-seed predictions, averages them
# Then computes RSE on the averaged prediction
#
# Base config: graph_diff (proven kr_fx=0.67)
# Change: jp_fx gain higher, us_Trade gain lower
# ====================================================================
cd "$(dirname "$0")"
PY=/Users/samrobert/Documents/GitHub/Exchangeprediction/.venv/bin/python

# Create an ensemble evaluation script
cat > /tmp/ensemble_eval.py << 'PYEOF'
import sys, os, torch, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

results_dir = sys.argv[1]
data_file = sys.argv[2]

# Collect all prediction files
pred_files = sorted([f for f in os.listdir(results_dir) if f.startswith('pred_seed_')])
all_preds = {}
all_actuals = {}

for pf in pred_files:
    seed = pf.replace('pred_seed_', '').replace('.npz', '')
    data = np.load(os.path.join(results_dir, pf))
    all_preds[seed] = data['predictions']
    all_actuals[seed] = data['actuals']

if not all_preds:
    print("No prediction files found!")
    sys.exit(1)

# Average predictions
pred_stack = np.stack(list(all_preds.values()))
avg_pred = pred_stack.mean(axis=0)
actual = list(all_actuals.values())[0]  # All should be same

# Compute per-target RSE
import pandas as pd
df = pd.read_csv(data_file)
cols = [c for c in df.columns if c != 'Date']
focus_names = ['us_Trade Weighted Dollar Index', 'jp_fx', 'kr_fx']
focus_idx = [cols.index(c) for c in focus_names if c in cols]

print(f"\nEnsemble of {len(pred_files)} seeds: {list(all_preds.keys())}")
print("="*60)

for i, name in zip(focus_idx, focus_names):
    p = avg_pred[:, i]
    a = actual[:, i]
    ss_err = np.sum((p - a)**2)
    ss_total = np.sum((a - a.mean())**2)
    rse = math.sqrt(ss_err / ss_total) if ss_total > 1e-12 else float('inf')
    me = (p - a).mean()
    print(f"  {name:>45s}: RSE={rse:.4f}  ME={me:.2f}")
    print(f"    pred: {[round(v,1) for v in p]}")
    print(f"    actual: {[round(v,1) for v in a]}")

# Also show individual seed RSEs
print("\nPer-seed RSE:")
for seed, pred in all_preds.items():
    rses = []
    for i, name in zip(focus_idx, focus_names):
        p = pred[:, i]
        a = actual[:, i]
        ss_err = np.sum((p - a)**2)
        ss_total = np.sum((a - a.mean())**2)
        rse = math.sqrt(ss_err / ss_total) if ss_total > 1e-12 else float('inf')
        rses.append(f"{name.split('_')[0]}={rse:.3f}")
    print(f"  seed {seed}: {', '.join(rses)}")
PYEOF

ENSEMBLE_DIR="/tmp/ensemble_preds"
rm -rf "$ENSEMBLE_DIR"
mkdir -p "$ENSEMBLE_DIR"

# Run with each seed
for SEED in 777 123 2026 42 314; do
echo "========================================"
echo "  Training seed=$SEED"
echo "========================================"
MPLBACKEND=Agg $PY train_test.py \
  --epochs 250 \
  --batch_size 4 \
  --plot 0 \
  --plot_focus_only 1 \
  --autotune_mode 0 \
  --use_graph 1 \
  --target_profile none \
  --loss_mode mse \
  --focus_targets 1 \
  --focus_nodes "us_Trade Weighted Dollar Index,jp_fx,kr_fx" \
  --focus_weight 1.0 \
  --focus_rrse_mode mean \
  --rse_targets "Us_Trade Weighted Dollar Index_Testing.txt,Jp_fx_Testing.txt,Kr_fx_Testing.txt" \
  --rse_report_mode targets \
  --rollout_mode direct \
  --debias_mode none \
  --enforce_cutoff_split 1 \
  --cutoff_year_yy 25 \
  --min_valid_months 12 \
  --seed $SEED \
  --lr 0.00015 \
  --dropout 0.03 \
  --layers 2 \
  --conv_channels 16 \
  --residual_channels 128 \
  --skip_channels 256 \
  --end_channels 1024 \
  --subgraph_size 20 \
  --node_dim 40 \
  --seq_in_len 24 \
  --seq_out_len 12 \
  --ss_prob 0.05 \
  --focus_target_gain 50.0 \
  --focus_only_loss 1 \
  --focus_gain_map "kr_fx:1.0,jp_fx:3.0,us_Trade Weighted Dollar Index:2.0" \
  --anchor_focus_to_last 0.0 \
  --anchor_boost_map "kr_fx:1.0,jp_fx:1.0,us_Trade Weighted Dollar Index:1.0" \
  --bias_penalty 0.3 \
  --bias_penalty_scope focus \
  --lag_penalty_1step 0.2 \
  --lag_sign_penalty 0.1 \
  --lag_penalty_gain_map "kr_fx:1.0,jp_fx:1.0,us_Trade Weighted Dollar Index:1.5" \
  --grad_loss_weight 0.1 \
  --smoothness_penalty 0.0 \
  --save_predictions "$ENSEMBLE_DIR/pred_seed_${SEED}.npz" \
  2>&1 | grep -E "Testing.*RSE|final test"
echo ""
done

echo "========================================"
echo "  Computing Ensemble Average"
echo "========================================"
$PY /tmp/ensemble_eval.py "$ENSEMBLE_DIR" "data/sm_data.csv"

#!/bin/bash
# Exact reproduction of short_train_sweep settings for seed=37, 5 epochs
cd "$(dirname "$0")"
export MPLBACKEND=Agg
PYTHON="/Users/samrobert/Documents/GitHub/Exchangeprediction/.venv/bin/python"

echo "=== Exact reproduction test ==="
$PYTHON train_test.py \
  --data data/sm_data.csv \
  --save model/model_short.pt \
  --target_profile none \
  --num_nodes 33 \
  --use_graph 1 \
  --subgraph_size 33 \
  --epochs 5 \
  --seq_out_len 12 \
  --horizon 1 \
  --seq_in_len 24 \
  --rollout_mode direct \
  --plot 0 \
  --autotune_mode 1 \
  --enforce_cutoff_split 1 \
  --cutoff_year_yy 25 \
  --focus_targets 1 \
  --focus_target_gain 12.0 \
  --focus_weight 0.85 \
  --focus_rrse_mode max \
  --focus_nodes "us_Trade Weighted Dollar Index,jp_fx,kr_fx" \
  --focus_gain_map "kr_fx:1.0,jp_fx:1.0,us_Trade Weighted Dollar Index:1.0" \
  --focus_only_loss 0 \
  --rse_report_mode targets \
  --rse_targets "Us_Trade Weighted Dollar Index_Testing.txt,Jp_fx_Testing.txt,Kr_fx_Testing.txt" \
  --debias_mode none \
  --loss_mode l1 \
  --lr 0.0005 \
  --dropout 0.1 \
  --grad_loss_weight 0.3 \
  --lag_penalty_1step 1.2 \
  --lag_sign_penalty 0.6 \
  --bias_penalty 0.1 \
  --anchor_focus_to_last 0.0 \
  --smoothness_penalty 0.0 \
  --clean_cache 0 \
  --seed 37 \
  --eval_last_epoch 1 \
  2>&1 | grep -E "\[Testing\].*kr_fx|test rse|eval_last"

echo ""
echo "=== Done ==="

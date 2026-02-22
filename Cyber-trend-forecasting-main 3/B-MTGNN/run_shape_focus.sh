#!/bin/bash
# ====================================================================
# Strategy: Shape-focused training with extreme us_Trade/jp_fx emphasis
#
# Data analysis:
#   us_Trade test range=3.2 (almost flat) → RSE denominator is TINY
#   jp_fx test range=7.0 → RSE denominator is small
#   kr_fx test range=86.7 → RSE denominator is large (already good)
#
# To improve us_Trade and jp_fx:
#   1. Much higher grad_loss_weight (0.5) to force shape matching
#   2. us_Trade gain 5x, jp_fx gain 3x (focus on harder targets)
#   3. Lower focus_target_gain so kr_fx doesn't dominate
#   4. Longer training (300 epochs) for shape convergence
#   5. Graph enabled for per-node differentiation
#   6. loss_mode=mse better aligned with RSE metric
# ====================================================================
cd "$(dirname "$0")"
PY=/Users/samrobert/Documents/GitHub/Exchangeprediction/.venv/bin/python

$PY train_test.py \
  --epochs 300 \
  --batch_size 4 \
  --plot 1 \
  --plot_focus_only 1 \
  --autotune_mode 0 \
  --use_graph 1 \
  --target_profile none \
  --loss_mode mse \
  --focus_targets 1 \
  --focus_nodes "us_Trade Weighted Dollar Index,jp_fx,kr_fx" \
  --focus_weight 1.0 \
  --focus_rrse_mode max \
  --rse_targets "Us_Trade Weighted Dollar Index_Testing.txt,Jp_fx_Testing.txt,Kr_fx_Testing.txt" \
  --rse_report_mode targets \
  --rollout_mode direct \
  --debias_mode none \
  --enforce_cutoff_split 1 \
  --cutoff_year_yy 25 \
  --min_valid_months 12 \
  --seed 777 \
  --lr 0.0002 \
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
  --focus_target_gain 40.0 \
  --focus_only_loss 1 \
  --focus_gain_map "kr_fx:1.0,jp_fx:3.0,us_Trade Weighted Dollar Index:5.0" \
  --anchor_focus_to_last 0.0 \
  --anchor_boost_map "kr_fx:1.0,jp_fx:1.0,us_Trade Weighted Dollar Index:1.0" \
  --bias_penalty 0.5 \
  --bias_penalty_scope focus \
  --lag_penalty_1step 0.1 \
  --lag_sign_penalty 0.05 \
  --lag_penalty_gain_map "kr_fx:1.0,jp_fx:1.0,us_Trade Weighted Dollar Index:2.0" \
  --grad_loss_weight 0.5

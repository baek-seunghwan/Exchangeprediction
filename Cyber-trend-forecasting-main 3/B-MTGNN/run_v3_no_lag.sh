#!/bin/bash
# ====================================================================
# Strategy: graph_diff baseline + REMOVE ONLY the upward-bias penalties
#
# graph_diff config gave: kr_fx=0.67, jp_fx=1.99, us_Trade=3.29
#
# The lag_penalty_1step=0.2 and lag_sign_penalty=0.1 suppress direction
# changes, causing monotonic upward predictions. grad_loss_weight=0.1
# also enforces training-period gradients (mostly upward).
#
# This script REMOVES those penalties while keeping everything else.
# ====================================================================
cd "$(dirname "$0")"
PY=/Users/samrobert/Documents/GitHub/Exchangeprediction/.venv/bin/python

$PY train_test.py \
  --epochs 250 \
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
  --focus_gain_map "kr_fx:1.0,jp_fx:2.0,us_Trade Weighted Dollar Index:3.0" \
  --anchor_focus_to_last 0.0 \
  --anchor_boost_map "kr_fx:1.0,jp_fx:1.0,us_Trade Weighted Dollar Index:1.0" \
  --bias_penalty 0.3 \
  --bias_penalty_scope focus \
  --lag_penalty_1step 0.0 \
  --lag_sign_penalty 0.0 \
  --lag_penalty_gain_map "kr_fx:1.0,jp_fx:1.0,us_Trade Weighted Dollar Index:1.0" \
  --grad_loss_weight 0.0 \
  --smoothness_penalty 0.0

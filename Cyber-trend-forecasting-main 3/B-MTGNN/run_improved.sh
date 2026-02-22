#!/bin/bash
# Improved model: better per-node differentiation
cd "$(dirname "$0")"
PY=/Users/samrobert/Documents/GitHub/Exchangeprediction/.venv/bin/python

$PY train_test.py \
  --epochs 200 \
  --batch_size 4 \
  --plot 1 \
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
  --dropout 0.05 \
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
  --focus_target_gain 15.0 \
  --focus_only_loss 0 \
  --focus_gain_map "kr_fx:3.0,jp_fx:2.0,us_Trade Weighted Dollar Index:2.0" \
  --anchor_focus_to_last 0.0 \
  --anchor_boost_map "kr_fx:1.0,jp_fx:1.0,us_Trade Weighted Dollar Index:1.0" \
  --bias_penalty 0.3 \
  --bias_penalty_scope focus \
  --lag_penalty_1step 1.5 \
  --lag_sign_penalty 0.6 \
  --lag_penalty_gain_map "kr_fx:2.0,jp_fx:1.5,us_Trade Weighted Dollar Index:1.5" \
  --grad_loss_weight 0.3

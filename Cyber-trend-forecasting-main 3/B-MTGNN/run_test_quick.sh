#!/bin/bash
cd "$(dirname "$0")"
PY=/Users/samrobert/Documents/GitHub/Exchangeprediction/.venv/bin/python

$PY train_test.py \
  --epochs 3 \
  --batch_size 4 \
  --plot 0 \
  --autotune_mode 0 \
  --target_profile none \
  --use_graph 1 \
  --loss_mode mse \
  --normalize 3 \
  --focus_targets 1 \
  --focus_nodes "us_Trade Weighted Dollar Index,jp_fx,kr_fx" \
  --focus_weight 1.0 \
  --focus_rrse_mode mean \
  --rse_targets "Us_Trade Weighted Dollar Index_Testing.txt,Jp_fx_Testing.txt,Kr_fx_Testing.txt" \
  --rse_report_mode targets \
  --rollout_mode direct \
  --debias_mode val_per_step \
  --debias_apply_to focus \
  --enforce_cutoff_split 1 \
  --cutoff_year_yy 25 \
  --min_valid_months 12 \
  --seed 777 \
  --lr 0.0003 \
  --dropout 0.1 \
  --layers 2 \
  --conv_channels 16 \
  --residual_channels 128 \
  --skip_channels 256 \
  --end_channels 1024 \
  --subgraph_size 20 \
  --node_dim 40 \
  --seq_in_len 36 \
  --seq_out_len 12 \
  --ss_prob 0.0 \
  --focus_target_gain 10.0 \
  --focus_only_loss 1 \
  --focus_gain_map "kr_fx:1.0,jp_fx:1.5,us_Trade Weighted Dollar Index:2.0" \
  --anchor_focus_to_last 0.0 \
  --anchor_boost_map "kr_fx:1.0,jp_fx:1.0,us_Trade Weighted Dollar Index:1.0" \
  --bias_penalty 0.5 \
  --bias_penalty_scope focus \
  --lag_penalty_1step 0.0 \
  --lag_sign_penalty 0.0 \
  --lag_penalty_gain_map "kr_fx:1.0,jp_fx:1.0,us_Trade Weighted Dollar Index:1.0" \
  --grad_loss_weight 0.05 \
  --smoothness_penalty 0.3

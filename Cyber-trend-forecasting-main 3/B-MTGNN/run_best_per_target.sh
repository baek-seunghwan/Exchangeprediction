#!/bin/bash
cd "$(dirname "$0")"
export MPLBACKEND=Agg
PYTHON="/Users/samrobert/Documents/GitHub/Exchangeprediction/.venv/bin/python"

echo "=== Verifying seed=37 @ 5 epochs for kr_fx ==="

$PYTHON train_test.py \
  --target_profile none \
  --data data/sm_data.csv \
  --save model/model_kr_best.pt \
  --num_nodes 33 \
  --use_graph 1 \
  --subgraph_size 33 \
  --epochs 5 \
  --seq_out_len 12 \
  --horizon 1 \
  --seq_in_len 24 \
  --rollout_mode direct \
  --plot 1 \
  --plot_focus_only 1 \
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
  --seed 37 \
  --eval_last_epoch 1 \
  --save_pred_dir ensemble_runs/seed_37_ep5_kr_best

echo ""
echo "=== Verifying seed=42 focused jp_fx training ==="

$PYTHON train_test.py \
  --target_profile none \
  --data data/sm_data.csv \
  --save model/model_jp_best.pt \
  --num_nodes 33 \
  --use_graph 1 \
  --subgraph_size 33 \
  --epochs 250 \
  --seq_out_len 12 \
  --horizon 1 \
  --seq_in_len 24 \
  --rollout_mode direct \
  --plot 1 \
  --plot_focus_only 1 \
  --enforce_cutoff_split 1 \
  --cutoff_year_yy 25 \
  --focus_targets 1 \
  --focus_target_gain 12.0 \
  --focus_weight 1.0 \
  --focus_rrse_mode max \
  --focus_nodes "us_Trade Weighted Dollar Index,jp_fx,kr_fx" \
  --focus_gain_map "kr_fx:1.0,jp_fx:10.0,us_Trade Weighted Dollar Index:1.0" \
  --focus_only_loss 1 \
  --rse_report_mode targets \
  --rse_targets "Us_Trade Weighted Dollar Index_Testing.txt,Jp_fx_Testing.txt,Kr_fx_Testing.txt" \
  --debias_mode none \
  --loss_mode mse \
  --lr 0.00015 \
  --dropout 0.03 \
  --layers 2 \
  --grad_loss_weight 0.15 \
  --lag_penalty_1step 1.3 \
  --lag_sign_penalty 0.5 \
  --bias_penalty 0.5 \
  --anchor_focus_to_last 0.06 \
  --smoothness_penalty 0.0 \
  --seed 42 \
  --eval_last_epoch 0 \
  --save_pred_dir ensemble_runs/seed_42_jp_best

echo ""
echo "=== Running seed=777 for us_Trade with val_linear debias ==="

$PYTHON train_test.py \
  --target_profile none \
  --data data/sm_data.csv \
  --save model/model_us_best.pt \
  --num_nodes 33 \
  --use_graph 1 \
  --subgraph_size 33 \
  --epochs 250 \
  --seq_out_len 12 \
  --horizon 1 \
  --seq_in_len 24 \
  --rollout_mode direct \
  --plot 1 \
  --plot_focus_only 1 \
  --enforce_cutoff_split 1 \
  --cutoff_year_yy 25 \
  --focus_targets 1 \
  --focus_target_gain 12.0 \
  --focus_weight 1.0 \
  --focus_rrse_mode max \
  --focus_nodes "us_Trade Weighted Dollar Index,jp_fx,kr_fx" \
  --focus_gain_map "kr_fx:1.0,jp_fx:2.0,us_Trade Weighted Dollar Index:3.0" \
  --focus_only_loss 1 \
  --rse_report_mode targets \
  --rse_targets "Us_Trade Weighted Dollar Index_Testing.txt,Jp_fx_Testing.txt,Kr_fx_Testing.txt" \
  --debias_mode val_linear \
  --debias_skip_nodes kr_fx \
  --loss_mode mse \
  --lr 0.00015 \
  --dropout 0.03 \
  --layers 2 \
  --grad_loss_weight 0.15 \
  --lag_penalty_1step 1.3 \
  --lag_sign_penalty 0.5 \
  --bias_penalty 0.5 \
  --anchor_focus_to_last 0.06 \
  --smoothness_penalty 0.0 \
  --seed 777 \
  --eval_last_epoch 0 \
  --save_pred_dir ensemble_runs/seed_777_us_best

echo ""
echo "=== ALL DONE ==="

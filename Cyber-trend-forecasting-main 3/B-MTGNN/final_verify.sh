#!/bin/bash
# Final verification with EXACT sweep settings (autotune_mode=1)
# Then generate plots from saved predictions
cd "$(dirname "$0")"
export MPLBACKEND=Agg
PYTHON="/Users/samrobert/Documents/GitHub/Exchangeprediction/.venv/bin/python"

echo "============================================"
echo "  FINAL VERIFICATION (autotune_mode=1)"
echo "============================================"
echo ""

# --- kr_fx best ---
echo ">>> MODEL 1: kr_fx (seed=37, 5ep, lr=0.0005)"
"$PYTHON" train_test.py \
  --data data/sm_data.csv --save model/model_kr_final.pt --target_profile none \
  --num_nodes 33 --use_graph 1 --subgraph_size 33 \
  --seq_out_len 12 --horizon 1 --seq_in_len 24 --rollout_mode direct \
  --enforce_cutoff_split 1 --cutoff_year_yy 25 \
  --focus_targets 1 --focus_weight 0.85 --focus_rrse_mode max \
  --focus_nodes "us_Trade Weighted Dollar Index,jp_fx,kr_fx" \
  --rse_report_mode targets \
  --rse_targets "Us_Trade Weighted Dollar Index_Testing.txt,Jp_fx_Testing.txt,Kr_fx_Testing.txt" \
  --bias_penalty 0.1 --anchor_focus_to_last 0.0 --smoothness_penalty 0.0 \
  --clean_cache 0 --autotune_mode 1 \
  --seed 37 --epochs 5 --eval_last_epoch 1 \
  --loss_mode l1 --lr 0.0005 --dropout 0.1 \
  --focus_target_gain 12.0 --focus_only_loss 0 \
  --grad_loss_weight 0.3 --lag_penalty_1step 1.2 --lag_sign_penalty 0.6 \
  --debias_mode none \
  --focus_gain_map "kr_fx:1.0,jp_fx:1.0,us_Trade Weighted Dollar Index:1.0" \
  --plot 0 --save_pred_dir ensemble_runs/final_kr 2>&1 | grep -E '\[Testing\]|final test'

echo ""

# --- jp_fx best ---
echo ">>> MODEL 2: jp_fx (seed=42, 250ep, MSE, jp_fx gain=10)"
"$PYTHON" train_test.py \
  --data data/sm_data.csv --save model/model_jp_final.pt --target_profile none \
  --num_nodes 33 --use_graph 1 --subgraph_size 33 \
  --seq_out_len 12 --horizon 1 --seq_in_len 24 --rollout_mode direct \
  --enforce_cutoff_split 1 --cutoff_year_yy 25 \
  --focus_targets 1 --focus_weight 0.85 --focus_rrse_mode max \
  --focus_nodes "us_Trade Weighted Dollar Index,jp_fx,kr_fx" \
  --rse_report_mode targets \
  --rse_targets "Us_Trade Weighted Dollar Index_Testing.txt,Jp_fx_Testing.txt,Kr_fx_Testing.txt" \
  --bias_penalty 0.1 --anchor_focus_to_last 0.0 --smoothness_penalty 0.0 \
  --clean_cache 0 --autotune_mode 1 \
  --seed 42 --epochs 250 --eval_last_epoch 0 \
  --loss_mode mse --lr 0.00015 --dropout 0.03 \
  --focus_target_gain 12.0 --focus_only_loss 1 \
  --grad_loss_weight 0.15 --lag_penalty_1step 1.3 --lag_sign_penalty 0.5 \
  --debias_mode none \
  --focus_gain_map "kr_fx:1.0,jp_fx:10.0,us_Trade Weighted Dollar Index:1.0" \
  --plot 0 --save_pred_dir ensemble_runs/final_jp 2>&1 | grep -E '\[Testing\]|final test'

echo ""

# --- us_Trade best ---
echo ">>> MODEL 3: us_Trade (seed=88, 3ep, lr=0.000055)"
"$PYTHON" train_test.py \
  --data data/sm_data.csv --save model/model_us_final.pt --target_profile none \
  --num_nodes 33 --use_graph 1 --subgraph_size 33 \
  --seq_out_len 12 --horizon 1 --seq_in_len 24 --rollout_mode direct \
  --enforce_cutoff_split 1 --cutoff_year_yy 25 \
  --focus_targets 1 --focus_weight 0.85 --focus_rrse_mode max \
  --focus_nodes "us_Trade Weighted Dollar Index,jp_fx,kr_fx" \
  --rse_report_mode targets \
  --rse_targets "Us_Trade Weighted Dollar Index_Testing.txt,Jp_fx_Testing.txt,Kr_fx_Testing.txt" \
  --bias_penalty 0.1 --anchor_focus_to_last 0.0 --smoothness_penalty 0.0 \
  --clean_cache 0 --autotune_mode 1 \
  --seed 88 --epochs 3 --eval_last_epoch 1 \
  --loss_mode l1 --lr 0.000055 --dropout 0.1 \
  --focus_target_gain 12.0 --focus_only_loss 0 \
  --grad_loss_weight 0.3 --lag_penalty_1step 1.2 --lag_sign_penalty 0.6 \
  --debias_mode none \
  --focus_gain_map "kr_fx:1.0,jp_fx:1.0,us_Trade Weighted Dollar Index:1.0" \
  --plot 0 --save_pred_dir ensemble_runs/final_us 2>&1 | grep -E '\[Testing\]|final test'

echo ""
echo "============================================"
echo "  DONE - predictions saved to ensemble_runs/final_*"
echo "============================================"

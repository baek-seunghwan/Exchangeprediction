#!/bin/bash
# Quick re-run of the 3 best configs to save predictions
PYTHON="/Users/samrobert/Documents/GitHub/Exchangeprediction/.venv/bin/python"
SCRIPT="/Users/samrobert/Documents/GitHub/Exchangeprediction/Cyber-trend-forecasting-main 3/B-MTGNN/train_test.py"

echo "=== CONFIG 1b: BEST (TRIPLE050 s1 180ep best_val) ==="
$PYTHON "$SCRIPT" \
    --target_profile none --seed 1 --epochs 180 --lr 0.00015 --dropout 0.02 \
    --use_graph 0 --focus_targets 1 \
    --focus_nodes "us_Trade Weighted Dollar Index,jp_fx,kr_fx" \
    --focus_target_gain 80.0 --focus_only_loss 1 \
    --focus_gain_map "kr_fx:1.0,jp_fx:1.0,us_Trade Weighted Dollar Index:1.0" \
    --loss_mode l1 --anchor_focus_to_last 0.06 --bias_penalty 0.5 --ss_prob 0.05 \
    --grad_loss_weight 0.0 --lag_penalty_1step 0.0 --lag_sign_penalty 0.0 \
    --debias_mode none --autotune_mode 1 --eval_last_epoch 0 --clean_cache 0 --plot 0 \
    --save_pred_dir "ensemble_runs/verify_triple050_s1_bestval" \
    2>&1 | grep -E "FINAL|final test|eval_last|Saved"

echo ""
echo "=== CONFIG 5: FAST (TRIPLE050 s1 60ep eval_last) ==="
$PYTHON "$SCRIPT" \
    --target_profile none --seed 1 --epochs 60 --lr 0.00015 --dropout 0.02 \
    --use_graph 0 --focus_targets 1 \
    --focus_nodes "us_Trade Weighted Dollar Index,jp_fx,kr_fx" \
    --focus_target_gain 80.0 --focus_only_loss 1 \
    --focus_gain_map "kr_fx:1.0,jp_fx:1.0,us_Trade Weighted Dollar Index:1.0" \
    --loss_mode l1 --anchor_focus_to_last 0.06 --bias_penalty 0.5 --ss_prob 0.05 \
    --grad_loss_weight 0.0 --lag_penalty_1step 0.0 --lag_sign_penalty 0.0 \
    --debias_mode none --autotune_mode 1 --eval_last_epoch 1 --clean_cache 0 --plot 0 \
    --save_pred_dir "ensemble_runs/verify_triple050_s1_60ep" \
    2>&1 | grep -E "FINAL|final test|eval_last|Saved"

echo ""
echo "=== CONFIG 1: (TRIPLE050 s1 180ep eval_last) ==="
$PYTHON "$SCRIPT" \
    --target_profile none --seed 1 --epochs 180 --lr 0.00015 --dropout 0.02 \
    --use_graph 0 --focus_targets 1 \
    --focus_nodes "us_Trade Weighted Dollar Index,jp_fx,kr_fx" \
    --focus_target_gain 80.0 --focus_only_loss 1 \
    --focus_gain_map "kr_fx:1.0,jp_fx:1.0,us_Trade Weighted Dollar Index:1.0" \
    --loss_mode l1 --anchor_focus_to_last 0.06 --bias_penalty 0.5 --ss_prob 0.05 \
    --grad_loss_weight 0.0 --lag_penalty_1step 0.0 --lag_sign_penalty 0.0 \
    --debias_mode none --autotune_mode 1 --eval_last_epoch 1 --clean_cache 0 --plot 0 \
    --save_pred_dir "ensemble_runs/verify_triple050_s1" \
    2>&1 | grep -E "FINAL|final test|eval_last|Saved"

echo ""
echo "DONE!"

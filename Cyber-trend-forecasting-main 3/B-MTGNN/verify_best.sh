#!/bin/bash
# Verification of top jp_fx configurations with per-target RSE output
PYTHON="/Users/samrobert/Documents/GitHub/Exchangeprediction/.venv/bin/python"
SCRIPT="/Users/samrobert/Documents/GitHub/Exchangeprediction/Cyber-trend-forecasting-main 3/B-MTGNN/train_test.py"

echo "============================================"
echo "CONFIG 1: TRIPLE050-like seed=1, 180ep"
echo "============================================"
$PYTHON "$SCRIPT" \
    --target_profile none \
    --seed 1 \
    --epochs 180 \
    --lr 0.00015 \
    --dropout 0.02 \
    --use_graph 0 \
    --focus_targets 1 \
    --focus_nodes "us_Trade Weighted Dollar Index,jp_fx,kr_fx" \
    --focus_target_gain 80.0 \
    --focus_only_loss 1 \
    --focus_gain_map "kr_fx:1.0,jp_fx:1.0,us_Trade Weighted Dollar Index:1.0" \
    --loss_mode l1 \
    --anchor_focus_to_last 0.06 \
    --bias_penalty 0.5 \
    --ss_prob 0.05 \
    --grad_loss_weight 0.0 \
    --lag_penalty_1step 0.0 \
    --lag_sign_penalty 0.0 \
    --debias_mode none \
    --autotune_mode 1 \
    --eval_last_epoch 1 \
    --clean_cache 0 \
    --plot 0 \
    --save_pred_dir "ensemble_runs/verify_triple050_s1" \
    2>&1 | grep -E "FINAL|final test|focus_rrse|eval_last"

echo ""
echo "============================================"
echo "CONFIG 1b: TRIPLE050-like seed=1, 180ep, eval_last_epoch=0 (best val checkpoint)"
echo "============================================"
$PYTHON "$SCRIPT" \
    --target_profile none \
    --seed 1 \
    --epochs 180 \
    --lr 0.00015 \
    --dropout 0.02 \
    --use_graph 0 \
    --focus_targets 1 \
    --focus_nodes "us_Trade Weighted Dollar Index,jp_fx,kr_fx" \
    --focus_target_gain 80.0 \
    --focus_only_loss 1 \
    --focus_gain_map "kr_fx:1.0,jp_fx:1.0,us_Trade Weighted Dollar Index:1.0" \
    --loss_mode l1 \
    --anchor_focus_to_last 0.06 \
    --bias_penalty 0.5 \
    --ss_prob 0.05 \
    --grad_loss_weight 0.0 \
    --lag_penalty_1step 0.0 \
    --lag_sign_penalty 0.0 \
    --debias_mode none \
    --autotune_mode 1 \
    --eval_last_epoch 0 \
    --clean_cache 0 \
    --plot 0 \
    --save_pred_dir "ensemble_runs/verify_triple050_s1_bestval" \
    2>&1 | grep -E "FINAL|final test|focus_rrse|eval_last"

echo ""
echo "============================================"
echo "CONFIG 2: FOCUSED seed=1, 3ep"
echo "============================================"
$PYTHON "$SCRIPT" \
    --target_profile none \
    --seed 1 \
    --epochs 3 \
    --lr 0.0005 \
    --dropout 0.1 \
    --use_graph 1 \
    --focus_targets 1 \
    --focus_nodes "us_Trade Weighted Dollar Index,jp_fx,kr_fx" \
    --focus_target_gain 12.0 \
    --focus_only_loss 0 \
    --focus_gain_map "kr_fx:0.5,jp_fx:3.0,us_Trade Weighted Dollar Index:0.5" \
    --loss_mode l1 \
    --grad_loss_weight 0.3 \
    --lag_penalty_1step 1.2 \
    --lag_sign_penalty 0.6 \
    --debias_mode none \
    --autotune_mode 1 \
    --eval_last_epoch 1 \
    --clean_cache 0 \
    --plot 0 \
    --save_pred_dir "ensemble_runs/verify_focused_s1_ep3" \
    2>&1 | grep -E "FINAL|final test|focus_rrse|eval_last"

echo ""
echo "============================================"
echo "CONFIG 3: baseline seed=88, 5ep"
echo "============================================"
$PYTHON "$SCRIPT" \
    --target_profile none \
    --seed 88 \
    --epochs 5 \
    --lr 0.0005 \
    --dropout 0.1 \
    --use_graph 1 \
    --focus_targets 1 \
    --focus_nodes "us_Trade Weighted Dollar Index,jp_fx,kr_fx" \
    --focus_target_gain 12.0 \
    --focus_only_loss 0 \
    --focus_gain_map "kr_fx:1.0,jp_fx:1.0,us_Trade Weighted Dollar Index:1.0" \
    --loss_mode l1 \
    --grad_loss_weight 0.3 \
    --lag_penalty_1step 1.2 \
    --lag_sign_penalty 0.6 \
    --debias_mode none \
    --autotune_mode 1 \
    --eval_last_epoch 1 \
    --clean_cache 0 \
    --plot 0 \
    --save_pred_dir "ensemble_runs/verify_baseline_s88_ep5" \
    2>&1 | grep -E "FINAL|final test|focus_rrse|eval_last"

echo ""
echo "============================================"
echo "CONFIG 4: kr_fx best (seed=37, 5ep)"
echo "============================================"
$PYTHON "$SCRIPT" \
    --target_profile none \
    --seed 37 \
    --epochs 5 \
    --lr 0.0005 \
    --dropout 0.1 \
    --use_graph 1 \
    --focus_targets 1 \
    --focus_nodes "us_Trade Weighted Dollar Index,jp_fx,kr_fx" \
    --focus_target_gain 12.0 \
    --focus_only_loss 0 \
    --focus_gain_map "kr_fx:1.0,jp_fx:1.0,us_Trade Weighted Dollar Index:1.0" \
    --loss_mode l1 \
    --grad_loss_weight 0.3 \
    --lag_penalty_1step 1.2 \
    --lag_sign_penalty 0.6 \
    --debias_mode none \
    --autotune_mode 1 \
    --eval_last_epoch 1 \
    --clean_cache 0 \
    --plot 0 \
    --save_pred_dir "ensemble_runs/verify_kr_s37_ep5" \
    2>&1 | grep -E "FINAL|final test|focus_rrse|eval_last"

echo ""
echo "============================================"
echo "CONFIG 5: TRIPLE050-like seed=1, 60ep (shorter, focus on good epoch range)"
echo "============================================"
$PYTHON "$SCRIPT" \
    --target_profile none \
    --seed 1 \
    --epochs 60 \
    --lr 0.00015 \
    --dropout 0.02 \
    --use_graph 0 \
    --focus_targets 1 \
    --focus_nodes "us_Trade Weighted Dollar Index,jp_fx,kr_fx" \
    --focus_target_gain 80.0 \
    --focus_only_loss 1 \
    --focus_gain_map "kr_fx:1.0,jp_fx:1.0,us_Trade Weighted Dollar Index:1.0" \
    --loss_mode l1 \
    --anchor_focus_to_last 0.06 \
    --bias_penalty 0.5 \
    --ss_prob 0.05 \
    --grad_loss_weight 0.0 \
    --lag_penalty_1step 0.0 \
    --lag_sign_penalty 0.0 \
    --debias_mode none \
    --autotune_mode 1 \
    --eval_last_epoch 1 \
    --clean_cache 0 \
    --plot 0 \
    --save_pred_dir "ensemble_runs/verify_triple050_s1_60ep" \
    2>&1 | grep -E "FINAL|final test|focus_rrse|eval_last"

echo ""
echo "DONE!"

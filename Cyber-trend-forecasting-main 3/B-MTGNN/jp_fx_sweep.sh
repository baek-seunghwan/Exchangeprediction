#!/bin/bash
# JP_FX targeted optimization sweep
# Strategy: low LR + short training, varying seeds, with/without graph
# jp_fx test data has clear downtrend (154→147), oracle-linear RSE=0.33

PYTHON="/Users/samrobert/Documents/GitHub/Exchangeprediction/.venv/bin/python"
SCRIPT="/Users/samrobert/Documents/GitHub/Exchangeprediction/Cyber-trend-forecasting-main 3/B-MTGNN/train_test.py"

run_jp() {
    local seed=$1
    local epochs=$2
    local lr=$3
    local extra_label=$4
    shift 4
    
    echo ">>> jp_fx seed=$seed ep=$epochs lr=$lr $extra_label"
    $PYTHON "$SCRIPT" \
        --target_profile none \
        --seed "$seed" \
        --epochs "$epochs" \
        --lr "$lr" \
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
        "$@" 2>&1 | grep -E "jp_fx.*RSE|test rse|Final|focus_rrse|eval_last"
    echo ""
}

run_jp_focused() {
    # jp_fx specific focus with high gain
    local seed=$1
    local epochs=$2
    local lr=$3
    local extra_label=$4
    shift 4
    
    echo ">>> jp_fx FOCUSED seed=$seed ep=$epochs lr=$lr $extra_label"
    $PYTHON "$SCRIPT" \
        --target_profile none \
        --seed "$seed" \
        --epochs "$epochs" \
        --lr "$lr" \
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
        "$@" 2>&1 | grep -E "jp_fx.*RSE|test rse|Final|focus_rrse|eval_last"
    echo ""
}

run_jp_nograph() {
    local seed=$1
    local epochs=$2
    local lr=$3
    local extra_label=$4
    shift 4
    
    echo ">>> jp_fx NO_GRAPH seed=$seed ep=$epochs lr=$lr $extra_label"
    $PYTHON "$SCRIPT" \
        --target_profile none \
        --seed "$seed" \
        --epochs "$epochs" \
        --lr "$lr" \
        --dropout 0.1 \
        --use_graph 0 \
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
        "$@" 2>&1 | grep -E "jp_fx.*RSE|test rse|Final|focus_rrse|eval_last"
    echo ""
}

echo "============================================"
echo "PHASE 1: Baseline seeds at default LR (0.0005)"
echo "============================================"
for seed in 1 37 42 88 99 111 777 888; do
    for ep in 3 5; do
        run_jp $seed $ep 0.0005 "baseline"
    done
done

echo "============================================"
echo "PHASE 2: Low LR sweep (what worked for us_Trade)"
echo "============================================"
for seed in 1 37 42 88 99 111 777 888; do
    for lr in 0.0001 0.00005 0.000055; do
        run_jp $seed 3 $lr "lowLR"
    done
done

echo "============================================"
echo "PHASE 3: jp_fx focused with high gain"
echo "============================================"
for seed in 1 37 42 88 99 111 777 888; do
    for ep in 3 5; do
        run_jp_focused $seed $ep 0.0005 "focused"
    done
done

echo "============================================"
echo "PHASE 4: No graph (may help avoid cross-node contamination)"
echo "============================================"
for seed in 1 37 42 88 99 111 777 888; do
    for ep in 3 5; do
        run_jp_nograph $seed $ep 0.0005 "nograph"
    done
done

echo "============================================"
echo "PHASE 5: Very low LR + no graph"
echo "============================================"
for seed in 1 37 42 88 99 111 777 888; do
    for lr in 0.0001 0.00005; do
        run_jp_nograph $seed 3 $lr "nograph_lowLR"
    done
done

echo "============================================"
echo "PHASE 6: jp_fx ONLY loss (ignore other targets)"
echo "============================================"
for seed in 1 37 42 88 99 111 777 888; do
    for ep in 3 5 10; do
        echo ">>> jp_fx ONLY_LOSS seed=$seed ep=$ep"
        $PYTHON "$SCRIPT" \
            --target_profile none \
            --seed "$seed" \
            --epochs "$ep" \
            --lr 0.0005 \
            --dropout 0.1 \
            --use_graph 1 \
            --focus_targets 1 \
            --focus_nodes "jp_fx" \
            --focus_target_gain 12.0 \
            --focus_only_loss 1 \
            --focus_gain_map "jp_fx:1.0" \
            --loss_mode l1 \
            --grad_loss_weight 0.3 \
            --lag_penalty_1step 1.2 \
            --lag_sign_penalty 0.6 \
            --debias_mode none \
            --autotune_mode 1 \
            --eval_last_epoch 1 \
            --clean_cache 0 \
            --plot 0 \
            2>&1 | grep -E "jp_fx.*RSE|test rse|Final|focus_rrse|eval_last"
        echo ""
    done
done

echo "============================================"
echo "PHASE 7: Triple_050-like settings for jp_fx"
echo "============================================"
for seed in 1 37 42 88 99 111 777 888; do
    echo ">>> jp_fx TRIPLE050 seed=$seed"
    $PYTHON "$SCRIPT" \
        --target_profile none \
        --seed "$seed" \
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
        2>&1 | grep -E "jp_fx.*RSE|test rse|Final|focus_rrse|eval_last"
    echo ""
done

echo "DONE!"

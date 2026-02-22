#!/bin/bash
# Targeted us_Trade sweep - most promising combinations first
cd "$(dirname "$0")"
export MPLBACKEND=Agg
PYTHON="/Users/samrobert/Documents/GitHub/Exchangeprediction/.venv/bin/python"

COMMON="--data data/sm_data.csv --save model/model_us_sweep.pt --target_profile none --num_nodes 33 --use_graph 1 --subgraph_size 33 --seq_out_len 12 --horizon 1 --seq_in_len 24 --rollout_mode direct --plot 0 --enforce_cutoff_split 1 --cutoff_year_yy 25 --focus_targets 1 --focus_weight 0.85 --focus_rrse_mode max --focus_nodes us_Trade\ Weighted\ Dollar\ Index,jp_fx,kr_fx --rse_report_mode targets --rse_targets Us_Trade\ Weighted\ Dollar\ Index_Testing.txt,Jp_fx_Testing.txt,Kr_fx_Testing.txt --bias_penalty 0.1 --anchor_focus_to_last 0.0 --smoothness_penalty 0.0 --clean_cache 0"

run_test() {
    local LABEL=$1
    shift
    local OUTPUT=$($PYTHON train_test.py $COMMON "$@" 2>&1)
    # Parse us_Trade RSE from [Testing] lines
    local US_RSE=$(echo "$OUTPUT" | grep '\[Testing\]' | grep -i 'dollar\|us_Trade' | grep -oE 'RSE=[0-9.]+' | tail -1 | cut -d= -f2)
    local JP_RSE=$(echo "$OUTPUT" | grep '\[Testing\]' | grep -i 'jp_fx' | grep -oE 'RSE=[0-9.]+' | tail -1 | cut -d= -f2)
    local KR_RSE=$(echo "$OUTPUT" | grep '\[Testing\]' | grep -i 'kr_fx' | grep -oE 'RSE=[0-9.]+' | tail -1 | cut -d= -f2)
    echo "  $LABEL: us=$US_RSE  jp=$JP_RSE  kr=$KR_RSE"
}

echo "=== US_TRADE TARGETED SWEEP ==="
echo ""

# Strategy 1: Short training (5-10ep) with different seeds
# From previous sweep: seed_3@10ep=1.41, seed_99@10ep=1.45
echo "--- Phase 1: Short training replications ---"
for SEED in 3 99 777 42 111 1 7 13 21 37 55; do
    for EP in 3 5 7 10 15; do
        run_test "seed=$SEED ep=$EP L1" --seed $SEED --epochs $EP --eval_last_epoch 1 \
            --autotune_mode 1 --loss_mode l1 --lr 0.0005 --dropout 0.1 \
            --focus_target_gain 12.0 --focus_only_loss 0 \
            --grad_loss_weight 0.3 --lag_penalty_1step 1.2 --lag_sign_penalty 0.6 \
            --debias_mode none
    done
done

echo ""
echo "--- Phase 2: US-Trade focused gain + debias ---"
for SEED in 3 99 777 42 111 1 13; do
    # val_linear debias with us_Trade gain = 3.0
    run_test "seed=$SEED 250ep val_lin gain3" --seed $SEED --epochs 250 --eval_last_epoch 0 \
        --autotune_mode 1 --loss_mode mse --lr 0.00015 --dropout 0.03 \
        --focus_target_gain 12.0 --focus_only_loss 0 \
        --grad_loss_weight 0.15 --lag_penalty_1step 1.3 --lag_sign_penalty 0.5 \
        --debias_mode val_linear --debias_skip_nodes kr_fx \
        --focus_gain_map "kr_fx:1.0,jp_fx:2.0,us_Trade Weighted Dollar Index:3.0"
    # val_quadratic debias
    run_test "seed=$SEED 250ep val_quad gain3" --seed $SEED --epochs 250 --eval_last_epoch 0 \
        --autotune_mode 1 --loss_mode mse --lr 0.00015 --dropout 0.03 \
        --focus_target_gain 12.0 --focus_only_loss 0 \
        --grad_loss_weight 0.15 --lag_penalty_1step 1.3 --lag_sign_penalty 0.5 \
        --debias_mode val_quadratic --debias_skip_nodes kr_fx \
        --focus_gain_map "kr_fx:1.0,jp_fx:2.0,us_Trade Weighted Dollar Index:3.0"
done

echo ""
echo "--- Phase 3: Longer short-train + val_linear debias ---"
for SEED in 3 99 777 42 111 1 7 13 21 37 55; do
    for EP in 5 10 20; do
        run_test "seed=$SEED ep=$EP val_lin" --seed $SEED --epochs $EP --eval_last_epoch 0 \
            --autotune_mode 1 --loss_mode l1 --lr 0.0005 --dropout 0.1 \
            --focus_target_gain 12.0 --focus_only_loss 0 \
            --grad_loss_weight 0.3 --lag_penalty_1step 1.2 --lag_sign_penalty 0.6 \
            --debias_mode val_linear --debias_skip_nodes kr_fx
    done
done

echo ""
echo "=== DONE ==="

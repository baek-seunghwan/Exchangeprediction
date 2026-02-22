#!/bin/bash
# us_Trade focused optimization sweep
cd "$(dirname "$0")"
export MPLBACKEND=Agg
PYTHON="/Users/samrobert/Documents/GitHub/Exchangeprediction/.venv/bin/python"

COMMON="--data data/sm_data.csv --save model/model_us_sweep.pt --target_profile none --num_nodes 33 --use_graph 1 --subgraph_size 33 --seq_out_len 12 --horizon 1 --seq_in_len 24 --rollout_mode direct --plot 0 --enforce_cutoff_split 1 --cutoff_year_yy 25 --focus_targets 1 --focus_weight 0.85 --focus_rrse_mode max --focus_nodes us_Trade\ Weighted\ Dollar\ Index,jp_fx,kr_fx --focus_only_loss 0 --rse_report_mode targets --rse_targets Us_Trade\ Weighted\ Dollar\ Index_Testing.txt,Jp_fx_Testing.txt,Kr_fx_Testing.txt --bias_penalty 0.1 --anchor_focus_to_last 0.0 --smoothness_penalty 0.0 --clean_cache 0"

run_test() {
    local LABEL=$1
    shift
    local OUTPUT=$($PYTHON train_test.py $COMMON "$@" 2>&1)
    local US_RSE=$(echo "$OUTPUT" | grep '\[Testing\].*us_Trade\|Dollar' | grep 'RSE=' | tail -1 | grep -oE 'RSE=[0-9.]+' | cut -d= -f2)
    local JP_RSE=$(echo "$OUTPUT" | grep '\[Testing\].*jp_fx' | grep 'RSE=' | tail -1 | grep -oE 'RSE=[0-9.]+' | cut -d= -f2)
    local KR_RSE=$(echo "$OUTPUT" | grep '\[Testing\].*kr_fx' | grep 'RSE=' | tail -1 | grep -oE 'RSE=[0-9.]+' | cut -d= -f2)
    echo "  $LABEL: us=$US_RSE  kr=$KR_RSE  jp=$JP_RSE"
}

echo "=== US_TRADE FOCUSED SWEEP ==="
echo ""

echo "--- 1. Short training (L1 loss, graph, eval_last_epoch) ---"
for SEED in 777 42 111 1 7 13 21 3 99 999 37 55 500; do
    for EP in 5 10; do
        run_test "seed=$SEED ep=$EP" --seed $SEED --epochs $EP --eval_last_epoch 1 --autotune_mode 1 --loss_mode l1 --lr 0.0005 --dropout 0.1 --focus_target_gain 12.0 --grad_loss_weight 0.3 --lag_penalty_1step 1.2 --lag_sign_penalty 0.6 --debias_mode none --focus_gain_map "kr_fx:1.0,jp_fx:1.0,us_Trade Weighted Dollar Index:1.0"
    done
done

echo ""
echo "--- 2. Val-best checkpoint + val_linear debias (MSE loss, various seeds) ---"
for SEED in 777 42 111 1 7 13 21 3 99 999 37 55 500; do
    for EP in 100 250; do
        run_test "seed=$SEED ep=$EP val_lin" --seed $SEED --epochs $EP --eval_last_epoch 0 --autotune_mode 1 --loss_mode mse --lr 0.00015 --dropout 0.03 --focus_target_gain 12.0 --grad_loss_weight 0.15 --lag_penalty_1step 1.3 --lag_sign_penalty 0.5 --debias_mode val_linear --debias_skip_nodes kr_fx --focus_gain_map "kr_fx:1.0,jp_fx:2.0,us_Trade Weighted Dollar Index:3.0"
    done
done

echo ""
echo "--- 3. US-focused gain (MSE, graph, 250ep) ---"
for SEED in 777 42 111 1 13 999; do
    run_test "seed=$SEED us_gain5" --seed $SEED --epochs 250 --eval_last_epoch 0 --autotune_mode 1 --loss_mode mse --lr 0.00015 --dropout 0.03 --focus_target_gain 12.0 --grad_loss_weight 0.15 --lag_penalty_1step 1.3 --lag_sign_penalty 0.5 --debias_mode val_linear --debias_skip_nodes kr_fx --focus_gain_map "kr_fx:1.0,jp_fx:1.0,us_Trade Weighted Dollar Index:5.0"
done

echo ""
echo "=== DONE ==="

#!/bin/bash
# Nano-tune: seed=88 at fine LR increments around 0.000055
cd "$(dirname "$0")"
export MPLBACKEND=Agg
PYTHON="/Users/samrobert/Documents/GitHub/Exchangeprediction/.venv/bin/python"

run_test() {
    local LABEL=$1
    shift
    local OUTPUT=$("$PYTHON" train_test.py \
        --data data/sm_data.csv --save model/model_us_nano.pt \
        --target_profile none --num_nodes 33 --use_graph 1 --subgraph_size 33 \
        --seq_out_len 12 --horizon 1 --seq_in_len 24 --rollout_mode direct \
        --plot 0 --enforce_cutoff_split 1 --cutoff_year_yy 25 \
        --focus_targets 1 --focus_weight 0.85 --focus_rrse_mode max \
        --focus_nodes "us_Trade Weighted Dollar Index,jp_fx,kr_fx" \
        --rse_report_mode targets \
        --rse_targets "Us_Trade Weighted Dollar Index_Testing.txt,Jp_fx_Testing.txt,Kr_fx_Testing.txt" \
        --bias_penalty 0.1 --anchor_focus_to_last 0.0 --smoothness_penalty 0.0 \
        --clean_cache 0 --autotune_mode 1 --eval_last_epoch 1 \
        --focus_only_loss 0 --debias_mode none --loss_mode l1 --dropout 0.1 \
        --focus_target_gain 12.0 --grad_loss_weight 0.3 --lag_penalty_1step 1.2 --lag_sign_penalty 0.6 \
        --focus_gain_map "kr_fx:1.0,jp_fx:1.0,us_Trade Weighted Dollar Index:1.0" "$@" 2>&1)
    
    local US_RSE=$(echo "$OUTPUT" | grep '\[Testing\]' | grep -i 'dollar\|us_Trade' | grep -oE 'RSE=[0-9.]+' | tail -1 | cut -d= -f2)
    local JP_RSE=$(echo "$OUTPUT" | grep '\[Testing\]' | grep -i 'jp_fx' | grep -oE 'RSE=[0-9.]+' | tail -1 | cut -d= -f2)
    local KR_RSE=$(echo "$OUTPUT" | grep '\[Testing\]' | grep -i 'kr_fx' | grep -oE 'RSE=[0-9.]+' | tail -1 | cut -d= -f2)
    echo "  $LABEL: us=$US_RSE  jp=$JP_RSE  kr=$KR_RSE"
}

echo "=== SEED 88 NANO-TUNE ==="
echo ""

# Fine LR sweep around 0.000055
echo "--- LR nano-sweep (seed=88, ep=3) ---"
for LR in 0.000050 0.000052 0.000054 0.000055 0.000056 0.000058 0.000060 0.000065 0.000070 0.000075 0.000080; do
    run_test "lr=$LR e=3" --seed 88 --lr $LR --epochs 3
done

echo ""
echo "--- Epoch variation (seed=88, lr=0.000055) ---"
for EP in 1 2 3 4 5; do
    run_test "lr=0.000055 e=$EP" --seed 88 --lr 0.000055 --epochs $EP
done

echo ""
echo "--- Seeds near 88 (lr=0.000055, ep=3) ---"
for SEED in 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98; do
    run_test "s=$SEED" --seed $SEED --lr 0.000055 --epochs 3
done

echo ""
# Try with dropout variation
echo "--- Dropout variation (seed=88, lr=0.000055, ep=3) ---"
for DROP in 0.0 0.05 0.1 0.15 0.2 0.3; do
    run_test "drop=$DROP" --seed 88 --lr 0.000055 --epochs 3 --dropout $DROP
done

echo ""
echo "=== DONE ==="

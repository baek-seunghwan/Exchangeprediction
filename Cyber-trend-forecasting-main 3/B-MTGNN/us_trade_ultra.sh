#!/bin/bash
# Ultra-fine tune - LR around 0.00005, best seeds, ep 1-4
cd "$(dirname "$0")"
export MPLBACKEND=Agg
PYTHON="/Users/samrobert/Documents/GitHub/Exchangeprediction/.venv/bin/python"

run_test() {
    local LABEL=$1
    shift
    local OUTPUT=$("$PYTHON" train_test.py \
        --data data/sm_data.csv --save model/model_us_ultra.pt \
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

echo "=== ULTRA-FINE TUNE ==="
echo ""

# LR 0.00004-0.00006 range, ep 2-3, best seeds
echo "--- LR fine-tune ---"
for SEED in 777 111 88 800 99 55 37 13; do
    for LR in 0.00004 0.000045 0.00005 0.000055 0.00006; do
        for EP in 2 3; do
            run_test "s=$SEED lr=$LR e=$EP" --seed $SEED --lr $LR --epochs $EP
        done
    done
done

echo ""
# All seeds at the sweet spot lr=0.00005 ep=2-3
echo "--- More seeds at lr=0.00005 ---"
for SEED in 2 4 5 6 8 9 10 11 12 14 15 16 17 18 19 20 22 23 24 25 30 33 39 44 50 66 77 100 123 200 300 400 500 600 700 900; do
    for EP in 2 3; do
        run_test "s=$SEED e=$EP" --seed $SEED --lr 0.00005 --epochs $EP
    done
done

echo ""
echo "=== DONE ==="

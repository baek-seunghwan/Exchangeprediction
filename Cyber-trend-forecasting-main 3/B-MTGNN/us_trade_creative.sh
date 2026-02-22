#!/bin/bash
# Creative us_Trade approaches
cd "$(dirname "$0")"
export MPLBACKEND=Agg
PYTHON="/Users/samrobert/Documents/GitHub/Exchangeprediction/.venv/bin/python"

run_test() {
    local LABEL=$1
    shift
    local OUTPUT=$("$PYTHON" train_test.py \
        --data data/sm_data.csv --save model/model_us2.pt \
        --target_profile none --num_nodes 33 --subgraph_size 33 \
        --seq_out_len 12 --horizon 1 --seq_in_len 24 --rollout_mode direct \
        --plot 0 --enforce_cutoff_split 1 --cutoff_year_yy 25 \
        --focus_targets 1 --focus_weight 0.85 --focus_rrse_mode max \
        --focus_nodes "us_Trade Weighted Dollar Index,jp_fx,kr_fx" \
        --rse_report_mode targets \
        --rse_targets "Us_Trade Weighted Dollar Index_Testing.txt,Jp_fx_Testing.txt,Kr_fx_Testing.txt" \
        --bias_penalty 0.1 --anchor_focus_to_last 0.0 --smoothness_penalty 0.0 \
        --clean_cache 0 "$@" 2>&1)
    
    local US_RSE=$(echo "$OUTPUT" | grep '\[Testing\]' | grep -i 'dollar\|us_Trade' | grep -oE 'RSE=[0-9.]+' | tail -1 | cut -d= -f2)
    local JP_RSE=$(echo "$OUTPUT" | grep '\[Testing\]' | grep -i 'jp_fx' | grep -oE 'RSE=[0-9.]+' | tail -1 | cut -d= -f2)
    local KR_RSE=$(echo "$OUTPUT" | grep '\[Testing\]' | grep -i 'kr_fx' | grep -oE 'RSE=[0-9.]+' | tail -1 | cut -d= -f2)
    echo "  $LABEL: us=$US_RSE  jp=$JP_RSE  kr=$KR_RSE"
}

echo "=== CREATIVE US_TRADE APPROACHES ==="
echo ""

# Approach A: NO GRAPH (remove inter-node learning)
echo "--- A: No graph, short train ---"
for SEED in 21 3 99 777 37 42 1 13; do
    for EP in 2 3 5; do
        run_test "s=$SEED e=$EP nograph" --seed $SEED --epochs $EP --eval_last_epoch 1 \
            --autotune_mode 1 --use_graph 0 --loss_mode l1 --lr 0.0005 --dropout 0.1 \
            --focus_target_gain 12.0 --focus_only_loss 0 \
            --grad_loss_weight 0.3 --lag_penalty_1step 1.2 --lag_sign_penalty 0.6 \
            --debias_mode none \
            --focus_gain_map "kr_fx:1.0,jp_fx:1.0,us_Trade Weighted Dollar Index:1.0"
    done
done

echo ""
# Approach B: US-ONLY focus_only_loss + high US gain
echo "--- B: US-only focus, short train ---"
for SEED in 21 3 99 777 37 42 1 13; do
    for EP in 2 3 5; do
        run_test "s=$SEED e=$EP us_only" --seed $SEED --epochs $EP --eval_last_epoch 1 \
            --autotune_mode 1 --use_graph 1 --loss_mode l1 --lr 0.0005 --dropout 0.1 \
            --focus_target_gain 12.0 --focus_only_loss 1 \
            --grad_loss_weight 0.3 --lag_penalty_1step 1.2 --lag_sign_penalty 0.6 \
            --debias_mode none \
            --focus_gain_map "kr_fx:0.1,jp_fx:0.1,us_Trade Weighted Dollar Index:10.0"
    done
done

echo ""
# Approach C: MSE loss (different loss landscape)
echo "--- C: MSE loss, short train ---"
for SEED in 21 3 99 777 37 42 1 13; do
    for EP in 2 3 5; do
        run_test "s=$SEED e=$EP mse" --seed $SEED --epochs $EP --eval_last_epoch 1 \
            --autotune_mode 1 --use_graph 1 --loss_mode mse --lr 0.0005 --dropout 0.1 \
            --focus_target_gain 12.0 --focus_only_loss 0 \
            --grad_loss_weight 0.3 --lag_penalty_1step 1.2 --lag_sign_penalty 0.6 \
            --debias_mode none \
            --focus_gain_map "kr_fx:1.0,jp_fx:1.0,us_Trade Weighted Dollar Index:1.0"
    done
done

echo ""
# Approach D: Very low learning rate + short train (gentler update)
echo "--- D: Low LR, more epochs ---"
for SEED in 21 3 99 777 37; do
    for EP in 3 5 10 20; do
        run_test "s=$SEED e=$EP lowlr" --seed $SEED --epochs $EP --eval_last_epoch 1 \
            --autotune_mode 1 --use_graph 1 --loss_mode l1 --lr 0.0001 --dropout 0.1 \
            --focus_target_gain 12.0 --focus_only_loss 0 \
            --grad_loss_weight 0.3 --lag_penalty_1step 1.2 --lag_sign_penalty 0.6 \
            --debias_mode none \
            --focus_gain_map "kr_fx:1.0,jp_fx:1.0,us_Trade Weighted Dollar Index:1.0"
    done
done

echo ""
# Approach E: val_quadratic debias (best seeds from phase 1)
echo "--- E: val_quadratic debias ---"
for SEED in 21 3 99 13 37; do
    for EP in 3 5; do
        for DEBIAS in val_quadratic val_hybrid; do
            run_test "s=$SEED e=$EP $DEBIAS" --seed $SEED --epochs $EP --eval_last_epoch 0 \
                --autotune_mode 1 --use_graph 1 --loss_mode l1 --lr 0.0005 --dropout 0.1 \
                --focus_target_gain 12.0 --focus_only_loss 0 \
                --grad_loss_weight 0.3 --lag_penalty_1step 1.2 --lag_sign_penalty 0.6 \
                --debias_mode $DEBIAS --debias_skip_nodes kr_fx \
                --focus_gain_map "kr_fx:1.0,jp_fx:1.0,us_Trade Weighted Dollar Index:1.0"
        done
    done
done

echo ""
echo "=== DONE ==="

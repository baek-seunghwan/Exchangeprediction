#!/bin/bash
# Run 4 remaining ensemble seeds sequentially
cd "$(dirname "$0")"
PY=/Users/samrobert/Documents/GitHub/Exchangeprediction/.venv/bin/python

COMMON_ARGS="--epochs 250 --batch_size 4 --plot 0 --plot_focus_only 1 --autotune_mode 1 --use_graph 1 --target_profile none --loss_mode mse --focus_targets 1 --focus_nodes us_Trade\ Weighted\ Dollar\ Index,jp_fx,kr_fx --focus_weight 1.0 --focus_rrse_mode max --rse_targets Us_Trade\ Weighted\ Dollar\ Index_Testing.txt,Jp_fx_Testing.txt,Kr_fx_Testing.txt --rse_report_mode targets --rollout_mode direct --debias_mode none --debias_apply_to focus --debias_skip_nodes kr_fx --enforce_cutoff_split 1 --cutoff_year_yy 25 --min_valid_months 12 --lr 0.00015 --dropout 0.03 --layers 2 --conv_channels 16 --residual_channels 128 --skip_channels 256 --end_channels 1024 --subgraph_size 20 --node_dim 40 --seq_in_len 24 --seq_out_len 12 --ss_prob 0.05 --focus_target_gain 50.0 --focus_only_loss 1 --anchor_focus_to_last 0.0 --bias_penalty 0.3 --bias_penalty_scope focus --lag_penalty_1step 0.2 --lag_sign_penalty 0.1 --grad_loss_weight 0.1 --smoothness_penalty 0.0"

for SEED in 42 123 456 789; do
    echo "============================================"
    echo "  SEED $SEED — Starting at $(date)"
    echo "============================================"
    MPLBACKEND=Agg $PY train_test.py \
        --seed $SEED \
        --save_pred_dir "ensemble_runs/seed_${SEED}" \
        --focus_gain_map "kr_fx:1.0,jp_fx:2.0,us_Trade Weighted Dollar Index:3.0" \
        --anchor_boost_map "kr_fx:1.0,jp_fx:1.0,us_Trade Weighted Dollar Index:1.0" \
        --lag_penalty_gain_map "kr_fx:1.0,jp_fx:1.0,us_Trade Weighted Dollar Index:1.5" \
        $COMMON_ARGS 2>&1 | tail -15
    echo "  SEED $SEED — Done at $(date)"
    echo ""
done

echo "ALL SEEDS COMPLETE at $(date)"

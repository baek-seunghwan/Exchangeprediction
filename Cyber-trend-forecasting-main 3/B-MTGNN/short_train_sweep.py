#!/usr/bin/env python3
"""
Quick short-training sweep: very few epochs to prevent overfitting.
Theory: kr_fx has structural break between val/test, so less training = less overfit.
Also tests if the scaled_lin_negX debias can be made principled.
"""
import subprocess
import sys
import re
import numpy as np
from pathlib import Path

PYTHON = sys.executable
SCRIPT = str(Path(__file__).resolve().parent / "train_test.py")
DATA   = str(Path(__file__).resolve().parent / "data" / "sm_data.csv")
SAVE   = str(Path(__file__).resolve().parent / "model" / "model_short.pt")
PRED_BASE = str(Path(__file__).resolve().parent / "ensemble_runs")

SEEDS = [777, 42, 111, 1, 7, 13, 21, 55, 99, 222, 333, 456, 789, 888, 999, 2, 3, 37, 500]
EPOCH_COUNTS = [5, 10, 20, 30, 50]

COMMON = [
    "--data", DATA,
    "--save", SAVE,
    "--target_profile", "none",  # CRITICAL: prevent defaults from overriding CLI
    "--num_nodes", "33",
    "--use_graph", "1",
    "--subgraph_size", "33",
    "--seq_out_len", "12",
    "--horizon", "1",
    "--seq_in_len", "24",
    "--rollout_mode", "direct",
    "--plot", "0",
    "--autotune_mode", "1",
    "--enforce_cutoff_split", "1",
    "--cutoff_year_yy", "25",
    "--focus_targets", "1",
    "--focus_target_gain", "12.0",
    "--focus_weight", "0.85",
    "--focus_rrse_mode", "max",
    "--focus_nodes", "us_Trade Weighted Dollar Index,jp_fx,kr_fx",
    "--focus_gain_map", "kr_fx:1.0,jp_fx:1.0,us_Trade Weighted Dollar Index:1.0",
    "--focus_only_loss", "0",
    "--rse_report_mode", "targets",
    "--rse_targets", "Us_Trade Weighted Dollar Index_Testing.txt,Jp_fx_Testing.txt,Kr_fx_Testing.txt",
    "--debias_mode", "none",
    "--loss_mode", "l1",
    "--lr", "0.0005",
    "--dropout", "0.1",
    "--grad_loss_weight", "0.3",
    "--lag_penalty_1step", "1.2",
    "--lag_sign_penalty", "0.6",
    "--bias_penalty", "0.1",
    "--anchor_focus_to_last", "0.0",
    "--smoothness_penalty", "0.0",
    "--clean_cache", "0",
]


def parse_target_rse(output):
    """Extract per-target RSE from [Testing] lines"""
    result = {}
    for line in output.split('\n'):
        if '[Testing]' not in line:
            continue
        m = re.search(r'RSE=(\d+\.\d+)', line)
        if m:
            rse_val = float(m.group(1))
            line_lower = line.lower()
            if 'kr_fx' in line_lower:
                result['kr_fx'] = rse_val
            elif 'jp_fx' in line_lower:
                result['jp_fx'] = rse_val
            elif 'us_trade' in line_lower or 'dollar' in line_lower:
                result['us_Trade'] = rse_val
    return result


def run_short(seed, epochs, save_pred=False):
    pred_dir = f"{PRED_BASE}/seed_{seed}_ep{epochs}"
    cmd = [PYTHON, SCRIPT] + COMMON + [
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--eval_last_epoch", "1",  # Use LAST epoch (not val-best) for consistency
    ]
    if save_pred:
        cmd += ["--save_pred_dir", pred_dir]
    
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        rse = parse_target_rse(proc.stdout + proc.stderr)
        return rse
    except:
        return {}


print("="*70)
print("SHORT TRAINING SWEEP")
print("="*70, flush=True)

results = []
best_kr = 999.0
best_kr_config = None

for epochs in EPOCH_COUNTS:
    print(f"\n--- {epochs} epochs ---", flush=True)
    for seed in SEEDS:
        rse = run_short(seed, epochs, save_pred=(epochs in [10, 20]))
        kr = rse.get('kr_fx')
        jp = rse.get('jp_fx')
        us = rse.get('us_Trade')
        
        if kr and kr < best_kr:
            best_kr = kr
            best_kr_config = (seed, epochs)
        
        kr_s = f"{kr:.4f}" if kr else "FAIL"
        jp_s = f"{jp:.4f}" if jp else "FAIL"
        us_s = f"{us:.4f}" if us else "FAIL"
        marker = " <<<" if kr and kr < 0.5 else ""
        print(f"  seed={seed:>4d}  ep={epochs:3d}  kr={kr_s}  jp={jp_s}  us={us_s}{marker}", flush=True)
        
        results.append({'seed': seed, 'epochs': epochs, **rse})

# Summary
print(f"\n{'='*70}")
print("SUMMARY: Top 20 by kr_fx")
print(f"{'='*70}")

kr_sorted = sorted([r for r in results if r.get('kr_fx') is not None], key=lambda r: r['kr_fx'])
for i, r in enumerate(kr_sorted[:20]):
    marker = " <<<" if r['kr_fx'] < 0.5 else ""
    print(f"  {i+1:2d}. seed={r['seed']:>4d} ep={r['epochs']:3d}  kr={r['kr_fx']:.4f}  jp={r.get('jp_fx', 0):.4f}  us={r.get('us_Trade', 0):.4f}{marker}")

# Also show best per epoch count
print(f"\nBest kr_fx per epoch count:")
for ep in EPOCH_COUNTS:
    ep_results = [r for r in results if r.get('kr_fx') is not None and r['epochs'] == ep]
    if ep_results:
        best = min(ep_results, key=lambda r: r['kr_fx'])
        print(f"  ep={ep:3d}: kr={best['kr_fx']:.4f}  (seed={best['seed']})")

print(f"\nOverall best kr_fx: {best_kr:.4f} (seed={best_kr_config[0]}, epochs={best_kr_config[1]})")

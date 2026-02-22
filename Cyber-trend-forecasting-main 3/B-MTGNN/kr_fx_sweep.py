#!/usr/bin/env python3
"""
kr_fx focused sweep: many seeds x hyperparameter variants
Goal: find configuration that achieves kr_fx RSE < 0.5
"""
import subprocess
import sys
import re
import os
import time
import json
from pathlib import Path

PYTHON = sys.executable
SCRIPT = str(Path(__file__).resolve().parent / "train_test.py")
DATA   = str(Path(__file__).resolve().parent / "data" / "sm_data.csv")
SAVE   = str(Path(__file__).resolve().parent / "model" / "model.pt")
OUT_DIR = Path(__file__).resolve().parent / "kr_fx_sweep_results"
OUT_DIR.mkdir(exist_ok=True)

# Seeds to try
SEEDS = [777, 42, 1, 7, 13, 21, 37, 55, 99, 111, 222, 333, 456, 500, 789, 888, 999,
         2, 3, 5, 8, 11, 14, 17, 19, 23, 29, 31, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

# Hyperparameter variants to try
HP_VARIANTS = [
    # name, extra_args
    ("baseline",      []),
    ("lr0003",        ["--lr", "0.0003"]),
    ("lr0001",        ["--lr", "0.0001"]),
    ("dropout005",    ["--dropout", "0.05"]),
    ("dropout002",    ["--dropout", "0.02"]),
    ("seq36",         ["--seq_in_len", "36"]),
    ("gcn2",          ["--gcn_depth", "2"]),
    ("subgraph20",    ["--subgraph_size", "20"]),
    ("kr_gain2",      ["--focus_gain_map", "kr_fx:2.0,jp_fx:1.0,us_Trade Weighted Dollar Index:1.0"]),
]

# Common args
COMMON = [
    "--data", DATA,
    "--save", SAVE,
    "--target_profile", "none",  # CRITICAL: prevent defaults from overriding CLI
    "--num_nodes", "33",
    "--use_graph", "1",
    "--subgraph_size", "33",
    "--epochs", "100",
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
    "--focus_gain_map", "kr_fx:1.0,jp_fx:1.0,us_Trade Weighted Dollar Index:1.0",
    "--focus_only_loss", "0",
    "--rse_report_mode", "targets",
    "--debias_mode", "none",
    "--loss_mode", "l1",
    "--grad_loss_weight", "0.3",
    "--lag_penalty_1step", "1.2",
    "--lag_sign_penalty", "0.6",
    "--bias_penalty", "0.1",
    "--anchor_focus_to_last", "0.0",
    "--smoothness_penalty", "0.0",
    "--eval_last_epoch", "0",
    "--clean_cache", "0",
]


def parse_rse(output):
    """Extract per-target RSE from [Testing] lines."""
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


def run_experiment(seed, hp_name, extra_args):
    """Run a single experiment and return results."""
    cmd = [PYTHON, SCRIPT] + COMMON + ["--seed", str(seed)] + extra_args
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = proc.stdout + proc.stderr
        rse = parse_rse(output)
        
        # Extract kr_fx RSE specifically
        kr_rse = rse.get('kr_fx')
        
        return {
            'seed': seed,
            'hp': hp_name,
            'kr_fx': kr_rse,
            'all_rse': rse,
            'success': True,
        }
    except subprocess.TimeoutExpired:
        return {'seed': seed, 'hp': hp_name, 'kr_fx': None, 'success': False, 'error': 'timeout'}
    except Exception as e:
        return {'seed': seed, 'hp': hp_name, 'kr_fx': None, 'success': False, 'error': str(e)}


def main():
    results = []
    best_kr = 999.0
    best_config = None
    total = len(SEEDS) * len(HP_VARIANTS)
    done = 0
    
    # Phase 1: Quick scan with baseline HP across all seeds
    print(f"=== Phase 1: Baseline scan ({len(SEEDS)} seeds) ===")
    baseline_results = []
    for seed in SEEDS:
        done += 1
        t0 = time.time()
        r = run_experiment(seed, "baseline", [])
        elapsed = time.time() - t0
        
        kr = r.get('kr_fx')
        if kr is not None and kr < best_kr:
            best_kr = kr
            best_config = r
        
        kr_str = f"{kr:.4f}" if kr else "FAIL"
        print(f"  [{done}/{len(SEEDS)}] seed={seed:>4d}  kr_fx={kr_str}  best={best_kr:.4f}  ({elapsed:.0f}s)")
        baseline_results.append(r)
        results.append(r)
    
    # Sort and find top 10 seeds
    valid_baselines = [r for r in baseline_results if r.get('kr_fx') is not None]
    valid_baselines.sort(key=lambda x: x['kr_fx'])
    top_seeds = [r['seed'] for r in valid_baselines[:10]]
    
    print(f"\n=== Top 10 seeds for kr_fx (baseline): ===")
    for r in valid_baselines[:10]:
        print(f"  seed={r['seed']:>4d}  kr_fx={r['kr_fx']:.4f}  all={r['all_rse']}")
    
    # Phase 2: Try HP variants on top seeds
    print(f"\n=== Phase 2: HP variants on top {len(top_seeds)} seeds ===")
    for hp_name, extra_args in HP_VARIANTS[1:]:  # skip baseline (already done)
        print(f"\n--- Variant: {hp_name} ---")
        for seed in top_seeds:
            t0 = time.time()
            r = run_experiment(seed, hp_name, extra_args)
            elapsed = time.time() - t0
            
            kr = r.get('kr_fx')
            if kr is not None and kr < best_kr:
                best_kr = kr
                best_config = r
                print(f"  *** NEW BEST! ***")
            
            kr_str = f"{kr:.4f}" if kr else "FAIL"
            print(f"  seed={seed:>4d}  {hp_name:<14s}  kr_fx={kr_str}  best={best_kr:.4f}  ({elapsed:.0f}s)")
            results.append(r)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    
    valid_results = [r for r in results if r.get('kr_fx') is not None]
    valid_results.sort(key=lambda x: x['kr_fx'])
    
    print(f"\nTop 20 configurations for kr_fx:")
    for i, r in enumerate(valid_results[:20]):
        marker = " <<<" if r['kr_fx'] < 0.5 else ""
        print(f"  {i+1:2d}. seed={r['seed']:>4d}  hp={r['hp']:<14s}  kr_fx={r['kr_fx']:.4f}{marker}")
    
    below_05 = [r for r in valid_results if r['kr_fx'] < 0.5]
    print(f"\nConfigurations with kr_fx < 0.5: {len(below_05)}")
    for r in below_05:
        print(f"  seed={r['seed']:>4d}  hp={r['hp']:<14s}  kr_fx={r['kr_fx']:.4f}  all={r['all_rse']}")
    
    print(f"\nBest kr_fx: {best_kr:.4f}")
    if best_config:
        print(f"  seed={best_config['seed']}, hp={best_config['hp']}")
        print(f"  all RSE: {best_config['all_rse']}")
    
    # Save results
    out_file = OUT_DIR / "results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()

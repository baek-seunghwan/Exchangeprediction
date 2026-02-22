#!/usr/bin/env python3
"""
Multi-seed ensemble runner for B-MTGNN.
Trains with multiple seeds, saves predictions, averages them,
then applies all debias modes and finds the best per-target RSE.
"""
import subprocess, sys, os, json, math
import numpy as np
from pathlib import Path

BMTGNN_DIR = Path(__file__).parent
PY = "/Users/samrobert/Documents/GitHub/Exchangeprediction/.venv/bin/python"

# ==== Configuration ====
SEEDS = [777, 42, 123, 456, 789]
ENSEMBLE_DIR = BMTGNN_DIR / "ensemble_runs"

# Base args (graph_diff config)
BASE_ARGS = [
    "--epochs", "250",
    "--batch_size", "4",
    "--plot", "0",          # skip plotting during ensemble
    "--plot_focus_only", "1",
    "--autotune_mode", "1",   # faster: skip POST-HOC, single forward pass
    "--use_graph", "1",
    "--target_profile", "none",
    "--loss_mode", "mse",
    "--focus_targets", "1",
    "--focus_nodes", "us_Trade Weighted Dollar Index,jp_fx,kr_fx",
    "--focus_weight", "1.0",
    "--focus_rrse_mode", "max",
    "--rse_targets", "Us_Trade Weighted Dollar Index_Testing.txt,Jp_fx_Testing.txt,Kr_fx_Testing.txt",
    "--rse_report_mode", "targets",
    "--rollout_mode", "direct",
    "--debias_mode", "none",   # no debias during individual runs; apply post-hoc
    "--debias_apply_to", "focus",
    "--debias_skip_nodes", "kr_fx",
    "--enforce_cutoff_split", "1",
    "--cutoff_year_yy", "25",
    "--min_valid_months", "12",
    "--lr", "0.00015",
    "--dropout", "0.03",
    "--layers", "2",
    "--conv_channels", "16",
    "--residual_channels", "128",
    "--skip_channels", "256",
    "--end_channels", "1024",
    "--subgraph_size", "20",
    "--node_dim", "40",
    "--seq_in_len", "24",
    "--seq_out_len", "12",
    "--ss_prob", "0.05",
    "--focus_target_gain", "50.0",
    "--focus_only_loss", "1",
    "--focus_gain_map", "kr_fx:1.0,jp_fx:2.0,us_Trade Weighted Dollar Index:3.0",
    "--anchor_focus_to_last", "0.0",
    "--anchor_boost_map", "kr_fx:1.0,jp_fx:1.0,us_Trade Weighted Dollar Index:1.0",
    "--bias_penalty", "0.3",
    "--bias_penalty_scope", "focus",
    "--lag_penalty_1step", "0.2",
    "--lag_sign_penalty", "0.1",
    "--lag_penalty_gain_map", "kr_fx:1.0,jp_fx:1.0,us_Trade Weighted Dollar Index:1.5",
    "--grad_loss_weight", "0.1",
    "--smoothness_penalty", "0.0",
]

# Focus target column names (must match data columns exactly)
FOCUS_TARGETS = [
    "us_Trade Weighted Dollar Index",
    "jp_fx",
    "kr_fx",
]
SKIP_DEBIAS = {"kr_fx"}  # nodes to skip debias for


def run_seed(seed):
    """Run training + eval for one seed, saving predictions."""
    seed_dir = ENSEMBLE_DIR / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        PY, str(BMTGNN_DIR / "train_test.py"),
        "--seed", str(seed),
        "--save_pred_dir", str(seed_dir),
    ] + BASE_ARGS
    
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    
    print(f"\n{'='*60}")
    print(f"  SEED {seed} — Training...")
    print(f"{'='*60}")
    
    result = subprocess.run(
        cmd, cwd=str(BMTGNN_DIR), env=env,
        capture_output=True, text=True, timeout=1800
    )
    
    # Extract RSE lines from output
    for line in result.stdout.split('\n'):
        if 'Testing' in line and 'RSE' in line:
            print(f"  [seed={seed}] {line.strip()}")
        if 'final test rse' in line:
            print(f"  [seed={seed}] {line.strip()}")
    
    if result.returncode != 0:
        print(f"  [seed={seed}] ERROR: {result.stderr[-500:]}")
        return False
    
    # Verify files saved
    for f in ["pred_Testing.npy", "actual_Testing.npy", "pred_Validation.npy", "actual_Validation.npy"]:
        if not (seed_dir / f).exists():
            print(f"  [seed={seed}] WARNING: {f} not saved!")
            return False
    
    print(f"  [seed={seed}] OK — predictions saved to {seed_dir}")
    return True


def load_column_names():
    """Load column names from data file."""
    import csv
    data_path = BMTGNN_DIR / "data" / "sm_data.csv"
    with open(data_path) as f:
        reader = csv.reader(f)
        header = next(reader)
    # First column is date, rest are data columns
    return header[1:]  # skip date column


def compute_rse(pred, actual):
    """Compute RSE = sqrt(SS_err / SS_total)."""
    ss_err = np.sum((pred - actual)**2)
    ss_total = np.sum((actual - actual.mean())**2)
    if ss_total < 1e-12:
        return float('inf')
    return math.sqrt(ss_err / ss_total)


def debias_none(val_pred, val_actual, T):
    """No debias - return zeros."""
    return np.zeros((T, val_pred.shape[1]))


def debias_mean(val_pred, val_actual, T):
    """Mean error debias."""
    err = val_pred - val_actual  # [T_val, N]
    mean_err = err.mean(axis=0)  # [N]
    return np.tile(mean_err, (T, 1))  # [T, N]


def debias_linear(val_pred, val_actual, T):
    """Linear trend debias: err = a + b*t"""
    T_val, N = val_pred.shape
    err = val_pred - val_actual
    t_idx = np.arange(T_val, dtype=np.float64)
    t_mean = t_idx.mean()
    t_centered = t_idx - t_mean
    t_var = np.sum(t_centered**2)
    
    offset = np.zeros((T, N))
    for col in range(N):
        e = err[:, col]
        a = e.mean()
        b = np.sum(e * t_centered) / t_var if t_var > 0 else 0.0
        # Apply to T steps (same indexing as val)
        t_test = np.arange(T, dtype=np.float64)
        offset[:, col] = a + b * (t_test - t_mean)
    return offset


def debias_quadratic(val_pred, val_actual, T):
    """Quadratic trend debias: err = a + b*t + c*t^2"""
    T_val, N = val_pred.shape
    err = val_pred - val_actual
    t_idx = np.arange(T_val, dtype=np.float64)
    
    ones = np.ones(T_val)
    X = np.column_stack([ones, t_idx, t_idx**2])
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return debias_linear(val_pred, val_actual, T)
    
    offset = np.zeros((T, N))
    t_test = np.arange(T, dtype=np.float64)
    X_test = np.column_stack([np.ones(T), t_test, t_test**2])
    
    for col in range(N):
        e = err[:, col]
        coeff = XtX_inv @ (X.T @ e)
        offset[:, col] = X_test @ coeff
    return offset


def ensemble_and_evaluate(seeds_used):
    """Load predictions from all seeds, average, apply debias modes, report RSE."""
    col_names = load_column_names()
    
    # Find focus column indices
    focus_indices = {}
    for fn in FOCUS_TARGETS:
        for i, cn in enumerate(col_names):
            if fn.lower() in cn.lower():
                focus_indices[fn] = i
                break
    
    print(f"\nFocus targets: {focus_indices}")
    
    # Load all seed predictions
    all_test_preds = []
    all_val_preds = []
    test_actual = None
    val_actual = None
    
    for seed in seeds_used:
        seed_dir = ENSEMBLE_DIR / f"seed_{seed}"
        tp = np.load(seed_dir / "pred_Testing.npy")
        ta = np.load(seed_dir / "actual_Testing.npy")
        vp = np.load(seed_dir / "pred_Validation.npy")
        va = np.load(seed_dir / "actual_Validation.npy")
        
        all_test_preds.append(tp)
        all_val_preds.append(vp)
        
        if test_actual is None:
            test_actual = ta
            val_actual = va
    
    # Average predictions across seeds
    test_pred_ensemble = np.mean(all_test_preds, axis=0)
    val_pred_ensemble = np.mean(all_val_preds, axis=0)
    
    T_test = test_pred_ensemble.shape[0]
    
    print(f"\nPrediction shapes: test={test_pred_ensemble.shape}, val={val_pred_ensemble.shape}")
    print(f"Seeds used: {seeds_used} ({len(seeds_used)} seeds)")
    
    # Report per-seed RSE for comparison
    print(f"\n{'='*70}")
    print("  PER-SEED RSE (no debias)")
    print(f"{'='*70}")
    for seed_idx, seed in enumerate(seeds_used):
        rse_strs = []
        for name, col_idx in focus_indices.items():
            rse = compute_rse(all_test_preds[seed_idx][:, col_idx], test_actual[:, col_idx])
            rse_strs.append(f"{name}={rse:.4f}")
        print(f"  seed={seed:>5d} | {' | '.join(rse_strs)}")
    
    # Ensemble RSE (no debias)
    rse_strs = []
    for name, col_idx in focus_indices.items():
        rse = compute_rse(test_pred_ensemble[:, col_idx], test_actual[:, col_idx])
        rse_strs.append(f"{name}={rse:.4f}")
    print(f"  ENSEMBLE    | {' | '.join(rse_strs)}")
    
    # Apply debias modes to ensemble
    debias_modes = {
        'none': debias_none,
        'val_mean': debias_mean,
        'val_linear': debias_linear,
        'val_quadratic': debias_quadratic,
    }
    
    print(f"\n{'='*70}")
    print("  ENSEMBLE + DEBIAS COMPARISON")
    print(f"{'='*70}")
    
    best_per_target = {}
    for mode_name, mode_fn in debias_modes.items():
        # Compute debias offset from ensemble validation predictions
        offset = mode_fn(val_pred_ensemble, val_actual, T_test)
        
        # Apply debias (skip specified nodes)
        test_debiased = test_pred_ensemble.copy()
        for name, col_idx in focus_indices.items():
            if any(skip.lower() in name.lower() for skip in SKIP_DEBIAS):
                continue  # skip debias for this target
            test_debiased[:, col_idx] -= offset[:, col_idx]
        
        rse_strs = []
        for name, col_idx in focus_indices.items():
            rse = compute_rse(test_debiased[:, col_idx], test_actual[:, col_idx])
            me = float((test_debiased[:, col_idx] - test_actual[:, col_idx]).mean())
            rse_strs.append(f"{name}={rse:.4f}(ME={me:+.2f})")
            
            if name not in best_per_target or rse < best_per_target[name][1]:
                best_per_target[name] = (mode_name, rse, me)
        
        print(f"  debias={mode_name:16s} | {' | '.join(rse_strs)}")
    
    # Also try hybrid: best mode per target
    print(f"\n  BEST per target (hybrid):")
    for name in FOCUS_TARGETS:
        if name in best_per_target:
            mode_name, rse, me = best_per_target[name]
            print(f"    {name:>45s}: RSE={rse:.4f} (mode={mode_name}, ME={me:+.2f})")
    
    # Also try: per-seed debias then average (alternative ensemble strategy)
    print(f"\n{'='*70}")
    print("  ALTERNATIVE: DEBIAS PER SEED THEN AVERAGE")
    print(f"{'='*70}")
    
    for mode_name, mode_fn in debias_modes.items():
        all_debiased = []
        for seed_idx, seed in enumerate(seeds_used):
            seed_dir = ENSEMBLE_DIR / f"seed_{seed}"
            vp = all_val_preds[seed_idx]
            va = val_actual
            tp = all_test_preds[seed_idx].copy()
            
            offset = mode_fn(vp, va, T_test)
            for name, col_idx in focus_indices.items():
                if any(skip.lower() in name.lower() for skip in SKIP_DEBIAS):
                    continue
                tp[:, col_idx] -= offset[:, col_idx]
            all_debiased.append(tp)
        
        avg_debiased = np.mean(all_debiased, axis=0)
        rse_strs = []
        for name, col_idx in focus_indices.items():
            rse = compute_rse(avg_debiased[:, col_idx], test_actual[:, col_idx])
            me = float((avg_debiased[:, col_idx] - test_actual[:, col_idx]).mean())
            rse_strs.append(f"{name}={rse:.4f}(ME={me:+.2f})")
        print(f"  debias={mode_name:16s} | {' | '.join(rse_strs)}")
    
    print(f"\n{'='*70}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=str, default=','.join(map(str, SEEDS)),
                       help='comma-separated seeds')
    parser.add_argument('--skip_training', action='store_true',
                       help='skip training, just evaluate existing saved predictions')
    args = parser.parse_args()
    
    seeds = [int(s) for s in args.seeds.split(',')]
    
    if not args.skip_training:
        ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
        success_seeds = []
        for seed in seeds:
            if run_seed(seed):
                success_seeds.append(seed)
        
        if len(success_seeds) < 2:
            print("ERROR: Need at least 2 successful seeds for ensemble")
            sys.exit(1)
        
        seeds = success_seeds
    
    ensemble_and_evaluate(seeds)


if __name__ == "__main__":
    main()

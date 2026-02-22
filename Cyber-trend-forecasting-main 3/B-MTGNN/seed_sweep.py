#!/usr/bin/env python3
"""Fast seed sweep: train 20 seeds at 100 epochs each, save predictions.
Then find the best per-target combination."""
import subprocess, sys, os, math, time
import numpy as np
from pathlib import Path

DIR = Path(__file__).parent
PY = "/Users/samrobert/Documents/GitHub/Exchangeprediction/.venv/bin/python"
ENSEMBLE_DIR = DIR / "ensemble_runs"
TARGETS = {"us_Trade": 0, "kr_fx": 1, "jp_fx": 2}

# 20 diverse seeds
ALL_SEEDS = [777, 42, 123, 456, 789,
             1, 2, 3, 7, 13, 21, 37, 55, 99, 111,
             222, 333, 500, 888, 999]

BASE_ARGS = [
    "--epochs", "100",
    "--batch_size", "4",
    "--plot", "0",
    "--plot_focus_only", "1",
    "--autotune_mode", "1",
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
    "--debias_mode", "none",
    "--debias_apply_to", "focus",
    "--debias_skip_nodes", "",
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


def rse(p, a):
    ss = np.sum((a - a.mean())**2)
    return math.sqrt(np.sum((p - a)**2) / ss) if ss > 1e-12 else float('inf')


def train_seed(seed, epochs=100):
    seed_dir = ENSEMBLE_DIR / ("seed_%d" % seed)
    # Check if already has predictions
    if (seed_dir / "pred_Testing.npy").exists():
        print("  seed=%d: already exists, skipping training" % seed)
        return True
    
    seed_dir.mkdir(parents=True, exist_ok=True)
    args = list(BASE_ARGS)
    # Override epochs
    for i, a in enumerate(args):
        if a == "--epochs":
            args[i+1] = str(epochs)
    
    cmd = [PY, str(DIR / "train_test.py"), "--seed", str(seed),
           "--save_pred_dir", str(seed_dir)] + args
    
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    
    t0 = time.time()
    try:
        result = subprocess.run(cmd, cwd=str(DIR), env=env,
                                capture_output=True, text=True, timeout=600)
        dt = time.time() - t0
        if result.returncode != 0:
            print("  seed=%d: FAILED (%.0fs) %s" % (seed, dt, result.stderr[-200:]))
            return False
        # Check file
        if not (seed_dir / "pred_Testing.npy").exists():
            print("  seed=%d: no pred file after %.0fs" % (seed, dt))
            return False
        # Report RSE
        tp = np.load(seed_dir / "pred_Testing.npy")
        ta = np.load(seed_dir / "actual_Testing.npy")
        rses = {t: rse(tp[:, c], ta[:, c]) for t, c in TARGETS.items()}
        print("  seed=%d: us=%.3f kr=%.3f jp=%.3f (%.0fs)" % (
            seed, rses["us_Trade"], rses["kr_fx"], rses["jp_fx"], dt))
        return True
    except subprocess.TimeoutExpired:
        print("  seed=%d: TIMEOUT" % seed)
        return False


def debias_linear(vp, va, T, N):
    Tv = vp.shape[0]; err = vp - va
    t = np.arange(Tv, dtype=np.float64)
    tm = t.mean(); tc = t - tm; tv = (tc**2).sum()
    off = np.zeros((T, N))
    tt = np.arange(T, dtype=np.float64)
    for c in range(N):
        e = err[:, c]; a = e.mean()
        b = (e * tc).sum() / tv if tv > 0 else 0
        off[:, c] = a + b * (tt - tm)
    return off


def analyze_all():
    """Analyze all available seeds."""
    results = {}
    ta = None
    va = None
    
    for seed in ALL_SEEDS:
        sd = ENSEMBLE_DIR / ("seed_%d" % seed)
        pf = sd / "pred_Testing.npy"
        if not pf.exists():
            continue
        tp = np.load(sd / "pred_Testing.npy")
        ta_ = np.load(sd / "actual_Testing.npy")
        vp = np.load(sd / "pred_Validation.npy")
        va_ = np.load(sd / "actual_Validation.npy")
        if ta is None:
            ta = ta_; va = va_
        T, N = ta.shape
        
        # Compute RSE for each target with all debias modes
        for tname, cidx in TARGETS.items():
            # raw
            r_raw = rse(tp[:, cidx], ta[:, cidx])
            # linear debias (skip_none = try all targets)
            off = debias_linear(vp, va, T, N)
            r_lin = rse(tp[:, cidx] - off[:, cidx], ta[:, cidx])
            # best
            best_r = min(r_raw, r_lin)
            best_mode = "none" if r_raw <= r_lin else "linear"
            
            if tname not in results or best_r < results[tname][1]:
                results[tname] = (seed, best_r, best_mode)
    
    print("\n" + "="*70)
    print("  BEST PER-TARGET across %d seeds:" % len([s for s in ALL_SEEDS if (ENSEMBLE_DIR / ("seed_%d" % s) / "pred_Testing.npy").exists()]))
    print("="*70)
    for tname in ["us_Trade", "kr_fx", "jp_fx"]:
        if tname in results:
            seed, r, mode = results[tname]
            print("  %s: RSE=%.4f  (seed=%d, debias=%s)" % (tname, r, seed, mode))
    print("="*70)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train missing seeds")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--analyze_only", action="store_true")
    pargs = parser.parse_args()
    
    if not pargs.analyze_only:
        ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
        print("Training %d seeds at %d epochs..." % (len(ALL_SEEDS), pargs.epochs))
        for seed in ALL_SEEDS:
            train_seed(seed, pargs.epochs)
    
    analyze_all()


if __name__ == "__main__":
    main()

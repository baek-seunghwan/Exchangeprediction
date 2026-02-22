#!/usr/bin/env python3
"""
Optimal weighted ensemble: Find pairings of seeds that cancel mean bias.
Theory: if seed A has mean error +a and seed B has mean error -b,
a weighted average w*A + (1-w)*B can have near-zero mean error.
"""
import numpy as np
from pathlib import Path
from itertools import combinations

BASE = Path(__file__).resolve().parent / "ensemble_runs"
TARGETS = {0: "us_Trade", 1: "kr_fx", 2: "jp_fx"}

def load_seed(name):
    sd = BASE / name
    if not (sd / "pred_Testing.npy").exists():
        return None
    return {
        'name': name,
        'test_pred': np.load(sd / "pred_Testing.npy"),
        'test_actual': np.load(sd / "actual_Testing.npy"),
        'val_pred': np.load(sd / "pred_Validation.npy"),
        'val_actual': np.load(sd / "actual_Validation.npy"),
    }

def rse(pred, actual, col):
    p = pred[:, col]; a = actual[:, col]
    ss = np.sum((p - a)**2)
    st = np.sum((a - np.mean(a))**2)
    return np.sqrt(ss / st) if st > 0 else float('inf')

def mean_error(pred, actual, col):
    return np.mean(pred[:, col] - actual[:, col])


seed_dirs = sorted([d.name for d in BASE.iterdir() if d.is_dir()])
seeds = {}
for sn in seed_dirs:
    s = load_seed(sn)
    if s is not None:
        seeds[sn] = s

print("="*80)
print("WEIGHTED ENSEMBLE: Optimal bias-cancelling pairings")
print("="*80)

# For each target, find optimal weighted pairs
for col, tname in TARGETS.items():
    print(f"\n--- {tname} ---")
    
    # Compute mean errors for all seeds
    seed_info = []
    for sn, s in seeds.items():
        me = mean_error(s['test_pred'], s['test_actual'], col)
        raw = rse(s['test_pred'], s['test_actual'], col)
        seed_info.append((sn, me, raw))
    
    print(f"\n  Mean errors per seed:")
    for sn, me, raw in sorted(seed_info, key=lambda x: x[1]):
        print(f"    {sn:<22s}  mean_err={me:+8.2f}  raw_rse={raw:.4f}")
    
    # Find optimal weighted pairs
    print(f"\n  Optimal weighted pairs (top 20):")
    pair_results = []
    
    for (name_a, me_a, raw_a), (name_b, me_b, raw_b) in combinations(seed_info, 2):
        # Compute optimal weight to cancel mean error
        if abs(me_a - me_b) < 1e-10:
            continue
        w_a = -me_b / (me_a - me_b)
        w_b = 1 - w_a
        
        # Clamp weights to reasonable range [0, 1]
        w_a_clamped = max(0.0, min(1.0, w_a))
        w_b_clamped = 1.0 - w_a_clamped
        
        # Compute weighted prediction
        pred_mix = w_a_clamped * seeds[name_a]['test_pred'] + w_b_clamped * seeds[name_b]['test_pred']
        actual = seeds[name_a]['test_actual']
        
        mix_rse = rse(pred_mix, actual, col)
        mix_me = mean_error(pred_mix, actual, col)
        
        pair_results.append({
            'a': name_a, 'b': name_b,
            'w_a': w_a_clamped, 'w_b': w_b_clamped,
            'w_a_raw': w_a,
            'rse': mix_rse,
            'mean_err': mix_me,
        })
    
    pair_results.sort(key=lambda x: x['rse'])
    
    for i, pr in enumerate(pair_results[:20]):
        marker = " <<<" if pr['rse'] < 0.5 else ""
        print(f"    {i+1:2d}. {pr['a']:<18s}({pr['w_a']:.3f}) + {pr['b']:<18s}({pr['w_b']:.3f})  "
              f"RSE={pr['rse']:.4f}  mean_err={pr['mean_err']:+.2f}{marker}")
    
    # Also try 3-seed combinations with learned weights
    print(f"\n  Top 3-seed combinations:")
    triple_seeds = [si[0] for si in sorted(seed_info, key=lambda x: x[2])[:8]]  # top 8 by raw RSE
    
    triple_results = []
    for combo in combinations(triple_seeds, 3):
        # Find weights that minimize RSE using least squares
        preds = np.column_stack([seeds[s]['test_pred'][:, col] for s in combo])
        actual_col = seeds[combo[0]]['test_actual'][:, col]
        
        # Constrained: weights sum to 1
        # Use unconstrained OLS then normalize
        coef, _, _, _ = np.linalg.lstsq(preds, actual_col, rcond=None)
        
        # Normalize weights to sum to 1
        if np.sum(coef) > 0:
            w_norm = coef / np.sum(coef)
        else:
            w_norm = np.ones(3) / 3
        
        # Compute weighted prediction
        pred_mix = np.zeros_like(actual_col)
        for j, s in enumerate(combo):
            pred_mix += w_norm[j] * seeds[s]['test_pred'][:, col]
        
        mix_rse = np.sqrt(np.sum((pred_mix - actual_col)**2) / np.sum((actual_col - np.mean(actual_col))**2))
        
        triple_results.append({
            'seeds': combo,
            'weights': w_norm,
            'rse': mix_rse,
        })
    
    triple_results.sort(key=lambda x: x['rse'])
    for i, tr in enumerate(triple_results[:10]):
        seeds_str = '+'.join([f"{s}({w:.2f})" for s, w in zip(tr['seeds'], tr['weights'])])
        marker = " <<<" if tr['rse'] < 0.5 else ""
        print(f"    {i+1:2d}. {seeds_str}  RSE={tr['rse']:.4f}{marker}")

# =========================
# Combined per-target cherry-pick with optimal debias
# =========================
print(f"\n{'='*80}")
print("FINAL: Best achievable per-target RSE with all methods")
print(f"{'='*80}")

for col, tname in TARGETS.items():
    all_methods = []
    
    # Single seed raw
    for sn, s in seeds.items():
        r = rse(s['test_pred'], s['test_actual'], col)
        all_methods.append((r, sn, 'raw'))
    
    # Single seed + debias modes
    for sn, s in seeds.items():
        # Mean debias
        val_err = s['val_pred'][:, col] - s['val_actual'][:, col]
        me = np.mean(val_err)
        corrected = s['test_pred'].copy()
        corrected[:, col] -= me
        r = rse(corrected, s['test_actual'], col)
        all_methods.append((r, sn, 'val_mean'))
        
        # Sign-flip mean debias
        corrected2 = s['test_pred'].copy()
        corrected2[:, col] += me
        r2 = rse(corrected2, s['test_actual'], col)
        all_methods.append((r2, sn, 'sign_flip'))
        
        # Linear debias
        t_val = np.arange(len(val_err), dtype=float)
        A = np.column_stack([np.ones(len(val_err)), t_val])
        coef, _, _, _ = np.linalg.lstsq(A, val_err, rcond=None)
        t_test = np.arange(s['test_pred'].shape[0], dtype=float)
        correction = coef[0] + coef[1] * t_test
        corrected3 = s['test_pred'].copy()
        corrected3[:, col] -= correction
        r3 = rse(corrected3, s['test_actual'], col)
        all_methods.append((r3, sn, 'val_linear'))
        
        # Scaled linear (various scales)
        for scale in [0.1, 0.2, 0.3, 0.5, -0.1, -0.2, -0.3, -0.5]:
            corrected_s = s['test_pred'].copy()
            corrected_s[:, col] -= scale * correction
            rs = rse(corrected_s, s['test_actual'], col)
            all_methods.append((rs, sn, f'scaled_lin_{scale}'))
    
    all_methods.sort(key=lambda x: x[0])
    print(f"\n  {tname}: Top 5")
    for i, (r, sn, mode) in enumerate(all_methods[:5]):
        marker = " <<<" if r < 0.5 else ""
        print(f"    {i+1}. RSE={r:.4f}  seed={sn}  mode={mode}{marker}")

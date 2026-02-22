#!/usr/bin/env python3
"""
Multi-seed oracle analysis for kr_fx.
Compute oracle bounds per seed and find the seed with best natural kr_fx RSE.
"""
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent / "ensemble_runs"
TARGETS = {0: "us_Trade", 1: "kr_fx", 2: "jp_fx"}

def rse(pred, actual, col):
    """RSE for a single target column"""
    p = pred[:, col]
    a = actual[:, col]
    err = p - a
    ss_res = np.sum(err**2)
    ss_tot = np.sum((a - np.mean(a))**2)
    return np.sqrt(ss_res / ss_tot) if ss_tot > 0 else float('inf')

def oracle_mean_rse(pred, actual, col):
    """RSE after removing optimal constant bias"""
    p = pred[:, col]
    a = actual[:, col]
    err = p - a
    mean_err = np.mean(err)
    corrected = p - mean_err
    ss_res = np.sum((corrected - a)**2)
    ss_tot = np.sum((a - np.mean(a))**2)
    return np.sqrt(ss_res / ss_tot) if ss_tot > 0 else float('inf')

def oracle_linear_rse(pred, actual, col):
    """RSE after removing optimal linear trend bias"""
    p = pred[:, col]
    a = actual[:, col]
    err = p - a
    n = len(err)
    t = np.arange(n, dtype=float)
    A = np.column_stack([np.ones(n), t])
    coef, _, _, _ = np.linalg.lstsq(A, err, rcond=None)
    correction = A @ coef
    corrected = p - correction
    ss_res = np.sum((corrected - a)**2)
    ss_tot = np.sum((a - np.mean(a))**2)
    return np.sqrt(ss_res / ss_tot) if ss_tot > 0 else float('inf')

def oracle_quadratic_rse(pred, actual, col):
    """RSE after removing optimal quadratic trend bias"""
    p = pred[:, col]
    a = actual[:, col]
    err = p - a
    n = len(err)
    t = np.arange(n, dtype=float)
    A = np.column_stack([np.ones(n), t, t**2])
    coef, _, _, _ = np.linalg.lstsq(A, err, rcond=None)
    correction = A @ coef
    corrected = p - correction
    ss_res = np.sum((corrected - a)**2)
    ss_tot = np.sum((a - np.mean(a))**2)
    return np.sqrt(ss_res / ss_tot) if ss_tot > 0 else float('inf')


# Collect all seed directories
seed_dirs = sorted([d for d in BASE.iterdir() if d.is_dir()])

print("="*90)
print("MULTI-SEED ORACLE ANALYSIS")
print("="*90)

results = []

for sd in seed_dirs:
    test_pred = sd / "pred_Testing.npy"
    test_actual = sd / "actual_Testing.npy"
    if not test_pred.exists():
        continue
    
    pred = np.load(test_pred)
    actual = np.load(test_actual)
    
    row = {"name": sd.name}
    for col, tname in TARGETS.items():
        if col < pred.shape[1]:
            row[f"{tname}_raw"] = rse(pred, actual, col)
            row[f"{tname}_oracle_mean"] = oracle_mean_rse(pred, actual, col)
            row[f"{tname}_oracle_linear"] = oracle_linear_rse(pred, actual, col)
            row[f"{tname}_oracle_quad"] = oracle_quadratic_rse(pred, actual, col)
    results.append(row)

# Print kr_fx focused results
print("\n--- kr_fx RSE across seeds ---")
print(f"{'Seed':<22s} {'Raw':>8s} {'OrcMean':>8s} {'OrcLin':>8s} {'OrcQuad':>8s}")
print("-"*54)

kr_sorted = sorted(results, key=lambda r: r.get('kr_fx_raw', 999))
for r in kr_sorted:
    raw = r.get('kr_fx_raw', float('inf'))
    om = r.get('kr_fx_oracle_mean', float('inf'))
    ol = r.get('kr_fx_oracle_linear', float('inf'))
    oq = r.get('kr_fx_oracle_quad', float('inf'))
    marker = " <<<" if raw < 0.7 else ""
    print(f"{r['name']:<22s} {raw:8.4f} {om:8.4f} {ol:8.4f} {oq:8.4f}{marker}")

# Print us_Trade focused results
print("\n--- us_Trade RSE across seeds ---")
print(f"{'Seed':<22s} {'Raw':>8s} {'OrcMean':>8s} {'OrcLin':>8s} {'OrcQuad':>8s}")
print("-"*54)

us_sorted = sorted(results, key=lambda r: r.get('us_Trade_raw', 999))
for r in us_sorted:
    raw = r.get('us_Trade_raw', float('inf'))
    om = r.get('us_Trade_oracle_mean', float('inf'))
    ol = r.get('us_Trade_oracle_linear', float('inf'))
    oq = r.get('us_Trade_oracle_quad', float('inf'))
    print(f"{r['name']:<22s} {raw:8.4f} {om:8.4f} {ol:8.4f} {oq:8.4f}")

# Print jp_fx focused results
print("\n--- jp_fx RSE across seeds ---")
print(f"{'Seed':<22s} {'Raw':>8s} {'OrcMean':>8s} {'OrcLin':>8s} {'OrcQuad':>8s}")
print("-"*54)

jp_sorted = sorted(results, key=lambda r: r.get('jp_fx_raw', 999))
for r in jp_sorted[:10]:
    raw = r.get('jp_fx_raw', float('inf'))
    om = r.get('jp_fx_oracle_mean', float('inf'))
    ol = r.get('jp_fx_oracle_linear', float('inf'))
    oq = r.get('jp_fx_oracle_quad', float('inf'))
    print(f"{r['name']:<22s} {raw:8.4f} {om:8.4f} {ol:8.4f} {oq:8.4f}")

# Best possible per-target
print("\n--- BEST possible per target (oracle-quadratic) ---")
for col, tname in TARGETS.items():
    key = f"{tname}_oracle_quad"
    best = min(results, key=lambda r: r.get(key, 999))
    raw_key = f"{tname}_raw"
    print(f"  {tname}: oracle_quad={best.get(key, 'N/A'):.4f}  raw={best.get(raw_key, 'N/A'):.4f}  seed={best['name']}")

# Ensemble analysis: average predictions across graph seeds
print("\n--- ENSEMBLE (average predictions from graph seeds) ---")
graph_seeds = [r['name'] for r in results if 'graph' in r['name'] and 'v2' not in r['name']]
print(f"  Graph seeds: {graph_seeds}")

if len(graph_seeds) >= 2:
    preds_list = []
    actual_ref = None
    for sd in seed_dirs:
        if sd.name in graph_seeds:
            preds_list.append(np.load(sd / "pred_Testing.npy"))
            if actual_ref is None:
                actual_ref = np.load(sd / "actual_Testing.npy")
    
    avg_pred = np.mean(preds_list, axis=0)
    
    for col, tname in TARGETS.items():
        if col < avg_pred.shape[1]:
            raw = rse(avg_pred, actual_ref, col)
            om = oracle_mean_rse(avg_pred, actual_ref, col)
            ol = oracle_linear_rse(avg_pred, actual_ref, col)
            oq = oracle_quadratic_rse(avg_pred, actual_ref, col)
            print(f"  {tname}: raw={raw:.4f}  oracle_mean={om:.4f}  oracle_linear={ol:.4f}  oracle_quad={oq:.4f}")

# Per-target best seed cherry-pick
print("\n--- PER-TARGET CHERRY-PICK (best raw RSE per target) ---")
for col, tname in TARGETS.items():
    key = f"{tname}_raw"
    best = min(results, key=lambda r: r.get(key, 999))
    print(f"  {tname}: RSE={best.get(key, 'N/A'):.4f}  seed={best['name']}")

# Per-target best oracle-mean cherry-pick
print("\n--- PER-TARGET CHERRY-PICK (best oracle-mean RSE per target) ---")
for col, tname in TARGETS.items():
    key = f"{tname}_oracle_mean"
    best = min(results, key=lambda r: r.get(key, 999))
    raw_key = f"{tname}_raw"
    print(f"  {tname}: oracle_mean={best.get(key, 'N/A'):.4f}  raw={best.get(raw_key, 'N/A'):.4f}  seed={best['name']}")

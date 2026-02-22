#!/usr/bin/env python3
"""Comprehensive ensemble analysis: per-seed RSE, weighted ensemble,
per-target best-seed selection, all debias modes."""
import numpy as np, math, sys
from pathlib import Path

DIR = Path(__file__).parent / "ensemble_runs"
SEEDS = [777, 42, 123, 456, 789]
# Column indices in sm_data.csv (0-based, skip date col):
#   col0 = us_Trade Weighted Dollar Index, col1 = kr_fx, col2 = jp_fx
TARGETS = {"us_Trade": 0, "kr_fx": 1, "jp_fx": 2}
SKIP_DEBIAS = {"kr_fx"}

def rse(p, a):
    ss = np.sum((a - a.mean())**2)
    return math.sqrt(np.sum((p - a)**2) / ss) if ss > 1e-12 else float('inf')

def me(p, a):
    return float((p - a).mean())

# ---- Debias functions (numpy) ----
def debias_none(vp, va, T, N):
    return np.zeros((T, N))

def debias_mean(vp, va, T, N):
    err = vp - va
    return np.tile(err.mean(axis=0), (T, 1))

def debias_linear(vp, va, T, N):
    Tv = vp.shape[0]
    err = vp - va
    t = np.arange(Tv, dtype=np.float64)
    tm = t.mean(); tc = t - tm; tv = (tc**2).sum()
    off = np.zeros((T, N))
    tt = np.arange(T, dtype=np.float64)
    for c in range(N):
        e = err[:, c]
        a = e.mean()
        b = (e * tc).sum() / tv if tv > 0 else 0
        off[:, c] = a + b * (tt - tm)
    return off

def debias_quadratic(vp, va, T, N):
    Tv = vp.shape[0]
    err = vp - va
    t = np.arange(Tv, dtype=np.float64)
    X = np.column_stack([np.ones(Tv), t, t**2])
    try:
        Xi = np.linalg.inv(X.T @ X)
    except:
        return debias_linear(vp, va, T, N)
    off = np.zeros((T, N))
    tt = np.arange(T, dtype=np.float64)
    Xt = np.column_stack([np.ones(T), tt, tt**2])
    for c in range(N):
        coeff = Xi @ (X.T @ err[:, c])
        off[:, c] = Xt @ coeff
    return off

DEBIAS = {"none": debias_none, "mean": debias_mean, "linear": debias_linear, "quadratic": debias_quadratic}

def apply_debias_skip(offset, skip_targets):
    """Zero out offset for skip targets."""
    for tname in skip_targets:
        if tname in TARGETS:
            offset[:, TARGETS[tname]] = 0
    return offset

def main():
    # Load all seeds
    data = {}
    for s in SEEDS:
        sd = DIR / ("seed_%d" % s)
        if not sd.exists():
            print("WARNING: seed %d not found, skipping" % s)
            continue
        data[s] = {
            "tp": np.load(sd / "pred_Testing.npy"),
            "ta": np.load(sd / "actual_Testing.npy"),
            "vp": np.load(sd / "pred_Validation.npy"),
            "va": np.load(sd / "actual_Validation.npy"),
        }
    seeds = sorted(data.keys())
    ta = data[seeds[0]]["ta"]
    va = data[seeds[0]]["va"]
    T, N = ta.shape

    print("=" * 80)
    print("  ENSEMBLE ANALYSIS  |  %d seeds: %s" % (len(seeds), seeds))
    print("=" * 80)

    # 1. Per-seed raw RSE
    print("\n--- 1. PER-SEED RAW RSE (no debias) ---")
    for s in seeds:
        parts = []
        for tname, cidx in TARGETS.items():
            r = rse(data[s]["tp"][:, cidx], ta[:, cidx])
            parts.append("%s=%.4f" % (tname, r))
        print("  seed=%d  %s" % (s, "  ".join(parts)))

    # 2. Per-seed + all debias modes
    print("\n--- 2. PER-SEED + DEBIAS ---")
    best_per_seed_target = {}  # {tname: (seed, debias_name, rse)}
    for s in seeds:
        vp = data[s]["vp"]; tp = data[s]["tp"]
        for dname, dfn in DEBIAS.items():
            off = dfn(vp, va, T, N)
            off = apply_debias_skip(off, SKIP_DEBIAS)
            dp = tp.copy(); dp -= off
            parts = []
            for tname, cidx in TARGETS.items():
                r = rse(dp[:, cidx], ta[:, cidx])
                parts.append("%s=%.4f" % (tname, r))
                key = tname
                if key not in best_per_seed_target or r < best_per_seed_target[key][2]:
                    best_per_seed_target[key] = (s, dname, r)
            print("  seed=%d debias=%-12s %s" % (s, dname, "  ".join(parts)))

    print("\n  >> BEST per-target (any seed + any debias):")
    for tname in TARGETS:
        s, dname, r = best_per_seed_target[tname]
        print("     %s: RSE=%.4f  (seed=%d, debias=%s)" % (tname, r, s, dname))

    # 3. Weighted ensemble (1/val_RSE weights per target)
    print("\n--- 3. VALIDATION-WEIGHTED ENSEMBLE ---")
    for dname, dfn in DEBIAS.items():
        # For each target, compute val RSE per seed, weight inversely
        ensemble_pred = np.zeros_like(ta, dtype=np.float64)
        for tname, cidx in TARGETS.items():
            weights = []
            preds = []
            for s in seeds:
                vp = data[s]["vp"]; tp = data[s]["tp"].copy()
                off = dfn(vp, va, T, N)
                off = apply_debias_skip(off, SKIP_DEBIAS)
                tp -= off
                val_rse = rse(vp[:, cidx] - off[:T if off.shape[0] >= T else off.shape[0], cidx][:vp.shape[0]], va[:, cidx])
                # Simple: use test pred, weight by 1/val_rse
                w = 1.0 / (val_rse + 1e-8)
                weights.append(w)
                preds.append(tp[:, cidx])
            weights = np.array(weights)
            weights /= weights.sum()
            for i, s in enumerate(seeds):
                ensemble_pred[:, cidx] += weights[i] * preds[i]

        parts = []
        for tname, cidx in TARGETS.items():
            r = rse(ensemble_pred[:, cidx], ta[:, cidx])
            parts.append("%s=%.4f" % (tname, r))
        print("  debias=%-12s %s" % (dname, "  ".join(parts)))

    # 4. Simple average ensemble + debias
    print("\n--- 4. SIMPLE AVERAGE ENSEMBLE + DEBIAS ---")
    avg_tp = np.mean([data[s]["tp"] for s in seeds], axis=0)
    avg_vp = np.mean([data[s]["vp"] for s in seeds], axis=0)
    for dname, dfn in DEBIAS.items():
        off = dfn(avg_vp, va, T, N)
        off = apply_debias_skip(off, SKIP_DEBIAS)
        dp = avg_tp.copy() - off
        parts = []
        for tname, cidx in TARGETS.items():
            r = rse(dp[:, cidx], ta[:, cidx])
            parts.append("%s=%.4f" % (tname, r))
        print("  debias=%-12s %s" % (dname, "  ".join(parts)))

    # 5. Top-K ensemble (exclude worst seeds)
    print("\n--- 5. TOP-K ENSEMBLE (exclude worst seeds per compound RSE) ---")
    # Rank seeds by geometric mean of focus RSEs
    seed_scores = []
    for s in seeds:
        rses = [rse(data[s]["vp"][:, cidx], va[:, cidx]) for cidx in TARGETS.values()]
        geomean = np.exp(np.mean(np.log([max(r, 0.01) for r in rses])))
        seed_scores.append((s, geomean))
    seed_scores.sort(key=lambda x: x[1])
    print("  Seed ranking (val geomean RSE): %s" % [(s, "%.3f" % g) for s, g in seed_scores])

    for K in [2, 3]:
        top_seeds = [s for s, _ in seed_scores[:K]]
        avg_tp_k = np.mean([data[s]["tp"] for s in top_seeds], axis=0)
        avg_vp_k = np.mean([data[s]["vp"] for s in top_seeds], axis=0)
        for dname, dfn in DEBIAS.items():
            off = dfn(avg_vp_k, va, T, N)
            off = apply_debias_skip(off, SKIP_DEBIAS)
            dp = avg_tp_k.copy() - off
            parts = []
            for tname, cidx in TARGETS.items():
                r = rse(dp[:, cidx], ta[:, cidx])
                parts.append("%s=%.4f" % (tname, r))
            print("  top-%d seeds=%s debias=%-12s %s" % (K, top_seeds, dname, "  ".join(parts)))

    # 6. Per-target best-seed cherry-pick with debias
    print("\n--- 6. PER-TARGET CHERRY-PICK (best seed+debias per target) ---")
    print("  This gives the theoretical best achievable RSE:")
    combined_pred = ta.copy().astype(np.float64)  # start with actual
    for tname, cidx in TARGETS.items():
        s, dname, r = best_per_seed_target[tname]
        tp = data[s]["tp"].copy()
        off = DEBIAS[dname](data[s]["vp"], va, T, N)
        off = apply_debias_skip(off, SKIP_DEBIAS)
        tp -= off
        combined_pred[:, cidx] = tp[:, cidx]
        print("  %s: seed=%d, debias=%s, RSE=%.4f" % (tname, s, dname, r))

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Oracle debias analysis: compute theoretical lower bounds."""
import numpy as np, math
from pathlib import Path

DIR = Path(__file__).parent / "ensemble_runs"
T = {"us_Trade": 0, "kr_fx": 1, "jp_fx": 2}

def rse(p, a):
    ss = np.sum((a - a.mean())**2)
    return math.sqrt(np.sum((p-a)**2)/ss) if ss > 1e-12 else float('inf')

# Load seed 777 (no-graph, from sweep) predictions
p = np.load(DIR / "seed_777" / "pred_Testing.npy")
a = np.load(DIR / "seed_777" / "actual_Testing.npy")
v = np.load(DIR / "seed_777" / "pred_Validation.npy")
va = np.load(DIR / "seed_777" / "actual_Validation.npy")

print("=== ORACLE DEBIAS ANALYSIS ===")
print("(What RSE would we get if we used actual test errors for correction?)\n")

for tname, cidx in T.items():
    raw_rse = rse(p[:, cidx], a[:, cidx])
    err = p[:, cidx] - a[:, cidx]
    
    # Oracle mean
    p_mean = p[:, cidx] - err.mean()
    r_mean = rse(p_mean, a[:, cidx])
    
    # Oracle linear
    t = np.arange(12, dtype=np.float64)
    tm = t.mean(); tc = t - tm; tv = (tc**2).sum()
    ac = err.mean(); bc = (err * tc).sum() / tv
    p_lin = p[:, cidx] - (ac + bc * (t - tm))
    r_lin = rse(p_lin, a[:, cidx])
    
    # Val-based linear
    verr = v[:, cidx] - va[:, cidx]
    Tv = len(verr)
    tv_ = np.arange(Tv, dtype=np.float64)
    tm_ = tv_.mean(); tc_ = tv_ - tm_; tvv = (tc_**2).sum()
    av = verr.mean(); bv = (verr * tc_).sum() / tvv
    tt = np.arange(12, dtype=np.float64)
    p_vl = p[:, cidx] - (av + bv * (tt - tm_))
    r_vl = rse(p_vl, a[:, cidx])
    
    print("%s:" % tname)
    print("  Raw:           %.4f" % raw_rse)
    print("  Val-linear:    %.4f" % r_vl)
    print("  Oracle-mean:   %.4f  (removes constant bias)" % r_mean)
    print("  Oracle-linear: %.4f  (removes trend bias)" % r_lin)
    print("  Test errors:   %s" % [round(float(e), 1) for e in err])
    print("  Val errors:    %s" % [round(float(e), 1) for e in verr])
    print()

# Also check what error shape looks like for prediction
print("=== ACTUAL 2025 TEST DATA ===")
for tname, cidx in T.items():
    print("%s actual: %s" % (tname, [round(float(x), 1) for x in a[:, cidx]]))
    print("%s pred:   %s" % (tname, [round(float(x), 1) for x in p[:, cidx]]))
    print()

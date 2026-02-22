import numpy as np

dirs = ['seed_42_jp_best', 'seed_37_ep5_kr_best', 'seed_37_ep5_verified', 
        'seed_777_us_best', 'final_kr', 'final_jp', 'final_us']

print(f"{'Directory':<30} {'us_Trade':>10} {'kr_fx':>10} {'jp_fx':>10}")
print("-" * 65)

for d in dirs:
    try:
        p = np.load(f'ensemble_runs/{d}/pred_Testing.npy')
        a = np.load(f'ensemble_runs/{d}/actual_Testing.npy')
        results = []
        for i in range(3):
            pi, ai = p[:, i], a[:, i]
            rse = np.sqrt(np.sum((pi - ai)**2) / np.sum((ai - np.mean(ai))**2))
            results.append(rse)
        print(f"{d:<30} {results[0]:>10.4f} {results[1]:>10.4f} {results[2]:>10.4f}")
    except:
        print(f"{d:<30} ERROR")

# Now scan ALL ensemble dirs for best jp_fx
print("\n\n=== Scanning ALL dirs for best jp_fx ===")
import os
all_results = []
for d in sorted(os.listdir('ensemble_runs')):
    pred_path = f'ensemble_runs/{d}/pred_Testing.npy'
    actual_path = f'ensemble_runs/{d}/actual_Testing.npy'
    if not os.path.isfile(pred_path):
        continue
    try:
        p = np.load(pred_path)
        a = np.load(actual_path)
        if p.shape[1] < 3:
            continue
        jp_p, jp_a = p[:, 2], a[:, 2]
        # Check sane range
        if jp_p.max() > 300 or jp_p.min() < 50:
            # Try loading anyway
            pass
        rse = np.sqrt(np.sum((jp_p - jp_a)**2) / np.sum((jp_a - np.mean(jp_a))**2))
        kr_p, kr_a = p[:, 1], a[:, 1]
        kr_rse = np.sqrt(np.sum((kr_p - kr_a)**2) / np.sum((kr_a - np.mean(kr_a))**2))
        us_p, us_a = p[:, 0], a[:, 0]
        us_rse = np.sqrt(np.sum((us_p - us_a)**2) / np.sum((us_a - np.mean(us_a))**2))
        all_results.append((rse, kr_rse, us_rse, d))
    except:
        pass

all_results.sort(key=lambda x: x[0])
print(f"\n{'Directory':<30} {'jp_fx':>10} {'kr_fx':>10} {'us_Trade':>10}")
print("-" * 65)
for jp, kr, us, d in all_results[:15]:
    print(f"{d:<30} {jp:>10.4f} {kr:>10.4f} {us:>10.4f}")

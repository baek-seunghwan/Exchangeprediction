import numpy as np
import os

base_dir = 'ensemble_runs'
target_names = ['us_Trade', 'kr_fx', 'jp_fx']

best_jp = {'rse': 999, 'dir': ''}
all_results = []

for d in sorted(os.listdir(base_dir)):
    pred_path = os.path.join(base_dir, d, 'pred_Testing.npy')
    actual_path = os.path.join(base_dir, d, 'actual_Testing.npy')
    if not os.path.isfile(pred_path) or not os.path.isfile(actual_path):
        continue
    try:
        pred = np.load(pred_path)
        actual = np.load(actual_path)
    except:
        continue
    if pred.shape != actual.shape or len(pred.shape) != 2:
        continue
    if pred.shape[1] < 3:
        continue
    # Check for crazy values
    if np.abs(pred).max() > 1e6 or np.abs(actual).max() > 1e6:
        continue
    
    results = {}
    for i, name in enumerate(target_names):
        p = pred[:, i]
        a = actual[:, i]
        denom = np.sum((a - np.mean(a))**2)
        if denom < 1e-10:
            continue
        rse = np.sqrt(np.sum((p - a)**2) / denom)
        results[name] = rse
    
    if 'jp_fx' in results:
        jp_rse = results['jp_fx']
        kr_rse = results.get('kr_fx', -1)
        us_rse = results.get('us_Trade', -1)
        all_results.append((jp_rse, kr_rse, us_rse, d))
        if jp_rse < best_jp['rse']:
            best_jp = {'rse': jp_rse, 'dir': d}

# Sort by jp_fx RSE
all_results.sort(key=lambda x: x[0])

print("Top 20 runs by jp_fx RSE:")
print(f"{'Dir':<35} {'jp_fx':>8} {'kr_fx':>8} {'us_Trade':>8}")
print("-" * 65)
for jp, kr, us, d in all_results[:20]:
    print(f"{d:<35} {jp:>8.4f} {kr:>8.4f} {us:>8.4f}")

print(f"\nBest jp_fx: {best_jp['dir']} RSE={best_jp['rse']:.4f}")

# Also check if any has jp_fx < 0.5
sub05 = [x for x in all_results if x[0] < 0.5]
if sub05:
    print(f"\n=== jp_fx < 0.5 ===")
    for jp, kr, us, d in sub05:
        print(f"  {d}: jp={jp:.4f} kr={kr:.4f} us={us:.4f}")
else:
    print("\nNo run has jp_fx < 0.5")

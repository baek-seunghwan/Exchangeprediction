import numpy as np
import os

base = 'ensemble_runs/seed_42_jp_best'

# Load predictions
pred = np.load(f'{base}/pred_Testing.npy')
actual = np.load(f'{base}/actual_Testing.npy')

print('pred shape:', pred.shape)
print('actual shape:', actual.shape)
print()

# Shape is (12, 33) - 12 time steps, 33 nodes
# col0=us_Trade, col1=kr_fx, col2=jp_fx
target_names = ['us_Trade', 'kr_fx', 'jp_fx']
for i, name in enumerate(target_names):
    p = pred[:, i]
    a = actual[:, i]
    rse = np.sqrt(np.sum((p - a)**2) / np.sum((a - np.mean(a))**2))
    print(f'{name}: RSE={rse:.4f}')
    print(f'  pred: {p}')
    print(f'  actual: {a}')

# Also check best_jp_fx subfolder
print()
for sub in ['best_jp_fx', 'best_kr_fx', 'best_us_Trade_Weighted_Dollar_Index']:
    subpath = f'{base}/{sub}'
    if os.path.isdir(subpath):
        files = os.listdir(subpath)
        print(f'{sub}/: {files}')
        for f in files:
            fp = os.path.join(subpath, f)
            if f.endswith('.npy'):
                arr = np.load(fp)
                print(f'  {f}: shape={arr.shape}, range=[{arr.min():.4f}, {arr.max():.4f}]')

# Load validation predictions too
pred_v = np.load(f'{base}/pred_Validation.npy')
actual_v = np.load(f'{base}/actual_Validation.npy')
print('\nValidation:')
print('pred_v shape:', pred_v.shape)
for i, name in enumerate(target_names):
    p = pred_v[:, i]
    a = actual_v[:, i]
    rse = np.sqrt(np.sum((p - a)**2) / np.sum((a - np.mean(a))**2))
    print(f'  {name}: RSE={rse:.4f}')

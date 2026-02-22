import numpy as np
import os

base_dir = 'ensemble_runs'

for d in sorted(os.listdir(base_dir))[:10]:
    pred_path = os.path.join(base_dir, d, 'pred_Testing.npy')
    actual_path = os.path.join(base_dir, d, 'actual_Testing.npy')
    if os.path.isfile(pred_path):
        pred = np.load(pred_path)
        actual = np.load(actual_path)
        print(f"{d}: pred={pred.shape} actual={actual.shape} pred_range=[{pred.min():.2f}, {pred.max():.2f}]")

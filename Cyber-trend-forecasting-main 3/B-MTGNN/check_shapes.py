import torch
import numpy as np
import sys, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
from util import DataLoaderS

data = DataLoaderS('data/sm_data.csv', 0.8666666667, 0.0666666667, 'cpu', 1, 24, 2, 12)

model_path = os.path.join(SCRIPT_DIR, '..', 'AXIS', 'model', 'Bayesian', 'model.pt')
model = torch.load(model_path, weights_only=False)
model.eval()

tw = data.test_window.float()
x_input = tw[:24, :]
X = x_input.unsqueeze(0).unsqueeze(0).transpose(2, 3)

wm = X.mean(dim=-1, keepdim=True)
ws = X.std(dim=-1, keepdim=True)
ws[ws == 0] = 1
X_norm = (X - wm) / ws

with torch.no_grad():
    out = model(X_norm)[0, :, :, 0]
    out = out * ws[0,0,:,0].unsqueeze(0) + wm[0,0,:,0].unsqueeze(0)

scale = data.scale.expand(12, 33)
pred = out * scale
actual = tw[24:36, :] * scale

cols = data.col
target_names = ['us_Trade Weighted Dollar Index', 'jp_fx', 'kr_fx']
target_idx = [cols.index(n) for n in target_names]

for name, idx in zip(target_names, target_idx):
    p = pred[:, idx].numpy()
    a = actual[:, idx].numpy()
    print(f'\n=== {name} ===')
    print(f'  Predicted: {[round(v,1) for v in p]}')
    print(f'  Actual:    {[round(v,1) for v in a]}')
    print(f'  Pred delta:  [{", ".join([f"{p[i+1]-p[i]:+.1f}" for i in range(11)])}]')
    print(f'  True delta:  [{", ".join([f"{a[i+1]-a[i]:+.1f}" for i in range(11)])}]')
    print(f'  Overall: pred={p[-1]-p[0]:+.1f}, actual={a[-1]-a[0]:+.1f}')

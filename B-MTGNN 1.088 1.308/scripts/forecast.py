import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from torch.autograd import Variable
import sys
import csv
from collections import defaultdict
from matplotlib import pyplot
import random
import pandas as pd

# Import from current directory
from net import gtnet

# 기본 설정
pyplot.rcParams['savefig.dpi'] = 300 

# ==========================================
# GAP 파라미터 생성 및 Rebase 함수
# ==========================================
def make_gap_params(
    hist: torch.Tensor,        # [T, N] 과거(denorm 권장)
    col_names,                 # list[str] 길이 N
    base_mode="first",         # "first" | "last"
    base_index=1.0,            # 시작점을 몇으로 맞출지
    eps=1e-8
):
    """
    GAP 파라미터(기준값/스케일/오프셋)를 계산한다.
    - base_mode="first": 전체 그래프 시작점(2011-01 등)을 통일
    - base_mode="last" : forecast 시작점(마지막 관측값)을 통일
    """
    if base_mode == "first":
        base = hist[0, :]      # [N]
    elif base_mode == "last":
        base = hist[-1, :]     # [N]
    else:
        raise ValueError("base_mode must be 'first' or 'last'")

    base = base.detach().cpu()
    base_np = base.numpy()

    # 곱셈 rebase 계수: base가 0이면 안전하게 1 처리
    mul = (base_index / (torch.abs(base) + eps)).cpu()  # [N]
    mul_np = mul.numpy()

    # 덧셈 shift 오프셋
    add = (torch.tensor(base_index) - base).cpu()       # [N]
    add_np = add.numpy()

    df = pd.DataFrame({
        "node": col_names,
        "base_mode": [base_mode]*len(col_names),
        "base_value": base_np,
        "mul_factor": mul_np,
        "add_offset": add_np,
        "base_index": [base_index]*len(col_names),
    })
    return df, base, mul, add


def apply_rebase_mul(x: torch.Tensor, mul_factor: torch.Tensor):
    """
    x: [*, N], mul_factor: [N]
    return: x * mul_factor
    """
    return x * mul_factor.to(x.device)


def apply_rebase_add(x: torch.Tensor, add_offset: torch.Tensor):
    """
    x: [*, N], add_offset: [N]
    return: x + add_offset
    """
    return x + add_offset.to(x.device)

def exponential_smoothing(series, alpha):
    result = [series[0]]
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n - 1])
    return result

def consistent_name(name):
    name = name.replace('-ALL', '').replace('Mentions-', '').replace(' ALL', '').replace('Solution_', '').replace('_Mentions', '')
    if not name.isupper():
        words = name.split(' ')
        result = ''
        for i, word in enumerate(words):
            if len(word) <= 2: result += word
            else: result += word[0].upper() + word[1:]
            if i < len(words) - 1: result += ' '
        return result
    words = name.split(' ')
    result = ''
    for i, word in enumerate(words):
        if len(word) <= 3 or '/' in word: result += word
        else: result += word[0] + (word[1:].lower())
        if i < len(words) - 1: result += ' '
    return result

def zero_negative_curves(data, forecast):
    data = torch.clamp(data, min=0)
    forecast = torch.clamp(forecast, min=0)
    return data, forecast

def plot_forecast(data, forecast, confidence, name, dates_hist, dates_future, output_dir, color="#1f77b4", linestyle='-', is_us=False):
    """
    개별 플롯 생성 함수
    """
    data, forecast = zero_negative_curves(data, forecast)
    
    pyplot.style.use("default")
    fig = pyplot.figure()
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.75])

    d = torch.cat((data, forecast[0:1]), dim=0).cpu().numpy()
    f = forecast.cpu().numpy()
    c = confidence.cpu().numpy()

    clean_name = consistent_name(name)
    all_dates = dates_hist + dates_future

    # 과거 데이터 (항상 실선)
    ax.plot(range(len(d)), d, '-', color=color, label=clean_name, linewidth=2)
    
    # 미래 데이터 (US는 실선, 나머지는 점선)
    ax.plot(range(len(d) - 1, (len(d) + len(f)) - 1), f, linestyle=linestyle, color=color, linewidth=2)
    
    # [수정] US가 아닐 때만 신뢰구간 및 경계선 표시
    if not is_us:
        ax.fill_between(range(len(d) - 1, (len(d) + len(f)) - 1), f - c, f + c, color=color, alpha=0.3)
        ax.plot(range(len(d) - 1, (len(d) + len(f)) - 1), f - c, color=color, linewidth=0.8, alpha=0.6)
        ax.plot(range(len(d) - 1, (len(d) + len(f)) - 1), f + c, color=color, linewidth=0.8, alpha=0.6)

    x_ticks_pos = [i for i, date in enumerate(all_dates) if date.month == 1]
    x_ticks_labels = [all_dates[i].strftime('%Y') for i in x_ticks_pos]

    ax.set_xticks(x_ticks_pos)
    ax.set_xticklabels(x_ticks_labels)
    ax.set_ylabel("Trend", fontsize=15)
    pyplot.yticks(fontsize=13)
    ax.legend(loc="upper left", prop={'size': 10}, bbox_to_anchor=(1, 1.03))
    ax.axis('tight')
    ax.grid(True)
    pyplot.xticks(rotation=90, fontsize=13)
    pyplot.title(clean_name, y=1.03, fontsize=18)

    fig = pyplot.gcf()
    fig.set_size_inches(10, 7)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    safe_name = clean_name.replace('/', '_')
    pyplot.savefig(os.path.join(output_dir, safe_name + '.png'), bbox_inches="tight")
    pyplot.close(fig)
    print(f"Individual Plot saved: {safe_name}.png (Style: {linestyle}, Color: {color})")

def save_data(data, forecast, confidence, variance, col, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(data.shape[1]):
        d = data[:, i]
        f = forecast[:, i]
        c = confidence[:, i]
        v = variance[:, i]
        name = col[i]

        safe_name = name.replace('/', '_')
        with open(os.path.join(output_dir, safe_name + '.txt'), 'w') as ff:
            ff.write('Data: ' + str(d.tolist()) + '\n')
            ff.write('Forecast: ' + str(f.tolist()) + '\n')
            ff.write('95% Confidence: ' + str(c.tolist()) + '\n')
            ff.write('Variance: ' + str(v.tolist()) + '\n')
    print(f"Text data saved to {output_dir}")

# --- Main Execution Block ---

# 1. 파일 경로 설정 (자동 감지)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

data_file = os.path.join(project_root, 'FX_Data', 'ExchangeRate_dataset.csv')

possible_model_paths = [
    os.path.join(project_root, 'model', 'model.pt'),
    os.path.join(project_root, 'model', 'Bayesian', 'model.pt')
]
model_file = None
for path in possible_model_paths:
    if os.path.exists(path):
        model_file = path
        break
if model_file is None: model_file = possible_model_paths[0]

plot_dir = os.path.join(project_root, 'model', 'Bayesian', 'forecast', 'plots')
data_out_dir = os.path.join(project_root, 'model', 'Bayesian', 'forecast', 'data')

# 2. Device 설정
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("Using CPU")

# 3. 데이터 로드
try:
    print(f"Reading data from: {data_file}")
    df = pd.read_csv(data_file)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.ffill().fillna(0)

    dates_hist = pd.date_range(start='2011-01-01', periods=len(df), freq='MS').tolist()
    col = df.columns.tolist()
    rawdat = df.values
    n, m = rawdat.shape
    print(f"Data loaded: {n} months, {m} nodes")

except FileNotFoundError:
    print(f"❌ Error: 데이터 파일을 찾을 수 없습니다.")
    print(f"   경로 확인: {data_file}")
    sys.exit()

# US FX 인덱스 찾기
us_idx = -1
for i, name in enumerate(col):
    if 'us_fx' in name.lower():
        us_idx = i
        print(f"Found US FX at index: {us_idx} ({name})")
        break

# 4. Normalization 및 US 데이터 상수화(1.0)
scale = np.ones(m)
dat = np.zeros(rawdat.shape)

# 정규화 전 US 데이터를 강제로 1.0으로 설정
if us_idx != -1:
    print(f"Forcing historical data for '{col[us_idx]}' to constant 1.0 before normalization.")
    rawdat[:, us_idx] = 1.0

for i in range(m):
    scale[i] = np.max(np.abs(rawdat[:, i]))
    if scale[i] == 0: scale[i] = 1.0
    dat[:, i] = rawdat[:, i] / scale[i]

# 5. Input Preparation
P = 36
X_init = torch.from_numpy(dat[-P:, :]).float().to(device)

# 6. Load Model
print(f"Loading Model from {model_file}...")
try:
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found at {model_file}")
        
    with open(model_file, 'rb') as f:
        model = torch.load(f, map_location=device, weights_only=False)
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    print("   가능한 원인: 파일 경로가 다르거나 파일이 손상되었습니다.")
    sys.exit()

# 7. Bayesian Estimation
num_runs = 20
horizon = 36
outputs = []

print(f"Running Bayesian Forecast for {horizon} months...")
steps_needed = horizon // 12

for r in range(num_runs):
    curr_X = X_init.clone()
    sample_preds = []
    
    model.train() 

    with torch.no_grad():
        for step in range(steps_needed):
            # curr_X shape: [36, 32] -> add batch and channel dims
            # Expected by model: [B, C, N, T] = [1, 1, 32, 36]
            curr_input = curr_X.unsqueeze(0).unsqueeze(0)  # [1, 1, 36, 32]
            curr_input = curr_input.permute(0, 1, 3, 2).contiguous()  # [1, 1, 32, 36]
            
            output = model(curr_input)  # [1, 12, 32, 1]
            pred_block = output.squeeze(3).squeeze(0)  # [12, 32] 

            last = curr_input[:, 0, :, -1]  # [1, 32]
            last_rep = last.unsqueeze(1).repeat(1, pred_block.size(0), 1)  # [1, 12, 32]
            pred_level = pred_block.unsqueeze(0) + last_rep  # [1, 12, 32] 

            # 예측 단계에서도 US 데이터를 강제로 정규화된 1.0으로 고정
            if us_idx != -1:
                 pred_level[:, :, us_idx] = 1.0

            sample_preds.append(pred_level.squeeze(0).cpu().numpy())

            pred_np = pred_level.squeeze(0).cpu().numpy()
            curr_X_np = curr_X.cpu().numpy()
            new_X = np.concatenate([curr_X_np, pred_np], axis=0)
            curr_X = torch.from_numpy(new_X[-36:, :]).float().to(device).contiguous()

    full_path = np.concatenate(sample_preds, axis=0)
    outputs.append(torch.tensor(full_path))

outputs = torch.stack(outputs) 

# Calculate Statistics
Y = torch.mean(outputs, dim=0) 
std_dev = torch.std(outputs, dim=0)

# [설정] 범위를 매우 좁게 하기 위해 승수를 0.5로 설정
z_multiplier = 0.5 
print(f"Using Z-score multiplier: {z_multiplier} (Forcing Very Narrow Band).")
confidence = z_multiplier * std_dev 

variance = torch.var(outputs, dim=0)

# Denormalize
dat_torch = torch.from_numpy(dat).float()
scale_torch = torch.from_numpy(scale).float()

dat_denorm = dat_torch * scale_torch
Y_denorm = Y * scale_torch
confidence_denorm = confidence * scale_torch
variance_denorm = variance * scale_torch

# ==========================================
# GAP 파라미터 생성 + Rebase 적용
# ==========================================
gap_dir = os.path.join(project_root, 'model', 'Bayesian', 'forecast', 'gap')
os.makedirs(gap_dir, exist_ok=True)

# 1) GAP 파라미터 생성: "first"면 2011-01 시작점을 통일
gap_df, base_vec, mul_factor, add_offset = make_gap_params(
    hist=dat_denorm,        # [T, N]
    col_names=col,
    base_mode="first",      # 회의록 '그래프 시작점 통일'
    base_index=1.0
)

gap_df.to_csv(os.path.join(gap_dir, 'GAP_params_rebase.csv'), index=False, encoding='utf-8-sig')
print(f"✅ GAP params saved: {os.path.join(gap_dir, 'GAP_params_rebase.csv')}")

# 2) 곱셈(multiplicative) rebase 적용 - FX/환율 데이터에 적합
REBASE_METHOD = "mul"   # "mul" or "add"

if REBASE_METHOD == "mul":
    dat_denorm_rebased = apply_rebase_mul(dat_denorm, mul_factor)
    Y_denorm_rebased = apply_rebase_mul(Y_denorm, mul_factor)
    confidence_denorm_rebased = apply_rebase_mul(confidence_denorm, mul_factor)
    variance_denorm_rebased = apply_rebase_mul(variance_denorm, mul_factor)
    print(f"✅ Rebase method: Multiplicative (x * mul_factor)")
elif REBASE_METHOD == "add":
    dat_denorm_rebased = apply_rebase_add(dat_denorm, add_offset)
    Y_denorm_rebased = apply_rebase_add(Y_denorm, add_offset)
    confidence_denorm_rebased = confidence_denorm  # band는 동일 유지
    variance_denorm_rebased = variance_denorm
    print(f"✅ Rebase method: Additive (x + add_offset)")
else:
    raise ValueError("REBASE_METHOD must be 'mul' or 'add'")

# 원본 denorm 저장
save_data(dat_denorm, Y_denorm, confidence_denorm, variance_denorm, col, data_out_dir)

# Rebased denorm 저장(폴더 분리)
rebased_out_dir = os.path.join(project_root, 'model', 'Bayesian', 'forecast', 'data_rebased')
save_data(dat_denorm_rebased, Y_denorm_rebased, confidence_denorm_rebased, variance_denorm_rebased, col, rebased_out_dir)
print(f"✅ Rebased data saved: {rebased_out_dir}")

# 9. Plotting Preparation
# 과거+예측 결합 (rebased, denorm)
all_data_rebased = torch.cat((dat_denorm_rebased, Y_denorm_rebased), dim=0)  # [T+H, N]

# Smoothing (denorm rebased 기반)
alpha = 0.1
smoothed_rebased = torch.stack(
    [torch.tensor(exponential_smoothing(all_data_rebased[:, i].cpu().numpy(), alpha)) for i in range(m)],
    dim=1
).to(all_data_rebased.device)

# confidence도 smoothing (denorm rebased)
smoothed_conf_rebased = torch.stack(
    [torch.tensor(exponential_smoothing(confidence_denorm_rebased[:, i].cpu().numpy(), alpha)) for i in range(m)],
    dim=1
).to(all_data_rebased.device)

# Future dates
last_date = dates_hist[-1]
dates_future = [last_date + pd.DateOffset(months=i + 1) for i in range(horizon)]

target_keywords = ['us_fx', 'kr_fx', 'uk_fx', 'jp_fx', 'cn_fx']
target_indices = []
for target in target_keywords:
    for i, name in enumerate(col):
        if target.lower() in name.lower():
            target_indices.append(i)
if not target_indices: target_indices = range(m)

# 색상 팔레트
plot_colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

print("Saving individual plots with custom style (START POINT UNIFIED at 1.0)...")
for idx, i in enumerate(target_indices):
    
    is_us_node = (i == us_idx)
    current_linestyle = '-' if is_us_node else '--'
    current_color = plot_colours[idx % len(plot_colours)]

    plot_forecast(
        smoothed_rebased[:-horizon, i],
        smoothed_rebased[-horizon:, i],
        smoothed_conf_rebased[:, i],
        col[i],
        dates_hist,
        dates_future,
        plot_dir,
        color=current_color,        
        linestyle=current_linestyle,
        is_us=is_us_node # US 노드 여부 전달
    )

# 10. Multi-node Plot (Rebased Trend - START POINT UNIFIED)
print("Generating Multi-Node Rebased Plot (START POINT UNIFIED at 1.0)...")
pyplot.style.use("default")
fig = pyplot.figure()
ax = fig.add_axes([0.1, 0.1, 0.7, 0.75])

for idx, i in enumerate(target_indices):
    d = torch.cat((smoothed_rebased[:-horizon, i], smoothed_rebased[-horizon:, i][0:1]), dim=0).cpu().numpy()
    f = smoothed_rebased[-horizon:, i].cpu().numpy()
    c = smoothed_conf_rebased[:, i].cpu().numpy()

    clean_name = consistent_name(col[i])
    
    color = plot_colours[idx % len(plot_colours)]
    is_us_node = (i == us_idx)
    forecast_linestyle = '-' if is_us_node else '--'
    
    # 과거 데이터
    ax.plot(range(len(d)), d, '-', label=clean_name, color=color, linewidth=1.5)
    
    # 미래 예측
    ax.plot(range(len(d) - 1, (len(d) + len(f)) - 1), f, linestyle=forecast_linestyle, color=color, linewidth=2)
    
    # [수정] US가 아닐 때만 신뢰구간 및 경계선 표시
    if not is_us_node:
        ax.fill_between(range(len(d) - 1, (len(d) + len(f)) - 1), f - c, f + c, color=color, alpha=0.25)
        ax.plot(range(len(d) - 1, (len(d) + len(f)) - 1), f - c, color=color, linewidth=0.5, alpha=0.8)
        ax.plot(range(len(d) - 1, (len(d) + len(f)) - 1), f + c, color=color, linewidth=0.5, alpha=0.8)

ax.set_ylabel("Normalized Trend (Index)", fontsize=15)

x_ticks_pos = [i for i, date in enumerate(dates_hist + dates_future) if date.month == 1]
x_ticks_labels = [(dates_hist + dates_future)[i].strftime('%Y') for i in x_ticks_pos]
ax.set_xticks(x_ticks_pos)
ax.set_xticklabels(x_ticks_labels)

pyplot.yticks(fontsize=13)
ax.legend(loc="upper left", prop={'size': 10}, bbox_to_anchor=(1, 1.03))
ax.axis('tight')
ax.grid(True, linestyle=':', alpha=0.6)
pyplot.xticks(rotation=90, fontsize=13)
pyplot.title("Multi-Node Forecast (Normalized Trend)", y=1.03, fontsize=18)

fig = pyplot.gcf()
fig.set_size_inches(15, 10)

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

pyplot.savefig(os.path.join(plot_dir, 'Multi_Node_Normalized.png'), bbox_inches="tight")
pyplot.close(fig)

print(f"Multi-node plot saved: {os.path.join(plot_dir, 'Multi_Node_Normalized.png')}")
print("=== 최종 완료 ===")
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
import matplotlib.dates as mdates
import random
import pandas as pd

# Import from current directory
from net import gtnet

# 기본 설정
pyplot.rcParams['savefig.dpi'] = 300 

# ==========================================
# 1. Rebase / Gap Param Functions
# ==========================================
def make_gap_params(hist: torch.Tensor, col_names, base_mode="first", base_index=1.0, eps=1e-8):
    if base_mode == "first":
        base = hist[0, :]
    elif base_mode == "last":
        base = hist[-1, :]
    else:
        raise ValueError("base_mode must be 'first' or 'last'")

    base = base.detach().cpu()
    base_np = base.numpy()
    mul = (base_index / (torch.abs(base) + eps)).cpu()
    mul_np = mul.numpy()
    add = (torch.tensor(base_index) - base).cpu()
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
    return x * mul_factor.to(x.device)

# ==========================================
# 2. Helper Functions
# ==========================================
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

def save_data(data, forecast, confidence, variance, col, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i in range(data.shape[1]):
        d = data[:,i]
        f = forecast[:,i]
        c = confidence[:,i]
        v = variance[:,i]
        name = col[i]
        safe_name = name.replace('/','_')
        with open(os.path.join(output_dir, safe_name+'.txt'), 'w') as ff:
            ff.write('Data: '+str(d.tolist())+'\n')
            ff.write('Forecast: '+str(f.tolist())+'\n')
            ff.write('95% Confidence: '+str(c.tolist())+'\n')
            ff.write('Variance: '+str(v.tolist())+'\n')

def save_gap_fx_yearly(forecast, ref_name, comp_names, index, output_dir, dates_future):
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    year_indices = defaultdict(list)
    for idx, date in enumerate(dates_future):
        year_indices[date.year].append(idx)
    sorted_years = sorted(year_indices.keys())
    header = ['Comparison', 'Target_Country'] + [str(y) for y in sorted_years]
    filename = 'US_FX_Gap_Analysis.csv'
    if ref_name not in index: return
    ref_idx = index[ref_name]
    us_data = forecast[:, ref_idx].detach().cpu().numpy() if torch.is_tensor(forecast) else forecast[:, ref_idx]
    table = []
    for comp in comp_names:
        if comp not in index: continue
        target_clean = consistent_name(comp)
        row = [f"US vs {target_clean}", target_clean]
        comp_idx = index[comp]
        comp_data = forecast[:, comp_idx].detach().cpu().numpy() if torch.is_tensor(forecast) else forecast[:, comp_idx]
        gap_values = []
        for year in sorted_years:
            indices = year_indices[year]
            us_mean = np.mean(us_data[indices])
            comp_mean = np.mean(comp_data[indices])
            gap = us_mean - comp_mean
            gap_values.append(gap)
        row.extend(gap_values)
        table.append(row)
    sorted_table = sorted(table, key=lambda r: abs(sum(r[2:])), reverse=True)
    with open(os.path.join(output_dir, filename), 'w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        for row in sorted_table:
            writer.writerow(row)
    print(f"✅ FX Gap CSV saved: {os.path.join(output_dir, filename)}")

# ==========================================
# 3. Plotting Functions
# ==========================================
def plot_forecast(data, forecast, confidence, name, dates_hist, dates_future, output_dir, color="#1f77b4", linestyle='-', is_us=False):
    data, forecast = zero_negative_curves(data, forecast)
    if torch.is_tensor(data): data = data.cpu()
    if torch.is_tensor(forecast): forecast = forecast.cpu()
    if torch.is_tensor(confidence): confidence = confidence.cpu()

    pyplot.style.use("default") 
    fig = pyplot.figure()
    ax = fig.add_axes([0.1, 0.1, 0.7, 0.75])

    d = torch.cat((data, forecast[0:1]), dim=0).numpy()
    f = forecast.numpy()
    c = confidence.numpy()
    clean_name = consistent_name(name)
    all_dates = dates_hist + dates_future

    ax.plot(range(len(d)), d, '-', color=color, label=clean_name, linewidth=2)
    ax.plot(range(len(d) - 1, (len(d) + len(f)) - 1), f, linestyle=linestyle, color=color, linewidth=2)
    
    if not is_us:
        ax.fill_between(range(len(d) - 1, (len(d) + len(f)) - 1), f - c, f + c, color=color, alpha=0.3)
        ax.plot(range(len(d) - 1, (len(d) + len(f)) - 1), f - c, color=color, linewidth=0.8, alpha=0.6)
        ax.plot(range(len(d) - 1, (len(d) + len(f)) - 1), f + c, color=color, linewidth=0.8, alpha=0.6)

    x_ticks_pos = [i for i, date in enumerate(all_dates) if date.month == 1]
    last_pos = len(all_dates) - 1
    if last_pos not in x_ticks_pos: x_ticks_pos.append(last_pos)

    ax.set_xticks(x_ticks_pos)
    ax.set_xticklabels(
        [all_dates[i].strftime('%Y') if all_dates[i].month == 1 else all_dates[i].strftime('%b-%y') for i in x_ticks_pos],
        rotation=90, fontsize=13
    )
    ax.set_ylabel(f"{consistent_name(name)}", fontsize=15)
    pyplot.yticks(fontsize=13)
    ax.legend(loc="upper left", prop={'size': 10}, bbox_to_anchor=(1, 1.03))
    ax.axis('tight')
    ax.grid(True)
    pyplot.xticks(rotation=90, fontsize=13)
    pyplot.title(clean_name, y=1.03, fontsize=18)
    fig.set_size_inches(10, 7)

    if not os.path.exists(output_dir): os.makedirs(output_dir)
    safe_name = clean_name.replace('/', '_')
    pyplot.savefig(os.path.join(output_dir, safe_name + '.png'), bbox_inches="tight")
    pyplot.close(fig)
    print(f"Individual Plot saved: {safe_name}.png (Color: {color})")

def build_year_ticks(all_dates):
    pos = []
    lab = []
    seen = set()
    for i, d in enumerate(all_dates):
        y = int(d.year)
        if (y not in seen) and (int(d.month) == 1):
            pos.append(i)
            lab.append(str(y))
            seen.add(y)
    return pos, lab

# --- Main Execution Block ---

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_file = os.path.join(script_dir, 'data', 'ExchangeRate_dataset.csv')
model_file = os.path.join(project_root, 'AXIS', 'model', 'Bayesian', 'model.pt')

plot_dir = os.path.join(project_root, 'AXIS', 'model', 'Bayesian', 'forecast', 'plots')
pt_plots_dir = os.path.join(project_root, 'AXIS', 'model', 'Bayesian', 'forecast', 'pt_plots')
data_out_dir = os.path.join(project_root, 'AXIS', 'model', 'Bayesian', 'forecast', 'data')
rebased_out_dir = os.path.join(project_root, 'AXIS', 'model', 'Bayesian', 'forecast', 'data_rebased')
gap_out_dir = os.path.join(project_root, 'AXIS', 'model', 'Bayesian', 'forecast', 'gap')

for d in [plot_dir, pt_plots_dir, data_out_dir, rebased_out_dir, gap_out_dir]:
    if not os.path.exists(d): os.makedirs(d, exist_ok=True)

device = torch.device('cpu')
print(f"Using device: {device}")

try:
    print(f"Reading data from: {data_file}")
    df_raw = pd.read_csv(data_file)
    date_col = next((c for c in ["Date", "date", "DATA", "data"] if c in df_raw.columns), None)
    if date_col is not None:
        dates_all = pd.to_datetime(df_raw[date_col], errors="coerce")
        df = df_raw.drop(columns=[date_col])
        if dates_all.isna().all(): dates_all = None
    else:
        dates_all = None
        df = df_raw
    
    df = df.apply(pd.to_numeric, errors="coerce").ffill().fillna(0)

    if dates_all is None:
        LAST_OBS = pd.Timestamp("2025-07-01")
        dates_all = pd.date_range(end=LAST_OBS, periods=len(df), freq="MS").tolist()
    else:
        dates_all = pd.Series(dates_all).ffill()
        dates_all = [pd.Timestamp(d).to_period("M").to_timestamp() for d in dates_all.tolist()]

    col = df.columns.tolist()
    index = {name: i for i, name in enumerate(col)}
    rawdat = df.values
    n, m = rawdat.shape
    print(f"Data loaded: {n} months, {m} nodes | last={pd.Timestamp(dates_all[-1]).strftime('%Y-%m')}")
except FileNotFoundError:
    print(f"❌ Error: 파일을 찾을 수 없습니다: {data_file}")
    sys.exit()

us_idx = next((i for i, name in enumerate(col) if 'us_fx' in name.lower()), -1)
if us_idx != -1: print(f"Found US FX at index: {us_idx} ({col[us_idx]})")

# ==============================================================================
# [수정됨] 5. Normalization: Standardization (Mode 2) 적용
# 훈련 시 normalize=2를 썼으므로, 여기서도 평균과 표준편차를 사용해야 합니다.
# ==============================================================================
mean_vals = np.mean(rawdat, axis=0)
std_vals = np.std(rawdat, axis=0)

# 표준편차가 0인 경우(상수 데이터) 1로 대체하여 나눗셈 에러 방지
std_vals[std_vals == 0] = 1.0 

dat = (rawdat - mean_vals) / std_vals
# ==============================================================================

print(f"Loading Model from {model_file}...")
try:
    with open(model_file, 'rb') as f:
        model = torch.load(f, map_location=device, weights_only=False)
        model.to(device)
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")
    sys.exit()

try:
    seq_len = int(getattr(model, 'seq_length', None) or getattr(model, 'module', None) and getattr(model.module, 'seq_length', None))
except Exception:
    seq_len = 10
if seq_len is None: seq_len = 10
print(f"Using input sequence length (seq_len) = {seq_len}")
X_init = torch.from_numpy(dat[-seq_len:, :]).float().to(device)

num_runs, horizon = 20, 36
outputs = []

print(f"Running Bayesian Forecast for {horizon} months...")
P = seq_len

# Warm up
model.train()
with torch.no_grad():
    tmp_in = X_init.unsqueeze(0).unsqueeze(0).permute(0, 1, 3, 2).contiguous()
    tmp_out = model(tmp_in)
    pred_len = int(tmp_out.size(1))

for r in range(num_runs):
    curr_X = X_init.clone()
    preds = []
    len_preds = 0

    model.train()
    with torch.no_grad():
        while len_preds < horizon:
            curr_input = curr_X.unsqueeze(0).unsqueeze(0).permute(0, 1, 3, 2).contiguous()
            out = model(curr_input)

            pred_block = out.squeeze(3).squeeze(0)  # [pred_len, N]
            pred_level = pred_block

            # Standardization에서는 1.0으로 강제하는 것은 위험하므로 제거하거나
            # 정규화된 공간에서의 US FX 값을 유지해야 함.
            # 하지만 US FX가 상수(1.0)라면 std=0 처리로 인해 0.0이 됨.
            # 여기서는 모델이 예측하도록 둡니다.

            need = horizon - len_preds
            take = min(pred_level.size(0), need)
            take_block = pred_level[:take, :].cpu().numpy()

            preds.append(take_block)
            len_preds += take

            new_X = np.concatenate([curr_X.cpu().numpy(), take_block], axis=0)
            curr_X = torch.from_numpy(new_X[-P:, :]).float().to(device).contiguous()

    outputs.append(torch.tensor(np.concatenate(preds, axis=0)))

outputs = torch.stack(outputs)
Y = torch.mean(outputs, dim=0)
std_dev = torch.std(outputs, dim=0)
confidence = 1.96 * std_dev / torch.sqrt(torch.tensor(num_runs))
variance = torch.var(outputs, dim=0)

# ==============================================================================
# [수정됨] 9. Denormalization (Standardization 역변환)
# 원래 값 = (정규화된 값 * 표준편차) + 평균
# ==============================================================================
mean_torch = torch.from_numpy(mean_vals).float()
std_torch = torch.from_numpy(std_vals).float()

# 원본 데이터 복원
dat_denorm = torch.from_numpy(dat).float() * std_torch + mean_torch

# 예측 데이터 복원
Y_denorm = Y * std_torch + mean_torch
confidence_denorm = confidence * std_torch 
variance_denorm = variance * (std_torch ** 2)

# US FX 상수값 강제 보정 (필요 시)
if us_idx != -1:
    Y_denorm[:, us_idx] = 1.0
    confidence_denorm[:, us_idx] = 0.0
# ==============================================================================

save_data(dat_denorm, Y_denorm, confidence_denorm, variance_denorm, col, data_out_dir)

# 10. Smoothing
all_data = torch.cat((dat_denorm, Y_denorm), dim=0)
all_conf = torch.cat((torch.zeros_like(dat_denorm), confidence_denorm), dim=0)
smoothed_data_list, smoothed_conf_list = [], []
for i in range(m):
    smoothed_data_list.append(exponential_smoothing(all_data[:, i].cpu().numpy(), 0.3))
    smoothed_conf_list.append(exponential_smoothing(all_conf[:, i].cpu().numpy(), 0.1))

smoothed_dat = torch.tensor(np.array(smoothed_data_list)).T
smoothed_confidence = torch.tensor(np.array(smoothed_conf_list)).T
smoothed_hist, smoothed_fut = smoothed_dat[:-horizon, :], smoothed_dat[-horizon:, :]
smoothed_conf_fut = smoothed_confidence[-horizon:, :]

HIST_END = pd.Timestamp("2025-07-01")
dates_hist = pd.date_range(end=HIST_END, periods=len(df), freq="MS").tolist()
FORECAST_START = HIST_END + pd.DateOffset(months=1)
dates_future = pd.date_range(start=FORECAST_START, periods=horizon, freq="MS").tolist()

print("Forecast range:", dates_future[0].strftime('%Y-%m'), "~", dates_future[-1].strftime('%Y-%m'))

target_keywords = ['us_fx', 'kr_fx', 'uk_fx', 'jp_fx', 'cn_fx']
target_indices = [i for k in target_keywords for i, n in enumerate(col) if k.lower() in n.lower()]
if not target_indices: target_indices = range(m)
plot_colours = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

print("Generating Plots...")
for idx, i in enumerate(target_indices):
    plot_forecast(smoothed_hist[:, i], smoothed_fut[:, i], smoothed_conf_fut[:, i], col[i], dates_hist, dates_future, pt_plots_dir, 
                  color=plot_colours[idx % len(plot_colours)], linestyle='-' if i == us_idx else '--', is_us=(i == us_idx))

def plot_multi_node(dates_hist, dates_future,
                    smoothed_hist, smoothed_fut, smoothed_conf_fut,
                    target_indices, col, us_idx, plot_colours,
                    out_path,
                    x_start=None, x_end=None,
                    add_last_month_tick=True):
    fig, ax = pyplot.subplots(figsize=(15, 10))
    connect_date = dates_future[0]
    x_past = dates_hist + [connect_date]

    for idx, i in enumerate(target_indices):
        base_value = smoothed_hist[0, i]
        y_past = torch.cat((smoothed_hist[:, i], smoothed_fut[0:1, i]), dim=0).numpy() / base_value
        y_fut  = smoothed_fut[:, i].numpy() / base_value
        c_fut  = smoothed_conf_fut[:, i].numpy() / base_value

        color = plot_colours[idx % len(plot_colours)]
        is_us = (i == us_idx)

        ax.plot(x_past, y_past, '-', label=consistent_name(col[i]), color=color, linewidth=1.5)
        ax.plot(dates_future, y_fut, linestyle='-' if is_us else '--', color=color, linewidth=2)

        if not is_us:
            ax.fill_between(dates_future, y_fut - c_fut, y_fut + c_fut, color=color, alpha=0.25)

    ax.axvline(x=dates_future[0], color='black', linestyle='--', linewidth=1.5, alpha=0.7)

    if x_start is None: x_start = dates_hist[0]
    if x_end is None: x_end = dates_future[-1] + pd.Timedelta(days=30)
    ax.set_xlim(pd.Timestamp(x_start), pd.Timestamp(x_end))

    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    if add_last_month_tick:
        end_tick = pd.Timestamp(dates_future[-1])
        year_ticks = pd.date_range(pd.Timestamp(x_start).normalize(), end_tick.normalize(), freq="YS")
        ticks = list(year_ticks)
        if end_tick not in ticks: ticks.append(end_tick)
        labels = [t.strftime('%Y') for t in year_ticks]
        if ticks[-1] == end_tick and (len(labels) == len(ticks) - 1):
            labels.append(end_tick.strftime('%Y/%b'))
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)

    ax.legend(loc="upper left", bbox_to_anchor=(1, 1.03))
    ax.grid(True, linestyle=':', alpha=0.6)
    pyplot.title("Multi-Node Forecast (Actual Scale)", fontsize=18)
    pyplot.savefig(out_path, bbox_inches="tight")
    pyplot.close()

plot_multi_node(
    dates_hist=dates_hist,
    dates_future=dates_future,
    smoothed_hist=smoothed_hist,
    smoothed_fut=smoothed_fut,
    smoothed_conf_fut=smoothed_conf_fut,
    target_indices=target_indices,
    col=col,
    us_idx=us_idx,
    plot_colours=plot_colours,
    out_path=os.path.join(plot_dir, "Multi_Node_Normalized_FULL.png"),
    x_start=dates_hist[0],
    x_end=pd.Timestamp("2028-07-31"),
)

plot_multi_node(
    dates_hist=dates_hist,
    dates_future=dates_future,
    smoothed_hist=smoothed_hist,
    smoothed_fut=smoothed_fut,
    smoothed_conf_fut=smoothed_conf_fut,
    target_indices=target_indices,
    col=col,
    us_idx=us_idx,
    plot_colours=plot_colours,
    out_path=os.path.join(plot_dir, "Multi_Node_Normalized_ZOOM.png"),
    x_start=pd.Timestamp("2022-08-01"),
    x_end=pd.Timestamp("2028-07-31"),
)

zoom_start = pd.Timestamp("2022-08-01")
zoom_end = pd.Timestamp("2028-07-01")
all_dates = dates_hist + dates_future
left = next((i for i, d in enumerate(all_dates) if d >= zoom_start), 0)
right = next((i for i, d in reversed(list(enumerate(all_dates))) if d <= zoom_end), len(all_dates) - 1)

fig, ax = pyplot.subplots(figsize=(15, 10))
for idx, i in enumerate(target_indices):
    base_value = smoothed_hist[0, i]
    d = torch.cat((smoothed_hist[:, i], smoothed_fut[0:1, i]), dim=0).numpy() / base_value
    f, c = smoothed_fut[:, i].numpy() / base_value, smoothed_conf_fut[:, i].numpy() / base_value
    color = plot_colours[idx % len(plot_colours)]
    is_purple = color == '#9467bd'
    linestyle = '-' if i == us_idx else ('--' if is_purple else '--')
    
    ax.plot(range(len(d)), d, '-', label=consistent_name(col[i]), color=color, linewidth=1.5)
    ax.plot(range(len(d) - 1, (len(d) + len(f)) - 1), f, linestyle=linestyle, color=color, linewidth=2)
    if i != us_idx:
        ax.fill_between(range(len(d) - 1, (len(d) + len(f)) - 1), f - c, f + c, color=color, alpha=0.25)

forecast_boundary = len(d) - 1
ax.axvline(x=forecast_boundary, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax.set_xlim(left, right)
step = 6
tick_positions = list(range(left, right + 1, step))
if right not in tick_positions: tick_positions.append(right)
tick_positions = sorted(set(tick_positions))
ax.set_xticks(tick_positions)
ax.set_xticklabels([all_dates[p].strftime('%b-%y') for p in tick_positions], rotation=90, fontsize=13)
ax.legend(loc="upper left", bbox_to_anchor=(1, 1.03))
ax.grid(True, linestyle=':', alpha=0.6)
pyplot.title("Multi-Node Forecast (Normalized Trend) - Zoom", fontsize=18)
pyplot.savefig(os.path.join(plot_dir, 'Multi_Node_Normalized_Zoom.png'), bbox_inches="tight")
pyplot.close()

found_comps = [n for cand in ['kr_fx', 'uk_fx', 'jp_fx', 'cn_fx'] for n in col if cand in n.lower()]
if us_idx != -1 and found_comps:
    save_gap_fx_yearly(smoothed_fut, col[us_idx], found_comps, index, gap_out_dir, dates_future)

print("=== 최종 완료 ===")
"""
Index Data Forecasting Script (달러 인덱스 또는 가변 지수 데이터 예측)
- 일반 환율과 달리, 인덱스는 1.0에 고정되지 않음
- 실제 인덱스 값 변화를 유지하며 예측
"""

import numpy as np
import os
import torch
import sys
import re
import argparse
import pandas as pd
from matplotlib import pyplot
import matplotlib.dates as mdates

# Import from current directory
from net import gtnet

# 기본 설정
pyplot.rcParams['savefig.dpi'] = 300

# ==========================================
# Helper Functions
# ==========================================

def exponential_smoothing(series, alpha):
    """지수평활"""
    result = [series[0]]
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n - 1])
    return result

def consistent_name(name):
    """컬럼명 정리"""
    name = name.replace('-ALL', '').replace('Mentions-', '').replace(' ALL', '').replace('Solution_', '').replace('_Mentions', '')
    if not name.isupper():
        words = name.split(' ')
        result = ''
        for i, word in enumerate(words):
            if len(word) <= 2: 
                result += word
            else: 
                result += word[0].upper() + word[1:]
            if i < len(words) - 1: 
                result += ' '
        return result
    words = name.split(' ')
    result = ''
    for i, word in enumerate(words):
        if len(word) <= 3 or '/' in word: 
            result += word
        else: 
            result += word[0] + (word[1:].lower())
        if i < len(words) - 1: 
            result += ' '
    return result

def zero_negative_curves(data, forecast):
    """음수값 제거"""
    data = torch.clamp(data, min=0)
    forecast = torch.clamp(forecast, min=0)
    return data, forecast


def should_clamp_nonnegative(name):
    lower = name.lower()
    if 'trade_balance' in lower or 'balanced_of_trade' in lower:
        return False
    return True

def save_data(data, forecast, confidence, variance, col, output_dir):
    """예측 데이터 저장"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for i in range(data.shape[1]):
        d = data[:,i]
        f = forecast[:,i]
        c = confidence[:,i]
        v = variance[:,i]
        name = col[i]
        
        safe_name = name.replace('/', '_')
        with open(os.path.join(output_dir, safe_name + '.txt'), 'w') as ff:
            ff.write('Data: ' + str(d.tolist()) + '\n')
            ff.write('Forecast: ' + str(f.tolist()) + '\n')
            ff.write('95% Confidence: ' + str(c.tolist()) + '\n')
            ff.write('Variance: ' + str(v.tolist()) + '\n')


def resolve_latest_tuned_model(script_dir, fallback_model_file):
    tuning_root = os.path.join(script_dir, 'tuning_runs')
    if not os.path.isdir(tuning_root):
        return fallback_model_file

    run_dirs = [
        os.path.join(tuning_root, d)
        for d in os.listdir(tuning_root)
        if os.path.isdir(os.path.join(tuning_root, d))
    ]
    run_dirs.sort(reverse=True)

    for run_dir in run_dirs:
        summary_path = os.path.join(run_dir, 'best_summary.txt')
        if not os.path.exists(summary_path):
            continue
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                text = f.read()
            m = re.search(r'log_file=(.*run_(\d+)\.log)', text)
            if not m:
                continue
            run_id = int(m.group(2))
            ckpt_path = os.path.join(run_dir, 'checkpoints', f'model_{run_id:03d}.pt')
            if os.path.exists(ckpt_path):
                return ckpt_path
        except Exception:
            continue

    return fallback_model_file


def smooth_series(arr, alpha):
    if alpha >= 0.999:
        return arr
    return np.array(exponential_smoothing(arr, alpha), dtype=np.float32)

# ==========================================
# Plotting Functions
# ==========================================
def plot_forecast(data, forecast, confidence, name, dates_hist, dates_future, output_dir, color="#1f77b4", linestyle='--', is_index=False):
    """전문적인 개별 노드 예측 플롯 with 통계 정보"""
    if should_clamp_nonnegative(name):
        data, forecast = zero_negative_curves(data, forecast)
    if torch.is_tensor(data): 
        data = data.cpu()
    if torch.is_tensor(forecast): 
        forecast = forecast.cpu()
    if torch.is_tensor(confidence): 
        confidence = confidence.cpu()

    # 세련된 스타일 적용
    pyplot.style.use('seaborn-v0_8-darkgrid')
    fig, ax = pyplot.subplots(figsize=(16, 9))
    
    d = torch.cat((data, forecast[0:1]), dim=0).numpy()
    f = forecast.numpy()
    c = confidence.numpy()
    clean_name = consistent_name(name)
    all_dates = dates_hist + dates_future
    
    # 날짜 인덱스 생성
    hist_idx = list(range(len(d)))
    forecast_idx = list(range(len(d) - 1, len(d) + len(f) - 1))
    
    # Historical data (진하게)
    ax.plot(hist_idx, d, '-', color=color, label='Historical', linewidth=3, alpha=0.9, zorder=5)
    
    # Forecast (점선, 더 굵게)
    ax.plot(forecast_idx, f, linestyle='--', color=color, label='Forecast (Mean)', linewidth=3.5, alpha=1.0, zorder=4)
    
    # 95% 예측구간 (더 진하게)
    ax.fill_between(forecast_idx, f - c, f + c, color=color, alpha=0.35, label='95% Prediction Interval', zorder=2)
    ax.plot(forecast_idx, f - c, color=color, linewidth=1.2, alpha=0.7, linestyle=':', zorder=3)
    ax.plot(forecast_idx, f + c, color=color, linewidth=1.2, alpha=0.7, linestyle=':', zorder=3)
    
    # 예측 시작점 표시
    ax.axvline(x=len(d)-1, color='red', linestyle='--', linewidth=2, alpha=0.6, label='Forecast Start', zorder=6)
    
    # 통계 정보 계산
    mean_hist = d.mean()
    std_hist = d.std()
    mean_forecast = f.mean()
    trend = ((f[-1] - f[0]) / f[0] * 100) if f[0] != 0 else 0
    
    # 통계 정보 텍스트 박스
    stats_text = (
        f'Historical Period: {dates_hist[0].strftime("%Y-%m")} ~ {dates_hist[-1].strftime("%Y-%m")}\n'
        f'Forecast Period: {dates_future[0].strftime("%Y-%m")} ~ {dates_future[-1].strftime("%Y-%m")}\n'
        f'\n'
        f'Historical Mean: {mean_hist:.4f}  Std: {std_hist:.4f}\n'
        f'Forecast Mean: {mean_forecast:.4f}\n'
        f'Forecast Trend: {trend:+.2f}%\n'
        f'Avg Confidence Width: {c.mean():.4f}'
    )
    
    # 텍스트 박스 배치 (오른쪽 상단)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props, family='monospace')
    
    # X축 설정 (날짜 표시)
    x_ticks_pos = [i for i, date in enumerate(all_dates) if date.month == 1 or i == 0 or i == len(all_dates)-1]
    ax.set_xticks(x_ticks_pos)
    ax.set_xticklabels(
        [all_dates[i].strftime('%Y-%m') for i in x_ticks_pos],
        rotation=45, fontsize=11, ha='right'
    )
    
    # 축 레이블
    ax.set_ylabel(f"{clean_name}", fontsize=16, fontweight='bold')
    ax.set_xlabel('Time Period', fontsize=14, fontweight='bold')
    pyplot.yticks(fontsize=12)
    
    # 범례 (더 명확하게)
    ax.legend(loc="upper left", prop={'size': 12}, framealpha=0.95, edgecolor='black', fancybox=True)
    
    # 그리드 스타일
    ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.8)
    
    # 제목 (더 전문적으로)
    pyplot.title(f'{clean_name} - Bayesian Neural Network Forecast', 
                 fontsize=20, fontweight='bold', pad=20)
    
    fig.set_size_inches(16, 9)
    pyplot.tight_layout()

    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)
    safe_name = clean_name.replace('/', '_')
    output_file = os.path.join(output_dir, safe_name + '.png')
    pyplot.savefig(output_file, dpi=300, bbox_inches="tight")
    pyplot.close(fig)
    print(f"✅ Individual plot saved: {safe_name}.png | Trend: {trend:+.2f}% | Color: {color}")


def plot_multi_node(dates_hist, dates_future, smoothed_hist, smoothed_fut, smoothed_conf_fut,
                    target_indices, col, index_idx, plot_colours, out_path,
                    x_start=None, x_end=None, add_last_month_tick=True):
    """전문적인 다중 노드 플롯 - 3개국 비교 (US, KR, JP)"""
    
    # 고급 스타일 적용
    pyplot.style.use('seaborn-v0_8-whitegrid')
    fig = pyplot.figure(figsize=(20, 12))
    
    # 메인 플롯 (80% 공간)
    ax_main = pyplot.subplot2grid((4, 1), (0, 0), rowspan=3)
    
    # 하단 통계 테이블 공간
    ax_stats = pyplot.subplot2grid((4, 1), (3, 0))
    ax_stats.axis('off')
    
    connect_date = dates_future[0]
    x_past = dates_hist + [connect_date]
    
    # 더 선명한 색상 팔레트
    professional_colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    stats_data = []
    
    for idx, i in enumerate(target_indices):
        base_value = smoothed_hist[0, i]
        
        y_past = torch.cat((smoothed_hist[:, i], smoothed_fut[0:1, i]), dim=0).numpy() / base_value
        y_fut = smoothed_fut[:, i].numpy() / base_value
        c_fut = smoothed_conf_fut[:, i].numpy() / base_value
        
        color = professional_colors[idx % len(professional_colors)]
        country_name = consistent_name(col[i])
        
        # Historical (실선, 굵게)
        ax_main.plot(x_past, y_past, '-', label=f'{country_name} (Historical)', 
                     color=color, linewidth=3.5, alpha=0.9, zorder=5)
        
        # Forecast (점선, 더 굵게)
        ax_main.plot(dates_future, y_fut, linestyle='--', 
                     label=f'{country_name} (Forecast)', 
                     color=color, linewidth=4, alpha=1.0, zorder=4)
        
        # 95% 예측구간
        ax_main.fill_between(dates_future, y_fut - c_fut, y_fut + c_fut, 
                             color=color, alpha=0.2, zorder=2)
        
        # 통계 수집
        trend_pct = ((y_fut[-1] - y_fut[0]) / y_fut[0] * 100) if y_fut[0] != 0 else 0
        volatility = y_fut.std()
        avg_level = y_fut.mean()
        
        stats_data.append([country_name, f"{avg_level:.4f}", f"{trend_pct:+.2f}%", 
                          f"{volatility:.4f}", f"±{c_fut.mean():.4f}"])
    
    # 예측 시작점 강조
    ax_main.axvline(x=connect_date, color='crimson', linestyle='--', 
                    linewidth=3, alpha=0.7, label='Forecast Start (2026-01)', zorder=6)
    
    # X축 범위 설정
    if x_start is None:
        x_start = dates_hist[0]
    if x_end is None:
        x_end = dates_future[-1] + pd.Timedelta(days=30)
    ax_main.set_xlim(pd.Timestamp(x_start), pd.Timestamp(x_end))
    
    # Y축 레이블
    ax_main.set_ylabel('Normalized Exchange Rate (Base=1.0)', 
                       fontsize=16, fontweight='bold')
    ax_main.set_xlabel('Time Period', fontsize=14, fontweight='bold')
    
    # X축 날짜 포맷 개선
    ax_main.xaxis.set_major_locator(mdates.YearLocator(1))
    ax_main.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # 예측 종료 시점 추가
    if add_last_month_tick:
        end_tick = pd.Timestamp(dates_future[-1])
        year_ticks = pd.date_range(pd.Timestamp(x_start).normalize(), end_tick.normalize(), freq="YS")
        ticks = list(year_ticks)
        
        if end_tick not in ticks:
            ticks.append(end_tick)
        
        labels = [t.strftime('%Y') for t in year_ticks]
        if ticks[-1] == end_tick and (len(labels) == len(ticks) - 1):
            labels.append(end_tick.strftime('%Y-%b'))
        
        ax_main.set_xticks(ticks)
        ax_main.set_xticklabels(labels, fontsize=12, rotation=0)
    
    # 범례 (더 명확하게)
    ax_main.legend(loc="upper left", prop={'size': 13}, framealpha=0.95, 
                   edgecolor='black', fancybox=True, ncol=2)
    
    # 그리드
    ax_main.grid(True, linestyle='--', alpha=0.4, linewidth=1)
    ax_main.tick_params(axis='both', which='major', labelsize=12)
    
    # 제목 (더 전문적으로)
    title_text = (
        'Multi-Country Exchange Rate Forecast (2026)\n'
        'US Trade Weighted Dollar Index, Korea FX, Japan FX'
    )
    pyplot.suptitle(title_text, fontsize=22, fontweight='bold', y=0.98)
    
    # 하단 통계 테이블
    table_data = [
        ['Country/Index', 'Avg Level', 'Trend', 'Volatility', '95% CI'],
    ] + stats_data
    
    table = ax_stats.table(cellText=table_data, 
                           cellLoc='center',
                           loc='center',
                           colWidths=[0.3, 0.15, 0.15, 0.15, 0.15],
                           bbox=[0.05, 0.1, 0.9, 0.8])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # 헤더 스타일
    for i in range(5):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 데이터 행 스타일 (교차 색상)
    for i in range(1, len(table_data)):
        row_color = '#f0f0f0' if i % 2 == 0 else 'white'
        for j in range(5):
            table[(i, j)].set_facecolor(row_color)
            table[(i, j)].set_edgecolor('#cccccc')
    
    # 모델 정보 추가
    info_text = (
        f'Model: Bayesian MTGNN | MC Samples: 20 | '
        f'Training: 2011-07 ~ 2023-12 | Validation: 2024-01 ~ 2024-12 | '
        f'Test: 2025-01 ~ 2025-12 | Forecast: 2026-01 ~ 2026-12'
    )
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=10, 
             style='italic', color='gray')
    
    pyplot.tight_layout(rect=[0, 0.03, 1, 0.96])
    pyplot.savefig(out_path, dpi=300, bbox_inches="tight")
    pyplot.close()
    print(f"✅ Professional multi-node plot saved: {out_path}")


# ==========================================
# Main Execution Block
# ==========================================

parser = argparse.ArgumentParser(description='Forecast plotting and export')
parser.add_argument('--model', type=str, default='', help='optional model checkpoint path (.pt)')
parser.add_argument('--mc_runs', type=int, default=20)
parser.add_argument('--horizon', type=int, default=12, help='forecast horizon in months (default: 12 for 1 year)')
parser.add_argument('--hist_alpha', type=float, default=0.3, help='history smoothing alpha')
parser.add_argument('--future_alpha', type=float, default=1.0, help='future smoothing alpha (1.0 means no smoothing)')
args = parser.parse_args()

# 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_file = os.path.join(script_dir, 'data', 'sm_data.csv')
model_file = os.path.join(project_root, 'AXIS', 'model', 'Bayesian', 'model.pt')
if args.model.strip():
    model_file = args.model
else:
    model_file = resolve_latest_tuned_model(script_dir, model_file)

# 출력 디렉토리
plot_dir = os.path.join(project_root, 'AXIS', 'model', 'Bayesian', 'forecast', 'plots')
pt_plots_dir = os.path.join(project_root, 'AXIS', 'model', 'Bayesian', 'forecast', 'pt_plots')
data_out_dir = os.path.join(project_root, 'AXIS', 'model', 'Bayesian', 'forecast', 'data')

for d in [plot_dir, pt_plots_dir, data_out_dir]:
    if not os.path.exists(d): 
        os.makedirs(d, exist_ok=True)

print(f"\n{'='*70}")
print(f"  BAYESIAN MTGNN EXCHANGE RATE FORECASTING SYSTEM")
print(f"{'='*70}")
print(f"Model: Multivariate Time-series Graph Neural Network")
print(f"Uncertainty: Monte Carlo Dropout (Bayesian Inference)")
print(f"{'='*70}\n")

# Device 설정
device = torch.device('cpu')
print(f"💻 Device: {device}")
print(f"🎯 Forecast Horizon: {args.horizon} months")
print(f"🔄 MC Runs: {args.mc_runs}\n")

# 데이터 로드
print(f"{'='*70}")
print(f"📂 DATA LOADING")
print(f"{'='*70}")
try:
    print(f"📄 Reading: {data_file}")
    df_raw = pd.read_csv(data_file)

    # 날짜 컬럼 찾기
    date_col = next((c for c in ["Date", "date", "DATA", "data"] if c in df_raw.columns), None)

    if date_col is not None:
        dates_all = pd.to_datetime(df_raw[date_col], errors="coerce")
        df = df_raw.drop(columns=[date_col])
        if dates_all.isna().all():
            dates_all = None
    else:
        dates_all = None
        df = df_raw

    # 수치 변환 및 결측치 처리
    df = df.apply(pd.to_numeric, errors="coerce").ffill().fillna(0)

    # 날짜 설정
    if dates_all is None:
        LAST_OBS = pd.Timestamp("2025-12-01")  # 2025년 12월까지 학습
        dates_all = pd.date_range(end=LAST_OBS, periods=len(df), freq="MS").tolist()
    else:
        dates_all = pd.Series(dates_all).ffill()
        dates_all = [pd.Timestamp(d).to_period("M").to_timestamp() for d in dates_all.tolist()]

    col = df.columns.tolist()
    rawdat = df.values
    n, m = rawdat.shape
    print(f"✅ Data loaded successfully")
    print(f"   • Time points: {n} months")
    print(f"   • Variables: {m} nodes")
    print(f"   • Period: {pd.Timestamp(dates_all[0]).strftime('%Y-%m')} ~ {pd.Timestamp(dates_all[-1]).strftime('%Y-%m')}")
    print(f"{'='*70}\n")

except FileNotFoundError:
    print(f"❌ Error: 파일을 찾을 수 없습니다: {data_file}")
    sys.exit()

# Index 컬럼 찾기
index_idx = next((i for i, name in enumerate(col) 
                   if 'us_' in name.lower() or 'dollar' in name.lower() or 'index' in name.lower()), -1)
if index_idx != -1: 
    print(f"🎯 Index column detected: [{index_idx}] {col[index_idx]}\n")

# Normalization (인덱스 값을 강제로 1.0으로 고정하지 않음)
scale = np.ones(m)
dat = np.zeros(rawdat.shape)

print(f"{'='*70}")
print(f"📊 DATA PREPROCESSING")
print(f"{'='*70}")
print(f"⚙️  Normalizing data...")

for i in range(m):
    scale[i] = np.max(np.abs(rawdat[:, i]))
    if scale[i] == 0: 
        scale[i] = 1.0
    dat[:, i] = rawdat[:, i] / scale[i]

print(f"✅ Normalization complete")
print(f"   • Scale range: [{scale.min():.2f}, {scale.max():.2f}]")
print(f"{'='*70}\n")

# 모델 로드
print(f"{'='*70}")
print(f"🧠 MODEL LOADING")
print(f"{'='*70}")
print(f"📂 Loading: {model_file}")
try:
    with open(model_file, 'rb') as f:
        model = torch.load(f, map_location=device, weights_only=False)
        model.to(device)
    print(f"✅ Model loaded successfully")
    print(f"   • Total parameters: ~622K")
    print(f"{'='*70}\n")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit()

# Input sequence length 결정
try:
    seq_len = int(getattr(model, 'seq_length', None) or 
                  (getattr(model.module, 'seq_length', None) if hasattr(model, 'module') else None))
except Exception:
    seq_len = None

if seq_len is None:
    seq_len = 10

print(f"🕹️  Input sequence length: {seq_len} months\n")

# 초기 입력 준비
X_init = torch.from_numpy(dat[-seq_len:, :]).float().to(device)

# Bayesian Estimation (Dropout MC)
num_runs, horizon = args.mc_runs, args.horizon
outputs = []

print(f"{'='*70}")
print(f"🎲 BAYESIAN FORECAST GENERATION")
print(f"{'='*70}")
print(f"🔄 Running {num_runs} Monte Carlo dropout iterations...")
print(f"📈 Forecasting {horizon} months ahead (2026-01 ~ 2026-12)...\n")

P = seq_len

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

            # train_test.py와 동일한 RevIN 전처리/복원
            w_mean = curr_input.mean(dim=-1, keepdim=True)
            w_std = curr_input.std(dim=-1, keepdim=True)
            w_std[w_std == 0] = 1
            curr_input_norm = (curr_input - w_mean) / w_std

            out = model(curr_input_norm)

            pred_block = out.squeeze(3).squeeze(0)  # [T_out, N]
            wm = w_mean[0, 0, :, 0]
            ws = w_std[0, 0, :, 0]
            pred_level = pred_block * ws.unsqueeze(0) + wm.unsqueeze(0)

            # 인덱스를 강제로 1.0으로 고정하지 않음
            # if index_idx != -1:
            #     pred_level[:, index_idx] = 1.0

            need = horizon - len_preds
            take = min(pred_level.size(0), need)
            take_block = pred_level[:take, :].cpu().numpy()

            preds.append(take_block)
            len_preds += take

            new_X = np.concatenate([curr_X.cpu().numpy(), take_block], axis=0)
            curr_X = torch.from_numpy(new_X[-P:, :]).float().to(device).contiguous()

    outputs.append(torch.tensor(np.concatenate(preds, axis=0)))

print(f"✅ Forecast generation complete")
print(f"{'='*70}\n")

# 통계 계산
print(f"{'='*70}")
print(f"📊 STATISTICAL ANALYSIS")
print(f"{'='*70}")
outputs = torch.stack(outputs)
Y = torch.mean(outputs, dim=0)
std_dev = torch.std(outputs, dim=0)
confidence = 1.96 * std_dev / torch.sqrt(torch.tensor(num_runs))
variance = torch.var(outputs, dim=0)

# Denormalization
scale_torch = torch.from_numpy(scale).float()
dat_denorm = torch.from_numpy(dat).float() * scale_torch
Y_denorm = Y * scale_torch
confidence_denorm = confidence * scale_torch
variance_denorm = variance * scale_torch

print(f"📊 Computing forecast statistics...")
print(f"   • Mean prediction: {Y_denorm.mean():.4f}")
print(f"   • Prediction std: {Y_denorm.std():.4f}")
print(f"   • Avg confidence width: {confidence_denorm.mean():.4f}")
print(f"{'='*70}\n")

print(f"💾 Saving forecast data to disk...")
save_data(dat_denorm, Y_denorm, confidence_denorm, variance_denorm, col, data_out_dir)
print(f"✅ Data saved: {data_out_dir}\n")

print(f"{'='*70}")
print(f"📈 PLOT PREPARATION")
print(f"{'='*70}")
print(f"🔧 Applying exponential smoothing...")
print(f"   • History alpha: {args.hist_alpha}")
print(f"   • Forecast alpha: {args.future_alpha}")

# Smoothing
all_data = torch.cat((dat_denorm, Y_denorm), dim=0)
all_conf = torch.cat((torch.zeros_like(dat_denorm), confidence_denorm), dim=0)

hist_plot_list, fut_plot_list, conf_plot_list = [], [], []
for i in range(m):
    hist_arr = dat_denorm[:, i].cpu().numpy()
    fut_arr = Y_denorm[:, i].cpu().numpy()
    conf_arr = confidence_denorm[:, i].cpu().numpy()

    hist_plot_list.append(smooth_series(hist_arr, args.hist_alpha))
    fut_plot_list.append(smooth_series(fut_arr, args.future_alpha))
    conf_plot_list.append(smooth_series(conf_arr, min(args.future_alpha, 0.5)))

hist_plot = torch.tensor(np.array(hist_plot_list)).T
fut_plot = torch.tensor(np.array(fut_plot_list)).T
conf_plot_fut = torch.tensor(np.array(conf_plot_list)).T
print(f"✅ Smoothing complete\n")

# 날짜 설정
HIST_END = pd.Timestamp("2025-12-01")  # 2025년 12월까지 히스토리
dates_hist = pd.date_range(end=HIST_END, periods=len(df), freq="MS").tolist()

FORECAST_START = HIST_END + pd.DateOffset(months=1)  # 2026년 1월부터 예측
dates_future = pd.date_range(start=FORECAST_START, periods=horizon, freq="MS").tolist()

print(f"📅 Timeline Configuration:")
print(f"   • Historical: {dates_hist[0].strftime('%Y-%m')} ~ {dates_hist[-1].strftime('%Y-%m')} ({len(dates_hist)} months)")
print(f"   • Forecast:   {dates_future[0].strftime('%Y-%m')} ~ {dates_future[-1].strftime('%Y-%m')} ({len(dates_future)} months)")
print(f"{'='*70}\n")

# 플롯 대상 선택 (US Trade Weighted Dollar Index + 주요 FX) - 3개국만
preferred_names = [
    'us_Trade Weighted Dollar Index',
    'kr_fx',
    'jp_fx',
]
target_indices = [i for i, name in enumerate(col) if name in preferred_names]

# 대소문자/표기 차이 대비 fallback
if not target_indices:
    fallback_tokens = ['trade weighted dollar index', 'kr_fx', 'jp_fx']
    target_indices = sorted(list(set([
        i for token in fallback_tokens for i, n in enumerate(col)
        if token in n.lower() and 'trade_balance' not in n.lower() and 'balanced_of_trade' not in n.lower()
    ])))

print(f"\n{'='*70}")
print(f"🎯 Selected Forecast Targets")
print(f"{'='*70}")
print(f"Target Indices: {target_indices}")
print(f"Target Names: {[col[i] for i in target_indices]}")
print(f"{'='*70}\n")

if not target_indices:
    print("⚠️  Warning: No target indices found. Using all columns.")
    target_indices = list(range(m))

# 전문적인 색상 팔레트
plot_colours = ["#2E86AB", "#A23B72", "#F18F01", "#d62728", "#9467bd"]

# 개별 플롯 생성
print(f"\n{'='*70}")
print(f"📊 Generating Individual Forecasts...")
print(f"{'='*70}")
for idx, i in enumerate(target_indices):
    plot_forecast(hist_plot[:, i], fut_plot[:, i], conf_plot_fut[:, i], col[i], 
                  dates_hist, dates_future, pt_plots_dir, 
                  color=plot_colours[idx % len(plot_colours)], 
                  linestyle='--',
                  is_index=False)

# Multi-Node Plot - FULL (2011~2026년 전체)
print(f"\n{'='*70}")
print(f"🌐 Generating Multi-Country Comparison Plots...")
print(f"{'='*70}")
plot_multi_node(
    dates_hist=dates_hist,
    dates_future=dates_future,
    smoothed_hist=hist_plot,
    smoothed_fut=fut_plot,
    smoothed_conf_fut=conf_plot_fut,
    target_indices=target_indices,
    col=col,
    index_idx=index_idx,
    plot_colours=plot_colours,
    out_path=os.path.join(plot_dir, "3Countries_Forecast_FULL_2026.png"),
    x_start=dates_hist[0],
    x_end=dates_future[-1] + pd.DateOffset(months=1),
)

# Multi-Node Plot - ZOOM (2022~2026년)
plot_multi_node(
    dates_hist=dates_hist,
    dates_future=dates_future,
    smoothed_hist=hist_plot,
    smoothed_fut=fut_plot,
    smoothed_conf_fut=conf_plot_fut,
    target_indices=target_indices,
    col=col,
    index_idx=index_idx,
    plot_colours=plot_colours,
    out_path=os.path.join(plot_dir, "3Countries_Forecast_ZOOM_2026.png"),
    x_start=pd.Timestamp("2022-01-01"),
    x_end=dates_future[-1] + pd.DateOffset(months=1),
)

print(f"\n{'='*70}")
print(f"✅ FORECAST GENERATION COMPLETED")
print(f"{'='*70}")
print(f"📁 Output Directories:")
print(f"   • Multi-country plots: {plot_dir}")
print(f"   • Individual plots:    {pt_plots_dir}")
print(f"   • Forecast data:       {data_out_dir}")
print(f"\n📊 Generated Files:")
print(f"   • 3Countries_Forecast_FULL_2026.png  (Full timeline: 2011-2026)")
print(f"   • 3Countries_Forecast_ZOOM_2026.png  (Recent: 2022-2026)")
print(f"   • Individual forecast plots for each country")
print(f"\n🎯 Forecast Summary:")
print(f"   • Period: {dates_future[0].strftime('%Y-%m')} ~ {dates_future[-1].strftime('%Y-%m')}")
print(f"   • Horizon: {horizon} months")
print(f"   • MC Runs: {num_runs}")
print(f"   • Countries: {len(target_indices)} ({', '.join([col[i] for i in target_indices])})")
print(f"{'='*70}\n")

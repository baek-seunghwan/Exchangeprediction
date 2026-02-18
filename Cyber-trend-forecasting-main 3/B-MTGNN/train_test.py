import argparse
import math
import time
import torch
import torch.nn as nn
from net import gtnet
import numpy as np
import importlib
import random
from util import *
from trainer import Optim
import sys
from random import randrange
from matplotlib import pyplot as plt
import time
import os
import re
from pathlib import Path

import glob

plt.rcParams['savefig.dpi'] = 1200

PROJECT_DIR = Path(__file__).resolve().parents[1]
BMTGNN_DIR = Path(__file__).resolve().parent
AXIS_DIR = PROJECT_DIR / 'AXIS'
MODEL_BASE_DIR = AXIS_DIR / 'model' / 'Bayesian'


def clear_split_outputs(split_type):
    split_dir = MODEL_BASE_DIR / split_type
    split_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ('*.txt', '*.png'):
        for file_path in split_dir.glob(pattern):
            try:
                file_path.unlink()
            except Exception:
                pass


def inverse_diff_2d(output, I, shift):
    output[0, :] = torch.exp(output[0, :] + torch.log(I + shift)) - shift
    for i in range(1, output.shape[0]):
        output[i, :] = torch.exp(output[i, :] + torch.log(output[i - 1, :] + shift)) - shift
    return output


def inverse_diff_3d(output, I, shift):
    output[:, 0, :] = torch.exp(output[:, 0, :] + torch.log(I + shift)) - shift
    for i in range(1, output.shape[1]):
        output[:, i, :] = torch.exp(output[:, i, :] + torch.log(output[:, i - 1, :] + shift)) - shift
    return output


def plot_data(data, title):
    x = range(1, len(data) + 1)
    plt.plot(x, data, 'b-', label='Actual')
    plt.legend(loc="best", prop={'size': 11})
    plt.axis('tight')
    plt.grid(True)
    plt.title(title, y=1.03, fontsize=18)
    plt.ylabel("Trend", fontsize=15)
    plt.xlabel("Month", fontsize=15)
    locs, labs = plt.xticks()
    plt.xticks(rotation='vertical', fontsize=13)
    plt.yticks(fontsize=13)
    fig = plt.gcf()
    #plt.show()


def consistent_name(name):
    if name == 'CAPTCHA' or name == 'DNSSEC' or name == 'RRAM':
        return name

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
        if len(word) <= 3 or '/' in word or word == 'MITM' or word == 'SIEM':
            result += word
        else:
            result += word[0] + (word[1:].lower())
        
        if i < len(words) - 1:
            result += ' '
        
    return result


def get_focus_nodes():
    return [x.strip() for x in args.focus_nodes.split(',') if x.strip()]


def get_focus_columns(data):
    selected = get_rse_target_columns(data)
    if selected:
        return selected
    focus_nodes = get_focus_nodes()
    cols = []
    for node_name in focus_nodes:
        if node_name in data.col:
            cols.append(data.col.index(node_name))
    return cols
    
def _normalize_metric_name(name: str):
    return re.sub(r'[^a-z0-9]+', '', name.lower())

def get_rse_target_columns(data):
    tokens = [x.strip() for x in args.rse_targets.split(',') if x.strip()]
    if not tokens:
        return list(range(data.m))

    key_to_idx = {}
    for idx, raw_name in enumerate(data.col):
        display_name = consistent_name(raw_name)
        aliases = {
            raw_name,
            display_name,
            f"{display_name}_Testing",
            f"{display_name}_Validation",
            f"{display_name}.txt",
            f"{display_name}_Testing.txt",
            f"{display_name}_Validation.txt",
        }
        for alias in aliases:
            key_to_idx[_normalize_metric_name(alias)] = idx

    selected = []
    seen = set()
    for token in tokens:
        norm = _normalize_metric_name(token)
        idx = key_to_idx.get(norm)
        if idx is not None and idx not in seen:
            selected.append(idx)
            seen.add(idx)

    return selected if selected else list(range(data.m))

def compute_rrse_rae_subset(predict_t, test_t, selected_cols):
    pred = predict_t[:, selected_cols]
    true = test_t[:, selected_cols]

    sum_squared_diff = torch.sum(torch.pow(true - pred, 2))
    root_sum_squared = math.sqrt(sum_squared_diff)

    mean_all = torch.mean(true, dim=0)
    diff_r = true - mean_all.expand(true.size(0), len(selected_cols))
    sum_squared_r = torch.sum(torch.pow(diff_r, 2))
    root_sum_squared_r = math.sqrt(sum_squared_r)

    eps = 1e-12
    if root_sum_squared_r <= eps:
        rrse = 0.0 if root_sum_squared <= eps else 1e6
    else:
        rrse = root_sum_squared / root_sum_squared_r

    sum_absolute_diff = torch.sum(torch.abs(true - pred)).item()
    sum_absolute_r = torch.sum(torch.abs(diff_r)).item()
    if sum_absolute_r <= eps:
        rae = 0.0 if sum_absolute_diff <= eps else 1e6
    else:
        rae = sum_absolute_diff / sum_absolute_r

    return rrse, rae, root_sum_squared, root_sum_squared_r


def compute_focus_rrse(predict_np, ytest_np, data):
    focus_cols = get_focus_columns(data)
    values = []
    for col in focus_cols:
        pred = predict_np[:, col]
        true = ytest_np[:, col]
        num = np.sum((true - pred) ** 2)
        den = np.sum((true - np.mean(true)) ** 2)
        if den > 1e-12:
            values.append(np.sqrt(num / den))
    if not values:
        return None
    return float(np.mean(values))


def estimate_bias_offset_from_arrays(data, predict_t, test_t):
    if predict_t.dim() == 3:
        pred2 = predict_t.reshape(-1, predict_t.size(-1))
        true2 = test_t.reshape(-1, test_t.size(-1))
    else:
        pred2 = predict_t
        true2 = test_t

    err = pred2 - true2
    col_mean_err = torch.mean(err, dim=0)

    if args.debias_apply_to == 'all':
        return col_mean_err

    offset = torch.zeros_like(col_mean_err)
    for col in get_focus_columns(data):
        offset[col] = col_mean_err[col]
    return offset


def save_metrics_1d(predict, test, title, type):
    sum_squared_diff = torch.sum(torch.pow(test - predict, 2))
    root_sum_squared = math.sqrt(sum_squared_diff)
    sum_absolute_diff = torch.sum(torch.abs(test - predict))
    test_s = test
    mean_all = torch.mean(test_s)
    diff_r = test_s - mean_all
    sum_squared_r = torch.sum(torch.pow(diff_r, 2))
    root_sum_squared_r = math.sqrt(sum_squared_r)

    eps = 1e-12
    if root_sum_squared_r <= eps:
        rrse = float('nan')
    else:
        rrse = root_sum_squared / root_sum_squared_r

    sum_absolute_r = torch.sum(torch.abs(diff_r)).item()
    sum_absolute_diff = sum_absolute_diff.item()
    if sum_absolute_r <= eps:
        rae = float('nan')
    else:
        rae = sum_absolute_diff / sum_absolute_r

    err = predict - test
    me = torch.mean(err).item()
    mae = torch.mean(torch.abs(err)).item()
    valid_mask = torch.abs(test) > eps
    if torch.any(valid_mask):
        mpe = torch.mean((err[valid_mask] / test[valid_mask]) * 100.0).item()
    else:
        mpe = float('nan')

    title = title.replace('/', '_')

    save_dir = MODEL_BASE_DIR / type
    save_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = save_dir / f"{title}_{type}.txt"
    
    with open(file_path, "w") as f:
        f.write('rse:' + str(rrse) + '\n')
        f.write('rae:' + str(rae) + '\n')
        f.write('me:' + str(me) + '\n')
        f.write('mae:' + str(mae) + '\n')
        f.write('mpe:' + str(mpe) + '\n')


def plot_predicted_actual(predicted, actual, title, type, variance, confidence_95):
    months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    M=[]
    # 2011년부터 2026년까지 넉넉하게 라벨 생성
    for year in range (11, 27):   
        for month in months:
            if year==11 and month not in ['Jul','Aug','Sep','Oct','Nov','Dec']:
                continue
            M.append(month+'-'+str(year))   
    M2=[]
    p=[]
    
    # === [수정] 날짜 필터링 로직 강화 ===
    target_year = '25' if type == 'Testing' else '24'
    
    # 1. 해당 연도 라벨을 Jan~Dec 고정 생성
    target_labels = [m for m in M if f'-{target_year}' in m][:12]

    # 2. 그래프는 해당 연도 12개월 기준으로 정렬
    #    - 예측 길이가 12보다 길면 마지막 12개 사용
    #    - 예측 길이가 12보다 짧으면 Jan부터 순서대로 라벨 사용
    if len(predicted) > 12:
        predicted = predicted[-12:]
        actual = actual[-12:]
        confidence_95 = confidence_95[-12:]

    label_len = min(len(predicted), len(target_labels))
    M = target_labels[:label_len]
    if len(predicted) > label_len:
        predicted = predicted[:label_len]
        actual = actual[:label_len]
        confidence_95 = confidence_95[:label_len]

    # X축 틱: 월별 모두 표시 (Jan 시작 보장)
    for index, value in enumerate(M):
        M2.append(value)
        p.append(index+1)

    # === 그래프 그리기 ===
    x = range(1, len(predicted) + 1)
    
    plt.plot(x, actual, 'b-', label='Actual')
    plt.plot(x, predicted, '--', color='purple', label='Predicted')
    if isinstance(confidence_95, torch.Tensor):
        confidence_95 = confidence_95.cpu().numpy()
    plt.fill_between(x, predicted - confidence_95, predicted + confidence_95, alpha=0.5, color='pink', label='95% Confidence')
    plt.legend(loc="best", prop={'size': 11})
    plt.axis('tight')
    plt.grid(True)
    plt.title(title, y=1.03, fontsize=18)
    plt.ylabel("Trend", fontsize=15)
    plt.xlabel("Month", fontsize=15)
    
    # X축 라벨 적용 (Validation: Jan-24~Dec-24, Testing: Jan-25~Dec-25)
    plt.xticks(ticks=p, labels=M2, rotation='vertical', fontsize=13)
    plt.yticks(fontsize=13)
    
    # === 파일 저장 ===
    fig = plt.gcf()
    title = title.replace('/', '_')
    
    save_dir = MODEL_BASE_DIR / type
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(save_dir / f"{title}_{type}.png", bbox_inches="tight")
    plt.show(block=False)
    plt.pause(2)
    plt.close()


def s_mape(yTrue, yPred):
    eps = 1e-12
    mape = 0
    for i in range(len(yTrue)):
        denom = abs(yTrue[i]) + abs(yPred[i])
        mape += abs(yTrue[i] - yPred[i]) / (denom + eps)
    mape /= max(len(yTrue), 1)
    return mape


def evaluate_sliding_window(data, test_window, model, evaluateL2, evaluateL1, n_input, is_plot, split_type='Testing', bias_offset=None, return_arrays=False):
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None
    variance = None
    confidence_95 = None

    # ============================================
    r = 5
    print('testing r=', str(r))
    test_window = test_window.to(data.device)
    
    # 초기 입력 데이터 설정
    x_input = test_window[0:n_input, :].clone()

    # [수정 1] fixed_wm, fixed_ws 변수 및 관련 로직 제거
    # 매 반복문(Sliding Window)마다 통계를 새로 계산해야 함

    for i in range(n_input, test_window.shape[0]):
        X = torch.unsqueeze(x_input, dim=0)
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)  # [1, 1, N, T]
        X = X.to(torch.float)

        # =====================================================
        # [수정 2] Dynamic RevIN: 현재 윈도우(X)의 통계 계산
        # =====================================================
        w_mean = X.mean(dim=-1, keepdim=True)  # [1, 1, N, 1]
        w_std = X.std(dim=-1, keepdim=True)    # [1, 1, N, 1]
        w_std[w_std == 0] = 1 # 0으로 나누기 방지
        
        # 정규화 (Normalization)
        X = (X - w_mean) / w_std
        
        # 나중에 복원을 위해 차원 축소해서 저장
        wm = w_mean[0, 0, :, 0]  # [N]
        ws = w_std[0, 0, :, 0]   # [N]
        # =====================================================

        y_true_full = test_window[i: i + data.out_len, :].clone()
        y_true = y_true_full[:1, :].clone()

        num_runs = 10
        outputs = []

        for _ in range(num_runs):
            with torch.no_grad():
                output = model(X)
                y_pred = output[-1, :, :, -1].clone()
                
                # =====================================================
                # [수정 3] 현재 윈도우의 통계로 복원 (Denormalization)
                # =====================================================
                y_pred = y_pred * ws + wm
                # =====================================================
                
                y_pred = y_pred[:1, :]
            outputs.append(y_pred)

        outputs = torch.stack(outputs)
        y_pred = torch.mean(outputs, dim=0)

        if args.anchor_focus_to_last > 0:
            focus_cols = get_focus_columns(data)
            if focus_cols:
                alpha = min(max(args.anchor_focus_to_last, 0.0), 1.0)
                focus_idx = torch.tensor(focus_cols, device=y_pred.device)
                last_obs = x_input[-1, :].to(y_pred.device)
                y_focus = torch.index_select(y_pred, dim=1, index=focus_idx)
                last_focus = torch.index_select(last_obs, dim=0, index=focus_idx).unsqueeze(0)
                y_focus = (1.0 - alpha) * y_focus + alpha * last_focus
                y_pred[:, focus_idx] = y_focus

        var = torch.var(outputs, dim=0)
        std_dev = torch.std(outputs, dim=0)

        z = 1.96
        confidence = z * std_dev / torch.sqrt(torch.tensor(num_runs))

        # 다음 스텝을 위한 입력 업데이트
        eval_recursive = (args.force_recursive_eval == 1 and split_type in ('Validation', 'Testing'))
        if eval_recursive or args.rollout_mode == 'recursive':
            next_chunk = y_pred
        else:
            # teacher-forced 1-step: 다음 시점 실제값을 사용해 윈도우 갱신
            next_chunk = y_true

        x_input = torch.cat([x_input[1:, :].clone(), next_chunk.clone()], dim=0)

        if predict is None:
            predict = y_pred
            test = y_true
            variance = var
            confidence_95 = confidence
        else:
            predict = torch.cat((predict, y_pred))
            test = torch.cat((test, y_true))
            variance = torch.cat((variance, var))
            confidence_95 = torch.cat((confidence_95, confidence))

    # 데이터 스케일(DataLoader의 scale/shift) 복원
    scale = data.scale.expand(test.size(0), data.m)

    predict = predict * scale
    test = test * scale
    variance *= scale
    confidence_95 *= scale

    if bias_offset is not None:
        b = bias_offset.to(predict.device)
        if b.dim() == 1:
            b = b.unsqueeze(0)
        predict = predict - b

    # --- Metrics 계산: target/all 동시 계산 후 report mode 선택 ---
    selected_cols = get_rse_target_columns(data)
    rrse_target, rae_target, root_sum_squared_target, root_sum_squared_r_target = compute_rrse_rae_subset(predict, test, selected_cols)

    all_cols = list(range(data.m))
    rrse_all, rae_all, root_sum_squared_all, root_sum_squared_r_all = compute_rrse_rae_subset(predict, test, all_cols)

    if args.rse_report_mode == 'all':
        rrse = rrse_all
        rae = rae_all
    else:
        rrse = rrse_target
        rae = rae_target

    print(
        f"rrse(target)={root_sum_squared_target} / {root_sum_squared_r_target} | "
        f"rrse(all)={root_sum_squared_all} / {root_sum_squared_r_all} | "
        f"report_mode={args.rse_report_mode}"
    )

    predict_t = predict.clone()
    test_t = test.clone()

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0) & (sigma_p != 0)
    if index.sum() > 0:
        with np.errstate(divide='ignore', invalid='ignore'):
            correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
        correlation = correlation[index]
        correlation = correlation[np.isfinite(correlation)]
        correlation = correlation.mean() if correlation.size > 0 else 0.0
    else:
        correlation = 0.0

    smape = 0
    for z in range(Ytest.shape[1]):
        smape += s_mape(Ytest[:, z], predict[:, z])
    smape /= Ytest.shape[1]

    focus_rrse = compute_focus_rrse(predict, Ytest, data)

    # --- Plotting (기존 코드 유지) ---
    counter = 0
    if is_plot:
        skipped_nodes = []
        focus_nodes = set(get_focus_nodes())
        for v in range(data.m):
            col = v
            raw_name = data.col[col]

            if args.plot_focus_only == 1 and raw_name not in focus_nodes:
                continue

            # Near-constant series는 RSE 분모가 0에 가까워 왜곡되므로 리포트에서 제외
            if np.std(Ytest[:, col]) < 1e-10:
                skipped_nodes.append(raw_name)
                continue

            node_name = raw_name.replace('-ALL', '').replace('Mentions-', 'Mentions of ').replace(' ALL', '').replace('Solution_', '').replace('_Mentions', '')
            node_name = consistent_name(node_name)
            
            save_metrics_1d(torch.from_numpy(predict[:, col]), torch.from_numpy(Ytest[:, col]), node_name, split_type)
            plot_predicted_actual(predict[:, col], Ytest[:, col], node_name, split_type, variance[:, col], confidence_95[:, col])
            counter += 1
        if skipped_nodes:
            print(f"[{split_type}] skipped near-constant nodes for per-node metrics: {skipped_nodes}")

    if return_arrays:
        return rrse, rae, correlation, smape, focus_rrse, predict_t.detach().cpu(), test_t.detach().cpu()
    return rrse, rae, correlation, smape, focus_rrse


def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size, is_plot):
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None
    variance = None
    confidence_95 = None
    sum_squared_diff = 0
    sum_absolute_diff = 0
    
    r = 5
    print('validation r=', str(r))

    for X, Y in data.get_batches(X, Y, batch_size, False):
        X_raw = X.clone()
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)  # [B, 1, N, T]

        # ===== RevIN: Per-Window Normalization =====
        w_mean = X.mean(dim=-1, keepdim=True)  # [B, 1, N, 1]
        w_std = X.std(dim=-1, keepdim=True)    # [B, 1, N, 1]
        w_std[w_std == 0] = 1
        X = (X - w_mean) / w_std

        wm = w_mean[:, 0, :, 0]  # [B, N]
        ws = w_std[:, 0, :, 0]   # [B, N]
        # ============================================

        num_runs = 10
        outputs = []

        with torch.no_grad():
            for _ in range(num_runs):
                output = model(X)
                output = output.squeeze(-1)  # [B, T, N, 1] → [B, T, N]
                if len(output.shape) == 2:
                    output = output.unsqueeze(dim=1)  # [B, N] → [B, 1, N]
                elif len(output.shape) == 1:
                    output = output.unsqueeze(dim=0).unsqueeze(dim=0)  # [N] → [1, 1, N]
                outputs.append(output)

        outputs = torch.stack(outputs)
        mean = torch.mean(outputs, dim=0)
        var = torch.var(outputs, dim=0)
        std_dev = torch.std(outputs, dim=0)

        z = 1.96
        confidence = z * std_dev / torch.sqrt(torch.tensor(num_runs))

        output = mean

        # ===== RevIN Denormalize (back to z-score space) =====
        output = output * ws.unsqueeze(1) + wm.unsqueeze(1)
        var = var * ws.unsqueeze(1)
        confidence = confidence * ws.unsqueeze(1)
        # ======================================================

        # =============================================
        # [중요] Global z-score Denormalize
        # =============================================
        scale = data.scale.expand(Y.size(0), Y.size(1), data.m)

        output = output * scale
        Y = Y * scale
        var *= scale
        confidence *= scale

        if args.anchor_focus_to_last > 0:
            focus_cols = get_focus_columns(data)
            if focus_cols:
                alpha = min(max(args.anchor_focus_to_last, 0.0), 1.0)
                focus_idx = torch.tensor(focus_cols, device=output.device)
                last_obs = X_raw[:, -1, :] * data.scale.expand(X_raw.size(0), data.m)
                y_focus = torch.index_select(output, dim=2, index=focus_idx)
                last_focus = torch.index_select(last_obs, dim=1, index=focus_idx).unsqueeze(1)
                y_focus = (1.0 - alpha) * y_focus + alpha * last_focus
                output[:, :, focus_idx] = y_focus

        if predict is None:
            predict = output
            test = Y
            variance = var
            confidence_95 = confidence
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))
            variance = torch.cat((variance, var))
            confidence_95 = torch.cat((confidence_95, confidence))

        if args.debug_eval == 1:
            print('EVALUATE RESULTS:')
            scale = data.scale.expand(Y.size(0), Y.size(1), data.m)
            y_pred_o = output
            y_true_o = Y
            for z in range(Y.shape[1]):
                print(y_pred_o[0, z, r], y_true_o[0, z, r])
        
        total_loss += evaluateL2(output, Y).item()
        total_loss_l1 += evaluateL1(output, Y).item()
        n_samples += (output.size(0) * output.size(1) * data.m)

        sum_squared_diff += torch.sum(torch.pow(Y - output, 2))
        sum_absolute_diff += torch.sum(torch.abs(Y - output))

    rse = math.sqrt(total_loss / n_samples) / data.rse 
    rae = (total_loss_l1 / n_samples) / data.rae 

    root_sum_squared = math.sqrt(sum_squared_diff)
    test_s = test
    mean_all = torch.mean(test_s, dim=(0, 1))
    diff_r = test_s - mean_all.expand(test_s.size(0), test_s.size(1), data.m)
    sum_squared_r = torch.sum(torch.pow(diff_r, 2))
    root_sum_squared_r = math.sqrt(sum_squared_r)

    eps = 1e-12
    if root_sum_squared_r <= eps:
        rrse = 0.0 if root_sum_squared <= eps else 1e6
    else:
        rrse = root_sum_squared / root_sum_squared_r

    sum_absolute_r = torch.sum(torch.abs(diff_r)).item()
    sum_absolute_diff = sum_absolute_diff.item()
    if sum_absolute_r <= eps:
        rae = 0.0 if sum_absolute_diff <= eps else 1e6
    else:
        rae = sum_absolute_diff / sum_absolute_r

    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0) & (sigma_p != 0)
    if index.sum() > 0:
        with np.errstate(divide='ignore', invalid='ignore'):
            correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
        correlation = correlation[index]
        correlation = correlation[np.isfinite(correlation)]
        correlation = correlation.mean() if correlation.size > 0 else 0.0
    else:
        correlation = 0.0

    smape = 0
    for x in range(Ytest.shape[0]):
        for z in range(Ytest.shape[2]):
            smape += s_mape(Ytest[x, :, z], predict[x, :, z])
    smape /= Ytest.shape[0] * Ytest.shape[2]

    # ===== jp_fx 개별 RSE 계산 (Best 모델 선택 기준용) =====
    jp_fx_rse = None
    for v in range(data.m):
        raw_name = data.col[v]
        if raw_name == 'jp_fx':
            # jp_fx 노드의 RSE 계산
            jp_pred = predict[:, 0, v]
            jp_true = Ytest[:, 0, v]
            jp_diff_sq = np.sum((jp_pred - jp_true) ** 2)
            jp_mean = np.mean(jp_true)
            jp_diff_from_mean_sq = np.sum((jp_true - jp_mean) ** 2)
            if jp_diff_from_mean_sq > 0:
                jp_fx_rse = np.sqrt(jp_diff_sq / jp_diff_from_mean_sq)
            else:
                jp_fx_rse = 0.0
            break
    if jp_fx_rse is None:
        jp_fx_rse = rrse  # jp_fx를 찾지 못한 경우 전체 RSE 사용
    # ========================================================

    counter = 0
    if is_plot:
        for v in range(data.m):
            col = v
            raw_name = data.col[col]

            node_name = raw_name.replace('-ALL', '').replace('Mentions-', 'Mentions of ').replace(' ALL', '').replace('Solution_', '').replace('_Mentions', '')
            node_name = consistent_name(node_name)
            save_metrics_1d(torch.from_numpy(predict[:, 0, col]), torch.from_numpy(Ytest[:, 0, col]), node_name, 'Validation')
            plot_predicted_actual(predict[:, 0, col], Ytest[:, 0, col], node_name, 'Validation', variance[:, 0, col], confidence_95[:, 0, col])
            counter += 1
    return rrse, rae, correlation, smape, jp_fx_rse


def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    iter = 0

    # ===== Target-Weighted Loss (optional) =====
    target_weight = torch.ones(data.m, device=device)
    if args.focus_targets == 1:
        focus_cols = get_focus_columns(data)
        for col in focus_cols:
            target_weight[col] = args.focus_target_gain
        print(f"[Target-Weighted Loss] weights applied: "
              f"{ {data.col[i]: target_weight[i].item() for i in range(data.m) if target_weight[i] > 1} }")
        if args.focus_only_loss == 1 and focus_cols:
            print(f"[Target-Only Loss] enabled for columns: {[data.col[i] for i in focus_cols]}")
    else:
        print("[Target-Weighted Loss] disabled (uniform weight for overall RSE optimization)")
    if args.bias_penalty > 0:
        if args.bias_penalty_scope == 'focus':
            scope_cols = get_focus_columns(data)
            scope_names = [data.col[i] for i in scope_cols] if scope_cols else []
            print(f"[Bias Penalty] enabled lambda={args.bias_penalty} scope=focus cols={scope_names}")
        else:
            print(f"[Bias Penalty] enabled lambda={args.bias_penalty} scope=all")
    else:
        print("[Bias Penalty] disabled")
    if args.trend_penalty > 0:
        if args.trend_penalty_scope == 'focus':
            scope_cols = get_focus_columns(data)
            scope_names = [data.col[i] for i in scope_cols] if scope_cols else []
            print(f"[Trend Penalty] enabled lambda={args.trend_penalty} scope=focus cols={scope_names}")
        else:
            print(f"[Trend Penalty] enabled lambda={args.trend_penalty} scope=all")
    else:
        print("[Trend Penalty] disabled")
    # ==================================

    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()
        X = torch.unsqueeze(X, dim=1)
        X = X.transpose(2, 3)  # [B, 1, N, T]

        # ===== RevIN: Per-Window Normalization =====
        w_mean = X.mean(dim=-1, keepdim=True)  # [B, 1, N, 1]
        w_std = X.std(dim=-1, keepdim=True)    # [B, 1, N, 1]
        w_std[w_std == 0] = 1
        X = (X - w_mean) / w_std

        # Normalize target Y using same window stats
        wm = w_mean[:, 0, :, 0]  # [B, N]
        ws = w_std[:, 0, :, 0]   # [B, N]
        Y = (Y - wm.unsqueeze(1)) / ws.unsqueeze(1)  # [B, T_out, N]
        # ============================================

        if iter % args.step_size == 0:
            perm = np.random.permutation(range(args.num_nodes))
        num_sub = int(args.num_nodes / args.num_split)

        for j in range(args.num_split):
            if j != args.num_split - 1:
                id = perm[j * num_sub:(j + 1) * num_sub]
            else:
                id = perm[j * num_sub:]

            id = torch.tensor(id).to(device)
            tx = X[:, :, :, :]
            ty = Y[:, :, :]

            if iter > 0 and args.ss_prob > 0 and random.random() < args.ss_prob:
                with torch.no_grad():
                    prev_output = model(tx)
                    prev_pred = torch.squeeze(prev_output, 3)  # [B, T_out, N]
                    if prev_pred.dim() == 2:
                        prev_pred = prev_pred.unsqueeze(1)
                    tail_steps = min(tx.size(3), prev_pred.size(1))
                    pred_tail = prev_pred[:, -tail_steps:, :].transpose(1, 2).unsqueeze(1)  # [B,1,N,tail]
                    tx = tx.clone()
                    tx[:, :, :, -tail_steps:] = pred_tail
            
            output = model(tx)
            output = torch.squeeze(output, 3)

            # ===== Target-Weighted Loss =====
            use_l1 = (args.loss_mode == 'l1')
            diff = torch.abs(output - ty) if use_l1 else (output - ty) ** 2
            w = target_weight.unsqueeze(0).unsqueeze(0)  # [1, 1, N]
            if args.focus_targets == 1 and args.focus_only_loss == 1:
                focus_cols = get_focus_columns(data)
                if focus_cols:
                    focus_idx = torch.tensor(focus_cols, device=device)
                    diff = torch.index_select(diff, dim=2, index=focus_idx)
                    w_focus = torch.index_select(w, dim=2, index=focus_idx)
                    loss = (diff * w_focus).mean() / torch.clamp(w_focus.mean(), min=1e-12)
                else:
                    loss = (diff * w).mean() / torch.clamp(w.mean(), min=1e-12)
            else:
                loss = (diff * w).mean() / torch.clamp(w.mean(), min=1e-12)

            if args.bias_penalty > 0:
                err_signed = output - ty
                if args.bias_penalty_scope == 'focus':
                    focus_cols = get_focus_columns(data)
                    if focus_cols:
                        focus_idx = torch.tensor(focus_cols, device=device)
                        err_signed = torch.index_select(err_signed, dim=2, index=focus_idx)
                bias_term = torch.mean(torch.mean(err_signed, dim=(0, 1)) ** 2)
                loss = loss + args.bias_penalty * bias_term

            if args.trend_penalty > 0:
                last_obs = tx[:, 0, :, -1].unsqueeze(1)
                pred_delta = output - last_obs
                true_delta = ty - last_obs
                trend_err = pred_delta - true_delta
                if args.trend_penalty_scope == 'focus':
                    focus_cols = get_focus_columns(data)
                    if focus_cols:
                        focus_idx = torch.tensor(focus_cols, device=device)
                        trend_err = torch.index_select(trend_err, dim=2, index=focus_idx)
                trend_term = torch.mean(trend_err ** 2)
                loss = loss + args.trend_penalty * trend_term
            # ==================================
            loss.backward()
            total_loss += loss.item()
            n_samples += (output.size(0) * output.size(1) * data.m)
            
            grad_norm = optim.step()

        if iter % 1 == 0:
            print('iter:{:3d} | loss: {:.3f}'.format(iter, loss.item() / (output.size(0) * output.size(1) * data.m)))
        iter += 1
    return total_loss / n_samples


DEFAULT_DATA_PATH = BMTGNN_DIR / 'data' / 'sm_data.csv'
DEFAULT_MODEL_SAVE = MODEL_BASE_DIR / 'model.pt'

parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, default=str(DEFAULT_DATA_PATH), help='location of the data file')
parser.add_argument('--log_interval', type=int, default=2000, metavar='N', help='report interval')
parser.add_argument('--save', type=str, default=str(DEFAULT_MODEL_SAVE), help='path to save the final model')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--loss_mode', type=str, default='l1', choices=['l1', 'mse'], help='training loss type; mse generally aligns better with RSE minimization')
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--device', type=str, default='cuda:1', help='')
parser.add_argument('--gcn_true', type=bool, default=False, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=bool, default=False, help='whether to construct adaptive adjacency matrix')
parser.add_argument('--use_graph', type=int, default=0, choices=[0, 1], help='1: use graph modules, 0: disable graph modules')
parser.add_argument('--gcn_depth', type=int, default=1, help='graph convolution depth')
parser.add_argument('--num_nodes', type=int, default=142, help='number of nodes/variables')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('--subgraph_size', type=int, default=40, help='k')
parser.add_argument('--node_dim', type=int, default=30, help='dim of nodes')
parser.add_argument('--dilation_exponential', type=int, default=2, help='dilation exponential')
parser.add_argument('--conv_channels', type=int, default=16, help='convolution channels')
parser.add_argument('--residual_channels', type=int, default=128, help='residual channels')
parser.add_argument('--skip_channels', type=int, default=256, help='skip channels')
parser.add_argument('--end_channels', type=int, default=1024, help='end channels')
parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')
parser.add_argument('--seq_in_len', type=int, default=24, help='input sequence length')
parser.add_argument('--seq_out_len', type=int, default=1, help='output sequence length (multi-step horizon)')
parser.add_argument('--horizon', type=int, default=1, help='forecast start offset (1=predict immediately after input)')
parser.add_argument('--layers', type=int, default=2, help='number of layers')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay rate')
parser.add_argument('--clip', type=int, default=10, help='clip')
parser.add_argument('--propalpha', type=float, default=0.6, help='prop alpha')
parser.add_argument('--tanhalpha', type=float, default=0.1, help='tanh alpha')
parser.add_argument('--epochs', type=int, default=320, help='')
parser.add_argument('--early_stop_patience', type=int, default=0, help='stop if no best-objective improvement for N epochs (0 disables)')
parser.add_argument('--early_stop_min_epochs', type=int, default=0, help='minimum epochs before early stopping is allowed')
parser.add_argument('--num_split', type=int, default=1, help='number of splits for graphs')
parser.add_argument('--step_size', type=int, default=100, help='step_size')
parser.add_argument('--ss_prob', type=float, default=0.2, help='scheduled sampling probability')
parser.add_argument('--train_ratio', type=float, default=0.8666666667, help='train split ratio')
parser.add_argument('--valid_ratio', type=float, default=0.0666666667, help='validation split ratio')
parser.add_argument('--fixed_eval_periods', type=int, default=1, choices=[0, 1], help='1 to force Validation/Test split by calendar years (valid_year/test_year)')
parser.add_argument('--valid_year', type=int, default=2024, help='validation target year when fixed_eval_periods=1')
parser.add_argument('--test_year', type=int, default=2025, help='testing target year when fixed_eval_periods=1')
parser.add_argument('--focus_targets', type=int, default=0, help='1 to upweight us/kr/jp target nodes')
parser.add_argument('--focus_nodes', type=str, default='us_Trade Weighted Dollar Index,jp_fx,kr_fx', help='priority nodes (comma separated)')
parser.add_argument('--focus_weight', type=float, default=0.7, help='priority weight for focus-node RRSE in model selection (0~1)')
parser.add_argument('--focus_target_gain', type=float, default=12.0, help='loss weight applied to focus target columns when focus_targets=1')
parser.add_argument('--focus_only_loss', type=int, default=0, choices=[0, 1], help='1 to optimize loss only on focus/rse target columns')
parser.add_argument('--anchor_focus_to_last', type=float, default=0.0, help='0~1 level anchoring strength for focus columns during evaluation/forecast')
parser.add_argument('--rse_targets', type=str, default='Us_Trade Weighted Dollar Index_Testing.txt,Jp_fx_Testing.txt,Kr_fx_Testing.txt', help='comma-separated target series/file names used for terminal RSE/RAE aggregation')
parser.add_argument('--rse_report_mode', type=str, default='targets', choices=['targets', 'all'], help='which RSE/RAE to report in terminal and final summary')
parser.add_argument('--debias_mode', type=str, default='none', choices=['none', 'val_mean_error'], help='bias correction mode for final evaluation/reporting')
parser.add_argument('--debias_apply_to', type=str, default='focus', choices=['focus', 'all'], help='apply debias offset to focus targets only or all series')
parser.add_argument('--bias_penalty', type=float, default=0.0, help='lambda for training-time signed-bias penalty (0 disables)')
parser.add_argument('--bias_penalty_scope', type=str, default='focus', choices=['focus', 'all'], help='scope for training-time bias penalty')
parser.add_argument('--trend_penalty', type=float, default=0.0, help='lambda for training-time delta/trend penalty (0 disables)')
parser.add_argument('--trend_penalty_scope', type=str, default='focus', choices=['focus', 'all'], help='scope for trend penalty')
parser.add_argument('--plot_focus_only', type=int, default=0, help='1 to plot/save only focus nodes')
parser.add_argument('--debug_eval', type=int, default=0, help='1 to print per-step eval tensors')
parser.add_argument('--rollout_mode', type=str, default='teacher_forced', choices=['teacher_forced', 'recursive'], help='test rollout mode')
parser.add_argument('--force_recursive_eval', type=int, default=1, choices=[0, 1], help='1 to always use recursive rollout for Validation/Testing plots and metrics')
parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--plot', type=int, default=1, help='1 to save plots, 0 to skip plotting')
parser.add_argument('--clean_cache', type=int, default=0, choices=[0, 1], help='1 to delete cached *.pt in data dir before training')
parser.add_argument('--autotune_mode', type=int, default=0, choices=[0, 1], help='1 to optimize for repeated auto-tuning runs')
parser.add_argument('--apply_best_tuning', type=int, default=0, choices=[0, 1], help='1 to override args with best tuning run values')
parser.add_argument('--eval_best_tuning', type=int, default=0, choices=[0, 1], help='1 to skip training and evaluate best tuned checkpoint with plotting')
parser.add_argument('--target_profile', type=str, default='triple_050', choices=['none', 'triple_050', 'run001_us'], help='preset for target-focused optimization setup')


args = parser.parse_args()
args.best_tuning_checkpoint = ''
if args.autotune_mode == 1:
    args.plot = 0
    args.clean_cache = 0
    args.use_graph = 0

# If requested, load best tuning run from tuning_runs and override matching args
if args.apply_best_tuning == 1:
    try:
        import json
        from pathlib import Path
        TR = Path(__file__).resolve().parent / 'tuning_runs'
        # find latest tuning run folder by name (timestamped folders)
        runs = sorted([p for p in TR.iterdir() if p.is_dir()])
        if runs:
            latest = runs[-1]
            results_file = latest / 'results.json'
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                # pick the run with smallest best_test_rse when available
                best = None
                for r in results:
                    if 'best_test_rse' in r and r.get('best_test_rse') is not None:
                        if best is None or r.get('best_test_rse') < best.get('best_test_rse'):
                            best = r
                if best is None and results:
                    best = results[0]

                if best is not None:
                    run_id = best.get('run_id')
                    log_path = latest / f"run_{int(run_id):03d}.log"
                    ckpt_path = latest / 'checkpoints' / f"model_{int(run_id):03d}.pt"
                    if log_path.exists():
                        # extract Namespace(...) line from log and convert to dict
                        import re
                        ns_line = None
                        with open(log_path, 'r') as lf:
                            for line in lf:
                                if line.strip().startswith('Namespace('):
                                    ns_line = line.strip()
                                    break
                        if ns_line:
                            m = re.search(r"Namespace\((.*)\)", ns_line)
                            if m:
                                inner = m.group(1)
                                try:
                                    parsed = eval('dict(' + inner + ')')
                                except Exception:
                                    parsed = None

                                if isinstance(parsed, dict):
                                    # Only override args that the parser defines.
                                    # Do not overwrite user-run controls like epochs/plot/autotune_mode/clean_cache
                                    skip_keys = {'epochs', 'plot', 'autotune_mode', 'clean_cache'}
                                    for k, v in parsed.items():
                                        if k in skip_keys:
                                            continue
                                        if hasattr(args, k):
                                            try:
                                                setattr(args, k, v)
                                            except Exception:
                                                pass
                                    # ensure autotune mode off when applying tuned HPs
                                    args.autotune_mode = 0
                                    if ckpt_path.exists():
                                        args.best_tuning_checkpoint = str(ckpt_path)
                                        args.save = str(ckpt_path)
                                    print(f"[apply_best_tuning] Applied params from {log_path} (preserved epochs/plot)")
                                else:
                                    print(f"[apply_best_tuning] failed to parse Namespace in {log_path}")
                        else:
                            print(f"[apply_best_tuning] Namespace line not found in {log_path}")
                    else:
                        print(f"[apply_best_tuning] log file not found: {log_path}")
            else:
                print(f"[apply_best_tuning] results.json not found in {latest}")
        else:
            print("[apply_best_tuning] no tuning_runs directories found")
    except Exception as e:
        print(f"[apply_best_tuning] error while loading tuning run: {e}")

# Optional target optimization profile
if args.target_profile == 'triple_050':
    args.focus_targets = 1
    args.focus_nodes = 'us_Trade Weighted Dollar Index,jp_fx,kr_fx'
    args.rse_targets = 'Us_Trade Weighted Dollar Index_Testing.txt,Jp_fx_Testing.txt,Kr_fx_Testing.txt'
    args.rse_report_mode = 'targets'
    args.loss_mode = 'mse'
    args.lr = 0.0003
    args.dropout = 0.1
    args.seq_in_len = 36
    args.ss_prob = 0.2
    args.seed = 123
    args.focus_weight = 1.0
    args.focus_target_gain = 40.0
    args.focus_only_loss = 1
    args.anchor_focus_to_last = 0.75
    args.rollout_mode = 'recursive'
    args.debias_mode = 'none'
    args.debias_apply_to = 'focus'
    args.bias_penalty = 0.0
    args.bias_penalty_scope = 'focus'
    args.use_graph = 0
    print('[target_profile] applied: triple_050')

if args.target_profile == 'run001_us':
    args.loss_mode = 'mse'
    args.use_graph = 0
    args.lr = 0.0002
    args.dropout = 0.1
    args.layers = 2
    args.seq_in_len = 24
    args.seq_out_len = 1
    args.ss_prob = 0.2
    args.epochs = 120
    args.seed = 2026
    args.focus_targets = 1
    args.focus_nodes = 'us_Trade Weighted Dollar Index'
    args.focus_weight = 1.0
    args.focus_target_gain = 40.0
    args.focus_only_loss = 1
    args.anchor_focus_to_last = 0.15
    args.rse_targets = 'Us_Trade Weighted Dollar Index_Testing.txt'
    args.rse_report_mode = 'targets'
    args.debias_mode = 'none'
    args.debias_apply_to = 'focus'
    args.rollout_mode = 'recursive'
    args.trend_penalty = 0.2
    args.trend_penalty_scope = 'focus'
    print('[target_profile] applied: run001_us')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available, using CPU")
    torch.set_num_threads(3)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


fixed_seed = args.seed


def main(experiment):
    set_random_seed(fixed_seed)

    # ===== Fixed HP (0209 최적 결과 기반) - no random search =====
    best_val = 10000000
    best_rse = 10000000
    best_rae = 10000000
    best_corr = -10000000
    best_smape = 10000000
    best_objective = 10000000
    best_test_rse = 10000000
    best_test_corr = -10000000

    best_hp = []

    for q in range(1):
        # 0209 최적 HP 사용 (고정)
        gcn_depth = args.gcn_depth
        lr = args.lr
        conv = args.conv_channels
        res = args.residual_channels
        skip = args.skip_channels
        end = args.end_channels
        layer = args.layers
        k = args.subgraph_size
        dropout = args.dropout
        dilation_ex = args.dilation_exponential
        node_dim = args.node_dim
        prop_alpha = args.propalpha
        tanh_alpha = args.tanhalpha

        # ============================================================
        # Cache Cleaning (optional)
        # ============================================================
        if args.clean_cache == 1:
            data_dir = os.path.dirname(args.data)
            pt_files = glob.glob(os.path.join(data_dir, "*.pt"))
            print(f"!!! Cleaning Cache Files in {data_dir} !!!")
            for file_path in pt_files:
                if "model" not in file_path:
                    try:
                        os.remove(file_path)
                        print(f"Deleted: {file_path}")
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")
            print("!!! Cache Clean Complete !!!")
        # ============================================================

        if args.train_ratio <= 0 or args.valid_ratio <= 0 or (args.train_ratio + args.valid_ratio) >= 1:
            raise ValueError("train_ratio + valid_ratio must be < 1 and both must be > 0")

        Data = DataLoaderS(
            args.data,
            args.train_ratio,
            args.valid_ratio,
            device,
            args.horizon,
            args.seq_in_len,
            args.normalize,
            args.seq_out_len,
            fixed_eval_periods=args.fixed_eval_periods,
            valid_year=args.valid_year,
            test_year=args.test_year,
        )

        print('train X:', Data.train[0].shape)
        print('train Y:', Data.train[1].shape)
        print('valid X:', Data.valid[0].shape)
        print('valid Y:', Data.valid[1].shape)
        print('test X:', Data.test[0].shape)
        print('test Y:', Data.test[1].shape)
        print('test window:', Data.test_window.shape)

        print('length of training set=', Data.train[0].shape[0])
        print('length of validation set=', Data.valid[0].shape[0])
        print('length of testing set=', Data.test[0].shape[0])
        print('valid=', int((args.train_ratio + args.valid_ratio) * Data.n))

        if len(Data.train[0].shape) == 4: 
             args.num_nodes = Data.train[0].shape[2]
        elif len(Data.train[0].shape) == 3: 
             args.num_nodes = Data.train[0].shape[2]

        print(f"Auto-detected num_nodes: {args.num_nodes}")

        use_graph = bool(args.use_graph)
        model = gtnet(use_graph, use_graph, gcn_depth, args.num_nodes,
                      device, Data.adj, dropout=dropout, subgraph_size=k,
                      node_dim=node_dim, dilation_exponential=dilation_ex,
                      conv_channels=conv, residual_channels=res,
                      skip_channels=skip, end_channels=end,
                      seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                      layers=layer, propalpha=prop_alpha, tanhalpha=tanh_alpha, layer_norm_affline=False)
        model = model.to(device)

        print(args)
        print('The recpetive field size is', model.receptive_field)
        nParams = sum([p.nelement() for p in model.parameters()])
        print('Number of model parameters is', nParams, flush=True)

        if args.loss_mode == 'l1':
            criterion = nn.L1Loss(reduction='sum').to(device)
        else:
            criterion = nn.MSELoss(reduction='sum').to(device)
        evaluateL2 = nn.MSELoss(reduction='sum').to(device)
        evaluateL1 = nn.L1Loss(reduction='sum').to(device)

        optim = Optim(
            model.parameters(), args.optim, lr, args.clip, lr_decay=args.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim.optimizer, mode='min', factor=0.5, patience=15
        )

        es_counter = 0
        try:
            if args.eval_best_tuning == 1:
                ckpt_to_load = args.best_tuning_checkpoint if args.best_tuning_checkpoint else args.save
                if ckpt_to_load and os.path.exists(ckpt_to_load):
                    with open(ckpt_to_load, 'rb') as f:
                        model = torch.load(f, weights_only=False)
                    model = model.to(device)
                    print(f"[eval_best_tuning] loaded checkpoint: {ckpt_to_load}")
                else:
                    raise FileNotFoundError(f"[eval_best_tuning] checkpoint not found: {ckpt_to_load}")
            else:
                print('begin training')
            for epoch in range(1, args.epochs + 1):
                if args.eval_best_tuning == 1:
                    break
                print('Experiment:', (experiment + 1))
                print('Iter:', q)
                print('epoch:', epoch)
                print('hp=', [gcn_depth, lr, conv, res, skip, end, k, dropout, dilation_ex, node_dim, prop_alpha, tanh_alpha, layer, epoch])
                print('best sum=', best_val)
                print('best rrse=', best_rse)
                print('best rrae=', best_rae)
                print('best corr=', best_corr)
                print('best smape=', best_smape)
                print('best objective=', best_objective)
                print('best hps=', best_hp)
                print('best test rse=', best_test_rse)
                print('best test corr=', best_test_corr)

                es_counter += 1

                epoch_start_time = time.time()
                train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)
                val_loss, val_rae, val_corr, val_smape, val_focus_rrse = evaluate_sliding_window(
                    Data, Data.valid_window, model, evaluateL2, evaluateL1, args.seq_in_len, False, 'Validation'
                )
                if val_focus_rrse is None:
                    val_focus_rrse = val_loss
                focus_weight = min(max(args.focus_weight, 0.0), 1.0)
                objective_score = (1.0 - focus_weight) * val_loss + focus_weight * val_focus_rrse
                print(
                    '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f} | valid smape  {:5.4f} | focus_rrse {:5.4f} | obj {:5.4f}'.format(
                        epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr, val_smape, val_focus_rrse, objective_score), flush=True)
                
                scheduler.step(objective_score)

                # ===== Best 모델 선택 기준: 전체 RSE + 우선노드 RSE 가중 합 =====
                safe_corr = val_corr if not math.isnan(val_corr) else 0.0
                sum_loss = val_loss + val_rae - safe_corr

                if objective_score < best_objective:
                    save_path = Path(args.save)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(save_path, 'wb') as f:
                        torch.save(model, f)
                    best_val = sum_loss
                    best_rse = val_loss
                    best_rae = val_rae
                    best_corr = val_corr
                    best_smape = val_smape
                    best_objective = objective_score

                    best_hp = [gcn_depth, lr, conv, res, skip, end, k, dropout, dilation_ex, node_dim, prop_alpha, tanh_alpha, layer, epoch]

                    es_counter = 0

                    test_acc, test_rae, test_corr, test_smape, test_focus_rrse = evaluate_sliding_window(
                        Data, Data.test_window, model, evaluateL2, evaluateL1, args.seq_in_len, False, 'Testing'
                    )
                    print('********************************************************************************************************')
                    print("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}| test smape {:5.4f} | test focus_rrse {:5.4f}".format(
                        test_acc, test_rae, test_corr, test_smape, test_focus_rrse if test_focus_rrse is not None else test_acc), flush=True)
                    print('********************************************************************************************************')
                    best_test_rse = test_acc
                    best_test_corr = test_corr

                if args.early_stop_patience > 0 and epoch >= args.early_stop_min_epochs and es_counter >= args.early_stop_patience:
                    print(f"[early_stop] no improvement for {es_counter} epochs (patience={args.early_stop_patience}).")
                    break

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    print('best val loss=', best_val)
    print('best hps=', best_hp)
    
    hp_save_path = MODEL_BASE_DIR / 'hp.txt'
    hp_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hp_save_path, "w") as f:
        f.write(str(best_hp))

    if os.path.exists(args.save):
        with open(args.save, 'rb') as f:
            model = torch.load(f, weights_only=False)
    else:
        print(f"Warning: checkpoint not found at {args.save}. Using current in-memory model.")
    # 로드한 모델도 학습 시와 같은 device로 이동
    model = model.to(device)

    if args.plot == 1:
        clear_split_outputs('Validation')
        clear_split_outputs('Testing')

    bias_offset = None
    if args.debias_mode == 'val_mean_error':
        _, _, _, _, _, v_pred_t, v_true_t = evaluate_sliding_window(
            Data, Data.valid_window, model, evaluateL2, evaluateL1, args.seq_in_len, False, 'Validation', return_arrays=True
        )
        bias_offset = estimate_bias_offset_from_arrays(Data, v_pred_t, v_true_t)
        focus_cols = get_focus_columns(Data)
        if focus_cols:
            summary = {Data.col[c]: float(bias_offset[c].item()) for c in focus_cols}
            print(f"[debias] mode={args.debias_mode} apply_to={args.debias_apply_to} offset={summary}")

    vtest_acc, vtest_rae, vtest_corr, vtest_smape, _ = evaluate_sliding_window(
        Data, Data.valid_window, model, evaluateL2, evaluateL1, args.seq_in_len, args.plot == 1, 'Validation', bias_offset=bias_offset
    )

    test_acc, test_rae, test_corr, test_smape, _ = evaluate_sliding_window(
        Data, Data.test_window, model, evaluateL2, evaluateL1, args.seq_in_len, args.plot == 1, 'Testing', bias_offset=bias_offset
    )
    print('********************************************************************************************************')
    print("final test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f} | test smape {:5.4f}".format(test_acc, test_rae, test_corr, test_smape))
    print('********************************************************************************************************')
    return vtest_acc, vtest_rae, vtest_corr, vtest_smape, test_acc, test_rae, test_corr, test_smape


if __name__ == "__main__":
    vacc = []
    vrae = []
    vcorr = []
    vsmape = []
    acc = []
    rae = []
    corr = []
    smape = []
    for i in range(1):
        val_acc, val_rae, val_corr, val_smape, test_acc, test_rae, test_corr, test_smape = main(i)
        vacc.append(val_acc)
        vrae.append(val_rae)
        vcorr.append(val_corr)
        vsmape.append(val_smape)
        acc.append(test_acc)
        rae.append(test_rae)
        corr.append(test_corr)
        smape.append(test_smape)
    print('\n\n')
    print('1 run average')
    print('\n\n')
    print("valid\trse\trae")
    print("mean\t{:5.4f}\t{:5.4f}".format(np.mean(vacc), np.mean(vrae)))
    print("std\t{:5.4f}\t{:5.4f}".format(np.std(vacc), np.std(vrae)))
    print('\n\n')
    print("test\trse\trae")
    print("mean\t{:5.4f}\t{:5.4f}".format(np.mean(acc), np.mean(rae)))
    print("std\t{:5.4f}\t{:5.4f}".format(np.std(acc), np.std(rae)))
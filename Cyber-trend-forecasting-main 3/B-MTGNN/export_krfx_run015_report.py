import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from util import DataLoaderS


def main():
    root = '/Users/samrobert/Documents/GitHub/Exchangeprediction/Cyber-trend-forecasting-main 3'
    bm_dir = os.path.join(root, 'B-MTGNN')
    data_path = os.path.join(bm_dir, 'data', 'sm_data.csv')
    model_path = os.path.join(bm_dir, 'tuning_runs', '20260217_134252', 'checkpoints', 'model_015.pt')
    out_dir = os.path.join(root, 'AXIS', 'model', 'Bayesian', 'Testing')
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cpu')
    model = torch.load(model_path, map_location=device, weights_only=False)
    model = model.to(device)

    seq_in_len = int(getattr(model, 'seq_length', 24) or 24)
    train_ratio = 0.8666666667
    valid_ratio = 0.0666666667
    horizon = 1
    normalize = 2
    out_len = 1

    data_loader = DataLoaderS(data_path, train_ratio, valid_ratio, device, horizon, seq_in_len, normalize, out_len)

    if 'kr_fx' not in data_loader.col:
        raise RuntimeError('kr_fx not found in columns')
    kr_idx = data_loader.col.index('kr_fx')

    num_runs = 10
    anchor_alpha = 0.6
    rollout_mode = 'teacher_forced'

    test_window = data_loader.test_window.to(device)
    x_input = test_window[0:seq_in_len, :].clone()

    pred_list = []
    true_list = []

    for i in range(seq_in_len, test_window.shape[0]):
        x_eval = torch.unsqueeze(x_input, dim=0)
        x_eval = torch.unsqueeze(x_eval, dim=1)
        x_eval = x_eval.transpose(2, 3).to(torch.float)

        w_mean = x_eval.mean(dim=-1, keepdim=True)
        w_std = x_eval.std(dim=-1, keepdim=True)
        w_std[w_std == 0] = 1
        x_eval_norm = (x_eval - w_mean) / w_std

        wm = w_mean[0, 0, :, 0]
        ws = w_std[0, 0, :, 0]

        y_true = test_window[i:i + out_len, :][:1, :].clone()

        outputs = []
        with torch.no_grad():
            model.train()
            for _ in range(num_runs):
                output = model(x_eval_norm)
                y_pred = output[-1, :, :, -1].clone()
                y_pred = y_pred * ws + wm
                y_pred = y_pred[:1, :]
                outputs.append(y_pred)

        outputs = torch.stack(outputs)
        y_pred = torch.mean(outputs, dim=0)

        last_obs = x_input[-1, :]
        y_pred[:, kr_idx] = (1 - anchor_alpha) * y_pred[:, kr_idx] + anchor_alpha * last_obs[kr_idx]

        pred_list.append(y_pred[0, kr_idx].item())
        true_list.append(y_true[0, kr_idx].item())

        next_chunk = y_true if rollout_mode == 'teacher_forced' else y_pred
        x_input = torch.cat([x_input[1:, :].clone(), next_chunk.clone()], dim=0)

    scale_kr = data_loader.scale[kr_idx].item()
    pred_arr = np.array(pred_list) * scale_kr
    true_arr = np.array(true_list) * scale_kr

    n = len(pred_arr)
    dates = pd.date_range(start=pd.Timestamp('2025-01-01'), periods=n, freq='MS')

    err = pred_arr - true_arr
    abs_err = np.abs(err)
    ape = np.where(np.abs(true_arr) > 1e-12, abs_err / np.abs(true_arr) * 100.0, np.nan)

    csv_path = os.path.join(out_dir, 'Kr_fx_Testing_monthly_run015.csv')
    df = pd.DataFrame({
        'month': dates.strftime('%Y-%m'),
        'actual': true_arr,
        'predicted': pred_arr,
        'error': err,
        'abs_error': abs_err,
        'abs_pct_error': ape,
    })
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(dates, true_arr, label='Actual', color='blue', linewidth=2)
    plt.plot(dates, pred_arr, label='Predicted (run_015)', color='purple', linestyle='--', linewidth=2)
    plt.fill_between(dates, np.minimum(true_arr, pred_arr), np.maximum(true_arr, pred_arr), alpha=0.12, color='purple')
    plt.title('Kr_fx Testing (run_015): Actual vs Predicted')
    plt.xlabel('Month')
    plt.ylabel('Kr_fx')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    plot_path = os.path.join(out_dir, 'Kr_fx_Testing_run015_compare.png')
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print('saved_csv:', csv_path)
    print('saved_plot:', plot_path)
    print('mean_abs_error:', float(np.nanmean(abs_err)))
    print('mean_abs_pct_error:', float(np.nanmean(ape)))


if __name__ == '__main__':
    main()

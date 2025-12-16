"""
GNN 모델 학습 스크립트 (MTGNN / gtnet)
- 5개 통화(USD, KRW, JPY, GBP, CNY)를 노드로 하는 멀티변량 시계열 예측
- 입력 텐서: [B, C, N, L]  (C=in_dim, N=num_nodes, L=seq_length)
- 출력 텐서: 보통 [B, H, N] 또는 [B, H, N, 1] (H=out_dim)
- 논문용 기본 원칙: scaler는 train 구간으로만 fit (data leakage 방지)
"""

from pathlib import Path
import sys
import os

# ---------------------------------------------------------------------
# Path bootstrap (Edit_Code를 패키지로 import하기 위한 설정)
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../B-MTGNN
sys.path.insert(0, str(PROJECT_ROOT))              # 핵심: PROJECT_ROOT를 넣어야 Edit_Code.net의 상대 import가 동작

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler as SKStandardScaler

# MTGNN
from Edit_Code.net import gtnet


def build_windows(data_2d: np.ndarray, seq_length: int, horizon: int):
    """
    data_2d: [T, N]
    return:
      X: [S, 1, N, L]
      Y: [S, H, N]
    """
    T, N = data_2d.shape
    S = T - seq_length - horizon + 1
    if S <= 0:
        raise ValueError(
            f"샘플을 만들 수 없습니다. T={T}, seq_length={seq_length}, horizon={horizon} -> S={S}"
        )

    X = np.zeros((S, 1, N, seq_length), dtype=np.float32)
    Y = np.zeros((S, horizon, N), dtype=np.float32)

    for i in range(S):
        x = data_2d[i:i + seq_length, :]              # [L, N]
        y = data_2d[i + seq_length:i + seq_length + horizon, :]  # [H, N]

        # [1, N, L]
        X[i, 0, :, :] = x.T
        # [H, N]
        Y[i, :, :] = y

    return X, Y


def split_by_time(X, Y, train_ratio=0.6, valid_ratio=0.2):
    """
    시간 순서 유지 split (논문용 권장)
    X: [S, 1, N, L], Y: [S, H, N]
    """
    S = X.shape[0]
    train_end = int(S * train_ratio)
    valid_end = train_end + int(S * valid_ratio)

    train_X, train_Y = X[:train_end], Y[:train_end]
    valid_X, valid_Y = X[train_end:valid_end], Y[train_end:valid_end]
    test_X,  test_Y  = X[valid_end:], Y[valid_end:]

    return train_X, train_Y, valid_X, valid_Y, test_X, test_Y


def normalize_train_only(df_fx: pd.DataFrame, train_rows: int):
    """
    df_fx: [T, N]
    train_rows: scaler fit에 사용할 row 수 (time split 기준)
    """
    scaler = SKStandardScaler()

    train_part = df_fx.iloc[:train_rows].values.astype(np.float32)
    scaler.fit(train_part)

    all_scaled = scaler.transform(df_fx.values.astype(np.float32)).astype(np.float32)
    return all_scaled, scaler


def adapt_output_to_y(output: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    output shape를 y shape([B,H,N])에 맞춰 최대한 안전하게 변환
    """
    # 가능한 케이스:
    # 1) [B, H, N]
    # 2) [B, H, N, 1]
    # 3) [B, N, H]
    # 4) [B, N, H, 1]
    if output.ndim == 4 and output.shape[-1] == 1:
        output = output[..., 0]

    if output.ndim == 3:
        if output.shape == y.shape:
            return output
        # [B, N, H] -> [B, H, N]
        if output.shape[0] == y.shape[0] and output.shape[1] == y.shape[2] and output.shape[2] == y.shape[1]:
            return output.permute(0, 2, 1).contiguous()

    raise RuntimeError(f"모델 출력 shape를 y에 맞출 수 없습니다. output={tuple(output.shape)}, y={tuple(y.shape)}")


def train_model(
    data_file: str,
    model_save_path: str,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 5e-4,
    train_ratio: float = 0.6,
    valid_ratio: float = 0.2,
    seq_length: int = 12,   # 월별 데이터면 12(1년) 추천
    horizon: int = 36,       # 36개월 예측
    use_diff_data: bool = True,
    shape_loss_weight: float = 0.2,
    usd_anchor_idx: int = 0,
    predict_delta_output: bool = True,
    delta_scale: float = 0.05,
    clamp_range=None,
):
    print("=" * 60)
    print("🚀 GNN(MTGNN) 모델 학습 시작")
    print("=" * 60)

    print("📊 데이터 로드 중...")
    df = pd.read_csv(data_file, index_col=0)

    # _fx 컬럼만, DUMMY_fx 제외
    fx_cols = [c for c in df.columns if c.endswith("_fx") and c != "DUMMY_fx"]
    num_nodes = len(fx_cols)

    print(f"[INFO] Using fx columns (len={num_nodes}): {fx_cols}")
    if num_nodes != 5:
        raise RuntimeError(f"[FATAL] 논문 설정은 5노드여야 합니다. 현재 num_nodes={num_nodes}, fx_cols={fx_cols}")

    df_fx = df[fx_cols].apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0)
    if use_diff_data:
        df_fx = df_fx.diff().iloc[1:].fillna(0.0)
        print(f"[INFO] Differencing enabled -> training on returns-like series, T={len(df_fx)}")

    # time split 기준으로 scaler fit: window 생성 전에, row 기준으로 train 범위 잡기
    T = len(df_fx)
    train_rows = int(T * train_ratio)  # 단순 row 기준 (window split과 약간 다를 수 있으나 leakage 방지 목적)
    if train_rows <= seq_length:
        raise RuntimeError(f"[FATAL] train_rows({train_rows})가 seq_length({seq_length})보다 작습니다. train_ratio 또는 seq_length 조정 필요")

    data_scaled, scaler = normalize_train_only(df_fx, train_rows=train_rows)

    # windowing (입력: [T,N] -> X:[S,1,N,L], Y:[S,H,N])
    X, Y = build_windows(data_scaled, seq_length=seq_length, horizon=horizon)
    print(f"[INFO] Windowed X: {X.shape}, Y: {Y.shape}")

    # split (window index 기준으로 time split)
    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = split_by_time(X, Y, train_ratio=train_ratio, valid_ratio=valid_ratio)
    print(f"[INFO] Train: {train_X.shape}, Valid: {valid_X.shape}, Test: {test_X.shape}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"💻 Device: {device}")

    print("🏗️  모델 초기화 중...")
    model = gtnet(
        gcn_true=True,
        buildA_true=True,
        gcn_depth=2,
        num_nodes=num_nodes,
        device=device,            # ✅ 너희 gtnet은 device 인자가 필수로 들어감
        predefined_A=None,
        static_feat=None,
        dropout=0.35,
        subgraph_size=min(20, num_nodes),
        node_dim=40,
        dilation_exponential=1,
        conv_channels=32,
        residual_channels=32,
        skip_channels=64,
        end_channels=128,
        seq_length=seq_length,
        in_dim=1,
        out_dim=horizon,          # 36
        layers=5,
        propalpha=0.05,
        tanhalpha=3,
        layer_norm_affline=True,
        attn_heads=4,
        feature_gru_hidden=48,
        predict_delta_output=predict_delta_output,
        delta_scale=delta_scale,
        clamp_range=clamp_range,
        usd_anchor_idx=usd_anchor_idx,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=learning_rate * 0.1)
    criterion = nn.MSELoss()

    def run_epoch(split_name, X_np, Y_np, train_mode: bool):
        if train_mode:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        n_batches = 0

        # 시간 순서를 크게 깨지 않기 위해: train만 셔플(논문에서도 허용됨), valid/test는 순차
        idx = np.arange(len(X_np))
        if train_mode:
            np.random.shuffle(idx)

        for s in range(0, len(X_np), batch_size):
            b = idx[s:s + batch_size]
            x = torch.from_numpy(X_np[b]).float().to(device)  # [B,1,N,L]
            y = torch.from_numpy(Y_np[b]).float().to(device)  # [B,H,N]

            if train_mode:
                optimizer.zero_grad()

            with torch.set_grad_enabled(train_mode):
                out = model(x)
                out = adapt_output_to_y(out, y)
                # exclude USD (anchor) from loss to stabilize baseline
                mask = torch.ones_like(y)
                if 0 <= usd_anchor_idx < y.size(2):
                    mask[:, :, usd_anchor_idx] = 0.0

                diff = (out - y) * mask
                mse_loss = (diff ** 2).sum() / (mask.sum() + 1e-9)
                y_diff = y[:, 1:, :] - y[:, :-1, :]
                out_diff = out[:, 1:, :] - out[:, :-1, :]
                # correlation-based shape loss (directional + volatility alignment)
                out_norm = F.normalize(out_diff.reshape(out_diff.size(0), -1), dim=1, eps=1e-6)
                y_norm = F.normalize(y_diff.reshape(y_diff.size(0), -1), dim=1, eps=1e-6)
                cos_sim = (out_norm * y_norm).sum(dim=1)
                shape_loss = (1 - cos_sim).mean()
                loss = mse_loss + shape_loss_weight * shape_loss

                if train_mode:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(1, n_batches)

    print(f"📈 학습 시작 (epochs={epochs})")
    for epoch in range(epochs):
        tr = run_epoch("train", train_X, train_Y, train_mode=True)
        va = run_epoch("valid", valid_X, valid_Y, train_mode=False)
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] train_loss={tr:.6f} valid_loss={va:.6f}")

    print("✅ 학습 완료!")

    # 저장 (state_dict + metadata)
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    save_obj = {
        "model_state_dict": model.state_dict(),
        "num_nodes": num_nodes,
        "fx_cols": fx_cols,
        "seq_length": seq_length,
        "horizon": horizon,
        "scaler_mean": scaler.mean_.astype(np.float32),
        "scaler_scale": scaler.scale_.astype(np.float32),
        "predict_delta_output": predict_delta_output,
        "delta_scale": float(delta_scale),
        "clamp_range": clamp_range,
        "usd_anchor_idx": int(usd_anchor_idx),
    }
    torch.save(save_obj, model_save_path)
    print(f"💾 모델(state_dict) 저장: {model_save_path}")

    return model_save_path


if __name__ == "__main__":
    data_file = str(PROJECT_ROOT / "FX_Data" / "ExchangeRate_dataset.csv")
    model_save_path = str(PROJECT_ROOT / "model" / "Bayesian" / "model_5node.pt")

    print("\n" + "=" * 60)
    print("🎯 GNN 모델 학습 스크립트")
    print("=" * 60)
    print(f"📁 데이터 파일: {data_file}")
    print(f"💾 모델 저장 경로: {model_save_path}")

    # 바로 학습 실행
    train_model(
        data_file=data_file,
        model_save_path=model_save_path,
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        train_ratio=0.6,
        valid_ratio=0.2,
        seq_length=12,
        horizon=36
    )

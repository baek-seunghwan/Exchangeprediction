"""
환율 예측 스크립트 (Exchange Rate Forecasting)
- MTGNN(gtnet) state_dict 체크포인트 로드 기반 GNN 예측
- 36개월(3년) 예측
"""

from pathlib import Path
import sys
import os

# ---------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # .../B-MTGNN
sys.path.insert(0, str(PROJECT_ROOT))  # Edit_Code를 패키지로 인식

import numpy as np
import pandas as pd
import torch

from Edit_Code.net import gtnet


def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _normalize_with_saved_scaler(df_fx: pd.DataFrame, scaler_mean: np.ndarray, scaler_scale: np.ndarray):
    """train.py에서 저장한 StandardScaler 파라미터로 동일 정규화 적용"""
    x = df_fx.values.astype(np.float32)
    scale = np.where(scaler_scale == 0, 1.0, scaler_scale).astype(np.float32)
    return ((x - scaler_mean.astype(np.float32)) / scale).astype(np.float32)


def _build_last_input(data_scaled_2d: np.ndarray, seq_length: int):
    """
    data_scaled_2d: [T, N]
    return inp: torch.Tensor [1,1,N,L]
    """
    T, N = data_scaled_2d.shape
    if T < seq_length:
        raise ValueError(f"데이터 길이(T={T})가 seq_length(L={seq_length})보다 짧습니다.")

    last = data_scaled_2d[-seq_length:, :]   # [L, N]
    inp = torch.tensor(last.T, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,N,L]
    return inp


def _parse_model_output(out: torch.Tensor, horizon: int, num_nodes: int) -> np.ndarray:
    """
    out을 [H, N] numpy로 변환
    허용 케이스:
      [1,H,N]
      [1,H,N,1]
      [1,N,H]
      [1,N,H,1]
    """
    if out.ndim == 4 and out.shape[-1] == 1:
        out = out[..., 0]

    if out.ndim != 3:
        raise RuntimeError(f"Unexpected model output shape: {tuple(out.shape)}")

    # [1,H,N]
    if out.shape[1] == horizon and out.shape[2] == num_nodes:
        pred = out[0].detach().cpu().numpy()  # [H,N]
        return pred

    # [1,N,H] -> [1,H,N]
    if out.shape[1] == num_nodes and out.shape[2] == horizon:
        pred = out[0].permute(1, 0).detach().cpu().numpy()  # [H,N]
        return pred

    raise RuntimeError(f"Output shape mismatch. out={tuple(out.shape)}, expected H={horizon}, N={num_nodes}")


def load_gnn_checkpoint(model_path: str, device: torch.device):
    """
    train.py에서 저장한 state_dict 체크포인트 로드
    반환: (model, meta_dict)
    """
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    required = ["model_state_dict", "num_nodes", "fx_cols", "seq_length", "horizon", "scaler_mean", "scaler_scale"]
    for k in required:
        if k not in ckpt:
            raise RuntimeError(f"Checkpoint missing key: {k}")

    num_nodes = int(ckpt["num_nodes"])
    seq_length = int(ckpt["seq_length"])
    horizon = int(ckpt["horizon"])

    model = gtnet(
        gcn_true=True,
        buildA_true=True,
        gcn_depth=2,
        num_nodes=num_nodes,
        device=device,
        predefined_A=None,
        static_feat=None,
        dropout=0.3,
        subgraph_size=min(20, num_nodes),
        node_dim=40,
        dilation_exponential=1,
        conv_channels=32,
        residual_channels=32,
        skip_channels=64,
        end_channels=128,
        seq_length=seq_length,
        in_dim=1,
        out_dim=horizon,
        layers=5,
        propalpha=0.05,
        tanhalpha=3,
        layer_norm_affline=True
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, ckpt


def generate_forecast(data_file: str, model_path: str, method: str = "gnn"):
    """
    환율 예측 생성 (GNN만: 실패 시 예외)
    반환: forecasts dict
    """
    print("=" * 60)
    print("🚀 환율 예측 생성")
    print("=" * 60)

    print("📊 데이터 로드 중...")
    df = pd.read_csv(data_file, index_col=0)

    # 기초 통화 매핑(표시용)
    currency_map = {
        "us_fx": "USD",
        "kr_fx": "KRW",
        "jp_fx": "JPY",
        "uk_fx": "GBP",
        "cn_fx": "CNY",
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if method.lower() != "gnn":
        raise RuntimeError("이 버전은 논문용으로 GNN만 허용합니다. method='gnn'만 사용하세요.")

    print(f"📥 모델 로드 중: {model_path}")
    model, meta = load_gnn_checkpoint(model_path, device=device)
    print("✅ 모델 로드 완료")

    fx_cols = [c for c in meta["fx_cols"] if c != "DUMMY_fx"]
    num_nodes = int(meta["num_nodes"])
    seq_length = int(meta["seq_length"])
    horizon = int(meta["horizon"])

    if len(fx_cols) != num_nodes:
        raise RuntimeError(f"Checkpoint fx_cols len({len(fx_cols)}) != num_nodes({num_nodes}). fx_cols={fx_cols}")

    # df에서 필요한 컬럼이 모두 존재해야 함
    missing = [c for c in fx_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"CSV에 체크포인트 컬럼이 없습니다: {missing}. CSV cols={list(df.columns)}")

    print(f"[DEBUG] final use_cols (len={len(fx_cols)}): {fx_cols}")

    df_fx = df[fx_cols].apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0)

    # 저장된 scaler로 동일 정규화
    data_scaled = _normalize_with_saved_scaler(
        df_fx,
        scaler_mean=np.array(meta["scaler_mean"], dtype=np.float32),
        scaler_scale=np.array(meta["scaler_scale"], dtype=np.float32),
    )

    # 입력 텐서 생성
    inp = _build_last_input(data_scaled, seq_length=seq_length).to(device)  # [1,1,N,L]

    print("\n📈 예측 방식: GNN")
    print("-" * 60)

    with torch.no_grad():
        out = model(inp)

    pred_mat = _parse_model_output(out, horizon=horizon, num_nodes=num_nodes)  # [H,N]

    # 날짜 구성
    dates = pd.date_range(start="2011-01", periods=len(df_fx), freq="MS")
    forecast_dates = pd.date_range(start=dates[-1], periods=horizon + 1, freq="MS")[1:]

    forecasts = {}
    # actual은 “스케일된 값”으로 저장(플롯 일관성). 원하면 역변환도 가능.
    for j, col in enumerate(fx_cols):
        cur = currency_map.get(col, col)
        forecasts[cur] = {
            "scaler_mean": float(meta["scaler_mean"][j]),
            "scaler_scale": float(meta["scaler_scale"][j]),
            "actual": data_scaled[:, j],
            "forecast": pred_mat[:, j],
            "actual_dates": dates,
            "forecast_dates": forecast_dates,
            "params": {
                "method": "standard_scaler(train-only)",
                "fx_col": col,
                "seq_length": seq_length,
                "horizon": horizon,
            },
        }
        print(f"✅ {cur}: {horizon}개월 GNN 예측 완료")

    print("-" * 60)
    print("✅ 전체 예측 완료!")
    return forecasts


if __name__ == "__main__":
    data_file = str(PROJECT_ROOT / "FX_Data" / "ExchangeRate_dataset.csv")
    model_path = str(PROJECT_ROOT / "model" / "Bayesian" / "model_5node.pt")

    _ = generate_forecast(
        data_file=data_file,
        model_path=model_path,
        method="gnn"
    )

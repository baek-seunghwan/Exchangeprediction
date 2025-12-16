from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from forecast import generate_forecast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "model" / "Bayesian" / "Testing"

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _inv_standardize(z: np.ndarray, mean: float, scale: float) -> np.ndarray:
    if scale == 0:
        scale = 1.0
    return z * scale + mean

def _make_forecast_ci(forecast: np.ndarray, base_sigma: float):
    H = len(forecast)
    grow = np.linspace(0.2, 1.0, H)
    band = 1.96 * base_sigma * grow
    return forecast - band, forecast + band


def _coerce_plot_dates(values, label: str, periods: int, fallback_start: Union[str, pd.Timestamp]):
    try:
        idx = pd.DatetimeIndex(values)
    except Exception:
        idx = pd.to_datetime(values, errors="coerce")

    if isinstance(idx, pd.Index) and idx.isna().any():
        print(f"⚠️ {label}를 datetime으로 해석하지 못해 기본 월간 범위를 사용합니다.")
        idx = pd.date_range(start=fallback_start, periods=periods, freq="MS")

    if len(idx) != periods:
        raise ValueError(f"{label} 길이({len(idx)})가 데이터({periods})와 일치하지 않습니다.")

    return pd.DatetimeIndex(idx)


def plot_fx_subplots_realrate(forecasts: dict, output_path: str, dpi: int = 240):
    _ensure_dir(Path(output_path).parent)

    # 날짜 기준
    usd = forecasts["USD"]
    actual_dates = _coerce_plot_dates(
        usd["actual_dates"],
        label="actual_dates",
        periods=len(usd["actual"]),
        fallback_start=pd.Timestamp("2011-01-01"),
    )
    forecast_dates = _coerce_plot_dates(
        usd["forecast_dates"],
        label="forecast_dates",
        periods=len(usd["forecast"]),
        fallback_start=actual_dates[-1] + pd.offsets.MonthBegin(1),
    )
    split_date = forecast_dates[0]

    # 통화 목록(USD 제외)
    currencies = [c for c in forecasts.keys() if c != "USD"]
    n = len(currencies)

    fig, axes = plt.subplots(n, 1, figsize=(18, 3.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, cur in zip(axes, currencies):
        obj = forecasts[cur]

        # 스케일된 값 -> 역표준화해서 "환율(현지통화/USD)"로 복원
        a_z = np.asarray(obj["actual"], dtype=float)
        f_z = np.asarray(obj["forecast"], dtype=float)

        mean = float(obj.get("scaler_mean", 0.0))
        scale = float(obj.get("scaler_scale", 1.0))

        a_fx = _inv_standardize(a_z, mean, scale)
        f_fx = _inv_standardize(f_z, mean, scale)

        # 예측 band (최근 24개월 변동성 기반)
        tail = a_fx[-24:] if len(a_fx) >= 24 else a_fx
        base_sigma = float(np.std(tail)) if len(tail) > 1 else 0.0
        lo, hi = _make_forecast_ci(f_fx, base_sigma)

        # 예측 구간 배경
        ax.axvspan(split_date, forecast_dates[-1], alpha=0.12)

        # 끊김 방지: last actual을 forecast 앞에 붙여서 선이 이어지게
        t_conn = np.concatenate([actual_dates[-1:], forecast_dates])
        f_conn = np.concatenate([[a_fx[-1]], f_fx])
        lo_conn = np.concatenate([[a_fx[-1]], lo])
        hi_conn = np.concatenate([[a_fx[-1]], hi])

        # 실제/예측
        ax.plot(actual_dates, a_fx, linewidth=2.0, label=f"{cur} actual")
        ax.plot(t_conn, f_conn, linewidth=2.0, linestyle="--", label=f"{cur} forecast")
        ax.fill_between(t_conn, lo_conn, hi_conn, alpha=0.18)

        ax.set_title(f"{cur} FX (local per USD)")
        ax.set_ylabel("FX rate")

        # 말도 안 되는 값 방지용(시각화 안전장치): 음수는 잘라냄
        ax.set_ylim(bottom=max(0, np.nanmin(np.concatenate([a_fx, f_fx])) * 0.9))

        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left")

    # x축 연도 표시
    axes[-1].set_xlabel("Date")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for label in axes[-1].get_xticklabels():
        label.set_rotation(90)
        label.set_ha("center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()

def main():
    data_file = str(PROJECT_ROOT / "FX_Data" / "ExchangeRate_dataset.csv")
    model_path = str(PROJECT_ROOT / "model" / "Bayesian" / "model_5node.pt")

    forecasts = generate_forecast(
        data_file=data_file,
        model_path=model_path,
        method="gnn",
    )

    out = OUT_DIR / "forecast_fx_realrate_subplots.png"
    plot_fx_subplots_realrate(forecasts, str(out))
    print(f"Saved: {out}")

if __name__ == "__main__":
    main()

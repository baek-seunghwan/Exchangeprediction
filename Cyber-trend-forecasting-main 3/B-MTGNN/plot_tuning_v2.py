import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--goal", type=float, default=0.5)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"results.csv not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("results.csv is empty")

    df = df.sort_values("run_id").reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(df["run_id"], df["test_focus_rrse"], marker="o", color="#2f6db3", label="focus_rrse")
    axes[0].plot(df["run_id"], df["final_test_rse"], marker="s", color="#666666", alpha=0.7, label="final_test_rse")
    axes[0].axhline(args.goal, color="#d62728", linestyle="--", linewidth=1.5, label=f"goal={args.goal}")
    axes[0].set_title("RSE Progress by Trial")
    axes[0].set_xlabel("run_id")
    axes[0].set_ylabel("RSE")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    best_idx = df["objective_total"].idxmin()
    best = df.loc[best_idx]

    bars = [best["final_test_rse"], best["test_focus_rrse"], args.goal]
    labels = ["best_final_test_rse", "best_focus_rrse", "goal"]
    colors = ["#4c78a8", "#f58518", "#e45756"]

    axes[1].bar(labels, bars, color=colors)
    axes[1].set_title("Best Trial vs Goal")
    axes[1].set_ylabel("RSE")
    axes[1].grid(True, axis="y", alpha=0.3)

    for i, v in enumerate(bars):
        axes[1].text(i, float(v) + 0.03, f"{float(v):.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    out_path = run_dir / "rse_analysis.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()

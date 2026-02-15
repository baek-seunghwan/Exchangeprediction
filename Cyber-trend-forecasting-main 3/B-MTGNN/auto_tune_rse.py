import argparse
import csv
import itertools
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path


def build_trials():
    return [
        {"lr": 2e-4, "dropout": 0.10, "layers": 2, "conv_channels": 16, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "seq_in_len": 24, "seq_out_len": 1, "ss_prob": 0.2},
        {"lr": 3e-4, "dropout": 0.10, "layers": 2, "conv_channels": 16, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "seq_in_len": 24, "seq_out_len": 1, "ss_prob": 0.3},
        {"lr": 5e-4, "dropout": 0.10, "layers": 2, "conv_channels": 16, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "seq_in_len": 24, "seq_out_len": 1, "ss_prob": 0.2},
        {"lr": 2e-4, "dropout": 0.05, "layers": 2, "conv_channels": 16, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "seq_in_len": 24, "seq_out_len": 1, "ss_prob": 0.2},
        {"lr": 2e-4, "dropout": 0.10, "layers": 2, "conv_channels": 16, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "seq_in_len": 12, "seq_out_len": 1, "ss_prob": 0.2},
        {"lr": 3e-4, "dropout": 0.05, "layers": 2, "conv_channels": 16, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "seq_in_len": 12, "seq_out_len": 1, "ss_prob": 0.3},
        {"lr": 2e-4, "dropout": 0.10, "layers": 1, "conv_channels": 8, "residual_channels": 64, "skip_channels": 128, "end_channels": 512, "seq_in_len": 24, "seq_out_len": 1, "ss_prob": 0.2},
        {"lr": 3e-4, "dropout": 0.10, "layers": 1, "conv_channels": 8, "residual_channels": 64, "skip_channels": 128, "end_channels": 512, "seq_in_len": 12, "seq_out_len": 1, "ss_prob": 0.2},
    ]


def parse_metrics(output_text: str):
    final_rse = None
    final_rae = None
    best_test_rse = None

    m_final = re.findall(r"final test rse\s+([0-9.]+)\s*\|\s*test rae\s+([0-9.]+)", output_text)
    if m_final:
        final_rse = float(m_final[-1][0])
        final_rae = float(m_final[-1][1])

    m_best = re.findall(r"best test rse=\s*([0-9.]+)", output_text)
    if m_best:
        best_test_rse = float(m_best[-1])

    return final_rse, final_rae, best_test_rse


def run_once(py_exec, script_path, common_args, trial, seed, run_dir, run_id):
    cmd = [
        py_exec,
        str(script_path),
        "--epochs", str(common_args.epochs),
        "--horizon", "1",
        "--normalize", "2",
        "--batch_size", str(common_args.batch_size),
        "--train_ratio", str(common_args.train_ratio),
        "--valid_ratio", str(common_args.valid_ratio),
        "--focus_targets", "0",
        "--debug_eval", "0",
        "--rollout_mode", common_args.rollout_mode,
        "--plot", "0",
        "--seed", str(seed),
        "--lr", str(trial["lr"]),
        "--dropout", str(trial["dropout"]),
        "--layers", str(trial["layers"]),
        "--conv_channels", str(trial["conv_channels"]),
        "--residual_channels", str(trial["residual_channels"]),
        "--skip_channels", str(trial["skip_channels"]),
        "--end_channels", str(trial["end_channels"]),
        "--seq_in_len", str(trial["seq_in_len"]),
        "--seq_out_len", str(trial["seq_out_len"]),
        "--ss_prob", str(trial["ss_prob"]),
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output = proc.stdout

    log_path = run_dir / f"run_{run_id:03d}.log"
    log_path.write_text(output, encoding="utf-8")

    final_rse, final_rae, best_test_rse = parse_metrics(output)

    return {
        "run_id": run_id,
        "seed": seed,
        "trial": trial,
        "return_code": proc.returncode,
        "final_test_rse": final_rse,
        "final_test_rae": final_rae,
        "best_test_rse": best_test_rse,
        "log_file": str(log_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--py", type=str, default="/Users/samrobert/Documents/GitHub/Exchangeprediction/.venv/bin/python")
    parser.add_argument("--epochs", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--train_ratio", type=float, default=0.8666666667)
    parser.add_argument("--valid_ratio", type=float, default=0.0666666667)
    parser.add_argument("--rollout_mode", type=str, default="teacher_forced", choices=["teacher_forced", "recursive"])
    parser.add_argument("--seeds", type=str, default="123,777,2026")
    parser.add_argument("--max_trials", type=int, default=8)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    train_script = root / "train_test.py"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / "tuning_runs" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    trials = build_trials()[: args.max_trials]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    results = []
    run_id = 1
    total = len(trials) * len(seeds)

    for t_idx, trial in enumerate(trials, 1):
        for seed in seeds:
            print(f"[{run_id}/{total}] trial={t_idx} seed={seed} start")
            res = run_once(args.py, train_script, args, trial, seed, out_dir, run_id)
            results.append(res)
            print(f"[{run_id}/{total}] done final_test_rse={res['final_test_rse']} best_test_rse={res['best_test_rse']}")
            run_id += 1

    valid = [r for r in results if r["final_test_rse"] is not None]
    valid_sorted = sorted(valid, key=lambda x: x["final_test_rse"]) if valid else []

    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "seed", "final_test_rse", "final_test_rae", "best_test_rse", "return_code", "trial_json", "log_file"])
        for r in results:
            w.writerow([
                r["run_id"],
                r["seed"],
                r["final_test_rse"],
                r["final_test_rae"],
                r["best_test_rse"],
                r["return_code"],
                json.dumps(r["trial"], ensure_ascii=False),
                r["log_file"],
            ])

    summary_path = out_dir / "best_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        if valid_sorted:
            best = valid_sorted[0]
            f.write(f"best_final_test_rse={best['final_test_rse']}\n")
            f.write(f"best_seed={best['seed']}\n")
            f.write(f"best_trial={json.dumps(best['trial'], ensure_ascii=False)}\n")
            f.write(f"log_file={best['log_file']}\n")
        else:
            f.write("No valid parsed runs. Check logs.\n")

    print("output_dir:", out_dir)
    if valid_sorted:
        print("best_final_test_rse:", valid_sorted[0]["final_test_rse"])
        print("best_seed:", valid_sorted[0]["seed"])
        print("best_trial:", valid_sorted[0]["trial"])
    else:
        print("No valid parsed runs.")


if __name__ == "__main__":
    main()

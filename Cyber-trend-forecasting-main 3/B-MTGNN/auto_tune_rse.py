import argparse
import csv
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


def build_trials():
    return [
        {"lr": 1.5e-4, "dropout": 0.10, "layers": 2, "conv_channels": 16, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "seq_in_len": 24, "seq_out_len": 1, "ss_prob": 0.1, "focus_target_gain": 30.0, "trend_penalty": 0.1, "anchor_focus_to_last": 0.05},
        {"lr": 2e-4, "dropout": 0.10, "layers": 2, "conv_channels": 16, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "seq_in_len": 24, "seq_out_len": 1, "ss_prob": 0.2, "focus_target_gain": 40.0, "trend_penalty": 0.2, "anchor_focus_to_last": 0.10},
        {"lr": 2.5e-4, "dropout": 0.10, "layers": 2, "conv_channels": 16, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "seq_in_len": 24, "seq_out_len": 1, "ss_prob": 0.2, "focus_target_gain": 50.0, "trend_penalty": 0.25, "anchor_focus_to_last": 0.10},
        {"lr": 3e-4, "dropout": 0.10, "layers": 2, "conv_channels": 16, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "seq_in_len": 24, "seq_out_len": 1, "ss_prob": 0.2, "focus_target_gain": 60.0, "trend_penalty": 0.3, "anchor_focus_to_last": 0.12},
        {"lr": 2e-4, "dropout": 0.05, "layers": 2, "conv_channels": 16, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "seq_in_len": 24, "seq_out_len": 1, "ss_prob": 0.1, "focus_target_gain": 50.0, "trend_penalty": 0.2, "anchor_focus_to_last": 0.05},
        {"lr": 3e-4, "dropout": 0.05, "layers": 2, "conv_channels": 16, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "seq_in_len": 24, "seq_out_len": 1, "ss_prob": 0.1, "focus_target_gain": 60.0, "trend_penalty": 0.25, "anchor_focus_to_last": 0.08},
        {"lr": 1.5e-4, "dropout": 0.10, "layers": 2, "conv_channels": 16, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "seq_in_len": 36, "seq_out_len": 1, "ss_prob": 0.2, "focus_target_gain": 30.0, "trend_penalty": 0.15, "anchor_focus_to_last": 0.05},
        {"lr": 2e-4, "dropout": 0.10, "layers": 2, "conv_channels": 16, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "seq_in_len": 36, "seq_out_len": 1, "ss_prob": 0.2, "focus_target_gain": 40.0, "trend_penalty": 0.2, "anchor_focus_to_last": 0.08},
        {"lr": 2.5e-4, "dropout": 0.10, "layers": 2, "conv_channels": 16, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "seq_in_len": 36, "seq_out_len": 1, "ss_prob": 0.2, "focus_target_gain": 50.0, "trend_penalty": 0.25, "anchor_focus_to_last": 0.10},
        {"lr": 2e-4, "dropout": 0.05, "layers": 2, "conv_channels": 16, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "seq_in_len": 36, "seq_out_len": 1, "ss_prob": 0.1, "focus_target_gain": 50.0, "trend_penalty": 0.2, "anchor_focus_to_last": 0.05},
        {"lr": 1.5e-4, "dropout": 0.10, "layers": 3, "conv_channels": 32, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "seq_in_len": 24, "seq_out_len": 1, "ss_prob": 0.2, "focus_target_gain": 30.0, "trend_penalty": 0.2, "anchor_focus_to_last": 0.05},
        {"lr": 2e-4, "dropout": 0.10, "layers": 3, "conv_channels": 32, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "seq_in_len": 24, "seq_out_len": 1, "ss_prob": 0.2, "focus_target_gain": 40.0, "trend_penalty": 0.25, "anchor_focus_to_last": 0.08},
        {"lr": 1.5e-4, "dropout": 0.05, "layers": 3, "conv_channels": 32, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "seq_in_len": 36, "seq_out_len": 1, "ss_prob": 0.1, "focus_target_gain": 30.0, "trend_penalty": 0.2, "anchor_focus_to_last": 0.05},
        {"lr": 2e-4, "dropout": 0.05, "layers": 3, "conv_channels": 32, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "seq_in_len": 36, "seq_out_len": 1, "ss_prob": 0.1, "focus_target_gain": 40.0, "trend_penalty": 0.25, "anchor_focus_to_last": 0.08},
        {"lr": 1e-4, "dropout": 0.10, "layers": 2, "conv_channels": 16, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "seq_in_len": 48, "seq_out_len": 1, "ss_prob": 0.1, "focus_target_gain": 20.0, "trend_penalty": 0.15, "anchor_focus_to_last": 0.05},
        {"lr": 1.5e-4, "dropout": 0.05, "layers": 2, "conv_channels": 16, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "seq_in_len": 48, "seq_out_len": 1, "ss_prob": 0.1, "focus_target_gain": 30.0, "trend_penalty": 0.2, "anchor_focus_to_last": 0.05},
    ]


def value_for_trial(trial, key, default):
    return trial[key] if key in trial else default


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


def run_once(py_exec, script_path, common_args, trial, seed, run_dir, run_id, data_path):
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"model_{run_id:03d}.pt"

    cmd = [
        py_exec,
        str(script_path),
        "--data", str(data_path),
        "--epochs", str(common_args.epochs),
        "--horizon", "1",
        "--normalize", "2",
        "--batch_size", str(common_args.batch_size),
        "--train_ratio", str(common_args.train_ratio),
        "--valid_ratio", str(common_args.valid_ratio),
        "--fixed_eval_periods", str(common_args.fixed_eval_periods),
        "--valid_year", str(common_args.valid_year),
        "--test_year", str(common_args.test_year),
        "--focus_targets", str(common_args.focus_targets),
        "--focus_nodes", common_args.focus_nodes,
        "--focus_weight", str(common_args.focus_weight),
        "--focus_target_gain", str(value_for_trial(trial, "focus_target_gain", common_args.focus_target_gain)),
        "--focus_only_loss", str(common_args.focus_only_loss),
        "--anchor_focus_to_last", str(value_for_trial(trial, "anchor_focus_to_last", common_args.anchor_focus_to_last)),
        "--rse_targets", common_args.rse_targets,
        "--rse_report_mode", common_args.rse_report_mode,
        "--loss_mode", common_args.loss_mode,
        "--target_profile", common_args.target_profile,
        "--bias_penalty", str(common_args.bias_penalty),
        "--bias_penalty_scope", common_args.bias_penalty_scope,
        "--trend_penalty", str(value_for_trial(trial, "trend_penalty", common_args.trend_penalty)),
        "--trend_penalty_scope", common_args.trend_penalty_scope,
        "--early_stop_patience", str(common_args.early_stop_patience),
        "--early_stop_min_epochs", str(common_args.early_stop_min_epochs),
        "--debug_eval", "0",
        "--rollout_mode", common_args.rollout_mode,
        "--force_recursive_eval", str(common_args.force_recursive_eval),
        "--plot", "0",
        "--use_graph", "0",
        "--autotune_mode", "1",
        "--save", str(ckpt_path),
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
        "data": str(data_path),
        "use_graph": 0,
        "trial": trial,
        "return_code": proc.returncode,
        "final_test_rse": final_rse,
        "final_test_rae": final_rae,
        "best_test_rse": best_test_rse,
        "log_file": str(log_path),
    }


def write_results_json_csv(out_dir: Path, results):
    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "seed", "data", "use_graph", "final_test_rse", "final_test_rae", "best_test_rse", "objective_rse", "return_code", "trial_json", "log_file"])
        for r in results:
            obj = r["best_test_rse"] if r["best_test_rse"] is not None else r["final_test_rse"]
            w.writerow([
                r["run_id"],
                r["seed"],
                r["data"],
                r["use_graph"],
                r["final_test_rse"],
                r["final_test_rae"],
                r["best_test_rse"],
                obj,
                r["return_code"],
                json.dumps(r["trial"], ensure_ascii=False),
                r["log_file"],
            ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--py", type=str, default="/Users/samrobert/Documents/GitHub/Exchangeprediction/.venv/bin/python")
    parser.add_argument("--epochs", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--train_ratio", type=float, default=0.8666666667)
    parser.add_argument("--valid_ratio", type=float, default=0.0666666667)
    parser.add_argument("--fixed_eval_periods", type=int, default=1, choices=[0, 1])
    parser.add_argument("--valid_year", type=int, default=2024)
    parser.add_argument("--test_year", type=int, default=2025)
    parser.add_argument("--focus_targets", type=int, default=1)
    parser.add_argument("--focus_nodes", type=str, default="us_Trade Weighted Dollar Index")
    parser.add_argument("--focus_weight", type=float, default=1.0)
    parser.add_argument("--focus_target_gain", type=float, default=40.0)
    parser.add_argument("--focus_only_loss", type=int, default=1, choices=[0, 1])
    parser.add_argument("--anchor_focus_to_last", type=float, default=0.1)
    parser.add_argument("--rse_targets", type=str, default="Us_Trade Weighted Dollar Index_Testing.txt")
    parser.add_argument("--rse_report_mode", type=str, default="targets", choices=["targets", "all"])
    parser.add_argument("--loss_mode", type=str, default="mse", choices=["l1", "mse"])
    parser.add_argument("--target_profile", type=str, default="none", choices=["none", "triple_050", "run001_us"])
    parser.add_argument("--bias_penalty", type=float, default=0.0)
    parser.add_argument("--bias_penalty_scope", type=str, default="focus", choices=["focus", "all"])
    parser.add_argument("--trend_penalty", type=float, default=0.2)
    parser.add_argument("--trend_penalty_scope", type=str, default="focus", choices=["focus", "all"])
    parser.add_argument("--early_stop_patience", type=int, default=20)
    parser.add_argument("--early_stop_min_epochs", type=int, default=30)
    parser.add_argument("--goal_rse", type=float, default=0.5, help="stop sweep early when objective_rse <= goal")
    parser.add_argument("--rollout_mode", type=str, default="recursive", choices=["teacher_forced", "recursive"])
    parser.add_argument("--force_recursive_eval", type=int, default=1, choices=[0, 1])
    parser.add_argument("--seeds", type=str, default="123,777,2026")
    parser.add_argument("--max_trials", type=int, default=16)
    parser.add_argument("--max_runtime_minutes", type=int, default=300, help="wall-clock cap for sweep; <=0 disables")
    parser.add_argument("--datasets", type=str, default="sm_data.csv")
    parser.add_argument("--resume_dir", type=str, default="", help="existing tuning_runs/<timestamp> directory to resume from")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    train_script = root / "train_test.py"

    if args.resume_dir.strip():
        out_dir = Path(args.resume_dir).expanduser().resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = root / "tuning_runs" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    trials = build_trials()[: args.max_trials]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]

    combos = []
    for dataset in datasets:
        for trial in trials:
            for seed in seeds:
                combos.append((dataset, trial, seed))

    results_path = out_dir / "results.json"
    if results_path.exists():
        try:
            results = json.loads(results_path.read_text(encoding="utf-8"))
            if not isinstance(results, list):
                results = []
        except Exception:
            results = []
    else:
        results = []

    done_keys = set()
    max_existing_run_id = 0
    for row in results:
        key = (str(row.get("data", "")), int(row.get("seed", -1)), json.dumps(row.get("trial", {}), sort_keys=True))
        done_keys.add(key)
        max_existing_run_id = max(max_existing_run_id, int(row.get("run_id", 0) or 0))

    run_id = max_existing_run_id + 1
    total = len(combos)
    done_count = len(done_keys)

    if done_count > 0:
        print(f"resume_dir: {out_dir}")
        print(f"already completed: {done_count}/{total}")

    sweep_start = time.time()
    for idx, (dataset, trial, seed) in enumerate(combos, 1):
        if args.max_runtime_minutes > 0:
            elapsed_min = (time.time() - sweep_start) / 60.0
            if elapsed_min >= args.max_runtime_minutes:
                print(f"time budget reached: elapsed={elapsed_min:.1f} min >= max_runtime_minutes={args.max_runtime_minutes}. stopping.")
                break

        key = (str(root / "data" / dataset), int(seed), json.dumps(trial, sort_keys=True))
        if key in done_keys:
            print(f"[{idx}/{total}] skip (already done) data={dataset} seed={seed}")
            continue

        data_path = root / "data" / dataset
        t_idx = trials.index(trial) + 1
        print(f"[{idx}/{total}] data={dataset} graph=0 trial={t_idx} seed={seed} start")
        try:
            res = run_once(args.py, train_script, args, trial, seed, out_dir, run_id, data_path)
        except KeyboardInterrupt:
            print("Interrupted by user. Partial results are saved.")
            write_results_json_csv(out_dir, results)
            raise

        results.append(res)
        done_keys.add(key)
        write_results_json_csv(out_dir, results)
        print(f"[{idx}/{total}] done final_test_rse={res['final_test_rse']} best_test_rse={res['best_test_rse']}")

        obj = res["best_test_rse"] if res["best_test_rse"] is not None else res["final_test_rse"]
        if obj is not None and obj <= args.goal_rse:
            print(f"goal reached: objective_rse={obj} <= goal_rse={args.goal_rse}. stopping early.")
            break

        run_id += 1

    valid = [r for r in results if (r["best_test_rse"] is not None or r["final_test_rse"] is not None)]
    def objective(row):
        return row["best_test_rse"] if row["best_test_rse"] is not None else row["final_test_rse"]
    valid_sorted = sorted(valid, key=objective) if valid else []

    write_results_json_csv(out_dir, results)

    summary_path = out_dir / "best_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        if valid_sorted:
            best = valid_sorted[0]
            best_obj = best["best_test_rse"] if best["best_test_rse"] is not None else best["final_test_rse"]
            f.write(f"best_objective_rse={best_obj}\n")
            f.write(f"best_final_test_rse={best['final_test_rse']}\n")
            f.write(f"best_best_test_rse={best['best_test_rse']}\n")
            f.write(f"best_seed={best['seed']}\n")
            f.write(f"best_data={best['data']}\n")
            f.write(f"best_use_graph={best['use_graph']}\n")
            f.write(f"best_trial={json.dumps(best['trial'], ensure_ascii=False)}\n")
            f.write(f"log_file={best['log_file']}\n")
        else:
            f.write("No valid parsed runs. Check logs.\n")

    print("output_dir:", out_dir)
    if valid_sorted:
        best = valid_sorted[0]
        best_obj = best["best_test_rse"] if best["best_test_rse"] is not None else best["final_test_rse"]
        print("best_objective_rse:", best_obj)
        print("best_final_test_rse:", best["final_test_rse"])
        print("best_best_test_rse:", best["best_test_rse"])
        print("best_seed:", best["seed"])
        print("best_data:", best["data"])
        print("best_use_graph:", best["use_graph"])
        print("best_trial:", best["trial"])
    else:
        print("No valid parsed runs.")


if __name__ == "__main__":
    main()

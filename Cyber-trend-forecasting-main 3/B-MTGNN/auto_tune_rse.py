import argparse
import csv
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


def build_trials(expand_factor=20):
    base_trials = [
        {"lr": 2.0e-4, "dropout": 0.05, "layers": 2, "conv_channels": 16, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "subgraph_size": 40, "node_dim": 30, "seq_in_len": 36, "seq_out_len": 1, "ss_prob": 0.15, "focus_target_gain": 45.0, "anchor_focus_to_last": 0.70, "bias_penalty": 0.30},
        {"lr": 1.5e-4, "dropout": 0.02, "layers": 2, "conv_channels": 16, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "subgraph_size": 40, "node_dim": 30, "seq_in_len": 48, "seq_out_len": 1, "ss_prob": 0.10, "focus_target_gain": 60.0, "anchor_focus_to_last": 0.80, "bias_penalty": 0.25},
        {"lr": 1.0e-4, "dropout": 0.02, "layers": 2, "conv_channels": 16, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "subgraph_size": 40, "node_dim": 30, "seq_in_len": 60, "seq_out_len": 1, "ss_prob": 0.05, "focus_target_gain": 80.0, "anchor_focus_to_last": 0.85, "bias_penalty": 0.35},
        {"lr": 1.5e-4, "dropout": 0.00, "layers": 3, "conv_channels": 32, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "subgraph_size": 30, "node_dim": 40, "seq_in_len": 48, "seq_out_len": 1, "ss_prob": 0.10, "focus_target_gain": 80.0, "anchor_focus_to_last": 0.85, "bias_penalty": 0.40},
        {"lr": 1.0e-4, "dropout": 0.00, "layers": 3, "conv_channels": 32, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "subgraph_size": 30, "node_dim": 40, "seq_in_len": 60, "seq_out_len": 1, "ss_prob": 0.05, "focus_target_gain": 90.0, "anchor_focus_to_last": 0.90, "bias_penalty": 0.45},
        {"lr": 2.5e-4, "dropout": 0.05, "layers": 2, "conv_channels": 16, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "subgraph_size": 40, "node_dim": 30, "seq_in_len": 24, "seq_out_len": 1, "ss_prob": 0.20, "focus_target_gain": 50.0, "anchor_focus_to_last": 0.70, "bias_penalty": 0.20},
        {"lr": 2.0e-4, "dropout": 0.02, "layers": 2, "conv_channels": 32, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "subgraph_size": 30, "node_dim": 40, "seq_in_len": 36, "seq_out_len": 1, "ss_prob": 0.10, "focus_target_gain": 70.0, "anchor_focus_to_last": 0.85, "bias_penalty": 0.35},
        {"lr": 1.2e-4, "dropout": 0.02, "layers": 2, "conv_channels": 16, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "subgraph_size": 20, "node_dim": 50, "seq_in_len": 48, "seq_out_len": 1, "ss_prob": 0.05, "focus_target_gain": 70.0, "anchor_focus_to_last": 0.90, "bias_penalty": 0.40},
        {"lr": 1.0e-4, "dropout": 0.05, "layers": 2, "conv_channels": 16, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "subgraph_size": 20, "node_dim": 50, "seq_in_len": 60, "seq_out_len": 1, "ss_prob": 0.05, "focus_target_gain": 60.0, "anchor_focus_to_last": 0.90, "bias_penalty": 0.30},
        {"lr": 1.5e-4, "dropout": 0.00, "layers": 3, "conv_channels": 32, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "subgraph_size": 20, "node_dim": 50, "seq_in_len": 36, "seq_out_len": 1, "ss_prob": 0.10, "focus_target_gain": 85.0, "anchor_focus_to_last": 0.95, "bias_penalty": 0.50},
        {"lr": 8.0e-5, "dropout": 0.00, "layers": 3, "conv_channels": 32, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "subgraph_size": 20, "node_dim": 50, "seq_in_len": 60, "seq_out_len": 1, "ss_prob": 0.05, "focus_target_gain": 100.0, "anchor_focus_to_last": 0.95, "bias_penalty": 0.55},
        {"lr": 2.0e-4, "dropout": 0.02, "layers": 2, "conv_channels": 16, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "subgraph_size": 40, "node_dim": 30, "seq_in_len": 48, "seq_out_len": 1, "ss_prob": 0.10, "focus_target_gain": 55.0, "anchor_focus_to_last": 0.80, "bias_penalty": 0.28},
    ]

    if expand_factor <= 1:
        return base_trials

    lr_mults = [0.85, 1.00, 1.15, 0.70, 1.30]
    drop_shifts = [0.00, -0.01, 0.01, -0.02, 0.02]
    ss_shifts = [0.00, -0.03, 0.03, -0.05, 0.05]
    anchor_shifts = [0.00, -0.03, 0.03, -0.05, 0.05]
    gain_mults = [1.00, 0.90, 1.10, 0.80, 1.20]
    bias_mults = [1.00, 0.80, 1.20, 0.60, 1.40]
    seq_choices = [24, 36, 48, 60]

    expanded = []
    for idx in range(expand_factor):
        for base in base_trials:
            trial = dict(base)
            k = idx % len(lr_mults)
            trial["lr"] = max(1e-5, min(5e-4, base["lr"] * lr_mults[k]))
            trial["dropout"] = max(0.0, min(0.2, base["dropout"] + drop_shifts[k]))
            trial["ss_prob"] = max(0.0, min(0.3, base["ss_prob"] + ss_shifts[k]))
            trial["anchor_focus_to_last"] = max(0.0, min(0.99, base["anchor_focus_to_last"] + anchor_shifts[k]))
            trial["focus_target_gain"] = max(20.0, min(140.0, base["focus_target_gain"] * gain_mults[k]))
            trial["bias_penalty"] = max(0.0, min(1.2, base.get("bias_penalty", 0.3) * bias_mults[k]))
            trial["seq_in_len"] = seq_choices[(idx + (base["seq_in_len"] // 12)) % len(seq_choices)]
            expanded.append(trial)

    return expanded


def value_for_trial(trial, key, default):
    return trial[key] if key in trial else default


def parse_metrics(output_text: str):
    final_rse = None
    final_rae = None
    best_test_rse = None
    per_target_rrse = None

    m_final = re.findall(r"final test rse\s+([0-9.]+)\s*\|\s*test rae\s+([0-9.]+)", output_text)
    if m_final:
        final_rse = float(m_final[-1][0])
        final_rae = float(m_final[-1][1])

    m_best = re.findall(r"best test rse=\s*([0-9.]+)", output_text)
    if m_best:
        best_test_rse = float(m_best[-1])

    m_targets = re.findall(r"\[Testing\]\s+per_target_rrse_json=(\{.*?\})", output_text)
    if m_targets:
        try:
            per_target_rrse = json.loads(m_targets[-1])
            if not isinstance(per_target_rrse, dict):
                per_target_rrse = None
        except Exception:
            per_target_rrse = None

    return final_rse, final_rae, best_test_rse, per_target_rrse


def validate_rollout_policy(output_text: str):
    if "rollout_mode='teacher_forced'" in output_text or 'rollout_mode="teacher_forced"' in output_text:
        return False
    return True


def run_once(py_exec, script_path, common_args, trial, seed, run_dir, run_id, data_path):
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"model_{run_id:03d}.pt"

    focus_gain = value_for_trial(trial, "focus_target_gain", common_args.focus_target_gain)
    focus_gain = float(focus_gain) * float(common_args.focus_gain_scale)
    focus_gain = max(1.0, min(300.0, focus_gain))

    bias_lambda = value_for_trial(trial, "bias_penalty", common_args.bias_penalty)
    bias_lambda = float(bias_lambda) * float(common_args.bias_penalty_scale)
    bias_lambda = max(0.0, min(2.0, bias_lambda))

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
        "--focus_targets", str(common_args.focus_targets),
        "--focus_nodes", common_args.focus_nodes,
        "--focus_weight", str(common_args.focus_weight),
        "--focus_target_gain", str(focus_gain),
        "--focus_only_loss", str(common_args.focus_only_loss),
        "--anchor_focus_to_last", str(value_for_trial(trial, "anchor_focus_to_last", common_args.anchor_focus_to_last)),
        "--rse_targets", common_args.rse_targets,
        "--rse_report_mode", common_args.rse_report_mode,
        "--loss_mode", common_args.loss_mode,
        "--target_profile", common_args.target_profile,
        "--bias_penalty", str(bias_lambda),
        "--bias_penalty_scope", common_args.bias_penalty_scope,
        "--debias_mode", common_args.debias_mode,
        "--debias_apply_to", common_args.debias_apply_to,
        "--debug_eval", "0",
        "--rollout_mode", common_args.rollout_mode,
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
        "--subgraph_size", str(value_for_trial(trial, "subgraph_size", 40)),
        "--node_dim", str(value_for_trial(trial, "node_dim", 30)),
        "--seq_in_len", str(trial["seq_in_len"]),
        "--seq_out_len", str(trial["seq_out_len"]),
        "--ss_prob", str(trial["ss_prob"]),
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output = proc.stdout

    policy_ok = True
    if int(common_args.strict_no_teacher_forced) == 1:
        policy_ok = validate_rollout_policy(output)
        if not policy_ok:
            output += "\n[policy_check] FAIL: teacher_forced detected in run output.\n"

    log_path = run_dir / f"run_{run_id:03d}.log"
    log_path.write_text(output, encoding="utf-8")

    final_rse, final_rae, best_test_rse, per_target_rrse = parse_metrics(output)
    max_target_rse = None
    if isinstance(per_target_rrse, dict) and len(per_target_rrse) > 0:
        vals = [float(v) for v in per_target_rrse.values() if v is not None]
        if vals:
            max_target_rse = max(vals)

    return {
        "run_id": run_id,
        "seed": seed,
        "data": str(data_path),
        "use_graph": 0,
        "trial": trial,
        "return_code": proc.returncode if policy_ok else 2,
        "policy_ok": policy_ok,
        "final_test_rse": final_rse,
        "final_test_rae": final_rae,
        "best_test_rse": best_test_rse,
        "per_target_rrse": per_target_rrse,
        "max_target_rse": max_target_rse,
        "log_file": str(log_path),
    }


def write_results_json_csv(out_dir: Path, results):
    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "seed", "data", "use_graph", "final_test_rse", "final_test_rae", "best_test_rse", "max_target_rse", "objective_rse", "return_code", "policy_ok", "trial_json", "per_target_rrse_json", "log_file"])
        for r in results:
            if r.get("max_target_rse") is not None:
                obj = r["max_target_rse"]
            else:
                obj = r["best_test_rse"] if r["best_test_rse"] is not None else r["final_test_rse"]
            w.writerow([
                r["run_id"],
                r["seed"],
                r["data"],
                r["use_graph"],
                r["final_test_rse"],
                r["final_test_rae"],
                r["best_test_rse"],
                r.get("max_target_rse"),
                obj,
                r["return_code"],
                r.get("policy_ok", True),
                json.dumps(r["trial"], ensure_ascii=False),
                json.dumps(r.get("per_target_rrse"), ensure_ascii=False),
                r["log_file"],
            ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--py", type=str, default="/Users/samrobert/Documents/GitHub/Exchangeprediction/.venv/bin/python")
    parser.add_argument("--epochs", type=int, default=320)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--train_ratio", type=float, default=0.8666666667)
    parser.add_argument("--valid_ratio", type=float, default=0.0666666667)
    parser.add_argument("--focus_targets", type=int, default=1)
    parser.add_argument("--focus_nodes", type=str, default="us_Trade Weighted Dollar Index,jp_fx,kr_fx")
    parser.add_argument("--focus_weight", type=float, default=1.0)
    parser.add_argument("--focus_target_gain", type=float, default=40.0)
    parser.add_argument("--focus_gain_scale", type=float, default=1.0, help="global multiplier for trial focus_target_gain")
    parser.add_argument("--focus_only_loss", type=int, default=1, choices=[0, 1])
    parser.add_argument("--anchor_focus_to_last", type=float, default=0.1)
    parser.add_argument("--rse_targets", type=str, default="Us_Trade Weighted Dollar Index_Testing.txt,Jp_fx_Testing.txt,Kr_fx_Testing.txt")
    parser.add_argument("--rse_report_mode", type=str, default="targets", choices=["targets", "all"])
    parser.add_argument("--loss_mode", type=str, default="mse", choices=["l1", "mse"])
    parser.add_argument("--target_profile", type=str, default="none", choices=["none", "triple_050", "run001_us"])
    parser.add_argument("--bias_penalty", type=float, default=0.3)
    parser.add_argument("--bias_penalty_scale", type=float, default=1.0, help="global multiplier for trial bias_penalty")
    parser.add_argument("--bias_penalty_scope", type=str, default="focus", choices=["focus", "all"])
    parser.add_argument("--debias_mode", type=str, default="val_mean_error", choices=["none", "val_mean_error", "val_affine"])
    parser.add_argument("--debias_apply_to", type=str, default="focus", choices=["focus", "all"])
    parser.add_argument("--goal_rse", type=float, default=0.95, help="stop sweep early when max target RSE (US/JP/KR) <= goal")
    parser.add_argument("--rollout_mode", type=str, default="recursive", choices=["recursive"])
    parser.add_argument("--seeds", type=str, default="777")
    parser.add_argument("--max_trials", type=int, default=240)
    parser.add_argument("--max_runtime_minutes", type=int, default=0, help="wall-clock cap for sweep; <=0 disables")
    parser.add_argument("--trial_expand_factor", type=int, default=20, help="expand base trial set by deterministic perturbations")
    parser.add_argument("--datasets", type=str, default="sm_data.csv")
    parser.add_argument("--resume_dir", type=str, default="", help="existing tuning_runs/<timestamp> directory to resume from")
    parser.add_argument("--strict_no_teacher_forced", type=int, default=1, choices=[0, 1], help="1: fail a run if output indicates teacher_forced rollout")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    train_script = root / "train_test.py"

    if args.resume_dir.strip():
        out_dir = Path(args.resume_dir).expanduser().resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = root / "tuning_runs" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    trials = build_trials(args.trial_expand_factor)[: args.max_trials]
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
        print(f"[{idx}/{total}] done final_test_rse={res['final_test_rse']} best_test_rse={res['best_test_rse']} max_target_rse={res.get('max_target_rse')}")

        obj = res.get("max_target_rse")
        if obj is None:
            obj = res["best_test_rse"] if res["best_test_rse"] is not None else res["final_test_rse"]
        if obj is not None and obj <= args.goal_rse:
            print(f"goal reached: objective_rse={obj} <= goal_rse={args.goal_rse}. stopping early.")
            break

        run_id += 1

    valid = [r for r in results if (r["best_test_rse"] is not None or r["final_test_rse"] is not None)]
    def objective(row):
        if row.get("max_target_rse") is not None:
            return row["max_target_rse"]
        return row["best_test_rse"] if row["best_test_rse"] is not None else row["final_test_rse"]
    valid_sorted = sorted(valid, key=objective) if valid else []

    write_results_json_csv(out_dir, results)

    summary_path = out_dir / "best_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        if valid_sorted:
            best = valid_sorted[0]
            best_obj = objective(best)
            f.write(f"best_objective_rse={best_obj}\n")
            f.write(f"best_final_test_rse={best['final_test_rse']}\n")
            f.write(f"best_best_test_rse={best['best_test_rse']}\n")
            f.write(f"best_max_target_rse={best.get('max_target_rse')}\n")
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
        best_obj = objective(best)
        print("best_objective_rse:", best_obj)
        print("best_final_test_rse:", best["final_test_rse"])
        print("best_best_test_rse:", best["best_test_rse"])
        print("best_max_target_rse:", best.get("max_target_rse"))
        print("best_seed:", best["seed"])
        print("best_data:", best["data"])
        print("best_use_graph:", best["use_graph"])
        print("best_trial:", best["trial"])
    else:
        print("No valid parsed runs.")


if __name__ == "__main__":
    main()

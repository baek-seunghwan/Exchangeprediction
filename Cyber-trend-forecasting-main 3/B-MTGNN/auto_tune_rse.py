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
        {"lr": 1.5e-4, "dropout": 0.00, "layers": 3, "conv_channels": 32, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "subgraph_size": 20, "node_dim": 50, "seq_in_len": 36, "seq_out_len": 1, "ss_prob": 0.10, "focus_target_gain": 100.0, "anchor_focus_to_last": 0.0, "bias_penalty": 0.60, "lag_penalty_1step": 0.8, "lag_sign_penalty": 0.3},
        {"lr": 1.2e-4, "dropout": 0.00, "layers": 3, "conv_channels": 32, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "subgraph_size": 20, "node_dim": 50, "seq_in_len": 48, "seq_out_len": 1, "ss_prob": 0.08, "focus_target_gain": 110.0, "anchor_focus_to_last": 0.0, "bias_penalty": 0.65, "lag_penalty_1step": 1.0, "lag_sign_penalty": 0.4},
        {"lr": 1.0e-4, "dropout": 0.00, "layers": 3, "conv_channels": 32, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "subgraph_size": 20, "node_dim": 50, "seq_in_len": 60, "seq_out_len": 1, "ss_prob": 0.05, "focus_target_gain": 120.0, "anchor_focus_to_last": 0.05, "bias_penalty": 0.70, "lag_penalty_1step": 1.2, "lag_sign_penalty": 0.5},
        {"lr": 8.0e-5, "dropout": 0.00, "layers": 3, "conv_channels": 32, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "subgraph_size": 20, "node_dim": 50, "seq_in_len": 60, "seq_out_len": 1, "ss_prob": 0.05, "focus_target_gain": 140.0, "anchor_focus_to_last": 0.10, "bias_penalty": 0.80, "lag_penalty_1step": 1.4, "lag_sign_penalty": 0.6},
        {"lr": 1.8e-4, "dropout": 0.02, "layers": 3, "conv_channels": 32, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "subgraph_size": 30, "node_dim": 40, "seq_in_len": 36, "seq_out_len": 1, "ss_prob": 0.10, "focus_target_gain": 90.0, "anchor_focus_to_last": 0.10, "bias_penalty": 0.55},
        {"lr": 1.5e-4, "dropout": 0.02, "layers": 3, "conv_channels": 32, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "subgraph_size": 30, "node_dim": 40, "seq_in_len": 48, "seq_out_len": 1, "ss_prob": 0.08, "focus_target_gain": 105.0, "anchor_focus_to_last": 0.15, "bias_penalty": 0.65},
        {"lr": 1.2e-4, "dropout": 0.02, "layers": 3, "conv_channels": 32, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "subgraph_size": 30, "node_dim": 40, "seq_in_len": 60, "seq_out_len": 1, "ss_prob": 0.05, "focus_target_gain": 120.0, "anchor_focus_to_last": 0.20, "bias_penalty": 0.75},
        {"lr": 2.0e-4, "dropout": 0.05, "layers": 2, "conv_channels": 16, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "subgraph_size": 40, "node_dim": 30, "seq_in_len": 24, "seq_out_len": 1, "ss_prob": 0.20, "focus_target_gain": 80.0, "anchor_focus_to_last": 0.0, "bias_penalty": 0.40},
        {"lr": 1.7e-4, "dropout": 0.03, "layers": 2, "conv_channels": 16, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "subgraph_size": 40, "node_dim": 30, "seq_in_len": 36, "seq_out_len": 1, "ss_prob": 0.15, "focus_target_gain": 90.0, "anchor_focus_to_last": 0.05, "bias_penalty": 0.50},
        {"lr": 1.4e-4, "dropout": 0.03, "layers": 2, "conv_channels": 16, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "subgraph_size": 40, "node_dim": 30, "seq_in_len": 48, "seq_out_len": 1, "ss_prob": 0.10, "focus_target_gain": 100.0, "anchor_focus_to_last": 0.10, "bias_penalty": 0.60},
        {"lr": 1.1e-4, "dropout": 0.03, "layers": 2, "conv_channels": 16, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "subgraph_size": 40, "node_dim": 30, "seq_in_len": 60, "seq_out_len": 1, "ss_prob": 0.06, "focus_target_gain": 115.0, "anchor_focus_to_last": 0.15, "bias_penalty": 0.70},
        {"lr": 1.6e-4, "dropout": 0.01, "layers": 3, "conv_channels": 32, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024, "subgraph_size": 20, "node_dim": 50, "seq_in_len": 36, "seq_out_len": 1, "ss_prob": 0.10, "focus_target_gain": 130.0, "anchor_focus_to_last": 0.20, "bias_penalty": 0.85},
    ]

    if expand_factor <= 1:
        return base_trials

    lr_mults = [0.85, 1.00, 1.15, 0.70, 1.30]
    drop_shifts = [0.00, -0.01, 0.01, -0.02, 0.02]
    ss_shifts = [0.00, -0.03, 0.03, -0.05, 0.05]
    anchor_shifts = [0.00, -0.02, 0.02, -0.04, 0.04]
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
            trial["anchor_focus_to_last"] = max(0.0, min(0.30, base["anchor_focus_to_last"] + anchor_shifts[k]))
            trial["focus_target_gain"] = max(20.0, min(140.0, base["focus_target_gain"] * gain_mults[k]))
            trial["bias_penalty"] = max(0.0, min(1.2, base.get("bias_penalty", 0.3) * bias_mults[k]))
            trial["seq_in_len"] = seq_choices[(idx + (base["seq_in_len"] // 12)) % len(seq_choices)]
            expanded.append(trial)

    return expanded


def unique_trials(trials):
    out = []
    seen = set()
    for t in trials:
        key = json.dumps(t, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def load_elite_trials_from_csv_dirs(root_dir: Path, csv_dirs, top_k=6):
    rows = []
    for d in csv_dirs:
        cpath = (root_dir / "tuning_runs" / d / "results.csv").resolve()
        if not cpath.exists():
            continue
        try:
            with cpath.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    trial = None
                    tj = row.get("trial_json")
                    if tj:
                        try:
                            parsed = json.loads(tj)
                            if isinstance(parsed, dict):
                                trial = parsed
                        except Exception:
                            trial = None
                    if trial is None:
                        continue

                    try:
                        max_target_rse = float(row.get("max_target_rse", "nan"))
                    except Exception:
                        max_target_rse = float("inf")
                    try:
                        best_test_rse = float(row.get("best_test_rse", "nan"))
                    except Exception:
                        best_test_rse = float("inf")

                    rows.append(
                        {
                            "trial": trial,
                            "max_target_rse": max_target_rse,
                            "best_test_rse": best_test_rse,
                            "source": d,
                        }
                    )
        except Exception:
            continue

    if not rows:
        return []

    rows.sort(key=lambda r: (r["max_target_rse"], r["best_test_rse"]))

    picked = []
    seen = set()
    for r in rows:
        t = r["trial"]
        key = json.dumps(t, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)

        t2 = dict(t)
        t2["anchor_focus_to_last"] = max(0.0, min(0.20, float(t2.get("anchor_focus_to_last", 0.0))))
        t2["focus_target_gain"] = max(45.0, min(120.0, float(t2.get("focus_target_gain", 80.0))))
        t2["bias_penalty"] = max(0.20, min(0.80, float(t2.get("bias_penalty", 0.4))))
        t2["dropout"] = max(0.0, min(0.08, float(t2.get("dropout", 0.02))))
        t2["lr"] = max(8e-5, min(2.2e-4, float(t2.get("lr", 1.2e-4))))

        if "lag_penalty_1step" not in t2:
            t2["lag_penalty_1step"] = 1.0
        if "lag_sign_penalty" not in t2:
            t2["lag_sign_penalty"] = 0.4

        picked.append(t2)
        if len(picked) >= int(top_k):
            break

    return picked


def value_for_trial(trial, key, default):
    return trial[key] if key in trial else default


def row_objective(row):
    if row.get("objective_total") is not None:
        return float(row["objective_total"])
    if row.get("objective_rse") is not None:
        return float(row["objective_rse"])
    if row.get("max_target_rse") is not None:
        return float(row["max_target_rse"])
    if row.get("best_test_rse") is not None:
        return float(row["best_test_rse"])
    if row.get("final_test_rse") is not None:
        return float(row["final_test_rse"])
    return float("inf")


def generate_recovery_trials(best_trial, phase=1, max_items=6):
    base = dict(best_trial)

    lr = float(base.get("lr", 1.5e-4))
    dropout = float(base.get("dropout", 0.0))
    ss_prob = float(base.get("ss_prob", 0.10))
    gain = float(base.get("focus_target_gain", 80.0))
    bias = float(base.get("bias_penalty", 0.5))
    seq_in_len = int(base.get("seq_in_len", 36))
    anchor = float(base.get("anchor_focus_to_last", 0.0))

    if phase <= 1:
        lr_mults = [0.85, 1.00, 1.15]
        gain_mults = [1.05, 1.15]
        bias_mults = [1.0, 1.2]
    else:
        lr_mults = [0.70, 0.90, 1.10, 1.30]
        gain_mults = [1.10, 1.25, 1.35]
        bias_mults = [1.0, 1.3]

    seq_candidates = sorted(set([24, 36, 48, 60, seq_in_len]))
    trials = []
    seen = set()

    for lr_m in lr_mults:
        for g_m in gain_mults:
            for b_m in bias_mults:
                for seq in seq_candidates:
                    t = dict(base)
                    t["lr"] = max(1e-5, min(5e-4, lr * lr_m))
                    t["focus_target_gain"] = max(40.0, min(180.0, gain * g_m))
                    t["bias_penalty"] = max(0.0, min(1.5, bias * b_m))
                    t["dropout"] = max(0.0, min(0.1, dropout))
                    t["ss_prob"] = max(0.0, min(0.25, ss_prob))
                    t["anchor_focus_to_last"] = max(0.0, min(0.20, anchor))
                    t["seq_in_len"] = int(seq)
                    key = json.dumps(t, sort_keys=True)
                    if key in seen:
                        continue
                    seen.add(key)
                    trials.append(t)
                    if len(trials) >= max_items:
                        return trials
    return trials


def parse_metrics(output_text: str):
    final_rse = None
    final_rae = None
    best_test_rse = None
    per_target_rrse = None
    per_target_lag_mae = None
    per_target_dir_mismatch = None

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

    m_lag = re.findall(r"\[Testing\]\s+per_target_lag_mae_json=(\{.*?\})", output_text)
    if m_lag:
        try:
            per_target_lag_mae = json.loads(m_lag[-1])
            if not isinstance(per_target_lag_mae, dict):
                per_target_lag_mae = None
        except Exception:
            per_target_lag_mae = None

    m_dir = re.findall(r"\[Testing\]\s+per_target_dir_mismatch_json=(\{.*?\})", output_text)
    if m_dir:
        try:
            per_target_dir_mismatch = json.loads(m_dir[-1])
            if not isinstance(per_target_dir_mismatch, dict):
                per_target_dir_mismatch = None
        except Exception:
            per_target_dir_mismatch = None

    return final_rse, final_rae, best_test_rse, per_target_rrse, per_target_lag_mae, per_target_dir_mismatch


def validate_rollout_policy(output_text: str):
    if "rollout_mode='teacher_forced'" in output_text or 'rollout_mode="teacher_forced"' in output_text:
        return False
    return True


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
        "--bias_penalty", str(value_for_trial(trial, "bias_penalty", common_args.bias_penalty)),
        "--bias_penalty_scope", common_args.bias_penalty_scope,
        "--debias_mode", common_args.debias_mode,
        "--debias_apply_to", common_args.debias_apply_to,
        "--lag_penalty_1step", str(value_for_trial(trial, "lag_penalty_1step", common_args.lag_penalty_1step)),
        "--lag_sign_penalty", str(value_for_trial(trial, "lag_sign_penalty", common_args.lag_sign_penalty)),
        "--debug_eval", "0",
        "--rollout_mode", common_args.rollout_mode,
        "--enforce_cutoff_split", "1",
        "--cutoff_year_yy", "25",
        "--min_valid_months", "12",
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

    final_rse, final_rae, best_test_rse, per_target_rrse, per_target_lag_mae, per_target_dir_mismatch = parse_metrics(output)
    max_target_rse = None
    if isinstance(per_target_rrse, dict) and len(per_target_rrse) > 0:
        vals = [float(v) for v in per_target_rrse.values() if v is not None]
        if vals:
            max_target_rse = max(vals)

    max_target_lag_mae = None
    if isinstance(per_target_lag_mae, dict) and len(per_target_lag_mae) > 0:
        vals = [float(v) for v in per_target_lag_mae.values() if v is not None]
        if vals:
            max_target_lag_mae = max(vals)

    max_target_dir_mismatch = None
    if isinstance(per_target_dir_mismatch, dict) and len(per_target_dir_mismatch) > 0:
        vals = [float(v) for v in per_target_dir_mismatch.values() if v is not None]
        if vals:
            max_target_dir_mismatch = max(vals)

    objective_rse = max_target_rse
    if objective_rse is None:
        objective_rse = best_test_rse if best_test_rse is not None else final_rse

    lag_penalty = 0.0
    if max_target_lag_mae is not None:
        lag_penalty += float(common_args.lag_penalty_weight) * max_target_lag_mae
    if max_target_dir_mismatch is not None:
        lag_penalty += float(common_args.dir_mismatch_weight) * max_target_dir_mismatch

    goal_shortfall_penalty = 0.0
    goal_val = float(common_args.goal_rse)
    if objective_rse is not None and objective_rse > goal_val:
        goal_shortfall_penalty = float(common_args.goal_shortfall_weight) * ((objective_rse - goal_val) ** 2)

    objective_total = None if objective_rse is None else (objective_rse + lag_penalty + goal_shortfall_penalty)

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
        "per_target_lag_mae": per_target_lag_mae,
        "max_target_lag_mae": max_target_lag_mae,
        "per_target_dir_mismatch": per_target_dir_mismatch,
        "max_target_dir_mismatch": max_target_dir_mismatch,
        "objective_rse": objective_rse,
        "objective_total": objective_total,
        "log_file": str(log_path),
    }


def write_results_json_csv(out_dir: Path, results):
    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "seed", "data", "use_graph", "final_test_rse", "final_test_rae", "best_test_rse", "max_target_rse", "max_target_lag_mae", "max_target_dir_mismatch", "objective_rse", "objective_total", "return_code", "policy_ok", "trial_json", "per_target_rrse_json", "per_target_lag_mae_json", "per_target_dir_mismatch_json", "log_file"])
        for r in results:
            obj_rse = r.get("objective_rse")
            if obj_rse is None:
                if r.get("max_target_rse") is not None:
                    obj_rse = r["max_target_rse"]
                else:
                    obj_rse = r["best_test_rse"] if r["best_test_rse"] is not None else r["final_test_rse"]
            w.writerow([
                r["run_id"],
                r["seed"],
                r["data"],
                r["use_graph"],
                r["final_test_rse"],
                r["final_test_rae"],
                r["best_test_rse"],
                r.get("max_target_rse"),
                r.get("max_target_lag_mae"),
                r.get("max_target_dir_mismatch"),
                obj_rse,
                r.get("objective_total"),
                r["return_code"],
                r.get("policy_ok", True),
                json.dumps(r["trial"], ensure_ascii=False),
                json.dumps(r.get("per_target_rrse"), ensure_ascii=False),
                json.dumps(r.get("per_target_lag_mae"), ensure_ascii=False),
                json.dumps(r.get("per_target_dir_mismatch"), ensure_ascii=False),
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
    parser.add_argument("--focus_only_loss", type=int, default=1, choices=[0, 1])
    parser.add_argument("--anchor_focus_to_last", type=float, default=0.1)
    parser.add_argument("--rse_targets", type=str, default="Us_Trade Weighted Dollar Index_Testing.txt,Jp_fx_Testing.txt,Kr_fx_Testing.txt")
    parser.add_argument("--rse_report_mode", type=str, default="targets", choices=["targets", "all"])
    parser.add_argument("--loss_mode", type=str, default="mse", choices=["l1", "mse"])
    parser.add_argument("--target_profile", type=str, default="none", choices=["none", "triple_050", "run001_us"])
    parser.add_argument("--bias_penalty", type=float, default=0.3)
    parser.add_argument("--bias_penalty_scope", type=str, default="focus", choices=["focus", "all"])
    parser.add_argument("--lag_penalty_1step", type=float, default=0.8)
    parser.add_argument("--lag_sign_penalty", type=float, default=0.3)
    parser.add_argument("--debias_mode", type=str, default="val_mean_error", choices=["none", "val_mean_error", "val_affine"])
    parser.add_argument("--debias_apply_to", type=str, default="focus", choices=["focus", "all"])
    parser.add_argument("--goal_rse", type=float, default=0.5, help="stop sweep early when max target RSE (US/JP/KR) <= goal")
    parser.add_argument("--lag_penalty_weight", type=float, default=0.25, help="weight for max target lag-mae penalty in tuning objective")
    parser.add_argument("--dir_mismatch_weight", type=float, default=0.10, help="weight for max target direction-mismatch penalty in tuning objective")
    parser.add_argument("--goal_shortfall_weight", type=float, default=6.0, help="extra penalty weight for (max_target_rse-goal_rse)^2 when goal is not met")
    parser.add_argument("--rollout_mode", type=str, default="recursive", choices=["recursive"])
    parser.add_argument("--seeds", type=str, default="777")
    parser.add_argument("--max_trials", type=int, default=240)
    parser.add_argument("--max_runtime_minutes", type=int, default=0, help="wall-clock cap for sweep; <=0 disables")
    parser.add_argument("--trial_expand_factor", type=int, default=20, help="expand base trial set by deterministic perturbations")
    parser.add_argument("--datasets", type=str, default="sm_data.csv")
    parser.add_argument("--resume_dir", type=str, default="", help="existing tuning_runs/<timestamp> directory to resume from")
    parser.add_argument("--strict_no_teacher_forced", type=int, default=1, choices=[0, 1], help="1: fail a run if output indicates teacher_forced rollout")
    parser.add_argument("--stagnation_patience", type=int, default=8, help="trigger recovery runs when no objective improvement for this many runs")
    parser.add_argument("--stagnation_min_delta", type=float, default=1e-3, help="minimum objective improvement considered meaningful")
    parser.add_argument("--stagnation_recovery_rounds", type=int, default=2, help="maximum number of automatic recovery rounds")
    parser.add_argument("--stagnation_recovery_trials", type=int, default=6, help="number of generated recovery trials per round")
    parser.add_argument("--stagnation_recovery_budget", type=int, default=18, help="maximum total recovery runs allowed")
    parser.add_argument("--elite_from_csv_dirs", type=str, default="20260220_153322,20260220_154634,20260220_175117,20260220_181241,20260220_183544,20260220_195447", help="comma-separated tuning_runs subdirs to mine elite trials from results.csv")
    parser.add_argument("--elite_top_k", type=int, default=8, help="number of elite trials to prepend from prior csv results")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    train_script = root / "train_test.py"

    if args.resume_dir.strip():
        out_dir = Path(args.resume_dir).expanduser().resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = root / "tuning_runs" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_trials = build_trials(args.trial_expand_factor)
    elite_dirs = [x.strip() for x in args.elite_from_csv_dirs.split(",") if x.strip()]
    elite_trials = load_elite_trials_from_csv_dirs(root, elite_dirs, top_k=args.elite_top_k)

    if elite_trials:
        print(f"elite trials loaded: {len(elite_trials)} from {elite_dirs}")
    else:
        print("elite trials loaded: 0 (fallback to built-in trials)")

    trials = unique_trials(elite_trials + seed_trials)[: args.max_trials]
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
    best_objective_so_far = float("inf")
    no_improve_runs = 0
    recovery_round = 0
    recovery_runs_used = 0

    for row in results:
        o = row_objective(row)
        if o < best_objective_so_far:
            best_objective_so_far = o

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
        print(f"[{idx}/{total}] done final_test_rse={res['final_test_rse']} best_test_rse={res['best_test_rse']} max_target_rse={res.get('max_target_rse')} max_target_lag_mae={res.get('max_target_lag_mae')} objective_total={res.get('objective_total')}")

        cur_obj = row_objective(res)
        if cur_obj + float(args.stagnation_min_delta) < best_objective_so_far:
            best_objective_so_far = cur_obj
            no_improve_runs = 0
        else:
            no_improve_runs += 1

        goal_obj = res.get("objective_rse")
        if goal_obj is None:
            goal_obj = res.get("max_target_rse")
        if goal_obj is None:
            goal_obj = res["best_test_rse"] if res["best_test_rse"] is not None else res["final_test_rse"]
        if goal_obj is not None and goal_obj <= args.goal_rse:
            print(f"goal reached: objective_rse={goal_obj} <= goal_rse={args.goal_rse}. stopping early.")
            break

        if (
            args.stagnation_patience > 0
            and no_improve_runs >= args.stagnation_patience
            and recovery_round < args.stagnation_recovery_rounds
            and recovery_runs_used < args.stagnation_recovery_budget
        ):
            valid_now = [r for r in results if (r.get("best_test_rse") is not None or r.get("final_test_rse") is not None)]
            if valid_now:
                best_now = sorted(valid_now, key=row_objective)[0]
                recovery_round += 1
                no_improve_runs = 0
                candidates = generate_recovery_trials(
                    best_now["trial"],
                    phase=recovery_round,
                    max_items=args.stagnation_recovery_trials,
                )

                if candidates:
                    print(
                        f"[stagnation] no improvement for {args.stagnation_patience} runs. "
                        f"launching recovery_round={recovery_round} with {len(candidates)} trials"
                    )

                for ridx, rtrial in enumerate(candidates, 1):
                    if recovery_runs_used >= args.stagnation_recovery_budget:
                        break
                    if args.max_runtime_minutes > 0:
                        elapsed_min = (time.time() - sweep_start) / 60.0
                        if elapsed_min >= args.max_runtime_minutes:
                            print("[stagnation] stop recovery due to time budget")
                            break

                    rkey = (str(root / "data" / dataset), int(seed), json.dumps(rtrial, sort_keys=True))
                    if rkey in done_keys:
                        continue

                    print(f"[recovery {recovery_round}:{ridx}] data={dataset} seed={seed} start")
                    try:
                        rres = run_once(args.py, train_script, args, rtrial, seed, out_dir, run_id, data_path)
                    except KeyboardInterrupt:
                        print("Interrupted by user during recovery. Partial results are saved.")
                        write_results_json_csv(out_dir, results)
                        raise

                    results.append(rres)
                    done_keys.add(rkey)
                    write_results_json_csv(out_dir, results)
                    recovery_runs_used += 1
                    run_id += 1

                    robj = row_objective(rres)
                    if robj + float(args.stagnation_min_delta) < best_objective_so_far:
                        best_objective_so_far = robj
                        no_improve_runs = 0

                    print(
                        f"[recovery {recovery_round}:{ridx}] done "
                        f"best_test_rse={rres.get('best_test_rse')} max_target_rse={rres.get('max_target_rse')} "
                        f"objective_total={rres.get('objective_total')}"
                    )

                    r_goal = rres.get("objective_rse")
                    if r_goal is None:
                        r_goal = rres.get("max_target_rse")
                    if r_goal is None:
                        r_goal = rres.get("best_test_rse") if rres.get("best_test_rse") is not None else rres.get("final_test_rse")
                    if r_goal is not None and r_goal <= args.goal_rse:
                        print(f"goal reached during recovery: objective_rse={r_goal} <= goal_rse={args.goal_rse}. stopping early.")
                        break

                if any(
                    (rr.get("objective_rse") if rr.get("objective_rse") is not None else rr.get("max_target_rse")) is not None
                    and ((rr.get("objective_rse") if rr.get("objective_rse") is not None else rr.get("max_target_rse")) <= args.goal_rse)
                    for rr in results
                ):
                    break

        run_id += 1

    valid = [r for r in results if (r["best_test_rse"] is not None or r["final_test_rse"] is not None)]
    def objective(row):
        if row.get("objective_total") is not None:
            return row["objective_total"]
        if row.get("objective_rse") is not None:
            return row["objective_rse"]
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

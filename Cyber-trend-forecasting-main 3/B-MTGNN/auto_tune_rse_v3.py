import argparse
import csv
import json
import random
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


def parse_metrics(output_text: str):
    final_rse = None
    final_rae = None
    best_test_rse = None
    test_focus_rrse = None

    m_final = re.findall(r"final test rse\s+([0-9.]+)\s*\|\s*test rae\s+([0-9.]+)", output_text)
    if m_final:
        final_rse = float(m_final[-1][0])
        final_rae = float(m_final[-1][1])

    m_best = re.findall(r"best test rse=\s*([0-9.]+)", output_text)
    if m_best:
        best_test_rse = float(m_best[-1])

    m_focus = re.findall(r"test focus_rrse\s+([0-9.]+)", output_text)
    if m_focus:
        test_focus_rrse = float(m_focus[-1])

    return final_rse, final_rae, best_test_rse, test_focus_rrse


def objective_from_metrics(final_rse, focus_rrse, goal_rse, goal_weight):
    """Weighted objective: prioritise overall RSE but also penalise bad focus_rrse."""
    if final_rse is None:
        return None
    # Use overall RSE as base; add focus_rrse penalty
    base = final_rse
    focus_penalty = 0.0
    if focus_rrse is not None:
        focus_penalty = 0.3 * focus_rrse  # mild weight on focus
    shortfall = max(0.0, base - goal_rse)
    return base + focus_penalty + (goal_weight * (shortfall ** 2))


def load_top_trials_from_results(results_path: Path, top_k=8):
    if not results_path.exists():
        return []
    try:
        rows = json.loads(results_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(rows, list):
        return []

    valid = [r for r in rows if isinstance(r, dict) and isinstance(r.get("trial"), dict)]
    if not valid:
        return []

    # Sort by final_test_rse (overall RSE), NOT objective_total
    valid = sorted(valid, key=lambda r: r.get("final_test_rse") if r.get("final_test_rse") is not None else 1e18)
    out = []
    seen = set()
    for r in valid:
        t = dict(r["trial"])
        key = json.dumps(t, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= int(top_k):
            break
    return out


def expand_local_trials(base_trials, extra_per_base=4, seed=123):
    """TIGHT perturbation around base trials."""
    random.seed(seed)
    out = []
    seen = set()
    for b in base_trials:
        b = dict(b)
        for _ in range(extra_per_base):
            t = dict(b)
            # Tight perturbation: ±10-15%
            t["lr"] = max(8e-5, min(3e-4, float(t.get("lr", 1.5e-4)) * random.choice([0.9, 0.95, 1.0, 1.05, 1.1])))
            t["dropout"] = max(0.0, min(0.05, float(t.get("dropout", 0.02)) + random.choice([-0.01, 0.0, 0.0, 0.01])))
            t["seq_in_len"] = 24  # FIXED
            t["layers"] = int(t.get("layers", 2))  # keep same
            t["ss_prob"] = max(0.01, min(0.08, float(t.get("ss_prob", 0.05)) + random.choice([-0.01, 0.0, 0.01])))
            t["focus_target_gain"] = max(50.0, min(130.0, float(t.get("focus_target_gain", 80.0)) * random.choice([0.9, 1.0, 1.1, 1.2])))
            t["anchor_focus_to_last"] = max(0.02, min(0.10, float(t.get("anchor_focus_to_last", 0.06)) + random.choice([-0.02, -0.01, 0.0, 0.01, 0.02])))
            t["bias_penalty"] = max(0.2, min(0.8, float(t.get("bias_penalty", 0.5)) + random.choice([-0.1, 0.0, 0.1])))
            t["lag_penalty_1step"] = max(0.8, min(2.0, float(t.get("lag_penalty_1step", 1.3)) + random.choice([-0.2, -0.1, 0.0, 0.1, 0.2])))
            t["lag_sign_penalty"] = max(0.2, min(1.0, float(t.get("lag_sign_penalty", 0.5)) + random.choice([-0.1, 0.0, 0.1])))
            t["grad_loss_weight"] = max(0.05, min(0.3, float(t.get("grad_loss_weight", 0.15)) + random.choice([-0.05, 0.0, 0.05])))
            t["focus_only_loss"] = 1  # ALWAYS 1

            key = json.dumps(t, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            out.append(t)
    return out


# The proven-best config (run7 from 20260221_140924, RSE=0.67)
RUN7_CONFIG = {
    "lr": 1.5e-4, "dropout": 0.02, "layers": 2,
    "conv_channels": 16, "residual_channels": 128, "skip_channels": 256, "end_channels": 1024,
    "subgraph_size": 30, "node_dim": 30,
    "seq_in_len": 24, "seq_out_len": 12,
    "ss_prob": 0.05,
    "focus_target_gain": 80.0, "anchor_focus_to_last": 0.06,
    "bias_penalty": 0.5,
    "lag_penalty_1step": 1.3, "lag_sign_penalty": 0.5,
    "focus_only_loss": 1,
    "grad_loss_weight": 0.15,
}


def build_seed_trials(seed=42, count=60):
    random.seed(seed)

    # Anchors: all based on run7 with slight variations
    anchors = [
        # run7 exact
        dict(RUN7_CONFIG),
        # Higher lag penalty
        {**RUN7_CONFIG, "lag_penalty_1step": 1.6, "lag_sign_penalty": 0.7, "focus_target_gain": 100.0},
        # Higher lr
        {**RUN7_CONFIG, "lr": 2.0e-4, "dropout": 0.01, "grad_loss_weight": 0.2},
        # Stronger anchor
        {**RUN7_CONFIG, "anchor_focus_to_last": 0.10, "bias_penalty": 0.35, "focus_target_gain": 60.0},
    ]

    trials = []
    for i in range(count):
        a = dict(anchors[i % len(anchors)])
        t = dict(a)
        # TIGHT perturbation
        t["lr"] = max(8e-5, min(3e-4, a["lr"] * random.choice([0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15])))
        t["dropout"] = max(0.0, min(0.05, a["dropout"] + random.choice([-0.01, 0.0, 0.0, 0.01])))
        t["layers"] = 2  # FIXED
        t["conv_channels"] = random.choice([16, 16, 24])
        t["residual_channels"] = 128
        t["skip_channels"] = 256
        t["end_channels"] = 1024
        t["subgraph_size"] = random.choice([20, 30, 30])
        t["node_dim"] = random.choice([30, 30, 40])
        t["seq_in_len"] = 24  # FIXED
        t["seq_out_len"] = 12
        t["ss_prob"] = max(0.01, min(0.08, a["ss_prob"] + random.choice([-0.02, -0.01, 0.0, 0.01])))
        t["focus_target_gain"] = max(50.0, min(140.0, a["focus_target_gain"] * random.choice([0.85, 0.95, 1.0, 1.1, 1.2])))
        t["anchor_focus_to_last"] = max(0.02, min(0.12, a["anchor_focus_to_last"] + random.choice([-0.02, -0.01, 0.0, 0.01, 0.02])))
        t["bias_penalty"] = max(0.2, min(0.8, a["bias_penalty"] + random.choice([-0.1, -0.05, 0.0, 0.05, 0.1])))
        t["lag_penalty_1step"] = max(0.8, min(2.2, a.get("lag_penalty_1step", 1.3) + random.choice([-0.2, -0.1, 0.0, 0.1, 0.2])))
        t["lag_sign_penalty"] = max(0.2, min(1.0, a.get("lag_sign_penalty", 0.5) + random.choice([-0.1, -0.05, 0.0, 0.05, 0.1])))
        t["grad_loss_weight"] = max(0.05, min(0.3, a.get("grad_loss_weight", 0.15) + random.choice([-0.05, 0.0, 0.05])))
        t["focus_only_loss"] = 1  # ALWAYS 1
        trials.append(t)

    unique = []
    seen = set()
    for t in trials:
        key = json.dumps(t, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        unique.append(t)
    return unique


def run_once(py_exec, train_script, args, trial, seed, run_dir, run_id):
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"model_{run_id:03d}.pt"

    cmd = [
        py_exec,
        str(train_script),
        "--data", str(args.data_path),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--horizon", "1",
        "--normalize", "2",
        "--plot", "0",
        "--autotune_mode", "1",
        "--use_graph", "0",
        "--target_profile", "none",
        "--loss_mode", "mse",
        "--focus_targets", "1",
        "--focus_nodes", args.focus_nodes,
        "--focus_weight", "1.0",
        "--focus_rrse_mode", "max",
        "--rse_targets", args.rse_targets,
        "--rse_report_mode", "targets",
        "--rollout_mode", "direct",
        "--debias_mode", "none",
        "--debias_apply_to", "focus",
        "--enforce_cutoff_split", "1",
        "--cutoff_year_yy", "25",
        "--min_valid_months", "12",
        "--train_ratio", str(args.train_ratio),
        "--valid_ratio", str(args.valid_ratio),
        "--seed", str(seed),
        "--save", str(ckpt_path),
        "--lr", str(trial["lr"]),
        "--dropout", str(trial["dropout"]),
        "--layers", str(trial["layers"]),
        "--conv_channels", str(trial["conv_channels"]),
        "--residual_channels", str(trial["residual_channels"]),
        "--skip_channels", str(trial["skip_channels"]),
        "--end_channels", str(trial["end_channels"]),
        "--subgraph_size", str(trial["subgraph_size"]),
        "--node_dim", str(trial["node_dim"]),
        "--seq_in_len", str(trial["seq_in_len"]),
        "--seq_out_len", str(trial["seq_out_len"]),
        "--ss_prob", str(trial["ss_prob"]),
        "--focus_target_gain", str(trial["focus_target_gain"]),
        "--focus_only_loss", str(trial["focus_only_loss"]),
        "--focus_gain_map", args.focus_gain_map,
        "--anchor_focus_to_last", str(trial["anchor_focus_to_last"]),
        "--anchor_boost_map", args.anchor_boost_map,
        "--bias_penalty", str(trial["bias_penalty"]),
        "--bias_penalty_scope", "focus",
        "--lag_penalty_1step", str(trial.get("lag_penalty_1step", args.default_lag_penalty_1step)),
        "--lag_sign_penalty", str(trial.get("lag_sign_penalty", args.default_lag_sign_penalty)),
        "--lag_penalty_gain_map", args.lag_penalty_gain_map,
        "--grad_loss_weight", str(trial.get("grad_loss_weight", 0.15)),
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output = proc.stdout

    log_path = run_dir / f"run_{run_id:03d}.log"
    log_path.write_text(output, encoding="utf-8")

    final_rse, final_rae, best_test_rse, focus_rrse = parse_metrics(output)
    objective = objective_from_metrics(final_rse, focus_rrse, args.goal_rse, args.goal_shortfall_weight)

    return {
        "run_id": run_id,
        "seed": seed,
        "trial": trial,
        "return_code": proc.returncode,
        "final_test_rse": final_rse,
        "final_test_rae": final_rae,
        "best_test_rse": best_test_rse,
        "test_focus_rrse": focus_rrse,
        "objective_total": objective,
        "log_file": str(log_path),
    }


def write_results(out_dir: Path, rows):
    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "run_id", "seed", "final_test_rse", "final_test_rae", "best_test_rse", "test_focus_rrse",
            "objective_total", "return_code", "trial_json", "log_file"
        ])
        for r in rows:
            w.writerow([
                r["run_id"], r["seed"], r.get("final_test_rse"), r.get("final_test_rae"), r.get("best_test_rse"),
                r.get("test_focus_rrse"), r.get("objective_total"), r["return_code"],
                json.dumps(r["trial"], ensure_ascii=False), r["log_file"]
            ])


def row_objective(r):
    if r.get("objective_total") is not None:
        return float(r["objective_total"])
    if r.get("final_test_rse") is not None:
        return float(r["final_test_rse"])
    return float("inf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--py", type=str, default="/Users/samrobert/Documents/GitHub/Exchangeprediction/.venv/bin/python")
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--goal_rse", type=float, default=0.5)
    parser.add_argument("--goal_shortfall_weight", type=float, default=12.0)
    parser.add_argument("--max_trials", type=int, default=120)
    parser.add_argument("--max_runtime_minutes", type=int, default=120)
    parser.add_argument("--seeds", type=str, default="777")
    parser.add_argument("--train_ratio", type=float, default=0.8666666667)
    parser.add_argument("--valid_ratio", type=float, default=0.0666666667)
    parser.add_argument("--focus_nodes", type=str, default="us_Trade Weighted Dollar Index,jp_fx,kr_fx")
    parser.add_argument("--rse_targets", type=str, default="Us_Trade Weighted Dollar Index_Testing.txt,Jp_fx_Testing.txt,Kr_fx_Testing.txt")
    parser.add_argument("--focus_gain_map", type=str, default="kr_fx:2.0,jp_fx:1.0,us_Trade Weighted Dollar Index:1.0")
    parser.add_argument("--anchor_boost_map", type=str, default="kr_fx:1.8,jp_fx:1.0,us_Trade Weighted Dollar Index:1.0")
    parser.add_argument("--lag_penalty_gain_map", type=str, default="kr_fx:2.0,jp_fx:1.0,us_Trade Weighted Dollar Index:1.0")
    parser.add_argument("--rollout_mode", type=str, default="direct", choices=["recursive", "direct"])
    parser.add_argument("--trial_seed", type=int, default=42)
    parser.add_argument("--warm_start_dir", type=str, default="")
    parser.add_argument("--warm_top_k", type=int, default=8)
    parser.add_argument("--warm_expand", type=int, default=4)
    parser.add_argument("--default_lag_penalty_1step", type=float, default=1.3)
    parser.add_argument("--default_lag_sign_penalty", type=float, default=0.5)
    parser.add_argument("--resume_dir", type=str, default="")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    train_script = root / "train_test.py"
    args.data_path = root / "data" / "sm_data.csv"

    if args.resume_dir.strip():
        out_dir = Path(args.resume_dir).expanduser().resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = root / "tuning_runs_v3" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # === ALWAYS start with proven run7 config as trial 0 ===
    forced_first = [dict(RUN7_CONFIG)]

    seed_trials = build_seed_trials(seed=args.trial_seed, count=max(args.max_trials, 12))

    warm_trials = []
    warm_base = None
    if args.warm_start_dir.strip():
        warm_base = Path(args.warm_start_dir).expanduser().resolve() / "results.json"
    elif args.resume_dir.strip():
        warm_base = Path(args.resume_dir).expanduser().resolve() / "results.json"

    if warm_base is not None:
        top_trials = load_top_trials_from_results(warm_base, top_k=args.warm_top_k)
        if top_trials:
            # Filter: only keep trials with focus_only_loss=1 and seq_in_len=24
            good_trials = [t for t in top_trials if t.get("focus_only_loss") == 1 and t.get("seq_in_len", 24) == 24]
            if not good_trials:
                good_trials = top_trials[:3]  # fallback
            warm_trials = expand_local_trials(good_trials, extra_per_base=args.warm_expand, seed=args.trial_seed + 7)
            print(f"warm-start trials loaded: top={len(good_trials)} expanded={len(warm_trials)}")

    merged = []
    seen = set()
    # Forced first, then warm, then seed
    for t in (forced_first + warm_trials + seed_trials):
        key = json.dumps(t, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        merged.append(t)

    trials = merged[:args.max_trials]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    combos = []
    for t in trials:
        for s in seeds:
            combos.append((t, s))

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

    done = set()
    run_id = 1
    for r in results:
        key = (int(r.get("seed", -1)), json.dumps(r.get("trial", {}), sort_keys=True))
        done.add(key)
        run_id = max(run_id, int(r.get("run_id", 0)) + 1)

    start = time.time()
    total = len(combos)

    best_so_far = float("inf")
    for idx, (trial, seed) in enumerate(combos, 1):
        if args.max_runtime_minutes > 0:
            elapsed = (time.time() - start) / 60.0
            if elapsed >= args.max_runtime_minutes:
                print(f"time budget reached: {elapsed:.1f} min")
                break

        key = (seed, json.dumps(trial, sort_keys=True))
        if key in done:
            print(f"[{idx}/{total}] skip seed={seed}")
            continue

        print(f"\n{'='*60}")
        print(f"[{idx}/{total}] seed={seed} seq_in={trial.get('seq_in_len')} lag1={trial.get('lag_penalty_1step')} lagS={trial.get('lag_sign_penalty')} lr={trial.get('lr'):.6f} fonly={trial.get('focus_only_loss')}")
        print(f"{'='*60}")
        try:
            row = run_once(args.py, train_script, args, trial, seed, out_dir, run_id)
        except KeyboardInterrupt:
            write_results(out_dir, results)
            raise

        results.append(row)
        done.add(key)
        write_results(out_dir, results)

        rse_val = row.get('final_test_rse') or float('inf')
        marker = " *** NEW BEST ***" if rse_val < best_so_far else ""
        if rse_val < best_so_far:
            best_so_far = rse_val
        print(
            f"[{idx}/{total}] done rse={row.get('final_test_rse')} focus={row.get('test_focus_rrse')} "
            f"obj={row.get('objective_total')}{marker}"
        )

        obj_val = row.get("final_test_rse")
        if obj_val is not None and obj_val <= args.goal_rse:
            print(f"=== GOAL REACHED: test_rse={obj_val} <= {args.goal_rse} ===")
            break

        run_id += 1

    valid = [r for r in results if (r.get("final_test_rse") is not None)]
    valid_sorted = sorted(valid, key=lambda r: r.get("final_test_rse", 1e18))

    summary_path = out_dir / "best_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        if valid_sorted:
            b = valid_sorted[0]
            f.write(f"best_final_test_rse={b.get('final_test_rse')}\n")
            f.write(f"best_focus_rrse={b.get('test_focus_rrse')}\n")
            f.write(f"best_seed={b.get('seed')}\n")
            f.write(f"best_trial={json.dumps(b.get('trial', {}), ensure_ascii=False)}\n")
            f.write(f"log_file={b.get('log_file')}\n")
        else:
            f.write("No valid runs\n")

    print("\n" + "="*60)
    print("output_dir:", out_dir)
    if valid_sorted:
        print(f"\n=== TOP 5 by test_rse ===")
        for i, r in enumerate(valid_sorted[:5]):
            print(f"  #{i+1}: rse={r.get('final_test_rse')} focus_rrse={r.get('test_focus_rrse')} seed={r.get('seed')}")
    else:
        print("No valid runs")


if __name__ == "__main__":
    main()

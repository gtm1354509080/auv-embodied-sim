#!/usr/bin/env python3
"""Aggregate evaluation CSVs in results/ into the paper's robustness matrix.

Scans results/eval_*.csv (non-recursive; archive_n20/ is a subdirectory and
is therefore excluded automatically), parses method / scenario / parameter
offsets / PID gains from filenames, computes per-file statistics, writes
results/robustness_matrix.csv and prints an aligned summary table.

Filename grammar:
    eval_{tag}_{algo}_{scenario}[_mass{s}][_drag{s}].csv
    eval_pid_{scenario}_kp{a}_ki{b}_kd{c}[_mass{s}][_drag{s}].csv

CSV format (per file):
    episode,reward,mae,hold_ratio,length
    0,455.2,0.0412,0.972,500
    ...
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
import glob
import re

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
OUT_CSV = os.path.join(RESULTS_DIR, "robustness_matrix.csv")

# tag -> canonical method name used in the paper
METHOD_MAP = {
    "base": "PPO-nominal",
    "pulseppo": "PPO-pulse",
    "dr": "DR-PPO",
    "sacv4": "SAC-pulse",
    "sacnom": "SAC-nominal",
}

# narrative order for printing / writing
METHOD_ORDER = ["PID", "PPO-nominal", "DR-PPO", "PPO-pulse", "SAC-pulse", "SAC-nominal"]
SCENARIO_ORDER = {"nominal": 0, "pulse": 1}

CSV_COLUMNS = [
    "file", "method", "scenario", "mass_scale", "drag_scale", "pid_gain",
    "n_episodes", "reward_mean", "reward_std",
    "mae_mean", "mae_std", "hold_mean", "hold_std",
]


def parse_filename(fname):
    """Parse an eval CSV basename.

    Returns dict(file, method, scenario, mass_scale, drag_scale, pid_gain),
    or None if the name does not match the expected grammar.

    NOTE: parsing is positional (split on "_"), never substring-based --
    the tag "pulseppo" itself contains "pulse", so any "_pulse in name"
    check would misclassify eval_pulseppo_ppo_nominal.csv.
    """
    name = fname[:-4] if fname.endswith(".csv") else fname
    if not name.startswith("eval_"):
        return None
    body = name[len("eval_"):]

    # 1) strip trailing parameter-offset tags (_mass{s} / _drag{s}, any order, possibly both)
    mass_scale, drag_scale = 1.0, 1.0
    while True:
        m = re.search(r"_(mass|drag)([0-9]*\.?[0-9]+)$", body)
        if not m:
            break
        if m.group(1) == "mass":
            mass_scale = float(m.group(2))
        else:
            drag_scale = float(m.group(2))
        body = body[: m.start()]

    # 2) parse the core positional grammar
    pid_gain = ""
    parts = body.split("_")

    if parts[0] == "pid":
        # pid_{scenario}_kp{a}_ki{b}_kd{c}
        if len(parts) < 2:
            return None
        method = "PID"
        scenario = parts[1]
        gm = re.search(r"_kp([0-9]*\.?[0-9]+)_ki([0-9]*\.?[0-9]+)_kd([0-9]*\.?[0-9]+)$", body)
        if gm:
            pid_gain = "{}/{}/{}".format(gm.group(1), gm.group(2), gm.group(3))
    else:
        # {tag}_{algo}_{scenario}
        if len(parts) < 3:
            return None
        tag, scenario = parts[0], parts[2]
        if tag not in METHOD_MAP:
            return None
        method = METHOD_MAP[tag]

    if scenario not in SCENARIO_ORDER:
        return None

    return {
        "file": name[len("eval_"):],  # stem minus eval_ prefix, offset tags kept
        "method": method,
        "scenario": scenario,
        "mass_scale": mass_scale,
        "drag_scale": drag_scale,
        "pid_gain": pid_gain,
    }


def compute_stats(path):
    """Read one eval CSV; return (n, reward_ms, mae_ms, hold_ms) or None if empty."""
    rewards, maes, holds = [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rewards.append(float(row["reward"]))
            maes.append(float(row["mae"]))
            holds.append(float(row["hold_ratio"]))

    n = len(rewards)
    if n == 0:
        return None

    def mean_std(xs):
        arr = np.asarray(xs, dtype=float)
        std = float(arr.std(ddof=1)) if n > 1 else 0.0
        return float(arr.mean()), std

    return n, mean_std(rewards), mean_std(maes), mean_std(holds)


def sort_key(row):
    try:
        m_idx = METHOD_ORDER.index(row["method"])
    except ValueError:
        m_idx = len(METHOD_ORDER)
    return (
        m_idx,
        SCENARIO_ORDER.get(row["scenario"], 99),
        row["mass_scale"],
        row["drag_scale"],
        row["file"],
    )


def main():
    paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "eval_*.csv")))  # non-recursive
    if not paths:
        print("[WARN] no eval_*.csv found under {}".format(RESULTS_DIR))
        return

    rows = []
    for path in paths:
        fname = os.path.basename(path)
        meta = parse_filename(fname)
        if meta is None:
            print("[WARN] cannot parse filename, skipped: {}".format(fname))
            continue
        stats = compute_stats(path)
        if stats is None:
            print("[WARN] empty CSV, skipped: {}".format(fname))
            continue
        n, (r_m, r_s), (m_m, m_s), (h_m, h_s) = stats
        rows.append({
            "file": meta["file"],
            "method": meta["method"],
            "scenario": meta["scenario"],
            "mass_scale": meta["mass_scale"],
            "drag_scale": meta["drag_scale"],
            "pid_gain": meta["pid_gain"],
            "n_episodes": n,
            "reward_mean": round(r_m, 4),
            "reward_std": round(r_s, 4),
            "mae_mean": round(m_m, 4),
            "mae_std": round(m_s, 4),
            "hold_mean": round(h_m, 4),
            "hold_std": round(h_s, 4),
        })

    rows.sort(key=sort_key)

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    # human-readable aligned table
    print()
    print("{:<15} {:<8} {:>5} {:>5} {:>4}  {:<17} {:<19} {:<15}".format(
        "method", "scenario", "mass", "drag", "n", "reward", "mae", "hold"))
    for r in rows:
        gain = " [{}]".format(r["pid_gain"]) if r["pid_gain"] else ""
        print("{:<15} {:<8} {:>5.2f} {:>5.2f} {:>4d}  {:7.2f} ± {:6.2f}  {:.4f} ± {:.4f}  {:.3f} ± {:.3f}{}".format(
            r["method"], r["scenario"], r["mass_scale"], r["drag_scale"], r["n_episodes"],
            r["reward_mean"], r["reward_std"],
            r["mae_mean"], r["mae_std"],
            r["hold_mean"], r["hold_std"], gain))
    print()
    print("已生成 results/robustness_matrix.csv ({} rows)".format(len(rows)))


if __name__ == "__main__":
    main()

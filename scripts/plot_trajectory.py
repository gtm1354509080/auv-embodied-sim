#!/usr/bin/env python3
"""
Fig 3 -- Depth trajectories under the force-pulse evaluation scenario.
Fig 5 -- Control actions (smoothness comparison).
"""
import argparse
import csv
import glob
import os
import statistics
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

TIME_CANDIDATES = ["time", "t", "timestamp", "time_s"]
DEPTH_CANDIDATES = ["depth", "z", "depth_m", "pos_z"]
ACTION_CANDIDATES = ["action", "a", "thruster", "u", "cmd"]
TARGET_CANDIDATES = ["target", "z_ref", "ref", "target_depth"]
PULSE_CANDIDATES = ["pulse", "pulse_force", "f_ext"]

METHOD_GUESS = [("pulseppo", "PPO-pulse"), ("pid", "PID"), ("sac", "SAC-pulse"),
                ("dr", "DR-PPO"), ("ppo", "PPO")]
PLOT_ORDER = ["PID", "PPO-pulse", "SAC-pulse", "DR-PPO", "PPO"]
COLORS = {"PID": "#8c8c8c", "PPO-pulse": "#c44e52", "SAC-pulse": "#8172b3",
          "DR-PPO": "#55a868", "PPO": "#4c72b0"}
HOLD_BAND = 0.05


def guess_method(fname):
    l = os.path.basename(fname).lower()
    for key, name in METHOD_GUESS:
        if key in l:
            return name
    return None


def pick(header_lower, header, candidates):
    for c in candidates:
        if c in header_lower:
            return header[header_lower.index(c)]
    return None


def read_traj(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        lower = [h.strip().lower() for h in header]
        c_time = pick(lower, header, TIME_CANDIDATES)
        c_depth = pick(lower, header, DEPTH_CANDIDATES)
        c_action = pick(lower, header, ACTION_CANDIDATES)
        c_target = pick(lower, header, TARGET_CANDIDATES)
        c_pulse = pick(lower, header, PULSE_CANDIDATES)
        if c_depth is None:
            print("[ERROR] no depth column in {}".format(path))
            return None
        cols = {k: [] for k in ("time", "depth", "action", "target", "pulse")}
        for row in reader:
            for key, col in (("time", c_time), ("depth", c_depth), ("action", c_action),
                             ("target", c_target), ("pulse", c_pulse)):
                if col is not None:
                    cols[key].append(float(row[col]))
    if not cols["time"]:
        cols["time"] = list(range(len(cols["depth"])))
    return cols


def apply_ieee_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.6, "lines.linewidth": 1.0,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    return plt


def reference_depth(series_list):
    targets = [v for s in series_list if s["target"] for v in s["target"]]
    if targets:
        return statistics.fmean(targets)
    steady = [v for s in series_list for v in s["depth"][-100:]]
    ref = statistics.median(steady)
    print("[INFO] no target column; inferred ref = {:.3f} m".format(ref))
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    ap.add_argument("--traj", action="append")
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    if args.traj:
        files = [item.split("=", 1)[1] for item in args.traj]
    else:
        files = sorted(glob.glob(os.path.join(args.dir, "traj_*.csv")))
    if not files:
        print("[ERROR] no traj_*.csv found")
        sys.exit(1)

    data = {}
    for path in files:
        method = guess_method(path)
        if method is None:
            print("[WARN] cannot guess method: {}".format(path))
            continue
        series = read_traj(path)
        if series is None:
            continue
        data[method] = series
        print("[INFO] {} -> {} ({} steps)".format(os.path.basename(path), method, len(series["depth"])))

    if not data:
        sys.exit(1)

    plt = apply_ieee_style()
    os.makedirs(args.outdir, exist_ok=True)
    order = [m for m in PLOT_ORDER if m in data] + [m for m in data if m not in PLOT_ORDER]

    ref = reference_depth([data[m] for m in order])
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    ax.axhspan(ref - HOLD_BAND, ref + HOLD_BAND, color="green", alpha=0.12, label="hold band")
    ax.axhline(ref, color="black", linewidth=0.7, linestyle="--", label="target")
    for m in order:
        s = data[m]
        ax.plot(s["time"], s["depth"], color=COLORS.get(m, None), label=m, alpha=0.9)
    ax.set_xlabel("Time (s)" if data[order[0]]["time"] != list(range(len(data[order[0]]["depth"]))) else "Step")
    ax.set_ylabel("Depth (m)")
    ax.grid(alpha=0.3, linewidth=0.4)
    ax.legend(frameon=False, ncol=2, loc="best")
    fig.tight_layout()
    out3 = os.path.join(args.outdir, "fig3_trajectories")
    fig.savefig(out3 + ".pdf"); fig.savefig(out3 + ".png", dpi=300)
    print("saved {}.pdf/.png".format(out3))

    have_action = [m for m in order if data[m]["action"]]
    if not have_action:
        print("[WARN] no action column; Fig 5 skipped")
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    for m in have_action:
        s = data[m]
        ax.plot(s["time"], s["action"], color=COLORS.get(m, None), label=m, alpha=0.85)
    ax.set_xlabel("Time (s)" if data[order[0]]["time"] != list(range(len(data[order[0]]["depth"]))) else "Step")
    ax.set_ylabel("Thruster command")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=0.3, linewidth=0.4)
    ax.legend(frameon=False, ncol=2, loc="best")
    fig.tight_layout()
    out5 = os.path.join(args.outdir, "fig5_actions")
    fig.savefig(out5 + ".pdf"); fig.savefig(out5 + ".png", dpi=300)
    print("saved {}.pdf/.png".format(out5))

    print("\nmean |delta action|:")
    for m in have_action:
        a = data[m]["action"]
        mad = statistics.fmean(abs(b - c) for b, c in zip(a[1:], a[:-1]))
        print("  {:<12} {:.4f}".format(m, mad))


if __name__ == "__main__":
    main()
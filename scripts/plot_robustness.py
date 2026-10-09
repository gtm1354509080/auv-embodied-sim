#!/usr/bin/env python3
"""Fig 4 -- Hold-ratio robustness comparison (grouped bar chart)."""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

METHODS = ["PID", "PPO-nominal", "DR-PPO", "PPO-pulse", "SAC-pulse"]
SCEN_KEYS = ["nominal", "pulse", "mass", "drag"]
SCEN_LABELS = {"nominal": "Nominal", "pulse": "Force pulse",
               "mass": "Mass +15%", "drag": "Drag \u221225%"}
HATCHES = ["", "//", "xx", "..", "\\\\", "oo"]
COLORS = ["#8c8c8c", "#4c72b0", "#55a868", "#c44e52", "#8172b3", "#cc9966"]


def scenario_key(row):
    if float(row["mass_scale"]) > 1.0:
        return "mass"
    if float(row["drag_scale"]) < 1.0:
        return "drag"
    return row["scenario"]


def build_values(csv_path, methods):
    values = {m: {} for m in methods}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            method = row["method"]
            if method not in values:
                continue
            key = scenario_key(row)
            if key not in SCEN_KEYS:
                continue
            values[method][key] = (float(row["hold_mean"]), float(row["hold_std"]))
    for m in methods:
        for k in SCEN_KEYS:
            if k not in values[m]:
                print("[WARN] missing cell: {} @ {}".format(m, k))
    return values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join("results", "robustness_matrix.csv"))
    ap.add_argument("--include-sacnom", action="store_true")
    ap.add_argument("--out", default=os.path.join("figures", "fig4_robustness"))
    args = ap.parse_args()

    methods = list(METHODS)
    if args.include_sacnom:
        methods.append("SAC-nominal")

    values = build_values(args.csv, methods)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    n_m, n_s = len(methods), len(SCEN_KEYS)
    width = 0.8 / n_m

    for i, m in enumerate(methods):
        xs, ys, es = [], [], []
        for j, k in enumerate(SCEN_KEYS):
            if k not in values[m]:
                continue
            xs.append(j - 0.4 + width * (i + 0.5))
            ys.append(values[m][k][0])
            es.append(values[m][k][1])
        ax.bar(xs, ys, width=width, yerr=es, capsize=1.5,
               color=COLORS[i % len(COLORS)], hatch=HATCHES[i % len(HATCHES)],
               edgecolor="black", linewidth=0.5, error_kw={"linewidth": 0.6},
               label=m)
        
    ax.set_xlim(-0.5, n_s - 0.5)
    ax.set_xticks(range(n_s))
    ax.set_xticklabels([SCEN_LABELS[k] for k in SCEN_KEYS])
    ax.set_ylabel("Hold ratio")
    ax.set_ylim(0.85, 1.0)
    ax.grid(axis="y", alpha=0.3, linewidth=0.4)
    ax.legend(frameon=False, ncol=3, loc="lower left", columnspacing=0.8, handlelength=1.4)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out + ".pdf")
    fig.savefig(args.out + ".png", dpi=300)
    print("saved {}.pdf/.png".format(args.out))


if __name__ == "__main__":
    main()
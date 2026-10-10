"""Fig 2: SAC training curves under two training regimes (from TensorBoard eval records)."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import glob
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.6, "lines.linewidth": 1.2,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load_scalar(logdir, tag):
    ea = EventAccumulator(logdir, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return None, None
    ev = ea.Scalars(tag)
    return np.array([e.step for e in ev]), np.array([e.value for e in ev])


def smooth(y, k=3):
    if k <= 1 or len(y) <= k:
        return y
    return np.convolve(y, np.ones(k) / k, mode="same")


def main():
    os.makedirs("figures", exist_ok=True)

    configs = [
        ("SAC-pulse (v4)", "tb/sac_v4/SAC_*", "C1"),
        ("SAC-nominal",    "tb/sac_nominal/SAC_*", "C0"),
    ]

    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    plotted = 0

    for label, pattern, color in configs:
        dirs = sorted(glob.glob(pattern))
        if not dirs:
            print(f"[WARN] no dir matched {pattern}")
            continue
        s, v = load_scalar(dirs[0], "eval/mean_reward")
        if s is None:
            print(f"[WARN] eval/mean_reward not found in {dirs[0]}")
            continue
        ax.plot(s, smooth(v), color=color, label=label)
        print(f"[INFO] {label}: {len(s)} points, final={v[-1]:.1f}")
        plotted += 1

    if plotted == 0:
        print("[ERROR] no SAC curve plotted")
        return

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Evaluation reward")
    ax.grid(alpha=0.3, linewidth=0.4)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig("figures/fig2_training_curves.pdf")
    fig.savefig("figures/fig2_training_curves.png", dpi=300)
    print("saved figures/fig2_training_curves.{pdf,png}")


if __name__ == "__main__":
    main()
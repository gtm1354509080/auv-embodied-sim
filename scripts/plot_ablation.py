"""Fig 6: Reward-weight sensitivity analysis (PPO ablation).

Reads TensorBoard training curves for the three PPO variants
(A = baseline, B = energy-focused, C = smoothness-focused), plots
rollout/ep_rew_mean vs. environment steps.

Output: figures/fig6_ablation.{pdf,png}
Style: IEEE single column (3.5 in), serif 8 pt, embedded fonts.
"""
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
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
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


def smooth(y, k=5):
    if k <= 1 or len(y) <= k:
        return y
    return np.convolve(y, np.ones(k) / k, mode="same")


VARIANTS = [
    ("A: baseline",   "tb/ablation_A/PPO_*", "C0"),
    ("B: energy",     "tb/ablation_B/PPO_*", "C1"),
    ("C: smoothness", "tb/ablation_C/PPO_*", "C2"),
]


def main():
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.5, 2.3))

    plotted = 0
    for label, pattern, color in VARIANTS:
        dirs = sorted(glob.glob(pattern))
        if not dirs:
            print(f"[WARN] no dir matched {pattern}")
            continue
        s, v = load_scalar(dirs[0], "rollout/ep_rew_mean")
        if s is None:
            print(f"[WARN] rollout/ep_rew_mean not found in {dirs[0]}")
            continue
        ax.plot(s, smooth(v), color=color, label=label)
        print(f"[INFO] {label}: {len(s)} points, final={v[-1]:.2f}")
        plotted += 1

    if plotted == 0:
        print("[ERROR] no ablation curves plotted; check tb/ablation_*/ exists")
        return

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Mean episode reward")
    ax.grid(alpha=0.3, linewidth=0.4)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig("figures/fig6_ablation.pdf")
    fig.savefig("figures/fig6_ablation.png", dpi=300)
    print("saved figures/fig6_ablation.{pdf,png}")


if __name__ == "__main__":
    main()
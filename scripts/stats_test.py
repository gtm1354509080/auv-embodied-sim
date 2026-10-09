"""论文关键对比的显著性检验。

主检验：Wilcoxon 符号秩检验（配对）。
  所有方法共用 seed 5678 评估协议，第 i 回合的初始状态跨方法一一配对，
  配对检验消去初始条件方差，功效高于非配对检验——对 "training regime"
  这种 ~0.7pp 的小效应尤其重要。
辅助检验：Welch t（非配对，正态近似），供参考。
hold ratio 有界且接近 1.0、分布偏斜，正文以非参数检验（Wilcoxon）为准。

用法：
  python scripts/stats_test.py              # 跑全部预定义对比
  python scripts/stats_test.py --col mae    # 换指标（reward/mae/hold）

数据前提：results/ 下为 n=50 统一重跑后的 CSV（--tag 命名）。
缺失文件会警告并跳过，不会中断。
"""
import argparse
import csv
import math
import os

import numpy as np

RESULTS = "results"

# (标签, 文件A, 文件B)——文件名对应 n=50 重跑的 --tag 命名
COMPARISONS = [
    # A. 头条：RL vs PID（脉冲场景）
    ("A1 RLvsPID  base-PPO @ pulse",  "eval_base_ppo_pulse.csv",                "eval_pid_pulse_kp3.0_ki0.0_kd1.5.csv"),
    ("A2 RLvsPID  SAC-v4   @ pulse",  "eval_sacv4_sac_pulse.csv",               "eval_pid_pulse_kp3.0_ki0.0_kd1.5.csv"),
    # B. 训练制度：pulse-PPO vs nominal-PPO（四个条件）
    ("B1 Regime   @ nominal",         "eval_pulseppo_ppo_nominal.csv",          "eval_base_ppo_nominal.csv"),
    ("B2 Regime   @ pulse",           "eval_pulseppo_ppo_pulse.csv",            "eval_base_ppo_pulse.csv"),
    ("B3 Regime   @ mass+15%",        "eval_pulseppo_ppo_nominal_mass1.15.csv", "eval_base_ppo_nominal_mass1.15.csv"),
    ("B4 Regime   @ drag-25%",        "eval_pulseppo_ppo_nominal_drag0.75.csv", "eval_base_ppo_nominal_drag0.75.csv"),
    # C. 算法：pulse-PPO vs pulse-SAC（四个条件）
    ("C1 Algo     @ nominal",         "eval_pulseppo_ppo_nominal.csv",          "eval_sacv4_sac_nominal.csv"),
    ("C2 Algo     @ pulse",           "eval_pulseppo_ppo_pulse.csv",            "eval_sacv4_sac_pulse.csv"),
    ("C3 Algo     @ mass+15%",        "eval_pulseppo_ppo_nominal_mass1.15.csv", "eval_sacv4_sac_nominal_mass1.15.csv"),
    ("C4 Algo     @ drag-25%",        "eval_pulseppo_ppo_nominal_drag0.75.csv", "eval_sacv4_sac_nominal_drag0.75.csv"),
    # D. DR：DR-PPO vs nominal-PPO（两个偏移条件）
    ("D1 DR       @ mass+15%",        "eval_dr_ppo_nominal_mass1.15.csv",       "eval_base_ppo_nominal_mass1.15.csv"),
    ("D2 DR       @ drag-25%",        "eval_dr_ppo_nominal_drag0.75.csv",       "eval_base_ppo_nominal_drag0.75.csv"),
]


def load_col(path, col):
    with open(path) as f:
        rows = list(csv.reader(f))
    header = [h.strip().lower() for h in rows[0]]
    idx = next((i for i, h in enumerate(header) if col in h), None)
    if idx is None:
        idx = {"reward": 1, "mae": 2, "hold": 3}[col]  # 位置兜底
    return np.array([float(r[idx]) for r in rows[1:] if r])


def norm_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def rankdata(a):
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def wilcoxon_p(x, y):
    """Wilcoxon 符号秩检验（配对），正态近似，双侧。"""
    d = x - y
    d = d[d != 0.0]
    n = len(d)
    if n < 5:
        return float("nan")
    r = rankdata(np.abs(d))
    W = float(r[d > 0].sum())
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (W - mu) / sigma
    return 2.0 * (1.0 - norm_cdf(abs(z)))


def welch_p(x, y):
    nx, ny = len(x), len(y)
    t = (x.mean() - y.mean()) / math.sqrt(x.var(ddof=1) / nx + y.var(ddof=1) / ny)
    return 2.0 * (1.0 - norm_cdf(abs(t)))


def sig(p):
    if math.isnan(p):
        return "n/a"
    if p < 0.001:
        return "p<0.001 ***"
    if p < 0.01:
        return f"p={p:.3f} **"
    if p < 0.05:
        return f"p={p:.3f} *"
    return f"p={p:.3f} n.s."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--col", default="hold", choices=["reward", "mae", "hold"])
    args = ap.parse_args()

    print(f"指标: {args.col} | 主检验 Wilcoxon(配对) | 辅助 Welch t")
    print("-" * 92)
    for label, fa, fb in COMPARISONS:
        pa, pb = os.path.join(RESULTS, fa), os.path.join(RESULTS, fb)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print(f"{label:34s} 缺文件，跳过 ({fa if not os.path.exists(pa) else fb})")
            continue
        x, y = load_col(pa, args.col), load_col(pb, args.col)
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]
        diff = x.mean() - y.mean()
        pw = wilcoxon_p(x, y)
        pt = welch_p(x, y)
        print(f"{label:34s} n={n:2d}  {x.mean():.4f}±{x.std(ddof=1):.4f} vs "
              f"{y.mean():.4f}±{y.std(ddof=1):.4f}  diff={diff:+.4f}  "
              f"W:{sig(pw):15s} t:{sig(pt)}")


if __name__ == "__main__":
    main()

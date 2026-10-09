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
    """Wilcoxon 符号秩检验（配对），正态近似，双侧。
    返回 (p, rank_biserial_r, n_used)。
    rank-biserial r = (W+ - W-) / (n(n+1)/2)，正值表示 x > y 方向。"""
    d = x - y
    d = d[d != 0.0]
    n = len(d)
    if n < 5:
        return float("nan"), float("nan"), n
    r = rankdata(np.abs(d))
    W = float(r[d > 0].sum())
    total = n * (n + 1) / 2.0
    r_rb = (2.0 * W - total) / total
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (W - mu) / sigma
    return 2.0 * (1.0 - norm_cdf(abs(z))), r_rb, n


def holm_adjust(pvals):
    """Holm-Bonferroni 校正。输入 p 列表（可含 nan），输出校正后 p 列表。"""
    m = sum(0 if math.isnan(p) else 1 for p in pvals)
    order = sorted((p, i) for i, p in enumerate(pvals) if not math.isnan(p))
    adj = [float("nan")] * len(pvals)
    running = 0.0
    for rank, (p, i) in enumerate(order):
        val = min(1.0, (m - rank) * p)
        running = max(running, val)
        adj[i] = running
    return adj


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

    rows = []
    for label, fa, fb in COMPARISONS:
        pa, pb = os.path.join(RESULTS, fa), os.path.join(RESULTS, fb)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print(f"{label:34s} 缺文件，跳过 ({fa if not os.path.exists(pa) else fb})")
            continue
        x, y = load_col(pa, args.col), load_col(pb, args.col)
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]
        pw, r_rb, n_eff = wilcoxon_p(x, y)
        pt = welch_p(x, y)
        rows.append({"label": label, "n": n,
                     "mx": x.mean(), "sx": x.std(ddof=1),
                     "my": y.mean(), "sy": y.std(ddof=1),
                     "diff": x.mean() - y.mean(),
                     "p_w": pw, "r_rb": r_rb, "p_t": pt})

    p_holm = holm_adjust([r["p_w"] for r in rows])
    for r, ph in zip(rows, p_holm):
        r["p_holm"] = ph

    print(f"指标: {args.col} | 主检验 Wilcoxon(配对) | r_rb=rank-biserial 效应量 | Holm=多重校正后 p | 辅助 Welch t")
    print("-" * 118)
    for r in rows:
        print(f"{r['label']:34s} n={r['n']:2d}  {r['mx']:.4f}±{r['sx']:.4f} vs "
              f"{r['my']:.4f}±{r['sy']:.4f}  diff={r['diff']:+.4f}  "
              f"W:{sig(r['p_w']):15s} r={r['r_rb']:+.2f}  Holm:{sig(r['p_holm']):15s} t:{sig(r['p_t'])}")

    out = os.path.join(RESULTS, f"stats_summary_{args.col}.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "n", "mean_x", "std_x", "mean_y", "std_y", "diff",
                    "p_wilcoxon", "rank_biserial_r", "p_holm", "p_welch"])
        for r in rows:
            w.writerow([r["label"], r["n"], f"{r['mx']:.6f}", f"{r['sx']:.6f}",
                        f"{r['my']:.6f}", f"{r['sy']:.6f}", f"{r['diff']:.6f}",
                        f"{r['p_w']:.3e}", f"{r['r_rb']:.4f}",
                        f"{r['p_holm']:.3e}" if not math.isnan(r["p_holm"]) else "",
                        f"{r['p_t']:.3e}"])
    print(f"\n已生成 {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

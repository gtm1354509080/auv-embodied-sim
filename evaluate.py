"""统一评估脚本 v2：PPO / SAC / PID 同一环境公平对比 + 参数偏移鲁棒性评估

相对 v1 的修正：
  1. 新增 --mass-scale / --drag-scale：物理参数偏移评估（DR 鲁棒性矩阵的核心）
  2. SB3 评估路径显式 venv.seed(seed) —— v1 未 seed，SB3 每组评估抽到的初始
     状态随机（PID 路径有 seed 所以两次复测一致，SB3 不可复现），这正是
     "pulse 场景数值优于 nominal" 伪影的根因。seed 后 nominal/pulse 逐回合
     初始状态配对，差异只来自扰动本身。

用法:
    # 标称
    python evaluate.py --algo sac --model sac_auv_depth_pulse_v4.zip \
        --vecnorm vecnormalize_v4.pkl --scenario nominal -n 20
    # 参数偏移（DR 评估）
    python evaluate.py --algo ppo --model ppo_auv_depth_dr_seed0.zip \
        --vecnorm vecnormalize_dr_ppo_seed0.pkl --scenario nominal \
        --mass-scale 1.15 -n 20
    python evaluate.py --algo pid --scenario nominal --drag-scale 0.75 -n 20
"""
import argparse
import csv
import os

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.auv_depth_env import AuvDepthEnv
from envs.pid_controller import PIDController
from envs.pulse_env import ForcePulseEnv

ALGOS = {"ppo": PPO, "sac": SAC}


def build_env(scenario, mass_scale=1.0, drag_scale=1.0):
    if scenario == "nominal":
        env = AuvDepthEnv()
    elif scenario == "pulse":
        env = ForcePulseEnv(
            pulse_force=0.3,
            pulse_duration=10,
            pulses_per_episode=(2, 2),
            obs_noise_std=0.0,
        )
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    env.mass *= mass_scale
    env.drag_coeff *= drag_scale
    return env


def eval_sb3(algo, model_path, vecnorm_path, scenario, n_episodes, seed,
             mass_scale, drag_scale):
    """SB3 评估 v3：单环境 + 手动 obs 归一化 + 每回合显式 env.reset(seed=...)。

    与 PID 路径完全同构的 seed 协议，不依赖 VecEnv.seed / model.load 的任何
    内部行为。归一化统计量直接读 pkl（training=False，统计量不更新）。
    """
    if not (vecnorm_path and os.path.exists(vecnorm_path)):
        raise FileNotFoundError(f"VecNormalize 文件缺失: {vecnorm_path}")
    vn = VecNormalize.load(
        vecnorm_path,
        DummyVecEnv([lambda: build_env(scenario, mass_scale, drag_scale)]),
    )
    vn.training = False
    vn.norm_reward = False
    obs_rms, clip_obs, eps = vn.obs_rms, vn.clip_obs, vn.epsilon

    model = ALGOS[algo].load(model_path)  # 不挂 env，避免 load 内部 force_reset

    def norm(obs):
        return np.clip(
            (obs - obs_rms.mean) / np.sqrt(obs_rms.var + eps),
            -clip_obs, clip_obs,
        ).astype(np.float32)

    ep_rewards, ep_mae, ep_hold, ep_len = [], [], [], []
    for ep in range(n_episodes):
        env = build_env(scenario, mass_scale, drag_scale)
        obs, info = env.reset(seed=seed + ep)  # 关键：每回合显式 seed
        obs_n = norm(obs)
        cur_r, cur_ae, cur_hold, cur_n = 0.0, 0.0, 0.0, 0
        for _ in range(600):
            action, _ = model.predict(obs_n.reshape(1, -1), deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action[0])
            obs_n = norm(obs)
            cur_r += float(reward)
            cur_ae += abs(info["error"])
            cur_hold += float(info["in_band"])
            cur_n += 1
            if terminated or truncated:
                break
        ep_rewards.append(cur_r)
        ep_mae.append(cur_ae / cur_n)
        ep_hold.append(cur_hold / cur_n)
        ep_len.append(cur_n)
    return ep_rewards, ep_mae, ep_hold, ep_len


def eval_pid(scenario, n_episodes, seed, mass_scale, drag_scale,
             kp=1.0, ki=0.0, kd=0.5):
    ep_rewards, ep_mae, ep_hold, ep_len = [], [], [], []
    for ep in range(n_episodes):
        env = build_env(scenario, mass_scale, drag_scale)
        _, info = env.reset(seed=seed + ep)
        pid = PIDController(kp=kp, ki=ki, kd=kd, dt=env.dt)
        pid.reset()
        cur_r, cur_ae, cur_hold, cur_n = 0.0, 0.0, 0.0, 0
        while True:
            action = np.array([pid.compute(info["target"], info["depth"])], dtype=np.float32)
            _, reward, terminated, truncated, info = env.step(action)
            cur_r += reward
            cur_ae += abs(info["error"])
            cur_hold += float(info["in_band"])
            cur_n += 1
            if terminated or truncated:
                break
        ep_rewards.append(cur_r)
        ep_mae.append(cur_ae / cur_n)
        ep_hold.append(cur_hold / cur_n)
        ep_len.append(cur_n)
    return ep_rewards, ep_mae, ep_hold, ep_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=["ppo", "sac", "pid"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--vecnorm", default=None)
    ap.add_argument("--scenario", default="nominal", choices=["nominal", "pulse"])
    ap.add_argument("--mass-scale", type=float, default=1.0, help="质量偏移倍率, 如 1.15")
    ap.add_argument("--drag-scale", type=float, default=1.0, help="阻尼偏移倍率, 如 0.75")
    ap.add_argument("-n", "--n-episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--kp", type=float, default=1.0, help="PID 增益（仅 algo=pid）")
    ap.add_argument("--ki", type=float, default=0.0)
    ap.add_argument("--kd", type=float, default=0.5)
    ap.add_argument("--tag", default="", help="CSV 文件名前缀标识（如 dr、base），防止不同模型互相覆盖")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    if args.algo == "pid":
        res = eval_pid(args.scenario, args.n_episodes, args.seed,
                       args.mass_scale, args.drag_scale,
                       kp=args.kp, ki=args.ki, kd=args.kd)
    else:
        assert args.model, "SB3 模型需要 --model"
        res = eval_sb3(args.algo, args.model, args.vecnorm,
                       args.scenario, args.n_episodes, args.seed,
                       args.mass_scale, args.drag_scale)

    ep_rewards, ep_mae, ep_hold, ep_len = res
    ms = lambda x: (float(np.mean(x)), float(np.std(x)))
    r_m, r_s = ms(ep_rewards)
    mae_m, mae_s = ms(ep_mae)
    h_m, h_s = ms(ep_hold)
    l_m, l_s = ms(ep_len)

    tag = f"{args.algo}_{args.scenario}"
    if args.tag:
        tag = f"{args.tag}_{tag}"
    if args.algo == "pid" and (args.kp, args.ki, args.kd) != (1.0, 0.0, 0.5):
        tag += f"_kp{args.kp}_ki{args.ki}_kd{args.kd}"
    if args.mass_scale != 1.0:
        tag += f"_mass{args.mass_scale}"
    if args.drag_scale != 1.0:
        tag += f"_drag{args.drag_scale}"

    print(f"\n=== {args.algo.upper()} @ {args.scenario} "
          f"(mass×{args.mass_scale}, drag×{args.drag_scale}, {len(ep_rewards)} eps) ===")
    print(f"episode reward : {r_m:10.2f} ± {r_s:.2f}")
    print(f"depth MAE (m)  : {mae_m:10.4f} ± {mae_s:.4f}")
    print(f"hold ratio     : {h_m:10.3f} ± {h_s:.3f}")
    print(f"episode length : {l_m:10.1f} ± {l_s:.1f}")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"eval_{tag}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "reward", "mae", "hold_ratio", "length"])
        for i, (r, a, h, le) in enumerate(zip(ep_rewards, ep_mae, ep_hold, ep_len)):
            w.writerow([i, r, a, h, le])
    print(f"已保存: {path}")


if __name__ == "__main__":
    main()

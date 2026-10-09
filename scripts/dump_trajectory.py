"""转储单回合轨迹：depth / target / action 时间序列，供 Fig 3、Fig 5 使用。

相对 DeepSeek 版的两处修正：
1. 时间轴对齐：原版把第 1 步之后的 depth 记在 t=0，整条曲线平移了一个 dt。
   本版先记录 reset 后的初始状态（t=0），再从 t=dt 开始记录。
2. 环境属性访问改用 get_attr()：原版 venv.envs[0] 只在 DummyVecEnv 上有效，
   被 VecNormalize 包装后会 AttributeError（本脚本 PID 路径恰好不加载 vecnorm
   所以能跑，但属于脆弱写法）。

脉冲参数与 evaluate.py 的评估场景一致（pulses_per_episode=(2,2)、
obs_noise_std=0.0）——轨迹图展示的是评估场景，这是正确的对应关系。
PID 增益 kp=3.0/ki=0.0/kd=1.5，与 Table 2 网格搜索终稿一致。
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.auv_depth_env import AuvDepthEnv
from envs.pulse_env import ForcePulseEnv
from envs.pid_controller import PIDController

DT = 0.1
STEPS = 500


def build(scenario, seed):
    if scenario == "pulse":
        env = ForcePulseEnv(
            pulse_force=0.3,
            pulse_duration=10,
            pulses_per_episode=(2, 2),
            obs_noise_std=0.0,
        )
    else:
        env = AuvDepthEnv()
    venv = DummyVecEnv([lambda: env])
    venv.seed(seed)
    return venv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True, choices=["ppo", "sac", "pid"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--vecnorm", default=None)
    ap.add_argument("--scenario", default="nominal", choices=["nominal", "pulse"])
    ap.add_argument("--seed", type=int, default=999)
    args = ap.parse_args()

    venv = build(args.scenario, args.seed)
    if args.vecnorm:
        venv = VecNormalize.load(args.vecnorm, venv)
        venv.training = False
        venv.norm_reward = False

    if args.algo == "pid":
        policy = None
    elif args.algo == "sac":
        policy = SAC.load(args.model)
    else:
        policy = PPO.load(args.model)

    obs = venv.reset()
    pid = PIDController(kp=3.0, ki=0.0, kd=1.5, dt=DT)
    traj = {"t": [], "depth": [], "target": [], "action": []}

    # t=0：reset 后的初始状态（动作记为初始 last_action=0）
    traj["t"].append(0.0)
    traj["depth"].append(float(venv.get_attr("depth")[0]))
    traj["target"].append(float(venv.get_attr("target_depth")[0]))
    traj["action"].append(0.0)

    for step in range(1, STEPS + 1):
        if args.algo == "pid":
            target = float(venv.get_attr("target_depth")[0])
            depth = float(venv.get_attr("depth")[0])
            a = np.array([[pid.compute(target, depth)]])
        else:
            a, _ = policy.predict(obs, deterministic=True)
        obs, r, done, infos = venv.step(a)
        traj["t"].append(step * DT)
        traj["depth"].append(float(infos[0]["depth"]))
        traj["target"].append(float(infos[0]["target"]))
        traj["action"].append(float(a[0][0]))
        if done[0]:
            break

    import csv
    csv_path = f"results/traj_{args.algo}_{args.scenario}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "depth", "target", "action"])
        for t, d, tg, a in zip(traj["t"], traj["depth"], traj["target"], traj["action"]):
            w.writerow([t, d, tg, a])
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()

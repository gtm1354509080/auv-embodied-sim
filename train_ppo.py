"""PPO 重训（环境已修复为"深度保持"任务，旧模型与新环境不可比，必须重训）

1M 步 CPU 约 10 分钟。seed 0 今晚跑，seed 1-2 明天后台补。

用法:
    python train_ppo.py --seed 0
"""
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.auv_depth_env import AuvDepthEnv


def make_env():
    return AuvDepthEnv()  # 标称环境（与 SAC v4 的评估环境一致）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1_000_000)
    args = ap.parse_args()

    env = DummyVecEnv([make_env for _ in range(4)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./tb/ppo_v2",
        seed=args.seed,
    )
    model.learn(total_timesteps=args.steps)
    model.save(f"ppo_auv_depth_v2_seed{args.seed}.zip")
    env.save(f"vecnormalize_ppo_v2_seed{args.seed}.pkl")
    print(f"完成: ppo_auv_depth_v2_seed{args.seed}.zip")


if __name__ == "__main__":
    main()

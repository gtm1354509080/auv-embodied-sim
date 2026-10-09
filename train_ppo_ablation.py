"""PPO 权重消融：3 组权重各跑 200k 步，用于 Fig 6"""
import argparse
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.auv_depth_env import AuvDepthEnv


WEIGHTS = {
    "A": [1.0, 0.1, 0.01, 0.001],   # baseline
    "B": [1.0, 0.1, 0.1, 0.01],     # energy-focused
    "C": [1.0, 0.01, 0.001, 0.1],   # smoothness-focused
}


def make_env(w):
    def _init():
        env = AuvDepthEnv()
        env.w1, env.w2, env.w3, env.w4 = w
        return env
    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(WEIGHTS.keys()))
    args = ap.parse_args()

    w = WEIGHTS[args.variant]
    env = DummyVecEnv([make_env(w) for _ in range(4)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO("MlpPolicy", env, verbose=1, seed=0,
                tensorboard_log=f"./tb/ablation_{args.variant}/")
    model.learn(total_timesteps=200_000)
    model.save(f"ppo_ablation_{args.variant}")
    env.save(f"vecnormalize_ablation_{args.variant}.pkl")
    print(f"完成: {args.variant} = {w}")


if __name__ == "__main__":
    main()
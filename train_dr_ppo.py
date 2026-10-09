"""DR-PPO 训练：域随机化环境

相对 DeepSeek 版的修正：n_envs 1 → 4，与 train_ppo.py 保持一致
（算法间训练条件可比，且约快 4 倍）。
"""
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.auv_depth_env import AuvDepthEnv
from envs.domain_randomize import DomainRandomizeWrapper


def make_env():
    return DomainRandomizeWrapper(
        AuvDepthEnv(), mass_std=0.10, drag_std=0.20, noise_std=0.01
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = DummyVecEnv([make_env for _ in range(4)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    env.seed(args.seed)  # VecEnv.seed 会给 4 个子环境分配 seed+idx，保证可复现

    model = PPO(
        "MlpPolicy", env, verbose=1,
        tensorboard_log="./tb/dr_ppo/", seed=args.seed,
        # 以下超参与 train_ppo.py（SB3 默认值）显式对齐，保证 PPO vs DR-PPO 可比
        n_steps=2048, batch_size=64, learning_rate=3e-4, gamma=0.99,
    )
    model.learn(total_timesteps=1_000_000)
    model.save(f"ppo_auv_depth_dr_seed{args.seed}")
    env.save(f"vecnormalize_dr_ppo_seed{args.seed}.pkl")
    print(f"完成: ppo_auv_depth_dr_seed{args.seed}.zip")


if __name__ == "__main__":
    main()

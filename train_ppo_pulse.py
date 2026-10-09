"""pulse-PPO：脉冲环境训练版 PPO，用于公平对比 SAC v4（训练制度 vs 算法）

关键：训练环境参数必须与 sac_fix/train_sac.py（SAC v4）逐字段一致——
对比的变量是"算法"，脉冲制度（幅度/时长/次数/起始步/观测噪声）必须完全相同，
否则只是用一个新混淆变量替换旧混淆变量。

SAC v4 训练制度（train_sac.py:27-33）：
  pulse_force=0.3, pulse_duration=10, pulses_per_episode=(1, 2),
  first_pulse_after=100, obs_noise_std=0.01

注意与 evaluate.py 的评估制度区分：评估用 pulses_per_episode=(2,2)、
obs_noise_std=0.0、无 first_pulse_after——训练/评估协议不同是正常的，
但两个"脉冲训练"的模型之间，训练制度必须一致。

超参：全部 SB3 默认（n_steps=2048, batch=64, lr=3e-4, gamma=0.99），
与 train_ppo.py 完全一致——DeepSeek 的提醒，防止超参混进"训练制度"变量。
"""
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.pulse_env import ForcePulseEnv


def make_env():
    return ForcePulseEnv(
        pulse_force=0.3,
        pulse_duration=10,
        pulses_per_episode=(1, 2),   # 与 SAC v4 训练一致（不是评估的 (2,2)）
        first_pulse_after=100,       # 与 SAC v4 训练一致
        obs_noise_std=0.01,          # 与 SAC v4 训练一致（评估才是 0.0）
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = DummyVecEnv([make_env for _ in range(4)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    env.seed(args.seed)  # 子环境可复现（与 train_dr_ppo 同一标准）

    model = PPO(
        "MlpPolicy", env, verbose=1,
        tensorboard_log="./tb/ppo_pulse/", seed=args.seed,
        # 显式对齐 train_ppo.py 的默认值，保证可比
        n_steps=2048, batch_size=64, learning_rate=3e-4, gamma=0.99,
    )
    model.learn(total_timesteps=1_000_000)
    model.save(f"ppo_auv_depth_pulse_seed{args.seed}")
    env.save(f"vecnormalize_ppo_pulse_seed{args.seed}.pkl")
    print(f"完成: ppo_auv_depth_pulse_seed{args.seed}.zip")


if __name__ == "__main__":
    main()

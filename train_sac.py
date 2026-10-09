"""SAC v4：外力脉冲环境下训练（今晚唯一一发，~4-5h @ gradient_steps=2）

用法:
    python train_sac.py                  # 默认 500k 步, gradient_steps=2
    python train_sac.py --gsteps 1       # 时间紧时减半到 ~2.1h
"""
import argparse
import os

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)

from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecNormalize,
    sync_envs_normalization,
)
from envs.auv_depth_env import AuvDepthEnv
from envs.pulse_env import ForcePulseEnv

LOG_DIR = "./tb/sac_v4"
CKPT_DIR = "./checkpoints"


def make_train_env():
    return ForcePulseEnv(
        pulse_force=0.3,
        pulse_duration=10,
        pulses_per_episode=(1, 2),
        first_pulse_after=100,
        obs_noise_std=0.01,
    )


def make_eval_env():
    return AuvDepthEnv()  # 标称环境（无扰动无噪声），训练期评估用


class SyncNormCallback(BaseCallback):
    """把训练环境的 VecNormalize 统计量同步到评估环境"""

    def __init__(self, source_env, target_env):
        super().__init__()
        self.source_env = source_env
        self.target_env = target_env

    def _on_step(self) -> bool:
        sync_envs_normalization(self.source_env, self.target_env)
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--gsteps", type=int, default=2, help="gradient_steps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)

    train_env = DummyVecEnv([make_train_env])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False, training=False, clip_obs=10.0
    )

    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=500_000,
        learning_starts=10_000,   # 关键修复: 默认 100 太小, 早期 Q 被噪声样本主导
        batch_size=256,
        gamma=0.98,               # 压低跨脉冲事件的 bootstrap 方差
        tau=0.005,
        ent_coef="auto",
        gradient_steps=args.gsteps,
        train_freq=1,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log=LOG_DIR,
        seed=args.seed,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(CKPT_DIR, "sac_v4_best"),
        log_path=os.path.join(CKPT_DIR, "sac_v4_eval"),
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
    )
    sync_cb = SyncNormCallback(train_env, eval_env)
    ckpt_cb = CheckpointCallback(
        save_freq=50_000, save_path=CKPT_DIR, name_prefix="sac_v4"
    )

    print(f"开始训练 SAC v4（外力脉冲, {args.steps} 步, gradient_steps={args.gsteps}）...")
    model.learn(total_timesteps=args.steps, callback=[eval_cb, sync_cb, ckpt_cb])
    model.save("sac_auv_depth_pulse_v4.zip")
    train_env.save("vecnormalize_v4.pkl")  # 评估时必须一起加载!
    print("完成: sac_auv_depth_pulse_v4.zip + vecnormalize_v4.pkl")


if __name__ == "__main__":
    main()

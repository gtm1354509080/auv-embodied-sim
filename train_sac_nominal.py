"""SAC 标称环境训练版（用于 Fig 2 训练曲线对标）

相对 DeepSeek 版的修正：
  加回 SyncNormCallback —— EvalCallback 使用独立的 VecNormalize(training=False)
  eval_env，不同步则 obs_rms 恒为初始值 = 评估时不归一化，而策略按归一化观测
  训练，导致 TensorBoard 的 eval/mean_reward 曲线错误、best_model 选错。
  （最终保存的模型 zip + train_env 的 pkl 不受影响，但评估曲线是废的。）
"""
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

LOG_DIR = "./tb/sac_nominal"
CKPT_DIR = "./checkpoints"


def make_env():
    return AuvDepthEnv()


class SyncNormCallback(BaseCallback):
    """把训练环境的 VecNormalize 统计同步到评估环境"""

    def __init__(self, source_env, target_env):
        super().__init__()
        self.source_env = source_env
        self.target_env = target_env

    def _on_step(self) -> bool:
        sync_envs_normalization(self.source_env, self.target_env)
        return True


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    train_env = DummyVecEnv([make_env])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env = DummyVecEnv([make_env])
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False, training=False, clip_obs=10.0
    )

    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=500_000,
        learning_starts=10_000,
        batch_size=256,
        gamma=0.98,
        tau=0.005,
        ent_coef="auto",
        gradient_steps=2,
        train_freq=1,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log=LOG_DIR,
        seed=0,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"{CKPT_DIR}/sac_nominal_best",
        log_path=f"{CKPT_DIR}/sac_nominal_eval",
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
    )
    sync_cb = SyncNormCallback(train_env, eval_env)
    ckpt_cb = CheckpointCallback(
        save_freq=50_000, save_path=CKPT_DIR, name_prefix="sac_nominal"
    )

    model.learn(total_timesteps=500_000, callback=[eval_cb, sync_cb, ckpt_cb])
    model.save("sac_auv_depth_nominal.zip")
    train_env.save("vecnormalize_sac_nominal.pkl")
    print("完成: sac_auv_depth_nominal.zip + vecnormalize_sac_nominal.pkl")


if __name__ == "__main__":
    main()

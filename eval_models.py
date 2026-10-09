import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from envs.auv_depth_env import AuvDepthEnv

gym.register(id="AuvDepth-v0", entry_point=AuvDepthEnv, max_episode_steps=500)


def make_env():
    return AuvDepthEnv()


def run_scene(model, vecnorm_path, target_depth=3.0, steps=500):
    env = DummyVecEnv([make_env])
    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False

    env.reset()
    
    # 设置目标深度
    for e in env.envs:
        e.target_depth = target_depth

    depths = [env.envs[0].depth]
    for step in range(steps):
        action, _ = model.predict(env.get_original_obs(), deterministic=True)
        obs, reward, done, info = env.step(action)
        depths.append(env.envs[0].depth)
        if done[0]:
            break
    return np.array(depths)


if __name__ == "__main__":
    ppo_model = PPO.load("ppo_auv_depth_v2_seed0")
    sac_model = SAC.load("sac_auv_depth_pulse_v4")

    ppo_depths = run_scene(ppo_model, "vecnormalize_ppo_v2_seed0.pkl", target_depth=3.0)
    sac_depths = run_scene(sac_model, "vecnormalize_v4.pkl", target_depth=3.0)

    np.savez("results_ppo_step_v2.npz", depths=ppo_depths)
    np.savez("results_sac_step_v4.npz", depths=sac_depths)

    print(f"PPO: 最终深度={ppo_depths[-1]:.4f}m, 步数={len(ppo_depths)}")
    print(f"SAC: 最终深度={sac_depths[-1]:.4f}m, 步数={len(sac_depths)}")
import gymnasium as gym
import numpy as np


class DomainRandomizeWrapper(gym.Wrapper):
    """
    域随机化：reset 时采样物理参数（修正版）

    相对 DeepSeek 版的修正：
      1. 随机数全部走 self.env.np_random —— seed 可复现
         （原版用 np.random，seed 失效，与旧 PulseTrainEnv 同病）
      2. 删除 buoyancy 随机化 —— v2 动力学中 buoyancy 不参与 acc 计算，
         随机化它是空操作，论文参数表必须与实际生效参数一致
      3. 观测噪声裁剪到 observation_space，并把采样到的参数写进 info（便于记录）
    """

    def __init__(self, env, mass_std=0.10, drag_std=0.20, noise_std=0.01):
        super().__init__(env)
        self.mass_std = mass_std
        self.drag_std = drag_std
        self.noise_std = noise_std
        self.nominal_mass = env.mass
        self.nominal_drag = env.drag_coeff

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        rng = self.env.np_random  # env.reset() 之后可用，seed 可复现
        self.env.mass = self.nominal_mass * (1 + rng.uniform(-self.mass_std, self.mass_std))
        self.env.drag_coeff = self.nominal_drag * (1 + rng.uniform(-self.drag_std, self.drag_std))
        info["mass"] = self.env.mass
        info["drag_coeff"] = self.env.drag_coeff
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.noise_std > 0.0:
            noise = self.env.np_random.normal(0.0, self.noise_std, size=obs.shape)
            obs = np.clip(
                obs + noise,
                self.env.observation_space.low,
                self.env.observation_space.high,
            ).astype(np.float32)
        return obs, reward, terminated, truncated, info

import numpy as np

from envs.auv_depth_env import AuvDepthEnv


class ForcePulseEnv(AuvDepthEnv):
    """
    外力脉冲扰动环境 v4（替代崩溃的 PulseTrainEnv）

    与旧 PulseTrainEnv 的根本区别:
      1. 脉冲 = 持续 pulse_duration 步的恒定外力，经基类 ext_force 通道
         在动力学积分【之前】进入状态更新 —— 返回的 obs 真实反映脉冲后的状态，
         (s, a, r, s') 闭合，不再向 replay buffer 写入断裂的 transition
      2. 全部随机数走 self.np_random，seed 可复现
      3. 每回合按 reset 时预生成的时刻表触发 1~2 次脉冲（替代每步伯努利采样），
         强度与频次可控，且前 first_pulse_after 步不扰动（让 agent 先到位）
      4. 观测噪声在状态更新之后、返回之前注入，并裁剪到 observation_space
    """

    def __init__(
        self,
        pulse_force: float = 0.3,      # 外力幅值（thrust∈[-1,1] 即 F_max=1 → 0.3×F_max）
        pulse_duration: int = 10,      # 持续步数（10 步 × dt=0.1s = 1.0s）
        pulses_per_episode=(1, 2),
        first_pulse_after: int = 100,  # 前 100 步不打脉冲
        obs_noise_std: float = 0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pulse_force = pulse_force
        self.pulse_duration = pulse_duration
        self.pulses_per_episode = tuple(pulses_per_episode)
        self.first_pulse_after = first_pulse_after
        self.obs_noise_std = obs_noise_std
        self._schedule = []
        self._pulse_left = 0
        self._pulse_f = 0.0
        self.pulse_count = 0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._pulse_left = 0
        self._pulse_f = 0.0
        self.pulse_count = 0
        lo, hi = self.pulses_per_episode
        n = int(self.np_random.integers(lo, hi + 1))
        start_min = self.first_pulse_after
        start_max = self.max_steps - self.pulse_duration - 1
        self._schedule = (
            sorted(self.np_random.integers(start_min, start_max, size=n).tolist())
            if start_max > start_min
            else []
        )
        return obs, info

    def step(self, action):
        # 1) 触发新脉冲
        if self._schedule and self.current_step >= self._schedule[0]:
            self._schedule.pop(0)
            self._pulse_left = self.pulse_duration
            sign = float(self.np_random.choice([-1.0, 1.0]))
            self._pulse_f = sign * self.pulse_force * self.np_random.uniform(0.8, 1.0)
            self.pulse_count += 1

        # 2) 经 ext_force 通道施加（super().step() 内完成积分后自动清零）
        if self._pulse_left > 0:
            self.ext_force = self._pulse_f
            self._pulse_left -= 1

        obs, reward, terminated, truncated, info = super().step(action)

        # 3) 观测噪声（在状态更新之后、返回之前）
        if self.obs_noise_std > 0.0:
            noise = self.np_random.normal(0.0, self.obs_noise_std, size=obs.shape)
            obs = np.clip(
                obs + noise,
                self.observation_space.low,
                self.observation_space.high,
            ).astype(np.float32)

        info["pulse_active"] = self._pulse_left > 0
        info["pulse_count"] = self.pulse_count
        return obs, reward, terminated, truncated, info

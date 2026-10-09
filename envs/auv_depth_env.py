import gymnasium as gym
from gymnasium import spaces
import numpy as np


class AuvDepthEnv(gym.Env):
    """
    AUV 深度控制环境 v2（修复版）

    状态: [depth_error, vel_z, pitch, roll, last_action] (5维)
    动作: [thruster] (1维, [-1, 1])
    动力学: z¨ = (thrust + ext_force - c·ż) / m，奖励前三项与 v1 完全一致。

    相对 v1 的变更（均不改变核心动力学与奖励前三项）:
      1. 进入目标带 (|e|<hold_band) 不再终止回合，改为每步 +hold_bonus 保持奖励
         —— 任务从"碰到一次"修正为"持续保持"，消除随机终止信号对 SAC bootstrap 的污染
      2. 姿态超限不再终止，clip 到 ±30° 并施加固定惩罚 -att_penalty
      3. 随机数统一走 self.np_random，seed 完全可复现
      4. 新增 ext_force 外力通道（标称环境恒为 0，动力学与 v1 逐位一致）
      5. fix_drag_sign=True 修正 v1 阻尼符号笔误（v1 实为正反馈）；
         如需复现 v1 行为传 fix_drag_sign=False
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    ATT_LIMIT = np.pi / 6  # 30°

    def __init__(
        self,
        max_depth: float = 5.0,
        dt: float = 0.1,
        max_steps: int = 500,
        hold_band: float = 0.05,
        hold_bonus: float = 1.0,
        att_penalty: float = 1.0,
        terminate_on_success: bool = False,   # v1 行为: True
        terminate_on_attitude: bool = False,  # v1 行为: True
        fix_drag_sign: bool = True,           # v1 行为(物理错误的正反馈): False
    ):
        super().__init__()
        self.max_depth = max_depth
        self.dt = dt
        self.max_steps = max_steps
        self.hold_band = hold_band
        self.hold_bonus = hold_bonus
        self.att_penalty = att_penalty
        self.terminate_on_success = terminate_on_success
        self.terminate_on_attitude = terminate_on_attitude
        self.fix_drag_sign = fix_drag_sign

        self.target_depth = 2.5
        self.current_step = 0

        self.mass = 1.0
        self.drag_coeff = 0.5
        self.buoyancy = 0.0

        self.depth = 0.0
        self.vel_z = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.last_action = 0.0
        self.ext_force = 0.0  # 外力通道（扰动子类使用），标称恒 0

        self.observation_space = spaces.Box(
            low=np.array([-10.0, -5.0, -np.pi, -np.pi, -1.0], dtype=np.float32),
            high=np.array([10.0, 5.0, np.pi, np.pi, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def _get_obs(self):
        depth_error = self.target_depth - self.depth
        return np.array(
            [depth_error, self.vel_z, self.pitch, self.roll, self.last_action],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.depth = self.np_random.uniform(0.0, self.max_depth)
        self.vel_z = 0.0
        self.pitch = self.np_random.uniform(-0.1, 0.1)
        self.roll = 0.0
        self.last_action = 0.0
        self.ext_force = 0.0
        self.target_depth = self.np_random.uniform(1.0, self.max_depth)
        return self._get_obs(), {"depth": self.depth, "target": self.target_depth}

    def step(self, action):
        self.current_step += 1
        thrust = float(np.clip(action[0], -1.0, 1.0))
        prev_action = self.last_action

        if self.fix_drag_sign:
            # 正确的阻尼: 与速度反向
            acc_z = (thrust + self.ext_force - self.drag_coeff * self.vel_z) / self.mass
        else:
            # v1 原式: 展开为 thrust + c·v (正反馈, 物理错误, 仅为复现保留)
            drag_force = -self.drag_coeff * self.vel_z
            acc_z = (thrust + self.ext_force - drag_force) / self.mass

        self.vel_z += acc_z * self.dt
        self.depth += self.vel_z * self.dt
        self.ext_force = 0.0  # 外力仅作用当前步，由子类在每步重新设置

        self.pitch += 0.01 * np.sin(self.current_step * 0.1)
        self.roll += 0.005 * np.cos(self.current_step * 0.1)

        if self.depth < 0.0:
            self.depth = 0.0
            self.vel_z = max(0.0, self.vel_z)
        if self.depth > self.max_depth + 1.0:
            self.depth = self.max_depth + 1.0
            self.vel_z = min(0.0, self.vel_z)

        self.last_action = thrust

        depth_error = self.target_depth - self.depth
        w1, w2, w3, w4 = 1.0, 0.1, 0.01, 0.001
        action_diff = thrust - prev_action
        reward = -(
            w1 * abs(depth_error)
            + w2 * abs(self.vel_z)
            + w3 * abs(thrust)
            + w4 * action_diff**2
        )

        terminated = False
        truncated = False
        in_band = abs(depth_error) < self.hold_band

        # 修复点 1: 目标带内每步给保持奖励，默认不终止
        if in_band:
            reward += self.hold_bonus
            if self.terminate_on_success:
                terminated = True

        # 修复点 2: 姿态超限 clip + 固定惩罚，默认不终止
        if abs(self.pitch) > self.ATT_LIMIT or abs(self.roll) > self.ATT_LIMIT:
            reward -= self.att_penalty
            if self.terminate_on_attitude:
                terminated = True
            else:
                self.pitch = float(np.clip(self.pitch, -self.ATT_LIMIT, self.ATT_LIMIT))
                self.roll = float(np.clip(self.roll, -self.ATT_LIMIT, self.ATT_LIMIT))

        if self.current_step >= self.max_steps:
            truncated = True

        info = {
            "depth": self.depth,
            "target": self.target_depth,
            "error": depth_error,
            "in_band": in_band,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        pass

import gymnasium as gym
import numpy as np
from envs.auv_depth_env import AuvDepthEnv
from envs.pid_controller import PIDController

gym.register(id="AuvDepth-v0", entry_point=AuvDepthEnv, max_episode_steps=500)

def run_pid_scene(env, controller, target_depth, scene_type="step", steps=500):
    obs, info = env.reset()
    env.target_depth = target_depth
    controller.reset()
    depths = [env.depth]
    actions = []
    
    for step in range(steps):
        if scene_type == "sine":
            current_target = target_depth + 1.0 * np.sin(0.05 * step)
        else:
            current_target = target_depth
            
        action = controller.compute(current_target, env.depth)
        
        # 【修改点1】扰动提前到第20步，幅度加大到3.0
        if scene_type == "impulse" and step == 20:
            env.vel_z +=0.2
            
        obs, reward, terminated, truncated, info = env.step(np.array([action], dtype=np.float32))
        depths.append(env.depth)
        actions.append(action)
        
        # 【修改点2】注释掉这行！
        # if terminated or truncated:
        #     break
            
    return np.array(depths), np.array(actions)

if __name__ == "__main__":
    env = AuvDepthEnv()
    pid = PIDController(kp=2.0, ki=0.0, kd=1.0, dt=0.1)

    # 阶跃
    depths_step, actions_step = run_pid_scene(env, pid, target_depth=3.0, scene_type="step")
    np.savez("results_pid_step.npz", depths=depths_step, actions=actions_step, target=3.0)

    # 脉冲
    depths_impulse, actions_impulse = run_pid_scene(env, pid, target_depth=3.0, scene_type="impulse")
    np.savez("results_pid_impulse.npz", depths=depths_impulse, actions=actions_impulse, target=3.0)
    
    print("PID 阶跃 + 脉冲数据已保存。")
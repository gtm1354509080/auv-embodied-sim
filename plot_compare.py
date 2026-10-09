import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['lines.linewidth'] = 1.2

def pad_to_length(arr, length=500):
    if len(arr) < length:
        return np.pad(arr, (0, length - len(arr)), 'edge')
    return arr[:length]

data_pid = pad_to_length(np.load("results_pid_step.npz")["depths"])
data_ppo = pad_to_length(np.load("results_ppo_step.npz")["depths"])
data_sac = pad_to_length(np.load("results_sac_step.npz")["depths"])

# 只取前 100 步，聚焦瞬态响应
zoom = 100
time_steps = np.arange(zoom)

plt.figure(figsize=(3.5, 2.5))
plt.plot(time_steps, data_pid[:zoom], label='PID', color='gray', linestyle='--')
plt.plot(time_steps, data_ppo[:zoom], label='PPO', color='#0072BD', linestyle='-')
plt.plot(time_steps, data_sac[:zoom], label='SAC', color='#D95319', linestyle='-')
plt.axhline(y=3.0, color='red', linestyle=':', linewidth=1.2, label='Target (3.0m)')

plt.xlabel('Time Step')
plt.ylabel('Depth (m)')
plt.xlim(0, zoom)
plt.ylim(0, 4.0)

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right', frameon=False)

plt.tight_layout()
plt.savefig('fig_depth_comparison.pdf', bbox_inches='tight')
plt.savefig('fig_depth_comparison.png', dpi=300, bbox_inches='tight')
print("对比图已保存。")
plt.show()
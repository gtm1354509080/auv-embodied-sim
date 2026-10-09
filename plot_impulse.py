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

data_pid = pad_to_length(np.load("results_pid_impulse.npz")["depths"])
data_ppo = pad_to_length(np.load("results_ppo_impulse.npz")["depths"])
data_sac = pad_to_length(np.load("results_sac_impulse.npz")["depths"])

time_steps = np.arange(500)

plt.figure(figsize=(3.5, 2.5))
plt.plot(time_steps, data_pid, label='PID', color='gray', linestyle='--')
plt.plot(time_steps, data_ppo, label='PPO', color='#0072BD', linestyle='-')
plt.plot(time_steps, data_sac, label='SAC', color='#D95319', linestyle='-')
plt.axhline(y=3.0, color='red', linestyle=':', linewidth=1.2, label='Target (3.0m)')

# 标出扰动发生的时间
plt.axvline(x=100, color='black', linestyle='-.', linewidth=1.0, alpha=0.7, label='Disturbance')

plt.xlabel('Time Step')
plt.ylabel('Depth (m)')
plt.xlim(0, 500)
plt.ylim(0, 5.0)

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='lower right', frameon=False)

plt.tight_layout()
plt.savefig('fig_impulse_robustness.pdf', bbox_inches='tight')
plt.savefig('fig_impulse_robustness.png', dpi=300, bbox_inches='tight')
print("抗扰动对比图已保存。")
plt.show()
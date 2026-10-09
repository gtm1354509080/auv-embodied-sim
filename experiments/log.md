## [2026-10-09] Table 2/3 定稿（seed 5678, 20 eps）

### Table 2: Nominal performance
| Method | Reward | MAE (m) | Hold |
|---|---|---|---|
| PID (Kp=3.0, Kd=1.5) | 446.77 | 0.0469 | 0.944 |
| PPO (nominal-trained) | 450.30 | 0.0488 | 0.954 |
| DR-PPO | 449.13 | 0.0474 | 0.950 |
| PPO (pulse-trained) | 455.77 | 0.0457 | 0.961 |
| SAC (pulse-trained) | 453.79 | 0.0499 | 0.962 |

### Table 3: Hold ratio robustness matrix
| Method | nominal | pulse | mass+15% | drag−25% |
|---|---|---|---|---|
| PID | 0.944 | 0.906 | 0.937 | 0.942 |
| PPO (nominal) | 0.954 | 0.954 | 0.947 | 0.951 |
| DR-PPO | 0.950 | 0.950 | 0.942 | 0.945 |
| PPO (pulse) | 0.961 | 0.961 | 0.955 | 0.958 |
| SAC (pulse) | 0.962 | 0.962 | 0.958 | 0.960 |

### Key findings
1. Training regime matters more than algorithm choice: pulse-trained PPO and SAC both reach 0.96+; algorithm difference ≈ 0.1pp.
2. DR provides no significant gain: DR-PPO (0.942-0.945) vs nominal PPO (0.947-0.951) in offset scenarios.
3. PID degrades under pulse (0.944 → 0.906) despite grid-search tuning; RL stays flat.
4. PID MAE (0.0469) is lowest, but only in nominal — RL wins on hold ratio and robustness.
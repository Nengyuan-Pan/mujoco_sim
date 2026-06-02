# RM-65 MPC+iLQR 网球击打 Pipeline

> 脚本文件：`scripts/rm65_mpc_ilqr_5_5.py`
> Robot 模型：`src/robot/rm65_model.xml`
> 方法：模型预测控制（MPC）+ 迭代线性二次调节器（iLQR）

---

## 一、整体 Pipeline

```
加载配置 → 场景初始化 → 球轨迹生成 → 击打点搜索 → 初始 Warm-start
  → MPC 主循环（每步可选重规划）→ 击打后仿真 → 评估输出 → 可视化回放
```

### 1.1 配置加载

加载 `configs/default.yaml` + `configs/mpc.yaml`，合并为完整配置字典。关键参数：

| 参数 | 来源 | 说明 |
|------|------|------|
| `dt` | `sim.dt` | 仿真步长 0.005s |
| `gravity` | `ball.gravity` | 重力 [0, 0, -9.81] |
| `racket_speed` | `hitting.racket_speed` | 期望击打速度 5.0 m/s |
| `horizon` | `ilqr.horizon` | 离线规划步数 80 |
| `workspace_radius` | `hitting.workspace_radius` | 0.85m |

### 1.2 场景初始化

| 变量 | 值 | 说明 |
|------|-----|------|
| `total_horizon` | 200 | MPC 总步数，200×5ms=1.0s |
| `replan_interval` | 10 | 每 10 步重规划一次 |
| `fixed_horizon` | 30 | iLQR 短地平线（可覆盖） |
| `init_q` | `[0.0, -1.2, 1.8, -0.6, 0.0, 0.0]` | 初始姿态 |
| `shoulder_pos` | `[-0.1, -0.227, 1.303]` | 肩关节世界坐标 |

### 1.3 球轨迹生成

调用 `generate_ball_to_target_box()`：

```
target_center = [-0.82765693, -0.47411682, 0.86947444] (世界坐标)
target_offset = 0.2  (±0.2m 立方体)
hit_time      = total_horizon * dt * U(0.3, 0.4)  → 约 0.15~0.20s

球从 -Y 方向 ~5m 远飞来，初速约 10~14 m/s
```

### 1.4 击打点搜索

`find_hitting_point_physics(env, p0, v0, shoulder_pos, workspace_radius, horizon)` 在 MuJoCo 中前向仿真球轨迹，遍历所有步找最佳击打点：
- 球距肩关节最近
- 高度在工作空间内 (0.3 < z < 肩高+0.55)
- 评分 = dist + height_penalty - front_bonus

运行机制如下：
1. 预测球轨迹
调用 env.predict_ball_trajectory(ball_pos, ball_vel, horizon) — 用 MuJoCo 物理引擎前向仿真球在重力+地面弹跳下的 horizon 步轨迹（禁用臂碰撞，避免球拍干扰）。输出每步的 (位置, 速度)。

2. 遍历所有步，筛选候选击打点
对每步 k，三个条件同时满足才进入候选：

dist  = ‖球 - 肩关节‖ < workspace_radius (0.90m)
z     = 球高度 > 0.3m（地面以上）
dz    = 球高度 - 肩高度 ∈ (-0.60, +0.55)
3. 评分选最优
对每个候选点计算得分（越低越好）：

score = dist                           ← 距离越近越好
      + height_above² × 5.0           ← 太高扣分（肩上方 0.2m 以上）
      - max(0, 球_X - 肩_X) × 0.3     ← 前方加分（X 方向自由度更大）
返回最小 score 的候选点，输出 {k_hit, p_hit, v_ball_hit, dist}。没有任何候选则返回 None（球不可达）。

关键点：score 的 front_bonus 项使得算法偏好球在机器人前方的击打点，因为 RM-65 臂在前方 (X+) 的可达空间更大。
返回：`{k_hit, p_hit, v_ball_hit, dist}`

### 1.5 初始 Warm-start

**后摆模式**（默认启用）：
1. IK 求解击打姿态 `q_hit`（关节6固定为-2.8 rad）
2. 反解击打时刻关节速度 `qdot_hit_ik = J_p^+ @ v_des`
3. 生成后摆轨迹：关节1用**五次多项式**从初态经后摆最低点再到击打姿态，其余关节线性插值
4. 后摆幅度自适应：球越远→后摆越大（0.4~1.0 rad）
5. 用 PD 跟踪后摆轨迹生成初始控制序列

**非后摆模式**：雅可比转置 `u = J_p^T @ (p_target - p_ee)` 生成初始控制

---

## 二、MPC 主循环

### 2.1 两步结构

每步（0..199）执行：

1. 获取当前球状态 `(ball_pos, ball_vel)`
2. 判断是否需要重规划（step%10==0 或 buffer耗尽）
3. 从 `U_buffer` 取一步控制指令
4. `env.step_full(u_cmd)` 执行一步仿真
5. 碰撞检测 → 弹性反弹

### 2.2 重规划逻辑

当触发重规划时：

```
STEP A: 重新搜索击打点
  find_hitting_point_physics() → (k_hit_new, p_hit_new, v_ball_hit_new)

  k_hit 暴跌检测：如果 k_hit_new < max(10, k_hit_old//4) 且旧值>30
    → 用线性预测修正 k_hit = k_hit_old - replan_interval

STEP B: 分支策略

  ┌─ 远距 (k_hit > far_threshold = k_hit_total) ──────────────┐
  │  compute_jacobian_init_control() 生成整个 pacing horizon   │
  │  的控制序列，直接填充 U_buffer（不调用 iLQR）               │
  └─────────────────────────────────────────────────────────────┘

  ┌─ 近距 (k_hit <= far_threshold) ────────────────────────────┐
  │  a) 权重调度：基于实际位置误差线性插值 Q_p/Q_v             │
  │     pos_err > 0.10m → Q_p=5.0, Q_v=3.0                    │
  │     pos_err < 0.05m → Q_p=5.0, Q_v=50.0                   │
  │                                                             │
  │  b) 更新代价函数目标                                       │
  │     p_follow = p_hit + hit_shift · d_hat  (随挥偏移 0.01m) │
  │     v_hit_desired, n_des                                    │
  │                                                             │
  │  c) 迭代次数自适应：                                        │
  │     - 首次规划: 50 次                                       │
  │     - k_hit ≤ near_threshold: 15 次                         │
  │     - 常规: 8 次                                            │
  │                                                             │
  │  d) Warm-start 生成                                        │
  │     - U_prev 够长 → resample_control_sequence() 插值续用    │
  │     - 否则 → 后摆 PD 跟踪 / 雅可比转置                      │
  │                                                             │
  │  e) iLQR 求解                                              │
  │     solver.solve_few_iters(env, cost_fn, x, U_warm, max_iter)│
  │                                                             │
  │  f) U_buffer = U_mpc[:replan_interval]   (执行用)           │
  │     U_prev   = U_mpc[replan_interval:]   (下次 warm-start)   │
  └─────────────────────────────────────────────────────────────┘
```

### 2.3 碰撞处理

- 当 `k_hit_new ≤ 10` 时启用球-球拍碰撞（`env.set_arm_collision(True)`）
- MuJoCo 检测到 `ball` ↔ `racket_face` / `racket_handle` contact 后：
  - 用碰撞**前**的球速计算弹性反弹：`v_new = v_pre - (1+e) · (v_rel_pre·n_hat) · n_hat`，`e=0.8`
  - 继续 5 步仿真后退出 MPC

### 2.4 R 退火

控制代价权重沿时间线性衰减：

```
R_schedule[k] = R           for k < (1-r_decay_ratio)*N
R_schedule[k] = R·(1-s)     for k ≥ (1-r_decay_ratio)*N
  其中 s = (k - (1-ratio)*N) / (ratio*N)  (线性衰减至 0)
```

关节1 (shoulder_pan) 额外指数加速衰减：`exp(-3·s)`。

---

## 三、iLQR 算法流程

### 3.1 求解器核心循环

```
输入: x0, U_init
输出: X_opt (N+1×12), U_opt (N×6)

1. X = rollout(env, x0, U)          # 前向仿真得到名义轨迹
2. cost_old = Σ running_cost + terminal_cost

FOR iteration = 1..max_iter:
    3. As, Bs, fs = linearize(X, U)  # 沿轨迹线性化动力学
    4. l_xs, l_us, l_xxs, ... = running_cost_derivatives(X, U)
       l_x_N, l_xx_N = terminal_derivatives(X[-1])

    5. Ks, ks = backward_pass(As, Bs, l_xs, l_us, l_xxs, ..., mu)
       若失败 → 增大正则化 mu *= δ₀，continue

    6. forward_pass_with_linesearch(X, U, Ks, ks, α_list, cost_old)
        → (X_new, U_new, cost_new, accepted)
       若 accepted → X=X_new, U=U_new, mu /= δ₀
       若 rejected → mu *= δ₀

    7. 收敛判断：相对改进 < tol → break
```

### 3.2 MPC 简化版 (`solve_few_iters`)

与完整版区别：
- 迭代次数由 MPC 决定（8/15/50），不检查收敛
- 默认跳过线搜索（固定步长 α=0.5）
- 只要轨迹有限即接受更新（不强制 cost 下降）
- 依赖重规划纠错（re-planning correction）

### 3.3 后向传递（Backward Pass）

从终端 `k=N-1` 到 `k=0`：

```
V_x = l_x_N,  V_xx = l_xx_N

FOR k = N-1 → 0:
    Q_x  = l_xs[k]  + A_k^T · V_x
    Q_u  = l_us[k]  + B_k^T · V_x
    Q_xx = l_xxs[k] + A_k^T · V_xx · A_k
    Q_ux = l_uxs[k] + B_k^T · V_xx · A_k
    Q_uu = l_uus[k] + B_k^T · V_xx · B_k + μ·I   (正则化)

    K_k = -Q_uu_inv @ Q_ux     # 6×12 反馈增益
    k_k = -Q_uu_inv @ Q_u      # 6×1  前馈增益

    V_x  = Q_x  - Q_ux^T @ Q_uu_inv @ Q_u
    V_xx = Q_xx - Q_ux^T @ Q_uu_inv @ Q_ux
```

### 3.4 前向传递（Forward Pass）

```
X_new[0] = X[0]
FOR k = 0 → N-1:
    dx = X_new[k] - X[k]
    U_new[k] = U[k] + α·k_k + K_k·dx     (裁剪到 ctrlrange)
    X_new[k+1] = env.step_from_state(X_new[k], U_new[k])
```

线搜索：遍历 α ∈ [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]，取第一个降低总代价的步长。

### 3.5 动力学线性化（解析法）

使用 MuJoCo 原生 API 解析求导（可选有限差分）：

```
A_c = [  0,           I        ]
      [ -M⁻¹·H_q,  -M⁻¹·H_qdot ]
B_c = [  0    ]
      [ M⁻¹   ]
A   = I + A_c·dt          (欧拉离散)
B   = B_c·dt

其中:
  M  = mj_fullM()        质量矩阵
  H_q  = ∂h/∂q           偏置力对 q 的偏导
  H_qdot = ∂h/∂qdot       偏置力对 qdot 的偏导
  h(q,qdot) = C·qdot + g(q)  科氏力+离心力+重力
```

### 3.6 代价函数

**终端代价**（k=N）：
```
l_N(x) = ½·(p_ee - p_hit)^T·Q_p·(p_ee - p_hit)
       + ½·(v_ee - v_hit)^T·Q_v·(v_ee - v_hit)
       + ½·Q_n·‖n - n_des‖²
```

**运行代价**（k=0..N-1）：
```
l_k(x,u) = ½·u^T·R_k·u                    (控制力矩)
         + ½·(p_ee - p_target)^T·Q_p_run·(p_ee - p_target)  (运行位置)
         + ½·Σ Q_joint_j·(q_j - q_des_j)^2                   (关节跟踪)
         + ½·Σ Q_p_run·(p_ee - p_ball)^T·(p_ee - p_ball)     (球追踪)
```

其中 `R_k` 时变（R退火），`Q_p`, `Q_v` 随距离动态缩放。

---

## 四、辅助函数

| 函数 | 功能 |
|------|------|
| `compute_jacobian_init_control` | 雅可比转置初始控制：`J_p^T @ Kp·(p_hit-p_ee) - Kd·qdot` |
| `generate_backswing_warm_start` | 后摆 warm-start：IK→五次多项式轨迹→PD跟踪控制 |
| `compute_joint1_backswing_trajectory` | 关节1后摆五次多项式（6边界条件） |
| `resample_control_sequence` | 控制序列线性插值重采样 |
| `compute_r_schedule` | R退火调度生成 |
| `fix_joint5_control` | 将关节6控制力矩替换为PD保持 |
| `ik_pd_step` | 单步IK+PD（buffer耗尽兜底） |
| `visualize_rm65_result` | MuJoCo查看器可视化回放（含灯光） |

---

## 五、可视化

`visualize_rm65_result()` 在 MuJoCo 被动查看器中以真实速度回放完整轨迹：
- 5灯布光（天光、主光、补光、背光）+ 球跟随灯
- 可配相机视角、回放倍率、循环模式
- 支持击打后球飞出效果

---

## 六、运行命令

```bash
# 基本击打
python scripts/rm65_mpc_ilqr_5_5.py --seed 42 --normal-flip --viewer

# 反弹球模式
python scripts/rm65_evaluate.py --bounce --normal-flip --viewer

# 评估模式（含图表）
python scripts/rm65_evaluate.py --seed 42 --normal-flip

# 关节限速模式
python scripts/rm65_joint_limit.py --seed 42 --clip-vel --racket-speed 2.0
```

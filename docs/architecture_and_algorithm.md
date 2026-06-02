# RM-65 实时 MPC 击球系统：架构与算法详解

## 概述

`rm65_mpc_tube_constraint_realtime.py` 是一个基于 **iLQR（迭代线性二次调节器）** 的模型预测控制（MPC）系统，用于控制 RM-65B 六自由度机械臂在网球飞来时自动挥拍击球。系统采用**混合实时策略**：远段使用 Jacobian 转矩控制（快速粗略），近段使用 iLQR 精细规划，满足 200Hz 控制频率下的实时约束。

---

## 1. 系统架构总览

```
┌─────────────────────────────────────────────────────────┐
│                    主循环 (200Hz)                         │
│                                                          │
│  ┌──────────┐    ┌───────────┐    ┌───────────────────┐  │
│  │ 球状态感知 │───>│ 击打点计算  │───>│ 控制策略选择       │  │
│  └──────────┘    └───────────┘    │                     │  │
│                                    │ k_hit > 50: JT控制 │  │
│                                    │ k_hit ≤ 50: iLQR   │  │
│                                    └─────────┬─────────┘  │
│                                              │            │
│                                    ┌─────────▼─────────┐  │
│                                    │  安全滤波器        │  │
│                                    │  ① X平面墙约束     │  │
│                                    │  ② 关节速度/力矩限制│  │
│                                    └─────────┬─────────┘  │
│                                              │            │
│                                    ┌─────────▼─────────┐  │
│                                    │  MuJoCo 物理仿真    │  │
│                                    │  env.step_full(u)   │  │
│                                    └───────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 1.1 状态向量

| 符号 | 维度 | 含义 |
|------|------|------|
| `x` | 12 | 状态向量 = [q(6), qdot(6)]，关节角度 + 关节角速度 |
| `u` | 6 | 控制向量 = τ(6)，6 个关节的力矩 |
| `k_hit` | 1 | 剩余击球步数（距球到达击打点的仿真步数） |
| `p_hit` | 3 | 击打点位置（笛卡尔坐标） |
| `v_ball_hit` | 3 | 球到达击打点时的速度 |

### 1.2 关键参数

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `dt` | 5ms | 仿真步长（200Hz） |
| `replan_interval` | 20（实时）/ 10（离线） | 每隔多少步触发一次重规划 |
| `far_threshold` | 50 | k_hit > 50 时使用 JT 控制 |
| `near_threshold` | max(50, k_hit_total/3) | 近段阈值 |
| `horizon` | 120 | iLQR 规划时域长度 |
| `first_plan_iters` | 30 | 首次冷启动 iLQR 迭代次数 |
| `max_iter_per_plan` | 10 | 稳态重规划 iLQR 迭代次数 |

---

## 2. 完整执行流程

### 2.1 初始化阶段（球到达前完成，不计入实时预算）

```
1. 加载配置 → 创建 MuJoCo 环境 → 设置机器人硬约束
2. 随机生成发球（位置、速度、方向）
3. 预测球轨迹 → 寻找击打点（find_hitting_point_physics）
   → 输出：k_hit_total（击球步数），p_hit（击球位置），v_ball_hit（球到达时速度）
4. IK 可达性后过滤：若击球点关节超限，搜索附近可行替代点
5. 构建 Tube 空间走廊（若启用）
6. 计算初始控制序列（Jacobian 转矩 或 引拍轨迹）
7. 首次 iLQR 规划（30 次迭代，~1300ms）→ 保存 U_prev
```

### 2.2 MPC 主循环（每步 5ms）

```
for step in range(total_horizon):
    ┌─ 1. 读取球状态，计算末端-球距离
    │
    ├─ 2. 判断是否需要重规划
    │     触发条件：step % replan_interval == 0 或 buffer 耗尽
    │
    ├─ 3. 【若需重规划】
    │     ├─ 重新预测击打点（基于当前球状态）
    │     ├─ 击球点可执行性过滤
    │     ├─ 分支选择：
    │     │   ├─ k_hit > 50（远段）：JT 控制，0ms
    │     │   └─ k_hit ≤ 50（近段）：iLQR 求解，50-80ms
    │     │      ├─ Warm-start 选择（U_prev 重用 / JT 初始化）
    │     │      ├─ solver.solve_few_iters(10 次迭代)
    │     │      └─ 保存 U_prev（尾部控制，供下次 warm-start）
    │     └─ 更新 U_buffer（控制指令缓冲）
    │
    ├─ 4. 从 U_buffer 取控制指令
    │     若 buffer 空 → JT 后备控制
    │
    ├─ 5. 硬约束层1：X 平面墙（防止手臂越过身体中线 X=0）
    │     试探 beta ∈ {1.0, 0.6, 0.3, 0.0}，选不越界的最大 beta
    │
    ├─ 6. 硬约束层2：安全滤波器（关节速度/力矩检查）
    │     check_one_step_feasibility → 若不通过，beta 缩放直到安全
    │
    ├─ 7. 物理仿真：env.step_full(u_final)
    │
    ├─ 8. 碰撞检测（k_hit ≤ 30 且距离 < 0.35m 时启用）
    │     检测到物理接触 → 计算弹性反弹
    │
    └─ 9. 实时节奏：sleep(dt - step_time)
```

---

## 3. 核心算法详解

### 3.1 击打点计算（find_hitting_point_physics）

**目标**：在球的预测轨迹上，找到机械臂可达的最佳击球点。

```python
# 伪代码
for k in range(1, total_horizon):
    ball_pos_k = predict_ball(ball_pos_0, ball_vel_0, k * dt)
    dist_to_shoulder = norm(ball_pos_k - shoulder_pos)
    height_diff = ball_pos_k[2] - shoulder_pos[2]
    if dist_to_shoulder < workspace_radius and 0.3 < ball_pos_k[2] < 1.5:
        return (k, ball_pos_k, ball_vel_k)  # (步数, 位置, 球速)
```

每次重规划时重新计算，因为实际球轨迹可能与预测有偏差。

### 3.2 击球点可执行性过滤（refine_hit_point）

即使物理上可达，某些击球点的 IK 解可能接近关节限位边缘。过滤逻辑：

1. 计算击球点的 IK 解
2. 检查所有关节到限位的最小裕度
3. 若裕度 < 3°，在 Tube 窗口内搜索更安全的替代击球时刻
4. 替换为裕度最大的候选点

此外还有一个动态过滤机制（HIT_SWAP / HIT_LOCK）：
- **HIT_SWAP**：运行中发现当前击球点的关节裕度不足 → 替换为更安全的时刻
- **HIT_LOCK**：k_hit ≤ 60 后锁定击球点不再替换，确保末端有足够时间收敛

### 3.3 混合控制策略

#### 远段（k_hit > 50）：Jacobian 转矩控制（JT）

```
u = gain × Jᵀ × (p_target - p_ee) - kd × qdot
```

- **不需要 iLQR 迭代**，只做一次矩阵乘法（<1ms）
- 效果：末端朝击球点方向粗略移动
- 足够好：球还远着，不需要精细轨迹

#### 近段（k_hit ≤ 50）：iLQR 求解

完整的后向-前向迭代优化：

```
后向传递（k = N → 0）:
  Q_x, Q_u, Q_xx, Q_ux, Q_uu = 展开的代价函数二阶导
  V_x, V_xx = 价值函数梯度/海森矩阵
  K_k = -Q_uu^{-1} × Q_ux   （反馈增益矩阵）
  k_k = -Q_uu^{-1} × Q_u     （前馈增益向量）

前向传递（k = 0 → N）:
  δu_k = K_k × δx_k + k_k × alpha  （alpha 为线搜索步长）
  x_{k+1} = dynamics(x_k, u_k + δu_k)
```

迭代策略根据 k_hit 和是否实时模式动态调整：

| 条件 | 迭代次数 | fast_lin | fp_limits |
|------|---------|----------|-----------|
| 首次规划 | 30 | False | None |
| k_hit > 30（近段外） | 10 | True | None |
| k_hit ≤ 30，实时 | 10 | False | robot_limits |
| k_hit ≤ 30，离线 | 20 | False | robot_limits |

### 3.4 Warm-start 机制

每次 iLQR 求解需要一个初始控制序列 `U_warm`，质量直接影响收敛速度。

**三级 warm-start 策略**（优先级从高到低）：

1. **上一次规划的尾部重用**（最优）
   - 条件：`len(U_prev) >= horizon_full // 3`
   - 方法：`resample_control_sequence(U_prev, new_horizon)`
   - 效果：上次轨迹形状基本正确，只需微调，10 次迭代即收敛

2. **Jacobian 转矩初始化**（次优）
   - 条件：U_prev 不足 1/3 时域
   - 方法：`compute_jacobian_init_control(env, x0, p_target, horizon, gain=30)`
   - 效果：从零生成粗略轨迹，需要更多迭代

3. **引拍 warm-start**（backswing 模式专用）
   - 条件：启用 `--backswing` 且 U_prev 不足
   - 方法：`generate_backswing_warm_start(...)` 生成含引拍-前挥的五次多项式轨迹

**U_prev 传递链路**：

```
首次规划(iter=30, JT初始化) → 保存 U_mpc[replan_interval:] 作为 U_prev
  ↓
远段 JT 控制 → U_prev 保留（不重置，保持热启动连续性）
  ↓
近段 iLQR(iter=10, U_prev warm-start) → 保存新 U_prev
  ↓
下次近段 iLQR(iter=10, U_prev warm-start) → ...
```

### 3.5 代价函数设计

```
总代价 = 位置代价 + 速度代价 + 控制代价 + 法向对齐代价 + 平滑代价
```

#### 位置代价（终端）

```python
l_terminal_p = Q_p_scale × ||p_ee(x_N) - p_target||²
```

- `p_target = p_hit + hit_shift × d_hat`（击打点 + 随挥偏移）
- `Q_p_scale` 动态调度：距击球点远时 = 5.0，近时线性增加到 8.0

#### 速度代价（终端）

```python
l_terminal_v = Q_v_scale × ||v_ee(x_N) - v_hit_desired||²
```

- `v_hit_desired = compute_desired_hit_velocity(hit_direction, racket_speed)`
- `Q_v_scale` 动态调度：远时 = 3.0，近时 = 120.0

#### 控制代价（运行）

```python
l_control = Σ_k R_schedule[k] × ||u_k||²
```

- R 退火：接近击球时 R 衰减到 0，允许更大控制力矩全力挥拍
- J0（肩偏航）额外衰减 ×10，允许更大范围的挥拍动作

#### 法向对齐代价（终端）

```python
l_normal = Q_n × ||n_racket - n_des||²
```

- `n_des = -v_ball_hit / ||v_ball_hit||`（球拍面法向应对准球来方向）
- 权重 `Q_n` 可通过 `--normal-weight` 调节

#### 平滑代价（运行）

```python
l_smooth = Q_qdot × ||qdot||² + Q_qddot × ||qddot||² + Q_du × ||u_k - u_{k-1}||²
```

分阶段调度：
- 远段（k_hit > 50）：满权重，保持轨迹平滑
- 中段（k_hit 20-50）：2× 权重，更严格的平滑
- 近段（k_hit ≤ 20）：权重归零，不压制挥拍速度

### 3.6 R 退火调度

接近击球时刻，控制代价 R 逐渐衰减到 0：

```python
R_schedule[k] = R_base × (1 - decay_progress)  # decay_progress: 0→1

# J0（肩偏航）额外加速衰减
R_schedule[k, 0] = R_base × (1 - decay_progress)^10
```

效果：远处保持控制量小（省力、平滑），击球前释放全部力矩预算（全力挥拍）。

---

## 4. 安全约束系统（三层防护）

### 4.1 层1：X 平面墙（硬约束）

**目的**：防止右臂越过身体中线（X=0），避免与左臂/躯干碰撞。

```python
for beta in [1.0, 0.6, 0.3, 0.0]:
    u_try = beta × u_cmd
    x_pred = step_from_state(x_current, u_try)
    if all(body_xpos[bid, 0] <= -0.1 for body_id in right_arm_bodies):
        u_xsafe = u_try
        break
```

遍历 4 个 beta 值，选不越界的最大值。极端情况下 beta=0.0（零控制）。

### 4.2 层2：关节安全滤波器（硬约束）

**目的**：防止单步执行后关节速度超限。

```python
x_next = step_from_state(x_current, u_try)
strict_braking_check(x_current, x_next, u_try, limits)
```

检查规则（`strict_braking_check`）：

| k_hit 范围 | 检查行为 |
|-----------|---------|
| > 40 | 严格 1.0× 限制，超速即拒绝 |
| 20-40 | 放宽到 1.0-1.3× 限制（线性过渡） |
| ≤ 20 | **豁免**（击球前 100ms 需要全力加速） |

对于 k_hit > 20：
- 未超速关节 → 不允许新进入超速（hard reject）
- 已超速关节 → 不允许继续加速（必须减速，hard reject）

不通过时，尝试 beta 缩放：`[0.8, 0.6, 0.4, 0.2, 0.0]`，选第一个安全的。
全部不安全 → 紧急制动（阻尼制动 `u = -20 × qdot`）。

### 4.3 层3：执行后检查 + PD 推回

每步执行后，检查所有右臂 body 的 X 坐标。若越界：

```python
u_push = 300.0 × (init_q - q_now) - 20.0 × qdot_now  # PD 推回初始位形
```

这是最后一道保险。

---

## 5. 动力学线性化

iLQR 需要动力学方程的一阶线性化：`x_{k+1} ≈ A_k δx + B_k δu + ...`

### 5.1 解析线性化（默认）

通过 MuJoCo 的内部函数直接计算雅可比矩阵：

```python
A = ∂f/∂x  # 12×12 状态转移矩阵
B = ∂f/∂u  # 12×6  控制输入矩阵
```

计算速度快（~2ms per step），精度高。

### 5.2 快速线性化（fast_lin，k_hit > 30 时启用）

跳过部分二阶项的精确计算，用近似方法加速：
- 适用于远段和近段中段，轨迹不太激进时误差可接受
- 速度提升约 2-3×

### 5.3 有限差分线性化（--fd，调试用）

```python
A[i,j] = (f(x + ε*e_j, u) - f(x - ε*e_j, u)) / (2ε)
B[i,j] = (f(x, u + ε*e_j) - f(x, u - ε*e_j)) / (2ε)
```

精度低、速度慢（~15ms per step），仅用于验证。

---

## 6. Tube 空间走廊（可选增强）

Tube 是一条围绕球预测轨迹的空间走廊，提供：
- 多个候选击球时刻（时间窗口）
- 法向量对齐要求
- 空间覆盖范围评估

当前配置中，Tube **仅用于评估和监控**，不注入 iLQR 代价函数（已解耦）。解耦原因：tube 的软投影矩阵 `P_soft` 会沿球来方向降低位置代价权重，与最优击球方向冲突。

---

## 7. 碰撞检测与弹性反弹

### 7.1 碰撞检测窗口

```python
enable_collision = (k_hit ≤ 30 and dist < 0.35m) or (k_hit ≤ 10)
```

只在击球前后启用碰撞检测（减少计算开销）。启用时，MuJoCo 的接触检测（`ncon > 0`）检查球与球拍的几何体是否接触。

### 7.2 弹性反弹

检测到球-拍接触时，应用弹性碰撞模型：

```python
v_ball_after = v_ball_before - (1 + e) × (v_ball · n) × n  # 简化模型
```

其中 `e ≈ 0.8`（恢复系数），`n` 为碰撞法向量。反弹后球速度增加约 2-4 m/s。

---

## 8. 异步重规划（实验性，当前未使用）

代码中保留了 `AsyncReplanner` 基础设施（`--async-replan` 标志），但实测发现异步模式存在**过时计划问题**：

- 后台线程完成 iLQR 时，主线程已经前进了 30-50 步
- 将旧计划 shift 到当前时刻，但轨迹已经与实际臂状态偏离
- 误差从 0.03m（同步）涨到 0.25m（异步）

当前主循环走同步路径，异步代码保留供未来研究。

---

## 9. 数据流图

```
               球初始状态 (p0, v0)
                      │
                      ▼
            ┌─────────────────────┐
            │ predict_ball_trajectory │  预测球在每一步的位置
            └─────────┬───────────┘
                      │
                      ▼
            ┌─────────────────────┐
            │ find_hitting_point    │  找到机械臂可达的击球时刻
            │ → k_hit, p_hit       │
            └─────────┬───────────┘
                      │
         ┌────────────┼────────────┐
         │            │            │
         ▼            ▼            ▼
   ┌──────────┐ ┌──────────┐ ┌──────────────┐
   │ JT 初始化 │ │ Tube 构建 │ │ IK 可达性检查 │
   │ U_init    │ │ (可选)    │ │ + 替代搜索    │
   └────┬─────┘ └──────────┘ └──────────────┘
        │
        ▼
   ┌─────────────────────────────────┐
   │ 首次 iLQR (30 iters, ~1300ms)    │
   │ → U_mpc, X_mpc, U_prev          │
   └──────────────┬──────────────────┘
                  │
    ┌─────────────┼─────────────────┐
    │             │                 │
    ▼ (k>50)      ▼ (k≤50)          │
 ┌──────┐   ┌───────────────┐       │
 │ JT控制│   │ iLQR (10 iters)│      │
 │ 0ms   │   │ 50-80ms       │      │
 └──┬───┘   └──────┬────────┘      │
    │              │                │
    └──────┬───────┘                │
           ▼                        │
    ┌──────────────┐                │
    │ U_buffer 缓冲 │◄──── 保存尾部 ──┘
    └──────┬───────┘         (U_prev)
           │
           ▼
    ┌──────────────┐
    │ 取 u_cmd      │  buffer_idx++
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ X 平面墙约束  │  beta ∈ {1.0, 0.6, 0.3, 0.0}
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ 安全滤波器    │  beta ∈ {0.8, 0.6, 0.4, 0.2, 0.0}
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ env.step_full │  MuJoCo 物理仿真
    │ → x_current   │
    └──────────────┘
```

---

## 10. 关键设计决策与理由

| 决策 | 理由 |
|------|------|
| Tube 代价与 solver 解耦 | Tube 的 `P_soft` 矩阵降低位置代价权重，与最优击球方向冲突。解耦后命中率从 17/20 提升到 19/20 |
| 远段 JT + 近段 iLQR 混合 | 远段 iLQR 耗时 200ms（超 100ms 预算），JT 仅 <1ms。近段 iLQR 50-80ms 在预算内 |
| Warm-start 尾部重用 | 上次规划的后半段轨迹形状正确，重采样到新时域后只需 10 次迭代微调 |
| k_hit ≤ 20 豁免速度检查 | 击球前 100ms 需要全力加速到 3-5 m/s，此时限制速度会严重降低击球精度 |
| U_history 存 u_final | 安全滤波后的控制，确保回放与实际执行一致 |
| 误差指标用球实际位置 | `‖p_ee - ball_actual‖` 而非 `‖p_ee - p_hit‖`，正确反映是否击中球 |

---

## 11. 文件依赖关系

```
rm65_mpc_tube_constraint_realtime.py
├── src/sim/rm65_env.py           # MuJoCo 环境封装（正运动学、仿真步进、球状态）
├── src/tennis/ball.py             # 球轨迹生成（发球、抛物线预测）
├── src/tennis/hitting.py          # 击打点计算、期望击球速度
├── src/ilqt/cost.py               # 代价函数（HittingCost, TubeHittingCostWrapper）
├── src/ilqt/solver.py             # iLQR 求解器（后向-前向迭代）
├── src/cpp/solver_cpp.py          # C++ 加速版求解器（优先使用）
├── src/ilqt/robot_limits.py       # 关节约束检查、安全滤波器
├── src/ilqt/async_replanner.py    # 异步重规划器（实验性）
├── src/ilqt/retiming.py           # 轨迹重定时（速度约束平滑）
├── src/dynamics/linearize.py      # 动力学线性化（解析/有限差分/快速）
├── src/robot/rm65_model.xml       # MuJoCo XML 模型定义
└── configs/default.yaml           # 默认参数配置
```

---

## 12. 性能指标（12 m/s 球速，--realtime 模式）

| 指标 | 数值 |
|------|------|
| 命中率 | 19/20 (95%) |
| 平均球-拍最近距离 | 60mm |
| 远段重规划耗时 | avg 30ms, max 45ms（预算 100ms） |
| 近段重规划耗时 | avg 55ms, max 87ms（预算 200ms） |
| 非重规划步耗时 | avg 0.8ms |
| 每步平均总耗时 | 3.6ms |
| iLQR 总迭代次数 | ~70 次（首次 30 + 稳态 4×10） |
| 最大关节速度倍率 | 2.5-3.5×（超 RM-65B 规格） |
| 最大 TCP 速度 | 4-5 m/s |

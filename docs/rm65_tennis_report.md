# RM-65 网球击打系统技术报告

## 1. 项目概述

本项目基于 MuJoCo 物理引擎和 iLQT（迭代线性二次跟踪器）最优控制方法，实现 RM-65 双臂机器人的网球击打任务。系统采用 MPC（模型预测控制）框架，在每个控制周期内：

1. 预测网球飞行轨迹，确定击打点
2. 使用 iLQT 求解最优挥拍轨迹
3. 执行控制并滚动优化

**核心场景**：RM-65 右臂（6-DOF，力矩 ±60/30/10 Nm）安装在垂直桩柱右侧，末端法兰连接网球拍，球从摄像头正对方向（-Y 方向）飞来，系统计算最优挥拍轨迹使球拍在正确的时间和位置以期望的速度击中网球。

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    MPC 主循环 (rm65_mpc_ilqt.py)                 │
│                                                                 │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────────┐      │
│  │ 球轨迹生成 │──>│ 击打点搜索   │──>│ 误差驱动权重调度   │      │
│  │ (ball.py) │   │ (hitting.py) │   │ (pos-err based)    │      │
│  └──────────┘   └──────────────┘   └────────┬───────────┘      │
│                                              v                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │              iLQT 求解器 (solver.py)                  │       │
│  │                                                       │       │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │       │
│  │  │ 动力学线性化 │  │ 击打代价函数 │  │ 前向传递   │ │       │
│  │  │(linearize.py)│  │  (cost.py)   │  │ (utils.py) │ │       │
│  │  │ A_c,B_c→A,B  │  │ Q_h, h_des  │  │ α=0.5/搜索│ │       │
│  │  └──────────────┘  └──────────────┘  └────────────┘ │       │
│  │         │                 │               │          │       │
│  │         v                 v               v          │       │
│  │     ┌─────────────────────────────────────────┐     │       │
│  │     │       后向传递 (Riccati 递推)            │     │       │
│  │     │  K_k = -Q_uu⁻¹ Q_ux                    │     │       │
│  │     │  k_k = -Q_uu⁻¹ Q_u                     │     │       │
│  │     └─────────────────────────────────────────┘     │       │
│  └──────────────────────────────────────────────────────┘       │
│                           │                                     │
│                           v                                     │
│  ┌──────────────────────────────────────────────────────┐       │
│  │              RM65Env (rm65_env.py)                    │       │
│  │  MuJoCo 物理：臂 + 球 + 球拍碰撞                      │       │
│  │  解析弹跳：v_z *= -0.75, v_xy *= 0.95                 │       │
│  │  左臂 PD 保持：kp=200, kd=20                          │       │
│  └──────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 机器人模型

### 3.1 RM-65 关节配置

| 关节编号 | 名称 | 类型 | 关节轴 | 角度范围 (°) | 力矩限制 (Nm) | 阻尼 | 惯量 |
|----------|------|------|--------|-------------|---------------|------|------|
| 0 | r_joint1 | 肩偏航 | 0 0 1 | -178~178 | ±60 | 2.0 | 0.2 |
| 1 | r_joint2 | 肩俯仰 | 0 0 1 | -130~130 | ±60 | 2.0 | 0.2 |
| 2 | r_joint3 | 肘关节 | 0 0 1 | -135~135 | ±30 | 1.0 | 0.1 |
| 3 | r_joint4 | 腕关节1 | 0 0 1 | -178~178 | ±10 | 0.5 | 0.05 |
| 4 | r_joint5 | 腕关节2 | 0 0 1 | -128~128 | ±10 | 0.3 | 0.03 |
| 5 | r_joint6 | 腕关节3 | 0 0 1 | -360~360 | ±10 | 0.2 | 0.02 |

### 3.2 状态与控制向量

$$
\mathbf{x} = \begin{bmatrix} \mathbf{q} \\ \dot{\mathbf{q}} \end{bmatrix} \in \mathbb{R}^{12}, \quad \mathbf{u} = \boldsymbol{\tau} \in \mathbb{R}^{6}
$$

其中 $\mathbf{q} \in \mathbb{R}^6$ 为关节角度，$\dot{\mathbf{q}} \in \mathbb{R}^6$ 为关节角速度，$\boldsymbol{\tau} \in \mathbb{R}^6$ 为关节力矩。

### 3.3 连杆尺寸

| 连杆 | 长度 (mm) | 说明 |
|------|-----------|------|
| r_link1 | 187 | 肩关节到肩俯仰 |
| r_link2 | 256 | 上臂 |
| r_link3 | 210 | 前臂 |
| r_link5 | 112 | 腕部 |
| 法兰 | 44.5 | 法兰厚度 |
| 球拍手柄 | 220 | 手柄+拍面 |
| **总臂展** | **~850** | 从肩到球拍面中心 |

### 3.4 安装位置

右臂肩关节世界坐标：

$$
\mathbf{p}_{\text{shoulder}} = \begin{bmatrix} -0.1 \\ -0.22693 \\ 1.302645 \end{bmatrix} \text{ m}
$$

球拍面中心 site `racket_center` 距法兰 0.25m。

### 3.5 初始构型

$$
\mathbf{q}_0 = \begin{bmatrix} 0.0 & -0.8 & 1.5 & -0.7 & 0.0 & 0.0 \end{bmatrix}^T
$$

此构型使球拍处于身体前方偏下位置，便于向各方向挥拍。

---

## 4. 网球轨迹预测与击打点搜索

### 4.1 抛物线运动模型

网球在重力作用下的运动方程：

$$
\mathbf{p}(t) = \mathbf{p}_0 + \mathbf{v}_0 t + \frac{1}{2}\mathbf{g}t^2
$$

$$
\mathbf{v}(t) = \mathbf{v}_0 + \mathbf{g}t
$$

其中 $\mathbf{g} = [0, 0, -9.81]^T$ m/s²。

### 4.2 地面弹跳模型

当球触地（$z < r_{\text{ball}}$ 且 $v_z < 0$）时，采用解析弹跳：

$$
v_z^+ = -v_z^- \times e, \quad v_x^+ = v_x^- \times 0.95, \quad v_y^+ = v_y^- \times 0.95
$$

其中 $e = 0.75$ 为恢复系数。

### 4.3 发球轨迹生成

采用**反推法**：先采样击打点，再反推球的初始状态。

**步骤**：

1. 在肩关节周围的可达区域内采样击打点偏移量：
   - $\Delta x \sim U(0.15, 0.50)$, $\Delta y \sim U(-0.20, 0.20)$, $\Delta z \sim U(-0.15, 0.25)$
   - $\mathbf{p}_{\text{hit}} = \mathbf{p}_{\text{shoulder}} + [\Delta x, \Delta y, \Delta z]^T$

2. 采样球的初始位置（从摄像头正对方向 -Y 飞来）：
   - $x_0 \sim U(-1.0, 1.0)$, $y_0 \sim U(-8.0, -3.0)$, $z_0 \sim U(1.5, 2.5)$

3. 反推初始速度，保证球在 $t_{\text{hit}}$ 时刻到达 $\mathbf{p}_{\text{hit}}$：

$$
\mathbf{v}_0 = \frac{\mathbf{p}_{\text{hit}} - \mathbf{p}_0 - \frac{1}{2}\mathbf{g}t_{\text{hit}}^2}{t_{\text{hit}}}
$$

### 4.4 击打点搜索

使用 MuJoCo 物理仿真前向模拟球轨迹，搜索最佳击打点：

**评分函数**：

$$
\text{score} = d + h_{\text{penalty}} - f_{\text{bonus}}
$$

其中：
- $d = \|\mathbf{p}_{\text{ball}} - \mathbf{p}_{\text{shoulder}}\|$ — 球到肩关节的距离
- $h_{\text{penalty}} = 5.0 \times \max(0, z_{\text{ball}} - z_{\text{shoulder}} - 0.2)^2$ — 过高惩罚
- $f_{\text{bonus}} = 0.3 \times \max(0, x_{\text{ball}} - x_{\text{shoulder}})$ — 前方奖励

**可达性约束**：$d < r_{\text{workspace}}$，$z_{\text{ball}} > 0.3$，$-0.40 < \Delta z < 0.55$。

选择得分最低（距离最近 + 高度适中 + 偏前方）的时刻作为击打点。

---

## 5. iLQT 算法推导

### 5.1 问题定义

iLQT 求解离散时间最优控制问题：

$$
\min_{\mathbf{u}_0, \ldots, \mathbf{u}_{N-1}} \left[ \sum_{k=0}^{N-1} l_k(\mathbf{x}_k, \mathbf{u}_k) + l_N(\mathbf{x}_N) \right]
$$

受动力学约束：

$$
\mathbf{x}_{k+1} = f(\mathbf{x}_k, \mathbf{u}_k)
$$

其中 $l_k$ 为运行代价，$l_N$ 为终端代价。

### 5.2 代价函数

#### 运行代价（控制代价）

$$
l_k(\mathbf{x}_k, \mathbf{u}_k) = \frac{1}{2} \mathbf{u}_k^T \mathbf{R} \mathbf{u}_k
$$

其中 $\mathbf{R} = r \cdot \mathbf{I}_{6 \times 6}$，$r = 0.0001$。

#### 终端代价（击打代价）

$$
l_N(\mathbf{x}_N) = \frac{1}{2} (\mathbf{h}(\mathbf{x}_N) - \mathbf{h}_{\text{des}})^T \mathbf{Q}_h (\mathbf{h}(\mathbf{x}_N) - \mathbf{h}_{\text{des}})
$$

其中：

$$
\mathbf{h}(\mathbf{x}) = \begin{bmatrix} \mathbf{p}_{\text{ee}}(\mathbf{x}) \\ \mathbf{v}_{\text{ee}}(\mathbf{x}) \end{bmatrix} \in \mathbb{R}^6, \quad
\mathbf{h}_{\text{des}} = \begin{bmatrix} \mathbf{p}_{\text{hit}} \\ \mathbf{v}_{\text{hit}} \end{bmatrix} \in \mathbb{R}^6
$$

$$
\mathbf{Q}_h = \begin{bmatrix} s_p \mathbf{Q}_p & \mathbf{0} \\ \mathbf{0} & s_v \mathbf{Q}_v \end{bmatrix} \in \mathbb{R}^{6 \times 6}
$$

- $\mathbf{Q}_p = \text{diag}(q_p, q_p, q_p)$：位置权重矩阵，$q_p = 50000 \times 2 = 100000$
- $\mathbf{Q}_v = \text{diag}(q_v, q_v, q_v)$：速度权重矩阵，$q_v = 200 \times 2 = 400$
- $s_p, s_v$：动态缩放因子（MPC 权重调度）

### 5.3 动力学线性化

#### 连续时间刚体动力学

$$
\mathbf{M}(\mathbf{q}) \ddot{\mathbf{q}} + \mathbf{h}(\mathbf{q}, \dot{\mathbf{q}}) = \boldsymbol{\tau}
$$

其中 $\mathbf{h}(\mathbf{q}, \dot{\mathbf{q}}) = \mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q})$ 包含科里奥利力和重力项。

#### 连续时间状态空间形式

$$
\dot{\mathbf{x}} = f_c(\mathbf{x}, \mathbf{u}) = \begin{bmatrix} \dot{\mathbf{q}} \\ \mathbf{M}^{-1}(\mathbf{q}) \left( \boldsymbol{\tau} - \mathbf{h}(\mathbf{q}, \dot{\mathbf{q}}) \right) \end{bmatrix}
$$

#### 解析线性化步骤

**1. 质量矩阵** $\mathbf{M}(\mathbf{q}) \in \mathbb{R}^{6 \times 6}$：通过 `mj_fullM` 从 MuJoCo 稀疏表示提取。

**2. 偏差力** $\mathbf{h}(\mathbf{q}, \dot{\mathbf{q}}) \in \mathbb{R}^6$：通过 `mj_rne` 计算（设 $\ddot{\mathbf{q}} = 0$）。

**3. 偏差力对 q 的偏导**：中心差分

$$
\frac{\partial \mathbf{h}}{\partial q_j} \approx \frac{\mathbf{h}(\mathbf{q} + \epsilon \mathbf{e}_j, \dot{\mathbf{q}}) - \mathbf{h}(\mathbf{q} - \epsilon \mathbf{e}_j, \dot{\mathbf{q}})}{2\epsilon}
$$

**4. 偏差力对 $\dot{\mathbf{q}}$ 的偏导**：同理

$$
\frac{\partial \mathbf{h}}{\partial \dot{q}_j} \approx \frac{\mathbf{h}(\mathbf{q}, \dot{\mathbf{q}} + \epsilon \mathbf{e}_j) - \mathbf{h}(\mathbf{q}, \dot{\mathbf{q}} - \epsilon \mathbf{e}_j)}{2\epsilon}
$$

**5. 组装连续时间雅可比矩阵**：

$$
\mathbf{A}_c = \begin{bmatrix} \mathbf{0}_{6 \times 6} & \mathbf{I}_{6 \times 6} \\ -\mathbf{M}^{-1} \frac{\partial \mathbf{h}}{\partial \mathbf{q}} & -\mathbf{M}^{-1} \frac{\partial \mathbf{h}}{\partial \dot{\mathbf{q}}} \end{bmatrix} \in \mathbb{R}^{12 \times 12}
$$

$$
\mathbf{B}_c = \begin{bmatrix} \mathbf{0}_{6 \times 6} \\ \mathbf{M}^{-1} \end{bmatrix} \in \mathbb{R}^{12 \times 6}
$$

**6. Euler 离散化**（时间步长 $\Delta t = 0.005$ s）：

$$
\mathbf{A} = \mathbf{I}_{12} + \mathbf{A}_c \Delta t, \quad \mathbf{B} = \mathbf{B}_c \Delta t
$$

### 5.4 后向传递（Riccati 递推）

初始化价值函数导数：

$$
\mathbf{V}_x = \frac{\partial l_N}{\partial \mathbf{x}}, \quad \mathbf{V}_{xx} = \frac{\partial^2 l_N}{\partial \mathbf{x}^2}
$$

从 $k = N-1$ 到 $k = 0$ 递推：

**步骤 1**：计算 Q 函数展开系数

$$
\mathbf{Q}_x = \mathbf{l}_x + \mathbf{A}_k^T \mathbf{V}_x
$$

$$
\mathbf{Q}_u = \mathbf{l}_u + \mathbf{B}_k^T \mathbf{V}_x
$$

$$
\mathbf{Q}_{xx} = \mathbf{l}_{xx} + \mathbf{A}_k^T \mathbf{V}_{xx} \mathbf{A}_k
$$

$$
\mathbf{Q}_{ux} = \mathbf{l}_{ux} + \mathbf{B}_k^T \mathbf{V}_{xx} \mathbf{A}_k
$$

$$
\mathbf{Q}_{uu} = \mathbf{l}_{uu} + \mathbf{B}_k^T \mathbf{V}_{xx} \mathbf{B}_k
$$

**步骤 2**：加入正则化

$$
\mathbf{Q}_{uu}^{\text{reg}} = \mathbf{Q}_{uu} + \mu \mathbf{I}
$$

**步骤 3**：计算反馈增益和前馈增益

$$
\mathbf{K}_k = -(\mathbf{Q}_{uu}^{\text{reg}})^{-1} \mathbf{Q}_{ux}
$$

$$
\mathbf{k}_k = -(\mathbf{Q}_{uu}^{\text{reg}})^{-1} \mathbf{Q}_u
$$

使用 `np.linalg.solve` 而非显式求逆，提高数值稳定性。

**步骤 4**：更新价值函数导数

$$
\mathbf{V}_x = \mathbf{Q}_x - \mathbf{Q}_{ux}^T (\mathbf{Q}_{uu}^{\text{reg}})^{-1} \mathbf{Q}_u
$$

$$
\mathbf{V}_{xx} = \mathbf{Q}_{xx} - \mathbf{Q}_{ux}^T (\mathbf{Q}_{uu}^{\text{reg}})^{-1} \mathbf{Q}_{ux}
$$

$$
\mathbf{V}_{xx} \leftarrow \frac{1}{2}(\mathbf{V}_{xx} + \mathbf{V}_{xx}^T) \quad \text{（对称化）}
$$

### 5.5 前向传递

#### 带线搜索的前向传递

对每个步长 $\alpha \in \{1.0, 0.5, 0.25, 0.1, 0.05, 0.01\}$：

$$
\mathbf{u}_k^{\text{new}} = \mathbf{u}_k + \alpha \mathbf{k}_k + \mathbf{K}_k (\mathbf{x}_k^{\text{new}} - \mathbf{x}_k)
$$

$$
\mathbf{u}_k^{\text{new}} \leftarrow \text{clip}(\mathbf{u}_k^{\text{new}}, \mathbf{u}_{\min}, \mathbf{u}_{\max})
$$

$$
\mathbf{x}_{k+1}^{\text{new}} = f(\mathbf{x}_k^{\text{new}}, \mathbf{u}_k^{\text{new}})
$$

若总代价 $J^{\text{new}} < J^{\text{old}}$，接受更新。

#### MPC 模式前向传递（固定步长）

$$
\mathbf{u}_k^{\text{new}} = \mathbf{u}_k + 0.5 \mathbf{k}_k + \mathbf{K}_k (\mathbf{x}_k^{\text{new}} - \mathbf{x}_k)
$$

始终接受更新（MPC 依赖重规划纠错），跳过代价计算以节省时间。

### 5.6 终端代价的 Gauss-Newton 近似

$$
\mathbf{J}_h \approx \begin{bmatrix} \mathbf{J}_p & \mathbf{0}_{3 \times 6} \\ \mathbf{0}_{3 \times 6} & \mathbf{J}_p \end{bmatrix} \in \mathbb{R}^{6 \times 12}
$$

其中 $\mathbf{J}_p \in \mathbb{R}^{3 \times 6}$ 是末端位置雅可比矩阵（MuJoCo `mj_jacSite`）。

一阶导数：

$$
\frac{\partial l_N}{\partial \mathbf{x}} = \mathbf{J}_h^T \mathbf{Q}_h (\mathbf{h}(\mathbf{x}) - \mathbf{h}_{\text{des}})
$$

二阶导数（Gauss-Newton 近似，忽略二阶项）：

$$
\frac{\partial^2 l_N}{\partial \mathbf{x}^2} \approx \mathbf{J}_h^T \mathbf{Q}_h \mathbf{J}_h
$$

---

## 6. MPC 控制框架

### 6.1 三层控制策略

```
┌─────────────────────────────────────────────────────────┐
│ 远距（k_hit > k_hit_total）                              │
│   雅可比转置控制器，仅位置追踪，无 iLQT 优化               │
│   τ = J_p^T · err · scale − 2.0 · q̇                    │
├─────────────────────────────────────────────────────────┤
│ 中距（near_threshold < k_hit ≤ k_hit_total）             │
│   iLQT 3 次迭代，位置权重为主                             │
│   Q_p_scale = 5.0, Q_v_scale = 2.0                      │
│   skip_linesearch = True                                 │
├─────────────────────────────────────────────────────────┤
│ 近距（k_hit ≤ near_threshold）                           │
│   iLQT 5 次迭代，速度权重增强                             │
│   Q_p_scale = 3.0, Q_v_scale = 12.0                     │
│   skip_linesearch = True                                 │
└─────────────────────────────────────────────────────────┘
```

### 6.2 误差驱动权重调度

实际使用**位置误差驱动**的权重切换（替代步数调度）：

$$
(s_p, s_v) = \begin{cases} (5.0, 2.0) & \text{if } \|\mathbf{p}_{\text{ee}} - \mathbf{p}_{\text{hit}}\| > 0.10 \\ (3.0 + 2.0r, \; 12.0 - 10.0r) & \text{otherwise} \end{cases}
$$

其中 $r = \|\mathbf{p}_{\text{ee}} - \mathbf{p}_{\text{hit}}\| / 0.10 \in [0, 1]$ 为误差比。

**设计理念**：
- 误差大时专注位置收敛（高 $s_p$，低 $s_v$）
- 误差小时加速度匹配（低 $s_p$，高 $s_v$）

### 6.3 期望击打速度

$$
\mathbf{v}_{\text{hit}} = \frac{\mathbf{d}_{\text{hit}}}{\|\mathbf{d}_{\text{hit}}\|} \times v_{\text{racket}}
$$

其中 $\mathbf{d}_{\text{hit}} = [-1.0, 0.0, 0.3]^T$ 为击打方向，$v_{\text{racket}} = 5.0$ m/s 为球拍速度。

### 6.4 雅可比转置初始控制

作为 iLQT 的 warm-start 生成器：

$$
\boldsymbol{\tau}_k = \mathbf{J}_p^T \cdot (\mathbf{p}_{\text{hit}} - \mathbf{p}_{\text{ee}}) \cdot s_{\text{gain}} \cdot \min(\|\mathbf{e}\|, 0.5) - 2.0 \dot{\mathbf{q}}
$$

其中 $s_{\text{gain}} = 30.0$（远距）或 $60.0$（首次规划）。

### 6.5 重规划策略

| 参数 | 值 | 说明 |
|------|-----|------|
| replan_interval | 10 步 | 每 10 步重规划一次 |
| max_iter_per_plan | 3 | 常规迭代数 |
| 首次规划迭代 | 5 | 首次规划更精细 |
| 近距迭代 | 5 | 近距增强迭代 |
| far_threshold | $k_{\text{hit,total}}$ | 远距阈值 = 初始击打步数 |
| near_threshold | $\max(40, k_{\text{hit,total}} / 4)$ | 近距阈值 |

### 6.6 击打后保持

击打时刻后执行 80 步 PD 保持（$\mathbf{q}_{\text{hold}}$ 为击打时刻关节角）：

$$
\boldsymbol{\tau}_{\text{hold}} = 100 \cdot (\mathbf{q}_{\text{hold}} - \mathbf{q}) - 10 \cdot \dot{\mathbf{q}}
$$

---

## 7. 碰撞检测方案

### 7.1 位掩码碰撞过滤

MuJoCo 中两个 geom 碰撞的充要条件：

$$
(\text{contype}_1 \wedge \text{conaffinity}_2) \neq 0 \;\; \lor \;\; (\text{contype}_2 \wedge \text{conaffinity}_1) \neq 0
$$

### 7.2 碰撞组定义

| 位 | 十六进制 | 碰撞组 | 参与者 |
|----|----------|--------|--------|
| bit 0 (1) | 0x01 | 地面-结构 | 地面、身体结构 |
| bit 1 (2) | 0x02 | 臂段自碰撞 | 右臂 link1-6、左臂 link1-6 |
| bit 2 (4) | 0x04 | 球拍-球 | 球拍、球 |
| bit 3 (8) | 0x08 | 球 | 球 |

### 7.3 碰撞矩阵

| 部位 | contype | conaffinity | 二进制 |
|------|---------|-------------|--------|
| 地面 | 1 | 15 (1+2+4+8) | 0001 / 1111 |
| 身体结构（底盘/躯干/头/臂基座） | 1 | 1 | 0001 / 0001 |
| 臂段（link1-6） | 2 | 2 | 0010 / 0010 |
| 球拍 | 4 | 8 | 0100 / 1000 |
| 球 | 8 | 5 (1+4) | 1000 / 0101 |
| 视觉网格 | 0 | 0 | 0000 / 0000 |

### 7.4 碰撞对验证

| 碰撞对 | ct₁∧ca₂ | ct₂∧ca₁ | 结果 |
|--------|---------|---------|------|
| 臂段↔臂段 | 2∧2=2 | 2∧2=2 | ✓ 碰撞 |
| 臂段↔地面 | 2∧15=2 | 1∧2=0 | ✓ 碰撞 |
| 臂段↔球拍 | 2∧8=0 | 4∧2=0 | ✗ 不碰 |
| 臂段↔球 | 2∧5=0 | 8∧2=0 | ✗ 不碰 |
| 球拍↔球 | 4∧5=4 | 8∧8=8 | ✓ 碰撞 |
| 球↔地面 | 8∧15=8 | 1∧5=1 | ✓ 碰撞 |
| 结构↔地面 | 1∧15=1 | 1∧1=1 | ✓ 碰撞 |

### 7.5 碰撞排除对

通过 `<contact><exclude>` 排除不应碰撞的体对：

- **相邻臂段**（共享关节）：`r_base↔r_link1`, `r_link1↔r_link2`, ..., `r_link5↔r_link6`（左右臂各 6 对）
- **同臂非相邻常穿透对**：`r_link4↔r_link6`, `l_link4↔l_link6`（腕部折叠时的胶囊重叠）

### 7.6 碰撞胶囊尺寸

| 部位 | 类型 | 参数 | 半径 |
|------|------|------|------|
| 底盘 | capsule | z: -0.2~0.2 | 0.12 |
| 躯干 | capsule | z: -0.05~0.85 | 0.10 |
| 平台 | capsule | z: -0.08~0.08 | 0.08 |
| 头部1 | capsule | z: -0.03~0.03 | 0.025 |
| 头部2 | capsule | z: -0.05~0.05 | 0.05 |
| 摄像头 | capsule | z: -0.03~0.03 | 0.02 |
| 臂基座 | capsule | z: -0.07~0.07 | 0.05 |
| link1 | capsule | z: -0.06~0.06 | 0.04 |
| link2 | capsule | z: -0.14~0.14 | 0.04 |
| link3 | capsule | z: -0.07~0.07 | 0.03 |
| link4 | capsule | z: -0.04~0.04 | 0.03 |
| link5 | capsule | z: -0.05~0.05 | 0.025 |
| link6 | capsule | z: -0.02~0.04 | 0.025 |
| 球拍手柄 | capsule | z: 0.03~0.22 | 0.012 |
| 球拍面 | ellipsoid | center z=0.25 | 0.005×0.10×0.12 |
| 球 | sphere | — | 0.033 |

### 7.7 左臂保持策略

左臂不驱动但需要抵抗碰撞推力，采用 PD 控制器保持零位：

$$
\boldsymbol{\tau}_L = 200 \cdot (0 - \mathbf{q}_L) - 20 \cdot \dot{\mathbf{q}}_L
$$

力矩裁剪到 $\pm 60/30/10$ Nm，同时每步强制归零位置和速度保证规划一致性。

---

## 8. 实验结果

### 8.1 测试配置

- 球从 -Y 方向（摄像头正对方向）飞来
- 球拍速度：5.0 m/s
- 击打方向：$[-1, 0, 0.3]^T$
- MPC 周期：5ms（200Hz）
- 总仿真步数：200 步（1.0s）
- 线性化方法：解析法

### 8.2 性能统计（15 个随机种子）

| 指标 | 无自碰撞 | 有自碰撞 |
|------|---------|---------|
| 平均位置误差 | 0.058m | 0.084m |
| <5cm 命中率 | 53% (8/15) | 33% (5/15) |
| <10cm 命中率 | 87% (13/15) | 60% (9/15) |
| <20cm 命中率 | 93% (14/15) | 93% (14/15) |
| 实时比率 | ~0.55x | ~0.50x |

### 8.3 典型结果示例

| 种子 | 位置误差 | 球速 | 判定 |
|------|---------|------|------|
| 100 | 0.026m | — | 命中 |
| 77 | 0.022m | — | 命中 |
| 99 | 0.015m | — | 命中 |
| 7 | 0.025m | — | 命中 |
| 200 | 0.037m | — | 命中 |
| 66 | 0.005m | — | 精准 |

---

## 9. 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 球-身体不碰撞 | conaffinity 位掩码 | 避免球弹到机器人身体导致 MPC 预测错误 |
| 预测时固定臂位置 | `predict_ball_trajectory` 中锁定臂 | 防止臂在重力下坠落碰到球 |
| 误差驱动权重调度 | 位置误差 <10cm 时切速度权重 | 避免过早引入速度代价导致过冲 |
| 首次规划用线搜索 | 5 次迭代 + 线搜索 | 确保初始轨迹质量 |
| 后续规划跳过线搜索 | α=0.5 始终接受 | 节省计算时间，依赖重规划纠错 |
| 左臂 PD 保持 | kp=200, kd=20 | 自碰撞时抵抗推力，避免干扰右臂 |
| u_hold bug 修复 | `q_hold = q.copy()` | 原代码 `x-x=0` 导致击打后臂坠落 |
| 球从 -Y 方向飞来 | `ball_direction="y"` | 匹配摄像头朝向，臂运动更自然 |
| 解析弹跳模型 | $v_z^+ = -0.75 v_z^-$ | MuJoCo 碰撞模型过于耗散 |
| Euler 离散化 | $A = I + A_c \Delta t$ | 简单有效，配合 `implicitfast` 积分器 |

---

## 10. 文件结构

```
tennis_robot/
├── src/
│   ├── robot/rm65_model.xml      # RM-65 MuJoCo 模型（碰撞方案 + 左臂执行器）
│   ├── sim/rm65_env.py            # RM65Env（PD保持 + 解析弹跳 + IK求解器）
│   ├── dynamics/linearize.py      # 解析线性化（M⁻¹H_q, M⁻¹H_qdot）
│   ├── ilqt/solver.py            # iLQT 求解器（Riccati + 线搜索）
│   ├── ilqt/cost.py              # HittingCost（Gauss-Newton 终端代价）
│   ├── ilqt/utils.py             # 前向传递（线搜索 / 固定步长）
│   ├── tennis/ball.py            # 球轨迹生成 + 反推初速度
│   └── tennis/hitting.py         # 击打点搜索 + 权重调度
├── scripts/rm65_mpc_ilqt.py      # MPC 主循环（误差驱动调度 + warm-start）
├── scripts/rm65_mpc_ilqr_5_5.py # 后摆+碰撞击打（解析弹性碰撞反弹球）
├── configs/default.yaml          # 默认超参数
└── docs/rm65_tennis_report.md    # 本文档
```
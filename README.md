# RM-65 网球机器人 — MPC+iLQR+Tube 击打仿真

RM-65 双臂机器人网球击打 MuJoCo 仿真项目。使用 **MPC（模型预测控制）** 作为外层闭环框架，**iLQR（迭代线性二次调节器）** 作为内层轨迹优化求解器，**Tube-based Robust Hitting** 实现时空鲁棒性，**多层安全滤波**保障关节/TCP 约束。球和机械臂均由 MuJoCo 物理引擎驱动，球拍-球碰撞产生真实击打效果。

---

## 环境安装

```bash
# 创建 conda 环境（推荐）
conda create -n mujoco_tennis python=3.11
conda activate mujoco_tennis
pip install -r requirements.txt

# 或直接 pip 安装
pip install -r requirements.txt
```

依赖：`mujoco>=3.0`, `numpy>=1.24`, `scipy>=1.10`, `matplotlib>=3.7`, `pyyaml>=6.0`

---

## 快速开始

### 主脚本：MPC+Tube 鲁棒击打仿真（V5）

```bash
# 默认参数运行（随机发球）
python scripts/rm65_mpc_tube_constraint_realtime_v5.py --viewer

# 使用长方体发球区 + 指定球速
python scripts/rm65_mpc_tube_constraint_realtime_v5.py --serve-box --ball-speed 12 --viewer
python scripts/rm65_mpc_tube_constraint_realtime_v5.py --serve-box --ball-speed 15 --viewer

# 指定随机种子
python scripts/rm65_mpc_tube_constraint_realtime_v5.py --serve-box --seed 42 --viewer

# 关闭 Tube（纯 MPC+iLQR 基线）
python scripts/rm65_mpc_tube_constraint_realtime_v5.py --use_tube false --viewer --seed 42

# 启用异步重规划 + 实时节奏
python scripts/rm65_mpc_tube_constraint_realtime_v5.py --serve-box --async-replan --realtime --viewer

# 施加时空扰动（鲁棒性测试）
python scripts/rm65_mpc_tube_constraint_realtime_v5.py --serve-box --time-perturb-ms 30 --space-perturb-m 0.05 --viewer

# TCP 限速实验
python scripts/rm65_mpc_tube_constraint_realtime_v5.py --serve-box --max-tcp 2.0 --viewer
```

### 关节调节查看器

```bash
python scripts/rm65_joint_viewer.py
```

打开后可拖动右侧 Control 面板滑条直接控制关节。按键：`R` 重置、`P` 打印、`F` 切换坐标轴。

---

## 命令行参数（V5 主脚本）

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--viewer` | flag | — | 计算完成后 MuJoCo 查看器回放 |
| `--seed` | int | None | 随机种子 |
| `--fd` | flag | — | 有限差分线性化（默认解析） |
| `--horizon` | int | None | 短地平线步数（覆盖配置文件） |
| `--iter` | int | None | 每次重规划迭代数 |
| `--use_tube` | str | true | 启用/关闭 Tube 鲁棒策略 |
| `--fix-joint5` | flag | — | 固定第 6 关节 |
| `--backswing` | float | 0.6 | 后摆幅度 (rad) |
| `--bs-ratio` | float | 0.35 | 后摆占比 (0~1) |
| `--no-backswing` | flag | — | 禁用后摆 |
| `--r-decay` | float | 0.40 | R 退火占比 (0~1) |
| `--no-r-decay` | flag | — | 禁用 R 退火 |
| `--hit-shift` | float | 0.01 | 随挥偏移距离 (m) |
| `--ball-speed` | float | None | 球到达击打点时水平速度 (m/s) |
| `--ball-distance` | float | None | 球起始位置到击打点的直线距离 (m) |
| `--approach-angle` | float | 0.0 | 球飞来方向角 (度)，0=-Y，90=-X |
| `--serve-box` | flag | — | 使用长方体发球区模式 |
| `--no-bounce` | flag | — | 禁用地面弹跳 |
| `--serve-distance` | float | 8.0 | 发球区 Y 方向距离 (m) |
| `--serve-height` | float | 1.2 | 发球区中心高度 (m) |
| `--normal-weight` | float | 500000 | 拍面法向量代价权重 (0=禁用) |
| `--normal-flip` | flag | — | 翻转法向量方向 |
| `--replan-interval` | int | None | 重规划间隔步数 |
| `--window-ms` | float | 50.0 | Tube 候选窗口半宽 (ms) |
| `--tube-cost-ratio` | float | 0.3 | Tube 代价占比 (0~1) |
| `--softmin-beta` | float | 5.0 | Softmin 温度参数 |
| `--no-softmin` | flag | — | 禁用 Softmin 多终端 |
| `--no-plot` | flag | — | 禁用 matplotlib 可视化 |
| `--realtime` | flag | — | 模拟实时节奏（5ms/步） |
| `--async-replan` | flag | — | 启用异步重规划（后台线程 iLQR） |
| `--time-perturb-ms` | float | 0.0 | 时间扰动量 (ms) |
| `--space-perturb-m` | float | 0.0 | 空间扰动量 (m) |
| `--ball-speed-perturb-pct` | float | 0.0 | 球速耦合扰动百分比 (%) |
| `--max-tcp` | float | None | TCP 线速度硬限制 (m/s) |
| `--terminal-exempt-steps` | int | None | 终段 qdot/TCP 豁免步数 |

---

## 核心算法

### MPC + iLQR 闭环架构

```
每 N 步重规划（由 replan-interval 控制）:
  1. 观测球当前位置和速度
  2. find_hitting_point_physics → 物理仿真预测击打点
  3. Softmin 多终端代价 → 允许在候选时间窗口内任意时刻击球
  4. 生成 Warm-start 控制序列（后摆轨迹 PD）
  5. solve_few_iters → iLQR 优化轨迹
  6. 安全滤波（关节/TCP/半空间约束）
  7. 执行第一个力矩指令
  8. 下一时间步重复
```

### Tube-based Robust Hitting

不确定性管道建模球到达时间的偏差，通过空间走廊式代价提升鲁棒性：

```
σ(t) = σ₀ + σᵥ·t + σₐ·t²       （不确定性管道）
候选击球窗口：以 best_k 为中心，window_half_ms 为半宽

走廊代价（不绑定时间-空间对应）:
  1. 垂直偏离代价（hinge loss）
  2. 速度方向代价（球拍沿球轨迹线方向运动）
  3. 法向量代价（拍面朝向来球方向）
```

### 多层安全滤波

```
1. 关节约束：位置/速度/加速度/力矩 四重限制
2. TCP 速度硬限制
3. X 平面墙：臂不越过身体中线（X=0）
4. 逐步安全滤波：β = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
```

### 代价函数

```
终端代价（Softmin 多终端）:
  J_hit = w_p·||p_ee − p_hit||² + w_v·||v_ee − v_des||² + w_n·||n − n_des||²

运行代价:
  J_run = Σ [ R_schedule_k·||u_k||² + Q_qdot·||qdot||² + Q_qddot·||qddot||² + Q_du·||Δu||² ]

R 退火: R 在接近击打时刻衰减到零，允许爆发力矩
```

### 击打点搜索

使用 MuJoCo 物理引擎前向仿真球的运动（含地面弹跳），逐帧打分：
```
score = 距离 + 高度惩罚 − 前方优先
```
选择 score 最低的帧作为击打点。

---

## 目录结构

```
mujoco_sim/
├── README.md
├── requirements.txt
├── AGENTS.md                          # 项目开发规范
├── configs/
│   ├── default.yaml                   # 基础仿真参数、代价权重、网球参数
│   ├── mpc.yaml                       # MPC 专用参数
│   ├── cost_hitting.yaml              # HittingCost 代价权重
│   ├── v4_follow_through.yaml         # V4 随挥配置
│   └── v5_active_hit.yaml             # V5 主动击打配置（继承 default）
│
├── scripts/
│   ├── rm65_mpc_tube_constraint_realtime_v5.py   # ★ 主脚本：MPC+iLQR+Tube+安全滤波+主动击打
│   ├── rm65_mpc_tube_constraint_realtime_v4.py   # V4：Softmin 多终端改进
│   ├── rm65_mpc_tube_constraint_realtime.py      # V1：Tube+硬半空间约束
│   ├── rm65_mpc_tube_constraint.py               # 离线版 Tube MPC
│   ├── rm65_mpc_tube.py                          # Tube 基线
│   ├── rm65_mpc_ilqr_5_5.py                     # MPC+iLQR 基线（后摆+退火+法向量）
│   ├── rm65_joint_viewer.py                      # 关节调节查看器
│   └── ...                                       # 实验/评估/绘图脚本
│
├── src/
│   ├── robot/
│   │   ├── rm65_model.xml             # RM-65 双臂 + 球拍 MuJoCo 模型
│   │   └── kinematics.py              # 正运动学 / 雅可比矩阵
│   ├── sim/
│   │   ├── rm65_env.py                # RM-65 MuJoCo 环境封装（双臂 + 球 + 碰撞）
│   │   └── viewer.py                  # 可视化工具
│   ├── ilqt/
│   │   ├── solver.py                  # ILQTSolver（后向递推 + solve_few_iters）
│   │   ├── utils.py                   # 前向传递 + 线搜索
│   │   ├── cost.py                    # HittingCost 代价函数
│   │   ├── async_replanner.py         # 异步重规划器（后台线程）
│   │   ├── retiming.py                # 轨迹重定时
│   │   └── robot_limits.py            # 多层安全滤波（关节/TCP/半空间）
│   ├── dynamics/
│   │   ├── linearize.py               # 解析/有限差分动力学线性化
│   │   └── simulate.py                # 前向仿真 / rollout
│   ├── tennis/
│   │   ├── ball.py                    # 网球抛物线轨迹 + 弹跳模型 + 发球生成
│   │   └── hitting.py                 # 击打点搜索 + 期望击打速度计算
│   ├── cpp/                           # C++ 加速实现
│   │   ├── core_ext.cpp               # iLQR 核心计算
│   │   ├── forward_pass.cpp           # 前向传递
│   │   ├── linearize.cpp              # 线性化
│   │   └── solver_cpp.py              # Python 绑定
│   └── utils/
│       └── math_utils.py              # 通用数学工具
│
├── tests/                             # 单元测试
├── skills/                            # Agent Skill 定义
└── docs/                              # 技术文档
```

---

## 配置文件

### `configs/default.yaml` — 基础参数

```yaml
sim:
  dt: 0.005              # 仿真步长 (s)

cost:
  Q_p: [50000, 50000, 50000]   # 终端位置代价权重
  Q_v: [200, 200, 200]         # 终端速度代价权重
  R: 0.0001                    # 控制代价权重

hitting:
  racket_speed: 1.8             # 期望击球速度 (m/s)
  hit_direction: [0.0, -1.0, 0.3]  # 挥拍方向
  workspace_radius: 0.85        # 工作空间半径 (m)
  shoulder_pos: [0.0, -0.12, 1.163]
```

### `configs/v5_active_hit.yaml` — V5 主动击打（覆盖 default）

```yaml
hitting:
  follow_through_length: 0.5      # 随挥距离 (m)
  follow_through_steps: 60        # 随挥步数 (300ms)
  follow_through_v_terminal: 0.3  # 随挥终端速度 (m/s)

cost:
  Q_qdot: 0.001                   # 关节速度代价
  Q_qddot: 0.0005                 # 关节加速度代价
  Q_du: 0.001                     # 控制增量代价
```

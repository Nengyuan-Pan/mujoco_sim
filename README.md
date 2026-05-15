# RM-65 网球机器人 — MPC+iLQR 击打仿真

RM-65 双臂机器人网球击打 MuJoCo 仿真项目。使用 **MPC（模型预测控制）** 作为外层闭环框架，**iLQR（迭代线性二次调节器）** 作为内层轨迹优化求解器。包含后摆 Warm-start、R 退火调度、拍面法向量约束等优化策略。球和机械臂均由 MuJoCo 物理引擎驱动，球拍-球碰撞产生真实击打效果。

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

### 主脚本：MPC 击打仿真

```bash
# 默认参数运行
python scripts/rm65_mpc_ilqr_5_5.py

# 带可视化回放
python scripts/rm65_mpc_ilqr_5_5.py --viewer

# 指定随机种子
python scripts/rm65_mpc_ilqr_5_5.py --seed 42 --viewer

# 调整后摆 + R 退火
python scripts/rm65_mpc_ilqr_5_5.py --seed 42 --r-decay 0.8 --bs-ratio 0.1 --viewer

# 调整拍面法向量约束
python scripts/rm65_mpc_ilqr_5_5.py --seed 42 --normal-weight 100000 --viewer
```

### 关节调节查看器

```bash
python scripts/rm65_joint_viewer.py
```

打开后可拖动右侧 Control 面板滑条直接控制关节。按键：`R` 重置、`P` 打印、`F` 切换坐标轴。

### 实时连续击打

```bash
python scripts/rm65_realtime_play.py
python scripts/rm65_realtime_play.py --interval 3.0   # 每 3 秒发球
python scripts/rm65_realtime_play.py --seed 42 --max-serves 10
```

---

## 命令行参数 (`rm65_mpc_ilqr_5_5.py`)

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--viewer` | flag | — | 计算完成后 MuJoCo 查看器回放 |
| `--seed` | int | None | 随机种子 |
| `--fd` | flag | — | 有限差分线性化（默认解析） |
| `--horizon` | int | None | 短地平线步数（覆盖 mpc.yaml） |
| `--iter` | int | None | 每次重规划迭代数（覆盖 mpc.yaml） |
| `--backswing` | float | 0.6 | 后摆幅度 (rad) |
| `--bs-ratio` | float | 0.35 | 后摆占比 (0~1) |
| `--no-backswing` | flag | — | 禁用后摆 |
| `--r-decay` | float | 0.30 | R 退火占比 (0~1) |
| `--no-r-decay` | flag | — | 禁用 R 退火 |
| `--normal-weight` | float | 500000 | 拍面法向量代价权重 (0=禁用) |
| `--normal-flip` | flag | — | 翻转法向量方向 |
| `--fix-joint5` | flag | — | 固定第 6 关节 |
| `--hit-shift` | float | 0.01 | 击打目标沿挥拍方向前移 (m) |
| `--ball-speed` | float | None | 球到达击打点时的速度 (m/s)，不指定则随机 |

---

## 目录结构

```
mujoco_sim/
├── README.md
├── requirements.txt
├── AGENTS.md                          # 项目开发规范（中文注释、类型提示等）
├── MUJOCO-LOG.TXT                     # MuJoCo 运行日志
│
├── configs/
│   ├── default.yaml                   # 仿真参数、代价权重、网球参数、初始位姿
│   ├── mpc.yaml                       # MPC 专用参数 + cost_type 切换
│   └── cost_hitting.yaml              # HittingCost 代价权重
│
├── scripts/
│   ├── rm65_mpc_ilqr_5_5.py           # ★ 主脚本：MPC+iLQR 击打（后摆+退火+法向量）
│   ├── rm65_mpc_ilqt.py               # MPC+iLQT 基线（无后摆优化）
│   ├── rm65_joint_viewer.py           # 关节调节查看器（位置执行器，可拖动滑条）
│   ├── rm65_realtime_play.py          # 实时连续击打（自动发球循环）
│   ├── run_rm65.py                    # 雅可比转置实时控制器
│   ├── train_ilqt.py                  # 离线 iLQT 规划器（UR5e 遗留）
│   ├── train_mpc.py                   # 离线 MPC 规划器（UR5e 遗留）
│   └── eval_sim.py                    # 仿真评估脚本（UR5e 遗留）
│
├── src/
│   ├── robot/
│   │   ├── rm65_model.xml             # RM-65 双臂 + 球拍 MuJoCo 模型（唯一真相源）
│   │   ├── model.xml                  # UR5e 右臂模型（遗留）
│   │   └── kinematics.py              # 正运动学 / 雅可比矩阵
│   ├── sim/
│   │   ├── rm65_env.py                # RM-65 MuJoCo 环境封装（双臂 + 球 + 碰撞 + 弹跳）
│   │   ├── env.py                     # UR5e MuJoCo 环境封装（遗留）
│   │   └── viewer.py                  # 可视化工具（离线回放 + matplotlib 绘图）
│   ├── ilqt/
│   │   ├── solver.py                  # ILQTSolver（后向递推 + solve_few_iters）
│   │   ├── utils.py                   # 前向传递 + 线搜索
│   │   ├── cost.py                    # 向后兼容 shim，重导出 HittingCost
│   │   └── costs/                     # ★ 代价函数子包（插件化架构）
│   │       ├── __init__.py            # COST_REGISTRY + create_cost 工厂
│   │       ├── base.py                # BaseCost(ABC) + EndEffectorCost
│   │       └── hitting.py             # HittingCost（终端位置+速度+法向量+运行代价）
│   ├── dynamics/
│   │   ├── linearize.py               # 解析/有限差分动力学线性化（A, B 矩阵）
│   │   └── simulate.py                # 前向仿真 / rollout
│   ├── tennis/
│   │   ├── ball.py                    # 网球抛物线轨迹 + 弹跳模型 + 发球生成
│   │   └── hitting.py                 # 击打点搜索（解析 + 物理仿真）+ 权重调度
│   └── utils/
│       └── math_utils.py              # 通用数学工具
│
├── tests/                             # 单元测试
│   ├── test_kinematics.py             # 运动学 + 模型加载测试
│   ├── test_ball.py                   # 球轨迹 + 弹跳模型测试
│   ├── test_linearize.py              # 解析 vs 有限差分线性化一致性测试
│   └── test_mpc.py                    # 权重调度 + MPC 组件测试
│
├── skills/                            # Agent Skill 定义
│   ├── framework_design.md            # 代码框架设计 skill
│   ├── file_management.md             # 文件管理 skill
│   └── sim_run.md                     # 仿真运行 skill
│
├── docs/
│   └── rm65_tennis_report.md          # 项目技术报告
│
└── assets/
    └── rm_65/                         # RM-65 机器人资产（URDF、STL 网格）
```

---

## 核心算法

### MPC + iLQR 闭环架构

```
每 15 步重规划（由 mpc.yaml 中 replan_interval 控制）:
  1. 观测球当前位置和速度
  2. find_hitting_point_physics → 物理仿真预测击打点
  3. 构建代价函数 HittingCost（终端位置+速度+法向量）
  4. 生成 Warm-start 控制序列（后摆轨迹 PD）
  5. solve_few_iters → iLQR 优化轨迹
  6. 执行第一个力矩指令
  7. 下一时间步重复
```

### 代价函数

代价函数采用插件化架构（`src/ilqt/costs/`），通过 `COST_REGISTRY` 注册表 + `mpc.yaml` 中 `cost_type` 字段切换。
新增代价类型只需：新建 `costs/xxx.py` + 新建 `configs/cost_xxx.yaml` + 注册表加一行。

当前默认使用 HittingCost：

```
终端代价:
  J_hit = w_p·||p_ee − p_hit||² + w_v·||v_ee − v_des||² + w_n·||n − n_des||²

运行代价:
  J_run = Σ [ R_schedule_k·||u_k||² + Q_joint·||q − q_des||² ]

R 退火: R 在接近击打时刻衰减到零，允许爆发力矩
Q_joint: 关节空间跟踪（保护后摆方向不被优化器洗掉）
```

### 击打点搜索

使用 MuJoCo 物理引擎前向仿真球的运动（含地面弹跳），逐帧打分：
```
score = 距离 + 高度惩罚 − 前方优先
```
选择 score 最低的帧作为击打点。

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
  racket_speed: 5.0             # 期望击球速度 (m/s)
  hit_direction: [0.0, -1.0, 0.3]  # 挥拍方向
  workspace_radius: 0.85        # 工作空间半径 (m)
  shoulder_pos: [0.0, -0.12, 1.163]  # 肩关节世界坐标
```

### `configs/mpc.yaml` — MPC 专用参数

```yaml
mpc:
  cost_type: hitting             # 代价函数类型（对应 configs/cost_*.yaml）
  total_horizon: 200           # 总仿真步数
  fixed_horizon: 20            # 短地平线步数
  replan_interval: 15          # 重规划间隔
  max_iter_per_plan: 2         # 每次重规划迭代数
  use_analytical: true         # 使用解析线性化
  use_bounce: true             # 启用弹跳模型
  bounce_restitution: 0.75     # 弹跳恢复系数
```

### `configs/cost_hitting.yaml` — HittingCost 代价权重

```yaml
terminal:
  Q_p: [50000.0, 50000.0, 50000.0]   # 终端位置代价权重
  Q_v: [200.0, 200.0, 200.0]         # 终端速度代价权重
  Q_n: 500000.0                       # 拍面法向量代价权重 (0=禁用)

control:
  R: 0.0001                           # 基础控制代价权重

running:
  Q_p_ratio: 0.0                      # 运行位置代价比例

joint_tracking:
  Q_joint: {0: 500.0}                 # 关节跟踪权重
```

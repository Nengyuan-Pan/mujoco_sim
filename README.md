# RM-65 网球机器人 — MPC+iLQR 击打仿真

RM-65 双臂机器人网球击打 MuJoCo 仿真项目。使用 **MPC（模型预测控制）** 作为外层闭环框架，**iLQR（迭代线性二次调节器）** 作为内层轨迹优化求解器。包含后摆 Warm-start、R 退火调度、拍面法向量约束等优化策略。

---

## 环境安装

```bash
# 安装 Python 依赖
pip install -r requirements.txt
```

依赖：`mujoco>=3.0`, `numpy`, `scipy`, `matplotlib`, `pyyaml`

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

---

## 命令行参数 (`rm65_mpc_ilqr_5_5.py`)

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--viewer` | flag | — | 计算完成后 MuJoCo 查看器回放 |
| `--seed` | int | None | 随机种子 |
| `--fd` | flag | — | 有限差分线性化（默认解析） |
| `--backswing` | float | 0.6 | 后摆幅度 (rad) |
| `--bs-ratio` | float | 0.35 | 后摆占比 (0~1) |
| `--no-backswing` | flag | — | 禁用后摆 |
| `--r-decay` | float | 0.70 | R 退火占比 (0~1) |
| `--no-r-decay` | flag | — | 禁用 R 退火 |
| `--normal-weight` | float | 100000 | 拍面法向量代价权重 (0=禁用) |
| `--normal-flip` | flag | — | 翻转法向量方向 |
| `--fix-joint5` | flag | — | 固定第 6 关节 |
| `--hit-shift` | float | 0.0 | 击打目标前移 (m) |

---

## 目录结构

```
tennis_robot/
├── README.md
├── requirements.txt
├── AGENTS.md                          # 项目开发规范（中文注释、类型提示等）
│
├── configs/
│   ├── default.yaml                   # 仿真参数、代价权重、网球参数、初始位姿
│   └── mpc.yaml                       # MPC 专用参数（horizon、replan_interval、退火等）
│
├── scripts/
│   ├── rm65_mpc_ilqr_5_5.py           # ★ 主脚本：MPC+iLQR 击打（后摆+退火+法向量）
│   ├── rm65_joint_viewer.py           # 关节调节查看器（位置执行器，可拖动滑条）
│   ├── rm65_mpc_ilqt.py               # 旧版 MPC 基线脚本
│   ├── run_rm65.py                    # 雅可比转置实时控制器
│   ├── eval_sim.py                    # 仿真评估脚本
│   ├── train_ilqt.py                  # 离线 iLQT 规划器（UR5e）
│   ├── train_mpc.py                   # 离线 MPC 规划器（UR5e）
│   └── rm65_realtime_play.py          # 实时播放器
│
├── src/
│   ├── robot/
│   │   ├── rm65_model.xml             # RM-65 双臂 + 球拍 MuJoCo 模型（唯一真相源）
│   │   ├── model.xml                  # UR5e 右臂模型
│   │   ├── kinematics.py              # 正运动学 / 雅可比矩阵
│   │   └── meshes/                    # STL 网格文件
│   ├── sim/
│   │   ├── rm65_env.py                # RM-65 MuJoCo 环境封装（双臂 + 球 + 碰撞 + 弹跳）
│   │   ├── env.py                     # UR5e MuJoCo 环境封装
│   │   └── viewer.py                  # 可视化工具（离线回放 + matplotlib 绘图）
│   ├── ilqt/
│   │   ├── solver.py                  # ILQTSolver（后向递推 + solve_few_iters）
│   │   ├── cost.py                    # HittingCost（终端位置+速度+法向量+运行代价）
│   │   └── utils.py                   # 前向传递 + 线搜索
│   ├── dynamics/
│   │   ├── linearize.py               # 解析/有限差分动力学线性化（A, B 矩阵）
│   │   └── simulate.py                # 前向仿真 / rollout
│   ├── tennis/
│   │   ├── ball.py                    # 网球抛物线轨迹生成 + 由目标反推初速
│   │   └── hitting.py                 # 击打点搜索（MuJoCo 物理仿真 + 打分筛选）
│   └── utils/
│       └── math_utils.py              # 通用数学工具
│
├── tests/                             # 单元测试
├── docs/                              # 文档
├── results/                           # 输出（图片、日志）
└── assets/rm_65/                      # RM-65 机器人资产（URDF、网格、感知模型）
```

---

## 核心算法

### MPC + iLQR 闭环架构

```
每 10 步重规划:
  1. 观测球当前位置和速度
  2. find_hitting_point_physics → 预测击打点
  3. 构建代价函数 HittingCost（终端位置+速度+法向量）
  4. 生成 Warm-start 控制序列（后摆轨迹 PD）
  5. solve_few_iters → iLQR 优化轨迹
  6. 执行第一个力矩指令
  7. 下一时间步重复
```

### 代价函数

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

## 配置文件 (`configs/default.yaml`)

```yaml
sim:
  dt: 0.005              # 仿真步长 (s)

cost:
  Q_p: [2000, 2000, 2000]       # 终端位置代价权重
  Q_v: [50000, 50000, 50000]    # 终端速度代价权重
  R: 0.000001                    # 控制代价权重

hitting:
  racket_speed: 10.0             # 期望击球速度 (m/s)
  hit_direction: [-1, 0, 0.3]    # 挥拍方向
  workspace_radius: 0.85         # 工作空间半径 (m)
```

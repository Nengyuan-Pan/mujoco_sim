# AGENTS.md - RM-65 网球机器人项目规范

## 项目概述
本项目使用 **MPC（模型预测控制）+ iLQR（迭代线性二次调节器）** 闭环架构，
解决 **RM-65 双臂人形机器人**网球击打场景。外层 MPC 每隔若干步重规划，
内层 iLQR 求解短地平线最优轨迹。包含后摆 Warm-start、R 退火调度、
拍面法向量约束等优化策略。球和机械臂均由 MuJoCo 物理引擎驱动，
球拍-球碰撞产生真实击打效果。

历史背景：项目最初基于 UR5e 单臂 + 纯 iLQT 离线规划（代码保留在 `model.xml` /
`env.py` / `train_ilqt.py` 中），现已全面迁移到 RM-65 双臂 + MPC+iLQR 在线闭环。

## 语言与注释规范
- **所有代码使用 Python 编写**
- **所有代码注释、docstring 必须使用中文**
- **所有深度思考、设计决策说明使用中文**
- 变量名、函数名、类名使用英文（遵循 Python 命名规范）
- 类型提示（type hints）必须标注在所有函数签名上
- 公有函数必须有中文 docstring（Google 风格）

## 技术栈
- **语言**: Python 3.9+
- **仿真**: MuJoCo（mujoco Python 包 >= 3.0）— 跨平台 Windows/Ubuntu
- **数值计算**: NumPy, SciPy
- **可视化**: MuJoCo 内置查看器 + matplotlib（轨迹绘图）
- **包管理**: pip + requirements.txt

## 机器人模型定义

### 主模型：RM-65 双臂（`src/robot/rm65_model.xml`）
RM-65 人形机器人，仅驱动右臂 6-DOF 挥拍，左臂 6-DOF 保持零位（PD 控制）。
底盘固定在地面上，腰部升降关节已固定。

#### 右臂关节分配

| 关节编号 | MuJoCo 关节名  | 说明                       | ctrlrange (Nm) | range (deg) |
|----------|----------------|----------------------------|----------------|-------------|
| 0        | r_joint1       | 肩关节偏航（绕局部 Z）     | ±60            | ±178        |
| 1        | r_joint2       | 肩关节俯仰（绕局部 Z）     | ±60            | ±130        |
| 2        | r_joint3       | 肘关节（绕局部 Z）         | ±30            | ±135        |
| 3        | r_joint4       | 腕关节 1（绕局部 Z）       | ±10            | ±178        |
| 4        | r_joint5       | 腕关节 2（绕局部 Z）       | ±10            | ±128        |
| 5        | r_joint6       | 腕关节 3 / 法兰旋转        | ±10            | ±360        |

左臂关节名：`l_joint1` ~ `l_joint6`，参数相同，但不参与规划。

#### 状态向量（`RM65Env` 视角）
- **右臂状态**（iLQR 规划用）: `x = [q(6), qdot(6)]` ∈ R^12
- **控制向量**: `u = tau(6)` ∈ R^6（右臂关节力矩）
- **MuJoCo qpos 布局**: 右臂 `[0:6]`，左臂 `[6:12]`，球 freejoint `[12:19]`
- **MuJoCo qvel 布局**: 右臂 `[0:6]`，左臂 `[6:12]`，球 `[12:18]`
- **执行器 ctrl 布局**: 右臂 `[0:6]`，左臂 `[6:12]`
- **末端执行器**: `racket_center` site（球拍面中心点）
- **球拍**: 从法兰沿手柄延伸，拍面法线沿 Y 方向（与法兰 Z 轴呈 90°）

#### 碰撞位掩码方案
| 对象           | contype | conaffinity | 碰撞对象           |
|----------------|---------|-------------|---------------------|
| 地面           | 1       | 15 (1+2+4+8)| 机器人+球拍+球      |
| 机器人结构体   | 1       | 1           | 地面                |
| 机械臂连杆     | 2       | 2           | 臂间自碰撞          |
| 球拍           | 4       | 8           | 仅球                |
| 球             | 8       | 5 (1+4)     | 地面+球拍           |
| 视觉网格       | 0       | 0           | 无碰撞（仅渲染）    |

### 遗留模型：UR5e（`src/robot/model.xml`）
UR5e 6-DOF 单臂模型，仅用于 `train_ilqt.py` 离线规划。MuJoCo 模型中关节名为
`shoulder_pan_joint`、`shoulder_lift_joint` 等（menagerie 风格，带 `_joint` 后缀）。
不再作为主要开发目标。

## 项目目录结构
```
mujoco_sim/
├── AGENTS.md                          # 本文件 — 项目规范与 Agent 指令
├── README.md                          # 项目说明文档
├── requirements.txt                   # Python 依赖
├── MUJOCO-LOG.TXT                     # MuJoCo 运行日志
│
├── skills/                            # Skill 定义，供 Agent 工作流调用
│   ├── framework_design.md            # 代码框架设计 skill
│   ├── file_management.md             # 文件管理 skill
│   └── sim_run.md                     # 仿真运行 skill
│
├── configs/
│   ├── default.yaml                   # 默认超参数（仿真、iLQT、代价、球、击打）
│   └── mpc.yaml                       # MPC 专用参数（horizon、分层策略、退火、发球）
│
├── src/
│   ├── __init__.py
│   ├── robot/                         # 机器人模型定义
│   │   ├── __init__.py
│   │   ├── rm65_model.xml             # ★ RM-65 双臂 + 球拍 MuJoCo 模型（主模型）
│   │   ├── model.xml                  # UR5e 单臂模型（遗留，仅 train_ilqt.py 使用）
│   │   └── kinematics.py              # 正运动学 / 雅可比矩阵 / 工作空间估计
│   ├── dynamics/                      # 动力学计算
│   │   ├── __init__.py
│   │   ├── linearize.py               # 动力学线性化（有限差分 + 解析两种实现）
│   │   └── simulate.py                # 前向仿真 / rollout
│   ├── ilqt/                          # iLQR/iLQT 求解器核心
│   │   ├── __init__.py
│   │   ├── solver.py                  # ILQTSolver（后向递推 + solve_few_iters）
│   │   ├── cost.py                    # HittingCost（终端位置+速度+法向量+运行代价）
│   │   └── utils.py                   # 前向传递 + 线搜索 + 正则化
│   ├── tennis/                        # 网球场景相关
│   │   ├── __init__.py
│   │   ├── ball.py                    # 抛物线轨迹 + 弹跳模型 + 发球生成
│   │   └── hitting.py                 # 击打点搜索（解析+物理仿真）+ 权重调度
│   ├── sim/                           # MuJoCo 仿真封装
│   │   ├── __init__.py
│   │   ├── rm65_env.py                # ★ RM65Env（双臂+球+碰撞+弹跳）
│   │   ├── env.py                     # MujocoEnv（UR5e 遗留环境）
│   │   └── viewer.py                  # 可视化（MuJoCo 回放 + matplotlib 绘图）
│   └── utils/
│       ├── __init__.py
│       └── math_utils.py              # 通用数学工具
│
├── scripts/
│   ├── rm65_mpc_ilqr_5_5.py           # ★ 主脚本：MPC+iLQR 击打（后摆+退火+法向量）
│   ├── rm65_mpc_ilqt.py               # MPC+iLQT 基线（RM-65，无后摆优化）
│   ├── rm65_joint_viewer.py           # RM-65 关节调节查看器（位置执行器，可拖动）
│   ├── rm65_realtime_play.py          # RM-65 实时连续击打（自动发球循环）
│   ├── run_rm65.py                    # RM-65 雅可比转置实时控制器
│   ├── train_ilqt.py                  # UR5e 离线 iLQT 规划（遗留）
│   ├── train_mpc.py                   # UR5e MPC 基线脚本（遗留）
│   └── eval_sim.py                    # UR5e 仿真评估（遗留）
│
├── tests/
│   ├── test_kinematics.py             # 运动学 + 模型加载测试
│   ├── test_ball.py                   # 球轨迹 + 弹跳模型测试
│   ├── test_linearize.py              # 解析线性化 vs 有限差分一致性测试
│   └── test_mpc.py                    # 权重调度 + MPC 组件测试
│
├── docs/
│   └── rm65_tennis_report.md          # 项目技术报告
│
└── assets/
    └── rm_65/                         # RM-65 机器人资产（URDF、STL 网格）
```

## 核心算法说明

### MPC + iLQR 闭环架构（主算法）
主脚本 `rm65_mpc_ilqr_5_5.py` 采用 MPC 外层闭环 + iLQR 内层轨迹优化：
```
每 replan_interval 步重规划:
  1. 观测球当前位置和速度（MuJoCo 物理状态）
  2. find_hitting_point_physics → 物理仿真预测击打点
  3. 构建代价函数 HittingCost（终端位置+速度+法向量）
  4. 生成 Warm-start 控制序列（后摆轨迹 PD）
  5. solve_few_iters → iLQR 优化短地平线轨迹
  6. 执行第一个力矩指令
  7. 下一时间步重复
```

### iLQR 求解器（`ILQTSolver`）
- **后向传递**: 从终端时刻到初始时刻，计算增益矩阵 K_k, k_k
- **前向传递**: 用线搜索（alpha_list）更新轨迹和控制序列
- **正则化**: Levenberg-Marquardt 风格（mu_min, mu_max, delta_0）
- **两种线性化**: 解析法（`linearize_analytical`，推荐）和有限差分法（`linearize_dynamics`，备用）
- **短地平线模式**: `solve_few_iters()` 专为 MPC 设计，只迭代 2~3 次

### 代价函数（`HittingCost`）
```
终端代价:
  J_terminal = ||p_ee - p_hit||²_Q_p + ||v_ee - v_des||²_Q_v + Q_n·||n - n_des||²

运行代价:
  J_running = Σ_k [ R_schedule_k · ||u_k||² + Q_p_running · ||p_ee - p_ball||²
                    + Σ_j Q_joint_j · (q_j - q_des_j)² ]

其中:
  - Q_n, n_des: 终端拍面法向量代价（默认 0 = 禁用）
  - R_schedule: 时变 R（退火调度，接近击打时衰减到零）
  - Q_joint: 关节空间跟踪（保护后摆方向不被优化器洗掉）
  - Q_p_running: 运行位置代价（跟踪球位置）
```

### 后摆 Warm-start
主脚本生成后摆→前挥初始控制序列，分两段：
- **后摆段**（前 bs_ratio 比例）：关节1 向后旋转 backswing 弧度
- **前挥段**（剩余比例）：PD 控制追踪击打点方向
- 效果：为 iLQR 提供物理合理的初始猜测，避免陷入局部最优

### R 退火调度
`R_schedule` 在接近击打时刻线性衰减到零：
- 前段高 R → 抑制力矩，轨迹平滑
- 后段低 R → 允许爆发力矩，精确击打
- `r_decay` 参数控制退火占比

### 拍面法向量约束
终端代价中加入拍面法向量惩罚：
- `Q_n` 权重控制法向量偏离惩罚强度
- `n_des` 为期望法向量方向
- `--normal-flip` 参数可翻转法向量方向

### 击打点搜索
两种实现：
1. **解析模型**（`find_hitting_point`）：抛物线公式 + 可选弹跳模型
2. **物理仿真**（`find_hitting_point_physics`，推荐）：使用 MuJoCo 物理引擎前向仿真球运动（含地面弹跳），逐帧打分筛选最佳击打时刻

评分函数：`score = 距离 + 高度惩罚 − 前方优先`

### 网球轨迹预测（`ball.py`）
- 基本抛物线：`p(t) = p0 + v0*t + 0.5*g*t²`
- 含弹跳模型：解析计算弹跳时刻，弹起后 Z 速度反转×恢复系数
- 发球生成：`generate_hittable_ball`、`generate_serve_ball`、`generate_ball_to_target_box`

## 构建与运行命令

### 环境配置
```bash
# 创建独立 conda 环境（推荐）
conda create -n mujoco_tennis python=3.11
conda activate mujoco_tennis
pip install -r requirements.txt

# 或直接 pip 安装
pip install -r requirements.txt
```

依赖：`mujoco>=3.0`, `numpy>=1.24`, `scipy>=1.10`, `matplotlib>=3.7`, `pyyaml>=6.0`,
`pytest>=7.0`, `ruff>=0.1`, `mypy>=1.5`

### 运行命令

#### 主脚本（RM-65 MPC+iLQR）
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

#### 其他脚本
```bash
# RM-65 关节调节查看器（拖动滑条直接控制关节）
python scripts/rm65_joint_viewer.py

# RM-65 实时连续击打
python scripts/rm65_realtime_play.py --interval 3.0

# UR5e 离线 iLQT 优化（遗留）
python scripts/train_ilqt.py --viewer
```

#### 开发命令
```bash
# 运行测试
pytest tests/

# 代码检查
ruff check src/ tests/ scripts/

# 类型检查
mypy src/
```

## 编码规范
- 所有代码注释、docstring 使用**中文**
- 使用 `numpy` 进行数组运算，禁止对数组使用原生 Python 循环
- RM-65 模型定义在 `src/robot/rm65_model.xml`，是 DOF 数、关节名和 ctrlrange 的唯一事实来源
- UR5e 模型定义在 `src/robot/model.xml`（遗留）
- 可调参数放在 `configs/*.yaml` 中，不要硬编码
- 日志使用 Python `logging` 模块，不要用 `print`
- 测试文件与 `src/` 结构对应，放在 `tests/` 下

## 跨平台注意事项
- **Windows**: MuJoCo 查看器原生支持，使用 `mujoco.viewer.launch_passive()`
- **Ubuntu**: 同样 API；无头服务器上设置 `MUJOCO_GL=osmesa` 或 `egl`
- 不要使用 `glx` 或平台特定的渲染调用
- 文件路径使用 `pathlib.Path`，不要拼接字符串

## MuJoCo 关键注意事项
- **`range` 属性使用角度（degrees）**：MuJoCo 3.8+ 的 `range` 属性以度为单位，
  会自动转换为弧度。例如 `range="-178 178"` 对应约 ±3.11 rad
- **`ctrlrange` 与力矩裁剪**：`RM65Env.step()` 中使用 `model.actuator_ctrlrange` 裁剪控制
- **碰撞位掩码**：RM-65 模型使用位掩码分级碰撞，不要随意修改 contype/conaffinity
- **积分器**：`rm65_model.xml` 使用 `integrator="implicitfast"`，提高大扭矩下的仿真稳定性
- **球拍-球碰撞**：`<pair>` 指定 `condim="1"`（仅法向力）+ `solref="-5000 0"`（高刚度零阻尼）
  实现近弹性碰撞
- **相邻臂段排除**：`<contact><exclude>` 排除共享关节的相邻连杆碰撞
- **球弹跳**：`RM65Env._handle_ball_bounce()` 在 MuJoCo 步后解析处理球触地弹跳

## 优化策略

### MPC 分层策略
- **远距**（球距击打点远）：雅可比转置开环控制，快速接近
- **中距**：iLQR 单次迭代（max_iter=1），轻量修正
- **近距**：iLQR 多次迭代（max_iter=2~3），精细优化

### 后摆 + R 退火 + 法向量（rm65_mpc_ilqr_5_5.py 独有）
- **后摆 Warm-start**: 关节1 先向后摆动再前挥，生成物理合理的初始轨迹
- **R 退火**: 控制代价权重在接近击打时刻衰减，允许爆发力矩
- **法向量约束**: 终端代价惩罚拍面朝向偏离，确保击打方向正确

### 遗留策略（train_ilqt.py）
- **两阶段 iLQT**：阶段1 仅位置代价（高 Q_p），快速到达击打点；阶段2 完整代价（位置+速度），热启动精细调整
- **雅可比转置初始控制**：`J^T * (p_hit - p_ee)` 生成初始控制序列

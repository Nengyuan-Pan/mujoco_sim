# AGENTS.md - Tennis Robot 项目规范

## 项目概述
本项目使用 **MPC + iLQR + Tube** 框架，解决 **RM-65B 双臂机械臂**挥拍击打网球的场景。
机器人模型为 **RM-65B（12自由度双臂工业机械臂）**，安装在垂直桩柱上方，
双末端各连接垂直网球拍（物理上仅右臂装拍，左臂保持零位），
在给定网球飞来轨迹的情况下，计算最优挥拍轨迹，使末端执行器（球拍面）在正确的时间和位置以期望的速度击中网球。

- 状态向量: x = [q(12), qdot(12)] ∈ R^24（双臂关节位置 + 关节速度）; 球自由关节另计
- 控制向量: u = tau(6) ∈ R^6（仅驱动右臂，左臂保持零位）
- 末端执行器: 球拍面中心点（racket_center site）
- 实际关节: r_joint1~r_joint6（右臂）+ l_joint1~l_joint6（左臂）
- MuJoCo 模型: nq=19(双臂 12+球 7), nv=18(双臂 12+球 6), nu=12(右臂 motor 6 + 左臂 motor 6)

## 语言与注释规范
- **所有代码使用 Python 编写**
- **所有代码注释、docstring 必须使用中文**
- **所有深度思考、设计决策说明使用中文**
- 变量名、函数名、类名使用英文（遵循 Python 命名规范）
- 类型提示（type hints）必须标注在所有函数签名上
- 公有函数必须有中文 docstring（Google 风格）

## 技术栈
- **语言**: Python 3.11+ (conda 环境 `mujoco_tennis`)
- **仿真**: MuJoCo 3.9+（mujoco Python 包）— 跨平台 Windows/Ubuntu
- **数值计算**: NumPy, SciPy
- **C++ 加速**: pybind11 — `src/cpp/` 下的 iLQR 核心循环（linearize_analytical_batch, forward_pass）
- **可视化**: MuJoCo 内置查看器 + matplotlib（轨迹绘图）
- **包管理**: conda + pip + requirements.txt
- **构建**: `setup.py build_ext --inplace` 编译 C++ 扩展

## 机器人模型定义
RM-65B 12 自由度（双臂各 6），关节分配如下：

### 右臂关节（驱动）
| 关节编号 | MuJoCo 关节名 | qpos 索引 | 说明 |
|----------|--------------|----------|------|
| 0 | r_joint1 | qpos[0] | 右肩偏航 |
| 1 | r_joint2 | qpos[1] | 右肩俯仰 |
| 2 | r_joint3 | qpos[2] | 右肘 |
| 3 | r_joint4 | qpos[3] | 右腕 1 |
| 4 | r_joint5 | qpos[4] | 右腕 2 |
| 5 | r_joint6 | qpos[5] | 右腕 3 |

### 左臂关节（不驱动，保持零位）
| 关节编号 | MuJoCo 关节名 | qpos 索引 |
|----------|--------------|----------|
| 6 | l_joint1 | qpos[6] |
| 7 | l_joint2 | qpos[7] |
| 8 | l_joint3 | qpos[8] |
| 9 | l_joint4 | qpos[9] |
| 10 | l_joint5 | qpos[10] |
| 11 | l_joint6 | qpos[11] |

### 球自由关节
- qpos[12:19] (7维 quaternion + xyz)
- qvel[12:18] (6维)

- MuJoCo 模型为 `src/robot/rm65_model.xml`，是 DOF 数和关节顺序的唯一事实来源
- 球拍: 连杆沿法兰局部方向延伸，球拍面在连杆末端（racket_center site）
- 臂展: 约 850mm（单臂）
- 终端执行器: `racket_center` site（右臂 r_flange → r_racket_body → r_racket → 球拍面）

## 资产目录
```
assets/rm_65/
├── urdf/                        # RM-65B URDF 源文件 + 网格
│   ├── meshes/*.STL, *.dae     # 机械臂视觉网格
│   ├── visual/*.STL            # 灵巧手 visual 网格
│   └── dh_robotics_ag95.urdf   # 灵巧手 URDF
├── realmanControlNode.py       # 真实机械臂控制节点
├── config.py                   # RM-65B 硬件参数
└── ...
```

## 项目目录结构
```
mujoco_sim/
├── AGENTS.md                          # 本文件 — 项目规范与 Agent 指令
├── setup.py                           # C++ 扩展构建脚本（pybind11）
├── requirements.txt                   # Python 依赖
├── skills/                            # Skill 定义（7 个，详见下方 Skills 表）
│   ├── framework_design.md            # 代码框架设计
│   ├── file_management.md             # 文件管理规范
│   ├── sim_run.md                     # 仿真运行流程
│   ├── experiment_design.md           # 实验设计与数据管理
│   ├── figure_generation.md           # 论文图表生成
│   ├── paper_writing.md               # 论文撰写
│   └── paper_review.md                # 论文审稿与迭代
├── src/
│   ├── __init__.py
│   ├── robot/                         # 机器人模型定义
│   │   ├── __init__.py
│   │   ├── rm65_model.xml             # MuJoCo XML 模型（双臂12DOF + 球拍 + 球）
│   │   ├── model.xml                  # 旧版单臂模型（左臂装在右侧桩柱）
│   │   └── kinematics.py              # 正运动学 / 雅可比矩阵工具
│   ├── dynamics/                      # 动力学计算
│   │   ├── __init__.py
│   │   ├── linearize.py               # 动力学线性化（fx, fu），供 iLQR 使用
│   │   └── simulate.py                # 前向仿真 / rollout
│   ├── ilqt/                          # iLQR 求解器核心
│   │   ├── __init__.py
│   │   ├── solver.py                  # iLQR 后向-前向迭代主循环（solve / solve_few_iters）
│   │   ├── cost.py                    # 代价函数（终端击打点代价 + 控制代价 + Tube 代价）
│   │   ├── utils.py                   # 增益计算、线搜索、正则化辅助函数
│   │   ├── robot_limits.py            # 关节约束 + 安全滤波（RobotLimits, strict_braking_check）
│   │   ├── retiming.py                # 时间重映射工具
│   │   ├── async_replanner.py         # 异步重规划器（后台线程 iLQR）
│   │   └── costs/                     # 模块化代价函数
│   │       ├── __init__.py
│   │       ├── base.py                # BaseCost 基类
│   │       └── hitting.py             # 击打场景专用代价
│   ├── cpp/                           # C++ 加速模块（pybind11）
│   │   ├── __init__.py
│   │   ├── solver_cpp.py              # Python 封装，桥接 C++ 和 Python
│   │   ├── core_ext.cpp               # pybind11 模块入口
│   │   ├── linearize.cpp              # 解析动力学线性化（批量）
│   │   └── forward_pass.cpp           # 前向传递（单步 + 线搜索）
│   ├── sim/                           # MuJoCo 仿真封装
│   │   ├── __init__.py
│   │   ├── env.py                     # MujocoEnv 基类
│   │   ├── rm65_env.py                # RM65Env 双臂环境封装
│   │   └── viewer.py                  # 可视化工具
│   ├── tennis/                        # 网球场景相关
│   │   ├── __init__.py
│   │   ├── ball.py                    # 网球抛物线轨迹预测
│   │   └── hitting.py                 # 击打点计算 & 球拍-球接触判断
│   └── utils/
│       ├── __init__.py
│       ├── math_utils.py              # 通用数学工具
│       ├── mujoco_loader.py           # 跨平台安全模型加载器（处理中文路径）
│       └── noise.py                   # 噪声注入（观测/力矩/初始关节随机化）
├── configs/
│   ├── default.yaml                   # 默认超参数（时间步长、iLQR 参数、关节约束）
│   ├── mpc.yaml                       # MPC 专用参数
│   ├── cost_hitting.yaml              # 代价函数权重
│   ├── v4_follow_through.yaml         # V4 随挥策略配置
│   └── v5_active_hit.yaml             # V5 主动击球配置
├── scripts/
│   ├── rm65_mpc_tube_constraint.py               # 离线仿真（根，被 exp/ 包装 import）
│   ├── rm65_mpc_tube_constraint_realtime.py      # 实时 v1（根）
│   ├── rm65_mpc_tube_constraint_realtime_v2.py   # 实时 v2（根）
│   ├── rm65_mpc_tube.py / rm65_mpc_ilqr_5_5.py  # Tube/iLQR 基线（根）
│   ├── rm65_evaluate.py                          # 评估脚本（根）
│   ├── sim/            # 独立仿真 13 个（v4/v5/fast/ilqt/train）
│   ├── exp/            # 实验设施 25 个（包装·批量·运行器·提取）
│   ├── extract/        # 结果提取 4 个（日志→CSV）
│   ├── plot/           # 论文图表 6 个
│   ├── tools/          # 独立工具 7 个（查看器·扫描·可视化）
│   ├── test/           # 快速验证 9 个
│   └── README.md       # 完整清单与说明
├── tests/
│   ├── test_kinematics.py
│   ├── test_linearize.py
│   ├── test_mpc.py
│   ├── test_ball.py
│   └── test_noise.py
├── experiment_data/                  # 实验数据（按 exp1~exp6 组织）
│   └── README.md                     # 数据存储规范
├── paper/                            # 论文 LaTeX 工程
│   ├── main.tex
│   ├── references.bib
│   ├── sections/                     # 英文各节 .tex
│   ├── sections_zh/                  # 中文草稿 .md
│   └── figures/                      # 图表
├── docs/                             # 历史文档
├── results/                          # 输出目录（日志、轨迹、视频）
└── requirements.txt
```

## Skills 参考

项目包含 7 个 Skill，位于 `skills/` 目录。Agent 在对应场景下应加载相应 Skill：

| Skill | 文件 | 用途 | 触发条件 |
|-------|------|------|---------|
| 代码框架设计 | `skills/framework_design.md` | 设计代码架构、模块归属、接口定义 | 创建/重构模块时 |
| 文件管理 | `skills/file_management.md` | 文件创建/移动/命名规范、目录结构映射 | 添加/移动文件时 |
| 仿真运行 | `skills/sim_run.md` | iLQT 训练、MuJoCo 评估、轨迹回放 | 启动仿真/训练时 |
| 实验设计与数据管理 | `skills/experiment_design.md` | 6 组实验矩阵设计、批量运行、CSV/NPZ 数据管理 | 运行批量实验时 |
| 论文图表生成 | `skills/figure_generation.md` | 8 张 IEEE RAL 论文图（系统/算法/关节/命中率/Tube/实时/诊断）| 生成论文图表时 |
| 论文撰写 | `skills/paper_writing.md` | IEEE RAL 结构、中文草稿→英文翻译、符号表 | 撰写论文时 |
| 论文审稿与迭代 | `skills/paper_review.md` | 6 维自审查单、审稿报告模板、迭代工作流 | 审查论文草稿时 |
| 实验记录 | `skills/experiment_log.md` | 实验后自动生成记录：读取CSV→聚合统计→Agent 数据观察结论+人工分析决策→更新索引 | 运行批量实验后 / "记录实验" |

### 实验数据目录
- 所有实验数据存放在 `experiment_data/` 目录
- 按 `exp1~exp6` 编号组织，每组含 `config.yaml` + `results.csv` + `raw/`
- 详见 `experiment_data/README.md`

## 核心算法说明

### MPC + iLQR + Tube 三层框架

- **MPC 外循环**: 每 `replan_interval` 步重规划，分 far/mid/near 三阶段自适应迭代次数
- **iLQR 内循环**: 后向传递计算增益 K_k, k_k + 前向传递线搜索更新轨迹
- **Tube 鲁棒层**: 空间走廊式代价（不绑定时间-空间对应），候选击球窗口以 best_k 为中心

### iLQR（迭代线性二次调节器）
- **后向传递**: Riccati 递推，从终端时刻到初始时刻计算增益矩阵 K_k, k_k
- **前向传递**: 用线搜索更新轨迹和控制序列（MPC 模式跳过线搜索，固定 alpha=0.5）
- **终端代价**: 惩罚末端执行器偏离期望击打点（位置 + 速度 + 法向量）
- **运行代价**: 惩罚过大的控制力矩 + 关节加速度 + 控制变化率
- **正则化**: Levenberg-Marquardt 风格（mu_min=1e-6, mu_max=1e10, delta_0=1.6）
- 终端代价形式：
  ```
  l_terminal(x) = ||p_ee(x) - p_hit||^2_Q_p + ||v_ee(x) - v_hit||^2_Q_v + (1 - n_racket·n_des) * Q_n
  ```
  其中 p_ee 为末端位置，p_hit 为击打点位置，v_ee 为末端速度，v_hit 为期望击打速度

### Tube-based Robust Hitting
- 不确定性管道：σ(t) = σ₀ + σᵥ·t + σₐ·t²
- 候选击球窗口：以 best_k 为中心，window_half_ms 为半宽（默认 50ms）
- 空间走廊式代价（不绑定时间-空间对应）：
  1. 垂直偏离代价（hinge loss）：球拍超出走廊半径即惩罚
  2. 速度方向代价（球拍沿球轨迹线方向运动）
  3. 法向量代价（拍面朝向来球方向）
- Softmin 终端聚合：多个候选终端代价加权，β 控制锐度
- 不确定性管道：σ(t) = σ₀ + σᵥ·t + σₐ·t²

### 多层安全滤波
- **X 平面墙预判**：臂不越过身体中线（X≥-0.1），越界 PD 推回
- **关节约束**：位置/速度/加速度/力矩四重限制
- **TCP 速度硬限制**：max_tcp_speed = 1.8 m/s
- **逐步安全滤波**：β = [0.8, 0.6, 0.4, 0.2, 0.0]，找到最大可行控制
- **终段豁免**：击球前 terminal_exempt_steps 步跳过速度检查（默认 20 步）
- **紧急制动**：所有 β 均失败时施加阻尼力矩 u = -20·qdot

### 动力学线性化
- 通过 MuJoCo 的 `mj_jac` 和 `mj_rne` 计算雅可比和动力学偏导
- 解析线性化（C++ `linearize_analytical_batch`）— MPC 默认
- 有限差分法数值线性化（开发初期，`--fd` 标志）— 较慢但更鲁棒
- 线性化结果：x_{k+1} ≈ f(x_k, u_k) = A_k δx + B_k δu + ...

### 网球轨迹预测
- 假设网球在重力作用下做抛物线运动（忽略空气阻力）
- 给定球的初始位置和速度，预测球在任意时刻的位置
- 计算击打时刻和击打点：球到达球拍可及范围内的时间点
- serve_box 模式：从 8m×0.2m×0.3m 范围内随机发球

## 优化策略
- **雅可比转置初始控制（JT warm-start）**：使用 `J^T * (p_hit - p_ee)` 生成初始控制序列，远优于零/常数力矩初始猜测
- **分阶段迭代**：far 阶段仅 JT 控制（零 iLQR 开销），near 阶段减少迭代数 + 启用 hard_constraints
- **R 退火**：控制代价 R 从击球前逐步衰减（r_decay_ratio=0.40），关节1额外衰减 10×
- **后摆策略**：五次多项式后摆轨迹，增大挥拍行程达到更高末端速度
- **随挥（V5）**：击球后 60 步（300ms）内末端沿来球反方向加速，随挥长度 0.5m
- **权重调度**：far 阶段 Q_p×5, Q_v×3；near 阶段 Q_p×8, Q_v×120

## 构建与运行命令
- 创建 conda 环境: `conda create -n mujoco_tennis python=3.11 -y`
- 激活环境: `conda activate mujoco_tennis`
- 安装依赖: `pip install -r requirements.txt`
- 编译 C++ 扩展: `python setup.py build_ext --inplace`
- 运行 MPC 仿真（当前活跃版本）: `python scripts/rm65_mpc_tube_constraint_realtime_v5.py --serve-box --ball-speed 7`
- 离线测试: `python scripts/rm65_mpc_tube_constraint.py --serve-box --ball-speed 9`
- 关节安全扫描: `python scripts/scan_joint_safety.py`
- 运行测试: `pytest tests/`
- 代码检查: `ruff check src/ tests/ scripts/`
- 类型检查: `mypy src/`

## 批量实验架构

新增实验不再需要手动写 PowerShell 循环、处理编码问题、逐个提取结果。
已有标准化的三层模板，复制参考文件后只改 3 个参数。

### 三层架构

| 层 | 文件模板 | 职责 | 改什么 |
|----|---------|------|--------|
| 包装 | `scripts/_run_expX_*.py` | monkey-patch 约束 → 构建 sys.argv → 调主脚本 | 约束参数、独有 CLI flag（如 `--no-bounce`） |
| 运行 | `scripts/run_expX_batch.py` | 遍历参数矩阵 → subprocess → UTF-8 日志 | `SPEEDS`、`SEEDS`、`TUBE_MODES` 三个列表 |
| 提取 | `scripts/extract_expX_results.py` | regex 解析日志 → results.csv + 命中率汇总 | 离线/实时格式选择、额外指标 regex |

### 参考实现（复制即改）

| 实验类型 | 包装参考 | 运行参考 | 提取参考 |
|---------|---------|---------|---------|
| 豁免约束 + 离线 | `_run_exp1_v3_exempt.py` | `run_exp1_v3_batch.py` | `extract_exp1_v3_results.py` |
| 严格约束 + 离线 | `_run_exp1_exempt.py`（改 margin→1.0） | `run_exp2_v2_batch.py` | `extract_exp2_v2_results.py` |
| 实时脚本（有 `__RESULT__`） | 不需要 | `run_exp1_batch.py` | `extract_exp2_results.py` |

### 新建实验只需 3 步

1. **建目录**：`experiment_data/expN_<name>/raw/` + `config.yaml`
2. **复制包装**：从参考中选一个 `_run_expX_*.py`，改 `constraints` 和 `sys.argv` 中的独有 flag
3. **改运行器**：复制 `run_expX_v3_batch.py`，改 `SPEEDS`、`SEEDS`、`TUBE_MODES` 三个列表

提取脚本通常无需改动，直接复用对应格式的版本。

### 已验证的效率

| 指标 | 手动模式 | 批量模式 |
|------|---------|---------|
| 单次运行 | `python script.py args > log` | 自动 subprocess |
| 多参循环 | PowerShell 手写嵌套循环 | 改一行列表 |
| 日志编码 | Tee-Object 产生 UTF-16LE 乱码 | 统一 UTF-8 |
| 并行加速 | 不支持 | `--workers 4`，540 runs 135min→15.6min |
| 断点续传 | 手工跳过已跑组合 | `log_path.exists()` 自动跳过 |
| 结果提取 | 手动 grep + Excel | 一条命令输出 CSV |
| 3 次实验总计 | 预估 6+ 小时手工作业 | **实际 2 小时全自动** |

### 常见坑

| 坑 | 现象 | 原因 | 解决 |
|----|------|------|------|
| UTF-16LE 日志 | regex 匹配不到中文 | PowerShell `Tee-Object` 默认编码 | 用 Python `subprocess` + `encoding="utf-8"` |
| 离线脚本无 `__RESULT__` | 提取脚本报 KeyError | 离线脚本只输出 step log | 用离线专用提取脚本（解析 `球拍击球!` 行） |
| monkey-patch 不生效 | 约束未改变 | import 顺序错误 | patch 必须在 `import main_mod` **之前** |
| 并行跑崩 | MuJoCo segfault | 多进程共享 GL context | 确保 `--no-plot` 关掉所有渲染 |

## 编码规范
- **所有代码注释、docstring 使用中文**
- 使用 `numpy` 进行数组运算，禁止对数组使用原生 Python 循环
- MuJoCo 模型定义在 `src/robot/rm65_model.xml`，是 DOF 数和关节顺序的唯一事实来源
- 可调参数放在 `configs/*.yaml` 中，不要硬编码
- 日志使用 Python `logging` 模块，不要用 `print`
- 测试文件与 `src/` 结构对应，放在 `tests/` 下
- 导入使用绝对路径：`from src.ilqt.solver import ILQTSolver`
- 文件路径使用 `pathlib.Path`，不拼接字符串

## 跨平台注意事项
- **Windows**: MuJoCo 查看器原生支持，使用 `mujoco.viewer.launch_passive()`
- **Ubuntu**: 同样 API；无头服务器上设置 `MUJOCO_GL=osmesa` 或 `egl`
- 不要使用 `glx` 或平台特定的渲染调用
- 文件路径使用 `pathlib.Path`，不要拼接字符串
- **中文路径问题**：Windows 上 MuJoCo C 层 (`mj_loadXML`) 无法打开含非 ASCII 字符的路径。
  所有 `mujoco.MjModel.from_xml_path()` 调用必须替换为 `load_mujoco_model()`（位于 `src/utils/mujoco_loader.py`），
  该函数在 Win32 + 非 ASCII 路径下自动复制模型到临时 ASCII 目录加载，Linux 上直接加载零开销。

## MuJoCo 关键注意事项
- **`range` 属性使用角度（degrees）**：MuJoCo 3.8+ 的 `range` 属性以度为单位，会自动转换为弧度。例如 `range="-180 180"` 对应 ±π，而非 `range="-3.14 3.14"`（后者仅给出 ±3.14°）
- **`ctrlrange` 与力矩裁剪**：前向传递和 `env.step()` 中必须使用 `model.actuator_ctrlrange` 裁剪控制，不要硬编码
- **自碰撞**：手臂和躯干的 geom 需设置 `contype="0" conaffinity="0"` 以避免碰撞约束阻止运动
- **积分器**：使用 `integrator="implicitfast"` 可提高大扭矩下的仿真稳定性
- **模型加载**：始终通过 `src/utils/mujoco_loader.py` 的 `load_mujoco_model()` 加载模型，而非直接调用 `mujoco.MjModel.from_xml_path()`

## 核心算法脚本参考

本项目的核心算法实现在以下脚本中：

| 脚本 | 用途 | 关键特性 |
|------|------|---------|
| `scripts/rm65_mpc_tube_constraint.py` | 离线仿真主脚本 | MPC+iLQR+Tube+硬约束+X平面墙 |
| `scripts/rm65_mpc_tube_constraint_realtime_v5.py` | ★ 当前活跃版本 | 主动击球+随挥+空间走廊Tube+多层安全滤波+异步重规划 |
| `scripts/rm65_mpc_tube_constraint_realtime.py` | 实时仿真 v1 | 异步重规划+buffer机制 |
| `scripts/run_tcp_limit_experiment.py` | TCP 限速实验 | monkey-patch 安全滤波器注入 TCP 检查 |
| `scripts/rm65_mpc_ilqt.py` | 简化 MPC+iLQR | 无 Tube，基础两阶段 iLQR |
| `scripts/train_ilqt.py` | 离线训练入口 | 单次 iLQR 优化 + 保存轨迹 |
| `scripts/rm65_joint_viewer.py` | 关节调节查看器 | position 执行器，拖动滑条控制关节角 |

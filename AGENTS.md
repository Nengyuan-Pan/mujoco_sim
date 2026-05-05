# AGENTS.md - Tennis Robot 项目规范

## 项目概述
本项目使用 iLQT（迭代线性二次跟踪器）传统最优控制方法，解决 UR5e 机械臂挥拍击打网球的场景。
机器人模型为**UR5e（6自由度工业机械臂）**，安装在垂直桩柱右侧，末端法兰上连接垂直网球拍，
在给定网球飞来轨迹的情况下，计算最优挥拍轨迹，使末端执行器（球拍面）在正确的时间和位置以期望的速度击中网球。

## 语言与注释规范
- **所有代码使用 Python 编写**
- **所有代码注释、docstring 必须使用中文**
- **所有深度思考、设计决策说明使用中文**
- 变量名、函数名、类名使用英文（遵循 Python 命名规范）
- 类型提示（type hints）必须标注在所有函数签名上
- 公有函数必须有中文 docstring（Google 风格）

## 技术栈
- **语言**: Python 3.9+
- **仿真**: MuJoCo（mujoco Python 包）— 跨平台 Windows/Ubuntu
- **数值计算**: NumPy, SciPy
- **可视化**: MuJoCo 内置查看器 + matplotlib（轨迹绘图）
- **包管理**: pip + requirements.txt

## 机器人模型定义
UR5e 6 自由度，关节分配如下：

| 关节编号 | 关节名称             | 说明                           | 力矩限制 (Nm) |
|----------|----------------------|--------------------------------|---------------|
| 0        | shoulder_pan         | 肩关节偏航（绕 Z 旋转）        | ±150          |
| 1        | shoulder_lift        | 肩关节俯仰（绕 Y 旋转）        | ±150          |
| 2        | elbow                | 肘关节（绕 Y 旋转）            | ±150          |
| 3        | wrist_1              | 腕关节 1（绕 Y 旋转）          | ±28           |
| 4        | wrist_2              | 腕关节 2（绕 Z 旋转）          | ±28           |
| 5        | wrist_3              | 腕关节 3（绕 Y 旋转）          | ±28           |

- 状态向量: x = [q(6), qdot(6)] ∈ R^12  （关节位置 + 关节速度）
- 控制向量: u = tau(6) ∈ R^6            （关节力矩）
- 末端执行器: 球拍面中心点（racket center）
- UR5e 臂展: 850mm
- 球拍: 连杆沿法兰局部 X 方向延伸（垂直于法兰轴），球拍面在连杆末端

## 项目目录结构
```
tennis_robot/
├── AGENTS.md                # 本文件 — 项目规范与 Agent 指令
├── skills/                  # Skill 定义，供 Agent 工作流调用
│   ├── framework_design.md  # 代码框架设计 skill
│   ├── file_management.md   # 文件管理 skill
│   └── sim_run.md           # 仿真运行 skill
├── src/
│   ├── __init__.py
│   ├── robot/               # 机器人模型定义
│   │   ├── __init__.py
│   │   ├── model.xml        # MuJoCo XML 模型（右臂6DOF + 球拍）
│   │   └── kinematics.py    # 正运动学 / 雅可比矩阵工具
│   ├── dynamics/            # 动力学计算
│   │   ├── __init__.py
│   │   ├── linearize.py     # 动力学线性化（fx, fu），供 iLQT 使用
│   │   └── simulate.py      # 前向仿真 / rollout
│   ├── ilqt/                # iLQT 求解器核心
│   │   ├── __init__.py
│   │   ├── solver.py        # iLQT 后向-前向迭代主循环
│   │   ├── cost.py          # 代价函数（末端击打点代价 + 控制代价）
│   │   └── utils.py         # 增益计算、线搜索、正则化辅助函数
│   ├── tennis/              # 网球场景相关
│   │   ├── __init__.py
│   │   ├── ball.py          # 网球抛物线轨迹预测
│   │   └── hitting.py       # 击打点计算 & 球拍-球接触判断
│   ├── sim/                 # MuJoCo 仿真封装
│   │   ├── __init__.py
│   │   ├── env.py           # MuJoCo 环境封装类
│   │   └── viewer.py        # 可视化工具
│   └── utils/
│       ├── __init__.py
│       └── math_utils.py    # 通用数学工具
├── configs/
│   └── default.yaml         # 默认超参数（时间步长、权重等）
├── scripts/
│   ├── train_ilqt.py        # 主入口：运行 iLQT 优化
│   ├── eval_sim.py          # 在 MuJoCo 仿真中评估优化轨迹
│   └── plot_results.py      # 绘制轨迹、代价等图表
├── tests/
│   ├── test_kinematics.py
│   ├── test_ilqt.py
│   └── test_dynamics.py
├── results/                 # 输出目录（日志、轨迹、视频）
└── requirements.txt
```

## 核心算法说明

### iLQT（迭代线性二次跟踪器）
- **后向传递**: 从终端时刻到初始时刻，计算增益矩阵 K_k, k_k
- **前向传递**: 用线搜索更新轨迹和控制序列
- **终端代价**: 惩罚末端执行器偏离期望击打点（位置 + 速度）
- **运行代价**: 惩罚过大的控制力矩
- 终端代价形式：
  ```
  l_terminal(x) = ||p_ee(x) - p_hit||^2_Q_p + ||v_ee(x) - v_hit||^2_Q_v
  ```
  其中 p_ee 为末端位置，p_hit 为击打点位置，v_ee 为末端速度，v_hit 为期望击打速度
- **两阶段优化**：
  - 阶段1：仅位置代价（Q_p×5, Q_v≈0, R×0.1），使用雅可比转矩初始控制，快速收敛到击打点附近
  - 阶段2：完整代价（Q_p, Q_v, R），从阶段1结果热启动，精细调整末端速度

### 动力学线性化
- 通过 MuJoCo 的 `mj_jac` 和 `mj_rne` 计算雅可比和动力学偏导
- 或使用有限差分法数值线性化（开发初期更快）
- 线性化结果：x_{k+1} ≈ f(x_k, u_k) = A_k δx + B_k δu + ...

### 网球轨迹预测
- 假设网球在重力作用下做抛物线运动（忽略空气阻力）
- 给定球的初始位置和速度，预测球在任意时刻的位置
- 计算击打时刻和击打点：球到达球拍可及范围内的时间点

## 构建与运行命令

### 环境配置
- **Windows / 通用**: 直接 `pip install -r requirements.txt`
- **Ubuntu（仅限本机工作站）**: 使用 `env_isaaclab` conda 环境（Python 3.11，已预装 numpy/scipy/matplotlib）
  - 激活: `conda activate env_isaaclab`
  - 首次安装依赖: `pip install mujoco>=3.0.0 pyyaml>=6.0 pytest>=7.0 ruff>=0.1 mypy>=1.5`
  - 后续可直接: `pip install -r requirements.txt`

### 运行命令
- 运行 iLQT 优化: `python scripts/train_ilqt.py`
- 仿真评估: `python scripts/eval_sim.py`
- 运行测试: `pytest tests/`
- 代码检查: `ruff check src/ tests/ scripts/`
- 类型检查: `mypy src/`

## 编码规范
- 所有代码注释、docstring 使用**中文**
- 使用 `numpy` 进行数组运算，禁止对数组使用原生 Python 循环
- MuJoCo 模型定义在 `src/robot/model.xml`，是 DOF 数和关节顺序的唯一事实来源
- 可调参数放在 `configs/*.yaml` 中，不要硬编码
- 日志使用 Python `logging` 模块，不要用 `print`
- 测试文件与 `src/` 结构对应，放在 `tests/` 下

## 跨平台注意事项
- **Windows**: MuJoCo 查看器原生支持，使用 `mujoco.viewer.launch_passive()`
- **Ubuntu**: 同样 API；无头服务器上设置 `MUJOCO_GL=osmesa` 或 `egl`
- 不要使用 `glx` 或平台特定的渲染调用
- 文件路径使用 `pathlib.Path`，不要拼接字符串

## MuJoCo 关键注意事项
- **`range` 属性使用角度（degrees）**：MuJoCo 3.8+ 的 `range` 属性以度为单位，会自动转换为弧度。例如 `range="-180 180"` 对应 ±π，而非 `range="-3.14 3.14"`（后者仅给出 ±3.14°）
- **`ctrlrange` 与力矩裁剪**：前向传递和 `env.step()` 中必须使用 `model.actuator_ctrlrange` 裁剪控制，不要硬编码 ±100
- **自碰撞**：手臂和躯干的 geom 需设置 `contype="0" conaffinity="0"` 以避免碰撞约束阻止运动
- **积分器**：使用 `integrator="implicitfast"` 可提高大扭矩下的仿真稳定性

## 优化策略
- **两阶段 iLQT**：阶段1 仅位置代价（高权重 Q_p），快速到达击打点；阶段2 完整代价（位置+速度），热启动精细调整
- **雅可比转置初始控制**：使用 `J^T * (p_hit - p_ee)` 生成初始控制序列，比常数力矩初始猜测效果好得多
- **控制范围**：肩关节 ±300~500 Nm，肘关节 ±200 Nm，腕关节 ±50 Nm

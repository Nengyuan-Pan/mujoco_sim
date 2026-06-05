# Skill: 文件管理（file_management）

## 目的
管理项目文件：创建、移动、重命名、删除文件，确保符合 AGENTS.md 定义的项目结构。
调用时机：添加新文件、重组模块、清理项目时。

## 文件命名规范

| 类型         | 命名规则               | 示例                      |
|--------------|------------------------|---------------------------|
| Python 模块  | `snake_case.py`        | `kinematics.py`           |
| MuJoCo 模型  | `snake_case.xml`       | `model.xml`               |
| 配置文件     | `snake_case.yaml`      | `default.yaml`            |
| 测试文件     | `test_<module>.py`     | `test_kinematics.py`      |
| 脚本文件     | `snake_case.py`        | `train_ilqt.py`           |
| Skill 文件   | `snake_case.md`        | `framework_design.md`     |

## 核心规则

### 禁止事项
1. **禁止未经用户同意修改 `src/robot/model.xml`** — 它是 DOF 数和关节顺序的唯一真值来源
2. **禁止未经用户同意删除文件** — 必须先确认
3. **禁止硬编码路径** — 使用 `pathlib.Path` 处理文件路径
4. **禁止硬编码参数** — 可调参数放 `configs/*.yaml`
5. **禁止在源码中使用 `print`** — 使用 `logging` 模块

### 必须事项
1. **新建 Python 包目录必须创建 `__init__.py`**
2. **新建源文件必须在 `tests/` 下创建对应测试**
3. **导入使用绝对路径**：`from src.ilqt.solver import ILQTSolver`
4. **文件路径使用 `pathlib.Path`**，不拼接字符串
5. **中文 docstring 和中文注释**，英文变量名

## 文件创建检查清单

### 创建 Python 源文件
- [ ] 文件在正确的 `src/` 子目录下
- [ ] 所在目录有 `__init__.py`
- [ ] 函数签名有类型提示
- [ ] 公有函数有中文 docstring
- [ ] 核心逻辑有中文注释
- [ ] 对应 `tests/test_*.py` 已创建
- [ ] 无循环导入

### 创建 MuJoCo 模型文件
- [ ] 放在 `src/robot/` 目录
- [ ] 关节数量 = 6（右臂6DOF）
- [ ] 关节顺序与 AGENTS.md 中的定义一致
- [ ] 包含球拍几何体作为末端执行器
- [ ] 关节限位合理（防止 iLQT 生成不可行轨迹）

### 创建配置文件
- [ ] 放在 `configs/` 目录
- [ ] 使用 YAML 格式
- [ ] 包含所有可调参数（时间步长、iLQT 迭代次数、代价权重等）
- [ ] 有中文注释说明各参数含义

### 创建脚本文件
- [ ] 放在正确的 `scripts/` 子目录（见下方分类规则）
- [ ] 使用 `argparse` 解析命令行参数
- [ ] 支持 `--config` 指定配置文件路径
- [ ] 日志输出到 `results/` 目录
- [ ] 路径使用 `Path(__file__).resolve().parent...` 相对定位，层数与目录深度匹配
- [ ] 创建/移动后更新 `scripts/README.md` 和 `AGENTS.md`

## scripts/ 目录分类规则

### 根目录（被引用的核心仿真脚本）

**原则：被其他脚本通过 Python `import` 或 `subprocess` 调用的脚本留在根目录，不可移动。**

| 脚本 | 引用方式 |
|------|---------|
| `rm65_mpc_tube_constraint.py` | exp/ 包装脚本 Python import |
| `rm65_mpc_tube_constraint_realtime.py` | TCP 限速实验 import |
| `rm65_mpc_tube_constraint_realtime_v2.py` | 多个 exp/ 脚本 subprocess |
| `rm65_mpc_tube.py` | scan_ball_params import |
| `rm65_mpc_ilqr_5_5.py` | realtime_batch import |
| `rm65_evaluate.py` | realtime_batch import |
| `rm65_mpc_v6.py` | 10+ 个 run_exp_* 脚本 subprocess |
| `rm65_mpc_v7.py` | 4+ 个 run_exp_* 脚本 subprocess |
| `rm65_mpc_v8.py` | run_20hits_video.py Python import + run_v8_exp.py subprocess |
| `rm65_mpc_v9.py` | 最新迭代，与 v6/v7/v8 同类 |

### 子目录分类

| 目录 | 用途 | 判定条件 |
|------|------|---------|
| `sim/` | 独立仿真脚本（变体、benchmark、训练） | 可独立运行的仿真，不被其他脚本引用 |
| `exp/` | 实验运行器（包装·批量·扫参） | 通过 subprocess 调用根目录仿真脚本 |
| `extract/` | 结果提取（日志 → CSV） | 解析实验日志输出结构化数据 |
| `plot/` | 论文图表生成 | 读取数据文件生成 matplotlib 图表 |
| `tools/` | 独立工具（查看器·扫描·诊断） | 一次性分析/可视化工具 |
| `test/` | 快速验证脚本 | 通过 subprocess 调用仿真做快速测试 |

### 新建脚本的分类决策树

1. **被其他脚本引用？** → 留根目录
2. **是仿真主脚本（MPC/iLQR/控制循环）？** → `sim/`
3. **批量运行多个条件并收集结果？** → `exp/`
4. **解析日志输出 CSV/统计？** → `extract/`
5. **读取数据生成图表？** → `plot/`
6. **查看器/扫描器/一次性诊断？** → `tools/`
7. **快速验证某功能是否正常？** → `test/`

## 脚本移动检查清单

- [ ] 确认目标文件未被其他脚本引用（`grep` 搜索文件名）
- [ ] 移动到正确子目录
- [ ] 修复 `Path(__file__).resolve().parent` 层数（根=2层, 子目录=3层）
- [ ] 修复 docstring/usage 中的路径示例（`scripts/xxx` → `scripts/sub/xxx`）
- [ ] 更新 `scripts/README.md` 详细清单
- [ ] 更新 `AGENTS.md` 目录结构树和核心脚本参考表

## 目录结构映射

源文件与测试文件的对应关系：

```
src/robot/kinematics.py      → tests/test_kinematics.py
src/dynamics/linearize.py    → tests/test_dynamics.py
src/dynamics/simulate.py     → tests/test_dynamics.py
src/ilqt/solver.py           → tests/test_ilqt.py
src/ilqt/cost.py             → tests/test_ilqt.py
src/ilqt/utils.py            → tests/test_ilqt.py
src/tennis/ball.py           → tests/test_tennis.py
src/tennis/hitting.py        → tests/test_tennis.py
src/sim/env.py               → tests/test_sim.py
src/utils/math_utils.py      → tests/test_math_utils.py
```

## 结果目录规范
- `results/` 目录存放运行输出（日志、轨迹数据、图表）
- 输出文件按时间戳组织：`results/YYYYMMDD_HHMMSS/`
- 每次运行保存：`trajectory.npy`, `controls.npy`, `cost_history.npy`, `config_used.yaml`
- `results/` 目录不纳入版本控制

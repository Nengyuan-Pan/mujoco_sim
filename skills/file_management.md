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
- [ ] 放在 `scripts/` 目录
- [ ] 使用 `argparse` 解析命令行参数
- [ ] 支持 `--config` 指定配置文件路径
- [ ] 日志输出到 `results/` 目录

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

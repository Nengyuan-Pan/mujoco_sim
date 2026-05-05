# Skill: 代码框架设计（framework_design）

## 目的
设计并搭建 tennis robot iLQT 项目的代码架构。
调用时机：创建新模块、定义类层次结构、重构代码时。

## 工作流程

### 1. 确认模块归属
根据 AGENTS.md 中的项目目录结构，确认要创建/修改的模块：

| 模块            | 职责                                     |
|-----------------|------------------------------------------|
| `src/robot/`    | MuJoCo 模型定义、正运动学、雅可比计算    |
| `src/dynamics/` | 动力学线性化（A, B 矩阵）、前向仿真      |
| `src/ilqt/`     | iLQT 求解器、代价函数、增益与正则化      |
| `src/tennis/`   | 网球轨迹预测、击打点计算                  |
| `src/sim/`      | MuJoCo 环境封装、可视化                   |
| `src/utils/`    | 通用数学工具                              |

### 2. 检查现有代码
- 查看目标模块中已有的文件和接口
- 遵循已有的代码风格和命名约定
- 如果是新模块，参考相邻模块的风格

### 3. 定义接口
- 先写函数签名（含类型提示）和中文 docstring
- 复杂逻辑先用 `raise NotImplementedError` 占位
- 确保模块可被正确 import

### 4. 实现骨架
- 按接口定义填充基本实现
- 核心算法部分添加中文注释说明思路
- 保持函数粒度适中，单一职责

### 5. 添加测试
- 在 `tests/` 下创建对应的测试文件
- 至少覆盖：构造函数、核心方法的输入输出形状

## 核心设计原则

### 模块依赖关系
```
robot/model.xml ──► sim/env.py ──► dynamics/linearize.py ──► ilqt/solver.py
                                                └──► dynamics/simulate.py
tennis/ball.py ──► tennis/hitting.py ──► ilqt/cost.py ──► ilqt/solver.py
```

**依赖方向必须从上到下，禁止反向依赖或循环依赖。**

### MuJoCo 作为唯一真值来源
- 机器人模型、关节顺序、DOF 数量由 `src/robot/model.xml` 定义
- 动力学计算通过 MuJoCo API 完成，不自行实现前向动力学
- 正运动学和雅可比通过 MuJoCo 的 `mj_jac` 系列函数获取

### 状态与控制约定
- 状态向量：`x = np.concatenate([q, qdot])`  形状 `(12,)`
- 控制向量：`u = tau`  形状 `(6,)`
- 轨迹矩阵：`X` 形状 `(N+1, 12)`，`U` 形状 `(N, 6)`
- 时刻下标：`k = 0, 1, ..., N`，`x_0` 为初始状态

### iLQT 求解器接口设计
```python
class ILQTSolver:
    def __init__(self, dynamics: DynamicsFunc, cost: CostFunc, config: dict) -> None:
        """初始化 iLQT 求解器。"""

    def solve(self, x0: np.ndarray, X_init: np.ndarray, U_init: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """运行 iLQT 优化，返回最优轨迹和控制序列。"""

    def backward_pass(self, X: np.ndarray, U: np.ndarray) -> tuple[list, list]:
        """后向传递：计算增益矩阵 K_k, k_k。"""

    def forward_pass(self, X: np.ndarray, U: np.ndarray, K: list, k: list, alpha: float) -> tuple[np.ndarray, np.ndarray, float]:
        """前向传递：用线搜索更新轨迹。"""
```

### 代价函数接口设计
```python
class CostFunc:
    def running_cost(self, x: np.ndarray, u: np.ndarray) -> float:
        """计算运行代价 l(x, u)。"""

    def terminal_cost(self, x: np.ndarray) -> float:
        """计算终端代价 l_N(x)。"""

    def running_derivatives(self, x: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """返回 (lx, lu, lxx, luu)。"""

    def terminal_derivatives(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回 (lx_N, lxx_N)。"""
```

## 新增模块检查清单
- [ ] 文件位于正确的 `src/` 子目录
- [ ] `__init__.py` 已创建并导出公有接口
- [ ] 类型提示标注在所有函数签名上
- [ ] 中文 docstring 写在所有公有函数上
- [ ] 核心逻辑有中文注释
- [ ] `tests/` 下有对应测试文件
- [ ] 无循环依赖

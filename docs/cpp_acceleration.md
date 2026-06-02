# C++ iLQR 热路径加速模块 — 构建与使用指南

## 一、文件结构

```
src/cpp/
├── __init__.py          # Python 包入口
├── solver_cpp.py        # C++ 加速版 iLQR 求解器（自动回退）
├── core_ext.cpp         # pybind11 绑定 + 所有 C++ 实现入口
├── types.h              # 共享类型定义、工具函数
├── linearize.cpp        # 解析动力学线性化
└── forward_pass.cpp     # 前向传递（单步 + 线搜索）
setup.py                 # 构建脚本
```

## 二、依赖安装

### 2.1 Python 包

```bash
pip install pybind11 numpy mujoco
```

### 2.2 C++ 编译器

| 平台 | 编译器 | 安装方式 |
|------|--------|---------|
| Windows | MSVC 2019+ | 安装 Visual Studio 2019/2022，勾选 "Desktop development with C++" |
| Linux | GCC 9+ 或 Clang 10+ | `sudo apt install build-essential` |
| macOS | Clang (Xcode) | `xcode-select --install` |

验证编译器可用：

```bash
# Windows (Developer PowerShell)
cl /?

# Linux / macOS
g++ --version
```

## 三、编译

```bash
# 在项目根目录下执行：

# 方式 1：开发模式安装（推荐）
pip install -e .

# 方式 2：仅编译（不安装）
python setup.py build_ext --inplace
```

编译成功后将生成 `src/cpp/iLQR_Core.cp310-win_amd64.pyd`（或对应平台的 `.so`）。

## 四、使用方式

### 4.1 自动集成（无需改代码）

`rm65_mpc_fast.py` 已自动支持 C++ 加速：

```bash
python scripts/rm65_mpc_fast.py --seed 42 --normal-flip
```

脚本启动时会打印：
- `iLQR C++ 加速模块已加载` — C++ 加速生效
- `C++ 加速模块未找到，使用纯 Python iLQR` — 自动回退

### 4.2 手动使用

```python
from src.cpp.solver_cpp import ILQTSolver

solver = ILQTSolver(ilqt_config, use_analytical=True)
X, U, costs = solver.solve_few_iters(
    env, cost_fn, x0, U_warm,
    max_iter=8,
    skip_linesearch=True,
)
```

### 4.3 检查 C++ 模块是否可用

```python
from src.cpp import is_available
if is_available():
    print("C++ 加速已启用")
```

## 五、性能对比

| 操作 | Python | C++ | 加速比 |
|------|--------|-----|--------|
| 线性化 (horizon=40) | ~180ms | ~18ms | **10×** |
| 前向传递 (horizon=40) | ~30ms | ~5ms | **6×** |
| 单次 iLQR 迭代 | ~215ms | ~60ms | **3.6×** |
| 首次规划 (15次) | ~350ms | ~100ms | **3.5×** |
| MPC 总计算 | ~700ms | ~200ms | **3.5×** |

## 六、注意事项

1. **仅加速解析线性化**：有限差分模式（`--fd`）不走 C++，保持原速
2. **仅加速右臂 6-DOF**：关节数固定为 6
3. **MuJoCo 版本**：需 `mujoco>=3.0.0`，推荐 `>=3.2.0`
4. **回退安全**：C++ 编译失败只影响性能，不影响功能
5. **线搜索代价评估仍在 Python**：代价函数逻辑复杂，保留 Python 实现

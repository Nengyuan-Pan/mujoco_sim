"""C++ 加速的 iLQR 热路径模块。

本模块提供三个 C++ 加速函数：
  - linearize_analytical_batch:  批量解析动力学线性化
  - forward_pass_single:          单步前向传递（MPC）
  - forward_pass_linesearch:      带线搜索前向传递

用法：
    from src.cpp.iLQR_Core import (
        linearize_analytical_batch,
        forward_pass_single,
        forward_pass_linesearch,
    )
"""

# 检查模块是否可用
try:
    from .iLQR_Core import (  # type: ignore
        linearize_analytical_batch,
        forward_pass_single,
        forward_pass_linesearch,
    )
    _available = True
except ImportError:
    _available = False


def is_available() -> bool:
    """C++ 加速模块是否可用。"""
    return _available

"""兼容重导：从 src.ilqt.costs 导入 HittingCost。

旧代码 `from src.ilqt.cost import HittingCost` 继续有效。
新代码建议使用 `from src.ilqt.costs import HittingCost` 或 `from src.ilqt.costs.hitting import HittingCost`。
"""
from src.ilqt.costs.hitting import HittingCost  # noqa: F401

"""代价函数子包：注册表 + 工厂函数。

新增代价类型只需：
1. 在 costs/ 下新建 xxx.py，实现 BaseCost 子类
2. 在 COST_REGISTRY 加一行
3. 新建 configs/cost_xxx.yaml
"""

from src.ilqt.costs.base import BaseCost
from src.ilqt.costs.base import EndEffectorCost as EndEffectorCost
from src.ilqt.costs.hitting import HittingCost

COST_REGISTRY: dict[str, type[BaseCost]] = {
    "hitting": HittingCost,
}


def create_cost(
    cost_type: str,
    env,
    config: dict,
    **runtime_kwargs,
) -> BaseCost:
    """工厂函数：从注册表查找代价类并用配置实例化。

    Args:
        cost_type: 代价类型名称（对应 COST_REGISTRY 的 key）。
        env: MuJoCo 环境实例。
        config: 代价配置字典（来自 yaml 文件）。
        **runtime_kwargs: 运行时参数透传给 from_config。

    Returns:
        代价函数实例。

    Raises:
        KeyError: 若 cost_type 不在注册表中。
    """
    if cost_type not in COST_REGISTRY:
        raise KeyError(
            f"未知代价类型 '{cost_type}'，可用: {list(COST_REGISTRY.keys())}"
        )
    cls = COST_REGISTRY[cost_type]
    return cls.from_config(env, config, **runtime_kwargs)

"""前向仿真 / rollout 工具。"""

import numpy as np
from src.sim.env import MujocoEnv


def rollout(
    env: MujocoEnv,
    x0: np.ndarray,
    U: np.ndarray,
) -> np.ndarray:
    """从初始状态 x0 施加控制序列 U 进行前向仿真。

    Args:
        env: MuJoCo 环境实例。
        x0: 初始臂状态，形状 (12,)。
        U: 控制序列，形状 (N, 6)。

    Returns:
        状态轨迹 X，形状 (N+1, 12)。
    """
    N = len(U)
    X = np.zeros((N + 1, env.NX))
    env.set_arm_state(x0)
    X[0] = x0.copy()

    for k in range(N):
        env.set_arm_state(X[k])
        X[k + 1] = env.step(U[k])

    return X

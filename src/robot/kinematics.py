"""正运动学与雅可比矩阵工具。"""

import mujoco
import numpy as np
from src.sim.env import MujocoEnv


def forward_kinematics(env: MujocoEnv, q: np.ndarray) -> np.ndarray:
    """计算给定关节角度下的末端执行器位置。

    Args:
        env: MuJoCo 环境实例。
        q: 关节角度，形状 (6,)。

    Returns:
        末端执行器位置，形状 (3,)。
    """
    x = np.zeros(env.NX)
    x[: env.NQ] = q
    env.set_arm_state(x)
    return env.get_ee_pos()


def compute_jacobian(env: MujocoEnv, q: np.ndarray) -> np.ndarray:
    """计算给定关节角度下的位置雅可比矩阵。

    Args:
        env: MuJoCo 环境实例。
        q: 关节角度，形状 (6,)。

    Returns:
        位置雅可比矩阵，形状 (3, 6)。
    """
    x = np.zeros(env.NX)
    x[: env.NQ] = q
    env.set_arm_state(x)
    return env.get_ee_jacp()


def compute_workspace_reach(env: MujocoEnv, n_samples: int = 500) -> float:
    """通过采样估计机械臂的最大可达距离。

    Args:
        env: MuJoCo 环境实例。
        n_samples: 采样数。

    Returns:
        最大可达距离（米）。
    """
    max_dist = 0.0
    rng = np.random.default_rng(42)
    q0 = np.zeros(env.NQ)

    for _ in range(n_samples):
        q = q0.copy()
        # 随机关节角度（在限位内）
        for j in range(env.NQ):
            jnt_id = j
            range_lo = env.model.jnt_range[jnt_id, 0]
            range_hi = env.model.jnt_range[jnt_id, 1]
            q[j] = rng.uniform(range_lo, range_hi)

        p = forward_kinematics(env, q)
        # 肩关节位置
        shoulder_pos = env.data.body(
            mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "shoulder_link")
        ).xpos
        dist = np.linalg.norm(p - shoulder_pos)
        max_dist = max(max_dist, dist)

    return max_dist

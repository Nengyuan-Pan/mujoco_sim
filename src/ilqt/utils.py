"""iLQT 辅助函数：正则化、线搜索。"""

import numpy as np
from src.sim.env import MujocoEnv
from src.ilqt.cost import HittingCost


def compute_total_cost(
    env: MujocoEnv,
    cost_fn: HittingCost,
    X: np.ndarray,
    U: np.ndarray,
) -> float:
    """计算轨迹的总代价。

    Args:
        env: MuJoCo 环境（未使用，保留接口一致性）。
        cost_fn: 代价函数实例。
        X: 状态轨迹，形状 (N+1, 12)。
        U: 控制轨迹，形状 (N, 6)。

    Returns:
        总代价值。
    """
    total = 0.0
    for k in range(len(U)):
        total += cost_fn.running_cost(X[k], U[k], k)
    total += cost_fn.terminal_cost(X[-1])
    return total


def forward_pass_with_linesearch(
    env: MujocoEnv,
    cost_fn: HittingCost,
    X: np.ndarray,
    U: np.ndarray,
    Ks: list[np.ndarray],
    ks: list[np.ndarray],
    alpha_list: list[float],
    cost_old: float,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """带线搜索的前向传递。

    Args:
        env: MuJoCo 环境实例。
        cost_fn: 代价函数实例。
        X: 名义状态轨迹，形状 (N+1, 12)。
        U: 名义控制轨迹，形状 (N, 6)。
        Ks: 反馈增益列表，每个形状 (6, 12)。
        ks: 前馈增益列表，每个形状 (6,)。
        alpha_list: 线搜索步长列表。
        cost_old: 旧轨迹的总代价。

    Returns:
        (X_new, U_new, cost_new, accepted): 新轨迹和新代价，以及是否被接受。
    """
    N = len(U)
    n_u = env.NU

    ctrl_lo = env.model.actuator_ctrlrange[:n_u, 0]
    ctrl_hi = env.model.actuator_ctrlrange[:n_u, 1]

    has_collision_ctrl = hasattr(env, "set_arm_collision")
    if has_collision_ctrl:
        env.set_arm_collision(False)

    for alpha in alpha_list:
        X_new = np.zeros_like(X)
        U_new = np.zeros_like(U)
        X_new[0] = X[0].copy()

        valid = True
        for k in range(N):
            dx = X_new[k] - X[k]
            U_new[k] = U[k] + alpha * ks[k] + Ks[k] @ dx
            U_new[k] = np.clip(U_new[k], ctrl_lo, ctrl_hi)
            X_new[k + 1] = env.step_from_state(X_new[k], U_new[k])

            if not np.all(np.isfinite(X_new[k + 1])):
                valid = False
                break

        if not valid:
            continue

        cost_new = compute_total_cost(env, cost_fn, X_new, U_new)
        if cost_new < cost_old:
            if has_collision_ctrl:
                env.set_arm_collision(True)
            return X_new, U_new, cost_new, True

    if has_collision_ctrl:
        env.set_arm_collision(True)
    return X.copy(), U.copy(), cost_old, False


def forward_pass_single(
    env: MujocoEnv,
    cost_fn: HittingCost,
    X: np.ndarray,
    U: np.ndarray,
    Ks: list[np.ndarray],
    ks: list[np.ndarray],
    alpha: float = 0.5,
    skip_cost: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """固定步长前向传递（MPC 模式，不搜索）。

    Args:
        env: MuJoCo 环境实例。
        cost_fn: 代价函数实例。
        X: 名义状态轨迹，形状 (N+1, 12)。
        U: 名义控制轨迹，形状 (N, 6)。
        Ks: 反馈增益列表，每个形状 (6, 12)。
        ks: 前馈增益列表，每个形状 (6,)。
        alpha: 固定步长（默认 0.5，阻尼防振荡）。
        skip_cost: 是否跳过代价计算（MPC 模式默认 True）。

    Returns:
        (X_new, U_new, cost_new): 新轨迹、新控制、新代价。
    """
    N = len(U)
    n_u = env.NU

    ctrl_lo = env.model.actuator_ctrlrange[:n_u, 0]
    ctrl_hi = env.model.actuator_ctrlrange[:n_u, 1]

    X_new = np.zeros_like(X)
    U_new = np.zeros_like(U)
    X_new[0] = X[0].copy()

    has_collision_ctrl = hasattr(env, "set_arm_collision")
    if has_collision_ctrl:
        env.set_arm_collision(False)

    for k in range(N):
        dx = X_new[k] - X[k]
        U_new[k] = U[k] + alpha * ks[k] + Ks[k] @ dx
        U_new[k] = np.clip(U_new[k], ctrl_lo, ctrl_hi)
        X_new[k + 1] = env.step_from_state(X_new[k], U_new[k])

    if has_collision_ctrl:
        env.set_arm_collision(True)

        if not np.all(np.isfinite(X_new[k + 1])):
            return X.copy(), U.copy(), float("inf")

    cost_new = 0.0 if skip_cost else compute_total_cost(env, cost_fn, X_new, U_new)
    return X_new, U_new, cost_new

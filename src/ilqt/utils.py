"""iLQT 辅助函数：正则化、线搜索 + 真实机器人硬约束检查。"""

import logging
from collections import deque

import numpy as np
import mujoco
from src.sim.env import MujocoEnv
from src.ilqt.cost import HittingCost
from src.ilqt.robot_limits import (
    RobotLimits,
    check_step_feasibility,
    build_qdot_history,
    compute_trajectory_metrics,
    TrajectoryMetrics,
)

logger = logging.getLogger(__name__)


def check_constraints_x(
    env: MujocoEnv,
    x: np.ndarray,
    joint_limits: dict | None = None,
    x_limit: float | None = None,
    x_body_ids: list | None = None,
) -> bool:
    """检查状态 x 是否违反硬约束。

    三级联查（便宜→贵）:
      1. 关节范围
      2. 控制范围（由 env.step 自动处理，此处跳过）
      3. body X 坐标

    Args:
        env: 环境实例。
        x: 臂状态 (12,)。
        joint_limits: {idx: (lo, hi)} 字典，None 表示无限制。
        x_limit: X 墙上限，None 表示不检查。
        x_body_ids: 需检查的 body ID 列表。

    Returns:
        True 表示通过所有约束，False 表示违反。
    """
    n_q = env.NQ
    q = x[:n_q]

    # Level 1: 关节范围
    if joint_limits:
        for j, (lo, hi) in joint_limits.items():
            if lo is not None and q[j] < lo:
                return False
            if hi is not None and q[j] > hi:
                return False

    # Level 2: 控制范围 — 由 env.step 的 ctrlrange clip 自动处理

    # Level 3: body X 坐标（仅当接近边界时检查）
    if x_limit is not None and x_body_ids:
        env.set_arm_state(x)
        for bid in x_body_ids:
            if env.data.xpos[bid, 0] > x_limit:
                return False

    return True


def check_constraints_trajectory_x(
    env: MujocoEnv,
    X: np.ndarray,
    joint_limits: dict | None = None,
    x_limit: float | None = None,
    x_body_ids: list | None = None,
    check_every: int = 2,
) -> bool:
    """检查整条轨迹 X 的每一帧是否违反约束。

    对每 check_every 帧执行 body FK 检查（其他帧仅检查关节范围）。
    """
    for k in range(len(X)):
        if k % check_every == 0:
            if not check_constraints_x(env, X[k], joint_limits, x_limit, x_body_ids):
                return False
        else:
            # 仅快速检查关节范围
            if joint_limits:
                q = X[k, :env.NQ]
                for j, (lo, hi) in joint_limits.items():
                    if lo is not None and q[j] < lo:
                        return False
                    if hi is not None and q[j] > hi:
                        return False
    return True


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
    limits: RobotLimits | None = None,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """带线搜索的前向传递（含真实机器人硬约束检查）。

    Args:
        env: MuJoCo 环境实例。
        cost_fn: 代价函数实例。
        X: 名义状态轨迹，形状 (N+1, 12)。
        U: 名义控制轨迹，形状 (N, 6)。
        Ks: 反馈增益列表，每个形状 (6, 12)。
        ks: 前馈增益列表，每个形状 (6,)。
        alpha_list: 线搜索步长列表。
        cost_old: 旧轨迹的总代价。
        limits: 真实机器人硬约束参数。None 表示不启用。

    Returns:
        (X_new, U_new, cost_new, accepted): 新轨迹和新代价，以及是否被接受。
    """
    N = len(U)
    n_u = env.NU
    dt = env.dt

    ctrl_lo = env.model.actuator_ctrlrange[:n_u, 0]
    ctrl_hi = env.model.actuator_ctrlrange[:n_u, 1]

    has_collision_ctrl = hasattr(env, "set_arm_collision")
    if has_collision_ctrl:
        env.set_arm_collision(False)

    best_result: tuple[np.ndarray | None, np.ndarray | None, float, str] = (
        None, None, float("inf"), ""
    )
    fp_margin = limits.forward_pass_margin if limits is not None else 1.0
    fp_q_tol = limits.forward_pass_q_tol_rad if limits is not None else 0.0
    actuator_mode = getattr(env, 'actuator_mode', 0)

    for alpha in alpha_list:
        X_new = np.zeros_like(X)
        U_new = np.zeros_like(U)
        X_new[0] = X[0].copy()

        qdot_hist: deque = deque(maxlen=limits.qddot_window_size + 1 if limits is not None else 6)
        if limits is not None:
            build_qdot_history(qdot_hist, X_new[0][6:], limits.qddot_window_size)

        valid = True
        reject_reason = ""

        for k in range(N):
            dx = X_new[k] - X[k]
            U_new[k] = U[k] + alpha * ks[k] + Ks[k] @ dx
            U_new[k] = np.clip(U_new[k], ctrl_lo, ctrl_hi)
            X_new[k + 1] = env.step_from_state(X_new[k], U_new[k])

            if not np.all(np.isfinite(X_new[k + 1])):
                valid = False
                reject_reason = f"NaN in state at k={k}"
                break

            # 真实机器人硬约束: q/qdot/u 硬检查 + qddot 滑动窗口（Phase1仅日志）
            if limits is not None:
                build_qdot_history(qdot_hist, X_new[k + 1][6:], limits.qddot_window_size)
                ok, reason = check_step_feasibility(
                    X_new[k], X_new[k + 1], U_new[k], limits, dt,
                    step=k, margin=fp_margin,
                    skip_qdot='braking', skip_qddot=False,
                    qdot_history=qdot_hist,
                    fp_q_tol=fp_q_tol,
                    actuator_mode=actuator_mode,
                )
                if not ok:
                    valid = False
                    reject_reason = reason
                    logger.warning(
                        "[REJECT_LOG] alpha=%.2f rejected at k=%d: %s",
                        alpha, k, reason,
                    )
                    break

        if not valid:
            continue

        cost_new = compute_total_cost(env, cost_fn, X_new, U_new)
        if cost_new < cost_old:
            if has_collision_ctrl:
                env.set_arm_collision(True)
            # 计算并记录轨迹约束指标
            if limits is not None:
                _log_trajectory_metrics(env, X_new, U_new, limits, dt, alpha)
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
    limits: RobotLimits | None = None,
    skip_cost: bool = True,
) -> tuple[np.ndarray | None, np.ndarray | None, float, str]:
    """固定步长前向传递（MPC 模式，含真实机器人硬约束 + alpha 回退）。

    先尝试传入的 alpha。若违反硬约束（且 limits 不为 None），
    自动按 limits.alpha_fallback 列表回退尝试更小的 alpha。
    全部失败则返回 (None, None, inf, reason)。

    Args:
        env: MuJoCo 环境实例。
        cost_fn: 代价函数实例。
        X: 名义状态轨迹，形状 (N+1, 12)。
        U: 名义控制轨迹，形状 (N, 6)。
        Ks: 反馈增益列表，每个形状 (6, 12)。
        ks: 前馈增益列表，每个形状 (6,)。
        alpha: 首选固定步长（默认 0.5）。
        limits: 真实机器人硬约束参数。None 表示不启用。
        skip_cost: 是否跳过代价计算（MPC 模式默认 True）。

    Returns:
        (X_new, U_new, cost_new, reject_reason):
          - 成功: (X_array, U_array, cost_float, "")
          - 失败: (None, None, inf, "qdot limit exceeded, joint=2")
    """
    N = len(U)
    n_u = env.NU
    dt = env.dt

    ctrl_lo = env.model.actuator_ctrlrange[:n_u, 0]
    ctrl_hi = env.model.actuator_ctrlrange[:n_u, 1]

    has_collision_ctrl = hasattr(env, "set_arm_collision")
    if has_collision_ctrl:
        env.set_arm_collision(False)

    alphas_to_try = [alpha]
    if limits is not None:
        # 合并用户给出的 alpha 和 fallback 列表（去重，保持顺序）
        for a in limits.alpha_fallback:
            if a not in alphas_to_try:
                alphas_to_try.append(a)

    fp_margin = limits.forward_pass_margin if limits is not None else 1.0
    fp_q_tol = limits.forward_pass_q_tol_rad if limits is not None else 0.0
    actuator_mode = getattr(env, 'actuator_mode', 0)

    for alpha_try in alphas_to_try:
        X_new = np.zeros_like(X)
        U_new = np.zeros_like(U)
        X_new[0] = X[0].copy()

        qdot_hist: deque = deque(maxlen=limits.qddot_window_size + 1 if limits is not None else 6)
        if limits is not None:
            build_qdot_history(qdot_hist, X_new[0][6:], limits.qddot_window_size)

        valid = True
        reject_reason = ""

        for k in range(N):
            dx = X_new[k] - X[k]
            U_new[k] = U[k] + alpha_try * ks[k] + Ks[k] @ dx
            U_new[k] = np.clip(U_new[k], ctrl_lo, ctrl_hi)
            X_new[k + 1] = env.step_from_state(X_new[k], U_new[k])

            if not np.all(np.isfinite(X_new[k + 1])):
                valid = False
                reject_reason = f"NaN in state at k={k}"
                break

            # 真实机器人硬约束: q/qdot/u 硬检查 + qddot 滑动窗口（Phase1仅日志）
            if limits is not None:
                build_qdot_history(qdot_hist, X_new[k + 1][6:], limits.qddot_window_size)
                ok, reason = check_step_feasibility(
                    X_new[k], X_new[k + 1], U_new[k], limits, dt,
                    step=k, margin=fp_margin,
                    skip_qdot='braking', skip_qddot=False,
                    qdot_history=qdot_hist,
                    fp_q_tol=fp_q_tol,
                    actuator_mode=actuator_mode,
                )
                if not ok:
                    valid = False
                    reject_reason = reason
                    if alpha_try != alpha or len(alphas_to_try) > 1:
                        logger.warning(
                            "[REJECT_LOG] alpha=%.2f rejected at k=%d: %s",
                            alpha_try, k, reason,
                        )
                    break

        if not valid:
            continue

        # 通过所有约束
        if has_collision_ctrl:
            env.set_arm_collision(True)

        cost_new = 0.0 if skip_cost else compute_total_cost(env, cost_fn, X_new, U_new)
        # 计算并记录轨迹约束指标
        if limits is not None:
            _log_trajectory_metrics(env, X_new, U_new, limits, dt, alpha_try)
        return X_new, U_new, cost_new, ""

    # 全部 alpha 失败
    if has_collision_ctrl:
        env.set_arm_collision(True)

    return None, None, float("inf"), reject_reason


def _log_trajectory_metrics(
    env, X: np.ndarray, U: np.ndarray,
    limits: RobotLimits, dt: float, alpha: float,
) -> None:
    """记录前向传递轨迹的约束指标（DEBUG 级别）。"""
    m = compute_trajectory_metrics(X, U, limits, dt, env=env)
    logger.debug(
        "[PLAN_METRICS] alpha=%.2f | qdot=%.2fx(j%d) qddot=%.2fx(j%d) "
        "qddot_100ms_mean=%.2fx peak_before_hit=%.2fx "
        "joint_speed=%.1frad/s tcp=%.1fm/s face=%.1fm/s",
        alpha, m.max_qdot_ratio, m.max_qdot_joint,
        m.max_qddot_ratio, m.max_qddot_joint,
        m.mean_qddot_ratio_100ms, m.qddot_peak_before_hit,
        m.max_joint_speed_rad_s, m.tcp_speed_max, m.racket_face_speed_max,
    )


def apply_control_beta(
    u: np.ndarray,
    q: np.ndarray,
    beta: float,
    is_position: bool,
) -> np.ndarray:
    """对控制量施加 beta 缩放（力矩/位置模式兼容）。

    力矩模式: u_out = beta * u（线性缩放控制量）
    位置模式: u_out = q + beta * (u - q)（向目标角度插值）

    beta=0: 力矩模式 → 零力矩; 位置模式 → 保持当前角度
    beta=1: 力矩模式 → 完整力矩; 位置模式 → 设置目标角度

    Args:
        u: 控制量 (NU,) — 力矩模式下为 tau，位置模式下为 q_desired。
        q: 当前关节角度 (NU,) — 仅位置模式使用。
        beta: 缩放系数 ∈ [0, 1]。
        is_position: 是否为位置模式。

    Returns:
        缩放后的控制量 (NU,)。
    """
    if is_position:
        return q + beta * (u - q)
    return beta * u

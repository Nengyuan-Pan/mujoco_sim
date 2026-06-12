"""位置模式下的雅可比转置初始控制序列生成。

力矩模式的 JT 初始控制（compute_jacobian_init_control）输出力矩 τ = J^T * err * scale。
位置模式需要输出目标角度 q_desired，本模块提供角度增量版本的 JT 初始控制。
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.sim.rm65_env import RM65Env


def _get_jnt_range(env: RM65Env) -> tuple[np.ndarray, np.ndarray]:
    """获取右臂关节限位 (jnt_lo, jnt_hi)，各形状 (6,)。"""
    jnt_lo = np.zeros(env.NU)
    jnt_hi = np.zeros(env.NU)
    for i in range(env.NU):
        jnt_id = env.model.actuator_trnid[i, 0]
        jnt_lo[i] = env.model.jnt_range[jnt_id, 0]
        jnt_hi[i] = env.model.jnt_range[jnt_id, 1]
    return jnt_lo, jnt_hi


def _compute_joint1_backswing_trajectory(
    q1_current: float,
    qdot1_current: float,
    q1_hit: float,
    qdot1_hit: float,
    horizon: int,
    backswing_offset: float = -0.6,
    backswing_ratio: float = 0.35,
) -> np.ndarray:
    """生成关节1的"后摆→前挥"五次多项式轨迹。

    与 V11 中 compute_joint1_backswing_trajectory 相同的算法，
    此处复制以避免循环依赖。

    Args:
        q1_current: 当前关节1角度。
        qdot1_current: 当前关节1角速度。
        q1_hit: 击球时刻关节1角度。
        qdot1_hit: 击球时刻关节1角速度。
        horizon: 轨迹步数。
        backswing_offset: 后摆偏移量（弧度）。
        backswing_ratio: 后摆占比（0~1）。

    Returns:
        关节1轨迹，形状 (horizon,)。
    """
    if horizon <= 0:
        return np.zeros(0)

    T = float(horizon)
    alpha = float(np.clip(backswing_ratio, 0.05, 0.95))
    q_mid = q1_current + backswing_offset

    a0 = q1_current
    a1 = qdot1_current * T

    alpha2, alpha3, alpha4, alpha5 = alpha**2, alpha**3, alpha**4, alpha**5

    A = np.array([
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 3.0, 4.0, 5.0],
        [alpha2, alpha3, alpha4, alpha5],
        [2 * alpha, 3 * alpha2, 4 * alpha3, 5 * alpha4],
    ])
    b = np.array([
        q1_hit - a0 - a1,
        qdot1_hit * T - a1,
        q_mid - a0 - a1 * alpha,
        -a1,
    ])
    coeffs_high = np.linalg.solve(A, b)
    a2, a3, a4, a5 = coeffs_high[0], coeffs_high[1], coeffs_high[2], coeffs_high[3]

    q1_traj = np.zeros(horizon)
    for k in range(horizon):
        tau = (k + 1) / T
        q1_traj[k] = a0 + a1 * tau + a2 * tau**2 + a3 * tau**3 + a4 * tau**4 + a5 * tau**5
    return q1_traj


def compute_jacobian_init_control_position(
    env: RM65Env,
    x0: np.ndarray,
    p_hit: np.ndarray,
    horizon: int,
    gain: float = 0.5,
    fix_joint5_angle: float | None = None,
    damp: float = 1e-6,
) -> np.ndarray:
    """位置模式初始控制：JT → 角度增量序列。

    输出 q_desired[k]，使末端逐步趋近 p_hit。每步：
      1. set_arm_state([q, zeros]) → 计算 p_ee 和 J_p
      2. dq = J^T * solve(J*J^T + λI, err) * gain
      3. q_desired = q + dq，clip 到 jnt_range
      4. step_from_state 物理仿真更新状态
      5. q = 仿真后真实角度

    Args:
        env: RM65Env 实例（需已配置位置模式）。
        x0: 初始右臂状态 [q, qdot]，形状 (12,)。
        p_hit: 目标击打点位置，形状 (3,)。
        horizon: 控制序列长度。
        gain: 角度增量步长（弧度），默认 0.5。
        fix_joint5_angle: 若非 None，第 5 关节固定为该角度。
        damp: 阻尼因子，防止雅可比奇异。

    Returns:
        控制序列 U，形状 (horizon, 6)，每行为 q_desired。
    """
    NQ = env.NQ
    NU = env.NU
    U = np.zeros((horizon, NU))
    x = x0.copy()

    jnt_lo, jnt_hi = _get_jnt_range(env)

    has_collision_ctrl = hasattr(env, "set_arm_collision")
    if has_collision_ctrl:
        env.set_arm_collision(False)

    for k in range(horizon):
        env.set_arm_state(x)
        p_ee = env.get_ee_pos()
        J_p = env.get_ee_jacp()

        err = p_hit - p_ee
        JJT = J_p @ J_p.T + damp * np.eye(3)
        dq = J_p.T @ np.linalg.solve(JJT, err) * gain

        q_current = x[:NQ]
        q_desired = q_current + dq
        q_desired = np.clip(q_desired, jnt_lo, jnt_hi)

        if fix_joint5_angle is not None:
            q_desired[5] = fix_joint5_angle

        U[k] = q_desired
        x = env.step_from_state(x, q_desired)

    if has_collision_ctrl:
        env.set_arm_collision(True)
    return U


def generate_backswing_warm_start_position(
    env: RM65Env,
    x0: np.ndarray,
    p_hit: np.ndarray,
    v_hit_desired: np.ndarray,
    horizon: int,
    backswing_offset: float = 0.0,
    backswing_ratio: float = 0.0,
    fix_joint5_angle: float | None = None,
    n_des: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """位置模式的后摆 warm-start：直接输出角度轨迹作为控制序列。

    与力矩版 generate_backswing_warm_start 对应，但输出 q_desired 而非 PD 力矩。
    位置模式下无需 PD 跟踪仿真，直接输出 q_des_traj 即可，
    因为 MuJoCo 内部 PD 执行器会自动跟踪。

    Args:
        env: RM65Env 实例（需已配置位置模式）。
        x0: 初始右臂状态 [q, qdot]，形状 (12,)。
        p_hit: 目标击打点位置，形状 (3,)。
        v_hit_desired: 期望击球速度，形状 (3,)。
        horizon: 控制序列长度。
        backswing_offset: 关节 1 后摆偏移量（弧度）。
        backswing_ratio: 后摆占比（0~1）。
        fix_joint5_angle: 若非 None，第 5 关节固定为该角度。
        n_des: 期望法向量，形状 (3,)。

    Returns:
        (U, q_des_traj): U 形状 (horizon, 6)，q_des_traj 形状 (horizon, 6)。
    """
    NQ = env.NQ
    NU = env.NU

    if horizon <= 0:
        return np.zeros((0, NU)), np.zeros((0, NQ))

    q_hit = env.solve_ik(p_hit, q_init=x0[:NQ], max_iter=200, eps=1e-3)
    if fix_joint5_angle is not None:
        q_hit[5] = fix_joint5_angle

    if n_des is not None:
        wrist_joints = [3, 4, 5]
        for _ in range(20):
            env.set_arm_state(np.concatenate([q_hit, np.zeros(NQ)]))
            n_cur = env.get_ee_normal()
            n_err = n_cur - n_des
            err_norm = np.linalg.norm(n_err)
            if err_norm < 0.01:
                break
            J_omega = env.get_ee_jacr()
            nx, ny, nz = -n_cur[0], -n_cur[1], -n_cur[2]
            skew = np.array([[0, -nz, ny], [nz, 0, -nx], [-ny, nx, 0]])
            J_n = skew @ J_omega
            J_n_wrist = J_n[:, wrist_joints]
            dq_wrist = -np.linalg.lstsq(J_n_wrist, n_err, rcond=None)[0]
            dq_wrist *= min(1.0, 0.02 / (np.linalg.norm(dq_wrist) + 1e-12))
            q_hit[wrist_joints] += dq_wrist

    env.set_arm_state(np.concatenate([q_hit, np.zeros(NQ)]))
    J_p_hit = env.get_ee_jacp()
    qdot_hit = np.linalg.lstsq(J_p_hit, v_hit_desired, rcond=None)[0]
    max_qdot = 3.0
    qdot_norm = np.linalg.norm(qdot_hit)
    if qdot_norm > max_qdot:
        qdot_hit *= max_qdot / qdot_norm

    q1_traj = _compute_joint1_backswing_trajectory(
        x0[0], x0[NQ], q_hit[0], qdot_hit[0],
        horizon,
        backswing_offset=backswing_offset,
        backswing_ratio=backswing_ratio,
    )

    q_des_traj = np.zeros((horizon, NQ))
    q_des_traj[:, 0] = q1_traj
    for j in range(1, NQ):
        q_des_traj[:, j] = np.linspace(x0[j], q_hit[j], horizon)

    if fix_joint5_angle is not None:
        q_des_traj[:, 5] = fix_joint5_angle

    jnt_lo, jnt_hi = _get_jnt_range(env)
    q_des_traj = np.clip(q_des_traj, jnt_lo, jnt_hi)

    return q_des_traj.copy(), q_des_traj


def fix_joint5_control_trajectory_position(
    U: np.ndarray,
    q_fixed: float,
) -> np.ndarray:
    """位置模式下将控制序列的第 5 关节替换为固定角度。

    力矩版用 PD 力矩保持，位置版直接设 q_desired。

    Args:
        U: 控制序列，形状 (horizon, 6)。
        q_fixed: 固定的关节 5 角度。

    Returns:
        修改后的 U（副本）。
    """
    U = U.copy()
    U[:, 5] = q_fixed
    return U

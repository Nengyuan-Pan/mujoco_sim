"""RM-65 Tube-based Robust Hitting + 右臂硬半空间约束 实验脚本（V2 改进版）。

基于 rm65_mpc_tube_constraint_realtime.py 的 Softmin 终端改进：
  Softmin 多终端代价 — 允许在多个候选时刻之一击球，时间/空间鲁棒性大幅提升
  （经消融实验验证：per-step sigma 走廊贡献为零，已移除）

在 rm65_mpc_tube.py 基础上新增执行层硬约束：
  每步执行后检查 r_link3 (肘)、r_link5 (腕)、r_racket_body (球拍) 的 X 坐标。
  若任何 body 越过身体中线 X=0，立即用 PD 控制推回安全位形一步。
  不依赖代价函数软惩罚 —— 直接在物理层面阻止越界。

用法:
  python scripts/rm65_mpc_tube_constraint_realtime_v4.py --serve-box --ball-speed 12 --viewer
  python scripts/rm65_mpc_tube_constraint_realtime_v4.py --serve-box --ball-speed 15 --viewer
  python scripts/rm65_mpc_tube_constraint_realtime_v4.py --use_tube false --viewer --seed 42
"""

from __future__ import annotations

import sys
import time
import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.sim.rm65_env import RM65Env
from src.tennis.ball import (
    generate_ball_to_target_box,
    generate_ball_from_serve_box,
)
from src.tennis.hitting import (
    find_hitting_point_physics,
    compute_desired_hit_velocity,
)
from src.ilqt.cost import HittingCost
from src.ilqt.robot_limits import (
    RobotLimits,
    check_step_feasibility,
    check_one_step_feasibility,
    compute_trajectory_metrics,
    compute_tcp_speed_from_env,
    ExecutionMetrics,
    strict_braking_check,
)
from src.ilqt.async_replanner import AsyncReplanner, PlanRequest, PlanResult
try:
    from src.ilqt.retiming import retime_trajectory as _retime_impl
except ImportError:
    def _retime_impl(X, U, dt, limits, k_hit):  # type: ignore[misc]
        return X, U, dt, False
try:
    from src.cpp.solver_cpp import ILQTSolver
except ImportError:
    from src.ilqt.solver import ILQTSolver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
# 抑制 robot_limits 的 DEBUG 级别日志（避免 qddot 每步检查泛滥）
logging.getLogger("src.ilqt.robot_limits").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ==============================================================================
# 辅助函数（从 rm65_mpc_fast.py 复用）
# ==============================================================================

def fix_joint5_control(
    u: np.ndarray,
    q_fixed: float,
    x_current: np.ndarray,
    nq: int,
    kp: float = 300.0,
    kd: float = 30.0,
) -> np.ndarray:
    """将第 6 关节（索引 5）的控制力矩替换为 PD 保持力矩。"""
    u = u.copy()
    q5_err = q_fixed - x_current[:nq][5]
    q5dot_err = -x_current[nq:][5]
    tau5 = kp * q5_err + kd * q5dot_err
    if u.ndim == 1:
        u[5] = tau5
    else:
        u[:, 5] = tau5
    return u


def fix_joint5_control_trajectory(
    U: np.ndarray,
    x0: np.ndarray,
    env: RM65Env,
    q_fixed: float,
    kp: float = 300.0,
    kd: float = 30.0,
) -> np.ndarray:
    """将整个控制序列的第 6 关节替换为 PD 保持力矩。"""
    U = U.copy()
    x = x0.copy()
    has_collision_ctrl = hasattr(env, "set_arm_collision")
    if has_collision_ctrl:
        env.set_arm_collision(False)
    bq = env.BALL_QPOS_START
    bv = env.BALL_QVEL_START
    ball_qpos_save = env.data.qpos[bq:bq + 7].copy()
    ball_qvel_save = env.data.qvel[bv:bv + 6].copy()

    for k in range(len(U)):
        q5_err = q_fixed - x[:env.NQ][5]
        q5dot_err = -x[env.NQ:][5]
        U[k, 5] = kp * q5_err + kd * q5dot_err
        x = env.step_from_state(x, U[k])

    env.data.qpos[bq:bq + 7] = ball_qpos_save
    env.data.qvel[bv:bv + 6] = ball_qvel_save

    if has_collision_ctrl:
        env.set_arm_collision(True)
    return U


def load_config(config_path: Path) -> dict:
    """加载 YAML 配置文件。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_configs(base: dict, override: dict) -> dict:
    """递归合并两个配置字典，override 覆盖 base。"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def compute_jacobian_init_control(
    env: RM65Env,
    x0: np.ndarray,
    p_hit: np.ndarray,
    horizon: int,
    gain: float = 50.0,
    fix_joint5_angle: float | None = None,
) -> np.ndarray:
    """基于雅可比转置法计算初始控制序列。"""
    U = np.zeros((horizon, env.NU))
    x = x0.copy()
    ctrl_lo = env.model.actuator_ctrlrange[:env.NU, 0]
    ctrl_hi = env.model.actuator_ctrlrange[:env.NU, 1]

    has_collision_ctrl = hasattr(env, "set_arm_collision")
    if has_collision_ctrl:
        env.set_arm_collision(False)

    for k in range(horizon):
        env.set_arm_state(x)
        p_ee = env.get_ee_pos()
        J_p = env.get_ee_jacp()

        err = p_hit - p_ee
        dist = np.linalg.norm(err)
        scale = gain * min(dist, 0.5)
        tau = J_p.T @ err * scale
        tau -= 2.0 * x[env.NQ:]
        tau = np.clip(tau, ctrl_lo, ctrl_hi)
        U[k] = tau

        if fix_joint5_angle is not None:
            U[k, 5] = 300.0 * (fix_joint5_angle - x[:env.NQ][5]) - 30.0 * x[env.NQ:][5]
        x = env.step_from_state(x, U[k])

    if has_collision_ctrl:
        env.set_arm_collision(True)
    return U


def compute_joint1_backswing_trajectory(
    q1_current: float,
    qdot1_current: float,
    q1_hit: float,
    qdot1_hit: float,
    horizon: int,
    backswing_offset: float = -0.6,
    backswing_ratio: float = 0.35,
) -> np.ndarray:
    """生成关节1的"后摆→前挥"五次多项式轨迹。"""
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


def generate_backswing_warm_start(
    env: RM65Env,
    x0: np.ndarray,
    p_hit: np.ndarray,
    v_hit_desired: np.ndarray,
    horizon: int,
    backswing_offset: float = 0,
    backswing_ratio: float = 0,
    kp: float = 150.0,
    kd: float = 15.0,
    fix_joint5_angle: float | None = None,
    n_des: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """生成带后摆的关节空间轨迹 + PD 跟踪初始控制序列。"""
    NQ = env.NQ
    NU = env.NU
    ctrl_lo = env.model.actuator_ctrlrange[:NU, 0]
    ctrl_hi = env.model.actuator_ctrlrange[:NU, 1]

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

    q1_traj = compute_joint1_backswing_trajectory(
        x0[0], x0[NQ], q_hit[0], qdot_hit[0],
        horizon,
        backswing_offset=backswing_offset,
        backswing_ratio=backswing_ratio,
    )

    q_des_traj = np.zeros((horizon, NQ))
    for j in range(NQ):
        if j == 0:
            q_des_traj[:, j] = q1_traj
        else:
            q_des_traj[:, j] = np.linspace(x0[j], q_hit[j], horizon)

    U = np.zeros((horizon, NU))
    x = x0.copy()

    has_collision_ctrl = hasattr(env, "set_arm_collision")
    if has_collision_ctrl:
        env.set_arm_collision(False)

    for k in range(horizon):
        q_des_k = q_des_traj[k]
        if k < horizon - 1:
            qdot_des_k = (q_des_traj[k + 1] - q_des_k) / env.dt
        else:
            qdot_des_k = qdot_hit

        tau = kp * (q_des_k - x[:NQ]) + kd * (qdot_des_k - x[NQ:])
        if fix_joint5_angle is not None:
            tau[5] = 300.0 * (fix_joint5_angle - x[:NQ][5]) - 30.0 * x[NQ:][5]
        tau = np.clip(tau, ctrl_lo, ctrl_hi)
        U[k] = tau
        x = env.step_from_state(x, U[k])

    if has_collision_ctrl:
        env.set_arm_collision(True)
    return U, q_des_traj


def resample_control_sequence(U_old: np.ndarray, new_horizon: int) -> np.ndarray:
    """将旧控制序列重采样到新 horizon（线性插值）。"""
    old_horizon = len(U_old)
    if old_horizon == new_horizon:
        return U_old.copy()
    if old_horizon == 0:
        return np.zeros((new_horizon, U_old.shape[1]))
    n_u = U_old.shape[1]
    U_new = np.zeros((new_horizon, n_u))
    for k in range(new_horizon):
        t_frac = k / max(new_horizon - 1, 1) * (old_horizon - 1)
        idx_lo = int(np.floor(t_frac))
        idx_hi = min(idx_lo + 1, old_horizon - 1)
        alpha = t_frac - idx_lo
        U_new[k] = (1.0 - alpha) * U_old[idx_lo] + alpha * U_old[idx_hi]
    return U_new


def compute_r_schedule(
    steps_remaining: int,
    base_R: float,
    decay_ratio: float = 0.30,
    joint1_extra_decay: float = 10.0,
) -> np.ndarray:
    """生成 R 退火调度。"""
    if steps_remaining <= 0:
        return np.zeros((0, 6))
    decay_ratio = float(np.clip(decay_ratio, 0.0, 1.0))
    R_schedule = np.full((steps_remaining, 6), base_R)
    decay_start = int(steps_remaining * (1.0 - decay_ratio))
    if decay_start < steps_remaining:
        decay_len = steps_remaining - decay_start
        R_other = base_R * (1.0 - np.linspace(0.0, 1.0, decay_len))
        R_joint1 = base_R * (1.0 - np.linspace(0.0, 1.0, decay_len)) ** joint1_extra_decay
        R_schedule[decay_start:, 0] = R_joint1
        for j in range(1, 6):
            R_schedule[decay_start:, j] = R_other
    return R_schedule


def visualize_rm65_result(
    env: RM65Env,
    X: np.ndarray,
    U: np.ndarray,
    ball_positions_phys: np.ndarray,
    config: dict,
    init_q_left: np.ndarray,
    post_hit_steps: int = 80,
) -> None:
    """在 MuJoCo 查看器中可视化 RM-65 击打结果（含击打后球飞出效果）。

    Args:
        env: RM-65 环境实例。
        X: 右臂状态轨迹，形状 (N+1, 12)。
        U: 控制轨迹，形状 (N, 6)。
        ball_positions_phys: MuJoCo 物理球轨迹，形状 (M, 3)。
        config: 可视化配置。
        init_q_left: 左臂初始关节角度。
        post_hit_steps: 击打后额外仿真步数。
    """
    import mujoco
    import mujoco.viewer

    N = len(U)
    dt = env.dt
    viewer_cfg = config.get("viewer", {})
    playback_speed = viewer_cfg.get("playback_speed", 1.0)
    loop = viewer_cfg.get("loop", True)

    cam_distance = viewer_cfg.get("camera_distance", 3.5)
    cam_elevation = viewer_cfg.get("camera_elevation", -15)
    cam_azimuth = viewer_cfg.get("camera_azimuth", 135)

    total_frames = len(ball_positions_phys)

    bq = env.BALL_QPOS_START
    NQ = env.NQ
    data = env.data
    model = env.model

    data.qpos[:NQ] = X[0, :NQ]
    data.qvel[:NQ] = X[0, NQ:]
    data.qpos[NQ:NQ + env.LEFT_ARM_NQ] = init_q_left
    data.qpos[bq:bq + 3] = ball_positions_phys[0]
    data.qpos[bq + 3:bq + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)

    last_idx = -1

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = cam_distance
        viewer.cam.elevation = cam_elevation
        viewer.cam.azimuth = cam_azimuth
        viewer.cam.lookat[:] = [0.0, 0.0, 1.0]

        model.light_pos[0] = [0.0, 0.0, 8.0]
        model.light_dir[0] = [0.0, 0.0, -1.0]
        model.light_diffuse[0] = [1.4, 1.45, 1.55]
        model.light_ambient[0] = [0.3, 0.3, 0.35]
        model.light_specular[0] = [0.5, 0.5, 0.5]
        if model.nlight > 1:
            model.light_pos[1] = [2.0, -2.0, 3.0]
            model.light_dir[1] = [-0.4, 0.3, -0.8]
            model.light_diffuse[1] = [1.2, 1.15, 1.05]
            model.light_ambient[1] = [0.0, 0.0, 0.0]
            model.light_specular[1] = [0.6, 0.6, 0.6]
            model.light_active[1] = True
        if model.nlight > 2:
            model.light_pos[2] = [-1.5, -1.0, 2.5]
            model.light_dir[2] = [0.3, 0.2, -0.7]
            model.light_diffuse[2] = [0.8, 0.85, 0.95]
            model.light_ambient[2] = [0.0, 0.0, 0.0]
            model.light_specular[2] = [0.4, 0.4, 0.4]
            model.light_active[2] = True
        if model.nlight > 3:
            model.light_pos[3] = [0.0, 2.0, 2.0]
            model.light_dir[3] = [0.0, -0.5, -0.6]
            model.light_diffuse[3] = [0.5, 0.5, 0.55]
            model.light_ambient[3] = [0.0, 0.0, 0.0]
            model.light_specular[3] = [0.3, 0.3, 0.3]
            model.light_active[3] = True

        start_time = time.perf_counter()

        while viewer.is_running():
            elapsed = time.perf_counter() - start_time
            sim_time = elapsed * playback_speed
            idx = int(sim_time / dt)

            if idx >= total_frames:
                if loop:
                    start_time = time.perf_counter()
                    idx = 0
                else:
                    idx = total_frames - 1

            if idx != last_idx:
                last_idx = idx

                if idx <= N:
                    arm_x = X[idx]
                else:
                    arm_x = X[-1]

                data.qpos[:NQ] = arm_x[:NQ]
                data.qvel[:NQ] = arm_x[NQ:]
                data.qpos[NQ:NQ + env.LEFT_ARM_NQ] = init_q_left

                if idx < len(ball_positions_phys):
                    bp = ball_positions_phys[idx]
                    data.qpos[bq: bq + 3] = bp
                    data.qpos[bq + 3: bq + 7] = [1, 0, 0, 0]

                mujoco.mj_forward(model, data)

            viewer.sync()
            time.sleep(1.0 / 120.0)


# ==============================================================================
# Tube 框架 Dataclasses
# ==============================================================================

@dataclass
class TubeConfig:
    """Tube-based robust hitting 配置参数。"""

    window_half_ms: float = 50.0
    """候选窗口半宽（毫秒），以 best_k 为中心前后扩展。"""

    contact_offset: float = 0.0
    """球拍接触点偏移（米），球拍中心到击球面的偏移。"""

    Q_p_tube: float = 50000.0
    """Tube 位置代价权重。"""

    Q_v_tube: float = 200.0
    """Tube 速度代价权重。"""

    Q_n_tube: float = 100000.0
    """Tube 法向量代价权重。"""

    tube_cost_mode: str = "weighted_sum"
    """Tube 代价聚合模式: 'weighted_sum' 或 'softmin'（暂仅支持 weighted_sum）。"""

    tube_cost_ratio: float = 1.0
    """Tube 代价占总代价的比例（0~1），剩余来自原 HittingCost 终端代价。"""

    softmin_beta: float = 5.0
    """终端 softmin 锐度参数 β。β 越大越接近 hard-min（只选最优候选），
    β 越小越接近均匀平均（所有候选等权）。建议范围 1.0~20.0。"""

    use_softmin_terminal: bool = True
    """是否启用多终端 softmin 代价（P0-2 改进）。
    True: 终端代价在多个候选位置上 softmin，容忍时间不确定性。
    False: 仅在 best_k 单点终端代价（原始行为）。"""


@dataclass
class BallTrajectoryTube:
    """带不确定性半径的球轨迹管道。"""

    positions: np.ndarray
    """球位置轨迹，形状 (N, 3)。"""

    velocities: np.ndarray
    """球速度轨迹，形状 (N, 3)。"""

    times: np.ndarray
    """时间序列，形状 (N,)。"""


@dataclass
class HitWindow:
    """候选击球时间窗口。"""

    best_k: int
    """最佳击球步数（find_hitting_point_physics 的结果）。"""

    k_candidates: np.ndarray
    """候选击球步数列表。"""

    p_ball_candidates: np.ndarray
    """候选时刻球位置，形状 (M, 3)。"""

    v_ball_candidates: np.ndarray
    """候选时刻球速度，形状 (M, 3)。"""

    weights: np.ndarray
    """候选权重（高斯衰减），形状 (M,)。"""


@dataclass
class HittingTube:
    """击球管道：包含多个候选时刻的期望球拍状态。"""

    k_candidates: np.ndarray
    """候选击球步数列表。"""

    p_racket_des: np.ndarray
    """期望球拍中心位置，形状 (M, 3)。"""

    v_racket_des: np.ndarray
    """期望球拍速度，形状 (M, 3)。"""

    n_racket_des: np.ndarray
    """期望球拍法向量，形状 (M, 3)。"""

    p_ball: np.ndarray
    """候选时刻球位置，形状 (M, 3)。"""

    v_ball: np.ndarray
    """候选时刻球速度，形状 (M, 3)。"""

    weights: np.ndarray
    """候选权重，形状 (M,)。"""

    best_k: int


# ==============================================================================
# Tube 构建函数
# ==============================================================================

def build_ball_trajectory_tube(
    ball_positions: np.ndarray,
    ball_velocities: np.ndarray,
    dt: float,
    config: TubeConfig,
) -> BallTrajectoryTube:
    """将确定球轨迹转换为管道（仅保留位置和速度信息）。"""
    N = len(ball_positions)
    times = np.arange(N) * dt
    return BallTrajectoryTube(
        positions=ball_positions.copy(),
        velocities=ball_velocities.copy(),
        times=times.copy(),
    )


def search_hit_window(
    env: RM65Env,
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    shoulder_pos: np.ndarray,
    workspace_radius: float,
    horizon: int,
    config: TubeConfig,
    ball_direction: str = "y",
    current_step: int = 0,
    robot_limits: "RobotLimits | None" = None,
    init_q: "np.ndarray | None" = None,
) -> HitWindow | None:
    """搜索候选击球时间窗口。

    先通过 find_hitting_point_physics 得到 best_k，然后以 best_k 为中心
    扩展 candidate_range 步的候选窗口，过滤不满足条件的步。

    返回的 k_candidates 已减去 current_step，为 iLQR 规划地平线内的相对步索引。

    Args:
        env: RM-65 环境实例。
        ball_pos: 球当前位置。
        ball_vel: 球当前速度。
        shoulder_pos: 肩关节世界坐标。
        workspace_radius: 工作空间半径。
        horizon: 规划步数上限。
        config: Tube 配置。
        ball_direction: 球飞来方向。
        current_step: 当前 MPC 绝对仿真步（用于将绝对步转为 iLQR 相对步）。

    Returns:
        HitWindow 或 None（若最佳击打球步不在工作空间内）。
    """
    # 1. 先找最佳击球步
    hit_info = find_hitting_point_physics(
        env, ball_pos, ball_vel, shoulder_pos, workspace_radius, horizon
    )
    if hit_info is None:
        return None

    best_k_abs = hit_info["k_hit"]
    dt = env.dt

    # 2. 窗口扩展范围（步数）——基于绝对步
    window_half_steps = int(round(config.window_half_ms / 1000.0 / dt))
    k_min_abs = max(1, best_k_abs - window_half_steps)
    k_max_abs = min(horizon, best_k_abs + window_half_steps)

    # 3. 预测球轨迹（仅预测窗口范围内的）
    n_pred = k_max_abs + 5
    ball_positions, ball_velocities = env.predict_ball_trajectory(
        ball_pos, ball_vel, n_pred
    )

    # 4. 筛选候选时刻（绝对步）
    candidates_k_abs: list[int] = []
    candidates_p: list[np.ndarray] = []
    candidates_v: list[np.ndarray] = []

    for k in range(k_min_abs, k_max_abs + 1):
        if k < 1 or k > len(ball_positions):
            continue
        p_ball = ball_positions[k - 1]  # predict_ball_trajectory 从 k=0 开始
        v_ball = ball_velocities[k - 1]

        dist = np.linalg.norm(p_ball - shoulder_pos)
        dz = p_ball[2] - shoulder_pos[2]

        # 可达性检查
        if not (dist < workspace_radius and p_ball[2] > 0.3 and -0.60 < dz < 0.55):
            continue

        # ball_direction="y" 时，球从 -Y 飞来，Y 坐标应 < shoulder_pos[1]（前方）
        if ball_direction == "y":
            dy = p_ball[1] - shoulder_pos[1]
            if dy > 0.6:
                continue
        else:
            dx = p_ball[0] - shoulder_pos[0]
            if dx > 0.6:
                continue

        # IK 可达性过滤：排除关节限制超限的点
        if robot_limits is not None and init_q is not None:
            q_ik = env.solve_ik(p_ball, q_init=init_q, max_iter=30, eps=2e-2)
            m_low_deg = (q_ik - robot_limits.q_lower) * 180.0 / np.pi
            m_up_deg = (robot_limits.q_upper - q_ik) * 180.0 / np.pi
            min_margin_deg = float(np.min(np.minimum(m_low_deg, m_up_deg)))
            if min_margin_deg < 3.0:
                continue

        candidates_k_abs.append(k)
        candidates_p.append(p_ball.copy())
        candidates_v.append(v_ball.copy())

    if len(candidates_k_abs) == 0:
        # 回退：至少包含 best_k_abs
        k = best_k_abs
        if 1 <= k <= len(ball_positions):
            p_ball = ball_positions[k - 1]
            v_ball = ball_velocities[k - 1]
            candidates_k_abs.append(k)
            candidates_p.append(p_ball.copy())
            candidates_v.append(v_ball.copy())

    # 5. 计算高斯衰减权重：exp(-0.5 * ((k - best_k) / half_window)^2)
    half_ws = max(window_half_steps, 1)
    k_arr = np.array(candidates_k_abs, dtype=np.float64)
    weights = np.exp(-0.5 * ((k_arr - best_k_abs) / half_ws) ** 2)
    weights /= weights.sum()  # 归一化

    # 6. 将绝对步转为 iLQR 相对步（减去 current_step）
    best_k_rel = best_k_abs - current_step
    k_candidates_rel = np.array(candidates_k_abs, dtype=int) - current_step

    return HitWindow(
        best_k=best_k_rel,
        k_candidates=k_candidates_rel,
        p_ball_candidates=np.array(candidates_p),
        v_ball_candidates=np.array(candidates_v),
        weights=weights,
    )


def build_hitting_tube(
    hit_window: HitWindow,
    desired_speed: float,
    hit_direction: np.ndarray,
    config: TubeConfig,
) -> HittingTube:
    """为每个候选时刻生成期望球拍状态。

    对每个候选时刻 k：
      - p_racket_des[k] = p_ball[k] + contact_offset * target_direction
      - v_racket_des[k] = desired_speed * target_direction
      - n_racket_des[k] = -normalize(v_ball[k])  （拍面朝向来球方向）

    Args:
        hit_window: 候选击球窗口。
        desired_speed: 期望击球速度（标量）。
        hit_direction: 期望击球方向，形状 (3,)。
        config: Tube 配置。

    Returns:
        HittingTube 实例。
    """
    M = len(hit_window.k_candidates)
    d_hat = hit_direction / (np.linalg.norm(hit_direction) + 1e-8)

    p_racket_des = np.zeros((M, 3))
    v_racket_des = np.zeros((M, 3))
    n_racket_des = np.zeros((M, 3))

    for i in range(M):
        v_ball = hit_window.v_ball_candidates[i]
        v_ball_norm = np.linalg.norm(v_ball)
        if v_ball_norm > 1e-6:
            n_des = -v_ball / v_ball_norm
        else:
            n_des = d_hat

        n_racket_des[i] = n_des
        p_racket_des[i] = hit_window.p_ball_candidates[i] + config.contact_offset * d_hat
        v_racket_des[i] = desired_speed * d_hat

    return HittingTube(
        k_candidates=hit_window.k_candidates.copy(),
        p_racket_des=p_racket_des,
        v_racket_des=v_racket_des,
        n_racket_des=n_racket_des,
        p_ball=hit_window.p_ball_candidates.copy(),
        v_ball=hit_window.v_ball_candidates.copy(),
        weights=hit_window.weights.copy(),
        best_k=hit_window.best_k,
    )


# ==============================================================================
# TubeHittingCostWrapper — 与现有 iLQT solver 兼容的 Tube 代价包装器
# ==============================================================================

class TubeHittingCostWrapper:
    """包装 HittingCost，在候选击球窗口内施加空间走廊式 tube 代价。

    空间走廊（hinge loss）：
      - 走廊半宽 = RACKET_RADIUS（固定）
      - hinge loss: margin = perp_dist - RACKET_RADIUS
      - 走廊内零代价，走廊外二次惩罚

    Softmin 多终端代价：
      - 终端代价在所有候选击球位置上取 softmin
      - 求解器不需要精确预测"何时"击球，只需到达某个候选位置即可获得低代价
      - 球早到/晚到时，对应的候选位置提供低代价路径

    原始设计（空间重合，而非时间追踪）：
    - 提取球在窗口内的轨迹线方向 d_ball，构建"空间走廊"
    - 在 tube 窗口内的每个 iLQR 步 k，注入三类代价：
      1. 垂直偏离代价（hinge loss）：不绑定时间-空间对应
      2. 速度方向代价：鼓励球拍沿球轨迹线方向运动
      3. 法向量代价：拍面朝向来球方向
    - 兼容 ILQTSolver 的接口
    """

    RACKET_RADIUS: float = 0.12

    def __init__(
        self,
        env: RM65Env,
        base_cost: HittingCost,
        hitting_tube: HittingTube,
        horizon: int,
        config: TubeConfig,
    ) -> None:
        """初始化 Tube 代价包装器。

        Args:
            env: RM-65 环境实例。
            base_cost: 原始 HittingCost 实例（提供终端代价和基础运行代价）。
            hitting_tube: 击球管道。
            horizon: 规划地平线步数。
            config: Tube 配置。
        """
        self.env = env
        self.base_cost = base_cost
        self.hitting_tube = hitting_tube
        self.horizon = horizon
        self.config = config

        self._tube_ratio = config.tube_cost_ratio
        self._current_ratio = config.tube_cost_ratio
        self._anchor_alpha: float = 0.9
        self._Q_p_tube = config.Q_p_tube
        self._Q_v_tube = config.Q_v_tube
        self._Q_n_tube = config.Q_n_tube

        # P0-2: softmin 参数
        self._use_softmin = config.use_softmin_terminal
        self._softmin_beta = config.softmin_beta

        self._tube_steps: set[int] = set()
        self._tube_weight_scales: dict[int, float] = {}
        self._d_ball: np.ndarray = np.zeros(3)
        self._P_perp: np.ndarray = np.zeros((3, 3))
        self._p_ball_ref: np.ndarray = np.zeros(3)
        self._n_des_common: np.ndarray = np.zeros(3)
        # P0-2: 多终端候选信息（用于 softmin）
        self._p_ball_candidates: np.ndarray = np.zeros((0, 3))
        self._v_des_candidates: np.ndarray = np.zeros((0, 3))
        self._n_des_candidates: np.ndarray = np.zeros((0, 3))
        self._candidate_weights: np.ndarray = np.zeros(0)
        # 可解释性日志：最近一次 softmin 权重和候选代价
        self._last_softmin_alphas: np.ndarray = np.zeros(0)
        self._last_softmin_costs: np.ndarray = np.zeros(0)
        # 可解释性日志：最近一次 tube 走廊 margin
        self._last_tube_margins: dict[int, float] = {}
        self._rebuild_tube_maps(hitting_tube, horizon)

    def _rebuild_tube_maps(self, tube: HittingTube, horizon: int) -> None:
        """重建 tube 步集合、权重缓存和终端候选信息。"""
        self._tube_steps.clear()
        self._tube_weight_scales.clear()

        if len(tube.k_candidates) == 0:
            self._d_ball = np.zeros(3)
            self._P_perp = np.zeros((3, 3))
            self._p_ball_ref = np.zeros(3)
            self._n_des_common = np.zeros(3)
            self._p_ball_candidates = np.zeros((0, 3))
            self._v_des_candidates = np.zeros((0, 3))
            self._n_des_candidates = np.zeros((0, 3))
            self._candidate_weights = np.zeros(0)
            return

        for i in range(len(tube.k_candidates)):
            k = int(tube.k_candidates[i])
            if 0 <= k < horizon:
                self._tube_steps.add(k)
                self._tube_weight_scales[k] = float(tube.weights[i]) * self._current_ratio

        # 球轨迹线方向：用窗口内所有候选球速度的加权平均方向
        weights = tube.weights[:, np.newaxis]
        v_ball_mean = np.sum(weights * tube.v_ball, axis=0)
        v_norm = np.linalg.norm(v_ball_mean)
        if v_norm > 1e-6:
            self._d_ball = v_ball_mean / v_norm
        else:
            self._d_ball = np.array([0.0, -1.0, 0.0])

        # 垂直投影矩阵：P_perp = I - d_ball @ d_ball.T
        self._P_perp = np.eye(3) - np.outer(self._d_ball, self._d_ball)

        # 参考点：best_k 时刻的球位置（走廊中心线上的参考）
        best_idx = int(np.argmin(np.abs(tube.k_candidates - tube.best_k)))
        self._p_ball_ref = tube.p_ball[best_idx].copy()

        # 法向量：用 best_k 对应的拍面法向
        self._n_des_common = tube.n_racket_des[best_idx].copy()

        # P0-2: 保存所有候选位置用于多终端 softmin
        self._p_ball_candidates = tube.p_ball.copy()
        self._v_des_candidates = tube.v_racket_des.copy()
        self._n_des_candidates = tube.n_racket_des.copy()
        self._candidate_weights = tube.weights.copy()

        if len(self._candidate_weights) > 0:
            top3 = np.argsort(self._candidate_weights)[-min(3, len(self._candidate_weights)):][::-1]
            w_str = ", ".join(
                f"k={int(tube.k_candidates[i])}:w={self._candidate_weights[i]:.3f}"
                for i in top3
            )
            logger.info(f"[Softmin诊断] 候选权重TOP3: {w_str}")

    def running_cost(self, x: np.ndarray, u: np.ndarray, k: int | None = None) -> float:
        """计算运行代价 = 原始运行代价 + tube 代价（若 k 在候选窗口内）。"""
        cost = self.base_cost.running_cost(x, u, k)
        if k is not None and k in self._tube_steps:
            cost += self._compute_tube_cost_at_k(x, k)
        return cost

    def terminal_cost(self, x: np.ndarray) -> float:
        """计算终端代价。

        V2 改进（P0-2）：
        若启用 softmin，终端代价在所有候选击球位置上取 softmin：
          cost = -log(Σ_i w_i * exp(-β * c_i)) / β
        其中 c_i = ||p_ee - p_ball[i]||²_Qp + ||v_ee - v_des[i]||²_Qv
        这允许求解器"选择"在任意候选时刻击球，容忍时间不确定性。

        若未启用 softmin，退化为原始单点终端代价。
        """
        self.env.set_arm_state(x)
        p_ee = self.env.get_ee_pos()
        v_ee = self.env.get_ee_vel()
        n_rack = self.env.get_ee_normal()

        if self._use_softmin and len(self._p_ball_candidates) > 1:
            return self._compute_softmin_terminal(p_ee, v_ee, n_rack)
        else:
            return self._compute_single_terminal(p_ee, v_ee, n_rack)

    def _compute_single_terminal(
        self, p_ee: np.ndarray, v_ee: np.ndarray, n_rack: np.ndarray
    ) -> float:
        """原始单点终端代价（best_k 处的精确击打约束）。"""
        dp = p_ee - self.base_cost.p_hit
        cost_p = 0.5 * float(dp @ self.base_cost.Q_p @ dp)

        dv = v_ee - self.base_cost.v_hit
        cost_v = 0.5 * float(dv @ self.base_cost.Q_v @ dv)

        cost = cost_p + cost_v
        if self.base_cost.n_des is not None and self.base_cost.Q_n > 0:
            n_err = n_rack - self.base_cost.n_des
            cost += 0.5 * self.base_cost.Q_n * float(n_err @ n_err)
        return (1.0 - self._current_ratio) * cost

    def _compute_softmin_terminal(
        self, p_ee: np.ndarray, v_ee: np.ndarray, n_rack: np.ndarray
    ) -> float:
        """P0-2: 多终端 softmin 代价。

        对每个候选位置 i 计算：
          c_i = 0.5 * [ ||p_ee - p_ball_i||²_Qp + ||v_ee - v_des_i||²_Qv
                        + Q_n * ||n_rack - n_des_i||² ]
        然后取 softmin:
          cost = -log(Σ_i w_i * exp(-β * c_i)) / β

        softmin 的效果：只有代价最低的候选（最接近的候选位置）主导结果，
        但梯度从所有候选流向最优点附近的候选，保证光滑可微。

        Returns:
            终端代价值（已乘以 tube_ratio 缩放）。
        """
        M = len(self._p_ball_candidates)
        Q_p = self.base_cost.Q_p
        Q_v = self.base_cost.Q_v
        Q_n = self.base_cost.Q_n

        costs_i = np.zeros(M)
        for i in range(M):
            dp = p_ee - self._p_ball_candidates[i]
            costs_i[i] = 0.5 * float(dp @ Q_p @ dp)

            dv = v_ee - self._v_des_candidates[i]
            costs_i[i] += 0.5 * float(dv @ Q_v @ dv)

            if Q_n > 0:
                n_err = n_rack - self._n_des_candidates[i]
                costs_i[i] += 0.5 * Q_n * float(n_err @ n_err)

        # softmin: -log(Σ w_i * exp(-β * c_i)) / β
        # 数值稳定：减去最大值避免溢出
        beta = self._softmin_beta
        weighted_neg_costs = -beta * costs_i + np.log(self._candidate_weights + 1e-30)
        max_wnc = np.max(weighted_neg_costs)
        log_sum = max_wnc + np.log(np.sum(np.exp(weighted_neg_costs - max_wnc)))
        softmin_val = -log_sum / beta

        return (1.0 - self._current_ratio) * softmin_val

    def running_derivatives(
        self, x: np.ndarray, u: np.ndarray, k: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """计算运行代价导数 = 原始导数 + tube 导数（若 k 在候选窗口内）。"""
        l_x, l_u, l_xx, l_ux, l_uu = self.base_cost.running_derivatives(x, u, k)
        if k is not None and k in self._tube_steps:
            tl_x, tl_xx = self._compute_tube_derivatives_at_k(x, k)
            l_x += tl_x
            l_xx += tl_xx
        return l_x, l_u, l_xx, l_ux, l_uu

    def terminal_derivatives(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """终端代价导数。

        V2 改进（P0-2）：softmin 加权导数。
        softmin 的梯度 = Σ_i α_i * ∂c_i/∂x，其中 α_i 是 softmin 权重。
        """
        n_x = self.env.NX
        n_q = self.env.NQ

        self.env.set_arm_state(x)
        p_ee = self.env.get_ee_pos()
        v_ee = self.env.get_ee_vel()
        n_rack = self.env.get_ee_normal()
        J_p = self.env.get_ee_jacp()

        if self._use_softmin and len(self._p_ball_candidates) > 1:
            l_x, l_xx = self._compute_softmin_terminal_derivatives(
                p_ee, v_ee, n_rack, J_p, n_x, n_q
            )
        else:
            l_x, l_xx = self._compute_single_terminal_derivatives(
                p_ee, v_ee, n_rack, J_p, n_x, n_q
            )

        scale = 1.0 - self._current_ratio
        return scale * l_x, scale * l_xx

    def _compute_single_terminal_derivatives(
        self,
        p_ee: np.ndarray,
        v_ee: np.ndarray,
        n_rack: np.ndarray,
        J_p: np.ndarray,
        n_x: int,
        n_q: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """原始单点终端代价导数。"""
        K_p = self.base_cost.Q_p
        dp = p_ee - self.base_cost.p_hit

        l_x = np.zeros(n_x)
        l_xx = np.zeros((n_x, n_x))

        l_x[:n_q] = J_p.T @ K_p @ dp
        l_xx[:n_q, :n_q] = J_p.T @ K_p @ J_p

        dv = v_ee - self.base_cost.v_hit
        l_x[n_q:] = J_p.T @ self.base_cost.Q_v @ dv
        l_xx[n_q:, n_q:] = J_p.T @ self.base_cost.Q_v @ J_p

        if self.base_cost.n_des is not None and self.base_cost.Q_n > 0:
            J_omega = self.env.get_ee_jacr()
            nx, ny, nz = -n_rack[0], -n_rack[1], -n_rack[2]
            skew = np.array([
                [0, -nz, ny],
                [nz, 0, -nx],
                [-ny, nx, 0],
            ])
            J_n = np.zeros((3, n_x))
            J_n[:, :n_q] = skew @ J_omega
            n_err = n_rack - self.base_cost.n_des
            l_x += self.base_cost.Q_n * (J_n.T @ n_err)
            l_xx += self.base_cost.Q_n * (J_n.T @ J_n)

        return l_x, l_xx

    def _compute_softmin_terminal_derivatives(
        self,
        p_ee: np.ndarray,
        v_ee: np.ndarray,
        n_rack: np.ndarray,
        J_p: np.ndarray,
        n_x: int,
        n_q: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """P0-2: 多终端 softmin 代价的 Gauss-Newton 近似导数。

        softmin(c_1, ..., c_M) = -log(Σ w_i * exp(-β * c_i)) / β

        梯度：∂softmin/∂x = Σ_i α_i * ∂c_i/∂x
        Hessian 近似：Σ_i α_i * ∂²c_i/∂x²
        其中 α_i = w_i * exp(-β * c_i) / Σ_j w_j * exp(-β * c_j)  是 softmin 权重。

        Returns:
            (l_x, l_xx) — 终端代价对状态的一阶和二阶导数。
        """
        M = len(self._p_ball_candidates)
        Q_p = self.base_cost.Q_p
        Q_v = self.base_cost.Q_v
        Q_n = self.base_cost.Q_n
        beta = self._softmin_beta

        # 计算每个候选的代价
        costs_i = np.zeros(M)
        for i in range(M):
            dp = p_ee - self._p_ball_candidates[i]
            costs_i[i] = 0.5 * float(dp @ Q_p @ dp)
            dv = v_ee - self._v_des_candidates[i]
            costs_i[i] += 0.5 * float(dv @ Q_v @ dv)
            if Q_n > 0:
                n_err = n_rack - self._n_des_candidates[i]
                costs_i[i] += 0.5 * Q_n * float(n_err @ n_err)

        # 计算 softmin 权重 α_i
        weighted_neg_costs = -beta * costs_i + np.log(self._candidate_weights + 1e-30)
        max_wnc = np.max(weighted_neg_costs)
        exp_wnc = np.exp(weighted_neg_costs - max_wnc)
        alpha_i = exp_wnc / np.sum(exp_wnc)

        # 可解释性：缓存 softmin 权重和候选代价
        self._last_softmin_alphas = alpha_i.copy()
        self._last_softmin_costs = costs_i.copy()

        # 加权组合各候选的导数
        l_x = np.zeros(n_x)
        l_xx = np.zeros((n_x, n_x))

        # 法向量雅可比（对所有候选共用）
        J_omega = None
        J_n = None
        if Q_n > 0:
            J_omega = self.env.get_ee_jacr()
            nx, ny, nz = -n_rack[0], -n_rack[1], -n_rack[2]
            skew = np.array([
                [0, -nz, ny],
                [nz, 0, -nx],
                [-ny, nx, 0],
            ])
            J_n = np.zeros((3, n_x))
            J_n[:, :n_q] = skew @ J_omega

        for i in range(M):
            a_i = alpha_i[i]
            if a_i < 1e-12:
                continue

            # 位置导数
            dp = p_ee - self._p_ball_candidates[i]
            l_x[:n_q] += a_i * (J_p.T @ Q_p @ dp)
            l_xx[:n_q, :n_q] += a_i * (J_p.T @ Q_p @ J_p)

            # 速度导数
            dv = v_ee - self._v_des_candidates[i]
            l_x[n_q:] += a_i * (J_p.T @ Q_v @ dv)
            l_xx[n_q:, n_q:] += a_i * (J_p.T @ Q_v @ J_p)

            # 法向量导数
            if Q_n > 0 and J_n is not None:
                n_err = n_rack - self._n_des_candidates[i]
                l_x += a_i * Q_n * (J_n.T @ n_err)
                l_xx += a_i * Q_n * (J_n.T @ J_n)

        # Hessian 修正：加入 α_i 的一阶项（交叉项）
        # 对于 softmin，完整的 Hessian 包含 Σ_i α_i * (∂c_i/∂x)(∂c_i/∂x)^T
        # 减去 (Σ_i α_i * ∂c_i/∂x)(Σ_i α_i * ∂c_i/∂x)^T 乘以 β
        # 但这在 Gauss-Newton 近似中通常省略，因为 β 不太大时影响有限
        # 此处保留简化版本，仅用加权二阶项

        return l_x, l_xx

    def update_target(self, p_hit: np.ndarray, v_hit: np.ndarray, n_des: np.ndarray | None = None) -> None:
        """委托给 base_cost 更新终端目标。"""
        self.base_cost.update_target(p_hit, v_hit, n_des=n_des)

    def update_weights(self, Q_p_scale: float = 1.0, Q_v_scale: float = 1.0) -> None:
        """委托给 base_cost 更新权重。"""
        self.base_cost.update_weights(Q_p_scale, Q_v_scale)

    def set_q_des_traj(self, q_des_traj: np.ndarray | None, Q_joint: dict | None = None) -> None:
        """委托给 base_cost 设置关节轨迹。"""
        self.base_cost.set_q_des_traj(q_des_traj, Q_joint)

    def set_R_schedule(self, R_schedule: np.ndarray | None) -> None:
        """委托给 base_cost 设置 R 调度。"""
        self.base_cost.set_R_schedule(R_schedule)

    def update_hitting_tube(self, hitting_tube: HittingTube, horizon: int | None = None) -> None:
        """更新击球管道（用于 MPC 重规划）。

        Args:
            hitting_tube: 新的击球管道（k_candidates 应为 iLQR 相对步）。
            horizon: 新的规划地平线步数。None 表示保持原值。
        """
        self.hitting_tube = hitting_tube
        if horizon is not None:
            self.horizon = horizon
        self._rebuild_tube_maps(hitting_tube, self.horizon)

    def update_tube_params(self, ratio: float, anchor_alpha: float) -> None:
        """更新 tube 代价参数（用于渐进衰减策略）。

        Args:
            ratio: 新的有效 tube 代价比例 (0~1)。
            anchor_alpha: 终端锚定强度 (0~1)。0=全约束，1=沿d_ball完全自由。
        """
        self._current_ratio = max(0.0, min(1.0, ratio))
        self._anchor_alpha = max(0.0, min(1.0, anchor_alpha))
        # 用新比例重建 tube 权重
        self._rebuild_tube_maps(self.hitting_tube, self.horizon)

    def _compute_tube_cost_at_k(self, x: np.ndarray, k: int) -> float:
        """计算步骤 k 处的空间走廊 tube 代价值。

        走廊半宽 = RACKET_RADIUS（固定）。
        走廊内（perp_dist < RACKET_RADIUS）零代价，
        走廊外二次惩罚。

        三项代价：
        1. 垂直偏离代价（hinge loss）：margin > 0 时惩罚
        2. 速度方向代价：球拍速度垂直于球轨迹方向的分量
        3. 法向量代价：拍面法向与期望法向的对齐程度
        """
        self.env.set_arm_state(x)
        p_ee = self.env.get_ee_pos()
        v_ee = self.env.get_ee_vel()
        n_rack = self.env.get_ee_normal()

        # 1. 垂直偏离代价（hinge loss）
        dp = p_ee - self._p_ball_ref
        dp_perp = self._P_perp @ dp
        perp_dist = float(np.linalg.norm(dp_perp))
        margin = perp_dist - self.RACKET_RADIUS
        pos_err = max(0.0, margin)
        pos_cost = self._Q_p_tube * pos_err**2

        # 2. 速度方向代价：球拍速度垂直分量应尽量小
        v_perp = self._P_perp @ v_ee
        vel_cost = self._Q_v_tube * float(v_perp @ v_perp)

        # 3. 法向量代价
        n_dot = float(n_rack @ self._n_des_common)
        normal_cost = self._Q_n_tube * (1.0 - n_dot)

        scale = self._tube_weight_scales.get(k, 1.0)
        return 0.5 * scale * (pos_cost + vel_cost + normal_cost)

    def _compute_tube_derivatives_at_k(
        self, x: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """计算步骤 k 处空间走廊 tube 代价的 Gauss-Newton 近似导数。

        hinge loss 的导数仅在 margin > 0 时激活，
        走廊内梯度为零（不干扰优化），走廊外提供平滑梯度引导球拍回到走廊内。

        Returns:
            (l_x_tube, l_xx_tube)
        """
        self.env.set_arm_state(x)
        n_x = self.env.NX
        n_q = self.env.NQ

        p_ee = self.env.get_ee_pos()
        v_ee = self.env.get_ee_vel()
        n_rack = self.env.get_ee_normal()
        J_p = self.env.get_ee_jacp()

        scale = self._tube_weight_scales.get(k, 1.0)

        l_x_tube = np.zeros(n_x)
        l_xx_tube = np.zeros((n_x, n_x))

        # ---- 1. 垂直偏离代价（hinge loss）----
        # cost = Q_p * max(0, perp_dist - RACKET_RADIUS)^2
        # 当 margin > 0 时有梯度，否则为零
        dp = p_ee - self._p_ball_ref
        dp_perp = self._P_perp @ dp
        perp_dist = float(np.linalg.norm(dp_perp))
        margin = perp_dist - self.RACKET_RADIUS

        if margin > 0.0 and perp_dist > 1e-8:
            # 梯度方向：dp_perp 的单位向量
            dp_perp_hat = dp_perp / perp_dist
            # ∂perp_dist/∂q = dp_perp_hat^T @ P_perp @ J_p
            grad_perp_q = dp_perp_hat @ self._P_perp @ J_p  # (n_q,)
            l_x_tube[:n_q] += self._Q_p_tube * margin * grad_perp_q
            l_xx_tube[:n_q, :n_q] += self._Q_p_tube * np.outer(grad_perp_q, grad_perp_q)

        # 可解释性：缓存走廊 margin
        self._last_tube_margins[k] = margin

        # ---- 2. 速度方向代价 (v_perp = P_perp @ v_ee) ----
        Jp_perp = self._P_perp @ J_p
        v_perp = self._P_perp @ v_ee
        l_x_tube[n_q:] += self._Q_v_tube * (Jp_perp.T @ v_perp)
        l_xx_tube[n_q:, n_q:] += self._Q_v_tube * (Jp_perp.T @ Jp_perp)

        # ---- 3. 法向量代价 ----
        J_omega = self.env.get_ee_jacr()
        nx, ny, nz = -n_rack[0], -n_rack[1], -n_rack[2]
        skew = np.array([
            [0, -nz, ny],
            [nz, 0, -nx],
            [-ny, nx, 0],
        ])
        J_n = np.zeros((3, n_x))
        J_n[:, :n_q] = skew @ J_omega

        n_dot = float(n_rack @ self._n_des_common)
        n_err_val = 1.0 - n_dot
        if abs(n_err_val) > 1e-8:
            dn_dx = -J_n.T @ self._n_des_common
            l_x_tube += self._Q_n_tube * n_err_val * dn_dx
            l_xx_tube += self._Q_n_tube * np.outer(dn_dx, dn_dx)

        l_x_tube *= scale
        l_xx_tube *= scale

        return l_x_tube, l_xx_tube


# ==============================================================================
# 可视化函数
# ==============================================================================

def plot_tube_results(
    results_dir: Path,
    tag: str,
    ball_positions: np.ndarray,
    racket_positions: np.ndarray,
    hit_window: HitWindow | None,
    distances: list[float],
    normal_alignments: list[float],
    ball_near_flags: list[bool],
    tube_ready_flags: list[bool],
    k_hit_history: list[int],
    pos_errors: list[float],
) -> None:
    """保存 Tube 实验的可视化图像（非阻塞）。

    Args:
        results_dir: 输出目录。
        tag: 文件名标签。
        ball_positions: 球轨迹，形状 (N, 3)。
        racket_positions: 球拍中心轨迹，形状 (N, 3)。
        hit_window: 击球窗口。
        distances: 球拍-球距离序列。
        normal_alignments: 法向量对齐序列。
        ball_near_flags: 球物理上在拍附近的布尔序列。
        tube_ready_flags: 球拍在 tube 窗口内保持击球姿态的布尔序列。
        k_hit_history: 每步的 k_hit 估计。
        pos_errors: 位置误差序列。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib 未安装，跳过可视化")
        return

    results_dir.mkdir(parents=True, exist_ok=True)

    N = min(len(distances), len(normal_alignments), len(ball_near_flags), len(tube_ready_flags))
    t_axis = np.arange(N)

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(f"Tube-Based Robust Hitting Results [{tag}]", fontsize=14)

    # 子图 1: 球轨迹 + 球拍中心轨迹 3D
    ax3d = fig.add_subplot(3, 2, 1, projection="3d")
    ax3d.plot(ball_positions[:len(racket_positions), 0],
              ball_positions[:len(racket_positions), 1],
              ball_positions[:len(racket_positions), 2], "b-", alpha=0.5, label="Ball")
    ax3d.plot(racket_positions[:, 0], racket_positions[:, 1], racket_positions[:, 2],
              "r-", alpha=0.8, label="Racket")
    if hit_window is not None and len(hit_window.p_ball_candidates) > 0:
        ax3d.scatter(hit_window.p_ball_candidates[:, 0],
                     hit_window.p_ball_candidates[:, 1],
                     hit_window.p_ball_candidates[:, 2],
                     c="orange", s=20, marker="o", label="Hit Window")
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.legend()
    ax3d.set_title("Ball & Racket Trajectory")

    # 子图 2: 球拍-球距离
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(t_axis, distances[:N], "g-", linewidth=1)
    ax2.axhline(y=0.033 + 0.12, color="gray", linestyle="--", label="Racket+ball radius")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Distance (m)")
    ax2.set_title("Racket-Ball Distance")
    ax2.legend()

    # 子图 3: 法向量对齐
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(t_axis, normal_alignments[:N], "m-", linewidth=1)
    ax3.axhline(y=0.9, color="gray", linestyle="--", label="90% alignment")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("dot(n_rack, n_des)")
    ax3.set_title("Normal Alignment")
    ax3.set_ylim(-1.05, 1.05)
    ax3.legend()

    # 子图 4: ball_near vs tube_ready 双指标
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.fill_between(t_axis, 0, np.array(ball_near_flags[:N], dtype=float),
                     step="mid", alpha=0.4, color="orange", label="ball_near")
    ax4.fill_between(t_axis, 0, np.array(tube_ready_flags[:N], dtype=float),
                     step="mid", alpha=0.3, color="cyan", label="tube_ready")
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Flag")
    ax4.set_title("ball_near vs tube_ready")
    ax4.set_ylim(-0.1, 1.3)
    ax4.legend()

    # 子图 5: k_hit 估计变化
    ax5 = fig.add_subplot(3, 2, 5)
    ax5.plot(range(len(k_hit_history)), k_hit_history, "b.-", markersize=3)
    if hit_window is not None:
        ax5.axhline(y=hit_window.best_k, color="orange", linestyle="--", label=f"best_k={hit_window.best_k}")
        k_low = hit_window.k_candidates[0] if len(hit_window.k_candidates) > 0 else hit_window.best_k
        k_high = hit_window.k_candidates[-1] if len(hit_window.k_candidates) > 0 else hit_window.best_k
        ax5.fill_between(range(len(k_hit_history)), k_low, k_high, alpha=0.15, color="green", label="Hit Window")
    ax5.set_xlabel("Replan")
    ax5.set_ylabel("k_hit")
    ax5.set_title("Hitting Step Estimation")
    ax5.legend()

    # 子图 6: 位置误差
    ax6 = fig.add_subplot(3, 2, 6)
    ax6.plot(range(len(pos_errors)), pos_errors, "r.-", markersize=3)
    ax6.set_xlabel("Step")
    ax6.set_ylabel("Position Error (m)")
    ax6.set_title("Position Error over Time")
    ax6.axhline(y=0.05, color="gray", linestyle="--", label="5cm")
    ax6.legend()

    plt.tight_layout()
    out_path = results_dir / f"tube_results_{tag}.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    logger.info(f"可视化已保存到 {out_path}")


# ==============================================================================
# 异步重规划：状态 + 规划函数
# ==============================================================================

@dataclass
class ReplanState:
    """重规划所需的可变状态（主线程和后台线程共享的快照）。"""
    k_hit_new: int = 0
    p_hit_new: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v_ball_hit_new: np.ndarray = field(default_factory=lambda: np.zeros(3))
    current_n_des: np.ndarray = field(default_factory=lambda: np.zeros(3))
    U_prev: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    is_first_plan: bool = True
    hitting_tube: object = None
    cost_fn_type: str = ""


def do_replan(
    request: PlanRequest,
    env_plan: RM65Env,
    state: ReplanState,
    cfg: dict,
) -> PlanResult:
    """在后台线程中执行完整的重规划流程。

    使用独立的 env_plan（独立 MjData），不与主线程共享可变 MuJoCo 状态。

    Args:
        request: 规划请求（球状态、臂状态等）。
        env_plan: 独立 MjData 的规划环境。
        state: 当前的重规划状态快照。
        cfg: 配置字典，包含所有规划参数。

    Returns:
        PlanResult: 规划结果（新控制序列、击打点等）。
    """
    result = PlanResult()
    x_current = request.x_current.copy()
    ball_pos = request.ball_pos.copy()
    ball_vel = request.ball_vel.copy()
    step = request.step
    k_hit_new = state.k_hit_new

    remaining_horizon = cfg["total_horizon"] - step

    # 1. 查找击打点
    env_plan.set_ball_state(ball_pos, ball_vel)
    hit_info_new = find_hitting_point_physics(
        env_plan, ball_pos, ball_vel, cfg["shoulder_pos"], cfg["workspace_radius"],
        remaining_horizon,
    )
    if hit_info_new is None:
        logger.warning(
            f"ASYNC 步 {step}: 球不可达, ball_pos={ball_pos}, ball_vel={ball_vel}, "
            f"remaining_horizon={remaining_horizon}"
        )
        result.ball_unreachable = True
        return result

    k_hit_candidate = hit_info_new["k_hit"]
    if k_hit_candidate < max(10, k_hit_new // 4) and k_hit_new > 30:
        k_hit_candidate = max(1, k_hit_new - cfg["replan_interval"])

    # 衰减式扰动：计算衰减系数（模拟预测精度随观测逐步提高）
    decay_alpha = 1.0
    if abs(cfg.get("time_perturb_s", 0.0)) > 1e-6 or abs(cfg.get("space_perturb_m", 0.0)) > 1e-6:
        r_interval = cfg.get("replan_interval", 20)
        r_total = max(1, cfg.get("k_hit_total", 100) // r_interval)
        r_count = max(1, (cfg.get("k_hit_total", 100) - remaining_horizon) // r_interval + 1)
        decay_alpha = max(cfg.get("perturb_alpha_min", 0.0), 1.0 - r_count / r_total)

    # 时间扰动（衰减式）
    if abs(cfg.get("time_perturb_s", 0.0)) > 1e-6:
        effective_perturb = cfg["time_perturb_s"] * decay_alpha
        perturb_steps = int(round(effective_perturb / cfg["dt"]))
        if perturb_steps != 0:
            k_hit_candidate = k_hit_candidate - perturb_steps
            k_hit_candidate = max(5, min(k_hit_candidate, remaining_horizon - 1))

    p_hit_new = hit_info_new["p_hit"].copy()
    v_ball_hit_new = hit_info_new["v_ball_hit"].copy()
    k_hit_new = k_hit_candidate

    # 2. 击球点可执行性后过滤
    q_hit_feas = env_plan.solve_ik(p_hit_new, q_init=x_current[:env_plan.NQ], max_iter=50, eps=1e-2)
    env_plan.set_arm_state(np.concatenate([q_hit_feas, np.zeros(env_plan.NQ)]))
    J_p_feas = env_plan.get_ee_jacp()
    max_ee_v = float(np.linalg.norm(np.abs(J_p_feas) @ cfg["robot_limits"].qdot_max))
    ball_spd = float(np.linalg.norm(v_ball_hit_new))
    if ball_spd > max_ee_v * 2.0:
        logger.warning(f"ASYNC 步 {step}: 球速 {ball_spd:.1f}m/s 超过限速 {max_ee_v:.1f}m/s")

    # 3. 空间偏移（衰减式）
    if abs(cfg.get("space_perturb_m", 0.0)) > 1e-6:
        effective_sp = cfg["space_perturb_m"] * decay_alpha
        if abs(effective_sp) > 1e-6:
            d_ball_hit = v_ball_hit_new / (np.linalg.norm(v_ball_hit_new) + 1e-8)
            lateral = np.cross(d_ball_hit, np.array([0.0, 0.0, 1.0]))
            lateral_norm = np.linalg.norm(lateral)
            if lateral_norm > 1e-6:
                lateral /= lateral_norm
            else:
                lateral = np.array([1.0, 0.0, 0.0])
            p_hit_new = p_hit_new + lateral * effective_sp

    # 4. 法向量和随挥目标
    n_des_new = -v_ball_hit_new / (np.linalg.norm(v_ball_hit_new) + 1e-8)
    if cfg.get("normal_flip", False):
        n_des_new = -n_des_new

    # v5: 终端目标 = 随挥终点（击球点前方 follow_through_length）
    p_follow_new = p_hit_new + cfg["hit_shift"] * cfg["d_hat"]

    # v5: 扩展 horizon 包含随挥段，击球时刻变为轨迹中段
    follow_through_steps_cfg = cfg.get("follow_through_steps", 60)
    follow_through_length_cfg = cfg.get("follow_through_length", 0.5)
    follow_through_v_terminal_cfg = cfg.get("follow_through_v_terminal", 0.3)
    horizon_full = k_hit_new + follow_through_steps_cfg  # v5: 终端在随挥终点
    horizon_plan = min(horizon_full, cfg["fixed_horizon"])

    # v5: 终端目标 = 随挥终点（低速），iLQR 规划"加速→击球→减速→终点"
    p_terminal_v5 = p_hit_new + follow_through_length_cfg * cfg["d_hat"]
    v_terminal_v5 = follow_through_v_terminal_cfg * cfg["d_hat"]

    # 6. 位置误差 → 权重调度
    env_plan.set_arm_state(x_current)
    env_plan.update_kinematics()
    pos_err_now = float(np.linalg.norm(env_plan.get_ee_pos() - p_hit_new))

    if pos_err_now > 0.10:
        Q_p_scale = cfg["Q_p_scale_far"]
        Q_v_scale = cfg["Q_v_scale_far"]
    else:
        ratio = pos_err_now / 0.10
        Q_p_scale = cfg["Q_p_scale_near"] + (cfg["Q_p_scale_far"] - cfg["Q_p_scale_near"]) * ratio
        Q_v_scale = cfg["Q_v_scale_near"] + (cfg["Q_v_scale_far"] - cfg["Q_v_scale_near"]) * ratio

    # 7. 迭代策略（与主循环同步策略保持一致）
    near_threshold = cfg["near_threshold"]
    iters_plan = cfg["max_iter_per_plan"]
    skip_ls = True
    fp_limits = cfg["robot_limits"]
    fast_lin = False

    if request.is_first_plan:
        iters_plan = cfg["first_plan_iters"]
        skip_ls = True
        fp_limits = None
    elif k_hit_new <= near_threshold:
        iters_plan = cfg["near_plan_iters"]
        if k_hit_new > 30:
            fast_lin = True
            fp_limits = None
    else:
        iters_plan = cfg["max_iter_per_plan"]
        fast_lin = True

    # 8. 计算 warm start
    U_prev = request.U_prev.copy()
    fix_joint5_angle = cfg.get("fix_joint5_angle")

    if not cfg.get("use_backswing", False):
        if len(U_prev) >= horizon_full // 3:
            U_warm = resample_control_sequence(U_prev, horizon_full)[:horizon_plan]
            if fix_joint5_angle is not None:
                U_warm = fix_joint5_control_trajectory(U_warm, x_current, env_plan, fix_joint5_angle)
        else:
            U_warm = compute_jacobian_init_control(
                env_plan, x_current, p_follow_new, horizon_full, gain=30.0,
                fix_joint5_angle=fix_joint5_angle,
            )[:horizon_plan]
    else:
        if len(U_prev) >= horizon_full // 3:
            U_warm = resample_control_sequence(U_prev, horizon_full)[:horizon_plan]
            if fix_joint5_angle is not None:
                U_warm = fix_joint5_control_trajectory(U_warm, x_current, env_plan, fix_joint5_angle)
        else:
            U_warm_full, _ = generate_backswing_warm_start(
                env_plan, x_current, p_follow_new, cfg.get("v_hit_at_contact", cfg["v_hit_desired"]), horizon_full,
                backswing_offset=cfg.get("backswing_offset", 0.0),
                backswing_ratio=cfg.get("backswing_ratio", 0.3),
                fix_joint5_angle=fix_joint5_angle,
                n_des=n_des_new,
            )
            U_warm = U_warm_full[:horizon_plan]

    # 9. 构建 cost_fn（使用临时 cost_fn，env_plan 独立 MjData）
    Q_p_mat = Q_p_scale * np.eye(3)
    # v4: 方向对齐速度代价 — 只惩罚沿 d_follow 方向
    d_follow = cfg.get("d_follow", np.array([0.0, -1.0, 0.0]))
    Q_v_scalar = cfg.get("Q_v_scalar", 400.0)
    Q_v_mat = Q_v_scale * Q_v_scalar * np.outer(d_follow, d_follow)
    R_mat = cfg["R"] * np.eye(env_plan.NU)
    cost_fn_plan = HittingCost(
        env_plan, p_terminal_v5, v_terminal_v5, Q_p_mat, Q_v_mat, R_mat,
        n_des=n_des_new if cfg.get("Q_n", 0) > 0 else None,
    )

    if cfg.get("use_r_decay", False):
        R_schedule = compute_r_schedule(
            horizon_full, cfg["R"],
            decay_ratio=cfg.get("r_decay_ratio", 0.3),
        )[:horizon_plan]
        cost_fn_plan.set_R_schedule(R_schedule)

    # v5: 中途位置目标 — 在 k_hit 步强制经过击球位置 + 鼓励高速
    if k_hit_new > 0 and k_hit_new < horizon_plan:
        Q_midpoint = Q_p_mat * 2.0
        Q_midpoint_v = Q_v_mat * 5.0  # 速度代价权重（高权重鼓励击球时高速）
        cost_fn_plan.set_midpoint_target(k_hit_new, p_hit_new, Q_midpoint,
                                         v_target=cfg["v_hit_at_contact"],
                                         Q_midpoint_v=Q_midpoint_v)

    # 分阶段软平滑权重调度
    if hasattr(cost_fn_plan, 'set_smoothness_scale'):
        if k_hit_new > 50:
            sq = cfg.get("smooth_far", {"Q_qdot_mult": 0.01, "Q_qddot_mult": 0.01, "Q_du_mult": 0.1})
        elif k_hit_new > 20:
            sq = cfg.get("smooth_mid", {"Q_qdot_mult": 0.1,  "Q_qddot_mult": 0.1,  "Q_du_mult": 0.2})
        else:
            sq = cfg.get("smooth_near", {"Q_qdot_mult": 1.0,  "Q_qddot_mult": 1.0,  "Q_du_mult": 0.5})
        cost_fn_plan.set_smoothness_scale(
            float(sq["Q_qdot_mult"]), float(sq["Q_qddot_mult"]), float(sq["Q_du_mult"]),
        )

    # 10. 求解 iLQR
    ball_pos_save, ball_vel_save = env_plan.get_ball_state()

    X_mpc, U_mpc, iter_costs, solver_ok = cfg["solver"].solve_few_iters(
        env_plan, cost_fn_plan, x_current, U_warm,
        max_iter=iters_plan,
        skip_linesearch=skip_ls,
        limits=fp_limits,
        use_fast_lin=fast_lin,
    )

    env_plan.set_ball_state(ball_pos_save, ball_vel_save)
    env_plan.set_arm_state(x_current)

    # 11. 构建 PlanResult
    result.request_step = step
    result.k_hit_new = k_hit_new
    result.p_hit_new = p_hit_new
    result.v_ball_hit_new = v_ball_hit_new
    result.n_des_new = n_des_new
    result.solver_ok = solver_ok
    result.iters_plan = iters_plan
    result.horizon_plan = horizon_plan
    result.fast_lin = fast_lin
    result.fp_limits_was_none = (fp_limits is None)
    result.U_mpc_full = U_mpc.copy()

    if solver_ok:
        if fix_joint5_angle is not None:
            U_mpc = fix_joint5_control_trajectory(U_mpc, x_current, env_plan, fix_joint5_angle)
            result.U_mpc_full = U_mpc.copy()

        # U_prev：保存规划尾部，用于下次 warm start
        if len(U_mpc) > cfg["replan_interval"]:
            result.U_prev = U_mpc[cfg["replan_interval"]:].copy()
        elif len(U_mpc) > 0:
            result.U_prev = U_mpc[1:].copy()

        # U_buffer：异步模式需更长 buffer 覆盖后台规划延迟（≈horizon×0.2ms/步）
        # 远段 horizon~100 → ~200ms ≈ 40步，buffer 取 60步（×1.5余量）
        buffer_interval = cfg["replan_interval"]
        if len(U_mpc) >= buffer_interval * 6:
            result.U_buffer = U_mpc[:buffer_interval * 6].copy()
        elif len(U_mpc) >= buffer_interval * 4:
            result.U_buffer = U_mpc[:buffer_interval * 4].copy()
        elif k_hit_new <= 30 and len(U_mpc) >= buffer_interval * 2:
            result.U_buffer = U_mpc[:buffer_interval * 2].copy()
        else:
            result.U_buffer = U_mpc[:min(len(U_mpc), buffer_interval)].copy()
    else:
        # fallback: JT 控制
        u_jt = compute_jacobian_init_control(
            env_plan, x_current, p_follow_new, horizon=cfg["replan_interval"], gain=40.0,
        )
        result.U_buffer = u_jt[:cfg["replan_interval"]].copy()
        result.U_prev = np.zeros((0, env_plan.NU))

    return result


# ==============================================================================
# 主函数
# ==============================================================================

def main() -> None:
    """RM-65 Tube-Based Robust Hitting 主函数。"""
    parser = argparse.ArgumentParser(description="RM-65 Tube-Based MPC+iLQT 网球击打")
    parser.add_argument("--use_tube", type=str, default="true",
                        help="是否启用 tube-based 鲁棒击球 (true/false)")
    parser.add_argument("--viewer", action="store_true", help="计算完成后以真实速度回放")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--fd", action="store_true", help="使用有限差分线性化")
    parser.add_argument("--horizon", type=int, default=None, help="短地平线步数")
    parser.add_argument("--iter", type=int, default=None, help="每次重规划迭代数")
    parser.add_argument("--fix-joint5", action="store_true", help="固定第 6 关节")
    parser.add_argument("--backswing", type=float, default=0.6, help="后摆幅度 (rad)")
    parser.add_argument("--bs-ratio", type=float, default=0.35, help="后摆占比")
    parser.add_argument("--no-backswing", action="store_true", help="禁用后摆")
    parser.add_argument("--r-decay", type=float, default=0.40, help="R 衰减占比")
    parser.add_argument("--no-r-decay", action="store_true", help="禁用 R 退火")
    parser.add_argument("--hit-shift", type=float, default=0.01, help="随挥偏移距离 (m)")
    parser.add_argument("--ball-speed", type=float, default=None, help="球到达击打点时水平速度 (m/s)")
    parser.add_argument("--ball-distance", type=float, default=None, help="球起始位置到击打点的直线距离 (m)")
    parser.add_argument("--approach-angle", type=float, default=0.0, help="球飞来方向角 (度)，0=-Y方向，90=-X方向")
    parser.add_argument("--serve-box", action="store_true", help="使用长方体发球区模式（替代默认随机发球）")
    parser.add_argument("--no-bounce", action="store_true", help="禁用地面弹跳（默认弹跳打开）")
    parser.add_argument("--serve-distance", type=float, default=8.0, help="发球区 Y 方向距离 (m)")
    parser.add_argument("--serve-height", type=float, default=1.2, help="发球区中心高度 (m)")
    parser.add_argument("--serve-x-size", type=float, default=8.0, help="发球区 X 轴全长 (m)")
    parser.add_argument("--serve-y-size", type=float, default=0.2, help="发球区 Y 轴全长 (m)")
    parser.add_argument("--serve-z-size", type=float, default=0.3, help="发球区 Z 轴全长 (m)")
    parser.add_argument("--normal-weight", type=float, default=500000.0, help="拍面法向量代价权重")
    parser.add_argument("--normal-flip", action="store_true", help="翻转法向量方向")
    parser.add_argument("--replan-interval", type=int, default=None, help="重规划间隔步数")
    parser.add_argument("--window-ms", type=float, default=50.0, help="Tube 候选窗口半宽 (ms)")
    parser.add_argument("--tube-cost-ratio", type=float, default=0.3, help="Tube 代价占比 (0~1)")
    parser.add_argument("--softmin-beta", type=float, default=5.0,
                        help="终端 softmin 锐度 β (1~20)，越大越接近 hard-min")
    parser.add_argument("--no-softmin", action="store_true",
                        help="禁用多终端 softmin（退化为单点终端代价）")
    parser.add_argument("--no-plot", action="store_true", help="禁用 matplotlib 可视化")
    parser.add_argument("--dump-trajectory", type=str, default=None,
                        help="将轨迹数据保存到指定 pickle 文件路径")
    parser.add_argument("--realtime", action="store_true", help="模拟实时节奏（5ms/步），使异步重规划结果可被应用")
    parser.add_argument("--async-replan", action="store_true", help="启用异步重规划（后台线程 iLQR，主线程不阻塞）")
    parser.add_argument("--time-perturb-ms", type=float, default=0.0,
                        help="球到达时间预测扰动 (ms): 正值=MPC认为球早到，负值=认为球晚到")
    parser.add_argument("--space-perturb-m", type=float, default=0.0,
                        help="击打点空间偏移 (m): 对 p_hit 施加侧向偏移，测试 tube 空间走廊的鲁棒性")
    parser.add_argument("--perturb-alpha-min", type=float, default=0.0,
                        help="衰减扰动保底值 (0~1): alpha=max(alpha_min, 1-count/total), 0=衰减到0, 0.3=保留30%%残余偏差")
    parser.add_argument("--ball-speed-perturb-pct", type=float, default=0.0,
                        help="球速耦合扰动百分比 (%%): 实际发球速度=ball_speed*(1+pct/100), MPC规划仍用标称球速")
    parser.add_argument("--max-tcp", type=float, default=None,
                        help="TCP 线速度硬限制 (m/s), 默认从配置文件读取, 0=不限速")
    parser.add_argument("--terminal-exempt-steps", type=int, default=None,
                        help="终段 qdot/TCP 豁免步数 (0=全程硬限), 默认从配置文件读取")
    args = parser.parse_args()

    use_tube = args.use_tube.lower() in ("true", "1", "yes")
    use_analytical = not args.fd
    time_perturb_s = args.time_perturb_ms / 1000.0
    space_perturb_m = args.space_perturb_m
    perturb_alpha_min = args.perturb_alpha_min

    # 加载配置
    base_path = Path(__file__).resolve().parent.parent.parent / "configs"
    config_dict = load_config(base_path / "default.yaml")
    # v5: 叠加 v5 主动击打配置
    v5_config_path = base_path / "v5_active_hit.yaml"
    if v5_config_path.exists():
        v5_config = load_config(v5_config_path)
        config_dict = merge_configs(config_dict, v5_config)
    mpc_config_path = base_path / "mpc.yaml"
    if mpc_config_path.exists():
        mpc_config = load_config(mpc_config_path)
        config_dict = merge_configs(config_dict, mpc_config)

    dt = float(config_dict["sim"]["dt"])
    g = np.array(config_dict["ball"]["gravity"], dtype=np.float64)

    shoulder_pos = np.array([-0.1, -0.22693, 1.302645], dtype=np.float64)
    workspace_radius = 0.90

    mpc_cfg = config_dict.get("mpc", {})
    total_horizon = 200
    fixed_horizon = 40
    replan_interval = args.replan_interval if args.replan_interval is not None else 30
    max_iter_per_plan = 5
    Q_p_scale_far = 5.0
    Q_v_scale_far = 3.0
    Q_p_scale_near = 8.0
    Q_v_scale_near = 120.0
    first_plan_iters = 15
    near_plan_iters = 20

    if args.horizon is not None:
        fixed_horizon = args.horizon
    if args.iter is not None:
        max_iter_per_plan = args.iter
    if args.replan_interval is not None:
        replan_interval = args.replan_interval

    # serve-box 模式：球从 8m 远处飞来，需更长规划前瞻
    if args.serve_box:
        if args.horizon is None:
            fixed_horizon = 120
        if args.iter is None:
            max_iter_per_plan = 10
        if args.replan_interval is None:
            replan_interval = 20 if args.realtime else 10
        first_plan_iters = max(first_plan_iters, 30)
        total_horizon = max(total_horizon, 250)
        logger.info(f"serve-box auto params: horizon={fixed_horizon}, iter={max_iter_per_plan}, "
                     f"first_plan_iters={first_plan_iters}, total_horizon={total_horizon}, "
                     f"replan_interval={replan_interval}")

    init_q = np.array([-1.5, 1.57, -0.236, 0.404, 0.446, 2.45], dtype=np.float64)
    init_q_left = np.array([-0.373, -1.57, 0.236, -0.404, -0.446, -2.45], dtype=np.float64)

    fix_joint5_angle: float | None = init_q[5] if args.fix_joint5 else None
    use_backswing = not args.no_backswing
    backswing_offset = -abs(args.backswing)
    backswing_ratio = args.bs_ratio
    use_r_decay = not args.no_r_decay
    r_decay_ratio = args.r_decay

    # Tube 配置
    tube_cfg = TubeConfig(
        window_half_ms=args.window_ms,
        Q_p_tube=float(config_dict["cost"]["Q_p"][0]) * 2.0,
        Q_v_tube=float(config_dict["cost"]["Q_v"][0]) * 50.0,
        Q_n_tube=args.normal_weight,
        tube_cost_ratio=args.tube_cost_ratio,
        softmin_beta=args.softmin_beta,
        use_softmin_terminal=not args.no_softmin,
    )

    # 初始化 RM-65 环境
    model_path = Path(__file__).resolve().parent.parent.parent / "src" / "robot" / "rm65_model.xml"
    env = RM65Env(model_path, dt=dt)
    env.init_q_left = init_q_left

    # 加载真实机器人硬约束
    rl_cfg = config_dict.get("robot_limits", {})
    if args.max_tcp is not None:
        if args.max_tcp == 0:
            rl_cfg["max_tcp_speed"] = float("inf")
        else:
            rl_cfg["max_tcp_speed"] = args.max_tcp
    if args.terminal_exempt_steps is not None:
        rl_cfg["terminal_exempt_steps"] = args.terminal_exempt_steps
    robot_limits = RobotLimits.from_config(
        rl_cfg, dt=dt,
        ctrlrange=env.model.actuator_ctrlrange[:env.NU],
    )
    logger.info(
        "RobotLimits: q_lower=%.1f°(j0)/%.1f°(j3), qdot_max=%.1f°/s (scaled), "
        "qddot_max=%.1f°/s² (scaled), u_range=[%.1f, %.1f]Nm, "
        "max_tcp=%.1f m/s, exempt=%d步",
        float(robot_limits.q_lower[0] * 180 / np.pi),
        float(robot_limits.q_lower[3] * 180 / np.pi),
        float(np.max(robot_limits.qdot_max) * 180 / np.pi),
        float(np.max(robot_limits.qddot_max) * 180 / np.pi) if np.isfinite(robot_limits.qddot_max[0]) else float("inf"),
        float(robot_limits.u_min[0]), float(robot_limits.u_max[0]),
        robot_limits.max_tcp_speed if np.isfinite(robot_limits.max_tcp_speed) else -1.0,
        robot_limits.terminal_exempt_steps,
    )

    # 硬约束 body ID 缓存（初始化一次）
    import mujoco as _mj
    _hard_x_body_ids = [
        _mj.mj_name2id(env.model, _mj.mjtObj.mjOBJ_BODY, n)
        for n in ("r_link1", "r_link2", "r_link3", "r_link4",
                   "r_link5", "r_link6", "r_flange", "r_racket_body")
    ]

    x0 = np.zeros(env.NX)
    x0[:env.NQ] = init_q

    # ===== 生成发球轨迹 =====
    rng = np.random.default_rng(args.seed)
    hit_cfg = config_dict.get("hitting", {})

    env.reset(init_q)
    env.data.qpos[env.NQ:env.NQ + env.LEFT_ARM_NQ] = init_q_left
    env.data.qvel[env.NQ:env.NQ + env.LEFT_ARM_NQ] = 0.0
    env.update_kinematics()
    p_racket_init = env.get_ee_pos().copy()
    logger.info(f"球拍初始位置: {p_racket_init}")

    target_center = np.array([-0.82765693, -0.47411682, 0.86947444])
    target_offset = 0.10

    use_bounce = not args.no_bounce  # 默认弹跳打开

    if args.serve_box:
        # ── 长方体发球区模式 ──
        serve_half_x = args.serve_x_size / 2.0
        serve_half_y = args.serve_y_size / 2.0
        serve_half_z = args.serve_z_size / 2.0
        logger.info(
            f"发球区模式: center=(0,{-args.serve_distance},{args.serve_height}), "
            f"half=({serve_half_x},{serve_half_y},{serve_half_z}), "
            f"bounce={'ON' if use_bounce else 'OFF'}"
        )
        p0, v0, p_hit_expected = generate_ball_from_serve_box(
            serve_box_center=(0.0, -args.serve_distance, args.serve_height),
            serve_box_halfsize=(serve_half_x, serve_half_y, serve_half_z),
            target_center=target_center,
            target_offset=target_offset,
            shoulder_pos=shoulder_pos,
            workspace_radius=workspace_radius,
            g=g,
            ball_speed=args.ball_speed,
            speed_range=(8.0, 18.0),
            use_bounce=use_bounce,
            bounce_restitution=0.75,
            rng=rng,
        )
        implied_speed = float(np.linalg.norm(v0[:2]))
        logger.info(
            f"生成发球(serve_box): 起始={np.round(p0, 2)}, "
            f"初速={np.round(v0, 1)}, 水平球速≈{implied_speed:.1f}m/s, "
            f"期望击打点={np.round(p_hit_expected, 2)}, bounce={'ON' if use_bounce else 'OFF'}"
        )
    else:
        hit_time = total_horizon * dt * rng.uniform(0.3, 0.4)
        p0, v0, p_hit_expected = generate_ball_to_target_box(
            target_center, target_offset, hit_time, g,
            shoulder_pos=shoulder_pos, workspace_radius=workspace_radius,
            ball_speed=args.ball_speed,
            ball_distance=args.ball_distance,
            approach_angle_deg=args.approach_angle,
            rng=rng,
            ball_direction="y",
            ball_start_y_range=(-5.5, -4.5),
            ball_start_z_range=(1.4, 1.8),
        )
        if args.ball_distance is not None:
            implied_speed = np.linalg.norm(v0[:2])
            logger.info(
                f"生成发球: 距离={args.ball_distance:.1f}m, 角度={args.approach_angle}°, "
                f"初始位置={np.round(p0, 2)}, 初始速度={np.round(v0, 1)}, "
                f"水平球速≈{implied_speed:.1f}m/s, 期望击打点={np.round(p_hit_expected, 2)}"
            )
        else:
            logger.info(f"生成发球: 初始位置={p0}, 初始速度={v0}, 期望击打点={p_hit_expected}")

    # ===== 寻找击打点（基于预测轨迹） =====
    hit_info = find_hitting_point_physics(
        env, p0, v0, shoulder_pos, workspace_radius, total_horizon
    )
    if hit_info is None:
        print("\n========================================")
        print("  网球不在工作空间内，机械臂不击打！")
        print("========================================\n")
        return

    k_hit_total = hit_info["k_hit"]
    p_hit = hit_info["p_hit"]
    v_ball_hit = hit_info["v_ball_hit"]

    # IK 可达性后过滤：若初始击球点关节超限，搜索附近可行点
    ball_positions_all, ball_velocities_all = env.predict_ball_trajectory(p0, v0, total_horizon)
    q_ik_init = env.solve_ik(p_hit, q_init=init_q, max_iter=50, eps=1e-2)
    m_low_deg = (q_ik_init - robot_limits.q_lower) * 180.0 / np.pi
    m_up_deg = (robot_limits.q_upper - q_ik_init) * 180.0 / np.pi
    min_margin_deg = float(np.min(np.minimum(m_low_deg, m_up_deg)))
    if min_margin_deg < 3.0:
        logger.warning(
            "初始击球点 IK 超限 (min_margin=%.1f°)，搜索可行替代点",
            min_margin_deg,
        )
        search_range = 30
        best_alt_k = k_hit_total
        best_alt_margin = min_margin_deg
        for dk in range(-search_range, search_range + 1):
            kk = k_hit_total + dk
            if kk < 1 or kk > len(ball_positions_all):
                continue
            p_alt = ball_positions_all[kk - 1]
            dist_alt = np.linalg.norm(p_alt - shoulder_pos)
            dz = p_alt[2] - shoulder_pos[2]
            if not (dist_alt < workspace_radius and p_alt[2] > 0.3 and -0.60 < dz < 0.55):
                continue
            q_alt = env.solve_ik(p_alt, q_init=init_q, max_iter=30, eps=2e-2)
            m_low_a = (q_alt - robot_limits.q_lower) * 180.0 / np.pi
            m_up_a = (robot_limits.q_upper - q_alt) * 180.0 / np.pi
            m_a = float(np.min(np.minimum(m_low_a, m_up_a)))
            if m_a > best_alt_margin:
                best_alt_margin = m_a
                best_alt_k = kk
        if best_alt_k != k_hit_total:
            p_hit = ball_positions_all[best_alt_k - 1].copy()
            v_ball_hit = ball_velocities_all[best_alt_k - 1].copy()
            old_k = k_hit_total
            k_hit_total = best_alt_k
            logger.info(
                "击球点修正: k %d→%d, min_margin %.1f°→%.1f°",
                old_k, best_alt_k, min_margin_deg, best_alt_margin,
            )

    if use_backswing:
        p_ee_init = env.get_ee_pos()
        dist_to_ball = np.linalg.norm(p_hit - p_ee_init)
        bs_scale = np.clip((dist_to_ball - 0.8) / (1.5 - 0.8), 0.0, 1.0)
        adaptive_bs = 0.4 + bs_scale * 0.6
        backswing_offset = -adaptive_bs
        logger.info(f"自适应后摆: dist={dist_to_ball:.3f}m, backswing={adaptive_bs:.2f}rad")

    n_des_single = -v_ball_hit / (np.linalg.norm(v_ball_hit) + 1e-8)
    if args.normal_flip:
        n_des_single = -n_des_single

    near_threshold = max(50, k_hit_total // 3)
    far_threshold = 50  # 实时模式：远段用 JT 控制（0ms），仅近段用 iLQR

    hit_direction = np.array(config_dict["hitting"]["hit_direction"], dtype=np.float64)
    racket_speed = float(config_dict["hitting"]["racket_speed"])

    # ===== v4: 随挥方向 = 来球反方向 =====
    d_follow = -v_ball_hit / (np.linalg.norm(v_ball_hit) + 1e-8)
    d_hat = d_follow  # v4: 用来球反方向替代固定 hit_direction

    # v4: 击球时刻期望速度 = TCP 限制
    v_hit_at_contact = racket_speed * d_follow  # 1.8 m/s 沿来球反方向

    # v4: 随挥管道参数
    follow_through_length = float(config_dict["hitting"].get("follow_through_length", 0.4))
    follow_through_steps = int(config_dict["hitting"].get("follow_through_steps", 40))
    follow_through_v_terminal = float(config_dict["hitting"].get("follow_through_v_terminal", 0.3))

    # v4: 终端目标点 = 击球点 + 小偏移（与 v2 相同）
    p_follow = p_hit + 0.01 * d_follow
    # v4: 终端速度 = 击球时刻期望速度（1.8 m/s），iLQR 在终端（=击打时刻）最大化此速度
    v_hit_desired = v_hit_at_contact  # 1.8 m/s 沿来球反方向

    hit_shift = 0.01  # v4: 与 v2 相同，小偏移

    # v4: 随挥管道参数（击打后 PD 控制）
    follow_through_length = float(config_dict["hitting"].get("follow_through_length", 0.4))
    follow_through_steps = int(config_dict["hitting"].get("follow_through_steps", 40))
    follow_through_v_terminal = float(config_dict["hitting"].get("follow_through_v_terminal", 0.3))

    # ===== 构建初始 Tube（若启用） =====
    hit_window: HitWindow | None = None
    hitting_tube: HittingTube | None = None
    if use_tube:
        hit_window = search_hit_window(
            env, p0, v0, shoulder_pos, workspace_radius,
            k_hit_total + 30, tube_cfg,
            ball_direction="y",
            current_step=0,
            robot_limits=robot_limits,
            init_q=init_q,
        )
        if hit_window is not None:
            hitting_tube = build_hitting_tube(
                hit_window, racket_speed, d_follow, tube_cfg,
            )
            logger.info(
                f"Tube 已构建: best_k={hitting_tube.best_k}, "
                f"candidates={len(hitting_tube.k_candidates)} "
                f"[{hitting_tube.k_candidates[0]}..{hitting_tube.k_candidates[-1]}]"
            )
        else:
            logger.warning("Tube 构建失败，fallback 到 single-hit-point 模式")
            use_tube = False

    logger.info(f"击打步数: {k_hit_total}, 击打位置: {p_hit}")
    logger.info(f"[v5] 随挥方向(来球反方向): {np.round(d_follow, 3)}")
    logger.info(f"[v5] 击球时刻期望速度: {racket_speed} m/s")
    logger.info(f"[v5] 终端目标点: {np.round(p_follow, 3)}, 终端速度: {np.linalg.norm(v_hit_desired):.1f} m/s")
    logger.info(f"[v5] 随挥距离: {follow_through_length:.2f}m, 随挥步数: {follow_through_steps}")
    logger.info(f"Tube 模式: {'启用' if use_tube else '禁用'}")
    logger.info(f"线性化: {'解析' if use_analytical else '有限差分'}, horizon={fixed_horizon}")

    # ===== 初始化 =====
    Q_p = np.array(config_dict["cost"]["Q_p"], dtype=np.float64) * 2.0
    # v4: 方向对齐速度代价 — 只惩罚沿 d_follow 方向的速度偏差
    # Q_v_scalar 远大于原始 Q_v，但只作用在 1 个方向上，等效应力 ≈ Q_v_scalar
    Q_v_scalar = 10000.0  # v4: 方向对齐速度代价标量（远大于原始 400）
    Q_v = Q_v_scalar * np.outer(d_follow, d_follow)  # (3,3) 秩1矩阵
    logger.info(f"[v5] Q_v 方向对齐: scalar={Q_v_scalar}, d_follow={np.round(d_follow, 3)}")
    logger.info(f"[v5] 终端 = 随挥终点: p_terminal={np.round(p_hit + follow_through_length * d_follow, 3)}, "
                f"v_terminal={follow_through_v_terminal:.1f} m/s, horizon=+{follow_through_steps}步")
    R = float(config_dict["cost"]["R"])
    ilqt_cfg = dict(config_dict["ilqt"])

    p_target_init = p_follow

    if use_backswing:
        U_prev, q_des_traj_init = generate_backswing_warm_start(
            env, x0, p_target_init, v_hit_at_contact, k_hit_total,
            backswing_offset=backswing_offset,
            backswing_ratio=backswing_ratio,
            fix_joint5_angle=fix_joint5_angle,
            n_des=n_des_single,
        )
        logger.info(
            f"已生成后摆 Warm-start: offset={backswing_offset:.2f}rad, "
            f"ratio={backswing_ratio:.1%}"
        )
    else:
        U_prev = compute_jacobian_init_control(
            env, x0, p_target_init, k_hit_total, gain=60.0,
            fix_joint5_angle=fix_joint5_angle,
        )
        q_des_traj_init = None
        logger.info("已计算雅可比转置初始控制序列")

    r_joint_scale: dict[int, float] = {}
    if use_backswing:
        r_joint_scale[0] = 0.3
    if fix_joint5_angle is not None:
        r_joint_scale[5] = 1000.0

    Q_joint: dict[int, float] | None = None

    R_schedule_init = (
        compute_r_schedule(k_hit_total, R, decay_ratio=r_decay_ratio)
        if use_r_decay else None
    )

    # 创建基础代价函数（无软约束，硬约束在执行层处理）
    base_cost_fn = HittingCost(
        env, p_follow, v_hit_desired, Q_p, Q_v, R,
        Q_p_running=0.0,
        R_joint_scale=r_joint_scale if r_joint_scale else None,
        q_des_traj=q_des_traj_init,
        Q_joint=Q_joint,
        R_schedule=R_schedule_init,
        Q_n=args.normal_weight,
        n_des=n_des_single,
        Q_qdot=float(config_dict["cost"].get("Q_qdot", 0.0)),
        Q_qddot=float(config_dict["cost"].get("Q_qddot", 0.0)),
        Q_du=float(config_dict["cost"].get("Q_du", 0.0)),
    )

    # v5: 初始中途目标 — 在 k_hit 步强制经过击球位置 + 鼓励高速
    base_cost_fn.set_midpoint_target(k_hit_total, p_hit, Q_p * 2.0,
                                     v_target=v_hit_at_contact,
                                     Q_midpoint_v=Q_v * 5.0)

    # 创建 Tube 代价包装器（若启用）
    tube_cost_fn: TubeHittingCostWrapper | None = None
    if use_tube and hitting_tube is not None:
        tube_cost_fn = TubeHittingCostWrapper(
            env, base_cost_fn, hitting_tube, k_hit_total, tube_cfg,
        )
        cost_fn = tube_cost_fn
        logger.info("TubeHittingCostWrapper 已创建")
    else:
        cost_fn = base_cost_fn  # type: ignore[assignment]
        logger.info("使用标准 HittingCost（single-hit-point）")

    # ===== 球速耦合扰动：沿飞行方向平移发球位置，保持初速度不变 =====
    # 正 pct: 发球点沿飞行方向前移 → 球提前到达 → 时间+空间耦合偏移
    # 负 pct: 发球点沿飞行方向后退 → 球延迟到达
    ball_speed_perturb_pct = getattr(args, 'ball_speed_perturb_pct', 0.0)
    if abs(ball_speed_perturb_pct) > 0.01:
        v0_norm = np.linalg.norm(v0)
        if v0_norm > 0.1:
            v0_dir = v0 / v0_norm
            offset_m = ball_speed_perturb_pct / 100.0 * 2.0  # ±10% → ±2.0m 位移
            p0_real = p0 + v0_dir * offset_m
            v0_real = v0.copy()
            logger.info(
                f"发球位置耦合扰动: offset={offset_m:+.3f}m 沿飞行方向, "
                f"p0变化={np.round(p0_real - p0, 3)}"
            )
        else:
            p0_real = p0
            v0_real = v0
    else:
        p0_real = p0
        v0_real = v0

    solver = ILQTSolver(ilqt_cfg, use_analytical=use_analytical)

    env.reset(init_q)
    env.data.qpos[env.NQ:env.NQ + env.LEFT_ARM_NQ] = init_q_left
    env.data.qvel[env.NQ:env.NQ + env.LEFT_ARM_NQ] = 0.0
    env.update_kinematics()
    env.set_ball_state(p0_real, v0_real)
    if abs(time_perturb_s) > 1e-6:
        logger.info(f"球时间预测扰动: {args.time_perturb_ms:+.1f}ms "
                     "(仅影响 MPC 预测，球实际轨迹不变)")
    if abs(space_perturb_m) > 1e-6:
        logger.info(f"击打目标空间偏移: {space_perturb_m:+.3f}m")

    # ===== MPC 主循环 =====
    x_current = x0.copy()
    X_history = [x0.copy()]
    U_history: list[np.ndarray] = []
    ball_pos_history = [p0.copy()]
    cost_history: list[float] = []
    pos_error_history: list[float] = []

    # 衰减式扰动：记录重规划次数，计算衰减系数
    replan_count = 0
    if k_hit_total > 0:
        total_expected_replans = max(1, k_hit_total // replan_interval)
    else:
        total_expected_replans = 1

    # Tube 专用记录
    distances_history: list[float] = []
    normal_align_history: list[float] = []
    ball_near_history: list[bool] = []
    tube_ready_history: list[bool] = []
    valid_hit_history: list[bool] = []
    k_hit_steps_history: list[int] = []
    # 保存初始 tube 信息（用于最终评估显示和 tube_ready 判定）
    initial_tube_n_candidates = 0
    initial_tube_k_range = ""
    initial_hitting_tube: HittingTube | None = None
    if hitting_tube is not None:
        initial_tube_n_candidates = len(hitting_tube.k_candidates)
        initial_tube_k_range = f"[{hitting_tube.k_candidates[0]}..{hitting_tube.k_candidates[-1]}]"
        initial_hitting_tube = hitting_tube

    t_total_start = time.perf_counter()
    step_times: list[float] = []
    replan_times: list[float] = []
    step_is_replan: list[bool] = []
    replan_k_hit_history: list[int] = []

    U_buffer: np.ndarray = np.zeros((0, env.NU))
    buffer_idx: int = 0
    is_first_plan: bool = True
    p_hit_new = p_hit.copy()
    k_hit_new = k_hit_total
    iters = 0
    hit_step = -1
    p_ee_at_hit = None
    total_sleep_time = 0.0
    ball_pos_at_hit = None
    q_ik_cache: np.ndarray | None = None
    ball_was_hit = False
    follow_through_start = -1  # v4: 随挥开始步

    buffer_exhaustion_count = 0
    current_n_des = n_des_single.copy()
    U_prev = np.zeros((0, env.NU))
    current_n_des = n_des_single
    v_ball_hit_new = v_ball_hit.copy()

    # ==========================================================================
    # 击球点可执行性后过滤（本地函数，在 MPC replan 中调用）
    # ==========================================================================
    hit_lock_active = False            # 防抖锁定：k_hit ≤ 60 后不再换点
    hit_lock_threshold = 60            # 锁定阈值步数
    last_p_hit: np.ndarray | None = None  # 用于 hysteresis 比较

    def refine_hit_point(
        p_hit: np.ndarray,
        k_hit: int,
        remaining: int,
        env_local: RM65Env,
    ) -> tuple[np.ndarray, int, str]:
        """对 find_hitting_point_physics 返回的击球点做可执行性后过滤。

        在 tube 候选窗口内搜索关节裕度更充足的替代点。
        仅在 MPC replan 段调用，不修改 hitting.py。

        Returns:
            (p_hit_refined, k_hit_refined, log_message)
        """
        nonlocal hit_lock_active, last_p_hit

        # 防抖锁定：末段不再换点
        if k_hit <= hit_lock_threshold:
            if not hit_lock_active:
                hit_lock_active = True
                logger.info("[HIT_LOCK] k_hit=%d ≤ %d, 锁定击球点不再替换", k_hit, hit_lock_threshold)
            return p_hit, k_hit, "locked"

        # 快速 IK 检查当前点的双边裕度
        q_ik = env_local.solve_ik(p_hit, q_init=x_current[:env_local.NQ], max_iter=50, eps=1e-2)
        margin_lower_deg = (q_ik - robot_limits.q_lower) * 180.0 / np.pi
        margin_upper_deg = (robot_limits.q_upper - q_ik) * 180.0 / np.pi
        margin_min_deg = float(np.min(np.minimum(margin_lower_deg, margin_upper_deg)))
        margin_j1_deg = float(min(margin_lower_deg[1], margin_upper_deg[1]))

        # 分级判定
        hard_margin_deg = 2.0       # 低于此 → 高风险，必须尝试替换
        warn_margin_deg = 5.0       # 低于此 → 中风险，记录警告
        j1_warn_margin_deg = 8.0    # J1 单独更严格

        high_risk = (margin_min_deg < hard_margin_deg)
        j1_near = (margin_j1_deg < j1_warn_margin_deg)
        medium_risk = (margin_min_deg < warn_margin_deg)

        if not high_risk and not j1_near:
            if medium_risk:
                logger.info(
                    "[HIT_KEEP] p=%s min_margin=%.1f° j1=%.1f° → feasible (medium risk)",
                    np.round(p_hit, 3), margin_min_deg, margin_j1_deg,
                )
            else:
                logger.debug(
                    "[HIT_KEEP] p=%s min_margin=%.1f° j1=%.1f° → feasible",
                    np.round(p_hit, 3), margin_min_deg, margin_j1_deg,
                )
            return p_hit, k_hit, "feasible"

        # 高风险：先尝试微调位置偏移（保持同一时间点），再搜索 tube 窗口
        logger.warning(
            "[HIT_RISK] p=%s k=%d min_margin=%.1f° j1=%.1f° → searching alternatives",
            np.round(p_hit, 3), k_hit, margin_min_deg, margin_j1_deg,
        )

        # 策略1：微调位置偏移（保持时间不变，偏移3-8cm获取关节裕度）
        best_candidate: tuple | None = None
        best_score = -1e9

        if j1_near:
            j1_dir = 1.0 if margin_j1_deg == margin_lower_deg[1] else -1.0
            for offset_cm in [3, 5, 8, 12]:
                offset_m = offset_cm / 100.0
                p_shifted = p_hit.copy()
                p_shifted[1] += j1_dir * offset_m
                dist_s = np.linalg.norm(p_shifted - shoulder_pos)
                if dist_s > workspace_radius or p_shifted[2] < 0.3:
                    continue
                q_s = env_local.solve_ik(
                    p_shifted, q_init=x_current[:env_local.NQ], max_iter=30, eps=2e-2,
                )
                m_low_s = (q_s - robot_limits.q_lower) * 180.0 / np.pi
                m_up_s = (robot_limits.q_upper - q_s) * 180.0 / np.pi
                m_min_s = float(np.min(np.minimum(m_low_s, m_up_s)))
                m_j1_s = float(min(m_low_s[1], m_up_s[1]))
                if m_min_s < margin_min_deg - 0.5:
                    continue
                if m_j1_s < j1_warn_margin_deg:
                    continue
                score_s = (
                    2.0 * m_min_s
                    + 3.0 * m_j1_s
                    - 50.0 * np.linalg.norm(p_shifted - p_hit)
                )
                if score_s > best_score:
                    best_score = score_s
                    best_candidate = (p_shifted.copy(), k_hit, m_min_s, m_j1_s)

        # 策略2：tube 窗口搜索（改变时间点，仅当策略1无法满足时）
        ball_positions_pred, _ = env_local.predict_ball_trajectory(
            env_local.get_ball_pos(), env_local.get_ball_vel(),
            min(remaining + 30, 300),
        )
        window_half_steps = 15
        k_min = max(1, k_hit - window_half_steps)
        k_max = min(len(ball_positions_pred), k_hit + window_half_steps)

        for k_cand in range(k_min, k_max + 1):
            if k_cand == k_hit:
                continue  # 策略1已处理同时间点
            p_cand = ball_positions_pred[k_cand - 1]
            dist_cand = np.linalg.norm(p_cand - shoulder_pos)
            if dist_cand > workspace_radius * 1.1 or p_cand[2] < 0.3:
                continue

            q_cand = env_local.solve_ik(
                p_cand, q_init=x_current[:env_local.NQ], max_iter=30, eps=2e-2,
            )
            m_low = (q_cand - robot_limits.q_lower) * 180.0 / np.pi
            m_up = (robot_limits.q_upper - q_cand) * 180.0 / np.pi
            m_min = float(np.min(np.minimum(m_low, m_up)))
            m_j1 = float(min(m_low[1], m_up[1]))

            if m_min < margin_min_deg - 0.5:
                continue

            y_risk = max(0.0, (shoulder_pos[1] - 0.40) - p_cand[1])

            score = (
                2.0 * m_min
                + 3.0 * m_j1
                - 1.0 * abs(k_cand - k_hit)
                - 30.0 * np.linalg.norm(p_cand - p_hit)
                - 10.0 * y_risk
            )

            if score > best_score:
                best_score = score
                best_candidate = (p_cand.copy(), k_cand, m_min, m_j1)

        if best_candidate is not None:
            p_new, k_new, m_min_new, m_j1_new = best_candidate
            # hysteresis: 新点需显著优于旧点
            score_original = (2.0 * margin_min_deg + 3.0 * margin_j1_deg)
            if best_score > score_original + 10.0:
                logger.warning(
                    "[HIT_SWAP] reason=%s, k %d→%d, "
                    "min_margin %.1f°→%.1f°, j1 %.1f°→%.1f°",
                    "j1_near" if j1_near else "high_risk",
                    k_hit, k_new,
                    margin_min_deg, m_min_new,
                    margin_j1_deg, m_j1_new,
                )
                last_p_hit = p_new.copy()
                return p_new, k_new, "swapped"
            else:
                logger.info(
                    "[HIT_RISK] no significantly better candidate, "
                    "best_score=%.1f vs original=%.1f",
                    best_score, score_original,
                )

        logger.warning(
            "[HIT_RISK] min_margin=%.1f° j1=%.1f°, "
            "no safer candidate found in local window (±%d steps)",
            margin_min_deg, margin_j1_deg, window_half_steps,
        )
        return p_hit, k_hit, "risk_kept"

    # ==========================================================================
    # MPC 主循环
    # ==========================================================================

    # 执行层指标累积
    exec_metrics = ExecutionMetrics()
    terminal_fallback_count = 0       # k_hit ≤ 20 时的 solver 失败次数
    active_contact = False            # 主动击球判定 (racket_speed > 2m/s at contact)
    passive_contact = False           # 被动接触（球撞拍）
    # 分阶段权重配置
    stage_cfg = config_dict.get("stage_weights", {})
    far_stage = stage_cfg.get("far", {"Q_qdot_mult": 1.0, "Q_qddot_mult": 1.0, "Q_du_mult": 1.0})
    mid_stage = stage_cfg.get("mid", {"Q_qdot_mult": 2.0, "Q_qddot_mult": 2.0, "Q_du_mult": 2.0})
    near_stage = stage_cfg.get("near", {"Q_qdot_mult": 2.0, "Q_qddot_mult": 2.0, "Q_du_mult": 3.0})

    # ---- 异步重规划状态与配置 ----
    replan_state = ReplanState(
        k_hit_new=k_hit_total,
        p_hit_new=p_hit.copy(),
        v_ball_hit_new=v_ball_hit.copy(),
        current_n_des=n_des_single.copy(),
        U_prev=np.zeros((0, env.NU)),
        is_first_plan=True,
    )
    replan_cfg: dict = {
        "total_horizon": total_horizon,
        "fixed_horizon": fixed_horizon,
        "replan_interval": replan_interval,
        "max_iter_per_plan": max_iter_per_plan,
        "first_plan_iters": first_plan_iters,
        "near_plan_iters": near_plan_iters,
        "near_threshold": near_threshold,
        "dt": dt,
        "shoulder_pos": shoulder_pos,
        "workspace_radius": workspace_radius,
        "robot_limits": robot_limits,
        "solver": solver,
        "R": R,
        "Q_p_scale_far": Q_p_scale_far,
        "Q_v_scale_far": Q_v_scale_far,
        "Q_p_scale_near": Q_p_scale_near,
        "Q_v_scale_near": Q_v_scale_near,
        "hit_shift": hit_shift,
        "d_hat": d_hat,
        "v_hit_desired": v_hit_desired,
        "v_hit_at_contact": v_hit_at_contact,
        "d_follow": d_follow,
        "Q_v_scalar": Q_v_scalar,
        "follow_through_length": follow_through_length,
        "follow_through_steps": follow_through_steps,
        "follow_through_v_terminal": follow_through_v_terminal,
        "use_backswing": use_backswing,
        "use_r_decay": use_r_decay,
        "r_decay_ratio": r_decay_ratio,
        "time_perturb_s": time_perturb_s,
        "space_perturb_m": space_perturb_m,
        "perturb_alpha_min": perturb_alpha_min,
        "normal_flip": args.normal_flip,
        "fix_joint5_angle": fix_joint5_angle,
        "backswing_offset": backswing_offset,
        "backswing_ratio": backswing_ratio,
        "k_hit_total": k_hit_total,
        "smooth_far": far_stage,
        "smooth_mid": mid_stage,
        "smooth_near": near_stage,
    }
    async_replanner = AsyncReplanner(env, do_replan, replan_cfg, state=replan_state, model_path=model_path)
    async_replanner.start()
    # 确保 env_plan 已创建（用于同步首次规划）
    _ = async_replanner._ensure_env_plan()

    async_mode = args.async_replan

    # 首次规划同步完成（离线阶段）
    t_first_start = time.perf_counter()
    ball_pos_init, ball_vel_init = env.get_ball_state()
    first_request = PlanRequest(
        x_current=x_current.copy(),
        ball_pos=ball_pos_init,
        ball_vel=ball_vel_init,
        step=0,
        k_hit_current=k_hit_total,
        U_prev=np.zeros((0, env.NU)),
        p_hit_current=p_hit.copy(),
        v_hit_desired=v_hit_at_contact,  # v4: 用击球速度（1.8 m/s）做初始控制参考
        n_des_current=n_des_single.copy(),
        is_first_plan=True,
    )
    first_result = do_replan(first_request, async_replanner.env_plan, replan_state, replan_cfg)
    t_first_dur = time.perf_counter() - t_first_start
    replan_times.append(t_first_dur)
    replan_k_hit_history.append(first_result.k_hit_new)
    logger.info(
        f"REPLAN step=0 k_hit={first_result.k_hit_new} iters={first_result.iters_plan} "
        f"horizon={first_result.horizon_plan} t={t_first_dur*1000:.0f}ms "
        f"fp_limits={not first_result.fp_limits_was_none} fast_lin={first_result.fast_lin}"
    )
    replan_state.is_first_plan = False
    replan_state.k_hit_new = first_result.k_hit_new
    replan_state.p_hit_new = first_result.p_hit_new.copy()
    replan_state.v_ball_hit_new = first_result.v_ball_hit_new.copy()
    replan_state.current_n_des = first_result.n_des_new.copy()
    replan_state.U_prev = first_result.U_prev.copy()

    # 从首次规划结果覆盖控制缓冲
    U_buffer = first_result.U_buffer.copy()
    buffer_idx = 0
    async_replan_submitted = False

    # 状态变量（异步模式陈旧规划调整用）
    k_hit_new = first_result.k_hit_new
    p_hit_new = first_result.p_hit_new.copy()
    v_ball_hit_new = first_result.v_ball_hit_new.copy()
    current_n_des = first_result.n_des_new.copy()
    p_follow_new = p_hit_new + hit_shift * d_hat
    n_des_new = current_n_des.copy()
    U_prev = first_result.U_prev.copy()

    # 立即提交第一个异步重规划请求
    if async_mode:
        first_async_request = PlanRequest(
            x_current=x_current.copy(),
            ball_pos=ball_pos_init,
            ball_vel=ball_vel_init,
            step=0,
            k_hit_current=k_hit_new,
            U_prev=U_prev.copy(),
            p_hit_current=p_hit_new.copy(),
            v_hit_desired=v_hit_at_contact,  # v4: 初始控制参考用击球速度
            n_des_current=current_n_des.copy(),
            is_first_plan=False,
        )
        async_replanner.submit(first_async_request)
        async_replan_submitted = True

    total_horizon += follow_through_steps  # v4: 扩展以包含随挥段
    logger.info(f"开始 MPC 循环，总步数={total_horizon}，击打步数={k_hit_total}，随挥步数={follow_through_steps}")

    for step in range(total_horizon):
        t_step_start = time.perf_counter()

        ball_pos, ball_vel = env.get_ball_state()
        env.update_kinematics()

        # 记录 tube 诊断数据
        p_ee_cur = env.get_ee_pos()
        n_rack_cur = env.get_ee_normal()
        dist_cur = np.linalg.norm(p_ee_cur - ball_pos)
        distances_history.append(dist_cur)

        if hitting_tube is not None and len(hitting_tube.n_racket_des) > 0:
            n_des_cur = hitting_tube.n_racket_des[0]  # 用第一个候选
        else:
            n_des_cur = current_n_des
        n_align = float(n_rack_cur @ n_des_cur)
        normal_align_history.append(n_align)

        # 有效击球检测（拆分为两个独立指标）：
        # (a) ball_near：球物理上在球拍附近（纯碰撞检测，不依赖 tube）
        is_ball_near = (dist_cur < 0.033 + 0.12 + 0.03) and (abs(n_align) > 0.7)
        ball_near_history.append(is_ball_near)

        # (b) tube_ready：球拍在空间走廊内保持击球姿态
        #     当前仿真步落在 tube 时间窗口内，且球拍垂直偏离球轨迹线 < 15cm，法向对齐 > 0.7
        is_tube_ready = False
        if initial_hitting_tube is not None and len(initial_hitting_tube.p_ball) > 0:
            window_half_steps = int(round(tube_cfg.window_half_ms / 1000.0 / dt))
            tube_center = initial_hitting_tube.best_k
            if abs(step - tube_center) <= window_half_steps:
                # 计算球轨迹线方向和垂直投影
                v_ball_mean = np.mean(initial_hitting_tube.v_ball, axis=0)
                v_norm = np.linalg.norm(v_ball_mean)
                if v_norm > 1e-6:
                    d_ball = v_ball_mean / v_norm
                else:
                    d_ball = np.array([0.0, -1.0, 0.0])
                P_perp = np.eye(3) - np.outer(d_ball, d_ball)
                best_idx = int(np.argmin(np.abs(initial_hitting_tube.k_candidates - tube_center)))
                p_ref = initial_hitting_tube.p_ball[best_idx]
                dp = p_ee_cur - p_ref
                perp_dist = float(np.linalg.norm(P_perp @ dp))
                if perp_dist < 0.15 and abs(n_align) > 0.7:
                    is_tube_ready = True
        tube_ready_history.append(is_tube_ready)

        valid_hit_history.append(is_ball_near or is_tube_ready)

        need_replan = (step % replan_interval == 0) or (step == 0) or (buffer_idx >= len(U_buffer))

        # ---- 异步模式：检查结果 / 提交请求 ----
        if async_mode:
            # 检查已有异步结果
            if async_replanner.has_new_plan():
                result = async_replanner.apply_new_plan()
                if result is not None and result.request_step >= 0 and result.k_hit_new > 0:
                    async_replan_submitted = False
                    elapsed = step - result.request_step
                    if elapsed < len(result.U_mpc_full) and elapsed < result.k_hit_new:
                        U_shifted = result.U_mpc_full[elapsed:]
                        k_hit_adjusted = max(1, result.k_hit_new - elapsed)
                        if len(U_shifted) >= replan_interval:
                            if len(U_shifted) >= replan_interval * 6:
                                U_buffer = U_shifted[:replan_interval * 6]
                            elif len(U_shifted) >= replan_interval * 4:
                                U_buffer = U_shifted[:replan_interval * 4]
                            elif len(U_shifted) >= replan_interval * 2:
                                U_buffer = U_shifted[:replan_interval * 2]
                            else:
                                U_buffer = U_shifted[:replan_interval]
                            buffer_idx = 0
                            k_hit_new = k_hit_adjusted
                            p_hit_new = result.p_hit_new.copy()
                            v_ball_hit_new = result.v_ball_hit_new.copy()
                            current_n_des = result.n_des_new.copy()
                            n_des_new = current_n_des.copy()
                            p_follow_new = p_hit_new + hit_shift * d_hat
                            p_terminal_async = p_hit_new + follow_through_length * d_hat
                            v_terminal_async = follow_through_v_terminal * d_hat
                            base_cost_fn.update_target(p_terminal_async, v_terminal_async, n_des=n_des_new)
                            replan_state.k_hit_new = k_hit_new
                            replan_state.p_hit_new = p_hit_new.copy()
                            replan_state.v_ball_hit_new = v_ball_hit_new.copy()
                            replan_state.current_n_des = current_n_des.copy()
                            replan_state.U_prev = result.U_prev.copy()
                            replan_times.append(result.plan_duration_ms / 1000.0)
                            replan_k_hit_history.append(result.k_hit_new)
                            logger.info(
                                f"ASYNC_APPLY step={step} result_k_hit={result.k_hit_new} "
                                f"elapsed={elapsed} k_hit_adj={k_hit_adjusted} "
                                f"t={result.plan_duration_ms:.0f}ms"
                            )
                        else:
                            logger.warning(
                                f"ASYNC_DISCARD step={step}: 结果陈旧 "
                                f"elapsed={elapsed}/{len(result.U_mpc_full)}"
                            )
                    else:
                        logger.warning(
                            f"ASYNC_DISCARD step={step}: 结果过于陈旧 "
                            f"elapsed={elapsed} > {len(result.U_mpc_full)}"
                        )

            # 提交新异步请求
            can_submit = (not async_replan_submitted and not async_replanner.is_planning()
                          and step > 0)
            if need_replan and can_submit:
                request = PlanRequest(
                    x_current=x_current.copy(),
                    ball_pos=ball_pos.copy(),
                    ball_vel=ball_vel.copy(),
                    step=step,
                    k_hit_current=k_hit_new,
                    U_prev=U_prev.copy() if len(U_prev) > 0 else np.zeros((0, env.NU)),
                    p_hit_current=p_hit_new.copy(),
                    v_hit_desired=v_hit_desired,
                    n_des_current=current_n_des.copy(),
                    is_first_plan=False,
                )
                if async_replanner.submit(request):
                    async_replan_submitted = True

        # ---- 同步模式：阻塞执行重规划 ----
        if need_replan:
            t_replan_start = time.perf_counter()
            remaining_horizon = total_horizon - step

            # 始终使用实际球观测进行规划（tube 走廊基于正确轨迹）
            hit_info_new = find_hitting_point_physics(
                env, ball_pos, ball_vel, shoulder_pos, workspace_radius, remaining_horizon
            )

            if hit_info_new is None:
                if ball_was_hit and follow_through_start < 0:
                    follow_through_start = step
                    logger.info(f"步 {step}: 球已击中且飞走，开始随挥 ({follow_through_steps} 步)")
                elif not ball_was_hit:
                    logger.info(f"步 {step}: 球不再可达, 停止 MPC")
                    break
                hit_info_new = None  # v4: 保持 None，下方检查跳过重规划

            if hit_info_new is not None:
                k_hit_candidate = hit_info_new["k_hit"]
                if k_hit_candidate < max(10, k_hit_new // 4) and k_hit_new > 30:
                    k_hit_candidate = max(1, k_hit_new - replan_interval)

                # 对击球时刻施加时间预测扰动（衰减式）
                # 扰动随重规划次数线性衰减：初始满扰动，最后衰减到 alpha_min
                # 模拟预测精度随观测积累逐步提高的真实过程（保留残余系统性偏差）
                if abs(time_perturb_s) > 1e-6:
                    replan_count += 1
                    decay_alpha = max(perturb_alpha_min, 1.0 - replan_count / total_expected_replans)
                    effective_time_perturb = time_perturb_s * decay_alpha
                    perturb_steps = int(round(effective_time_perturb / dt))
                    if perturb_steps != 0:
                        k_hit_candidate = k_hit_candidate - perturb_steps
                        k_hit_candidate = max(5, min(k_hit_candidate, remaining_horizon - 1))

                p_hit_new = hit_info_new["p_hit"]
                v_ball_hit_new = hit_info_new["v_ball_hit"]
                k_hit_new = k_hit_candidate

                # 击球点可执行性后过滤：高风险点自动替换为 tube 窗口内更安全的候选
                p_hit_new, k_hit_new, hit_refine_log = refine_hit_point(
                    p_hit_new, k_hit_new,
                    remaining_horizon, env,
                )

                # 球速可行性检查：用 FK 估算关节限速下最大末端速度
                q_hit_feas = env.solve_ik(p_hit_new, q_init=x_current[:env.NQ], max_iter=50, eps=1e-2)
                env.set_arm_state(np.concatenate([q_hit_feas, np.zeros(env.NQ)]))
                J_p_feas = env.get_ee_jacp()
                max_ee_v = float(np.linalg.norm(np.abs(J_p_feas) @ robot_limits.qdot_max))
                ball_spd = float(np.linalg.norm(v_ball_hit_new))
                if ball_spd > max_ee_v * 2.0 and step % 40 == 0:
                    logger.warning(
                        f"步 {step}: 球速 {ball_spd:.1f}m/s 超过关节限速下最大末端速度 "
                        f"{max_ee_v:.1f}m/s (×1.5={max_ee_v*1.5:.1f})，"
                        f"建议 --ball-speed {max_ee_v:.0f} 或提前击球"
                    )

                # 对击打点施加空间偏移（衰减式）
                if abs(space_perturb_m) > 1e-6:
                    decay_alpha_sp = max(perturb_alpha_min, 1.0 - replan_count / total_expected_replans)
                    effective_space_perturb = space_perturb_m * decay_alpha_sp
                    if abs(effective_space_perturb) > 1e-6:
                        d_ball_hit = v_ball_hit_new / (np.linalg.norm(v_ball_hit_new) + 1e-8)
                        lateral = np.cross(d_ball_hit, np.array([0.0, 0.0, 1.0]))
                        lateral_norm = np.linalg.norm(lateral)
                        if lateral_norm > 1e-6:
                            lateral /= lateral_norm
                        else:
                            lateral = np.array([1.0, 0.0, 0.0])
                        p_hit_new = p_hit_new + lateral * effective_space_perturb

                n_des_new = -v_ball_hit_new / (np.linalg.norm(v_ball_hit_new) + 1e-8)
                if args.normal_flip:
                    n_des_new = -n_des_new
                current_n_des = n_des_new
                q_ik_cache = None

                p_follow_new = p_hit_new + hit_shift * d_hat

                k_hit_steps_history.append(k_hit_new)

                # v5: 扩展 horizon 包含随挥段
                horizon_full = k_hit_new + follow_through_steps
                horizon_plan = min(horizon_full, fixed_horizon)

                # v5: 终端目标 = 随挥终点
                p_terminal_v5 = p_hit_new + follow_through_length * d_hat
                v_terminal_v5 = follow_through_v_terminal * d_hat
                base_cost_fn.update_target(p_terminal_v5, v_terminal_v5, n_des=n_des_new)

                # v5: 中途目标 — 在 k_hit 步强制经过击球位置 + 鼓励高速
                if k_hit_new > 0 and k_hit_new < horizon_plan:
                    base_cost_fn.set_midpoint_target(k_hit_new, p_hit_new, Q_p * 2.0,
                                                     v_target=v_hit_at_contact,
                                                     Q_midpoint_v=Q_v * 5.0)

                if use_tube and tube_cost_fn is not None:
                    hit_window = search_hit_window(
                        env, ball_pos, ball_vel, shoulder_pos, workspace_radius,
                        remaining_horizon, tube_cfg,
                        ball_direction="y",
                        current_step=0,
                        robot_limits=robot_limits,
                        init_q=x_current[:env.NQ].copy(),
                    )
                    if hit_window is not None:
                        hitting_tube = build_hitting_tube(
                            hit_window, racket_speed, d_follow, tube_cfg,
                        )
                        tube_cost_fn.update_hitting_tube(hitting_tube, horizon=horizon_plan)
                        tube_cost_fn.update_target(p_terminal_v5, v_terminal_v5, n_des=n_des_new)
                    else:
                        logger.info(f"步 {step}: Tube 构建失败")

                # 求解器使用 tube_cost_fn（若启用）或 base_cost_fn
                cost_fn = tube_cost_fn if (use_tube and tube_cost_fn is not None) else base_cost_fn

                if k_hit_new > far_threshold:
                    ball_pos_save_far, ball_vel_save_far = env.get_ball_state()
                    p_target_jt = p_follow_new if use_tube and hitting_tube is not None else p_follow_new
                    u_jt = compute_jacobian_init_control(
                        env, x_current, p_target_jt, replan_interval, gain=60.0,
                        fix_joint5_angle=fix_joint5_angle,
                    )
                    env.set_ball_state(ball_pos_save_far, ball_vel_save_far)
                    env.set_arm_state(x_current)
                    U_buffer = u_jt
                    buffer_idx = 0
                    U_prev = U_prev if len(U_prev) > 0 else np.zeros((0, env.NU))
                    iters = 0
                    is_first_plan = False
                    replan_times.append(time.perf_counter() - t_replan_start)
                    replan_k_hit_history.append(k_hit_new)
                else:
                    ball_pos_save, ball_vel_save = env.get_ball_state()
                    env.set_arm_state(x_current)
                    env.update_kinematics()
                    pos_err_now = np.linalg.norm(env.get_ee_pos() - p_hit_new)

                    if pos_err_now > 0.10:
                        Q_p_scale = Q_p_scale_far
                        Q_v_scale = Q_v_scale_far
                    else:
                        ratio = pos_err_now / 0.10
                        Q_p_scale = Q_p_scale_near + (Q_p_scale_far - Q_p_scale_near) * ratio
                        Q_v_scale = Q_v_scale_near + (Q_v_scale_far - Q_v_scale_near) * ratio

                    cost_fn.update_weights(Q_p_scale, Q_v_scale)

                    # 分阶段软平滑权重调度
                    if k_hit_new > 50:
                        s = far_stage
                    elif k_hit_new > 20:
                        s = mid_stage
                    else:
                        s = near_stage
                    if hasattr(cost_fn, 'set_smoothness_scale'):
                        cost_fn.set_smoothness_scale(
                            float(s["Q_qdot_mult"]),
                            float(s["Q_qddot_mult"]),
                            float(s["Q_du_mult"]),
                        )

                    if use_r_decay:
                        R_schedule_new = compute_r_schedule(
                            horizon_full, R, decay_ratio=r_decay_ratio,
                        )[:horizon_plan]
                        cost_fn.set_R_schedule(R_schedule_new)
                    else:
                        cost_fn.set_R_schedule(None)

                    iters_plan = max_iter_per_plan
                    skip_ls = True
                    fp_limits = robot_limits
                    fast_lin = False
                    if is_first_plan:
                        iters_plan = first_plan_iters
                        skip_ls = True
                        fp_limits = None
                        is_first_plan = False
                    elif k_hit_new <= near_threshold:
                        if k_hit_new > 30:
                            iters_plan = min(near_plan_iters, max(max_iter_per_plan, 10))
                            fast_lin = True
                            fp_limits = None
                        else:
                            iters_plan = min(near_plan_iters, 10) if args.realtime else near_plan_iters
                    else:
                        iters_plan = max_iter_per_plan
                        fast_lin = True

                    if use_backswing:
                        q_hit_new_ik = env.solve_ik(
                            p_hit_new, q_init=x_current[:env.NQ],
                            max_iter=150, eps=1e-3,
                        )
                        if fix_joint5_angle is not None:
                            q_hit_new_ik[5] = fix_joint5_angle

                        env.set_arm_state(np.concatenate([q_hit_new_ik, np.zeros(env.NQ)]))
                        J_p_new = env.get_ee_jacp()
                        qdot_hit_new = np.linalg.lstsq(J_p_new, v_hit_at_contact, rcond=None)[0]
                        max_qdot = 3.0
                        qdot_norm = np.linalg.norm(qdot_hit_new)
                        if qdot_norm > max_qdot:
                            qdot_hit_new *= max_qdot / qdot_norm

                        backswing_scale = horizon_full / max(k_hit_total, 1)
                        q_des_traj_full = np.zeros((horizon_full, env.NQ))
                        q_des_traj_full[:, 0] = compute_joint1_backswing_trajectory(
                            x_current[0], x_current[env.NQ],
                            q_hit_new_ik[0], qdot_hit_new[0],
                            horizon_full,
                            backswing_offset=backswing_offset * backswing_scale,
                            backswing_ratio=backswing_ratio,
                        )
                        for j in range(1, env.NQ):
                            q_des_traj_full[:, j] = np.linspace(x_current[j], q_hit_new_ik[j], horizon_full)

                        q_des_traj_new = q_des_traj_full[:horizon_plan]
                        cost_fn.set_q_des_traj(q_des_traj_new, Q_joint=Q_joint)

                        if len(U_prev) >= horizon_full // 3:
                            U_warm = resample_control_sequence(U_prev, horizon_full)[:horizon_plan]
                            if fix_joint5_angle is not None:
                                U_warm = fix_joint5_control_trajectory(
                                    U_warm, x_current, env, fix_joint5_angle,
                                )
                        else:
                            U_warm_full, _ = generate_backswing_warm_start(
                                env, x_current, p_follow_new, v_hit_desired, horizon_full,
                                backswing_offset=backswing_offset * backswing_scale,
                                backswing_ratio=backswing_ratio,
                                fix_joint5_angle=fix_joint5_angle,
                                n_des=n_des_new,
                            )
                            U_warm = U_warm_full[:horizon_plan]
                    else:
                        if len(U_prev) >= horizon_full // 3:
                            U_warm = resample_control_sequence(U_prev, horizon_full)[:horizon_plan]
                            if fix_joint5_angle is not None:
                                U_warm = fix_joint5_control_trajectory(
                                    U_warm, x_current, env, fix_joint5_angle,
                                )
                        else:
                            U_warm = compute_jacobian_init_control(
                                env, x_current, p_follow_new, horizon_full, gain=30.0,
                                fix_joint5_angle=fix_joint5_angle,
                            )[:horizon_plan]

                    X_mpc, U_mpc, iter_costs, solver_ok = solver.solve_few_iters(
                        env, cost_fn, x_current, U_warm,
                        max_iter=iters_plan,
                        skip_linesearch=skip_ls,
                        limits=fp_limits,
                        use_fast_lin=fast_lin,
                    )

                    env.set_ball_state(ball_pos_save, ball_vel_save)
                    env.set_arm_state(x_current)

                    if not solver_ok:
                        logger.warning(
                            f"步 {step}: iLQR 部分迭代被拒绝，仍使用部分收敛结果"
                        )

                    if fix_joint5_angle is not None:
                        U_mpc = fix_joint5_control_trajectory(
                            U_mpc, x_current, env, fix_joint5_angle,
                        )

                    if len(U_mpc) > replan_interval:
                        U_prev = U_mpc[replan_interval:]
                    elif len(U_mpc) > 0:
                        U_prev = U_mpc[1:]
                    else:
                        U_prev = np.zeros((0, env.NU))

                    U_buffer = U_mpc[:replan_interval]
                    buffer_idx = 0

                    # 近距阶段：延长 U_buffer 减少重规划频率
                    if k_hit_new <= 30 and len(U_mpc) >= replan_interval * 2:
                        U_buffer = U_mpc[:replan_interval * 2]
                        buffer_idx = 0
                        logger.debug(
                            f"步 {step}: near_hit, U_buffer extended to "
                            f"{len(U_buffer)} steps"
                        )

                    if len(U_mpc) > 0 and len(X_mpc) > len(U_mpc):
                            m = compute_trajectory_metrics(X_mpc, U_mpc, robot_limits, dt)
                            logger.debug(
                                f"步 {step} 求解指标: max_qdot={m.max_qdot_ratio:.2f}x(j{m.max_qdot_joint}), "
                                f"max_qddot={m.max_qddot_ratio:.2f}x(j{m.max_qddot_joint}), "
                                f"joint_speed={m.max_joint_speed_rad_s:.1f}rad/s"
                            )

                    if len(iter_costs) > 0:
                        cost_history.append(iter_costs[-1])

                    iters = iters_plan
                    t_replan_dur = time.perf_counter() - t_replan_start
                    replan_times.append(t_replan_dur)
                    replan_k_hit_history.append(k_hit_new)
                    logger.info(
                        f"REPLAN step={step} k_hit={k_hit_new} iters={iters_plan} "
                        f"horizon={horizon_plan} t={t_replan_dur*1000:.0f}ms "
                        f"fp_limits={fp_limits is not None} fast_lin={fast_lin}"
                    )
                    # 可解释性日志：softmin 权重和走廊 margin 摘要
                    if (use_tube and tube_cost_fn is not None
                            and len(tube_cost_fn._last_softmin_alphas) > 0):
                        alphas = tube_cost_fn._last_softmin_alphas
                        costs_s = tube_cost_fn._last_softmin_costs
                        dom_idx = int(np.argmax(alphas))
                        logger.info(
                            f"[Softmin] dominant_α={alphas[dom_idx]:.3f} "
                            f"cost={costs_s[dom_idx]:.1f} | "
                            f"top3: {', '.join(f'{a:.3f}' for a in sorted(alphas, reverse=True)[:3])}"
                        )
                    if (use_tube and tube_cost_fn is not None
                            and len(tube_cost_fn._last_tube_margins) > 0):
                        margins = np.array(list(tube_cost_fn._last_tube_margins.values()))
                        n_active = int(np.sum(margins > 0))
                        logger.info(
                            f"[Corridor] margin: min={margins.min()*1000:.1f}mm "
                            f"max={margins.max()*1000:.1f}mm "
                            f"active={n_active}/{len(margins)} (hinge>0)"
                        )

        step_is_replan.append(bool(need_replan))

        if buffer_idx < len(U_buffer):
            u_cmd = U_buffer[buffer_idx]
            buffer_idx += 1
        else:
            buffer_exhaustion_count += 1
            u_cmd = np.zeros(env.NU)
            if k_hit_new > 0:
                # 缓冲耗尽：雅可比转矩后备
                env.set_arm_state(x_current)
                p_ee = env.get_ee_pos()
                J_p = env.get_ee_jacp()
                err = p_hit_new - p_ee
                tau_backup = J_p.T @ err * 30.0
                tau_backup -= 2.0 * x_current[env.NQ:]
                ctrl_lo = env.model.actuator_ctrlrange[:env.NU, 0]
                ctrl_hi = env.model.actuator_ctrlrange[:env.NU, 1]
                u_cmd = np.clip(tau_backup, ctrl_lo, ctrl_hi)

        if fix_joint5_angle is not None:
            u_cmd = fix_joint5_control(u_cmd, fix_joint5_angle, x_current, env.NQ)

        # ---- 碰撞检测：提前开窗（30步），末段无条件 ----
        enable_collision = False
        if not ball_was_hit:
            if k_hit_new <= 30 and dist_cur < 0.35:
                enable_collision = True
            elif k_hit_new <= 10:
                enable_collision = True

        if hasattr(env, "set_arm_collision"):
            env.set_arm_collision(enable_collision)

        # ---- 硬约束层1: X 平面墙预判（先于安全滤波器）----
        ball_save_x = env.get_ball_state()
        x_save_x = x_current.copy()
        arm_save_x = env.get_arm_state().copy()

        beta_list_x = [1.0, 0.6, 0.3, 0.0]
        u_xsafe = u_cmd
        for beta_x in beta_list_x:
            u_try_x = beta_x * u_cmd
            ctrl_lo = env.model.actuator_ctrlrange[:env.NU, 0]
            ctrl_hi = env.model.actuator_ctrlrange[:env.NU, 1]
            u_try_x = np.clip(u_try_x, ctrl_lo, ctrl_hi)
            if fix_joint5_angle is not None:
                u_try_x = fix_joint5_control(u_try_x, fix_joint5_angle, x_current, env.NQ)

            x_pred = env.step_from_state(x_current, u_try_x)
            env.update_kinematics()
            ok_x = all(env.data.xpos[bid, 0] <= -0.1 for bid in _hard_x_body_ids)

            env.set_ball_state(*ball_save_x)
            env.set_arm_state(x_save_x)

            if ok_x:
                u_xsafe = u_try_x
                break

        # ---- 硬约束层2: 安全滤波器 robot safety filter（最后防线）----
        ball_save_sf = env.get_ball_state()
        arm_save_sf = x_current.copy()

        def _safety_step(x: np.ndarray, u_val: np.ndarray) -> np.ndarray:
            x_next = env.step_from_state(x, u_val)
            env.set_ball_state(*ball_ref)
            return x_next

        ball_ref = ball_save_sf
        ok_filter, reason_filter = check_one_step_feasibility(
            x_current, u_xsafe, robot_limits, dt,
            step_predictor=_safety_step,
            k_hit_remaining=k_hit_new,
            env=env,
        )
        if not ok_filter:
            safety_beta_list = [0.8, 0.6, 0.4, 0.2, 0.0]
            found_safe = False
            for beta_s in safety_beta_list:
                u_try_s = beta_s * u_xsafe
                u_try_s = np.clip(u_try_s, ctrl_lo, ctrl_hi)
                if fix_joint5_angle is not None:
                    u_try_s = fix_joint5_control(u_try_s, fix_joint5_angle, x_current, env.NQ)
                env.set_arm_state(arm_save_sf)
                ok_s, _ = check_one_step_feasibility(
                    x_current, u_try_s, robot_limits, dt,
                    step_predictor=_safety_step,
                    k_hit_remaining=k_hit_new,
                    env=env,
                )
                if ok_s:
                    u_xsafe = u_try_s
                    found_safe = True
                    if beta_s < 1.0:
                        logger.info(
                            "[SAFETY_FILTER] beta=%.1f: %s",
                            beta_s, reason_filter,
                        )
                    break
            if not found_safe:
                u_xsafe = -20.0 * x_current[env.NQ:]
                exec_metrics.emergency_stop_count += 1
                logger.warning(
                    "[EMERGENCY_STOP] 步 %d: %s, safe_hold 阻尼制动",
                    step, reason_filter,
                )

        env.set_ball_state(*ball_save_sf)
        env.set_arm_state(arm_save_sf)

        u_final = u_xsafe

        ball_vel_before_step = ball_vel.copy() if enable_collision else ball_vel
        x_current, ball_pos, ball_vel = env.step_full(u_final)

        # ---- 硬约束：执行后检查，越界则 PD 推回一步 ----
        env.update_kinematics()
        violated = []
        for bid in _hard_x_body_ids:
            if env.data.xpos[bid, 0] > -0.1:
                name = _mj.mj_id2name(env.model, _mj.mjtObj.mjOBJ_BODY, bid)
                violated.append(name)
        if violated:
            q_now = x_current[:env.NQ]
            qdot_now = x_current[env.NQ:]
            u_push = 300.0 * (init_q - q_now) - 20.0 * qdot_now
            u_push = np.clip(u_push,
                             env.model.actuator_ctrlrange[:env.NU, 0],
                             env.model.actuator_ctrlrange[:env.NU, 1])
            if fix_joint5_angle is not None:
                u_push = fix_joint5_control(u_push, fix_joint5_angle, x_current, env.NQ)
            x_current, ball_pos, ball_vel = env.step_full(u_push)
            logger.warning(
                f"步 {step}: 臂越界 ({len(violated)} bodies: {', '.join(violated[:3])}...), "
                f"PD推回一步"
            )

        # 碰撞检测
        ball_racket_hit = False
        if enable_collision and not ball_was_hit:
            n_contacts = env.data.ncon
            if n_contacts > 0:
                for ci in range(n_contacts):
                    c = env.data.contact[ci]
                    g1 = env.model.geom(c.geom1).name
                    g2 = env.model.geom(c.geom2).name
                    if "ball" in g1 or "ball" in g2:
                        if "racket" in g1 or "racket" in g2:
                            ball_racket_hit = True
                            ee_vel = env.get_ee_vel()
                            ee_speed = np.linalg.norm(ee_vel)
                            ball_spd = np.linalg.norm(ball_vel)
                            if ee_speed > 2.0:
                                active_contact = True
                                contact_type = "主动击球"
                            else:
                                passive_contact = True
                                contact_type = "被动接触"
                            logger.info(
                                f"步 {step}: 球拍击球! {g1}<->{g2}, "
                                f"球拍速度={ee_speed:.2f}m/s, 球速={ball_spd:.2f}m/s "
                                f"[{contact_type}]"
                            )

        if hasattr(env, "set_arm_collision"):
            env.set_arm_collision(True)

        X_history.append(x_current.copy())
        U_history.append(u_final.copy())
        ball_pos_history.append(ball_pos.copy())

        env.update_kinematics()
        pos_err = np.linalg.norm(env.get_ee_pos() - p_hit_new)
        pos_error_history.append(pos_err)

        step_time = time.perf_counter() - t_step_start
        step_times.append(step_time)

        # 实时节奏：模拟 5ms/步，给异步重规划足够时间完成
        if args.realtime and step_time < dt:
            sleep_dur = dt - step_time
            time.sleep(sleep_dur)
            total_sleep_time += sleep_dur

        if step % 20 == 0 or k_hit_new <= 5:
            tube_info = ""
            if use_tube and hitting_tube is not None:
                n_valid = sum(valid_hit_history[-20:]) if len(valid_hit_history) >= 20 else sum(valid_hit_history)
                tube_info = f", valid_hit={n_valid}"
            # 实时约束指标
            qdot_cur = x_current[env.NQ:]
            qdot_ratio = float(np.max(np.abs(qdot_cur) / np.maximum(robot_limits.qdot_max, 1e-8)))
            racket_speed = float(np.linalg.norm(env.get_ee_vel()))
            face_speed = env.get_racket_face_speed()
            # 更新执行层指标
            exec_metrics.max_qdot_ratio = max(exec_metrics.max_qdot_ratio, qdot_ratio)
            exec_metrics.max_tcp_speed = max(exec_metrics.max_tcp_speed, racket_speed)
            exec_metrics.max_racket_face_speed = max(exec_metrics.max_racket_face_speed, face_speed)
            exec_metrics.total_mpc_steps += 1
            logger.info(
                f"步 {step}: 剩余={k_hit_new}, 误差={pos_err:.4f}m, "
                f"距离={dist_cur:.4f}m, 迭代={iters}, 步耗时={step_time*1000:.1f}ms{tube_info}, "
                f"max_qdot={qdot_ratio:.2f}x, TCP={racket_speed:.1f}m/s Face={face_speed:.1f}m/s"
            )

        if ball_racket_hit and not ball_was_hit:
            ball_was_hit = True
            hit_step = step
            env.update_kinematics()
            p_ee_at_hit = env.get_ee_pos().copy()
            ball_pos_at_hit = ball_pos.copy()

            n_racket = env.get_ee_normal()
            n_hat = n_racket / (np.linalg.norm(n_racket) + 1e-8)
            v_ee = env.get_ee_vel()
            v_ball_pre = ball_vel_before_step
            v_rel_n = np.dot(v_ball_pre - v_ee, n_hat)
            e = 0.8
            v_ball_new = v_ball_pre - (1 + e) * v_rel_n * n_hat
            logger.info(
                f"  弹性反弹: v_ball_before={v_ball_pre}, v_ball_after={v_ball_new}, "
                f"球速: {np.linalg.norm(v_ball_pre):.2f}->{np.linalg.norm(v_ball_new):.2f} m/s"
            )
            env.set_ball_vel(v_ball_new)

        # v4: 碰撞后延迟几步再开始随挥（让碰撞物理完成）
        if ball_was_hit and follow_through_start < 0 and hit_step >= 0 and (step - hit_step) >= 3:
            follow_through_start = step
            logger.info(f"步 {step}: 碰撞后开始随挥 ({follow_through_steps} 步)")

        # v4: 到达击打时刻后不 break，继续执行随挥段（只触发一次）
        if k_hit_new <= 1 and follow_through_start < 0:
            logger.info(f"步 {step}: 到达击打时刻，开始随挥 ({follow_through_steps} 步)")
            hit_step = step if hit_step < 0 else hit_step
            env.update_kinematics()
            if p_ee_at_hit is None:
                p_ee_at_hit = env.get_ee_pos().copy()
            if ball_pos_at_hit is None:
                ball_pos_at_hit = ball_pos.copy()
            follow_through_remain = follow_through_steps
            follow_through_start = step

        # v4: 随挥段 — 使用 PD 控制沿 d_follow 方向匀减速
        if follow_through_start >= 0:
            dt_follow = step - follow_through_start
            if dt_follow < follow_through_steps:
                # 匀减速轨迹：s(dt) = v_max * dt - 0.5 * a * dt^2
                # a = v_max / T_follow（从 v_max 匀减到 0）
                v_max_follow = np.linalg.norm(v_hit_at_contact)  # 1.8 m/s
                T_follow = follow_through_steps * dt  # 总随挥时间（秒）
                a_follow = v_max_follow / T_follow if T_follow > 0 else 0.0
                t_elapsed = dt_follow * dt  # 已过时间（秒）
                # 期望位置：从击球点沿 d_follow 移动
                p_des_follow = p_ee_at_hit + d_follow * (v_max_follow * t_elapsed - 0.5 * a_follow * t_elapsed ** 2)
                # 期望速度：匀减速
                v_des_follow = d_follow * max(v_max_follow - a_follow * t_elapsed, 0.0)

                # 任务空间 PD + 雅可比转置映射到关节空间
                env.update_kinematics()
                p_ee_cur = env.get_ee_pos()
                J_p = env.get_ee_jacp()[:, :env.NQ]
                dp = p_des_follow - p_ee_cur
                Kp_follow = 200.0
                Kd_follow = 20.0
                F_follow = Kp_follow * dp - Kd_follow * J_p @ x_current[env.NQ:]
                u_follow = J_p.T @ F_follow
                u_follow = np.clip(u_follow,
                                   env.model.actuator_ctrlrange[:env.NU, 0],
                                   env.model.actuator_ctrlrange[:env.NU, 1])
                # v4: TCP 速度限制 — 随挥阶段同样受限于 max_tcp
                v_ee_follow = J_p @ x_current[env.NQ:]
                tcp_speed_follow = float(np.linalg.norm(v_ee_follow))
                max_tcp_limit = float(args.max_tcp) if args.max_tcp and args.max_tcp > 0 else float('inf')
                if tcp_speed_follow > max_tcp_limit:
                    scale_factor = max_tcp_limit / tcp_speed_follow
                    u_follow *= scale_factor
                x_current = env.step(u_follow)
                U_history.append(u_follow.copy())
                X_history.append(x_current.copy())
                ball_pos_history.append(env.get_ball_pos().copy())
                continue  # 跳过 MPC 控制，直接进入下一步
            else:
                logger.info(f"步 {step}: 随挥完成 ({follow_through_steps} 步)")
                break

    # ===== 停止异步重规划器 =====
    if async_mode:
        async_replanner.stop()

    # ===== 记录 MPC 阶段结束时间快照 =====
    t_mpc_end = time.perf_counter()
    n_mpc_steps = len(U_history)                   # MPC 步数（不含后仿真）
    n_replans = len(replan_times)                  # 总重规划次数

    # ===== 击打后继续仿真 =====
    # v4: 随挥段已在 MPC 循环内完成，post_hit 仅做短暂停留
    post_hit_steps = 20
    logger.info(f"击打后继续仿真 {post_hit_steps} 步...")
    for _ in range(post_hit_steps):
        q_hold = x_current[:env.NQ].copy()
        u_hold = 100.0 * (q_hold - x_current[:env.NQ]) - 10.0 * x_current[env.NQ:]
        u_hold = np.clip(u_hold,
                         env.model.actuator_ctrlrange[:env.NU, 0],
                         env.model.actuator_ctrlrange[:env.NU, 1])
        if fix_joint5_angle is not None:
            u_hold = fix_joint5_control(u_hold, fix_joint5_angle, x_current, env.NQ)
        x_current, ball_pos, _ = env.step_full(u_hold)
        X_history.append(x_current.copy())
        U_history.append(u_hold.copy())
        ball_pos_history.append(ball_pos.copy())

    # ===== 评估 =====
    t_total = time.perf_counter() - t_total_start
    t_mpc = t_mpc_end - t_total_start                         # MPC 阶段墙钟时间
    t_post_hit = t_total - t_mpc                               # 后仿真阶段墙钟时间
    n_steps = len(U_history)
    n_post_actual = n_steps - n_mpc_steps                       # 实际后仿真步数

    # ---- 逐步耗时分类 ----
    _s_arr = np.array(step_times)
    _r_mask = np.array(step_is_replan)
    avg_step_ms = np.mean(_s_arr) * 1000 if len(_s_arr) > 0 else 0
    avg_non_replan_step_ms = np.mean(_s_arr[~_r_mask]) * 1000 if np.any(~_r_mask) else 0
    max_step_ms = np.max(_s_arr) * 1000 if len(_s_arr) > 0 else 0
    if np.any(~_r_mask):
        max_non_replan_ms = np.max(_s_arr[~_r_mask]) * 1000
        min_non_replan_ms = np.min(_s_arr[~_r_mask]) * 1000
    else:
        max_non_replan_ms = 0.0
        min_non_replan_ms = 0.0

    # ---- 真实机器人延迟估算 ----
    real_robot_overhead_ms = 2.0    # 传感器读取+通信延迟估算 (ms)
    est_real_non_replan_ms = avg_non_replan_step_ms + real_robot_overhead_ms

    # ---- 重规划指标（区分首次/稳态，区分 far/near 阶段） ----
    _rt = np.array(replan_times) * 1000   # 全部转为 ms
    _rk = np.array(replan_k_hit_history)  # 每次重规划时的 k_hit_new

    # 首次规划（冷启动，实际部署中在球到达前完成，不纳入实时预算分析）
    first_replan_ms = float(_rt[0]) if len(_rt) > 0 else 0.0

    # 稳态重规划（排除首次）
    steady_rt = _rt[1:] if len(_rt) > 1 else np.array([])
    steady_rk = _rk[1:] if len(_rk) > 1 else np.array([])
    avg_steady_replan_ms = float(np.mean(steady_rt)) if len(steady_rt) > 0 else 0.0
    max_steady_replan_ms = float(np.max(steady_rt)) if len(steady_rt) > 0 else 0.0

    # far 阶段预算 = replan_interval * dt，near 阶段预算 = 2 * replan_interval * dt
    far_budget_ms = replan_interval * dt * 1000
    near_budget_ms = 2.0 * replan_interval * dt * 1000
    near_k_threshold = near_threshold  # 与 MPC 迭代策略一致

    # 按阶段分类稳态重规划
    far_rt = steady_rt[steady_rk > near_k_threshold] if len(steady_rk) > 0 else np.array([])
    near_rt = steady_rt[steady_rk <= near_k_threshold] if len(steady_rk) > 0 else np.array([])
    avg_far_replan_ms = float(np.mean(far_rt)) if len(far_rt) > 0 else 0.0
    max_far_replan_ms = float(np.max(far_rt)) if len(far_rt) > 0 else 0.0
    avg_near_replan_ms = float(np.mean(near_rt)) if len(near_rt) > 0 else 0.0
    max_near_replan_ms = float(np.max(near_rt)) if len(near_rt) > 0 else 0.0

    # 稳态超预算判断：用最大值 vs 对应阶段预算
    far_over_budget = max_far_replan_ms > far_budget_ms if len(far_rt) > 0 else False
    near_over_budget = max_near_replan_ms > near_budget_ms if len(near_rt) > 0 else False
    steady_realtime_ok = not far_over_budget and not near_over_budget

    # 保留向后兼容的简单指标
    avg_replan_ms = float(np.mean(_rt)) if len(_rt) > 0 else 0.0
    max_replan_ms = float(np.max(_rt)) if len(_rt) > 0 else 0.0

    # ---- 核心实时比率（仅 MPC 控制阶段，不含后仿真） ----
    mpc_sim_time = n_mpc_steps * dt
    t_mpc_compute = t_mpc - total_sleep_time
    mpc_realtime_ratio = mpc_sim_time / t_mpc_compute if t_mpc_compute > 0 else 0
    mpc_wall_ratio = mpc_sim_time / t_mpc if t_mpc > 0 else 0

    logger.info(
        f"MPC 完成: MPC={t_mpc:.2f}s/{n_mpc_steps}步, "
        f"后仿真={t_post_hit:.2f}s/{n_post_actual}步, "
        f"MPC实时比率={mpc_realtime_ratio:.2f}x(纯计算)/{mpc_wall_ratio:.2f}x(含sleep), "
        f"重规划={n_replans}次 首次={first_replan_ms:.0f}ms 稳态avg={avg_steady_replan_ms:.0f}ms"
    )

    if p_ee_at_hit is not None:
        p_ee_final = p_ee_at_hit
    else:
        env.set_arm_state(x_current)
        p_ee_final = env.get_ee_pos()
    v_ee_final = env.get_ee_vel()
    pos_error_plan = np.linalg.norm(p_ee_final - p_hit)
    if ball_pos_at_hit is not None:
        pos_error = np.linalg.norm(p_ee_final - ball_pos_at_hit)
    else:
        pos_error = pos_error_plan
    # v4: 速度误差对比击球时刻期望速度（1.8 m/s），而不是终端速度
    vel_error = np.linalg.norm(v_ee_final - v_hit_at_contact)

    ball_vel_after = env.get_ball_vel()
    ball_speed_after = np.linalg.norm(ball_vel_after)
    ball_vel_before = v_ball_hit_new
    speed_before = np.linalg.norm(ball_vel_before)
    v_ee_speed = np.linalg.norm(v_ee_final)

    # ---- Tube 专用评估指标 ----
    d_arr = np.array(distances_history)
    min_dist = float(np.min(d_arr))

    # ball_near 指标：球物理上在拍附近的步数
    ball_near_arr = np.array(ball_near_history, dtype=bool)
    ball_near_duration = int(np.sum(ball_near_arr))
    ball_near_ms = ball_near_duration * dt * 1000

    # tube_ready 指标：球拍在 tube 窗口内保持击球姿态的步数
    tube_ready_arr = np.array(tube_ready_history, dtype=bool)
    tube_ready_duration = int(np.sum(tube_ready_arr))
    tube_ready_ms = tube_ready_duration * dt * 1000

    # 最长连续 tube_ready 窗口
    longest_tube_ready = 0
    cur_len = 0
    for v in tube_ready_history:
        if v:
            cur_len += 1
        else:
            longest_tube_ready = max(longest_tube_ready, cur_len)
            cur_len = 0
    longest_tube_ready = max(longest_tube_ready, cur_len)
    longest_tube_ready_ms = longest_tube_ready * dt * 1000

    # tube_ready 起止步
    tr_indices = [i for i, v in enumerate(tube_ready_history) if v]
    tube_ready_range_str = ""
    if tr_indices:
        tube_ready_range_str = f"[{tr_indices[0]}..{tr_indices[-1]}] = {tr_indices[-1]-tr_indices[0]+1} 步"

    tube_margin = float(np.min(d_arr - (0.033 + 0.12))) if len(d_arr) > 0 else 0.0

    # 命中时刻误差
    if p_ee_at_hit is not None and hit_step >= 0:
        hit_time_actual = hit_step * dt
        hit_time_expected = k_hit_total * dt
        hit_time_error = abs(hit_time_actual - hit_time_expected) * 1000  # ms
        hit_position_error = float(np.linalg.norm(p_ee_at_hit - ball_pos_at_hit)) if ball_pos_at_hit is not None else float(np.linalg.norm(p_ee_at_hit - p_hit))
    else:
        hit_time_error = 0.0
        hit_position_error = pos_error

    best_candidate_cost = 0.0
    weighted_tube_cost = 0.0
    if hitting_tube is not None and p_ee_at_hit is not None and len(hitting_tube.k_candidates) > 0:
        for i in range(len(hitting_tube.k_candidates)):
            p_des = hitting_tube.p_racket_des[i]
            v_des = hitting_tube.v_racket_des[i]
            n_des = hitting_tube.n_racket_des[i]
            w = hitting_tube.weights[i]
            pos_cost = np.linalg.norm(p_ee_at_hit - p_des)**2
            vel_cost = np.linalg.norm(v_ee_final - v_des)**2
            n_dot = float(env.get_ee_normal() @ n_des)
            n_cost = 1.0 - n_dot
            total_i = 0.5 * (tube_cfg.Q_p_tube * pos_cost + tube_cfg.Q_v_tube * vel_cost + tube_cfg.Q_n_tube * n_cost)
            weighted_tube_cost += w * total_i
            if i == 0 or total_i < best_candidate_cost:
                best_candidate_cost = total_i

    # ---- 打印评估结果 ----
    ball_racket_threshold = 0.033 + 0.12  # 球半径 + 球拍半径
    print("\n========================================")
    if pos_error < 0.05:
        print("  RM-65 击打成功！（球-拍距离 < 5cm，精准命中）")
    elif pos_error < ball_racket_threshold:
        print(f"  RM-65 击打命中！（球-拍距离 {pos_error:.4f}m < {ball_racket_threshold:.3f}m 物理接触阈值）")
    elif pos_error < 0.1:
        print("  RM-65 击打接近！（球-拍距离 < 10cm，未物理接触但接近）")
    else:
        print("  RM-65 击打偏差较大，需要调整参数。")

    print(f"  Tube 模式: {'启用' if use_tube or hitting_tube is not None else '禁用'}")
    # ---- v2 可解释性诊断 ----
    if use_tube and tube_cost_fn is not None:
        print(f"  --- v2 Tube 可解释性诊断 ---")
        if len(tube_cost_fn._last_softmin_alphas) > 0:
            alphas = tube_cost_fn._last_softmin_alphas
            costs_s = tube_cost_fn._last_softmin_costs
            print(f"  P0-2 Softmin终端: 候选数={len(alphas)}, beta={tube_cost_fn._softmin_beta:.1f}")
            top_n = min(5, len(alphas))
            top_idx = np.argsort(alphas)[::-1][:top_n]
            for rank, i in enumerate(top_idx):
                print(f"    #{rank+1}: α={alphas[i]:.4f}, cost={costs_s[i]:.1f}")
        if len(tube_cost_fn._last_tube_margins) > 0:
            margins = np.array(list(tube_cost_fn._last_tube_margins.values()))
            n_active = int(np.sum(margins > 0))
            print(f"  走廊激活: {n_active}/{len(margins)} 步 margin>0 (hinge激活) | "
                  f"margin范围=[{margins.min()*1000:.1f}, {margins.max()*1000:.1f}]mm")
    if initial_tube_n_candidates > 0:
        print(f"  初始候选窗口: {initial_tube_n_candidates} 步 {initial_tube_k_range} "
              f"(半宽 {args.window_ms:.0f}ms)")
    if abs(args.time_perturb_ms) > 0.01:
        print(f"  时间扰动: {args.time_perturb_ms:+.1f} ms")
    if abs(args.space_perturb_m) > 0.001:
        print(f"  球空间偏移: {args.space_perturb_m:+.3f} m (实际球轨迹侧偏，MPC用原始预测)")
    print(f"  击打目标位置: {np.round(p_hit_new, 3)}")
    print(f"  末端实际位置: {np.round(p_ee_final, 3)}")
    if ball_pos_at_hit is not None:
        print(f"  击球时刻球位置: {np.round(ball_pos_at_hit, 3)}")
        print(f"  实际球-拍距离: {pos_error:.4f} m")
    print(f"  规划跟踪误差: {pos_error_plan:.4f} m")
    print(f"  速度误差: {vel_error:.4f} m/s")
    print(f"  击打后球速: {ball_speed_after:.2f} m/s")
    print(f"  MPC 总步数: {n_steps}")
    print(f"  线性化: {'解析' if use_analytical else '有限差分'}")
    print(f"  后摆: {'启用' if use_backswing else '禁用'}")
    print(f"  随挥偏移: {hit_shift:.3f}m")
    print(f"  R 退火: {'启用' if use_r_decay else '禁用'}")
    print(f"  --- Tube 专用指标 ---")
    print(f"  最小球拍-球距离: {min_dist:.4f} m")
    print(f"  ball_near 步数: {ball_near_duration} = {ball_near_ms:.1f} ms  (球物理上在拍附近)")
    print(f"  tube_ready 步数: {tube_ready_duration} = {tube_ready_ms:.1f} ms  (球拍在窗口内保持击球姿态)")
    if tube_ready_range_str:
        print(f"  tube_ready 起止步: {tube_ready_range_str}  (best_k={initial_hitting_tube.best_k if initial_hitting_tube else '?'})")
    print(f"  最长连续 tube_ready: {longest_tube_ready} 步 = {longest_tube_ready_ms:.1f} ms")
    print(f"  Tube margin: {tube_margin:.4f} m")
    print(f"  Best candidate cost: {best_candidate_cost:.2f}")
    print(f"  Weighted tube cost: {weighted_tube_cost:.2f}")
    print(f"  击打时间误差: {hit_time_error:.1f} ms")
    print(f"  击打位置误差: {hit_position_error:.4f} m")
    print(f"  --- 计算性能 ---")
    print(f"  总墙钟时间: {t_total:.2f}s")
    print(f"    MPC 计算阶段: {t_mpc:.2f}s ({n_mpc_steps}步, 仿真时间={mpc_sim_time:.3f}s)")
    print(f"    MPC 实时比率: {mpc_realtime_ratio:.2f}x (纯计算, >1=超实时)")
    print(f"    MPC 含sleep:  {mpc_wall_ratio:.2f}x (含 sleep={total_sleep_time*1000:.0f}ms)")
    print(f"    后仿真阶段:   {t_post_hit:.2f}s ({n_post_actual}步, PD保持)")
    print(f"  --- 重规划性能 ({n_replans}次) ---")
    print(f"    首次规划(冷启动): {first_replan_ms:.0f}ms (不纳入实时预算, 部署时球到达前完成)")
    n_steady = len(steady_rt)
    if n_steady > 0:
        print(f"    稳态重规划({n_steady}次): avg={avg_steady_replan_ms:.0f}ms, max={max_steady_replan_ms:.0f}ms")
    if len(far_rt) > 0:
        far_tag = "[!!] 超预算" if far_over_budget else "[OK]"
        print(f"    far阶段({len(far_rt)}次, k_hit>{near_k_threshold}): avg={avg_far_replan_ms:.0f}ms, max={max_far_replan_ms:.0f}ms / 预算={far_budget_ms:.0f}ms {far_tag}")
    if len(near_rt) > 0:
        near_tag = "[!!] 超预算" if near_over_budget else "[OK]"
        print(f"    near阶段({len(near_rt)}次, k_hit<={near_k_threshold}): avg={avg_near_replan_ms:.0f}ms, max={max_near_replan_ms:.0f}ms / 预算={near_budget_ms:.0f}ms(buffer扩展2x) {near_tag}")
    if steady_realtime_ok and n_steady > 0:
        print(f"    稳态实时性: [OK] 所有稳态重规划在预算内")
    elif n_steady > 0:
        print(f"    稳态实时性: [!!] 存在超预算重规划, 需增大 replan_interval 或优化 iLQR")
    print(f"  --- 逐步执行性能 ---")
    print(f"    MPC 每步平均:     {avg_step_ms:.1f}ms ({n_mpc_steps}步)")
    n_non_replan = n_mpc_steps - n_replans
    if avg_non_replan_step_ms > 0:
        print(f"    非重规划步:       avg={avg_non_replan_step_ms:.1f}ms ({n_non_replan}步, 范围 {min_non_replan_ms:.1f}~{max_non_replan_ms:.1f}ms)")
        print(f"    真实机器人预估:   {est_real_non_replan_ms:.1f}ms (仿真{avg_non_replan_step_ms:.1f}ms + 传感器/通信{real_robot_overhead_ms:.0f}ms)")
        if est_real_non_replan_ms > dt * 1000:
            print(f"                      [!!] 预估超预算 (>{dt*1000:.0f}ms)")
        else:
            print(f"                      [OK] 预估在预算内 (<{dt*1000:.0f}ms)")
    print(f"    最慢步耗时:       {max_step_ms:.1f}ms")
    print(f"  --- 执行层约束 ---")
    hit_type = "主动击球" if active_contact else ("被动接触" if passive_contact else "未触球")
    print(f"  max_qdot={exec_metrics.max_qdot_ratio:.2f}x, max_tcp={exec_metrics.max_tcp_speed:.1f}m/s, "
          f"max_face={exec_metrics.max_racket_face_speed:.1f}m/s")
    print(f"  fallback={exec_metrics.fallback_count}(terminal={terminal_fallback_count}), "
          f"emerg_stop={exec_metrics.emergency_stop_count}, "
          f"buffer_exhaust={buffer_exhaustion_count}, "
          f"mpc_steps={exec_metrics.total_mpc_steps}")
    print(f"  击球类型: {hit_type}")
    print("========================================\n")

    # 结构化结果行（供批量实验脚本解析，固定英文格式）
    hit_type_en = "active" if active_contact else ("passive" if passive_contact else "miss")
    print(f"__RESULT__: pos_error={pos_error:.6f} vel_error={vel_error:.6f} "
          f"min_dist={min_dist:.6f} ball_near_ms={ball_near_ms:.1f} "
          f"tube_ready_ms={tube_ready_ms:.1f} max_tcp={exec_metrics.max_tcp_speed:.2f} "
          f"max_qdot={exec_metrics.max_qdot_ratio:.2f} hit_type={hit_type_en} "
          f"hit_time_error_ms={hit_time_error:.1f} hit_pos_error={hit_position_error:.6f}")

    # ===== 保存轨迹 =====
    if args.dump_trajectory:
        import pickle as _pickle
        traj_data = {
            "X_history": X_history,
            "U_history": U_history,
            "ball_pos_history": ball_pos_history,
            "init_q": init_q,
            "init_q_left": init_q_left,
            "pos_error": pos_error,
            "hit_type": hit_type_en,
            "p0": p0_real if 'p0_real' in dir() else p0,
            "v0": v0_real if 'v0_real' in dir() else v0,
            "hit_step": hit_step,
            "post_hit_steps": post_hit_steps,
        }
        dump_path = Path(args.dump_trajectory)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dump_path, "wb") as _f:
            _pickle.dump(traj_data, _f)

    # ===== 可视化 =====
    if not args.no_plot:
        results_dir = Path(__file__).resolve().parent.parent.parent / "results"
        tag = f"tube_{args.seed or 'default'}"
        ball_arr = np.array(ball_pos_history)
        racket_arr = np.array([X_history[i][:3] if i < len(X_history) else np.zeros(3)
                               for i in range(len(X_history))])
        # 从环境读取球拍位置轨迹
        racket_pos_list: list[np.ndarray] = []
        env.reset(init_q)
        env.data.qpos[env.NQ:env.NQ + env.LEFT_ARM_NQ] = init_q_left
        env.update_kinematics()
        for i in range(min(len(X_history), len(U_history) + 1)):
            if i < len(X_history):
                env.set_arm_state(X_history[i])
            racket_pos_list.append(env.get_ee_pos().copy())

        racket_pos_arr = np.array(racket_pos_list)

        plot_tube_results(
            results_dir, tag,
            ball_arr, racket_pos_arr, hit_window,
            distances_history, normal_align_history,
            ball_near_history, tube_ready_history,
            k_hit_steps_history,
            pos_error_history,
        )

    # ===== 真实速度回放 =====
    if args.viewer:
        logger.info("MPC 计算完成，开始真实速度回放（含击打后球飞出）...")

        X_arr = np.array(X_history)
        U_arr = np.array(U_history) if len(U_history) > 0 else np.zeros((0, env.NU))

        env.reset(init_q)
        env.data.qpos[env.NQ:env.NQ + env.LEFT_ARM_NQ] = init_q_left
        env.data.qvel[env.NQ:env.NQ + env.LEFT_ARM_NQ] = 0.0
        env.update_kinematics()
        env.set_ball_state(p0, v0)

        X_replay = [env.get_arm_state().copy()]
        ball_replay = [env.get_ball_pos().copy()]
        rebound_applied = False

        for i, u_cmd in enumerate(U_arr):
            # 回放碰撞窗口：与 MPC 执行完全相同的逻辑
            # 用 hit_step - i 模拟 k_hit_new（剩余击球步数）
            k_hit_remaining = max(0, hit_step - i) if hit_step >= 0 else 0
            ball_pos_rp = env.get_ball_pos()
            racket_pos_rp = env.get_ee_pos()
            dist_rp = np.linalg.norm(racket_pos_rp - ball_pos_rp)
            enable_collision_rp = False
            if not rebound_applied and hit_step >= 0:
                if k_hit_remaining <= 10:
                    enable_collision_rp = True
                elif k_hit_remaining <= 30 and dist_rp < 0.35:
                    enable_collision_rp = True
            if hasattr(env, "set_arm_collision"):
                env.set_arm_collision(enable_collision_rp)
            ball_vel_pre = env.get_ball_vel().copy() if enable_collision_rp else np.zeros(3)
            env.step(u_cmd)
            if enable_collision_rp and not rebound_applied and hit_step >= 0:
                ncon = env.data.ncon
                for ci in range(ncon):
                    c = env.data.contact[ci]
                    g1 = env.model.geom(c.geom1).name
                    g2 = env.model.geom(c.geom2).name
                    if ("ball" in g1 or "ball" in g2) and ("racket" in g1 or "racket" in g2):
                        n_racket = env.get_ee_normal()
                        n_hat = n_racket / (np.linalg.norm(n_racket) + 1e-8)
                        v_ee = env.get_ee_vel()
                        v_rel_n = np.dot(ball_vel_pre - v_ee, n_hat)
                        e = 0.8
                        v_ball_new = ball_vel_pre - (1 + e) * v_rel_n * n_hat
                        env.set_ball_vel(v_ball_new)
                        rebound_applied = True
                        logger.info(
                            f"  回放弹性反弹: 球速 {np.linalg.norm(ball_vel_pre):.2f}"
                            f"->{np.linalg.norm(v_ball_new):.2f} m/s"
                        )
                        break
            X_replay.append(env.get_arm_state().copy())
            ball_replay.append(env.get_ball_pos().copy())

        X_replay = np.array(X_replay)
        ball_replay_arr = np.array(ball_replay)

        if hasattr(env, "set_arm_collision"):
            env.set_arm_collision(True)

        visualize_rm65_result(
            env, X_replay, U_arr, ball_replay_arr, config_dict,
            init_q_left=init_q_left,
            post_hit_steps=post_hit_steps,
        )


if __name__ == "__main__":
    main()

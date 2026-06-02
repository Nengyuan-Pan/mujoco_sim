"""RM-65 Tube-based Robust Hitting 实验脚本。

在 rm65_mpc_fast.py 的基础上，引入 HittingTube 鲁棒击球框架：
将确定性的单点击球扩展为一段候选击球时间窗口，在窗口内对球拍
位置、速度、拍面法向量施加分布式代价，使球拍在一小段时间内都
尽量保持可击球状态，提升对球到达时间/位置不确定性的鲁棒性。

用法:
  python scripts/rm65_mpc_tube.py --use_tube true --viewer --seed 42
  python scripts/rm65_mpc_tube.py --use_tube false --viewer  # fallback 模式
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
try:
    from src.cpp.solver_cpp import ILQTSolver
except ImportError:
    from src.ilqt.solver import ILQTSolver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
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
    backswing_offset: float = -0.6,
    backswing_ratio: float = 0.35,
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

    sigma0: float = 0.02
    """初始位置不确定性半径（米）。"""

    sigma_v: float = 0.008
    """速度不确定性增长率（米/秒/秒）。"""

    sigma_a: float = 0.001
    """加速度不确定性增长率（米/秒²/秒²）。"""

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


@dataclass
class BallTrajectoryTube:
    """带不确定性半径的球轨迹管道。"""

    positions: np.ndarray
    """球位置轨迹，形状 (N, 3)。"""

    velocities: np.ndarray
    """球速度轨迹，形状 (N, 3)。"""

    times: np.ndarray
    """时间序列，形状 (N,)。"""

    pos_sigma: np.ndarray
    """每步位置不确定性半径，形状 (N,)。"""


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

    uncertainty_radius: np.ndarray
    """每步不确定性半径，形状 (M,)。"""


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

    uncertainty_radius: np.ndarray
    """不确定性半径，形状 (M,)。"""

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
    """将确定球轨迹转换为带不确定性半径的管道。

    pos_sigma[k] = sigma0 + sigma_v * t_k + sigma_a * t_k^2

    Args:
        ball_positions: 球位置轨迹，形状 (N, 3)。
        ball_velocities: 球速度轨迹，形状 (N, 3)。
        dt: 仿真步长。
        config: Tube 配置。

    Returns:
        BallTrajectoryTube 实例。
    """
    N = len(ball_positions)
    times = np.arange(N) * dt
    pos_sigma = config.sigma0 + config.sigma_v * times + config.sigma_a * times**2
    return BallTrajectoryTube(
        positions=ball_positions.copy(),
        velocities=ball_velocities.copy(),
        times=times.copy(),
        pos_sigma=pos_sigma,
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
    candidates_u: list[float] = []

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

        candidates_k_abs.append(k)
        candidates_p.append(p_ball.copy())
        candidates_v.append(v_ball.copy())

        # 不确定性半径
        t = k * dt
        sigma = config.sigma0 + config.sigma_v * t + config.sigma_a * t**2
        candidates_u.append(sigma)

    if len(candidates_k_abs) == 0:
        # 回退：至少包含 best_k_abs
        k = best_k_abs
        if 1 <= k <= len(ball_positions):
            p_ball = ball_positions[k - 1]
            v_ball = ball_velocities[k - 1]
            t = k * dt
            sigma = config.sigma0 + config.sigma_v * t + config.sigma_a * t**2
            candidates_k_abs.append(k)
            candidates_p.append(p_ball.copy())
            candidates_v.append(v_ball.copy())
            candidates_u.append(sigma)

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
        uncertainty_radius=np.array(candidates_u),
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
        uncertainty_radius=hit_window.uncertainty_radius.copy(),
        best_k=hit_window.best_k,
    )


# ==============================================================================
# TubeHittingCostWrapper — 与现有 iLQT solver 兼容的 Tube 代价包装器
# ==============================================================================

class TubeHittingCostWrapper:
    """包装 HittingCost，在候选击球窗口内施加空间走廊式 tube 代价。

    设计思路（空间重合，而非时间追踪）：
    - 提取球在窗口内的轨迹线方向 d_ball，构建"空间走廊"
    - 在 tube 窗口内的每个 iLQR 步 k，注入三类代价：
      1. 垂直偏离代价：球拍到球轨迹线的垂直距离（hinge loss），
         不绑定"第 k 步必须到 p_ball(k)"的时间-空间对应
      2. 速度方向代价：球拍速度在垂直于 d_ball 方向的分量应尽量小，
         鼓励球拍沿球轨迹线方向运动（相向扫过），实现空间轨迹重合
      3. 法向量代价：拍面朝向来球方向
    - 终端代价保留原有 HittingCost（乘以 1 - tube_ratio），位置部分放宽
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
        self._current_ratio = config.tube_cost_ratio  # 当前有效比例（可随时间衰减）
        self._anchor_alpha: float = 0.9  # 终端锚定强度（0=全约束, 1=沿d_ball完全自由）
        self._Q_p_tube = config.Q_p_tube
        self._Q_v_tube = config.Q_v_tube
        self._Q_n_tube = config.Q_n_tube

        self._tube_steps: set[int] = set()
        self._tube_weight_scales: dict[int, float] = {}
        self._d_ball: np.ndarray = np.zeros(3)
        self._P_perp: np.ndarray = np.zeros((3, 3))
        self._p_ball_ref: np.ndarray = np.zeros(3)
        self._n_des_common: np.ndarray = np.zeros(3)
        self._max_uncertainty: float = 0.0
        self._rebuild_tube_maps(hitting_tube, horizon)

    def _rebuild_tube_maps(self, tube: HittingTube, horizon: int) -> None:
        """重建 tube 步集合、权重缓存和空间走廊几何信息。"""
        self._tube_steps.clear()
        self._tube_weight_scales.clear()

        if len(tube.k_candidates) == 0:
            self._d_ball = np.zeros(3)
            self._P_perp = np.zeros((3, 3))
            self._p_ball_ref = np.zeros(3)
            self._n_des_common = np.zeros(3)
            self._max_uncertainty = 0.0
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

        # 最大不确定性半径（走廊半宽）
        self._max_uncertainty = float(np.max(tube.uncertainty_radius))

    def running_cost(self, x: np.ndarray, u: np.ndarray, k: int | None = None) -> float:
        """计算运行代价 = 原始运行代价 + tube 代价（若 k 在候选窗口内）。"""
        cost = self.base_cost.running_cost(x, u, k)
        if k is not None and k in self._tube_steps:
            cost += self._compute_tube_cost_at_k(x, k)
        return cost

    def terminal_cost(self, x: np.ndarray) -> float:
        """终端代价 = (1 - tube_ratio) * 软投影终端代价。

        位置代价：垂直球轨迹方向全惩罚，沿球轨迹方向保留 10% 弱锚定。
        效果：球拍目标是一个线段而非一个点，可沿球轨迹线扫过，
        但又不会完全漂移。
        """
        # 软投影矩阵：锚定强度由 _anchor_alpha 控制
        # alpha=0.9: 沿d_ball保留10%锚定（自由扫过）
        # alpha=0.3: 沿d_ball保留70%锚定（精确定位）
        d_outer = np.outer(self._d_ball, self._d_ball)
        alpha = self._anchor_alpha
        P_soft = np.eye(3) - alpha * d_outer
        if np.linalg.norm(self._d_ball) < 1e-6:
            P_soft = np.eye(3)

        self.env.set_arm_state(x)
        p_ee = self.env.get_ee_pos()
        v_ee = self.env.get_ee_vel()
        n_rack = self.env.get_ee_normal()

        dp = p_ee - self.base_cost.p_hit
        cost_p = 0.5 * float((P_soft @ dp) @ self.base_cost.Q_p @ (P_soft @ dp))

        dv = v_ee - self.base_cost.v_hit
        cost_v = 0.5 * float(dv @ self.base_cost.Q_v @ dv)

        cost = cost_p + cost_v
        if self.base_cost.n_des is not None and self.base_cost.Q_n > 0:
            n_err = n_rack - self.base_cost.n_des
            cost += 0.5 * self.base_cost.Q_n * float(n_err @ n_err)
        return (1.0 - self._current_ratio) * cost

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
        """终端代价导数 = (1 - tube_ratio) * 软投影终端导数。"""
        d_outer = np.outer(self._d_ball, self._d_ball)
        alpha = self._anchor_alpha
        P_soft = np.eye(3) - alpha * d_outer
        if np.linalg.norm(self._d_ball) < 1e-6:
            P_soft = np.eye(3)

        n_x = self.env.NX
        n_q = self.env.NQ

        self.env.set_arm_state(x)
        p_ee = self.env.get_ee_pos()
        v_ee = self.env.get_ee_vel()
        n_rack = self.env.get_ee_normal()
        J_p = self.env.get_ee_jacp()

        K_p = P_soft.T @ self.base_cost.Q_p @ P_soft
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

        scale = 1.0 - self._current_ratio
        return scale * l_x, scale * l_xx

    def update_target(self, p_hit: np.ndarray, v_hit: np.ndarray, n_des: np.ndarray | None = None) -> None:
        """委托给 base_cost 更新终端目标。"""
        self.base_cost.update_target(p_hit, v_hit, n_des=n_des)

    def update_weights(self, Q_p_scale: float = 1.0, Q_v_scale: float = 1.0) -> None:
        """委托给 base_cost 更新权重。"""
        self.base_cost.update_weights(Q_p_scale, Q_p_scale)

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

        三项代价：
        1. 垂直偏离代价（hinge loss）：球拍到球轨迹线的垂直距离
        2. 速度方向代价：球拍速度垂直于球轨迹方向的分量
        3. 法向量代价：拍面法向与期望法向的对齐程度
        """
        self.env.set_arm_state(x)
        p_ee = self.env.get_ee_pos()
        v_ee = self.env.get_ee_vel()
        n_rack = self.env.get_ee_normal()

        # 1. 垂直偏离代价（hinge loss）
        dp = p_ee - self._p_ball_ref
        perp_dist = float(np.linalg.norm(self._P_perp @ dp))
        margin = perp_dist - self.RACKET_RADIUS - self._max_uncertainty
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

        # ---- 1. 垂直偏离代价 (hinge loss on perp distance) ----
        dp = p_ee - self._p_ball_ref
        dp_perp = self._P_perp @ dp
        perp_dist = float(np.linalg.norm(dp_perp))
        margin = perp_dist - self.RACKET_RADIUS - self._max_uncertainty

        if margin > 0 and perp_dist > 1e-8:
            # d(perp_dist)/dp = dp_perp / perp_dist  (因为 dp_perp 已经是垂直分量)
            # dp_perp / perp_dist 是垂直平面内的单位方向
            dir_perp = dp_perp / perp_dist
            # l_x 的位置部分: Q_p * margin * J_p^T @ dir_perp
            Jp_dir = J_p.T @ dir_perp
            l_x_tube[:n_q] += self._Q_p_tube * margin * Jp_dir
            # l_xx 的位置部分: Q_p * outer(J_p^T @ dir_perp, J_p^T @ dir_perp)
            l_xx_tube[:n_q, :n_q] += self._Q_p_tube * np.outer(Jp_dir, Jp_dir)

        # ---- 2. 速度方向代价 (v_perp = P_perp @ v_ee) ----
        # v_perp = P_perp @ v_ee
        # dv_perp/dqdot = P_perp @ J_p  (因为 v_ee ≈ J_p @ qdot，位置雅可比也可用于线速度)
        Jp_perp = self._P_perp @ J_p  # (3, n_q)
        v_perp = self._P_perp @ v_ee   # (3,)
        # l_x 速度部分: Q_v * Jp_perp^T @ v_perp  (作用于 qdot 部分，即 x[n_q:])
        l_x_tube[n_q:] += self._Q_v_tube * (Jp_perp.T @ v_perp)
        # l_xx 速度部分: Q_v * Jp_perp^T @ Jp_perp
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
    parser.add_argument("--r-decay", type=float, default=0.30, help="R 衰减占比")
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
    parser.add_argument("--replan-interval", type=int, default=20, help="重规划间隔步数")
    parser.add_argument("--window-ms", type=float, default=50.0, help="Tube 候选窗口半宽 (ms)")
    parser.add_argument("--sigma0", type=float, default=0.02, help="初始不确定性半径 (m)")
    parser.add_argument("--tube-cost-ratio", type=float, default=0.3, help="Tube 代价占比 (0~1)")
    parser.add_argument("--no-plot", action="store_true", help="禁用 matplotlib 可视化")
    parser.add_argument("--time-perturb-ms", type=float, default=0.0,
                        help="球到达时间预测扰动 (ms): 正值=MPC认为球早到，负值=认为球晚到")
    parser.add_argument("--space-perturb-m", type=float, default=0.0,
                        help="击打点空间偏移 (m): 对 p_hit 施加侧向偏移，测试 tube 空间走廊的鲁棒性")
    args = parser.parse_args()

    use_tube = args.use_tube.lower() in ("true", "1", "yes")
    use_analytical = not args.fd
    time_perturb_s = args.time_perturb_ms / 1000.0
    space_perturb_m = args.space_perturb_m

    # 加载配置
    base_path = Path(__file__).resolve().parent.parent / "configs"
    config_dict = load_config(base_path / "default.yaml")
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
    replan_interval = 20
    max_iter_per_plan = 5
    Q_p_scale_far = 5.0
    Q_v_scale_far = 3.0
    Q_p_scale_near = 5.0
    Q_v_scale_near = 50.0
    first_plan_iters = 15
    near_plan_iters = 8

    if args.horizon is not None:
        fixed_horizon = args.horizon
    if args.iter is not None:
        max_iter_per_plan = args.iter
    if args.replan_interval is not None:
        replan_interval = args.replan_interval

    # serve-box 模式：球从 8m 远处飞来，需更长规划前瞻且较少迭代（避免球状态漂移）
    if args.serve_box:
        if args.horizon is None:
            fixed_horizon = 80
        if args.iter is None:
            max_iter_per_plan = 3
        first_plan_iters = max(first_plan_iters, 20)
        total_horizon = max(total_horizon, 250)
        logger.info(f"serve-box auto params: horizon={fixed_horizon}, iter={max_iter_per_plan}, "
                     f"first_plan_iters={first_plan_iters}, total_horizon={total_horizon}")

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
        sigma0=args.sigma0,
        Q_p_tube=float(config_dict["cost"]["Q_p"][0]) * 2.0,
        Q_v_tube=float(config_dict["cost"]["Q_v"][0]) * 50.0,  # 强化速度方向对齐
        Q_n_tube=args.normal_weight,
        tube_cost_ratio=args.tube_cost_ratio,
    )

    # 初始化 RM-65 环境
    model_path = Path(__file__).resolve().parent.parent / "src" / "robot" / "rm65_model.xml"
    env = RM65Env(model_path, dt=dt)
    env.init_q_left = init_q_left

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
    target_offset = 0.0

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

    far_threshold = k_hit_total
    near_threshold = max(40, k_hit_total // 4)

    hit_direction = np.array(config_dict["hitting"]["hit_direction"], dtype=np.float64)
    racket_speed = float(config_dict["hitting"]["racket_speed"])
    v_hit_desired = compute_desired_hit_velocity(hit_direction, racket_speed)

    hit_shift = args.hit_shift
    d_hat = hit_direction / (np.linalg.norm(hit_direction) + 1e-8)
    p_follow = p_hit + hit_shift * d_hat

    # ===== 构建初始 Tube（若启用） =====
    hit_window: HitWindow | None = None
    hitting_tube: HittingTube | None = None
    if use_tube:
        hit_window = search_hit_window(
            env, p0, v0, shoulder_pos, workspace_radius,
            k_hit_total + 30, tube_cfg,
            ball_direction="y",
            current_step=0,
        )
        if hit_window is not None:
            hitting_tube = build_hitting_tube(
                hit_window, racket_speed, hit_direction, tube_cfg,
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
    if hit_shift > 0:
        logger.info(f"随挥偏移: {hit_shift:.3f}m, 随挥目标: {p_follow}")
    logger.info(f"Tube 模式: {'启用' if use_tube else '禁用'}")
    logger.info(f"线性化: {'解析' if use_analytical else '有限差分'}, horizon={fixed_horizon}")

    # ===== 初始化 =====
    Q_p = np.array(config_dict["cost"]["Q_p"], dtype=np.float64) * 2.0
    Q_v = np.array(config_dict["cost"]["Q_v"], dtype=np.float64) * 2.0
    R = float(config_dict["cost"]["R"])
    ilqt_cfg = dict(config_dict["ilqt"])

    p_target_init = p_follow

    if use_backswing:
        U_prev, q_des_traj_init = generate_backswing_warm_start(
            env, x0, p_target_init, v_hit_desired, k_hit_total,
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
            env, x0, p_target_init, k_hit_total, gain=30.0,
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

    # 创建基础代价函数
    base_cost_fn = HittingCost(
        env, p_follow, v_hit_desired, Q_p, Q_v, R,
        Q_p_running=0.0,
        R_joint_scale=r_joint_scale if r_joint_scale else None,
        q_des_traj=q_des_traj_init,
        Q_joint=Q_joint,
        R_schedule=R_schedule_init,
        Q_n=args.normal_weight,
        n_des=n_des_single,
    )

    # 创建 Tube 代价包装器（若启用）
    if use_tube and hitting_tube is not None:
        cost_fn = TubeHittingCostWrapper(
            env, base_cost_fn, hitting_tube, k_hit_total, tube_cfg,
        )
        logger.info("TubeHittingCostWrapper 已创建")
    else:
        cost_fn = base_cost_fn  # type: ignore[assignment]
        logger.info("使用标准 HittingCost（single-hit-point）")

    solver = ILQTSolver(ilqt_cfg, use_analytical=use_analytical)

    # 设置球的初始状态（始终使用原始 p0, v0）
    env.reset(init_q)
    env.data.qpos[env.NQ:env.NQ + env.LEFT_ARM_NQ] = init_q_left
    env.data.qvel[env.NQ:env.NQ + env.LEFT_ARM_NQ] = 0.0
    env.update_kinematics()
    env.set_ball_state(p0, v0)
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

    U_buffer: np.ndarray = np.zeros((0, env.NU))
    buffer_idx: int = 0
    is_first_plan: bool = True
    p_hit_new = p_hit.copy()
    k_hit_new = k_hit_total
    iters = 0
    hit_step = -1
    p_ee_at_hit = None
    q_ik_cache: np.ndarray | None = None
    ball_was_hit = False
    current_n_des = n_des_single
    v_ball_hit_new = v_ball_hit.copy()

    logger.info(f"开始 MPC 循环，总步数={total_horizon}，击打步数={k_hit_total}")

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

        if need_replan:
            t_replan_start = time.perf_counter()
            remaining_horizon = total_horizon - step

            # 始终使用实际球观测进行规划（tube 走廊基于正确轨迹）
            hit_info_new = find_hitting_point_physics(
                env, ball_pos, ball_vel, shoulder_pos, workspace_radius, remaining_horizon
            )

            if hit_info_new is None:
                logger.info(f"步 {step}: 球不再可达, 停止 MPC")
                break

            k_hit_candidate = hit_info_new["k_hit"]
            if k_hit_candidate < max(10, k_hit_new // 4) and k_hit_new > 30:
                k_hit_candidate = max(1, k_hit_new - replan_interval)

            # 对击球时刻施加时间预测扰动（仅偏移 k_hit，不改变 p_hit 和球轨迹）
            # 正扰动 = MPC 认为球早到 → k_hit 减小 → iLQR 终端步提前
            # 负扰动 = MPC 认为球晚到 → k_hit 增大 → iLQR 终端步延后
            # 球实际轨迹、tube 空间走廊均不受影响
            if abs(time_perturb_s) > 1e-6:
                perturb_steps = int(round(time_perturb_s / dt))
                k_hit_candidate = k_hit_candidate - perturb_steps
                k_hit_candidate = max(5, min(k_hit_candidate, remaining_horizon - 1))

            p_hit_new = hit_info_new["p_hit"]
            v_ball_hit_new = hit_info_new["v_ball_hit"]
            k_hit_new = k_hit_candidate

            # 对击打点施加空间偏移（仅偏移 p_hit，球轨迹和 tube 走廊不变）
            # 测试 tube 空间走廊能否覆盖终端目标偏移
            if abs(space_perturb_m) > 1e-6:
                d_ball_hit = v_ball_hit_new / (np.linalg.norm(v_ball_hit_new) + 1e-8)
                lateral = np.cross(d_ball_hit, np.array([0.0, 0.0, 1.0]))
                lateral_norm = np.linalg.norm(lateral)
                if lateral_norm > 1e-6:
                    lateral /= lateral_norm
                else:
                    lateral = np.array([1.0, 0.0, 0.0])
                p_hit_new = p_hit_new + lateral * space_perturb_m

            n_des_new = -v_ball_hit_new / (np.linalg.norm(v_ball_hit_new) + 1e-8)
            if args.normal_flip:
                n_des_new = -n_des_new
            current_n_des = n_des_new
            q_ik_cache = None

            p_follow_new = p_hit_new + hit_shift * d_hat

            k_hit_steps_history.append(k_hit_new)

            # ---- 更新 Tube ----
            # 先计算 iLQR 规划地平线（需要知道 horizon_plan 以便 tube 映射）
            horizon_full = k_hit_new
            horizon_plan = min(horizon_full, fixed_horizon)

            if use_tube:
                # 如果已经是纯终端模式，跳过 tube 重建
                if isinstance(cost_fn, TubeHittingCostWrapper):
                    hit_window = search_hit_window(
                        env, ball_pos, ball_vel, shoulder_pos, workspace_radius,
                        remaining_horizon, tube_cfg,
                        ball_direction="y",
                        current_step=0,
                    )
                    if hit_window is not None:
                        hitting_tube = build_hitting_tube(
                            hit_window, racket_speed, hit_direction, tube_cfg,
                        )
                        cost_fn.update_hitting_tube(hitting_tube, horizon=horizon_plan)
                        cost_fn.update_target(p_follow_new, v_hit_desired, n_des=n_des_new)
                    else:
                        cost_fn = base_cost_fn  # type: ignore[assignment]
                        cost_fn.update_target(p_follow_new, v_hit_desired, n_des=n_des_new)
                        logger.info(f"步 {step}: Tube 构建失败，回退到 single-hit-point")
                else:
                    # 之前已切换到纯终端模式
                    cost_fn.update_target(p_follow_new, v_hit_desired, n_des=n_des_new)
            else:
                cost_fn.update_target(p_follow_new, v_hit_desired, n_des=n_des_new)

            # ---- 渐进衰减策略：tube_ratio 和 anchor_alpha 随 k_hit 平滑变化 ----
            # 距离远时：tube 强引导 + 终端弱锚定（允许沿球轨迹扫过）
            # 距离近时：tube 弱引导 + 终端强锚定（精确定位到 p_hit）
            if isinstance(cost_fn, TubeHittingCostWrapper) and k_hit_new <= far_threshold:
                # ratio: k_hit=50→0.3, k_hit=40→0.15, k_hit≤30→0.0
                ratio_decay = max(0.0, min(1.0, (k_hit_new - 30) / 20))
                eff_ratio = tube_cfg.tube_cost_ratio * ratio_decay
                # anchor_alpha: k_hit=50→0.9(10%锚), k_hit≤30→0.3(70%锚)
                eff_anchor = 0.9 - 0.6 * (1.0 - ratio_decay)
                cost_fn.update_tube_params(eff_ratio, eff_anchor)

            if k_hit_new > far_threshold:
                p_target_jt = p_follow_new if use_tube and hitting_tube is not None else p_follow_new
                u_jt = compute_jacobian_init_control(
                    env, x_current, p_target_jt, replan_interval, gain=60.0,
                    fix_joint5_angle=fix_joint5_angle,
                )
                U_buffer = u_jt
                buffer_idx = 0
                U_prev = np.zeros((0, env.NU))
                iters = 0
                replan_times.append(time.perf_counter() - t_replan_start)
            else:
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

                if use_r_decay:
                    R_schedule_new = compute_r_schedule(
                        horizon_full, R, decay_ratio=r_decay_ratio,
                    )[:horizon_plan]
                    cost_fn.set_R_schedule(R_schedule_new)
                else:
                    cost_fn.set_R_schedule(None)

                iters_plan = max_iter_per_plan
                skip_ls = True
                if is_first_plan:
                    iters_plan = first_plan_iters
                    skip_ls = False
                    is_first_plan = False
                elif k_hit_new <= near_threshold:
                    iters_plan = near_plan_iters

                if use_backswing:
                    q_hit_new_ik = env.solve_ik(
                        p_hit_new, q_init=x_current[:env.NQ],
                        max_iter=150, eps=1e-3,
                    )
                    if fix_joint5_angle is not None:
                        q_hit_new_ik[5] = fix_joint5_angle

                    env.set_arm_state(np.concatenate([q_hit_new_ik, np.zeros(env.NQ)]))
                    J_p_new = env.get_ee_jacp()
                    qdot_hit_new = np.linalg.lstsq(J_p_new, v_hit_desired, rcond=None)[0]
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

                ball_pos_save, ball_vel_save = env.get_ball_state()

                X_mpc, U_mpc, iter_costs = solver.solve_few_iters(
                    env, cost_fn, x_current, U_warm,
                    max_iter=iters_plan,
                    skip_linesearch=skip_ls,
                )

                if fix_joint5_angle is not None:
                    U_mpc = fix_joint5_control_trajectory(
                        U_mpc, x_current, env, fix_joint5_angle,
                    )

                env.set_ball_state(ball_pos_save, ball_vel_save)
                env.set_arm_state(x_current)

                if len(iter_costs) > 0:
                    cost_history.append(iter_costs[-1])

                if len(U_mpc) > replan_interval:
                    U_prev = U_mpc[replan_interval:]
                elif len(U_mpc) > 0:
                    U_prev = U_mpc[1:]
                else:
                    U_prev = np.zeros((0, env.NU))

                U_buffer = U_mpc[:replan_interval]
                buffer_idx = 0
                iters = iters_plan
                replan_times.append(time.perf_counter() - t_replan_start)

        if buffer_idx < len(U_buffer):
            u_cmd = U_buffer[buffer_idx]
            buffer_idx += 1
        else:
            u_cmd = np.zeros(env.NU)
            if k_hit_new > 0:
                # 简单的雅可比转置后备
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

        # ---- 碰撞检测：仅在球物理靠近时启用，避免远距离虚假碰撞 ----
        enable_collision = False
        if not ball_was_hit:
            dist_threshold = 0.25
            if k_hit_new <= 8 and dist_cur < dist_threshold:
                enable_collision = True
            elif k_hit_new <= 5:
                enable_collision = True  # 末段无条件启用

        if hasattr(env, "set_arm_collision"):
            env.set_arm_collision(enable_collision)

        ball_vel_before_step = ball_vel.copy() if enable_collision else ball_vel
        x_current, ball_pos, ball_vel = env.step_full(u_cmd)

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
                            logger.info(
                                f"步 {step}: 球拍击球! {g1}<->{g2}, "
                                f"球拍速度={ee_speed:.2f}m/s, 球速={ball_spd:.2f}m/s"
                            )

        if hasattr(env, "set_arm_collision"):
            env.set_arm_collision(True)

        X_history.append(x_current.copy())
        U_history.append(u_cmd.copy())
        ball_pos_history.append(ball_pos.copy())

        env.update_kinematics()
        pos_err = np.linalg.norm(env.get_ee_pos() - p_hit_new)
        pos_error_history.append(pos_err)

        step_time = time.perf_counter() - t_step_start
        step_times.append(step_time)

        if step % 20 == 0 or k_hit_new <= 5:
            tube_info = ""
            if use_tube and hitting_tube is not None:
                n_valid = sum(valid_hit_history[-20:]) if len(valid_hit_history) >= 20 else sum(valid_hit_history)
                tube_info = f", valid_hit={n_valid}"
            logger.info(
                f"步 {step}: 剩余={k_hit_new}, 误差={pos_err:.4f}m, "
                f"距离={dist_cur:.4f}m, 迭代={iters}, 步耗时={step_time*1000:.1f}ms{tube_info}"
            )

        if ball_racket_hit and not ball_was_hit:
            ball_was_hit = True
            hit_step = step
            env.update_kinematics()
            p_ee_at_hit = env.get_ee_pos().copy()

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

        if ball_was_hit and (step - hit_step) >= 5:
            logger.info(f"步 {step}: 碰撞完成，停止MPC")
            break

        if k_hit_new <= 1:
            logger.info(f"步 {step}: 到达击打时刻")
            hit_step = step if hit_step < 0 else hit_step
            env.update_kinematics()
            if p_ee_at_hit is None:
                p_ee_at_hit = env.get_ee_pos().copy()
            break

    # ===== 击打后继续仿真 =====
    post_hit_steps = 80
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
    n_steps = len(U_history)
    avg_step_ms = np.mean(step_times) * 1000 if step_times else 0
    avg_replan_ms = np.mean(replan_times) * 1000 if replan_times else 0
    max_step_ms = np.max(step_times) * 1000 if step_times else 0
    real_time_ratio = (n_steps * dt) / t_total if t_total > 0 else 0

    logger.info(
        f"MPC 完成: 总耗时={t_total:.2f}s, 平均每步={avg_step_ms:.1f}ms, "
        f"平均重规划={avg_replan_ms:.1f}ms, 最慢步={max_step_ms:.1f}ms, "
        f"实时比率={real_time_ratio:.2f}x"
    )

    if p_ee_at_hit is not None:
        p_ee_final = p_ee_at_hit
    else:
        env.set_arm_state(x_current)
        p_ee_final = env.get_ee_pos()
    v_ee_final = env.get_ee_vel()
    pos_error = np.linalg.norm(p_ee_final - p_hit)
    vel_error = np.linalg.norm(v_ee_final - v_hit_desired)

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
        hit_position_error = float(np.linalg.norm(p_ee_at_hit - p_hit))
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
    print("\n========================================")
    if pos_error < 0.05:
        print("  RM-65 击打成功！（误差 < 5cm）")
    elif pos_error < 0.1:
        print("  RM-65 击打基本命中！（误差 < 10cm）")
    else:
        print("  RM-65 击打偏差较大，需要调整参数。")

    print(f"  Tube 模式: {'启用' if use_tube or hitting_tube is not None else '禁用'}")
    if initial_tube_n_candidates > 0:
        print(f"  初始候选窗口: {initial_tube_n_candidates} 步 {initial_tube_k_range} "
              f"(半宽 {args.window_ms:.0f}ms)")
    if abs(args.time_perturb_ms) > 0.01:
        print(f"  时间扰动: {args.time_perturb_ms:+.1f} ms")
    if abs(args.space_perturb_m) > 0.001:
        print(f"  球空间偏移: {args.space_perturb_m:+.3f} m (实际球轨迹侧偏，MPC用原始预测)")
    print(f"  击打目标位置: {np.round(p_hit_new, 3)}")
    print(f"  末端实际位置: {np.round(p_ee_final, 3)}")
    print(f"  位置误差: {pos_error:.4f} m")
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
    print(f"  总计算时间: {t_total:.2f}s")
    print(f"  平均每步: {avg_step_ms:.1f}ms (实时需 {dt*1000:.1f}ms)")
    print(f"  实时比率: {real_time_ratio:.2f}x")
    print("========================================\n")

    # ===== 可视化 =====
    if not args.no_plot:
        results_dir = Path(__file__).resolve().parent.parent / "results"
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
            enable_collision = (hit_step >= 0 and abs(i - hit_step) <= 5)
            if hasattr(env, "set_arm_collision"):
                env.set_arm_collision(enable_collision)
            ball_vel_pre = env.get_ball_vel().copy() if enable_collision else np.zeros(3)
            env.step(u_cmd)
            env._handle_ball_bounce()
            if enable_collision and not rebound_applied and hit_step >= 0:
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

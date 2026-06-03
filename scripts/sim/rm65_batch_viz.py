"""RM-65 批量网球击打 + 回放 + MP4 录制脚本。

连续多回合随机发球，每回合执行完整 MPC 规划并收集回放轨迹数据，
规划完成后在一个 MuJoCo 查看器会话中按顺序逐段播放，
并可选用离屏渲染导出 MP4 视频。

用法：
  python scripts/rm65_batch_viz.py --viewer --save-video
  python scripts/rm65_batch_viz.py --episodes 5 --seed 10 --no-viewer --save-video
"""

import sys
import time
import argparse
import logging
import numpy as np
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.sim.rm65_env import RM65Env
from src.tennis.ball import (
    generate_ball_to_target_box,
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


def fix_joint5_control(
    u: np.ndarray,
    q_fixed: float,
    x_current: np.ndarray,
    nq: int,
    kp: float = 300.0,
    kd: float = 30.0,
) -> np.ndarray:
    """将第 6 关节（索引 5）的控制力矩替换为 PD 保持力矩。

    Args:
        u: 原始控制力矩，形状 (6,) 或 (N, 6)。
        q_fixed: 第 6 关节固定角度。
        x_current: 当前臂状态，形状 (12,)。
        nq: 关节数（6）。
        kp: 位置增益。
        kd: 速度增益。

    Returns:
        修改后的控制力矩。
    """
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
    """将整个控制序列的第 6 关节替换为 PD 保持力矩。

    通过逐步仿真获取每步的状态来计算 PD。

    Args:
        U: 原始控制序列，形状 (N, 6)。
        x0: 初始臂状态，形状 (12,)。
        env: 环境实例。
        q_fixed: 第 6 关节固定角度。
        kp: 位置增益。
        kd: 速度增益。

    Returns:
        修改后的控制序列。
    """
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
    """基于雅可比转置法计算初始控制序列。

    Args:
        env: RM-65 环境实例。
        x0: 初始臂状态，形状 (12,)。
        p_hit: 目标击打位置，形状 (3,)。
        horizon: 规划步数。
        gain: 雅可比转置增益。
        fix_joint5_angle: 若非 None，将第 6 关节固定在此角度。

    Returns:
        初始控制序列，形状 (horizon, 6)。
    """
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


def ik_pd_step(
    env: RM65Env,
    x_current: np.ndarray,
    p_target: np.ndarray,
    q_ref: np.ndarray | None = None,
    kp: float = 120.0,
    kd: float = 12.0,
    fix_joint5_angle: float | None = None,
) -> np.ndarray:
    """基于 IK + PD 的单步控制（远距模式）。

    Args:
        env: RM-65 环境实例。
        x_current: 当前右臂状态，形状 (12,)。
        p_target: 目标位置，形状 (3,)。
        q_ref: IK 初始猜测，形状 (6,)。若为 None 则用当前 q。
        kp: PD 位置增益。
        kd: PD 速度增益。
        fix_joint5_angle: 若非 None，将第 6 关节固定在此角度。

    Returns:
        单步控制力矩，形状 (6,)。
    """
    q_ref = q_ref if q_ref is not None else x_current[:env.NQ]
    q_target = env.solve_ik(p_target, q_init=q_ref, max_iter=100, eps=2e-3)

    if fix_joint5_angle is not None:
        q_target[5] = fix_joint5_angle

    q_err = q_target - x_current[:env.NQ]
    qdot_err = -x_current[env.NQ:]
    tau = kp * q_err + kd * qdot_err

    ctrl_lo = env.model.actuator_ctrlrange[:env.NU, 0]
    ctrl_hi = env.model.actuator_ctrlrange[:env.NU, 1]
    return np.clip(tau, ctrl_lo, ctrl_hi)


def compute_r_schedule(
    steps_remaining: int,
    base_R: float,
    decay_ratio: float = 0.30,
    joint1_extra_decay: float = 10.0,
) -> np.ndarray:
    """生成 R 退火调度 — 前段恒定，后段衰减到零，关节1衰减更快。

    Args:
        steps_remaining: 剩余步数。
        base_R: 基础控制代价（来自 config）。
        decay_ratio: 衰减段占总步数的比例，默认 0.30。
        joint1_extra_decay: 关节1额外衰减倍率，默认 10。

    Returns:
        R_schedule: 形状 (steps_remaining, 6)，每步每关节的控制代价权重。
    """
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


def resample_control_sequence(
    U_old: np.ndarray,
    new_horizon: int,
) -> np.ndarray:
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


def compute_joint1_backswing_trajectory(
    q1_current: float,
    qdot1_current: float,
    q1_hit: float,
    qdot1_hit: float,
    horizon: int,
    backswing_offset: float = -0.6,
    backswing_ratio: float = 0.35,
) -> np.ndarray:
    """生成关节1的"后摆→前挥"五次多项式轨迹。

    Args:
        q1_current: 关节1当前角度 (rad)。
        qdot1_current: 关节1当前速度 (rad/s)。
        q1_hit: 击打位关节1角度 (rad)。
        qdot1_hit: 击打位关节1速度 (rad/s)。
        horizon: 规划步数。
        backswing_offset: 后摆角度偏移（负值=远离球方向），默认 -0.6。
        backswing_ratio: 后摆占比 (0~1)，默认 0.35。

    Returns:
        q1_traj: 形状 (horizon,)，关节1期望角度。
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
    """生成带后摆的关节空间轨迹 + PD 跟踪初始控制序列。

    Args:
        env: RM-65 环境实例。
        x0: 初始右臂状态，形状 (12,)。
        p_hit: 目标击打位置，形状 (3,)。
        v_hit_desired: 期望击打速度，形状 (3,)。
        horizon: 规划步数。
        backswing_offset: 后摆角度偏移。
        backswing_ratio: 后摆占比。
        kp: PD 位置增益。
        kd: PD 速度增益。
        fix_joint5_angle: 若非 None，固定第 6 关节。
        n_des: 期望拍面法向量。

    Returns:
        (U_warm, q_des_traj): 控制序列 (N, 6)，关节轨迹 (N, 6)。
    """
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

    q1_current = x0[0]
    qdot1_current = x0[NQ]
    q1_traj = compute_joint1_backswing_trajectory(
        q1_current, qdot1_current,
        q_hit[0], qdot_hit[0],
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

        err_joints = q_des_k - x[:NQ]
        err_joints_dot = qdot_des_k - x[NQ:]
        tau = kp * err_joints + kd * err_joints_dot

        if fix_joint5_angle is not None:
            tau[5] = 300.0 * (fix_joint5_angle - x[:NQ][5]) - 30.0 * x[NQ:][5]

        tau = np.clip(tau, ctrl_lo, ctrl_hi)
        U[k] = tau

        x = env.step_from_state(x, U[k])

    if has_collision_ctrl:
        env.set_arm_collision(True)

    return U, q_des_traj


def setup_camera_lights(
    model,
    cam_distance: float = 3.5,
    cam_elevation: float = -15,
    cam_azimuth: float = 135,
) -> None:
    """配置模型灯光（四灯布光方案）。

    Args:
        model: MuJoCo 模型实例。
        cam_distance: 相机距离。
        cam_elevation: 相机仰角。
        cam_azimuth: 相机方位角。
    """
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


def run_single_episode(
    env: RM65Env,
    episode_idx: int,
    seed: int,
    config: dict,
    args: argparse.Namespace,
    use_analytical: bool,
) -> dict | None:
    """运行单回合 MPC 规划并收集回放数据。

    Args:
        env: RM-65 环境实例。
        episode_idx: 回合索引（0 起始）。
        seed: 本回合随机种子。
        config: 合并后的配置字典。
        args: 命令行参数。
        use_analytical: 是否使用解析线性化。

    Returns:
        回合结果字典，若球不可达则返回 None。
    """
    dt = float(config["sim"]["dt"])
    g = np.array(config["ball"]["gravity"], dtype=np.float64)

    shoulder_pos = np.array([-0.1, -0.22693, 1.302645], dtype=np.float64)
    workspace_radius = 0.90

    total_horizon = 200
    _fixed_horizon = 40
    horizon_cap = args.horizon_cap
    replan_interval = args.replan_interval
    max_iter_per_plan = 5
    Q_p_scale_far = 5.0
    Q_v_scale_far = 3.0
    Q_p_scale_near = args.q_p_near
    Q_v_scale_near = 50.0
    first_plan_iters = args.first_iters
    near_plan_iters = args.near_iters

    init_q = np.array([-1.5, 1.57, -0.236, 0.404, 0.446, 2.45], dtype=np.float64)
    init_q_left = np.array([-0.373, -1.57, 0.236, -0.404, -0.446, -2.45], dtype=np.float64)

    fix_joint5_angle: float | None = init_q[5]

    use_backswing = True
    backswing_offset = -abs(args.backswing)
    backswing_ratio = args.bs_ratio

    use_r_decay = True
    r_decay_ratio = args.r_decay

    hit_shift = args.hit_shift

    x0 = np.zeros(env.NX)
    x0[:env.NQ] = init_q

    rng = np.random.default_rng(seed)

    env.reset(init_q)
    env.data.qpos[env.NQ:env.NQ + env.LEFT_ARM_NQ] = init_q_left
    env.data.qvel[env.NQ:env.NQ + env.LEFT_ARM_NQ] = 0.0
    env.update_kinematics()
    p_racket = env.get_ee_pos().copy()
    logger.info(f"[回合 {episode_idx+1}] 球拍初始位置: {p_racket}")

    target_center = np.array([-0.82765693, -0.47411682, 0.86947444])
    target_offset = 0.10

    hit_time = total_horizon * dt * rng.uniform(0.3, 0.4)
    p0, v0, p_hit_expected = generate_ball_to_target_box(
        target_center, target_offset, hit_time, g,
        shoulder_pos=shoulder_pos, workspace_radius=workspace_radius,
        ball_speed=None,
        rng=rng,
        ball_direction="y",
        ball_start_y_range=(-5.5, -4.5),
        ball_start_z_range=(1.4, 1.8),
    )
    logger.info(f"[回合 {episode_idx+1}] 发球: p0={np.round(p0,3)}, v0={np.round(v0,2)}, 期望击打={np.round(p_hit_expected,3)}")

    hit_info = find_hitting_point_physics(
        env, p0, v0, shoulder_pos, workspace_radius, total_horizon
    )

    if hit_info is None:
        logger.warning(f"[回合 {episode_idx+1}] 球不在工作空间内，跳过！")
        return None

    k_hit_total = hit_info["k_hit"]
    p_hit = hit_info["p_hit"]
    v_ball_hit = hit_info["v_ball_hit"]

    if use_backswing:
        p_ee_init = env.get_ee_pos()
        dist_to_ball = np.linalg.norm(p_hit - p_ee_init)
        bs_scale = np.clip((dist_to_ball - 0.8) / (1.5 - 0.8), 0.0, 1.0)
        adaptive_bs = 0.4 + bs_scale * 0.6
        backswing_offset = -adaptive_bs
        logger.info(f"[回合 {episode_idx+1}] 自适应后摆: dist={dist_to_ball:.3f}m, backswing={adaptive_bs:.2f}rad")

    n_des = -v_ball_hit / (np.linalg.norm(v_ball_hit) + 1e-8)

    far_threshold = k_hit_total
    near_threshold = max(40, k_hit_total // 4)

    hit_direction = np.array(config["hitting"]["hit_direction"], dtype=np.float64)
    racket_speed = float(config["hitting"]["racket_speed"])
    v_hit_desired = compute_desired_hit_velocity(hit_direction, racket_speed)

    d_hat = hit_direction / (np.linalg.norm(hit_direction) + 1e-8)
    p_follow = p_hit + hit_shift * d_hat

    logger.info(f"[回合 {episode_idx+1}] 击打步数={k_hit_total}, 位置={np.round(p_hit,3)}")

    Q_p = np.array(config["cost"]["Q_p"], dtype=np.float64) * args.q_p_scale
    Q_v = np.array(config["cost"]["Q_v"], dtype=np.float64) * args.q_p_scale
    R = float(config["cost"]["R"])

    ilqt_cfg = dict(config["ilqt"])

    p_target_init = p_follow
    if use_backswing:
        U_prev, q_des_traj_init = generate_backswing_warm_start(
            env, x0, p_target_init, v_hit_desired, k_hit_total,
            backswing_offset=backswing_offset,
            backswing_ratio=backswing_ratio,
            fix_joint5_angle=fix_joint5_angle,
            n_des=n_des,
        )
    else:
        U_prev = compute_jacobian_init_control(
            env, x0, p_target_init, k_hit_total, gain=30.0,
            fix_joint5_angle=fix_joint5_angle,
        )
        q_des_traj_init = None

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

    cost_fn = HittingCost(
        env, p_follow, v_hit_desired, Q_p, Q_v, R,
        Q_p_running=0.0,
        R_joint_scale=r_joint_scale if r_joint_scale else None,
        q_des_traj=q_des_traj_init,
        Q_joint=Q_joint,
        R_schedule=R_schedule_init,
        Q_n=args.normal_weight,
        n_des=n_des,
    )

    solver = ILQTSolver(ilqt_cfg, use_analytical=use_analytical)

    env.reset(init_q)
    env.data.qpos[env.NQ:env.NQ + env.LEFT_ARM_NQ] = init_q_left
    env.data.qvel[env.NQ:env.NQ + env.LEFT_ARM_NQ] = 0.0
    env.update_kinematics()
    env.set_ball_state(p0, v0)

    x_current = x0.copy()
    X_history = [x0.copy()]
    U_history: list[np.ndarray] = []
    ball_pos_history: list[np.ndarray] = [p0.copy()]
    pos_error_history: list[float] = []

    U_buffer: np.ndarray = np.zeros((0, env.NU))
    buffer_idx: int = 0
    is_first_plan: bool = True
    p_hit_new = p_hit.copy()
    k_hit_new = k_hit_total
    _iters = 0
    hit_step = -1
    p_ee_at_hit = None
    ball_was_hit = False

    for step in range(total_horizon):
        ball_pos, ball_vel = env.get_ball_state()

        need_replan = (step % replan_interval == 0) or (step == 0) or (buffer_idx >= len(U_buffer))

        if need_replan:
            remaining_horizon = total_horizon - step
            hit_info_new = find_hitting_point_physics(
                env, ball_pos, ball_vel, shoulder_pos, workspace_radius, remaining_horizon
            )

            if hit_info_new is None:
                logger.info(f"[回合 {episode_idx+1}] 步 {step}: 球不再可达，停止 MPC")
                break

            k_hit_candidate = hit_info_new["k_hit"]

            if k_hit_candidate < max(10, k_hit_new // 4) and k_hit_new > 30:
                k_hit_candidate = max(1, k_hit_new - replan_interval)

            p_hit_new = hit_info_new["p_hit"]
            k_hit_new = k_hit_candidate
            v_ball_hit_new = hit_info_new["v_ball_hit"]
            n_des_new = -v_ball_hit_new / (np.linalg.norm(v_ball_hit_new) + 1e-8)
            _q_ik_cache = None

            p_follow_new = p_hit_new + hit_shift * d_hat

            if k_hit_new > far_threshold:
                u_jt = compute_jacobian_init_control(
                    env, x_current, p_follow_new, replan_interval, gain=60.0,
                    fix_joint5_angle=fix_joint5_angle,
                )
                U_buffer = u_jt
                buffer_idx = 0
                U_prev = np.zeros((0, env.NU))
                _iters = 0
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

                p_follow_new = p_hit_new + hit_shift * d_hat
                cost_fn.update_target(p_follow_new, v_hit_desired, n_des=n_des_new)
                cost_fn.update_weights(Q_p_scale, Q_v_scale)

                horizon_full = k_hit_new
                horizon_plan = min(horizon_full, horizon_cap)

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

                if len(U_mpc) > replan_interval:
                    U_prev = U_mpc[replan_interval:]
                elif len(U_mpc) > 0:
                    U_prev = U_mpc[1:]
                else:
                    U_prev = np.zeros((0, env.NU))

                U_buffer = U_mpc[:replan_interval]
                buffer_idx = 0
                _iters = iters_plan
        else:
            _q_ik_cache = None
            _iters = 0

        if buffer_idx < len(U_buffer):
            u_cmd = U_buffer[buffer_idx]
            buffer_idx += 1
        else:
            u_cmd = ik_pd_step(env, x_current, p_hit_new, fix_joint5_angle=fix_joint5_angle)

        if fix_joint5_angle is not None:
            u_cmd = fix_joint5_control(u_cmd, fix_joint5_angle, x_current, env.NQ)

        enable_collision = (k_hit_new <= 10)
        if hasattr(env, "set_arm_collision"):
            env.set_arm_collision(enable_collision)
        ball_vel_before_step = ball_vel.copy() if enable_collision else ball_vel
        x_current, ball_pos, ball_vel = env.step_full(u_cmd)

        ball_racket_hit = False
        if enable_collision and not ball_was_hit:
            n_contacts = env.data.ncon
            if n_contacts > 0:
                for ci in range(n_contacts):
                    c = env.data.contact[ci]
                    g1 = env.model.geom(c.geom1).name
                    g2 = env.model.geom(c.geom2).name
                    if 'ball' in g1 or 'ball' in g2:
                        if 'racket' in g1 or 'racket' in g2:
                            ball_racket_hit = True
                            ee_vel = env.get_ee_vel()
                            ee_speed = np.linalg.norm(ee_vel)
                            logger.info(f"[回合 {episode_idx+1}] 步 {step}: 球拍击球! 速度={ee_speed:.2f}m/s")

        if hasattr(env, "set_arm_collision"):
            env.set_arm_collision(True)

        X_history.append(x_current.copy())
        U_history.append(u_cmd.copy())
        ball_pos_history.append(ball_pos.copy())

        env.update_kinematics()
        pos_err = np.linalg.norm(env.get_ee_pos() - p_hit_new)
        pos_error_history.append(pos_err)

        if ball_racket_hit and k_hit_new <= 10 and not ball_was_hit:
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
            env.set_ball_vel(v_ball_new)

        if ball_was_hit and (step - hit_step) >= 5:
            break

        if k_hit_new <= 1:
            hit_step = step
            env.update_kinematics()
            p_ee_at_hit = env.get_ee_pos().copy()
            break

    # 击打后继续仿真，让球拍碰撞把球打飞
    post_hit_steps = 80
    for _ in range(post_hit_steps):
        q_hold = x_current[:env.NQ].copy()
        u_hold = 100.0 * (q_hold - x_current[:env.NQ]) - 10.0 * x_current[env.NQ:]
        u_hold = np.clip(u_hold, env.model.actuator_ctrlrange[:env.NU, 0], env.model.actuator_ctrlrange[:env.NU, 1])

        if fix_joint5_angle is not None:
            u_hold = fix_joint5_control(u_hold, fix_joint5_angle, x_current, env.NQ)

        x_current, ball_pos, _ = env.step_full(u_hold)
        X_history.append(x_current.copy())
        U_history.append(u_hold.copy())
        ball_pos_history.append(ball_pos.copy())

    # 评估
    X_arr = np.array(X_history)
    U_arr = np.array(U_history) if len(U_history) > 0 else np.zeros((0, env.NU))
    ball_pos_arr = np.array(ball_pos_history)

    if p_ee_at_hit is not None:
        p_ee_final = p_ee_at_hit
    else:
        env.set_arm_state(x_current)
        p_ee_final = env.get_ee_pos()
    v_ee_final = env.get_ee_vel()
    pos_error = np.linalg.norm(p_ee_final - p_hit)
    vel_error = np.linalg.norm(v_ee_final - v_hit_desired)

    success = pos_error < 0.05
    near_hit = pos_error < 0.10

    logger.info(
        f"[回合 {episode_idx+1}] {'成功' if success else '基本命中' if near_hit else '偏差大'}"
        f" | 位置误差={pos_error:.4f}m, 速度误差={vel_error:.4f}m/s"
    )

    return {
        "episode": episode_idx,
        "X": X_arr,
        "U": U_arr,
        "ball_pos": ball_pos_arr,
        "p0": p0,
        "v0": v0,
        "p_hit": p_hit,
        "v_hit_desired": v_hit_desired,
        "pos_error": pos_error,
        "vel_error": vel_error,
        "success": success,
        "near_hit": near_hit,
        "hit_step": hit_step,
        "fix_joint5_angle": fix_joint5_angle,
        "post_hit_steps": post_hit_steps,
        "init_q": init_q,
        "init_q_left": init_q_left,
        "n_mpc_steps": len(U_arr),
    }


def build_replay_data(
    env: RM65Env,
    result: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """为单个回合构建回放轨迹（含碰撞后弹性反弹）。

    与 rm65_mpc_fast.py 的 viewer 回放段逻辑完全一致。

    Args:
        env: RM-65 环境实例。
        result: run_single_episode 返回的结果字典。

    Returns:
        (X_replay, ball_replay): 臂状态轨迹和球位置轨迹。
    """
    _X_arr = result["X"]
    U_arr = result["U"]
    p0 = result["p0"]
    v0 = result["v0"]
    hit_step = result["hit_step"]
    post_hit_steps = result["post_hit_steps"]
    init_q = result["init_q"]
    init_q_left = result["init_q_left"]
    fix_joint5_angle = result["fix_joint5_angle"]

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
                if ('ball' in g1 or 'ball' in g2) and ('racket' in g1 or 'racket' in g2):
                    n_racket = env.get_ee_normal()
                    n_hat = n_racket / (np.linalg.norm(n_racket) + 1e-8)
                    v_ee = env.get_ee_vel()
                    v_rel_n = np.dot(ball_vel_pre - v_ee, n_hat)
                    e = 0.8
                    v_ball_new = ball_vel_pre - (1 + e) * v_rel_n * n_hat
                    env.set_ball_vel(v_ball_new)
                    rebound_applied = True
                    break
        X_replay.append(env.get_arm_state().copy())
        ball_replay.append(env.get_ball_pos().copy())

    # 击打后继续 post_hit_steps 步仿真
    x_current = env.get_arm_state()
    for _ in range(post_hit_steps):
        q_hold = x_current[:env.NQ].copy()
        u_hold = 100.0 * (q_hold - x_current[:env.NQ]) - 10.0 * x_current[env.NQ:]
        u_hold = np.clip(u_hold, env.model.actuator_ctrlrange[:env.NU, 0], env.model.actuator_ctrlrange[:env.NU, 1])
        if fix_joint5_angle is not None:
            u_hold = fix_joint5_control(u_hold, fix_joint5_angle, x_current, env.NQ)
        x_current, ball_pos, _ = env.step_full(u_hold)
        X_replay.append(x_current.copy())
        ball_replay.append(ball_pos.copy())

    if hasattr(env, "set_arm_collision"):
        env.set_arm_collision(True)

    return np.array(X_replay), np.array(ball_replay)


def generate_return_trajectory(
    env: RM65Env,
    x_current: np.ndarray,
    x_target: np.ndarray,
    n_frames: int = 30,
    kp: float = 150.0,
    kd: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """生成从当前位姿回到初始位姿的平滑过渡轨迹。

    Args:
        env: RM-65 环境实例。
        x_current: 当前臂状态。
        x_target: 目标臂状态（通常为初始位姿）。
        n_frames: 过渡帧数。
        kp: PD 位置增益。
        kd: PD 速度增益。

    Returns:
        (X_return, ball_return): 臂状态轨迹和球位置轨迹。
    """
    X_return = [x_current.copy()]
    ball_return = [env.get_ball_pos().copy()]
    x = x_current.copy()

    for _ in range(n_frames):
        q_des = x_target[:env.NQ]
        qdot_des = x_target[env.NQ:]
        tau = kp * (q_des - x[:env.NQ]) + kd * (qdot_des - x[env.NQ:])
        tau = np.clip(tau, env.model.actuator_ctrlrange[:env.NU, 0], env.model.actuator_ctrlrange[:env.NU, 1])
        x, ball_pos, _ = env.step_full(tau)
        X_return.append(x.copy())
        ball_return.append(ball_pos.copy())

    return np.array(X_return), np.array(ball_return)


def render_offscreen(
    env: RM65Env,
    segments: list[dict],
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    """离屏渲染所有段并写入 MP4 视频。

    Args:
        env: RM-65 环境实例。
        segments: 段列表，每段包含 X_replay, ball_replay, init_q_left 等。
        args: 命令行参数。
        output_path: 输出 MP4 文件路径。
    """
    import mujoco
    import imageio

    width = args.width
    height = args.height
    fps = args.fps
    dt = env.dt

    # 动态调整 MuJoCo 离屏帧缓冲大小以匹配请求分辨率
    env.model.vis.global_.offwidth = width
    env.model.vis.global_.offheight = height

    renderer = mujoco.Renderer(env.model, width=width, height=height)
    setup_camera_lights(env.model)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 3.5
    cam.elevation = -15
    cam.azimuth = 135
    cam.lookat[:] = [0.0, 0.0, 1.0]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        output_params=["-crf", "20", "-pix_fmt", "yuv420p"],
    )

    logger.info(f"开始离屏渲染，输出: {output_path} ({width}x{height}@{fps}fps)")

    # 按时间戳精确采样：每 1/fps 秒输出1帧
    video_dt = 1.0 / fps

    for seg_idx, seg in enumerate(segments):
        X_replay = seg["X_replay"]
        ball_replay = seg["ball_replay"]
        init_q_left = seg["init_q_left"]
        NQ = env.NQ
        bq = env.BALL_QPOS_START

        total_frames = len(ball_replay)
        next_video_time = 0.0
        n_written = 0

        for idx in range(total_frames):
            sim_time = idx * dt
            if sim_time < next_video_time:
                continue

            if idx < len(X_replay):
                arm_x = X_replay[idx]
            else:
                arm_x = X_replay[-1]

            env.data.qpos[:NQ] = arm_x[:NQ]
            env.data.qvel[:NQ] = arm_x[NQ:]
            env.data.qpos[NQ:NQ + env.LEFT_ARM_NQ] = init_q_left

            if idx < len(ball_replay):
                bp = ball_replay[idx]
                env.data.qpos[bq: bq + 3] = bp
                env.data.qpos[bq + 3: bq + 7] = [1, 0, 0, 0]

            mujoco.mj_forward(env.model, env.data)
            renderer.update_scene(env.data, camera=cam)
            pixels = renderer.render()
            writer.append_data(pixels)
            next_video_time += video_dt
            n_written += 1

        logger.info(f"  段 {seg_idx+1}/{len(segments)} 渲染完成 ({total_frames} 仿真步, {n_written} 视频帧)")

    writer.close()
    logger.info(f"MP4 保存完成: {output_path}")


def viewer_playback(
    env: RM65Env,
    segments: list[dict],
    args: argparse.Namespace,
) -> None:
    """在 MuJoCo 查看器中逐段播放所有回合。

    Args:
        env: RM-65 环境实例。
        segments: 段列表，每段包含 X_replay, ball_replay, init_q_left 等。
        args: 命令行参数。
    """
    import mujoco
    import mujoco.viewer

    dt = env.dt
    NQ = env.NQ
    bq = env.BALL_QPOS_START
    data = env.data
    model = env.model

    setup_camera_lights(model)

    init_q_left = segments[0]["init_q_left"] if segments else env.init_q_left

    # 初始化到第一段的第一帧
    first_seg = segments[0]
    X0 = first_seg["X_replay"][0]
    bp0 = first_seg["ball_replay"][0]
    data.qpos[:NQ] = X0[:NQ]
    data.qvel[:NQ] = X0[NQ:]
    data.qpos[NQ:NQ + env.LEFT_ARM_NQ] = init_q_left
    data.qpos[bq:bq + 3] = bp0
    data.qpos[bq + 3:bq + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)

    # 展平所有帧到统一播放序列
    all_frames: list[tuple[int, int, int]] = []  # (seg_idx, frame_idx_in_seg, total_frame_idx)
    for seg_idx, seg in enumerate(segments):
        total_frames = len(seg["ball_replay"])
        for f in range(total_frames):
            all_frames.append((seg_idx, f, len(all_frames)))

    total_all = len(all_frames)
    playback_speed = 1.0
    loop = True

    last_idx = -1

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.5
        viewer.cam.elevation = -15
        viewer.cam.azimuth = 135
        viewer.cam.lookat[:] = [0.0, 0.0, 1.0]

        start_time = time.perf_counter()

        while viewer.is_running():
            elapsed = time.perf_counter() - start_time
            sim_time = elapsed * playback_speed
            idx = int(sim_time / dt)

            if idx >= total_all:
                if loop:
                    start_time = time.perf_counter()
                    idx = 0
                else:
                    idx = total_all - 1

            if idx != last_idx:
                last_idx = idx
                seg_idx, f_idx, _ = all_frames[idx]
                seg = segments[seg_idx]
                X_replay = seg["X_replay"]
                ball_replay = seg["ball_replay"]
                seg_init_q_left = seg["init_q_left"]

                if f_idx < len(X_replay):
                    arm_x = X_replay[f_idx]
                else:
                    arm_x = X_replay[-1]

                data.qpos[:NQ] = arm_x[:NQ]
                data.qvel[:NQ] = arm_x[NQ:]
                data.qpos[NQ:NQ + env.LEFT_ARM_NQ] = seg_init_q_left

                if f_idx < len(ball_replay):
                    bp = ball_replay[f_idx]
                    data.qpos[bq: bq + 3] = bp
                    data.qpos[bq + 3: bq + 7] = [1, 0, 0, 0]

                mujoco.mj_forward(model, data)

            viewer.sync()
            time.sleep(1.0 / 120.0)


def main() -> None:
    """RM-65 批量击打 + 回放 + 录制主函数。"""
    parser = argparse.ArgumentParser(description="RM-65 批量网球击打 + 回放 + MP4 录制")
    parser.add_argument("--episodes", type=int, default=10, help="击打回合数（默认10）")
    parser.add_argument("--no-viewer", dest="viewer", action="store_false", help="关闭 MuJoCo 查看器回放")
    parser.add_argument("--viewer", dest="viewer", action="store_true", help="开启 MuJoCo 查看器回放（默认开启）")
    parser.set_defaults(viewer=True)
    parser.add_argument("--save-video", action="store_true", help="保存 MP4 视频（默认关闭）")
    parser.add_argument("--output", type=str, default="results/batch_hit.mp4", help="MP4 输出路径（默认 results/batch_hit.mp4）")
    parser.add_argument("--fps", type=int, default=60, help="视频帧率（默认60）")
    parser.add_argument("--width", type=int, default=1920, help="视频宽度（默认1920）")
    parser.add_argument("--height", type=int, default=1080, help="视频高度（默认1080）")
    parser.add_argument("--seed", type=int, default=0, help="起始随机种子（默认0）")
    parser.add_argument("--backswing", type=float, default=0.6, help="后摆幅度 (rad)")
    parser.add_argument("--bs-ratio", type=float, default=0.35, help="后摆占比 (0~1)")
    parser.add_argument("--r-decay", type=float, default=0.30, help="R 衰减占比 (0~1)")
    parser.add_argument("--hit-shift", type=float, default=0.02, help="随挥偏移距离 (m)")
    parser.add_argument("--normal-weight", type=float, default=500000.0, help="拍面法向量代价权重")
    parser.add_argument("--replan-interval", type=int, default=10, help="MPC 重规划间隔步数（默认10）")
    parser.add_argument("--first-iters", type=int, default=20, help="首次规划迭代数（默认20）")
    parser.add_argument("--near-iters", type=int, default=10, help="近距规划迭代数（默认10）")
    parser.add_argument("--q-p-near", type=float, default=10.0, help="近距位置代价缩放（默认10）")
    parser.add_argument("--horizon-cap", type=int, default=40, help="iLQR 规划步数封顶（默认40）")
    parser.add_argument("--q-p-scale", type=float, default=2.0, help="Q_p/Q_v 基础缩放因子（默认2.0）")
    args = parser.parse_args()

    use_analytical = True

    # 加载配置
    base_path = Path(__file__).resolve().parent.parent.parent / "configs"
    config = load_config(base_path / "default.yaml")
    mpc_config_path = base_path / "mpc.yaml"
    if mpc_config_path.exists():
        mpc_config = load_config(mpc_config_path)
        config = merge_configs(config, mpc_config)

    # 初始化 RM-65 环境
    model_path = Path(__file__).resolve().parent.parent.parent / "src" / "robot" / "rm65_model.xml"
    env = RM65Env(model_path, dt=float(config["sim"]["dt"]))

    # ===== 批量 MPC 规划 =====
    results: list[dict] = []
    n_episodes = args.episodes

    logger.info(f"开始批量规划: {n_episodes} 回合, 起始种子={args.seed}")
    t_batch_start = time.perf_counter()

    for ep in range(n_episodes):
        seed = args.seed + ep
        logger.info(f"========== 回合 {ep+1}/{n_episodes} (seed={seed}) ==========")
        result = run_single_episode(env, ep, seed, config, args, use_analytical)
        if result is not None:
            results.append(result)

    t_batch = time.perf_counter() - t_batch_start
    logger.info(f"批量规划完成: {len(results)}/{n_episodes} 回合成功, 总耗时={t_batch:.2f}s")

    # ===== 构建回放数据 =====
    segments: list[dict] = []
    init_q = np.array([-1.5, 1.57, -0.236, 0.404, 0.446, 2.45], dtype=np.float64)
    x0 = np.zeros(env.NX)
    x0[:env.NQ] = init_q

    for i, result in enumerate(results):
        logger.info(f"构建回放: 回合 {i+1}/{len(results)}")
        X_replay, ball_replay = build_replay_data(env, result)
        segments.append({
            "X_replay": X_replay,
            "ball_replay": ball_replay,
            "init_q_left": result["init_q_left"],
        })

        # 段间过渡：回到初始位姿（30 帧）
        if i < len(results) - 1:
            x_last = X_replay[-1]
            X_return, ball_return = generate_return_trajectory(env, x_last, x0, n_frames=30)
            segments.append({
                "X_replay": X_return,
                "ball_replay": ball_return,
                "init_q_left": result["init_q_left"],
            })

    # ===== 汇总统计 =====
    if len(results) > 0:
        pos_errors = [r["pos_error"] for r in results]
        vel_errors = [r["vel_error"] for r in results]
        n_success = sum(1 for r in results if r["success"])
        n_near = sum(1 for r in results if r["near_hit"])

        print("\n========================================")
        print(f"  批量击打统计 ({len(results)} 回合)")
        print("========================================")
        print(f"  成功 (<5cm):  {n_success}/{len(results)} ({100*n_success/len(results):.0f}%)")
        print(f"  命中 (<10cm): {n_near}/{len(results)} ({100*n_near/len(results):.0f}%)")
        print(f"  位置误差: {np.mean(pos_errors):.4f} ± {np.std(pos_errors):.4f} m")
        print(f"  速度误差: {np.mean(vel_errors):.4f} ± {np.std(vel_errors):.4f} m/s")
        print(f"  总规划时间: {t_batch:.2f}s")
        for i, r in enumerate(results):
            tag = "成功" if r["success"] else "命中" if r["near_hit"] else "偏"
            print(f"  回合 {i+1}: {tag} | pos={r['pos_error']:.4f}m, vel={r['vel_error']:.4f}m/s")
        print("========================================\n")

    # ===== MP4 渲染（在查看器之前，因为查看器是阻塞的） =====
    if args.save_video and len(segments) > 0:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = Path(__file__).resolve().parent.parent.parent / output_path
        render_offscreen(env, segments, args, output_path)

    # ===== 查看器回放 =====
    if args.viewer and len(segments) > 0:
        logger.info("开始 MuJoCo 查看器回放...")
        viewer_playback(env, segments, args)


if __name__ == "__main__":
    main()

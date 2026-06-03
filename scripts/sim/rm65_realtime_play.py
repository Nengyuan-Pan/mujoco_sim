"""RM-65 双臂机器人实时 MPC+iLQT 网球连续击打脚本。

功能：
1. 使用 RM-65 双臂模型（仅驱动右臂 6-DOF）
2. MuJoCo 物理仿真实时运行，查看器同步显示
3. 每隔 serve_interval 秒自动发球，球落点在球拍附近
4. 击打后机械臂自动回到初始位姿，等待下一球
5. 循环往复，连续打球

用法：
  python scripts/rm65_realtime_play.py
  python scripts/rm65_realtime_play.py --interval 3.0   # 每3秒发球
  python scripts/rm65_realtime_play.py --seed 42
  python scripts/rm65_realtime_play.py --fd              # 有限差分线性化
"""

import sys
import time
import argparse
import logging
import numpy as np
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import mujoco
import mujoco.viewer

from src.sim.rm65_env import RM65Env
from src.tennis.ball import generate_hittable_ball
from src.tennis.hitting import (
    find_hitting_point_physics,
    compute_desired_hit_velocity,
)
from src.ilqt.cost import HittingCost
from src.ilqt.solver import ILQTSolver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


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
) -> np.ndarray:
    """基于雅可比转置法计算初始控制序列。"""
    U = np.zeros((horizon, env.NU))
    x = x0.copy()
    ctrl_lo = env.model.actuator_ctrlrange[:env.NU, 0]
    ctrl_hi = env.model.actuator_ctrlrange[:env.NU, 1]

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
        x = env.step_from_state(x, tau)

    return U


def ik_pd_step(
    env: RM65Env,
    x_current: np.ndarray,
    p_target: np.ndarray,
    kp: float = 120.0,
    kd: float = 12.0,
) -> np.ndarray:
    """基于 IK + PD 的单步控制。"""
    q_ref = x_current[:env.NQ]
    q_target = env.solve_ik(p_target, q_init=q_ref, max_iter=100, eps=2e-3)
    q_err = q_target - x_current[:env.NQ]
    qdot_err = -x_current[env.NQ:]
    tau = kp * q_err + kd * qdot_err
    ctrl_lo = env.model.actuator_ctrlrange[:env.NU, 0]
    ctrl_hi = env.model.actuator_ctrlrange[:env.NU, 1]
    return np.clip(tau, ctrl_lo, ctrl_hi)


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


def compute_hold_control(
    env: RM65Env,
    x_current: np.ndarray,
    q_target: np.ndarray,
    kp: float = 100.0,
    kd: float = 10.0,
) -> np.ndarray:
    """计算保持机械臂在目标关节角的 PD 控制力矩。"""
    q_err = q_target - x_current[:env.NQ]
    qdot_err = -x_current[env.NQ:]
    tau = kp * q_err + kd * qdot_err
    ctrl_lo = env.model.actuator_ctrlrange[:env.NU, 0]
    ctrl_hi = env.model.actuator_ctrlrange[:env.NU, 1]
    return np.clip(tau, ctrl_lo, ctrl_hi)


class BallRally:
    """单次击球的状态机。"""

    IDLE = "idle"
    TRACKING = "tracking"
    HIT = "hit"
    RETURNING = "returning"

    def __init__(self) -> None:
        self.state = self.IDLE
        self.k_hit_remaining = 0
        self.p_hit = np.zeros(3)
        self.U_buffer: np.ndarray = np.zeros((0, 6))
        self.buffer_idx = 0
        self.U_prev: np.ndarray = np.zeros((0, 6))
        self.is_first_plan = True
        self.hit_pos_error = 0.0
        self.return_step = 0


def main() -> None:
    """RM-65 实时连续击打主函数。"""
    parser = argparse.ArgumentParser(description="RM-65 实时连续击打")
    parser.add_argument("--interval", type=float, default=5.0, help="发球间隔（秒）")
    parser.add_argument("--max-serves", type=int, default=20, help="最大发球次数")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--fd", action="store_true", help="使用有限差分线性化")
    parser.add_argument("--horizon", type=int, default=None, help="短地平线步数")
    parser.add_argument("--iter", type=int, default=None, help="每次重规划迭代数")
    args = parser.parse_args()

    use_analytical = not args.fd

    # ===== 加载配置 =====
    base_path = Path(__file__).resolve().parent.parent.parent / "configs"
    config = load_config(base_path / "default.yaml")
    mpc_config_path = base_path / "mpc.yaml"
    if mpc_config_path.exists():
        mpc_config = load_config(mpc_config_path)
        config = merge_configs(config, mpc_config)

    dt = float(config["sim"]["dt"])
    g = np.array(config["ball"]["gravity"], dtype=np.float64)

    shoulder_pos = np.array([-0.1, -0.22693, 1.302645], dtype=np.float64)
    workspace_radius = 0.90

    mpc_cfg = config.get("mpc", {})
    total_horizon = 200
    fixed_horizon = int(mpc_cfg.get("fixed_horizon", 20))
    replan_interval = 10
    max_iter_per_plan = 3
    Q_p_scale_far = 5.0
    Q_v_scale_far = 2.0
    Q_p_scale_near = 3.0
    Q_v_scale_near = 12.0

    if args.horizon is not None:
        fixed_horizon = args.horizon
    if args.iter is not None:
        max_iter_per_plan = args.iter

    # ===== 初始位姿 =====
    init_q = np.array([0.373, 1.57, -0.236, 0.404, 0.446, 2.45], dtype=np.float64)
    init_q_left = np.array([-0.373, -1.57, 0.236, -0.404, -0.446, -2.45], dtype=np.float64)

    # ===== 初始化环境 =====
    model_path = Path(__file__).resolve().parent.parent.parent / "src" / "robot" / "rm65_model.xml"
    env = RM65Env(model_path, dt=dt)
    env.init_q_left = init_q_left

    env.reset(init_q)
    env.data.qpos[env.NQ:env.NQ + env.LEFT_ARM_NQ] = init_q_left
    env.data.qvel[env.NQ:env.NQ + env.LEFT_ARM_NQ] = 0.0
    env.update_kinematics()
    p_racket = env.get_ee_pos().copy()
    logger.info(f"球拍初始位置: {p_racket}")

    racket_offset = p_racket - shoulder_pos
    # 缩小偏移范围确保击打点在工作空间内（球拍在肩关节下方0.43m，距离0.73m）
    # 限制偏移使 p_hit 与肩关节距离 < workspace_radius
    hit_offset_ranges = {
        "x": [max(racket_offset[0] - 0.15, -0.65), min(racket_offset[0] + 0.15, 0.55)],
        "y": [max(racket_offset[1] - 0.15, -0.50), min(racket_offset[1] + 0.15, 0.30)],
        "z": [max(racket_offset[2] - 0.15, -0.40), min(racket_offset[2] + 0.15, 0.20)],
    }

    # ===== 初始化 iLQT =====
    Q_p = np.array(config["cost"]["Q_p"], dtype=np.float64) * 2.0
    Q_v = np.array(config["cost"]["Q_v"], dtype=np.float64) * 2.0
    R = float(config["cost"]["R"])
    ilqt_cfg = dict(config["ilqt"])

    hit_direction = np.array(config["hitting"]["hit_direction"], dtype=np.float64)
    racket_speed = float(config["hitting"]["racket_speed"])
    v_hit_desired = compute_desired_hit_velocity(hit_direction, racket_speed)

    cost_fn = HittingCost(env, p_racket.copy(), v_hit_desired, Q_p, Q_v, R, Q_p_running=0.20)
    solver = ILQTSolver(ilqt_cfg, use_analytical=use_analytical, horizon_override=fixed_horizon)

    rng = np.random.default_rng(args.seed)
    serve_interval = args.interval
    max_serves = args.max_serves

    # ===== 状态 =====
    x_current = np.zeros(env.NX)
    x_current[:env.NQ] = init_q

    rally = BallRally()
    rally.state = BallRally.IDLE

    last_serve_wall_time = time.perf_counter()
    sim_time = 0.0
    hit_count = 0
    serve_count = 0
    total_pos_error = 0.0
    success_count = 0

    return_steps_max = int(1.5 / dt)  # 1.5秒回到初始位姿
    far_threshold = total_horizon

    print("=" * 60)
    print("  RM-65 实时连续击打")
    print(f"  发球间隔: {serve_interval}s | 最大发球: {max_serves}次")
    print(f"  球拍初始位置: {np.round(p_racket, 3)}")
    print("  关闭查看器窗口退出")
    print("=" * 60)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.distance = 3.5
        viewer.cam.elevation = -15
        viewer.cam.azimuth = 135
        viewer.cam.lookat[:] = [0.0, 0.0, 1.0]

        # 设置初始帧
        env.data.qpos[:env.NQ] = init_q
        env.data.qpos[env.NQ:env.NQ + env.LEFT_ARM_NQ] = init_q_left
        env.data.qvel[:env.NQ] = 0.0
        env.data.qvel[env.NQ:env.NQ + env.LEFT_ARM_NQ] = 0.0
        mujoco.mj_forward(env.model, env.data)
        viewer.sync()

        while viewer.is_running():
            # ===== 状态机: IDLE — 等待发球 =====
            if rally.state == BallRally.IDLE:
                if serve_count >= max_serves:
                    # 达到最大发球次数，继续运行查看器但不发球
                    u_cmd = compute_hold_control(env, x_current, init_q)
                    x_current, _, _ = env.step_full(u_cmd)
                    sim_time += dt
                    mujoco.mj_forward(env.model, env.data)
                    viewer.sync()
                    continue

                time_since_serve = time.perf_counter() - last_serve_wall_time
                if time_since_serve >= serve_interval:
                    # 生成新球
                    hit_time = total_horizon * dt * rng.uniform(0.55, 0.85)
                    p0, v0, p_hit_expected = generate_hittable_ball(
                        shoulder_pos, workspace_radius, hit_time, g,
                        hit_offset_ranges=hit_offset_ranges, rng=rng,
                        ball_direction="y",
                    )

                    env.set_ball_state(p0, v0)

                    hit_info = find_hitting_point_physics(
                        env, p0, v0, shoulder_pos, workspace_radius, total_horizon
                    )

                    # 无论可达与否都更新发球时间，防止疯狂重试
                    last_serve_wall_time = time.perf_counter()
                    serve_count += 1

                    if hit_info is None:
                        logger.info("球不可达，跳过本次发球")
                    else:
                        rally.state = BallRally.TRACKING
                        rally.k_hit_remaining = hit_info["k_hit"]
                        rally.p_hit = hit_info["p_hit"].copy()
                        rally.U_buffer = np.zeros((0, env.NU))
                        rally.buffer_idx = 0
                        rally.U_prev = np.zeros((0, env.NU))
                        rally.is_first_plan = True

                        cost_fn.update_target(rally.p_hit, v_hit_desired)

                        far_threshold = rally.k_hit_remaining
                        hit_count += 1

                        logger.info(
                            f"[球#{hit_count}] 击打步数={rally.k_hit_remaining}, "
                            f"击打位置={np.round(rally.p_hit, 3)}"
                        )

                # 等待期间保持初始位姿
                u_cmd = compute_hold_control(env, x_current, init_q)
                x_current, _, _ = env.step_full(u_cmd)

            # ===== 状态机: TRACKING — MPC 跟踪击打 =====
            elif rally.state == BallRally.TRACKING:
                ball_pos, ball_vel = env.get_ball_state()

                need_replan = (
                    rally.buffer_idx >= len(rally.U_buffer)
                    or rally.buffer_idx == 0
                )

                if need_replan:
                    remaining_horizon = total_horizon
                    hit_info_new = find_hitting_point_physics(
                        env, ball_pos, ball_vel, shoulder_pos,
                        workspace_radius, remaining_horizon,
                    )

                    if hit_info_new is None:
                        logger.info(f"[球#{hit_count}] 球不再可达，停止跟踪")
                        rally.state = BallRally.RETURNING
                        rally.return_step = 0
                        u_cmd = compute_hold_control(env, x_current, init_q)
                        x_current, _, _ = env.step_full(u_cmd)
                        sim_time += dt
                        mujoco.mj_forward(env.model, env.data)
                        viewer.sync()
                        continue

                    rally.p_hit = hit_info_new["p_hit"].copy()
                    rally.k_hit_remaining = hit_info_new["k_hit"]

                    if rally.k_hit_remaining > far_threshold:
                        u_jt = compute_jacobian_init_control(
                            env, x_current, rally.p_hit,
                            replan_interval, gain=60.0,
                        )
                        rally.U_buffer = u_jt
                        rally.buffer_idx = 0
                        rally.U_prev = np.zeros((0, env.NU))
                    else:
                        env.set_arm_state(x_current)
                        env.update_kinematics()
                        pos_err_now = np.linalg.norm(env.get_ee_pos() - rally.p_hit)

                        if pos_err_now > 0.10:
                            Q_p_scale = Q_p_scale_far
                            Q_v_scale = Q_v_scale_far
                        else:
                            ratio = pos_err_now / 0.10
                            Q_p_scale = Q_p_scale_near + (Q_p_scale_far - Q_p_scale_near) * ratio
                            Q_v_scale = Q_v_scale_near + (Q_v_scale_far - Q_v_scale_near) * ratio

                        cost_fn.update_target(rally.p_hit, v_hit_desired)
                        cost_fn.update_weights(Q_p_scale, Q_v_scale)

                        # MPC 短地平线：始终使用 fixed_horizon，避免长地平线计算慢
                        horizon_plan = min(rally.k_hit_remaining, fixed_horizon)

                        iters_plan = max_iter_per_plan
                        skip_ls = not rally.is_first_plan
                        if rally.is_first_plan:
                            iters_plan = 5
                            rally.is_first_plan = False
                        elif rally.k_hit_remaining <= max(40, rally.k_hit_remaining // 4):
                            iters_plan = min(5, max_iter_per_plan + 2)

                        if len(rally.U_prev) >= horizon_plan // 2:
                            U_warm = resample_control_sequence(rally.U_prev, horizon_plan)
                        else:
                            U_warm = compute_jacobian_init_control(
                                env, x_current, rally.p_hit, horizon_plan, gain=30.0,
                            )

                        ball_pos_save, ball_vel_save = env.get_ball_state()

                        X_mpc, U_mpc, iter_costs = solver.solve_few_iters(
                            env, cost_fn, x_current, U_warm,
                            max_iter=iters_plan,
                            skip_linesearch=skip_ls,
                        )

                        env.set_ball_state(ball_pos_save, ball_vel_save)
                        env.set_arm_state(x_current)

                        if len(U_mpc) > replan_interval:
                            rally.U_prev = U_mpc[replan_interval:]
                        elif len(U_mpc) > 0:
                            rally.U_prev = U_mpc[1:]
                        else:
                            rally.U_prev = np.zeros((0, env.NU))

                        rally.U_buffer = U_mpc[:replan_interval]
                        rally.buffer_idx = 0

                # 执行控制
                if rally.buffer_idx < len(rally.U_buffer):
                    u_cmd = rally.U_buffer[rally.buffer_idx]
                    rally.buffer_idx += 1
                else:
                    u_cmd = ik_pd_step(env, x_current, rally.p_hit)

                x_current, ball_pos, ball_vel = env.step_full(u_cmd)

                # 递减剩余击打步数
                rally.k_hit_remaining -= 1

                # 检查是否到达击打时刻
                if rally.k_hit_remaining <= 1:
                    env.update_kinematics()
                    p_ee = env.get_ee_pos()
                    rally.hit_pos_error = np.linalg.norm(p_ee - rally.p_hit)
                    total_pos_error += rally.hit_pos_error

                    if rally.hit_pos_error < 0.05:
                        success_count += 1
                        logger.info(
                            f"[球#{hit_count}] 击打成功! 误差={rally.hit_pos_error:.4f}m"
                        )
                    elif rally.hit_pos_error < 0.10:
                        success_count += 1
                        logger.info(
                            f"[球#{hit_count}] 基本命中! 误差={rally.hit_pos_error:.4f}m"
                        )
                    else:
                        logger.info(
                            f"[球#{hit_count}] 偏差较大 误差={rally.hit_pos_error:.4f}m"
                        )

                    rally.state = BallRally.HIT
                    rally.return_step = 0

                elif rally.k_hit_remaining <= 10 and rally.k_hit_remaining % 5 == 0:
                    env.update_kinematics()
                    pe = np.linalg.norm(env.get_ee_pos() - rally.p_hit)
                    logger.info(f"[球#{hit_count}] 剩余={rally.k_hit_remaining}, 误差={pe:.4f}m")

            # ===== 状态机: HIT — 击打后短暂保持，观察球飞出 =====
            elif rally.state == BallRally.HIT:
                # 保持当前位置约 0.4 秒（让球拍碰撞把球打飞）
                q_hold = x_current[:env.NQ].copy()
                u_cmd = compute_hold_control(env, x_current, q_hold, kp=150.0, kd=15.0)
                x_current, _, _ = env.step_full(u_cmd)

                rally.return_step += 1
                hold_steps = int(0.4 / dt)
                if rally.return_step >= hold_steps:
                    rally.state = BallRally.RETURNING
                    rally.return_step = 0
                    logger.info(f"[球#{hit_count}] 开始回到初始位姿...")

            # ===== 状态机: RETURNING — 回到初始位姿 =====
            elif rally.state == BallRally.RETURNING:
                u_cmd = compute_hold_control(env, x_current, init_q, kp=120.0, kd=12.0)
                x_current, _, _ = env.step_full(u_cmd)

                rally.return_step += 1
                q_err = np.linalg.norm(x_current[:env.NQ] - init_q)

                if q_err < 0.05 or rally.return_step >= return_steps_max:
                    rally.state = BallRally.IDLE
                    avg_err = total_pos_error / hit_count if hit_count > 0 else 0
                    logger.info(
                        f"[球#{hit_count}] 已回位 | "
                        f"累计平均误差={avg_err:.4f}m"
                    )

            sim_time += dt
            mujoco.mj_forward(env.model, env.data)
            viewer.sync()

    print("\n" + "=" * 60)
    print(f"  发球次数: {serve_count}")
    print(f"  击打次数: {hit_count}")
    if hit_count > 0:
        print(f"  成功击打: {success_count}/{hit_count} ({success_count/hit_count*100:.1f}%)")
        print(f"  平均位置误差: {total_pos_error / hit_count:.4f}m")
    else:
        print("  无成功击打")
    print("=" * 60)


if __name__ == "__main__":
    main()

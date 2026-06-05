"""RM-65 实时批量击打评估脚本。

在 rm65_mpc_fast.py 基础上，连续发 20 球，自动收集并分析各项指标。
用法：python scripts/rm65_realtime_batch.py

输出：
  - 每球实时日志 + 击打结果
  - 20 球汇总统计（成功率、位置误差均值±std、球速、实时比率）
  - results/rm65_batch_summary.png（汇总图表）
"""

import sys
import time
import argparse
import logging
import numpy as np
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    matplotlib = None
    plt = None
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.sim.rm65_env import RM65Env
from src.tennis.ball import generate_ball_to_target_box
from src.tennis.hitting import find_hitting_point_physics, compute_desired_hit_velocity
from src.ilqt.cost import HittingCost
try:
    from src.cpp.solver_cpp import ILQTSolver
except ImportError:
    from src.ilqt.solver import ILQTSolver
from src.ilqt.utils import compute_total_cost, forward_pass_single, forward_pass_with_linesearch

# 从原始脚本导入辅助函数（仅函数定义，无副作用）
from scripts.rm65_mpc_ilqr_5_5 import (
    generate_backswing_warm_start,
    compute_jacobian_init_control,
    compute_joint1_backswing_trajectory,
    resample_control_sequence,
    compute_r_schedule,
    ik_pd_step,
)
from scripts.rm65_evaluate import (
    evaluate_trajectory,
    print_evaluation_report,
    plot_evaluation,
    RM65_JOINT_NAMES,
    RM65_JOINT_LIMIT_DEG,
    RM65_JOINT_VEL_LIMIT,
    RM65_TORQUE_LIMIT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 快速版 MPC 参数 ──
REPLAN_INTERVAL = 10
MAX_ITER_PER_PLAN = 5
FIRST_PLAN_ITERS = 3
NEAR_PLAN_ITERS = 5
HORIZON_CAP = 40
TOTAL_HORIZON = 200
DT = 0.005


def load_config(path: Path) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_configs(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            merge_configs(base[k], v)
        else:
            base[k] = v
    return base


def fix_joint5_control(u, q_fixed, x_current, nq, kp=300.0, kd=30.0):
    u = u.copy()
    err_q = q_fixed - x_current[:nq][5]
    err_qd = -x_current[nq:][5]
    u[5] = kp * err_q + kd * err_qd
    return u


def run_single(seed: int, args: argparse.Namespace, base_config: dict) -> dict | None:
    """执行单次击打并返回评估数据。"""
    config = merge_configs(base_config.copy(), base_config)
    dt = float(config["sim"]["dt"])
    g = np.array(config["ball"]["gravity"], dtype=np.float64)
    shoulder_pos = np.array([-0.1, -0.22693, 1.302645], dtype=np.float64)
    workspace_radius = 0.90

    rng = np.random.default_rng(seed)
    init_q = np.array([-1.5, 1.57, -0.236, 0.404, 0.446, 2.45], dtype=np.float64)
    init_q_left = np.array([-0.373, -1.57, 0.236, -0.404, -0.446, -2.45], dtype=np.float64)

    model_path = Path(__file__).resolve().parent.parent.parent / "src" / "robot" / "rm65_model.xml"
    env = RM65Env(str(model_path), dt=dt)
    env.init_q_left = init_q_left
    x0 = np.zeros(env.NX)
    x0[:env.NQ] = init_q

    # ── 发球 ──
    target_center = np.array([-0.82765693, -0.47411682, 0.86947444])
    target_offset = 0.2
    hit_time = TOTAL_HORIZON * dt * rng.uniform(0.3, 0.4)
    p0, v0, _ = generate_ball_to_target_box(
        target_center, target_offset, hit_time, g,
        shoulder_pos=shoulder_pos, workspace_radius=workspace_radius,
        ball_speed=None, rng=rng, ball_direction="y",
        ball_start_y_range=(-5.5, -4.5), ball_start_z_range=(1.4, 1.8),
    )

    env.reset(init_q)
    env.data.qpos[env.NQ:env.NQ + env.LEFT_ARM_NQ] = init_q_left
    env.data.qvel[env.NQ:env.NQ + env.LEFT_ARM_NQ] = 0.0
    env.update_kinematics()

    hit_info = find_hitting_point_physics(env, p0, v0, shoulder_pos, workspace_radius, TOTAL_HORIZON)
    if hit_info is None:
        logger.warning(f"  Seed {seed}: 球不可达，跳过")
        return None

    k_hit_total = hit_info["k_hit"]
    p_hit = hit_info["p_hit"]
    v_ball_hit = hit_info["v_ball_hit"]

    # 自适应后摆
    p_ee_init = env.get_ee_pos()
    dist_to_ball = np.linalg.norm(p_hit - p_ee_init)
    bs_scale = np.clip((dist_to_ball - 0.8) / (1.5 - 0.8), 0.0, 1.0)
    adaptive_bs = 0.4 + bs_scale * 0.6
    backswing_offset = -adaptive_bs

    far_threshold = k_hit_total
    near_threshold = max(40, k_hit_total // 4)

    hit_direction = np.array([0.0, -1.0, 0.3], dtype=np.float64)
    racket_speed = 5.0
    v_hit_desired = compute_desired_hit_velocity(hit_direction, racket_speed)
    d_hat = hit_direction / (np.linalg.norm(hit_direction) + 1e-8)

    n_des = -v_ball_hit / (np.linalg.norm(v_ball_hit) + 1e-8)
    if args.normal_flip:
        n_des = -n_des

    # ── 初始化代价与求解器 ──
    Q_p = np.array(config["cost"]["Q_p"], dtype=np.float64) * 2.0
    Q_v = np.array(config["cost"]["Q_v"], dtype=np.float64) * 2.0
    R = float(config["cost"]["R"])
    ilqt_cfg = dict(config["ilqt"])
    solver = ILQTSolver(ilqt_cfg, use_analytical=True)

    # Warm-start
    U_prev, q_des_traj_init = generate_backswing_warm_start(
        env, x0, p_hit, v_hit_desired, k_hit_total,
        backswing_offset=backswing_offset, backswing_ratio=0.35,
        fix_joint5_angle=None, n_des=n_des,
    )

    r_joint_scale = {0: 0.3}
    R_schedule = None
    hit_shift = 0.01
    p_follow = p_hit + hit_shift * d_hat

    cost_fn = HittingCost(
        env, p_follow, v_hit_desired, Q_p, Q_v, R,
        Q_p_running=0.0, R_joint_scale=r_joint_scale,
        q_des_traj=q_des_traj_init, Q_joint=None,
        R_schedule=R_schedule, Q_n=500000.0, n_des=n_des,
    )

    # ── MPC 主循环 ──
    env.set_ball_state(p0, v0)
    x_current = x0.copy()
    X_history = [x0.copy()]
    U_history = []
    ball_pos_history = [p0.copy()]

    U_buffer = np.zeros((0, env.NU))
    buffer_idx = 0
    is_first_plan = True
    p_hit_new = p_hit.copy()
    k_hit_new = k_hit_total
    iters = 0
    hit_step = -1
    ball_was_hit = False

    t0 = time.perf_counter()

    for step in range(TOTAL_HORIZON):
        ball_pos, ball_vel = env.get_ball_state()
        need_replan = (step % REPLAN_INTERVAL == 0) or step == 0 or buffer_idx >= len(U_buffer)

        if need_replan:
            remaining = TOTAL_HORIZON - step
            hi = find_hitting_point_physics(env, ball_pos, ball_vel, shoulder_pos, workspace_radius, remaining)
            if hi is None:
                break
            k_hit_candidate = hi["k_hit"]
            if k_hit_candidate < max(10, k_hit_new // 4) and k_hit_new > 30:
                k_hit_candidate = max(1, k_hit_new - REPLAN_INTERVAL)
            p_hit_new = hi["p_hit"]
            k_hit_new = k_hit_candidate
            v_ball_hit_new = hi["v_ball_hit"]
            n_des_new = -v_ball_hit_new / (np.linalg.norm(v_ball_hit_new) + 1e-8)
            if args.normal_flip:
                n_des_new = -n_des_new
            p_follow_new = p_hit_new + hit_shift * d_hat

            if k_hit_new > far_threshold:
                U_buffer = compute_jacobian_init_control(
                    env, x_current, p_follow_new, REPLAN_INTERVAL, gain=60.0)
                buffer_idx = 0
                U_prev = np.zeros((0, env.NU))
                iters = 0
            else:
                env.set_arm_state(x_current)
                env.update_kinematics()
                pos_err_now = np.linalg.norm(env.get_ee_pos() - p_hit_new)
                if pos_err_now > 0.10:
                    Q_p_scale, Q_v_scale = 5.0, 3.0
                else:
                    r = pos_err_now / 0.10
                    Q_p_scale = 5.0 + (5.0 - 5.0) * r
                    Q_v_scale = 50.0 + (3.0 - 50.0) * r

                cost_fn.update_target(p_follow_new, v_hit_desired, n_des=n_des_new)
                cost_fn.update_weights(Q_p_scale, Q_v_scale)

                horizon_full = k_hit_new
                horizon_plan = min(horizon_full, HORIZON_CAP)

                iters_plan = MAX_ITER_PER_PLAN
                skip_ls = True
                if is_first_plan:
                    iters_plan = FIRST_PLAN_ITERS
                    skip_ls = False
                    is_first_plan = False
                elif k_hit_new <= near_threshold:
                    iters_plan = NEAR_PLAN_ITERS

                # Warm-start (use pre-imported functions)
                q_hit_ik = env.solve_ik(p_hit_new, q_init=x_current[:env.NQ], max_iter=150, eps=1e-3)
                env.set_arm_state(np.concatenate([q_hit_ik, np.zeros(env.NQ)]))
                J_p_new = env.get_ee_jacp()
                qdot_hit_new = np.linalg.lstsq(J_p_new, v_hit_desired, rcond=None)[0]
                qdot_norm = np.linalg.norm(qdot_hit_new)
                if qdot_norm > 3.0:
                    qdot_hit_new *= 3.0 / qdot_norm

                bs_scale_plan = horizon_full / max(k_hit_total, 1)
                q_des_traj_full = np.zeros((horizon_full, env.NQ))
                q_des_traj_full[:, 0] = compute_joint1_backswing_trajectory(
                    x_current[0], x_current[env.NQ], q_hit_ik[0], qdot_hit_new[0],
                    horizon_full, backswing_offset * bs_scale_plan, 0.35)
                for j in range(1, env.NQ):
                    q_des_traj_full[:, j] = np.linspace(x_current[j], q_hit_ik[j], horizon_full)
                cost_fn.set_q_des_traj(q_des_traj_full[:horizon_plan], Q_joint=None)

                if len(U_prev) >= horizon_full // 3:
                    U_warm = resample_control_sequence(U_prev, horizon_full)[:horizon_plan]
                else:
                    U_warm_full, _ = generate_backswing_warm_start(
                        env, x_current, p_follow_new, v_hit_desired, horizon_full,
                        backswing_offset * bs_scale_plan, 0.35, None, n_des_new)
                    U_warm = U_warm_full[:horizon_plan]

                ball_pos_save, ball_vel_save = env.get_ball_state()
                X_mpc, U_mpc, _ = solver.solve_few_iters(
                    env, cost_fn, x_current, U_warm,
                    max_iter=iters_plan, skip_linesearch=skip_ls)
                env.set_ball_state(ball_pos_save, ball_vel_save)
                env.set_arm_state(x_current)

                if len(U_mpc) > REPLAN_INTERVAL:
                    U_prev = U_mpc[REPLAN_INTERVAL:]
                elif len(U_mpc) > 0:
                    U_prev = U_mpc[1:]
                else:
                    U_prev = np.zeros((0, env.NU))
                U_buffer = U_mpc[:REPLAN_INTERVAL]
                buffer_idx = 0
                iters = iters_plan
        else:
            iters = 0

        if buffer_idx < len(U_buffer):
            u_cmd = U_buffer[buffer_idx]
            buffer_idx += 1
        else:
            u_cmd = ik_pd_step(env, x_current, p_hit_new)

        enable_collision = (k_hit_new <= 10)
        if hasattr(env, "set_arm_collision"):
            env.set_arm_collision(enable_collision)
        ball_vel_before_step = ball_vel.copy() if enable_collision else ball_vel
        x_current, ball_pos, ball_vel = env.step_full(u_cmd)

        if enable_collision and not ball_was_hit:
            for ci in range(env.data.ncon):
                c = env.data.contact[ci]
                g1, g2 = env.model.geom(c.geom1).name, env.model.geom(c.geom2).name
                if ('ball' in g1 or 'ball' in g2) and ('racket' in g1 or 'racket' in g2):
                    ball_was_hit = True
                    hit_step = step
                    n_racket = env.get_ee_normal()
                    n_hat = n_racket / (np.linalg.norm(n_racket) + 1e-8)
                    v_ee = env.get_ee_vel()
                    v_rel_n = np.dot(ball_vel_before_step - v_ee, n_hat)
                    e = 0.8
                    env.set_ball_vel(ball_vel_before_step - (1 + e) * v_rel_n * n_hat)

        if hasattr(env, "set_arm_collision"):
            env.set_arm_collision(True)

        X_history.append(x_current.copy())
        U_history.append(u_cmd.copy())
        ball_pos_history.append(ball_pos.copy())

        if ball_was_hit and (step - hit_step) >= 5:
            break
        if k_hit_new <= 1:
            hit_step = step if hit_step < 0 else hit_step
            break

    # 击打后仿真
    for _ in range(80):
        q_hold = x_current[:env.NQ].copy()
        u_hold = 100.0 * (q_hold - x_current[:env.NQ]) - 10.0 * x_current[env.NQ:]
        u_hold = np.clip(u_hold, env.model.actuator_ctrlrange[:env.NU, 0],
                         env.model.actuator_ctrlrange[:env.NU, 1])
        x_current, ball_pos, _ = env.step_full(u_hold)
        X_history.append(x_current.copy())
        U_history.append(u_hold.copy())
        ball_pos_history.append(ball_pos.copy())

    elapsed = time.perf_counter() - t0
    n_steps = len(U_history)
    real_time_ratio = (n_steps * dt) / elapsed if elapsed > 0 else 0

    # ── 评估 ──
    X_arr = np.array(X_history)
    U_arr = np.array(U_history) if U_history else np.zeros((0, env.NU))
    ball_pos_arr = np.array(ball_pos_history)

    if hit_step >= 0 and hit_step < len(X_arr):
        env.set_arm_state(X_arr[hit_step])
        p_ee_final = env.get_ee_pos()
        v_ee_final = env.get_ee_vel()
    else:
        env.set_arm_state(x_current)
        p_ee_final = env.get_ee_pos()
        v_ee_final = env.get_ee_vel()

    pos_error = np.linalg.norm(p_ee_final - p_hit_new)
    vel_error = np.linalg.norm(v_ee_final - v_hit_desired)
    ball_speed_after = np.linalg.norm(env.get_ball_vel())
    ball_speed_before = np.linalg.norm(v_ball_hit) if 'v_ball_hit' in dir() else 0

    eval_data = evaluate_trajectory(
        X_arr, U_arr, env, ball_pos_arr,
        hit_step=hit_step, p_hit=p_hit_new, v_hit_desired=v_hit_desired, dt=dt)

    return {
        "seed": seed,
        "pos_error": pos_error,
        "vel_error": vel_error,
        "ball_speed_before": ball_speed_before,
        "ball_speed_after": ball_speed_after,
        "elapsed": elapsed,
        "n_steps": n_steps,
        "real_time_ratio": real_time_ratio,
        "hit": pos_error < 0.05,
        "fair": 0.05 <= pos_error < 0.10,
        "k_hit_total": k_hit_total,
        "eval_data": eval_data,
    }


def print_summary(results: list[dict]) -> None:
    """打印 20 球汇总统计。"""
    valid = [r for r in results if r is not None]
    if not valid:
        print("无有效结果")
        return

    pos_errors = np.array([r["pos_error"] for r in valid])
    vel_errors = np.array([r["vel_error"] for r in valid])
    speeds_before = np.array([r["ball_speed_before"] for r in valid])
    speeds_after = np.array([r["ball_speed_after"] for r in valid])
    elapsed_times = np.array([r["elapsed"] for r in valid])
    rt_ratios = np.array([r["real_time_ratio"] for r in valid])
    n_hit = sum(1 for r in valid if r["hit"])
    n_fair = sum(1 for r in valid if r["fair"])

    # 汇总关节指标
    all_peak_qdot = []
    all_peak_torque = []
    all_peak_qacc = []
    for r in valid:
        ed = r["eval_data"]
        all_peak_qdot.append(ed["peak_qdot"])
        all_peak_torque.append(ed["peak_torque"])
        # 从 qdot_traj 计算加速度峰值
        qdot = ed["qdot_traj"]
        if len(qdot) > 1:
            qacc = np.abs((qdot[1:] - qdot[:-1]) / DT).max(axis=0)
        else:
            qacc = np.zeros(6)
        all_peak_qacc.append(qacc)
    all_peak_qdot = np.array(all_peak_qdot)
    all_peak_torque = np.array(all_peak_torque)
    all_peak_qacc = np.array(all_peak_qacc)

    print("\n" + "=" * 70)
    print("  RM-65 实时批量评估 — 20 球汇总")
    print("=" * 70)
    print(f"  总球数: {len(results)}, 有效: {len(valid)}")
    print(f"  击打成功 (<5cm): {n_hit}/{len(valid)} ({n_hit/len(valid)*100:.0f}%)")
    print(f"  基本命中 (<10cm): {n_hit+n_fair}/{len(valid)} ({(n_hit+n_fair)/len(valid)*100:.0f}%)")
    print(f"\n  位置误差: 均值={pos_errors.mean()*100:.2f}cm, std={pos_errors.std()*100:.2f}cm")
    print(f"  速度误差: 均值={vel_errors.mean():.2f}m/s, std={vel_errors.std():.2f}m/s")
    print(f"  击打前球速: {speeds_before.mean():.1f}±{speeds_before.std():.1f} m/s")
    print(f"  击打后球速: {speeds_after.mean():.1f}±{speeds_after.std():.1f} m/s")
    print(f"  计算时间: {elapsed_times.mean():.2f}±{elapsed_times.std():.2f}s")
    print(f"  实时比率: {rt_ratios.mean():.2f}±{rt_ratios.std():.2f}x")

    print(f"\n  关节速度峰值 (deg/s) vs 限速:")
    for j in range(6):
        limit = np.rad2deg(RM65_JOINT_VEL_LIMIT[j])
        peak = np.rad2deg(all_peak_qdot[:, j])
        ratio = peak / limit * 100
        n_exceed = sum(1 for p in peak if p > limit)
        print(f"    J{j+1} ({RM65_JOINT_NAMES[j]:14s}): "
              f"峰值={peak.mean():.0f}±{peak.std():.0f}, 限速={limit:.0f}, "
              f"超限={n_exceed}/{len(valid)}")

    print(f"\n  关节力矩峰值 (Nm) vs 限值:")
    for j in range(6):
        limit = RM65_TORQUE_LIMIT[j]
        peak = all_peak_torque[:, j]
        n_exceed = sum(1 for p in peak if p > limit)
        print(f"    J{j+1} ({RM65_JOINT_NAMES[j]:14s}): "
              f"峰值={peak.mean():.1f}±{peak.std():.1f}, 限值={limit:.0f}, "
              f"超限={n_exceed}/{len(valid)}")

    print(f"\n  关节加速度峰值 (deg/s²) vs 600:")
    for j in range(6):
        peak = np.rad2deg(all_peak_qacc[:, j])
        n_exceed = sum(1 for p in peak if p > 600)
        print(f"    J{j+1} ({RM65_JOINT_NAMES[j]:14s}): "
              f"峰值={peak.mean():.0f}±{peak.std():.0f}")
    print("=" * 70 + "\n")


def plot_summary(results: list[dict], save_path: str) -> None:
    """绘制汇总图。"""
    if plt is None:
        logger.warning("matplotlib 未安装，跳过汇总图绘制")
        return

    valid = [r for r in results if r is not None]
    if len(valid) < 2:
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    seeds = np.array([r["seed"] for r in valid])

    # 1. 位置误差
    ax = axes[0, 0]
    pos_err = np.array([r["pos_error"]*100 for r in valid])
    colors = ['g' if p < 5 else ('y' if p < 10 else 'r') for p in pos_err]
    ax.bar(range(len(valid)), pos_err, color=colors)
    ax.axhline(5, color='g', linestyle='--', alpha=0.5, label='5cm')
    ax.axhline(10, color='r', linestyle='--', alpha=0.5, label='10cm')
    ax.set_xlabel("Trial")
    ax.set_ylabel("Position Error (cm)")
    ax.set_title("Position Error per Trial")
    ax.legend()

    # 2. 实时比率
    ax = axes[0, 1]
    rt = np.array([r["real_time_ratio"] for r in valid])
    colors = ['g' if r >= 1.0 else 'r' for r in rt]
    ax.bar(range(len(valid)), rt, color=colors)
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='1.0x')
    ax.set_xlabel("Trial")
    ax.set_ylabel("Real-time Ratio")
    ax.set_title("Real-time Ratio per Trial")
    ax.legend()

    # 3. 球速
    ax = axes[0, 2]
    sb = np.array([r["ball_speed_before"] for r in valid])
    sa = np.array([r["ball_speed_after"] for r in valid])
    x = np.arange(len(valid))
    ax.bar(x - 0.2, sb, 0.4, label='Before', color='steelblue')
    ax.bar(x + 0.2, sa, 0.4, label='After', color='orange')
    ax.set_xlabel("Trial")
    ax.set_ylabel("Ball Speed (m/s)")
    ax.set_title("Ball Speed per Trial")
    ax.legend()

    # 4. 关节速度
    ax = axes[1, 0]
    for r in valid:
        peak = np.rad2deg(r["eval_data"]["peak_qdot"])
        ax.plot(range(6), peak, 'o-', alpha=0.3, color='steelblue', markersize=4)
    limits = np.rad2deg(RM65_JOINT_VEL_LIMIT)
    ax.plot(range(6), limits, 'r--', linewidth=2, label='Limit')
    ax.set_xticks(range(6))
    ax.set_xticklabels(RM65_JOINT_NAMES, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Peak Velocity (deg/s)")
    ax.set_title("Joint Velocity Peaks")
    ax.legend()

    # 5. 关节力矩
    ax = axes[1, 1]
    for r in valid:
        peak = r["eval_data"]["peak_torque"]
        ax.plot(range(6), peak, 'o-', alpha=0.3, color='coral', markersize=4)
    ax.plot(range(6), RM65_TORQUE_LIMIT, 'r--', linewidth=2, label='Limit')
    ax.set_xticks(range(6))
    ax.set_xticklabels(RM65_JOINT_NAMES, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Peak Torque (Nm)")
    ax.set_title("Joint Torque Peaks")
    ax.legend()

    # 6. 成功率饼图
    ax = axes[1, 2]
    n_hit = sum(1 for r in valid if r["hit"])
    n_fair = sum(1 for r in valid if r["fair"])
    n_miss = len(valid) - n_hit - n_fair
    ax.pie([n_hit, n_fair, n_miss], labels=['Hit (<5cm)', 'Fair (<10cm)', 'Miss'],
           colors=['green', 'gold', 'red'], autopct='%1.0f%%')
    ax.set_title(f"Success Rate ({len(valid)} trials)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"汇总图表已保存: {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RM-65 实时批量评估")
    parser.add_argument("--seeds", type=int, default=20, help="评估球数（默认20）")
    parser.add_argument("--normal-flip", action="store_true", help="翻转法向量方向")
    parser.add_argument("--no-plot", action="store_true", help="不生成汇总图表")
    args = parser.parse_args()

    base_path = Path(__file__).resolve().parent.parent.parent / "configs"
    base_config = load_config(base_path / "default.yaml")
    mpc_path = base_path / "mpc.yaml"
    if mpc_path.exists():
        base_config = merge_configs(base_config, load_config(mpc_path))

    results = []
    t_batch = time.perf_counter()

    for i in range(args.seeds):
        seed = i
        logger.info(f"\n── 第 {i+1}/{args.seeds} 球 (seed={seed}) ──")
        r = run_single(seed, args, base_config)
        results.append(r)
        if r:
            status = "✅ 成功" if r["hit"] else ("⚠ 基本" if r["fair"] else "❌ 偏差")
            logger.info(
                f"  Seed {seed}: {status}, 位置误差={r['pos_error']*100:.2f}cm, "
                f"球速={r['ball_speed_before']:.1f}→{r['ball_speed_after']:.1f}m/s, "
                f"耗时={r['elapsed']:.2f}s, 实时={r['real_time_ratio']:.2f}x"
            )

    t_total = time.perf_counter() - t_batch
    logger.info(f"\n批量评估完成，总耗时: {t_total:.1f}s")

    print_summary(results)

    if not args.no_plot:
        results_dir = Path(__file__).resolve().parent.parent.parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        plot_summary(results, str(results_dir / "rm65_batch_summary.png"))


if __name__ == "__main__":
    main()

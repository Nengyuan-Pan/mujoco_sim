"""MPC + iLQR 实时滚动规划主入口。

功能：
1. 生成发球轨迹（含弹跳）
2. MPC 循环：分层策略（远距雅可比 / 中距单次迭代 / 近距多次迭代）
3. 球和机械臂均由 MuJoCo 物理引擎驱动
4. 计算完成后以真实速度回放可视化（可选）
5. 击打精度评估 + 详细计时统计

用法：
  python scripts/train_mpc.py                          # 默认：解析线性化，horizon=25，1次迭代
  python scripts/train_mpc.py --viewer                 # + 真实速度回放
  python scripts/train_mpc.py --fd                     # 有限差分线性化（对比用）
  python scripts/train_mpc.py --horizon 50 --iter 3   # 自定义参数
  python scripts/train_mpc.py --seed 42                # 指定随机种子
"""

import sys
import time
import argparse
import logging
import numpy as np
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sim.env import MujocoEnv
from src.tennis.ball import (
    generate_serve_ball,
    generate_unreachable_ball,
)
from src.tennis.hitting import (
    find_hitting_point_physics,
    compute_desired_hit_velocity,
)
from src.ilqt.cost import HittingCost
from src.ilqt.solver import ILQTSolver
from src.sim.viewer import visualize_result, plot_results

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
    env: MujocoEnv,
    x0: np.ndarray,
    p_hit: np.ndarray,
    horizon: int,
    gain: float = 50.0,
) -> np.ndarray:
    """基于雅可比转置法计算初始控制序列。

    Args:
        env: MuJoCo 环境实例。
        x0: 初始臂状态，形状 (12,)。
        p_hit: 目标击打位置，形状 (3,)。
        horizon: 规划步数。
        gain: 力矩增益。

    Returns:
        初始控制序列，形状 (horizon, 6)。
    """
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


def jacobian_transpose_step(
    env: MujocoEnv,
    x_current: np.ndarray,
    p_hit: np.ndarray,
    gain: float = 80.0,
) -> np.ndarray:
    """单步雅可比转置控制（远距模式）。

    Args:
        env: MuJoCo 环境实例。
        x_current: 当前臂状态，形状 (12,)。
        p_hit: 目标击打位置，形状 (3,)。
        gain: 力矩增益。

    Returns:
        单步控制力矩，形状 (6,)。
    """
    env.set_arm_state(x_current)
    p_ee = env.get_ee_pos()
    J_p = env.get_ee_jacp()

    err = p_hit - p_ee
    dist = np.linalg.norm(err)
    scale = gain * min(dist, 0.5)
    tau = J_p.T @ err * scale
    tau -= 2.0 * x_current[env.NQ:]

    ctrl_lo = env.model.actuator_ctrlrange[:env.NU, 0]
    ctrl_hi = env.model.actuator_ctrlrange[:env.NU, 1]
    return np.clip(tau, ctrl_lo, ctrl_hi)


def schedule_mpc_weights(
    steps_remaining: int,
    total_steps: int,
    Q_p_base: np.ndarray,
    Q_v_base: np.ndarray,
    far_threshold: int = 200,
    near_threshold: int = 25,
    Q_p_scale_far: float = 5.0,
    Q_v_scale_far: float = 0.1,
    Q_p_scale_near: float = 1.0,
    Q_v_scale_near: float = 10.0,
) -> tuple[float, float]:
    """根据剩余步数调度 Q_p 和 Q_v 权重。

    远距：高位置权重、低速度权重，重点朝目标行进。
    近距：提升速度权重，精细匹配击球条件。

    Args:
        steps_remaining: 剩余步数。
        total_steps: 总击打步数。
        Q_p_base: 基础位置权重。
        Q_v_base: 基础速度权重。
        far_threshold: 远距离阈值步数。
        near_threshold: 近距离阈值步数。
        Q_p_scale_far: 远距离 Q_p 倍率。
        Q_v_scale_far: 远距离 Q_v 倍率。
        Q_p_scale_near: 近距离 Q_p 倍率。
        Q_v_scale_near: 近距离 Q_v 倍率。

    Returns:
        (Q_p_scale, Q_v_scale): 位置和速度权重缩放因子。
    """
    if steps_remaining > far_threshold:
        return Q_p_scale_far, Q_v_scale_far
    if steps_remaining <= near_threshold:
        return Q_p_scale_near, Q_v_scale_near
    # 线性插值
    ratio = (steps_remaining - near_threshold) / max(far_threshold - near_threshold, 1)
    Q_p_scale = Q_p_scale_near + (Q_p_scale_far - Q_p_scale_near) * ratio
    Q_v_scale = Q_v_scale_near + (Q_v_scale_far - Q_v_scale_near) * ratio
    return Q_p_scale, Q_v_scale


def resample_control_sequence(
    U_old: np.ndarray,
    new_horizon: int,
) -> np.ndarray:
    """将旧控制序列重采样到新 horizon（线性插值）。

    Args:
        U_old: 旧控制序列，形状 (old_N, 6)。
        new_horizon: 新步数。

    Returns:
        新控制序列，形状 (new_horizon, 6)。
    """
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


def main() -> None:
    """MPC 主函数。"""
    parser = argparse.ArgumentParser(description="MPC+iLQR 网球击打实时规划")
    parser.add_argument("--viewer", action="store_true", help="计算完成后以真实速度回放")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--fd", action="store_true", help="使用有限差分线性化（默认解析）")
    parser.add_argument("--analytical", action="store_true", default=True, help="使用解析线性化（默认）")
    parser.add_argument("--horizon", type=int, default=None, help="短地平线步数（覆盖配置）")
    parser.add_argument("--iter", type=int, default=None, help="每次重规划迭代数（覆盖配置）")
    args = parser.parse_args()

    use_analytical = not args.fd

    # 加载配置
    base_path = Path(__file__).resolve().parent.parent / "configs"
    config = load_config(base_path / "default.yaml")
    mpc_config_path = base_path / "mpc.yaml"
    if mpc_config_path.exists():
        mpc_config = load_config(mpc_config_path)
        config = merge_configs(config, mpc_config)

    dt = float(config["sim"]["dt"])
    g = np.array(config["ball"]["gravity"], dtype=np.float64)
    shoulder_pos = np.array(config["hitting"]["shoulder_pos"], dtype=np.float64)
    workspace_radius = config["hitting"]["workspace_radius"]
    bounce_restitution = float(config["ball"].get("bounce_restitution", 0.75))

    mpc_cfg = config.get("mpc", {})
    total_horizon = int(mpc_cfg.get("total_horizon", 200))
    fixed_horizon = int(mpc_cfg.get("fixed_horizon", 25))
    replan_interval = int(mpc_cfg.get("replan_interval", 10))
    max_iter_per_plan = int(mpc_cfg.get("max_iter_per_plan", 1))
    far_threshold = int(mpc_cfg.get("far_threshold", 200))
    near_threshold = int(mpc_cfg.get("near_threshold", 25))
    Q_p_scale_far = float(mpc_cfg.get("Q_p_scale_far", 5.0))
    Q_v_scale_far = float(mpc_cfg.get("Q_v_scale_far", 0.1))
    Q_p_scale_near = float(mpc_cfg.get("Q_p_scale_near", 1.0))
    Q_v_scale_near = float(mpc_cfg.get("Q_v_scale_near", 10.0))

    # 命令行覆盖
    if args.horizon is not None:
        fixed_horizon = args.horizon
    if args.iter is not None:
        max_iter_per_plan = args.iter

    serve_cfg = config.get("serve", mpc_cfg.get("serve", {}))
    serve_distance = float(serve_cfg.get("serve_distance", 22.0))
    serve_height_range = tuple(serve_cfg.get("serve_height", [2.5, 3.0]))
    reachable_probability = float(serve_cfg.get("reachable_probability", 0.8))

    # 初始化环境
    model_path = Path(__file__).resolve().parent.parent / "src" / "robot" / "model.xml"
    env = MujocoEnv(model_path, dt=dt)

    # 初始臂状态
    init_q = np.array(config["init_q"], dtype=np.float64)
    x0 = np.zeros(env.NX)
    x0[:env.NQ] = init_q

    # ===== 生成发球轨迹 =====
    rng = np.random.default_rng(args.seed)
    generate_reachable = rng.random() < reachable_probability

    hit_cfg = config.get("hitting", {})
    hit_offset_ranges = {
        "x": hit_cfg.get("hit_offset_x_range", [0.10, 0.50]),
        "y": hit_cfg.get("hit_offset_y_range", [-0.35, 0.35]),
        "z": hit_cfg.get("hit_offset_z_range", [-0.20, 0.55]),
    }

    if generate_reachable:
        hit_time = total_horizon * dt * rng.uniform(0.6, 0.9)
        p0, v0, p_hit_expected = generate_serve_ball(
            shoulder_pos, workspace_radius, g, hit_time,
            serve_distance=serve_distance,
            serve_height_range=serve_height_range,
            bounce_restitution=bounce_restitution,
            hit_offset_ranges=hit_offset_ranges,
            rng=rng,
        )
        logger.info(f"生成可达发球: 初始位置={p0}, 初始速度={v0}")
    else:
        p0, v0 = generate_unreachable_ball(
            shoulder_pos, workspace_radius, total_horizon * dt, g, rng
        )
        logger.info(f"生成不可达球: 初始位置={p0}, 初始速度={v0}")

    # ===== 寻找击打点 =====
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

    hit_direction = np.array(config["hitting"]["hit_direction"], dtype=np.float64)
    racket_speed = float(config["hitting"]["racket_speed"])
    v_hit_desired = compute_desired_hit_velocity(hit_direction, racket_speed)

    logger.info(f"击打步数: {k_hit_total}, 击打位置: {p_hit}")
    logger.info(f"线性化: {'解析' if use_analytical else '有限差分'}, horizon={fixed_horizon}, iter={max_iter_per_plan}")

    # ===== 初始化 =====
    Q_p = np.array(config["cost"]["Q_p"], dtype=np.float64)
    Q_v = np.array(config["cost"]["Q_v"], dtype=np.float64)
    R = float(config["cost"]["R"])

    ilqt_cfg = dict(config["ilqt"])

    # 初始雅可比转矩控制
    U_prev = compute_jacobian_init_control(env, x0, p_hit, k_hit_total, gain=80.0)
    logger.info("已计算雅可比转置初始控制序列")

    # 初始化代价函数
    cost_fn = HittingCost(env, p_hit, v_hit_desired, Q_p, Q_v, R)

    # 初始化求解器
    solver = ILQTSolver(ilqt_cfg, use_analytical=use_analytical)

    # 设置球的初始状态
    env.reset(init_q)
    env.set_ball_state(p0, v0)

    # ===== MPC 主循环 =====
    x_current = x0.copy()
    X_history = [x0.copy()]
    U_history: list[np.ndarray] = []
    ball_pos_history: list[np.ndarray] = [p0.copy()]
    cost_history: list[float] = []
    pos_error_history: list[float] = []

    # 计时统计
    t_total_start = time.perf_counter()
    step_times: list[float] = []
    replan_times: list[float] = []
    linearize_times: list[float] = []
    backward_times: list[float] = []
    forward_times: list[float] = []

    # 缓存
    U_buffer: np.ndarray = np.zeros((0, env.NU))
    buffer_idx: int = 0
    is_first_plan: bool = True
    p_hit_new = p_hit.copy()
    k_hit_new = k_hit_total
    iters = 0

    logger.info(f"开始 MPC 循环，总步数={total_horizon}，击打步数={k_hit_total}")

    for step in range(total_horizon):
        t_step_start = time.perf_counter()

        # 获取球当前状态
        ball_pos, ball_vel = env.get_ball_state()

        # 判断是否需要重规划：定期 或 缓存耗尽
        need_replan = (step % replan_interval == 0) or (step == 0) or (buffer_idx >= len(U_buffer))

        if need_replan:
            t_replan_start = time.perf_counter()

            # 重新预测击打点
            remaining_horizon = total_horizon - step
            hit_info_new = find_hitting_point_physics(
                env, ball_pos, ball_vel, shoulder_pos, workspace_radius, remaining_horizon
            )

            if hit_info_new is None:
                logger.info(f"步 {step}: 球不再可达，停止 MPC")
                break

            p_hit_new = hit_info_new["p_hit"]
            k_hit_new = hit_info_new["k_hit"]

            # 分层策略
            if k_hit_new > far_threshold:
                # 远距：直接雅可比转矩
                u_cmd_jac = jacobian_transpose_step(env, x_current, p_hit_new)
                U_buffer = np.tile(u_cmd_jac, (replan_interval, 1))
                buffer_idx = 0
                iters = 0
                replan_times.append(time.perf_counter() - t_replan_start)
            else:
                # 中/近距：iLQT 规划
                # 权重调度
                Q_p_scale, Q_v_scale = schedule_mpc_weights(
                    k_hit_new, k_hit_total, Q_p, Q_v,
                    far_threshold, near_threshold,
                    Q_p_scale_far, Q_v_scale_far,
                    Q_p_scale_near, Q_v_scale_near,
                )
                cost_fn.update_target(p_hit_new, v_hit_desired)
                cost_fn.update_weights(Q_p_scale, Q_v_scale)

                # 地平线：取到击打点的剩余步数
                # 完整地平线保证终端代价在正确时刻施加
                horizon_plan = k_hit_new

                # 迭代次数：首次规划用线搜索保证质量，近距精细匹配
                iters_plan = max_iter_per_plan
                use_ls = not is_first_plan
                if is_first_plan:
                    iters_plan = max(3, max_iter_per_plan)
                    use_ls = True
                    is_first_plan = False
                elif k_hit_new <= near_threshold:
                    iters_plan = min(4, max_iter_per_plan + 2)

                # warm-start
                if len(U_prev) > 0:
                    U_warm = resample_control_sequence(U_prev, horizon_plan)
                else:
                    U_warm = compute_jacobian_init_control(
                        env, x_current, p_hit_new, horizon_plan, gain=80.0
                    )

                # 保存球状态
                ball_pos_save, ball_vel_save = env.get_ball_state()

                # iLQT 迭代 — 首次规划用线搜索保证质量，后续用阻尼直出
                X_mpc, U_mpc, iter_costs = solver.solve_few_iters(
                    env, cost_fn, x_current, U_warm,
                    max_iter=iters_plan,
                    skip_linesearch=use_ls,
                )

                # 恢复状态
                env.set_ball_state(ball_pos_save, ball_vel_save)
                env.set_arm_state(x_current)

                if len(iter_costs) > 0:
                    cost_history.append(iter_costs[-1])

                # 更新 warm-start
                if len(U_mpc) > replan_interval:
                    U_prev = U_mpc[replan_interval:]
                elif len(U_mpc) > 0:
                    U_prev = U_mpc[1:]
                else:
                    U_prev = np.zeros((0, env.NU))

                # 缓存控制
                U_buffer = U_mpc[:replan_interval]
                buffer_idx = 0
                iters = iters_plan

                replan_times.append(time.perf_counter() - t_replan_start)
        else:
            iters = 0

        # 执行控制
        if buffer_idx < len(U_buffer):
            u_cmd = U_buffer[buffer_idx]
            buffer_idx += 1
        else:
            u_cmd = jacobian_transpose_step(env, x_current, p_hit_new)

        x_current, ball_pos, ball_vel = env.step_full(u_cmd)

        X_history.append(x_current.copy())
        U_history.append(u_cmd.copy())
        ball_pos_history.append(ball_pos.copy())

        env.update_kinematics()
        pos_err = np.linalg.norm(env.get_ee_pos() - p_hit_new)
        pos_error_history.append(pos_err)

        step_time = time.perf_counter() - t_step_start
        step_times.append(step_time)

        if step % 20 == 0 or k_hit_new <= 5:
            logger.info(
                f"步 {step}: 剩余={k_hit_new}, 误差={pos_err:.4f}m, "
                f"迭代={iters}, 步耗时={step_time*1000:.1f}ms"
            )

        if k_hit_new <= 1:
            logger.info(f"步 {step}: 到达击打时刻")
            break

    # ===== 计时统计 =====
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

    # ===== 评估 =====
    X_arr = np.array(X_history)
    U_arr = np.array(U_history) if len(U_history) > 0 else np.zeros((0, env.NU))

    env.set_arm_state(x_current)
    p_ee_final = env.get_ee_pos()
    v_ee_final = env.get_ee_vel()
    pos_error = np.linalg.norm(p_ee_final - p_hit)
    vel_error = np.linalg.norm(v_ee_final - v_hit_desired)

    logger.info(f"最终位置误差: {pos_error:.4f} m")
    logger.info(f"最终速度误差: {vel_error:.4f} m/s")

    print("\n========================================")
    if pos_error < 0.05:
        print("  MPC 击打成功！（误差 < 5cm）")
    elif pos_error < 0.1:
        print("  MPC 击打基本命中！（误差 < 10cm）")
    else:
        print("  MPC 击打偏差较大，需要调整参数。")
    print(f"  击打目标位置: {np.round(p_hit, 3)}")
    print(f"  末端实际位置: {np.round(p_ee_final, 3)}")
    print(f"  位置误差: {pos_error:.4f} m")
    print(f"  速度误差: {vel_error:.4f} m/s")
    print(f"  MPC 总步数: {n_steps}")
    print(f"  线性化: {'解析' if use_analytical else '有限差分'}")
    print(f"  Horizon: {fixed_horizon}, Iter/plan: {max_iter_per_plan}")
    print(f"  总计算时间: {t_total:.2f}s")
    print(f"  平均每步: {avg_step_ms:.1f}ms (实时需 {dt*1000:.1f}ms)")
    print(f"  平均重规划: {avg_replan_ms:.1f}ms")
    print(f"  实时比率: {real_time_ratio:.2f}x (>1.0 表示可实时)")
    print("========================================\n")

    # ===== matplotlib 绘图 =====
    plot_results(
        X_arr, U_arr, p0, v0, g,
        p_hit=p_hit, v_hit=v_hit_desired,
        cost_history=cost_history if cost_history else [0.0],
        dt=dt,
        save_path=str(Path(__file__).resolve().parent.parent / "results" / "mpc_result"),
    )

    # ===== 真实速度回放 =====
    if args.viewer and len(U_history) > 0:
        logger.info("MPC 计算完成，开始真实速度回放...")

        env.reset(init_q)
        env.set_ball_state(p0, v0)

        X_replay = [env.get_arm_state().copy()]
        ball_replay = [env.get_ball_pos().copy()]

        for u_cmd in U_arr:
            env.step(u_cmd)
            X_replay.append(env.get_arm_state().copy())
            ball_replay.append(env.get_ball_pos().copy())

        X_replay = np.array(X_replay)
        ball_replay_arr = np.array(ball_replay)

        N_extra = 30
        X_vis = np.vstack([X_replay, np.tile(X_replay[-1:], (N_extra, 1))])
        U_vis = np.vstack([U_arr, np.zeros((N_extra, env.NU))])
        ball_vis = np.vstack([ball_replay_arr, np.tile(ball_replay_arr[-1:], (N_extra, 1))])
        visualize_result(env, X_vis, U_vis, p0, v0, g, config, ball_positions_phys=ball_vis)


if __name__ == "__main__":
    main()

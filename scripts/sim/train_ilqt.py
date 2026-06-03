"""主入口：运行 iLQT 优化并可视化网球击打场景。

功能：
1. 随机生成网球飞行轨迹（大概率落在工作空间内）
2. 判断球是否在机械臂工作空间内
3. 若可达，用 iLQT 规划挥拍轨迹（两阶段优化）
4. 若不可达，输出提示并不击打
5. 用 matplotlib 绘制结果图表 + 可选 MuJoCo 查看器可视化

两阶段优化策略：
- 阶段1：仅位置代价，高权重，快速到达击打点附近
- 阶段2：位置+速度代价，从阶段1结果热启动，精细调整

用法：
  python scripts/train_ilqt.py             # 优化 + 绘图
  python scripts/train_ilqt.py --viewer    # 优化 + 绘图 + MuJoCo交互查看器
"""

import sys
import argparse
import logging
import numpy as np
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.sim.env import MujocoEnv
from src.tennis.ball import (
    generate_hittable_ball,
    generate_unreachable_ball,
)
from src.tennis.hitting import (
    find_hitting_point,
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


def compute_jacobian_init_control(
    env: MujocoEnv,
    x0: np.ndarray,
    p_hit: np.ndarray,
    horizon: int,
    gain: float = 50.0,
) -> np.ndarray:
    """基于雅可比转置法计算初始控制序列。

    通过在每个时间步施加将末端执行器拉向目标位置的力矩，
    生成一个物理上合理的初始轨迹。

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

        # 雅可比转置控制：力矩方向 = J^T * (p_hit - p_ee)
        err = p_hit - p_ee
        # 逐步缩小增益（接近目标时减速）
        dist = np.linalg.norm(err)
        scale = gain * min(dist, 0.5)
        tau = J_p.T @ err * scale

        # 添加轻柔的关节阻尼（模拟自然运动）
        tau -= 2.0 * x[env.NQ:]

        tau = np.clip(tau, ctrl_lo, ctrl_hi)
        U[k] = tau

        # 仿真一步以更新状态
        x = env.step_from_state(x, tau)

    return U


def main() -> None:
    """主函数。"""
    parser = argparse.ArgumentParser(description="iLQT 网球击打优化")
    parser.add_argument("--viewer", action="store_true", help="打开 MuJoCo 交互查看器")
    parser.add_argument("--seed", type=int, default=None, help="随机种子（可复现）")
    parser.add_argument("--single-phase", action="store_true",
                        help="使用单阶段优化（跳过位置预热阶段）")
    args = parser.parse_args()

    # 加载配置
    config_path = Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml"
    config = load_config(config_path)

    dt = float(config["sim"]["dt"])
    g = np.array(config["ball"]["gravity"], dtype=np.float64)
    shoulder_pos = np.array(config["hitting"]["shoulder_pos"], dtype=np.float64)
    workspace_radius = config["hitting"]["workspace_radius"]
    horizon = int(config["ilqt"]["horizon"])

    # 初始化环境
    model_path = Path(__file__).resolve().parent.parent.parent / "src" / "robot" / "model.xml"
    env = MujocoEnv(model_path, dt=dt)

    # 初始臂状态
    init_q = np.array(config["init_q"], dtype=np.float64)
    x0 = np.zeros(env.NX)
    x0[:env.NQ] = init_q

    # ===== 随机生成网球轨迹 =====
    rng = np.random.default_rng(args.seed)

    generate_reachable = rng.random() < 0.8

    hit_cfg = config.get("hitting", {})
    hit_offset_ranges = {
        "x": hit_cfg.get("hit_offset_x_range", [0.05, 0.35]),
        "y": hit_cfg.get("hit_offset_y_range", [-0.25, 0.25]),
        "z": hit_cfg.get("hit_offset_z_range", [-0.2, 0.5]),
    }

    if generate_reachable:
        hit_time = horizon * dt * rng.uniform(0.6, 0.9)
        p0, v0, p_hit_expected = generate_hittable_ball(
            shoulder_pos, workspace_radius, hit_time, g, hit_offset_ranges, rng
        )
        logger.info(f"生成可达球: 初始位置={p0}, 初始速度={v0}")
        logger.info(f"期望击打位置: {p_hit_expected}")
    else:
        p0, v0 = generate_unreachable_ball(
            shoulder_pos, workspace_radius, horizon * dt, g, rng
        )
        logger.info(f"生成不可达球: 初始位置={p0}, 初始速度={v0}")

    # ===== 判断是否可达 =====
    hit_info = find_hitting_point(
        p0, v0, g, shoulder_pos, workspace_radius, dt, horizon
    )

    if hit_info is None:
        print("\n========================================")
        print("  网球不在工作空间内，机械臂不击打！")
        print("========================================\n")
        U_zero = np.zeros((horizon, env.NU))
        X_idle = np.zeros((horizon + 1, env.NX))
        env.set_arm_state(x0)
        X_idle[0] = x0.copy()
        for k in range(horizon):
            X_idle[k + 1] = env.step_from_state(X_idle[k], U_zero[k])
        plot_results(
            X_idle, U_zero, p0, v0, g,
            p_hit=np.zeros(3), v_hit=np.zeros(3),
            cost_history=[0.0], dt=dt,
        )
        if args.viewer:
            visualize_result(env, X_idle, U_zero, p0, v0, g, config)
        return

    # ===== 计算期望击打速度 =====
    k_hit = hit_info["k_hit"]
    p_hit = hit_info["p_hit"]
    v_ball_hit = hit_info["v_ball_hit"]

    hit_direction = np.array(config["hitting"]["hit_direction"], dtype=np.float64)
    racket_speed = float(config["hitting"]["racket_speed"])
    v_hit_desired = compute_desired_hit_velocity(hit_direction, racket_speed)

    logger.info(f"击打步数: {k_hit}, 击打位置: {p_hit}")
    logger.info(f"球在击打时刻速度: {v_ball_hit}")
    logger.info(f"期望击打速度: {v_hit_desired}")

    # ===== 计算雅可比初始控制 =====
    actual_horizon = k_hit
    ilqt_cfg = dict(config["ilqt"])
    ilqt_cfg["horizon"] = actual_horizon

    Q_p = np.array(config["cost"]["Q_p"], dtype=np.float64)
    Q_v = np.array(config["cost"]["Q_v"], dtype=np.float64)
    R = float(config["cost"]["R"])

    U_jac = compute_jacobian_init_control(env, x0, p_hit, actual_horizon, gain=80.0)
    logger.info("已计算雅可比转置初始控制序列")

    if args.single_phase:
        # ===== 单阶段优化 =====
        cost_fn = HittingCost(env, p_hit, v_hit_desired, Q_p, Q_v, R)
        solver = ILQTSolver(ilqt_cfg)
        logger.info(f"开始 iLQT 优化（单阶段），规划步数={actual_horizon}...")
        X_opt, U_opt, cost_history = solver.solve(env, cost_fn, x0, U_jac)
    else:
        # ===== 两阶段优化 =====
        # 阶段1：仅位置代价，高权重，快速到达击打点附近
        Q_p_phase1 = Q_p * 5.0
        Q_v_phase1 = np.ones(3) * 1.0
        R_phase1 = R * 0.1
        cost_fn_p1 = HittingCost(env, p_hit, np.zeros(3), Q_p_phase1, Q_v_phase1, R_phase1)

        ilqt_cfg_p1 = dict(ilqt_cfg)
        ilqt_cfg_p1["max_iter"] = 100
        ilqt_cfg_p1["tol"] = 1e-3
        solver_p1 = ILQTSolver(ilqt_cfg_p1)

        logger.info(f"阶段1：位置预热，规划步数={actual_horizon}...")
        X_p1, U_p1, cost_p1 = solver_p1.solve(env, cost_fn_p1, x0, U_jac)

        env.set_arm_state(X_p1[-1])
        pos_err_p1 = np.linalg.norm(env.get_ee_pos() - p_hit)
        logger.info(f"阶段1完成，位置误差: {pos_err_p1:.4f} m")

        # 阶段2：完整代价（位置+速度），从阶段1热启动
        cost_fn_p2 = HittingCost(env, p_hit, v_hit_desired, Q_p, Q_v, R)
        ilqt_cfg_p2 = dict(ilqt_cfg)
        ilqt_cfg_p2["mu_init"] = max(ilqt_cfg["mu_init"], 0.1)
        solver_p2 = ILQTSolver(ilqt_cfg_p2)

        logger.info(f"阶段2：精细调整（位置+速度），规划步数={actual_horizon}...")
        X_opt, U_opt, cost_history_p2 = solver_p2.solve(env, cost_fn_p2, x0, U_p1)

        # 合并代价历史
        cost_history = cost_p1 + cost_history_p2

    logger.info(f"iLQT 优化完成，最终代价: {cost_history[-1]:.6f}")

    # ===== 评估结果 =====
    env.set_arm_state(X_opt[-1])
    p_ee_final = env.get_ee_pos()
    v_ee_final = env.get_ee_vel()
    pos_error = np.linalg.norm(p_ee_final - p_hit)
    vel_error = np.linalg.norm(v_ee_final - v_hit_desired)

    logger.info(f"末端位置误差: {pos_error:.4f} m")
    logger.info(f"末端速度误差: {vel_error:.4f} m/s")

    print("\n========================================")
    if pos_error < 0.05:
        print("  击打成功！（误差 < 5cm）")
    elif pos_error < 0.1:
        print("  击打基本命中！（误差 < 10cm）")
    else:
        print("  击打偏差较大，需要调整参数。")
    print(f"  击打目标位置: {np.round(p_hit, 3)}")
    print(f"  末端实际位置: {np.round(p_ee_final, 3)}")
    print(f"  位置误差: {pos_error:.4f} m")
    print(f"  速度误差: {vel_error:.4f} m/s")
    print(f"  iLQT 迭代次数: {len(cost_history) - 1}")
    print("========================================\n")

    # ===== matplotlib 绘图 =====
    plot_results(
        X_opt, U_opt, p0, v0, g,
        p_hit=p_hit, v_hit=v_hit_desired,
        cost_history=cost_history, dt=dt,
    )

    # ===== MuJoCo 查看器（可选） =====
    if args.viewer:
        N_extra = horizon - actual_horizon + 20
        X_vis = np.vstack([X_opt, np.tile(X_opt[-1:], (N_extra, 1))])
        U_vis = np.vstack([U_opt, np.zeros((N_extra, env.NU))])
        visualize_result(env, X_vis, U_vis, p0, v0, g, config)


if __name__ == "__main__":
    main()

"""球参数网格扫描脚本：评估不同球速和飞来距离组合的可达性和精度。

用法：
  # 快速扫描（仅检查可达性和球速）：
  python scripts/scan_ball_params.py --quick

  # 快速扫描，自定义范围和步长：
  python scripts/scan_ball_params.py --quick --speeds 5,10,15,20 --distances 2,4,6,8 --angles 0,30

  # 完整 MPC 评估（慢，但给出实际击打精度）：
  python scripts/scan_ball_params.py --full --seed 42 --viewer

  # 输出到指定目录：
  python scripts/scan_ball_params.py --quick --out results/ball_scan
"""

from __future__ import annotations

import sys
import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tennis.ball import generate_ball_to_target_box, generate_ball_from_serve_box
from src.tennis.hitting import find_hitting_point_physics, compute_desired_hit_velocity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class BallScanResult:
    """单次扫描结果。"""

    ball_speed: float
    ball_distance: float
    approach_angle_deg: float
    reachable: bool = False
    k_hit: int = 0
    p_hit: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v_ball_at_hit: np.ndarray = field(default_factory=lambda: np.zeros(3))
    actual_speed_at_hit: float = 0.0
    dist_shoulder_to_hit: float = 0.0
    hit_time: float = 0.0
    p0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v0: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # 完整 MPC 评估结果
    pos_error: float = float("nan")
    vel_error: float = float("nan")
    ball_was_hit: bool = False
    mpc_success: bool = False
    mpc_time_s: float = float("nan")


def load_config(config_path: Path) -> dict:
    """加载 YAML 配置文件。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_configs(base: dict, override: dict) -> dict:
    """递归合并两个配置字典。"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def serve_box_quick_scan(
    speeds: list[float],
    config_dict: dict,
    serve_dist: float = 8.0,
    serve_height: float = 1.2,
    seed: int = 42,
) -> list[BallScanResult]:
    """发球区快速扫描：使用 generate_ball_from_serve_box 检查可达性。

    Args:
        speeds: 球速列表 (m/s)。
        config_dict: 配置字典。
        serve_dist: 发球区 Y 距离 (m)。
        serve_height: 发球区中心高度 (m)。
        seed: 随机种子。

    Returns:
        BallScanResult 列表。
    """
    dt = float(config_dict["sim"]["dt"])
    g = np.array(config_dict["ball"]["gravity"], dtype=np.float64)
    shoulder_pos = np.array([-0.1, -0.22693, 1.302645], dtype=np.float64)
    workspace_radius = 0.90

    results: list[BallScanResult] = []

    for i, speed in enumerate(speeds):
        if speed <= 0:
            continue

        rng = np.random.default_rng(seed + i * 1000)

        p0, v0, p_hit = generate_ball_from_serve_box(
            serve_box_center=(0.0, -serve_dist, serve_height),
            serve_box_halfsize=(4.0, 0.1, 0.15),
            target_center=np.array([-0.83, -0.47, 0.87], dtype=np.float64),
            target_offset=0.0,
            shoulder_pos=shoulder_pos,
            workspace_radius=workspace_radius,
            g=g,
            ball_speed=speed,
            use_bounce=True,
            max_retries=800,
            rng=rng,
        )

        result = BallScanResult(
            ball_speed=speed,
            ball_distance=float(np.linalg.norm(p_hit[:2] - p0[:2])),
            approach_angle_deg=float(np.rad2deg(np.arctan2(
                p_hit[1] - p0[1], p_hit[0] - p0[0]
            ))),
            hit_time=float(np.linalg.norm(p_hit[:2] - p0[:2])) / speed,
            p0=p0,
            v0=v0,
        )

        # 用 MuJoCo 物理检查可达性
        try:
            from src.sim.rm65_env import RM65Env

            model_path = Path(__file__).resolve().parent.parent / "src" / "robot" / "rm65_model.xml"
            env = RM65Env(model_path, dt=dt)
            env.reset(np.zeros(6))
            env.set_ball_state(p0, v0)

            hit_info = find_hitting_point_physics(
                env, p0, v0, shoulder_pos, workspace_radius,
                int(np.linalg.norm(p_hit[:2] - p0[:2]) / speed / dt) + 50,
            )
            if hit_info is not None:
                result.reachable = True
                result.k_hit = hit_info["k_hit"]
                result.p_hit = hit_info["p_hit"].copy()
                result.v_ball_at_hit = hit_info["v_ball_hit"].copy()
                result.actual_speed_at_hit = float(np.linalg.norm(hit_info["v_ball_hit"]))
                result.dist_shoulder_to_hit = float(hit_info["dist"])
        except ImportError:
            result.reachable = True
            result.p_hit = p_hit.copy()
            result.actual_speed_at_hit = float(np.linalg.norm(v0[:2]))

        results.append(result)
        logger.debug(
            f"serve_box speed={speed:.1f} reachable={result.reachable} k_hit={result.k_hit}"
        )

    return results


def quick_scan(
    speeds: list[float],
    distances: list[float],
    angles: list[float],
    config_dict: dict,
    seed: int = 42,
) -> list[BallScanResult]:
    """快速扫描：仅生成球轨迹并检查可达性，不运行 MPC。

    Args:
        speeds: 球速列表 (m/s)。
        distances: 球起始距离列表 (m)。
        angles: 角度列表 (度)。
        config_dict: 配置字典。
        seed: 随机种子。

    Returns:
        BallScanResult 列表。
    """
    dt = float(config_dict["sim"]["dt"])
    g = np.array(config_dict["ball"]["gravity"], dtype=np.float64)
    hitting_cfg = config_dict.get("hitting", {})
    shoulder_pos = np.array(hitting_cfg.get("shoulder_pos", [-0.1, -0.23, 1.30]), dtype=np.float64)
    workspace_radius = float(hitting_cfg.get("workspace_radius", 0.90))
    target_center = np.array([-0.83, -0.47, 0.87], dtype=np.float64)
    target_offset = 0.0

    results: list[BallScanResult] = []
    rng = np.random.default_rng(seed)

    for speed in speeds:
        for dist in distances:
            for angle in angles:
                if speed <= 0 or dist <= 0:
                    continue

                hit_time = dist / speed  # 物理一致性
                p0, v0, p_hit = generate_ball_to_target_box(
                    target_center, target_offset, hit_time, g,
                    shoulder_pos=shoulder_pos, workspace_radius=workspace_radius,
                    ball_speed=speed,
                    ball_distance=dist,
                    approach_angle_deg=angle,
                    rng=rng,
                    ball_direction="y",
                )

                result = BallScanResult(
                    ball_speed=speed,
                    ball_distance=dist,
                    approach_angle_deg=angle,
                    hit_time=hit_time,
                    p0=p0,
                    v0=v0,
                )

                # 检查可达性（需要 RM65Env 实例—延迟导入避免循环依赖）
                try:
                    from src.sim.rm65_env import RM65Env

                    model_path = Path(__file__).resolve().parent.parent / "src" / "robot" / "rm65_model.xml"
                    env = RM65Env(model_path, dt=dt)
                    env.reset(np.zeros(6))
                    env.set_ball_state(p0, v0)

                    hit_info = find_hitting_point_physics(
                        env, p0, v0, shoulder_pos, workspace_radius,
                        int(hit_time / dt) + 50,
                    )
                    if hit_info is not None:
                        result.reachable = True
                        result.k_hit = hit_info["k_hit"]
                        result.p_hit = hit_info["p_hit"].copy()
                        result.v_ball_at_hit = hit_info["v_ball_hit"].copy()
                        result.actual_speed_at_hit = float(np.linalg.norm(hit_info["v_ball_hit"]))
                        result.dist_shoulder_to_hit = float(hit_info["dist"])
                except ImportError as e:
                    logger.warning(f"无法导入 RM65Env: {e}，跳过硬仿真正向检查")
                    result.reachable = True  # 假定可达
                    result.p_hit = p_hit.copy()
                    result.actual_speed_at_hit = float(np.linalg.norm(v0[:2]))

                results.append(result)
                logger.debug(
                    f"speed={speed:.1f} dist={dist:.1f} angle={angle}° "
                    f"reachable={result.reachable} k_hit={result.k_hit}"
                )

    return results


def run_mpc_evaluation(
    result: BallScanResult,
    seed: int,
    config_dict: dict,
    viewer: bool = False,
) -> BallScanResult:
    """对单个参数组合运行完整 MPC 评估。

    Args:
        result: 包含 ball_speed/ball_distance 的待填充结果。
        seed: 随机种子。
        config_dict: 配置字典。
        viewer: 是否显示 MuJoCo 查看器。

    Returns:
        填充了 MPC 评估结果的 BallScanResult。
    """
    dt = float(config_dict["sim"]["dt"])
    g = np.array(config_dict["ball"]["gravity"], dtype=np.float64)
    hitting_cfg = config_dict.get("hitting", {})
    shoulder_pos = np.array(hitting_cfg.get("shoulder_pos", [-0.1, -0.23, 1.30]), dtype=np.float64)
    workspace_radius = float(hitting_cfg.get("workspace_radius", 0.90))
    target_center = np.array([-0.83, -0.47, 0.87], dtype=np.float64)
    target_offset = 0.0

    rng = np.random.default_rng(seed)

    hit_time = result.ball_distance / result.ball_speed
    p0, v0, _p_hit_expected = generate_ball_to_target_box(
        target_center, target_offset, hit_time, g,
        shoulder_pos=shoulder_pos, workspace_radius=workspace_radius,
        ball_speed=result.ball_speed,
        ball_distance=result.ball_distance,
        approach_angle_deg=result.approach_angle_deg,
        rng=rng,
        ball_direction="y",
    )

    try:
        from src.sim.rm65_env import RM65Env
        from src.ilqt.cost import HittingCost
        try:
            from src.cpp.solver_cpp import ILQTSolver
        except ImportError:
            from src.ilqt.solver import ILQTSolver

        mpc_cfg = config_dict.get("mpc", {})

        init_q = np.array([-1.5, 1.57, -0.236, 0.404, 0.446, 2.45], dtype=np.float64)
        model_path = Path(__file__).resolve().parent.parent / "src" / "robot" / "rm65_model.xml"
        env = RM65Env(model_path, dt=dt)

        env.reset(init_q)
        env.set_ball_state(p0, v0)
        env.update_kinematics()

        hit_info = find_hitting_point_physics(
            env, p0, v0, shoulder_pos, workspace_radius, 200,
        )
        if hit_info is None:
            result.reachable = False
            result.mpc_success = False
            return result

        result.reachable = True
        result.k_hit = hit_info["k_hit"]
        result.p_hit = hit_info["p_hit"].copy()
        result.v_ball_at_hit = hit_info["v_ball_hit"].copy()

        Q_p = np.array(config_dict["cost"]["Q_p"], dtype=np.float64) * 2.0
        Q_v = np.array(config_dict["cost"]["Q_v"], dtype=np.float64) * 2.0
        R = float(config_dict["cost"]["R"])
        ilqt_cfg = dict(config_dict["ilqt"])

        hit_direction = np.array(hitting_cfg["hit_direction"], dtype=np.float64)
        racket_speed = float(hitting_cfg["racket_speed"])
        v_hit_desired = compute_desired_hit_velocity(hit_direction, racket_speed)

        cost_fn = HittingCost(
            env, hit_info["p_hit"], v_hit_desired, Q_p, Q_v, R,
        )

        x0 = np.zeros(env.NX)
        x0[:env.NQ] = init_q

        # 简单雅可比初始控制
        from scripts.rm65_mpc_tube import compute_jacobian_init_control
        U_init = compute_jacobian_init_control(
            env, x0, hit_info["p_hit"], 40, gain=30.0,
        )

        solver = ILQTSolver(ilqt_cfg, use_analytical=True)

        t_start = time.perf_counter()
        _X, _U, _ = solver.solve_few_iters(
            env, cost_fn, x0, U_init,
            max_iter=5,
            skip_linesearch=True,
        )

        env.set_arm_state(x0)
        env.set_ball_state(p0, v0)

        x_cur = x0.copy()
        hit_detected = False
        pos_errors: list[float] = []

        for k in range(min(80, len(_U))):
            x_cur, ball_pos, ball_vel = env.step_full(_U[k])
            env.update_kinematics()
            p_ee = env.get_ee_pos()
            err = np.linalg.norm(p_ee - hit_info["p_hit"])
            pos_errors.append(err)

            if k >= hit_info["k_hit"] - 3 and k <= hit_info["k_hit"] + 3:
                n_contacts = env.data.ncon
                for ci in range(n_contacts):
                    c = env.data.contact[ci]
                    g1 = env.model.geom(c.geom1).name
                    g2 = env.model.geom(c.geom2).name
                    if ("ball" in g1 or "ball" in g2) and ("racket" in g1 or "racket" in g2):
                        hit_detected = True
                        break

        t_end = time.perf_counter()
        result.mpc_time_s = t_end - t_start

        if pos_errors:
            result.pos_error = float(np.min(pos_errors))
        env.set_arm_state(x_cur)
        v_ee = env.get_ee_vel()
        result.vel_error = float(np.linalg.norm(v_ee - v_hit_desired))
        result.ball_was_hit = hit_detected
        result.mpc_success = result.pos_error < 0.10 and result.ball_was_hit

    except ImportError as e:
        logger.warning(f"完整 MPC 评估失败 (import): {e}")
    except Exception as e:
        logger.warning(f"完整 MPC 评估异常 ({result.ball_speed}m/s, {result.ball_distance}m): {e}")

    return result


def print_results_table(results: list[BallScanResult]) -> None:
    """打印扫描结果表格。"""
    reached = [r for r in results if r.reachable]
    unreached = [r for r in results if not r.reachable]
    succeeded = [r for r in results if r.mpc_success]

    print("\n" + "=" * 90)
    print("  Ball Parameter Scan Results")
    print("=" * 90)
    print(f"  {'Speed(m/s)':>10} {'Dist(m)':>10} {'Angle':>8} {'Reach':>6} {'k_hit':>6} "
          f"{'ActSpeed':>10} {'ShDist(m)':>10} {'PosErr(m)':>10} {'Hit':>6}")
    print("-" * 90)

    for r in sorted(results, key=lambda x: (x.ball_speed, x.ball_distance)):
        pos_err_str = f"{r.pos_error:.4f}" if np.isfinite(r.pos_error) else "  N/A"
        actual_spd = r.actual_speed_at_hit if r.actual_speed_at_hit > 0 else float(np.linalg.norm(r.v0[:2]))
        print(
            f"  {r.ball_speed:10.1f} {r.ball_distance:10.1f} {r.approach_angle_deg:8.0f} "
            f"{'Yes' if r.reachable else 'No':>6} {r.k_hit:6d} "
            f"{actual_spd:10.1f} {r.dist_shoulder_to_hit:10.3f} "
            f"{pos_err_str:>10} {'Yes' if r.ball_was_hit else 'No':>6}"
        )

    print("-" * 90)
    print(f"  Total: {len(results)} combinations")
    print(f"  Reachable: {len(reached)} / {len(results)}")
    if succeeded:
        print(f"  MPC Success: {len(succeeded)} / {len(reached)}")
    print("=" * 90 + "\n")


def plot_heatmaps(results: list[BallScanResult], out_dir: Path, tag: str = "") -> None:
    """绘制扫描热力图。

    Args:
        results: 扫描结果列表。
        out_dir: 输出目录。
        tag: 文件名标签。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib 未安装，跳过可视化")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    reached = [r for r in results if r.reachable]

    if len(reached) < 3:
        logger.warning("可达组合太少，跳过热力图")
        return

    speeds = sorted(set(r.ball_speed for r in results))
    distances = sorted(set(r.ball_distance for r in results))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Ball Parameter Scan Heatmap [{tag}]", fontsize=14)

    # 子图 1: 可达性
    ax1 = axes[0, 0]
    reach_map = np.zeros((len(distances), len(speeds)))
    for r in results:
        si = speeds.index(r.ball_speed)
        di = distances.index(r.ball_distance)
        reach_map[di, si] = 1.0 if r.reachable else 0.0
    im1 = ax1.imshow(reach_map, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto", origin="lower")
    ax1.set_xticks(range(len(speeds)))
    ax1.set_xticklabels([f"{s:.0f}" for s in speeds])
    ax1.set_yticks(range(len(distances)))
    ax1.set_yticklabels([f"{d:.1f}" for d in distances])
    ax1.set_xlabel("Ball Speed (m/s)")
    ax1.set_ylabel("Distance (m)")
    ax1.set_title("Reachability (1=reachable)")
    plt.colorbar(im1, ax=ax1)

    # 子图 2: 实际球速
    ax2 = axes[0, 1]
    speed_map = np.full((len(distances), len(speeds)), np.nan)
    for r in reached:
        si = speeds.index(r.ball_speed)
        di = distances.index(r.ball_distance)
        speed_map[di, si] = r.actual_speed_at_hit
    im2 = ax2.imshow(speed_map, cmap="viridis", aspect="auto", origin="lower")
    ax2.set_xticks(range(len(speeds)))
    ax2.set_xticklabels([f"{s:.0f}" for s in speeds])
    ax2.set_yticks(range(len(distances)))
    ax2.set_yticklabels([f"{d:.1f}" for d in distances])
    ax2.set_xlabel("Ball Speed (m/s)")
    ax2.set_ylabel("Distance (m)")
    ax2.set_title("Actual Speed at Hit (m/s)")
    plt.colorbar(im2, ax=ax2)

    # 子图 3: k_hit
    ax3 = axes[1, 0]
    khit_map = np.full((len(distances), len(speeds)), np.nan)
    for r in reached:
        si = speeds.index(r.ball_speed)
        di = distances.index(r.ball_distance)
        khit_map[di, si] = r.k_hit
    im3 = ax3.imshow(khit_map, cmap="plasma", aspect="auto", origin="lower")
    ax3.set_xticks(range(len(speeds)))
    ax3.set_xticklabels([f"{s:.0f}" for s in speeds])
    ax3.set_yticks(range(len(distances)))
    ax3.set_yticklabels([f"{d:.1f}" for d in distances])
    ax3.set_xlabel("Ball Speed (m/s)")
    ax3.set_ylabel("Distance (m)")
    ax3.set_title("Hitting Step k_hit")
    plt.colorbar(im3, ax=ax3)

    # 子图 4: 位置误差
    ax4 = axes[1, 1]
    err_map = np.full((len(distances), len(speeds)), np.nan)
    for r in results:
        si = speeds.index(r.ball_speed)
        di = distances.index(r.ball_distance)
        if np.isfinite(r.pos_error):
            err_map[di, si] = r.pos_error
    im4 = ax4.imshow(err_map, cmap="RdYlGn_r", aspect="auto", origin="lower", vmin=0, vmax=0.2)
    ax4.set_xticks(range(len(speeds)))
    ax4.set_xticklabels([f"{s:.0f}" for s in speeds])
    ax4.set_yticks(range(len(distances)))
    ax4.set_yticklabels([f"{d:.1f}" for d in distances])
    ax4.set_xlabel("Ball Speed (m/s)")
    ax4.set_ylabel("Distance (m)")
    ax4.set_title("Position Error (m)")
    plt.colorbar(im4, ax=ax4)

    plt.tight_layout()
    out_path = out_dir / f"ball_scan_heatmap_{tag}.png" if tag else out_dir / "ball_scan_heatmap.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    logger.info(f"热力图已保存到 {out_path}")


def main() -> None:
    """球参数网格扫描主函数。"""
    parser = argparse.ArgumentParser(description="球参数网格扫描")
    parser.add_argument("--quick", action="store_true", help="快速扫描：仅检查可达性")
    parser.add_argument("--full", action="store_true", help="完整 MPC 评估（较慢）")
    parser.add_argument("--serve-box", action="store_true", help="使用发球区模式（generate_ball_from_serve_box）")
    parser.add_argument("--speeds", type=str, default="8,10,12,15",
                        help="球速列表，逗号分隔 (m/s)")
    parser.add_argument("--distances", type=str, default="3,4,5,6",
                        help="起始距离列表，逗号分隔 (m)。serve-box 模式下忽略此参数")
    parser.add_argument("--angles", type=str, default="0",
                        help="角度列表，逗号分隔 (度)。serve-box 模式下忽略此参数")
    parser.add_argument("--serve-dist", type=float, default=8.0, help="发球区 Y 方向距离 (m)")
    parser.add_argument("--serve-height", type=float, default=1.2, help="发球区中心高度 (m)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--out", type=str, default="results/ball_scan", help="输出目录")
    parser.add_argument("--viewer", action="store_true", help="显示 MuJoCo 查看器（仅 full 模式）")
    parser.add_argument("--no-plot", action="store_true", help="禁用 matplotlib 可视化")
    args = parser.parse_args()

    speeds = [float(s) for s in args.speeds.split(",")]
    distances = [float(d) for d in args.distances.split(",")]
    angles = [float(a) for a in args.angles.split(",")]

    base_path = Path(__file__).resolve().parent.parent / "configs"
    config_dict = load_config(base_path / "default.yaml")
    mpc_config_path = base_path / "mpc.yaml"
    if mpc_config_path.exists():
        mpc_config = load_config(mpc_config_path)
        config_dict = merge_configs(config_dict, mpc_config)

    if args.serve_box:
        logger.info(f"发球区模式: 球速={speeds}, center=(0,-{args.serve_dist},{args.serve_height})")
        logger.info(f"总计 {len(speeds)} 组合")
        results = serve_box_quick_scan(
            speeds, config_dict,
            serve_dist=args.serve_dist,
            serve_height=args.serve_height,
            seed=args.seed,
        )
    else:
        logger.info(f"扫描范围: 球速={speeds}, 距离={distances}, 角度={angles}")
        logger.info(f"总计 {len(speeds) * len(distances) * len(angles)} 组合")
        results = quick_scan(speeds, distances, angles, config_dict, seed=args.seed)

    logger.info(f"扫描完成: {len(results)} 组合, {sum(1 for r in results if r.reachable)} 可达")

    # 完整 MPC 评估
    if args.full:
        logger.info("开始完整 MPC 评估...")
        for r in results:
            if not r.reachable:
                continue
            logger.info(f"评估: speed={r.ball_speed}m/s, dist={r.ball_distance}m, angle={r.approach_angle_deg}°")
            run_mpc_evaluation(r, seed=args.seed, config_dict=config_dict, viewer=args.viewer)
        logger.info("完整 MPC 评估完成")

    # 输出
    print_results_table(results)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 保存 CSV
    csv_path = out_dir / "ball_scan_results.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        headers = ["ball_speed", "ball_distance", "angle_deg", "reachable", "k_hit",
                    "actual_speed", "dist_shoulder", "pos_error", "vel_error",
                    "ball_was_hit", "mpc_success", "hit_time", "mpc_time_s"]
        f.write(",".join(headers) + "\n")
        for r in results:
            row = [
                f"{r.ball_speed:.1f}",
                f"{r.ball_distance:.1f}",
                f"{r.approach_angle_deg:.0f}",
                str(int(r.reachable)),
                str(r.k_hit),
                f"{r.actual_speed_at_hit:.1f}",
                f"{r.dist_shoulder_to_hit:.3f}",
                f"{r.pos_error:.4f}" if np.isfinite(r.pos_error) else "nan",
                f"{r.vel_error:.4f}" if np.isfinite(r.vel_error) else "nan",
                str(int(r.ball_was_hit)),
                str(int(r.mpc_success)),
                f"{r.hit_time:.3f}",
                f"{r.mpc_time_s:.2f}" if np.isfinite(r.mpc_time_s) else "nan",
            ]
            f.write(",".join(row) + "\n")
    logger.info(f"CSV 已保存到 {csv_path}")

    # 可视化
    if not args.no_plot:
        tag = f"s{min(speeds):.0f}_{max(speeds):.0f}_d{min(distances):.0f}_{max(distances):.0f}"
        plot_heatmaps(results, out_dir, tag)


if __name__ == "__main__":
    main()

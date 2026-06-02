"""连续击打 20 次并保存 MP4 视频。

全程硬约束：TCP <= 1.8 m/s + 关节速度 <= 1.0x，无豁免窗口。
采用两阶段流程：
  1. 运行 v3 仿真收集轨迹（使用 --no-plot 加速）
  2. 用 MuJoCo 离屏渲染生成连续 MP4 视频

用法:
    python scripts/run_20hits_video.py
    python scripts/run_20hits_video.py --ball-speed 9
    python scripts/run_20hits_video.py --no-video   # 仅跑仿真不渲染
    python scripts/run_20hits_video.py --start-seed 5 --n-runs 10
"""
import sys
import re
import csv
import time
import logging
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("20hits")

# ===== 参数 =====
parser = argparse.ArgumentParser(description="连续击打 N 次并保存视频")
parser.add_argument("--ball-speed", type=float, default=8)
parser.add_argument("--n-runs", type=int, default=20)
parser.add_argument("--start-seed", type=int, default=0)
parser.add_argument("--max-tcp", type=float, default=1.8)
parser.add_argument("--use-tube", type=str, default="true")
parser.add_argument("--no-video", action="store_true", help="不渲染视频，仅保存数据")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--fps", type=int, default=60)
parser.add_argument("--pause-frames", type=int, default=30, help="两次击打间黑帧数")
parser.add_argument("--output-dir", type=str, default=None)
args = parser.parse_args()

output_dir = Path(args.output_dir) if args.output_dir else Path("results/20hits_speed%d" % int(args.ball_speed))
output_dir.mkdir(parents=True, exist_ok=True)

# 发球距离映射
SERVE_DIST_MAP = {5: 5.7, 6: 6.8, 7: 8.0}
serve_dist = SERVE_DIST_MAP.get(int(args.ball_speed), 9.5)


def run_v3_simulation(seed: int) -> dict:
    """运行一次 v3 仿真，返回 __RESULT__ 字典。"""
    import subprocess

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "run_tcp_limit_experiment_v3.py"),
        "--ball-speed", str(args.ball_speed),
        "--seed", str(seed),
        "--max-tcp", str(args.max_tcp),
        "--use-tube", args.use_tube,
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=180,
                       cwd=str(Path(__file__).resolve().parent.parent))
    out = r.stdout.decode("utf-8", errors="replace")

    # 保存日志
    log_path = output_dir / ("hit%02d_seed%d.log" % (seed - args.start_seed, seed))
    log_path.write_bytes(out.encode("utf-8"))

    # 解析 __RESULT__
    result = {"seed": seed}
    m = re.search(r"__RESULT__:\s+(.*)", out)
    if m:
        for pair in m.group(1).strip().split():
            if "=" in pair:
                k, v = pair.split("=", 1)
                try:
                    result[k] = float(v)
                except ValueError:
                    result[k] = v
    result["hit"] = result.get("pos_error", 999) < 0.153

    return result


def run_v3_with_trajectory(seed: int):
    """运行 v3 并收集完整轨迹数据。"""
    from src.tennis import ball as _ball_mod
    from src.sim.rm65_env import RM65Env

    # 构造 v2 参数
    import scripts.rm65_mpc_tube_constraint_realtime_v2 as main_mod

    saved_argv = sys.argv[:]
    sys.argv = [
        "rm65_mpc_tube_constraint_realtime_v2.py",
        "--serve-box",
        "--ball-speed", str(args.ball_speed),
        "--seed", str(seed),
        "--serve-distance", str(serve_dist),
        "--no-backswing",
        "--no-plot",
        "--realtime",
        "--use_tube", args.use_tube,
        "--max-tcp", str(args.max_tcp),
        "--terminal-exempt-steps", "0",
    ]

    # Monkey-patch 以收集轨迹
    original_main_body = main_mod.main

    collected = {}

    # 无法直接 patch main 内部变量
    # 改用方法：运行两次 —— 第一次收集结果，第二次用 --viewer 获取轨迹
    # 但 --viewer 会弹窗口...

    # 最优方案：直接修改 sys.argv 调 main，然后从 X_history / U_history 提取
    # 由于 main() 是一个巨型函数，我们用更简单的方式：

    # 方案：运行仿真，通过 env 回放收集轨迹
    # 先运行一次获取控制序列
    try:
        main_mod.main()
    except SystemExit:
        pass
    finally:
        sys.argv = saved_argv

    return None


def render_mp4_from_segments(segments: list, video_path: Path) -> None:
    """从轨迹段列表渲染连续 MP4。"""
    import mujoco
    from src.sim.rm65_env import RM65Env

    env = RM65Env()
    model = env.model
    data = env.data
    dt = env.dt
    NQ = env.NQ
    bq = env.BALL_QPOS_START

    model.vis.global_.offwidth = args.width
    model.vis.global_.offheight = args.height

    renderer = mujoco.Renderer(model, width=args.width, height=args.height)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 3.5
    cam.elevation = -15
    cam.azimuth = 135
    cam.lookat[:] = [0.0, 0.0, 1.0]

    # 灯光
    model.light_pos[0] = [0.0, 0.0, 8.0]
    model.light_dir[0] = [0.0, 0.0, -1.0]
    model.light_diffuse[0] = [1.4, 1.45, 1.55]
    model.light_ambient[0] = [0.3, 0.3, 0.35]
    if model.nlight > 1:
        model.light_pos[1] = [2.0, -2.0, 3.0]
        model.light_dir[1] = [-0.4, 0.3, -0.8]
        model.light_diffuse[1] = [1.2, 1.15, 1.05]
        model.light_active[1] = True

    import imageio
    writer = imageio.get_writer(
        str(video_path),
        fps=args.fps,
        output_params=["-crf", "18", "-pix_fmt", "yuv420p"],
    )

    black_frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)

    for seg_idx, seg in enumerate(segments):
        X = seg["X"]
        ball_pos = seg["ball"]
        init_q = seg["init_q"]
        init_q_left = seg["init_q_left"]
        p0 = seg["p0"]
        v0 = seg["v0"]

        # 初始化环境
        env.reset(init_q)
        data.qpos[NQ:NQ + env.LEFT_ARM_NQ] = init_q_left
        data.qvel[NQ:NQ + env.LEFT_ARM_NQ] = 0.0
        env.update_kinematics()
        env.set_ball_state(p0, v0)

        U = seg["U"]
        post_steps = seg.get("post_steps", 40)
        total_steps = len(U) + post_steps

        for step in range(total_steps):
            if step < len(U):
                env.step(U[step])
            else:
                # 击打后自由运动（零力矩）
                env.step(np.zeros(env.NU))

            # 渲染帧
            renderer.update_scene(data, camera=cam)
            writer.append_data(renderer.render())

        # 黑帧间隔
        for _ in range(args.pause_frames):
            writer.append_data(black_frame)

        hit_str = "HIT" if seg.get("hit", False) else "MISS"
        logger.info("  段 %d/%d 渲染完成 (%s, %d 帧)" % (
            seg_idx + 1, len(segments), hit_str, total_steps))

    writer.close()
    renderer.close()
    logger.info("视频已保存: %s" % video_path)


def run_full_simulation_with_trajectory(seed: int) -> dict:
    """运行完整仿真并返回轨迹段数据。"""
    # 需要直接调用 v2 main 的核心逻辑
    # 最简方案：import 后通过 monkey-patch 收集
    import scripts.rm65_mpc_tube_constraint_realtime_v2 as main_mod
    from src.sim.rm65_env import RM65Env

    env = RM65Env()
    dt = env.dt
    NQ = env.NQ

    saved_argv = sys.argv[:]
    sys.argv = [
        "rm65_mpc_tube_constraint_realtime_v2.py",
        "--serve-box",
        "--ball-speed", str(args.ball_speed),
        "--seed", str(seed),
        "--serve-distance", str(serve_dist),
        "--no-backswing",
        "--no-plot",
        "--realtime",
        "--use_tube", args.use_tube,
        "--max-tcp", str(args.max_tcp),
        "--terminal-exempt-steps", "0",
    ]

    # Patch env.step 收集轨迹
    X_traj = []
    ball_traj = []
    U_traj = []

    original_step = env.step
    original_reset = env.reset

    def patched_step(u):
        result = original_step(u)
        X_traj.append(env.get_arm_state().copy())
        ball_traj.append(env.get_ball_pos().copy())
        U_traj.append(u.copy())
        return result

    env.step = patched_step

    # 运行主函数
    result_data = {"hit": False, "pos_error": None, "seed": seed}
    try:
        main_mod.main()
    except SystemExit:
        pass
    except Exception as e:
        logger.warning("seed=%d 仿真异常: %s" % (seed, e))
    finally:
        sys.argv = saved_argv
        env.step = original_step

    # 从 stdout 或内部变量提取结果
    # main_mod 的 main() 中有全局变量可以访问
    # 但 main() 是函数，内部变量不可直接访问

    # 通过重新解析日志获取结果
    # 更简单的做法：直接使用 X_traj 和 ball_traj 来判断命中
    if len(X_traj) > 0 and len(ball_traj) > 0:
        min_dist = min(
            np.linalg.norm(X_traj[i][:NQ][:3] - ball_traj[i])
            if i < len(ball_traj) else 999
            for i in range(min(len(X_traj), len(ball_traj)))
        )
        # 用末端执行器位置而非关节位置计算距离
        env.reset(X_traj[0][:NQ])
        ee_dists = []
        for i in range(min(len(X_traj), len(ball_traj))):
            env.set_arm_state(X_traj[i])
            env.update_kinematics()
            ee_pos = env.get_ee_pos()
            ball_p = ball_traj[i]
            ee_dists.append(np.linalg.norm(ee_pos - ball_p))
        min_ee_dist = min(ee_dists) if ee_dists else 999
        result_data["min_ee_dist"] = min_ee_dist
        result_data["hit"] = min_ee_dist < 0.153

    return {
        "result": result_data,
        "X": np.array(X_traj) if X_traj else np.zeros((0, env.NX)),
        "ball": np.array(ball_traj) if ball_traj else np.zeros((0, 3)),
        "U": np.array(U_traj) if U_traj else np.zeros((0, env.NU)),
        "init_q": X_traj[0][:NQ].copy() if len(X_traj) > 0 else np.zeros(NQ),
        "init_q_left": np.zeros(env.LEFT_ARM_NQ),
    }


def main():
    logger.info("=" * 60)
    logger.info("连续击打测试: %d 次, 球速=%.1f m/s" % (args.n_runs, args.ball_speed))
    logger.info("约束: TCP<=%.1f m/s, qdot<=1.0x, 无豁免" % args.max_tcp)
    logger.info("输出: %s" % output_dir)
    logger.info("=" * 60)

    if args.no_video:
        # ===== 仅仿真模式 =====
        results = []
        t0 = time.perf_counter()
        for i in range(args.n_runs):
            seed = args.start_seed + i
            result = run_v3_simulation(seed)
            results.append(result)
            hit_str = "HIT" if result.get("hit") else "MISS"
            pe = result.get("pos_error", 0)
            qd = result.get("max_qdot", 0)
            logger.info("[%2d/%d] seed=%2d %s  err=%.1fmm  qdot=%.2fx  (%.0fs)" % (
                i + 1, args.n_runs, seed, hit_str,
                pe * 1000 if pe else 0, qd, time.perf_counter() - t0))

        # 保存 CSV
        csv_path = output_dir / "results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["seed", "hit", "pos_error", "vel_error",
                                               "min_dist", "max_qdot", "max_tcp", "hit_type",
                                               "hit_time_error_ms", "ball_near_ms"],
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(results)

        hits = sum(1 for r in results if r.get("hit"))
        logger.info("结果: %d/%d 命中 (%.0f%%)" % (hits, args.n_runs, hits / args.n_runs * 100))
        logger.info("CSV: %s" % csv_path)
    else:
        # ===== 仿真 + 视频模式 =====
        segments = []
        results = []
        t0 = time.perf_counter()

        for i in range(args.n_runs):
            seed = args.start_seed + i
            logger.info("[%2d/%d] 仿真 seed=%d ..." % (i + 1, args.n_runs, seed))

            seg = run_full_simulation_with_trajectory(seed)

            result = seg["result"]
            results.append(result)
            seg["hit"] = result.get("hit", False)
            segments.append(seg)

            hit_str = "HIT" if result.get("hit") else "MISS"
            logger.info("[%2d/%d] %s  (%.0fs)" % (
                i + 1, args.n_runs, hit_str, time.perf_counter() - t0))

        # 保存 CSV
        csv_path = output_dir / "results.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["seed", "hit", "min_ee_dist"],
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(results)

        # 保存轨迹数据
        npz_path = output_dir / "trajectories.npz"
        save_data = {}
        for i, seg in enumerate(segments):
            save_data["X_%02d" % i] = seg["X"]
            save_data["ball_%02d" % i] = seg["ball"]
            save_data["U_%02d" % i] = seg["U"]
        np.savez_compressed(str(npz_path), **save_data)
        logger.info("轨迹已保存: %s" % npz_path)

        # 渲染视频
        valid_segments = [s for s in segments if len(s["X"]) > 0]
        if valid_segments:
            video_path = output_dir / "20hits_continuous.mp4"
            # 需要设置球的初始状态
            for seg in valid_segments:
                if len(seg["ball"]) > 0:
                    seg["p0"] = seg["ball"][0]
                    seg["v0"] = np.zeros(3)  # 近似
                else:
                    seg["p0"] = np.zeros(3)
                    seg["v0"] = np.zeros(3)
            render_mp4_from_segments(valid_segments, video_path)
        else:
            logger.warning("无有效轨迹数据，跳过视频渲染")

    logger.info("完成! 总耗时: %.0fs" % (time.perf_counter() - t0))


if __name__ == "__main__":
    main()

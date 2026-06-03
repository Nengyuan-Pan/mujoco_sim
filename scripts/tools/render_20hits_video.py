"""从 v2 MPC 管线渲染连续 MP4 视频（对齐 rm65_batch_viz.py 质量）。

通过逐次运行 v2（--dump-trajectory），收集完整轨迹数据，
然后用 MuJoCo 离屏渲染器以运动学回放方式录制（含碰撞反弹、段间过渡）。

特性:
  - 1920x1080 @ 60fps
  - 四灯布光 + specular
  - 碰撞弹性反弹（球拍击球后球反弹飞出）
  - 段间 PD 回位过渡（无黑帧跳变）
  - 按仿真 dt 精确时间采样

用法:
    python scripts/render_20hits_video.py
    python scripts/render_20hits_video.py --ball-speed 8
    python scripts/render_20hits_video.py --ball-speed 8 --seeds 0 1 2 3 4
    python scripts/render_20hits_video.py --ball-speed 7 --episodes 10
"""
import sys
import pickle
import time
import logging
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("render_video")

parser = argparse.ArgumentParser(description="v2 20次击打视频渲染")
parser.add_argument("--ball-speed", type=float, default=8)
parser.add_argument("--max-tcp", type=float, default=1.8)
parser.add_argument("--use-tube", type=str, default="true")
parser.add_argument("--seeds", type=int, nargs="+", default=None)
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
parser.add_argument("--fps", type=int, default=60)
parser.add_argument("--output", type=str, default=None)
parser.add_argument("--return-frames", type=int, default=30, help="段间回位过渡帧数")
args = parser.parse_args()

SERVE_DIST_MAP = {5: 5.7, 6: 6.8, 7: 8.0}
serve_dist = SERVE_DIST_MAP.get(int(args.ball_speed), 9.5)

TEMP_DIR = Path("results/_traj_temp")

INIT_Q = np.array([-1.5, 1.57, -0.236, 0.404, 0.446, 2.45], dtype=np.float64)
INIT_Q_LEFT = np.array([-0.373, -1.57, 0.236, -0.404, -0.446, -2.45], dtype=np.float64)


def run_one_and_dump(seed: int) -> dict | None:
    """运行一次 v2 仿真，通过 --dump-trajectory 保存轨迹，返回轨迹数据。"""
    traj_path = TEMP_DIR / f"seed_{seed:03d}.pkl"

    saved_argv = sys.argv[:]
    sys.argv = [
        "rm65_mpc_tube_constraint_realtime_v2.py",
        "--serve-box", "--ball-speed", str(args.ball_speed),
        "--seed", str(seed), "--serve-distance", str(serve_dist),
        "--no-backswing", "--no-plot", "--realtime",
        "--use_tube", args.use_tube,
        "--max-tcp", str(args.max_tcp),
        "--terminal-exempt-steps", "0",
        "--dump-trajectory", str(traj_path),
    ]

    import scripts.rm65_mpc_tube_constraint_realtime_v2 as main_mod
    try:
        main_mod.main()
    except SystemExit:
        pass
    except Exception as e:
        logger.warning("seed=%d 异常: %s" % (seed, e))
    finally:
        sys.argv = saved_argv

    if traj_path.exists():
        with open(traj_path, "rb") as f:
            data = pickle.load(f)
        traj_path.unlink()
        return data
    else:
        logger.warning("seed=%d 未生成轨迹文件" % seed)
        return None


def setup_camera_lights(model) -> None:
    """配置四灯布光方案（与 rm65_batch_viz.py 一致）。"""
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


def build_replay_data(env, traj: dict) -> tuple[np.ndarray, np.ndarray]:
    """为单次击打构建回放轨迹（含碰撞后弹性反弹）。

    与 rm65_batch_viz.py 的 build_replay_data 逻辑一致。

    Args:
        env: RM65Env 实例。
        traj: run_one_and_dump 返回的轨迹字典。

    Returns:
        (X_replay, ball_replay): 臂状态轨迹和球位置轨迹。
    """
    U_arr = np.array(traj["U_history"])
    p0 = traj["p0"]
    v0 = traj["v0"]
    hit_step = traj.get("hit_step", -1)
    post_hit_steps = traj.get("post_hit_steps", 80)
    init_q = traj["init_q"]
    init_q_left = traj["init_q_left"]

    NQ = env.NQ

    env.reset(init_q)
    env.data.qpos[NQ:NQ + env.LEFT_ARM_NQ] = init_q_left
    env.data.qvel[NQ:NQ + env.LEFT_ARM_NQ] = 0.0
    env.update_kinematics()
    env.set_ball_state(p0, v0)

    X_replay = [env.get_arm_state().copy()]
    ball_replay = [env.get_ball_pos().copy()]
    rebound_applied = False

    for i in range(len(U_arr)):
        u_cmd = U_arr[i]
        enable_collision = (hit_step >= 0 and abs(i - hit_step) <= 5)
        if hasattr(env, "set_arm_collision"):
            env.set_arm_collision(enable_collision)
        ball_vel_pre = env.get_ball_vel().copy() if enable_collision else np.zeros(3)
        env.step(u_cmd)
        if hasattr(env, "_handle_ball_bounce"):
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
                    break
        X_replay.append(env.get_arm_state().copy())
        ball_replay.append(env.get_ball_pos().copy())

    # 击打后继续仿真（PD 保持）
    x_current = env.get_arm_state()
    for _ in range(post_hit_steps):
        q_hold = x_current[:NQ].copy()
        u_hold = 100.0 * (q_hold - x_current[:NQ]) - 10.0 * x_current[NQ:]
        u_hold = np.clip(u_hold,
                         env.model.actuator_ctrlrange[:env.NU, 0],
                         env.model.actuator_ctrlrange[:env.NU, 1])
        x_current, ball_pos, _ = env.step_full(u_hold)
        X_replay.append(x_current.copy())
        ball_replay.append(ball_pos.copy())

    if hasattr(env, "set_arm_collision"):
        env.set_arm_collision(True)

    return np.array(X_replay), np.array(ball_replay)


def generate_return_trajectory(
    env,
    x_current: np.ndarray,
    x_target: np.ndarray,
    n_frames: int = 30,
    kp: float = 150.0,
    kd: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """生成从当前位姿回到初始位姿的平滑过渡轨迹。

    Args:
        env: RM65Env 实例。
        x_current: 当前臂状态。
        x_target: 目标臂状态。
        n_frames: 过渡帧数。
        kp: PD 位置增益。
        kd: PD 速度增益。

    Returns:
        (X_return, ball_return): 臂状态和球位置轨迹。
    """
    X_return = [x_current.copy()]
    ball_return = [env.get_ball_pos().copy()]
    x = x_current.copy()

    for _ in range(n_frames):
        q_des = x_target[:env.NQ]
        qdot_des = x_target[env.NQ:]
        tau = kp * (q_des - x[:env.NQ]) + kd * (qdot_des - x[env.NQ:])
        tau = np.clip(tau,
                      env.model.actuator_ctrlrange[:env.NU, 0],
                      env.model.actuator_ctrlrange[:env.NU, 1])
        x, ball_pos, _ = env.step_full(tau)
        X_return.append(x.copy())
        ball_return.append(ball_pos.copy())

    return np.array(X_return), np.array(ball_return)


def render_offscreen(
    env,
    segments: list[dict],
    video_path: Path,
) -> None:
    """离屏渲染所有段并写入 MP4 视频。

    按仿真 dt 精确时间采样，与 rm65_batch_viz.py 的 render_offscreen 一致。

    Args:
        env: RM65Env 实例。
        segments: 段列表，每段包含 X_replay, ball_replay, init_q_left 等。
        video_path: 输出 MP4 文件路径。
    """
    import mujoco
    import imageio

    width = args.width
    height = args.height
    fps = args.fps
    dt = env.dt
    model = env.model
    data = env.data

    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height

    renderer = mujoco.Renderer(model, width=width, height=height)
    setup_camera_lights(model)

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 3.5
    cam.elevation = -15
    cam.azimuth = 135
    cam.lookat[:] = [0.0, 0.0, 1.0]

    writer = imageio.get_writer(
        str(video_path),
        fps=fps,
        output_params=["-crf", "18", "-pix_fmt", "yuv420p"],
    )

    video_dt = 1.0 / fps
    NQ = env.NQ
    bq = env.BALL_QPOS_START

    logger.info("开始离屏渲染: %dx%d @ %dfps, 输出: %s" % (width, height, fps, video_path))

    for seg_idx, seg in enumerate(segments):
        X_replay = seg["X_replay"]
        ball_replay = seg["ball_replay"]
        init_q_left = seg["init_q_left"]

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

            data.qpos[:NQ] = arm_x[:NQ]
            data.qvel[:NQ] = arm_x[NQ:]
            data.qpos[NQ:NQ + env.LEFT_ARM_NQ] = init_q_left
            data.qvel[NQ:NQ + env.LEFT_ARM_NQ] = 0.0

            if idx < len(ball_replay):
                bp = ball_replay[idx]
                data.qpos[bq:bq + 3] = bp
                data.qpos[bq + 3:bq + 7] = [1, 0, 0, 0]

            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            pixels = renderer.render()
            writer.append_data(pixels)
            next_video_time += video_dt
            n_written += 1

        hit_str = seg.get("hit_label", "?")
        logger.info("  段 %d/%d 渲染完成 (%s, %d 仿真步, %d 视频帧)" % (
            seg_idx + 1, len(segments), hit_str, total_frames, n_written))

    writer.close()
    renderer.close()
    logger.info("MP4 保存完成: %s" % video_path)


def main():
    output_path = Path(args.output) if args.output else Path("results/20hits_speed%d.mp4" % int(args.ball_speed))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    seeds = args.seeds if args.seeds is not None else list(range(args.episodes))
    logger.info("渲染视频: %d 次击打, 球速=%.1f m/s" % (len(seeds), args.ball_speed))
    logger.info("输出: %s (%dx%d @ %dfps)" % (output_path, args.width, args.height, args.fps))

    # ===== 阶段1: 运行仿真，收集轨迹 =====
    trajectories = []
    t0 = time.perf_counter()

    for i, seed in enumerate(seeds):
        logger.info("[%d/%d] 仿真 seed=%d ..." % (i + 1, len(seeds), seed))
        traj = run_one_and_dump(seed)
        if traj is not None:
            trajectories.append(traj)
            hit_str = "HIT" if traj.get("hit_type") != "miss" else "MISS"
            pos_err = traj.get("pos_error", 0)
            logger.info("[%d/%d] seed=%d %s  pos_err=%.1fmm  (%.0fs)" % (
                i + 1, len(trajectories), seed, hit_str,
                pos_err * 1000 if pos_err else 0, time.perf_counter() - t0))
        else:
            logger.warning("[%d/%d] seed=%d 无轨迹数据" % (i + 1, len(seeds), seed))

    hits = sum(1 for t in trajectories if t.get("hit_type") != "miss")
    logger.info("汇总: %d/%d 命中 (%.0f%%)" % (hits, len(trajectories), hits / max(len(trajectories), 1) * 100))

    if not trajectories:
        logger.error("无轨迹数据，退出")
        return

    # ===== 阶段2: 构建回放数据 =====
    from src.sim.rm65_env import RM65Env

    model_path = Path(__file__).resolve().parent.parent.parent / "src" / "robot" / "rm65_model.xml"
    env = RM65Env(model_path)

    x0 = np.zeros(env.NX)
    x0[:env.NQ] = INIT_Q

    segments = []
    logger.info("构建回放数据...")

    for i, traj in enumerate(trajectories):
        logger.info("  回放 %d/%d ..." % (i + 1, len(trajectories)))
        X_replay, ball_replay = build_replay_data(env, traj)
        hit_label = "HIT" if traj.get("hit_type") != "miss" else "MISS"
        segments.append({
            "X_replay": X_replay,
            "ball_replay": ball_replay,
            "init_q_left": traj["init_q_left"],
            "hit_label": hit_label,
        })

        # 段间过渡：PD 回位
        if i < len(trajectories) - 1:
            x_last = X_replay[-1]
            X_return, ball_return = generate_return_trajectory(
                env, x_last, x0, n_frames=args.return_frames)
            segments.append({
                "X_replay": X_return,
                "ball_replay": ball_return,
                "init_q_left": traj["init_q_left"],
                "hit_label": "过渡",
            })

    # ===== 阶段3: 渲染 MP4 =====
    logger.info("开始渲染视频 (%d 段)..." % len(segments))
    render_offscreen(env, segments, output_path)

    logger.info("完成! 总耗时: %.0fs" % (time.perf_counter() - t0))

    # 清理临时目录
    if TEMP_DIR.exists():
        for f in TEMP_DIR.iterdir():
            f.unlink()
        TEMP_DIR.rmdir()


if __name__ == "__main__":
    main()

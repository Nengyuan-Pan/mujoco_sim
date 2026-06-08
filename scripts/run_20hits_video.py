"""连续运行 20 次 v9 击打仿真并保存 MP4 视频（单进程内循环）。

设计:
  - 单进程内复用 env / solver / cost_fn
  - 循环 20 次：发球 -> MPC 击打 -> 随挥 -> PD 回初始位 -> 下一球
  - 每个仿真步同时离屏渲染一帧（真实速率，dt=5ms/帧）
  - 击打间隙无黑帧，机械臂自然回到准备姿态

用法:
    python scripts/run_20hits_video.py --serve-box --ball-speed 7
    python scripts/run_20hits_video.py --no-video          # 仅仿真不渲染
    python scripts/run_20hits_video.py --n-runs 10
"""

from __future__ import annotations

import sys
import csv
import time
import logging
import argparse
from pathlib import Path

import numpy as np

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
from src.ilqt.robot_limits import (
    RobotLimits,
    check_one_step_feasibility,
    ExecutionMetrics,
)
from src.ilqt.async_replanner import AsyncReplanner, PlanRequest, PlanResult
try:
    from src.cpp.solver_cpp import ILQTSolver
except ImportError:
    from src.ilqt.solver import ILQTSolver

from scripts.rm65_mpc_v9 import (
    fix_joint5_control,
    fix_joint5_control_trajectory,
    load_config,
    merge_configs,
    compute_jacobian_init_control,
    compute_joint1_backswing_trajectory,
    generate_backswing_warm_start,
    resample_control_sequence,
    compute_r_schedule,
    TubeConfig,
    TubeHittingCostWrapper,
    TubeOnlyCost,
    SoftminOnlyCost,
    search_hit_window,
    build_hitting_tube,
    HitWindow,
    HittingTube,
    ReplanState,
    do_replan,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("src.ilqt.robot_limits").setLevel(logging.WARNING)
logger = logging.getLogger("20hits_v9")

parser = argparse.ArgumentParser(description="V9 连续 20 次击打 + MP4 视频")
parser.add_argument("--ball-speed", type=float, default=7)
parser.add_argument("--n-runs", type=int, default=20)
parser.add_argument("--start-seed", type=int, default=0)
parser.add_argument("--max-tcp", type=float, default=1.8)
parser.add_argument("--no-video", action="store_true", help="不渲染视频")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--no-backswing", action="store_true")
parser.add_argument("--softmin-beta", type=float, default=5.0)
parser.add_argument("--window-ms", type=float, default=50.0)
parser.add_argument("--target-speed", type=float, default=1.8)
parser.add_argument("--serve-box", action="store_true", default=True)
parser.add_argument("--no-serve-box", action="store_true")
parser.add_argument("--serve-distance", type=float, default=8.0)
parser.add_argument("--output-dir", type=str, default=None)
parser.add_argument("--return-steps", type=int, default=80,
                    help="击打后 PD 回初始位步数（默认80）")
parser.add_argument("--pause-steps", type=int, default=40,
                    help="回初始位后静止等待步数（默认40）")
parser.add_argument("--ablation", choices=["full", "tube_only", "softmin_only", "none"],
                    default="full",
                    help="消融模式: full=Tube+Softmin(默认), tube_only, softmin_only, none")
parser.add_argument("--no-follow-through", action="store_true",
                    help="禁用随挥（消融实验用）")
parser.add_argument("--follow-trigger", choices=["planned", "contact"],
                    default="planned", help="随挥触发方式")
parser.add_argument("--hit-shift", type=float, default=0.0, help="终端偏移距离 (m)")
parser.add_argument("--normal-weight", type=float, default=500000.0)
parser.add_argument("--replan-interval", type=int, default=None)
parser.add_argument("--fix-joint5", action="store_true")
parser.add_argument("--backswing", type=float, default=0.6)
parser.add_argument("--bs-ratio", type=float, default=0.35)
parser.add_argument("--no-r-decay", action="store_true")
parser.add_argument("--r-decay", type=float, default=0.40)
parser.add_argument("--terminal-exempt-steps", type=int, default=0)
parser.add_argument("--fps", type=int, default=None,
                    help="输出视频帧率（默认=1/dt=200fps）")
args = parser.parse_args()

if args.no_serve_box:
    args.serve_box = False

output_dir = Path(args.output_dir) if args.output_dir else Path(
    "results/20hits_v9_speed%d" % int(args.ball_speed))
output_dir.mkdir(parents=True, exist_ok=True)

SERVE_DIST_MAP = {5: 5.7, 6: 6.8, 7: 8.0, 8: 9.0, 9: 9.5, 10: 10.0}

ablation_mode = args.ablation
use_tube = ablation_mode in ("full", "tube_only")
need_candidates = ablation_mode in ("full", "tube_only", "softmin_only")
logger.info("[ablation] mode=%s (Tube=%s, Softmin=%s)",
            ablation_mode,
            "ON" if use_tube else "OFF",
            "ON" if ablation_mode in ("full", "softmin_only") else "OFF")


def run_batch() -> None:
    """主函数：初始化环境，循环 N 次击打。"""
    import mujoco
    import mujoco.viewer

    base_path = Path(__file__).resolve().parent.parent / "configs"
    config_dict = load_config(base_path / "default.yaml")
    v5_config_path = base_path / "v5_active_hit.yaml"
    if v5_config_path.exists():
        config_dict = merge_configs(config_dict, load_config(v5_config_path))
    mpc_config_path = base_path / "mpc.yaml"
    if mpc_config_path.exists():
        config_dict = merge_configs(config_dict, load_config(mpc_config_path))

    dt = float(config_dict["sim"]["dt"])
    g = np.array(config_dict["ball"]["gravity"], dtype=np.float64)

    shoulder_pos = np.array([-0.1, -0.22693, 1.302645], dtype=np.float64)
    workspace_radius = 0.90

    total_horizon = 200
    fixed_horizon = 40
    replan_interval = args.replan_interval if args.replan_interval is not None else 30
    max_iter_per_plan = 5
    first_plan_iters = 15
    near_plan_iters = 20

    if args.serve_box:
        fixed_horizon = 120
        total_horizon = 250
        serve_dist = args.serve_distance if args.serve_distance != 8.0 else SERVE_DIST_MAP.get(
            int(args.ball_speed), 9.5)
        replan_interval = 20
        max_iter_per_plan = 10
        first_plan_iters = 30
        near_plan_iters = 5
        logger.info("serve-box: distance=%.1fm, horizon=%d, replan=%d",
                     serve_dist, fixed_horizon, replan_interval)

    init_q = np.array([-1.5, 1.57, -0.236, 0.404, 0.446, 2.45], dtype=np.float64)
    init_q_left = np.array([-0.373, -1.57, 0.236, -0.404, -0.446, -2.45],
                           dtype=np.float64)

    fix_joint5_angle: float | None = init_q[5] if args.fix_joint5 else None
    use_backswing = not args.no_backswing
    backswing_offset = -abs(args.backswing)
    backswing_ratio = args.bs_ratio
    use_r_decay = not args.no_r_decay
    r_decay_ratio = args.r_decay

    if args.no_follow_through:
        follow_through_length = 0.0
        follow_through_steps = 0
        follow_through_v_terminal = 0.0
    else:
        follow_through_length = args.hit_shift
        follow_through_steps = int(
            config_dict["hitting"].get("follow_through_steps", 160))
        follow_through_v_terminal = float(
            config_dict["hitting"].get("follow_through_v_terminal", 0.3))

    tube_cfg = TubeConfig(
        window_half_ms=args.window_ms,
        Q_p_tube=500.0,
        Q_v_tube=0.0,
        Q_n_tube=0.0,
        tube_cost_ratio=1.0,
        softmin_beta=args.softmin_beta,
        use_softmin_terminal=ablation_mode in ("full", "softmin_only"),
    )

    model_path = Path(__file__).resolve().parent / "src" / "robot" / "rm65_model.xml"
    if not model_path.exists():
        model_path = Path(__file__).resolve().parent.parent / "src" / "robot" / "rm65_model.xml"
    env = RM65Env(model_path, dt=dt)
    env.init_q_left = init_q_left
    NQ = env.NQ
    NU = env.NU
    NX = env.NX
    LEFT_NQ = env.LEFT_ARM_NQ

    rl_cfg = config_dict.get("robot_limits", {})
    if args.max_tcp > 0:
        rl_cfg["max_tcp_speed"] = args.max_tcp
    if args.terminal_exempt_steps > 0:
        rl_cfg["terminal_exempt_steps"] = args.terminal_exempt_steps
    robot_limits = RobotLimits.from_config(
        rl_cfg, dt=dt, ctrlrange=env.model.actuator_ctrlrange[:NU],
    )

    import mujoco as _mj
    _hard_x_body_ids = [
        _mj.mj_name2id(env.model, _mj.mjtObj.mjOBJ_BODY, n)
        for n in ("r_link1", "r_link2", "r_link3", "r_link4",
                   "r_link5", "r_link6", "r_flange", "r_racket_body")
    ]

    ctrl_lo = env.model.actuator_ctrlrange[:NU, 0]
    ctrl_hi = env.model.actuator_ctrlrange[:NU, 1]

    Q_p = np.array(config_dict["cost"]["Q_p"], dtype=np.float64) * 2.0
    Q_v = np.array(config_dict["cost"]["Q_v"], dtype=np.float64) * 2.0
    R = float(config_dict["cost"]["R"])
    ilqt_cfg = dict(config_dict["ilqt"])

    solver = ILQTSolver(ilqt_cfg, use_analytical=True)

    hit_direction = np.array(config_dict["hitting"]["hit_direction"],
                             dtype=np.float64)
    racket_speed = float(config_dict["hitting"]["racket_speed"])
    target_center = np.array([-0.82765693, -0.47411682, 0.86947444])
    target_offset = 0.10

    stage_cfg = config_dict.get("stage_weights", {})
    far_stage = stage_cfg.get("far", {"Q_qdot_mult": 1.0, "Q_qddot_mult": 1.0, "Q_du_mult": 1.0})
    mid_stage = stage_cfg.get("mid", {"Q_qdot_mult": 2.0, "Q_qddot_mult": 2.0, "Q_du_mult": 2.0})
    near_stage = stage_cfg.get("near", {"Q_qdot_mult": 0.0, "Q_qddot_mult": 0.0, "Q_du_mult": 0.0})

    rng = np.random.default_rng(args.start_seed)

    do_render = not args.no_video
    renderer = None
    writer = None

    if do_render:
        import imageio

        env.model.vis.global_.offwidth = args.width
        env.model.vis.global_.offheight = args.height
        renderer = mujoco.Renderer(env.model, width=args.width, height=args.height)

        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = 3.5
        cam.elevation = -15
        cam.azimuth = 135
        cam.lookat[:] = [0.0, 0.0, 1.0]

        env.model.light_pos[0] = [0.0, 0.0, 8.0]
        env.model.light_dir[0] = [0.0, 0.0, -1.0]
        env.model.light_diffuse[0] = [1.4, 1.45, 1.55]
        env.model.light_ambient[0] = [0.3, 0.3, 0.35]
        env.model.light_specular[0] = [0.5, 0.5, 0.5]
        if env.model.nlight > 1:
            env.model.light_pos[1] = [2.0, -2.0, 3.0]
            env.model.light_dir[1] = [-0.4, 0.3, -0.8]
            env.model.light_diffuse[1] = [1.2, 1.15, 1.05]
            env.model.light_ambient[1] = [0.0, 0.0, 0.0]
            env.model.light_specular[1] = [0.6, 0.6, 0.6]
            env.model.light_active[1] = True
        if env.model.nlight > 2:
            env.model.light_pos[2] = [-1.5, -1.0, 2.5]
            env.model.light_dir[2] = [0.3, 0.2, -0.7]
            env.model.light_diffuse[2] = [0.8, 0.85, 0.95]
            env.model.light_ambient[2] = [0.0, 0.0, 0.0]
            env.model.light_specular[2] = [0.4, 0.4, 0.4]
            env.model.light_active[2] = True

        fps = args.fps if args.fps else int(1.0 / dt)
        video_path = output_dir / "20hits_v9_continuous.mp4"
        writer = imageio.get_writer(
            str(video_path),
            fps=fps,
            output_params=["-crf", "18", "-pix_fmt", "yuv420p"],
        )
    else:
        cam = None

    def render_frame() -> None:
        if renderer is None:
            return
        renderer.update_scene(env.data, camera=cam)
        writer.append_data(renderer.render())

    def pd_return_to_init(x_cur: np.ndarray, n_steps: int) -> np.ndarray:
        x = x_cur.copy()
        for _ in range(n_steps):
            q_err = init_q - x[:NQ]
            qd_err = -x[NQ:]
            u_ret = 200.0 * q_err + 20.0 * qd_err
            u_ret = np.clip(u_ret, ctrl_lo, ctrl_hi)
            x, bp, bv = env.step_full(u_ret)
            if do_render:
                render_frame()
        return x

    x0 = np.zeros(NX)
    x0[:NQ] = init_q
    env.reset(init_q)
    env.data.qpos[NQ:NQ + LEFT_NQ] = init_q_left
    env.data.qvel[NQ:NQ + LEFT_NQ] = 0.0
    env.update_kinematics()

    results = []
    t0 = time.perf_counter()

    for i in range(args.n_runs):
        seed = args.start_seed + i
        t_hit_start = time.perf_counter()

        if args.serve_box:
            p0, v0, p_hit_expected = generate_ball_from_serve_box(
                serve_box_center=(0.0, -serve_dist, 1.2),
                serve_box_halfsize=(4.0, 0.1, 0.15),
                target_center=target_center,
                target_offset=target_offset,
                shoulder_pos=shoulder_pos,
                workspace_radius=workspace_radius,
                g=g,
                ball_speed=args.ball_speed,
                speed_range=(8.0, 18.0),
                use_bounce=True,
                bounce_restitution=0.75,
                rng=rng,
            )
        else:
            hit_time = total_horizon * dt * rng.uniform(0.3, 0.4)
            p0, v0, p_hit_expected = generate_ball_to_target_box(
                target_center, target_offset, hit_time, g,
                shoulder_pos=shoulder_pos, workspace_radius=workspace_radius,
                ball_speed=args.ball_speed, rng=rng, ball_direction="y",
                ball_start_y_range=(-5.5, -4.5), ball_start_z_range=(1.4, 1.8),
            )

        hit_info = find_hitting_point_physics(
            env, p0, v0, shoulder_pos, workspace_radius, total_horizon
        )
        if hit_info is None:
            logger.warning("[%2d/%d] seed=%d ball unreachable, skip",
                           i + 1, args.n_runs, seed)
            results.append({"seed": seed, "hit": False, "pos_error": 999.0,
                            "hit_type": "unreachable"})
            continue

        k_hit_total = hit_info["k_hit"]
        p_hit = hit_info["p_hit"]
        v_ball_hit = hit_info["v_ball_hit"]

        ball_positions_all, ball_velocities_all = env.predict_ball_trajectory(
            p0, v0, total_horizon)

        q_ik_init = env.solve_ik(p_hit, q_init=init_q, max_iter=50, eps=1e-2)
        m_low_deg = (q_ik_init - robot_limits.q_lower) * 180.0 / np.pi
        m_up_deg = (robot_limits.q_upper - q_ik_init) * 180.0 / np.pi
        min_margin_deg = float(np.min(np.minimum(m_low_deg, m_up_deg)))
        if min_margin_deg < 3.0:
            search_range = 30
            best_alt_k = k_hit_total
            best_alt_margin = min_margin_deg
            for dk in range(-search_range, search_range + 1):
                kk = k_hit_total + dk
                if kk < 1 or kk > len(ball_positions_all):
                    continue
                p_alt = ball_positions_all[kk - 1]
                dist_alt = np.linalg.norm(p_alt - shoulder_pos)
                dz_alt = p_alt[2] - shoulder_pos[2]
                if not (dist_alt < workspace_radius and p_alt[2] > 0.3
                        and -0.60 < dz_alt < 0.55):
                    continue
                q_alt = env.solve_ik(p_alt, q_init=init_q, max_iter=30, eps=2e-2)
                m_a = float(np.min(np.minimum(
                    (q_alt - robot_limits.q_lower) * 180.0 / np.pi,
                    (robot_limits.q_upper - q_alt) * 180.0 / np.pi,
                )))
                if m_a > best_alt_margin:
                    best_alt_margin = m_a
                    best_alt_k = kk
            if best_alt_k != k_hit_total:
                p_hit = ball_positions_all[best_alt_k - 1].copy()
                v_ball_hit = ball_velocities_all[best_alt_k - 1].copy()
                k_hit_total = best_alt_k

        n_des_single = -v_ball_hit / (np.linalg.norm(v_ball_hit) + 1e-8)
        d_follow = -v_ball_hit / (np.linalg.norm(v_ball_hit) + 1e-8)
        d_hat = d_follow
        v_hit_at_contact = args.target_speed * d_follow
        v_hit_desired = v_hit_at_contact
        hit_shift = follow_through_length

        near_threshold = max(50, k_hit_total // 3)
        far_threshold = 50

        p_follow = p_hit + hit_shift * d_follow

        p_ee_init = env.get_ee_pos()
        dist_to_ball = np.linalg.norm(p_hit - p_ee_init)
        bs_scale = np.clip((dist_to_ball - 0.8) / (1.5 - 0.8), 0.0, 1.0)
        adaptive_bs = 0.4 + bs_scale * 0.6
        cur_backswing_offset = -adaptive_bs

        hit_window: HitWindow | None = None
        hitting_tube: HittingTube | None = None
        if need_candidates:
            hit_window = search_hit_window(
                env, p0, v0, shoulder_pos, workspace_radius,
                k_hit_total + 30, tube_cfg, ball_direction="y",
                current_step=0, robot_limits=robot_limits, init_q=init_q,
            )
            if hit_window is not None:
                hitting_tube = build_hitting_tube(
                    hit_window, racket_speed, d_follow, tube_cfg)

        r_joint_scale: dict[int, float] = {}
        if use_backswing:
            r_joint_scale[0] = 0.3
        if fix_joint5_angle is not None:
            r_joint_scale[5] = 1000.0

        R_schedule_init = (
            compute_r_schedule(k_hit_total, R, decay_ratio=r_decay_ratio)
            if use_r_decay else None
        )

        base_cost_fn = HittingCost(
            env, p_follow, v_hit_desired, Q_p, Q_v, R,
            Q_p_running=0.0,
            R_joint_scale=r_joint_scale if r_joint_scale else None,
            q_des_traj=None,
            Q_joint=None,
            R_schedule=R_schedule_init,
            Q_n=args.normal_weight,
            n_des=n_des_single,
            Q_qdot=float(config_dict["cost"].get("Q_qdot", 0.0)),
            Q_qddot=float(config_dict["cost"].get("Q_qddot", 0.0)),
            Q_du=float(config_dict["cost"].get("Q_du", 0.0)),
        )

        if hitting_tube is not None:
            if ablation_mode == "full":
                cost_fn = TubeHittingCostWrapper(
                    env, base_cost_fn, hitting_tube, k_hit_total, tube_cfg)
            elif ablation_mode == "tube_only":
                cost_fn = TubeOnlyCost(
                    env, base_cost_fn, hitting_tube, k_hit_total, tube_cfg)
            elif ablation_mode == "softmin_only":
                cost_fn = SoftminOnlyCost(
                    env, base_cost_fn, hitting_tube, k_hit_total, tube_cfg)
            else:
                cost_fn = base_cost_fn
        else:
            cost_fn = base_cost_fn

        if use_backswing:
            U_prev, q_des_traj_init = generate_backswing_warm_start(
                env, x0, p_follow, v_hit_at_contact, k_hit_total,
                backswing_offset=cur_backswing_offset,
                backswing_ratio=backswing_ratio,
                fix_joint5_angle=fix_joint5_angle,
                n_des=n_des_single,
            )
        else:
            U_prev = compute_jacobian_init_control(
                env, x0, p_follow, k_hit_total, gain=60.0,
                fix_joint5_angle=fix_joint5_angle,
            )

        env.reset(init_q)
        env.data.qpos[NQ:NQ + LEFT_NQ] = init_q_left
        env.data.qvel[NQ:NQ + LEFT_NQ] = 0.0
        env.update_kinematics()
        env.set_ball_state(p0, v0)

        ball_pos_init, ball_vel_init = env.get_ball_state()
        first_request = PlanRequest(
            x_current=x0.copy(), ball_pos=ball_pos_init, ball_vel=ball_vel_init,
            step=0, k_hit_current=k_hit_total,
            U_prev=np.zeros((0, NU)), p_hit_current=p_hit.copy(),
            v_hit_desired=v_hit_at_contact, n_des_current=n_des_single.copy(),
            is_first_plan=True,
        )
        replan_state = ReplanState(
            k_hit_new=k_hit_total, p_hit_new=p_hit.copy(),
            v_ball_hit_new=v_ball_hit.copy(), current_n_des=n_des_single.copy(),
            U_prev=np.zeros((0, NU)), is_first_plan=True,
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
            "Q_p_scale_far": 5.0, "Q_v_scale_far": 3.0,
            "Q_p_scale_near": 8.0, "Q_v_scale_near": 120.0,
            "hit_shift": hit_shift, "d_hat": d_hat,
            "v_hit_desired": v_hit_desired, "v_hit_at_contact": v_hit_at_contact,
            "d_follow": d_follow,
            "follow_through_length": follow_through_length,
            "follow_through_steps": follow_through_steps,
            "follow_through_v_terminal": follow_through_v_terminal,
            "use_backswing": use_backswing,
            "use_r_decay": use_r_decay,
            "r_decay_ratio": r_decay_ratio,
            "time_perturb_s": 0.0, "space_perturb_m": 0.0,
            "perturb_alpha_min": 0.0,
            "normal_flip": False,
            "fix_joint5_angle": fix_joint5_angle,
            "backswing_offset": cur_backswing_offset,
            "backswing_ratio": backswing_ratio,
            "k_hit_total": k_hit_total,
            "smooth_far": far_stage, "smooth_mid": mid_stage,
            "smooth_near": near_stage,
            "Q_tcp_soft": 5000.0, "Q_qdot_limit": 1000.0,
            "softmin_beta": args.softmin_beta,
            "ball_positions_all": ball_positions_all,
            "max_tcp_speed": args.max_tcp,
            "no_v_maximize": False,
            "normal_weight": args.normal_weight,
        }

        env_plan = AsyncReplanner(env, do_replan, replan_cfg,
                                  state=replan_state, model_path=model_path)
        _ = env_plan._ensure_env_plan()

        first_result = do_replan(first_request, env_plan.env_plan,
                                 replan_state, replan_cfg)
        replan_state.is_first_plan = False
        replan_state.k_hit_new = first_result.k_hit_new
        replan_state.p_hit_new = first_result.p_hit_new.copy()
        replan_state.v_ball_hit_new = first_result.v_ball_hit_new.copy()
        replan_state.current_n_des = first_result.n_des_new.copy()
        replan_state.U_prev = first_result.U_prev.copy()

        U_buffer = first_result.U_buffer.copy()
        buffer_idx = 0
        U_prev = first_result.U_prev.copy()

        k_hit_new = first_result.k_hit_new
        p_hit_new = first_result.p_hit_new.copy()
        v_ball_hit_new = first_result.v_ball_hit_new.copy()
        current_n_des = first_result.n_des_new.copy()
        is_first_plan = False

        x_current = x0.copy()
        hit_step = -1
        ball_was_hit = False
        p_ee_at_hit = None
        v_ee_at_hit = None
        ball_pos_at_hit = None
        follow_through_start = -1
        active_contact = False
        passive_contact = False
        buffer_exhaustion_count = 0
        exec_metrics = ExecutionMetrics()

        mpc_horizon = total_horizon
        effective_total = total_horizon + follow_through_steps

        for step in range(effective_total):
            ball_pos, ball_vel = env.get_ball_state()
            p_ee_cur = env.get_ee_pos()
            dist_cur = np.linalg.norm(p_ee_cur - ball_pos)

            remaining_horizon = max(5, min(effective_total - step,
                                           mpc_horizon - step))

            need_replan = ((step % replan_interval == 0) or (step == 0)
                           or (buffer_idx >= len(U_buffer)))

            if need_replan:
                hit_info_new = find_hitting_point_physics(
                    env, ball_pos, ball_vel, shoulder_pos,
                    workspace_radius, remaining_horizon
                )

                if hit_info_new is None:
                    if ball_was_hit and follow_through_start < 0:
                        follow_through_start = step
                    elif not ball_was_hit:
                        break
                    hit_info_new = None

                if hit_info_new is not None:
                    k_hit_candidate = hit_info_new["k_hit"]
                    if (k_hit_candidate < max(10, k_hit_new // 4)
                            and k_hit_new > 30):
                        k_hit_candidate = max(1, k_hit_new - replan_interval)

                    p_hit_new = hit_info_new["p_hit"]
                    v_ball_hit_new = hit_info_new["v_ball_hit"]
                    k_hit_new = k_hit_candidate

                    n_des_new = -v_ball_hit_new / (
                        np.linalg.norm(v_ball_hit_new) + 1e-8)
                    current_n_des = n_des_new
                    p_follow_new = p_hit_new + hit_shift * d_hat

                    horizon_full = k_hit_new
                    horizon_plan = min(horizon_full, fixed_horizon)

                    p_terminal_v5 = p_hit_new + follow_through_length * d_hat
                    v_terminal_v5 = v_hit_at_contact
                    base_cost_fn.update_target(p_terminal_v5, v_terminal_v5,
                                               n_des=n_des_new)
                    base_cost_fn.set_midpoint_target(None, None)

                    if need_candidates:
                        hw = search_hit_window(
                            env, ball_pos, ball_vel, shoulder_pos,
                            workspace_radius, remaining_horizon, tube_cfg,
                            ball_direction="y", current_step=0,
                            robot_limits=robot_limits,
                            init_q=x_current[:NQ].copy(),
                        )
                        if hw is not None:
                            ht = build_hitting_tube(hw, racket_speed,
                                                    d_follow, tube_cfg)
                            if ablation_mode == "full":
                                cost_fn = TubeHittingCostWrapper(
                                    env, base_cost_fn, ht, k_hit_new, tube_cfg)
                            elif ablation_mode == "tube_only":
                                cost_fn = TubeOnlyCost(
                                    env, base_cost_fn, ht, k_hit_new, tube_cfg)
                            elif ablation_mode == "softmin_only":
                                cost_fn = SoftminOnlyCost(
                                    env, base_cost_fn, ht, k_hit_new, tube_cfg)
                            else:
                                cost_fn = base_cost_fn
                        else:
                            cost_fn = base_cost_fn

                    if k_hit_new > far_threshold and not is_first_plan:
                        ball_save_far, ball_vel_save_far = env.get_ball_state()
                        u_jt = compute_jacobian_init_control(
                            env, x_current, p_follow_new, replan_interval,
                            gain=60.0,
                            fix_joint5_angle=fix_joint5_angle,
                        )
                        env.set_ball_state(ball_save_far, ball_vel_save_far)
                        env.set_arm_state(x_current)
                        U_buffer = u_jt
                        buffer_idx = 0
                        U_prev = U_prev if len(U_prev) > 0 else np.zeros(
                            (0, NU))
                        continue

                    env.set_arm_state(x_current)
                    env.update_kinematics()
                    pos_err_now = np.linalg.norm(
                        env.get_ee_pos() - p_hit_new)

                    if pos_err_now > 0.10:
                        Q_p_scale = 5.0
                        Q_v_scale = 3.0
                    else:
                        ratio_s = pos_err_now / 0.10
                        Q_p_scale = 8.0 + (5.0 - 8.0) * ratio_s
                        Q_v_scale = 120.0 + (3.0 - 120.0) * ratio_s

                    cost_fn.update_weights(Q_p_scale, Q_v_scale)

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
                            horizon_full, R, decay_ratio=r_decay_ratio
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
                        fast_lin = True
                        fp_limits = None
                        if k_hit_new > 30:
                            iters_plan = min(near_plan_iters,
                                             max(max_iter_per_plan, 10))
                        else:
                            iters_plan = near_plan_iters
                    else:
                        fast_lin = True

                    if use_backswing and horizon_full > 0:
                        q_hit_ik = env.solve_ik(
                            p_hit_new, q_init=x_current[:NQ],
                            max_iter=150, eps=1e-3)
                        env.set_arm_state(
                            np.concatenate([q_hit_ik, np.zeros(NQ)]))
                        J_p_new = env.get_ee_jacp()
                        qdot_hit_new = np.linalg.lstsq(
                            J_p_new, v_hit_at_contact, rcond=None)[0]
                        qd_n = np.linalg.norm(qdot_hit_new)
                        if qd_n > 3.0:
                            qdot_hit_new *= 3.0 / qd_n
                        bs_scale_plan = horizon_full / max(k_hit_total, 1)
                        q_des_full = np.zeros((horizon_full, NQ))
                        q_des_full[:, 0] = compute_joint1_backswing_trajectory(
                            x_current[0], x_current[NQ], q_hit_ik[0],
                            qdot_hit_new[0], horizon_full,
                            backswing_offset=cur_backswing_offset * bs_scale_plan,
                            backswing_ratio=backswing_ratio,
                        )
                        for j in range(1, NQ):
                            q_des_full[:, j] = np.linspace(
                                x_current[j], q_hit_ik[j], horizon_full)
                        cost_fn.set_q_des_traj(q_des_full[:horizon_plan],
                                                Q_joint=None)

                    if len(U_prev) >= horizon_full // 3:
                        U_warm = resample_control_sequence(
                            U_prev, horizon_full)[:horizon_plan]
                    elif use_backswing and horizon_full > 0:
                        U_warm_full, _ = generate_backswing_warm_start(
                            env, x_current, p_follow_new, v_hit_desired,
                            horizon_full, cur_backswing_offset,
                            backswing_ratio, fix_joint5_angle, n_des_new,
                        )
                        U_warm = U_warm_full[:horizon_plan]
                    else:
                        U_warm = compute_jacobian_init_control(
                            env, x_current, p_follow_new, horizon_full,
                            gain=30.0,
                            fix_joint5_angle=fix_joint5_angle,
                        )[:horizon_plan]

                    ball_pos_save, ball_vel_save = env.get_ball_state()
                    X_mpc, U_mpc, iter_costs, solver_ok = solver.solve_few_iters(
                        env, cost_fn, x_current, U_warm,
                        max_iter=iters_plan,
                        skip_linesearch=skip_ls,
                        limits=fp_limits,
                        use_fast_lin=fast_lin,
                    )
                    env.set_ball_state(ball_pos_save, ball_vel_save)
                    env.set_arm_state(x_current)

                    if len(U_mpc) > replan_interval:
                        U_prev = U_mpc[replan_interval:]
                    elif len(U_mpc) > 0:
                        U_prev = U_mpc[1:]
                    else:
                        U_prev = np.zeros((0, NU))

                    U_buffer = U_mpc[:replan_interval]
                    buffer_idx = 0

                    if k_hit_new <= 30 and len(U_mpc) >= replan_interval * 2:
                        U_buffer = U_mpc[:replan_interval * 2]
                        buffer_idx = 0

            if buffer_idx < len(U_buffer):
                u_cmd = U_buffer[buffer_idx]
                buffer_idx += 1
            else:
                buffer_exhaustion_count += 1
                u_cmd = np.zeros(NU)
                if k_hit_new > 0:
                    env.set_arm_state(x_current)
                    p_ee = env.get_ee_pos()
                    J_p = env.get_ee_jacp()
                    err = p_hit_new - p_ee
                    tau_backup = J_p.T @ err * 30.0
                    tau_backup -= 2.0 * x_current[NQ:]
                    u_cmd = np.clip(tau_backup, ctrl_lo, ctrl_hi)

            if fix_joint5_angle is not None:
                u_cmd = fix_joint5_control(u_cmd, fix_joint5_angle,
                                           x_current, NQ)

            enable_collision = False
            if not ball_was_hit:
                if k_hit_new <= 30 and dist_cur < 0.35:
                    enable_collision = True
                elif k_hit_new <= 10:
                    enable_collision = True
            if hasattr(env, "set_arm_collision"):
                env.set_arm_collision(enable_collision)

            ball_save_x = env.get_ball_state()
            x_save_x = x_current.copy()
            for beta_x in [1.0, 0.6, 0.3, 0.0]:
                u_try = beta_x * u_cmd
                u_try = np.clip(u_try, ctrl_lo, ctrl_hi)
                x_pred = env.step_from_state(x_current, u_try)
                env.update_kinematics()
                ok_x = all(env.data.xpos[bid, 0] <= -0.1
                           for bid in _hard_x_body_ids)
                env.set_ball_state(*ball_save_x)
                env.set_arm_state(x_save_x)
                if ok_x:
                    u_cmd = u_try
                    break

            ball_save_sf = env.get_ball_state()
            arm_save_sf = x_current.copy()
            ball_ref = ball_save_sf

            def _safety_step(x_s, u_s):
                x_n = env.step_from_state(x_s, u_s)
                env.set_ball_state(*ball_ref)
                return x_n

            ok_f, reason_f = check_one_step_feasibility(
                x_current, u_cmd, robot_limits, dt,
                step_predictor=_safety_step,
                k_hit_remaining=k_hit_new, env=env,
            )
            if not ok_f:
                for beta_s in [0.8, 0.6, 0.4, 0.2, 0.0]:
                    u_s = beta_s * u_cmd
                    u_s = np.clip(u_s, ctrl_lo, ctrl_hi)
                    env.set_arm_state(arm_save_sf)
                    ok_ss, _ = check_one_step_feasibility(
                        x_current, u_s, robot_limits, dt,
                        step_predictor=_safety_step,
                        k_hit_remaining=k_hit_new, env=env,
                    )
                    if ok_ss:
                        u_cmd = u_s
                        break
                else:
                    u_cmd = -20.0 * x_current[NQ:]

            env.set_ball_state(*ball_save_sf)
            env.set_arm_state(arm_save_sf)

            ball_vel_before_step = ball_vel.copy() if enable_collision else ball_vel
            x_current, ball_pos, ball_vel = env.step_full(u_cmd)

            ball_racket_hit = False
            if enable_collision and not ball_was_hit:
                for ci in range(env.data.ncon):
                    c = env.data.contact[ci]
                    g1 = env.model.geom(c.geom1).name
                    g2 = env.model.geom(c.geom2).name
                    if (("ball" in g1 or "ball" in g2)
                            and ("racket" in g1 or "racket" in g2)):
                        ball_racket_hit = True
                        ee_spd = np.linalg.norm(env.get_ee_vel())
                        v_ee_at_hit = ee_spd
                        if ee_spd > 0.3:
                            active_contact = True
                        else:
                            passive_contact = True
                        n_racket = env.get_ee_normal()
                        n_hat = n_racket / (np.linalg.norm(n_racket) + 1e-8)
                        v_ee = env.get_ee_vel()
                        v_rel_n = np.dot(ball_vel_before_step - v_ee, n_hat)
                        env.set_ball_vel(
                            ball_vel_before_step - (1 + 0.8) * v_rel_n * n_hat)

            if hasattr(env, "set_arm_collision"):
                env.set_arm_collision(True)

            if do_render:
                render_frame()

            if ball_racket_hit and not ball_was_hit:
                ball_was_hit = True
                hit_step = step
                p_ee_at_hit = env.get_ee_pos().copy()
                ball_pos_at_hit = ball_pos.copy()

            if (ball_was_hit and follow_through_start < 0
                    and hit_step >= 0 and (step - hit_step) >= 3):
                follow_through_start = step

            if (args.follow_trigger == "planned"
                    and k_hit_new <= 1 and follow_through_start < 0):
                hit_step = step if hit_step < 0 else hit_step
                if p_ee_at_hit is None:
                    p_ee_at_hit = env.get_ee_pos().copy()
                if ball_pos_at_hit is None:
                    ball_pos_at_hit = ball_pos.copy()
                follow_through_start = step

            if follow_through_start >= 0 and step > follow_through_start:
                dt_follow = step - follow_through_start
                if dt_follow <= follow_through_steps:
                    v_max_follow = np.linalg.norm(v_hit_at_contact)
                    T_follow = follow_through_steps * dt
                    a_follow = v_max_follow / T_follow if T_follow > 0 else 0.0
                    t_elapsed = dt_follow * dt
                    p_des_follow = p_ee_at_hit + d_follow * (
                        v_max_follow * t_elapsed
                        - 0.5 * a_follow * t_elapsed ** 2)
                    v_des_follow = d_follow * max(
                        v_max_follow - a_follow * t_elapsed, 0.0)

                    env.update_kinematics()
                    J_p_f = env.get_ee_jacp()[:, :NQ]
                    dp_f = p_des_follow - env.get_ee_pos()
                    F_follow = 200.0 * dp_f - 20.0 * J_p_f @ x_current[NQ:]
                    u_follow = J_p_f.T @ F_follow
                    u_follow = np.clip(u_follow, ctrl_lo, ctrl_hi)
                    x_current, ball_pos, ball_vel = env.step_full(u_follow)
                    if do_render:
                        render_frame()
                    continue
                else:
                    break

        for _ in range(20):
            q_hold = x_current[:NQ].copy()
            u_hold = 100.0 * (init_q - q_hold) - 10.0 * x_current[NQ:]
            u_hold = np.clip(u_hold, ctrl_lo, ctrl_hi)
            x_current, bp_ret, _ = env.step_full(u_hold)
            if do_render:
                render_frame()

        x_current = pd_return_to_init(x_current, args.return_steps)
        for _ in range(args.pause_steps):
            if do_render:
                render_frame()

        if p_ee_at_hit is not None and ball_pos_at_hit is not None:
            pos_error = float(np.linalg.norm(p_ee_at_hit - ball_pos_at_hit))
        else:
            env.set_arm_state(x_current)
            pos_error = float(np.linalg.norm(env.get_ee_pos() - p_hit))

        hit_type_en = ("active" if active_contact
                       else ("passive" if passive_contact else "miss"))
        is_hit = pos_error < 0.153

        result = {
            "seed": seed,
            "pos_error": pos_error,
            "hit_type": hit_type_en,
            "hit": is_hit,
            "v_racket_at_hit": v_ee_at_hit if v_ee_at_hit is not None else 0.0,
        }
        results.append(result)

        hit_str = "HIT" if is_hit else "MISS"
        t_elapsed = time.perf_counter() - t0
        logger.info(
            "[%2d/%d] seed=%2d %s  err=%.1fmm  type=%s  v_rack=%.2fm/s  (%.0fs)",
            i + 1, args.n_runs, seed, hit_str,
            pos_error * 1000, hit_type_en,
            result["v_racket_at_hit"],
            t_elapsed,
        )

    csv_path = output_dir / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["seed", "hit", "pos_error", "hit_type",
                         "v_racket_at_hit"],
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(results)

    hits = sum(1 for r in results if r.get("hit"))
    logger.info("=" * 60)
    logger.info("Result: %d/%d hits (%.0f%%)",
                hits, args.n_runs, hits / max(args.n_runs, 1) * 100)
    logger.info("CSV: %s", csv_path)

    if do_render and writer is not None:
        writer.close()
        logger.info("Video saved: %s", video_path)
    if renderer is not None:
        renderer.close()

    logger.info("Done! Total: %.0fs", time.perf_counter() - t0)


if __name__ == "__main__":
    run_batch()

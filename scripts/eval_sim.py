"""加载 MPC 结果做仿真评估。

用法：
  python scripts/eval_sim.py                 # 运行 MPC 并在 MuJoCo 中评估
  python scripts/eval_sim.py --viewer        # 同时打开实时查看器
  python scripts/eval_sim.py --seed 42       # 指定随机种子
"""

import sys
import argparse
import logging
import numpy as np
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sim.env import MujocoEnv
from src.tennis.ball import generate_serve_ball
from src.tennis.hitting import find_hitting_point
from src.sim.viewer import RealTimeViewer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """加载 YAML 配置文件。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    """评估主函数。"""
    parser = argparse.ArgumentParser(description="MPC 仿真评估")
    parser.add_argument("--viewer", action="store_true", help="打开 MuJoCo 查看器")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    # 加载配置
    base_path = Path(__file__).resolve().parent.parent / "configs"
    config = load_config(base_path / "default.yaml")
    mpc_config_path = base_path / "mpc.yaml"
    if mpc_config_path.exists():
        mpc_cfg = load_config(mpc_config_path)
        config = merge_configs(config, mpc_cfg)

    dt = float(config["sim"]["dt"])
    g = np.array(config["ball"]["gravity"], dtype=np.float64)
    shoulder_pos = np.array(config["hitting"]["shoulder_pos"], dtype=np.float64)
    workspace_radius = config["hitting"]["workspace_radius"]
    bounce_restitution = float(config["ball"].get("bounce_restitution", 0.75))

    mpc_cfg = config.get("mpc", {})
    total_horizon = int(mpc_cfg.get("total_horizon", 200))
    use_bounce = mpc_cfg.get("use_bounce", True)

    # 初始化环境
    model_path = Path(__file__).resolve().parent.parent / "src" / "robot" / "model.xml"
    env = MujocoEnv(model_path, dt=dt)

    # 初始臂状态
    init_q = np.array(config["init_q"], dtype=np.float64)
    x0 = np.zeros(env.NX)
    x0[:env.NQ] = init_q

    # 生成球轨迹
    rng = np.random.default_rng(args.seed)
    hit_cfg = config.get("hitting", {})
    hit_offset_ranges = {
        "x": hit_cfg.get("hit_offset_x_range", [0.10, 0.50]),
        "y": hit_cfg.get("hit_offset_y_range", [-0.35, 0.35]),
        "z": hit_cfg.get("hit_offset_z_range", [-0.20, 0.55]),
    }

    hit_time = total_horizon * dt * 0.8
    p0, v0, p_hit_expected = generate_serve_ball(
        shoulder_pos, workspace_radius, g, hit_time,
        serve_distance=config["ball"].get("serve_distance", 22.0),
        serve_height_range=tuple(config["ball"].get("serve_height", [2.5, 3.0])),
        bounce_restitution=bounce_restitution,
        hit_offset_ranges=hit_offset_ranges,
        rng=rng,
    )

    hit_info = find_hitting_point(
        p0, v0, g, shoulder_pos, workspace_radius, dt, total_horizon,
        use_bounce=use_bounce, bounce_restitution=bounce_restitution,
    )

    if hit_info is None:
        logger.info("球不可达，退出评估")
        return

    p_hit = hit_info["p_hit"]
    k_hit = hit_info["k_hit"]

    logger.info(f"评估: 击打步数={k_hit}, 击打位置={p_hit}")

    # 设置初始状态
    env.reset(init_q)
    env.set_ball_state(p0, v0)

    # 打开查看器
    viewer = None
    if args.viewer:
        viewer = RealTimeViewer(env, config)
        viewer.start()

    # 在 MuJoCo 中仿真球飞行
    for step in range(total_horizon):
        # 不施加任何臂控制（零力矩），仅观察球运动
        u_zero = np.zeros(env.NU)
        env.step(u_zero)

        if viewer is not None and viewer.is_running():
            viewer.sync()

        # 检查球是否到达击打点附近
        ball_pos, _ = env.get_ball_state()
        if np.linalg.norm(ball_pos - p_hit) < 0.1:
            logger.info(f"步 {step}: 球到达击打点附近")
            break

    logger.info("评估完成")


def merge_configs(base: dict, override: dict) -> dict:
    """递归合并两个配置字典。"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


if __name__ == "__main__":
    main()

"""RM-65 机器人部位可视化脚本。

在 MuJoCo 查看器中用彩色标记球标注每个 body 的位置和名称，
控制台同步打印颜色 → 名称对照图例。

用法:
  # 默认初始姿势
  python scripts/visualize_robot_parts.py

  # 自定义关节角度 (6 个右臂关节，弧度)
  python scripts/visualize_robot_parts.py --q0 -1.5 1.57 -0.236 0.404 0.446 2.45

  # 仅打印控制台图例（不启动查看器）
  python scripts/visualize_robot_parts.py --console-only
"""

from __future__ import annotations

import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.mujoco_loader import load_mujoco_model

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ==============================================================================
# Body 定义：名称、RGB 颜色、描述、分组
# ==============================================================================
BODY_INFO: list[tuple[str, str, tuple[float, float, float, float], str, str]] = [
    # ── 底盘 / 躯干 (蓝色系) ──
    ("base_link_underpan",  "#3355CC", (0.20, 0.33, 0.80, 0.75), "底盘底座",        "底盘/躯干"),
    ("body_base_link",      "#4466DD", (0.27, 0.40, 0.87, 0.75), "躯干柱",          "底盘/躯干"),
    ("platform_base_link",  "#5577EE", (0.33, 0.47, 0.93, 0.75), "肩部平台",        "底盘/躯干"),
    ("head_link1",          "#6688FF", (0.40, 0.53, 1.00, 0.75), "头部1段",         "底盘/躯干"),
    ("head_link2",          "#7799FF", (0.47, 0.60, 1.00, 0.75), "头部2段",         "底盘/躯干"),
    ("camera_link",         "#88AAFF", (0.53, 0.67, 1.00, 0.75), "摄像头",          "底盘/躯干"),

    # ── 右臂 (红 → 橙渐变) ──
    ("r_base_link1",  "#CC1111", (0.80, 0.07, 0.07, 0.75), "右肩底座 (J0 固定座)",   "右臂"),
    ("r_link1",       "#DD3322", (0.87, 0.20, 0.13, 0.75), "J1 shoulder_pan 肩偏航",  "右臂"),
    ("r_link2",       "#EE5533", (0.93, 0.33, 0.20, 0.75), "J2 shoulder_lift 肩俯仰", "右臂"),
    ("r_link3",       "#FF7744", (1.00, 0.47, 0.27, 0.75), "J3 elbow 肘 [规避检查点]", "右臂"),
    ("r_link4",       "#FF9944", (1.00, 0.60, 0.27, 0.75), "J4 wrist_1 腕1",          "右臂"),
    ("r_link5",       "#FFBB55", (1.00, 0.73, 0.33, 0.75), "J5 wrist_2 腕2 [规避检查点]", "右臂"),
    ("r_link6",       "#FFDD66", (1.00, 0.87, 0.40, 0.75), "J6 wrist_3 腕3",          "右臂"),
    ("r_flange",      "#EE44EE", (0.93, 0.27, 0.93, 0.75), "法兰",                    "右臂"),
    ("r_racket_body", "#AA22CC", (0.67, 0.13, 0.80, 0.75), "球拍体 (拍面中心→ racket_center)", "右臂"),

    # ── 左臂 (绿色系) ──
    ("l_base_link1", "#22AA22", (0.13, 0.67, 0.13, 0.75), "左肩底座 (J0 固定座)",   "左臂"),
    ("l_link1",      "#33BB33", (0.20, 0.73, 0.20, 0.75), "左肩关节",                "左臂"),
    ("l_link2",      "#44CC44", (0.27, 0.80, 0.27, 0.75), "左肩俯仰",                "左臂"),
    ("l_link3",      "#55DD55", (0.33, 0.87, 0.33, 0.75), "左肘",                    "左臂"),
    ("l_link4",      "#66DD55", (0.40, 0.87, 0.33, 0.75), "左腕1",                   "左臂"),
    ("l_link5",      "#77EE66", (0.47, 0.93, 0.40, 0.75), "左腕2",                   "左臂"),
    ("l_link6",      "#88EE77", (0.53, 0.93, 0.47, 0.75), "左腕3",                   "左臂"),

    # ── 其他 ──
    ("ball", "#DDDD00", (0.87, 0.87, 0.00, 0.75), "网球 (freejoint, 6DOF)", "其他"),
]
"""每个 body 的元信息：(MuJoCo名称, 十六进制颜色, (R,G,B,A), 中文描述, 分组)。"""

INIT_Q_DEFAULT = np.array([-1.5, 1.57, -0.236, 0.404, 0.446, 2.45], dtype=np.float64)


def print_legend() -> None:
    """打印彩色图例到控制台。"""
    gray = "\033[90m"
    reset = "\033[0m"
    body_color = lambda r, g, b: f"\033[38;2;{int(r*255)};{int(g*255)};{int(b*255)}m"

    print()
    print("=" * 80)
    print("  RM-65 机器人部位标注图例")
    print("=" * 80)

    current_group = ""
    for name, hex_str, rgba, desc, group in BODY_INFO:
        if group != current_group:
            current_group = group
            print(f"\n── {group} ──")
        r, g, b = rgba[0], rgba[1], rgba[2]
        marker = body_color(r, g, b) + "●" + reset
        print(f"  {marker} {hex_str}  {name:<22s} {desc}")

    print()
    print(f"{gray}提示: 在 MuJoCo 查看器中旋转/缩放观察各部位位置。关闭窗口退出。{reset}")
    print("=" * 80)
    print()


def get_body_positions(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, np.ndarray]:
    """获取所有定义 body 的世界坐标位置。

    Args:
        model: MuJoCo 模型。
        data: MuJoCo 数据。

    Returns:
        {body_name: world_position (3,)} 字典。
    """
    positions: dict[str, np.ndarray] = {}
    for name, _, _, _, _ in BODY_INFO:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        positions[name] = data.xpos[body_id].copy()
    return positions


def add_body_markers(
    scene: mujoco.MjvScene,
    positions: dict[str, np.ndarray],
    marker_radius: float = 0.018,
) -> None:
    """在 scene 中为每个 body 添加彩色标记球。

    Args:
        scene: MuJoCo 可视化场景 (viewer.user_scn)。
        positions: body 世界坐标字典。
        marker_radius: 标记球半径 (m)。
    """
    for name, _, rgba, _, _ in BODY_INFO:
        if name not in positions:
            continue
        pos = positions[name]

        idx = scene.ngeom
        if idx >= scene.maxgeom:
            logger.warning("scene geom 已满，部分标记未显示")
            break

        mujoco.mjv_initGeom(
            scene.geoms[idx],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([marker_radius, 0.0, 0.0]),
            pos=pos,
            mat=np.eye(3).flatten(),
            rgba=np.array(rgba, dtype=np.float64),
        )
        scene.ngeom += 1


def main() -> None:
    """RM-65 机器人部位可视化主函数。"""
    parser = argparse.ArgumentParser(description="RM-65 机器人部位可视化")
    parser.add_argument(
        "--q0", type=float, nargs=6, default=INIT_Q_DEFAULT.tolist(),
        help="右臂初始关节角度 (6 个值，弧度)。默认: -1.5 1.57 -0.236 0.404 0.446 2.45",
    )
    parser.add_argument(
        "--console-only", action="store_true",
        help="仅打印控制台图例，不启动 MuJoCo 查看器",
    )
    args = parser.parse_args()

    init_q_right = np.array(args.q0, dtype=np.float64)
    init_q_left = np.array([-0.373, -1.57, 0.236, -0.404, -0.446, -2.45], dtype=np.float64)

    # 打印图例
    print_legend()

    if args.console_only:
        return

    # 加载模型
    model_path = Path(__file__).resolve().parent.parent / "src" / "robot" / "rm65_model.xml"
    model = load_mujoco_model(model_path)
    data = mujoco.MjData(model)

    # 设置初始关节角度（右臂 + 左臂）
    data.qpos[:6] = init_q_right
    data.qpos[6:12] = init_q_left
    mujoco.mj_forward(model, data)

    # 启动查看器
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 调整摄像机
        viewer.cam.distance = 3.5
        viewer.cam.elevation = -15
        viewer.cam.azimuth = 135
        viewer.cam.lookat[:] = [0.0, -0.3, 1.2]

        logger.info("查看器已启动。旋转/缩放观察各部位标记球，关闭窗口退出。")

        while viewer.is_running():
            # 每帧清除并重建标记（确保关节变化时标记跟随）
            viewer.user_scn.ngeom = 0
            positions = get_body_positions(model, data)
            add_body_markers(viewer.user_scn, positions)

            viewer.sync()


if __name__ == "__main__":
    main()

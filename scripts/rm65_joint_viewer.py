"""RM-65 双臂机器人关节调节查看器（位置控制版）。

与 rm65_mpc_ilqt.py 使用完全一致的模型和初始位姿，
将 motor 执行器替换为 position 执行器，支持 MuJoCo 控制面板直接拖动关节角。

操作说明：
  - MuJoCo 右侧 Control 面板直接拖动滑条控制关节角度（弧度）
  - 拖动后关节会物理驱动到目标角度
  - 按键 R 重置到初始位姿
  - 按键 P 打印当前关节角度（Python 数组格式）
  - 按键 F 切换坐标系显示（World / Racket Center / 关闭）
  - 红=X轴，绿=Y轴，蓝=Z轴

用法：
  python scripts/rm65_joint_viewer.py
"""

import sys
import re
import time
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import mujoco
import mujoco.viewer
from src.utils.mujoco_loader import load_mujoco_model


INIT_Q_RIGHT = np.array([0.373, 1.57, -0.236, 0.404, 0.446, 2.45], dtype=np.float64)
INIT_Q_LEFT = np.array([0.0, -2.0, 0.7, 0.7, 1.0, 0.0], dtype=np.float64)

POSITION_ACTUATORS = """  <actuator>
    <!-- 右臂：位置执行器，ctrl 直接控制目标关节角（弧度） -->
    <position name="pos_r_joint1" joint="r_joint1" ctrlrange="-3.11 3.11" kp="200" damping="20"/>
    <position name="pos_r_joint2" joint="r_joint2" ctrlrange="-2.27 2.27" kp="200" damping="20"/>
    <position name="pos_r_joint3" joint="r_joint3" ctrlrange="-2.36 2.36" kp="100" damping="10"/>
    <position name="pos_r_joint4" joint="r_joint4" ctrlrange="-3.11 3.11" kp="50" damping="5"/>
    <position name="pos_r_joint5" joint="r_joint5" ctrlrange="-2.23 2.23" kp="50" damping="5"/>
    <position name="pos_r_joint6" joint="r_joint6" ctrlrange="-6.28 6.28" kp="20" damping="2"/>
    <!-- 左臂 -->
    <position name="pos_l_joint1" joint="l_joint1" ctrlrange="-3.11 3.11" kp="200" damping="20"/>
    <position name="pos_l_joint2" joint="l_joint2" ctrlrange="-2.27 2.27" kp="200" damping="20"/>
    <position name="pos_l_joint3" joint="l_joint3" ctrlrange="-2.36 2.36" kp="100" damping="10"/>
    <position name="pos_l_joint4" joint="l_joint4" ctrlrange="-3.11 3.11" kp="50" damping="5"/>
    <position name="pos_l_joint5" joint="l_joint5" ctrlrange="-2.23 2.23" kp="50" damping="5"/>
    <position name="pos_l_joint6" joint="l_joint6" ctrlrange="-6.28 6.28" kp="20" damping="2"/>
  </actuator>"""


def build_position_actuator_model() -> tuple[mujoco.MjModel, Path]:
    """读取原始模型 XML，将 motor 执行器替换为 position 执行器，返回新模型。"""
    model_path = Path(__file__).resolve().parent.parent / "src" / "robot" / "rm65_model.xml"
    xml_str = model_path.read_text(encoding="utf-8")

    new_xml = re.sub(
        r"<actuator>.*?</actuator>",
        POSITION_ACTUATORS,
        xml_str,
        flags=re.DOTALL,
    )

    tmp_path = model_path.parent / "_rm65_position_actuator_tmp.xml"
    tmp_path.write_text(new_xml, encoding="utf-8")

    model = load_mujoco_model(tmp_path)
    return model, tmp_path


def main() -> None:
    model, tmp_path = build_position_actuator_model()
    data = mujoco.MjData(model)

    data.qpos[:6] = INIT_Q_RIGHT
    data.qpos[6:12] = INIT_Q_LEFT
    data.ctrl[:6] = INIT_Q_RIGHT
    data.ctrl[6:12] = INIT_Q_LEFT
    data.qpos[12:15] = [5.0, 0.0, 2.0]
    data.qpos[15:19] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)

    racket_site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "racket_center"
    )

    # 坐标系显示状态: 0=关闭, 1=World, 2=RacketCenter
    frame_mode = [1]

    print("=" * 60)
    print("  RM-65 关节调节查看器（位置控制版）")
    print("=" * 60)
    print(f"  右臂初始位姿: {np.round(INIT_Q_RIGHT, 4).tolist()}")
    print(f"  左臂初始位姿: {np.round(INIT_Q_LEFT, 4).tolist()}")
    print("-" * 60)
    print("  操作:")
    print("    右侧 Control 面板拖动滑条 → 直接控制关节角（弧度）")
    print("    R  重置到初始位姿")
    print("    P  打印当前关节角度")
    print("    F  切换坐标系 (World → Racket → 关闭)")
    print("    红=X轴, 绿=Y轴, 蓝=Z轴")
    print("=" * 60)

    def key_callback(keycode: int) -> None:
        if keycode == ord("R"):
            data.qpos[:6] = INIT_Q_RIGHT
            data.qpos[6:12] = INIT_Q_LEFT
            data.qvel[:6] = 0.0
            data.qvel[6:12] = 0.0
            data.ctrl[:6] = INIT_Q_RIGHT
            data.ctrl[6:12] = INIT_Q_LEFT
            mujoco.mj_forward(model, data)
            print("  [重置] 所有关节已回到初始位姿")
        elif keycode == ord("P"):
            q_r = data.qpos[:6].copy()
            q_l = data.qpos[6:12].copy()
            p_ee = data.site_xpos[racket_site_id].copy()
            print()
            print(f"  右臂 init_q = np.array("
                  f"{np.round(q_r, 4).tolist()}, dtype=np.float64)")
            print(f"  左臂 init_q_left = np.array("
                  f"{np.round(q_l, 4).tolist()}, dtype=np.float64)")
            print(f"  球拍中心位置: {np.round(p_ee, 4).tolist()}")
            print()
        elif keycode == ord("F"):
            frame_mode[0] = (frame_mode[0] + 1) % 3
            labels = ["关闭", "世界坐标系 (World)", "球拍中心坐标系 (Racket)"]
            print(f"  [坐标系] {labels[frame_mode[0]]}")

    try:
        with mujoco.viewer.launch_passive(
            model, data, key_callback=key_callback
        ) as viewer:
            viewer.cam.distance = 3.5
            viewer.cam.elevation = -15
            viewer.cam.azimuth = 135
            viewer.cam.lookat[:] = [0.0, 0.0, 1.0]

            # 获取 racket_center site 的 muJoCo ID
            racket_site_mj_id = racket_site_id

            while viewer.is_running():
                # 坐标系显示
                if frame_mode[0] == 0:
                    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
                elif frame_mode[0] == 1:
                    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
                elif frame_mode[0] == 2:
                    # 只显示 racket_center 的 local frame
                    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(1.0 / 240.0)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


if __name__ == "__main__":
    main()

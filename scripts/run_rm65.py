"""在 MuJoCo 查看器中运行 RM-65 网球机器人仿真。

使用方法:
    python scripts/run_rm65.py [--model MODEL] [--viewer]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.mujoco_loader import load_mujoco_model
import mujoco
import mujoco.viewer
import numpy as np


def jacobian_transpose_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    racket_site_id: int,
    target_pos: np.ndarray,
    gain: float = 50.0,
) -> np.ndarray:
    """雅可比转置控制：将球拍中心移向目标位置。

    Args:
        model: MuJoCo 模型。
        data: MuJoCo 数据。
        racket_site_id: 球拍中心 site ID。
        target_pos: 目标位置 (3,)。
        gain: 增益。

    Returns:
        关节力矩 (6,)。
    """
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, racket_site_id)

    ee_pos = data.site_xpos[racket_site_id].copy()
    error = target_pos - ee_pos

    nu = 6
    jac_arm = jacp[:, :nu]
    tau = gain * jac_arm.T @ error
    return tau


def main() -> None:
    parser = argparse.ArgumentParser(description="RM-65 网球机器人 MuJoCo 仿真")
    parser.add_argument(
        "--model",
        type=str,
        default="src/robot/rm65_model.xml",
        help="MuJoCo XML 模型路径",
    )
    parser.add_argument(
        "--target",
        type=float,
        nargs=3,
        default=[0.0, -0.4, 1.3],
        help="球拍目标位置 x y z",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"模型文件不存在: {model_path}")
        sys.exit(1)

    model = load_mujoco_model(model_path)
    data = mujoco.MjData(model)

    racket_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "racket_center")
    target_pos = np.array(args.target, dtype=np.float64)

    # 初始关节角：右臂稍微弯曲准备挥拍
    data.qpos[:6] = [0.0, -0.8, 1.5, -0.7, 0.0, 0.0]
    data.qpos[6:12] = [0.0, 0.8, -1.5, 0.7, 0.0, 0.0]

    # 球放到前方
    ball_qpos_adr = model.jnt_qposadr[12]
    data.qpos[ball_qpos_adr : ball_qpos_adr + 3] = [1.0, 0.0, 1.5]
    data.qpos[ball_qpos_adr + 3 : ball_qpos_adr + 7] = [1, 0, 0, 0]

    mujoco.mj_forward(model, data)

    print(f"模型: {model_path}")
    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print(f"球拍初始位置: {data.site_xpos[racket_site_id]}")
    print(f"目标位置: {target_pos}")
    print("启动查看器...")

    step_count = [0]

    def control_callback(m: mujoco.MjModel, d: mujoco.MjData) -> None:
        step_count[0] += 1

        # 雅可比转置控制右臂
        tau = jacobian_transpose_step(m, d, racket_site_id, target_pos, gain=80.0)

        ctrl_lo = model.actuator_ctrlrange[:6, 0]
        ctrl_hi = model.actuator_ctrlrange[:6, 1]
        d.ctrl[:6] = np.clip(tau, ctrl_lo, ctrl_hi)

        # 每 500 步打印一次球拍误差
        if step_count[0] % 500 == 0:
            err = np.linalg.norm(d.site_xpos[racket_site_id] - target_pos)
            print(f"  步 {step_count[0]}: 球拍误差={err:.4f}m")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            control_callback(model, data)
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()

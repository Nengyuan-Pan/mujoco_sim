"""可视化工具：回放轨迹（支持 MuJoCo 查看器和 matplotlib 绘图）。"""

import time
import logging
import numpy as np
import mujoco
import mujoco.viewer
from src.sim.env import MujocoEnv
from src.tennis.ball import ball_trajectory

logger = logging.getLogger(__name__)


def visualize_result(
    env: MujocoEnv,
    X: np.ndarray,
    U: np.ndarray,
    p0: np.ndarray,
    v0: np.ndarray,
    g: np.ndarray,
    config: dict,
    ball_positions_phys: np.ndarray | None = None,
) -> None:
    """在 MuJoCo 查看器中可视化 iLQT 优化结果。

    同时显示机械臂运动和网球飞行轨迹。

    Args:
        env: MuJoCo 环境实例。
        X: 状态轨迹，形状 (N+1, 12)。
        U: 控制轨迹，形状 (N, 6)。
        p0: 球初始位置，形状 (3,)。
        v0: 球初始速度，形状 (3,)。
        g: 重力加速度，形状 (3,)。
        config: 可视化配置字典。
        ball_positions_phys: MuJoCo 物理仿真得到的球轨迹，形状 (M, 3)。
            若提供则优先使用（比解析公式更真实），否则用解析抛物线公式。
    """
    N = len(U)
    dt = env.dt
    viewer_cfg = config.get("viewer", {})
    playback_speed = viewer_cfg.get("playback_speed", 1.0)
    loop = viewer_cfg.get("loop", True)

    cam_distance = viewer_cfg.get("camera_distance", 2.5)
    cam_elevation = viewer_cfg.get("camera_elevation", -20)
    cam_azimuth = viewer_cfg.get("camera_azimuth", 135)

    # 球轨迹：优先使用 MuJoCo 物理仿真结果，否则回退到解析公式
    total_frames = N + 30
    ball_positions = np.zeros((total_frames, 3))
    if ball_positions_phys is not None and len(ball_positions_phys) > 0:
        n_phys = min(len(ball_positions_phys), total_frames)
        ball_positions[:n_phys] = ball_positions_phys[:n_phys]
        # 剩余帧用最后位置填充
        if n_phys < total_frames:
            ball_positions[n_phys:] = ball_positions_phys[-1]
    else:
        for k in range(total_frames):
            ball_positions[k] = ball_trajectory(p0, v0, g, k * dt)

    NQ = env.NQ
    data = env.data
    model = env.model

    # 用墙钟时间驱动帧索引，确保真实速度播放
    start_time = time.perf_counter()
    last_idx = -1

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = cam_distance
        viewer.cam.elevation = cam_elevation
        viewer.cam.azimuth = cam_azimuth
        viewer.cam.lookat[:] = [0.0, 0.0, 1.0]

        while viewer.is_running():
            elapsed = time.perf_counter() - start_time
            sim_time = elapsed * playback_speed
            idx = int(sim_time / dt)

            if idx >= total_frames:
                if loop:
                    start_time = time.perf_counter()
                    idx = 0
                else:
                    idx = total_frames - 1

            if idx != last_idx:
                last_idx = idx

                arm_x = X[idx] if idx <= N else X[-1]
                data.qpos[:NQ] = arm_x[:NQ]
                data.qvel[:NQ] = arm_x[NQ:]

                bp = ball_positions[idx]
                data.qpos[NQ:NQ + 3] = bp
                data.qpos[NQ + 3:NQ + 7] = [1, 0, 0, 0]
                data.qvel[NQ:NQ + 3] = 0.0
                data.qvel[NQ + 3:NQ + 6] = 0.0

                mujoco.mj_forward(model, data)

            viewer.sync()
            time.sleep(1.0 / 120.0)


class RealTimeViewer:
    """MPC 实时可视化查看器。

    在 MPC 循环中持续运行，每一步调用 sync() 更新画面。
    支持 overlay 文字显示剩余时间、误差等信息。
    """

    def __init__(
        self,
        env: MujocoEnv,
        config: dict,
    ) -> None:
        """初始化实时查看器。

        Args:
            env: MuJoCo 环境实例。
            config: 配置字典。
        """
        self.env = env
        viewer_cfg = config.get("viewer", {})

        self.cam_distance = viewer_cfg.get("camera_distance", 3.5)
        self.cam_elevation = viewer_cfg.get("camera_elevation", -15)
        self.cam_azimuth = viewer_cfg.get("camera_azimuth", 135)
        self.cam_lookat = viewer_cfg.get("camera_lookat", [0.0, 0.0, 1.0])

        self._viewer = None

    def start(self) -> None:
        """启动查看器窗口。"""
        self._viewer = mujoco.viewer.launch_passive(
            self.env.model, self.env.data
        )
        self._viewer.cam.distance = self.cam_distance
        self._viewer.cam.elevation = self.cam_elevation
        self._viewer.cam.azimuth = self.cam_azimuth
        self._viewer.cam.lookat[:] = self.cam_lookat

    def sync(self) -> None:
        """同步渲染一帧画面。"""
        if self._viewer is not None and self._viewer.is_running():
            mujoco.mj_forward(self.env.model, self.env.data)
            self._viewer.sync()

    def is_running(self) -> bool:
        """检查查看器是否仍在运行。"""
        if self._viewer is None:
            return False
        return self._viewer.is_running()

    def close(self) -> None:
        """关闭查看器。"""
        self._viewer = None


def plot_results(
    X: np.ndarray,
    U: np.ndarray,
    p0: np.ndarray,
    v0: np.ndarray,
    g: np.ndarray,
    p_hit: np.ndarray,
    v_hit: np.ndarray,
    cost_history: list[float],
    dt: float,
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """用 matplotlib 绘制优化结果图表。

    Args:
        X: 状态轨迹，形状 (N+1, 12)。
        U: 控制轨迹，形状 (N, 6)。
        p0: 球初始位置。
        v0: 球初始速度。
        g: 重力加速度。
        p_hit: 期望击打位置。
        v_hit: 期望击打速度。
        cost_history: 代价历史。
        dt: 时间步长。
        save_path: 图片保存路径前缀。若为 None 则自动保存到 results/。
        show: 是否交互显示（会阻塞）。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if save_path is None:
        from pathlib import Path
        results_dir = Path(__file__).resolve().parent.parent.parent / "results"
        results_dir.mkdir(exist_ok=True)
        save_path = str(results_dir / "ilqt_result")

    N = len(U)
    joint_names = [
        "shoulder_pan", "shoulder_lift", "elbow",
        "wrist_1", "wrist_2", "wrist_3",
    ]
    t_arr = np.arange(N + 1) * dt
    t_u = np.arange(N) * dt

    # 关节角度 + 速度
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("iLQT Optimization Results", fontsize=14)
    for j in range(6):
        ax = axes[j // 3, j % 3]
        ax.plot(t_arr, X[:, j], label=f"{joint_names[j]} (q)")
        ax.plot(t_arr, X[:, j + 6], "--", label=f"{joint_names[j]} (qdot)")
        ax.set_title(joint_names[j])
        ax.set_xlabel("Time (s)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}_joints.png", dpi=150)
    plt.close()

    # 控制力矩
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Control Torques", fontsize=14)
    for j in range(6):
        ax = axes[j // 3, j % 3]
        ax.plot(t_u, U[:, j])
        ax.set_title(f"tau_{joint_names[j]}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Torque (N*m)")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path}_controls.png", dpi=150)
    plt.close()

    # 3D 轨迹
    from src.sim.env import MujocoEnv
    from pathlib import Path

    model_path = Path(__file__).resolve().parent.parent / "robot" / "model.xml"
    env_temp = MujocoEnv(model_path, dt=dt)

    ee_positions = []
    for k in range(N + 1):
        env_temp.set_arm_state(X[k])
        ee_positions.append(env_temp.get_ee_pos())
    ee_positions = np.array(ee_positions)

    ball_positions = []
    for k in range(N + 1):
        t = k * dt
        ball_positions.append(ball_trajectory(p0, v0, g, t))
    ball_positions = np.array(ball_positions)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2],
            "b-", linewidth=2, label="Racket")
    ax.plot(ball_positions[:, 0], ball_positions[:, 1], ball_positions[:, 2],
            "y--", linewidth=2, label="Tennis Ball")
    ax.scatter(*p_hit, color="r", s=100, marker="*", label="Hit Target")
    ax.scatter(*ee_positions[-1], color="b", s=100, marker="o", label="EE Final Pos")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D Trajectories")
    ax.legend()
    plt.savefig(f"{save_path}_3d.png", dpi=150)
    plt.close()

    # 代价收敛
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cost_history, "b-", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total Cost")
    ax.set_title("iLQT Cost Convergence")
    ax.grid(True, alpha=0.3)
    plt.savefig(f"{save_path}_cost.png", dpi=150)
    plt.close()

    print(f"Plots saved to {save_path}_*.png")

    if show:
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt2
        figs = []
        for suffix in ["_joints.png", "_controls.png", "_3d.png", "_cost.png"]:
            img = plt2.imread(f"{save_path}{suffix}")
            fig, ax = plt2.subplots(1, 1, figsize=(12, 8))
            ax.imshow(img)
            ax.axis("off")
            figs.append(fig)
        plt2.show()

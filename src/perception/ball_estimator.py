"""6-DOF 网球状态卡尔曼滤波器。

向前兼容设计：
  - 零外部依赖（仅 numpy），不访问 MuJoCo 或环境
  - update() 是唯一公共入口
  - 弹跳保护内置，外部无需感知
"""

import numpy as np
from numpy.typing import NDArray
import time


class BallEstimator:
    """6-DOF 网球状态卡尔曼滤波器。

    状态向量:  [x, y, z, vx, vy, vz]
    观测向量:  [x, y, z, vx, vy, vz]（直接观测全状态）
    过程模型:  匀速 + 重力加速度（线性）
    弹跳处理:  检测 vz 由负转正 → slam 状态 → 重置 Z-vel 协方差
    """

    def __init__(
        self,
        dt: float,
        pos_noise_std: float = 0.0,
        vel_noise_std: float = 0.0,
        pos_noise_xyz: tuple[float, float, float] | None = None,
        vel_noise_xyz: tuple[float, float, float] | None = None,
        process_noise_pos: float = 0.001,
        process_noise_vel: float = 0.01,
        bounce_restitution: float = 0.75,
        g: float = 9.80665,
    ) -> None:
        """初始化 6D 卡尔曼滤波器。

        Args:
            dt: 仿真时间步长（s）。
            pos_noise_std: 位置观测噪声标准差 (m)，标量模式。
            vel_noise_std: 速度观测噪声标准差 (m/s)，标量模式。
            pos_noise_xyz: per-axis 位置噪声 (σx, σy, σz)，优先于标量。
            vel_noise_xyz: per-axis 速度噪声 (σx, σy, σz)，优先于标量。
            process_noise_pos: 位置过程噪声标准差 (m)。
            process_noise_vel: 速度过程噪声标准差 (m/s)。
            bounce_restitution: 弹跳恢复系数。
            g: 重力加速度 (m/s²)，向下为正。
        """
        self._dt = dt
        self._g = g
        self._bounce_restitution = bounce_restitution

        self._initialized = False
        self._last_update_time: float | None = None

        self._n = 6
        self._I = np.eye(self._n)

        self._R = self._build_R(pos_noise_std, vel_noise_std,
                                pos_noise_xyz, vel_noise_xyz)
        self._Q = self._build_Q(process_noise_pos, process_noise_vel)
        self._H = np.eye(self._n)

        self._x = np.zeros(self._n)
        self._P = np.eye(self._n) * 100.0

    @property
    def initialized(self) -> bool:
        """滤波器是否已接收过至少一次观测。"""
        return self._initialized

    def _build_R(
        self, pos_std: float, vel_std: float,
        pos_xyz: tuple | None, vel_xyz: tuple | None,
    ) -> NDArray[np.floating]:
        """构造观测噪声协方差矩阵 R（per-axis 优先）。"""
        R = np.zeros((self._n, self._n))
        if pos_xyz is not None:
            R[0, 0], R[1, 1], R[2, 2] = pos_xyz[0]**2, pos_xyz[1]**2, pos_xyz[2]**2
        else:
            R[0, 0] = R[1, 1] = R[2, 2] = pos_std ** 2
        if vel_xyz is not None:
            R[3, 3], R[4, 4], R[5, 5] = vel_xyz[0]**2, vel_xyz[1]**2, vel_xyz[2]**2
        else:
            R[3, 3] = R[4, 4] = R[5, 5] = vel_std ** 2
        return R

    def _build_Q(
        self, pos_std: float, vel_std: float,
    ) -> NDArray[np.floating]:
        """构造过程噪声协方差矩阵 Q。"""
        Q = np.zeros((self._n, self._n))
        Q[0, 0] = Q[1, 1] = Q[2, 2] = pos_std ** 2
        Q[3, 3] = Q[4, 4] = Q[5, 5] = vel_std ** 2
        return Q

    def _predict(self, dt: float) -> None:
        """卡尔曼预测步：x̄ = F·x + Bu_g, P̄ = F·P·F' + Q。"""
        I3 = np.eye(3)
        Z3 = np.zeros((3, 3))
        F = np.block([[I3, dt * I3], [Z3, I3]])

        Bu_g = np.zeros(self._n)
        Bu_g[2] = -0.5 * self._g * dt * dt
        Bu_g[5] = -self._g * dt

        self._x = F @ self._x + Bu_g
        self._P = F @ self._P @ F.T + self._Q

    def update(
        self, z_pos: NDArray[np.floating], z_vel: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """用新观测更新状态。

        Args:
            z_pos: 观测球位置 (3,)。
            z_vel: 观测球速度 (3,)。

        Returns:
            (filtered_pos, filtered_vel) 滤波后的球位置和速度。
        """
        # R=0 直通透传：观测无噪时不运行预测/校正循环
        if np.allclose(self._R, 0):
            self._x = np.hstack([z_pos, z_vel])
            self._initialized = True
            self._last_update_time = time.perf_counter()
            return z_pos.copy(), z_vel.copy()

        if not self._initialized:
            self._x = np.hstack([z_pos, z_vel])
            self._initialized = True
            self._last_update_time = time.perf_counter()
            return z_pos.copy(), z_vel.copy()

        now = time.perf_counter()
        elapsed = self._dt
        if self._last_update_time is not None:
            elapsed = max(now - self._last_update_time, self._dt)
        self._last_update_time = now

        self._predict(elapsed)

        # 弹跳检测：预测 vz 为负且观测 vz 为正 → 弹跳发生
        pred_vz = self._x[5]
        obs_vz = z_vel[2]
        if pred_vz < -1.0 and obs_vz > 1.0:
            self._x[2] = max(z_pos[2], 0.02)
            self._x[5] = -pred_vz * self._bounce_restitution
            self._P[5, :] = 0.0
            self._P[:, 5] = 0.0
            self._P[5, 5] = 100.0

        z = np.hstack([z_pos, z_vel])

        # K = P·H'·(H·P·H' + R)⁻¹
        H = self._H
        P_HT = self._P @ H.T
        S = H @ P_HT + self._R
        K = P_HT @ np.linalg.inv(S)

        # x = x̄ + K·(z - H·x̄)
        innovation = z - H @ self._x
        self._x = self._x + K @ innovation

        # P = (I - K·H)·P
        self._P = (self._I - K @ H) @ self._P

        return self._x[:3].copy(), self._x[3:].copy()

    def update_noise_params(
        self,
        pos_noise_std: float | None = None,
        vel_noise_std: float | None = None,
        pos_noise_xyz: tuple[float, float, float] | None = None,
        vel_noise_xyz: tuple[float, float, float] | None = None,
    ) -> None:
        """动态更新观测噪声协方差矩阵 R。

        Args:
            pos_noise_std: 新的位置标量噪声 std，None 表示不修改。
            vel_noise_std: 新的速度标量噪声 std，None 表示不修改。
            pos_noise_xyz: 新的 per-axis 位置噪声，None 表示保持当前值。
            vel_noise_xyz: 新的 per-axis 速度噪声，None 表示保持当前值。
        """
        if pos_noise_xyz is not None:
            self._R[0, 0] = pos_noise_xyz[0] ** 2
            self._R[1, 1] = pos_noise_xyz[1] ** 2
            self._R[2, 2] = pos_noise_xyz[2] ** 2
        elif pos_noise_std is not None:
            self._R[0, 0] = self._R[1, 1] = self._R[2, 2] = pos_noise_std ** 2
        if vel_noise_xyz is not None:
            self._R[3, 3] = vel_noise_xyz[0] ** 2
            self._R[4, 4] = vel_noise_xyz[1] ** 2
            self._R[5, 5] = vel_noise_xyz[2] ** 2
        elif vel_noise_std is not None:
            self._R[3, 3] = self._R[4, 4] = self._R[5, 5] = vel_noise_std ** 2

    def reset(self) -> None:
        """完全重置滤波器（用于新的发球/实验）。"""
        self._initialized = False
        self._last_update_time = None
        self._x = np.zeros(self._n)
        self._P = np.eye(self._n) * 100.0

    @property
    def state(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """滤波后的球状态 (pos, vel)。"""
        return self._x[:3].copy(), self._x[3:].copy()

    @property
    def covariance(self) -> NDArray[np.floating]:
        """状态协方差矩阵 P。"""
        return self._P.copy()

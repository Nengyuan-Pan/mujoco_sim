"""观测频率门控：模拟低频摄像机的观测约束。

MuJoCo 物理引擎以 200Hz 运行，观测门控决定 MPC 在哪些步能"看到"球真值，
其余步使用抛物线预测或 KF predict-only。
"""

import numpy as np
from numpy.typing import NDArray

from src.perception.ball_estimator import BallEstimator
from src.utils.noise import add_observation_noise


_G = 9.80665


class BallObservationGate:
    """观测频率门控：模拟低频摄像机的观测约束。

    观测步：读 MuJoCo 真值 → 可选加噪 → 可选 KF update → 缓存 → 返回
    非观测步：基于上次观测值做抛物线预测 或 KF predict-only → 返回
    """

    def __init__(
        self,
        obs_freq: float,
        physics_dt: float,
        noise_pos: float = 0.0,
        noise_vel: float = 0.0,
        pos_std_xyz: tuple[float, float, float] | None = None,
        vel_std_xyz: tuple[float, float, float] | None = None,
        kf: BallEstimator | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        """初始化观测门控。

        Args:
            obs_freq: 观测频率 (Hz)。200=每步观测（等价无门控）。
            physics_dt: 物理仿真时间步长 (s)，通常 0.005。
            noise_pos: 位置观测噪声 std (m)，0=不加噪。
            noise_vel: 速度观测噪声 std (m/s)，0=不加噪。
            pos_std_xyz: per-axis 位置噪声，优先于 noise_pos。
            vel_std_xyz: per-axis 速度噪声，优先于 noise_vel。
            kf: BallEstimator 实例，None 时使用内部抛物线预测。
            rng: 随机数生成器。
        """
        self.obs_interval = max(1, round(1.0 / (obs_freq * physics_dt)))
        self.physics_dt = physics_dt
        self.noise_pos = noise_pos
        self.noise_vel = noise_vel
        self.pos_std_xyz = pos_std_xyz
        self.vel_std_xyz = vel_std_xyz
        self.kf = kf
        self.rng = rng or np.random.default_rng(0)

        self.last_obs_pos: NDArray[np.floating] | None = None
        self.last_obs_vel: NDArray[np.floating] | None = None
        self.last_obs_step: int = -1

    def _is_obs_step(self, step: int) -> bool:
        """判断当前步是否为观测步。"""
        return step == 0 or step % self.obs_interval == 0

    def _parabolic_predict(
        self, elapsed_steps: int,
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """基于上次观测值的抛物线预测。

        Args:
            elapsed_steps: 距上次观测经过的物理步数。

        Returns:
            (pred_pos, pred_vel) 预测的球位置和速度。
        """
        dt_total = elapsed_steps * self.physics_dt
        assert self.last_obs_pos is not None and self.last_obs_vel is not None
        pred_pos = self.last_obs_pos + self.last_obs_vel * dt_total
        pred_pos = pred_pos.copy()
        pred_pos[2] += 0.5 * (-_G) * dt_total**2
        pred_vel = self.last_obs_vel.copy()
        pred_vel[2] += (-_G) * dt_total
        return pred_pos, pred_vel

    def get_state(
        self,
        step: int,
        true_pos: NDArray[np.floating],
        true_vel: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """返回门控后的球状态。

        Args:
            step: 当前物理仿真步数。
            true_pos: MuJoCo 球真值位置 (3,)。
            true_vel: MuJoCo 球真值速度 (3,)。

        Returns:
            (pos, vel) 门控后的球位置和速度。
        """
        if self._is_obs_step(step):
            pos = true_pos.copy()
            vel = true_vel.copy()

            if self.noise_pos > 0 or self.noise_vel > 0 or self.pos_std_xyz is not None or self.vel_std_xyz is not None:
                pos, vel = add_observation_noise(
                    pos, vel, self.rng,
                    pos_std=self.noise_pos, vel_std=self.noise_vel,
                    pos_std_xyz=self.pos_std_xyz, vel_std_xyz=self.vel_std_xyz,
                )

            if self.kf is not None:
                pos, vel = self.kf.update(pos, vel)

            self.last_obs_pos = pos.copy()
            self.last_obs_vel = vel.copy()
            self.last_obs_step = step
            return pos, vel
        else:
            elapsed = step - self.last_obs_step

            if self.kf is not None:
                pos, vel = self.kf.predict_only(self.physics_dt)
                return pos, vel
            else:
                return self._parabolic_predict(elapsed)

    def reset(self) -> None:
        """重置门控状态（新发球/新 episode 时调用）。"""
        self.last_obs_pos = None
        self.last_obs_vel = None
        self.last_obs_step = -1
        if self.kf is not None:
            self.kf.reset()

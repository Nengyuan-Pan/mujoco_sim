"""仿真噪声注入工具函数。"""

import numpy as np
from numpy.typing import NDArray


_GROUND_Z = 0.01


def _apply_noise(
    arr: NDArray[np.floating],
    rng: np.random.Generator,
    scalar_std: float,
    per_axis_std: tuple[float, float, float] | None,
) -> NDArray[np.floating]:
    """对标量或 per-axis 标准差施加高斯噪声。per-axis 优先。"""
    if per_axis_std is not None:
        stds = np.array(per_axis_std)
        if np.any(stds > 0):
            return arr + rng.normal(0, 1, size=arr.shape) * stds
        return arr.copy()
    if scalar_std > 0:
        return arr + rng.normal(0, scalar_std, size=arr.shape)
    return arr.copy()


def add_observation_noise(
    ball_pos: NDArray[np.floating],
    ball_vel: NDArray[np.floating],
    rng: np.random.Generator,
    pos_std: float = 0.0,
    vel_std: float = 0.0,
    pos_std_xyz: tuple[float, float, float] | None = None,
    vel_std_xyz: tuple[float, float, float] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """给球位置/速度观测加高斯噪声。

    支持两种模式：
    - 标量模式：pos_std/vel_std 对三轴使用相同标准差（向后兼容）
    - per-axis 模式：pos_std_xyz/vel_std_xyz 各轴独立标准差

    per-axis 优先：若 pos_std_xyz 非 None，忽略 pos_std。
    位置 Z 坐标不低于地面（0.01m）。

    Args:
        ball_pos: 球真实位置 (3,)。
        ball_vel: 球真实速度 (3,)。
        rng: 随机数生成器。
        pos_std: 位置噪声标准差 (m)，0 表示不加噪声。
        vel_std: 速度噪声标准差 (m/s)，0 表示不加噪声。
        pos_std_xyz: 位置各轴标准差 (σx, σy, σz)，None 表示使用 pos_std。
        vel_std_xyz: 速度各轴标准差 (σx, σy, σz)，None 表示使用 vel_std。

    Returns:
        (noisy_pos, noisy_vel)，std=0 时原样返回。
    """
    pos_out = _apply_noise(ball_pos, rng, pos_std, pos_std_xyz)
    pos_out[2] = max(float(pos_out[2]), _GROUND_Z)
    vel_out = _apply_noise(ball_vel, rng, vel_std, vel_std_xyz)
    return pos_out, vel_out


def add_torque_noise(
    u: NDArray[np.floating],
    rng: np.random.Generator,
    torque_std: float = 0.0,
) -> NDArray[np.floating]:
    """给力矩控制加高斯噪声。

    Args:
        u: 控制力矩 (6,)。
        rng: 随机数生成器。
        torque_std: 力矩噪声标准差 (Nm)，0 表示不加噪声。

    Returns:
        含噪声的力矩，std=0 时原样返回。
    """
    if torque_std > 0:
        return u + rng.normal(0, torque_std, size=u.shape)
    return u.copy()


def randomize_init_q(
    init_q: NDArray[np.floating],
    rng: np.random.Generator,
    noise_rad: float = 0.0,
) -> NDArray[np.floating]:
    """给初始关节角度加高斯噪声。

    Args:
        init_q: 初始关节角度 (6,)。
        rng: 随机数生成器。
        noise_rad: 角度噪声标准差 (rad)，0 表示不加噪声。

    Returns:
        含噪声的角度，noise_rad=0 时原样返回。
    """
    if noise_rad > 0:
        return init_q + rng.normal(0, noise_rad, size=init_q.shape)
    return init_q.copy()

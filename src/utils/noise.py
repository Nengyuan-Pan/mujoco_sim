"""仿真噪声注入工具函数。"""

import numpy as np
from numpy.typing import NDArray


def add_observation_noise(
    ball_pos: NDArray[np.floating],
    ball_vel: NDArray[np.floating],
    rng: np.random.Generator,
    pos_std: float = 0.0,
    vel_std: float = 0.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """给球位置/速度观测加高斯噪声。

    Args:
        ball_pos: 球真实位置 (3,)。
        ball_vel: 球真实速度 (3,)。
        rng: 随机数生成器。
        pos_std: 位置噪声标准差 (m)，0 表示不加噪声。
        vel_std: 速度噪声标准差 (m/s)，0 表示不加噪声。

    Returns:
        (noisy_pos, noisy_vel)，std=0 时原样返回。
    """
    pos_out = ball_pos.copy()
    vel_out = ball_vel.copy()
    if pos_std > 0:
        pos_out = pos_out + rng.normal(0, pos_std, size=3)
    if vel_std > 0:
        vel_out = vel_out + rng.normal(0, vel_std, size=3)
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

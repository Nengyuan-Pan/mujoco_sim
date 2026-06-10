"""exp8 实验配置工厂：替代 monkey-patch 的纯函数。

提供 make_preprocessor() 和 make_estimator() 两个工厂函数，
用于根据噪声模式创建 RM65Env 所需的 preprocessor 和 BallEstimator。
"""

import numpy as np
from src.perception.ball_estimator import BallEstimator
from src.utils.noise import add_observation_noise

DT = 0.005


def make_preprocessor(noise_mode: str, seed: int):
    """根据噪声模式创建观测预处理回调。

    Args:
        noise_mode: off | lo | mid | hi | anis
        seed: 随机种子基准。

    Returns:
        回调 (pos, vel) → (noisy_pos, noisy_vel)，off 模式返回 None。
    """
    if noise_mode == "off":
        return None

    rng = np.random.default_rng(seed + 7000)
    if noise_mode == "lo":
        pos_std, vel_std = 0.02, 0.2
        pos_xyz = vel_xyz = None
    elif noise_mode == "mid":
        pos_std, vel_std = 0.05, 0.5
        pos_xyz = vel_xyz = None
    elif noise_mode == "hi":
        pos_std, vel_std = 0.10, 1.0
        pos_xyz = vel_xyz = None
    elif noise_mode == "anis":
        pos_std, vel_std = 0.0, 0.0
        pos_xyz = (0.03, 0.10, 0.03)
        vel_xyz = (0.3, 1.0, 0.3)
    else:
        raise ValueError(f"未知噪声模式: {noise_mode}")

    def preprocessor(pos, vel):
        return add_observation_noise(
            pos, vel, rng,
            pos_std=pos_std, vel_std=vel_std,
            pos_std_xyz=pos_xyz, vel_std_xyz=vel_xyz,
        )

    return preprocessor


def make_estimator(noise_mode: str) -> BallEstimator:
    """根据噪声模式创建 BallEstimator。

    Args:
        noise_mode: off | lo | mid | hi | anis

    Returns:
        配置好的 BallEstimator。off 模式使用 R=0 直通透传。
    """
    if noise_mode == "off":
        return BallEstimator(dt=DT, pos_noise_std=0.0, vel_noise_std=0.0)
    elif noise_mode == "lo":
        return BallEstimator(dt=DT, pos_noise_std=0.02, vel_noise_std=0.2)
    elif noise_mode == "mid":
        return BallEstimator(dt=DT, pos_noise_std=0.05, vel_noise_std=0.5)
    elif noise_mode == "hi":
        return BallEstimator(dt=DT, pos_noise_std=0.10, vel_noise_std=1.0)
    elif noise_mode == "anis":
        return BallEstimator(
            dt=DT,
            pos_noise_xyz=(0.03, 0.10, 0.03),
            vel_noise_xyz=(0.3, 1.0, 0.3),
        )
    raise ValueError(f"未知噪声模式: {noise_mode}")

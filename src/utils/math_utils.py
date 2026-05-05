"""通用数学工具函数。"""

import numpy as np
from numpy.typing import NDArray


def normalize(v: NDArray[np.floating]) -> NDArray[np.floating]:
    """归一化向量。

    Args:
        v: 输入向量。

    Returns:
        单位向量。若输入为零向量，返回零向量。
    """
    n = np.linalg.norm(v)
    if n < 1e-10:
        return v.copy()
    return v / n


def rotation_matrix_x(angle: float) -> NDArray[np.float64]:
    """绕 x 轴的旋转矩阵。"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rotation_matrix_y(angle: float) -> NDArray[np.float64]:
    """绕 y 轴的旋转矩阵。"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotation_matrix_z(angle: float) -> NDArray[np.float64]:
    """绕 z 轴的旋转矩阵。"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

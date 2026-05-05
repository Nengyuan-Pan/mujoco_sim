"""网球抛物线轨迹预测与随机生成（含弹跳模型）。"""

import numpy as np
from numpy.typing import NDArray


def ball_trajectory(
    p0: NDArray[np.floating],
    v0: NDArray[np.floating],
    g: NDArray[np.floating],
    t: float,
) -> NDArray[np.floating]:
    """计算网球在时刻 t 的位置（抛物线运动）。

    Args:
        p0: 初始位置，形状 (3,)。
        v0: 初始速度，形状 (3,)。
        g: 重力加速度，形状 (3,)。
        t: 时间（秒）。

    Returns:
        时刻 t 的球位置，形状 (3,)。
    """
    return p0 + v0 * t + 0.5 * g * t**2


def ball_velocity(
    v0: NDArray[np.floating],
    g: NDArray[np.floating],
    t: float,
) -> NDArray[np.floating]:
    """计算网球在时刻 t 的速度。

    Args:
        v0: 初始速度，形状 (3,)。
        g: 重力加速度，形状 (3,)。
        t: 时间（秒）。

    Returns:
        时刻 t 的球速度，形状 (3,)。
    """
    return v0 + g * t


def ball_trajectory_with_bounce(
    p0: NDArray[np.floating],
    v0: NDArray[np.floating],
    g: NDArray[np.floating],
    t: float,
    bounce_restitution: float = 0.75,
) -> NDArray[np.floating]:
    """计算含地面弹跳的网球在时刻 t 的位置。

    简化弹跳模型：球触地时 Z 分量速度反向并乘以恢复系数，
    弹起后继续受重力飞行。仅计算一次弹跳。

    Args:
        p0: 初始位置，形状 (3,)。
        v0: 初始速度，形状 (3,)。
        g: 重力加速度，形状 (3,)。
        t: 时间（秒）。
        bounce_restitution: 弹跳恢复系数，默认 0.75。

    Returns:
        时刻 t 的球位置，形状 (3,)。
    """
    # 计算弹跳时刻：p0[2] + v0[2]*t + 0.5*g[2]*t^2 = 0
    # 0.5*g[2]*t_b^2 + v0[2]*t_b + p0[2] = 0
    a = 0.5 * g[2]
    b = v0[2]
    c = p0[2]

    discriminant = b * b - 4.0 * a * c
    t_bounce = None

    if discriminant >= 0 and a != 0:
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b + sqrt_disc) / (2.0 * a)
        t2 = (-b - sqrt_disc) / (2.0 * a)
        # 取最小的正根
        candidates = [t_val for t_val in [t1, t2] if t_val > 1e-6]
        if candidates:
            t_bounce = min(candidates)

    if t_bounce is None or t < t_bounce:
        # 弹跳前，正常抛物线
        return p0 + v0 * t + 0.5 * g * t**2

    # 弹跳后：从弹跳点出发，Z 速度反转并乘恢复系数
    p_bounce = p0 + v0 * t_bounce + 0.5 * g * t_bounce**2
    v_bounce = v0 + g * t_bounce
    v_bounce[2] = -v_bounce[2] * bounce_restitution

    dt_after = t - t_bounce
    return p_bounce + v_bounce * dt_after + 0.5 * g * dt_after**2


def ball_velocity_with_bounce(
    p0: NDArray[np.floating],
    v0: NDArray[np.floating],
    g: NDArray[np.floating],
    t: float,
    bounce_restitution: float = 0.75,
) -> NDArray[np.floating]:
    """计算含弹跳的网球在时刻 t 的速度。

    Args:
        p0: 初始位置，形状 (3,)。
        v0: 初始速度，形状 (3,)。
        g: 重力加速度，形状 (3,)。
        t: 时间（秒）。
        bounce_restitution: 弹跳恢复系数。

    Returns:
        时刻 t 的球速度，形状 (3,)。
    """
    a = 0.5 * g[2]
    b = v0[2]
    c = p0[2]

    discriminant = b * b - 4.0 * a * c
    t_bounce = None

    if discriminant >= 0 and a != 0:
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b + sqrt_disc) / (2.0 * a)
        t2 = (-b - sqrt_disc) / (2.0 * a)
        candidates = [t_val for t_val in [t1, t2] if t_val > 1e-6]
        if candidates:
            t_bounce = min(candidates)

    if t_bounce is None or t < t_bounce:
        return v0 + g * t

    v_bounce = v0 + g * t_bounce
    v_bounce[2] = -v_bounce[2] * bounce_restitution
    dt_after = t - t_bounce
    return v_bounce + g * dt_after


def compute_bounce_time(
    p0: NDArray[np.floating],
    v0: NDArray[np.floating],
    g: NDArray[np.floating],
) -> float | None:
    """计算球第一次触地的时间。

    Args:
        p0: 初始位置，形状 (3,)。
        v0: 初始速度，形状 (3,)。
        g: 重力加速度，形状 (3,)。

    Returns:
        弹跳时刻（秒），若不触地则返回 None。
    """
    a = 0.5 * g[2]
    b = v0[2]
    c = p0[2]

    discriminant = b * b - 4.0 * a * c
    if discriminant < 0 or a == 0:
        return None

    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b + sqrt_disc) / (2.0 * a)
    t2 = (-b - sqrt_disc) / (2.0 * a)
    candidates = [t_val for t_val in [t1, t2] if t_val > 1e-6]
    if not candidates:
        return None
    return min(candidates)


def generate_serve_ball(
    shoulder_pos: NDArray[np.floating],
    workspace_radius: float,
    g: NDArray[np.floating],
    hit_time: float,
    serve_distance: float = 22.0,
    serve_height_range: tuple[float, float] = (2.5, 3.0),
    bounce_restitution: float = 0.75,
    target_speed_range: tuple[float, float] = (15.0, 25.0),
    hit_offset_ranges: dict | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """生成发球机发出的网球轨迹（含弹跳）。

    球从约 22m 远处发出，飞行约 0.3-0.5s 后触地弹起，继续飞向机器人。
    到达工作区域时速度约 15-25 m/s。

    Args:
        shoulder_pos: 肩关节世界坐标位置，形状 (3,)。
        workspace_radius: 工作空间半径（米）。
        g: 重力加速度，形状 (3,)。
        hit_time: 期望击打时间（秒）。
        serve_distance: 发球 X 距离（米）。
        serve_height_range: 发球高度范围 (min, max)。
        bounce_restitution: 弹跳恢复系数。
        target_speed_range: 到达工作区的速度范围 (min, max)。
        hit_offset_ranges: 击打点偏移范围字典。
        rng: 随机数生成器。

    Returns:
        (p0, v0, p_hit): 球初始位置、初始速度、击打点位置。
    """
    if rng is None:
        rng = np.random.default_rng()

    # 发球位置：X ≈ -serve_distance
    x0 = -serve_distance + rng.uniform(-1.0, 1.0)
    y0 = rng.uniform(-0.8, 0.8)
    z0 = rng.uniform(serve_height_range[0], serve_height_range[1])
    p0 = np.array([x0, y0, z0])

    # 弹跳时刻：球在 0.3-0.5s 触地
    t_bounce_desired = rng.uniform(0.3, 0.5)
    # 从 z0 下落到地面：z0 + vz0*t_b + 0.5*gz*t_b^2 = 0
    vz0 = (-z0 - 0.5 * g[2] * t_bounce_desired**2) / t_bounce_desired

    # 弹跳后的 Z 速度
    vz_at_bounce = vz0 + g[2] * t_bounce_desired
    vz_after_bounce = -vz_at_bounce * bounce_restitution

    # 弹跳后飞行时间
    t_after_bounce = hit_time - t_bounce_desired
    if t_after_bounce <= 0:
        t_after_bounce = 0.1

    # 弹跳后的 Z 位置
    p_hit_z = vz_after_bounce * t_after_bounce + 0.5 * g[2] * t_after_bounce**2

    # X/Y 速度：弹跳不改变 X/Y 速度，全程恒速
    # 需要球在 hit_time 时刻到达工作区域附近
    # 先采样目标 X（在工作空间内）
    offset_x = rng.uniform(0.15, 0.55)
    offset_y = rng.uniform(-0.25, 0.25)
    p_hit_x = shoulder_pos[0] + offset_x
    p_hit_y = shoulder_pos[1] + offset_y
    p_hit_z = max(p_hit_z, 0.4)  # 确保在地面以上
    p_hit_z = min(p_hit_z, shoulder_pos[2] + 0.5)  # 不超过肩上方太多

    vx0 = (p_hit_x - x0) / hit_time
    vy0 = (p_hit_y - y0) / hit_time

    p_hit = np.array([p_hit_x, p_hit_y, p_hit_z])
    v0 = np.array([vx0, vy0, vz0])

    return p0, v0, p_hit


def generate_hittable_ball(
    shoulder_pos: NDArray[np.floating],
    workspace_radius: float,
    hit_time: float,
    g: NDArray[np.floating],
    hit_offset_ranges: dict | None = None,
    rng: np.random.Generator | None = None,
    ball_direction: str = "x",
    ball_start_y_range: tuple[float, float] | None = None,
    ball_start_z_range: tuple[float, float] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """生成一条大概率落在工作空间内的网球轨迹。

    策略：先在工作空间内采样击打点，再反推球的初始状态。

    Args:
        shoulder_pos: 肩关节世界坐标位置，形状 (3,)。
        workspace_radius: 工作空间半径（米）。
        hit_time: 期望击打时间（秒）。
        g: 重力加速度，形状 (3,)。
        hit_offset_ranges: 击打点偏移范围字典，包含 x/y/z 的 [min, max]。
        rng: 随机数生成器。
        ball_direction: 球飞来的主方向，"x" 表示从 -X 方向飞来（默认），
            "y" 表示从 -Y 方向飞来（摄像头正对方向）。
        ball_start_y_range: 球初始 Y 坐标范围 (min, max)。不传则用默认值。
        ball_start_z_range: 球初始 Z 坐标范围 (min, max)。不传则用默认值。

    Returns:
        (p0, v0, p_hit): 球初始位置、初始速度、击打点位置。
    """
    if rng is None:
        rng = np.random.default_rng()

    if hit_offset_ranges is None:
        hit_offset_ranges = {
            "x": [0.05, 0.35],
            "y": [-0.25, 0.25],
            "z": [-0.2, 0.5],
        }

    # 在可达区域内采样击打点
    offset_x = rng.uniform(hit_offset_ranges["x"][0], hit_offset_ranges["x"][1])
    offset_y = rng.uniform(hit_offset_ranges["y"][0], hit_offset_ranges["y"][1])
    offset_z = rng.uniform(hit_offset_ranges["z"][0], hit_offset_ranges["z"][1])
    p_hit = shoulder_pos + np.array([offset_x, offset_y, offset_z])

    # 球的初始位置：在远处，沿主方向
    if ball_direction == "y":
        # 球从 -Y 方向飞来（摄像头正对方向）
        x0 = rng.uniform(-1.0, 1.0)
        y_range = ball_start_y_range if ball_start_y_range is not None else (-8.0, -3.0)
        y0 = rng.uniform(y_range[0], y_range[1])
    else:
        # 球从 -X 方向飞来（默认）
        x0 = rng.uniform(-8.0, -3.0)
        y0 = rng.uniform(-1.0, 1.0)
    z_range = ball_start_z_range if ball_start_z_range is not None else (1.5, 2.5)
    z0 = rng.uniform(z_range[0], z_range[1])
    p0 = np.array([x0, y0, z0])

    # 反推初速度：使球在 hit_time 时刻到达 p_hit
    # p_hit = p0 + v0 * t + 0.5 * g * t^2
    v0 = (p_hit - p0 - 0.5 * g * hit_time**2) / hit_time

    return p0, v0, p_hit


def generate_ball_to_target_box(
    target_center: NDArray[np.floating],
    target_offset: float,
    hit_time: float,
    g: NDArray[np.floating],
    shoulder_pos: NDArray[np.floating] | None = None,
    workspace_radius: float = 0.90,
    ball_speed: float | None = None,
    rng: np.random.Generator | None = None,
    ball_direction: str = "y",
    ball_start_y_range: tuple[float, float] | None = None,
    ball_start_z_range: tuple[float, float] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """生成一条落入目标立方体区域的网球轨迹。

    在 target_center 为中心、边长 2*target_offset 的立方体内随机采样击打点，
    然后反推球的初始位置和速度，使球在 hit_time 时刻到达该点。
    若提供 shoulder_pos，会约束采样点在工作空间内。
    若提供 ball_speed，则根据目标球速反推起始位置，而非随机采样。

    Args:
        target_center: 目标立方体中心，形状 (3,)。
        target_offset: 立方体半边长（各轴相同），例如 0.3 表示 ±0.3m。
        hit_time: 期望击打时间（秒）。
        g: 重力加速度，形状 (3,)。
        shoulder_pos: 肩关节世界坐标，用于工作空间约束。None 则不约束。
        workspace_radius: 工作空间半径（米）。
        ball_speed: 期望球到达击打点时的速度大小 (m/s)。
            若提供，根据此速度反推球的起始位置，使球速精确匹配。
            若为 None，则随机采样起始位置。
        rng: 随机数生成器。
        ball_direction: 球飞来的主方向，"x" 从 -X 方向飞来，"y" 从 -Y 方向飞来。
        ball_start_y_range: 球初始 Y 坐标范围 (min, max)。仅在 ball_speed=None 时使用。
        ball_start_z_range: 球初始 Z 坐标范围 (min, max)。仅在 ball_speed=None 时使用。

    Returns:
        (p0, v0, p_hit): 球初始位置、初始速度、击打点位置。
    """
    if rng is None:
        rng = np.random.default_rng()

    # 在目标立方体内采样，最多重试 50 次确保在工作空间内
    for _ in range(50):
        p_hit = target_center + rng.uniform(-target_offset, target_offset, size=3)
        p_hit[2] = max(p_hit[2], 0.3)

        if shoulder_pos is not None:
            dist = np.linalg.norm(p_hit - shoulder_pos)
            dz = p_hit[2] - shoulder_pos[2]
            if dist < workspace_radius and -0.60 < dz < 0.55:
                break
    else:
        # 全部重试失败，钳位到最近的工作空间内点
        if shoulder_pos is not None:
            direction = p_hit - shoulder_pos
            dist = np.linalg.norm(direction)
            if dist > workspace_radius * 0.95:
                p_hit = shoulder_pos + direction / dist * workspace_radius * 0.9
            dz = p_hit[2] - shoulder_pos[2]
            if dz <= -0.60:
                p_hit[2] = shoulder_pos[2] - 0.55
            elif dz >= 0.55:
                p_hit[2] = shoulder_pos[2] + 0.50
            p_hit[2] = max(p_hit[2], 0.3)

    if ball_speed is not None:
        # 根据目标球速反推起始位置
        # 球到达击打点时的速度 v_hit = v0 + g * hit_time
        # |v_hit| = ball_speed
        # 策略：让球沿主方向以 ball_speed 飞来，Z 方向速度由重力自然决定
        # 水平速度方向：从起始位置指向击打点
        if ball_direction == "y":
            vx_hit = rng.uniform(-0.3, 0.3)
            vy_hit = -ball_speed
        else:
            vx_hit = -ball_speed
            vy_hit = rng.uniform(-0.3, 0.3)

        # Z 方向速度由重力自由落地近似
        vz_hit = rng.uniform(-1.0, 1.0)

        # 击打时刻速度
        v_hit = np.array([vx_hit, vy_hit, vz_hit])
        v_hit[:2] = v_hit[:2] / np.linalg.norm(v_hit[:2]) * ball_speed
        v_hit[2] = vz_hit

        # 反推初速度：v0 = v_hit - g * hit_time
        v0 = v_hit - g * hit_time

        # 反推起始位置：p_hit = p0 + v0 * hit_time + 0.5 * g * hit_time^2
        p0 = p_hit - v0 * hit_time - 0.5 * g * hit_time**2
    else:
        # 球的初始位置：在远处，沿主方向
        if ball_direction == "y":
            x0 = rng.uniform(-1.0, 1.0)
            y_range = ball_start_y_range if ball_start_y_range is not None else (-5.5, -3.0)
            y0 = rng.uniform(y_range[0], y_range[1])
        else:
            x0 = rng.uniform(-8.0, -3.0)
            y0 = rng.uniform(-1.0, 1.0)
        z_range = ball_start_z_range if ball_start_z_range is not None else (1.4, 2.0)
        z0 = rng.uniform(z_range[0], z_range[1])
        p0 = np.array([x0, y0, z0])

        # 反推初速度：p_hit = p0 + v0 * t + 0.5 * g * t^2
        v0 = (p_hit - p0 - 0.5 * g * hit_time**2) / hit_time

    return p0, v0, p_hit


def generate_unreachable_ball(
    shoulder_pos: NDArray[np.floating],
    workspace_radius: float,
    hit_time: float,
    g: NDArray[np.floating],
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """生成一条不在工作空间内的网球轨迹。

    Args:
        shoulder_pos: 肩关节世界坐标位置，形状 (3,)。
        workspace_radius: 工作空间半径（米）。
        hit_time: 期望时间（秒）。
        g: 重力加速度，形状 (3,)。
        rng: 随机数生成器。

    Returns:
        (p0, v0): 球初始位置和初始速度。
    """
    if rng is None:
        rng = np.random.default_rng()

    # 球飞向远离机器人的方向
    choice = rng.integers(0, 3)
    if choice == 0:
        # 飞向机器人侧面远处
        p0 = np.array([rng.uniform(-3, -1), rng.uniform(-2, -1.5), rng.uniform(1.5, 2.5)])
        v0 = np.array([rng.uniform(1, 3), rng.uniform(-3, -1), rng.uniform(-2, 0)])
    elif choice == 1:
        # 飞过高
        p0 = np.array([rng.uniform(-3, -1), rng.uniform(-0.5, 0.5), rng.uniform(2, 3)])
        v0 = np.array([rng.uniform(2, 5), rng.uniform(-1, 1), rng.uniform(2, 5)])
    else:
        # 飞过头顶
        p0 = np.array([rng.uniform(-3, -1), rng.uniform(-0.5, 0.5), rng.uniform(1.5, 2.5)])
        v0 = np.array([rng.uniform(3, 6), rng.uniform(-0.5, 0.5), rng.uniform(3, 6)])

    return p0, v0

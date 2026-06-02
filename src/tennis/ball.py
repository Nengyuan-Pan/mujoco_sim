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
    ball_distance: float | None = None,
    approach_angle_deg: float = 0.0,
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
    若提供 ball_distance，则根据指定距离反推起始位置，精确控制球飞来的远近。

    Args:
        target_center: 目标立方体中心，形状 (3,)。
        target_offset: 立方体半边长（各轴相同），例如 0.3 表示 ±0.3m。
        hit_time: 期望击打时间（秒）。
        g: 重力加速度，形状 (3,)。
        shoulder_pos: 肩关节世界坐标，用于工作空间约束。None 则不约束。
        workspace_radius: 工作空间半径（米）。
        ball_speed: 期望球到达击打点时的水平速度大小 (m/s)。
            若提供，根据此速度反推球的起始位置，使球速精确匹配。
            若为 None，则随机采样起始位置。
        ball_distance: 球起始位置到击打点的直线距离 (m)。
            若提供，球从距击打点刚好此距离处出发，配合 approach_angle_deg 控制方向。
            若同时提供 ball_speed，hit_time 会被自动调整以保持物理一致性
            （expected_hit_time = ball_distance / ball_speed）。
            若为 None，则使用原有随机逻辑。
        approach_angle_deg: 球飞来方向角（度），仅在 ball_distance 非 None 时生效。
            0° 表示球从 -Y 方向（正前方）飞来，90° 表示从 -X 方向（右侧）飞来。
        rng: 随机数生成器。
        ball_direction: 球飞来的主方向，"x" 从 -X 方向飞来，"y" 从 -Y 方向飞来。
        ball_start_y_range: 球初始 Y 坐标范围 (min, max)。仅在未使用 ball_distance 且 ball_speed=None 时使用。
        ball_start_z_range: 球初始 Z 坐标范围 (min, max)。仅在未使用 ball_distance 且 ball_speed=None 时使用。

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

    if ball_distance is not None:
        # ── 基于距离参数反推起始位置 ──
        # 计算有效飞行时间：若同时提供 ball_speed，物理一致性要求
        # hit_time ≈ ball_distance / ball_speed，不匹配时以 ball_distance 为准调整
        effective_hit_time = hit_time
        if ball_speed is not None:
            expected_hit_time = ball_distance / ball_speed
            if abs(expected_hit_time - hit_time) / max(hit_time, 0.001) > 0.5:
                # 飞行时间差距过大，以 ball_distance/ball_speed 为准
                effective_hit_time = expected_hit_time
            else:
                # 差距在合理范围，保持原始 hit_time，微调 ball_speed 警告
                effective_hit_time = hit_time

        # 接近方向单位向量（水平面内）
        angle_rad = np.deg2rad(approach_angle_deg)
        approach_dir_xy = np.array([np.sin(angle_rad), np.cos(angle_rad)], dtype=np.float64)

        # 水平初始位置：从击打点沿反方向退 ball_distance
        p0_xy = p_hit[:2] - approach_dir_xy * ball_distance

        # 水平初速度（匀速）：p_hit_xy = p0_xy + v0_xy * effective_hit_time
        v0_xy = (p_hit[:2] - p0_xy) / effective_hit_time
        v_hit_xy = v0_xy.copy()  # 水平速度恒定

        # Z 分量：物理一致性推导 — 设定合理的 vz_hit，反推 p0_z
        # 球到达击打点时 Z 速度接近 0（抛物线顶端附近经过），保证轨迹自然
        if ball_speed is not None:
            vz_hit = rng.uniform(-1.0, 1.0)
        else:
            vz_hit = rng.uniform(-2.0, 2.0)
        v0_z = vz_hit - g[2] * effective_hit_time
        p0_z = p_hit[2] - v0_z * effective_hit_time - 0.5 * g[2] * effective_hit_time**2
        # 钳位到合理发球高度范围（0.3m ~ 5m）
        if p0_z < 0.3 or p0_z > 5.0:
            p0_z = float(np.clip(p0_z, 0.3, 5.0))
            v0_z = (p_hit[2] - p0_z - 0.5 * g[2] * effective_hit_time**2) / effective_hit_time

        p0 = np.array([p0_xy[0], p0_xy[1], p0_z], dtype=np.float64)
        v0 = np.array([v0_xy[0], v0_xy[1], v0_z], dtype=np.float64)

        # 水平球速 = 距离 / 时间（仅用于日志参考，不作为返回值）
        implied_speed = float(np.linalg.norm(v_hit_xy))
        if ball_speed is not None and abs(implied_speed - ball_speed) / max(ball_speed, 0.001) > 0.3:
            import logging
            _log = logging.getLogger(__name__)
            _log.warning(
                f"ball_speed={ball_speed}m/s 与 ball_distance/effective_hit_time="
                f"{implied_speed:.1f}m/s 偏差过大 (effective_hit_time={effective_hit_time:.3f}s)，"
                f"实际水平球速 ≈ {implied_speed:.1f} m/s"
            )
        return p0, v0, p_hit

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


def generate_ball_from_serve_box(
    serve_box_center: tuple[float, float, float] = (0.0, -8.0, 1.2),
    serve_box_halfsize: tuple[float, float, float] = (4.0, 0.1, 0.15),
    target_center: NDArray[np.floating] | None = None,
    target_offset: float = 0.0,
    shoulder_pos: NDArray[np.floating] | None = None,
    workspace_radius: float = 0.90,
    g: NDArray[np.floating] | None = None,
    ball_speed: float | None = None,
    speed_range: tuple[float, float] = (8.0, 18.0),
    use_bounce: bool = True,
    bounce_restitution: float = 0.75,
    bounce_friction: float = 0.95,
    rng: np.random.Generator | None = None,
    max_retries: int = 500,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """从长方体发球区生成网球轨迹，保证球经过右臂工作区。

    在发球长方体区域内均匀采样球初始位置 p0，在工作区目标盒内采样击打点 p_hit，
    反推初始速度 v0 使得球在受重力（和可选地面弹跳）作用下到达击打点。

    物理推导：
      - 水平面：匀速运动，弹跳后 XY 速度乘 friction（0.95）减速。
        d_xy = ball_speed * (t_before + friction * t_after)
      - Z 方向：抛物线运动，含一次可选地面弹跳。
        触地时 vz 反向并乘恢复系数。
      - 弹跳点 XY = p0_xy + v_hat * ball_speed * t_before
      - 弹跳后 v_xy = v_hat * ball_speed * friction

    Args:
        serve_box_center: 发球区中心坐标 (x, y, z)，默认 (0, -8, 1.2)。
        serve_box_halfsize: 发球区半尺寸 (hx, hy, hz)，默认 (4, 0.1, 0.15)。
            X 轴 ±4m 跨球场横向，Y 轴 ±0.1m 纵深，Z 轴 ±0.15m 高度。
        target_center: 工作区目标中心，形状 (3,)。默认 [-0.83, -0.47, 0.87]。
        target_offset: 目标盒半边长（米），默认 0（精确点）。
        shoulder_pos: 肩关节世界坐标，形状 (3,)。用于约束采样点在工作空间内。
        workspace_radius: 工作空间半径（米），默认 0.90。
        g: 重力加速度，形状 (3,)。默认 [0, 0, -9.81]。
        ball_speed: 指定水平球速 (m/s)。若为 None，从 speed_range 随机选取。
        speed_range: 球速范围 (min, max)，默认 (8, 18) m/s。
        use_bounce: 是否使用地面弹跳模型。True 时可降低最低可行球速。
        bounce_restitution: 弹跳恢复系数，默认 0.75。
        rng: 随机数生成器。
        max_retries: 最大重试次数。

    Returns:
        (p0, v0, p_hit): 球初始位置、初始速度、击打点位置。

    Raises:
        RuntimeError: 超过 max_retries 仍未找到可行轨迹。
    """
    import logging
    _log = logging.getLogger(__name__)

    if rng is None:
        rng = np.random.default_rng()
    if g is None:
        g_arr = np.array([0.0, 0.0, -9.81], dtype=np.float64)
    else:
        g_arr = np.asarray(g, dtype=np.float64)
    if target_center is None:
        tgt = np.array([-0.83, -0.47, 0.87], dtype=np.float64)
    else:
        tgt = np.asarray(target_center, dtype=np.float64)
    if shoulder_pos is None:
        sh_pos = np.array([-0.1, -0.23, 1.30], dtype=np.float64)
    else:
        sh_pos = np.asarray(shoulder_pos, dtype=np.float64)

    center = np.array(serve_box_center, dtype=np.float64)
    half = np.array(serve_box_halfsize, dtype=np.float64)

    gz = float(g_arr[2])
    min_speed, max_speed = speed_range
    if ball_speed is not None:
        min_speed = max_speed = ball_speed

    for attempt in range(max_retries):
        # 1. 选择球速
        spd = rng.uniform(min_speed, max_speed)

        # 2. 在工作区目标盒采样 p_hit
        p_hit = tgt + rng.uniform(-target_offset, target_offset, size=3)
        p_hit[2] = max(p_hit[2], 0.3)

        # 工作空间约束
        if shoulder_pos is not None:
            d_to_shoulder = np.linalg.norm(p_hit - sh_pos)
            dz = p_hit[2] - sh_pos[2]
            if d_to_shoulder >= workspace_radius or dz <= -0.60 or dz >= 0.55:
                continue

        # 3. 采样发球区 XY
        p0_xy = center[:2] + rng.uniform(-half[:2], half[:2])

        # 4. 水平动力学
        d_xy_vec = p_hit[:2] - p0_xy
        d_xy = float(np.linalg.norm(d_xy_vec))
        if d_xy < 0.5:
            continue
        v_hat_xy = d_xy_vec / d_xy

        if use_bounce:
            # ── 弹跳模式：二分搜索 t2（反弹后飞行时间），使弹跳物理自洽 ──
            # 球轨迹: 发球区→地面弹跳(restitution=0.75, friction=0.95)→工作区
            # 约束方程组 (BALL_RADIUS=0.033):
            #   水平: d_xy = spd·t1 + spd·friction·t2  →  t1 = t_total - friction·t2
            #   反弹后Z: p_hit_z = BR + vz_after·t2 + ½gz·t2²  →  vz_after
            #   弹跳: vz_before = -vz_after / restitution  (反弹前向下)
            #   反弹前Z: BR = p0_z + v0_z·t1 + ½gz·t1²  →  p0_z_req
            #   速度连续: v0_z = vz_before - gz·t1
            # 目标: 找到 t2 使 p0_z_req 落在发球区 Z 范围内
            BR = 0.033  # BALL_RADIUS
            t_total = d_xy / spd
            if t_total < 0.15 or t_total > 2.0:
                continue

            # t2 范围：不能小于 0.05s，不能使 t1 为负
            t2_max = min(0.60, (t_total - 0.02) / bounce_friction)
            t2_min = 0.05
            if t2_max <= t2_min:
                continue

            # 均匀采样 t2，搜索使 p0_z_req 落在发球区范围内的值
            z_lo = center[2] - half[2]
            z_hi = center[2] + half[2]
            best_t2 = None
            best_p0_z_req = None
            best_v0_z = None
            best_p0_z_err = float("inf")

            for _ in range(60):
                t2 = rng.uniform(t2_min, t2_max)
                t1 = t_total - bounce_friction * t2
                if t1 <= 0.02:
                    continue

                # 反弹后 Z 速度
                # p_hit_z = BR + vz_after·t2 + 0.5·gz·t2²
                vz_after = (p_hit[2] - BR - 0.5 * gz * t2 ** 2) / t2
                if vz_after <= 0.5:  # 必须向上且有足够速度
                    continue

                # 反弹前 Z 速度（向下为负）
                vz_before = -vz_after / bounce_restitution
                if vz_before >= 0.0:
                    continue

                # 反弹前初速度与高度
                v0_z = vz_before - gz * t1
                # BR = p0_z + v0_z·t1 + 0.5·gz·t1²
                p0_z_req = BR - v0_z * t1 - 0.5 * gz * t1 ** 2

                # 检查 p0_z_req 合理性
                if p0_z_req < 0.2 or p0_z_req > 4.0:
                    continue

                # v0_z 必须为负（球初始向下运动，才能击中地面）
                if v0_z >= 0.2:
                    continue

                # 距离发球区中心 Z 的误差
                z_err = abs(p0_z_req - center[2])
                if z_err < best_p0_z_err:
                    best_p0_z_err = z_err
                    best_t2 = t2
                    best_p0_z_req = p0_z_req
                    best_v0_z = v0_z

            # 检查找到的最佳值是否可接受
            if best_p0_z_req is not None and z_lo - 1.5 <= best_p0_z_req <= z_hi + 1.5:
                t1_best = t_total - bounce_friction * best_t2
                p0 = np.array([p0_xy[0], p0_xy[1], best_p0_z_req], dtype=np.float64)
                v0 = np.array([v_hat_xy[0] * spd, v_hat_xy[1] * spd, best_v0_z], dtype=np.float64)

                # MuJoCo 预测验证
                try:
                    from src.sim.rm65_env import RM65Env
                    from pathlib import Path
                    _env = RM65Env(
                        Path(__file__).resolve().parent.parent / "src" / "robot" / "rm65_model.xml",
                        dt=0.005,
                    )
                    _env.reset(np.zeros(6))
                    _env.set_ball_state(p0.copy(), v0.copy())
                    pred_pos, _ = _env.predict_ball_trajectory(
                        p0.copy(), v0.copy(), int(t_total / 0.005) + 50
                    )
                    found = any(
                        float(np.linalg.norm(pred_pos[k] - sh_pos)) < workspace_radius
                        and pred_pos[k, 2] > 0.3
                        and -0.60 < (pred_pos[k, 2] - sh_pos[2]) < 0.55
                        for k in range(len(pred_pos))
                    )
                    _env.close() if hasattr(_env, 'close') else None
                    if not found:
                        continue
                except Exception:
                    pass

                _log.debug(
                    f"serve_box bounce (attempt={attempt}): p0_z={best_p0_z_req:.2f}, "
                    f"vz0={best_v0_z:.1f}, spd={spd:.1f}, t1={t1_best:.3f}, t2={best_t2:.3f}, "
                    f"z_err={best_p0_z_err:.2f}"
                )
                return p0, v0, p_hit
        else:
            # ── 无弹跳：直接反解 v0 ──
            # p_hit = p0 + v0*t + 0.5*g*t² → v0 = (p_hit - p0 - 0.5*g*t²) / t
            p0_z = center[2] + rng.uniform(-half[2], half[2])
            t_total = d_xy / spd
            if t_total < 0.1 or t_total > 2.0:
                continue

            v0_z = (p_hit[2] - p0_z - 0.5 * gz * t_total ** 2) / t_total
            # 合理性检查：v0_z 不能太极端
            if abs(v0_z) > 25.0:
                continue

            p0 = np.array([p0_xy[0], p0_xy[1], p0_z], dtype=np.float64)
            v0 = np.array([v_hat_xy[0] * spd, v_hat_xy[1] * spd, v0_z], dtype=np.float64)
            _log.debug(
                f"serve_box direct OK (attempt={attempt}): p0={np.round(p0,2)}, "
                f"spd={spd:.1f}m/s, t_total={t_total:.3f}s"
            )
            return p0, v0, p_hit

    raise RuntimeError(
        f"generate_ball_from_serve_box: 在 {max_retries} 次重试后未找到可行轨迹。"
        f"speed_range={speed_range}, bounce={use_bounce}"
    )


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


def generate_bounce_ball(
    shoulder_pos: NDArray[np.floating],
    workspace_radius: float,
    hit_time: float,
    g: NDArray[np.floating],
    serve_distance: float = 20.0,
    bounce_restitution: float = 0.75,
    target_speed_range: tuple[float, float] = (16.0, 22.0),
    hit_offset_ranges: dict | None = None,
    rng: np.random.Generator | None = None,
    ball_direction: str = "y",
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """生成从远处飞来、在地面反弹一次后到达击打区域的网球轨迹。

    球从约 20m 远处发出，飞行过程中触地弹跳一次（弹跳恢复系数 0.75），
    弹起后继续飞向机器人工作区域，到达时速度在 target_speed_range 范围内。

    算法与 ball_trajectory_with_bounce / ball_velocity_with_bounce 完全一致：
    弹跳时仅 Z 速度反转乘恢复系数，XY 速度不变。

    反推流程：
    1. 在工作空间内采样击打点 p_hit
    2. 设定弹跳时刻 t_bounce（总飞行时间的前 40%-60%）
    3. 反推弹跳后的速度 v_after（使球从弹跳点飞到 p_hit）
    4. 反推弹跳前的速度 v_before（Z 反转除以恢复系数，XY 不变）
    5. 反推初始速度 v0 = v_before - g * t_bounce
    6. 反推初始位置 p0 = p_bounce - v0 * t_bounce - 0.5 * g * t_bounce^2

    Args:
        shoulder_pos: 肩关节世界坐标位置，形状 (3,)。
        workspace_radius: 工作空间半径（米）。
        hit_time: 期望从发球到击打的总飞行时间（秒）。
        g: 重力加速度，形状 (3,)，通常 [0, 0, -9.81]。
        serve_distance: 发球距离（米），球从 -Y 或 -X 方向约此距离处发出。
        bounce_restitution: 弹跳恢复系数，默认 0.75。
        target_speed_range: 球到达击打区时的速度范围 (min, max) m/s。
        hit_offset_ranges: 击打点相对肩关节的偏移范围字典。
        rng: 随机数生成器。
        ball_direction: 球飞来的主方向，"y" 从 -Y 飞来，"x" 从 -X 飞来。

    Returns:
        (p0, v0, p_hit): 球初始位置、初始速度、击打点位置。
    """
    if rng is None:
        rng = np.random.default_rng()

    if hit_offset_ranges is None:
        hit_offset_ranges = {
            "x": [0.05, 0.45],
            "y": [-0.30, 0.30],
            "z": [-0.15, 0.40],
        }

    # 1. 在工作空间内采样击打点
    offset_x = rng.uniform(hit_offset_ranges["x"][0], hit_offset_ranges["x"][1])
    offset_y = rng.uniform(hit_offset_ranges["y"][0], hit_offset_ranges["y"][1])
    offset_z = rng.uniform(hit_offset_ranges["z"][0], hit_offset_ranges["z"][1])
    p_hit = shoulder_pos + np.array([offset_x, offset_y, offset_z])
    p_hit[2] = max(p_hit[2], 0.4)
    p_hit[2] = min(p_hit[2], shoulder_pos[2] + 0.45)

    # 2. 弹跳时刻：总飞行时间的前 40%-60%
    t_bounce_ratio = rng.uniform(0.40, 0.60)
    t_bounce = hit_time * t_bounce_ratio
    t_after_bounce = hit_time - t_bounce

    # 3. 反推弹跳后的速度 v_after
    # p_hit = p_bounce + v_after * t_after + 0.5 * g * t_after^2
    # 弹跳点 z ≈ 0（地面）
    # p_hit_z = 0 + vz_after * t_after + 0.5 * g[2] * t_after^2
    vz_after = (p_hit[2] - 0.5 * g[2] * t_after_bounce**2) / t_after_bounce
    vz_after = max(vz_after, 1.0)

    # 目标球速
    target_speed = rng.uniform(target_speed_range[0], target_speed_range[1])
    speed_horizontal = np.sqrt(max(target_speed**2 - vz_after**2, 0.0))

    if ball_direction == "y":
        vx_after = rng.uniform(-0.5, 0.5)
        vy_after = speed_horizontal
    else:
        vx_after = speed_horizontal
        vy_after = rng.uniform(-0.5, 0.5)

    # 归一化水平分量
    vh = np.sqrt(vx_after**2 + vy_after**2)
    if vh > 1e-6:
        vx_after *= speed_horizontal / vh
        vy_after *= speed_horizontal / vh

    v_after = np.array([vx_after, vy_after, vz_after])

    # 弹跳点水平位置
    p_bounce_x = p_hit[0] - vx_after * t_after_bounce
    p_bounce_y = p_hit[1] - vy_after * t_after_bounce

    # 4. 反推弹跳前速度（与解析模型一致：XY不变，Z反转除恢复系数）
    vz_before = -vz_after / bounce_restitution
    v_before = np.array([vx_after, vy_after, vz_before])

    # 5. 反推初始速度 v0 = v_before - g * t_bounce
    v0 = v_before - g * t_bounce

    # 6. 反推初始位置 p0 = p_bounce - v0 * t_bounce - 0.5 * g * t_bounce^2
    p0 = np.array([
        p_bounce_x - v0[0] * t_bounce - 0.5 * g[0] * t_bounce**2,
        p_bounce_y - v0[1] * t_bounce - 0.5 * g[1] * t_bounce**2,
        0.0 - v0[2] * t_bounce - 0.5 * g[2] * t_bounce**2,
    ])

    # 确保初始位置在合理范围
    p0[2] = max(p0[2], 0.5)
    p0[2] = min(p0[2], 4.0)

    return p0, v0, p_hit

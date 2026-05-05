"""击打点计算与工作空间可达性判定（含弹跳支持）。"""

import numpy as np
from numpy.typing import NDArray

from src.tennis.ball import (
    ball_trajectory,
    ball_velocity,
    ball_trajectory_with_bounce,
    ball_velocity_with_bounce,
)


def find_hitting_point(
    p0: NDArray[np.floating],
    v0: NDArray[np.floating],
    g: NDArray[np.floating],
    shoulder_pos: NDArray[np.floating],
    workspace_radius: float,
    dt: float,
    horizon: int,
    use_bounce: bool = False,
    bounce_restitution: float = 0.75,
) -> dict | None:
    """在规划时间窗口内寻找最佳击打点（解析模型）。

    遍历所有时间步，找到球距肩关节最近且在工作空间内的时刻。

    Args:
        p0: 球初始位置，形状 (3,)。
        v0: 球初始速度，形状 (3,)。
        g: 重力加速度，形状 (3,)。
        shoulder_pos: 肩关节世界坐标，形状 (3,)。
        workspace_radius: 工作空间半径（米）。
        dt: 时间步长（秒）。
        horizon: 规划步数。
        use_bounce: 是否使用弹跳模型。
        bounce_restitution: 弹跳恢复系数。

    Returns:
        若可达，返回字典：
            {
                "t_hit": 击打时间（秒）,
                "k_hit": 击打步数,
                "p_hit": 击打位置 (3,),
                "v_ball_hit": 击打时刻球速 (3,),
                "dist": 球到肩的距离,
            }
        若不可达，返回 None。
    """
    best_k = None
    best_score = np.inf
    best_p = None
    best_v_ball = None
    best_dist = np.inf

    for k in range(1, horizon + 1):
        t = k * dt
        if use_bounce:
            p_ball = ball_trajectory_with_bounce(p0, v0, g, t, bounce_restitution)
            v_ball = ball_velocity_with_bounce(p0, v0, g, t, bounce_restitution)
        else:
            p_ball = ball_trajectory(p0, v0, g, t)
            v_ball = ball_velocity(v0, g, t)
        dist = np.linalg.norm(p_ball - shoulder_pos)

        # 球在工作空间内且在地面上方，且高度在肩关节附近
        dz = p_ball[2] - shoulder_pos[2]
        if dist < workspace_radius and p_ball[2] > 0.3 and -0.60 < dz < 0.55:
            # 可达性评分：距离越近越好，偏好前方
            height_above = max(0.0, p_ball[2] - shoulder_pos[2] - 0.2)
            height_penalty = height_above ** 2 * 5.0
            front_bonus = max(0.0, p_ball[0] - shoulder_pos[0]) * 0.3
            score = dist + height_penalty - front_bonus
            if score < best_score:
                best_score = score
                best_dist = dist
                best_k = k
                best_p = p_ball.copy()
                best_v_ball = v_ball.copy()

    if best_k is None:
        return None

    return {
        "t_hit": best_k * dt,
        "k_hit": best_k,
        "p_hit": best_p,
        "v_ball_hit": best_v_ball,
        "dist": best_dist,
    }


def find_hitting_point_physics(
    env,
    ball_pos: NDArray[np.floating],
    ball_vel: NDArray[np.floating],
    shoulder_pos: NDArray[np.floating],
    workspace_radius: float,
    horizon: int,
) -> dict | None:
    """在规划时间窗口内寻找最佳击打点（MuJoCo 物理仿真）。

    使用 MuJoCo 物理引擎前向仿真球的运动轨迹，比解析模型更真实。

    Args:
        env: MuJoCo 环境实例。
        ball_pos: 球当前位置，形状 (3,)。
        ball_vel: 球当前速度，形状 (3,)。
        shoulder_pos: 肩关节世界坐标，形状 (3,)。
        workspace_radius: 工作空间半径（米）。
        horizon: 规划步数。

    Returns:
        若可达，返回字典（同 find_hitting_point）；若不可达，返回 None。
    """
    ball_positions, ball_velocities = env.predict_ball_trajectory(
        ball_pos, ball_vel, horizon
    )

    best_k = None
    best_score = np.inf
    best_p = None
    best_v_ball = None
    best_dist = np.inf

    for k in range(horizon):
        p_ball = ball_positions[k]
        v_ball = ball_velocities[k]
        dist = np.linalg.norm(p_ball - shoulder_pos)

        dz = p_ball[2] - shoulder_pos[2]
        if dist < workspace_radius and p_ball[2] > 0.3 and -0.60 < dz < 0.55:
            height_above = max(0.0, p_ball[2] - shoulder_pos[2] - 0.2)
            height_penalty = height_above ** 2 * 5.0
            front_bonus = max(0.0, p_ball[0] - shoulder_pos[0]) * 0.3
            score = dist + height_penalty - front_bonus
            if score < best_score:
                best_score = score
                best_dist = dist
                best_k = k + 1
                best_p = p_ball.copy()
                best_v_ball = v_ball.copy()

    if best_k is None:
        return None

    return {
        "t_hit": best_k * env.dt,
        "k_hit": best_k,
        "p_hit": best_p,
        "v_ball_hit": best_v_ball,
        "dist": best_dist,
    }


def compute_desired_hit_velocity(
    hit_direction: NDArray[np.floating],
    racket_speed: float,
) -> NDArray[np.floating]:
    """计算期望的球拍击打速度。

    Args:
        hit_direction: 期望击打方向，形状 (3,)。
        racket_speed: 期望击打球速（米/秒）。

    Returns:
        期望的末端执行器速度，形状 (3,)。
    """
    d = hit_direction / (np.linalg.norm(hit_direction) + 1e-8)
    return d * racket_speed


def is_reachable(
    p0: NDArray[np.floating],
    v0: NDArray[np.floating],
    g: NDArray[np.floating],
    shoulder_pos: NDArray[np.floating],
    workspace_radius: float,
    dt: float,
    horizon: int,
    use_bounce: bool = False,
    bounce_restitution: float = 0.75,
) -> bool:
    """快速判断球是否在工作空间内可达。

    Args:
        参数同 find_hitting_point。

    Returns:
        True 如果球在规划窗口内经过工作空间。
    """
    for k in range(1, horizon + 1):
        t = k * dt
        if use_bounce:
            p_ball = ball_trajectory_with_bounce(p0, v0, g, t, bounce_restitution)
        else:
            p_ball = ball_trajectory(p0, v0, g, t)
        dist = np.linalg.norm(p_ball - shoulder_pos)
        if dist < workspace_radius and p_ball[2] > 0.3:
            return True
    return False


def schedule_weights(
    t_remaining: float,
    t_total: float,
) -> tuple[float, float]:
    """根据剩余时间计算代价权重缩放因子。

    远离击打时刻时 Q_p 权重大（精确到达位置），
    接近击打时刻时 Q_v 权重增大（精确匹配速度）。

    Args:
        t_remaining: 剩余时间（秒）。
        t_total: 总规划时间（秒）。

    Returns:
        (Q_p_scale, Q_v_scale): 位置和速度权重缩放因子。
    """
    if t_total < 1e-6:
        return 1.0, 1.0
    ratio = t_remaining / t_total
    # Q_p: 随接近从 1.0 渐降到 0.5
    Q_p_scale = 0.5 + 0.5 * ratio
    # Q_v: 随接近从 0.5 渐升到 2.0
    Q_v_scale = 2.0 - 1.5 * ratio
    return Q_p_scale, Q_v_scale

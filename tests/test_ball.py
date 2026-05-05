"""网球弹跳轨迹测试。"""

import numpy as np

from src.tennis.ball import (
    ball_trajectory,
    ball_velocity,
    ball_trajectory_with_bounce,
    ball_velocity_with_bounce,
    compute_bounce_time,
    generate_serve_ball,
)


class TestBallTrajectory:
    """测试基本抛物线轨迹。"""

    def test_trajectory_at_t0(self) -> None:
        """t=0 时位置等于初始位置。"""
        p0 = np.array([1.0, 2.0, 3.0])
        v0 = np.array([1.0, 0.0, 0.0])
        g = np.array([0.0, 0.0, -9.81])
        p = ball_trajectory(p0, v0, g, 0.0)
        np.testing.assert_allclose(p, p0)

    def test_velocity_at_t0(self) -> None:
        """t=0 时速度等于初始速度。"""
        v0 = np.array([5.0, 3.0, 2.0])
        g = np.array([0.0, 0.0, -9.81])
        v = ball_velocity(v0, g, 0.0)
        np.testing.assert_allclose(v, v0)

    def test_free_fall(self) -> None:
        """自由落体：从静止下落 1 秒。"""
        p0 = np.array([0.0, 0.0, 10.0])
        v0 = np.zeros(3)
        g = np.array([0.0, 0.0, -9.81])
        p = ball_trajectory(p0, v0, g, 1.0)
        np.testing.assert_allclose(p[2], 10.0 - 0.5 * 9.81, atol=1e-10)


class TestBounceTrajectory:
    """测试含弹跳的轨迹。"""

    def test_bounce_time(self) -> None:
        """计算弹跳时刻。"""
        p0 = np.array([0.0, 0.0, 2.8])
        v0 = np.array([22.0, 0.0, -6.28])
        g = np.array([0.0, 0.0, -9.81])
        t_bounce = compute_bounce_time(p0, v0, g)
        assert t_bounce is not None
        assert 0.3 < t_bounce < 0.4

    def test_bounce_position_near_ground(self) -> None:
        """弹跳时刻球的位置接近地面。"""
        p0 = np.array([0.0, 0.0, 2.8])
        v0 = np.array([22.0, 0.0, -6.28])
        g = np.array([0.0, 0.0, -9.81])
        t_bounce = compute_bounce_time(p0, v0, g)
        assert t_bounce is not None
        p = ball_trajectory_with_bounce(p0, v0, g, t_bounce, 0.75)
        assert abs(p[2]) < 0.01

    def test_no_bounce_when_high(self) -> None:
        """球在地下且向下飞时不会弹跳到地面。"""
        g = np.array([0.0, 0.0, -9.81])
        p0_neg = np.array([0.0, 0.0, -1.0])
        v0_neg = np.array([0.0, 0.0, -1.0])
        t_bounce_neg = compute_bounce_time(p0_neg, v0_neg, g)
        assert t_bounce_neg is None  # 球在地下且向下飞，不会弹到地面

    def test_bounce_preserves_xy_speed(self) -> None:
        """弹跳不改变 X/Y 速度。"""
        p0 = np.array([-10.0, 0.0, 2.0])
        v0 = np.array([20.0, 1.0, -5.0])
        g = np.array([0.0, 0.0, -9.81])
        t_bounce = compute_bounce_time(p0, v0, g)
        assert t_bounce is not None
        # 弹跳后的速度
        v_after = ball_velocity_with_bounce(p0, v0, g, t_bounce + 0.01, 0.75)
        np.testing.assert_allclose(v_after[0], v0[0], atol=1e-6)
        np.testing.assert_allclose(v_after[1], v0[1], atol=1e-6)
        # Z 速度应该为正（弹起）
        assert v_after[2] > 0


class TestServeBall:
    """测试发球生成。"""

    def test_serve_ball_reachable(self) -> None:
        """生成的发球应在工作空间内可达。"""
        from src.tennis.hitting import find_hitting_point

        g = np.array([0.0, 0.0, -9.81])
        shoulder_pos = np.array([0.0, -0.15, 1.163])
        ws = 0.85
        dt = 0.005
        horizon = 200

        rng = np.random.default_rng(42)
        p0, v0, p_hit = generate_serve_ball(
            shoulder_pos, ws, g, 0.8,
            serve_distance=22.0,
            serve_height_range=(2.5, 3.0),
            bounce_restitution=0.75,
            rng=rng,
        )

        hit_info = find_hitting_point(
            p0, v0, g, shoulder_pos, ws, dt, horizon,
            use_bounce=True, bounce_restitution=0.75,
        )
        assert hit_info is not None
        assert hit_info["k_hit"] > 0

    def test_serve_ball_start_x(self) -> None:
        """发球起始 X 应约 -22m。"""
        g = np.array([0.0, 0.0, -9.81])
        shoulder_pos = np.array([0.0, -0.15, 1.163])
        rng = np.random.default_rng(7)

        p0, v0, _ = generate_serve_ball(
            shoulder_pos, 0.85, g, 0.8,
            serve_distance=22.0,
            rng=rng,
        )
        assert -25 < p0[0] < -19

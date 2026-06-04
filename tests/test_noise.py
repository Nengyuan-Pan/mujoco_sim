"""噪声注入功能测试。"""

import numpy as np
import pytest

from src.utils.noise import add_observation_noise, add_torque_noise, randomize_init_q


class TestObservationNoiseZeroStd:
    """std=0 时噪声函数不应修改输入。"""

    def test_pos_unchanged_when_zero_std(self) -> None:
        """pos_std=0 时球位置应原样返回。"""
        rng = np.random.default_rng(42)
        ball_pos = np.array([0.5, -0.3, 1.2])
        ball_vel = np.array([9.0, 0.1, -2.0])

        pos_out, vel_out = add_observation_noise(
            ball_pos, ball_vel, rng, pos_std=0.0, vel_std=0.0,
        )

        np.testing.assert_array_equal(pos_out, ball_pos)
        np.testing.assert_array_equal(vel_out, ball_vel)

    def test_only_pos_noise_when_vel_std_zero(self) -> None:
        """pos_std>0 且 vel_std=0 时仅位置有噪声，速度不变。"""
        rng = np.random.default_rng(42)
        ball_pos = np.array([0.5, -0.3, 1.2])
        ball_vel = np.array([9.0, 0.1, -2.0])

        pos_out, vel_out = add_observation_noise(
            ball_pos, ball_vel, rng, pos_std=0.02, vel_std=0.0,
        )

        np.testing.assert_array_equal(vel_out, ball_vel)
        assert not np.allclose(pos_out, ball_pos, atol=1e-10)


class TestTorqueNoiseZeroStd:
    """torque_std=0 时力矩不应被修改。"""

    def test_torque_unchanged_when_zero_std(self) -> None:
        """torque_std=0 时控制力矩应原样返回。"""
        rng = np.random.default_rng(42)
        u = np.array([10.0, -20.0, 15.0, -5.0, 2.0, -1.0])

        u_out = add_torque_noise(u, rng, torque_std=0.0)

        np.testing.assert_array_equal(u_out, u)


class TestInitQZeroNoise:
    """noise_rad=0 时初始角度不应被修改。"""

    def test_init_q_unchanged_when_zero_noise(self) -> None:
        """noise_rad=0 时初始关节角度应原样返回。"""
        rng = np.random.default_rng(42)
        init_q = np.array([-1.5, 1.57, -0.236, 0.404, 0.446, 2.45])

        q_out = randomize_init_q(init_q, rng, noise_rad=0.0)

        np.testing.assert_array_equal(q_out, init_q)


class TestObservationNoiseStatistics:
    """观测噪声的统计特性。"""

    def test_pos_noise_mean_near_zero(self) -> None:
        """大量样本的位置噪声均值应接近零。"""
        rng = np.random.default_rng(42)
        ball_pos = np.array([0.0, 0.0, 1.2])
        ball_vel = np.zeros(3)
        pos_std = 0.02

        samples = []
        for _ in range(5000):
            pos_out, _ = add_observation_noise(
                ball_pos, ball_vel, rng, pos_std=pos_std, vel_std=0.0,
            )
            samples.append(pos_out.copy())
        samples = np.array(samples)

        np.testing.assert_allclose(samples.mean(axis=0), ball_pos, atol=0.001)
        np.testing.assert_allclose(samples.std(axis=0), pos_std, atol=0.003)


class TestTorqueNoiseStatistics:
    """力矩噪声的统计特性。"""

    def test_torque_noise_mean_near_zero(self) -> None:
        """大量样本的力矩噪声均值应接近零。"""
        rng = np.random.default_rng(42)
        u = np.zeros(6)
        torque_std = 0.5

        samples = []
        for _ in range(5000):
            u_out = add_torque_noise(u, rng, torque_std=torque_std)
            samples.append(u_out.copy())
        samples = np.array(samples)

        np.testing.assert_allclose(samples.mean(axis=0), 0.0, atol=0.02)
        np.testing.assert_allclose(samples.std(axis=0), torque_std, atol=0.05)


class TestInitQNoiseStatistics:
    """初始角度噪声的统计特性。"""

    def test_init_q_noise_mean_near_nominal(self) -> None:
        """大量样本的噪声后角度均值应接近标称值。"""
        rng = np.random.default_rng(42)
        init_q = np.array([-1.5, 1.57, -0.236, 0.404, 0.446, 2.45])
        noise_rad = 0.02

        samples = []
        for _ in range(5000):
            q_out = randomize_init_q(init_q, rng, noise_rad=noise_rad)
            samples.append(q_out.copy())
        samples = np.array(samples)

        np.testing.assert_allclose(samples.mean(axis=0), init_q, atol=0.003)
        np.testing.assert_allclose(samples.std(axis=0), noise_rad, atol=0.003)


class TestReproducibility:
    """同 seed 同参数结果应完全一致。"""

    def test_observation_noise_reproducible(self) -> None:
        """同 seed 调用两次 add_observation_noise 结果应完全相同。"""
        ball_pos = np.array([0.5, -0.3, 1.2])
        ball_vel = np.array([9.0, 0.1, -2.0])

        rng1 = np.random.default_rng(123)
        pos1, vel1 = add_observation_noise(
            ball_pos, ball_vel, rng1, pos_std=0.02, vel_std=0.1,
        )

        rng2 = np.random.default_rng(123)
        pos2, vel2 = add_observation_noise(
            ball_pos, ball_vel, rng2, pos_std=0.02, vel_std=0.1,
        )

        np.testing.assert_array_equal(pos1, pos2)
        np.testing.assert_array_equal(vel1, vel2)

    def test_torque_noise_reproducible(self) -> None:
        """同 seed 调用两次 add_torque_noise 结果应完全相同。"""
        u = np.array([10.0, -20.0, 15.0, -5.0, 2.0, -1.0])

        rng1 = np.random.default_rng(456)
        u1 = add_torque_noise(u, rng1, torque_std=0.5)

        rng2 = np.random.default_rng(456)
        u2 = add_torque_noise(u, rng2, torque_std=0.5)

        np.testing.assert_array_equal(u1, u2)

    def test_init_q_reproducible(self) -> None:
        """同 seed 调用两次 randomize_init_q 结果应完全相同。"""
        init_q = np.array([-1.5, 1.57, -0.236, 0.404, 0.446, 2.45])

        rng1 = np.random.default_rng(789)
        q1 = randomize_init_q(init_q, rng1, noise_rad=0.02)

        rng2 = np.random.default_rng(789)
        q2 = randomize_init_q(init_q, rng2, noise_rad=0.02)

        np.testing.assert_array_equal(q1, q2)


class TestPerAxisPosNoise:
    """pos_std_xyz 各轴独立标准差。"""

    def test_per_axis_pos_std(self) -> None:
        """pos_std_xyz=(0.01, 0.01, 0.05) 时 Z 轴 std≈0.05，X/Y≈0.01。"""
        rng = np.random.default_rng(42)
        ball_pos = np.array([0.0, 0.0, 1.2])
        ball_vel = np.zeros(3)
        samples = []
        for _ in range(5000):
            pos, _ = add_observation_noise(
                ball_pos, ball_vel, rng,
                pos_std_xyz=(0.01, 0.01, 0.05),
            )
            samples.append(pos.copy())
        samples = np.array(samples)
        np.testing.assert_allclose(samples.std(axis=0)[0], 0.01, atol=0.003)
        np.testing.assert_allclose(samples.std(axis=0)[1], 0.01, atol=0.003)
        np.testing.assert_allclose(samples.std(axis=0)[2], 0.05, atol=0.005)


class TestPerAxisAllZero:
    """per-axis 全零等价于不加噪声。"""

    def test_xyz_all_zero_is_identity(self) -> None:
        """pos_std_xyz=(0,0,0) 时位置不变。"""
        rng = np.random.default_rng(42)
        ball_pos = np.array([0.5, -0.3, 1.2])
        ball_vel = np.array([9.0, 0.1, -2.0])

        pos, vel = add_observation_noise(
            ball_pos, ball_vel, rng,
            pos_std_xyz=(0.0, 0.0, 0.0),
            vel_std_xyz=(0.0, 0.0, 0.0),
        )

        np.testing.assert_array_equal(pos, ball_pos)
        np.testing.assert_array_equal(vel, ball_vel)


class TestZClamp:
    """噪声后球 Z 坐标不低于地面。"""

    def test_z_clamped_above_ground(self) -> None:
        """Z 接近地面加大噪声后，Z 始终 >= 0.01。"""
        rng = np.random.default_rng(42)
        ball_pos = np.array([0.5, -0.3, 0.02])
        ball_vel = np.zeros(3)
        for _ in range(1000):
            pos, _ = add_observation_noise(
                ball_pos, ball_vel, rng,
                pos_std=0.05,
            )
            assert pos[2] >= 0.01


class TestPerAxisOverridesScalar:
    """per-axis 参数非 None 时忽略标量 pos_std/vel_std。"""

    def test_xyz_overrides_scalar_pos(self) -> None:
        """pos_std=0.02 且 pos_std_xyz 非 None 时使用 per-axis 值。"""
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        ball_pos = np.array([1.0, 2.0, 3.0])
        ball_vel = np.zeros(3)

        pos1, _ = add_observation_noise(
            ball_pos, ball_vel, rng1,
            pos_std=0.02,
            pos_std_xyz=(0.01, 0.01, 0.05),
        )
        pos2, _ = add_observation_noise(
            ball_pos, ball_vel, rng2,
            pos_std_xyz=(0.01, 0.01, 0.05),
        )
        np.testing.assert_array_equal(pos1, pos2)


class TestPerAxisVelNoise:
    """vel_std_xyz 各轴独立标准差。"""

    def test_per_axis_vel_std(self) -> None:
        """vel_std_xyz=(0.1, 0.1, 0.3) 时 Z 轴 std≈0.3。"""
        rng = np.random.default_rng(42)
        ball_pos = np.zeros(3)
        ball_vel = np.zeros(3)
        samples = []
        for _ in range(5000):
            _, vel = add_observation_noise(
                ball_pos, ball_vel, rng,
                vel_std_xyz=(0.1, 0.1, 0.3),
            )
            samples.append(vel.copy())
        samples = np.array(samples)
        np.testing.assert_allclose(samples.std(axis=0)[0], 0.1, atol=0.01)
        np.testing.assert_allclose(samples.std(axis=0)[2], 0.3, atol=0.02)


class TestDifferentSeedDifferentResult:
    """不同 seed 应产生不同结果。"""

    def test_different_seed_different_noise(self) -> None:
        """不同 seed 产生的噪声应不同。"""
        ball_pos = np.zeros(3)
        ball_vel = np.zeros(3)

        rng1 = np.random.default_rng(1)
        pos1, vel1 = add_observation_noise(
            ball_pos, ball_vel, rng1, pos_std=0.02, vel_std=0.1,
        )

        rng2 = np.random.default_rng(2)
        pos2, vel2 = add_observation_noise(
            ball_pos, ball_vel, rng2, pos_std=0.02, vel_std=0.1,
        )

        assert not np.allclose(pos1, pos2)

    def test_no_mutation_of_input(self) -> None:
        """噪声函数不应修改原始输入数组。"""
        rng = np.random.default_rng(42)
        ball_pos = np.array([0.5, -0.3, 1.2])
        ball_vel = np.array([9.0, 0.1, -2.0])
        u = np.array([10.0, -20.0, 15.0, -5.0, 2.0, -1.0])
        init_q = np.array([-1.5, 1.57, -0.236, 0.404, 0.446, 2.45])

        ball_pos_orig = ball_pos.copy()
        ball_vel_orig = ball_vel.copy()
        u_orig = u.copy()
        init_q_orig = init_q.copy()

        add_observation_noise(ball_pos, ball_vel, rng, pos_std=0.02, vel_std=0.1)
        add_torque_noise(u, rng, torque_std=0.5)
        randomize_init_q(init_q, rng, noise_rad=0.02)

        np.testing.assert_array_equal(ball_pos, ball_pos_orig)
        np.testing.assert_array_equal(ball_vel, ball_vel_orig)
        np.testing.assert_array_equal(u, u_orig)
        np.testing.assert_array_equal(init_q, init_q_orig)

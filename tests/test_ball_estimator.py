"""感知模块单元测试 — BallEstimator 卡尔曼滤波器 + RM65Env 集成。"""

import numpy as np
from pathlib import Path

from src.sim.rm65_env import RM65Env
from src.perception.ball_estimator import BallEstimator
from src.perception import BallEstimator as BallEstimatorTopLevel

MODEL_PATH = Path(__file__).parent.parent / "src" / "robot" / "rm65_model.xml"


class TestPublicImport:
    """感知模块顶层导出。"""

    def test_top_level_import(self) -> None:
        """from src.perception import BallEstimator 可用且等价。"""
        assert BallEstimatorTopLevel is BallEstimator


class TestBackwardCompatNoEstimator:
    """向后兼容基线：无 estimator 时行为与今日完全相同。"""

    def test_rm65env_no_estimator_works(self) -> None:
        """RM65Env(model_path) 无 estimator 参数时应正常初始化。"""
        env = RM65Env(MODEL_PATH, dt=0.005)
        assert env._estimator is None

    def test_get_ball_state_returns_raw(self) -> None:
        """无 estimator 时 get_ball_state 返回 MuJoCo 原始数据。"""
        env = RM65Env(MODEL_PATH, dt=0.005)
        env.reset()
        pos, vel = env.get_ball_state()
        assert pos.shape == (3,)
        assert vel.shape == (3,)
        # 球初始位置在 [5.0, 0.0, 2.0]（reset 中的默认值）
        np.testing.assert_allclose(pos, [5.0, 0.0, 2.0], atol=0.01)

    def test_get_ball_pos_vel_independent(self) -> None:
        """无 estimator 时 get_ball_pos/vel 各自返回独立分量。"""
        env = RM65Env(MODEL_PATH, dt=0.005)
        env.reset()
        pos = env.get_ball_pos()
        vel = env.get_ball_vel()
        assert pos.shape == (3,)
        assert vel.shape == (3,)
        np.testing.assert_allclose(pos, [5.0, 0.0, 2.0], atol=0.01)


class TestBallEstimatorInit:
    """BallEstimator 初始化和首次观测。"""

    def test_first_update_slams_state(self) -> None:
        """首次 update() 应以观测值 slam 初始化状态。"""
        est = BallEstimator(dt=0.005, pos_noise_std=0.02, vel_noise_std=0.2)
        pos, vel = est.update(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
        np.testing.assert_allclose(pos, [1.0, 2.0, 3.0], atol=1e-6)
        np.testing.assert_allclose(vel, [0.1, 0.2, 0.3], atol=1e-6)


class TestBallEstimatorConvergence:
    """BallEstimator 收敛性测试。"""

    def test_repeated_update_converges(self) -> None:
        """滤波状态沿着同一物理轨迹收敛于真值。"""
        est = BallEstimator(dt=0.005, pos_noise_std=0.02, vel_noise_std=0.1)
        traces = []
        for i in range(50):
            t = i * 0.005
            true_z = 1.2 - 3.0 * t - 0.5 * 9.81 * t * t
            true_vz = -3.0 - 9.81 * t
            pos_in = np.array([0.35 + -1.5 * t, -7.91 + 9.9 * t, true_z])
            vel_in = np.array([-1.5, 9.9, true_vz])
            est.update(pos_in, vel_in)
            traces.append(np.trace(est.covariance))

        final_pos, final_vel = est.state
        t_end = 49 * 0.005
        true_p = np.array([0.35 + -1.5 * t_end, -7.91 + 9.9 * t_end,
                           1.2 - 3.0 * t_end - 0.5 * 9.81 * t_end * t_end])
        true_v = np.array([-1.5, 9.9, -3.0 - 9.81 * t_end])
        np.testing.assert_allclose(final_pos, true_p, atol=0.10)
        np.testing.assert_allclose(final_vel, true_v, atol=0.30)
        assert traces[-1] < traces[0]


class TestBallEstimatorAttenuation:
    """BallEstimator 噪声衰减能力。"""

    def test_filter_attenuates_noise(self) -> None:
        """滤波后 RMSE < 原始观测 RMSE 的 1/3。"""
        rng = np.random.default_rng(42)
        dt = 0.005
        est = BallEstimator(dt=dt, pos_noise_std=0.05, vel_noise_std=0.3,
                            pos_noise_xyz=None, vel_noise_xyz=None)

        n = 200
        true_pos = np.zeros((n, 3))
        true_vel = np.zeros((n, 3))
        obs_pos = np.zeros((n, 3))
        obs_vel = np.zeros((n, 3))
        filt_pos = np.zeros((n, 3))
        filt_vel = np.zeros((n, 3))

        v0 = np.array([-1.5, 9.9, -3.0], dtype=float)
        g = np.array([0.0, 0.0, -9.81])
        for i in range(n):
            t = i * dt
            true_pos[i] = np.array([0.35, -7.91, 1.2]) + v0 * t + 0.5 * g * t * t
            true_vel[i] = v0 + g * t

            obs_pos[i] = true_pos[i] + rng.normal(0, 0.05, 3)
            obs_vel[i] = true_vel[i] + rng.normal(0, 0.3, 3)

            fp, fv = est.update(obs_pos[i], obs_vel[i])
            filt_pos[i] = fp
            filt_vel[i] = fv

        raw_rmse_pos = np.sqrt(np.mean((obs_pos - true_pos) ** 2))
        filt_rmse_pos = np.sqrt(np.mean((filt_pos - true_pos) ** 2))
        assert filt_rmse_pos < raw_rmse_pos * 0.35

        raw_rmse_vel = np.sqrt(np.mean((obs_vel - true_vel) ** 2))
        filt_rmse_vel = np.sqrt(np.mean((filt_vel - true_vel) ** 2))
        assert filt_rmse_vel < raw_rmse_vel * 0.35


class TestBallEstimatorBounce:
    """BallEstimator 弹跳保护。"""

    def test_bounce_detection_handles_vz_flip(self) -> None:
        """弹跳后滤波器快速跟踪 vz 翻转，不卡在负值。"""
        rng = np.random.default_rng(42)
        dt = 0.005
        g = 9.80665
        est = BallEstimator(dt=dt, pos_noise_std=0.02, vel_noise_std=0.2, g=g)

        z = 1.2
        vz = -3.0
        total_steps = 200
        true_vz = np.zeros(total_steps)
        filt_vz = np.zeros(total_steps)
        bounce_happened = False

        for i in range(total_steps):
            z += vz * dt
            vz += -g * dt
            if z < 0.033 and vz < 0 and not bounce_happened:
                z = 0.033
                vz = -vz * 0.75
                bounce_happened = True
            true_vz[i] = vz
            z_obs = z + rng.normal(0, 0.02)
            vz_obs = vz + rng.normal(0, 0.2)
            _, fv = est.update(np.array([0.0, 0.0, z_obs]),
                               np.array([0.0, 0.0, vz_obs]))
            filt_vz[i] = fv[2]

        assert bounce_happened
        # 弹跳后 30~40 步滤波 vz 应转为正值
        peak_idx = int(0.3 / dt)  # ~60 steps after start
        assert np.mean(filt_vz[peak_idx:peak_idx + 10]) > 0.0
        assert not np.any(np.isnan(est.covariance))


class TestBallEstimatorIdentity:
    """BallEstimator 零噪声恒等。"""

    def test_zero_noise_is_identity(self) -> None:
        """σ=0 时滤波输出 = 观测（不引入偏差）。"""
        est = BallEstimator(dt=0.005, pos_noise_std=0.0, vel_noise_std=0.0)
        for _ in range(10):
            fp, fv = est.update(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
            np.testing.assert_allclose(fp, [1.0, 2.0, 3.0], atol=0.01)
            np.testing.assert_allclose(fv, [0.1, 0.2, 0.3], atol=0.01)


class TestBallEstimatorVariableDt:
    """BallEstimator 自适应帧间隔。"""

    def test_variable_dt_handling(self) -> None:
        """变速输入（5ms / 33ms / 跳帧200ms）不导致发散或无 NaN。"""
        dt_default = 0.005
        est = BallEstimator(dt=dt_default, pos_noise_std=0.01, vel_noise_std=0.05)

        pos = np.array([0.35, -7.91, 1.2], dtype=float)
        vel = np.array([-1.5, 9.9, -3.0], dtype=float)

        for step in range(100):
            intervals = [0.005, 0.005, 0.005, 0.033, 0.033, 0.005, 0.005, 0.200]
            dt_i = intervals[step % len(intervals)]
            pos += vel * dt_i
            vel[2] += -9.81 * dt_i

            fp, fv = est.update(pos, vel)
            assert not np.any(np.isnan(est.covariance))
            assert not np.any(np.isinf(fp))
            assert not np.any(np.isinf(fv))


class TestRM65EnvIntegration:
    """RM65Env 与 BallEstimator 集成。"""

    def test_rm65env_with_estimator_config(self) -> None:
        """通过 estimator_config 字典启用滤波。"""
        env = RM65Env(MODEL_PATH, dt=0.005,
                      estimator_config={"pos_noise_std": 0.02, "vel_noise_std": 0.1})
        assert env._estimator is not None
        env.reset()
        pos, vel = env.get_ball_state()
        assert pos.shape == (3,)
        assert vel.shape == (3,)

    def test_get_ball_pos_vel_consistent(self) -> None:
        """get_ball_pos/vel 与 get_ball_state 返回一致滤波值。"""
        env = RM65Env(MODEL_PATH, dt=0.005,
                      estimator_config={"pos_noise_std": 0.02, "vel_noise_std": 0.1})
        env.reset()
        state_pos, state_vel = env.get_ball_state()
        pos = env.get_ball_pos()
        vel = env.get_ball_vel()
        np.testing.assert_allclose(pos, state_pos, atol=1e-6)
        np.testing.assert_allclose(vel, state_vel, atol=1e-6)

    def test_estimator_instance_direct(self) -> None:
        """直接传入预配置 BallEstimator 实例。"""
        est = BallEstimator(dt=0.005, pos_noise_std=0.05)
        env = RM65Env(MODEL_PATH, dt=0.005, estimator=est)
        assert env._estimator is est

    def test_reset_clears_estimator_state(self) -> None:
        """env.reset() 后 estimator 回到未初始化状态，P 恢复初始值。"""
        est = BallEstimator(dt=0.005, pos_noise_std=0.02, vel_noise_std=0.1)
        env = RM65Env(MODEL_PATH, dt=0.005, estimator=est)
        env.reset()
        _ = env.get_ball_state()
        assert est._initialized is True

        env.reset()
        assert est._initialized is False
        np.testing.assert_allclose(est.covariance, np.eye(6) * 100.0)


class TestBallEstimatorDynamicR:
    """BallEstimator 动态观测噪声参数。"""

    def test_update_noise_params_dynamic_R(self) -> None:
        """缩小噪声参数后协方差进一步减小。"""
        est = BallEstimator(dt=0.005, pos_noise_std=0.1, vel_noise_std=0.5)

        for _ in range(5):
            est.update(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))

        est.update_noise_params(pos_noise_std=0.001, vel_noise_std=0.005)
        assert est._R[0, 0] == 0.001 ** 2
        assert est._R[3, 3] == 0.005 ** 2

        p_before = np.trace(est.covariance)
        for _ in range(5):
            est.update(np.array([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
        p_after = np.trace(est.covariance)
        assert p_after < p_before

    def test_incremental_update_preserves_per_axis(self) -> None:
        """per-axis R 设置后，增量更新未传的轴保持原值。"""
        est = BallEstimator(dt=0.005, pos_noise_xyz=(0.02, 0.05, 0.03))
        est.update_noise_params(vel_noise_std=0.1)
        assert est._R[0, 0] == 0.02 ** 2
        assert est._R[1, 1] == 0.05 ** 2
        assert est._R[2, 2] == 0.03 ** 2
        assert est._R[3, 3] == est._R[4, 4] == est._R[5, 5] == 0.1 ** 2

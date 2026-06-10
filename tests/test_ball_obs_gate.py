"""BallObservationGate 单元测试。"""

import numpy as np
import pytest

from src.perception.ball_estimator import BallEstimator
from src.perception.ball_obs_gate import BallObservationGate


DT = 0.005
G = 9.80665


class TestObsInterval:
    """观测间隔计算。"""

    def test_200hz_interval_1(self):
        gate = BallObservationGate(200, DT)
        assert gate.obs_interval == 1

    def test_60hz_interval(self):
        gate = BallObservationGate(60, DT)
        assert gate.obs_interval == round(1.0 / (60 * DT))

    def test_30hz_interval_7(self):
        gate = BallObservationGate(30, DT)
        assert gate.obs_interval == 7

    def test_10hz_interval_20(self):
        gate = BallObservationGate(10, DT)
        assert gate.obs_interval == 20


class TestObsStepDetection:
    """观测步检测逻辑。"""

    def test_step0_always_obs(self):
        gate = BallObservationGate(30, DT)
        assert gate._is_obs_step(0)

    def test_30hz_obs_steps(self):
        gate = BallObservationGate(30, DT)
        interval = gate.obs_interval
        assert gate._is_obs_step(0)
        assert not gate._is_obs_step(1)
        assert not gate._is_obs_step(6)
        assert gate._is_obs_step(interval)
        assert gate._is_obs_step(2 * interval)

    def test_200hz_every_step(self):
        gate = BallObservationGate(200, DT)
        for s in range(20):
            assert gate._is_obs_step(s)


class TestNoNoiseNoKf:
    """无噪声无 KF：观测步返回真值，非观测步返回抛物线预测。"""

    def test_obs_step_returns_true_value(self):
        gate = BallObservationGate(30, DT)
        pos = np.array([1.0, 2.0, 3.0])
        vel = np.array([0.1, -5.0, 2.0])
        out_pos, out_vel = gate.get_state(0, pos, vel)
        np.testing.assert_allclose(out_pos, pos, atol=1e-10)
        np.testing.assert_allclose(out_vel, vel, atol=1e-10)

    def test_non_obs_step_parabolic(self):
        gate = BallObservationGate(30, DT)
        p0 = np.array([0.0, 3.0, 2.0])
        v0 = np.array([0.0, -5.0, 3.0])
        gate.get_state(0, p0, v0)

        elapsed = 5
        pred_pos, pred_vel = gate.get_state(elapsed, p0, v0)
        dt_total = elapsed * DT
        expected_pos = p0 + v0 * dt_total
        expected_pos[2] += 0.5 * (-G) * dt_total**2
        expected_vel = v0.copy()
        expected_vel[2] += (-G) * dt_total
        np.testing.assert_allclose(pred_pos, expected_pos, atol=1e-10)
        np.testing.assert_allclose(pred_vel, expected_vel, atol=1e-10)

    def test_200hz_always_true_value(self):
        gate = BallObservationGate(200, DT)
        pos = np.array([1.0, 2.0, 3.0])
        vel = np.array([0.0, -4.0, 1.0])
        for step in range(20):
            out_pos, out_vel = gate.get_state(step, pos, vel)
            np.testing.assert_allclose(out_pos, pos, atol=1e-10)
            np.testing.assert_allclose(out_vel, vel, atol=1e-10)


class TestNoiseInjection:
    """噪声注入：仅观测步有噪声。"""

    def test_obs_step_has_noise(self):
        rng = np.random.default_rng(42)
        gate = BallObservationGate(30, DT, noise_pos=0.1, noise_vel=1.0, rng=rng)
        pos = np.array([1.0, 2.0, 3.0])
        vel = np.array([0.0, -5.0, 2.0])
        out_pos, out_vel = gate.get_state(0, pos, vel)
        assert not np.allclose(out_pos, pos, atol=1e-6)
        assert not np.allclose(out_vel, vel, atol=1e-6)

    def test_non_obs_step_no_extra_noise(self):
        rng = np.random.default_rng(42)
        gate = BallObservationGate(30, DT, noise_pos=0.1, noise_vel=1.0, rng=rng)
        p0 = np.array([1.0, 2.0, 3.0])
        v0 = np.array([0.0, -5.0, 2.0])
        obs_pos, obs_vel = gate.get_state(0, p0, v0)

        pred_pos, pred_vel = gate.get_state(3, p0, v0)
        dt_total = 3 * DT
        expected_pos = obs_pos + obs_vel * dt_total
        expected_pos = expected_pos.copy()
        expected_pos[2] += 0.5 * (-G) * dt_total**2
        assert np.allclose(pred_pos, expected_pos, atol=1e-10)

    def test_per_axis_noise(self):
        rng = np.random.default_rng(42)
        gate = BallObservationGate(
            30, DT,
            pos_std_xyz=(0.01, 0.05, 0.01),
            vel_std_xyz=(0.1, 0.5, 0.1),
            rng=rng,
        )
        pos = np.array([1.0, 2.0, 3.0])
        vel = np.array([0.0, -5.0, 2.0])
        out_pos, out_vel = gate.get_state(0, pos, vel)
        assert not np.allclose(out_pos, pos, atol=1e-6)


class TestWithKF:
    """KF 模式：观测步 update，非观测步 predict-only。"""

    def test_kf_update_on_obs_step(self):
        kf = BallEstimator(DT, pos_noise_std=0.02, vel_noise_std=0.2)
        gate = BallObservationGate(30, DT, noise_pos=0.02, noise_vel=0.2, kf=kf, rng=np.random.default_rng(0))
        pos = np.array([1.0, 2.0, 3.0])
        vel = np.array([0.0, -5.0, 2.0])
        out_pos, out_vel = gate.get_state(0, pos, vel)
        assert kf.initialized

    def test_kf_predict_only_on_non_obs(self):
        kf = BallEstimator(DT, pos_noise_std=0.02, vel_noise_std=0.2)
        gate = BallObservationGate(30, DT, noise_pos=0.02, noise_vel=0.2, kf=kf, rng=np.random.default_rng(0))
        p0 = np.array([1.0, 2.0, 3.0])
        v0 = np.array([0.0, -5.0, 2.0])
        gate.get_state(0, p0, v0)
        pred_pos, pred_vel = gate.get_state(3, p0, v0)
        assert pred_pos is not None
        assert pred_vel is not None

    def test_kf_covariance_propagates(self):
        kf = BallEstimator(DT, pos_noise_std=0.05, vel_noise_std=0.5)
        gate = BallObservationGate(30, DT, noise_pos=0.05, noise_vel=0.5, kf=kf, rng=np.random.default_rng(0))
        p0 = np.array([1.0, 2.0, 3.0])
        v0 = np.array([0.0, -5.0, 2.0])
        gate.get_state(0, p0, v0)
        P_after_obs = kf.covariance.copy()
        gate.get_state(1, p0, v0)
        P_after_pred = kf.covariance
        assert np.trace(P_after_pred) > np.trace(P_after_obs)


class TestReset:
    """重置测试。"""

    def test_reset_clears_state(self):
        gate = BallObservationGate(30, DT)
        pos = np.array([1.0, 2.0, 3.0])
        vel = np.array([0.0, -5.0, 2.0])
        gate.get_state(0, pos, vel)
        assert gate.last_obs_pos is not None
        gate.reset()
        assert gate.last_obs_pos is None
        assert gate.last_obs_step == -1

    def test_reset_clears_kf(self):
        kf = BallEstimator(DT, pos_noise_std=0.05, vel_noise_std=0.5)
        gate = BallObservationGate(30, DT, noise_pos=0.05, noise_vel=0.5, kf=kf, rng=np.random.default_rng(0))
        gate.get_state(0, np.array([1.0, 2.0, 3.0]), np.array([0.0, -5.0, 2.0]))
        assert kf.initialized
        gate.reset()
        assert not kf.initialized


class TestParabolicAccuracy:
    """抛物线预测精度。"""

    def test_freefall_accuracy(self):
        gate = BallObservationGate(30, DT)
        p0 = np.array([0.0, 3.0, 2.0])
        v0 = np.array([0.0, -5.0, 0.0])
        gate.get_state(0, p0, v0)

        for elapsed in [1, 3, 6]:
            pred_pos, pred_vel = gate.get_state(elapsed, p0, v0)
            dt_total = elapsed * DT
            expected_z = p0[2] + 0.5 * (-G) * dt_total**2
            expected_vz = (-G) * dt_total
            assert abs(pred_pos[2] - expected_z) < 1e-10
            assert abs(pred_vel[2] - expected_vz) < 1e-10

    def test_horizontal_motion(self):
        gate = BallObservationGate(30, DT)
        p0 = np.array([0.0, 3.0, 2.0])
        v0 = np.array([1.0, -5.0, 3.0])
        gate.get_state(0, p0, v0)

        elapsed = 4
        pred_pos, _ = gate.get_state(elapsed, p0, v0)
        dt_total = elapsed * DT
        assert abs(pred_pos[0] - (p0[0] + v0[0] * dt_total)) < 1e-10
        assert abs(pred_pos[1] - (p0[1] + v0[1] * dt_total)) < 1e-10


class TestContinuousSequence:
    """连续序列测试：模拟完整 MPC 循环。"""

    def test_30hz_gate_sequence(self):
        gate = BallObservationGate(30, DT)
        pos = np.array([0.0, 3.0, 2.0])
        vel = np.array([0.0, -5.0, 3.0])
        interval = gate.obs_interval

        obs_count = 0
        for step in range(50):
            out_pos, out_vel = gate.get_state(step, pos, vel)
            if gate._is_obs_step(step):
                obs_count += 1
                np.testing.assert_allclose(out_pos, pos, atol=1e-10)
        assert obs_count == pytest.approx(50 / interval + 1, abs=1)

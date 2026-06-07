"""观测模块集成测试 — 噪声注入→卡尔曼滤波完整 pipeline。

验证三层架构的端到端可靠性：
  仿真层(MuJoCo真值) → 实验层(add_observation_noise) → 感知层(BallEstimator) → 输出

测试矩阵：
  - 空中飞行 + lo/mid/hi/anis 噪声
  - 弹跳场景
  - NumPy 解析抛物线补充
  - RM65Env 全链路 smoke test
"""

import numpy as np
import pytest
from pathlib import Path

from src.sim.rm65_env import RM65Env
from src.perception.ball_estimator import BallEstimator
from src.utils.noise import add_observation_noise

MODEL_PATH = Path(__file__).parent.parent / "src" / "robot" / "rm65_model.xml"
DT = 0.005


def _run_mujoco_pipeline(
    ball_pos0: np.ndarray,
    ball_vel0: np.ndarray,
    n_steps: int,
    rng: np.random.Generator,
    pos_std: float = 0.0,
    vel_std: float = 0.0,
    pos_std_xyz: tuple[float, float, float] | None = None,
    vel_std_xyz: tuple[float, float, float] | None = None,
    estimator_pos_noise: float = 0.03,
    estimator_vel_noise: float = 0.3,
) -> dict:
    """运行 MuJoCo 球飞行 + 噪声注入 + KF 滤波完整 pipeline。

    Args:
        ball_pos0: 球初始位置 (3,)。
        ball_vel0: 球初始速度 (3,)。
        n_steps: 仿真步数。
        rng: 随机数生成器。
        pos_std: 位置噪声标量 std。
        vel_std: 速度噪声标量 std。
        pos_std_xyz: per-axis 位置噪声，优先于标量。
        vel_std_xyz: per-axis 速度噪声，优先于标量。
        estimator_pos_noise: BallEstimator 的 R 矩阵位置噪声参数。
        estimator_vel_noise: BallEstimator 的 R 矩阵速度噪声参数。

    Returns:
        包含 true/obs/filt 轨迹和 RMSE 的字典。
    """
    est = BallEstimator(
        dt=DT, pos_noise_std=estimator_pos_noise, vel_noise_std=estimator_vel_noise
    )
    bq = RM65Env.BALL_QPOS_START
    bv = RM65Env.BALL_QVEL_START

    true_pos_arr = np.zeros((n_steps, 3))
    true_vel_arr = np.zeros((n_steps, 3))
    obs_pos_arr = np.zeros((n_steps, 3))
    obs_vel_arr = np.zeros((n_steps, 3))
    filt_pos_arr = np.zeros((n_steps, 3))
    filt_vel_arr = np.zeros((n_steps, 3))

    env = RM65Env(MODEL_PATH, dt=DT)
    env.reset()
    env.set_ball_state(ball_pos0, ball_vel0)

    for i in range(n_steps):
        true_pos = env.data.qpos[bq:bq + 3].copy()
        true_vel = env.data.qvel[bv:bv + 3].copy()

        noisy_pos, noisy_vel = add_observation_noise(
            true_pos, true_vel, rng,
            pos_std=pos_std, vel_std=vel_std,
            pos_std_xyz=pos_std_xyz, vel_std_xyz=vel_std_xyz,
        )

        filt_pos, filt_vel = est.update(noisy_pos, noisy_vel)

        true_pos_arr[i] = true_pos
        true_vel_arr[i] = true_vel
        obs_pos_arr[i] = noisy_pos
        obs_vel_arr[i] = noisy_vel
        filt_pos_arr[i] = filt_pos
        filt_vel_arr[i] = filt_vel

        env.step(np.zeros(RM65Env.NU))

    raw_rmse_pos = np.sqrt(np.mean((obs_pos_arr - true_pos_arr) ** 2))
    filt_rmse_pos = np.sqrt(np.mean((filt_pos_arr - true_pos_arr) ** 2))
    raw_rmse_vel = np.sqrt(np.mean((obs_vel_arr - true_vel_arr) ** 2))
    filt_rmse_vel = np.sqrt(np.mean((filt_vel_arr - true_vel_arr) ** 2))

    return {
        "true_pos": true_pos_arr,
        "true_vel": true_vel_arr,
        "obs_pos": obs_pos_arr,
        "obs_vel": obs_vel_arr,
        "filt_pos": filt_pos_arr,
        "filt_vel": filt_vel_arr,
        "raw_rmse_pos": raw_rmse_pos,
        "filt_rmse_pos": filt_rmse_pos,
        "raw_rmse_vel": raw_rmse_vel,
        "filt_rmse_vel": filt_rmse_vel,
    }


def _find_bounce_step(true_vel: np.ndarray) -> int | None:
    """找到 vz 由负变正的帧（弹跳时刻）。"""
    vz = true_vel[:, 2]
    for i in range(1, len(vz)):
        if vz[i - 1] < 0 and vz[i] > 0:
            return i
    return None


# ================================================================
# MuJoCo 层 pipeline 测试
# ================================================================


class TestPipelineMuJoCo:
    """MuJoCo 物理仿真 + 噪声注入 + KF 滤波完整 pipeline。"""

    def test_flight_lo_noise(self) -> None:
        """空中飞行(无弹跳) + lo噪声: RMSE衰减, 终点<10cm, 无NaN。"""
        result = _run_mujoco_pipeline(
            ball_pos0=np.array([2.5, -4.0, 2.0]),
            ball_vel0=np.array([-1.5, 8.0, -2.0]),
            n_steps=200,
            rng=np.random.default_rng(42),
            pos_std=0.03,
            vel_std=0.3,
            estimator_pos_noise=0.03,
            estimator_vel_noise=0.3,
        )

        assert not np.any(np.isnan(result["filt_pos"]))
        assert not np.any(np.isnan(result["filt_vel"]))
        assert result["filt_rmse_pos"] < result["raw_rmse_pos"] * 0.5
        assert result["filt_rmse_vel"] < result["raw_rmse_vel"] * 0.5

        final_err = np.linalg.norm(result["filt_pos"][-1] - result["true_pos"][-1])
        assert final_err < 0.10, f"终点位置误差 {final_err:.3f}m > 10cm"

    def test_bounce_mid_noise(self) -> None:
        """弹跳场景 + mid噪声: RMSE衰减, 弹跳vz跟踪不滞后>20步, 无NaN。"""
        result = _run_mujoco_pipeline(
            ball_pos0=np.array([1.0, -3.0, 1.0]),
            ball_vel0=np.array([-0.3, 5.0, -1.5]),
            n_steps=400,
            rng=np.random.default_rng(42),
            pos_std=0.05,
            vel_std=0.5,
            estimator_pos_noise=0.05,
            estimator_vel_noise=0.5,
        )

        assert not np.any(np.isnan(result["filt_pos"]))
        assert not np.any(np.isnan(result["filt_vel"]))
        assert result["filt_rmse_pos"] < result["raw_rmse_pos"] * 0.5

        bounce_step = _find_bounce_step(result["true_vel"])
        assert bounce_step is not None, "球未弹跳，测试条件不满足"

        post_bounce_filt_vz = result["filt_vel"][bounce_step:bounce_step + 20, 2]
        assert np.mean(post_bounce_filt_vz) > 0.0, (
            f"弹跳后20步内滤波vz均值={np.mean(post_bounce_filt_vz):.3f} <= 0，"
            "滤波器未跟踪弹跳"
        )

    def test_flight_hi_noise(self) -> None:
        """高噪声压力测试: 不发散(无NaN/Inf), 终点<30cm。"""
        result = _run_mujoco_pipeline(
            ball_pos0=np.array([2.5, -4.0, 2.0]),
            ball_vel0=np.array([-1.5, 8.0, -2.0]),
            n_steps=200,
            rng=np.random.default_rng(42),
            pos_std=0.10,
            vel_std=1.0,
            estimator_pos_noise=0.10,
            estimator_vel_noise=1.0,
        )

        assert not np.any(np.isnan(result["filt_pos"]))
        assert not np.any(np.isinf(result["filt_pos"]))
        assert not np.any(np.isnan(result["filt_vel"]))
        assert not np.any(np.isinf(result["filt_vel"]))

        final_err = np.linalg.norm(result["filt_pos"][-1] - result["true_pos"][-1])
        assert final_err < 0.30, f"终点位置误差 {final_err:.3f}m > 30cm"

    def test_flight_anisotropic(self) -> None:
        """各向异性噪声(深度方向大): RMSE衰减, 无NaN。"""
        result = _run_mujoco_pipeline(
            ball_pos0=np.array([2.5, -4.0, 2.0]),
            ball_vel0=np.array([-1.5, 8.0, -2.0]),
            n_steps=200,
            rng=np.random.default_rng(42),
            pos_std_xyz=(0.02, 0.08, 0.02),
            vel_std_xyz=(0.3, 1.0, 0.3),
            estimator_pos_noise=0.04,
            estimator_vel_noise=0.5,
        )

        assert not np.any(np.isnan(result["filt_pos"]))
        assert not np.any(np.isnan(result["filt_vel"]))
        assert result["filt_rmse_pos"] < result["raw_rmse_pos"] * 0.5
        assert result["filt_rmse_vel"] < result["raw_rmse_vel"] * 0.5


# ================================================================
# NumPy 解析抛物线补充测试
# ================================================================


class TestPipelineNumPy:
    """纯 NumPy 解析抛物线 + 噪声 + KF 滤波，无 MuJoCo 依赖。"""

    def test_analytical_parabola_lo_noise(self) -> None:
        """解析抛物线 + lo噪声: 终点pos<5cm, vel<0.3m/s, RMSE衰减。"""
        dt = DT
        g = 9.80665
        n_steps = 200
        rng = np.random.default_rng(42)

        est = BallEstimator(dt=dt, pos_noise_std=0.03, vel_noise_std=0.3)

        p0 = np.array([2.5, -4.0, 2.0])
        v0 = np.array([-1.5, 8.0, -2.0])
        grav = np.array([0.0, 0.0, -g])

        true_pos_arr = np.zeros((n_steps, 3))
        filt_pos_arr = np.zeros((n_steps, 3))
        true_vel_arr = np.zeros((n_steps, 3))
        filt_vel_arr = np.zeros((n_steps, 3))
        obs_pos_arr = np.zeros((n_steps, 3))
        obs_vel_arr = np.zeros((n_steps, 3))

        for i in range(n_steps):
            t = i * dt
            true_pos = p0 + v0 * t + 0.5 * grav * t * t
            true_vel = v0 + grav * t

            noisy_pos = true_pos + rng.normal(0, 0.03, 3)
            noisy_vel = true_vel + rng.normal(0, 0.3, 3)

            fp, fv = est.update(noisy_pos, noisy_vel)

            true_pos_arr[i] = true_pos
            true_vel_arr[i] = true_vel
            obs_pos_arr[i] = noisy_pos
            obs_vel_arr[i] = noisy_vel
            filt_pos_arr[i] = fp
            filt_vel_arr[i] = fv

        final_pos_err = np.linalg.norm(filt_pos_arr[-1] - true_pos_arr[-1])
        final_vel_err = np.linalg.norm(filt_vel_arr[-1] - true_vel_arr[-1])

        assert final_pos_err < 0.05, f"终点位置误差 {final_pos_err:.3f}m > 5cm"
        assert final_vel_err < 0.3, f"终点速度误差 {final_vel_err:.3f}m/s > 0.3m/s"

        raw_rmse = np.sqrt(np.mean((obs_pos_arr - true_pos_arr) ** 2))
        filt_rmse = np.sqrt(np.mean((filt_pos_arr - true_pos_arr) ** 2))
        assert filt_rmse < raw_rmse * 0.5


# ================================================================
# RM65Env 全链路 smoke test
# ================================================================


class TestPipelineSmoke:
    """RM65Env + estimator_config 全链路 smoke: 确认数据通路不报错。"""

    def test_env_estimator_noise_pipeline(self) -> None:
        """env 配 estimator → get_ball_state 全链路 10 步不报错, 形状正确。"""
        env = RM65Env(
            MODEL_PATH, dt=DT,
            estimator_config={"pos_noise_std": 0.03, "vel_noise_std": 0.3}
        )
        env.reset()

        for _ in range(10):
            pos, vel = env.get_ball_state()
            assert pos.shape == (3,)
            assert vel.shape == (3,)
            assert not np.any(np.isnan(pos))
            assert not np.any(np.isnan(vel))
            env.step(np.zeros(RM65Env.NU))

    def test_env_reset_reinitializes_pipeline(self) -> None:
        """reset 后 estimator 重新初始化，pipeline 可复用。"""
        env = RM65Env(
            MODEL_PATH, dt=DT,
            estimator_config={"pos_noise_std": 0.03, "vel_noise_std": 0.3}
        )

        for trial in range(3):
            env.reset()
            for _ in range(5):
                pos, vel = env.get_ball_state()
                assert pos.shape == (3,)
                assert vel.shape == (3,)
                env.step(np.zeros(RM65Env.NU))

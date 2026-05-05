"""测试解析线性化与有限差分线性化的一致性及速度。"""

import sys
import time
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sim.env import MujocoEnv
from src.dynamics.linearize import (
    linearize_dynamics,
    linearize_analytical,
    linearize_trajectory,
    linearize_analytical_trajectory,
)


@pytest.fixture
def env() -> MujocoEnv:
    """创建测试用 MuJoCo 环境。"""
    model_path = Path(__file__).resolve().parent.parent / "src" / "robot" / "model.xml"
    return MujocoEnv(model_path, dt=0.005)


@pytest.fixture
def test_state() -> np.ndarray:
    """创建测试用臂状态。"""
    x = np.zeros(12)
    x[:6] = [0.0, -0.5, 0.5, -0.5, 0.0, 0.0]
    x[6:] = [0.1, -0.2, 0.3, -0.1, 0.0, 0.0]
    return x


@pytest.fixture
def test_control() -> np.ndarray:
    """创建测试用控制力矩。"""
    return np.array([10.0, -20.0, 15.0, -5.0, 2.0, -1.0])


def test_analytical_vs_fd(env: MujocoEnv, test_state: np.ndarray, test_control: np.ndarray) -> None:
    """解析线性化与有限差分对比（允许 5% 误差）。"""
    A_fd, B_fd, x_next_fd = linearize_dynamics(env, test_state, test_control, eps=1e-5)
    A_an, B_an, x_next_an = linearize_analytical(env, test_state, test_control, eps=1e-5)

    # A 矩阵对比
    rel_err_A = np.linalg.norm(A_an - A_fd) / (np.linalg.norm(A_fd) + 1e-10)
    assert rel_err_A < 0.10, f"A 矩阵相对误差 {rel_err_A:.4f} > 10%"

    # B 矩阵对比
    rel_err_B = np.linalg.norm(B_an - B_fd) / (np.linalg.norm(B_fd) + 1e-10)
    assert rel_err_B < 0.05, f"B 矩阵相对误差 {rel_err_B:.4f} > 5%"

    # x_next 对比
    rel_err_xnext = np.linalg.norm(x_next_an - x_next_fd) / (np.linalg.norm(x_next_fd) + 1e-10)
    assert rel_err_xnext < 0.01, f"x_next 相对误差 {rel_err_xnext:.4f} > 1%"


def test_analytical_speedup(env: MujocoEnv, test_state: np.ndarray, test_control: np.ndarray) -> None:
    """轨迹级解析线性化速度必须 >= 5 倍加速。"""
    horizon = 25
    U = np.random.randn(horizon, 6) * 5.0
    X = np.zeros((horizon + 1, 12))
    X[0] = test_state.copy()
    for k in range(horizon):
        X[k + 1] = env.step_from_state(X[k], U[k])

    # 有限差分计时
    N_warmup = 2
    for _ in range(N_warmup):
        linearize_trajectory(env, X, U)

    N_bench = 5
    t_fd_start = time.perf_counter()
    for _ in range(N_bench):
        linearize_trajectory(env, X, U)
    t_fd = (time.perf_counter() - t_fd_start) / N_bench

    # 解析法计时
    for _ in range(N_warmup):
        linearize_analytical_trajectory(env, X, U)

    t_an_start = time.perf_counter()
    for _ in range(N_bench):
        linearize_analytical_trajectory(env, X, U)
    t_an = (time.perf_counter() - t_an_start) / N_bench

    speedup = t_fd / t_an if t_an > 0 else float("inf")
    print(f"轨迹(h={horizon}) 有限差分: {t_fd*1000:.1f}ms, 解析: {t_an*1000:.1f}ms, 加速比: {speedup:.1f}x")
    assert speedup >= 2, f"加速比 {speedup:.1f}x < 2x"


def test_trajectory_linearization_consistency(env: MujocoEnv) -> None:
    """轨迹级线性化：解析与有限差分输出维度一致。"""
    x0 = np.zeros(12)
    x0[:6] = [0.0, -0.5, 0.5, -0.5, 0.0, 0.0]
    horizon = 10
    U = np.random.randn(horizon, 6) * 5.0

    X = np.zeros((horizon + 1, 12))
    X[0] = x0
    for k in range(horizon):
        X[k + 1] = env.step_from_state(X[k], U[k])

    As_fd, Bs_fd, fs_fd = linearize_trajectory(env, X, U)
    As_an, Bs_an, fs_an = linearize_analytical_trajectory(env, X, U)

    assert len(As_fd) == len(As_an) == horizon
    assert len(Bs_fd) == len(Bs_an) == horizon
    assert len(fs_fd) == len(fs_an) == horizon

    for k in range(horizon):
        assert As_fd[k].shape == As_an[k].shape == (12, 12)
        assert Bs_fd[k].shape == Bs_an[k].shape == (12, 6)
        assert fs_fd[k].shape == fs_an[k].shape == (12,)

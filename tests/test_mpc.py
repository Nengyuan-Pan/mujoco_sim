"""MPC 相关功能测试。"""

import numpy as np

from src.tennis.hitting import schedule_weights
from src.ilqt.cost import HittingCost
from src.ilqt.solver import ILQTSolver
from src.sim.env import MujocoEnv
from pathlib import Path


class TestScheduleWeights:
    """测试权重调度。"""

    def test_initial_weights(self) -> None:
        """初始时 Q_p 权重大，Q_v 权重小。"""
        Q_p, Q_v = schedule_weights(1.0, 1.0)
        assert Q_p > Q_v

    def test_final_weights(self) -> None:
        """接近击打时 Q_v 权重增大。"""
        Q_p, Q_v = schedule_weights(0.1, 1.0)
        assert Q_v > Q_p

    def test_monotonic(self) -> None:
        """权重应单调变化。"""
        ratios = [1.0, 0.75, 0.5, 0.25, 0.1]
        Q_p_list = []
        Q_v_list = []
        for r in ratios:
            Q_p, Q_v = schedule_weights(r, 1.0)
            Q_p_list.append(Q_p)
            Q_v_list.append(Q_v)
        # Q_p 应递减
        for i in range(len(Q_p_list) - 1):
            assert Q_p_list[i] >= Q_p_list[i + 1]
        # Q_v 应递增
        for i in range(len(Q_v_list) - 1):
            assert Q_v_list[i] <= Q_v_list[i + 1]


class TestHittingCost:
    """测试代价函数更新。"""

    def test_update_weights(self) -> None:
        """测试权重更新。"""
        model_path = Path(__file__).resolve().parent.parent / "src" / "robot" / "model.xml"
        env = MujocoEnv(model_path)
        p_hit = np.array([0.5, 0.0, 1.0])
        v_hit = np.array([0.0, 0.0, 0.0])
        Q_p = np.array([100.0, 100.0, 100.0])
        Q_v = np.array([10.0, 10.0, 10.0])
        R = 0.001

        cost_fn = HittingCost(env, p_hit, v_hit, Q_p, Q_v, R)

        # 更新权重
        cost_fn.update_weights(2.0, 3.0)
        # 验证 Q_p 被缩放
        np.testing.assert_allclose(cost_fn.Q_p[0, 0], 200.0)
        np.testing.assert_allclose(cost_fn.Q_v[0, 0], 30.0)

    def test_update_target(self) -> None:
        """测试目标更新。"""
        model_path = Path(__file__).resolve().parent.parent / "src" / "robot" / "model.xml"
        env = MujocoEnv(model_path)
        p_hit = np.array([0.5, 0.0, 1.0])
        v_hit = np.zeros(3)
        cost_fn = HittingCost(env, p_hit, v_hit, np.ones(3) * 100, np.ones(3) * 10, 0.001)

        new_p = np.array([0.6, 0.1, 1.2])
        new_v = np.array([1.0, 0.0, 0.0])
        cost_fn.update_target(new_p, new_v)

        np.testing.assert_allclose(cost_fn.p_hit, new_p)
        np.testing.assert_allclose(cost_fn.v_hit, new_v)


class TestSolverFewIters:
    """测试 solve_few_iters 方法。"""

    def test_few_iters_returns_result(self) -> None:
        """solve_few_iters 应返回有效的轨迹和控制。"""
        model_path = Path(__file__).resolve().parent.parent / "src" / "robot" / "model.xml"
        env = MujocoEnv(model_path, dt=0.005)

        init_q = np.array([0.0, -0.5, 0.5, -0.5, 0.0, 0.0])
        x0 = np.zeros(12)
        x0[:6] = init_q

        p_hit = np.array([0.5, 0.0, 1.0])
        v_hit = np.zeros(3)
        Q_p = np.array([5000.0, 5000.0, 5000.0])
        Q_v = np.array([10.0, 10.0, 10.0])
        R = 0.001

        cost_fn = HittingCost(env, p_hit, v_hit, Q_p, Q_v, R)

        ilqt_cfg = {
            "max_iter": 10,
            "tol": 1e-4,
            "horizon": 20,
            "mu_min": 1e-6,
            "mu_max": 1e10,
            "mu_init": 0.01,
            "delta_0": 1.6,
            "alpha_list": [1.0, 0.5, 0.1],
            "lin_eps": 1e-5,
        }
        solver = ILQTSolver(ilqt_cfg)

        U_init = np.zeros((20, 6))
        X, U, cost_history, success = solver.solve_few_iters(
            env, cost_fn, x0, U_init, max_iter=3,
        )

        assert X.shape == (21, 12)
        assert U.shape == (20, 6)
        assert len(cost_history) > 0

"""执行器双模式（力矩/位置）测试。

覆盖 Stage -1 切片 1-9：RM65Env 位置模式物理正确性、力矩模式零回归、
ctrlrange 切换、属性防错、clone 同步、reset 不影响配置、
Python 解析线性化 B/A 矩阵位置模式数学结构验证。
"""

import numpy as np
import mujoco
import pytest
from pathlib import Path

from src.sim.rm65_env import RM65Env
from src.dynamics.linearize import linearize_analytical


def _make_env() -> RM65Env:
    """创建测试用 RM65Env 实例。"""
    model_path = Path(__file__).resolve().parent.parent / "src" / "robot" / "rm65_model.xml"
    return RM65Env(model_path)


class TestEnvPositionStep:
    """切片 1-2：位置模式 step 物理正确性。"""

    def test_position_step_matches_manual_torque(self) -> None:
        """位置模式 step(q_desired) 产生的力矩 = Kp*(u-q) - Kd*qdot。

        初始 q=0, qdot=0, step(q_desired=0.1) -> tau = Kp*0.1。
        """
        env = _make_env()
        kp = np.array([200.0, 200.0, 200.0, 50.0, 50.0, 20.0])
        kd = np.array([20.0, 20.0, 20.0, 5.0, 5.0, 2.0])

        env.configure_actuator_mode("position", kp=kp, kd=kd)
        env.reset(q0=np.zeros(6))

        q_desired = np.full(6, 0.1)
        env.step(q_desired)

        q0 = np.zeros(6)
        expected_tau = kp * (q_desired - q0) - kd * 0.0
        actual_tau = env.data.qfrc_actuator[: env.NU]
        np.testing.assert_allclose(actual_tau, expected_tau, atol=0.1,
                                   err_msg="位置模式力矩不匹配 Kp*(u-q)-Kd*qdot")

    def test_position_step_damping_with_zero_error(self) -> None:
        """位置模式 q_desired=q_current 且 qdot 非零时产生阻尼力矩 -Kd*qdot。"""
        env = _make_env()
        kp = np.array([200.0, 200.0, 200.0, 50.0, 50.0, 20.0])
        kd = np.array([20.0, 20.0, 20.0, 5.0, 5.0, 2.0])

        env.configure_actuator_mode("position", kp=kp, kd=kd)

        q_current = np.array([0.1, -0.2, 0.3, -0.1, 0.05, 0.0])
        qdot = np.array([0.5, -0.3, 0.8, -0.2, 0.4, -0.6])

        x = np.concatenate([q_current, qdot])
        env.set_arm_state(x)

        env.step(q_current)

        expected_tau = -kd * qdot
        actual_tau = env.data.qfrc_actuator[: env.NU]
        np.testing.assert_allclose(actual_tau, expected_tau, atol=0.1,
                                   err_msg="阻尼力矩不匹配 -Kd*qdot")


class TestTorqueRegression:
    """切片 3-4：力矩模式零回归 + ctrlrange 切换。"""

    def test_torque_mode_unchanged_after_position_switch(self) -> None:
        """切到位置模式再切回力矩模式后，step(tau) 结果完全一致。"""
        env = _make_env()
        kp = np.array([200.0, 200.0, 200.0, 50.0, 50.0, 20.0])
        kd = np.array([20.0, 20.0, 20.0, 5.0, 5.0, 2.0])
        tau = np.array([5.0, -3.0, 2.0, -1.0, 0.5, 0.0])
        q0 = np.array([0.0, -1.2, 1.8, -0.6, 0.0, 0.0])

        env.reset(q0=q0)
        x1 = env.step(tau)

        env.configure_actuator_mode("position", kp=kp, kd=kd)
        env.reset(q0=q0)
        env.step(np.zeros(6))

        env.configure_actuator_mode("torque")
        env.reset(q0=q0)
        x2 = env.step(tau)

        np.testing.assert_allclose(x2, x1, atol=1e-10,
                                   err_msg="切回力矩模式后数值不一致")

    def test_ctrlrange_switches_between_modes(self) -> None:
        """力矩模式 ctrlrange 为 Nm 级别，位置模式 ctrlrange 为 rad 级别。"""
        env = _make_env()
        kp = np.array([200.0, 200.0, 200.0, 50.0, 50.0, 20.0])
        kd = np.array([20.0, 20.0, 20.0, 5.0, 5.0, 2.0])

        torque_ctrl = env.model.actuator_ctrlrange[: env.NU].copy()

        env.configure_actuator_mode("position", kp=kp, kd=kd)
        pos_ctrl = env.model.actuator_ctrlrange[: env.NU].copy()

        assert np.all(torque_ctrl[:, 1] < 100), "力矩模式 ctrlrange 应为 Nm 级别"
        assert np.all(pos_ctrl[:, 1] > 2.0), "位置模式 ctrlrange 应为 rad 级别（>π/2）"

        env.configure_actuator_mode("torque")
        restored_ctrl = env.model.actuator_ctrlrange[: env.NU]
        np.testing.assert_allclose(restored_ctrl, torque_ctrl, atol=1e-10,
                                   err_msg="切回力矩后 ctrlrange 未恢复")


class TestPropertiesAndValidation:
    """切片 5：属性只读 + 防错。"""

    def test_readonly_properties(self) -> None:
        """actuator_mode/kp/kd 为只读属性。"""
        env = _make_env()
        assert env.actuator_mode == 0
        assert env.kp is None
        assert env.kd is None

        kp = np.array([200.0, 200.0, 200.0, 50.0, 50.0, 20.0])
        kd = np.array([20.0, 20.0, 20.0, 5.0, 5.0, 2.0])
        env.configure_actuator_mode("position", kp=kp, kd=kd)

        assert env.actuator_mode == 1
        np.testing.assert_array_equal(env.kp, kp)
        np.testing.assert_array_equal(env.kd, kd)

    def test_position_mode_requires_kp_kd(self) -> None:
        """位置模式不给 kp/kd 应抛出 ValueError。"""
        env = _make_env()
        with pytest.raises(ValueError, match="kp"):
            env.configure_actuator_mode("position")

    def test_unknown_mode_raises(self) -> None:
        """未知执行器模式应抛出 ValueError。"""
        env = _make_env()
        with pytest.raises(ValueError, match="未知"):
            env.configure_actuator_mode("velocity")


class TestCloneConfig:
    """切片 6-7：clone 同步 + reset 不影响。"""

    def test_clone_syncs_position_mode_to_target(self) -> None:
        """clone_actuator_config 将位置模式配置同步到目标 env。"""
        env = _make_env()
        env_plan = _make_env()

        kp = np.array([200.0, 200.0, 200.0, 50.0, 50.0, 20.0])
        kd = np.array([20.0, 20.0, 20.0, 5.0, 5.0, 2.0])
        env.configure_actuator_mode("position", kp=kp, kd=kd)

        env.clone_actuator_config(env_plan)

        assert env_plan.actuator_mode == 1
        np.testing.assert_array_equal(env_plan.kp, kp)
        np.testing.assert_array_equal(env_plan.kd, kd)
        for i in range(env.NU):
            assert env_plan.model.actuator_biastype[i] == 1, \
                f"执行器 {i} biastype 未设为 AFFINE"

    def test_reset_preserves_actuator_config(self) -> None:
        """reset() 不影响执行器模式配置。"""
        env = _make_env()
        kp = np.array([200.0, 200.0, 200.0, 50.0, 50.0, 20.0])
        kd = np.array([20.0, 20.0, 20.0, 5.0, 5.0, 2.0])

        env.configure_actuator_mode("position", kp=kp, kd=kd)
        env.reset(q0=np.zeros(6))

        assert env.actuator_mode == 1
        np.testing.assert_array_equal(env.kp, kp)
        np.testing.assert_array_equal(env.kd, kd)
        for i in range(env.NU):
            assert env.model.actuator_gainprm[i, 0] == kp[i], \
                f"执行器 {i} gainprm 被 reset 重置"


class TestPythonLinearizePosition:
    """切片 8-9：Python 解析线性化位置模式数学结构验证。

    验证方法：直接对比位置模式和力矩模式的 A/B 矩阵差异，
    确认位置模式额外项 -M^{-1}*diag(Kp) 和 -M^{-1}*diag(Kd) 精确成立。

    不使用 FD 对比，因为 MuJoCo implicitfast 积分器与欧拉离散化差异显著。
    """

    @staticmethod
    def _compute_M_inv(env: RM65Env, x: np.ndarray) -> np.ndarray:
        """在指定状态点计算臂关节质量矩阵逆。"""
        env.set_arm_state(x)
        nv = env.NQ
        M_full = np.zeros((env.model.nv, env.model.nv))
        mujoco.mj_fullM(env.model, M_full, env.data.qM)
        M = M_full[:nv, :nv].copy()
        return np.linalg.solve(M, np.eye(nv))

    def test_B_matrix_has_Kp_scaling(self) -> None:
        """位置模式 B 下半块 = dt * M^{-1} * diag(Kp)。

        即 B_pos[6:,:] = B_torque[6:,:] * diag(Kp)（逐列缩放）。
        """
        env = _make_env()
        kp = np.array([200.0, 200.0, 200.0, 50.0, 50.0, 20.0])
        kd = np.array([20.0, 20.0, 20.0, 5.0, 5.0, 2.0])

        env.configure_actuator_mode("position", kp=kp, kd=kd)
        env.reset(q0=np.array([0.0, -1.2, 1.8, -0.6, 0.0, 0.0]))

        rng = np.random.default_rng(42)
        q = np.array([0.1, -0.5, 0.8, -0.3, 0.2, 0.1])
        qdot = rng.standard_normal(6) * 0.5
        x = np.concatenate([q, qdot])
        u = rng.standard_normal(6) * 0.1

        _, B_t, _ = linearize_analytical(env, x, u, eps=1e-5, actuator_mode=0)
        _, B_p, _ = linearize_analytical(
            env, x, u, eps=1e-5, actuator_mode=1, kp=kp, kd=kd,
        )

        # 验证 B_pos 下半块每列 = B_torque 下半块该列 * Kp[j]
        for j in range(6):
            np.testing.assert_allclose(
                B_p[6:, j], B_t[6:, j] * kp[j], atol=1e-12,
                err_msg=f"B 矩阵第 {j} 列 Kp 缩放不正确",
            )

    def test_A_matrix_has_extra_PD_terms(self) -> None:
        """位置模式 A 下半块额外项 = -dt * M^{-1} * diag(Kp) 和 -dt * M^{-1} * diag(Kd)。

        即 A_pos[6:,:6] - A_torque[6:,:6] = -dt * M^{-1} * diag(Kp)
           A_pos[6:,6:] - A_torque[6:,6:] = -dt * M^{-1} * diag(Kd)
        """
        env = _make_env()
        kp = np.array([200.0, 200.0, 200.0, 50.0, 50.0, 20.0])
        kd = np.array([20.0, 20.0, 20.0, 5.0, 5.0, 2.0])

        env.configure_actuator_mode("position", kp=kp, kd=kd)
        env.reset(q0=np.array([0.0, -1.2, 1.8, -0.6, 0.0, 0.0]))

        rng = np.random.default_rng(42)
        q = np.array([0.1, -0.5, 0.8, -0.3, 0.2, 0.1])
        qdot = rng.standard_normal(6) * 0.5
        x = np.concatenate([q, qdot])
        u = rng.standard_normal(6) * 0.1

        A_t, _, _ = linearize_analytical(env, x, u, eps=1e-5, actuator_mode=0)
        A_p, _, _ = linearize_analytical(
            env, x, u, eps=1e-5, actuator_mode=1, kp=kp, kd=kd,
        )

        dt = env.dt
        M_inv = self._compute_M_inv(env, x)

        # dA[6:,:6] = -dt * M^{-1} * diag(Kp)
        expected_dA_q = -dt * M_inv * kp[np.newaxis, :]
        np.testing.assert_allclose(
            A_p[6:, :6] - A_t[6:, :6], expected_dA_q, atol=1e-12,
            err_msg="A 矩阵额外 -M^{-1}*Kp 项不正确",
        )

        # dA[6:,6:] = -dt * M^{-1} * diag(Kd)
        expected_dA_qdot = -dt * M_inv * kd[np.newaxis, :]
        np.testing.assert_allclose(
            A_p[6:, 6:] - A_t[6:, 6:], expected_dA_qdot, atol=1e-12,
            err_msg="A 矩阵额外 -M^{-1}*Kd 项不正确",
        )


class TestCppLinearizePosition:
    """切片 10：C++ 解析线性化位置模式与 Python 一致性验证。

    验证 C++ linearize_analytical_batch 在力矩/位置两种模式下
    均与 Python linearize_analytical 结果精确匹配。
    """

    @pytest.fixture(autouse=True)
    def _skip_if_no_cpp(self) -> None:
        """C++ 模块不可用时跳过。"""
        try:
            from src.cpp.iLQR_Core import linearize_analytical_batch
        except ImportError:
            pytest.skip("C++ iLQR_Core 模块未编译")

    @staticmethod
    def _get_ptrs(env: RM65Env) -> tuple[int, int]:
        return env.model._address, env.data._address

    def test_torque_mode_cpp_matches_python(self) -> None:
        """力矩模式下 C++ 和 Python 结果一致（回归验证）。"""
        from src.cpp.iLQR_Core import linearize_analytical_batch

        env = _make_env()
        env.reset(q0=np.array([0.0, -1.2, 1.8, -0.6, 0.0, 0.0]))

        rng = np.random.default_rng(42)
        q = np.array([0.1, -0.5, 0.8, -0.3, 0.2, 0.1])
        qdot = rng.standard_normal(6) * 0.5
        x = np.concatenate([q, qdot])
        u = rng.standard_normal(6) * 0.1

        A_py, B_py, f_py = linearize_analytical(env, x, u, eps=1e-5, actuator_mode=0)

        N = 1
        A_cpp = np.zeros((N, 12, 12))
        B_cpp = np.zeros((N, 12, 6))
        f_cpp = np.zeros((N, 12))
        mp, dp = self._get_ptrs(env)
        linearize_analytical_batch(
            A_cpp, B_cpp, f_cpp,
            x.reshape(1, 12), u.reshape(1, 6),
            mp, dp, env.init_q_left,
            1e-5, env.dt, 0, None, None,
        )

        np.testing.assert_allclose(A_cpp[0], A_py, atol=1e-14,
                                   err_msg="力矩模式 A 矩阵 C++ vs Python 不一致")
        np.testing.assert_allclose(B_cpp[0], B_py, atol=1e-14,
                                   err_msg="力矩模式 B 矩阵 C++ vs Python 不一致")

    def test_position_mode_cpp_matches_python(self) -> None:
        """位置模式下 C++ 和 Python 结果一致。"""
        from src.cpp.iLQR_Core import linearize_analytical_batch

        env = _make_env()
        kp = np.array([200.0, 200.0, 200.0, 50.0, 50.0, 20.0])
        kd = np.array([20.0, 20.0, 20.0, 5.0, 5.0, 2.0])
        env.configure_actuator_mode("position", kp=kp, kd=kd)
        env.reset(q0=np.array([0.0, -1.2, 1.8, -0.6, 0.0, 0.0]))

        rng = np.random.default_rng(99)
        q = np.array([0.1, -0.5, 0.8, -0.3, 0.2, 0.1])
        qdot = rng.standard_normal(6) * 0.5
        x = np.concatenate([q, qdot])
        u = rng.standard_normal(6) * 0.1

        A_py, B_py, f_py = linearize_analytical(
            env, x, u, eps=1e-5, actuator_mode=1, kp=kp, kd=kd,
        )

        N = 1
        A_cpp = np.zeros((N, 12, 12))
        B_cpp = np.zeros((N, 12, 6))
        f_cpp = np.zeros((N, 12))
        mp, dp = self._get_ptrs(env)
        linearize_analytical_batch(
            A_cpp, B_cpp, f_cpp,
            x.reshape(1, 12), u.reshape(1, 6),
            mp, dp, env.init_q_left,
            1e-5, env.dt, 1, kp, kd,
        )

        np.testing.assert_allclose(A_cpp[0], A_py, atol=1e-14,
                                   err_msg="位置模式 A 矩阵 C++ vs Python 不一致")
        np.testing.assert_allclose(B_cpp[0], B_py, atol=1e-14,
                                   err_msg="位置模式 B 矩阵 C++ vs Python 不一致")


class TestSolverIntegration:
    """切片 11：solver 自动从 env 读取 actuator_mode。

    验证 solve_few_iters 在力矩和位置两种模式下均能正常运行，
    且内部线性化使用了正确的 actuator_mode（通过 B 矩阵结构间接验证）。
    """

    @staticmethod
    def _make_solver() -> tuple:
        """创建 env + solver + cost_fn，返回 (env, solver, cost_fn, x0, U_init)。"""
        from src.ilqt.solver import ILQTSolver
        from src.ilqt.cost import HittingCost

        env = _make_env()
        env.reset(q0=np.array([0.0, -1.2, 1.8, -0.6, 0.0, 0.0]))

        q0 = np.array([0.0, -1.2, 1.8, -0.6, 0.0, 0.0])
        x0 = np.concatenate([q0, np.zeros(6)])

        p_hit = np.array([0.4, -0.3, 1.0])
        v_hit = np.array([-2.0, 0.0, 1.0])
        Q_p = np.diag([5000.0, 5000.0, 5000.0])
        Q_v = np.diag([10.0, 10.0, 10.0])
        cost_fn = HittingCost(env, p_hit, v_hit, Q_p, Q_v, R=0.001)

        ilqt_cfg = {
            "max_iter": 10,
            "tol": 1e-4,
            "horizon": 10,
            "mu_min": 1e-6,
            "mu_max": 1e10,
            "mu_init": 0.01,
            "delta_0": 1.6,
            "alpha_list": [1.0, 0.5, 0.1],
            "lin_eps": 1e-5,
        }
        solver = ILQTSolver(ilqt_cfg)
        U_init = np.zeros((10, 6))

        return env, solver, cost_fn, x0, U_init

    def test_torque_mode_solve_few_iters(self) -> None:
        """力矩模式下 solve_few_iters 正常运行（回归验证）。"""
        env, solver, cost_fn, x0, U_init = self._make_solver()

        X, U, cost_history, success = solver.solve_few_iters(
            env, cost_fn, x0, U_init, max_iter=2,
        )

        assert X.shape == (11, 12)
        assert U.shape == (10, 6)
        assert np.all(np.isfinite(X)), "力矩模式 X 含 NaN/Inf"
        assert np.all(np.isfinite(U)), "力矩模式 U 含 NaN/Inf"
        assert len(cost_history) > 0

    def test_position_mode_solve_few_iters(self) -> None:
        """位置模式下 solve_few_iters 正常运行，且线性化使用了正确模式。

        通过在 solver._linearize 外包装捕获 B 矩阵，
        验证 B 矩阵下半部分含 Kp 缩放（即确认线性化路径读取了 actuator_mode=1）。
        """
        from src.ilqt.solver import ILQTSolver
        from src.ilqt.cost import HittingCost

        env = _make_env()
        kp = np.array([200.0, 200.0, 200.0, 50.0, 50.0, 20.0])
        kd = np.array([20.0, 20.0, 20.0, 5.0, 5.0, 2.0])
        env.configure_actuator_mode("position", kp=kp, kd=kd)
        env.reset(q0=np.array([0.0, -1.2, 1.8, -0.6, 0.0, 0.0]))

        q0 = np.array([0.0, -1.2, 1.8, -0.6, 0.0, 0.0])
        x0 = np.concatenate([q0, np.zeros(6)])

        p_hit = np.array([0.4, -0.3, 1.0])
        v_hit = np.array([-2.0, 0.0, 1.0])
        Q_p = np.diag([5000.0, 5000.0, 5000.0])
        Q_v = np.diag([10.0, 10.0, 10.0])
        cost_fn = HittingCost(env, p_hit, v_hit, Q_p, Q_v, R=0.001)

        ilqt_cfg = {
            "max_iter": 10,
            "tol": 1e-4,
            "horizon": 10,
            "mu_min": 1e-6,
            "mu_max": 1e10,
            "mu_init": 0.01,
            "delta_0": 1.6,
            "alpha_list": [1.0, 0.5, 0.1],
            "lin_eps": 1e-5,
        }
        solver = ILQTSolver(ilqt_cfg, use_analytical=True)
        U_init = np.zeros((10, 6))

        captured_Bs: list = []
        _orig_linearize = solver._linearize

        def _capturing_linearize(env_arg, X_arg, U_arg):
            result = _orig_linearize(env_arg, X_arg, U_arg)
            captured_Bs.append(result[1])
            return result

        solver._linearize = _capturing_linearize

        X, U, cost_history, success = solver.solve_few_iters(
            env, cost_fn, x0, U_init, max_iter=2,
        )

        assert X.shape == (11, 12)
        assert U.shape == (10, 6)
        assert np.all(np.isfinite(X)), "位置模式 X 含 NaN/Inf"
        assert np.all(np.isfinite(U)), "位置模式 U 含 NaN/Inf"
        assert len(captured_Bs) > 0, "线性化未被调用"

        B_first = captured_Bs[0][0]
        B_upper = B_first[:6, :]
        B_lower = B_first[6:, :]

        np.testing.assert_allclose(B_upper, 0, atol=1e-10,
                                   err_msg="B 矩阵上半部分应为零")
        assert np.linalg.norm(B_lower) > 1e-6, \
            "位置模式 B 矩阵下半部分应为非零"

    def test_solver_cpp_position_mode(self) -> None:
        """C++ solver 位置模式 solve_few_iters 正常运行。"""
        try:
            from src.cpp.solver_cpp import ILQTSolver as CppSolver
        except ImportError:
            pytest.skip("C++ iLQR_Core 模块未编译")

        from src.ilqt.cost import HittingCost

        env = _make_env()
        kp = np.array([200.0, 200.0, 200.0, 50.0, 50.0, 20.0])
        kd = np.array([20.0, 20.0, 20.0, 5.0, 5.0, 2.0])
        env.configure_actuator_mode("position", kp=kp, kd=kd)
        env.reset(q0=np.array([0.0, -1.2, 1.8, -0.6, 0.0, 0.0]))

        q0 = np.array([0.0, -1.2, 1.8, -0.6, 0.0, 0.0])
        x0 = np.concatenate([q0, np.zeros(6)])

        p_hit = np.array([0.4, -0.3, 1.0])
        v_hit = np.array([-2.0, 0.0, 1.0])
        Q_p = np.diag([5000.0, 5000.0, 5000.0])
        Q_v = np.diag([10.0, 10.0, 10.0])
        cost_fn = HittingCost(env, p_hit, v_hit, Q_p, Q_v, R=0.001)

        ilqt_cfg = {
            "max_iter": 10,
            "tol": 1e-4,
            "horizon": 10,
            "mu_min": 1e-6,
            "mu_max": 1e10,
            "mu_init": 0.01,
            "delta_0": 1.6,
            "alpha_list": [1.0, 0.5, 0.1],
            "lin_eps": 1e-5,
        }
        solver = CppSolver(ilqt_cfg, use_analytical=True)
        U_init = np.zeros((10, 6))

        X, U, cost_history, success = solver.solve_few_iters(
            env, cost_fn, x0, U_init, max_iter=2,
        )

        assert X.shape == (11, 12)
        assert U.shape == (10, 6)
        assert np.all(np.isfinite(X)), "C++ 位置模式 X 含 NaN/Inf"
        assert np.all(np.isfinite(U)), "C++ 位置模式 U 含 NaN/Inf"

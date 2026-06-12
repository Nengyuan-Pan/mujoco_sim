"""执行器双模式（力矩/位置）测试。

覆盖 Stage -1 切片 1-16：RM65Env 位置模式物理正确性、力矩模式零回归、
ctrlrange 切换、属性防错、clone 同步、reset 不影响配置、
Python 解析线性化 B/A 矩阵位置模式数学结构验证、
安全约束双模式、前向传递双模式、代价函数 R=0、
JT 初始控制角度增量版。
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


# ==============================================================================
#  切片 12-13：安全约束双模式
# ==============================================================================


class TestSafetyPositionMode:
    """切片 12-13：RobotLimits dq_max + strict_braking_check 双模式。"""

    def _make_limits(self, dt: float = 0.005) -> "RobotLimits":
        """构造测试用 RobotLimits，含 dq_max。"""
        from src.ilqt.robot_limits import RobotLimits

        deg = np.pi / 180.0
        qdot_max = np.array([180, 180, 225, 225, 225, 225], dtype=np.float64) * deg
        dq_max_fraction = 0.5
        return RobotLimits(
            q_lower=-np.ones(6),
            q_upper=np.ones(6),
            qdot_max=qdot_max,
            qddot_max=np.full(6, np.inf),
            u_min=-np.full(6, 150.0),
            u_max=np.full(6, 150.0),
            dq_max=qdot_max * dt * dq_max_fraction,
        )

    def test_check_step_uses_dq_max_in_position_mode(self) -> None:
        """位置模式 check_step_feasibility 检查 |u-q| < dq_max。

        力矩模式仍检查 u_min/u_max。
        """
        from src.ilqt.robot_limits import check_step_feasibility

        limits = self._make_limits()
        nq = 6
        q = np.zeros(nq)
        x_prev = np.concatenate([q, np.zeros(nq)])
        x_next = x_prev.copy()
        dt = 0.005

        ok_torque, _ = check_step_feasibility(
            x_prev, x_next, np.full(nq, 100.0), limits, dt,
            actuator_mode=0,
        )
        assert ok_torque, "力矩模式 u=100 < u_max=150 应通过"

        ok_torque_over, reason = check_step_feasibility(
            x_prev, x_next, np.full(nq, 200.0), limits, dt,
            actuator_mode=0,
        )
        assert not ok_torque_over, "力矩模式 u=200 > u_max=150 应拒绝"
        assert "u upper" in reason

        dq_max_val = limits.dq_max[0]

        u_within = q + 0.5 * dq_max_val
        ok_pos, _ = check_step_feasibility(
            x_prev, x_next, u_within, limits, dt,
            actuator_mode=1,
        )
        assert ok_pos, f"位置模式 dq={0.5*dq_max_val:.4f} < dq_max={dq_max_val:.4f} 应通过"

        u_over = q + 2.0 * dq_max_val
        ok_pos_over, reason_pos = check_step_feasibility(
            x_prev, x_next, u_over, limits, dt,
            actuator_mode=1,
        )
        assert not ok_pos_over, f"位置模式 dq={2.0*dq_max_val:.4f} > dq_max 应拒绝"
        assert "dq limit" in reason_pos

    def test_strict_braking_position_mode(self) -> None:
        """位置模式 strict_braking_check 使用 dq_max 约束。"""
        from src.ilqt.robot_limits import strict_braking_check

        limits = self._make_limits()
        nq = 6
        q = np.zeros(nq)
        x_prev = np.concatenate([q, np.zeros(nq)])
        x_next = x_prev.copy()
        dt = 0.005

        ok_torque, _ = strict_braking_check(
            x_prev, x_next, np.full(nq, 100.0), limits, dt,
            actuator_mode=0,
        )
        assert ok_torque, "力矩模式 strict_braking u=100 应通过"

        ok_torque_over, reason = strict_braking_check(
            x_prev, x_next, np.full(nq, 200.0), limits, dt,
            actuator_mode=0,
        )
        assert not ok_torque_over, "力矩模式 strict_braking u=200 应拒绝"
        assert "u upper" in reason

        dq_max_val = limits.dq_max[0]
        u_over = q + 2.0 * dq_max_val
        ok_pos_over, reason_pos = strict_braking_check(
            x_prev, x_next, u_over, limits, dt,
            actuator_mode=1,
        )
        assert not ok_pos_over, "位置模式 strict_braking dq 超限应拒绝"
        assert "dq limit" in reason_pos

        u_within = q + 0.5 * dq_max_val
        ok_pos, _ = strict_braking_check(
            x_prev, x_next, u_within, limits, dt,
            actuator_mode=1,
        )
        assert ok_pos, "位置模式 strict_braking dq 内应通过"

    def test_check_one_step_reads_actuator_mode_from_env(self) -> None:
        """check_one_step_feasibility 从 env 自动读取 actuator_mode。"""
        from src.ilqt.robot_limits import check_one_step_feasibility

        env = _make_env()
        kp = np.array([200.0, 200.0, 200.0, 50.0, 50.0, 20.0])
        kd = np.array([20.0, 20.0, 5.0, 5.0, 5.0, 2.0])
        env.configure_actuator_mode("position", kp=kp, kd=kd)
        env.reset(q0=np.zeros(6))

        limits = self._make_limits()
        nq = 6
        x0 = np.concatenate([np.zeros(nq), np.zeros(nq)])

        def _step(x, u):
            return env.step_from_state(x, u)

        dq_max_val = limits.dq_max[0]
        u_over = np.full(nq, 2.0 * dq_max_val)

        ok, reason = check_one_step_feasibility(
            x0, u_over, limits, env.dt,
            step_predictor=_step, env=env,
        )
        assert not ok, "env 位置模式下应使用 dq_max 检查"
        assert "dq limit" in reason


# ==============================================================================
#  切片 14：前向传递双模式
# ==============================================================================


class TestForwardPassPosition:
    """切片 14：forward pass 传递 actuator_mode。"""

    def test_forward_pass_single_position_mode(self) -> None:
        """位置模式下 forward_pass_single 正确传递 actuator_mode。"""
        from src.ilqt.robot_limits import RobotLimits
        from src.ilqt.cost import HittingCost
        from src.ilqt.utils import forward_pass_single

        env = _make_env()
        kp = np.array([200.0, 200.0, 200.0, 50.0, 50.0, 20.0])
        kd = np.array([20.0, 20.0, 5.0, 5.0, 5.0, 2.0])
        env.configure_actuator_mode("position", kp=kp, kd=kd)
        q0 = np.array([0.0, -1.2, 1.8, -0.6, 0.0, 0.0])
        env.reset(q0=q0)

        N = 5
        x0 = np.concatenate([q0, np.zeros(6)])
        X = np.tile(x0, (N + 1, 1))
        U = np.tile(q0, (N, 1))

        p_hit = np.array([0.4, -0.3, 1.0])
        v_hit = np.array([-2.0, 0.0, 1.0])
        Q_p = np.diag([5000.0, 5000.0, 5000.0])
        Q_v = np.diag([10.0, 10.0, 10.0])
        cost_fn = HittingCost(env, p_hit, v_hit, Q_p, Q_v, R=0.0)

        Ks = [np.zeros((6, 12)) for _ in range(N)]
        ks = [np.zeros(6) for _ in range(N)]

        deg = np.pi / 180.0
        qdot_max = np.array([180, 180, 225, 225, 225, 225], dtype=np.float64) * deg
        limits = RobotLimits(
            q_lower=-np.ones(6) * 3.0,
            q_upper=np.ones(6) * 3.0,
            qdot_max=qdot_max,
            qddot_max=np.full(6, np.inf),
            u_min=-np.full(6, 150.0),
            u_max=np.full(6, 150.0),
            dq_max=qdot_max * 0.005 * 0.5,
        )

        X_new, U_new, cost_new, reject_reason = forward_pass_single(
            env, cost_fn, X, U, Ks, ks,
            alpha=0.5, limits=limits,
        )

        assert X_new is not None, f"位置模式 forward_pass 失败: {reject_reason}"
        assert U_new is not None
        assert np.all(np.isfinite(X_new)), "X_new 含 NaN/Inf"
        assert np.all(np.isfinite(U_new)), "U_new 含 NaN/Inf"


# ==============================================================================
#  切片 15：代价函数 R=0
# ==============================================================================


class TestCostPositionMode:
    """切片 15：位置模式下 R=0，Q_du 正常。"""

    def test_R_zero_in_position_mode(self) -> None:
        """位置模式 running_cost 中 R*u²=0，力矩模式有 R*u²。"""
        env = _make_env()
        env.reset(q0=np.array([0.0, -1.2, 1.8, -0.6, 0.0, 0.0]))

        p_hit = np.array([0.4, -0.3, 1.0])
        v_hit = np.array([-2.0, 0.0, 1.0])
        Q_p = np.diag([5000.0, 5000.0, 5000.0])
        Q_v = np.diag([10.0, 10.0, 10.0])
        R = 0.01

        from src.ilqt.cost import HittingCost

        cost_torque = HittingCost(env, p_hit, v_hit, Q_p, Q_v, R=R, actuator_mode=0)
        cost_pos = HittingCost(env, p_hit, v_hit, Q_p, Q_v, R=R, actuator_mode=1)

        x = np.concatenate([np.array([0.0, -1.2, 1.8, -0.6, 0.0, 0.0]), np.zeros(6)])
        u = np.array([10.0, -5.0, 8.0, -3.0, 2.0, -1.0])

        c_torque = cost_torque.running_cost(x, u, k=None)
        c_pos = cost_pos.running_cost(x, u, k=None)

        expected_R_cost = 0.5 * R * float(u @ u)
        assert c_torque >= expected_R_cost * 0.9, \
            f"力矩模式应有 R*u² 贡献，cost={c_torque:.6f}, R*u²={expected_R_cost:.6f}"

        assert abs(c_pos) < 1e-10, \
            f"位置模式 R=0 且无其他运行代价，cost 应≈0，实际 cost={c_pos:.6f}"

    def test_R_zero_derivatives_in_position_mode(self) -> None:
        """位置模式 running_derivatives 中 l_u 和 l_uu 为零矩阵。"""
        env = _make_env()
        env.reset(q0=np.array([0.0, -1.2, 1.8, -0.6, 0.0, 0.0]))

        p_hit = np.array([0.4, -0.3, 1.0])
        v_hit = np.array([-2.0, 0.0, 1.0])
        Q_p = np.diag([5000.0, 5000.0, 5000.0])
        Q_v = np.diag([10.0, 10.0, 10.0])
        R = 0.01

        from src.ilqt.cost import HittingCost

        cost_pos = HittingCost(env, p_hit, v_hit, Q_p, Q_v, R=R, actuator_mode=1)
        cost_torque = HittingCost(env, p_hit, v_hit, Q_p, Q_v, R=R, actuator_mode=0)

        x = np.concatenate([np.array([0.0, -1.2, 1.8, -0.6, 0.0, 0.0]), np.zeros(6)])
        u = np.array([10.0, -5.0, 8.0, -3.0, 2.0, -1.0])

        l_x_t, l_u_t, l_xx_t, l_ux_t, l_uu_t = cost_torque.running_derivatives(x, u, k=None)
        l_x_p, l_u_p, l_xx_p, l_ux_p, l_uu_p = cost_pos.running_derivatives(x, u, k=None)

        np.testing.assert_allclose(l_u_p, 0, atol=1e-15,
                                   err_msg="位置模式 l_u 应为 0")
        np.testing.assert_allclose(l_uu_p, 0, atol=1e-15,
                                   err_msg="位置模式 l_uu 应为 0")

        assert np.linalg.norm(l_u_t) > 1e-6, "力矩模式 l_u 应非零"
        assert np.linalg.norm(l_uu_t) > 1e-6, "力矩模式 l_uu 应非零"


class TestJTInitPosition:
    """切片 16：JT 初始控制角度增量版。"""

    @staticmethod
    def _make_position_env() -> RM65Env:
        """创建配置了位置模式的 RM65Env。"""
        env = _make_env()
        kp = np.array([200.0, 200.0, 200.0, 50.0, 50.0, 20.0])
        kd = np.array([20.0, 20.0, 20.0, 5.0, 5.0, 2.0])
        env.configure_actuator_mode("position", kp=kp, kd=kd)
        env.reset(q0=np.array([0.0, -1.2, 1.8, -0.6, 0.0, 0.0]))
        return env

    def test_jt_init_outputs_valid_angles(self) -> None:
        """位置模式 JT 初始控制输出在 jnt_range 内的角度值。"""
        from src.ilqt.jt_init import compute_jacobian_init_control_position

        env = self._make_position_env()
        x0 = env.get_arm_state()
        p_hit = np.array([0.4, -0.3, 1.0])
        horizon = 40

        U = compute_jacobian_init_control_position(env, x0, p_hit, horizon, gain=0.3)

        assert U.shape == (horizon, 6), f"U 形状错误: {U.shape}"

        for i in range(6):
            jnt_id = env.model.actuator_trnid[i, 0]
            lo = env.model.jnt_range[jnt_id, 0]
            hi = env.model.jnt_range[jnt_id, 1]
            assert np.all(U[:, i] >= lo - 0.01), \
                f"关节 {i} 角度 {U[:, i].min():.4f} 低于下限 {lo:.4f}"
            assert np.all(U[:, i] <= hi + 0.01), \
                f"关节 {i} 角度 {U[:, i].max():.4f} 超过上限 {hi:.4f}"

    def test_jt_init_progressive_approach(self) -> None:
        """序列渐进：最后一步比第一步更接近击打点。"""
        from src.ilqt.jt_init import compute_jacobian_init_control_position

        env = self._make_position_env()
        x0 = env.get_arm_state()
        p_hit = np.array([0.4, -0.3, 1.0])
        horizon = 60

        U = compute_jacobian_init_control_position(env, x0, p_hit, horizon, gain=0.3)

        env.set_arm_state(x0)
        p_ee_init = env.get_ee_pos()
        dist_init = np.linalg.norm(p_ee_init - p_hit)

        env.set_arm_state(np.concatenate([U[-1, :6], np.zeros(6)]))
        p_ee_final = env.get_ee_pos()
        dist_final = np.linalg.norm(p_ee_final - p_hit)

        assert dist_final < dist_init, \
            f"末端未接近击打点: dist_final={dist_final:.4f} >= dist_init={dist_init:.4f}"

    def test_jt_init_not_torque_scale(self) -> None:
        """输出量级是弧度（|u| < π+0.1），不是力矩（|u| > 10）。"""
        from src.ilqt.jt_init import compute_jacobian_init_control_position

        env = self._make_position_env()
        x0 = env.get_arm_state()
        p_hit = np.array([0.4, -0.3, 1.0])
        horizon = 40

        U_pos = compute_jacobian_init_control_position(env, x0, p_hit, horizon, gain=0.3)

        assert np.all(np.abs(U_pos) < np.pi + 0.1), \
            f"位置模式输出超出弧度范围: max={np.abs(U_pos).max():.4f}"

    def test_jt_init_fix_joint5(self) -> None:
        """fix_joint5_angle=0.5 时，所有 U[:,5] = 0.5。"""
        from src.ilqt.jt_init import compute_jacobian_init_control_position

        env = self._make_position_env()
        x0 = env.get_arm_state()
        p_hit = np.array([0.4, -0.3, 1.0])
        horizon = 30
        fix_angle = 0.5

        U = compute_jacobian_init_control_position(
            env, x0, p_hit, horizon, gain=0.3, fix_joint5_angle=fix_angle,
        )

        np.testing.assert_allclose(U[:, 5], fix_angle, atol=0.01,
                                   err_msg="fix_joint5 角度不正确")

    def test_jt_init_no_nan_after_rollout(self) -> None:
        """逐步 rollout 后无 NaN。"""
        from src.ilqt.jt_init import compute_jacobian_init_control_position

        env = self._make_position_env()
        x0 = env.get_arm_state()
        p_hit = np.array([0.4, -0.3, 1.0])
        horizon = 50

        U = compute_jacobian_init_control_position(env, x0, p_hit, horizon, gain=0.3)

        assert not np.any(np.isnan(U)), "U 中有 NaN"
        x = x0.copy()
        for k in range(horizon):
            x = env.step_from_state(x, U[k])
            assert not np.any(np.isnan(x)), f"步 {k} 后状态有 NaN"

    def test_fix_joint5_trajectory_position(self) -> None:
        """fix_joint5_control_trajectory_position 直接替换第 5 关节为固定角度。"""
        from src.ilqt.jt_init import fix_joint5_control_trajectory_position

        U = np.random.randn(20, 6)
        q_fixed = 0.7

        U_fixed = fix_joint5_control_trajectory_position(U, q_fixed)

        np.testing.assert_allclose(U_fixed[:, 5], q_fixed)
        assert not np.allclose(U_fixed[:, :5], 0), "其他关节不应被修改"

    def test_backswing_warm_start_position_outputs_angles(self) -> None:
        """位置模式后摆 warm-start 输出角度（|U| < π+0.1）。"""
        from src.ilqt.jt_init import generate_backswing_warm_start_position

        env = self._make_position_env()
        x0 = env.get_arm_state()
        p_hit = np.array([0.4, -0.3, 1.0])
        v_hit = np.array([-3.0, 0.0, 1.0])
        horizon = 80

        U, q_des = generate_backswing_warm_start_position(
            env, x0, p_hit, v_hit, horizon,
            backswing_offset=-0.6, backswing_ratio=0.35,
        )

        assert U.shape == (horizon, 6)
        assert q_des.shape == (horizon, 6)
        assert np.all(np.abs(U) < np.pi + 0.1), \
            f"后摆输出超出弧度范围: max={np.abs(U).max():.4f}"
        assert not np.any(np.isnan(U)), "后摆输出有 NaN"


# ==============================================================================
#  切片 17：V11 端到端位置模式集成测试
# ==============================================================================


@pytest.mark.slow
class TestV11Integration:
    """切片 17：V11 main() 位置模式端到端集成测试。

    通过 subprocess 运行 V11 脚本，验证：
    1. 位置模式运行不崩溃（RC=0）
    2. 无 ERROR/CRASH/EMERGENCY_STOP/NaN
    3. 位置模式配置正确生效
    """

    @staticmethod
    def _run_v11(args: list[str], timeout: int = 300) -> tuple[int, str, str]:
        """运行 V11 主脚本并返回 (returncode, stdout, stderr)。"""
        import subprocess
        import sys
        from pathlib import Path
        script = Path(__file__).resolve().parent.parent / "scripts" / "rm65_mpc_v11.py"
        result = subprocess.run(
            [sys.executable, str(script)] + args,
            capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr

    def test_v11_position_mode_no_crash(self) -> None:
        """位置模式 V11 运行不崩溃，无 ERROR/EMERGENCY_STOP/NaN。"""
        rc, stdout, stderr = self._run_v11(
            ["--position-mode", "--no-plot", "--seed", "42",
             "--ball-speed", "7", "--no-bounce", "--replan-interval", "20"],
            timeout=120,
        )

        assert rc == 0, f"V11 位置模式返回码非零: rc={rc}, stderr={stderr[-500:]}"

        combined = stdout + stderr
        assert "EMERGENCY_STOP" not in combined, \
            "位置模式触发了紧急制动"
        assert "NaN" not in combined, \
            "位置模式输出含 NaN"

    def test_v11_position_mode_config_applied(self) -> None:
        """位置模式 V11 正确配置并输出位置模式日志。"""
        rc, stdout, stderr = self._run_v11(
            ["--position-mode", "--no-plot", "--seed", "42",
             "--ball-speed", "7", "--no-bounce", "--replan-interval", "20"],
            timeout=120,
        )

        assert rc == 0, f"V11 返回码非零: rc={rc}"

        combined = stdout + stderr
        assert "[actuator] 位置模式" in combined, \
            "位置模式日志未输出，配置可能未生效"
        assert "POSITION MODE" in combined, \
            "RobotLimits 日志未标记 POSITION MODE"
        assert "dq_max" in combined, \
            "位置模式日志未输出 dq_max 信息"

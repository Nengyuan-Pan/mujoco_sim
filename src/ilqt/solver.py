"""iLQT（迭代线性二次跟踪器）求解器。"""

import logging
import numpy as np
from src.sim.env import MujocoEnv
from src.dynamics.linearize import linearize_trajectory, linearize_analytical_trajectory
from src.ilqt.cost import HittingCost
from src.ilqt.robot_limits import RobotLimits
from src.ilqt.utils import (
    compute_total_cost,
    forward_pass_with_linesearch,
    forward_pass_single,
)

logger = logging.getLogger(__name__)


class ILQTSolver:
    """iLQT 求解器：后向-前向迭代主循环。"""

    def __init__(
        self,
        config: dict,
        use_analytical: bool = True,
        horizon_override: int | None = None,
    ) -> None:
        """初始化求解器。

        Args:
            config: iLQT 超参数字典，包含：
                max_iter, tol, horizon, mu_min, mu_max, mu_init,
                delta_0, alpha_list, lin_eps。
            use_analytical: 是否使用解析线性化（True=快速，False=有限差分）。
            horizon_override: 覆盖 config 中的 horizon 值（用于 MPC 短地平线）。
        """
        self.max_iter = int(config["max_iter"])
        self.tol = float(config["tol"])
        self.horizon = horizon_override if horizon_override is not None else int(config["horizon"])
        self.mu_min = float(config["mu_min"])
        self.mu_max = float(config["mu_max"])
        self.mu_init = float(config["mu_init"])
        self.delta_0 = float(config["delta_0"])
        self.alpha_list = [float(a) for a in config["alpha_list"]]
        self.lin_eps = float(config["lin_eps"])
        self.use_analytical = use_analytical

    def _linearize(self, env: MujocoEnv, X: np.ndarray, U: np.ndarray,
                   ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """根据配置选择线性化方法。"""
        if self.use_analytical:
            return linearize_analytical_trajectory(env, X, U, self.lin_eps)
        return linearize_trajectory(env, X, U, self.lin_eps)

    def solve(
        self,
        env: MujocoEnv,
        cost_fn: HittingCost,
        x0: np.ndarray,
        U_init: np.ndarray | None = None,
        limits: RobotLimits | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[float]]:
        """运行 iLQT 优化。

        Args:
            env: MuJoCo 环境实例。
            cost_fn: 代价函数实例。
            x0: 初始臂状态，形状 (12,)。
            U_init: 初始控制序列，形状 (N, 6)。若为 None 则用零初始化。
            limits: 真实机器人硬约束参数。None 表示不启用。

        Returns:
            (X_opt, U_opt, cost_history): 最优轨迹、最优控制、代价历史。
        """
        N = self.horizon
        n_u = env.NU

        if U_init is None:
            U = np.zeros((N, n_u))
        else:
            U = U_init.copy()

        X = self._rollout(env, x0, U)
        cost_old = compute_total_cost(env, cost_fn, X, U)
        cost_history: list[float] = [cost_old]

        mu = self.mu_init

        for iteration in range(self.max_iter):
            As, Bs, fs = self._linearize(env, X, U)

            l_xs, l_us, l_xxs, l_uxs, l_uus = self._running_cost_derivatives(
                cost_fn, X, U
            )
            l_x_N, l_xx_N = cost_fn.terminal_derivatives(X[-1])

            result = self._backward_pass(
                As, Bs, l_xs, l_us, l_xxs, l_uxs, l_uus, l_x_N, l_xx_N, mu
            )

            if result is None:
                mu = min(mu * self.delta_0, self.mu_max)
                if mu >= self.mu_max:
                    logger.warning("正则化参数达到上限，停止迭代")
                    break
                continue

            Ks, ks = result

            X_new, U_new, cost_new, accepted = forward_pass_with_linesearch(
                env, cost_fn, X, U, Ks, ks, self.alpha_list, cost_old,
                limits=limits,
            )

            if accepted:
                X = X_new
                U = U_new
                mu = max(mu / self.delta_0, self.mu_min)

                rel_improve = abs(cost_old - cost_new) / max(abs(cost_old), 1.0)
                cost_old = cost_new
                cost_history.append(cost_old)

                logger.info(
                    f"迭代 {iteration}: 代价 = {cost_old:.6f}, "
                    f"相对改进 = {rel_improve:.2e}, mu = {mu:.2e}"
                )

                if rel_improve < self.tol:
                    logger.info(f"收敛于迭代 {iteration}")
                    break
            else:
                mu = min(mu * self.delta_0, self.mu_max)
                cost_history.append(cost_old)
                logger.info(
                    f"迭代 {iteration}: 未改进, mu = {mu:.2e}"
                )
                if mu >= self.mu_max:
                    logger.warning("正则化参数达到上限，停止迭代")
                    break

        return X, U, cost_history

    def solve_few_iters(
        self,
        env: MujocoEnv,
        cost_fn: HittingCost,
        x0: np.ndarray,
        U_init: np.ndarray,
        max_iter: int = 3,
        skip_linesearch: bool = True,
        limits: RobotLimits | None = None,
        use_fast_lin: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, list[float], bool]:
        """运行指定次数的 iLQT 迭代（用于 MPC 实时规划）。

        Args:
            env: MuJoCo 环境实例。
            cost_fn: 代价函数实例。
            x0: 初始臂状态，形状 (12,)。
            U_init: 初始控制序列，形状 (N, 6)。
            max_iter: 最大迭代次数。
            skip_linesearch: 是否跳过线搜索（MPC 模式默认 True）。
            limits: 真实机器人硬约束参数。None 表示不启用。
            use_fast_lin: 是否使用快速线性化（跳过 ∂h/∂q, ∂h/∂qdot）。

        Returns:
            (X_opt, U_opt, cost_history, success):
              success=False 表示硬约束饱和，所有迭代 forward pass 均被拒绝。
        """
        U = U_init.copy()
        X = self._rollout(env, x0, U)
        cost_old = compute_total_cost(env, cost_fn, X, U)
        cost_history: list[float] = [cost_old]

        mu = self.mu_init
        rejection_streak: int = 0
        max_rejection_streak: int = 3
        solver_success: bool = True

        for iteration in range(max_iter):
            if use_fast_lin:
                As, Bs, fs = self._linearize_fast(env, X, U)
            else:
                As, Bs, fs = self._linearize(env, X, U)

            l_xs, l_us, l_xxs, l_uxs, l_uus = self._running_cost_derivatives(
                cost_fn, X, U
            )
            l_x_N, l_xx_N = cost_fn.terminal_derivatives(X[-1])

            result = self._backward_pass(
                As, Bs, l_xs, l_us, l_xxs, l_uxs, l_uus, l_x_N, l_xx_N, mu
            )

            if result is None:
                mu = min(mu * self.delta_0, self.mu_max)
                continue

            Ks, ks = result

            if skip_linesearch:
                X_new, U_new, cost_new, reject_reason = forward_pass_single(
                    env, cost_fn, X, U, Ks, ks, alpha=0.5,
                    limits=limits,
                )
                if X_new is None:
                    # 所有 alpha 被硬约束拒绝
                    rejection_streak += 1
                    mu = min(mu * self.delta_0, self.mu_max)
                    cost_history.append(cost_old)
                    logger.warning(
                        "迭代 %d: forward pass 硬约束拒绝 (%s), "
                        "rejection_streak=%d, mu=%.2e",
                        iteration, reject_reason, rejection_streak, mu,
                    )
                    if rejection_streak >= max_rejection_streak:
                        solver_success = False
                        logger.warning(
                            "迭代 %d: 连续 %d 次硬约束拒绝，求解失败",
                            iteration, rejection_streak,
                        )
                        break
                    continue
                else:
                    rejection_streak = 0

                # MPC 模式：始终接受更新，依赖重规划纠错
                if np.isfinite(X_new[-1]).all():
                    X = X_new
                    U = U_new
                    mu = max(mu / self.delta_0, self.mu_min)
                    cost_history.append(cost_new)
                else:
                    mu = min(mu * self.delta_0, self.mu_max)
                    cost_history.append(cost_old)
            else:
                X_new, U_new, cost_new, accepted = forward_pass_with_linesearch(
                    env, cost_fn, X, U, Ks, ks, self.alpha_list, cost_old,
                    limits=limits,
                )

                if accepted:
                    X = X_new
                    U = U_new
                    mu = max(mu / self.delta_0, self.mu_min)
                    cost_old = cost_new
                    cost_history.append(cost_old)
                    rejection_streak = 0
                else:
                    rejection_streak += 1
                    mu = min(mu * self.delta_0, self.mu_max)
                    cost_history.append(cost_old)
                    if rejection_streak >= max_rejection_streak:
                        solver_success = False
                        break
                    if mu >= self.mu_max:
                        solver_success = False
                        break

        return X, U, cost_history, solver_success

    def _rollout(
        self, env: MujocoEnv, x0: np.ndarray, U: np.ndarray
    ) -> np.ndarray:
        """前向仿真获取轨迹（规划期间禁用球拍碰撞，避免干扰球轨迹）。"""
        has_collision_ctrl = hasattr(env, "set_arm_collision")
        if has_collision_ctrl:
            env.set_arm_collision(False)
        N = len(U)
        X = np.zeros((N + 1, env.NX))
        env.set_arm_state(x0)
        X[0] = x0.copy()
        for k in range(N):
            X[k + 1] = env.step_from_state(X[k], U[k])
        if has_collision_ctrl:
            env.set_arm_collision(True)
        return X

    def _running_cost_derivatives(
        self, cost_fn: HittingCost, X: np.ndarray, U: np.ndarray
    ) -> tuple[list, list, list, list, list]:
        """计算所有时间步的运行代价导数。"""
        N = len(U)
        l_xs, l_us, l_xxs, l_uxs, l_uus = [], [], [], [], []
        for k in range(N):
            if k > 0 and hasattr(cost_fn, 'set_u_prev'):
                cost_fn.set_u_prev(U[k - 1])
            lx, lu, lxx, lux, luu = cost_fn.running_derivatives(X[k], U[k], k)
            l_xs.append(lx)
            l_us.append(lu)
            l_xxs.append(lxx)
            l_uxs.append(lux)
            l_uus.append(luu)
        return l_xs, l_us, l_xxs, l_uxs, l_uus

    def _linearize_fast(
        self, env: MujocoEnv, X: np.ndarray, U: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """快速线性化（仅 M^{-1}，跳过 ∂h/∂q, ∂h/∂qdot 有限差分）。"""
        from src.dynamics.linearize import linearize_fast_trajectory
        return linearize_fast_trajectory(env, X, U)

    def _backward_pass(
        self,
        As: list[np.ndarray],
        Bs: list[np.ndarray],
        l_xs: list[np.ndarray],
        l_us: list[np.ndarray],
        l_xxs: list[np.ndarray],
        l_uxs: list[np.ndarray],
        l_uus: list[np.ndarray],
        l_x_N: np.ndarray,
        l_xx_N: np.ndarray,
        mu: float,
    ) -> tuple[list[np.ndarray], list[np.ndarray]] | None:
        """后向传递：计算增益矩阵 K_k, k_k。

        使用 np.linalg.solve 替代 Cholesky 分解求逆，减少开销。

        Args:
            As, Bs: 动力学线性化矩阵列表。
            l_xs, l_us, ...: 运行代价导数列表。
            l_x_N, l_xx_N: 终端代价导数。
            mu: 正则化参数。

        Returns:
            (Ks, ks) 增益列表，或 None（若正则化失败）。
        """
        N = len(As)
        n_u = Bs[0].shape[1]
        reg = mu * np.eye(n_u)

        Ks: list[np.ndarray] = []
        ks: list[np.ndarray] = []

        V_x = l_x_N.copy()
        V_xx = l_xx_N.copy()

        for k in range(N - 1, -1, -1):
            A_T = As[k].T
            B_T = Bs[k].T

            Q_x = l_xs[k] + A_T @ V_x
            Q_u = l_us[k] + B_T @ V_x
            Q_xx = l_xxs[k] + A_T @ V_xx @ As[k]
            Q_ux = l_uxs[k] + B_T @ V_xx @ As[k]
            Q_uu = l_uus[k] + B_T @ V_xx @ Bs[k]

            Q_uu_reg = Q_uu + reg

            try:
                # solve(Q_uu_reg, I) 等价于 Q_uu_reg^{-1}，但数值更稳定
                Q_uu_inv = np.linalg.solve(Q_uu_reg, np.eye(n_u))
            except np.linalg.LinAlgError:
                return None

            K_k = -Q_uu_inv @ Q_ux
            k_k = -Q_uu_inv @ Q_u

            Ks.insert(0, K_k)
            ks.insert(0, k_k)

            Q_ux_T = Q_ux.T
            V_x = Q_x - Q_ux_T @ Q_uu_inv @ Q_u
            V_xx = Q_xx - Q_ux_T @ Q_uu_inv @ Q_ux
            V_xx = 0.5 * (V_xx + V_xx.T)

        return Ks, ks

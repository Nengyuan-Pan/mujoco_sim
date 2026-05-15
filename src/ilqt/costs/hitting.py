"""网球击打场景的 iLQR 代价函数。

终端代价：惩罚末端位置、速度、拍面法向量偏离期望值。
运行代价：惩罚末端位置偏离 + 控制力矩 + 关节跟踪。
"""

from __future__ import annotations

import numpy as np

from src.ilqt.costs.base import EndEffectorCost


class HittingCost(EndEffectorCost):
    """网球击打场景代价函数。

    继承 EndEffectorCost 的末端执行器计算方法，
    实现位置/速度/法向量/控制力矩的复合代价。
    """

    def __init__(
        self,
        env,
        p_hit: np.ndarray,
        v_hit: np.ndarray,
        Q_p: np.ndarray,
        Q_v: np.ndarray,
        R: float,
        Q_p_running: float | None = None,
        R_joint_scale: dict[int, float] | None = None,
        q_des_traj: np.ndarray | None = None,
        Q_joint: dict[int, float] | None = None,
        R_schedule: np.ndarray | None = None,
        p_ball_running: np.ndarray | None = None,
        Q_n: float = 0.0,
        n_des: np.ndarray | None = None,
    ) -> None:
        """初始化代价函数。

        Args:
            env: MuJoCo 环境实例。
            p_hit: 期望击打位置（终端代价目标），形状 (3,)。
            v_hit: 期望击打速度，形状 (3,)。
            Q_p: 位置代价权重，形状 (3,) 或 (3,3)。
            Q_v: 速度代价权重，形状 (3,) 或 (3,3)。
            R: 控制代价权重（标量）。
            Q_p_running: 运行位置代价权重缩放（相对于 Q_p 的比例）。
            R_joint_scale: 关节控制代价缩放，格式 {关节索引: 缩放因子}。
            q_des_traj: 期望关节角度轨迹，形状 (N, 6)。
            Q_joint: 关节跟踪权重，格式 {关节索引: 权重}。
            R_schedule: 时变控制代价权重，形状 (N,)。
            p_ball_running: 运行位置代价目标，形状 (3,)。
            Q_n: 终端拍面法向量代价权重（标量，0=禁用）。
            n_des: 期望拍面法向量，形状 (3,)。若不传则不参与代价。
        """
        super().__init__(env)
        self.p_hit = p_hit.copy()
        self.v_hit = v_hit.copy()
        self._p_ball_running = p_ball_running.copy() if p_ball_running is not None else None
        self._Q_p_base = np.diag(Q_p) if Q_p.ndim == 1 else Q_p.copy()
        self._Q_v_base = np.diag(Q_v) if Q_v.ndim == 1 else Q_v.copy()
        self.Q_p = self._Q_p_base.copy()
        self.Q_v = self._Q_v_base.copy()
        self.R_mat = R * np.eye(self.NU)
        self._R_scalar = R
        self._R_joint_scale = R_joint_scale or {}
        for j_idx, scale in self._R_joint_scale.items():
            self.R_mat[j_idx, j_idx] *= scale
        self._Q_p_running_ratio = Q_p_running
        self._Q_p_running: np.ndarray | None = None
        self._rebuild_running_weight()
        self._q_des_traj = q_des_traj
        self._Q_joint = Q_joint or {}
        self._R_schedule = R_schedule
        self.Q_n = Q_n
        self.n_des = n_des.copy() if n_des is not None else None
        self._rebuild_combined_weight()

    @classmethod
    def from_config(
        cls,
        env,
        cfg: dict,
        p_hit: np.ndarray | None = None,
        v_hit: np.ndarray | None = None,
        n_des: np.ndarray | None = None,
    ) -> HittingCost:
        """从配置字典构造代价函数实例。

        Args:
            env: MuJoCo 环境实例。
            cfg: 代价配置字典（来自 yaml 文件的 grouped 结构）。
            p_hit: 期望击打位置，形状 (3,)。运行时参数。
            v_hit: 期望击打速度，形状 (3,)。运行时参数。
            n_des: 期望拍面法向量，形状 (3,)。运行时参数。

        Returns:
            HittingCost 实例。
        """
        terminal = cfg.get("terminal", {})
        Q_p = np.array(terminal.get("Q_p", [50000.0, 50000.0, 50000.0]))
        Q_v = np.array(terminal.get("Q_v", [200.0, 200.0, 200.0]))
        Q_n = float(terminal.get("Q_n", 0.0))

        control = cfg.get("control", {})
        R = float(control.get("R", 0.0001))

        running = cfg.get("running", {})
        Q_p_ratio = float(running.get("Q_p_ratio", 0.0))

        joint_tracking = cfg.get("joint_tracking", {})
        raw_qj = joint_tracking.get("Q_joint", {})
        Q_joint = {int(k): float(v) for k, v in raw_qj.items()} if raw_qj else None

        return cls(
            env=env,
            p_hit=p_hit if p_hit is not None else np.zeros(3),
            v_hit=v_hit if v_hit is not None else np.zeros(3),
            Q_p=Q_p,
            Q_v=Q_v,
            R=R,
            Q_p_running=Q_p_ratio,
            Q_n=Q_n,
            n_des=n_des,
            Q_joint=Q_joint,
        )

    def _rebuild_running_weight(self) -> None:
        """重建运行位置代价权重。"""
        if self._Q_p_running_ratio is not None and self._Q_p_running_ratio > 0:
            self._Q_p_running = self._Q_p_base * self._Q_p_running_ratio
        else:
            self._Q_p_running = None

    def _rebuild_combined_weight(self) -> None:
        """重建组合权重矩阵和期望向量。"""
        self.Q_h = np.block([
            [self.Q_p, np.zeros((3, 3))],
            [np.zeros((3, 3)), self.Q_v],
        ])
        self.h_des = np.concatenate([self.p_hit, self.v_hit])

    def update_weights(
        self,
        Q_p_scale: float = 1.0,
        Q_v_scale: float = 1.0,
    ) -> None:
        """更新代价权重缩放因子（用于 MPC 权重调度）。

        根据距离击打时刻的远近动态调整位置和速度权重。

        Args:
            Q_p_scale: 位置权重缩放因子。
            Q_v_scale: 速度权重缩放因子。
        """
        self.Q_p = self._Q_p_base * Q_p_scale
        self.Q_v = self._Q_v_base * Q_v_scale
        self._rebuild_running_weight()
        self._rebuild_combined_weight()

    def set_q_des_traj(
        self,
        q_des_traj: np.ndarray | None,
        Q_joint: dict[int, float] | None = None,
    ) -> None:
        """更新期望关节轨迹（MPC 每步重规划时调用）。

        Args:
            q_des_traj: 期望关节角度轨迹，形状 (N, 6)。None 表示清除。
            Q_joint: 关节跟踪权重，例如 {0: 500.0}。
        """
        self._q_des_traj = q_des_traj
        if Q_joint is not None:
            self._Q_joint = Q_joint

    def set_R_schedule(self, R_schedule: np.ndarray | None) -> None:
        """更新时变 R 调度（MPC 每步重规划时调用）。

        Args:
            R_schedule: 时变 R 值，形状 (N,)。None 表示恢复常数 R。
        """
        self._R_schedule = R_schedule

    def update_target(
        self,
        p_hit: np.ndarray,
        v_hit: np.ndarray,
        p_ball_running: np.ndarray | None = None,
        n_des: np.ndarray | None = None,
    ) -> None:
        """更新击打目标（用于 MPC 重新规划）。

        Args:
            p_hit: 新的期望击打位置（终端代价目标），形状 (3,)。
            v_hit: 新的期望击打速度，形状 (3,)。
            p_ball_running: 新的运行位置代价目标，形状 (3,)。None 表示保持不变。
            n_des: 新的期望拍面法向量，形状 (3,)。None 表示保持不变。
        """
        self.p_hit = p_hit.copy()
        self.v_hit = v_hit.copy()
        if p_ball_running is not None:
            self._p_ball_running = p_ball_running.copy()
        if n_des is not None:
            self.n_des = n_des.copy()
        self._rebuild_combined_weight()

    def running_cost(self, x: np.ndarray, u: np.ndarray, k: int | None = None) -> float:
        """计算运行代价 l(x, u) = 0.5 * (p_ee - p_hit)^T Q_p_run (p_ee - p_hit) + 0.5 * u^T R u。

        Args:
            x: 臂状态，形状 (12,)。
            u: 控制力矩，形状 (6,)。
            k: 当前时间索引。None 表示使用常数 R，且不计算关节轨迹跟踪代价。

        Returns:
            运行代价值。
        """
        if k is not None and self._R_schedule is not None and k < len(self._R_schedule):
            R_k = self._R_schedule[k]
            if np.ndim(R_k) == 0:
                cost = 0.5 * R_k * (u @ u)
            else:
                cost = 0.5 * float(u @ (R_k * u))
        else:
            cost = 0.5 * u @ self.R_mat @ u
        if self._Q_p_running is not None:
            self.env.set_arm_state(x)
            p_ee = self.env.get_ee_pos()
            p_running_target = self._p_ball_running if self._p_ball_running is not None else self.p_hit
            dp = p_ee - p_running_target
            cost += 0.5 * dp @ self._Q_p_running @ dp
        if k is not None and self._q_des_traj is not None and k < len(self._q_des_traj):
            q_des = self._q_des_traj[k]
            dq = x[:self.NQ] - q_des
            for j_idx, weight in self._Q_joint.items():
                cost += 0.5 * weight * dq[j_idx] ** 2
        return cost

    def terminal_cost(self, x: np.ndarray) -> float:
        """计算终端代价 l_N(x)。

        l_N = 0.5*(h(x)-h_des)^T Q_h (h(x)-h_des) + 0.5*Q_n*(1-(n·n_des)²)
         其中 h(x)=[p_ee; v_ee]，n 为拍面法向量。

        Args:
            x: 臂状态，形状 (12,)。

        Returns:
            终端代价值。
        """
        h = self._compute_h(x)
        diff = h - self.h_des
        cost = 0.5 * diff @ self.Q_h @ diff
        if self.n_des is not None and self.Q_n > 0:
            n = self._compute_n(x)
            n_err = n - self.n_des
            cost += 0.5 * self.Q_n * float(n_err @ n_err)
        return cost

    def running_derivatives(
        self, x: np.ndarray, u: np.ndarray, k: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """计算运行代价的一阶和二阶导数。

        Args:
            x: 臂状态，形状 (12,)。
            u: 控制力矩，形状 (6,)。
            k: 当前时间索引。None 表示使用常数 R，且不计算关节轨迹跟踪导数。

        Returns:
            (l_x, l_u, l_xx, l_ux, l_uu)。
        """
        l_x = np.zeros(self.NX)
        l_xx = np.zeros((self.NX, self.NX))
        l_ux = np.zeros((self.NU, self.NX))

        if k is not None and self._R_schedule is not None and k < len(self._R_schedule):
            R_k = self._R_schedule[k]
            if np.ndim(R_k) == 0:
                l_u = R_k * u
                l_uu = R_k * np.eye(self.NU)
            else:
                l_u = R_k * u
                l_uu = np.diag(R_k)
        else:
            l_u = self.R_mat @ u
            l_uu = self.R_mat.copy()

        if self._Q_p_running is not None:
            self.env.set_arm_state(x)
            p_ee = self.env.get_ee_pos()
            J_p = self.env.get_ee_jacp()
            p_running_target = self._p_ball_running if self._p_ball_running is not None else self.p_hit
            dp = p_ee - p_running_target
            Q_run = self._Q_p_running

            l_x[:self.NQ] += J_p.T @ (Q_run @ dp)
            l_xx[:self.NQ, :self.NQ] += J_p.T @ Q_run @ J_p

        if k is not None and self._q_des_traj is not None and k < len(self._q_des_traj):
            q_des = self._q_des_traj[k]
            dq = x[:self.NQ] - q_des
            for j_idx, weight in self._Q_joint.items():
                l_x[j_idx] += weight * dq[j_idx]
                l_xx[j_idx, j_idx] += weight

        return l_x, l_u, l_xx, l_ux, l_uu

    def terminal_derivatives(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """计算终端代价的一阶和二阶导数（Gauss-Newton 近似）。

        Returns:
            (l_x_N, l_xx_N)。
        """
        h = self._compute_h(x)
        J_h = self._compute_jacobian_h(x)
        diff = h - self.h_des

        l_x = J_h.T @ self.Q_h @ diff
        l_xx = J_h.T @ self.Q_h @ J_h

        if self.n_des is not None and self.Q_n > 0:
            n = self._compute_n(x)
            J_n = self._compute_jacobian_n(x)
            n_err = n - self.n_des
            l_x += self.Q_n * (J_n.T @ n_err)
            l_xx += self.Q_n * (J_n.T @ J_n)

        return l_x, l_xx

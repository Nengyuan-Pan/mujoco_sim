"""iLQT 代价函数：终端击打点代价 + 拍面法向量代价 + 运行代价 + 身体硬约束。"""

import numpy as np
from src.sim.env import MujocoEnv


class HittingCost:
    """网球击打场景的 iLQT 代价函数。

    终端代价：惩罚末端位置、速度、拍面法向量偏离期望值。
    运行代价：惩罚末端位置偏离 + 控制力矩 + 关节跟踪。
    """

    def __init__(
        self,
        env: MujocoEnv,
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
        body_avoidance: dict | None = None,
        joint_limits: dict[int, tuple[float | None, float | None]] | None = None,
        Q_joint_limit: float = 100000.0,
        right_arm_x_limit: dict | None = None,
        Q_qdot: float = 0.0,
        Q_qddot: float = 0.0,
        Q_du: float = 0.0,
        softmin_candidates: np.ndarray | None = None,
        softmin_beta: float = 5.0,
        softmin_weights: np.ndarray | None = None,
        maximize_v_at_midpoint: bool = False,
        v_maximize_direction: np.ndarray | None = None,
        Q_v_max: float = 5000.0,
        Q_v_max_eps: float = 0.01,
        Q_tcp_soft: float = 0.0,
        tcp_threshold: float = 1.44,
        max_tcp_speed: float = 1.8,
        Q_qdot_limit: float = 0.0,
        qdot_limit_thresholds: np.ndarray | None = None,
        actuator_mode: int = 0,
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
            body_avoidance: 身体硬约束配置字典。
            joint_limits: 关节角度安全范围，{索引: (下界, 上界)}。
                None 表示无边，如 {0: (None, -0.3)} 表示 q0 ≤ -0.3。
            Q_joint_limit: 关节限制约束权重，默认 100k。越高越硬。
            Q_qdot: 关节速度软平滑权重（0=禁用）。用于平滑轨迹而非安全保障。
            Q_qddot: 关节加速度软平滑权重（0=禁用）。θ̈ ≈ θ̇/dt。
            Q_du: 控制变化率软平滑权重（0=禁用）。||u_k - u_{k-1}||²。
            softmin_candidates: v6 多终端候选球位置，形状 (M, 3)。None 表示使用单一 p_hit。
            softmin_beta: v6 softmin 温度参数，值越大越接近 hard min。
            softmin_weights: v6 高斯权重，形状 (M,)。None 表示均匀权重。
            maximize_v_at_midpoint: v6 是否在中途步最大化速度（负线性代价）。
            v_maximize_direction: v6 速度最大化方向，形状 (3,)。
            Q_v_max: v6 速度最大化线性代价权重。
            Q_v_max_eps: v6 速度最大化二次正则化系数（防止无界速度）。
            Q_tcp_soft: v6 TCP 速度软惩罚权重（0=禁用）。
            tcp_threshold: v6 TCP 速度阈值（超过此值开始惩罚）。
            max_tcp_speed: v6 TCP 最大允许速度（用于导数归一化）。
            Q_qdot_limit: v6 关节速度阈值软惩罚权重（0=禁用）。
            qdot_limit_thresholds: v6 各关节速度阈值，形状 (6,)。超过此值开始惩罚。
        """
        self.env = env
        self.p_hit = p_hit.copy()
        self.v_hit = v_hit.copy()
        self._p_ball_running = p_ball_running.copy() if p_ball_running is not None else None
        self._Q_p_base = np.diag(Q_p) if Q_p.ndim == 1 else Q_p.copy()
        self._Q_v_base = np.diag(Q_v) if Q_v.ndim == 1 else Q_v.copy()
        self.Q_p = self._Q_p_base.copy()
        self.Q_v = self._Q_v_base.copy()
        self._actuator_mode = actuator_mode
        if actuator_mode == 1:
            self.R_mat = np.zeros((env.NU, env.NU))
        else:
            self.R_mat = R * np.eye(env.NU)
        self._R_scalar = R
        # 关节级控制代价缩放
        self._R_joint_scale = R_joint_scale or {}
        for j_idx, scale in self._R_joint_scale.items():
            self.R_mat[j_idx, j_idx] *= scale
        # 运行位置代价
        self._Q_p_running_ratio = Q_p_running
        self._Q_p_running: np.ndarray | None = None
        self._rebuild_running_weight()
        # 关节空间代价（后摆跟踪）
        self._q_des_traj = q_des_traj
        self._Q_joint = Q_joint or {}
        # 时变 R 调度（退火）
        self._R_schedule = R_schedule
        # 终端拍面法向量代价
        self.Q_n = Q_n
        self.n_des = n_des.copy() if n_des is not None else None
        # 身体圆柱体硬约束（四次方惩罚，近似硬约束）
        self._body_enabled: bool = False
        self._body_center: np.ndarray = np.zeros(2)
        self._body_radius: float = 0.0
        self._Q_body: float = 0.0
        self._avoid_body_names: list[str] = []
        self._avoid_body_ids: list[int] = []
        if body_avoidance is not None and body_avoidance.get("enabled", True):
            self._body_enabled = True
            self._body_center = np.array(body_avoidance.get("center_xy", [0.0, -0.08]), dtype=np.float64)
            self._body_radius = float(body_avoidance.get("radius", 0.18))
            self._Q_body = float(body_avoidance.get("Q_body", 500000.0))
            self._avoid_body_names = list(body_avoidance.get("avoid_points", ["r_link3", "r_link5"]))
        # 关节安全范围约束（二次型惩罚，防止臂摆入躯干区域）
        self._joint_limits: dict[int, tuple[float | None, float | None]] = joint_limits or {}
        self._Q_joint_limit = Q_joint_limit
        # X 平面墙约束：右臂关键 body 的 X 坐标必须 ≤ limit_x，防止臂越过身体中线
        self._x_limit_enabled: bool = False
        self._x_limit: float = 0.0
        self._Q_x_limit: float = 0.0
        self._x_limit_body_ids: list[int] = []
        self._x_limit_body_names: list[str] = []
        if right_arm_x_limit is not None and right_arm_x_limit.get("enabled", True):
            self._x_limit_enabled = True
            self._x_limit = float(right_arm_x_limit.get("limit_x", 0.0))
            self._Q_x_limit = float(right_arm_x_limit.get("Q", 500000.0))
            self._x_limit_body_names = list(right_arm_x_limit.get(
                "check_bodies", ["r_link3", "r_link5", "r_racket_body"]
            ))
        # 软平滑项（默认 0，不影响现有行为）
        self._Q_qdot = max(0.0, Q_qdot)
        self._Q_qddot = max(0.0, Q_qddot)
        self._Q_du = max(0.0, Q_du)
        self._Q_qdot_effective = self._Q_qdot
        self._Q_qddot_effective = self._Q_qddot
        self._Q_du_effective = self._Q_du
        self._u_prev: np.ndarray | None = None
        # v5: 中途位置目标（在 k_hit 步强制经过击球位置）
        self._midpoint_step: int | None = None
        self._midpoint_target: np.ndarray | None = None
        self._Q_midpoint: np.ndarray | None = None
        # v5: 中途速度目标（在 k_hit 步鼓励高速）
        self._midpoint_v_target: np.ndarray | None = None
        self._Q_midpoint_v: np.ndarray | None = None
        # v6: softmin 多终端候选
        self._softmin_candidates: np.ndarray | None = (
            softmin_candidates.copy() if softmin_candidates is not None else None
        )
        self._softmin_beta: float = softmin_beta
        self._softmin_weights: np.ndarray | None = (
            softmin_weights.copy() if softmin_weights is not None else None
        )
        self._softmin_v_des: np.ndarray | None = None
        self._softmin_n_des: np.ndarray | None = None
        self._softmin_alpha_cache: np.ndarray | None = None
        self._softmin_costs_cache: np.ndarray | None = None
        # v6: 速度最大化（负代价）
        self._maximize_v_at_midpoint: bool = maximize_v_at_midpoint
        self._v_maximize_direction: np.ndarray | None = (
            v_maximize_direction.copy() if v_maximize_direction is not None else None
        )
        self._Q_v_max: float = Q_v_max
        self._Q_v_max_eps: float = Q_v_max_eps
        # v6: TCP 速度软惩罚
        self._Q_tcp_soft: float = max(0.0, Q_tcp_soft)
        self._tcp_threshold: float = tcp_threshold
        self._max_tcp_speed: float = max_tcp_speed
        # v6: 关节速度阈值软惩罚
        self._Q_qdot_limit: float = max(0.0, Q_qdot_limit)
        self._qdot_limit_thresholds: np.ndarray | None = (
            qdot_limit_thresholds.copy() if qdot_limit_thresholds is not None else None
        )
        # 组合权重
        self._rebuild_combined_weight()

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
        ])  # (6, 6)
        self.h_des = np.concatenate([self.p_hit, self.v_hit])  # (6,)

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

    def set_u_prev(self, u_prev: np.ndarray) -> None:
        """设置上一帧控制量（用于 Q_du 软平滑项）。

        需要在 _running_cost_derivatives 中每步调用以更新状态。

        Args:
            u_prev: 上一步控制力矩，形状 (6,)。
        """
        self._u_prev = u_prev.copy()

    def set_midpoint_target(
        self,
        step: int | None,
        target: np.ndarray | None,
        Q_midpoint: np.ndarray | None = None,
        v_target: np.ndarray | None = None,
        Q_midpoint_v: np.ndarray | None = None,
    ) -> None:
        """设置中途位置+速度目标（v5: 在指定步强制经过击球位置并鼓励高速）。

        Args:
            step: 中途步索引。None 表示清除。
            target: 中途目标位置 (3,)。None 表示清除。
            Q_midpoint: 中途位置代价权重 (3,3) 或 (3,) 或 None 使用 Q_p。
            v_target: 中途目标速度 (3,)。
            Q_midpoint_v: 中途速度代价权重 (3,3) 或 (3,) 或 None 使用 Q_v。
        """
        self._midpoint_step = step
        self._midpoint_target = target.copy() if target is not None else None
        if Q_midpoint is not None:
            self._Q_midpoint = np.diag(Q_midpoint) if Q_midpoint.ndim == 1 else Q_midpoint.copy()
        else:
            self._Q_midpoint = self._Q_p_base.copy()
        self._midpoint_v_target = v_target.copy() if v_target is not None else None
        if Q_midpoint_v is not None:
            self._Q_midpoint_v = np.diag(Q_midpoint_v) if np.ndim(Q_midpoint_v) == 1 else Q_midpoint_v.copy()
        else:
            self._Q_midpoint_v = self._Q_v_base.copy()

    def set_smoothness_scale(
        self, qdot_scale: float, qddot_scale: float, du_scale: float,
    ) -> None:
        """动态调整软平滑项权重（用于分阶段权重策略）。

        实际权重 = 基准值 × scale。不影响 q/u 硬约束。

        Args:
            qdot_scale: Q_qdot 缩放因子 (>0)。
            qddot_scale: Q_qddot 缩放因子 (>0)。
            du_scale: Q_du 缩放因子 (>0)。
        """
        self._Q_qdot_effective = self._Q_qdot * qdot_scale
        self._Q_qddot_effective = self._Q_qddot * qddot_scale
        self._Q_du_effective = self._Q_du * du_scale

    def set_softmin_candidates(
        self,
        candidates: np.ndarray | None,
        v_des_candidates: np.ndarray | None = None,
        n_des_candidates: np.ndarray | None = None,
        weights: np.ndarray | None = None,
        beta: float | None = None,
    ) -> None:
        """动态更新 softmin 候选（重规划时调用）。

        每个候选拥有独立的位置、期望速度和期望法向量，
        softmin 在三维（位置+速度+法向量）上进行匹配。

        Args:
            candidates: 候选球位置，形状 (M, 3)。None 表示清除 softmin。
            v_des_candidates: 每候选的期望击球速度，形状 (M, 3)。
                None 表示所有候选共享 self.v_hit。
            n_des_candidates: 每候选的期望拍面法向量，形状 (M, 3)。
                None 表示所有候选共享 self.n_des。
            weights: 高斯权重，形状 (M,)。None 表示均匀权重。
            beta: softmin 温度参数。None 表示保持不变。
        """
        self._softmin_candidates = candidates.copy() if candidates is not None else None
        self._softmin_v_des = v_des_candidates.copy() if v_des_candidates is not None else None
        self._softmin_n_des = n_des_candidates.copy() if n_des_candidates is not None else None
        self._softmin_weights = weights.copy() if weights is not None else None
        if beta is not None:
            self._softmin_beta = beta

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
        if self._actuator_mode == 1:
            cost = 0.0
        elif k is not None and self._R_schedule is not None and k < len(self._R_schedule):
            R_k = self._R_schedule[k]
            if np.ndim(R_k) == 0:
                cost = 0.5 * R_k * (u @ u)
            else:
                cost = 0.5 * float(u @ (R_k * u))
        else:
            cost = 0.5 * u @ self.R_mat @ u

        need_fk = (
            self._Q_p_running is not None
            or self._body_enabled
            or self._x_limit_enabled
            or self._Q_tcp_soft > 0
        )
        if need_fk:
            self.env.set_arm_state(x)

        if self._Q_p_running is not None:
            p_ee = self.env.get_ee_pos()
            p_running_target = self._p_ball_running if self._p_ball_running is not None else self.p_hit
            dp = p_ee - p_running_target
            cost += 0.5 * dp @ self._Q_p_running @ dp
        if k is not None and self._q_des_traj is not None and k < len(self._q_des_traj):
            q_des = self._q_des_traj[k]
            dq = x[:self.env.NQ] - q_des
            for j_idx, weight in self._Q_joint.items():
                cost += 0.5 * weight * dq[j_idx] ** 2
        if self._joint_limits:
            q = x[:self.env.NQ]
            for j, (lo, hi) in self._joint_limits.items():
                if lo is not None:
                    m = max(0.0, lo - q[j])
                    cost += 0.5 * self._Q_joint_limit * m * m
                if hi is not None:
                    m = max(0.0, q[j] - hi)
                    cost += 0.5 * self._Q_joint_limit * m * m
        cost += self._add_body_avoidance_cost(x, skip_set_state=True)
        cost += self._add_x_limit_cost(x, skip_set_state=True)
        # 软平滑项
        cost += self._add_smoothness_cost(x, u, k)
        # v6: TCP 速度软惩罚
        if self._Q_tcp_soft > 0:
            J_p_tcp = self.env.get_ee_jacp()[:, :self.env.NQ]
            tcp_vel = J_p_tcp @ x[self.env.NQ:]
            tcp_speed = float(np.linalg.norm(tcp_vel))
            if tcp_speed > self._tcp_threshold:
                excess = tcp_speed - self._tcp_threshold
                cost += 0.5 * self._Q_tcp_soft * excess * excess
        # v6: 关节速度阈值软惩罚
        if self._Q_qdot_limit > 0 and self._qdot_limit_thresholds is not None:
            qdot = x[self.env.NQ:]
            for j in range(self.env.NQ):
                excess = max(0.0, abs(qdot[j]) - self._qdot_limit_thresholds[j])
                if excess > 0:
                    cost += 0.5 * self._Q_qdot_limit * excess * excess
        # v5: 中途位置代价（v6: 支持速度最大化模式）
        is_midpoint = (k is not None and self._midpoint_step is not None
                       and k == self._midpoint_step)
        if is_midpoint:
            need_midpoint_fk = (self._midpoint_target is not None
                                or self._maximize_v_at_midpoint
                                or self._midpoint_v_target is not None)
            if need_midpoint_fk and not need_fk:
                self.env.set_arm_state(x)
            if self._midpoint_target is not None:
                p_ee = self.env.get_ee_pos()
                dp = p_ee - self._midpoint_target
                cost += 0.5 * float(dp @ self._Q_midpoint @ dp)
            if self._maximize_v_at_midpoint and self._v_maximize_direction is not None:
                v_ee = self.env.get_ee_vel()
                cost += -self._Q_v_max * np.dot(v_ee, self._v_maximize_direction)
                cost += 0.5 * self._Q_v_max_eps * np.dot(v_ee, v_ee)
            elif self._midpoint_v_target is not None:
                v_ee = self.env.get_ee_vel()
                dv = v_ee - self._midpoint_v_target
                cost += 0.5 * float(dv @ self._Q_midpoint_v @ dv)
        return cost

    def terminal_cost(self, x: np.ndarray) -> float:
        """计算终端代价 l_N(x)。

        当 softmin 候选启用时，使用 softmin 聚合多候选代价：
            cost = -log(Σ w_i * exp(-β * c_i)) / β
        其中 c_i = 0.5*||p-p_i||²_Qp + 0.5*||v-v_i||²_Qv + 0.5*Q_n*||n-n_i||²。
        每个候选拥有独立的位置、期望速度和期望法向量。

        Args:
            x: 臂状态，形状 (12,)。

        Returns:
            终端代价值。
        """
        self.env.set_arm_state(x)
        # softmin 多终端候选（per-candidate v/n）
        if self._softmin_candidates is not None and len(self._softmin_candidates) > 1:
            p_ee = self.env.get_ee_pos()
            v_ee = self.env.get_ee_vel()
            n_rack = self._compute_n_no_set(x) if (self.Q_n > 0 and self._softmin_n_des is not None) else None
            M = len(self._softmin_candidates)
            costs_i = np.zeros(M)
            for i in range(M):
                dp = p_ee - self._softmin_candidates[i]
                v_des_i = self._softmin_v_des[i] if self._softmin_v_des is not None else self.v_hit
                dv = v_ee - v_des_i
                costs_i[i] = 0.5 * dp @ self.Q_p @ dp + 0.5 * dv @ self.Q_v @ dv
                if self.Q_n > 0 and self._softmin_n_des is not None:
                    n_err = n_rack - self._softmin_n_des[i]
                    costs_i[i] += 0.5 * self.Q_n * float(n_err @ n_err)
            w = self._softmin_weights if self._softmin_weights is not None else np.ones(M) / M
            beta = self._softmin_beta
            neg_beta_ci = -beta * costs_i
            m = np.max(neg_beta_ci)
            log_sum = m + np.log(np.sum(w * np.exp(neg_beta_ci - m)))
            cost = -log_sum / beta
            return cost
        # 原有单一目标终端代价
        h = self._compute_h_no_set(x)
        diff = h - self.h_des
        cost = 0.5 * diff @ self.Q_h @ diff
        if self.n_des is not None and self.Q_n > 0:
            n = self._compute_n_no_set(x)
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
        n_x = self.env.NX
        n_u = self.env.NU
        l_x = np.zeros(n_x)
        l_xx = np.zeros((n_x, n_x))
        l_ux = np.zeros((n_u, n_x))

        # 统一设置一次 arm state，后续所有 helper 不再重复调用
        need_fk = (
            self._Q_p_running is not None
            or self._body_enabled
            or self._x_limit_enabled
            or self._Q_tcp_soft > 0
        )
        if need_fk:
            self.env.set_arm_state(x)

        # 时变 R 调度（位置模式 R=0）
        if self._actuator_mode == 1:
            l_u = np.zeros(n_u)
            l_uu = np.zeros((n_u, n_u))
        elif k is not None and self._R_schedule is not None and k < len(self._R_schedule):
            R_k = self._R_schedule[k]
            if np.ndim(R_k) == 0:
                l_u = R_k * u
                l_uu = R_k * np.eye(n_u)
            else:
                l_u = R_k * u
                l_uu = np.diag(R_k)
        else:
            l_u = self.R_mat @ u
            l_uu = self.R_mat.copy()

        if self._Q_p_running is not None:
            p_ee = self.env.get_ee_pos()
            J_p = self.env.get_ee_jacp()
            p_running_target = self._p_ball_running if self._p_ball_running is not None else self.p_hit
            dp = p_ee - p_running_target
            Q_run = self._Q_p_running

            l_x[:6] += J_p.T @ (Q_run @ dp)
            l_xx[:6, :6] += J_p.T @ Q_run @ J_p

        # 关节空间跟踪代价（用于后摆引导）
        if k is not None and self._q_des_traj is not None and k < len(self._q_des_traj):
            q_des = self._q_des_traj[k]
            dq = x[:self.env.NQ] - q_des
            for j_idx, weight in self._Q_joint.items():
                l_x[j_idx] += weight * dq[j_idx]
                l_xx[j_idx, j_idx] += weight

        # 关节安全范围约束（二次型惩罚）
        if self._joint_limits:
            q = x[:self.env.NQ]
            for j, (lo, hi) in self._joint_limits.items():
                if lo is not None:
                    m = max(0.0, lo - q[j])
                    if m > 0:
                        l_x[j] -= self._Q_joint_limit * m
                        l_xx[j, j] += self._Q_joint_limit
                if hi is not None:
                    m = max(0.0, q[j] - hi)
                    if m > 0:
                        l_x[j] += self._Q_joint_limit * m
                        l_xx[j, j] += self._Q_joint_limit

        # 身体硬约束
        self._add_body_avoidance_derivatives(x, l_x, l_xx, skip_set_state=True)
        # X 平面墙约束（右臂不得越过中线）
        self._add_x_limit_derivatives(x, l_x, l_xx, skip_set_state=True)
        # 软平滑项
        self._add_smoothness_derivatives(x, u, k, l_x, l_u, l_xx, l_uu)
        # v6: TCP 速度软惩罚导数
        if self._Q_tcp_soft > 0:
            J_p_tcp = self.env.get_ee_jacp()[:, :self.env.NQ]
            tcp_vel = J_p_tcp @ x[self.env.NQ:]
            tcp_speed = float(np.linalg.norm(tcp_vel))
            if tcp_speed > self._tcp_threshold:
                excess = tcp_speed - self._tcp_threshold
                if tcp_speed > 1e-8:
                    grad_speed = tcp_vel / tcp_speed
                    nq = self.env.NQ
                    l_x[nq:] += self._Q_tcp_soft * excess * (J_p_tcp.T @ grad_speed)
                    H_tcp = self._Q_tcp_soft * np.outer(grad_speed, grad_speed) / tcp_speed
                    l_xx[nq:, nq:] += J_p_tcp.T @ H_tcp @ J_p_tcp
        # v6: 关节速度阈值软惩罚导数
        if self._Q_qdot_limit > 0 and self._qdot_limit_thresholds is not None:
            nq = self.env.NQ
            qdot = x[nq:]
            for j in range(nq):
                excess = max(0.0, abs(qdot[j]) - self._qdot_limit_thresholds[j])
                if excess > 0:
                    sign = 1.0 if qdot[j] >= 0 else -1.0
                    l_x[nq + j] += self._Q_qdot_limit * excess * sign
                    l_xx[nq + j, nq + j] += self._Q_qdot_limit

        # v5: 中途位置+速度代价导数（v6: 支持速度最大化模式）
        is_midpoint = (k is not None and self._midpoint_step is not None
                       and k == self._midpoint_step)
        if is_midpoint:
            nq = self.env.NQ
            need_midpoint_fk = (self._midpoint_target is not None
                                or self._maximize_v_at_midpoint
                                or self._midpoint_v_target is not None)
            if need_midpoint_fk and not need_fk:
                self.env.set_arm_state(x)
            # 位置代价导数
            if self._midpoint_target is not None:
                J_p = self.env.get_ee_jacp()
                p_ee = self.env.get_ee_pos()
                dp = p_ee - self._midpoint_target
                Q_mid = self._Q_midpoint
                l_x[:nq] += J_p.T @ (Q_mid @ dp)
                l_xx[:nq, :nq] += J_p.T @ Q_mid @ J_p
            # 速度代价导数（v6: 最大化模式 vs 跟踪模式）
            if self._maximize_v_at_midpoint and self._v_maximize_direction is not None:
                J_p = self.env.get_ee_jacp()
                d_follow = self._v_maximize_direction
                l_x[nq:] += -self._Q_v_max * (J_p[:, :nq].T @ d_follow)
                l_x[nq:] += self._Q_v_max_eps * (J_p[:, :nq].T @ J_p[:, :nq] @ x[nq:])
                l_xx[nq:, nq:] += self._Q_v_max_eps * (J_p[:, :nq].T @ J_p[:, :nq])
            elif self._midpoint_v_target is not None:
                J_p = self.env.get_ee_jacp()
                v_ee = self.env.get_ee_vel()
                dv = v_ee - self._midpoint_v_target
                Q_mid_v = self._Q_midpoint_v
                l_x[nq:] += J_p.T @ (Q_mid_v @ dv)
                l_xx[nq:, nq:] += J_p.T @ Q_mid_v @ J_p

        return l_x, l_u, l_xx, l_ux, l_uu

    def terminal_derivatives(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """计算终端代价的一阶和二阶导数（Gauss-Newton 近似）。

        当 softmin 候选启用时，使用加权平均导数：
            l_x = Σ α_i * (J_h^T @ Q_h @ diff_i + Q_n * J_n^T @ n_err_i)
            l_xx = Σ α_i * (J_h^T @ Q_h @ J_h + Q_n * J_n^T @ J_n)
                   = J_h^T @ Q_h @ J_h + Q_n * J_n^T @ J_h（Σα_i=1）
        其中 α_i = w_i * exp(-β*c_i) / Σ_j w_j * exp(-β*c_j)，
        diff_i = [p_ee - p_i; v_ee - v_des_i]，n_err_i = n - n_des_i。

        Returns:
            (l_x_N, l_xx_N)。
        """
        self.env.set_arm_state(x)

        # softmin 多终端候选导数（per-candidate v/n）
        if self._softmin_candidates is not None and len(self._softmin_candidates) > 1:
            p_ee = self.env.get_ee_pos()
            v_ee = self.env.get_ee_vel()
            h = np.concatenate([p_ee, v_ee])
            J_h = self._compute_jacobian_h_no_set(x)
            n_rack = self._compute_n_no_set(x) if (self.Q_n > 0 and self._softmin_n_des is not None) else None
            J_n = self._compute_jacobian_n_no_set(x) if (self.Q_n > 0 and self._softmin_n_des is not None) else None
            M = len(self._softmin_candidates)
            costs_i = np.zeros(M)
            diffs_i = np.zeros((M, 6))
            for i in range(M):
                v_des_i = self._softmin_v_des[i] if self._softmin_v_des is not None else self.v_hit
                h_des_i = np.concatenate([self._softmin_candidates[i], v_des_i])
                diff_i = h - h_des_i
                diffs_i[i] = diff_i
                costs_i[i] = 0.5 * diff_i @ self.Q_h @ diff_i
                if self.Q_n > 0 and self._softmin_n_des is not None:
                    n_err_i = n_rack - self._softmin_n_des[i]
                    costs_i[i] += 0.5 * self.Q_n * float(n_err_i @ n_err_i)
            w = self._softmin_weights if self._softmin_weights is not None else np.ones(M) / M
            beta = self._softmin_beta
            neg_beta_ci = -beta * costs_i
            m = np.max(neg_beta_ci)
            exp_vals = w * np.exp(neg_beta_ci - m)
            alpha = exp_vals / np.sum(exp_vals)
            # 缓存诊断信息
            self._softmin_alpha_cache = alpha.copy()
            self._softmin_costs_cache = costs_i.copy()
            # 加权梯度（位置+速度）
            weighted_diff = np.sum(alpha[:, None] * diffs_i, axis=0)
            l_x = J_h.T @ self.Q_h @ weighted_diff
            l_xx = J_h.T @ self.Q_h @ J_h
            # 拍面法向量代价导数（per-candidate，按 alpha 加权）
            if self.Q_n > 0 and self._softmin_n_des is not None and J_n is not None:
                weighted_n_err = np.sum(alpha[:, None] * (n_rack[None, :] - self._softmin_n_des), axis=0)
                l_x += self.Q_n * (J_n.T @ weighted_n_err)
                l_xx += self.Q_n * (J_n.T @ J_n)
            return l_x, l_xx

        # 原有单一目标终端导数
        h = self._compute_h_no_set(x)
        J_h = self._compute_jacobian_h_no_set(x)
        diff = h - self.h_des

        l_x = J_h.T @ self.Q_h @ diff
        l_xx = J_h.T @ self.Q_h @ J_h

        if self.n_des is not None and self.Q_n > 0:
            n = self._compute_n_no_set(x)
            J_n = self._compute_jacobian_n_no_set(x)
            n_err = n - self.n_des
            l_x += self.Q_n * (J_n.T @ n_err)
            l_xx += self.Q_n * (J_n.T @ J_n)

        return l_x, l_xx

    def _compute_h(self, x: np.ndarray) -> np.ndarray:
        """计算 h(x) = [p_ee; v_ee]，形状 (6,)。"""
        self.env.set_arm_state(x)
        return self._compute_h_no_set(x)

    def _compute_h_no_set(self, x: np.ndarray) -> np.ndarray:
        """计算 h(x) = [p_ee; v_ee]，不调用 set_arm_state（调用方已设置）。"""
        p_ee = self.env.get_ee_pos()
        v_ee = self.env.get_ee_vel()
        return np.concatenate([p_ee, v_ee])

    def _compute_n(self, x: np.ndarray) -> np.ndarray:
        """计算球拍面法向量，形状 (3,)。"""
        self.env.set_arm_state(x)
        return self._compute_n_no_set(x)

    def _compute_n_no_set(self, x: np.ndarray) -> np.ndarray:
        """计算球拍面法向量，不调用 set_arm_state。"""
        return self.env.get_ee_normal()

    def _compute_jacobian_n(self, x: np.ndarray) -> np.ndarray:
        """计算球拍法向量对状态的雅可比 (3, 12)。

        使用 MuJoCo 旋转雅可比 J_ω 分析求导，避免有限差分的噪声：
            Δn ≈ skew(−n) @ Δθ
            Δθ = J_ω @ Δq
            ∴ ∂n/∂q ≈ skew(−n) @ J_ω

        法向量不依赖 q̇，故后 6 列为零。
        """
        self.env.set_arm_state(x)
        return self._compute_jacobian_n_no_set(x)

    def _compute_jacobian_n_no_set(self, x: np.ndarray) -> np.ndarray:
        """计算球拍法向量雅可比，不调用 set_arm_state。"""
        n = self._compute_n_no_set(x)
        J_omega = self.env.get_ee_jacr()  # (3, 6)
        nx, ny, nz = -n[0], -n[1], -n[2]
        skew = np.array([
            [0, -nz,  ny],
            [nz,  0, -nx],
            [-ny, nx,  0],
        ])
        J_n = np.zeros((3, self.env.NX))
        J_n[:, :self.env.NQ] = skew @ J_omega
        return J_n

    def _compute_jacobian_h(self, x: np.ndarray) -> np.ndarray:
        """计算 h(x) 对 x 的雅可比矩阵（近似），形状 (6, 12)。

        使用 Gauss-Newton 近似：
        J_h ≈ [J_p,  0  ]
              [ 0,  J_p ]
        其中 J_p 是位置雅可比 (3, 6)。
        """
        self.env.set_arm_state(x)
        return self._compute_jacobian_h_no_set(x)

    def _compute_jacobian_h_no_set(self, x: np.ndarray) -> np.ndarray:
        """计算 h(x) 雅可比，不调用 set_arm_state。"""
        J_p = self.env.get_ee_jacp()  # (3, 6)
        n_x = self.env.NX
        J_h = np.zeros((6, n_x))
        J_h[:3, :6] = J_p
        J_h[3:, 6:] = J_p
        return J_h

    # ------------------------------------------------------------------
    #  身体硬约束（四次方势垒，近似硬约束）
    #  将躯干建模为竖直圆柱体：center_xy + radius。
    #  惩罚臂关键点 (r_link3, r_link5) 进入圆柱体半径内。
    # ------------------------------------------------------------------

    def _init_body_avoidance_ids(self) -> None:
        """延迟初始化规避关键点的 MuJoCo body ID。"""
        if self._avoid_body_ids or not self._avoid_body_names:
            return
        import mujoco
        model = self.env.model
        for name in self._avoid_body_names:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            self._avoid_body_ids.append(body_id)

    def _add_body_avoidance_cost(
        self, x: np.ndarray, skip_set_state: bool = False,
    ) -> float:
        """计算身体规避代价（四次方势垒）。

        cost = Σ 0.25 * Q_body * max(0, radius - d)⁴
        当 d ≪ radius 时代价急剧增长，形成近似的硬约束。
        """
        if not self._body_enabled:
            return 0.0
        self._init_body_avoidance_ids()
        if not skip_set_state:
            self.env.set_arm_state(x)
        cost = 0.0
        for body_id in self._avoid_body_ids:
            p = self.env.get_body_pos_by_id(body_id)
            dx = p[0] - self._body_center[0]
            dy = p[1] - self._body_center[1]
            d = np.sqrt(dx * dx + dy * dy)
            if d < self._body_radius:
                margin = self._body_radius - d
                cost += 0.25 * self._Q_body * margin ** 4
        return cost

    def _add_body_avoidance_derivatives(
        self, x: np.ndarray, l_x: np.ndarray, l_xx: np.ndarray,
        skip_set_state: bool = False,
    ) -> None:
        """计算身体规避代价的导数并累加到 l_x, l_xx（Gauss-Newton 近似）。

        四次方势垒 c = 0.25 * Q * m⁴，其中 m = max(0, radius - d)。
        ∂c/∂p = -Q * m³ * e/d     (e = [dx, dy, 0])
        GN Hessian: H_p ≈ Q * m² * (e e^T) / d²
        """
        if not self._body_enabled:
            return
        self._init_body_avoidance_ids()
        if not skip_set_state:
            self.env.set_arm_state(x)
        n_q = self.env.NQ
        for body_id in self._avoid_body_ids:
            p = self.env.get_body_pos_by_id(body_id)
            dx = p[0] - self._body_center[0]
            dy = p[1] - self._body_center[1]
            d = np.sqrt(dx * dx + dy * dy)
            if d < self._body_radius:
                margin = self._body_radius - d
                e = np.array([dx, dy, 0.0], dtype=np.float64)
                e_hat = e / (d + 1e-12)
                J_p = self.env.get_body_jacp_by_id(body_id)

                # 梯度: ∂c/∂p = -Q * m³ * e_hat
                grad_p = -self._Q_body * margin ** 3 * e_hat
                l_x[:n_q] += J_p.T @ grad_p

                # GN Hessian: H_p ≈ Q * m² * (e_hat)(e_hat)^T
                H_p = self._Q_body * margin ** 2 * np.outer(e_hat, e_hat)
                l_xx[:n_q, :n_q] += J_p.T @ H_p @ J_p

    # ------------------------------------------------------------------
    #  X 平面墙约束：右臂关键 body 的 X ≤ limit_x
    #  直接限制肘、腕、球拍的 X 坐标，防止臂越过身体中线。
    # ------------------------------------------------------------------

    def _init_x_limit_ids(self) -> None:
        """延迟初始化 X 墙约束的 body ID。"""
        if self._x_limit_body_ids or not self._x_limit_body_names:
            return
        import mujoco
        model = self.env.model
        for name in self._x_limit_body_names:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            self._x_limit_body_ids.append(body_id)

    def _add_x_limit_cost(self, x: np.ndarray, skip_set_state: bool = False) -> float:
        """计算 X 墙约束代价（二次型惩罚）。

        cost = Σ 0.5 * Q * max(0, px - limit_x)²
        仅 X 方向，比圆柱体规避更直接高效。
        """
        if not self._x_limit_enabled:
            return 0.0
        self._init_x_limit_ids()
        if not skip_set_state:
            self.env.set_arm_state(x)
        cost = 0.0
        for body_id in self._x_limit_body_ids:
            px = self.env.get_body_pos_by_id(body_id)[0]
            if px > self._x_limit:
                margin = px - self._x_limit
                cost += 0.5 * self._Q_x_limit * margin * margin
        return cost

    def _add_x_limit_derivatives(
        self, x: np.ndarray, l_x: np.ndarray, l_xx: np.ndarray,
        skip_set_state: bool = False,
    ) -> None:
        """计算 X 墙约束的导数并累加。

        ∂c/∂x[0] = Q * margin * [1, 0, 0] @ J_p
        GN Hessian 只有 (0,0) 分量非零。
        """
        if not self._x_limit_enabled:
            return
        self._init_x_limit_ids()
        if not skip_set_state:
            self.env.set_arm_state(x)
        n_q = self.env.NQ
        for body_id in self._x_limit_body_ids:
            p = self.env.get_body_pos_by_id(body_id)
            if p[0] > self._x_limit:
                margin = p[0] - self._x_limit
                grad_p = self._Q_x_limit * margin * np.array([1.0, 0.0, 0.0], dtype=np.float64)
                J_p = self.env.get_body_jacp_by_id(body_id)
                l_x[:n_q] += J_p.T @ grad_p
                # GN Hessian: Q * diag(1,0,0)
                H_p = np.zeros((3, 3), dtype=np.float64)
                H_p[0, 0] = self._Q_x_limit
                l_xx[:n_q, :n_q] += J_p.T @ H_p @ J_p

    # ------------------------------------------------------------------
    #  软平滑项：qdot / qddot / du（默认权重 0，不影响现有行为）
    #  仅用于平滑轨迹和提前引导，不作为安全保障。
    # ------------------------------------------------------------------

    def _add_smoothness_cost(self, x: np.ndarray, u: np.ndarray, k: int | None) -> float:
        """计算软平滑代价。

        Q_qdot:  0.5 * Q_qdot * ||qdot||²
        Q_qddot: 0.5 * Q_qddot * ||qdot / dt||² （粗近似）
        Q_du:    0.5 * Q_du * ||u - u_prev||²
        """
        cost = 0.0
        nq = self.env.NQ
        dt = self.env.dt

        if self._Q_qdot_effective > 0:
            qdot = x[nq:]
            cost += 0.5 * self._Q_qdot_effective * float(qdot @ qdot)

        if self._Q_qddot_effective > 0:
            qdot = x[nq:]
            qddot_proxy = qdot / dt
            cost += 0.5 * self._Q_qddot_effective * float(qddot_proxy @ qddot_proxy)

        if self._Q_du_effective > 0 and k is not None and k > 0 and self._u_prev is not None:
            du = u - self._u_prev
            cost += 0.5 * self._Q_du_effective * float(du @ du)

        return cost

    def _add_smoothness_derivatives(
        self,
        x: np.ndarray,
        u: np.ndarray,
        k: int | None,
        l_x: np.ndarray,
        l_u: np.ndarray,
        l_xx: np.ndarray,
        l_uu: np.ndarray,
    ) -> None:
        """计算软平滑项的一阶和二阶导数并累加。

        Q_qdot:  Q_qdot * qdot ∈ l_x[nq:], Q_qdot * I ∈ l_xx[nq:, nq:]
        Q_qddot: Q_qddot * qdot / dt² ∈ l_x[nq:], Q_qddot/dt² * I ∈ l_xx[nq:, nq:]
        Q_du:    Q_du * (u - u_prev) ∈ l_u, Q_du * I ∈ l_uu
        """
        nq = self.env.NQ
        dt = self.env.dt

        if self._Q_qdot_effective > 0:
            qdot = x[nq:]
            l_x[nq:] += self._Q_qdot_effective * qdot
            l_xx_diag = np.eye(nq) * self._Q_qdot_effective
            l_xx[nq:, nq:] += l_xx_diag

        if self._Q_qddot_effective > 0:
            qdot = x[nq:]
            qddot_proxy = qdot / dt
            l_x[nq:] += self._Q_qddot_effective * qddot_proxy / dt
            l_xx_diag = np.eye(nq) * (self._Q_qddot_effective / (dt * dt))
            l_xx[nq:, nq:] += l_xx_diag

        if self._Q_du_effective > 0 and k is not None and k > 0 and self._u_prev is not None:
            du = u - self._u_prev
            l_u += self._Q_du_effective * du
            l_uu += self._Q_du_effective * np.eye(len(u))

"""代价函数抽象基类：定义 iLQR/iLQT 求解器所需的核心接口。"""

from abc import ABC, abstractmethod

import numpy as np


class BaseCost(ABC):
    """所有代价函数的抽象基类。

    定义 iLQR 求解器所需的 4 个核心方法（抽象），
    以及 MPC 运行时可选的 4 个 mutation 方法（no-op 默认）。
    """

    @abstractmethod
    def running_cost(self, x: np.ndarray, u: np.ndarray, k: int | None = None) -> float:
        """计算运行代价 l(x, u, k)。

        Args:
            x: 臂状态，形状 (12,)。
            u: 控制力矩，形状 (6,)。
            k: 当前时间索引。None 表示使用常数参数。

        Returns:
            运行代价值。
        """
        ...

    @abstractmethod
    def terminal_cost(self, x: np.ndarray) -> float:
        """计算终端代价 l_N(x)。

        Args:
            x: 臂状态，形状 (12,)。

        Returns:
            终端代价值。
        """
        ...

    @abstractmethod
    def running_derivatives(
        self, x: np.ndarray, u: np.ndarray, k: int | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """计算运行代价的一阶和二阶导数。

        Args:
            x: 臂状态，形状 (12,)。
            u: 控制力矩，形状 (6,)。
            k: 当前时间索引。

        Returns:
            (l_x, l_u, l_xx, l_ux, l_uu)。
        """
        ...

    @abstractmethod
    def terminal_derivatives(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """计算终端代价的一阶和二阶导数。

        Args:
            x: 臂状态，形状 (12,)。

        Returns:
            (l_x_N, l_xx_N)。
        """
        ...

    # ---- MPC 运行时 mutation 方法（no-op 默认） ----

    def update_weights(
        self,
        Q_p_scale: float = 1.0,
        Q_v_scale: float = 1.0,
    ) -> None:
        """更新代价权重缩放因子（用于 MPC 权重调度）。"""

    def update_target(
        self,
        p_hit: np.ndarray,
        v_hit: np.ndarray,
        p_ball_running: np.ndarray | None = None,
        n_des: np.ndarray | None = None,
    ) -> None:
        """更新击打目标（用于 MPC 重新规划）。"""

    def set_R_schedule(self, R_schedule: np.ndarray | None) -> None:
        """更新时变 R 调度（MPC 每步重规划时调用）。"""

    def set_q_des_traj(
        self,
        q_des_traj: np.ndarray | None,
        Q_joint: dict[int, float] | None = None,
    ) -> None:
        """更新期望关节轨迹（MPC 每步重规划时调用）。"""


class EndEffectorCost(BaseCost):
    """末端执行器代价函数中间基类。

    持有 env 引用和维度常量，提供末端位置/速度/法向量/雅可比的通用计算方法。
    所有需要末端执行器信息的代价函数（HittingCost 等）继承此类。
    """

    def __init__(self, env) -> None:
        """初始化。

        Args:
            env: MuJoCo 环境实例（MujocoEnv 或 RM65Env）。
        """
        self.env = env
        self.NX: int = env.NX
        self.NU: int = env.NU
        self.NQ: int = env.NQ

    def _compute_h(self, x: np.ndarray) -> np.ndarray:
        """计算 h(x) = [p_ee; v_ee]，形状 (6,)。"""
        self.env.set_arm_state(x)
        p_ee = self.env.get_ee_pos()
        v_ee = self.env.get_ee_vel()
        return np.concatenate([p_ee, v_ee])

    def _compute_n(self, x: np.ndarray) -> np.ndarray:
        """计算球拍面法向量，形状 (3,)。"""
        self.env.set_arm_state(x)
        return self.env.get_ee_normal()

    def _compute_jacobian_h(self, x: np.ndarray) -> np.ndarray:
        """计算 h(x) 对 x 的雅可比矩阵（近似），形状 (6, 12)。

        使用 Gauss-Newton 近似：
        J_h ≈ [J_p,  0  ]
              [ 0,  J_p ]
        其中 J_p 是位置雅可比 (3, 6)。
        """
        self.env.set_arm_state(x)
        J_p = self.env.get_ee_jacp()
        J_h = np.zeros((6, self.NX))
        J_h[:3, :self.NQ] = J_p
        J_h[3:, self.NQ:] = J_p
        return J_h

    def _compute_jacobian_n(self, x: np.ndarray) -> np.ndarray:
        """计算球拍法向量对状态的雅可比 (3, NX)。

        使用 MuJoCo 旋转雅可比 J_ω 分析求导：
            Δn ≈ skew(−n) @ Δθ
            Δθ = J_ω @ Δq
            ∴ ∂n/∂q ≈ skew(−n) @ J_ω

        法向量不依赖 q̇，故后 NQ 列为零。
        """
        n = self._compute_n(x)
        J_omega = self.env.get_ee_jacr()
        nx, ny, nz = -n[0], -n[1], -n[2]
        skew = np.array([
            [0, -nz, ny],
            [nz, 0, -nx],
            [-ny, nx, 0],
        ])
        J_n = np.zeros((3, self.NX))
        J_n[:, :self.NQ] = skew @ J_omega
        return J_n

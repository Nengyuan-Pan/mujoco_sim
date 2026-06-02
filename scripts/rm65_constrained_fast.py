"""RM-65 MPC+iLQR 网球击打（关节限速/限加速度硬约束版）。

在 rm65_mpc_fast.py 基础上新增：
  - 硬裁剪关节速度（J1-J2: 180°/s, J3-J6: 225°/s）
  - 硬裁剪关节加速度（600°/s²/步，RM65-B 额定值）
  - 代价函数软惩罚高速关节速度（引导 iLQR 自主回避超速）
  - 力矩裁剪（ctrlrange 硬约束）

用法：
  python scripts/rm65_constrained_fast.py --seed 42 --normal-flip --viewer
"""

import sys
import time
import argparse
import logging
import numpy as np
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sim.rm65_env import RM65Env
from src.tennis.ball import (
    generate_hittable_ball,
    generate_ball_to_target_box,
)
from src.tennis.hitting import (
    find_hitting_point_physics,
    compute_desired_hit_velocity,
)
from src.ilqt.cost import HittingCost
# C++ 版优先，回退到 Python 版
try:
    from src.cpp.solver_cpp import ILQTSolver
except ImportError:
    from src.ilqt.solver import ILQTSolver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── 关节约束常量 ──
RM65_VEL_LIMIT_RAD = np.deg2rad(np.array([180.0, 180.0, 225.0, 225.0, 225.0, 225.0]))
RM65_ACC_LIMIT_RAD = np.deg2rad(600.0)

# ── 身体圆柱体规避参数 ──
BODY_CENTER = np.array([0.0, -0.08], dtype=np.float64)  # 躯干竖直轴中心 (X, Y)
BODY_RADIUS = 0.18  # 躯干半径 0.10m + 安全余量 0.08m
BODY_AVOID_POINTS = [  # 需检查的臂关键点（规避碰撞）
    "r_link3",    # 肘关节（后摆时可能撞到躯干）
    "r_link5",    # 腕关节 2（前端时距身体最近）
]


class ConstrainedHittingCost(HittingCost):
    """带关节速度软惩罚 + 身体碰撞规避的代价函数。

    身体规避：将躯干区域建模为竖直圆柱体，惩罚臂关键点进入圆柱体半径内。
    """

    def __init__(
        self,
        *args,
        Q_qdot: float = 0.0,
        Q_body: float = 0.0,
        body_avoid_points: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._Q_qdot = Q_qdot
        self._vel_limit = RM65_VEL_LIMIT_RAD

        # 身体圆柱体碰撞规避
        self._Q_body = Q_body
        self._body_center = BODY_CENTER
        self._body_radius = BODY_RADIUS
        self._body_avoid_points: list[str] = body_avoid_points or []
        # 缓存 body id 和 geom id 用于位置计算
        self._avoid_body_ids: list[int] = []

    def _init_avoid_ids(self) -> None:
        """初始化规避关键点的 MuJoCo body ID 缓存。"""
        if self._avoid_body_ids or not self._body_avoid_points:
            return
        import mujoco
        for name in self._body_avoid_points:
            body_id = mujoco.mj_name2id(
                self.env.model, mujoco.mjtObj.mjOBJ_BODY, name
            )
            self._avoid_body_ids.append(body_id)

    def _body_avoidance_single(
        self, p: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """计算单点对身体的规避代价和任务空间梯度。

        Args:
            p: 检查点的世界坐标，形状 (3,)。

        Returns:
            (cost, grad_p): 代价标量和任务空间梯度 (3,)。
        """
        dx = p[0] - self._body_center[0]
        dy = p[1] - self._body_center[1]
        d = np.sqrt(dx * dx + dy * dy)

        if d >= self._body_radius:
            return 0.0, np.zeros(3)

        margin = self._body_radius - d
        cost = 0.5 * self._Q_body * margin * margin

        e = np.array([dx, dy, 0.0], dtype=np.float64)
        # ∂cost/∂p = −Q_body · margin · e / d
        grad_p = -self._Q_body * margin * e / (d + 1e-12)

        return cost, grad_p

    def _body_hessian_p(self, p: np.ndarray) -> np.ndarray:
        """计算身体规避代价对任务空间位置的 Gauss-Newton Hessian。

        H_p = Q_body · (e·e^T) / d²
        """
        dx = p[0] - self._body_center[0]
        dy = p[1] - self._body_center[1]
        d = np.sqrt(dx * dx + dy * dy)

        if d >= self._body_radius:
            return np.zeros((3, 3))

        e = np.array([dx, dy, 0.0], dtype=np.float64)
        return self._Q_body * np.outer(e, e) / (d * d + 1e-12)

    def running_cost(self, x, u, k=None):
        cost = super().running_cost(x, u, k)
        if self._Q_qdot > 0:
            qdot = x[self.env.NQ:]
            excess = np.maximum(np.abs(qdot) - self._vel_limit, 0.0)
            cost += 0.5 * self._Q_qdot * float(excess @ excess)
        if self._Q_body > 0 and self._body_avoid_points:
            self._init_avoid_ids()
            self.env.set_arm_state(x)
            for body_id in self._avoid_body_ids:
                p = self.env.get_body_pos_by_id(body_id)
                c, _ = self._body_avoidance_single(p)
                cost += c
        return cost

    def running_derivatives(self, x, u, k=None):
        l_x, l_u, l_xx, l_ux, l_uu = super().running_derivatives(x, u, k)
        if self._Q_qdot > 0:
            NQ = self.env.NQ
            qdot = x[NQ:]
            excess = np.maximum(np.abs(qdot) - self._vel_limit, 0.0)
            sign = np.sign(qdot)
            l_x[NQ:] += self._Q_qdot * sign * excess
            for j in range(self.env.NU):
                if excess[j] > 0:
                    l_xx[NQ + j, NQ + j] += self._Q_qdot
        if self._Q_body > 0 and self._body_avoid_points:
            self._init_avoid_ids()
            self.env.set_arm_state(x)
            for body_id in self._avoid_body_ids:
                p = self.env.get_body_pos_by_id(body_id)
                J_p = self.env.get_body_jacp_by_id(body_id)  # (3, 6)
                dx = p[0] - self._body_center[0]
                dy = p[1] - self._body_center[1]
                d = np.sqrt(dx * dx + dy * dy)
                if d < self._body_radius:
                    margin = self._body_radius - d
                    e = np.array([dx, dy, 0.0], dtype=np.float64)
                    grad_p = -self._Q_body * margin * e / (d + 1e-12)
                    l_x[:6] += J_p.T @ grad_p
                    H_p = self._Q_body * np.outer(e, e) / (d * d + 1e-12)
                    l_xx[:6, :6] += J_p.T @ H_p @ J_p
        return l_x, l_u, l_xx, l_ux, l_uu


def clip_joint_velocity(qdot: np.ndarray) -> np.ndarray:
    """硬裁剪关节速度到额定限速。"""
    return np.clip(qdot, -RM65_VEL_LIMIT_RAD, RM65_VEL_LIMIT_RAD)


def clip_joint_acceleration(qdot: np.ndarray, qdot_prev: np.ndarray, dt: float) -> np.ndarray:
    """硬裁剪关节加速度（单步速度变化量）。"""
    max_delta = RM65_ACC_LIMIT_RAD * dt
    delta = qdot - qdot_prev
    return qdot_prev + np.clip(delta, -max_delta, max_delta)


def fix_joint5_control(
    u: np.ndarray,
    q_fixed: float,
    x_current: np.ndarray,
    nq: int,
    kp: float = 300.0,
    kd: float = 30.0,
) -> np.ndarray:
    """将第 6 关节（索引 5）的控制力矩替换为 PD 保持力矩。

    Args:
        u: 原始控制力矩，形状 (6,) 或 (N, 6)。
        q_fixed: 第 6 关节固定角度。
        x_current: 当前臂状态，形状 (12,)。
        nq: 关节数（6）。
        kp: 位置增益。
        kd: 速度增益。

    Returns:
        修改后的控制力矩。
    """
    u = u.copy()
    q5_err = q_fixed - x_current[:nq][5]
    q5dot_err = -x_current[nq:][5]
    tau5 = kp * q5_err + kd * q5dot_err
    if u.ndim == 1:
        u[5] = tau5
    else:
        u[:, 5] = tau5
    return u


def fix_joint5_control_trajectory(
    U: np.ndarray,
    x0: np.ndarray,
    env: RM65Env,
    q_fixed: float,
    kp: float = 300.0,
    kd: float = 30.0,
) -> np.ndarray:
    """将整个控制序列的第 6 关节替换为 PD 保持力矩。

    通过逐步仿真获取每步的状态来计算 PD。

    Args:
        U: 原始控制序列，形状 (N, 6)。
        x0: 初始臂状态，形状 (12,)。
        env: 环境实例。
        q_fixed: 第 6 关节固定角度。
        kp: 位置增益。
        kd: 速度增益。

    Returns:
        修改后的控制序列。
    """
    U = U.copy()
    x = x0.copy()
    has_collision_ctrl = hasattr(env, "set_arm_collision")
    if has_collision_ctrl:
        env.set_arm_collision(False)
    # 保存球状态（fix_joint5_control_trajectory 会调用 step_from_state，会改变球状态）
    bq = env.BALL_QPOS_START
    bv = env.BALL_QVEL_START
    ball_qpos_save = env.data.qpos[bq:bq + 7].copy()
    ball_qvel_save = env.data.qvel[bv:bv + 6].copy()

    for k in range(len(U)):
        q5_err = q_fixed - x[:env.NQ][5]
        q5dot_err = -x[env.NQ:][5]
        U[k, 5] = kp * q5_err + kd * q5dot_err
        x = env.step_from_state(x, U[k])

    # 恢复球状态
    env.data.qpos[bq:bq + 7] = ball_qpos_save
    env.data.qvel[bv:bv + 6] = ball_qvel_save

    if has_collision_ctrl:
        env.set_arm_collision(True)
    return U


def load_config(config_path: Path) -> dict:
    """加载 YAML 配置文件。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_configs(base: dict, override: dict) -> dict:
    """递归合并两个配置字典，override 覆盖 base。"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def compute_jacobian_init_control(
    env: RM65Env,
    x0: np.ndarray,
    p_hit: np.ndarray,
    horizon: int,
    gain: float = 50.0,
    fix_joint5_angle: float | None = None,
) -> np.ndarray:
    """基于雅可比转置法计算初始控制序列。

    Args:
        env: RM-65 环境实例。
        x0: 初始臂状态，形状 (12,)。
        p_hit: 目标击打位置，形状 (3,)。
        horizon: 规划步数。
        gain: 雅可比转置增益。
        fix_joint5_angle: 若非 None，将第 6 关节固定在此角度。

    Returns:
        初始控制序列，形状 (horizon, 6)。
    """
    U = np.zeros((horizon, env.NU))
    x = x0.copy()
    ctrl_lo = env.model.actuator_ctrlrange[:env.NU, 0]
    ctrl_hi = env.model.actuator_ctrlrange[:env.NU, 1]

    has_collision_ctrl = hasattr(env, "set_arm_collision")
    if has_collision_ctrl:
        env.set_arm_collision(False)

    for k in range(horizon):
        env.set_arm_state(x)
        p_ee = env.get_ee_pos()
        J_p = env.get_ee_jacp()

        err = p_hit - p_ee
        dist = np.linalg.norm(err)
        scale = gain * min(dist, 0.5)
        tau = J_p.T @ err * scale

        tau -= 2.0 * x[env.NQ:]

        tau = np.clip(tau, ctrl_lo, ctrl_hi)
        U[k] = tau

        if fix_joint5_angle is not None:
            U[k, 5] = 300.0 * (fix_joint5_angle - x[:env.NQ][5]) - 30.0 * x[env.NQ:][5]

        x = env.step_from_state(x, U[k])

    if has_collision_ctrl:
        env.set_arm_collision(True)

    return U


def compute_ik_trajectory_init_control(
    env: RM65Env,
    x0: np.ndarray,
    p_hit: np.ndarray,
    v_hit_desired: np.ndarray,
    horizon: int,
    kp: float = 150.0,
    kd: float = 15.0,
    fix_joint5_angle: float | None = None,
) -> np.ndarray:
    """基于 IK 轨迹 + 速度前馈的初始控制序列。

    1. 求解目标位置 IK 得到 q_hit
    2. 在关节空间生成平滑轨迹：从 q0 到 q_hit，末端速度匹配 v_hit
    3. 用 PD + 前馈控制跟踪该轨迹

    Args:
        env: RM-65 环境实例。
        x0: 初始右臂状态，形状 (12,)。
        p_hit: 目标击打位置，形状 (3,)。
        v_hit_desired: 期望击打速度，形状 (3,)。
        horizon: 规划步数。
        kp: PD 位置增益。
        kd: PD 速度增益。
        fix_joint5_angle: 若非 None，将第 6 关节固定在此角度。

    Returns:
        初始控制序列，形状 (horizon, 6)。
    """
    NQ = env.NQ
    ctrl_lo = env.model.actuator_ctrlrange[:env.NU, 0]
    ctrl_hi = env.model.actuator_ctrlrange[:env.NU, 1]

    # IK 求解目标关节角
    q_hit = env.solve_ik(p_hit, q_init=x0[:NQ], max_iter=200, eps=1e-3)

    # 期望末端速度 → 期望关节速度（雅可比伪逆）
    env.set_arm_state(np.concatenate([q_hit, np.zeros(NQ)]))
    J_p = env.get_ee_jacp()
    qdot_hit = np.linalg.lstsq(J_p, v_hit_desired, rcond=None)[0]

    # 限制关节速度避免过大
    max_qdot = 3.0
    qdot_norm = np.linalg.norm(qdot_hit)
    if qdot_norm > max_qdot:
        qdot_hit *= max_qdot / qdot_norm

    # 关节空间五次多项式轨迹：q(t) = q0 + c1*α + c2*α² + c3*α³
    # 边界条件：q(0)=q0, qdot(0)=qdot0, q(1)=q_hit, qdot(1)=qdot_hit
    q0 = x0[:NQ]
    qdot0 = x0[NQ:]
    T = horizon

    # 求解五次多项式系数
    # q(α) = a0 + a1*α + a2*α² + a3*α³ + a4*α⁴ + a5*α⁵
    # q(0)=q0 → a0 = q0
    # q'(0)=qdot0/T → a1 = qdot0*T... 不对，α = t/T
    # dα = dt/T, 所以 dq/dt = (dq/dα) * (1/T)
    # qdot(0) = a1/T = qdot0 → a1 = qdot0 * T... 也不对
    # 用 α ∈ [0,1], t = α*T
    # q(α) = a0 + a1*α + a2*α² + a3*α³ + a4*α⁴ + a5*α⁵
    # q(0) = a0 = q0
    # q'(0)/T = a1/T = qdot0 → a1 = qdot0 * T
    # q(1) = q0 + a1 + a2 + a3 + a4 + a5 = q_hit
    # q'(1)/T = (a1 + 2*a2 + 3*a3 + 4*a4 + 5*a5)/T = qdot_hit

    # 加两个约束：q''(0)=0, q''(1)=0（零加速度边界）
    # q''(0)/T² = 2*a2/T² = 0 → a2 = 0
    # q''(1)/T² = (2*a2 + 6*a3 + 12*a4 + 20*a5)/T² = 0

    # 所以：a2 = 0
    # q0 + a1 + 0 + a3 + a4 + a5 = q_hit → a3 + a4 + a5 = q_hit - q0 - a1
    # (a1 + 2*0 + 3*a3 + 4*a4 + 5*a5)/T = qdot_hit → 3*a3 + 4*a4 + 5*a5 = qdot_hit*T - a1
    # (0 + 6*a3 + 12*a4 + 20*a5)/T² = 0 → 6*a3 + 12*a4 + 20*a5 = 0

    # 3 equations, 3 unknowns (a3, a4, a5)
    a0 = q0
    a1 = qdot0 * T

    dq = q_hit - q0 - a1
    dv = qdot_hit * T - a1

    # 解线性方程组：
    # a3 + a4 + a5 = dq
    # 3*a3 + 4*a4 + 5*a5 = dv
    # 6*a3 + 12*a4 + 20*a5 = 0
    A_mat = np.array([
        [1, 1, 1],
        [3, 4, 5],
        [6, 12, 20],
    ])
    b_vec = np.array([dq, dv, np.zeros(NQ)])

    # 对每个关节分别求解
    coeffs = np.zeros((3, NQ))
    for j in range(NQ):
        b_j = np.array([dq[j], dv[j], 0.0])
        coeffs[:, j] = np.linalg.solve(A_mat, b_j)

    a3 = coeffs[0]
    a4 = coeffs[1]
    a5 = coeffs[2]

    U = np.zeros((horizon, env.NU))
    x = x0.copy()

    has_collision_ctrl = hasattr(env, "set_arm_collision")
    if has_collision_ctrl:
        env.set_arm_collision(False)

    for k in range(horizon):
        alpha = (k + 1) / T
        alpha = min(alpha, 1.0)

        q_des = a0 + a1 * alpha + a3 * alpha**3 + a4 * alpha**4 + a5 * alpha**5
        qdot_des = (a1 + 3 * a3 * alpha**2 + 4 * a4 * alpha**3 + 5 * a5 * alpha**4) / T

        q_err = q_des - x[:NQ]
        qdot_err = qdot_des - x[NQ:]
        tau = kp * q_err + kd * qdot_err

        tau = np.clip(tau, ctrl_lo, ctrl_hi)
        U[k] = tau

        if fix_joint5_angle is not None:
            U[k, 5] = 300.0 * (fix_joint5_angle - x[:NQ][5]) - 30.0 * x[NQ:][5]

        x = env.step_from_state(x, U[k])

    if has_collision_ctrl:
        env.set_arm_collision(True)

    return U


def ik_pd_step(
    env: RM65Env,
    x_current: np.ndarray,
    p_target: np.ndarray,
    q_ref: np.ndarray | None = None,
    kp: float = 120.0,
    kd: float = 12.0,
    fix_joint5_angle: float | None = None,
) -> np.ndarray:
    """基于 IK + PD 的单步控制（远距模式）。

    先用 IK 求解目标关节角度，再用 PD 控制器跟踪。

    Args:
        env: RM-65 环境实例。
        x_current: 当前右臂状态，形状 (12,)。
        p_target: 目标位置，形状 (3,)。
        q_ref: IK 初始猜测，形状 (6,)。若为 None 则用当前 q。
        kp: PD 位置增益。
        kd: PD 速度增益。
        fix_joint5_angle: 若非 None，将第 6 关节固定在此角度。

    Returns:
        单步控制力矩，形状 (6,)。
    """
    q_ref = q_ref if q_ref is not None else x_current[:env.NQ]
    q_target = env.solve_ik(p_target, q_init=q_ref, max_iter=100, eps=2e-3)

    if fix_joint5_angle is not None:
        q_target[5] = fix_joint5_angle

    q_err = q_target - x_current[:env.NQ]
    qdot_err = -x_current[env.NQ:]
    tau = kp * q_err + kd * qdot_err

    ctrl_lo = env.model.actuator_ctrlrange[:env.NU, 0]
    ctrl_hi = env.model.actuator_ctrlrange[:env.NU, 1]
    return np.clip(tau, ctrl_lo, ctrl_hi)


def schedule_mpc_weights(
    steps_remaining: int,
    total_steps: int,
    Q_p_base: np.ndarray,
    Q_v_base: np.ndarray,
    far_threshold: int = 200,
    near_threshold: int = 25,
    Q_p_scale_far: float = 5.0,
    Q_v_scale_far: float = 0.1,
    Q_p_scale_near: float = 1.0,
    Q_v_scale_near: float = 10.0,
) -> tuple[float, float]:
    """根据剩余步数调度 Q_p 和 Q_v 权重。"""
    if steps_remaining > far_threshold:
        return Q_p_scale_far, Q_v_scale_far
    if steps_remaining <= near_threshold:
        return Q_p_scale_near, Q_v_scale_near
    ratio = (steps_remaining - near_threshold) / max(far_threshold - near_threshold, 1)
    Q_p_scale = Q_p_scale_near + (Q_p_scale_far - Q_p_scale_near) * ratio
    Q_v_scale = Q_v_scale_near + (Q_v_scale_far - Q_v_scale_near) * ratio
    return Q_p_scale, Q_v_scale


def compute_r_schedule(
    steps_remaining: int,
    base_R: float,
    decay_ratio: float = 0.30,
    joint1_extra_decay: float = 10.0,
) -> np.ndarray:
    """生成 R 退火调度 — 前段恒定，后段衰减到零，关节1衰减更快。

    网球挥拍：关节1（shoulder_pan）主导水平旋转，最后段需要比其他关节
    更激进地解除控制惩罚，才能甩出大摆幅高加速度。

    Args:
        steps_remaining: 剩余步数。
        base_R: 基础控制代价（来自 config）。
        decay_ratio: 衰减段占总步数的比例，默认 0.30。
        joint1_extra_decay: 关节1额外衰减倍率，默认 5（比其余关节快5倍）。

    Returns:
        R_schedule: 形状 (steps_remaining, 6)，每步每关节的控制代价权重。
    """
    if steps_remaining <= 0:
        return np.zeros((0, 6))

    decay_ratio = float(np.clip(decay_ratio, 0.0, 1.0))
    R_schedule = np.full((steps_remaining, 6), base_R)
    decay_start = int(steps_remaining * (1.0 - decay_ratio))
    if decay_start < steps_remaining:
        decay_len = steps_remaining - decay_start
        # 其余关节：线性衰减到 0
        R_other = base_R * (1.0 - np.linspace(0.0, 1.0, decay_len))
        # 关节1：指数倍更快衰减
        R_joint1 = base_R * (1.0 - np.linspace(0.0, 1.0, decay_len)) ** joint1_extra_decay
        R_schedule[decay_start:, 0] = R_joint1
        for j in range(1, 6):
            R_schedule[decay_start:, j] = R_other
    return R_schedule


def resample_control_sequence(
    U_old: np.ndarray,
    new_horizon: int,
) -> np.ndarray:
    """将旧控制序列重采样到新 horizon（线性插值）。"""
    old_horizon = len(U_old)
    if old_horizon == new_horizon:
        return U_old.copy()
    if old_horizon == 0:
        return np.zeros((new_horizon, U_old.shape[1]))

    n_u = U_old.shape[1]
    U_new = np.zeros((new_horizon, n_u))
    for k in range(new_horizon):
        t_frac = k / max(new_horizon - 1, 1) * (old_horizon - 1)
        idx_lo = int(np.floor(t_frac))
        idx_hi = min(idx_lo + 1, old_horizon - 1)
        alpha = t_frac - idx_lo
        U_new[k] = (1.0 - alpha) * U_old[idx_lo] + alpha * U_old[idx_hi]

    return U_new


def compute_joint1_backswing_trajectory(
    q1_current: float,
    qdot1_current: float,
    q1_hit: float,
    qdot1_hit: float,
    horizon: int,
    backswing_offset: float = -0.6,
    backswing_ratio: float = 0.35,
) -> np.ndarray:
    """生成关节1的“后摆→前挥”五次多项式轨迹。

    在三段时间区间上定义一条平滑连续轨迹：
      [0, α*h]:   后摆段 — q1从当前值向后摆动到 q1_mid = q1_current + backswing_offset
      [α*h, h]:   前挥段 — q1从后摆最深点加速向前到击打位

    简化为两段边界条件下的单条五次多项式：
      6个边界条件：q(0), q'(0), q(α*T)=q_mid, q'(α*T)=0, q(T)=q_hit, q'(T)=qdot_hit

    用归一化时间 τ = t/T 上的五次多项式 q(τ) = a0 + a1*τ + a2*τ² + a3*τ³ + a4*τ⁴ + a5*τ⁵
    6个边界条件 → 解 6×6 线性方程组。

    Args:
        q1_current: 关节1当前角度 (rad)。
        qdot1_current: 关节1当前速度 (rad/s)。
        q1_hit: 击打位关节1角度 (rad)。
        qdot1_hit: 击打位关节1速度 (rad/s)。
        horizon: 规划步数。
        backswing_offset: 后摆角度偏移（负值=远离球方向），默认 -0.6。
        backswing_ratio: 后摆占比 (0~1)，默认 0.35。

    Returns:
        q1_traj: 形状 (horizon,)，关节1期望角度。
    """
    if horizon <= 0:
        return np.zeros(0)

    T = float(horizon)
    alpha = float(np.clip(backswing_ratio, 0.05, 0.95))
    q_mid = q1_current + backswing_offset

    # 归一化时间轴 τ = t/T
    # 6个边界条件：
    #   q(0) = a0 = q1_current
    #   q'(0)/T = a1/T = qdot1_current  → a1 = qdot1_current * T
    #   q(α) = a0 + a1*α + a2*α² + a3*α³ + a4*α⁴ + a5*α⁵ = q_mid
    #   q'(α)/T = (a1 + 2*a2*α + 3*a3*α² + 4*a4*α³ + 5*a5*α⁴)/T = 0
    #   q(1) = a0 + a1 + a2 + a3 + a4 + a5 = q1_hit
    #   q'(1)/T = (a1 + 2*a2 + 3*a3 + 4*a4 + 5*a5)/T = qdot1_hit
    a0 = q1_current
    a1 = qdot1_current * T

    # 构造线性系统 A * [a2, a3, a4, a5]^T = b
    # q(1):   a2 + a3 + a4 + a5 = q1_hit - a0 - a1
    # q'(1):  2*a2 + 3*a3 + 4*a4 + 5*a5 = qdot1_hit*T - a1
    # q(α):   α²*a2 + α³*a3 + α⁴*a4 + α⁵*a5 = q_mid - a0 - a1*α
    # q'(α):  2α*a2 + 3α²*a3 + 4α³*a4 + 5α⁴*a5 = -a1
    alpha2, alpha3, alpha4, alpha5 = alpha**2, alpha**3, alpha**4, alpha**5

    A = np.array([
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 3.0, 4.0, 5.0],
        [alpha2, alpha3, alpha4, alpha5],
        [2 * alpha, 3 * alpha2, 4 * alpha3, 5 * alpha4],
    ])

    b = np.array([
        q1_hit - a0 - a1,
        qdot1_hit * T - a1,
        q_mid - a0 - a1 * alpha,
        -a1,
    ])

    coeffs_high = np.linalg.solve(A, b)
    a2, a3, a4, a5 = coeffs_high[0], coeffs_high[1], coeffs_high[2], coeffs_high[3]

    q1_traj = np.zeros(horizon)
    for k in range(horizon):
        tau = (k + 1) / T
        q1_traj[k] = a0 + a1 * tau + a2 * tau**2 + a3 * tau**3 + a4 * tau**4 + a5 * tau**5

    return q1_traj


def generate_backswing_warm_start(
    env: RM65Env,
    x0: np.ndarray,
    p_hit: np.ndarray,
    v_hit_desired: np.ndarray,
    horizon: int,
    backswing_offset: float = -0.6,
    backswing_ratio: float = 0.35,
    kp: float = 150.0,
    kd: float = 15.0,
    fix_joint5_angle: float | None = None,
    n_des: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """生成带后摆的关节空间轨迹 + PD 跟踪初始控制序列。

    步骤：
    1. IK 求解击打位关节角 q_hit
    2. 生成关节1的“后摆→前挥”五次多项式轨迹
    3. 其余关节从当前值线性插值到 q_hit
    4. 沿轨迹用 PD 控制生成 U_warm（A 方案：Warm-start）

    Args:
        env: RM-65 环境实例。
        x0: 初始右臂状态，形状 (12,)。
        p_hit: 目标击打位置，形状 (3,)。
        v_hit_desired: 期望击打速度，形状 (3,)。
        horizon: 规划步数。
        backswing_offset: 后摆角度偏移。
        backswing_ratio: 后摆占比。
        kp: PD 位置增益。
        kd: PD 速度增益。
        fix_joint5_angle: 若非 None，固定第 6 关节。

    Returns:
        (U_warm, q_des_traj): 控制序列 (N, 6)，关节轨迹 (N, 6)。
    """
    NQ = env.NQ
    NU = env.NU
    ctrl_lo = env.model.actuator_ctrlrange[:NU, 0]
    ctrl_hi = env.model.actuator_ctrlrange[:NU, 1]

    if horizon <= 0:
        return np.zeros((0, NU)), np.zeros((0, NQ))

    # 1. IK 求解击打位关节角
    q_hit = env.solve_ik(p_hit, q_init=x0[:NQ], max_iter=200, eps=1e-3)

    if fix_joint5_angle is not None:
        q_hit[5] = fix_joint5_angle

    # 1b. 腕关节调姿：将拍面法向量朝 n_des 方向对齐
    if n_des is not None:
        wrist_joints = [3, 4, 5]
        for _ in range(20):
            env.set_arm_state(np.concatenate([q_hit, np.zeros(NQ)]))
            n_cur = env.get_ee_normal()
            n_err = n_cur - n_des
            err_norm = np.linalg.norm(n_err)
            if err_norm < 0.01:
                break
            J_omega = env.get_ee_jacr()  # (3, 6)
            nx, ny, nz = -n_cur[0], -n_cur[1], -n_cur[2]
            skew = np.array([[0, -nz, ny], [nz, 0, -nx], [-ny, nx, 0]])
            J_n = skew @ J_omega  # (3, 6)
            J_n_wrist = J_n[:, wrist_joints]
            dq_wrist = -np.linalg.lstsq(J_n_wrist, n_err, rcond=None)[0]
            dq_wrist *= min(1.0, 0.02 / (np.linalg.norm(dq_wrist) + 1e-12))
            q_hit[wrist_joints] += dq_wrist

    # 2. 击打位关节速度（雅可比伪逆）
    env.set_arm_state(np.concatenate([q_hit, np.zeros(NQ)]))
    J_p_hit = env.get_ee_jacp()
    qdot_hit = np.linalg.lstsq(J_p_hit, v_hit_desired, rcond=None)[0]
    max_qdot = 3.0
    qdot_norm = np.linalg.norm(qdot_hit)
    if qdot_norm > max_qdot:
        qdot_hit *= max_qdot / qdot_norm

    # 3. 关节1后摆轨迹
    q1_current = x0[0]
    qdot1_current = x0[NQ]
    q1_traj = compute_joint1_backswing_trajectory(
        q1_current, qdot1_current,
        q_hit[0], qdot_hit[0],
        horizon,
        backswing_offset=backswing_offset,
        backswing_ratio=backswing_ratio,
    )

    # 4. 其余关节线性插值
    q_des_traj = np.zeros((horizon, NQ))
    for j in range(NQ):
        if j == 0:
            q_des_traj[:, j] = q1_traj
        else:
            q_des_traj[:, j] = np.linspace(x0[j], q_hit[j], horizon)

    # 5. PD 跟踪生成控制序列
    U = np.zeros((horizon, NU))
    x = x0.copy()

    has_collision_ctrl = hasattr(env, "set_arm_collision")
    if has_collision_ctrl:
        env.set_arm_collision(False)

    for k in range(horizon):
        q_des_k = q_des_traj[k]

        # 计算期望关节速度（差分）
        if k < horizon - 1:
            qdot_des_k = (q_des_traj[k + 1] - q_des_k) / env.dt
        else:
            qdot_des_k = qdot_hit

        err_joints = q_des_k - x[:NQ]
        err_joints_dot = qdot_des_k - x[NQ:]
        tau = kp * err_joints + kd * err_joints_dot

        if fix_joint5_angle is not None:
            tau[5] = 300.0 * (fix_joint5_angle - x[:NQ][5]) - 30.0 * x[NQ:][5]

        tau = np.clip(tau, ctrl_lo, ctrl_hi)
        U[k] = tau

        x = env.step_from_state(x, U[k])

    if has_collision_ctrl:
        env.set_arm_collision(True)

    return U, q_des_traj


def compute_task_space_trajectory(
    p0: np.ndarray,
    v0: np.ndarray,
    p_hit: np.ndarray,
    v_hit: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """在任务空间生成五次多项式轨迹，满足边界位置和速度约束。

    Args:
        p0: 初始末端位置，形状 (3,)。
        v0: 初始末端速度，形状 (3,)。
        p_hit: 目标击打位置，形状 (3,)。
        v_hit: 目标击打速度，形状 (3,)。
        horizon: 轨迹步数。

    Returns:
        (p_traj, v_traj): 位置和速度轨迹，形状均为 (horizon+1, 3)。
    """
    T = horizon
    p_traj = np.zeros((horizon + 1, 3))
    v_traj = np.zeros((horizon + 1, 3))

    dp = p_hit - p0
    dv = v_hit - v0

    for k in range(horizon + 1):
        t = k / T
        t = min(t, 1.0)

        # 五次多项式：p(t) = p0 + a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵
        # 边界条件：p(0)=p0, p'(0)=v0, p(1)=p_hit, p'(1)=v_hit, p''(0)=0, p''(1)=0
        a0 = p0
        a1 = v0 * T
        a2 = np.zeros(3)
        # 解 a3, a4, a5:
        # a3 + a4 + a5 = dp - a1
        # 3*a3 + 4*a4 + 5*a5 = dv*T - a1
        # 6*a3 + 12*a4 + 20*a5 = 0
        A_mat = np.array([[1, 1, 1], [3, 4, 5], [6, 12, 20]], dtype=np.float64)
        for dim in range(3):
            b = np.array([dp[dim] - a1[dim], dv[dim] * T - a1[dim], 0.0])
            coeffs = np.linalg.solve(A_mat, b)
            a3, a4, a5 = coeffs

            p_traj[k, dim] = a0[dim] + a1[dim]*t + a2[dim]*t**2 + a3*t**3 + a4*t**4 + a5*t**5
            v_traj[k, dim] = (a1[dim] + 2*a2[dim]*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4) / T

    return p_traj, v_traj


def impedance_control_step(
    env: RM65Env,
    x_current: np.ndarray,
    p_des: np.ndarray,
    v_des: np.ndarray,
    kp: float = 2000.0,
    kd: float = 80.0,
) -> np.ndarray:
    """任务空间阻抗控制器：J^T * (Kp*e_p + Kd*e_v) + 关节阻尼。

    Args:
        env: RM-65 环境实例。
        x_current: 当前右臂状态，形状 (12,)。
        p_des: 期望末端位置，形状 (3,)。
        v_des: 期望末端速度，形状 (3,)。
        kp: 位置刚度。
        kd: 速度阻尼。

    Returns:
        控制力矩，形状 (6,)。
    """
    env.set_arm_state(x_current)
    env.update_kinematics()
    p_ee = env.get_ee_pos()
    v_ee = env.get_ee_vel()
    J_p = env.get_ee_jacp()

    e_p = p_des - p_ee
    e_v = v_des - v_ee

    f_task = kp * e_p + kd * e_v
    tau = J_p.T @ f_task

    # 关节空间阻尼
    tau -= 5.0 * x_current[env.NQ:]

    ctrl_lo = env.model.actuator_ctrlrange[:env.NU, 0]
    ctrl_hi = env.model.actuator_ctrlrange[:env.NU, 1]
    return np.clip(tau, ctrl_lo, ctrl_hi)


def visualize_rm65_result(
    env: RM65Env,
    X: np.ndarray,
    U: np.ndarray,
    ball_positions_phys: np.ndarray,
    config: dict,
    init_q_left: np.ndarray,
    post_hit_steps: int = 80,
) -> None:
    """在 MuJoCo 查看器中可视化 RM-65 击打结果（含击打后球飞出效果）。

    Args:
        env: RM-65 环境实例。
        X: 右臂状态轨迹，形状 (N+1, 12)。
        U: 控制轨迹，形状 (N, 6)。
        ball_positions_phys: MuJoCo 物理球轨迹，形状 (M, 3)。
        config: 可视化配置。
        post_hit_steps: 击打后额外仿真步数。
    """
    import mujoco
    import mujoco.viewer

    N = len(U)
    dt = env.dt
    viewer_cfg = config.get("viewer", {})
    playback_speed = viewer_cfg.get("playback_speed", 1.0)
    loop = viewer_cfg.get("loop", True)

    cam_distance = viewer_cfg.get("camera_distance", 3.5)
    cam_elevation = viewer_cfg.get("camera_elevation", -15)
    cam_azimuth = viewer_cfg.get("camera_azimuth", 135)

    total_frames = len(ball_positions_phys)

    bq = env.BALL_QPOS_START
    NQ = env.NQ
    data = env.data
    model = env.model

    # 先将 data 重置到第 0 帧，避免查看器打开时显示终端状态
    data.qpos[:NQ] = X[0, :NQ]
    data.qvel[:NQ] = X[0, NQ:]
    data.qpos[NQ:NQ + env.LEFT_ARM_NQ] = init_q_left
    data.qpos[bq:bq + 3] = ball_positions_phys[0]
    data.qpos[bq + 3:bq + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)

    last_idx = -1

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = cam_distance
        viewer.cam.elevation = cam_elevation
        viewer.cam.azimuth = cam_azimuth
        viewer.cam.lookat[:] = [0.0, 0.0, 1.0]

        # 多灯布光（天光+主光+补光+背光+球灯）
        # 0: 天光 - 正上方模拟天空漫射
        model.light_pos[0] = [0.0, 0.0, 8.0]
        model.light_dir[0] = [0.0, 0.0, -1.0]
        model.light_diffuse[0] = [1.4, 1.45, 1.55]
        model.light_ambient[0] = [0.3, 0.3, 0.35]
        model.light_specular[0] = [0.5, 0.5, 0.5]
        # 1: 主光 - 右上方暖白
        if model.nlight > 1:
            model.light_pos[1] = [2.0, -2.0, 3.0]
            model.light_dir[1] = [-0.4, 0.3, -0.8]
            model.light_diffuse[1] = [1.2, 1.15, 1.05]
            model.light_ambient[1] = [0.0, 0.0, 0.0]
            model.light_specular[1] = [0.6, 0.6, 0.6]
            model.light_active[1] = True
        # 2: 补光 - 左前方冷白
        if model.nlight > 2:
            model.light_pos[2] = [-1.5, -1.0, 2.5]
            model.light_dir[2] = [0.3, 0.2, -0.7]
            model.light_diffuse[2] = [0.8, 0.85, 0.95]
            model.light_ambient[2] = [0.0, 0.0, 0.0]
            model.light_specular[2] = [0.4, 0.4, 0.4]
            model.light_active[2] = True
        # 3: 背光 - 后方轮廓
        if model.nlight > 3:
            model.light_pos[3] = [0.0, 2.0, 2.0]
            model.light_dir[3] = [0.0, -0.5, -0.6]
            model.light_diffuse[3] = [0.5, 0.5, 0.55]
            model.light_ambient[3] = [0.0, 0.0, 0.0]
            model.light_specular[3] = [0.3, 0.3, 0.3]
            model.light_active[3] = True

        start_time = time.perf_counter()

        while viewer.is_running():
            elapsed = time.perf_counter() - start_time
            sim_time = elapsed * playback_speed
            idx = int(sim_time / dt)

            if idx >= total_frames:
                if loop:
                    start_time = time.perf_counter()
                    idx = 0
                else:
                    idx = total_frames - 1

            if idx != last_idx:
                last_idx = idx

                if idx <= N:
                    arm_x = X[idx]
                else:
                    arm_x = X[-1]

                data.qpos[:NQ] = arm_x[:NQ]
                data.qvel[:NQ] = arm_x[NQ:]
                data.qpos[NQ:NQ + env.LEFT_ARM_NQ] = init_q_left

                if idx < len(ball_positions_phys):
                    bp = ball_positions_phys[idx]
                    data.qpos[bq: bq + 3] = bp
                    data.qpos[bq + 3: bq + 7] = [1, 0, 0, 0]

                mujoco.mj_forward(model, data)

            viewer.sync()
            time.sleep(1.0 / 120.0)


def main() -> None:
    """RM-65 MPC 主函数。"""
    parser = argparse.ArgumentParser(description="RM-65 MPC+iLQT 网球击打")
    parser.add_argument("--viewer", action="store_true", help="计算完成后以真实速度回放")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--fd", action="store_true", help="使用有限差分线性化")
    parser.add_argument("--horizon", type=int, default=None, help="短地平线步数")
    parser.add_argument("--iter", type=int, default=None, help="每次重规划迭代数")
    parser.add_argument("--fix-joint5", action="store_true", help="固定第 6 关节（wrist_3）")
    parser.add_argument("--backswing", type=float, default=0.6, help="后摆幅度 (rad, 正数表示关节1向后转的弧度)")
    parser.add_argument("--bs-ratio", type=float, default=0.35, help="后摆占比 (0~1, 默认0.35即前35%时间做后摆)")
    parser.add_argument("--no-backswing", action="store_true", help="禁用后摆（退化为普通 MPC）")
    parser.add_argument("--r-decay", type=float, default=0.30, help="R 衰减占比 (0~1, 默认0.30即后30%衰减)")
    parser.add_argument("--no-r-decay", action="store_true", help="禁用 R 退火（全程常数 R）")
    parser.add_argument("--hit-shift", type=float, default=0.01, help="击打目标沿挥拍方向前移距离 (m)，0=不偏移，默认0.01")
    parser.add_argument("--ball-speed", type=float, default=None, help="球到达击打点时的速度 (m/s)，不指定则随机")
    parser.add_argument("--normal-weight", type=float, default=500000.0, help="拍面法向量代价权重 (0=禁用，默认500000)")
    parser.add_argument("--normal-flip", action="store_true", help="翻转法向量方向（当拍面朝向反了时使用）")
    parser.add_argument("--replan-interval", type=int, default=20, help="重规划间隔步数（快速版默认20，原版默认10）")
    parser.add_argument("--no-clip", action="store_true", help="禁用关节速度/加速度硬裁剪（仅用软惩罚）")
    parser.add_argument("--qdot-weight", type=float, default=5000.0, help="关节速度软惩罚权重（默认5000, 0=禁用）")
    parser.add_argument("--vel-scale", type=float, default=1.0, help="关节速度限速缩放倍数（默认1.0=额定, 2.0=2倍额定）")
    parser.add_argument("--acc-scale", type=float, default=1.0, help="关节加速度限值缩放倍数（默认1.0=额定600°/s²）")
    parser.add_argument("--body-avoid-weight", type=float, default=50000.0, help="身体碰撞规避代价权重（0=禁用，默认50000）")
    args = parser.parse_args()

    use_analytical = not args.fd
    vel_limit_scaled = RM65_VEL_LIMIT_RAD * args.vel_scale
    acc_limit_scaled = RM65_ACC_LIMIT_RAD * args.acc_scale

    # 本地裁剪函数
    def _clip_vel(qd): return np.clip(qd, -vel_limit_scaled, vel_limit_scaled)
    def _clip_acc(qd, qd_prev):
        max_d = acc_limit_scaled * dt
        return qd_prev + np.clip(qd - qd_prev, -max_d, max_d)

    # 加载配置
    base_path = Path(__file__).resolve().parent.parent / "configs"
    config = load_config(base_path / "default.yaml")
    mpc_config_path = base_path / "mpc.yaml"
    if mpc_config_path.exists():
        mpc_config = load_config(mpc_config_path)
        config = merge_configs(config, mpc_config)

    dt = float(config["sim"]["dt"])
    g = np.array(config["ball"]["gravity"], dtype=np.float64)
    bounce_restitution = float(config["ball"].get("bounce_restitution", 0.75))

    # RM-65 肩关节位置（右臂 base_link1 安装位置）
    shoulder_pos = np.array([-0.1, -0.22693, 1.302645], dtype=np.float64)
    workspace_radius = 0.90

    # RM-65 专用 MPC 参数（快速版）
    mpc_cfg = config.get("mpc", {})
    total_horizon = 200
    fixed_horizon = 30
    replan_interval = 20               # 快速版：重规划间隔从 10 → 20
    max_iter_per_plan = 5              # 快速版：远距迭代从 8 → 5
    Q_p_scale_far = 5.0
    Q_v_scale_far = 3.0
    Q_p_scale_near = 5.0
    Q_v_scale_near = 50.0
    # 快速版：迭代次数控制
    first_plan_iters = 10              # 首次迭代6次
    near_plan_iters = 5                # 近距迭代2次

    if args.horizon is not None:
        fixed_horizon = args.horizon
    if args.iter is not None:
        max_iter_per_plan = args.iter
    if args.replan_interval is not None:
        replan_interval = args.replan_interval

    # 初始右臂状态
    # init_q = np.array([0.373, 1.57, -0.236, 0.404, 0.446, 2.45], dtype=np.float64)
    init_q = np.array([-1.5, 1.57, -0.236, 0.404, 0.446, 2.45], dtype=np.float64)
    # 初始左臂状态
    init_q_left = np.array([-0.373, -1.57, 0.236, -0.404, -0.446, -2.45], dtype=np.float64)

    # 第 6 关节固定角度（若启用 --fix-joint5）
    fix_joint5_angle: float | None = init_q[5] if args.fix_joint5 else None

    # 后摆参数
    use_backswing = not args.no_backswing
    backswing_offset = -abs(args.backswing)
    backswing_ratio = args.bs_ratio

    # R 退火参数
    use_r_decay = not args.no_r_decay
    r_decay_ratio = args.r_decay

    # 初始化 RM-65 环境
    model_path = Path(__file__).resolve().parent.parent / "src" / "robot" / "rm65_model.xml"
    env = RM65Env(model_path, dt=dt)
    env.init_q_left = init_q_left

    x0 = np.zeros(env.NX)
    x0[:env.NQ] = init_q

    # ===== 生成发球轨迹 =====
    rng = np.random.default_rng(args.seed)

    hit_cfg = config.get("hitting", {})

    # 目标立方体：以球拍初始位置为中心，offset 可调
    env.reset(init_q)
    env.data.qpos[env.NQ:env.NQ + env.LEFT_ARM_NQ] = init_q_left
    env.data.qvel[env.NQ:env.NQ + env.LEFT_ARM_NQ] = 0.0
    env.update_kinematics()
    p_racket = env.get_ee_pos().copy()
    logger.info(f"球拍初始位置: {p_racket}")

    # target_center = np.array([-0.82765693, -0.47411682, 0.86947444])
    target_center = np.array([-0.82765693, -0.47411682, 0.86947444])
    target_offset = 0.1

    hit_time = total_horizon * dt * rng.uniform(0.3, 0.4)
    p0, v0, p_hit_expected = generate_ball_to_target_box(
        target_center, target_offset, hit_time, g,
        shoulder_pos=shoulder_pos, workspace_radius=workspace_radius,
        ball_speed=args.ball_speed,
        rng=rng,
        ball_direction="y",
        ball_start_y_range=(-5.5, -4.5),
        ball_start_z_range=(1.4, 1.8),
    )
    logger.info(f"生成发球: 初始位置={p0}, 初始速度={v0}, 期望击打点={p_hit_expected}")

    # ===== 寻找击打点 =====
    hit_info = find_hitting_point_physics(
        env, p0, v0, shoulder_pos, workspace_radius, total_horizon
    )

    if hit_info is None:
        print("\n========================================")
        print("  网球不在工作空间内，机械臂不击打！")
        print("========================================\n")
        return

    k_hit_total = hit_info["k_hit"]
    p_hit = hit_info["p_hit"]
    v_ball_hit = hit_info["v_ball_hit"]

    # 自适应后摆幅度：球越远（需要更大旋转）→ 后摆幅度越大
    if use_backswing:
        p_ee_init = env.get_ee_pos()
        dist_to_ball = np.linalg.norm(p_hit - p_ee_init)
        # 距离 0.8~1.5m 映射到 backswing 0.4~1.0 rad
        bs_scale = np.clip((dist_to_ball - 0.8) / (1.5 - 0.8), 0.0, 1.0)
        adaptive_bs = 0.4 + bs_scale * 0.6
        backswing_offset = -adaptive_bs
        logger.info(
            f"自适应后摆: dist={dist_to_ball:.3f}m, backswing={adaptive_bs:.2f}rad"
        )

    # 期望拍面法向量：朝向来球方向（球速反向）
    n_des = -v_ball_hit / (np.linalg.norm(v_ball_hit) + 1e-8)
    if args.normal_flip:
        n_des = -n_des

    # 基于击打步数设定远近阈值
    far_threshold = k_hit_total
    near_threshold = max(40, k_hit_total // 4)

    hit_direction = np.array(config["hitting"]["hit_direction"], dtype=np.float64)
    racket_speed = float(config["hitting"]["racket_speed"])
    v_hit_desired = compute_desired_hit_velocity(hit_direction, racket_speed)

    # ===== 随挥偏移：将 iLQT 终端目标沿挥拍方向前移 =====
    hit_shift = args.hit_shift
    d_hat = hit_direction / (np.linalg.norm(hit_direction) + 1e-8)
    p_follow = p_hit + hit_shift * d_hat

    # 随挥偏移仅用于 warm-start 和 IK 引导方向，终端代价仍瞄准 p_hit
    # 这样求解器在 warm-start 的随挥趋势下优化，同时保证击打精度

    logger.info(f"击打步数: {k_hit_total}, 击打位置: {p_hit}")
    if hit_shift > 0:
        logger.info(f"随挥偏移: {hit_shift:.3f}m, 随挥目标: {p_follow}")
    logger.info(f"线性化: {'解析' if use_analytical else '有限差分'}, horizon={fixed_horizon}, iter={max_iter_per_plan}")
    if fix_joint5_angle is not None:
        logger.info(f"第 6 关节固定: angle={fix_joint5_angle:.3f} rad")

    # ===== 初始化 =====
    Q_p = np.array(config["cost"]["Q_p"], dtype=np.float64) * 2.0
    Q_v = np.array(config["cost"]["Q_v"], dtype=np.float64) * 2.0
    R = float(config["cost"]["R"])

    ilqt_cfg = dict(config["ilqt"])

    # 初始 Warm-start（后摆 或 雅可比转置），目标用随挥点
    p_target_init = p_follow
    if use_backswing:
        U_prev, q_des_traj_init = generate_backswing_warm_start(
            env, x0, p_target_init, v_hit_desired, k_hit_total,
            backswing_offset=backswing_offset,
            backswing_ratio=backswing_ratio,
            fix_joint5_angle=fix_joint5_angle,
            n_des=n_des,
        )
        logger.info(
            f"已生成后摆 Warm-start: offset={backswing_offset:.2f}rad, "
            f"ratio={backswing_ratio:.1%}"
        )
    else:
        U_prev = compute_jacobian_init_control(
            env, x0, p_target_init, k_hit_total, gain=30.0,
            fix_joint5_angle=fix_joint5_angle,
        )
        q_des_traj_init = None
        logger.info("已计算雅可比转置初始控制序列")

    # 关节控制代价
    r_joint_scale: dict[int, float] = {}
    if use_backswing:
        r_joint_scale[0] = 0.3
    if fix_joint5_angle is not None:
        r_joint_scale[5] = 1000.0

    # 关节空间跟踪权重（B 方案）— 设为 None，仅用后摆 warm-start 引导
    Q_joint: dict[int, float] | None = None

    # 初始 R 退火调度
    R_schedule_init = (
        compute_r_schedule(k_hit_total, R, decay_ratio=r_decay_ratio)
        if use_r_decay else None
    )

    # 初始化代价函数（带关节速度软惩罚 + 身体碰撞规避）
    cost_fn = ConstrainedHittingCost(
        env, p_follow, v_hit_desired, Q_p, Q_v, R,
        Q_p_running=0.0,
        R_joint_scale=r_joint_scale if r_joint_scale else None,
        q_des_traj=q_des_traj_init,
        Q_joint=Q_joint,
        R_schedule=R_schedule_init,
        Q_n=args.normal_weight,
        n_des=n_des,
        Q_qdot=args.qdot_weight,
        Q_body=args.body_avoid_weight,
        body_avoid_points=BODY_AVOID_POINTS if args.body_avoid_weight > 0 else None,
    )
    cost_fn._vel_limit = vel_limit_scaled  # 使用缩放后的限值

    # 初始化求解器
    solver = ILQTSolver(ilqt_cfg, use_analytical=use_analytical)

    # 设置球的初始状态
    env.reset(init_q)
    env.data.qpos[env.NQ:env.NQ + env.LEFT_ARM_NQ] = init_q_left
    env.data.qvel[env.NQ:env.NQ + env.LEFT_ARM_NQ] = 0.0
    env.update_kinematics()
    env.set_ball_state(p0, v0)

    # ===== MPC 主循环 =====
    x_current = x0.copy()
    X_history = [x0.copy()]
    U_history: list[np.ndarray] = []
    ball_pos_history: list[np.ndarray] = [p0.copy()]
    cost_history: list[float] = []
    pos_error_history: list[float] = []

    t_total_start = time.perf_counter()
    step_times: list[float] = []
    replan_times: list[float] = []

    U_buffer: np.ndarray = np.zeros((0, env.NU))
    buffer_idx: int = 0
    is_first_plan: bool = True
    p_hit_new = p_hit.copy()
    k_hit_new = k_hit_total
    iters = 0
    hit_step = -1
    p_ee_at_hit = None
    q_ik_cache: np.ndarray | None = None
    ball_was_hit = False

    logger.info(f"开始 MPC 循环，总步数={total_horizon}，击打步数={k_hit_total}")

    for step in range(total_horizon):
        t_step_start = time.perf_counter()

        ball_pos, ball_vel = env.get_ball_state()

        need_replan = (step % replan_interval == 0) or (step == 0) or (buffer_idx >= len(U_buffer))

        if need_replan:
            t_replan_start = time.perf_counter()

            remaining_horizon = total_horizon - step
            hit_info_new = find_hitting_point_physics(
                env, ball_pos, ball_vel, shoulder_pos, workspace_radius, remaining_horizon
            )

            if hit_info_new is None:
                logger.info(f"步 {step}: 球不再可达 (ball_pos={np.round(ball_pos,3)}, ball_vel={np.round(ball_vel,2)}), 停止 MPC")
                break

            k_hit_candidate = hit_info_new["k_hit"]

            # 防止误检：若新 k_hit 暴跌（<上次的 1/4），可能是球拍碰撞干扰
            # 使用线性预测作为下界
            if k_hit_candidate < max(10, k_hit_new // 4) and k_hit_new > 30:
                k_hit_candidate = max(1, k_hit_new - replan_interval)

            p_hit_new = hit_info_new["p_hit"]
            k_hit_new = k_hit_candidate
            v_ball_hit_new = hit_info_new["v_ball_hit"]
            n_des_new = -v_ball_hit_new / (np.linalg.norm(v_ball_hit_new) + 1e-8)
            if args.normal_flip:
                n_des_new = -n_des_new
            q_ik_cache = None

            # 随挥偏移：iLQT 目标沿挥拍方向前移
            p_follow_new = p_hit_new + hit_shift * d_hat

            if k_hit_new > far_threshold:
                # 远距：雅可比转置控制器（位置追踪，目标用随挥点）
                u_jt = compute_jacobian_init_control(
                    env, x_current, p_follow_new, replan_interval, gain=60.0,
                    fix_joint5_angle=fix_joint5_angle,
                )
                U_buffer = u_jt
                buffer_idx = 0
                U_prev = np.zeros((0, env.NU))
                iters = 0
                replan_times.append(time.perf_counter() - t_replan_start)
            else:
                # 基于实际位置误差的权重调度
                env.set_arm_state(x_current)
                env.update_kinematics()
                pos_err_now = np.linalg.norm(env.get_ee_pos() - p_hit_new)

                if pos_err_now > 0.10:
                    Q_p_scale = Q_p_scale_far
                    Q_v_scale = Q_v_scale_far
                else:
                    ratio = pos_err_now / 0.10
                    Q_p_scale = Q_p_scale_near + (Q_p_scale_far - Q_p_scale_near) * ratio
                    Q_v_scale = Q_v_scale_near + (Q_v_scale_far - Q_v_scale_near) * ratio

                # 终端目标用随挥点（略偏球位前方）
                p_follow_new = p_hit_new + hit_shift * d_hat
                cost_fn.update_target(p_follow_new, v_hit_desired, n_des=n_des_new)
                cost_fn.update_weights(Q_p_scale, Q_v_scale)

                horizon_full = k_hit_new           # 后摆生成用完整地平线
                horizon_plan = min(horizon_full, 40)  # iLQR 规划封顶 40 步

                # 更新 R 退火调度
                if use_r_decay:
                    R_schedule_new = compute_r_schedule(
                        horizon_full, R, decay_ratio=r_decay_ratio,
                    )[:horizon_plan]
                    cost_fn.set_R_schedule(R_schedule_new)
                else:
                    cost_fn.set_R_schedule(None)

                iters_plan = max_iter_per_plan
                skip_ls = True               # 快速版：非首次跳线搜索
                if is_first_plan:
                    iters_plan = first_plan_iters
                    skip_ls = False           # 首次保留线搜索保收敛
                    is_first_plan = False
                elif k_hit_new <= near_threshold:
                    iters_plan = near_plan_iters

                if use_backswing:
                    q_hit_new_ik = env.solve_ik(
                        p_hit_new, q_init=x_current[:env.NQ],
                        max_iter=150, eps=1e-3,
                    )
                    if fix_joint5_angle is not None:
                        q_hit_new_ik[5] = fix_joint5_angle

                    env.set_arm_state(np.concatenate([q_hit_new_ik, np.zeros(env.NQ)]))
                    J_p_new = env.get_ee_jacp()
                    qdot_hit_new = np.linalg.lstsq(J_p_new, v_hit_desired, rcond=None)[0]
                    max_qdot = 3.0
                    qdot_norm = np.linalg.norm(qdot_hit_new)
                    if qdot_norm > max_qdot:
                        qdot_hit_new *= max_qdot / qdot_norm

                    backswing_scale = horizon_full / max(k_hit_total, 1)
                    q_des_traj_full = np.zeros((horizon_full, env.NQ))
                    q_des_traj_full[:, 0] = compute_joint1_backswing_trajectory(
                        x_current[0], x_current[env.NQ],
                        q_hit_new_ik[0], qdot_hit_new[0],
                        horizon_full,
                        backswing_offset=backswing_offset * backswing_scale,
                        backswing_ratio=backswing_ratio,
                    )
                    for j in range(1, env.NQ):
                        q_des_traj_full[:, j] = np.linspace(x_current[j], q_hit_new_ik[j], horizon_full)

                    q_des_traj_new = q_des_traj_full[:horizon_plan]
                    cost_fn.set_q_des_traj(q_des_traj_new, Q_joint=Q_joint)

                    # 用上一次规划的热启动，生成 full horizon 后截断
                    if len(U_prev) >= horizon_full // 3:
                        U_warm = resample_control_sequence(U_prev, horizon_full)[:horizon_plan]
                        if fix_joint5_angle is not None:
                            U_warm = fix_joint5_control_trajectory(
                                U_warm, x_current, env, fix_joint5_angle,
                            )
                    else:
                        U_warm_full, _ = generate_backswing_warm_start(
                            env, x_current, p_follow_new, v_hit_desired, horizon_full,
                            backswing_offset=backswing_offset * backswing_scale,
                            backswing_ratio=backswing_ratio,
                            fix_joint5_angle=fix_joint5_angle,
                            n_des=n_des_new,
                        )
                        U_warm = U_warm_full[:horizon_plan]
                else:
                    if len(U_prev) >= horizon_full // 3:
                        U_warm = resample_control_sequence(U_prev, horizon_full)[:horizon_plan]
                        if fix_joint5_angle is not None:
                            U_warm = fix_joint5_control_trajectory(
                                U_warm, x_current, env, fix_joint5_angle,
                            )
                    else:
                        U_warm = compute_jacobian_init_control(
                            env, x_current, p_follow_new, horizon_full, gain=30.0,
                            fix_joint5_angle=fix_joint5_angle,
                        )[:horizon_plan]

                ball_pos_save, ball_vel_save = env.get_ball_state()

                X_mpc, U_mpc, iter_costs = solver.solve_few_iters(
                    env, cost_fn, x_current, U_warm,
                    max_iter=iters_plan,
                    skip_linesearch=skip_ls,
                )

                if fix_joint5_angle is not None:
                    U_mpc = fix_joint5_control_trajectory(
                        U_mpc, x_current, env, fix_joint5_angle,
                    )

                env.set_ball_state(ball_pos_save, ball_vel_save)
                env.set_arm_state(x_current)

                if len(iter_costs) > 0:
                    cost_history.append(iter_costs[-1])

                if len(U_mpc) > replan_interval:
                    U_prev = U_mpc[replan_interval:]
                elif len(U_mpc) > 0:
                    U_prev = U_mpc[1:]
                else:
                    U_prev = np.zeros((0, env.NU))

                U_buffer = U_mpc[:replan_interval]
                buffer_idx = 0
                iters = iters_plan

                replan_times.append(time.perf_counter() - t_replan_start)
        else:
            q_ik_cache = None
            iters = 0

        if buffer_idx < len(U_buffer):
            u_cmd = U_buffer[buffer_idx]
            buffer_idx += 1
        else:
            u_cmd = ik_pd_step(env, x_current, p_hit_new, fix_joint5_angle=fix_joint5_angle)

        if fix_joint5_angle is not None:
            u_cmd = fix_joint5_control(u_cmd, fix_joint5_angle, x_current, env.NQ)

        # 接近击打时启用碰撞，让球拍物理击球（k_hit_new<=10 保证碰撞窗口足够）
        enable_collision = (k_hit_new <= 10)
        if hasattr(env, "set_arm_collision"):
            env.set_arm_collision(enable_collision)
        # 保存碰撞前球速，用于弹性反弹计算
        ball_vel_before_step = ball_vel.copy() if enable_collision else ball_vel
        qdot_before = x_current[env.NQ:].copy()
        x_current, ball_pos, ball_vel = env.step_full(u_cmd)

        # ── 关节速度/加速度硬裁剪 ──
        if not args.no_clip:
            qdot_clipped = _clip_vel(x_current[env.NQ:])
            qdot_clipped = _clip_acc(qdot_clipped, qdot_before)
            x_current[env.NQ:] = qdot_clipped
            env.set_arm_state(x_current)

        # 碰撞诊断：检查球-球拍接触
        ball_racket_hit = False
        if enable_collision and not ball_was_hit:
            n_contacts = env.data.ncon
            if n_contacts > 0:
                for ci in range(n_contacts):
                    c = env.data.contact[ci]
                    g1 = env.model.geom(c.geom1).name
                    g2 = env.model.geom(c.geom2).name
                    if 'ball' in g1 or 'ball' in g2:
                        if 'racket' in g1 or 'racket' in g2:
                            ball_racket_hit = True
                            ee_vel = env.get_ee_vel()
                            ee_speed = np.linalg.norm(ee_vel)
                            ball_spd = np.linalg.norm(ball_vel)
                            logger.info(f"步 {step}: 球拍击球! {g1}<->{g2}, 球拍速度={ee_speed:.2f}m/s [{ee_vel[0]:.2f},{ee_vel[1]:.2f},{ee_vel[2]:.2f}], 球速={ball_spd:.2f}m/s [{ball_vel[0]:.2f},{ball_vel[1]:.2f},{ball_vel[2]:.2f}]")

        if hasattr(env, "set_arm_collision"):
            env.set_arm_collision(True)

        X_history.append(x_current.copy())
        U_history.append(u_cmd.copy())
        ball_pos_history.append(ball_pos.copy())

        env.update_kinematics()
        pos_err = np.linalg.norm(env.get_ee_pos() - p_hit_new)
        pos_error_history.append(pos_err)

        step_time = time.perf_counter() - t_step_start
        step_times.append(step_time)

        if step % 20 == 0 or k_hit_new <= 5:
            logger.info(
                f"步 {step}: 剩余={k_hit_new}, 误差={pos_err:.4f}m, "
                f"迭代={iters}, 步耗时={step_time*1000:.1f}ms"
            )

        if ball_racket_hit and k_hit_new <= 10 and not ball_was_hit:
            logger.info(f"步 {step}: 球拍物理碰撞击球，继续仿真让碰撞完成")
            ball_was_hit = True
            hit_step = step
            env.update_kinematics()
            p_ee_at_hit = env.get_ee_pos().copy()

            # 解析弹性碰撞：用碰撞前的球速计算正确反弹
            # MuJoCo 碰撞响应偏弱，用手动计算覆盖
            n_racket = env.get_ee_normal()
            n_hat = n_racket / (np.linalg.norm(n_racket) + 1e-8)
            v_ee = env.get_ee_vel()
            v_ball_pre = ball_vel_before_step
            v_rel_n = np.dot(v_ball_pre - v_ee, n_hat)
            e = 0.8
            v_ball_new = v_ball_pre - (1 + e) * v_rel_n * n_hat
            logger.info(f"  弹性反弹: v_ball_before={v_ball_pre}, v_ball_after={v_ball_new}")
            logger.info(f"  球速: {np.linalg.norm(v_ball_pre):.2f}->{np.linalg.norm(v_ball_new):.2f} m/s, v_rel_n={v_rel_n:.2f}")
            env.set_ball_vel(v_ball_new)

        # 碰撞后继续几步再停止，让碰撞响应完成
        if ball_was_hit and (step - hit_step) >= 5:
            logger.info(f"步 {step}: 碰撞完成，停止MPC")
            break

        if k_hit_new <= 1:
            logger.info(f"步 {step}: 到达击打时刻")
            hit_step = step
            env.update_kinematics()
            p_ee_at_hit = env.get_ee_pos().copy()
            break

    # ===== 击打后继续仿真，让球拍碰撞把球打飞 =====
    post_hit_steps = 80
    logger.info(f"击打后继续仿真 {post_hit_steps} 步，观察球飞出效果...")

    for _ in range(post_hit_steps):
        q_hold = x_current[:env.NQ].copy()
        u_hold = 100.0 * (q_hold - x_current[:env.NQ]) - 10.0 * x_current[env.NQ:]
        u_hold = np.clip(u_hold, env.model.actuator_ctrlrange[:env.NU, 0], env.model.actuator_ctrlrange[:env.NU, 1])

        if fix_joint5_angle is not None:
            u_hold = fix_joint5_control(u_hold, fix_joint5_angle, x_current, env.NQ)

        x_current, ball_pos, _ = env.step_full(u_hold)
        if not args.no_clip:
            x_current[env.NQ:] = _clip_vel(x_current[env.NQ:])
            env.set_arm_state(x_current)
        X_history.append(x_current.copy())
        U_history.append(u_hold.copy())
        ball_pos_history.append(ball_pos.copy())

    # ===== 计时统计 =====
    t_total = time.perf_counter() - t_total_start
    n_steps = len(U_history)
    avg_step_ms = np.mean(step_times) * 1000 if step_times else 0
    avg_replan_ms = np.mean(replan_times) * 1000 if replan_times else 0
    max_step_ms = np.max(step_times) * 1000 if step_times else 0
    real_time_ratio = (n_steps * dt) / t_total if t_total > 0 else 0

    logger.info(
        f"MPC 完成: 总耗时={t_total:.2f}s, 平均每步={avg_step_ms:.1f}ms, "
        f"平均重规划={avg_replan_ms:.1f}ms, 最慢步={max_step_ms:.1f}ms, "
        f"实时比率={real_time_ratio:.2f}x"
    )

    # ===== 评估 =====
    X_arr = np.array(X_history)
    U_arr = np.array(U_history) if len(U_history) > 0 else np.zeros((0, env.NU))
    ball_pos_arr = np.array(ball_pos_history)

    if p_ee_at_hit is not None:
        p_ee_final = p_ee_at_hit
    else:
        env.set_arm_state(x_current)
        p_ee_final = env.get_ee_pos()
    v_ee_final = env.get_ee_vel()
    pos_error = np.linalg.norm(p_ee_final - p_hit)
    vel_error = np.linalg.norm(v_ee_final - v_hit_desired)

    # 检查击打后球的速度变化
    ball_vel_after = env.get_ball_vel()
    ball_speed_after = np.linalg.norm(ball_vel_after)
    ball_vel_before = v_ball_hit if 'v_ball_hit' in dir() else np.zeros(3)
    if 'v_ball_hit_new' in dir():
        ball_vel_before = v_ball_hit_new
    speed_before = np.linalg.norm(ball_vel_before)
    v_ee_speed = np.linalg.norm(v_ee_final)

    # 反弹诊断：沿球拍面法向的速度变化
    if ball_was_hit and p_ee_at_hit is not None:
        n_racket = env.get_ee_normal()
        n_hat = n_racket / (np.linalg.norm(n_racket) + 1e-8)
        v_n_before = np.dot(ball_vel_before, n_hat)
        v_n_after = np.dot(ball_vel_after, n_hat)
        logger.info(f"反弹诊断(沿拍面法向): 碰撞前={v_n_before:+.2f} m/s, 碰撞后={v_n_after:+.2f} m/s, 反弹比={abs(v_n_after/v_n_before):.2f}")

    logger.info(f"最终位置误差: {pos_error:.4f} m")
    logger.info(f"最终速度误差: {vel_error:.4f} m/s")
    logger.info(f"击打前球速: {speed_before:.2f} m/s, 方向: {ball_vel_before/(speed_before+1e-8)}")
    logger.info(f"击打后球速: {ball_speed_after:.2f} m/s, 方向: {ball_vel_after/(ball_speed_after+1e-8)}")
    logger.info(f"球拍末端速度: {v_ee_speed:.2f} m/s, 方向: {v_ee_final/(v_ee_speed+1e-8)}")
    speed_change = ball_speed_after - speed_before
    logger.info(f"球速大小变化: {speed_change:+.2f} m/s")

    # 法向量对齐诊断
    if args.normal_weight > 0 and p_ee_at_hit is not None:
        env.set_arm_state(x_current)
        n_actual = env.get_ee_normal()
        n_desired = -v_ball_hit_new / (np.linalg.norm(v_ball_hit_new) + 1e-8) if 'v_ball_hit_new' in dir() else -v_ball_hit / (np.linalg.norm(v_ball_hit) + 1e-8)
        cos_angle = float(n_actual @ n_desired)
        angle_deg = np.degrees(np.arccos(np.clip(abs(cos_angle), -1.0, 1.0)))
        logger.info(
            f"法向量对齐: cos²={cos_angle**2:.4f}, 夹角={angle_deg:.1f}° "
            f"(实际={np.round(n_actual, 3)}, 期望={np.round(n_desired, 3)})"
        )

    # ===== 关节约束分析 =====
    if not args.no_clip:
        qdot_arr = X_arr[:, env.NQ:]
        peak_qdot = np.abs(qdot_arr).max(axis=0)
        peak_qdot_deg = np.rad2deg(peak_qdot)
        vel_limits_deg = np.rad2deg(vel_limit_scaled)
        print(f"\n  关节速度硬约束分析 (限速={vel_limits_deg.astype(int)} deg/s):")
        for j in range(6):
            n_violate = sum(1 for qd in qdot_arr[:, j] if abs(qd) > vel_limit_scaled[j])
            status = "OK" if n_violate == 0 else f"超限{n_violate}步/{len(qdot_arr)}"
            print(f"    J{j+1}: 峰值={peak_qdot_deg[j]:.0f} deg/s, 限速={vel_limits_deg[j]:.0f}, {status}")

    print("\n========================================")
    if pos_error < 0.05:
        print("  RM-65 击打成功！（误差 < 5cm）")
    elif pos_error < 0.1:
        print("  RM-65 击打基本命中！（误差 < 10cm）")
    else:
        print("  RM-65 击打偏差较大，需要调整参数。")
    print(f"  击打目标位置: {np.round(p_hit, 3)}")
    print(f"  末端实际位置: {np.round(p_ee_final, 3)}")
    print(f"  位置误差: {pos_error:.4f} m")
    print(f"  速度误差: {vel_error:.4f} m/s")
    print(f"  击打后球速: {ball_speed_after:.2f} m/s")
    print(f"  MPC 总步数: {n_steps}")
    print(f"  线性化: {'解析' if use_analytical else '有限差分'}")
    print(f"  后摆: {'启用' if use_backswing else '禁用'} (offset={backswing_offset:.2f}rad, ratio={backswing_ratio:.1%})")
    print(f"  随挥偏移: {hit_shift:.3f}m (iLQT目标={np.round(p_follow, 3)})")
    print(f"  R 退火: {'启用' if use_r_decay else '禁用'} (decay={r_decay_ratio:.1%})")
    if args.normal_weight > 0 and p_ee_at_hit is not None:
        print(f"  法向量: 权重={args.normal_weight:.0f}, 翻转={'是' if args.normal_flip else '否'}")
    print(f"  总计算时间: {t_total:.2f}s")
    print(f"  平均每步: {avg_step_ms:.1f}ms (实时需 {dt*1000:.1f}ms)")
    print(f"  实时比率: {real_time_ratio:.2f}x")
    print("========================================\n")

    # ===== 真实速度回放 =====
    if args.viewer:
        logger.info("MPC 计算完成，开始真实速度回放（含击打后球飞出）...")

        env.reset(init_q)
        env.data.qpos[env.NQ:env.NQ + env.LEFT_ARM_NQ] = init_q_left
        env.data.qvel[env.NQ:env.NQ + env.LEFT_ARM_NQ] = 0.0
        env.update_kinematics()
        env.set_ball_state(p0, v0)

        # 重新仿真获取完整轨迹（包含碰撞效果）
        X_replay = [env.get_arm_state().copy()]
        ball_replay = [env.get_ball_pos().copy()]
        rebound_applied = False

        for i, u_cmd in enumerate(U_arr):
            # 只在击打前 2 步到击打后 2 步开启碰撞，其余步关闭
            # 避免球在接近过程中被球拍提前碰飞
            enable_collision = (hit_step >= 0 and abs(i - hit_step) <= 5)
            if hasattr(env, "set_arm_collision"):
                env.set_arm_collision(enable_collision)
            # 保存碰撞前球速
            ball_vel_pre = env.get_ball_vel().copy() if enable_collision else np.zeros(3)
            env.step(u_cmd)
            env._handle_ball_bounce()
            # 碰撞检测 + 手动弹性反弹
            if enable_collision and not rebound_applied and hit_step >= 0:
                ncon = env.data.ncon
                for ci in range(ncon):
                    c = env.data.contact[ci]
                    g1 = env.model.geom(c.geom1).name
                    g2 = env.model.geom(c.geom2).name
                    if ('ball' in g1 or 'ball' in g2) and ('racket' in g1 or 'racket' in g2):
                        n_racket = env.get_ee_normal()
                        n_hat = n_racket / (np.linalg.norm(n_racket) + 1e-8)
                        v_ee = env.get_ee_vel()
                        v_rel_n = np.dot(ball_vel_pre - v_ee, n_hat)
                        e = 0.8
                        v_ball_new = ball_vel_pre - (1 + e) * v_rel_n * n_hat
                        env.set_ball_vel(v_ball_new)
                        rebound_applied = True
                        logger.info(f"  回放弹性反弹: 球速 {np.linalg.norm(ball_vel_pre):.2f}->{np.linalg.norm(v_ball_new):.2f} m/s")
                        break
            X_replay.append(env.get_arm_state().copy())
            ball_replay.append(env.get_ball_pos().copy())

        X_replay = np.array(X_replay)
        ball_replay_arr = np.array(ball_replay)

        # 恢复碰撞状态（确保查看器正常显示）
        if hasattr(env, "set_arm_collision"):
            env.set_arm_collision(True)

        visualize_rm65_result(
            env, X_replay, U_arr, ball_replay_arr, config,
            init_q_left=init_q_left,
            post_hit_steps=post_hit_steps,
        )


if __name__ == "__main__":
    main()

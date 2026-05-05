"""动力学线性化（有限差分法 + 解析法）。"""

import numpy as np
import mujoco
from src.sim.env import MujocoEnv


def linearize_dynamics(
    env: MujocoEnv,
    x: np.ndarray,
    u: np.ndarray,
    eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在 (x, u) 处用中心差分线性化动力学。

    计算 A = ∂f/∂x, B = ∂f/∂u, 以及 f(x, u) 的基准下一步状态。

    Args:
        env: MuJoCo 环境实例。
        x: 当前臂状态，形状 (12,)。
        u: 当前控制，形状 (6,)。
        eps: 有限差分步长。

    Returns:
        (A, B, x_next): A 形状 (12,12)，B 形状 (12,6)，x_next 形状 (12,)。
    """
    n_x = env.NX
    n_u = env.NU
    A = np.zeros((n_x, n_x))
    B = np.zeros((n_x, n_u))

    # 基准下一步
    x_next_base = env.step_from_state(x, u)

    # ∂f/∂x
    for j in range(n_x):
        x_plus = x.copy()
        x_plus[j] += eps
        x_next_plus = env.step_from_state(x_plus, u)

        x_minus = x.copy()
        x_minus[j] -= eps
        x_next_minus = env.step_from_state(x_minus, u)

        A[:, j] = (x_next_plus - x_next_minus) / (2.0 * eps)

    # ∂f/∂u
    for j in range(n_u):
        u_plus = u.copy()
        u_plus[j] += eps
        x_next_plus = env.step_from_state(x, u_plus)

        u_minus = u.copy()
        u_minus[j] -= eps
        x_next_minus = env.step_from_state(x, u_minus)

        B[:, j] = (x_next_plus - x_next_minus) / (2.0 * eps)

    return A, B, x_next_base


def linearize_trajectory(
    env: MujocoEnv,
    X: np.ndarray,
    U: np.ndarray,
    eps: float = 1e-5,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """沿整条轨迹线性化动力学（有限差分法）。

    Args:
        env: MuJoCo 环境实例。
        X: 状态轨迹，形状 (N+1, 12)。
        U: 控制轨迹，形状 (N, 6)。
        eps: 有限差分步长。

    Returns:
        (As, Bs, fs): 每步的 A_k, B_k, f_k 列表，各长度 N。
    """
    N = len(U)
    As: list[np.ndarray] = []
    Bs: list[np.ndarray] = []
    fs: list[np.ndarray] = []

    for k in range(N):
        A_k, B_k, f_k = linearize_dynamics(env, X[k], U[k], eps)
        As.append(A_k)
        Bs.append(B_k)
        fs.append(f_k)

    return As, Bs, fs


def linearize_analytical(
    env: MujocoEnv,
    x: np.ndarray,
    u: np.ndarray,
    eps: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """使用 MuJoCo 解析函数线性化臂动力学。

    计算 A = I + A_c * dt, B = B_c * dt，其中 A_c, B_c 从
    连续时间刚体动力学 M*q̈ + h(q,q̇) = τ 推导得到。

    连续状态空间：
      dx/dt = [q̇; M^{-1} * (τ - h(q, q̇))]
    其中 h(q, q̇) = C(q,q̇)*q̇ + g(q)

    方法：
    1. mj_fullM 一次求 M（解析质量矩阵）
    2. mj_rne 多次求 h 和其偏导数 H_q, H_q̇（中心差分）
    3. 组装 A_c, B_c 后用欧拉法离散化

    Args:
        env: MuJoCo 环境实例。
        x: 当前臂状态 [q; q̇]，形状 (12,)。
        u: 当前控制力矩 τ，形状 (6,)。
        eps: 有限差分步长（仅用于计算 h 的偏导）。

    Returns:
        (A, B, x_next): A (12,12), B (12,6), x_next (12,)。
    """
    nv = env.NQ
    n_x = env.NX
    n_u = env.NU
    dt = env.dt
    model = env.model
    data = env.data

    # 设置臂状态并前向运动学
    env.set_arm_state(x)

    # ---- 1. 计算质量矩阵 M (nv x nv)，仅取臂关节子块 ----
    M_full = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, M_full, data.qM)
    M = M_full[:nv, :nv].copy()
    M_inv = np.linalg.solve(M, np.eye(nv))

    # ---- 2. 计算基准偏置力 h(q, qdot) ----
    # mj_rne(flq_acc=0) 返回 h = C*qdot + g(q)
    data.qacc[:] = 0.0
    h_base = np.zeros(model.nv)
    mujoco.mj_rne(model, data, 0, h_base)
    h_arm = h_base[:nv].copy()

    # ---- 3. 计算 ∂h/∂q（中心差分） ----
    H_q = np.zeros((nv, nv))
    q_orig = data.qpos[:nv].copy()
    for j in range(nv):
        # 扰动 q[j] + eps
        data.qpos[:nv] = q_orig.copy()
        data.qpos[j] += eps
        mujoco.mj_forward(model, data)
        data.qacc[:] = 0.0
        h_plus = np.zeros(model.nv)
        mujoco.mj_rne(model, data, 0, h_plus)

        # 扰动 q[j] - eps
        data.qpos[:nv] = q_orig.copy()
        data.qpos[j] -= eps
        mujoco.mj_forward(model, data)
        data.qacc[:] = 0.0
        h_minus = np.zeros(model.nv)
        mujoco.mj_rne(model, data, 0, h_minus)

        H_q[:, j] = (h_plus[:nv] - h_minus[:nv]) / (2.0 * eps)

    # 恢复 q
    data.qpos[:nv] = q_orig
    mujoco.mj_forward(model, data)

    # ---- 4. 计算 ∂h/∂qdot（中心差分） ----
    H_qdot = np.zeros((nv, nv))
    qdot_orig = data.qvel[:nv].copy()
    for j in range(nv):
        # 扰动 qdot[j] + eps
        data.qvel[:nv] = qdot_orig.copy()
        data.qvel[j] += eps
        mujoco.mj_forward(model, data)
        data.qacc[:] = 0.0
        h_plus = np.zeros(model.nv)
        mujoco.mj_rne(model, data, 0, h_plus)

        # 扰动 qdot[j] - eps
        data.qvel[:nv] = qdot_orig.copy()
        data.qvel[j] -= eps
        mujoco.mj_forward(model, data)
        data.qacc[:] = 0.0
        h_minus = np.zeros(model.nv)
        mujoco.mj_rne(model, data, 0, h_minus)

        H_qdot[:, j] = (h_plus[:nv] - h_minus[:nv]) / (2.0 * eps)

    # 恢复 qdot
    data.qvel[:nv] = qdot_orig
    mujoco.mj_forward(model, data)

    # ---- 5. 组装连续时间 A_c, B_c ----
    # dx/dt = [qdot; M^{-1}*(tau - h(q, qdot))]
    # A_c = [0, I; -M^{-1}*H_q, -M^{-1}*H_qdot]
    # B_c = [0; M^{-1}]
    A_c = np.zeros((n_x, n_x))
    A_c[:nv, nv:] = np.eye(nv)
    A_c[nv:, :nv] = -M_inv @ H_q
    A_c[nv:, nv:] = -M_inv @ H_qdot

    B_c = np.zeros((n_x, n_u))
    B_c[nv:, :] = M_inv

    # ---- 6. 欧拉法离散化 ----
    A = np.eye(n_x) + A_c * dt
    B = B_c * dt

    # ---- 7. 基准下一步状态 ----
    x_next = env.step_from_state(x, u)

    return A, B, x_next


def linearize_analytical_trajectory(
    env: MujocoEnv,
    X: np.ndarray,
    U: np.ndarray,
    eps: float = 1e-5,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """沿整条轨迹线性化动力学（解析法）。

    Args:
        env: MuJoCo 环境实例。
        X: 状态轨迹，形状 (N+1, 12)。
        U: 控制轨迹，形状 (N, 6)。
        eps: 有限差分步长（仅用于 h 的偏导）。

    Returns:
        (As, Bs, fs): 每步的 A_k, B_k, f_k 列表，各长度 N。
    """
    N = len(U)
    As: list[np.ndarray] = []
    Bs: list[np.ndarray] = []
    fs: list[np.ndarray] = []

    for k in range(N):
        A_k, B_k, f_k = linearize_analytical(env, X[k], U[k], eps)
        As.append(A_k)
        Bs.append(B_k)
        fs.append(f_k)

    return As, Bs, fs

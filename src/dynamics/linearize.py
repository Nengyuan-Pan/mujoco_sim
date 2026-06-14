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
    actuator_mode: int = 0,
    kp: np.ndarray | None = None,
    kd: np.ndarray | None = None,
    use_feedforward: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """使用 MuJoCo 解析函数线性化臂动力学。

    计算 A = I + A_c * dt, B = B_c * dt，其中 A_c, B_c 从
    连续时间刚体动力学 M*q̈ + h(q,q̇) = τ 推导得到。

    连续状态空间：
      dx/dt = [q̇; M^{-1} * (τ - h(q, q̇))]
    其中 h(q, q̇) = C(q,q̇)*q̇ + g(q)

    力矩模式 (actuator_mode=0):
      τ = u（直接施加力矩）
      A_c = [0, I; -M^{-1}*H_q, -M^{-1}*H_qdot]
      B_c = [0; M^{-1}]

    位置模式 (actuator_mode=1, use_feedforward=False):
      τ = Kp*(u - q) - Kd*qdot（PD 位置控制）
      A_c = [0, I; -M^{-1}*(H_q + diag(Kp)), -M^{-1}*(H_qdot + diag(Kd))]
      B_c = [0; M^{-1}*diag(Kp)]

    位置模式+前馈 (actuator_mode=1, use_feedforward=True):
      τ = Kp*(u - q) - Kd*qdot + h(q,qdot)（PD + 重力/科氏力补偿）
      前馈抵消偏置力后等效动力学: M*q̈ = Kp*(u-q) - Kd*q̇
      A_c = [0, I; -M^{-1}*diag(Kp), -M^{-1}*diag(Kd)]
      B_c = [0; M^{-1}*diag(Kp)]
      跳过 H_q/H_qdot 计算（约 50% 加速）。

    方法：
    1. mj_fullM 一次求 M（解析质量矩阵）
    2. mj_rne 多次求 h 和其偏导数 H_q, Ḣ（中心差分，前馈模式跳过）
    3. 组装 A_c, B_c 后用欧拉法离散化

    Args:
        env: MuJoCo 环境实例。
        x: 当前臂状态 [q; q̇]，形状 (12,)。
        u: 当前控制，形状 (6,)。力矩模式为力矩，位置模式为期望角度。
        eps: 有限差分步长（仅用于计算 h 的偏导）。
        actuator_mode: 0=力矩模式, 1=位置模式。
        kp: 位置模式比例增益 (6,)。位置模式时必填。
        kd: 位置模式微分增益 (6,)。位置模式时必填。
        use_feedforward: 位置模式下是否启用前馈补偿。True 时跳过 H_q/H_qdot。

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

    # ---- 3-4. 计算 ∂h/∂q 和 ∂h/∂qdot（前馈模式跳过） ----
    H_q = np.zeros((nv, nv))
    H_qdot = np.zeros((nv, nv))
    if not use_feedforward:
        # ∂h/∂q（中心差分）
        q_orig = data.qpos[:nv].copy()
        for j in range(nv):
            data.qpos[:nv] = q_orig.copy()
            data.qpos[j] += eps
            mujoco.mj_forward(model, data)
            data.qacc[:] = 0.0
            h_plus = np.zeros(model.nv)
            mujoco.mj_rne(model, data, 0, h_plus)

            data.qpos[:nv] = q_orig.copy()
            data.qpos[j] -= eps
            mujoco.mj_forward(model, data)
            data.qacc[:] = 0.0
            h_minus = np.zeros(model.nv)
            mujoco.mj_rne(model, data, 0, h_minus)

            H_q[:, j] = (h_plus[:nv] - h_minus[:nv]) / (2.0 * eps)
        data.qpos[:nv] = q_orig
        mujoco.mj_forward(model, data)

        # ∂h/∂qdot（中心差分）
        qdot_orig = data.qvel[:nv].copy()
        for j in range(nv):
            data.qvel[:nv] = qdot_orig.copy()
            data.qvel[j] += eps
            mujoco.mj_forward(model, data)
            data.qacc[:] = 0.0
            h_plus = np.zeros(model.nv)
            mujoco.mj_rne(model, data, 0, h_plus)

            data.qvel[:nv] = qdot_orig.copy()
            data.qvel[j] -= eps
            mujoco.mj_forward(model, data)
            data.qacc[:] = 0.0
            h_minus = np.zeros(model.nv)
            mujoco.mj_rne(model, data, 0, h_minus)

            H_qdot[:, j] = (h_plus[:nv] - h_minus[:nv]) / (2.0 * eps)
        data.qvel[:nv] = qdot_orig
        mujoco.mj_forward(model, data)

    # ---- 5. 组装连续时间 A_c, B_c（分模式） ----
    A_c = np.zeros((n_x, n_x))
    A_c[:nv, nv:] = np.eye(nv)
    # 前馈模式下 H_q=H_qdot=0，等效于跳过偏置力项
    A_c[nv:, :nv] = -M_inv @ H_q
    A_c[nv:, nv:] = -M_inv @ H_qdot

    B_c = np.zeros((n_x, n_u))
    if actuator_mode == 0:
        # 力矩模式: B_c = [0; M^{-1}]
        B_c[nv:, :] = M_inv
    else:
        # 位置模式:
        #   B_c = [0; M^{-1}*diag(Kp)]
        #   A_c 额外: -M^{-1}*diag(Kp) → A_c[nv:, :nv]
        #   A_c 额外: -M^{-1}*diag(Kd) → A_c[nv:, nv:]
        kp_row = kp[np.newaxis, :]  # (1, 6) 广播
        kd_row = kd[np.newaxis, :]
        B_c[nv:, :] = M_inv * kp_row
        A_c[nv:, :nv] -= M_inv * kp_row
        A_c[nv:, nv:] -= M_inv * kd_row

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
    actuator_mode: int = 0,
    kp: np.ndarray | None = None,
    kd: np.ndarray | None = None,
    use_feedforward: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """沿整条轨迹线性化动力学（解析法）。

    Args:
        env: MuJoCo 环境实例。
        X: 状态轨迹，形状 (N+1, 12)。
        U: 控制轨迹，形状 (N, 6)。
        eps: 有限差分步长（仅用于 h 的偏导）。
        actuator_mode: 0=力矩模式, 1=位置模式。
        kp: 位置模式比例增益 (6,)。
        kd: 位置模式微分增益 (6,)。
        use_feedforward: 位置模式下是否启用前馈补偿。

    Returns:
        (As, Bs, fs): 每步的 A_k, B_k, f_k 列表，各长度 N。
    """
    N = len(U)
    As: list[np.ndarray] = []
    Bs: list[np.ndarray] = []
    fs: list[np.ndarray] = []

    for k in range(N):
        A_k, B_k, f_k = linearize_analytical(
            env, X[k], U[k], eps,
            actuator_mode=actuator_mode, kp=kp, kd=kd,
            use_feedforward=use_feedforward,
        )
        As.append(A_k)
        Bs.append(B_k)
        fs.append(f_k)

    return As, Bs, fs


def linearize_fast_trajectory(
    env: MujocoEnv,
    X: np.ndarray,
    U: np.ndarray,
    actuator_mode: int = 0,
    kp: np.ndarray | None = None,
    kd: np.ndarray | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """沿轨迹快速线性化（仅计算 M^{-1}，跳过 ∂h/∂q 和 ∂h/∂qdot 有限差分）。

    力矩模式:
      A ≈ [I, dt*I; 0, I]
      B = [0; dt*M^{-1}]

    位置模式:
      A ≈ [I, dt*I; -dt*M^{-1}*diag(Kp), I - dt*M^{-1}*diag(Kd)]
      B = [0; dt*M^{-1}*diag(Kp)]

    f 通过完整动力学 step_from_state 计算，保证前向传递精度。

    适用于 far 阶段 MPC 重规划：线性模型精度较低但计算速度约快 10×。

    Args:
        env: MuJoCo 环境实例。
        X: 状态轨迹，形状 (N+1, 12)。
        U: 控制轨迹，形状 (N, 6)。
        actuator_mode: 0=力矩模式, 1=位置模式。
        kp: 位置模式比例增益 (6,)。
        kd: 位置模式微分增益 (6,)。

    Returns:
        (As, Bs, fs): 每步的 A_k, B_k, f_k 列表，各长度 N。
    """
    nv = env.NQ
    n_x = env.NX
    n_u = env.NU
    dt = env.dt
    model = env.model
    data = env.data

    As: list[np.ndarray] = []
    Bs: list[np.ndarray] = []
    fs: list[np.ndarray] = []

    for k in range(len(U)):
        env.set_arm_state(X[k])

        M_full = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, M_full, data.qM)
        M = M_full[:nv, :nv].copy()
        M_inv = np.linalg.solve(M, np.eye(nv))

        A_k = np.eye(n_x)
        A_k[:nv, nv:] = dt * np.eye(nv)

        B_k = np.zeros((n_x, n_u))
        B_k[nv:, :] = dt * M_inv

        if actuator_mode == 1:
            kp_row = kp[np.newaxis, :]
            kd_row = kd[np.newaxis, :]
            A_k[nv:, :nv] -= dt * M_inv * kp_row
            A_k[nv:, nv:] -= dt * M_inv * kd_row
            B_k[nv:, :] = dt * M_inv * kp_row

        f_k = env.step_from_state(X[k], U[k])

        As.append(A_k)
        Bs.append(B_k)
        fs.append(f_k)

    return As, Bs, fs

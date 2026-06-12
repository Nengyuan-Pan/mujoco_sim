"""RM-65 双臂机器人 MuJoCo 仿真环境封装。

与 MujocoEnv 的区别：
- 右臂关节 qpos[0:6], qvel[0:6]
- 左臂关节 qpos[6:12], qvel[6:12]（不驱动，保持零位）
- 球 freejoint qpos[12:19], qvel[12:18]
- 执行器 ctrl[0:6] 仅控制右臂
"""

import numpy as np
import mujoco
from pathlib import Path
from typing import Callable

from src.utils.mujoco_loader import load_mujoco_model
from src.perception.ball_estimator import BallEstimator


class RM65Env:
    """RM-65 双臂机器人 MuJoCo 仿真环境封装类。"""

    NQ: int = 6
    NX: int = 12
    NU: int = 6
    BALL_RADIUS: float = 0.033
    BOUNCE_RESTITUTION: float = 0.75
    LEFT_ARM_NQ: int = 6
    BALL_QPOS_START: int = 12
    BALL_QVEL_START: int = 12

    def __init__(
        self,
        model_path: Path,
        dt: float | None = None,
        estimator: BallEstimator | None = None,
        estimator_config: dict | None = None,
        preprocessor: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        """初始化 RM-65 MuJoCo 环境。

        Args:
            model_path: MuJoCo XML 模型文件路径。
            dt: 可选的覆盖时间步长。
            estimator: 预配置的 BallEstimator 实例，None 或不存在时不滤波。
            estimator_config: BallEstimator 参数字典，estimator 非 None 时忽略。
            preprocessor: 观测预处理回调 (pos, vel) -> (pos, vel)，
                          在 KF 之前调用，用于噪声注入等外部处理。
        """
        self.model = load_mujoco_model(model_path)
        if dt is not None:
            self.model.opt.timestep = dt
        self.data = mujoco.MjData(self.model)

        self.racket_center_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "racket_center"
        )
        self.ball_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "ball"
        )
        self.hit_target_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "hit_target"
        )

        self.init_q_left: np.ndarray = np.zeros(self.LEFT_ARM_NQ)
        self._preprocessor = preprocessor
        self._estimator = None
        if estimator is not None:
            self._estimator = estimator
        elif estimator_config is not None:
            self._estimator = BallEstimator(self.dt, **estimator_config)
        self._cached_ball_state: tuple[np.ndarray, np.ndarray] | None = None

        # 保存原始力矩模式参数（configure_actuator_mode 切换时恢复用）
        self._torque_ctrlrange = self.model.actuator_ctrlrange[:self.NU].copy()
        self._torque_gainprm = self.model.actuator_gainprm[:self.NU].copy()
        self._torque_biasprm = self.model.actuator_biasprm[:self.NU].copy()
        self._torque_biastype = self.model.actuator_biastype[:self.NU].copy()
        self._actuator_mode: int = 0  # 0=力矩, 1=位置
        self._kp: np.ndarray | None = None
        self._kd: np.ndarray | None = None

    def observe(self) -> tuple[np.ndarray, np.ndarray]:
        """推进观测处理管线：MuJoCo → preprocessor → KF → 缓存。

        每 MPC 步仅应调用一次，后续用 get_ball_state() 读缓存。

        Returns:
            (pos, vel) 处理后的球状态（滤波值，或无 KF 时为 MuJoCo 真值）。
        """
        bq = self.BALL_QPOS_START
        bv = self.BALL_QVEL_START
        pos = self.data.qpos[bq: bq + 3].copy()
        vel = self.data.qvel[bv: bv + 3].copy()
        if self._preprocessor is not None:
            pos, vel = self._preprocessor(pos, vel)
        if self._estimator is not None:
            pos, vel = self._estimator.update(pos, vel)
        self._cached_ball_state = (pos, vel)
        return pos, vel

    def reset(self, q0: np.ndarray | None = None) -> np.ndarray:
        """重置仿真状态。

        Args:
            q0: 右臂初始关节角度，形状 (6,)。

        Returns:
            初始右臂状态 x0，形状 (12,)。
        """
        mujoco.mj_resetData(self.model, self.data)
        if q0 is not None:
            self.data.qpos[: self.NQ] = q0
        self.data.qvel[: self.NQ] = 0.0
        self._set_ball_pos_vel(np.array([5.0, 0.0, 2.0]), np.zeros(3))
        mujoco.mj_forward(self.model, self.data)
        if self._estimator is not None:
            self._estimator.reset()
        self._cached_ball_state = None
        return self.get_arm_state()

    def get_arm_state(self) -> np.ndarray:
        """获取右臂状态 [q, qdot]，形状 (12,)。"""
        return np.concatenate(
            [self.data.qpos[: self.NQ].copy(), self.data.qvel[: self.NQ].copy()]
        )

    def set_arm_state(self, x: np.ndarray) -> None:
        """设置右臂状态 [q, qdot] 并刷新运动学。

        同时将左臂归到 init_q_left，防止 iLQT 规划时左臂漂移干扰右臂动力学。

        Args:
            x: 右臂状态，形状 (12,)。
        """
        self.data.qpos[: self.NQ] = x[: self.NQ]
        self.data.qvel[: self.NQ] = x[self.NQ :]
        self.data.qpos[self.NQ: self.NQ + self.LEFT_ARM_NQ] = self.init_q_left
        self.data.qvel[self.NQ: self.NQ + self.LEFT_ARM_NQ] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def update_kinematics(self) -> None:
        """轻量运动学刷新。"""
        mujoco.mj_kinematics(self.model, self.data)

    def step(self, u: np.ndarray) -> np.ndarray:
        """施加右臂控制力矩并前进一步。

        同时保持左臂零位（PD 控制器 + 位置归零），防止碰撞推力导致漂移。

        Args:
            u: 右臂关节力矩，形状 (6,)。

        Returns:
            新的右臂状态，形状 (12,)。
        """
        ctrl_range = self.model.actuator_ctrlrange[: self.NU]
        u_clipped = np.clip(u, ctrl_range[:, 0], ctrl_range[:, 1])
        self.data.ctrl[: self.NU] = u_clipped
        # 先强制左臂位置和速度，再计算 PD（此时误差为 0，ctrl=0）
        self.data.qpos[self.NQ: self.NQ + self.LEFT_ARM_NQ] = self.init_q_left
        self.data.qvel[self.NQ: self.NQ + self.LEFT_ARM_NQ] = 0.0
        l_q = self.data.qpos[self.NQ: self.NQ + self.LEFT_ARM_NQ]
        l_qd = self.data.qvel[self.NQ: self.NQ + self.LEFT_ARM_NQ]
        l_tau = 200.0 * (self.init_q_left - l_q) - 20.0 * l_qd
        l_ctrl_range = self.model.actuator_ctrlrange[self.NU: self.NU + self.LEFT_ARM_NQ]
        l_tau = np.clip(l_tau, l_ctrl_range[:, 0], l_ctrl_range[:, 1])
        self.data.ctrl[self.NU: self.NU + self.LEFT_ARM_NQ] = l_tau
        mujoco.mj_step(self.model, self.data)
        self._cached_ball_state = None  # 球物理状态已变，缓存作废
        self._handle_ball_bounce()
        return self.get_arm_state()

    def set_arm_collision(self, enabled: bool) -> None:
        """启用或禁用机械臂 geom 碰撞（用于规划时避免球拍-球碰撞干扰）。"""
        ball_geom_start = self.model.body("ball").geomadr[0]
        if enabled:
            self.model.geom_contype[:ball_geom_start] = self._arm_contype_save
            self.model.geom_conaffinity[:ball_geom_start] = self._arm_conaffinity_save
        else:
            if not hasattr(self, "_arm_contype_save"):
                self._arm_contype_save = self.model.geom_contype[:ball_geom_start].copy()
                self._arm_conaffinity_save = self.model.geom_conaffinity[:ball_geom_start].copy()
            self.model.geom_contype[:ball_geom_start] = 0
            self.model.geom_conaffinity[:ball_geom_start] = 0

    def step_from_state(self, x: np.ndarray, u: np.ndarray,
                        preserve_cache: bool = False) -> np.ndarray:
        """从指定右臂状态出发，施加控制并前进一步。

        Args:
            x: 右臂状态 [q, qdot]，形状 (12,)。
            u: 控制力矩，形状 (NU,)。
            preserve_cache: 为 True 时保留球状态缓存，
                用于 trial step 场景（X 平面墙 / safety filter），
                调用端会在之后恢复球状态，缓存应随之保留。
        """
        cache = self._cached_ball_state if preserve_cache else None
        self.set_arm_state(x)
        result = self.step(u)
        if preserve_cache:
            self._cached_ball_state = cache
        return result

    def _handle_ball_bounce(self) -> None:
        """检测球是否触碰地面并应用解析弹跳。"""
        bq = self.BALL_QPOS_START
        bv = self.BALL_QVEL_START

        ball_z = self.data.qpos[bq + 2]
        ball_vz = self.data.qvel[bv + 2]

        if ball_z < self.BALL_RADIUS and ball_vz < 0:
            self.data.qpos[bq + 2] = self.BALL_RADIUS
            self.data.qvel[bv + 2] = -ball_vz * self.BOUNCE_RESTITUTION
            friction_factor = 0.95
            self.data.qvel[bv] *= friction_factor
            self.data.qvel[bv + 1] *= friction_factor

    def get_ee_pos(self) -> np.ndarray:
        """获取球拍中心世界坐标位置，形状 (3,)。"""
        return self.data.site_xpos[self.racket_center_id].copy()

    def get_ee_vel(self) -> np.ndarray:
        """获取球拍中心线速度，形状 (3,)。"""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(
            self.model, self.data, jacp, jacr, self.racket_center_id
        )
        return (jacp[:, : self.NQ] @ self.data.qvel[: self.NQ]).copy()

    def get_racket_face_speed(self) -> float:
        """获取球拍面中心线速度标量 [m/s]。

        v_face = v_tcp + ω × racket_offset
        其中 racket_offset = [0, 0, 0.25]（球拍中心相对法兰的偏移）。
        """
        v_tcp = self.get_ee_vel()
        omega = self.get_ee_angular_vel()
        racket_offset_local = np.array([0.0, 0.0, 0.25])
        v_face = v_tcp + np.cross(omega, racket_offset_local)
        return float(np.linalg.norm(v_face))

    def get_ee_angular_vel(self) -> np.ndarray:
        """获取球拍中心角速度，形状 (3,)。"""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(
            self.model, self.data, jacp, jacr, self.racket_center_id
        )
        return (jacr[:, : self.NQ] @ self.data.qvel[: self.NQ]).copy()

    def get_ee_jacp(self) -> np.ndarray:
        """获取球拍中心位置雅可比矩阵，形状 (3, 6)。"""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(
            self.model, self.data, jacp, jacr, self.racket_center_id
        )
        return jacp[:, : self.NQ].copy()

    def get_ee_jacr(self) -> np.ndarray:
        """获取球拍中心旋转雅可比矩阵，形状 (3, 6)。

        返回 J_ω 满足 ω = J_ω @ q̇。
        """
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(
            self.model, self.data, jacp, jacr, self.racket_center_id
        )
        return jacr[:, : self.NQ].copy()

    def get_body_pos_by_id(self, body_id: int) -> np.ndarray:
        """获取指定刚体帧原点的世界坐标位置。

        Args:
            body_id: MuJoCo body ID。

        Returns:
            世界坐标位置，形状 (3,)。
        """
        return self.data.xpos[body_id].copy()

    def get_body_jacp_by_id(self, body_id: int) -> np.ndarray:
        """获取指定刚体帧原点的位置雅可比矩阵。

        使用 mj_jac 计算 body 局部坐标零点相对于关节速度的雅可比。

        Args:
            body_id: MuJoCo body ID。

        Returns:
            位置雅可比矩阵，形状 (3, 6)。
        """
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jacp, jacr, np.zeros(3), body_id)
        return jacp[:, : self.NQ].copy()

    def get_ee_normal(self) -> np.ndarray:
        """获取球拍面法向量（球拍局部 X 轴在世界坐标系中的方向）。

        RM-65 模型中球拍面椭球体 size="0.005 0.10 0.12"，
        最薄方向是局部 X 轴，即 X 轴为拍面法向量。
        r_racket_body 的旋转矩阵取第一列 (X 轴) 即为世界坐标系法向量。

        Returns:
            法向量，形状 (3,)，单位向量。
        """
        racket_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "r_racket_body"
        )
        R = self.data.xmat[racket_body_id].reshape(3, 3)
        normal = R[:, 0].copy()
        normal /= np.linalg.norm(normal) + 1e-12
        return normal

    def _set_ball_pos_vel(self, pos: np.ndarray, vel: np.ndarray) -> None:
        """设置球的位置和速度。"""
        bq = self.BALL_QPOS_START
        bv = self.BALL_QVEL_START
        self.data.qpos[bq: bq + 3] = pos
        self.data.qpos[bq + 3: bq + 7] = [1, 0, 0, 0]
        self.data.qvel[bv: bv + 3] = vel
        self.data.qvel[bv + 3: bv + 6] = 0.0

    def set_ball_state(self, pos: np.ndarray, vel: np.ndarray) -> None:
        """设置球的位置和速度并刷新。"""
        self._set_ball_pos_vel(pos, vel)
        mujoco.mj_forward(self.model, self.data)

    def set_ball_vel(self, vel: np.ndarray) -> None:
        """仅设置球的速度（不改变位置）。"""
        bv = self.BALL_QVEL_START
        self.data.qvel[bv:bv + 3] = vel[:3]
        if vel.shape[0] >= 6:
            self.data.qvel[bv + 3:bv + 6] = vel[3:6]

    def get_ball_state(self) -> tuple[np.ndarray, np.ndarray]:
        """获取球的当前位置和速度（如有缓存则返回缓存值，否则读 MuJoCo）。"""
        if self._cached_ball_state is not None:
            return self._cached_ball_state
        bq = self.BALL_QPOS_START
        bv = self.BALL_QVEL_START
        pos = self.data.qpos[bq: bq + 3].copy()
        vel = self.data.qvel[bv: bv + 3].copy()
        if self._estimator is not None:
            pos, vel = self._estimator.update(pos, vel)
            self._cached_ball_state = (pos, vel)
        return pos, vel

    def get_ball_pos(self) -> np.ndarray:
        """获取球的当前世界坐标位置（如有缓存则优先返回缓存滤波值）。

        不会驱动滤波器前进一步。主循环应使用 observe() 以确保每帧滤波推进。
        """
        if self._cached_ball_state is not None:
            return self._cached_ball_state[0]
        if self._estimator is not None and self._estimator.initialized:
            return self._estimator.state[0]
        bq = self.BALL_QPOS_START
        return self.data.qpos[bq: bq + 3].copy()

    def get_ball_vel(self) -> np.ndarray:
        """获取球的当前线速度（如有缓存则优先返回缓存滤波值）。

        不会驱动滤波器前进一步。主循环应使用 observe() 以确保每帧滤波推进。
        """
        if self._cached_ball_state is not None:
            return self._cached_ball_state[1]
        if self._estimator is not None and self._estimator.initialized:
            return self._estimator.state[1]
        bv = self.BALL_QVEL_START
        return self.data.qvel[bv: bv + 3].copy()

    def set_hit_target(self, pos: np.ndarray) -> None:
        """设置击打目标标记位置。"""
        self._hit_target_pos = pos.copy()

    @property
    def dt(self) -> float:
        """仿真时间步长。"""
        return self.model.opt.timestep

    @property
    def actuator_mode(self) -> int:
        """执行器模式：0=力矩, 1=位置。"""
        return self._actuator_mode

    @property
    def kp(self) -> np.ndarray | None:
        """位置模式比例增益，形状 (6,)。力矩模式下为 None。"""
        return self._kp

    @property
    def kd(self) -> np.ndarray | None:
        """位置模式速度增益，形状 (6,)。力矩模式下为 None。"""
        return self._kd

    def configure_actuator_mode(
        self,
        mode: str,
        kp: np.ndarray | None = None,
        kd: np.ndarray | None = None,
    ) -> None:
        """配置执行器模式，修改 MuJoCo model 参数。

        调用后所有后续 step() 和线性化操作都将使用新模式。
        mj_resetData() 不会恢复此设置。

        Args:
            mode: "torque" 或 "position"。
            kp: (6,) 位置增益。position 模式必须提供。
            kd: (6,) 速度增益。position 模式必须提供。
        """
        if mode == "torque":
            self._actuator_mode = 0
            self.model.actuator_ctrlrange[:self.NU] = self._torque_ctrlrange
            self.model.actuator_gainprm[:self.NU] = self._torque_gainprm
            self.model.actuator_biasprm[:self.NU] = self._torque_biasprm
            self.model.actuator_biastype[:self.NU] = self._torque_biastype
            self._kp = None
            self._kd = None

        elif mode == "position":
            if kp is None or kd is None:
                raise ValueError("位置模式必须提供 kp 和 kd")
            kp = np.asarray(kp, dtype=np.float64).reshape(self.NU)
            kd = np.asarray(kd, dtype=np.float64).reshape(self.NU)
            self._actuator_mode = 1
            self._kp = kp.copy()
            self._kd = kd.copy()

            for i in range(self.NU):
                self.model.actuator_gainprm[i, 0] = kp[i]
                self.model.actuator_biasprm[i, 0] = 0.0
                self.model.actuator_biasprm[i, 1] = -kp[i]
                self.model.actuator_biasprm[i, 2] = -kd[i]
                self.model.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_AFFINE
                jnt_id = self.model.actuator_trnid[i, 0]
                self.model.actuator_ctrlrange[i] = self.model.jnt_range[jnt_id]

            for i in range(self.NU):
                actual = self.model.actuator_gainprm[i, 0]
                assert abs(actual - kp[i]) < 1e-10, \
                    f"执行器 {i} gain 写入失败: expected {kp[i]}, got {actual}"
        else:
            raise ValueError(f"未知执行器模式: {mode}")

    def clone_actuator_config(self, target_env: "RM65Env") -> None:
        """将当前 env 的执行器配置复制到目标 env。

        用于异步规划中同步 env_plan 的执行器配置。

        Args:
            target_env: 目标环境实例。
        """
        if self._actuator_mode == 0:
            target_env.configure_actuator_mode("torque")
        else:
            target_env.configure_actuator_mode("position", self._kp, self._kd)

    def step_full(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """施加控制力矩并前进一步，返回右臂状态和球状态。

        调用 step() 后自动推进观测管线，保持缓存与物理状态同步。
        """
        x_arm = self.step(u)
        ball_pos, ball_vel = self.observe()
        return x_arm, ball_pos, ball_vel

    def predict_ball_trajectory(
        self,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        n_steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """前向仿真球的物理轨迹，不改变当前仿真状态。

        预测期间固定臂关节位置，并禁用机械臂碰撞，避免球拍碰撞干扰轨迹。
        """
        qpos_save = self.data.qpos.copy()
        qvel_save = self.data.qvel.copy()
        ctrl_save = self.data.ctrl.copy()

        bq = self.BALL_QPOS_START
        bv = self.BALL_QVEL_START
        NQ = self.NQ
        arm_qpos_save = self.data.qpos[:NQ].copy()

        # 临时禁用机械臂 geom 碰撞，防止球拍碰撞干扰预测
        contype_save = self.model.geom_contype.copy()
        conaffinity_save = self.model.geom_conaffinity.copy()
        # 保留球 geom 的碰撞（最后几个 geom），禁用其他所有
        ball_geom_start = self.model.body('ball').geomadr[0]
        self.model.geom_contype[:ball_geom_start] = 0
        self.model.geom_conaffinity[:ball_geom_start] = 0

        self.data.qpos[bq: bq + 3] = ball_pos
        self.data.qpos[bq + 3: bq + 7] = [1, 0, 0, 0]
        self.data.qvel[bv: bv + 3] = ball_vel
        self.data.qvel[bv + 3: bv + 6] = 0.0
        self.data.ctrl[:] = 0.0

        positions = np.zeros((n_steps, 3))
        velocities = np.zeros((n_steps, 3))

        for k in range(n_steps):
            self.data.qpos[:NQ] = arm_qpos_save
            self.data.qvel[:NQ] = 0.0
            self.data.qpos[NQ:NQ + self.LEFT_ARM_NQ] = self.init_q_left
            self.data.qvel[NQ:NQ + self.LEFT_ARM_NQ] = 0.0
            self.data.ctrl[:] = 0.0

            mujoco.mj_step(self.model, self.data)
            self._handle_ball_bounce()
            positions[k] = self.data.qpos[bq: bq + 3].copy()
            velocities[k] = self.data.qvel[bv: bv + 3].copy()

        # 恢复碰撞设置
        self.model.geom_contype[:] = contype_save
        self.model.geom_conaffinity[:] = conaffinity_save

        self.data.qpos[:] = qpos_save
        self.data.qvel[:] = qvel_save
        self.data.ctrl[:] = ctrl_save
        mujoco.mj_forward(self.model, self.data)

        return positions, velocities

    def solve_ik(
        self,
        target_pos: np.ndarray,
        q_init: np.ndarray | None = None,
        max_iter: int = 200,
        eps: float = 1e-3,
        damp: float = 1e-6,
        step_size: float = 0.1,
    ) -> np.ndarray:
        """阻尼最小二乘逆运动学求解器（基于 MuJoCo 雅可比）。

        使用与 assets/rm_65/controller.py 中 find_arm_inverse_kinematics
        相同的阻尼伪逆算法，但基于 MuJoCo 的 mj_jacSite 雅可比计算，
        保证与 MuJoCo 模型一致。

        Args:
            target_pos: 目标末端位置，形状 (3,)。
            q_init: 初始关节角度，形状 (6,)。若为 None 则使用零位。
            max_iter: 最大迭代次数。
            eps: 收敛阈值（位置误差，米）。
            damp: 阻尼因子，防止奇异。
            step_size: 每步步长缩放。

        Returns:
            求解的关节角度，形状 (6,)。
        """
        if q_init is None:
            q = np.zeros(self.NQ)
        else:
            q = q_init.copy()

        for _ in range(max_iter):
            x = np.zeros(self.NX)
            x[:self.NQ] = q
            self.set_arm_state(x)

            p_ee = self.get_ee_pos()
            err = target_pos - p_ee
            err_norm = np.linalg.norm(err)
            if err_norm < eps:
                break

            J = self.get_ee_jacp()  # (3, 6)
            JJT = J @ J.T + damp * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, err)
            q = q + step_size * dq

        return q

    def solve_ik_trajectory(
        self,
        waypoints: np.ndarray,
        q_init: np.ndarray | None = None,
        max_iter_per_wp: int = 100,
        eps: float = 2e-3,
    ) -> np.ndarray:
        """沿路标点序列求解逆运动学轨迹。

        每个路标点的 IK 解作为下一个路标点的初始猜测（热启动）。

        Args:
            waypoints: 路标点位置序列，形状 (N, 3)。
            q_init: 初始关节角度，形状 (6,)。
            max_iter_per_wp: 每个路标点最大迭代数。
            eps: 收敛阈值。

        Returns:
            关节角度轨迹，形状 (N, 6)。
        """
        N = len(waypoints)
        Q = np.zeros((N, self.NQ))
        q = q_init.copy() if q_init is not None else np.zeros(self.NQ)

        for k in range(N):
            q = self.solve_ik(
                waypoints[k], q_init=q,
                max_iter=max_iter_per_wp, eps=eps,
            )
            Q[k] = q.copy()

        return Q

"""MuJoCo 仿真环境封装。"""

import numpy as np
import mujoco
from pathlib import Path


class MujocoEnv:
    """MuJoCo 仿真环境封装类，管理模型加载、状态读写、仿真步进。"""

    # 臂关节数量
    NQ: int = 6
    # 臂状态维度 (q + qdot)
    NX: int = 12
    # 控制维度
    NU: int = 6
    # 球半径
    BALL_RADIUS: float = 0.033
    # 弹跳恢复系数（真实网球硬地约 0.75）
    BOUNCE_RESTITUTION: float = 0.75

    def __init__(self, model_path: Path, dt: float | None = None) -> None:
        """初始化 MuJoCo 环境。

        Args:
            model_path: MuJoCo XML 模型文件路径。
            dt: 可选的覆盖时间步长，若提供则覆盖模型中的设定。
        """
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        if dt is not None:
            self.model.opt.timestep = dt
        self.data = mujoco.MjData(self.model)

        # 末端执行器 site id
        self.racket_center_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "racket_center"
        )
        # 球体 body id
        self.ball_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "ball"
        )
        # 击打目标 site id
        self.hit_target_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "hit_target"
        )
        # 关节名称列表（与 menagerie ur5e.xml 一致，带 _joint 后缀）
        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        # 执行器名称列表
        self.actuator_names = [
            "torque_shoulder_pan",
            "torque_shoulder_lift",
            "torque_elbow",
            "torque_wrist_1",
            "torque_wrist_2",
            "torque_wrist_3",
        ]

    def reset(self, q0: np.ndarray | None = None) -> np.ndarray:
        """重置仿真状态。

        Args:
            q0: 初始关节角度，形状 (6,)。若为 None 则使用零角度。

        Returns:
            初始臂状态 x0，形状 (12,)。
        """
        mujoco.mj_resetData(self.model, self.data)
        if q0 is not None:
            self.data.qpos[: self.NQ] = q0
        self.data.qvel[: self.NQ] = 0.0
        # 把球放到远处
        self._set_ball_pos_vel(np.array([5.0, 0.0, 2.0]), np.zeros(3))
        mujoco.mj_forward(self.model, self.data)
        return self.get_arm_state()

    def get_arm_state(self) -> np.ndarray:
        """获取臂状态 [q, qdot]，形状 (12,)。"""
        return np.concatenate(
            [self.data.qpos[: self.NQ].copy(), self.data.qvel[: self.NQ].copy()]
        )

    def set_arm_state(self, x: np.ndarray) -> None:
        """设置臂状态 [q, qdot] 并刷新运动学。

        Args:
            x: 臂状态，形状 (12,)。
        """
        self.data.qpos[: self.NQ] = x[: self.NQ]
        self.data.qvel[: self.NQ] = x[self.NQ :]
        mujoco.mj_forward(self.model, self.data)

    def update_kinematics(self) -> None:
        """轻量运动学刷新（仅更新位置量，不含约束/接触计算）。

        在 mj_step 之后调用，确保 site_xpos 等派生量与当前 qpos 一致。
        比 mj_forward 快 ~5x，适合 MPC 循环中使用。
        """
        mujoco.mj_kinematics(self.model, self.data)

    def step(self, u: np.ndarray) -> np.ndarray:
        """施加控制力矩并前进一步（球和臂都由 MuJoCo 物理仿真）。

        球碰地面时自动应用解析弹跳恢复系数，确保弹跳行为真实。

        Args:
            u: 关节力矩，形状 (6,)。

        Returns:
            新的臂状态，形状 (12,)。
        """
        ctrl_range = self.model.actuator_ctrlrange[: self.NU]
        u_clipped = np.clip(u, ctrl_range[:, 0], ctrl_range[:, 1])
        self.data.ctrl[: self.NU] = u_clipped
        mujoco.mj_step(self.model, self.data)
        self._handle_ball_bounce()
        return self.get_arm_state()

    def step_from_state(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """从指定状态出发，施加控制并前进一步。

        Args:
            x: 当前臂状态，形状 (12,)。
            u: 关节力矩，形状 (6,)。

        Returns:
            新的臂状态，形状 (12,)。
        """
        self.set_arm_state(x)
        return self.step(u)

    def _handle_ball_bounce(self) -> None:
        """检测球是否触碰地面，若触碰则应用解析弹跳。

        球与地面的碰撞被禁用（contype=0, conaffinity=0），
        由本方法手动处理弹跳，确保恢复系数与真实网球一致。
        弹跳时同时施加水平摩擦减速。
        """
        qpos_ball_start = self.NQ
        qvel_ball_start = self.NQ

        ball_z = self.data.qpos[qpos_ball_start + 2]
        ball_vz = self.data.qvel[qvel_ball_start + 2]

        # 球心低于球半径即视为触地
        if ball_z < self.BALL_RADIUS and ball_vz < 0:
            # 弹跳：Z 速度反转并乘恢复系数
            self.data.qpos[qpos_ball_start + 2] = self.BALL_RADIUS
            self.data.qvel[qvel_ball_start + 2] = -ball_vz * self.BOUNCE_RESTITUTION
            # 水平摩擦减速（模拟弹跳时的摩擦损耗）
            friction_factor = 0.95
            self.data.qvel[qvel_ball_start] *= friction_factor
            self.data.qvel[qvel_ball_start + 1] *= friction_factor

    def get_ee_pos(self) -> np.ndarray:
        """获取球拍中心世界坐标位置，形状 (3,)。"""
        return self.data.site_xpos[self.racket_center_id].copy()

    def get_ee_vel(self) -> np.ndarray:
        """获取球拍中心线速度，形状 (3,)。

        通过雅可比矩阵和关节速度计算：v = J_p @ qdot。
        """
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(
            self.model, self.data, jacp, jacr, self.racket_center_id
        )
        return (jacp[:, : self.NQ] @ self.data.qvel[: self.NQ]).copy()

    def get_ee_jacp(self) -> np.ndarray:
        """获取球拍中心位置雅可比矩阵（线速度部分），形状 (3, 6)。"""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(
            self.model, self.data, jacp, jacr, self.racket_center_id
        )
        return jacp[:, : self.NQ].copy()

    def _set_ball_pos_vel(self, pos: np.ndarray, vel: np.ndarray) -> None:
        """设置球的位置和速度。

        Args:
            pos: 球的世界坐标位置，形状 (3,)。
            vel: 球的线速度，形状 (3,)。
        """
        qpos_ball_start = self.NQ
        qvel_ball_start = self.NQ
        # freejoint qpos: [x, y, z, qw, qx, qy, qz]
        self.data.qpos[qpos_ball_start : qpos_ball_start + 3] = pos
        self.data.qpos[qpos_ball_start + 3 : qpos_ball_start + 7] = [1, 0, 0, 0]
        # freejoint qvel: [vx, vy, vz, wx, wy, wz]
        self.data.qvel[qvel_ball_start : qvel_ball_start + 3] = vel
        self.data.qvel[qvel_ball_start + 3 : qvel_ball_start + 6] = 0.0

    def set_ball_state(self, pos: np.ndarray, vel: np.ndarray) -> None:
        """设置球的位置和速度并刷新。

        Args:
            pos: 球的世界坐标位置，形状 (3,)。
            vel: 球的线速度，形状 (3,)。
        """
        self._set_ball_pos_vel(pos, vel)
        mujoco.mj_forward(self.model, self.data)

    def get_ball_state(self) -> tuple[np.ndarray, np.ndarray]:
        """获取球的当前位置和速度。

        Returns:
            (pos, vel): 球位置形状 (3,)，球速度形状 (3,)。
        """
        qpos_ball_start = self.NQ
        qvel_ball_start = self.NQ
        pos = self.data.qpos[qpos_ball_start : qpos_ball_start + 3].copy()
        vel = self.data.qvel[qvel_ball_start : qvel_ball_start + 3].copy()
        return pos, vel

    def set_hit_target(self, pos: np.ndarray) -> None:
        """设置击打目标标记的世界坐标位置。

        Args:
            pos: 目标位置，形状 (3,)。
        """
        # 更新 site 的 worldpos 通过修改关联 body 的 mocap 或者直接修改 site_xpos
        # MuJoCo 的 site_xpos 是只读的计算结果，需要通过修改 parent body 实现
        # 这里用简单方式：把 site_xpos 写入 data（对于非 anchored site 无效）
        # 更好的方案：用一个独立的 body + mocap 或直接移动一个 geom
        # 暂存目标位置，供 viewer 使用
        self._hit_target_pos = pos.copy()

    @property
    def dt(self) -> float:
        """仿真时间步长。"""
        return self.model.opt.timestep

    def step_full(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """施加控制力矩并前进一步，同时返回臂状态和球状态。

        球和臂都由 MuJoCo 物理引擎驱动，不做任何解析覆盖。

        Args:
            u: 关节力矩，形状 (6,)。

        Returns:
            (x_arm, ball_pos, ball_vel): 臂状态(12,)，球位置(3,)，球速度(3,)。
        """
        x_arm = self.step(u)
        ball_pos, ball_vel = self.get_ball_state()
        return x_arm, ball_pos, ball_vel

    def get_ball_pos(self) -> np.ndarray:
        """获取球的当前世界坐标位置，形状 (3,)。"""
        qpos_ball_start = self.NQ
        return self.data.qpos[qpos_ball_start : qpos_ball_start + 3].copy()

    def get_ball_vel(self) -> np.ndarray:
        """获取球的当前线速度，形状 (3,)。"""
        qvel_ball_start = self.NQ
        return self.data.qvel[qvel_ball_start : qvel_ball_start + 3].copy()

    def predict_ball_trajectory(
        self,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        n_steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """从指定球状态出发，前向仿真球的物理轨迹。

        不改变当前仿真状态（保存/恢复机制）。
        臂施加零力矩，仅让球自由飞行。

        Args:
            ball_pos: 球初始位置，形状 (3,)。
            ball_vel: 球初始速度，形状 (3,)。
            n_steps: 前向仿真步数。

        Returns:
            (positions, velocities): 位置数组形状 (n_steps, 3)，速度数组形状 (n_steps, 3)。
        """
        # 保存当前完整状态
        qpos_save = self.data.qpos.copy()
        qvel_save = self.data.qvel.copy()
        ctrl_save = self.data.ctrl.copy()

        # 设置球状态，臂保持当前
        self.data.qpos[self.NQ:self.NQ + 3] = ball_pos
        self.data.qpos[self.NQ + 3:self.NQ + 7] = [1, 0, 0, 0]
        self.data.qvel[self.NQ:self.NQ + 3] = ball_vel
        self.data.qvel[self.NQ + 3:self.NQ + 6] = 0.0
        self.data.ctrl[:] = 0.0

        positions = np.zeros((n_steps, 3))
        velocities = np.zeros((n_steps, 3))

        for k in range(n_steps):
            mujoco.mj_step(self.model, self.data)
            self._handle_ball_bounce()
            positions[k] = self.data.qpos[self.NQ:self.NQ + 3].copy()
            velocities[k] = self.data.qvel[self.NQ:self.NQ + 3].copy()

        # 恢复状态
        self.data.qpos[:] = qpos_save
        self.data.qvel[:] = qvel_save
        self.data.ctrl[:] = ctrl_save
        mujoco.mj_forward(self.model, self.data)

        return positions, velocities

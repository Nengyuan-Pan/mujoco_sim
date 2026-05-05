"""运动学与模型测试。"""

import numpy as np

from src.sim.env import MujocoEnv
from src.robot.kinematics import forward_kinematics, compute_workspace_reach
from pathlib import Path


class TestModel:
    """测试 MuJoCo 模型加载和关节名。"""

    def test_model_loads(self) -> None:
        """模型应能成功加载。"""
        model_path = Path(__file__).resolve().parent.parent / "src" / "robot" / "model.xml"
        env = MujocoEnv(model_path)
        assert env.NQ == 6
        assert env.NX == 12
        assert env.NU == 6

    def test_joint_names(self) -> None:
        """关节名应带 _joint 后缀（menagerie 风格）。"""
        model_path = Path(__file__).resolve().parent.parent / "src" / "robot" / "model.xml"
        env = MujocoEnv(model_path)
        for name in env.joint_names:
            assert name.endswith("_joint")

    def test_racket_center_exists(self) -> None:
        """racket_center site 应存在。"""
        model_path = Path(__file__).resolve().parent.parent / "src" / "robot" / "model.xml"
        env = MujocoEnv(model_path)
        assert env.racket_center_id >= 0


class TestKinematics:
    """测试运动学功能。"""

    def test_forward_kinematics(self) -> None:
        """正运动学在零位时应返回合理位置。"""
        model_path = Path(__file__).resolve().parent.parent / "src" / "robot" / "model.xml"
        env = MujocoEnv(model_path)
        q = np.zeros(6)
        p = forward_kinematics(env, q)
        assert p.shape == (3,)
        assert p[2] > 0  # 末端应在地面以上

    def test_workspace_reach(self) -> None:
        """工作空间半径应合理（~1m）。"""
        model_path = Path(__file__).resolve().parent.parent / "src" / "robot" / "model.xml"
        env = MujocoEnv(model_path)
        ws = compute_workspace_reach(env, n_samples=50)
        assert 0.8 < ws < 1.2

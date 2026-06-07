"""exp8 KF 滤波恢复包装脚本：monkey-patch RM65Env + BallEstimator。

用法:
    python scripts/exp/_run_exp7_kf.py <ball-speed> <seed> <use_tube> <noise_mode> <estimator_mode>

noise_mode: off | lo | mid | hi | anis
estimator_mode: kf | nokf

数据流 (kf 组):
    MuJoCo 真值 → add_observation_noise() → BallEstimator.update(noisy) → 滤波输出
    get_ball_pos/vel → 返回 estimator.state 缓存（不推进）
    env.step() → 重置 _kf_consumed flag
"""

import sys
import time as _time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ball_speed = sys.argv[1]
seed = sys.argv[2]
use_tube = sys.argv[3]
noise_mode = sys.argv[4]
estimator_mode = sys.argv[5]

# ============================================================
# 模块级噪声配置
# ============================================================
_NOISE_RNG: np.random.Generator | None = None
_NOISE_POS_STD: float = 0.0
_NOISE_VEL_STD: float = 0.0
_NOISE_POS_XYZ: tuple[float, float, float] | None = None
_NOISE_VEL_XYZ: tuple[float, float, float] | None = None

if noise_mode != "off":
    _NOISE_RNG = np.random.default_rng(int(seed) + 7000)
    if noise_mode == "lo":
        _NOISE_POS_STD = 0.02
        _NOISE_VEL_STD = 0.2
    elif noise_mode == "mid":
        _NOISE_POS_STD = 0.05
        _NOISE_VEL_STD = 0.5
    elif noise_mode == "hi":
        _NOISE_POS_STD = 0.10
        _NOISE_VEL_STD = 1.0
    elif noise_mode == "anis":
        _NOISE_POS_XYZ = (0.03, 0.10, 0.03)
        _NOISE_VEL_XYZ = (0.3, 1.0, 0.3)

# ============================================================
# 模块级 BallEstimator（kf 组专用）
# ============================================================
_estimator = None
_kf_consumed = False

if estimator_mode == "kf":
    from src.perception.ball_estimator import BallEstimator

    _kf_kwargs: dict = {}
    if noise_mode == "off":
        _kf_kwargs = {"pos_noise_std": 0.001, "vel_noise_std": 0.01}
    elif noise_mode == "lo":
        _kf_kwargs = {"pos_noise_std": 0.02, "vel_noise_std": 0.2}
    elif noise_mode == "mid":
        _kf_kwargs = {"pos_noise_std": 0.05, "vel_noise_std": 0.5}
    elif noise_mode == "hi":
        _kf_kwargs = {"pos_noise_std": 0.10, "vel_noise_std": 1.0}
    elif noise_mode == "anis":
        _kf_kwargs = {
            "pos_noise_xyz": (0.03, 0.10, 0.03),
            "vel_noise_xyz": (0.3, 1.0, 0.3),
        }

    _estimator = BallEstimator(dt=0.005, **_kf_kwargs)

# ============================================================
# Monkey-patch: RM65Env 方法
# ============================================================
from src.sim.rm65_env import RM65Env                     # noqa: E402
from src.utils.noise import add_observation_noise        # noqa: E402

_orig_get_ball_state = RM65Env.get_ball_state
_orig_get_ball_pos = RM65Env.get_ball_pos
_orig_get_ball_vel = RM65Env.get_ball_vel
_orig_step = RM65Env.step


def _patched_step(self, u):
    global _kf_consumed
    result = _orig_step(self, u)
    _kf_consumed = False
    return result


def _patched_get_ball_state(self) -> tuple[np.ndarray, np.ndarray]:
    global _kf_consumed
    pos, vel = _orig_get_ball_state(self)

    if _estimator is not None:
        if not _kf_consumed:
            if _NOISE_RNG is not None:
                pos, vel = add_observation_noise(
                    pos, vel, _NOISE_RNG,
                    pos_std=_NOISE_POS_STD,
                    vel_std=_NOISE_VEL_STD,
                    pos_std_xyz=_NOISE_POS_XYZ,
                    vel_std_xyz=_NOISE_VEL_XYZ,
                )
            _estimator._last_update_time = _time.perf_counter() - _estimator._dt
            pos, vel = _estimator.update(pos, vel)
            _kf_consumed = True
        else:
            pos, vel = _estimator.state
    elif _NOISE_RNG is not None:
        pos, vel = add_observation_noise(
            pos, vel, _NOISE_RNG,
            pos_std=_NOISE_POS_STD,
            vel_std=_NOISE_VEL_STD,
            pos_std_xyz=_NOISE_POS_XYZ,
            vel_std_xyz=_NOISE_VEL_XYZ,
        )

    return pos, vel


def _patched_get_ball_pos(self):
    if _estimator is not None:
        if not _estimator.initialized:
            return _orig_get_ball_pos(self)
        return _estimator.state[0]
    pos = _orig_get_ball_pos(self)
    if _NOISE_RNG is not None:
        pos, _ = add_observation_noise(
            pos, np.zeros(3), _NOISE_RNG,
            pos_std=_NOISE_POS_STD, vel_std=0.0,
            pos_std_xyz=_NOISE_POS_XYZ, vel_std_xyz=None,
        )
    return pos


def _patched_get_ball_vel(self):
    if _estimator is not None:
        if not _estimator.initialized:
            return _orig_get_ball_vel(self)
        return _estimator.state[1]
    vel = _orig_get_ball_vel(self)
    if _NOISE_RNG is not None:
        _, vel = add_observation_noise(
            np.zeros(3), vel, _NOISE_RNG,
            pos_std=0.0, vel_std=_NOISE_VEL_STD,
            pos_std_xyz=None, vel_std_xyz=_NOISE_VEL_XYZ,
        )
    return vel


RM65Env.step = _patched_step                               # type: ignore[method-assign]
RM65Env.get_ball_state = _patched_get_ball_state            # type: ignore[method-assign]
RM65Env.get_ball_pos = _patched_get_ball_pos                # type: ignore[method-assign]
RM65Env.get_ball_vel = _patched_get_ball_vel                # type: ignore[method-assign]

# ============================================================
# Monkey-patch RobotLimits: 速度豁免（与 exp7 一致）
# ============================================================
from src.ilqt.robot_limits import RobotLimits              # noqa: E402

_orig_from_config = RobotLimits.from_config


@classmethod
def _speed_exempt(cls, config, dt, ctrlrange):             # noqa: N805
    config = dict(config)
    config["forward_pass_margin"] = 3.0
    config["qdot_scale"] = 0.95
    config["forward_pass_q_tol_deg"] = 5.0
    config["max_tcp_speed"] = float("inf")
    return _orig_from_config(config, dt, ctrlrange)


RobotLimits.from_config = _speed_exempt                    # type: ignore[method-assign]

# ============================================================
# 构建 sys.argv 并调用主脚本
# ============================================================
sys.argv = [
    "rm65_mpc_tube_constraint.py",
    "--serve-box",
    "--ball-speed", ball_speed,
    "--seed", seed,
    "--use_tube", use_tube,
    "--no-backswing",
    "--no-plot",
]

import scripts.rm65_mpc_tube_constraint as main_mod         # noqa: E402

# ============================================================
# Monkey-patch: find_hitting_point_physics 缓存回退（与 exp7 一致）
# ============================================================
_last_hit_cache: dict | None = None
_last_horizon: int = 0

_orig_find_hitting = main_mod.find_hitting_point_physics


def _robust_find_hitting(env, ball_pos, ball_vel, shoulder_pos, workspace_radius, horizon):
    global _last_hit_cache, _last_horizon
    result = _orig_find_hitting(env, ball_pos, ball_vel, shoulder_pos, workspace_radius, horizon)
    if result is not None:
        _last_hit_cache = dict(result)
        _last_horizon = horizon
        return result
    if _last_hit_cache is not None:
        cached = dict(_last_hit_cache)
        step_delta = _last_horizon - horizon
        cached["k_hit"] = max(1, cached["k_hit"] - step_delta)
        _last_horizon = horizon
        return cached
    return None


main_mod.find_hitting_point_physics = _robust_find_hitting

main_mod.main()

"""exp8 KF 滤波恢复包装脚本：工厂函数 + __init__ monkey-patch。

用法:
    python scripts/exp/_run_exp7_kf.py <ball-speed> <seed> <use_tube> <noise_mode> <estimator_mode>

noise_mode: off | lo | mid | hi | anis
estimator_mode: kf | nokf

数据流 (kf 组):
    每 MPC 步: env.observe() → MuJoCo → preprocessor(加噪) → KF → 缓存
    规划内:     env.get_ball_state() → 返回缓存 (不推进)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ball_speed = sys.argv[1]
seed = int(sys.argv[2])
use_tube = sys.argv[3]
noise_mode = sys.argv[4]
estimator_mode = sys.argv[5]

# ============================================================
# 工厂函数：创建 preprocessor + BallEstimator（无 monkey-patch）
# ============================================================
from scripts.exp._exp8_config import make_preprocessor, make_estimator  # noqa: E402

_preprocessor = make_preprocessor(noise_mode, seed) if noise_mode != "off" else None
_estimator = make_estimator(noise_mode) if estimator_mode == "kf" else None

# ============================================================
# Monkey-patch: RM65Env.__init__ — 注入 preprocessor / estimator
# （仅此一处 patch，其余方法通过新增 API observe() / 缓存机制工作）
# ============================================================
from src.sim.rm65_env import RM65Env                     # noqa: E402

_orig_init = RM65Env.__init__


def _patched_init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    if _preprocessor is not None:
        self._preprocessor = _preprocessor
    if _estimator is not None:
        self._estimator = _estimator


RM65Env.__init__ = _patched_init                         # type: ignore[method-assign]

# ============================================================
# Monkey-patch: step_from_state 缓存保留（防止双重 KF 更新）
# 硬约束层的 trial step 之后总是恢复球状态，缓存应随之保留。
# ============================================================
_orig_step_from_state = RM65Env.step_from_state


def _cache_preserving_step_from_state(self, x, u, **kwargs):
    cache = self._cached_ball_state
    result = _orig_step_from_state(self, x, u, **kwargs)
    self._cached_ball_state = cache
    return result


RM65Env.step_from_state = _cache_preserving_step_from_state  # type: ignore[method-assign]

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
    "--ball-speed", str(ball_speed),
    "--seed", str(seed),
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

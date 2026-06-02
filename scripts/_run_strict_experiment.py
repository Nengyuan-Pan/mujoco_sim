"""严格约束实验脚本（自动生成）。

与 rm65_mpc_tube_constraint_realtime.py 相同，但：
- strict_braking_check 无豁免窗口（全程检查 qdot）
- 前向传递 margin=1.0
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ===== Monkey-patch: 取消安全滤波器的速度豁免 =====
from src.ilqt import robot_limits as _rl

_original_strict_braking_check = _rl.strict_braking_check

def _strict_braking_check_no_exemption(
    x_prev, x_next, u_try, limits, dt, k_hit_remaining=99,
):
    """全程严格 qdot 检查，无豁免窗口。"""
    import numpy as np
    nq = 6
    q_next = x_next[:nq]
    qdot_prev = x_prev[nq:]
    qdot_next = x_next[nq:]

    for j in range(nq):
        if q_next[j] < limits.q_lower[j]:
            return False, f"q lower bound violated, joint={j}"
        if q_next[j] > limits.q_upper[j]:
            return False, f"q upper bound violated, joint={j}"

    for j in range(nq):
        if u_try[j] < limits.u_min[j]:
            return False, f"u lower bound violated, joint={j}"
        if u_try[j] > limits.u_max[j]:
            return False, f"u upper bound violated, joint={j}"

    # 全程严格 1.0x，无豁免
    for j in range(nq):
        abs_prev = abs(qdot_prev[j])
        abs_next = abs(qdot_next[j])
        limit_j = limits.qdot_max[j]  # 1.0x，无放宽

        if abs_prev <= limit_j:
            if abs_next > limit_j:
                return False, (
                    f"qdot entering overspeed, joint={j}, "
                    f"|qdot|={abs_next:.2f} > {limit_j:.2f}"
                )
        else:
            if abs_next >= abs_prev:
                return False, (
                    f"qdot overspeeding+not-decelerating, joint={j}, "
                    f"|cur|={abs_prev:.2f} -> |next|={abs_next:.2f} > {limit_j:.2f}"
                )

    return True, ""

_rl.strict_braking_check = _strict_braking_check_no_exemption

# ===== Monkey-patch: 放宽 speed_range 以支持低速球 =====
from src.tennis import ball as _ball_mod
_original_gen = _ball_mod.generate_ball_from_serve_box

def _gen_wide_speed_range(*args, **kwargs):
    kwargs["speed_range"] = (2.0, 30.0)
    return _original_gen(*args, **kwargs)

_ball_mod.generate_ball_from_serve_box = _gen_wide_speed_range

# ===== Monkey-patch: 前向传递也严格检查 qdot =====
# forward_pass_margin 设为 1.0，不放宽

# ===== 导入并修改参数后运行主脚本 =====
# 修改 sys.argv 以传递参数给主脚本
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ball-speed", type=float, default=12)
parser.add_argument("--seed", type=int, default=0)
args, _ = parser.parse_known_args()

# 构造主脚本参数
sys.argv = [
    "rm65_mpc_tube_constraint_realtime.py",
    "--serve-box",
    "--ball-speed", str(args.ball_speed),
    "--seed", str(args.seed),
    "--no-backswing",
    "--no-plot",
    "--realtime",
]

# 导入主脚本并修改 forward_pass_margin
# 通过修改默认配置实现
import scripts.rm65_mpc_tube_constraint_realtime as main_mod

# 拦截 RobotLimits.from_config，强制 forward_pass_margin=1.0
_original_from_config = main_mod.RobotLimits.from_config

@classmethod
def _strict_from_config(cls, config, dt, ctrlrange):
    config = dict(config)
    config["forward_pass_margin"] = 1.0
    config["qdot_scale"] = 1.0
    config["forward_pass_q_tol_deg"] = 0.0
    return _original_from_config(config, dt, ctrlrange)

main_mod.RobotLimits.from_config = _strict_from_config

print(f"[STRICT MODE] ball_speed={args.ball_speed} seed={args.seed}")
print("[STRICT MODE] qdot <= 1.0x 全程，无豁免窗口")

main_mod.main()

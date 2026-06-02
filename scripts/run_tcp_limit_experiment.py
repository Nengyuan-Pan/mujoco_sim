"""TCP+关节双硬约束实验脚本。

在严格关节约束（qdot ≤ 1.0x）基础上，增加 TCP 线速度硬限制。
通过 monkey-patch 安全滤波器实现，不修改主脚本源码。

用法:
    python scripts/run_tcp_limit_experiment.py --ball-speed 9 --seed 0 --max-tcp 1.8 --viewer
    python scripts/run_tcp_limit_experiment.py --ball-speed 9 --seed 0 --max-tcp 1.8
"""
import sys
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ===== 参数解析 =====
parser = argparse.ArgumentParser(description="TCP+关节双硬约束实验")
parser.add_argument("--ball-speed", type=float, default=9)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max-tcp", type=float, default=1.8, help="TCP 线速度硬限制 (m/s)")
parser.add_argument("--viewer", action="store_true", help="启用 MuJoCo 回放")
parser.add_argument("--time-perturb-ms", type=float, default=0.0,
                    help="球到达时间预测扰动 (ms)")
parser.add_argument("--space-perturb-m", type=float, default=0.0,
                    help="击打点空间侧向偏移 (m)")
parser.add_argument("--use-tube", type=str, default="true",
                    help="是否启用 tube (true/false)")
parser.add_argument("--perturb-alpha-min", type=float, default=0.0,
                    help="衰减扰动保底值 (0~1)")
args, _ = parser.parse_known_args()

USE_TUBE = args.use_tube

MAX_TCP_SPEED = args.max_tcp
_env_ref = None

# ===== Monkey-patch 1: 严格关节约束（无豁免窗口）=====
from src.ilqt import robot_limits as _rl

def _strict_braking_check_all(x_prev, x_next, u_try, limits, dt, k_hit_remaining=99):
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

    for j in range(nq):
        abs_prev = abs(qdot_prev[j])
        abs_next = abs(qdot_next[j])
        limit_j = limits.qdot_max[j]
        if abs_prev <= limit_j:
            if abs_next > limit_j:
                return False, f"qdot entering overspeed, joint={j}, |qdot|={abs_next:.2f} > {limit_j:.2f}"
        else:
            if abs_next >= abs_prev:
                return False, f"qdot overspeeding+not-decelerating, joint={j}, |cur|={abs_prev:.2f} -> |next|={abs_next:.2f}"

    # TCP 速度硬限制
    global _env_ref
    if _env_ref is not None:
        _env_ref.set_arm_state(x_next)
        _env_ref.update_kinematics()
        tcp_vel = _env_ref.get_ee_vel()
        tcp_speed = float(np.linalg.norm(tcp_vel))
        if tcp_speed > MAX_TCP_SPEED:
            return False, f"tcp speed {tcp_speed:.2f} > {MAX_TCP_SPEED:.2f} m/s"

    return True, ""

_rl.strict_braking_check = _strict_braking_check_all

# ===== Monkey-patch 2: 放宽 speed_range 以支持低速球 =====
from src.tennis import ball as _ball_mod
_original_gen = _ball_mod.generate_ball_from_serve_box

def _gen_wide_speed_range(*a, **kw):
    kw["speed_range"] = (2.0, 30.0)
    return _original_gen(*a, **kw)

_ball_mod.generate_ball_from_serve_box = _gen_wide_speed_range

# ===== 构造主脚本参数 =====
sys.argv = [
    "rm65_mpc_tube_constraint_realtime.py",
    "--serve-box",
    "--ball-speed", str(args.ball_speed),
    "--seed", str(args.seed),
    "--no-backswing",
    "--no-plot",
    "--realtime",
    "--use_tube", USE_TUBE,
]
if args.viewer:
    sys.argv.append("--viewer")
if abs(args.time_perturb_ms) > 0.01:
    sys.argv.extend(["--time-perturb-ms", str(args.time_perturb_ms)])
if abs(args.space_perturb_m) > 0.001:
    sys.argv.extend(["--space-perturb-m", str(args.space_perturb_m)])
if args.perturb_alpha_min > 0.001:
    sys.argv.extend(["--perturb-alpha-min", str(args.perturb_alpha_min)])

# ===== 导入主脚本 =====
import scripts.rm65_mpc_tube_constraint_realtime as main_mod

# ===== Monkey-patch 3: RobotLimits.from_config 严格参数 =====
_original_from_config = main_mod.RobotLimits.from_config

@classmethod
def _strict_from_config(cls, config, dt, ctrlrange):
    config = dict(config)
    config["forward_pass_margin"] = 1.0
    config["qdot_scale"] = 1.0
    config["forward_pass_q_tol_deg"] = 0.0
    return _original_from_config(config, dt, ctrlrange)

main_mod.RobotLimits.from_config = _strict_from_config

# ===== Monkey-patch 4: 拦截主循环，注入 env 引用到安全滤波器 =====
_original_main = main_mod.main

def _patched_main():
    global _env_ref

    # 拦截 env.reset 和 env.step_full 来获取 env 引用
    # 更简洁的方式：直接 patch 主循环的局部变量
    # 由于无法直接 patch 局部变量，我们 patch env 的 step_from_state

    _original_step_from_state = main_mod.RM65Env.step_from_state

    # 直接运行原始 main，但在此之前 hook env 创建
    # 用 set_arm_state 的调用来检测 env 是否可用
    pass

# 放弃 patch main，改用更简单的方式：
# 在 strict_braking_check 中直接从 env 的全局实例获取 TCP 速度
# 需要在 main() 运行前设置 _env_ref

# 方案：patch main_mod.main，在调用原始 main 之前，
# 通过 hook env.reset 来捕获 env 实例

_original_rm65env_init = main_mod.RM65Env.__init__

def _hooked_init(self, *a, **kw):
    global _env_ref
    _original_rm65env_init(self, *a, **kw)
    _env_ref = self
    print(f"[TCP_LIMIT] env hooked, max_tcp={MAX_TCP_SPEED} m/s")

main_mod.RM65Env.__init__ = _hooked_init

# 但 __init__ 只在创建时调用，可能 step_from_state 恢复了状态
# 更好的做法是 patch step_from_state 使其在计算后恢复球状态
# 实际上 strict_braking_check 中的 set_arm_state 已经在安全滤波器中被调用
# check_one_step_feasibility 内部调用 step_predictor -> step_from_state
# step_from_state 会修改 env 的内部状态
# 所以在 strict_braking_check 中设置 arm state + update_kinematics 是安全的
# 因为安全滤波器之后会 restore arm state

# 唯一问题：球状态。step_from_state 不影响球，但 set_arm_state 不恢复球。
# 安全滤波器保存了 ball_save_sf，会在最后恢复。
# strict_braking_check 在 check_one_step_feasibility 内被调用，
# 此时 env 的状态是 step_from_state 之后的，我们额外做 set_arm_state(x_next)
# 来计算 TCP 速度。这是正确的，因为 x_next 就是预测的下一步状态。
# 但球状态可能被 step_from_state 改变（如果 step_from_state 内部调了 mj_step）。
# 需要保存/恢复球状态。

# 最安全的做法：在 strict_braking_check 中保存/恢复球状态
_original_get_ball_state = None
_original_set_ball_state = None

def _strict_braking_check_with_tcp(x_prev, x_next, u_try, limits, dt, k_hit_remaining=99):
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

    for j in range(nq):
        abs_prev = abs(qdot_prev[j])
        abs_next = abs(qdot_next[j])
        limit_j = limits.qdot_max[j]
        if abs_prev <= limit_j:
            if abs_next > limit_j:
                return False, f"qdot entering overspeed, joint={j}, |qdot|={abs_next:.2f} > {limit_j:.2f}"
        else:
            if abs_next >= abs_prev:
                return False, f"qdot overspeeding+not-decelerating, joint={j}, |cur|={abs_prev:.2f} -> |next|={abs_next:.2f}"

    # TCP 速度硬限制
    if _env_ref is not None:
        ball_saved = _env_ref.get_ball_state()
        _env_ref.set_arm_state(x_next)
        _env_ref.update_kinematics()
        tcp_vel = _env_ref.get_ee_vel()
        tcp_speed = float(np.linalg.norm(tcp_vel))
        _env_ref.set_ball_state(*ball_saved)
        if tcp_speed > MAX_TCP_SPEED:
            return False, f"tcp speed {tcp_speed:.2f} > {MAX_TCP_SPEED:.2f} m/s"

    return True, ""

_rl.strict_braking_check = _strict_braking_check_with_tcp

print(f"[TCP_LIMIT EXPERIMENT] ball_speed={args.ball_speed} seed={args.seed} max_tcp={MAX_TCP_SPEED} m/s")
print(f"[TCP_LIMIT] qdot <= 1.0x + TCP <= {MAX_TCP_SPEED} m/s, 全程无豁免")

main_mod.main()

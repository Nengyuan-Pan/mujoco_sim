"""TCP+关节双硬约束实验包装脚本。

v2 主脚本已内置 TCP 速度限制（--max-tcp）和严格关节约束（qdot_scale=1.0）。
本脚本现在仅作为兼容层，将参数转发给 v2 主脚本。

用法:
    python scripts/run_tcp_limit_experiment_v2.py --ball-speed 9 --seed 0 --max-tcp 1.8 --viewer
    python scripts/run_tcp_limit_experiment_v2.py --ball-speed 9 --seed 0 --max-tcp 1.8
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

parser = argparse.ArgumentParser(description="TCP+关节双硬约束实验（兼容层）")
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
parser.add_argument("--ablation", type=str, default=None,
                    help="消融模式: 'sigma-only'(禁用softmin)")
args, _ = parser.parse_known_args()

# ===== 放宽 speed_range 以支持低速球 =====
from src.tennis import ball as _ball_mod
_original_gen = _ball_mod.generate_ball_from_serve_box

def _gen_wide_speed_range(*a, **kw):
    kw["speed_range"] = (2.0, 30.0)
    return _original_gen(*a, **kw)

_ball_mod.generate_ball_from_serve_box = _gen_wide_speed_range

# ===== 构造 v2 主脚本参数 =====
sys.argv = [
    "rm65_mpc_tube_constraint_realtime_v2.py",
    "--serve-box",
    "--ball-speed", str(args.ball_speed),
    "--seed", str(args.seed),
    "--no-backswing",
    "--no-plot",
    "--realtime",
    "--use_tube", args.use_tube,
    "--max-tcp", str(args.max_tcp),
]
if args.viewer:
    sys.argv.append("--viewer")
if abs(args.time_perturb_ms) > 0.01:
    sys.argv.extend(["--time-perturb-ms", str(args.time_perturb_ms)])
if abs(args.space_perturb_m) > 0.001:
    sys.argv.extend(["--space-perturb-m", str(args.space_perturb_m)])
if args.perturb_alpha_min > 0.001:
    sys.argv.extend(["--perturb-alpha-min", str(args.perturb_alpha_min)])
if args.ablation == "sigma-only":
    sys.argv.append("--no-softmin")

print(f"[TCP_LIMIT EXPERIMENT] ball_speed={args.ball_speed} seed={args.seed} "
      f"max_tcp={args.max_tcp} m/s (built-in constraints)")

import scripts.rm65_mpc_tube_constraint_realtime_v2 as main_mod
main_mod.main()

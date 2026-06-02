"""v3: 全程硬约束实验（无终段豁免）。

关节速度 qdot ≤ 1.0x + TCP 线速度 ≤ 1.8 m/s，全程无豁免窗口。
与 v2 的区别：terminal_exempt_steps=0，击球阶段同样受限。

用法:
    python scripts/run_tcp_limit_experiment_v3.py --ball-speed 8 --seed 0
    python scripts/run_tcp_limit_experiment_v3.py --ball-speed 8 --seed 0 --viewer
    python scripts/run_tcp_limit_experiment_v3.py --ball-speed 8 --seed 0 --max-tcp 2.0
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

parser = argparse.ArgumentParser(description="v3: 全程硬约束实验（无终段豁免）")
parser.add_argument("--ball-speed", type=float, default=9)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--max-tcp", type=float, default=1.8, help="TCP 线速度硬限制 (m/s)")
parser.add_argument("--serve-distance", type=float, default=None,
                    help="发球区 Y 距离 (m), 默认根据球速自动选择")
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
parser.add_argument("--ball-speed-perturb-pct", type=float, default=0.0,
                    help="球速耦合扰动百分比 (%%): 实际发球速度=ball_speed*(1+pct/100)")
parser.add_argument("--no-softmin", action="store_true",
                    help="禁用多终端 softmin（退化为单点终端代价）")
parser.add_argument("--tube-cost-ratio", type=float, default=None,
                    help="Tube 走廊代价占比 (0~1)")
args, _ = parser.parse_known_args()

# ===== 放宽 speed_range 以支持低速球 =====
from src.tennis import ball as _ball_mod
_original_gen = _ball_mod.generate_ball_from_serve_box

def _gen_wide_speed_range(*a, **kw):
    kw["speed_range"] = (2.0, 30.0)
    return _original_gen(*a, **kw)

_ball_mod.generate_ball_from_serve_box = _gen_wide_speed_range

# ===== 根据球速自动选择发球距离 =====
if args.serve_distance is not None:
    serve_dist = args.serve_distance
elif args.ball_speed <= 5:
    serve_dist = 5.7
elif args.ball_speed <= 6:
    serve_dist = 6.8
elif args.ball_speed <= 7:
    serve_dist = 8.0
else:
    serve_dist = 9.5

# ===== 构造 v2 主脚本参数 =====
sys.argv = [
    "rm65_mpc_tube_constraint_realtime_v2.py",
    "--serve-box",
    "--ball-speed", str(args.ball_speed),
    "--seed", str(args.seed),
    "--serve-distance", str(serve_dist),
    "--no-backswing",
    "--no-plot",
    "--realtime",
    "--use_tube", args.use_tube,
    "--max-tcp", str(args.max_tcp),
    "--terminal-exempt-steps", "0",
]
if args.viewer:
    sys.argv.append("--viewer")
if abs(args.time_perturb_ms) > 0.01:
    sys.argv.extend(["--time-perturb-ms", str(args.time_perturb_ms)])
if abs(args.space_perturb_m) > 0.001:
    sys.argv.extend(["--space-perturb-m", str(args.space_perturb_m)])
if args.perturb_alpha_min > 0.001:
    sys.argv.extend(["--perturb-alpha-min", str(args.perturb_alpha_min)])
if args.ablation == "sigma-only" or args.no_softmin:
    sys.argv.append("--no-softmin")
if args.tube_cost_ratio is not None:
    sys.argv.extend(["--tube-cost-ratio", str(args.tube_cost_ratio)])
if abs(args.ball_speed_perturb_pct) > 0.01:
    sys.argv.extend(["--ball-speed-perturb-pct", str(args.ball_speed_perturb_pct)])

print(f"[V3 全程硬约束] ball_speed={args.ball_speed} seed={args.seed} "
      f"max_tcp={args.max_tcp} m/s, serve_dist={serve_dist}m, 终段豁免=0步(全程硬限)"
      f"{', ball_speed_perturb=%+.1f%%' % args.ball_speed_perturb_pct if abs(args.ball_speed_perturb_pct) > 0.01 else ''}")

import scripts.rm65_mpc_tube_constraint_realtime_v2 as main_mod
main_mod.main()

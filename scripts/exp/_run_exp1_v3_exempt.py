"""实验1 v3 辅助包装脚本：速度豁免 + no-bounce 球生成。

用法:
    python scripts/_run_exp1_v3_exempt.py <ball-speed> <seed> <use_tube>
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ball_speed = sys.argv[1]
seed = sys.argv[2]
use_tube = sys.argv[3]

# Monkey-patch RobotLimits for speed exemption
from src.ilqt.robot_limits import RobotLimits

_orig_from_config = RobotLimits.from_config


@classmethod
def _speed_exempt(cls, config, dt, ctrlrange):
    """注入速度豁免参数。"""
    config = dict(config)
    config["forward_pass_margin"] = 3.0
    config["qdot_scale"] = 0.95
    config["forward_pass_q_tol_deg"] = 5.0
    config["max_tcp_speed"] = float("inf")
    return _orig_from_config(config, dt, ctrlrange)


RobotLimits.from_config = _speed_exempt

# Override argv for the main script
sys.argv = [
    "rm65_mpc_tube_constraint.py",
    "--serve-box",
    "--no-bounce",
    "--ball-speed", ball_speed,
    "--seed", seed,
    "--use_tube", use_tube,
    "--no-backswing",
    "--no-plot",
]

import scripts.rm65_mpc_tube_constraint as main_mod  # noqa: E402
main_mod.main()

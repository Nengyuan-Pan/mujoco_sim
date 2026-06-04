"""实验2 v3 辅助包装脚本：严格关节约束 + no-bounce。

用法:
    python scripts/exp/_run_exp2_v3_strict.py <ball-speed> <seed> <use_tube>
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ball_speed = sys.argv[1]
seed = sys.argv[2]
use_tube = sys.argv[3]

from src.ilqt.robot_limits import RobotLimits

_orig_from_config = RobotLimits.from_config


@classmethod
def _strict_joint(cls, config, dt, ctrlrange):
    """注入严格关节约束参数。"""
    config = dict(config)
    config["forward_pass_margin"] = 1.0
    config["qdot_scale"] = 1.0
    config["forward_pass_q_tol_deg"] = 0.0
    config["max_tcp_speed"] = 1.8
    config["qddot_scale"] = 0.85
    return _orig_from_config(config, dt, ctrlrange)


RobotLimits.from_config = _strict_joint

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

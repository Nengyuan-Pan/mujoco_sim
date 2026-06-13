"""exp10 PD 扫描包装脚本。

用法:
    python scripts/exp/_run_exp10_pd_scan.py <kp_base> <kd_ratio> <dq_max_fraction> <seed> <ratio_mode>

ratio_mode:
    0 — current [1, 1, 1, 0.25, 0.25, 0.1]（当前默认，腕关节大幅降权）
    1 — uniform [1, 1, 1, 1, 1, 1]（均匀，最大化腕关节跟踪刚度）
    2 — torque-prop [1, 1, 0.5, 1/6, 1/6, 1/6]（按力矩限制比例）

Kd = Kp × kd_ratio（各关节独立计算）
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

kp_base = float(sys.argv[1])
kd_ratio = float(sys.argv[2])
dq_frac = float(sys.argv[3])
seed = int(sys.argv[4])
ratio_mode = int(sys.argv[5])

_RATIOS = {
    0: [1, 1, 1, 0.25, 0.25, 0.1],
    1: [1, 1, 1, 1, 1, 1],
    2: [1, 1, 0.5, 1 / 6, 1 / 6, 1 / 6],
}
ratio = _RATIOS[ratio_mode]
kp = [kp_base * r for r in ratio]
kd = [v * kd_ratio for v in kp]

sys.argv = [
    "rm65_mpc_v11.py",
    "--position-mode",
    "--serve-box",
    "--ball-speed", "7",
    "--seed", str(seed),
    "--no-plot",
    "--kp", *[str(v) for v in kp],
    "--kd", *[str(v) for v in kd],
    "--dq-max-fraction", str(dq_frac),
    "--replan-interval", "20",
]

import scripts.rm65_mpc_v11 as main_mod  # noqa: E402
main_mod.main()

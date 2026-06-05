"""v8 批量实验脚本：运行 6 组消融实验并汇总。"""

import subprocess
import sys
import csv
import re
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "rm65_mpc_v8.py"
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "exp_v8"
SEEDS = list(range(50))

BASE_ARGS = [
    sys.executable, str(SCRIPT),
    "--serve-box", "--ball-speed", "7",
    "--no-backswing", "--realtime",
    "--target-speed", "5.0",
    "--fixed-direction", "--hit-shift", "0.01",
    "--near-iters", "20", "--terminal-exempt-steps", "0",
    "--no-plot",
]

PERTURB_ARGS = [
    "--time-perturb-ms", "300",
    "--space-perturb-m", "0.15",
    "--perturb-alpha-min", "1.0",
    "--ball-speed-perturb-pct", "0",
]

EXPERIMENTS = [
    ("v8_default_no_perturb", []),
    ("v8_default_perturb", PERTURB_ARGS),
    ("v8_notube_no_perturb", ["--no-tube", "--no-softmin"]),
    ("v8_notube_perturb", ["--no-tube", "--no-softmin"] + PERTURB_ARGS),
    ("v8_nosoftmin_no_perturb", ["--no-softmin"]),
    ("v8_nosoftmin_perturb", ["--no-softmin"] + PERTURB_ARGS),
]

RESULT_RE = re.compile(
    r"pos_error=([\d.]+).*?min_dist=([\d.]+).*?hit_type=(\w+).*?"
    r"v_racket_at_hit=([\d.]+).*?max_tcp=([\d.]+).*?max_qdot=([\d.]+).*?"
    r"tube_ready_ms=([\d.]+).*?hit_time_error_ms=([\d.]+)",
    re.DOTALL,
)


def run_one(seed: int, extra_args: list[str]) -> dict | None:
    """运行单次实验，返回结果字典。"""
    cmd = BASE_ARGS + extra_args + ["--seed", str(seed)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    output = proc.stdout + proc.stderr
    m = RESULT_RE.search(output)
    if m:
        return {
            "seed": seed,
            "pos_error": float(m.group(1)),
            "min_dist": float(m.group(2)),
            "hit_type": m.group(3),
            "v_racket": float(m.group(4)),
            "max_tcp": float(m.group(5)),
            "max_qdot": float(m.group(6)),
            "tube_ready_ms": float(m.group(7)),
            "hit_time_error_ms": float(m.group(8)),
        }
    print(f"  [WARN] seed={seed} 无结果")
    return None


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) > 1:
        run_indices = [int(x) for x in sys.argv[1:]]
    else:
        run_indices = list(range(len(EXPERIMENTS)))

    for idx in run_indices:
        name, extra_args = EXPERIMENTS[idx]
        csv_path = RESULTS_DIR / f"{name}.csv"
        print(f"\n{'='*60}")
        print(f"实验 {idx+1}/{len(EXPERIMENTS)}: {name}")
        print(f"{'='*60}")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "seed", "pos_error", "min_dist", "hit_type", "v_racket",
                "max_tcp", "max_qdot", "tube_ready_ms", "hit_time_error_ms",
            ])
            writer.writeheader()

            hit_count = 0
            active_count = 0
            pos_sum = 0.0
            v_sum = 0.0

            for seed in SEEDS:
                row = run_one(seed, extra_args)
                if row is not None:
                    writer.writerow(row)
                    pos_sum += row["pos_error"]
                    v_sum += row["v_racket"]
                    if row["hit_type"] in ("active", "passive"):
                        hit_count += 1
                    if row["hit_type"] == "active":
                        active_count += 1
                else:
                    writer.writerow({"seed": seed, "hit_type": "error"})

            n = len(SEEDS)
            print(f"\n{name}:")
            print(f"  命中: {hit_count}/{n}")
            print(f"  Active: {active_count}/{n}")
            print(f"  avg pos_error: {pos_sum/n*100:.1f} cm")
            print(f"  avg v_racket: {v_sum/n:.2f} m/s")

    print("\n全部完成！")


if __name__ == "__main__":
    main()

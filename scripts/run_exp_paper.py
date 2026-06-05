"""论文实验：v2/v6/v7 对比，串行执行 50 seed。

条件矩阵：
  - v2: 无 tube, 无 softmin, 固定方向, target_speed=5.0
  - v6: 无 tube, 无 softmin, 固定方向, target_speed=5.0, n_des+smoothness in base_cost_fn
  - v7: corridor, 无 softmin, near_plan_iters=20, n_des+smoothness in base_cost_fn

每种跑 50 seed（无扰动 + 有扰动），共 6 组。

输出: results/exp_paper_20260604/<condition>_<perturb>.csv

用法:
  python scripts/run_exp_paper.py
  python scripts/run_exp_paper.py --seeds 10
  python scripts/run_exp_paper.py --conditions v2 v6
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "exp_paper_20260604"


def get_perturbations(n_seeds: int, rng_seed: int = 42) -> tuple[list[float], list[float]]:
    rng = np.random.default_rng(rng_seed)
    t_perturbs = rng.uniform(-300, 300, n_seeds).tolist()
    s_perturbs = rng.uniform(-0.15, 0.15, n_seeds).tolist()
    return t_perturbs, s_perturbs


def run_single(
    script: str,
    seed: int,
    t_perturb_ms: float | None,
    s_perturb_m: float | None,
    extra_args: list[str],
) -> dict[str, str] | None:
    cmd = [sys.executable, script]
    cmd += [
        "--serve-box", "--ball-speed", "7",
        "--seed", str(seed),
        "--no-backswing", "--realtime",
        "--terminal-exempt-steps", "0",
        "--no-plot", "--no-softmin",
    ]
    cmd += extra_args

    if t_perturb_ms is not None:
        cmd += ["--time-perturb-ms", f"{t_perturb_ms:.4f}"]
    if s_perturb_m is not None:
        cmd += ["--space-perturb-m", f"{s_perturb_m:.6f}"]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, cwd=str(ROOT),
        )
    except subprocess.TimeoutExpired:
        return {"seed": str(seed), "hit": "timeout"}

    output = result.stdout + result.stderr
    for line in output.split("\n"):
        if "__RESULT__" in line:
            fields: dict[str, str] = {"seed": str(seed)}
            for m in re.finditer(r"(\w+)=(\S+)", line):
                fields[m.group(1)] = m.group(2)
            return fields

    return {"seed": str(seed), "hit": "no_result", "output_tail": output[-200:]}


def parse_hit(fields: dict[str, str]) -> str:
    pos_err = float(fields.get("hit_pos_error", fields.get("pos_error", "999")))
    return "hit" if pos_err < 0.12 else "miss"


CONDITIONS = {
    "v2": {
        "script": "scripts/rm65_mpc_tube_constraint_realtime_v2.py",
        "extra": ["--use_tube", "false", "--max-tcp", "1.8"],
    },
    "v6": {
        "script": "scripts/rm65_mpc_v6.py",
        "extra": ["--target-speed", "5.0", "--fixed-direction", "--hit-shift", "0.01",
                   "--near-iters", "20"],
    },
    "v7": {
        "script": "scripts/rm65_mpc_v7.py",
        "extra": ["--hit-shift", "0.01", "--near-iters", "20"],
    },
}

FIELDS = [
    "seed", "pos_error", "vel_error", "min_dist", "ball_near_ms",
    "tube_ready_ms", "max_tcp", "max_qdot", "max_face", "hit_type",
    "hit_time_error_ms", "hit_pos_error", "v_racket_at_hit",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS.keys()),
                        choices=list(CONDITIONS.keys()))
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t_perturbs, s_perturbs = get_perturbations(args.seeds)

    crashed_seeds = {20, 21, 27}
    valid_seeds = [s for s in range(args.seeds) if s not in crashed_seeds]

    for cond_name in args.conditions:
        cond = CONDITIONS[cond_name]
        for perturb_label, with_perturb in [("no_perturb", False), ("perturb", True)]:
            csv_path = RESULTS_DIR / f"{cond_name}_{perturb_label}.csv"

            if args.skip_existing and csv_path.exists():
                print(f"[SKIP] {csv_path}")
                continue

            print(f"\n{'='*60}")
            print(f"  {cond_name} / {perturb_label}  ({len(valid_seeds)} seeds)")
            print(f"{'='*60}")

            rows: list[dict[str, str]] = []
            t0 = time.time()

            for seed in valid_seeds:
                t_p = t_perturbs[seed] if with_perturb else None
                s_p = s_perturbs[seed] if with_perturb else None
                fields = run_single(cond["script"], seed, t_p, s_p, cond["extra"])
                if fields is None:
                    fields = {"seed": str(seed), "hit": "error"}
                fields["hit"] = parse_hit(fields) if "hit" not in fields else fields["hit"]
                rows.append(fields)

                hit_tag = fields.get("hit", "?")
                pos_err = float(fields.get("hit_pos_error", fields.get("pos_error", "99")))
                v_rack = float(fields.get("v_racket_at_hit", "0"))
                print(f"  seed={seed:2d}  {hit_tag:4s}  pos={pos_err*100:5.1f}cm  "
                      f"v_rack={v_rack:.2f}m/s  ({time.time()-t0:.0f}s)")

            n_hit = sum(1 for r in rows if r.get("hit") == "hit")
            n_active = sum(1 for r in rows if r.get("hit_type") == "active" and r.get("hit") == "hit")
            v_racks = [float(r.get("v_racket_at_hit", "0")) for r in rows if r.get("hit") == "hit"]
            avg_v = np.mean(v_racks) if v_racks else 0
            print(f"  => {n_hit}/{len(rows)} hit ({100*n_hit/len(rows):.0f}%), "
                  f"{n_active} active, avg_v_racket={avg_v:.2f} m/s")

            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
                writer.writeheader()
                for r in rows:
                    writer.writerow(r)
            print(f"  saved: {csv_path}")


if __name__ == "__main__":
    main()

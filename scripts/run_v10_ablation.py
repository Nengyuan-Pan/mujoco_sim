"""V10 消融实验批量脚本。

6 组实验，每组 50 seeds，4 workers 并行。
用法:
    python scripts/run_v10_ablation.py 0 1 2 3 4 5
"""
import subprocess
import sys
import os
import csv
import re
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT = Path(__file__).parent / "rm65_mpc_v10.py"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "v10_ablation"
SEEDS = list(range(50))

BASE_CMD = [
    sys.executable, str(SCRIPT),
    "--serve-box", "--ball-speed", "7",
    "--hit-shift", "0.40",
    "--near-iters", "20",
    "--no-plot",
]

PERTURB_ARGS = [
    "--time-perturb-ms", "300",
    "--space-perturb-m", "0.15",
    "--perturb-alpha-min", "1.0",
]

EXPERIMENTS = [
    ("v10_full_nominal", []),
    ("v10_full_perturb", PERTURB_ARGS),
    ("v10_notube_nominal", ["--no-tube", "--no-softmin"]),
    ("v10_notube_perturb", ["--no-tube", "--no-softmin"] + PERTURB_ARGS),
    ("v10_nosoftmin_nominal", ["--no-softmin"]),
    ("v10_nosoftmin_perturb", ["--no-softmin"] + PERTURB_ARGS),
]

RESULT_RE = re.compile(
    r"pos_error=([\d.]+).*?min_dist=([\d.]+).*?"
    r"max_tcp=([\d.]+).*?max_qdot=([\d.]+).*?"
    r"hit_type=(\w+).*?"
    r"hit_time_error_ms=([-\d.]+).*?"
    r"v_racket_at_hit=([\d.]+)",
    re.DOTALL,
)

FIELDNAMES = [
    "seed", "pos_error", "min_dist", "max_tcp", "max_qdot",
    "hit_type", "hit_time_error_ms", "v_racket",
]


def _run_seed(args: tuple) -> dict:
    seed, extra_args = args
    cmd = BASE_CMD + extra_args + ["--seed", str(seed)]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=120,
                              env={**os.environ, "PYTHONUTF8": "1"})
        output = proc.stdout.decode("utf-8", errors="replace")
        m = RESULT_RE.search(output)
        if m:
            return {
                "seed": seed,
                "pos_error": float(m.group(1)),
                "min_dist": float(m.group(2)),
                "max_tcp": float(m.group(3)),
                "max_qdot": float(m.group(4)),
                "hit_type": m.group(5),
                "hit_time_error_ms": float(m.group(6)),
                "v_racket": float(m.group(7)),
            }
    except subprocess.TimeoutExpired:
        pass
    return {"seed": seed, "hit_type": "error"}


def run_experiment(name: str, extra_args: list[str]):
    csv_path = RESULTS_DIR / f"{name}.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    tasks = [(s, extra_args) for s in SEEDS]
    rows: list[dict] = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_run_seed, t): t for t in tasks}
        for i, f in enumerate(as_completed(futures), 1):
            rows.append(f.result())
            if i % 10 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] elapsed={time.time()-t0:.0f}s")

    rows.sort(key=lambda r: r["seed"])
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)

    hit = sum(1 for r in rows if r["hit_type"] in ("active", "passive"))
    act = sum(1 for r in rows if r["hit_type"] == "active")
    avg_pos = sum(r["pos_error"] for r in rows if r["hit_type"] != "error") / len(rows) * 100
    avg_v = sum(r["v_racket"] for r in rows if r["hit_type"] != "error") / len(rows)
    elapsed = time.time() - t0
    print(f"  => hit={hit}/50 ({hit*2}%) active={act}/50 ({act*2}%) "
          f"pos={avg_pos:.1f}cm v={avg_v:.2f}m/s [{elapsed:.0f}s]")


def main():
    if len(sys.argv) > 1:
        indices = [int(x) for x in sys.argv[1:]]
    else:
        indices = list(range(len(EXPERIMENTS)))

    t_start = time.time()
    for idx in indices:
        name, extra = EXPERIMENTS[idx]
        run_experiment(name, extra)

    print(f"\n全部完成！总耗时 {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()

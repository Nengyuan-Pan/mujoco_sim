"""球晚到实验: 随挥触发策略消融。

测试场景: 球比计划晚到 +50~100ms, 空间偏移 +3~8cm。
对比 3 种随挥策略 × 2 条件 (nominal / late) = 6 组。

随挥策略:
  - no_follow: 无随挥 (baseline)
  - planned: 到达计划击打时刻即触发随挥 PD (当前默认)
  - contact: 仅在球拍击球后触发随挥 PD, 球晚到时 MPC 继续追踪

用法:
    python scripts/exp/run_v9_late_ball.py
"""
import subprocess
import sys
import os
import csv
import re
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = PROJECT_ROOT / "scripts" / "rm65_mpc_v9.py"
RESULTS_DIR = PROJECT_ROOT / "results" / "v9_late_ball"
SEEDS = list(range(50))

BASE_CMD = [
    sys.executable, str(SCRIPT),
    "--serve-box", "--ball-speed", "7",
    "--hit-shift", "0.20",
    "--near-iters", "20",
    "--no-plot",
    "--ablation", "full",
]

LATE_PERTURB = [
    "--random-perturb",
    "--time-perturb-ms", "100",
    "--time-perturb-min-ms", "50",
    "--space-perturb-m", "0.08",
    "--space-perturb-min-m", "0.03",
    "--perturb-sign", "positive",
    "--perturb-alpha-min", "1.0",
]

EXPERIMENTS = [
    ("no_follow_nominal", ["--no-follow-through"]),
    ("planned_nominal", ["--follow-trigger", "planned"]),
    ("contact_nominal", ["--follow-trigger", "contact"]),
    ("no_follow_late", ["--no-follow-through"] + LATE_PERTURB),
    ("planned_late", ["--follow-trigger", "planned"] + LATE_PERTURB),
    ("contact_late", ["--follow-trigger", "contact"] + LATE_PERTURB),
]

RESULT_RE = re.compile(
    r"__RESULT__:\s+"
    r"pos_error=([\d.]+).*?"
    r"min_dist=([\d.]+).*?"
    r"max_tcp=([\d.]+).*?"
    r"max_qdot=([\d.]+).*?"
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
    hit_rows = [r for r in rows if r["hit_type"] in ("active", "passive")]
    avg_pos = sum(r["pos_error"] for r in hit_rows) / max(len(hit_rows), 1) * 100
    avg_v = sum(r["v_racket"] for r in hit_rows) / max(len(hit_rows), 1)
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

    print(f"\n全部完成！总耗时 {(time.time()-t_start)/60:.1f}min")

    print(f"\n{'='*70}")
    print(f"  球晚到实验结果汇总")
    print(f"{'='*70}")
    print(f"{'策略':<20} {'Nominal命中':>12} {'Late命中':>12} {'Nominal主动':>12} {'Late主动':>12}")
    print("-" * 70)
    strategies = ["no_follow", "planned", "contact"]
    for s in strategies:
        nom_path = RESULTS_DIR / f"{s}_nominal.csv"
        late_path = RESULTS_DIR / f"{s}_late.csv"
        nom_hit = nom_act = late_hit = late_act = 0
        if nom_path.exists():
            with open(nom_path, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
                nom_hit = sum(1 for r in rows if r["hit_type"] in ("active", "passive"))
                nom_act = sum(1 for r in rows if r["hit_type"] == "active")
        if late_path.exists():
            with open(late_path, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
                late_hit = sum(1 for r in rows if r["hit_type"] in ("active", "passive"))
                late_act = sum(1 for r in rows if r["hit_type"] == "active")
        print(f"{s:<20} {nom_hit:>10}/50 {late_hit:>10}/50 {nom_act:>10}/50 {late_act:>10}/50")


if __name__ == "__main__":
    main()

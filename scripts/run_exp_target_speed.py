"""v6 终端速度消融实验：1.8 vs 3.0 vs 5.0 m/s × 50 seeds × 扰动。

用 subprocess.Popen 并行启动多个 v6 进程。
"""
import subprocess
import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "rm65_mpc_v6.py"
DATE = "20260604"
OUT_DIR = ROOT / "results" / f"exp_target_speed_{DATE}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BALL_SPEED = 7.0
SERVE_DIST = 8.0
N_SEEDS = 50
ALPHA_MIN = 1.0
N_WORKERS = 8

meta_rng = np.random.default_rng(42)
t_perturbs = meta_rng.uniform(-300.0, 300.0, N_SEEDS)
s_perturbs = meta_rng.uniform(-0.15, 0.15, N_SEEDS)
np.savez(OUT_DIR / "perturbations.npz", t_perturbs=t_perturbs, s_perturbs=s_perturbs)

CONDITIONS = [
    {"name": "speed_1.8", "target_speed": 1.8},
    {"name": "speed_3.0", "target_speed": 3.0},
    {"name": "speed_5.0", "target_speed": 5.0},
]


def parse_result(output):
    for line in output.splitlines():
        if "__RESULT__:" in line:
            parts = line.split("__RESULT__:")[1].strip()
            kv = {}
            for token in parts.split():
                if "=" in token:
                    k, v = token.split("=", 1)
                    try:
                        kv[k] = float(v)
                    except ValueError:
                        kv[k] = v
            return kv
    return None


def build_args(target_speed, seed, t_ms, s_m):
    return [
        sys.executable, str(SCRIPT),
        "--serve-box",
        "--ball-speed", str(BALL_SPEED),
        "--serve-distance", str(SERVE_DIST),
        "--seed", str(seed),
        "--perturb-alpha-min", str(ALPHA_MIN),
        "--time-perturb-ms", f"{t_ms:.2f}",
        "--space-perturb-m", f"{s_m:.4f}",
        "--terminal-exempt-steps", "0",
        "--target-speed", str(target_speed),
        "--no-plot",
    ]


all_results = {c["name"]: [None] * N_SEEDS for c in CONDITIONS}

# 构建所有任务
tasks = []
for cond in CONDITIONS:
    for i in range(N_SEEDS):
        tasks.append((cond["name"], cond["target_speed"], i, t_perturbs[i], s_perturbs[i]))

# 分批并行执行
total = len(tasks)
done = 0
batch_size = N_WORKERS

print(f"开始并行实验: {total} 个任务, {N_WORKERS} workers")

idx = 0
while idx < total:
    batch = tasks[idx:idx + batch_size]
    procs = []
    for task in batch:
        cond_name, target_speed, seed, t_ms, s_m = task
        args = build_args(target_speed, seed, t_ms, s_m)
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             cwd=str(ROOT), text=True)
        procs.append((p, task))

    for p, task in procs:
        output, _ = p.communicate(timeout=300)
        cond_name, _, seed, t_ms, s_m = task
        r = parse_result(output)
        all_results[cond_name][seed] = {
            "seed": seed,
            "time_perturb_ms": float(t_ms),
            "space_perturb_m": float(s_m),
            "result": r,
        }
        done += 1
        hit = r.get("hit_type", "") in ("active", "passive") if r else False
        if done % 20 == 0 or not hit:
            tag = "HIT" if hit else "MISS"
            pos = r.get("pos_error", -1) if r else -1
            print(f"  [{done:3d}/{total}] {cond_name:>10s} seed={seed:2d}: {tag} pos={pos:.4f}m")

    idx += batch_size

with open(OUT_DIR / "raw_data.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n=== 终端速度消融实验 汇总 ===")
print(f"{'条件':>12s}  {'命中率':>8s}  {'主动率':>8s}  {'位置误差(cm)':>14s}  {'v_racket(m/s)':>14s}")
print("-" * 65)
for cond in CONDITIONS:
    name = cond["name"]
    runs = [r for r in all_results[name] if r is not None and r["result"] is not None]
    hits = [r for r in runs if r["result"].get("hit_type") in ("active", "passive")]
    active = [r for r in runs if r["result"].get("hit_type") == "active"]
    rate = len(hits) / len(runs) * 100 if runs else 0
    pe = [r["result"]["pos_error"] * 100 for r in runs]
    vr = [r["result"].get("v_racket_at_hit", 0) for r in hits]
    print(f"{name:>12s}  {len(hits)}/{len(runs)} ({rate:>3.0f}%)  "
          f"{len(active)}/{len(runs)} ({len(active)/len(runs)*100:>3.0f}%)  "
          f"{np.mean(pe):>6.1f}+/-{np.std(pe):.1f}      "
          f"{np.mean(vr):>6.2f}+/-{np.std(vr):.2f}")

print(f"\n数据已保存到 {OUT_DIR}")

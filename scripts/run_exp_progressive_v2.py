"""v6 逐步移植 v2 特性消融实验。

4 组条件 × 50 seeds × 扰动，用 Popen 并行。
条件：
  1. v6_baseline: 当前 v6（near_plan_iters=5，无走廊，无n_des）
  2. v6_iters20: near_plan_iters=20
  3. v6_iters20_corridor: iters20 + 走廊代价
  4. v7_full: 全部 v2 特性（iters20+走廊+n_des+平滑代价）
"""
import subprocess
import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V6_SCRIPT = ROOT / "scripts" / "rm65_mpc_v6.py"
V7_SCRIPT = ROOT / "scripts" / "rm65_mpc_v7.py"
DATE = "20260604"
OUT_DIR = ROOT / "results" / f"exp_progressive_v2_{DATE}"
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
    {"name": "v6_baseline", "script": "rm65_mpc_v6.py", "extra_args": []},
    {"name": "v6_iters20", "script": "rm65_mpc_v6.py", "extra_args": ["--near-iters", "20"]},
    {"name": "v6_iters20_corridor", "script": "rm65_mpc_v6.py",
     "extra_args": ["--near-iters", "20",
                     "--Q-tcp-soft", "0", "--Q-qdot-limit", "0"]},
    {"name": "v7_full_v2feat", "script": "rm65_mpc_v7.py", "extra_args": []},
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


all_results = {c["name"]: [None] * N_SEEDS for c in CONDITIONS}

tasks = []
for cond in CONDITIONS:
    for i in range(N_SEEDS):
        tasks.append((cond["name"], cond["script"], cond["extra_args"], i, t_perturbs[i], s_perturbs[i]))

total = len(tasks)
done = 0
idx = 0
print(f"开始实验: {total} 任务, {N_WORKERS} workers")

while idx < total:
    batch = tasks[idx:idx + N_WORKERS]
    procs = []
    for task in batch:
        cond_name, script, extra_args, seed, t_ms, s_m = task
        args = [
            sys.executable, str(ROOT / "scripts" / script),
            "--serve-box",
            "--ball-speed", str(BALL_SPEED),
            "--serve-distance", str(SERVE_DIST),
            "--seed", str(seed),
            "--perturb-alpha-min", str(ALPHA_MIN),
            "--time-perturb-ms", f"{t_ms:.2f}",
            "--space-perturb-m", f"{s_m:.4f}",
            "--terminal-exempt-steps", "0",
            "--no-plot",
        ] + extra_args
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             cwd=str(ROOT), text=True)
        procs.append((p, task))

    for p, task in procs:
        output, _ = p.communicate(timeout=300)
        cond_name, _, _, seed, t_ms, s_m = task
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
            print(f"  [{done:3d}/{total}] {cond_name:>20s} seed={seed:2d}: {tag} pos={pos:.4f}m")

    idx += N_WORKERS

with open(OUT_DIR / "raw_data.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n=== 逐步移植 v2 特性 消融实验 汇总 ===")
print(f"{'条件':>20s}  {'命中率':>8s}  {'主动率':>8s}  {'位置误差(cm)':>14s}  {'v_racket(m/s)':>14s}")
print("-" * 70)
for cond in CONDITIONS:
    name = cond["name"]
    runs = [r for r in all_results[name] if r is not None and r["result"] is not None]
    hits = [r for r in runs if r["result"].get("hit_type") in ("active", "passive")]
    active = [r for r in runs if r["result"].get("hit_type") == "active"]
    rate = len(hits) / len(runs) * 100 if runs else 0
    pe = [r["result"]["pos_error"] * 100 for r in runs]
    vr = [r["result"].get("v_racket_at_hit", 0) for r in hits]
    print(f"{name:>20s}  {len(hits)}/{len(runs)} ({rate:>3.0f}%)  "
          f"{len(active)}/{len(runs)} ({len(active)/len(runs)*100:>3.0f}%)  "
          f"{np.mean(pe):>6.1f}+/-{np.std(pe):.1f}      "
          f"{np.mean(vr):>6.2f}+/-{np.std(vr):.2f}")

print(f"\n数据已保存到 {OUT_DIR}")

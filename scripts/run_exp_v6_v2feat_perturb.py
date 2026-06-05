"""v6 + v2全特性（n_des+平滑代价+5.0速度+固定方向+iters20）× 50 seeds × 扰动。
用 Popen 并行。
"""
import subprocess
import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "rm65_mpc_v6.py"
DATE = "20260604"
OUT_DIR = ROOT / "results" / f"exp_v6_v2feat_perturb_{DATE}"
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


all_results = [None] * N_SEEDS
tasks = []
for i in range(N_SEEDS):
    tasks.append((i, t_perturbs[i], s_perturbs[i]))

total = len(tasks)
done = 0
idx = 0
print(f"开始实验: {total} 任务, {N_WORKERS} workers")

while idx < total:
    batch = tasks[idx:idx + N_WORKERS]
    procs = []
    for task in batch:
        seed, t_ms, s_m = task
        args = [
            sys.executable, str(SCRIPT),
            "--serve-box",
            "--ball-speed", str(BALL_SPEED),
            "--serve-distance", str(SERVE_DIST),
            "--seed", str(seed),
            "--perturb-alpha-min", str(ALPHA_MIN),
            "--time-perturb-ms", f"{t_ms:.2f}",
            "--space-perturb-m", f"{s_m:.4f}",
            "--terminal-exempt-steps", "0",
            "--target-speed", "5.0",
            "--fixed-direction",
            "--hit-shift", "0.01",
            "--near-iters", "20",
            "--realtime",
            "--no-softmin",
            "--no-plot",
        ]
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             cwd=str(ROOT), text=True)
        procs.append((p, task))

    for p, task in procs:
        output, _ = p.communicate(timeout=300)
        seed, t_ms, s_m = task
        r = parse_result(output)
        all_results[seed] = {
            "seed": seed,
            "time_perturb_ms": float(t_ms),
            "space_perturb_m": float(s_m),
            "result": r,
        }
        done += 1
        hit = r.get("hit_type", "") in ("active", "passive") if r else False
        if done % 10 == 0 or not hit:
            tag = "HIT" if hit else "MISS"
            pos = r.get("pos_error", -1) if r else -1
            vr = r.get("v_racket_at_hit", -1) if r else -1
            print(f"  [{done:3d}/{total}] seed={seed:2d}: {tag} pos={pos:.4f}m vr={vr:.2f}")

    idx += N_WORKERS

with open(OUT_DIR / "raw_data.json", "w", encoding="utf-8") as f:
    json.dump({"v6_v2feat": all_results}, f, ensure_ascii=False, indent=2)

runs = [r for r in all_results if r is not None and r["result"] is not None]
hits = [r for r in runs if r["result"].get("hit_type") in ("active", "passive")]
active = [r for r in runs if r["result"].get("hit_type") == "active"]
rate = len(hits) / len(runs) * 100 if runs else 0
pe = [r["result"]["pos_error"] * 100 for r in runs]
vr = [r["result"].get("v_racket_at_hit", 0) for r in hits]

print(f"\n=== v6 + v2全特性（扰动） 汇总 ===")
print(f"命中率: {len(hits)}/{len(runs)} ({rate:.0f}%)")
print(f"主动率: {len(active)}/{len(runs)} ({len(active)/len(runs)*100:.0f}%)")
print(f"位置误差: {np.mean(pe):.1f}+/-{np.std(pe):.1f}cm")
print(f"v_racket: {np.mean(vr):.2f}+/-{np.std(vr):.2f}m/s")
print(f"\n数据已保存到 {OUT_DIR}")

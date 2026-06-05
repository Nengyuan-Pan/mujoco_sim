"""v7 走廊代价消融实验：v6(无走廊) vs v7(有走廊) × 扰动。

2 组条件 × 50 seeds，扰动 Δt∈[-300,+300]ms, Δs∈[-15,+15]cm。
v6 作为 baseline（无走廊代价），v7 恢复了走廊代价 + near_plan_iters=20 + n_des + 平滑代价。
"""
import subprocess
import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATE = "20260604"
OUT_DIR = ROOT / "results" / f"exp_corridor_ablation_{DATE}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BALL_SPEED = 7.0
SERVE_DIST = 8.0
N_SEEDS = 50
ALPHA_MIN = 1.0

meta_rng = np.random.default_rng(42)
t_perturbs = meta_rng.uniform(-300.0, 300.0, N_SEEDS)
s_perturbs = meta_rng.uniform(-0.15, 0.15, N_SEEDS)
np.savez(OUT_DIR / "perturbations.npz", t_perturbs=t_perturbs, s_perturbs=s_perturbs)

CONDITIONS = [
    {"name": "v6_no_corridor", "script": "rm65_mpc_v6.py"},
    {"name": "v7_corridor",    "script": "rm65_mpc_v7.py"},
]


def run_one(script_name, args_list):
    script = ROOT / "scripts" / script_name
    cmd = [sys.executable, str(script)] + args_list
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(ROOT))
        output = result.stdout + result.stderr
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
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
    return None


all_results = {}
for cond in CONDITIONS:
    name = cond["name"]
    script = cond["script"]
    runs = []
    for i in range(N_SEEDS):
        t_ms = t_perturbs[i]
        s_m = s_perturbs[i]

        args = [
            "--serve-box",
            "--ball-speed", str(BALL_SPEED),
            "--serve-distance", str(SERVE_DIST),
            "--seed", str(i),
            "--perturb-alpha-min", str(ALPHA_MIN),
            "--time-perturb-ms", f"{t_ms:.2f}",
            "--space-perturb-m", f"{s_m:.4f}",
            "--terminal-exempt-steps", "0",
            "--no-plot",
        ]

        r = run_one(script, args)
        runs.append({
            "seed": i,
            "time_perturb_ms": float(t_ms),
            "space_perturb_m": float(s_m),
            "result": r,
        })
        hit = r.get("hit_type", "") in ("active", "passive") if r else False
        pos = r.get("pos_error", -1) if r else -1
        vr = r.get("v_racket_at_hit", -1) if r else -1
        if i % 10 == 0:
            tag = "HIT" if hit else "MISS"
            print(f"  {name:>20s} seed={i:2d} t={t_ms:+7.1f}ms s={s_m*100:+6.1f}cm: "
                  f"{tag} pos={pos:.4f}m vr={vr:.2f}")

    all_results[name] = runs

    runs_ok = [r for r in runs if r["result"] is not None]
    hits = [r for r in runs_ok if r["result"].get("hit_type") in ("active", "passive")]
    active = [r for r in runs_ok if r["result"].get("hit_type") == "active"]
    rate = len(hits) / len(runs_ok) * 100 if runs_ok else 0
    pos_errs = [r["result"]["pos_error"] * 100 for r in runs_ok]
    v_rackets = [r["result"].get("v_racket_at_hit", 0) for r in hits]
    print(f"  >> {name:>20s}: {len(hits)}/{len(runs_ok)} ({rate:.0f}%) "
          f"active={len(active)}/{len(runs_ok)} "
          f"pos={np.mean(pos_errs):.1f}+/-{np.std(pos_errs):.1f}cm "
          f"v_racket={np.mean(v_rackets):.2f}+/-{np.std(v_rackets):.2f}m/s")

with open(OUT_DIR / "raw_data.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n=== 走廊代价消融实验 汇总 ===")
print(f"{'条件':>20s}  {'命中率':>8s}  {'主动率':>8s}  {'位置误差(cm)':>14s}  {'v_racket(m/s)':>14s}")
print("-" * 75)
for cond in CONDITIONS:
    name = cond["name"]
    runs = [r for r in all_results[name] if r["result"] is not None]
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

"""v6 方向×偏移 2×2 消融实验。

4 组条件 × 50 seeds，扰动 Δt∈[-300,+300]ms, Δs∈[-15,+15]cm。
验证固定方向 vs ball-reverse 方向、1cm vs 5cm 偏移对鲁棒性的影响。

条件：
  1. fixed_1cm   : 固定方向 + hit_shift=0.01  (≈v3 配置)
  2. fixed_5cm   : 固定方向 + hit_shift=0.05
  3. breverse_1cm: ball-reverse + hit_shift=0.01
  4. breverse_5cm: ball-reverse + hit_shift=0.05 (当前 v6 默认)
"""
import subprocess
import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "rm65_mpc_v6.py"
DATE = "20260604"
OUT_DIR = ROOT / "results" / f"exp_direction_shift_{DATE}"
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
    {"name": "fixed_1cm",    "fixed_dir": True,  "hit_shift": 0.01},
    {"name": "fixed_5cm",    "fixed_dir": True,  "hit_shift": 0.05},
    {"name": "breverse_1cm", "fixed_dir": False, "hit_shift": 0.01},
    {"name": "breverse_5cm", "fixed_dir": False, "hit_shift": 0.05},
]


def run_one(args_list):
    cmd = [sys.executable, str(SCRIPT)] + args_list
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
            "--hit-shift", str(cond["hit_shift"]),
            "--no-plot",
        ]
        if cond["fixed_dir"]:
            args.append("--fixed-direction")

        r = run_one(args)
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
            print(f"  {name:>16s} seed={i:2d} t={t_ms:+7.1f}ms s={s_m*100:+6.1f}cm: "
                  f"{tag} pos={pos:.4f}m vr={vr:.2f}")

    all_results[name] = runs

    runs_ok = [r for r in runs if r["result"] is not None]
    hits = [r for r in runs_ok if r["result"].get("hit_type") in ("active", "passive")]
    rate = len(hits) / len(runs_ok) * 100 if runs_ok else 0
    pos_errs = [r["result"]["pos_error"] * 100 for r in runs_ok]
    v_rackets = [r["result"].get("v_racket_at_hit", 0) for r in hits]
    print(f"  >> {name:>16s}: {len(hits)}/{len(runs_ok)} ({rate:.0f}%) "
          f"pos={np.mean(pos_errs):.1f}+/-{np.std(pos_errs):.1f}cm "
          f"v_racket={np.mean(v_rackets):.2f}+/-{np.std(v_rackets):.2f}m/s")

with open(OUT_DIR / "raw_data.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n=== v6 方向×偏移 消融实验 汇总 ===")
print(f"{'条件':>16s}  {'命中率':>8s}  {'位置误差(cm)':>14s}  {'v_racket(m/s)':>14s}")
print("-" * 60)
for cond in CONDITIONS:
    name = cond["name"]
    runs = [r for r in all_results[name] if r["result"] is not None]
    hits = [r for r in runs if r["result"].get("hit_type") in ("active", "passive")]
    rate = len(hits) / len(runs) * 100 if runs else 0
    pe = [r["result"]["pos_error"] * 100 for r in runs]
    vr = [r["result"].get("v_racket_at_hit", 0) for r in hits]
    print(f"{name:>16s}  {len(hits)}/{len(runs)} ({rate:>3.0f}%)  "
          f"{np.mean(pe):>6.1f}+/-{np.std(pe):.1f}      "
          f"{np.mean(vr):>6.2f}+/-{np.std(vr):.2f}")

print(f"\n数据已保存到 {OUT_DIR}")

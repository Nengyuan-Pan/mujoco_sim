"""v6 消融实验: Softmin × 随挥 独立贡献消融。

4 组条件，每组 50 seeds，随机扰动 Δt∈[-300,+300]ms, Δs∈[-15,+15]cm。
perturb_alpha_min=1.0（不衰减），v6 内置 TCP/qdot 软约束。
--terminal-exempt-steps 0: 全程硬约束，与 v5 消融实验对齐。

条件设计：
  1. Full v6       : softmin=ON,  follow_through=ON   (完整 v6)
  2. No Softmin    : softmin=OFF, follow_through=ON   (消融 softmin)
  3. No FollowThru : softmin=ON,  follow_through=OFF  (消融随挥)
  4. Baseline      : softmin=OFF, follow_through=OFF  (v6 最简配置)
"""
import subprocess
import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "rm65_mpc_v6.py"
DATE = "20260604"
OUT_DIR = ROOT / "results" / f"exp_ablation_v6_{DATE}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BALL_SPEED = 7.0
SERVE_DIST = 8.0
N_SEEDS = 50
ALPHA_MIN = 1.0

# 随机扰动采样（与之前实验相同的 meta seed）
meta_rng = np.random.default_rng(42)
t_perturbs = meta_rng.uniform(-300.0, 300.0, N_SEEDS)
s_perturbs = meta_rng.uniform(-0.15, 0.15, N_SEEDS)
np.savez(OUT_DIR / "perturbations.npz", t_perturbs=t_perturbs, s_perturbs=s_perturbs)

# 实验条件定义
CONDITIONS = [
    {"name": "full_v6",       "no_softmin": False, "no_follow_through": False},
    {"name": "no_softmin",    "no_softmin": True,  "no_follow_through": False},
    {"name": "no_follow_thru", "no_softmin": False, "no_follow_through": True},
    {"name": "baseline",      "no_softmin": True,  "no_follow_through": True},
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
            "--no-plot",
        ]
        if cond["no_softmin"]:
            args.append("--no-softmin")
        if cond["no_follow_through"]:
            args.append("--no-follow-through")

        r = run_one(args)
        runs.append({
            "seed": i,
            "time_perturb_ms": float(t_ms),
            "space_perturb_m": float(s_m),
            "result": r,
        })
        hit = r.get("hit_type", "") in ("active", "passive") if r else False
        pos = r.get("pos_error", -1) if r else -1
        if i % 10 == 0:
            print(f"  {name:>16s} seed={i:2d} t={t_ms:+7.1f}ms s={s_m*100:+6.1f}cm: "
                  f"{'HIT' if hit else 'MISS'} pos={pos:.4f}m")

    all_results[name] = runs

    runs_ok = [r for r in runs if r["result"] is not None]
    hits = [r for r in runs_ok if r["result"].get("hit_type") in ("active", "passive")]
    rate = len(hits) / len(runs_ok) * 100 if runs_ok else 0
    pos_errs = [r["result"]["pos_error"] * 100 for r in runs_ok]
    v_rackets = [r["result"].get("v_racket_at_hit", 0) for r in hits]
    print(f"  >> {name:>16s}: {len(hits)}/{len(runs_ok)} ({rate:.0f}%) "
          f"pos={np.mean(pos_errs):.1f}±{np.std(pos_errs):.1f}cm "
          f"v_racket={np.mean(v_rackets):.2f}±{np.std(v_rackets):.2f}m/s")

with open(OUT_DIR / "raw_data.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n=== v6 消融实验 汇总 ===")
print(f"{'条件':>16s} {'成功率':>8s} {'位置误差(cm)':>14s} {'v_racket(m/s)':>14s}")
print("-" * 60)
for cond in CONDITIONS:
    name = cond["name"]
    runs = [r for r in all_results[name] if r["result"] is not None]
    hits = [r for r in runs if r["result"].get("hit_type") in ("active", "passive")]
    rate = len(hits) / len(runs) * 100 if runs else 0
    pe = [r["result"]["pos_error"] * 100 for r in runs]
    vr = [r["result"].get("v_racket_at_hit", 0) for r in hits]
    print(f"{name:>16s} {len(hits)}/{len(runs)} ({rate:>3.0f}%) "
          f"{np.mean(pe):>6.1f}±{np.std(pe):.1f} "
          f"{np.mean(vr):>6.2f}±{np.std(vr):.2f}")

print(f"\n数据已保存到 {OUT_DIR}")

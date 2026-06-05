"""消融实验A: 走廊代价 vs Softmin 终端代价的分离消融。

8 组条件，每组 50 seeds，随机扰动 Δt∈[-300,+300]ms, Δs∈[-15,+15]cm。
perturb_alpha_min=1.0（不衰减），全程 TCP≤1.8m/s + qdot≤1.0× 硬约束。

条件设计：
  1. No-Tube          : softmin=OFF, corridor=OFF
  2. Corridor 1.0     : softmin=OFF, corridor=1.0
  3. Corridor 0.3     : softmin=OFF, corridor=0.3
  4. Softmin only     : softmin=ON,  corridor=0.0
  5. S+0.1C           : softmin=ON,  corridor=0.1
  6. S+0.3C (默认)     : softmin=ON,  corridor=0.3
  7. S+0.5C           : softmin=ON,  corridor=0.5
  8. S+1.0C           : softmin=ON,  corridor=1.0
"""
import subprocess
import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = ROOT / "scripts" / "run_tcp_limit_experiment_v3.py"
DATE = "20260602"
OUT_DIR = ROOT / "results" / f"exp_ablation_corridor_{DATE}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BALL_SPEED = 7.0
N_SEEDS = 50
ALPHA_MIN = 1.0

# 随机扰动采样
meta_rng = np.random.default_rng(42)
t_perturbs = meta_rng.uniform(-300.0, 300.0, N_SEEDS)
s_perturbs = meta_rng.uniform(-0.15, 0.15, N_SEEDS)
np.savez(OUT_DIR / "perturbations.npz", t_perturbs=t_perturbs, s_perturbs=s_perturbs)

# 实验条件定义
CONDITIONS = [
    {"name": "no_tube",        "use_tube": "false", "no_softmin": False, "ratio": None},
    {"name": "corridor_1.0",   "use_tube": "true",  "no_softmin": True,  "ratio": 1.0},
    {"name": "corridor_0.3",   "use_tube": "true",  "no_softmin": True,  "ratio": 0.3},
    {"name": "softmin_only",   "use_tube": "true",  "no_softmin": False, "ratio": 0.0},
    {"name": "s+c0.1",         "use_tube": "true",  "no_softmin": False, "ratio": 0.1},
    {"name": "s+c0.3",         "use_tube": "true",  "no_softmin": False, "ratio": 0.3},
    {"name": "s+c0.5",         "use_tube": "true",  "no_softmin": False, "ratio": 0.5},
    {"name": "s+c1.0",         "use_tube": "true",  "no_softmin": False, "ratio": 1.0},
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
                        try: kv[k] = float(v)
                        except ValueError: kv[k] = v
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
            "--ball-speed", str(BALL_SPEED),
            "--seed", str(i),
            "--use-tube", cond["use_tube"],
            "--perturb-alpha-min", str(ALPHA_MIN),
            "--time-perturb-ms", f"{t_ms:.2f}",
            "--space-perturb-m", f"{s_m:.4f}",
        ]
        if cond["no_softmin"]:
            args.append("--no-softmin")
        if cond["ratio"] is not None:
            args.extend(["--tube-cost-ratio", str(cond["ratio"])])

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
            print(f"  {name:>14s} seed={i:2d} t={t_ms:+7.1f}ms s={s_m*100:+6.1f}cm: "
                  f"{'HIT' if hit else 'MISS'} pos={pos:.4f}m")

    all_results[name] = runs

    runs_ok = [r for r in runs if r["result"] is not None]
    hits = [r for r in runs_ok if r["result"].get("hit_type") in ("active", "passive")]
    rate = len(hits) / len(runs_ok) * 100 if runs_ok else 0
    pos_errs = [r["result"]["pos_error"] * 100 for r in runs_ok]
    print(f"  >> {name:>14s}: {len(hits)}/{len(runs_ok)} ({rate:.0f}%) "
          f"pos={np.mean(pos_errs):.1f}±{np.std(pos_errs):.1f}cm")

with open(OUT_DIR / "raw_data.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n=== 消融实验A 汇总 ===")
print(f"{'条件':>14s} {'成功率':>8s} {'位置误差(cm)':>14s}")
print("-" * 42)
for cond in CONDITIONS:
    name = cond["name"]
    runs = [r for r in all_results[name] if r["result"] is not None]
    hits = [r for r in runs if r["result"].get("hit_type") in ("active", "passive")]
    rate = len(hits) / len(runs) * 100 if runs else 0
    pe = [r["result"]["pos_error"] * 100 for r in runs]
    print(f"{name:>14s} {len(hits)}/{len(runs)} ({rate:>3.0f}%) {np.mean(pe):>6.1f}±{np.std(pe):.1f}")

print(f"\n数据已保存到 {OUT_DIR}")

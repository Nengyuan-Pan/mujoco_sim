"""测试不同 near_iters 配置在多 seed 下的表现.
replan_interval=20 固定, 测试 near_iters=3/5/10/20.
"""
import subprocess
import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = ROOT / "scripts" / "exp" / "run_tcp_limit_experiment_v3.py"
V5_SCRIPT = ROOT / "scripts" / "sim" / "rm65_mpc_tube_constraint_realtime_v5.py"

BALL_SPEED = 7.0
N_SEEDS = 20
REPLAN_INTERVAL = 20

CONFIGS = [
    {"name": "near3", "near_iters": 3},
    {"name": "near5", "near_iters": 5},
    {"name": "near10", "near_iters": 10},
    {"name": "near20", "near_iters": 20},
]

meta_rng = np.random.default_rng(42)
seeds = list(range(N_SEEDS))

all_results = {}
for cfg in CONFIGS:
    name = cfg["name"]
    runs = []
    for seed in seeds:
        cmd = [
            sys.executable, str(V5_SCRIPT),
            "--serve-box",
            "--ball-speed", str(BALL_SPEED),
            "--seed", str(seed),
            "--replan-interval", str(REPLAN_INTERVAL),
            "--near-iters", str(cfg["near_iters"]),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=str(ROOT))
            output = result.stdout + result.stderr
            kv = {}
            for line in output.splitlines():
                if "__RESULT__:" in line:
                    parts = line.split("__RESULT__:")[1].strip()
                    for token in parts.split():
                        if "=" in token:
                            k, v = token.split("=", 1)
                            try:
                                kv[k] = float(v)
                            except ValueError:
                                kv[k] = v
            # 提取 near 阶段性能
            near_avg_ms = None
            near_max_ms = None
            total_time_s = None
            for line in output.splitlines():
                if "稳态重规划" in line and "次)" in line:
                    parts = line.split("avg=")
                    if len(parts) > 1:
                        near_avg_ms = float(parts[1].split("ms")[0].strip())
                if "near" in line and "avg=" in line and "k_hit<=" in line:
                    parts = line.split("avg=")
                    if len(parts) > 1:
                        near_avg_ms = float(parts[1].split("ms")[0].strip())
                    parts = line.split("max=")
                    if len(parts) > 1:
                        near_max_ms = float(parts[1].split("ms")[0].strip())
                if "MPC 完成:" in line or "MPC计算:" in line:
                    parts = line.split("MPC=")
                    if len(parts) > 1:
                        total_time_s = float(parts[1].split("s")[0].strip())

            runs.append({
                "seed": seed,
                "result": kv,
                "near_avg_ms": near_avg_ms,
                "near_max_ms": near_max_ms,
                "total_time_s": total_time_s,
            })
            hit = kv.get("hit_type", "") in ("active", "passive") if kv else False
            pos = kv.get("pos_error", -1) if kv else -1
            if seed % 5 == 0:
                print(f"  {name:>8s} seed={seed}: {'HIT' if hit else 'MISS'} pos={pos*100:.1f}cm "
                      f"near_avg={near_avg_ms}ms near_max={near_max_ms}ms total={total_time_s}s")
        except Exception as e:
            print(f"  {name:>8s} seed={seed}: ERROR {e}")
            runs.append({"seed": seed, "result": None, "near_avg_ms": None,
                         "near_max_ms": None, "total_time_s": None})

    all_results[name] = runs

    runs_ok = [r for r in runs if r["result"] is not None]
    hits = [r for r in runs_ok if r["result"].get("hit_type") in ("active", "passive")]
    rate = len(hits) / len(runs_ok) * 100 if runs_ok else 0
    pe = [r["result"]["pos_error"] * 100 for r in runs_ok]
    na = [r["near_avg_ms"] for r in runs if r["near_avg_ms"] is not None]
    nm = [r["near_max_ms"] for r in runs if r["near_max_ms"] is not None]
    tt = [r["total_time_s"] for r in runs if r["total_time_s"] is not None]
    budget_ok = sum(1 for m in nm if m is not None and m <= 200) / len(nm) * 100 if nm else 0
    print(f"  >> {name:>8s}: {len(hits)}/{len(runs_ok)} ({rate:.0f}%) "
          f"pos={np.mean(pe):.1f}±{np.std(pe):.1f}cm "
          f"near_avg={np.mean(na):.0f}ms near_max_avg={np.mean(nm):.0f}ms "
          f"budget_ok={budget_ok:.0f}% total={np.mean(tt):.1f}s")

OUT_DIR = ROOT / "results" / "exp_near_iters_20260602"
OUT_DIR.mkdir(parents=True, exist_ok=True)
with open(OUT_DIR / "raw_data.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n数据已保存到 {OUT_DIR}")
print("\n=== 汇总 ===")
print(f"{'配置':>8s} {'成功率':>8s} {'位置误差':>12s} {'near_avg':>10s} {'near_max':>10s} {'预算内':>8s} {'总时间':>8s}")
print("-" * 75)
for cfg in CONFIGS:
    name = cfg["name"]
    runs = [r for r in all_results[name] if r["result"] is not None]
    hits = [r for r in runs if r["result"].get("hit_type") in ("active", "passive")]
    rate = len(hits) / len(runs) * 100 if runs else 0
    pe = [r["result"]["pos_error"] * 100 for r in runs]
    na = [r for r in all_results[name] if r["near_avg_ms"] is not None]
    nm = [r for r in all_results[name] if r["near_max_ms"] is not None]
    tt = [r for r in all_results[name] if r["total_time_s"] is not None]
    budget_ok = sum(1 for r in nm if r["near_max_ms"] <= 200) / len(nm) * 100 if nm else 0
    print(f"{name:>8s} {len(hits)}/{len(runs)} ({rate:>3.0f}%) "
          f"{np.mean(pe):>5.1f}±{np.std(pe):.1f}cm "
          f"{np.mean([r['near_avg_ms'] for r in na]):>8.0f}ms "
          f"{np.mean([r['near_max_ms'] for r in nm]):>8.0f}ms "
          f"{budget_ok:>6.0f}% "
          f"{np.mean([r['total_time_s'] for r in tt]):>6.1f}s")

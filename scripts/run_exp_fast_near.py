"""测试 fast_lin + fp_limits=None 在 k_hit≤30 时的效果.
replan_interval=20, 多 seed, 对比 near_iters=5/10.
"""
import subprocess
import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
V5_SCRIPT = ROOT / "scripts" / "rm65_mpc_tube_constraint_realtime_v5.py"

BALL_SPEED = 7.0
N_SEEDS = 30
REPLAN_INTERVAL = 20

CONFIGS = [
    {"name": "fast_near5", "near_iters": 5},
    {"name": "fast_near10", "near_iters": 10},
]

all_results = {}
for cfg in CONFIGS:
    name = cfg["name"]
    runs = []
    for seed in range(N_SEEDS):
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

            near_avg_ms = None
            near_max_ms = None
            total_time_s = None
            budget_ok_str = ""
            for line in output.splitlines():
                if "near" in line and "avg=" in line and "k_hit<=" in line:
                    import re
                    m = re.search(r'avg=(\d+)ms', line)
                    if m:
                        near_avg_ms = float(m.group(1))
                    m = re.search(r'max=(\d+)ms', line)
                    if m:
                        near_max_ms = float(m.group(1))
                    if "[OK]" in line:
                        budget_ok_str = "OK"
                    elif "[!!]" in line:
                        budget_ok_str = "OVER"
                if "MPC" in line and ("完成" in line or "计算" in line) and "MPC=" in line:
                    m = re.search(r'MPC=([\d.]+)s', line)
                    if m:
                        total_time_s = float(m.group(1))

            runs.append({
                "seed": seed,
                "result": kv,
                "near_avg_ms": near_avg_ms,
                "near_max_ms": near_max_ms,
                "total_time_s": total_time_s,
                "budget_ok": budget_ok_str,
            })
            hit = kv.get("hit_type", "") in ("active", "passive") if kv else False
            pos = kv.get("pos_error", -1) if kv else -1
            qdot = kv.get("max_qdot", -1) if kv else -1
            if seed % 10 == 0:
                print(f"  {name:>12s} seed={seed:2d}: {'HIT' if hit else 'MISS'} pos={pos*100:.1f}cm "
                      f"qdot={qdot:.2f}x near_avg={near_avg_ms}ms near_max={near_max_ms}ms "
                      f"budget={budget_ok_str}")
        except Exception as e:
            print(f"  {name:>12s} seed={seed:2d}: ERROR {e}")
            runs.append({"seed": seed, "result": None, "near_avg_ms": None,
                         "near_max_ms": None, "total_time_s": None, "budget_ok": "ERR"})

    all_results[name] = runs
    runs_ok = [r for r in runs if r["result"] is not None and isinstance(r["result"], dict) and "pos_error" in r["result"]]
    hits = [r for r in runs_ok if r["result"].get("hit_type") in ("active", "passive")]
    rate = len(hits) / len(runs_ok) * 100 if runs_ok else 0
    pe = [r["result"]["pos_error"] * 100 for r in runs_ok]
    na = [r["near_avg_ms"] for r in runs if r["near_avg_ms"] is not None]
    nm = [r["near_max_ms"] for r in runs if r["near_max_ms"] is not None]
    qd = [r["result"].get("max_qdot", 0) for r in runs_ok]
    budget_ok_count = sum(1 for r in runs if r["budget_ok"] == "OK")
    print(f"  >> {name:>12s}: {len(hits)}/{len(runs_ok)} ({rate:.0f}%) "
          f"pos={np.mean(pe):.1f}±{np.std(pe):.1f}cm "
          f"near_avg={np.mean(na):.0f}ms near_max={np.mean(nm):.0f}ms "
          f"max_qdot={np.max(qd):.2f}x budget_ok={budget_ok_count}/{len(runs)}")

OUT_DIR = ROOT / "results" / "exp_near_iters_20260602"
OUT_DIR.mkdir(parents=True, exist_ok=True)
with open(OUT_DIR / "fast_lin_data.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print(f"\n数据已保存到 {OUT_DIR}")

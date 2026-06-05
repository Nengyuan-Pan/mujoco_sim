"""v3 实验2（重设计）: 球速耦合扰动实验。

物理自洽的扰动方式：对发球初速度施加比例扰动（±5%, ±10%），
实际球以扰动速度飞行，但 MPC 规划器仍使用标称球速。
球速变化自然导致飞行时间和落点位置同时偏移，避免独立时间/空间扰动的耦合假象。

实验矩阵：
  - 球速扰动: -10%, -5%, 0%, +5%, +10%
  - 模式: Tube / No-Tube
  - 每条件 20 seeds
  - 总计: 5 × 2 × 20 = 200 runs

输出: results/v3_exp2_coupled_raw.json
"""
import subprocess
import sys
import json
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = ROOT / "scripts" / "exp" / "run_tcp_limit_experiment_v3.py"

BALL_SPEED = 7.0
N_SEEDS = 20
PERTURB_PCTS = [-20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
MODES = ["tube", "no_tube"]


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

for mode in MODES:
    use_tube = "true" if mode == "tube" else "false"
    for pct in PERTURB_PCTS:
        key = f"{mode}/pct={pct}"
        runs = []
        for seed in range(N_SEEDS):
            args = [
                "--ball-speed", str(BALL_SPEED),
                "--seed", str(seed),
                "--use-tube", use_tube,
            ]
            if abs(pct) > 0.01:
                args.extend(["--ball-speed-perturb-pct", str(pct)])

            r = run_one(args)
            runs.append({"seed": seed, "result": r})

            hit = r.get("hit_type", "") in ("active", "passive") if r else False
            pos = r.get("pos_error", -1) if r else -1
            print(f"  {mode} pct={pct:+.0f}% seed={seed}: {'HIT' if hit else 'MISS'} pos={pos:.4f}m")

        all_results[key] = runs

# 保存原始数据
out_path = ROOT / "results" / "v3_exp2_coupled_raw.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print(f"\n原始数据已保存到 {out_path}")

# 打印统计
print("\n=== 统计结果 ===")
print(f"{'模式':<10} {'扰动':>6} {'成功率':>10} {'位置误差(cm)':>15}")
print("-" * 50)
for mode in MODES:
    for pct in PERTURB_PCTS:
        key = f"{mode}/pct={pct}"
        runs = [r["result"] for r in all_results[key] if r["result"] is not None]
        if not runs:
            print(f"{mode:<10} {pct:>+5.0f}%  {'N/A':>10}")
            continue
        hits = [r for r in runs if r.get("hit_type") in ("active", "passive")]
        pos_errs = [r["pos_error"] * 100 for r in runs]
        rate = len(hits) / len(runs) * 100
        mean_e = np.mean(pos_errs)
        std_e = np.std(pos_errs)
        print(f"{mode:<10} {pct:>+5.0f}%  {len(hits)}/{len(runs)} ({rate:>.0f}%)   "
              f"{mean_e:.1f}±{std_e:.1f}")

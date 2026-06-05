"""实验B: 衰减因子扫描，perturb_alpha_min 0.0~1.0，每组100 seeds。

固定时间扰动 +200ms，空间扰动 0cm（最能体现 Tube 优势的条件）。
对比不同衰减因子下 tube 和 no_tube 的表现。
"""
import subprocess
import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = ROOT / "scripts" / "exp" / "run_tcp_limit_experiment_v3.py"
DATE = "20260602"
OUT_DIR = ROOT / "results" / f"exp_random_robustness_{DATE}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BALL_SPEED = 7.0
N_SEEDS = 20
TIME_PERTURB = 200.0
SPACE_PERTURB = 0.0
ALPHAS = [round(a * 0.1, 1) for a in range(0, 11)]  # 0.0, 0.1, ..., 1.0
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
                        try: kv[k] = float(v)
                        except ValueError: kv[k] = v
                return kv
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
    return None


all_results = {}
for alpha in ALPHAS:
    for mode in MODES:
        use_tube = "true" if mode == "tube" else "false"
        key = f"{mode}/alpha={alpha}"
        runs = []
        for seed in range(N_SEEDS):
            args = [
                "--ball-speed", str(BALL_SPEED),
                "--seed", str(seed),
                "--use-tube", use_tube,
                "--perturb-alpha-min", str(alpha),
                "--time-perturb-ms", str(TIME_PERTURB),
            ]
            if abs(SPACE_PERTURB) > 0.001:
                args.extend(["--space-perturb-m", str(SPACE_PERTURB)])

            r = run_one(args)
            runs.append({"seed": seed, "result": r})
            hit = r.get("hit_type", "") in ("active", "passive") if r else False
            if seed % 10 == 0:
                pos = r.get("pos_error", -1) if r else -1
                print(f"  alpha={alpha:.1f} {mode} seed={seed:2d}: "
                      f"{'HIT' if hit else 'MISS'} pos={pos:.4f}m")

        all_results[key] = runs
        runs_ok = [r for r in runs if r["result"] is not None]
        hits = [r for r in runs_ok if r["result"].get("hit_type") in ("active", "passive")]
        rate = len(hits) / len(runs_ok) * 100 if runs_ok else 0
        print(f"  >> alpha={alpha:.1f} {mode}: {len(hits)}/{len(runs_ok)} ({rate:.0f}%)")

with open(OUT_DIR / "expB_raw.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print("\n=== 实验B 统计 ===")
print(f"{'alpha':>6} {'Tube':>8} {'NoTube':>8} {'差异':>6}")
print("-" * 35)
for alpha in ALPHAS:
    rates = {}
    for mode in MODES:
        key = f"{mode}/alpha={alpha}"
        runs = [r for r in all_results[key] if r["result"] is not None]
        hits = [r for r in runs if r["result"].get("hit_type") in ("active", "passive")]
        rates[mode] = len(hits) / len(runs) * 100 if runs else 0
    diff = rates["tube"] - rates["no_tube"]
    print(f"{alpha:>5.1f} {rates['tube']:>7.0f}% {rates['no_tube']:>7.0f}% {diff:>+5.0f}pp")

print(f"\n数据已保存到 {OUT_DIR}")

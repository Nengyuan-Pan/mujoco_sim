"""v3 实验2（最终版）: 纯时间预测误差实验。

物理自洽的扰动方式：球的实际轨迹不变，MPC 对到达时间的预测有误差。
通过 perturb_alpha_min=1.0 保持扰动不衰减，模拟持续性的预测偏差。

Tube 的 softmin 多终端机制天然对时间不确定性有鲁棒性，
此实验直接验证这一点。

实验矩阵：
  - 时间扰动: ±100ms, ±200ms, ±300ms, 0ms
  - 模式: Tube / No-Tube
  - 每条件 20 seeds
  - 总计: 7 × 2 × 20 = 280 runs
"""
import subprocess
import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = ROOT / "scripts" / "run_tcp_limit_experiment_v3.py"

BALL_SPEED = 7.0
N_SEEDS = 20
TIME_PERTURBS = [-300, -200, -100, 0, 100, 200, 300]
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
    for t_ms in TIME_PERTURBS:
        key = f"{mode}/t={t_ms}"
        runs = []
        for seed in range(N_SEEDS):
            args = [
                "--ball-speed", str(BALL_SPEED),
                "--seed", str(seed),
                "--use-tube", use_tube,
                "--perturb-alpha-min", "1.0",
            ]
            if abs(t_ms) > 0.01:
                args.extend(["--time-perturb-ms", str(t_ms)])

            r = run_one(args)
            runs.append({"seed": seed, "result": r})

            hit = r.get("hit_type", "") in ("active", "passive") if r else False
            pos = r.get("pos_error", -1) if r else -1
            print(f"  {mode} t={t_ms:+d}ms seed={seed}: {'HIT' if hit else 'MISS'} pos={pos:.4f}m")

        all_results[key] = runs

# 保存原始数据
out_path = ROOT / "results" / "v3_exp2_time_raw.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)
print(f"\n原始数据已保存到 {out_path}")

# 打印统计
print("\n=== 统计结果 ===")
print(f"{'模式':<10} {'扰动':>8} {'成功率':>10} {'位置误差(cm)':>15}")
print("-" * 55)
for mode in MODES:
    for t_ms in TIME_PERTURBS:
        key = f"{mode}/t={t_ms}"
        runs = [r["result"] for r in all_results[key] if r["result"] is not None]
        if not runs:
            print(f"{mode:<10} {t_ms:>+6d}ms  {'N/A':>10}")
            continue
        hits = [r for r in runs if r.get("hit_type") in ("active", "passive")]
        pos_errs = [r["pos_error"] * 100 for r in runs]
        rate = len(hits) / len(runs) * 100
        mean_e = np.mean(pos_errs)
        std_e = np.std(pos_errs)
        print(f"{mode:<10} {t_ms:>+6d}ms  {len(hits)}/{len(runs)} ({rate:>.0f}%)   "
              f"{mean_e:.1f}±{std_e:.1f}")

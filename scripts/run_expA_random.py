"""实验A: 随机时间+空间扰动，50 seeds，tube vs no_tube。

每次实验随机采样时间扰动 [-300ms, +300ms] 和空间扰动 [-15cm, +15cm]，
用相同的随机扰动分别跑 tube 和 no_tube，对比成功率。

perturb_alpha_min=1.0（扰动不衰减），模拟持续性感知偏差。
"""
import subprocess
import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "run_tcp_limit_experiment_v3.py"
DATE = "20260602"
OUT_DIR = ROOT / "results" / f"exp_random_robustness_{DATE}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BALL_SPEED = 7.0
N_SEEDS = 50
ALPHA_MIN = 1.0
T_RANGE = (-300.0, 300.0)
S_RANGE = (-0.15, 0.15)
MODES = ["tube", "no_tube"]

meta_rng = np.random.default_rng(42)
t_perturbs = meta_rng.uniform(T_RANGE[0], T_RANGE[1], N_SEEDS)
s_perturbs = meta_rng.uniform(S_RANGE[0], S_RANGE[1], N_SEEDS)

np.savez(OUT_DIR / "perturbations.npz", t_perturbs=t_perturbs, s_perturbs=s_perturbs)


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
for mode in MODES:
    use_tube = "true" if mode == "tube" else "false"
    runs = []
    for i in range(N_SEEDS):
        t_ms = t_perturbs[i]
        s_m = s_perturbs[i]
        args = [
            "--ball-speed", str(BALL_SPEED),
            "--seed", str(i),
            "--use-tube", use_tube,
            "--perturb-alpha-min", str(ALPHA_MIN),
            "--time-perturb-ms", f"{t_ms:.2f}",
            "--space-perturb-m", f"{s_m:.4f}",
        ]
        r = run_one(args)
        runs.append({
            "seed": i,
            "time_perturb_ms": float(t_ms),
            "space_perturb_m": float(s_m),
            "result": r,
        })
        hit = r.get("hit_type", "") in ("active", "passive") if r else False
        pos = r.get("pos_error", -1) if r else -1
        print(f"  {mode} seed={i:2d} t={t_ms:+7.1f}ms s={s_m*100:+6.1f}cm: "
              f"{'HIT' if hit else 'MISS'} pos={pos:.4f}m")
    all_results[mode] = runs

with open(OUT_DIR / "expA_raw.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print("\n=== 实验A 统计 ===")
for mode in MODES:
    runs = [r for r in all_results[mode] if r["result"] is not None]
    hits = [r for r in runs if r["result"].get("hit_type") in ("active", "passive")]
    pos_errs = [r["result"]["pos_error"] * 100 for r in runs]
    print(f"{mode}: {len(hits)}/{len(runs)} ({len(hits)/len(runs)*100:.0f}%) "
          f"pos={np.mean(pos_errs):.1f}±{np.std(pos_errs):.1f}cm")

print(f"\n数据已保存到 {OUT_DIR}")

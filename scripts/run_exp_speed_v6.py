"""v6 球速实验: 在不同球速下统计任务成功率。

球速范围: 5, 6, 7, 8, 9, 10, 11 m/s
每组 50 seeds，随机扰动 Δt∈[-300,+300]ms, Δs∈[-15,+15]cm。
如果某球速下目标区域不可达，自动扩大 Y 轴发球区域。
"""
import subprocess
import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "rm65_mpc_v6.py"
DATE = "20260603"
OUT_DIR = ROOT / "results" / f"exp_speed_v6_{DATE}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_SEEDS = 50
ALPHA_MIN = 1.0

# 球速 → 发球参数映射
BALL_SPEED_CONFIG = {
    5.0:  {"serve_dist": 5.7,  "serve_y_size": 0.2, "no_bounce": False},
    6.0:  {"serve_dist": 6.8,  "serve_y_size": 0.2, "no_bounce": False},
    7.0:  {"serve_dist": 8.0,  "serve_y_size": 0.2, "no_bounce": False},
    8.0:  {"serve_dist": 9.5,  "serve_y_size": 0.2, "no_bounce": False},
    9.0:  {"serve_dist": 11.0, "serve_y_size": 0.4, "no_bounce": True},
    10.0: {"serve_dist": 12.5, "serve_y_size": 0.6, "no_bounce": True},
    11.0: {"serve_dist": 14.0, "serve_y_size": 0.8, "no_bounce": True},
}

# 随机扰动采样
meta_rng = np.random.default_rng(42)
t_perturbs = meta_rng.uniform(-300.0, 300.0, N_SEEDS)
s_perturbs = meta_rng.uniform(-0.15, 0.15, N_SEEDS)
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
                        try:
                            kv[k] = float(v)
                        except ValueError:
                            kv[k] = v
                return kv
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
    return None


all_results = {}
for ball_speed, cfg in sorted(BALL_SPEED_CONFIG.items()):
    name = f"speed_{ball_speed:.0f}"
    serve_dist = cfg["serve_dist"]
    serve_y_size = cfg["serve_y_size"]
    use_no_bounce = cfg.get("no_bounce", False)
    runs = []

    print(f"\n--- 球速 {ball_speed:.0f} m/s (serve_dist={serve_dist}m, y_size={serve_y_size}m, bounce={'OFF' if use_no_bounce else 'ON'}) ---")

    for i in range(N_SEEDS):
        t_ms = t_perturbs[i]
        s_m = s_perturbs[i]

        args = [
            "--serve-box",
            "--ball-speed", str(ball_speed),
            "--serve-distance", str(serve_dist),
            "--serve-y-size", str(serve_y_size),
            "--seed", str(i),
            "--perturb-alpha-min", str(ALPHA_MIN),
            "--time-perturb-ms", f"{t_ms:.2f}",
            "--space-perturb-m", f"{s_m:.4f}",
            "--no-softmin",
            "--no-follow-through",
            "--no-plot",
        ]
        if use_no_bounce:
            args.append("--no-bounce")

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
            print(f"  speed={ball_speed:.0f} seed={i:2d} t={t_ms:+7.1f}ms s={s_m*100:+6.1f}cm: "
                  f"{'HIT' if hit else 'MISS'} pos={pos:.4f}m")

    all_results[name] = runs

    runs_ok = [r for r in runs if r["result"] is not None]
    hits = [r for r in runs_ok if r["result"].get("hit_type") in ("active", "passive")]
    rate = len(hits) / len(runs_ok) * 100 if runs_ok else 0
    pos_errs = [r["result"]["pos_error"] * 100 for r in runs_ok]
    face_speeds = [r["result"].get("max_face", 0) for r in runs_ok]
    print(f"  >> speed={ball_speed:.0f}m/s: {len(hits)}/{len(runs_ok)} ({rate:.0f}%) "
          f"pos={np.mean(pos_errs):.1f}±{np.std(pos_errs):.1f}cm "
          f"face={np.mean(face_speeds):.1f}m/s")

with open(OUT_DIR / "raw_data.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n=== v6 球速实验 汇总 ===")
print(f"{'球速(m/s)':>10s} {'成功率':>8s} {'位置误差(cm)':>14s} {'Face速度(m/s)':>14s}")
print("-" * 52)
for ball_speed in sorted(BALL_SPEED_CONFIG.keys()):
    name = f"speed_{ball_speed:.0f}"
    runs = [r for r in all_results[name] if r["result"] is not None]
    hits = [r for r in runs if r["result"].get("hit_type") in ("active", "passive")]
    rate = len(hits) / len(runs) * 100 if runs else 0
    pe = [r["result"]["pos_error"] * 100 for r in runs]
    fs = [r["result"].get("max_face", 0) for r in runs]
    print(f"{ball_speed:>10.0f} {len(hits)}/{len(runs)} ({rate:>3.0f}%) "
          f"{np.mean(pe):>6.1f}±{np.std(pe):.1f} "
          f"{np.mean(fs):>6.1f}±{np.std(fs):.1f}")

print(f"\n数据已保存到 {OUT_DIR}")

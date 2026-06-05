"""v2+v5 鲁棒性测试: 时间+空间扰动下的成功率。

50 seeds，随机扰动 Δt∈[-300,+300]ms, Δs∈[-15,+15]cm。
perturb_alpha_min=1.0（不衰减），全程硬约束 TCP≤1.8m/s + qdot≤1.0×。

条件：
  1. v2_softmin     : v2(v3 wrapper) + softmin=ON, corridor=0.0（之前96%的配置）
  2. v2_no_softmin  : v2(v3 wrapper) + softmin=OFF, 无 tube
  3. v5_softmin     : v5 直接跑 + softmin=ON, corridor=0.0
  4. v5_no_softmin  : v5 直接跑 + softmin=OFF, 无 tube
"""
import subprocess
import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
V2_SCRIPT = ROOT / "scripts" / "exp" / "run_tcp_limit_experiment_v3.py"
V5_SCRIPT = ROOT / "scripts" / "sim" / "rm65_mpc_tube_constraint_realtime_v5.py"
DATE = "20260603"
OUT_DIR = ROOT / "results" / f"exp_v5_robustness_{DATE}"
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
    {"name": "v2_softmin",    "script": "v2", "no_softmin": False, "use_tube": "true",
     "tube_cost_ratio": 0.0},
    {"name": "v2_no_softmin", "script": "v2", "no_softmin": True,  "use_tube": "false",
     "tube_cost_ratio": None},
    {"name": "v5_softmin",    "script": "v5", "no_softmin": False, "use_tube": "true",
     "tube_cost_ratio": 0.0},
    {"name": "v5_no_softmin", "script": "v5", "no_softmin": True,  "use_tube": "false",
     "tube_cost_ratio": None},
]


def run_one(args_list, script_path):
    cmd = [sys.executable, str(script_path)] + args_list
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
    script = V2_SCRIPT if cond["script"] == "v2" else V5_SCRIPT
    runs = []
    for i in range(N_SEEDS):
        t_ms = t_perturbs[i]
        s_m = s_perturbs[i]

        if cond["script"] == "v2":
            # v2 通过 v3 wrapper 调用，参数格式不同
            args = [
                "--ball-speed", str(BALL_SPEED),
                "--seed", str(i),
                "--perturb-alpha-min", str(ALPHA_MIN),
                "--time-perturb-ms", f"{t_ms:.2f}",
                "--space-perturb-m", f"{s_m:.4f}",
            ]
            if cond["no_softmin"]:
                args.append("--no-softmin")
            if cond.get("use_tube"):
                args.extend(["--use-tube", cond["use_tube"]])
            if cond.get("tube_cost_ratio") is not None:
                args.extend(["--tube-cost-ratio", str(cond["tube_cost_ratio"])])
        else:
            # v5 直接调用
            args = [
                "--serve-box",
                "--ball-speed", str(BALL_SPEED),
                "--serve-distance", str(SERVE_DIST),
                "--seed", str(i),
                "--no-backswing",
                "--no-plot",
                "--terminal-exempt-steps", "0",
                "--perturb-alpha-min", str(ALPHA_MIN),
                "--time-perturb-ms", f"{t_ms:.2f}",
                "--space-perturb-m", f"{s_m:.4f}",
            ]
            if cond["no_softmin"]:
                args.append("--no-softmin")
            if cond.get("use_tube"):
                args.extend(["--use_tube", cond["use_tube"]])
            if cond.get("tube_cost_ratio") is not None:
                args.extend(["--tube-cost-ratio", str(cond["tube_cost_ratio"])])

        r = run_one(args, script)
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
    vel_errs = [r["result"].get("vel_error", 0) for r in runs_ok]
    tcp_max = max([r["result"].get("max_tcp", 0) for r in runs_ok]) if runs_ok else 0
    print(f"  >> {name:>16s}: {len(hits)}/{len(runs_ok)} ({rate:.0f}%) "
          f"pos={np.mean(pos_errs):.1f}±{np.std(pos_errs):.1f}cm "
          f"vel_err={np.mean(vel_errs):.1f}m/s "
          f"tcp_max={tcp_max:.2f}m/s")

with open(OUT_DIR / "raw_data.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n=== v5 鲁棒性测试 汇总 ===")
print(f"{'条件':>16s} {'成功率':>8s} {'位置误差(cm)':>14s} {'vel_err(m/s)':>13s} {'TCP_max':>8s}")
print("-" * 66)
for cond in CONDITIONS:
    name = cond["name"]
    runs = [r for r in all_results[name] if r["result"] is not None]
    hits = [r for r in runs if r["result"].get("hit_type") in ("active", "passive")]
    rate = len(hits) / len(runs) * 100 if runs else 0
    pe = [r["result"]["pos_error"] * 100 for r in runs] if runs else [0]
    ve = [r["result"].get("vel_error", 0) for r in runs] if runs else [0]
    tcp = [r["result"].get("max_tcp", 0) for r in runs] if runs else [0]
    print(f"{name:>16s} {len(hits)}/{len(runs)} ({rate:>3.0f}%) "
          f"{np.mean(pe):>6.1f}±{np.std(pe):.1f} "
          f"{np.mean(ve):>6.1f}±{np.std(ve):.1f} "
          f"{max(tcp):>6.2f}")

print(f"\n数据已保存到 {OUT_DIR}")

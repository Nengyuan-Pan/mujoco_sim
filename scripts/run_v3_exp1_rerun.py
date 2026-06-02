"""v3 实验1: 球速 5/6/7/8 m/s, 自适应发球距离, 20 seeds each."""
import subprocess
import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "run_tcp_limit_experiment_v3.py"

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

speeds = [5, 6, 7, 8]
n_seeds = 20

tasks = []
for speed in speeds:
    for seed in range(n_seeds):
        tasks.append((speed, seed, ["--ball-speed", str(speed), "--seed", str(seed)]))

print(f"=== 球速 {speeds} m/s, 自适应发球距离, 每速 {n_seeds} seeds ===")
all_results = {}
for speed, seed, args in tasks:
    r = run_one(args)
    if speed not in all_results:
        all_results[speed] = []
    all_results[speed].append((seed, r))
    status = "HIT" if r and r.get("hit_type") in ("active", "passive") else "MISS"
    pos = r.get("pos_error", -1) if r else -1
    print(f"  speed={speed} seed={seed}: {status} pos_err={pos:.4f}m")

# 保存原始数据
out_path = ROOT / "results" / "v3_exp1_raw.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
serializable = {str(k): [{"seed": s, "result": r} for s, r in v] for k, v in all_results.items()}
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(serializable, f, ensure_ascii=False, indent=2)
print(f"\n原始数据已保存到 {out_path}")

# 科研格式统计输出
print("\n=== 科研格式统计 ===")
for speed in sorted(all_results.keys()):
    runs = [r for _, r in all_results[speed] if r is not None]
    hits = [r for r in runs if r.get("hit_type") in ("active", "passive")]
    pos_errs = [r["pos_error"] for r in runs]
    mean_e = np.mean(pos_errs)
    std_e = np.std(pos_errs)
    min_e = np.min(pos_errs)
    print(f"  speed={speed}: {len(hits)}/{len(runs)} ({len(hits)/len(runs)*100:.0f}%) "
          f"pos_err={mean_e:.3f}±{std_e:.3f} (min={min_e:.4f})")

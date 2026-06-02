"""实验1批量运行脚本：算法能力上限（速度豁免模式）。

用法:
    python scripts/run_exp1_batch.py
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "experiment_data" / "exp1_algorithm_capability" / "raw"
LOG_DIR.mkdir(parents=True, exist_ok=True)

BALL_SPEEDS = [9, 15, 18, 20]
SEEDS = list(range(20))
TUBE_MODES = ["true", "false"]

script = str(ROOT / "scripts" / "rm65_mpc_tube_constraint.py")

total = len(BALL_SPEEDS) * len(SEEDS) * len(TUBE_MODES)
done = 0
skipped = 0
failed = 0
t_start = time.perf_counter()

for speed in BALL_SPEEDS:
    for tube in TUBE_MODES:
        for seed in SEEDS:
            tag = f"speed{speed}_tube{tube.capitalize()}_seed{seed}"
            log_path = LOG_DIR / f"{tag}.log"

            if log_path.exists() and "max_qdot" in log_path.read_text(encoding="utf-8", errors="ignore"):
                skipped += 1
                done += 1
                continue

            cmd = [
                sys.executable, script,
                "--serve-box", "--ball-speed", str(speed),
                "--seed", str(seed),
                "--use_tube", tube,
                "--no-backswing", "--no-plot",
                "--horizon", "120", "--iter", "10",
                "--replan-interval", "10",
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=120,
                    cwd=str(ROOT),
                )
                log_path.write_text(result.stdout + result.stderr, encoding="utf-8")

                if "max_qdot" in result.stdout:
                    done += 1
                else:
                    failed += 1
                    done += 1
                    print(f"  [WARN] {tag}: no max_qdot in output")
            except subprocess.TimeoutExpired:
                failed += 1
                done += 1
                print(f"  [TIMEOUT] {tag}")
            except Exception as e:
                failed += 1
                done += 1
                print(f"  [ERROR] {tag}: {e}")

            elapsed = time.perf_counter() - t_start
            eta = elapsed / done * (total - done) if done > 0 else 0
            print(f"[{done}/{total}] speed={speed} tube={tube} seed={seed} "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

print(f"\n完成: {done}/{total}, 跳过(已有): {skipped}, 失败: {failed}")
print(f"总耗时: {time.perf_counter() - t_start:.1f}s")

"""批量运行 exp2_strict_joint_v2 实验（球速扫参 + Tube 对比）。

用法:
    python scripts/run_exp2_v2_batch.py
"""
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "experiment_data" / "exp2_strict_joint_v2" / "raw"
SCRIPT = PROJECT_ROOT / "scripts" / "rm65_mpc_tube_constraint.py"
PYTHON_EXE = Path(sys.executable)

SPEEDS = [5, 6, 7, 8]
SEEDS = list(range(10))
TUBE_MODES = ["true", "false"]


def run_one(speed: int, seed: int, tube: str) -> bool:
    """运行单次实验，返回 True 表示成功。"""
    log_path = RAW_DIR / f"speed{speed}_seed{seed}_tube_{tube}.log"
    if log_path.exists():
        print(f"  [SKIP] {log_path.name} — 已存在")
        return True

    cmd = [
        str(PYTHON_EXE),
        str(SCRIPT),
        "--serve-box",
        "--ball-speed", str(speed),
        "--seed", str(seed),
        "--use_tube", tube,
        "--no-backswing",
        "--no-plot",
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
            env={**os.environ, "PYTHONUTF8": "1"},
        )
        log_path.write_text(result.stdout, encoding="utf-8")
        if result.stderr:
            log_path.write_text(
                result.stdout + "\n\n=== STDERR ===\n" + result.stderr,
                encoding="utf-8",
            )
        return True
    except subprocess.TimeoutExpired:
        print(f"  [FAIL] {log_path.name} — 超时 (120s)")
        return False
    except Exception as e:
        print(f"  [FAIL] {log_path.name} — {e}")
        return False


def main() -> None:
    total = len(SPEEDS) * len(TUBE_MODES) * len(SEEDS)
    print(f"exp2_strict_joint_v2: {len(SPEEDS)} 球速 × {len(TUBE_MODES)} tube × {len(SEEDS)} seeds = {total} runs")
    print(f"日志目录: {RAW_DIR}")
    print()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    completed = 0
    skipped = 0
    failed = 0
    n = 0

    for speed in SPEEDS:
        for tube in TUBE_MODES:
            for seed in SEEDS:
                n += 1
                label = f"speed={speed} tube={tube} seed={seed}"
                print(f"[{n}/{total}] {label} ...", flush=True)
                success = run_one(speed, seed, tube)
                if success:
                    completed += 1
                else:
                    failed += 1

    print(f"\n完成: {completed} ok, {failed} failed")


if __name__ == "__main__":
    main()

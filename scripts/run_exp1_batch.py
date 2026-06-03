"""批量运行 exp1_algorithm_capability 实验（速度豁免 + 1m/s球速扫参）。

用法:
    python scripts/run_exp1_batch.py
"""
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "experiment_data" / "exp1_algorithm_capability" / "raw"
WRAPPER = PROJECT_ROOT / "scripts" / "_run_exp1_exempt.py"
PYTHON_EXE = Path(sys.executable)

SPEEDS = list(range(8, 19))  # 8-18 m/s
SEEDS = list(range(10))
TUBE_MODES = ["true", "false"]


def run_one(speed: int, seed: int, tube: str) -> bool:
    """运行单次实验。"""
    tag = f"speed{speed}_seed{seed}_tube_{tube}"
    log_path = RAW_DIR / f"{tag}.log"
    if log_path.exists():
        print(f"  [SKIP] {tag} — 已存在")
        return True

    cmd = [str(PYTHON_EXE), str(WRAPPER), str(speed), str(seed), tube]
    try:
        result = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT), capture_output=True,
            timeout=120, encoding="utf-8",
            env={**os.environ, "PYTHONUTF8": "1"},
        )
        content = result.stdout
        if result.stderr:
            # stderr might contain logging output (MuJoCo info goes to stderr)
            # Prioritize stderr for the offline script
            content = result.stderr if result.stderr.strip() else content
        log_path.write_text(content, encoding="utf-8")
        return True
    except subprocess.TimeoutExpired:
        print(f"  [FAIL] {tag} — 超时")
        return False
    except Exception as e:
        print(f"  [FAIL] {tag} — {e}")
        return False


def main() -> None:
    total = len(SPEEDS) * len(TUBE_MODES) * len(SEEDS)
    print(f"exp1_algorithm_capability (速度豁免): {len(SPEEDS)} 球速 × {len(TUBE_MODES)} tube × {len(SEEDS)} seeds = {total} runs")
    print(f"日志目录: {RAW_DIR}\n")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    n = 0
    ok = 0
    for speed in SPEEDS:
        for tube in TUBE_MODES:
            for seed in SEEDS:
                n += 1
                label = f"{speed}m/s tube={tube} seed={seed}"
                print(f"[{n}/{total}] {label} ...", flush=True)
                if run_one(speed, seed, tube):
                    ok += 1

    print(f"\n完成: {ok} ok, {total - ok} failed")


if __name__ == "__main__":
    main()

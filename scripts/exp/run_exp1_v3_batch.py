"""批量运行 exp1_v3_algorithm_capability 实验（多进程并行）。

用法:
    python scripts/run_exp1_v3_batch.py              # 默认 4 workers
    python scripts/run_exp1_v3_batch.py --workers 2   # 2 workers
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "experiment_data" / "exp1_v3_algorithm_capability" / "raw"
WRAPPER = PROJECT_ROOT / "scripts" / "_run_exp1_v3_exempt.py"
PYTHON_EXE = str(Path(sys.executable))

SPEEDS = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 25, 26, 28, 30]
SEEDS = list(range(15))
TUBE_MODES = ["true", "false"]


def run_one(args: tuple[int, int, str]) -> tuple[str, bool]:
    """在子进程中运行单次实验。"""
    speed, seed, tube = args
    tag = f"speed{speed}_seed{seed}_tube_{tube}"
    log_path = RAW_DIR / f"{tag}.log"
    if log_path.exists():
        return tag, True

    cmd = [PYTHON_EXE, str(WRAPPER), str(speed), str(seed), tube]
    try:
        result = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT), capture_output=True,
            timeout=180, encoding="utf-8",
            env={**os.environ, "PYTHONUTF8": "1"},
        )
        content = result.stderr if result.stderr.strip() else result.stdout
        log_path.write_text(content, encoding="utf-8")
        return tag, True
    except subprocess.TimeoutExpired:
        return tag, False
    except Exception as e:
        log_path.write_text(f"ERROR: {e}", encoding="utf-8")
        return tag, False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4, help="并行进程数")
    args = parser.parse_args()

    tasks = [(s, d, t) for s in SPEEDS for t in TUBE_MODES for d in SEEDS]
    total = len(tasks)
    print(f"exp1_v3 (no-bounce, 速度豁免): {len(SPEEDS)} 球速 × {len(TUBE_MODES)} tube × {len(SEEDS)} seeds = {total} runs")
    print(f"并行 workers: {args.workers}")
    print(f"日志目录: {RAW_DIR}\n")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ok = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, t): t for t in tasks}
        for i, f in enumerate(as_completed(futures), 1):
            tag, success = f.result()
            if success:
                ok += 1
            else:
                failed += 1
            elapsed = time.time() - t0
            eta = elapsed / i * (total - i) if i > 0 else 0
            if i % 20 == 0 or i == total:
                print(f"[{i}/{total}] ok={ok} fail={failed} elapsed={elapsed:.0f}s eta={eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\n完成: {ok} ok, {failed} failed, 耗时 {elapsed/60:.1f}min")


if __name__ == "__main__":
    main()

"""批量运行 exp11_regression 回归测试（多进程并行）。

实验矩阵: 5 球速 × 2 模式 × 2 噪声 × 50 seeds = 1000 runs

用法:
    python scripts/exp/run_exp11_batch.py              # 默认 8 workers
    python scripts/exp/run_exp11_batch.py --workers 8   # 指定 workers
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "experiment_data" / "exp11_regression" / "raw"
V11_SCRIPT = PROJECT_ROOT / "scripts" / "rm65_mpc_v11.py"
PYTHON_EXE = str(Path(sys.executable))

SPEEDS = [6, 7, 8, 9, 10]
MODES = ["torque", "position"]
NOISES = ["off", "lo"]
SEEDS = list(range(50))


def run_one(args: tuple[int, str, str, int]) -> tuple[str, bool]:
    """在子进程中运行单次实验。

    Args:
        args: (speed, mode, noise, seed) 元组

    Returns:
        (tag, success) 元组
    """
    speed, mode, noise, seed = args
    tag = f"speed{speed}_mode{mode}_noise{noise}_s{seed}"
    log_path = RAW_DIR / f"{tag}.log"
    if log_path.exists():
        return tag, True

    cmd = [
        PYTHON_EXE, str(V11_SCRIPT),
        "--serve-box", "--ball-speed", str(speed),
        "--seed", str(seed), "--no-plot",
    ]
    if mode == "position":
        cmd.append("--position-mode")
    if noise == "lo":
        cmd.extend(["--obs-noise-pos", "0.02", "--obs-noise-vel", "0.2"])

    try:
        result = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT), capture_output=True,
            timeout=120, encoding="utf-8",
            env={**os.environ, "PYTHONUTF8": "1"},
        )
        content = (result.stdout or "") + (result.stderr or "")
        log_path.write_text(content, encoding="utf-8")
        return tag, True
    except subprocess.TimeoutExpired:
        log_path.write_text("ERROR: timeout (120s)", encoding="utf-8")
        return tag, False
    except Exception as e:
        log_path.write_text(f"ERROR: {e}", encoding="utf-8")
        return tag, False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8, help="并行进程数")
    cli_args = parser.parse_args()

    tasks = [
        (speed, mode, noise, seed)
        for speed in SPEEDS
        for mode in MODES
        for noise in NOISES
        for seed in SEEDS
    ]
    total = len(tasks)
    print(f"exp11_regression 回归测试: {len(SPEEDS)} 球速 × {len(MODES)} 模式 "
          f"× {len(NOISES)} 噪声 × {len(SEEDS)} seeds = {total} runs")
    print(f"并行 workers: {cli_args.workers}")
    print(f"日志目录: {RAW_DIR}\n")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    ok = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=cli_args.workers) as pool:
        futures = {pool.submit(run_one, t): t for t in tasks}
        for i, f in enumerate(as_completed(futures), 1):
            tag, success = f.result()
            if success:
                ok += 1
            else:
                failed += 1
            elapsed = time.time() - t0
            eta = elapsed / i * (total - i) if i > 0 else 0
            if i % 100 == 0 or i == total:
                print(f"[{i}/{total}] ok={ok} fail={failed} "
                      f"elapsed={elapsed:.0f}s eta={eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\n完成: {ok} ok, {failed} failed, 耗时 {elapsed/60:.1f}min")


if __name__ == "__main__":
    main()

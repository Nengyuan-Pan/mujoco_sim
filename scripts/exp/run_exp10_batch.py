"""批量运行 exp10_pd_scan 精调实验（多进程并行）。

用法:
    python scripts/exp/run_exp10_batch.py              # 默认 8 workers
    python scripts/exp/run_exp10_batch.py --workers 8   # 指定 workers
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "experiment_data" / "exp10_pd_scan" / "raw"
WRAPPER = PROJECT_ROOT / "scripts" / "exp" / "_run_exp10_pd_scan.py"
PYTHON_EXE = str(Path(sys.executable))

KP_VALUES = [10, 15, 20, 25, 30, 35, 40, 50]
KD_RATIOS = [0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2]
DQ_FRACTIONS = [0.5]
RATIO_MODES = [0, 1, 2]
SEEDS = list(range(20))


def run_one(args: tuple[float, float, float, int, int]) -> tuple[str, bool]:
    """在子进程中运行单次实验。"""
    kp, kdr, dq, seed, rmode = args
    tag = f"kp{int(kp)}_kdr{kdr}_dq{dq}_r{rmode}_s{seed}"
    log_path = RAW_DIR / f"{tag}.log"
    if log_path.exists():
        return tag, True

    cmd = [PYTHON_EXE, str(WRAPPER), str(kp), str(kdr), str(dq), str(seed), str(rmode)]
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
        (kp, kdr, dq, s, rmode)
        for kp in KP_VALUES
        for kdr in KD_RATIOS
        for dq in DQ_FRACTIONS
        for rmode in RATIO_MODES
        for s in SEEDS
    ]
    total = len(tasks)
    print(f"exp10_pd_scan 精调: {len(KP_VALUES)} Kp × {len(KD_RATIOS)} Kd_ratio "
          f"× {len(DQ_FRACTIONS)} dq_frac × {len(RATIO_MODES)} ratio × {len(SEEDS)} seeds = {total} runs")
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
            if i % 200 == 0 or i == total:
                print(f"[{i}/{total}] ok={ok} fail={failed} "
                      f"elapsed={elapsed:.0f}s eta={eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\n完成: {ok} ok, {failed} failed, 耗时 {elapsed/60:.1f}min")


if __name__ == "__main__":
    main()

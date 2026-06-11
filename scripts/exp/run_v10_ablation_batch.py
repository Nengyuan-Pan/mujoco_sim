"""V10 消融实验批量运行器。

消融维度：tube(开/关) × softmin(开/关) × perturb(0%/10%) × ball_speed × seed

用法:
    python scripts/exp/run_v10_ablation_batch.py
    python scripts/exp/run_v10_ablation_batch.py --workers 4
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "experiment_data" / "v10_ablation" / "raw"
SCRIPT = PROJECT_ROOT / "scripts" / "rm65_mpc_v10.py"
PYTHON_EXE = str(Path(sys.executable))

SPEEDS = [7, 8, 9, 10, 11]
SEEDS = list(range(10))
TUBE_FLAGS = ["", "--no-tube"]
SOFTMIN_FLAGS = ["", "--no-softmin"]
PERTURBS = [0, 10]


def run_one(args: tuple) -> tuple[str, bool]:
    speed, seed, tube_flag, softmin_flag, perturb_pct = args
    parts = [f"s{speed}", f"seed{seed}"]
    if tube_flag:
        parts.append("notube")
    else:
        parts.append("tube")
    if softmin_flag:
        parts.append("nosoftmin")
    else:
        parts.append("softmin")
    if perturb_pct > 0:
        parts.append(f"p{perturb_pct}")
    else:
        parts.append("nominal")
    tag = "_".join(parts)
    log_path = RAW_DIR / f"{tag}.log"
    if log_path.exists():
        return tag, True

    cmd = [
        PYTHON_EXE, str(SCRIPT),
        "--serve-box",
        "--ball-speed", str(speed),
        "--seed", str(seed),
        "--no-plot",
        "--ball-speed-perturb-pct", str(perturb_pct),
    ]
    if tube_flag:
        cmd.append(tube_flag)
    if softmin_flag:
        cmd.append(softmin_flag)

    try:
        result = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT), capture_output=True,
            timeout=120, encoding="utf-8",
            env={**os.environ, "PYTHONUTF8": "1"},
        )
        content = result.stdout + "\n" + result.stderr
        log_path.write_text(content, encoding="utf-8")
        return tag, True
    except subprocess.TimeoutExpired:
        log_path.write_text("ERROR: timeout", encoding="utf-8")
        return tag, False
    except Exception as e:
        log_path.write_text(f"ERROR: {e}", encoding="utf-8")
        return tag, False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4, help="并行进程数")
    cli_args = parser.parse_args()

    tasks = [
        (s, d, tf, sf, p)
        for s in SPEEDS
        for d in SEEDS
        for tf in TUBE_FLAGS
        for sf in SOFTMIN_FLAGS
        for p in PERTURBS
    ]
    total = len(tasks)
    n_tube = len(TUBE_FLAGS)
    n_soft = len(SOFTMIN_FLAGS)
    n_perturb = len(PERTURBS)
    print(f"V10 消融: {len(SPEEDS)} speeds × {len(SEEDS)} seeds × {n_tube} tube × {n_soft} softmin × {n_perturb} perturb = {total} runs")
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
            if i % 50 == 0 or i == total:
                print(f"[{i}/{total}] ok={ok} fail={failed} elapsed={elapsed:.0f}s eta={eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\n完成: {ok} ok, {failed} failed, 耗时 {elapsed/60:.1f}min")


if __name__ == "__main__":
    main()

"""批量运行 exp9_obs_freq_robustness 实验（多进程并行）。

用法:
    python scripts/exp/run_exp9_batch.py              # 默认 4 workers
    python scripts/exp/run_exp9_batch.py --workers 8   # 8 workers

直接调用 V11 脚本（无需 wrapper），通过 CLI 参数控制观测门控。
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "experiment_data" / "exp9_obs_freq_robustness" / "raw"
V11_SCRIPT = PROJECT_ROOT / "scripts" / "rm65_mpc_v11.py"
PYTHON_EXE = str(Path(sys.executable))

SPEEDS = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
SEEDS = list(range(50))
FREQS = [200, 60, 30, 15, 10]
NOISES = [
    ("off", 0.0, 0.0),
    ("lo", 0.02, 0.2),
    ("mid", 0.05, 0.5),
]
KF_MODES = ["nokf", "kf"]
TUBE_MODES = ["full", "none"]


def run_one(args: tuple) -> tuple[str, bool]:
    """在子进程中运行单次实验。"""
    speed, seed, freq, noise_name, noise_pos, noise_vel, kf_mode, tube = args
    tag = f"ball{speed}_seed{seed}_f{freq}_{noise_name}_{kf_mode}_tube{'on' if tube == 'full' else 'off'}"
    log_path = RAW_DIR / f"{tag}.log"
    if log_path.exists():
        return tag, True

    cmd = [
        PYTHON_EXE, str(V11_SCRIPT),
        "--serve-box",
        "--ball-speed", str(speed),
        "--seed", str(seed),
        "--obs-freq", str(freq),
        "--ablation", tube,
        "--no-plot",
    ]
    if noise_pos > 0:
        cmd += ["--obs-noise-pos", str(noise_pos), "--obs-noise-vel", str(noise_vel)]
    if kf_mode == "kf":
        cmd += ["--obs-use-kf"]

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

    tasks = [
        (s, d, f, n_name, n_pos, n_vel, kf, t)
        for s in SPEEDS
        for f in FREQS
        for n_name, n_pos, n_vel in NOISES
        for kf in KF_MODES
        for t in TUBE_MODES
        for d in SEEDS
    ]
    total = len(tasks)
    print(f"exp9 (观测频率鲁棒性): {len(SPEEDS)} 球速 × {len(FREQS)} 频率 × "
          f"{len(NOISES)} 噪声 × {len(KF_MODES)} KF × {len(TUBE_MODES)} Tube × "
          f"{len(SEEDS)} seeds = {total} runs")
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
            if i % 100 == 0 or i == total:
                print(f"[{i}/{total}] ok={ok} fail={failed} elapsed={elapsed:.0f}s eta={eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\n完成: {ok} ok, {failed} failed, 耗时 {elapsed / 60:.1f}min")

    with open(RAW_DIR.parent / "_.COMPLETE", "w", encoding="utf-8") as f:
        import datetime
        f.write(f"DONE {datetime.datetime.now().strftime('%Y年 %m月 %d日 %A %H:%M:%S %Z')}\n")


if __name__ == "__main__":
    main()

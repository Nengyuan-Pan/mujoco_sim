"""批量运行 exp12 前馈补偿综合评估（多进程并行，3 phase 统一入口）。

用法:
    python scripts/exp/run_exp12_batch.py --phase A --workers 4
    python scripts/exp/run_exp12_batch.py --phase B --workers 4
    python scripts/exp/run_exp12_batch.py --phase C --workers 4
    python scripts/exp/run_exp12_batch.py --phase all --workers 4
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "experiment_data" / "exp12_feedforward"
V11 = PROJECT_ROOT / "scripts" / "rm65_mpc_v11.py"
PYTHON_EXE = str(Path(sys.executable))

# ==================== Phase A: 核心消融 ====================
PHASE_A_MODES = ["torque", "pos_ff", "pos_noff"]
PHASE_A_SPEEDS = [5, 6, 7, 8, 9, 10, 11, 12]
PHASE_A_SEEDS = list(range(50))
PHASE_A_KP_BASE = 20
PHASE_A_KD_RATIO = 0.08

# ==================== Phase B: PD 重扫描 ====================
PHASE_B_KP_BASES = [10, 15, 20, 25, 30]
PHASE_B_KD_RATIOS = [0.05, 0.08, 0.10, 0.12]
PHASE_B_SEEDS = list(range(20))
PHASE_B_BALL_SPEED = 7

# ==================== Phase C: 噪声鲁棒 ====================
PHASE_C_FF_MODES = ["on", "off"]
PHASE_C_NOISE_LEVELS = [
    ("clean", 0.0, 0.0, False),
    ("low", 0.03, 0.3, True),
    ("high", 0.05, 0.5, True),
]
PHASE_C_SEEDS = list(range(50))
PHASE_C_BALL_SPEED = 7
PHASE_C_KP_BASE = 20
PHASE_C_KD_RATIO = 0.08


def _make_kp_kd(kp_base: float, kd_ratio: float) -> tuple[list[float], list[float]]:
    """从 kp_base 和 kd_ratio 构造 uniform kp/kd 数组（6 维）。"""
    ratio = [1.0] * 6  # uniform ratio mode
    kp = [kp_base * r for r in ratio]
    kd = [v * kd_ratio for v in kp]
    return kp, kd


def _build_cmd_phase_a(mode: str, speed: int, seed: int) -> list[str]:
    """构造 Phase A 的 V11 命令行。"""
    cmd = [PYTHON_EXE, str(V11), "--serve-box",
           "--ball-speed", str(speed), "--seed", str(seed),
           "--no-plot", "--replan-interval", "20"]
    if mode == "pos_ff":
        kp, kd = _make_kp_kd(PHASE_A_KP_BASE, PHASE_A_KD_RATIO)
        cmd += ["--position-mode", "--kp", *[str(v) for v in kp],
                "--kd", *[str(v) for v in kd]]
    elif mode == "pos_noff":
        kp, kd = _make_kp_kd(PHASE_A_KP_BASE, PHASE_A_KD_RATIO)
        cmd += ["--position-mode", "--no-feedforward",
                "--kp", *[str(v) for v in kp],
                "--kd", *[str(v) for v in kd]]
    # torque: 无额外 flag
    return cmd


def _build_cmd_phase_b(kp_base: float, kd_ratio: float, seed: int) -> list[str]:
    """构造 Phase B 的 V11 命令行（全部 pos+FF）。"""
    kp, kd = _make_kp_kd(kp_base, kd_ratio)
    return [PYTHON_EXE, str(V11), "--serve-box",
            "--ball-speed", str(PHASE_B_BALL_SPEED), "--seed", str(seed),
            "--no-plot", "--replan-interval", "20",
            "--position-mode",
            "--kp", *[str(v) for v in kp],
            "--kd", *[str(v) for v in kd]]


def _build_cmd_phase_c(ff_mode: str, noise_name: str,
                       pos_std: float, vel_std: float, use_kf: bool,
                       seed: int) -> list[str]:
    """构造 Phase C 的 V11 命令行。"""
    kp, kd = _make_kp_kd(PHASE_C_KP_BASE, PHASE_C_KD_RATIO)
    cmd = [PYTHON_EXE, str(V11), "--serve-box",
           "--ball-speed", str(PHASE_C_BALL_SPEED), "--seed", str(seed),
           "--no-plot", "--replan-interval", "20",
           "--position-mode",
           "--kp", *[str(v) for v in kp],
           "--kd", *[str(v) for v in kd]]
    if ff_mode == "off":
        cmd.append("--no-feedforward")
    if pos_std > 0 or vel_std > 0:
        cmd += ["--obs-noise-pos", str(pos_std),
                "--obs-noise-vel", str(vel_std)]
    if use_kf:
        cmd.append("--obs-use-kf")
    return cmd


def run_one(args: tuple) -> tuple[str, bool]:
    """在子进程中运行单次实验。

    Args:
        args: (tag, cmd, raw_dir) 三元组。

    Returns:
        (tag, success) 二元组。
    """
    tag, cmd, raw_dir_str = args
    raw_dir = Path(raw_dir_str)
    log_path = raw_dir / f"{tag}.log"
    if log_path.exists():
        return tag, True

    try:
        result = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT), capture_output=True,
            timeout=120, encoding="utf-8",
            env={**os.environ, "PYTHONUTF8": "1"},
        )
        content = (result.stdout or "") + (result.stderr or "")
        if not content.strip():
            content = "ERROR: empty output"
            log_path.write_text(content, encoding="utf-8")
            return tag, False
        log_path.write_text(content, encoding="utf-8")
        return tag, True
    except subprocess.TimeoutExpired:
        log_path.write_text("ERROR: timeout (120s)", encoding="utf-8")
        return tag, False
    except Exception as e:
        log_path.write_text(f"ERROR: {e}", encoding="utf-8")
        return tag, False


def gen_tasks_phase_a(raw_dir: Path) -> list[tuple]:
    """生成 Phase A 任务列表。"""
    tasks = []
    for mode in PHASE_A_MODES:
        for speed in PHASE_A_SPEEDS:
            for seed in PHASE_A_SEEDS:
                tag = f"mode{mode}_speed{speed}_seed{seed}"
                cmd = _build_cmd_phase_a(mode, speed, seed)
                tasks.append((tag, cmd, str(raw_dir)))
    return tasks


def gen_tasks_phase_b(raw_dir: Path) -> list[tuple]:
    """生成 Phase B 任务列表。"""
    tasks = []
    for kp_base in PHASE_B_KP_BASES:
        for kd_ratio in PHASE_B_KD_RATIOS:
            for seed in PHASE_B_SEEDS:
                tag = f"kp{int(kp_base)}_kdr{kd_ratio}_seed{seed}"
                cmd = _build_cmd_phase_b(kp_base, kd_ratio, seed)
                tasks.append((tag, cmd, str(raw_dir)))
    return tasks


def gen_tasks_phase_c(raw_dir: Path) -> list[tuple]:
    """生成 Phase C 任务列表。"""
    tasks = []
    for ff_mode in PHASE_C_FF_MODES:
        for noise_name, pos_std, vel_std, use_kf in PHASE_C_NOISE_LEVELS:
            for seed in PHASE_C_SEEDS:
                tag = f"ff{ff_mode}_noise{noise_name}_seed{seed}"
                cmd = _build_cmd_phase_c(ff_mode, noise_name, pos_std, vel_std, use_kf, seed)
                tasks.append((tag, cmd, str(raw_dir)))
    return tasks


def run_phase(phase: str, workers: int) -> None:
    """运行指定 phase 的全部任务。"""
    phase_dir = DATA_DIR / f"phase{phase}"
    raw_dir = phase_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    gen_map = {"A": gen_tasks_phase_a, "B": gen_tasks_phase_b, "C": gen_tasks_phase_c}
    tasks = gen_map[phase](raw_dir)
    total = len(tasks)

    print(f"\n{'='*60}")
    print(f"Phase {phase}: {total} runs, {workers} workers")
    print(f"  raw_dir: {raw_dir}")
    print(f"{'='*60}")

    # 断点续传统计
    existing = len(list(raw_dir.glob("*.log")))
    if existing > 0:
        print(f"  已有 {existing} 个日志（断点续传）")

    t0 = time.time()
    ok, failed = 0, 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, t): t for t in tasks}
        for i, f in enumerate(as_completed(futures), 1):
            tag, success = f.result()
            if success:
                ok += 1
            else:
                failed += 1
            if i % 50 == 0 or i == total:
                elapsed = time.time() - t0
                eta = elapsed / i * (total - i) if i > 0 else 0
                print(f"  [{i}/{total}] ok={ok} fail={failed} "
                      f"elapsed={elapsed:.0f}s eta={eta:.0f}s")

    elapsed_min = (time.time() - t0) / 60
    print(f"Phase {phase} 完成: {ok} ok, {failed} failed, {elapsed_min:.1f}min")

    # 提取结果
    print(f"  提取结果...")
    extract = PROJECT_ROOT / "scripts" / "extract" / "extract_exp12_results.py"
    result = subprocess.run(
        [PYTHON_EXE, str(extract), "--phase", phase],
        cwd=str(PROJECT_ROOT), capture_output=True, encoding="utf-8",
    )
    print(result.stdout[-500:] if result.stdout else "(无输出)")


def main() -> None:
    parser = argparse.ArgumentParser(description="exp12 前馈补偿综合评估批量运行")
    parser.add_argument("--phase", choices=["A", "B", "C", "all"], default="all",
                        help="运行哪个 phase")
    parser.add_argument("--workers", type=int, default=4,
                        help="并行进程数（默认 4）")
    args = parser.parse_args()

    phases = ["A", "B", "C"] if args.phase == "all" else [args.phase]
    for phase in phases:
        run_phase(phase, args.workers)

    # 写完成标记
    if args.phase == "all":
        (DATA_DIR / "_.COMPLETE").write_text(
            f"DONE {time.strftime('%Y-%m-%dT%H:%M:%S')}\n", encoding="utf-8")
        print(f"\n全部完成！结果在 {DATA_DIR}/")


if __name__ == "__main__":
    main()

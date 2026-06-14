"""从 exp12 实验日志提取结果到 CSV（支持 3 个 phase）。

用法:
    python scripts/extract/extract_exp12_results.py --phase A
    python scripts/extract/extract_exp12_results.py --phase B
    python scripts/extract/extract_exp12_results.py --phase C
    python scripts/extract/extract_exp12_results.py --phase all
"""
import argparse
import csv
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "experiment_data" / "exp12_feedforward"
HIT_THRESHOLD = 0.153  # 命中判定：球半径 0.033 + 球拍半径 0.12

# __RESULT__ 行的正则模式
RESULT_PATTERN = re.compile(
    r"__RESULT__:\s*"
    r"pos_error=([\d.]+)\s+"
    r"vel_error=([\d.]+)\s+"
    r"min_dist=([\d.]+)\s+"
    r"ball_near_ms=([\d.]+)\s+"
    r"tube_ready_ms=([\d.]+)\s+"
    r"max_tcp=([\d.]+)\s+"
    r"max_qdot=([\d.]+)\s+"
    r"max_face=([\d.]+)\s+"
    r"hit_type=(\w+)\s+"
    r"hit_time_error_ms=([\d.-]+)\s+"
    r"hit_pos_error=([\d.]+)\s+"
    r"v_racket_at_hit=([\d.]+)"
)

# CSV 列名（通用）
FIELDNAMES = [
    "phase", "mode", "ball_speed", "seed",
    "kp_base", "kd_ratio", "ff_mode", "noise_level",
    "hit", "pos_error", "vel_error", "min_dist",
    "ball_near_ms", "tube_ready_ms",
    "max_tcp", "max_qdot", "max_face",
    "hit_type", "hit_time_error_ms", "hit_pos_error",
    "v_racket_at_hit",
]


def parse_result_line(text: str) -> dict | None:
    """从日志文本中提取 __RESULT__ 行的字段。

    Args:
        text: 日志文件全文。

    Returns:
        包含各指标的字典，未找到则 None。
    """
    m = RESULT_PATTERN.search(text)
    if not m:
        return None
    return {
        "pos_error": float(m.group(1)),
        "vel_error": float(m.group(2)),
        "min_dist": float(m.group(3)),
        "ball_near_ms": float(m.group(4)),
        "tube_ready_ms": float(m.group(5)),
        "max_tcp": float(m.group(6)),
        "max_qdot": float(m.group(7)),
        "max_face": float(m.group(8)),
        "hit_type": m.group(9),
        "hit_time_error_ms": float(m.group(10)),
        "hit_pos_error": float(m.group(11)),
        "v_racket_at_hit": float(m.group(12)),
    }


def parse_filename_phase_a(filename: str) -> dict:
    """从 Phase A 文件名解析元数据。

    格式: mode{mode}_speed{speed}_seed{seed}.log
    """
    m = re.match(r"mode(\w+)_speed(\d+)_seed(\d+)\.log", filename)
    if not m:
        return {}
    return {
        "mode": m.group(1),
        "ball_speed": int(m.group(2)),
        "seed": int(m.group(3)),
    }


def parse_filename_phase_b(filename: str) -> dict:
    """从 Phase B 文件名解析元数据。

    格式: kp{kp}_kdr{kdr}_seed{seed}.log
    """
    m = re.match(r"kp(\d+)_kdr([\d.]+)_seed(\d+)\.log", filename)
    if not m:
        return {}
    return {
        "kp_base": int(m.group(1)),
        "kd_ratio": float(m.group(2)),
        "seed": int(m.group(3)),
        "ball_speed": 7,
    }


def parse_filename_phase_c(filename: str) -> dict:
    """从 Phase C 文件名解析元数据。

    格式: ff{ff}_noise{noise}_seed{seed}.log
    """
    m = re.match(r"ff(\w+)_noise(\w+)_seed(\d+)\.log", filename)
    if not m:
        return {}
    return {
        "ff_mode": m.group(1),
        "noise_level": m.group(2),
        "seed": int(m.group(3)),
        "ball_speed": 7,
    }


def extract_phase(phase: str) -> int:
    """提取指定 phase 的全部日志到 CSV。

    Args:
        phase: "A", "B", 或 "C"。

    Returns:
        成功提取的记录数。
    """
    raw_dir = DATA_DIR / f"phase{phase}" / "raw"
    csv_path = DATA_DIR / f"phase{phase}" / "results.csv"

    if not raw_dir.exists():
        print(f"Phase {phase}: raw 目录不存在 {raw_dir}")
        return 0

    log_files = sorted(raw_dir.glob("*.log"))
    if not log_files:
        print(f"Phase {phase}: 无日志文件")
        return 0

    parse_fn = {"A": parse_filename_phase_a,
                "B": parse_filename_phase_b,
                "C": parse_filename_phase_c}[phase]

    rows = []
    failed = 0
    for lf in log_files:
        text = lf.read_text(encoding="utf-8", errors="ignore")
        result = parse_result_line(text)
        if result is None:
            failed += 1
            continue

        meta = parse_fn(lf.name)
        row: dict[str, Any] = {"phase": phase}
        row.update(meta)
        row.update(result)
        row["hit"] = float(row["pos_error"]) < HIT_THRESHOLD
        rows.append(row)

    # 写 CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # 汇总统计
    total = len(rows)
    hits = sum(1 for r in rows if r.get("hit"))
    hit_rate = hits / total * 100 if total > 0 else 0
    avg_pos_err = sum(r["pos_error"] for r in rows) / total if total > 0 else 0

    print(f"\nPhase {phase}: {total} 条结果（{failed} 条解析失败）")
    print(f"  命中率: {hits}/{total} = {hit_rate:.1f}%")
    print(f"  平均位置误差: {avg_pos_err * 1000:.1f} mm")
    print(f"  CSV: {csv_path}")

    # Phase A 额外按模式分组统计
    if phase == "A":
        _print_phase_a_summary(rows)
    elif phase == "B":
        _print_phase_b_summary(rows)
    elif phase == "C":
        _print_phase_c_summary(rows)

    return total


def _print_phase_a_summary(rows: list[dict]) -> None:
    """Phase A 按模式和球速分组统计。"""
    from collections import defaultdict
    by_mode = defaultdict(list)
    for r in rows:
        by_mode[r.get("mode", "?")].append(r)

    print(f"\n  {'模式':<12} {'命中率':>8} {'均值误差(mm)':>14} {'TCP(m/s)':>10}")
    print(f"  {'-'*48}")
    for mode in ["torque", "pos_ff", "pos_noff"]:
        rs = by_mode.get(mode, [])
        if not rs:
            continue
        hits = sum(1 for r in rs if r.get("hit"))
        hr = hits / len(rs) * 100
        avg_err = sum(r["pos_error"] for r in rs) / len(rs) * 1000
        avg_tcp = sum(r["max_tcp"] for r in rs) / len(rs)
        print(f"  {mode:<12} {hr:>7.1f}% {avg_err:>13.1f} {avg_tcp:>9.2f}")


def _print_phase_b_summary(rows: list[dict]) -> None:
    """Phase B 按 Kp/Kd_ratio 分组统计。"""
    from collections import defaultdict
    by_pd = defaultdict(list)
    for r in rows:
        key = (r.get("kp_base", "?"), r.get("kd_ratio", "?"))
        by_pd[key].append(r)

    print(f"\n  {'Kp':>4} {'Kd_r':>6} {'命中率':>8} {'均值误差(mm)':>14}")
    print(f"  {'-'*38}")
    for (kp, kdr), rs in sorted(by_pd.items()):
        hits = sum(1 for r in rs if r.get("hit"))
        hr = hits / len(rs) * 100
        avg_err = sum(r["pos_error"] for r in rs) / len(rs) * 1000
        print(f"  {kp:>4} {kdr:>6.2f} {hr:>7.1f}% {avg_err:>13.1f}")


def _print_phase_c_summary(rows: list[dict]) -> None:
    """Phase C 按 FF 和噪声分组统计。"""
    from collections import defaultdict
    by_cond = defaultdict(list)
    for r in rows:
        key = (r.get("ff_mode", "?"), r.get("noise_level", "?"))
        by_cond[key].append(r)

    print(f"\n  {'FF':>5} {'噪声':>7} {'命中率':>8} {'均值误差(mm)':>14}")
    print(f"  {'-'*40}")
    for (ff, noise), rs in sorted(by_cond.items()):
        hits = sum(1 for r in rs if r.get("hit"))
        hr = hits / len(rs) * 100
        avg_err = sum(r["pos_error"] for r in rs) / len(rs) * 1000
        print(f"  {ff:>5} {noise:>7} {hr:>7.1f}% {avg_err:>13.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="exp12 结果提取")
    parser.add_argument("--phase", choices=["A", "B", "C", "all"], default="all",
                        help="提取哪个 phase")
    args = parser.parse_args()

    phases = ["A", "B", "C"] if args.phase == "all" else [args.phase]
    total = 0
    for phase in phases:
        total += extract_phase(phase)

    print(f"\n总计提取 {total} 条结果")


if __name__ == "__main__":
    main()

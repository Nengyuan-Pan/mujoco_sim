"""提取 exp11_regression 回归测试结果。

日志文件名格式: speed{S}_mode{torque|position}_noise{off|lo}_s{N}.log
"""
import re
import csv
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "experiment_data" / "exp11_regression" / "raw"
CSV_PATH = RAW_DIR.parent / "results.csv"

SPEEDS = [6, 7, 8, 9, 10]
MODES = ["torque", "position"]
NOISES = ["off", "lo"]


def parse_log(log_path: Path) -> dict:
    """解析单次实验日志。

    Args:
        log_path: 日志文件路径

    Returns:
        包含所有提取指标的字典
    """
    raw = log_path.read_bytes()
    text = raw.decode("utf-8", errors="ignore")
    if not text.strip() or text.startswith("ERROR:"):
        return _parse_name(log_path, status="error")

    result = _parse_name(log_path, status="ok")
    if result["status"] != "ok":
        return result

    m_result = re.search(
        r"__RESULT__: pos_error=(\S+) vel_error=(\S+) min_dist=(\S+) "
        r"ball_near_ms=(\S+) tube_ready_ms=(\S+) max_tcp=(\S+) "
        r"max_qdot=(\S+) max_face=(\S+) hit_type=(\S+) "
        r"hit_time_error_ms=(\S+) hit_pos_error=(\S+) v_racket_at_hit=(\S+)",
        text,
    )
    if m_result:
        result["pos_error"] = float(m_result.group(1))
        result["vel_error"] = float(m_result.group(2))
        result["min_dist"] = float(m_result.group(3))
        result["ball_near_ms"] = float(m_result.group(4))
        result["tube_ready_ms"] = float(m_result.group(5))
        result["max_tcp"] = float(m_result.group(6))
        result["max_qdot"] = float(m_result.group(7))
        result["max_face"] = float(m_result.group(8))
        result["hit_type"] = m_result.group(9)
        result["hit_pos_error"] = float(m_result.group(11))
        result["v_racket_at_hit"] = float(m_result.group(12))
        result["hit"] = result["pos_error"] < 0.153
    else:
        result.update({
            "pos_error": 0, "vel_error": 0, "min_dist": 0,
            "ball_near_ms": 0, "tube_ready_ms": 0,
            "max_tcp": 0, "max_qdot": 0, "max_face": 0,
            "hit_type": "error", "hit_pos_error": 0,
            "v_racket_at_hit": 0, "hit": False,
        })

    m_wall = re.search(r"总墙钟时间: ([\d.]+)s", text)
    result["wall_time"] = float(m_wall.group(1)) if m_wall else 0

    m_emerg = re.search(r"emerg_stop=(\d+)", text)
    result["emerg_stop"] = int(m_emerg.group(1)) if m_emerg else 0

    result["has_nan"] = "NaN" in text

    return result


def _parse_name(log_path: Path, status: str = "error") -> dict:
    """从文件名解析参数（格式: speed{S}_mode{torque|position}_noise{off|lo}_s{N}）。"""
    name = log_path.stem
    m = re.match(r"speed(\d+)_mode(\w+)_noise(\w+)_s(\d+)", name)
    if m:
        speed = int(m.group(1))
        mode = m.group(2)
        noise = m.group(3)
        seed = int(m.group(4))
    else:
        speed, mode, noise, seed = 0, "unknown", "unknown", 0
    return {
        "speed": speed, "mode": mode, "noise": noise, "seed": seed,
        "status": status, "hit": False,
        "pos_error": 0, "vel_error": 0, "min_dist": 0,
        "ball_near_ms": 0, "tube_ready_ms": 0,
        "max_tcp": 0, "max_qdot": 0, "max_face": 0,
        "hit_type": "error", "hit_pos_error": 0,
        "v_racket_at_hit": 0, "wall_time": 0,
        "emerg_stop": 0, "has_nan": False,
    }


def _group_summary(results: list[dict], speed: int, mode: str, noise: str) -> dict:
    """计算一组实验的汇总统计。"""
    group = [r for r in results
             if r["speed"] == speed and r["mode"] == mode
             and r["noise"] == noise and r["status"] == "ok"]
    n = len(group)
    n_hit = sum(1 for r in group if r["hit"])
    hit_errs = [r["pos_error"] for r in group if r["hit"]]
    return {
        "n": n, "n_hit": n_hit,
        "hit_rate": n_hit / max(n, 1) * 100,
        "avg_err_mm": sum(hit_errs) / max(n_hit, 1) * 1000 if hit_errs else 0,
        "avg_tcp": sum(r["max_tcp"] for r in group) / max(n, 1) if group else 0,
        "avg_v_racket": (
            sum(r["v_racket_at_hit"] for r in group if r["hit"]) / max(n_hit, 1)
            if n_hit > 0 else 0
        ),
        "emerg": sum(r["emerg_stop"] for r in group),
        "nan": sum(1 for r in group if r["has_nan"]),
    }


def main() -> None:
    results = [parse_log(lf) for lf in sorted(RAW_DIR.glob("*.log"))]
    fieldnames = [
        "speed", "mode", "noise", "seed",
        "status", "hit",
        "pos_error", "vel_error", "min_dist",
        "ball_near_ms", "tube_ready_ms",
        "max_tcp", "max_qdot", "max_face",
        "hit_type", "hit_pos_error", "v_racket_at_hit",
        "wall_time", "emerg_stop", "has_nan",
    ]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    total = len(results)
    ok = sum(1 for r in results if r["status"] == "ok")
    hits = sum(1 for r in results if r["hit"])
    print(f"提取 {total} 行 -> {CSV_PATH}")
    print(f"  ok={ok}, error={total - ok}, hits={hits}/{ok}\n")

    # === A. 速度退化曲线（off 噪声）===
    print("=" * 80)
    print("A. 速度退化曲线（无噪声 off）")
    print("=" * 80)
    print(f"{'球速':>4} {'torque命中':>10} {'position命中':>12} {'比率':>6} "
          f"{'torque误差':>10} {'position误差':>12} {'误差比':>6}")
    print("-" * 80)
    for speed in SPEEDS:
        t = _group_summary(results, speed, "torque", "off")
        p = _group_summary(results, speed, "position", "off")
        ratio = p["hit_rate"] / max(t["hit_rate"], 0.01) * 100
        err_ratio = p["avg_err_mm"] / max(t["avg_err_mm"], 0.01)
        print(f"{speed:>4}m/s {t['n_hit']:>2}/{t['n']:<2} ({t['hit_rate']:>4.0f}%) "
              f"{p['n_hit']:>2}/{p['n']:<2} ({p['hit_rate']:>5.0f}%) "
              f"{ratio:>5.0f}% "
              f"{t['avg_err_mm']:>8.1f}mm {p['avg_err_mm']:>10.1f}mm "
              f"{err_ratio:>5.2f}x")

    # === B. 噪声鲁棒性（lo 噪声）===
    print(f"\n{'=' * 80}")
    print("B. 噪声鲁棒性（lo 噪声 σ_p=0.02, σ_v=0.2）")
    print("=" * 80)
    print(f"{'球速':>4} {'torque命中':>10} {'position命中':>12} {'比率':>6} "
          f"{'torque退化':>10} {'position退化':>12}")
    print("-" * 80)
    for speed in SPEEDS:
        t_off = _group_summary(results, speed, "torque", "off")
        t_lo = _group_summary(results, speed, "torque", "lo")
        p_off = _group_summary(results, speed, "position", "off")
        p_lo = _group_summary(results, speed, "position", "lo")
        ratio = p_lo["hit_rate"] / max(t_lo["hit_rate"], 0.01) * 100
        t_deg = t_lo["hit_rate"] - t_off["hit_rate"]
        p_deg = p_lo["hit_rate"] - p_off["hit_rate"]
        print(f"{speed:>4}m/s {t_lo['n_hit']:>2}/{t_lo['n']:<2} ({t_lo['hit_rate']:>4.0f}%) "
              f"{p_lo['n_hit']:>2}/{p_lo['n']:<2} ({p_lo['hit_rate']:>5.0f}%) "
              f"{ratio:>5.0f}% "
              f"{t_deg:>+8.0f}pp {p_deg:>+10.0f}pp")

    # === C. TCP 速度 + v_racket 对比 ===
    print(f"\n{'=' * 80}")
    print("C. TCP 峰值速度 + 球拍击球速度对比（off 噪声）")
    print("=" * 80)
    print(f"{'球速':>4} {'torque TCP':>10} {'position TCP':>12} "
          f"{'torque v_racket':>15} {'position v_racket':>17}")
    print("-" * 80)
    for speed in SPEEDS:
        t = _group_summary(results, speed, "torque", "off")
        p = _group_summary(results, speed, "position", "off")
        print(f"{speed:>4}m/s {t['avg_tcp']:>8.2f} {p['avg_tcp']:>10.2f} "
              f"{t['avg_v_racket']:>13.3f} {p['avg_v_racket']:>15.3f}")

    # === D. 验收判定 ===
    print(f"\n{'=' * 80}")
    print("D. 验收判定（@ 7m/s, off 噪声）")
    print("=" * 80)
    t7 = _group_summary(results, 7, "torque", "off")
    p7 = _group_summary(results, 7, "position", "off")
    hit_ratio = p7["hit_rate"] / max(t7["hit_rate"], 0.01) * 100
    err_ratio = p7["avg_err_mm"] / max(t7["avg_err_mm"], 0.01)
    emerg_total = sum(r["emerg_stop"] for r in results if r["status"] == "ok")
    nan_total = sum(1 for r in results if r["has_nan"])

    print(f"  命中率: torque={t7['hit_rate']:.1f}%, position={p7['hit_rate']:.1f}%, "
          f"比率={hit_ratio:.1f}% (>= 90%? {'PASS' if hit_ratio >= 90 else 'FAIL'})")
    print(f"  误差比: {err_ratio:.2f}x (<= 1.5? {'PASS' if err_ratio <= 1.5 else 'FAIL'})")
    print(f"  安全: emerg_stop={emerg_total}, NaN={nan_total} "
          f"({'PASS' if emerg_total == 0 and nan_total == 0 else 'FAIL'})")

    # === E. 安全全局检查 ===
    print(f"\n{'=' * 80}")
    print("E. 全局安全检查（1000 runs）")
    print("=" * 80)
    print(f"  emerg_stop 总数: {emerg_total}")
    print(f"  NaN 总数: {nan_total}")
    print(f"  error 总数: {total - ok}")


if __name__ == "__main__":
    main()

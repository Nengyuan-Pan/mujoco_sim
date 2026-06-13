"""提取 exp10_pd_scan 精调结果。

日志文件名格式: kp{K}_kdr{R}_dq{D}_r{M}_s{S}.log
"""
import re
import csv
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "experiment_data" / "exp10_pd_scan" / "raw"
CSV_PATH = RAW_DIR.parent / "results.csv"

KP_VALUES = [10, 15, 20, 25, 30, 35, 40, 50]
KD_RATIOS = [0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2]
DQ_FRACTIONS = [0.5]
RATIO_MODES = [0, 1, 2]

_RATIO_NAMES = {0: "current", 1: "uniform", 2: "torque-prop"}


def parse_log(log_path: Path) -> dict:
    """解析单次实验日志。"""
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
    """从文件名解析参数（格式: kp{K}_kdr{R}_dq{D}_r{M}_s{S}）。"""
    name = log_path.stem
    m = re.match(r"kp(\d+)_kdr([\d.]+)_dq([\d.]+)_r(\d+)_s(\d+)", name)
    if m:
        kp = int(m.group(1))
        kdr = float(m.group(2))
        dq = float(m.group(3))
        rmode = int(m.group(4))
        seed = int(m.group(5))
    else:
        kp, kdr, dq, rmode, seed = 0, 0.0, 0.0, 0, 0
    return {
        "kp": kp, "kd_ratio": kdr, "dq_max_fraction": dq,
        "ratio_mode": rmode, "seed": seed,
        "status": status, "hit": False,
        "pos_error": 0, "vel_error": 0, "min_dist": 0,
        "ball_near_ms": 0, "tube_ready_ms": 0,
        "max_tcp": 0, "max_qdot": 0, "max_face": 0,
        "hit_type": "error", "hit_pos_error": 0,
        "v_racket_at_hit": 0, "wall_time": 0,
        "emerg_stop": 0, "has_nan": False,
    }


def main() -> None:
    results = [parse_log(lf) for lf in sorted(RAW_DIR.glob("*.log"))]
    fieldnames = [
        "kp", "kd_ratio", "dq_max_fraction", "ratio_mode", "seed",
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

    # 按 ratio_mode 分组汇总
    for rmode in RATIO_MODES:
        rname = _RATIO_NAMES.get(rmode, f"r{rmode}")
        r_hits = sum(1 for r in results if r["hit"] and r["ratio_mode"] == rmode)
        r_ok = sum(1 for r in results if r["status"] == "ok" and r["ratio_mode"] == rmode)
        print(f"  ratio_mode={rmode} ({rname}): {r_hits}/{r_ok} hits ({r_hits/max(r_ok,1)*100:.1f}%)")

    print()

    # 详细的 Kp×Kd×ratio 表
    for rmode in RATIO_MODES:
        rname = _RATIO_NAMES.get(rmode, f"r{rmode}")
        print(f"\n=== ratio_mode={rmode} ({rname}) ===")
        print(f"{'Kp':>6} {'KdR':>5} {'命中':>6} {'命中率':>6} "
              f"{'pos_err':>8} {'TCP':>6} {'v_racket':>8}")
        print("-" * 60)
        for kp in KP_VALUES:
            for kdr in KD_RATIOS:
                group = [r for r in results
                         if r["kp"] == kp and abs(r["kd_ratio"] - kdr) < 1e-6
                         and r["ratio_mode"] == rmode
                         and r["status"] == "ok"]
                n = len(group)
                if n == 0:
                    continue
                n_hit = sum(1 for r in group if r["hit"])
                hit_rate = n_hit / n * 100
                avg_err = sum(r["pos_error"] for r in group if r["hit"]) / max(n_hit, 1) * 1000
                avg_tcp = sum(r["max_tcp"] for r in group) / n
                avg_vr = sum(r["v_racket_at_hit"] for r in group if r["hit"]) / max(n_hit, 1)
                marker = " ***" if hit_rate >= 90 else (" **" if hit_rate >= 70 else "")
                print(f"{kp:>6} {kdr:>5.3f} {n_hit:>2}/{n:<3} {hit_rate:>5.0f}% "
                      f"{avg_err:>7.1f}mm {avg_tcp:>5.2f} {avg_vr:>7.3f}m/s{marker}")


if __name__ == "__main__":
    main()

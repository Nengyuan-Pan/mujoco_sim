"""提取 exp1_algorithm_capability 结果到 CSV。"""
import re, csv
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "experiment_data" / "exp1_algorithm_capability" / "raw"
CSV_PATH = RAW_DIR.parent / "results.csv"


def parse_log(log_path: Path) -> dict:
    raw = log_path.read_bytes()
    text = raw.decode("utf-8", errors="ignore")
    if not text.strip() or ("runtimeerror" in text.lower() and "球拍击球" not in text):
        text = raw.decode("utf-16-le", errors="ignore")

    name = log_path.stem  # speed8_seed0_tube_true
    parts = name.split("_")
    speed = int(parts[0].replace("speed", ""))
    seed = int(parts[1].replace("seed", ""))
    tube_on = parts[-1] == "true"

    result = {
        "ball_speed": speed, "seed": seed,
        "use_tube": str(tube_on).lower(), "status": "ok",
    }

    if "RuntimeError" in text and "球拍击球" not in text:
        result.update({"hit": "False", "status": "generation_failed",
                       "pos_error": 0, "vel_error": 0, "min_distance": 0,
                       "max_qdot_ratio": 0, "max_tcp_speed": 0,
                       "hit_type": "n/a", "ball_near_ms": 0, "tube_ready_ms": 0,
                       "mpc_steps": 0, "wall_time": 0})
        return result

    hit_match = re.search(r"步 \d+: 球拍击球!", text)
    if hit_match:
        result["hit"] = "True"
        m_type = re.search(r"\[(.+?)\]", hit_match.group(0))
        result["hit_type"] = m_type.group(1) if m_type else "passive"

        idx = hit_match.start()
        m_step = re.search(
            r"步 \d+: 剩余=(\d+),\s*误差=([\d.]+)m,\s*距离=([\d.]+)m,\s*迭代=(\d+),\s*步耗时=([\d.]+)ms,.*?max_qdot=([\d.]+)x,\s*TCP=([\d.]+)m/s\s*Face=([\d.]+)m/s",
            text[idx:],
        )
        if m_step:
            result["pos_error"] = round(float(m_step.group(2)), 6)
            result["min_distance"] = round(float(m_step.group(3)), 6)
            result["max_qdot_ratio"] = round(float(m_step.group(6)), 3)
            result["max_tcp_speed"] = round(float(m_step.group(7)), 2)
        else:
            result["pos_error"] = 0; result["min_distance"] = 0
            result["max_qdot_ratio"] = 0; result["max_tcp_speed"] = 0
    else:
        result["hit"] = "False"; result["hit_type"] = "miss"
        result["pos_error"] = 0; result["min_distance"] = 0
        result["max_qdot_ratio"] = 0; result["max_tcp_speed"] = 0

    result["vel_error"] = 0
    m_bn = re.search(r"ball_near 步数:\s*\d+\s*=\s*([\d.]+)\s*ms", text)
    result["ball_near_ms"] = float(m_bn.group(1)) if m_bn else 0
    m_wall = re.search(r"MPC 完成: MPC=([\d.]+)s/(\d+)步", text)
    if m_wall:
        result["wall_time"] = round(float(m_wall.group(1)), 2)
        result["mpc_steps"] = int(m_wall.group(2))
    else:
        result["wall_time"] = 0; result["mpc_steps"] = 0

    return result


def main() -> None:
    results = [parse_log(lf) for lf in sorted(RAW_DIR.glob("*.log"))]

    fieldnames = [
        "ball_speed", "seed", "use_tube", "hit", "status",
        "pos_error", "vel_error", "min_distance",
        "max_qdot_ratio", "max_tcp_speed",
        "hit_type", "ball_near_ms", "tube_ready_ms",
        "mpc_steps", "wall_time",
    ]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    ok = [r for r in results if r.get("status") == "ok"]
    failed = [r for r in results if r.get("status") != "ok"]
    print(f"提取 {len(results)} 行 -> {CSV_PATH}")
    print(f"  ok={len(ok)} failed={len(failed)}")
    print(f"\n{'球速':>6} {'Tube ON':>12} {'Tube OFF':>12} {'总体':>10}")
    for speed in sorted(set(r["ball_speed"] for r in ok)):
        on = [r for r in ok if r["ball_speed"] == speed and r["use_tube"] == "true"]
        off = [r for r in ok if r["ball_speed"] == speed and r["use_tube"] == "false"]
        hit_on = sum(1 for r in on if r["hit"] == "True")
        hit_off = sum(1 for r in off if r["hit"] == "True")
        n_on, n_off = len(on), len(off)
        err_on = sum(r["pos_error"] for r in on if r["hit"] == "True") / max(hit_on, 1) * 1000
        err_off = sum(r["pos_error"] for r in off if r["hit"] == "True") / max(hit_off, 1) * 1000
        active_on = sum(1 for r in on if "主动" in str(r.get("hit_type", "")))
        total = hit_on + hit_off
        print(f"{speed:>4}m/s {hit_on:>2}/{n_on:<7} {hit_off:>2}/{n_off:<7} {total:>3}/{n_on+n_off:<4}  "
              f"err={err_on:.0f}/{err_off:.0f}mm  active={active_on}+{sum(1 for r in off if '主动' in str(r.get('hit_type','')))}")


if __name__ == "__main__":
    main()

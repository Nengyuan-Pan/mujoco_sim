"""提取 exp1_v3_algorithm_capability 结果。"""
import re, csv
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent.parent / "experiment_data" / "exp1_v3_algorithm_capability" / "raw"
CSV_PATH = RAW_DIR.parent / "results.csv"
SPEEDS = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 25, 26, 28, 30]


def parse_log(log_path: Path) -> dict:
    raw = log_path.read_bytes()
    text = raw.decode("utf-8", errors="ignore")
    if not text.strip() or ("runtimeerror" in text.lower() and "球拍击球" not in text):
        text = raw.decode("utf-16-le", errors="ignore")

    name = log_path.stem
    parts = name.split("_")
    speed = int(parts[0].replace("speed", ""))
    seed = int(parts[1].replace("seed", ""))
    tube_on = parts[-1] == "true"

    result = {"ball_speed": speed, "seed": seed, "use_tube": str(tube_on).lower(), "status": "ok"}

    if "RuntimeError" in text and "球拍击球" not in text:
        result.update({"hit": "False", "status": "generation_failed",
                       "pos_error": 0, "min_distance": 0,
                       "max_qdot_ratio": 0, "max_tcp_speed": 0,
                       "hit_type": "n/a", "mpc_steps": 0, "wall_time": 0})
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
        # Extract max qdot/tcp from all step logs for misses
        qdots = [float(x) for x in re.findall(r"max_qdot=([\d.]+)x", text)]
        tcps = [float(x) for x in re.findall(r"TCP=([\d.]+)m/s", text)]
        result["max_qdot_ratio"] = round(max(qdots), 3) if qdots else 0
        result["max_tcp_speed"] = round(max(tcps), 2) if tcps else 0

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
        "pos_error", "min_distance", "max_qdot_ratio", "max_tcp_speed",
        "hit_type", "mpc_steps", "wall_time",
    ]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"提取 {len(results)} 行 -> {CSV_PATH}\n")
    print(f"{'球速':>5} {'Tube ON':>14} {'Tube OFF':>14} {'总体':>10}  {'误差ON/OFF':>14}  {'主动ON/OFF':>12}  qdot")
    for speed in SPEEDS:
        on = [r for r in results if r["ball_speed"] == speed and r["use_tube"] == "true" and r["status"] == "ok"]
        off = [r for r in results if r["ball_speed"] == speed and r["use_tube"] == "false" and r["status"] == "ok"]
        n_on, n_off = len(on), len(off)
        hit_on = sum(1 for r in on if r["hit"] == "True")
        hit_off = sum(1 for r in off if r["hit"] == "True")
        err_on = sum(r["pos_error"] for r in on if r["hit"] == "True") / max(hit_on, 1) * 1000
        err_off = sum(r["pos_error"] for r in off if r["hit"] == "True") / max(hit_off, 1) * 1000
        active_on = sum(1 for r in on if "主动" in str(r.get("hit_type", "")))
        active_off = sum(1 for r in off if "主动" in str(r.get("hit_type", "")))
        qdot_on = max((r["max_qdot_ratio"] for r in on), default=0)
        total = hit_on + hit_off
        print(f"{speed:>4}m/s {hit_on:>2}/{n_on:<6} {hit_off:>2}/{n_off:<6} {total:>3}/{n_on+n_off:<3}  "
              f"{err_on:>5.0f}/{err_off:<5.0f}mm  act={active_on}+{active_off}    qdot={qdot_on:.1f}x")


if __name__ == "__main__":
    main()

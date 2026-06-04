"""提取 exp7_noise_tube_ablation 结果。

用法:
    python scripts/extract/extract_exp7_results.py
"""

import re
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "experiment_data" / "exp7_noise_tube_ablation" / "raw"
CSV_PATH = RAW_DIR.parent / "results.csv"
SPEEDS = [8, 10, 12, 14, 16, 18, 20]
NOISE_MODES = ["off", "lo", "mid", "hi", "anis"]
TUBE_MODES = ["true", "false"]


def parse_log(log_path: Path) -> dict:
    raw = log_path.read_bytes()
    text = raw.decode("utf-8", errors="ignore")
    if not text.strip() or ("runtimeerror" in text.lower() and "球拍击球" not in text):
        text = raw.decode("utf-16-le", errors="ignore")

    name = log_path.stem
    m_name = re.match(r"speed(\d+)_seed(\d+)_tube_(true|false)_noise_(.+)", name)
    if not m_name:
        return {"ball_speed": 0, "seed": 0, "use_tube": "false", "noise_mode": "unknown",
                "hit": "False", "status": "parse_error"}
    speed = int(m_name.group(1))
    seed = int(m_name.group(2))
    tube_on = m_name.group(3) == "true"
    noise_mode = m_name.group(4)

    result = {
        "ball_speed": speed,
        "seed": seed,
        "use_tube": str(tube_on).lower(),
        "noise_mode": noise_mode,
        "status": "ok",
    }

    if "RuntimeError" in text and "球拍击球" not in text:
        result.update({
            "hit": "False", "status": "generation_failed",
            "pos_error": 0, "min_distance": 0,
            "max_qdot_ratio": 0, "max_tcp_speed": 0,
            "hit_type": "n/a", "mpc_steps": 0, "wall_time": 0,
        })
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
            result["pos_error"] = 0
            result["min_distance"] = 0
            result["max_qdot_ratio"] = 0
            result["max_tcp_speed"] = 0
    else:
        result["hit"] = "False"
        result["hit_type"] = "miss"
        result["pos_error"] = 0
        result["min_distance"] = 0
        qdots = [float(x) for x in re.findall(r"max_qdot=([\d.]+)x", text)]
        tcps = [float(x) for x in re.findall(r"TCP=([\d.]+)m/s", text)]
        result["max_qdot_ratio"] = round(max(qdots), 3) if qdots else 0
        result["max_tcp_speed"] = round(max(tcps), 2) if tcps else 0

    m_wall = re.search(r"MPC 完成: MPC=([\d.]+)s/(\d+)步", text)
    if m_wall:
        result["wall_time"] = round(float(m_wall.group(1)), 2)
        result["mpc_steps"] = int(m_wall.group(2))
    else:
        result["wall_time"] = 0
        result["mpc_steps"] = 0

    return result


def main() -> None:
    results = [parse_log(lf) for lf in sorted(RAW_DIR.glob("*.log"))]
    fieldnames = [
        "ball_speed", "seed", "use_tube", "noise_mode", "hit", "status",
        "pos_error", "min_distance", "max_qdot_ratio", "max_tcp_speed",
        "hit_type", "mpc_steps", "wall_time",
    ]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"提取 {len(results)} 行 -> {CSV_PATH}\n")

    # 汇总表：噪声×Tube×球速
    print(f"{'噪声':>6} {'Tube':>6} {'球速':>5} {'命中':>6}  {'位置误差':>10}  {'关速':>6}  {'TCP':>7}  {'主动':>6}")
    print("-" * 80)

    for noise in NOISE_MODES:
        for tube_val in TUBE_MODES:
            tube_label = "ON" if tube_val == "true" else "OFF"
            for speed in SPEEDS:
                subset = [r for r in results
                          if r["ball_speed"] == speed
                          and r["use_tube"] == tube_val
                          and r["noise_mode"] == noise
                          and r["status"] == "ok"]
                n = len(subset)
                if n == 0:
                    continue
                hits = [r for r in subset if r["hit"] == "True"]
                n_hit = len(hits)
                err = (sum(r["pos_error"] for r in hits) / max(n_hit, 1)) * 1000
                qdot = max((r["max_qdot_ratio"] for r in subset), default=0)
                tcp = max((r["max_tcp_speed"] for r in subset), default=0)
                active = sum(1 for r in hits if "主动" in str(r.get("hit_type", "")))
                print(f"{noise:>6} {tube_label:>6} {speed:>4}m/s {n_hit:>2}/{n:<2}  "
                      f"{err:>6.0f}mm  {qdot:>6.1f}x  {tcp:>6.1f}m/s  "
                      f"主动={active}")

    # 聚合汇总：噪声×Tube 全局命中率
    print(f"\n{'噪声':>6} {'Tube':>6} {'总命中率':>10}  {'总位置误差':>10}  {'总runs':>8}")
    print("-" * 60)
    for noise in NOISE_MODES:
        for tube_val in TUBE_MODES:
            tube_label = "ON" if tube_val == "true" else "OFF"
            subset = [r for r in results
                      if r["use_tube"] == tube_val
                      and r["noise_mode"] == noise
                      and r["status"] == "ok"]
            n_total = len(subset)
            if n_total == 0:
                continue
            hits = [r for r in subset if r["hit"] == "True"]
            n_hit = len(hits)
            rate = n_hit / n_total * 100
            err = (sum(r["pos_error"] for r in hits) / max(n_hit, 1)) * 1000
            print(f"{noise:>6} {tube_label:>6} {n_hit:>2}/{n_total:<2} = {rate:>5.1f}%  "
                  f"{err:>6.0f}mm  {n_total}")


if __name__ == "__main__":
    main()

"""提取 exp8_estimator_recovery 结果。

用法:
    python scripts/extract/extract_exp8_results.py
"""

import re
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "experiment_data" / "exp8_estimator_recovery" / "raw"
CSV_PATH = RAW_DIR.parent / "results.csv"
SPEEDS = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
NOISE_MODES = ["off", "lo", "mid", "hi", "anis"]
TUBE_MODES = ["true", "false"]
ESTIMATOR_MODES = ["kf", "nokf"]


def parse_log(log_path: Path) -> dict:
    raw = log_path.read_bytes()
    text = raw.decode("utf-8", errors="ignore")
    if not text.strip() or ("runtimeerror" in text.lower() and "球拍击球" not in text):
        text = raw.decode("utf-16-le", errors="ignore")

    name = log_path.stem
    m_name = re.match(
        r"speed(\d+)_seed(\d+)_tube_(true|false)_noise_(.+?)_(kf|nokf)", name
    )
    if not m_name:
        return {
            "ball_speed": 0, "seed": 0, "use_tube": "false",
            "noise_mode": "unknown", "estimator": "unknown",
            "hit": "False", "status": "parse_error",
        }
    speed = int(m_name.group(1))
    seed = int(m_name.group(2))
    tube_on = m_name.group(3) == "true"
    noise_mode = m_name.group(4)
    estimator = m_name.group(5)

    result = {
        "ball_speed": speed,
        "seed": seed,
        "use_tube": str(tube_on).lower(),
        "noise_mode": noise_mode,
        "estimator": estimator,
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
        "ball_speed", "seed", "use_tube", "noise_mode", "estimator",
        "hit", "status",
        "pos_error", "min_distance", "max_qdot_ratio", "max_tcp_speed",
        "hit_type", "mpc_steps", "wall_time",
    ]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"提取 {len(results)} 行 -> {CSV_PATH}\n")

    # 汇总表：噪声 × Estimator × Tube × 球速
    print(f"{'噪声':>6} {'Est':>4} {'Tube':>5} {'球速':>5} {'命中':>8}  "
          f"{'位置误差':>8}  {'关速':>6}  {'TCP':>7}")
    print("-" * 80)

    for noise in NOISE_MODES:
        for est in ESTIMATOR_MODES:
            for tube_val in TUBE_MODES:
                tube_label = "ON" if tube_val == "true" else "OFF"
                for speed in SPEEDS:
                    subset = [r for r in results
                              if r["ball_speed"] == speed
                              and r["use_tube"] == tube_val
                              and r["noise_mode"] == noise
                              and r["estimator"] == est
                              and r["status"] == "ok"]
                    n = len(subset)
                    if n == 0:
                        continue
                    hits = [r for r in subset if r["hit"] == "True"]
                    n_hit = len(hits)
                    err = (sum(r["pos_error"] for r in hits) / max(n_hit, 1)) * 1000
                    qdot = max((r["max_qdot_ratio"] for r in subset), default=0)
                    tcp = max((r["max_tcp_speed"] for r in subset), default=0)
                    print(f"{noise:>6} {est:>4} {tube_label:>5} {speed:>4}m/s "
                          f"{n_hit:>2}/{n:<2}  {err:>6.0f}mm  "
                          f"{qdot:>6.1f}x  {tcp:>6.1f}m/s")

    # 聚合汇总：噪声 × Estimator × Tube
    print(f"\n{'噪声':>6} {'Est':>4} {'Tube':>5} {'总命中率':>12}  "
          f"{'位置误差':>8}  {'总runs':>8}")
    print("-" * 70)
    for noise in NOISE_MODES:
        for est in ESTIMATOR_MODES:
            for tube_val in TUBE_MODES:
                tube_label = "ON" if tube_val == "true" else "OFF"
                subset = [r for r in results
                          if r["use_tube"] == tube_val
                          and r["noise_mode"] == noise
                          and r["estimator"] == est
                          and r["status"] == "ok"]
                n_total = len(subset)
                if n_total == 0:
                    continue
                hits = [r for r in subset if r["hit"] == "True"]
                n_hit = len(hits)
                rate = n_hit / n_total * 100
                err = (sum(r["pos_error"] for r in hits) / max(n_hit, 1)) * 1000
                print(f"{noise:>6} {est:>4} {tube_label:>5} "
                      f"{n_hit:>3}/{n_total:<3} = {rate:>5.1f}%  "
                      f"{err:>6.0f}mm  {n_total}")

    # KF 恢复率分析
    print(f"\n{'='*70}")
    print("KF 恢复率分析")
    print(f"{'='*70}")
    print(f"{'噪声':>6} {'Tube':>5} "
          f"{'nokf':>8} {'kf':>8} {'绝对恢复':>10} {'相对恢复':>10}")
    print("-" * 60)

    for noise in NOISE_MODES:
        for tube_val in TUBE_MODES:
            tube_label = "ON" if tube_val == "true" else "OFF"

            def _rate(est_mode: str) -> float:
                subset = [r for r in results
                          if r["use_tube"] == tube_val
                          and r["noise_mode"] == noise
                          and r["estimator"] == est_mode
                          and r["status"] == "ok"]
                if not subset:
                    return -1.0
                return sum(1 for r in subset if r["hit"] == "True") / len(subset) * 100

            off_rate = _rate("nokf") if noise == "off" else None
            if noise != "off":
                off_subset = [r for r in results
                              if r["use_tube"] == tube_val
                              and r["noise_mode"] == "off"
                              and r["estimator"] == "nokf"
                              and r["status"] == "ok"]
                off_rate = sum(1 for r in off_subset if r["hit"] == "True") / max(len(off_subset), 1) * 100

            nokf_rate = _rate("nokf")
            kf_rate = _rate("kf")

            if nokf_rate < 0 or kf_rate < 0:
                continue

            abs_recovery = kf_rate - nokf_rate
            if off_rate is not None and (off_rate - nokf_rate) > 0.01:
                rel_recovery = abs_recovery / (off_rate - nokf_rate) * 100
            else:
                rel_recovery = 0.0

            print(f"{noise:>6} {tube_label:>5} "
                  f"{nokf_rate:>7.1f}% {kf_rate:>7.1f}% "
                  f"{abs_recovery:>+8.1f}pp {rel_recovery:>8.1f}%")


if __name__ == "__main__":
    main()

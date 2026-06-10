"""提取 exp9_obs_freq_robustness 结果。

用法:
    python scripts/extract/extract_exp9_results.py
"""

import re
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "experiment_data" / "exp9_obs_freq_robustness" / "raw"
CSV_PATH = RAW_DIR.parent / "results.csv"

SPEEDS = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
FREQS = [200, 60, 30, 15, 10]
NOISES = ["off", "lo", "mid"]
KF_MODES = ["kf", "nokf"]
TUBES = ["full", "none"]


def parse_log(log_path: Path) -> dict:
    raw = log_path.read_bytes()
    text = raw.decode("utf-8", errors="ignore")
    if not text.strip() or ("runtimeerror" in text.lower() and "球拍击球" not in text):
        text = raw.decode("utf-16-le", errors="ignore")

    name = log_path.stem
    m_name = re.match(
        r"ball(\d+)_seed(\d+)_f(\d+)_(off|lo|mid)_(kf|nokf)_tube(on|off)", name
    )
    if not m_name:
        return {
            "ball_speed": 0, "seed": 0, "obs_freq": 0,
            "noise_mode": "unknown", "use_kf": "unknown", "use_tube": "unknown",
            "hit": "False", "status": "parse_error",
        }

    speed = int(m_name.group(1))
    seed = int(m_name.group(2))
    freq = int(m_name.group(3))
    noise = m_name.group(4)
    kf = m_name.group(5)
    tube = m_name.group(6) == "on"

    result = {
        "ball_speed": speed,
        "seed": seed,
        "obs_freq": freq,
        "noise_mode": noise,
        "use_kf": kf,
        "use_tube": str(tube).lower(),
        "status": "ok",
    }

    if "RuntimeError" in text and "球拍击球" not in text:
        result.update({
            "hit": "False", "status": "generation_failed",
            "pos_error": 0, "min_dist": 0,
            "max_qdot_ratio": 0, "max_tcp_speed": 0,
            "hit_type": "n/a", "wall_time": 0,
        })
        return result

    hit_match = re.search(r"步 \d+: 球拍击球!", text)
    if hit_match:
        result["hit"] = "True"
        m_type = re.search(r"\[(.+?)\]", hit_match.group(0))
        result["hit_type"] = m_type.group(1) if m_type else "passive"
        m_step = re.search(
            r"步 \d+: 剩余=(\d+),\s*误差=([\d.]+)m,\s*距离=([\d.]+)m,"
            r"\s*迭代=(\d+),\s*步耗时=([\d.]+)ms,.*?max_qdot=([\d.]+)x,"
            r"\s*TCP=([\d.]+)m/s",
            text[hit_match.start():],
        )
        if m_step:
            result["pos_error"] = round(float(m_step.group(2)), 6)
            result["min_dist"] = round(float(m_step.group(3)), 6)
            result["max_qdot_ratio"] = round(float(m_step.group(6)), 3)
            result["max_tcp_speed"] = round(float(m_step.group(7)), 2)
        else:
            result["pos_error"] = 0
            result["min_dist"] = 0
            result["max_qdot_ratio"] = 0
            result["max_tcp_speed"] = 0
    else:
        result["hit"] = "False"
        result["hit_type"] = "miss"
        result["pos_error"] = 0
        result["min_dist"] = 0
        qdots = [float(x) for x in re.findall(r"max_qdot=([\d.]+)x", text)]
        tcps = [float(x) for x in re.findall(r"TCP=([\d.]+)m/s", text)]
        result["max_qdot_ratio"] = round(max(qdots), 3) if qdots else 0
        result["max_tcp_speed"] = round(max(tcps), 2) if tcps else 0

    m_result = re.search(r"__RESULT__:.*min_dist=([\d.]+).*max_tcp=([\d.]+).*max_qdot=([\d.]+).*hit_type=(\w+)", text)
    if m_result:
        result["min_dist"] = round(float(m_result.group(1)), 6)
        result["max_tcp_speed"] = round(float(m_result.group(2)), 2)
        result["max_qdot_ratio"] = round(float(m_result.group(3)), 3)
        result["hit_type"] = m_result.group(4)

    m_wall = re.search(r"MPC 完成: MPC=([\d.]+)s/(\d+)步", text)
    if m_wall:
        result["wall_time"] = round(float(m_wall.group(1)), 2)
    else:
        result["wall_time"] = 0

    return result


def main() -> None:
    results = [parse_log(lf) for lf in sorted(RAW_DIR.glob("*.log"))]
    fieldnames = [
        "ball_speed", "seed", "obs_freq", "noise_mode", "use_kf", "use_tube",
        "hit", "status",
        "pos_error", "min_dist", "max_qdot_ratio", "max_tcp_speed",
        "hit_type", "wall_time",
    ]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"提取 {len(results)} 行 -> {CSV_PATH}\n")

    # 汇总：频率 × 噪声 × KF × Tube → 命中率
    print(f"{'频率':>5} {'噪声':>5} {'KF':>4} {'Tube':>5} {'命中':>12}  {'误差':>8}")
    print("-" * 60)

    for freq in FREQS:
        for noise in NOISES:
            for kf in KF_MODES:
                for tube_val in TUBES:
                    tube_label = "ON" if tube_val == "full" else "OFF"
                    subset = [r for r in results
                              if r["obs_freq"] == freq
                              and r["noise_mode"] == noise
                              and r["use_kf"] == kf
                              and r["use_tube"] == str(tube_val == "full").lower()
                              and r["status"] == "ok"]
                    n = len(subset)
                    if n == 0:
                        continue
                    hits = [r for r in subset if r["hit"] == "True"]
                    n_hit = len(hits)
                    rate = n_hit / n * 100
                    err = (sum(r["pos_error"] for r in hits) / max(n_hit, 1)) * 1000
                    print(f"{freq:>4}Hz {noise:>5} {kf:>4} {tube_label:>5} "
                          f"{n_hit:>3}/{n:<3}={rate:>5.1f}%  {err:>6.0f}mm")
        print()

    # 频率退化分析
    print(f"\n{'='*70}")
    print("频率退化分析（200Hz 基线 vs 低频）")
    print(f"{'='*70}")
    print(f"{'噪声':>5} {'KF':>4} {'Tube':>5} "
          f"{'200Hz':>8} {'60Hz':>8} {'30Hz':>8} {'15Hz':>8} {'10Hz':>8}")
    print("-" * 65)

    for noise in NOISES:
        for kf in KF_MODES:
            for tube_val in TUBES:
                tube_label = "ON" if tube_val == "full" else "OFF"
                rates = {}
                for freq in FREQS:
                    subset = [r for r in results
                              if r["obs_freq"] == freq
                              and r["noise_mode"] == noise
                              and r["use_kf"] == kf
                              and r["use_tube"] == str(tube_val == "full").lower()
                              and r["status"] == "ok"]
                    if subset:
                        rates[freq] = sum(1 for r in subset if r["hit"] == "True") / len(subset) * 100
                    else:
                        rates[freq] = -1
                print(f"{noise:>5} {kf:>4} {tube_label:>5} " +
                      " ".join(f"{rates.get(f, -1):>7.1f}%" if rates.get(f, -1) >= 0 else f"{'n/a':>8}" for f in FREQS))


if __name__ == "__main__":
    main()

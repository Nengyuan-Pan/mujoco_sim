"""从 exp2 日志提取结果到 CSV（处理 UTF-16LE 编码）。"""
import re, csv
from pathlib import Path

raw_dir = Path(r"F:\刀客塔\研究\网球机器人\工程\Code\mujoco_sim\experiment_data\exp2_strict_joint\raw")
results = []

for log_path in sorted(raw_dir.glob("*.log")):
    # PowerShell Tee-Object 写 UTF-16LE，读取时需处理
    raw_bytes = log_path.read_bytes()
    # 检测 BOM
    if raw_bytes[:2] == b"\xff\xfe":
        text = raw_bytes.decode("utf-16-le", errors="ignore")
    else:
        text = raw_bytes.decode("utf-8", errors="ignore")

    name = log_path.stem  # seed0_tube_true
    parts = name.replace("seed", "").split("_")
    seed = int(parts[0])
    tube_on = name.endswith("_true")

    m = re.search(r"__RESULT__:\s*(.+)$", text, re.MULTILINE)
    if not m:
        print(f"WARN: no __RESULT__ in {name}")
        continue
    r = {}
    for kv in m.group(1).split():
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        r[k] = v

    pos_err = float(r.get("pos_error", 0))
    min_dist = float(r.get("min_dist", 0))
    is_hit = "True" if (pos_err < 0.05 and min_dist < 0.153) else "False"

    m_replan = re.search(r"重规划=(\d+)次.*首次=(\d+)ms", text)
    n_replans = int(m_replan.group(1)) if m_replan else 0
    first_ms = int(m_replan.group(2)) if m_replan else 0
    m_emerg = re.search(r"emerg_stop=(\d+)", text)
    emerg = int(m_emerg.group(1)) if m_emerg else 0

    results.append({
        "seed": seed,
        "ball_speed": 9.0,
        "use_tube": str(tube_on).lower(),
        "hit": is_hit,
        "pos_error": round(pos_err, 6),
        "vel_error": round(float(r.get("vel_error", 0)), 6),
        "min_distance": round(min_dist, 6),
        "max_qdot_ratio": round(float(r.get("max_qdot", 0)), 3),
        "max_tcp_speed": round(float(r.get("max_tcp", 0)), 2),
        "hit_type": r.get("hit_type", "miss"),
        "hit_time_error_ms": float(r.get("hit_time_error_ms", 0)),
        "hit_pos_error": round(float(r.get("hit_pos_error", 0)), 6),
        "tube_ready_ms": float(r.get("tube_ready_ms", 0)),
        "ball_near_ms": float(r.get("ball_near_ms", 0)),
        "n_replans": n_replans,
        "first_replan_ms": first_ms,
        "emerg_stop": emerg,
    })

csv_path = raw_dir.parent / "results.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)

for row in results:
    mark = "[HIT]" if row["hit"] == "True" else "[MISS]"
    print(
        f"{mark} seed={row['seed']} tube={row['use_tube']} "
        f"pos_err={row['pos_error']}m min_dist={row['min_distance']}m "
        f"type={row['hit_type']} tube_ready={row['tube_ready_ms']}ms"
    )

print(f"\nDone: {len(results)} rows -> {csv_path}")

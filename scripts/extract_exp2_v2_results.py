"""从 exp2_strict_joint_v2 日志提取结果（解析离线脚本格式）。"""
import re, csv
from pathlib import Path

RAW_DIR = Path(r"F:\刀客塔\研究\网球机器人\工程\Code\mujoco_sim\experiment_data\exp2_strict_joint_v2\raw")
CSV_PATH = RAW_DIR.parent / "results.csv"


def parse_log(log_path: Path) -> dict:
    raw = log_path.read_bytes()
    text = (raw.decode("utf-16-le") if raw[:2] == b"\xff\xfe" else raw.decode("utf-8"))

    name = log_path.stem
    parts = name.split("_")
    speed = int(parts[0].replace("speed", ""))
    seed = int(parts[1].replace("seed", ""))
    tube_on = parts[-1] == "true"

    result = {
        "ball_speed": speed,
        "seed": seed,
        "use_tube": str(tube_on).lower(),
    }

    # Check if simulation actually ran
    if "RuntimeError" in text:
        result.update({"hit": "False", "status": "generation_failed",
                       "pos_error": 0, "vel_error": 0, "min_distance": 0,
                       "max_qdot_ratio": 0, "max_tcp_speed": 0, "max_face_speed": 0,
                       "hit_type": "n/a", "ball_near_ms": 0, "tube_ready_ms": 0,
                       "mpc_steps": 0, "wall_time": 0})
        return result

    # Find the hit step: "步 X: 球拍击球!"
    hit_match = re.search(r"步 (\d+): 球拍击球!", text)
    if hit_match:
        result["hit"] = "True"
        hit_step = int(hit_match.group(1))
        # Extract hit type from the hit line itself: [被动接触] or [主动击球]
        m_type = re.search(r"\[(.+?)\]", hit_match.group(0))
        result["hit_type"] = m_type.group(1) if m_type else "passive"

        # Search for the NEXT step's detail log line (hit details appear
        # on the step after the collision, sometimes far away)
        # Note: valid_hit= is sometimes absent (Tube OFF)
        idx = hit_match.start()
        step_detail = re.search(
            r"步 \d+: 剩余=(\d+),\s*误差=([\d.]+)m,\s*距离=([\d.]+)m,\s*迭代=(\d+),\s*步耗时=([\d.]+)ms,.*?max_qdot=([\d.]+)x,\s*TCP=([\d.]+)m/s\s*Face=([\d.]+)m/s",
            text[idx:],
        )
        if step_detail:
            result["pos_error"] = round(float(step_detail.group(2)), 6)
            result["min_distance"] = round(float(step_detail.group(3)), 6)
            result["max_qdot_ratio"] = round(float(step_detail.group(6)), 3)
            result["max_tcp_speed"] = round(float(step_detail.group(7)), 2)
            result["max_face_speed"] = round(float(step_detail.group(8)), 2)
        else:
            # If no step log, extract from max values across all steps
            all_qdot = [float(x) for x in re.findall(r"max_qdot=([\d.]+)x", text)]
            all_tcp = [float(x) for x in re.findall(r"TCP=([\d.]+)m/s", text)]
            all_face = [float(x) for x in re.findall(r"Face=([\d.]+)m/s", text)]
            result["pos_error"] = 0
            result["min_distance"] = 0
            result["max_qdot_ratio"] = round(max(all_qdot), 3) if all_qdot else 0
            result["max_tcp_speed"] = round(max(all_tcp), 2) if all_tcp else 0
            result["max_face_speed"] = round(max(all_face), 2) if all_face else 0
    else:
        result["hit"] = "False"
        result["pos_error"] = 0
        result["min_distance"] = 0
        result["max_qdot_ratio"] = 0
        result["max_tcp_speed"] = 0
        result["max_face_speed"] = 0
        result["hit_type"] = "miss"

    # Extract from summary or step lines
    result["vel_error"] = 0  # offline script doesn't always report this

    # ball_near/tube_ready from step logs
    m_bn = re.search(r"ball_near 步数:\s*\d+\s*=\s*([\d.]+)\s*ms", text)
    result["ball_near_ms"] = float(m_bn.group(1)) if m_bn else 0
    m_tr = re.search(r"tube_ready 步数:\s*\d+\s*=\s*([\d.]+)\s*ms", text)
    result["tube_ready_ms"] = float(m_tr.group(1)) if m_tr else 0

    # wall_time and mpc_steps from MPC completion line
    m_time = re.search(r"MPC 完成: MPC=([\d.]+)s/(\d+)步", text)
    if m_time:
        result["wall_time"] = round(float(m_time.group(1)), 2)
        result["mpc_steps"] = int(m_time.group(2))
    else:
        result["wall_time"] = 0
        result["mpc_steps"] = 0

    result["status"] = "ok"
    return result


def main() -> None:
    log_files = sorted(RAW_DIR.glob("*.log"))
    results = [parse_log(lf) for lf in log_files]

    fieldnames = [
        "ball_speed", "seed", "use_tube", "hit", "status",
        "pos_error", "vel_error", "min_distance",
        "max_qdot_ratio", "max_tcp_speed", "max_face_speed",
        "hit_type", "ball_near_ms", "tube_ready_ms",
        "mpc_steps", "wall_time",
    ]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    # Stats
    ok = [r for r in results if r.get("status") == "ok"]
    failed = [r for r in results if r.get("status") != "ok"]
    print(f"提取 {len(results)} 行 (ok={len(ok)} failed={len(failed)}) -> {CSV_PATH}")

    speeds_ok = sorted(set(r["ball_speed"] for r in ok))
    if not speeds_ok:
        print("无有效运行数据")
        return

    for speed in speeds_ok:
        on = [r for r in ok if r["ball_speed"] == speed and r["use_tube"] == "true"]
        off = [r for r in ok if r["ball_speed"] == speed and r["use_tube"] == "false"]
        hit_on = sum(1 for r in on if r["hit"] == "True")
        hit_off = sum(1 for r in off if r["hit"] == "True")
        err_on = round(sum(r["pos_error"] for r in on) / len(on) * 1000, 1) if on else 0
        err_off = round(sum(r["pos_error"] for r in off) / len(off) * 1000, 1) if off else 0
        print(f"  {speed}m/s Tube=ON: {hit_on}/{len(on)} hits ({err_on}mm avg)  "
              f"Tube=OFF: {hit_off}/{len(off)} hits ({err_off}mm avg)")


if __name__ == "__main__":
    main()

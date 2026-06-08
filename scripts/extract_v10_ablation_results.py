"""V10 消融实验结果提取脚本。

从 raw/ 日志中提取 __RESULT__ 行，输出 results.csv 和汇总统计。

用法:
    python scripts/extract_v10_ablation_results.py
"""
import csv
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "experiment_data" / "v10_ablation" / "raw"
OUT_CSV = PROJECT_ROOT / "experiment_data" / "v10_ablation" / "results.csv"
OUT_RESULTS = PROJECT_ROOT / "results" / "v10_ablation"

RESULT_RE = re.compile(
    r"__RESULT__: pos_error=([\d.]+) vel_error=([\d.]+) "
    r"min_dist=([\d.]+) ball_near_ms=([\d.]+) "
    r"tube_ready_ms=([\d.]+) max_tcp=([\d.]+) "
    r"max_qdot=([\d.]+) max_face=([\d.]+) "
    r"hit_type=(\w+) "
    r"hit_time_error_ms=([-\d.]+) hit_pos_error=([\d.]+) "
    r"v_racket_at_hit=([\d.]+)"
)

FIELDNAMES = [
    "ball_speed", "seed", "tube", "softmin", "perturb_pct",
    "hit", "hit_type", "pos_error", "vel_error", "min_dist",
    "v_racket_at_hit", "hit_time_error_ms", "hit_pos_error",
    "max_tcp", "max_qdot", "max_face",
    "ball_near_ms", "tube_ready_ms",
]


def parse_tag(tag: str) -> dict:
    parts = tag.split("_")
    info = {}
    for p in parts:
        if p.startswith("s") and p[1:].isdigit():
            info["ball_speed"] = int(p[1:])
        elif p.startswith("seed") and p[4:].isdigit():
            info["seed"] = int(p[4:])
        elif p == "tube":
            info["tube"] = "true"
        elif p == "notube":
            info["tube"] = "false"
        elif p == "softmin":
            info["softmin"] = "true"
        elif p == "nosoftmin":
            info["softmin"] = "false"
        elif p == "nominal":
            info["perturb_pct"] = 0
        elif p.startswith("p") and p[1:].isdigit():
            info["perturb_pct"] = int(p[1:])
    return info


def extract_one(log_path: Path) -> dict | None:
    try:
        text = log_path.read_text(encoding="utf-8")
    except Exception:
        return None
    if text.startswith("ERROR"):
        return None
    m = RESULT_RE.search(text)
    if m is None:
        return None
    g = m.groups()
    return {
        "pos_error": float(g[0]),
        "vel_error": float(g[1]),
        "min_dist": float(g[2]),
        "ball_near_ms": float(g[3]),
        "tube_ready_ms": float(g[4]),
        "max_tcp": float(g[5]),
        "max_qdot": float(g[6]),
        "max_face": float(g[7]),
        "hit_type": g[8],
        "hit_time_error_ms": float(g[9]),
        "hit_pos_error": float(g[10]),
        "v_racket_at_hit": float(g[11]),
    }


def main() -> None:
    if not RAW_DIR.exists():
        print(f"日志目录不存在: {RAW_DIR}")
        sys.exit(1)

    log_files = sorted(RAW_DIR.glob("*.log"))
    print(f"找到 {len(log_files)} 个日志文件")

    rows = []
    for lp in log_files:
        tag = lp.stem
        info = parse_tag(tag)
        result = extract_one(lp)
        row = {
            "ball_speed": info.get("ball_speed", ""),
            "seed": info.get("seed", ""),
            "tube": info.get("tube", ""),
            "softmin": info.get("softmin", ""),
            "perturb_pct": info.get("perturb_pct", ""),
        }
        if result:
            row.update(result)
            row["hit"] = "1" if result["hit_type"] in ("active", "passive") else "0"
        else:
            row.update({f: "" for f in FIELDNAMES if f not in row})
            row["hit"] = "0"
        rows.append(row)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"已写入: {OUT_CSV} ({len(rows)} rows)")

    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    dest = OUT_RESULTS / "results.csv"
    dest.write_text(OUT_CSV.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"已复制: {dest}")

    print("\n" + "=" * 90)
    print("汇总统计（按 ball_speed × tube × softmin × perturb_pct 分组）")
    print("=" * 90)

    from collections import defaultdict

    groups = defaultdict(list)
    for r in rows:
        key = (r["ball_speed"], r["tube"], r["softmin"], r["perturb_pct"])
        groups[key].append(r)

    header = f"{'speed':>5} {'tube':>5} {'soft':>5} {'pert%':>5} | {'n':>3} {'hit%':>5} {'act%':>5} {'pos_err':>8} {'v_racket':>8} {'min_d':>8}"
    print(header)
    print("-" * len(header))

    for key in sorted(groups.keys()):
        rs = groups[key]
        n = len(rs)
        hits = [r for r in rs if r.get("hit") == "1"]
        actives = [r for r in rs if r.get("hit_type") == "active"]
        hit_rate = len(hits) / n * 100 if n > 0 else 0
        active_rate = len(actives) / n * 100 if n > 0 else 0
        pos_errs = [float(r["pos_error"]) for r in hits if r.get("pos_error")]
        avg_pos = sum(pos_errs) / len(pos_errs) * 1000 if pos_errs else float("nan")
        v_racks = [float(r["v_racket_at_hit"]) for r in hits if r.get("v_racket_at_hit")]
        avg_v = sum(v_racks) / len(v_racks) if v_racks else float("nan")
        min_ds = [float(r["min_dist"]) for r in hits if r.get("min_dist")]
        avg_min = sum(min_ds) / len(min_ds) * 1000 if min_ds else float("nan")

        speed, tube, soft, pert = key
        print(f"{speed:>5} {tube:>5} {soft:>5} {pert:>5} | {n:>3} {hit_rate:>5.1f} {active_rate:>5.1f} {avg_pos:>7.1f}mm {avg_v:>7.2f}m/s {avg_min:>7.1f}mm")


if __name__ == "__main__":
    main()

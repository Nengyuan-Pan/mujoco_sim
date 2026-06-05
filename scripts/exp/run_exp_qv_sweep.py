"""V7 Q_v/Q_p 调参实验: 5 组条件 × 50 seed，统计命中率和击球速度。
终端偏移=0，TCP/关节硬约束。

条件:
  1. baseline:  Q_p×2, Q_v×2, Q_v_scale_near=120
  2. Q_v×4:    Q_p×2, Q_v×8, Q_v_scale_near=120
  3. Q_v_near×2: Q_p×2, Q_v×2, Q_v_scale_near=240
  4. Q_p降低:  Q_p×1, Q_v×2, Q_v_scale_near=120
  5. Q_v×4+Q_p降低: Q_p×1, Q_v×8, Q_v_scale_near=120
"""
import subprocess
import sys
import re
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = ROOT / "scripts" / "rm65_mpc_v7.py"

BALL_SPEED = 7.0
N_SEEDS = 50
SEED_START = 1
SEED_END = SEED_START + N_SEEDS - 1

CONDITIONS = [
    {
        "name": "baseline",
        "Q_p_mult": 2.0,
        "Q_v_mult": 2.0,
        "Q_v_scale_near": 120.0,
    },
    {
        "name": "Q_vx4",
        "Q_p_mult": 2.0,
        "Q_v_mult": 8.0,
        "Q_v_scale_near": 120.0,
    },
    {
        "name": "Q_v_nearx2",
        "Q_p_mult": 2.0,
        "Q_v_mult": 2.0,
        "Q_v_scale_near": 240.0,
    },
    {
        "name": "Q_p_lower",
        "Q_p_mult": 1.0,
        "Q_v_mult": 2.0,
        "Q_v_scale_near": 120.0,
    },
    {
        "name": "Q_vx4_Q_p_lower",
        "Q_p_mult": 1.0,
        "Q_v_mult": 8.0,
        "Q_v_scale_near": 120.0,
    },
]


def apply_condition(script_text: str, cond: dict) -> str:
    """修改脚本中的 Q_p/Q_v 倍率和 Q_v_scale_near。"""
    # Q_p 倍率
    t, n = re.subn(
        r'(Q_p = np\.array\(config_dict\["cost"\]\["Q_p"\], dtype=np\.float64\) \*)\s*[\d.]+',
        rf'\g<1> {cond["Q_p_mult"]}',
        script_text,
    )
    if n != 1:
        raise RuntimeError(f"Q_p 替换失败, 找到 {n} 处")

    # Q_v 倍率
    t, n = re.subn(
        r'(Q_v = np\.array\(config_dict\["cost"\]\["Q_v"\], dtype=np\.float64\) \*)\s*[\d.]+',
        rf'\g<1> {cond["Q_v_mult"]}',
        t,
    )
    if n != 1:
        raise RuntimeError(f"Q_v 替换失败, 找到 {n} 处")

    # Q_v_scale_near
    t, n = re.subn(
        r'Q_v_scale_near = [\d.]+',
        f'Q_v_scale_near = {cond["Q_v_scale_near"]}',
        t,
    )
    if n != 1:
        raise RuntimeError(f"Q_v_scale_near 替换失败, 找到 {n} 处")

    return t


def run_single(seed: int) -> dict | None:
    """运行单个 seed，解析 __RESULT__ 行。"""
    cmd = [
        sys.executable, str(SCRIPT),
        "--serve-box",
        "--ball-speed", str(BALL_SPEED),
        "--seed", str(seed),
        "--no-backswing",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=str(ROOT))
    output = result.stdout + result.stderr

    for line in output.splitlines():
        if "__RESULT__" in line:
            m = re.search(
                r"pos_error=([0-9.]+).*hit_type=(\w+).*v_racket_at_hit=([0-9.]+)",
                line,
            )
            if m:
                return {
                    "pos_error": float(m.group(1)),
                    "hit_type": m.group(2),
                    "v_racket_at_hit": float(m.group(3)),
                }
    return None


def main() -> None:
    print(f"Q_v/Q_p 调参实验: {len(CONDITIONS)} 组 × {N_SEEDS} seeds, ball_speed={BALL_SPEED}")
    print("=" * 90)

    original_text = SCRIPT.read_text(encoding="utf-8")

    summary = []

    for cond in CONDITIONS:
        # 应用条件
        modified = apply_condition(original_text, cond)
        SCRIPT.write_text(modified, encoding="utf-8")

        print(f"\n--- {cond['name']}: Q_p×{cond['Q_p_mult']}, Q_v×{cond['Q_v_mult']}, "
              f"Q_v_scale_near={cond['Q_v_scale_near']} ---")

        results = []
        for seed in range(SEED_START, SEED_END + 1):
            r = run_single(seed)
            if r is not None:
                results.append(r)
                tag = "HIT" if r["hit_type"] != "miss" else "MISS"
                print(f"  seed={seed:3d} {tag:4s} pos_err={r['pos_error']:.4f} "
                      f"v_racket={r['v_racket_at_hit']:.3f}m/s")
            else:
                print(f"  seed={seed:3d} CRASH")

        # 统计
        n = len(results)
        hits = [r for r in results if r["hit_type"] != "miss"]
        hit_rate = len(hits) / n * 100 if n > 0 else 0

        if hits:
            avg_pos = np.mean([r["pos_error"] for r in hits])
            avg_v = np.mean([r["v_racket_at_hit"] for r in hits])
            std_v = np.std([r["v_racket_at_hit"] for r in hits])
            max_v = max(r["v_racket_at_hit"] for r in hits)
            min_v = min(r["v_racket_at_hit"] for r in hits)
        else:
            avg_pos = avg_v = std_v = max_v = min_v = 0.0

        row = {
            "name": cond["name"],
            "hit_rate": f"{len(hits)}/{n} ({hit_rate:.0f}%)",
            "avg_pos": f"{avg_pos:.4f}",
            "avg_v": f"{avg_v:.3f}",
            "std_v": f"{std_v:.3f}",
            "min_v": f"{min_v:.3f}",
            "max_v": f"{max_v:.3f}",
        }
        summary.append(row)

        print(f"\n  汇总 {cond['name']}:")
        print(f"    命中率: {len(hits)}/{n} = {hit_rate:.0f}%")
        print(f"    pos_error avg: {avg_pos:.4f}m")
        print(f"    球拍速度 avg: {avg_v:.3f} ± {std_v:.3f} m/s, range [{min_v:.3f}, {max_v:.3f}]")

    # 恢复原始脚本
    SCRIPT.write_text(original_text, encoding="utf-8")

    # 打印汇总表
    print("\n" + "=" * 90)
    print("汇总对比表:")
    print(f"{'条件':<20s} {'命中率':<15s} {'pos_err':<10s} {'v_racket(avg)':<15s} {'±std':<10s} {'range':<20s}")
    print("-" * 90)
    for row in summary:
        print(f"{row['name']:<20s} {row['hit_rate']:<15s} {row['avg_pos']:<10s} "
              f"{row['avg_v']:<15s} {row['std_v']:<10s} [{row['min_v']}, {row['max_v']}]")

    print("\n已恢复原始脚本参数")


if __name__ == "__main__":
    main()

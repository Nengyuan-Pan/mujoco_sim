"""偏移量扫描实验: 测试 1cm/2cm/3cm/4cm/5cm 偏移 × 50 seed。
统计命中率和击球瞬间球拍速度。
"""
import subprocess
import sys
import re
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = ROOT / "scripts" / "rm65_mpc_v6.py"

BALL_SPEED = 7.0
N_SEEDS = 50
SEED_START = 1
SEED_END = SEED_START + N_SEEDS - 1

OFFSETS = [0.01, 0.02, 0.03, 0.04, 0.05]


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
    print(f"偏移量扫描实验: offsets={OFFSETS}, seeds={N_SEEDS}, ball_speed={BALL_SPEED}")
    print("=" * 80)

    for offset in OFFSETS:
        # 修改脚本中的偏移量
        script_text = SCRIPT.read_text(encoding="utf-8")
        pattern = r'follow_through_length = 0\.\d+  # 终端偏移'
        replacement = f"follow_through_length = {offset:.2f}  # 终端偏移"
        new_text, count = re.subn(pattern, replacement, script_text, count=1)
        if count != 1:
            print(f"[ERROR] 无法替换 follow_through_length，找到 {count} 处")
            sys.exit(1)
        SCRIPT.write_text(new_text, encoding="utf-8")

        print(f"\n--- offset = {offset:.2f}m ---")
        results = []
        for seed in range(SEED_START, SEED_END + 1):
            r = run_single(seed)
            if r is not None:
                results.append(r)
                tag = "HIT" if r["hit_type"] != "miss" else "MISS"
                print(f"  seed={seed:3d} {tag} pos_err={r['pos_error']:.4f} "
                      f"v_racket={r['v_racket_at_hit']:.3f}m/s")
            else:
                print(f"  seed={seed:3d} CRASH")

        # 统计
        n = len(results)
        hits = [r for r in results if r["hit_type"] != "miss"]
        misses = [r for r in results if r["hit_type"] == "miss"]
        hit_rate = len(hits) / n * 100 if n > 0 else 0

        if hits:
            avg_pos = np.mean([r["pos_error"] for r in hits])
            avg_v = np.mean([r["v_racket_at_hit"] for r in hits])
            max_v = max(r["v_racket_at_hit"] for r in hits)
            min_v = min(r["v_racket_at_hit"] for r in hits)
        else:
            avg_pos = avg_v = max_v = min_v = 0.0

        print(f"\n  汇总 offset={offset:.2f}m:")
        print(f"    命中率: {len(hits)}/{n} = {hit_rate:.0f}%")
        print(f"    命中 pos_error avg: {avg_pos:.4f}m")
        print(f"    球拍速度 avg: {avg_v:.3f} m/s, min: {min_v:.3f}, max: {max_v:.3f}")

    # 恢复原始偏移量
    script_text = SCRIPT.read_text(encoding="utf-8")
    pattern = r'follow_through_length = 0\.\d+  # 终端偏移'
    replacement = "follow_through_length = 0.05  # 终端偏移"
    new_text, count = re.subn(pattern, replacement, script_text, count=1)
    SCRIPT.write_text(new_text, encoding="utf-8")
    print("\n已恢复 follow_through_length = 0.05")


if __name__ == "__main__":
    main()

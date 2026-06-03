"""逐步放宽关节角限制，测试不同 q_margin 配置下的击球精度。
使用固定 20 个 seed，统计命中率和均值误差。

命中判定：位置误差 < 0.10m（球拍面半径约 0.08m + 球半径 0.033m）
"""

import subprocess
import sys
import re
import json
import time
from pathlib import Path

SEEDS = [1, 3, 5, 7, 9, 13, 17, 21, 33, 42,
         55, 66, 77, 88, 99, 100, 123, 150, 200, 255]

HIT_THRESHOLD = 0.10  # 位置误差 < 10cm 视为命中

MARGIN_CONFIGS = [
    {"label": "原始",      "margins": [2, 1, 5, 10, 10, 10]},
    {"label": "保守",      "margins": [2, 1, 3, 3, 3, 3]},
    {"label": "中度",      "margins": [1, 0, 1, 1, 1, 1]},
    {"label": "激进",      "margins": [0, 0, 0, 0, 0, 0]},
]

YAML_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml"


def set_margins(margins: list[int]) -> None:
    """修改 default.yaml 中的 q_margin_deg。"""
    text = YAML_PATH.read_text(encoding="utf-8")
    pattern = r"q_margin_deg:\s*\[.*?\]"
    replacement = f"q_margin_deg: {margins}"
    text = re.sub(pattern, replacement, text)
    YAML_PATH.write_text(text, encoding="utf-8")


def run_seed(seed: int) -> dict:
    """运行单个 seed，返回 {pos_err, hit, fb}。"""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "rm65_mpc_tube_constraint.py"),
        "--serve-box", "--ball-speed", "15",
        "--seed", str(seed),
        "--no-backswing", "--no-plot",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=str(Path(__file__).resolve().parent.parent.parent),
                            timeout=120)
    out = result.stdout + result.stderr
    pos_err = None
    fb = 0
    m = re.search(r"位置误差:\s*([\d.]+)\s*m", out)
    if m:
        pos_err = float(m.group(1))
    m = re.search(r"fallback=(\d+)", out)
    if m:
        fb = int(m.group(1))
    hit = pos_err is not None and pos_err < HIT_THRESHOLD
    return {"pos_err": pos_err, "hit": hit, "fb": fb}


def run_config(label: str, margins: list[int]) -> dict:
    """运行一种 margin 配置，返回汇总。"""
    set_margins(margins)
    print(f"\n{'='*60}")
    print(f"配置: {label}, margins={margins}")
    print(f"{'='*60}")
    results = []
    t0 = time.time()
    for seed in SEEDS:
        try:
            r = run_seed(seed)
            r["seed"] = seed
        except Exception as e:
            r = {"seed": seed, "pos_err": None, "hit": False, "fb": -1, "error": str(e)}
        tag = "HIT " if r["hit"] else "MISS"
        err_str = f"{r['pos_err']:.4f}" if r["pos_err"] is not None else "FAIL"
        print(f"  seed={seed:3d}  pos_err={err_str}m  {tag}")
        results.append(r)
    elapsed = time.time() - t0

    valid = [r for r in results if r["pos_err"] is not None]
    hits = sum(1 for r in valid if r["hit"])
    mean_err = sum(r["pos_err"] for r in valid) / len(valid) if valid else float("inf")
    max_err = max(r["pos_err"] for r in valid) if valid else float("inf")
    min_err = min(r["pos_err"] for r in valid) if valid else float("inf")
    hit_rate = hits / len(valid) * 100 if valid else 0

    summary = {
        "label": label,
        "margins": margins,
        "mean_err": round(mean_err, 4),
        "min_err": round(min_err, 4),
        "max_err": round(max_err, 4),
        "hit_rate": f"{hit_rate:.0f}%",
        "hits": hits,
        "total": len(valid),
        "elapsed": round(elapsed, 1),
        "results": results,
    }
    print(f"\n  汇总: 均值={mean_err:.4f}m, 命中率={hits}/{len(valid)}={hit_rate:.0f}%, "
          f"耗时={elapsed:.0f}s")
    return summary


if __name__ == "__main__":
    all_results = []
    for cfg in MARGIN_CONFIGS:
        s = run_config(cfg["label"], cfg["margins"])
        all_results.append(s)

    print(f"\n\n{'='*60}")
    print("综合对比")
    print(f"{'='*60}")
    print(f"{'配置':<8} {'margins':<28} {'均值(m)':<10} {'命中率':<10} {'min':<8} {'max':<8}")
    print("-" * 80)
    for s in all_results:
        print(f"{s['label']:<8} {str(s['margins']):<28} {s['mean_err']:<10.4f} "
              f"{s['hit_rate']:<10} {s['min_err']:<8.4f} {s['max_err']:<8.4f}")

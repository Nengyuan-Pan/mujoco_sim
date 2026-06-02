"""算法能力与真实机械臂可执行性实验脚本。

实验A：速度豁免模式（当前默认行为），测试 9-30 m/s 球速下算法极限。
实验B：严格约束模式（qdot/qddot/torque ≤ 1.0x），测试 RM-65B 可执行的最高球速。

用法:
    python scripts/run_experiments.py --experiment A
    python scripts/run_experiments.py --experiment B
    python scripts/run_experiments.py --experiment A --speeds 9 12 15 18 20 25 30
    python scripts/run_experiments.py --experiment B --speeds 3 4 5 6 7 8
"""

from __future__ import annotations

import sys
import subprocess
import json
import re
import time
import argparse
import os
from pathlib import Path
from datetime import datetime

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent.parent
RESULT_DIR = SCRIPT_DIR / "results"


def run_single(args_dict: dict) -> dict:
    """运行单次实验，返回结果字典。"""
    cmd = [sys.executable, "scripts/rm65_mpc_tube_constraint_realtime.py"]
    for k, v in args_dict.items():
        if v is True:
            cmd.append(f"--{k}")
        elif v is not False and v is not None:
            cmd.append(f"--{k}")
            cmd.append(str(v))
    
    result = {
        "seed": args_dict.get("seed", 0),
        "ball_speed": args_dict.get("ball_speed", 12),
        "hit": False,
        "hit_type": "miss",
        "ball_dist": None,
        "plan_err": None,
        "min_dist": None,
        "max_qdot": None,
        "max_tcp": None,
        "near_avg_ms": None,
        "near_max_ms": None,
        "far_avg_ms": None,
        "far_max_ms": None,
        "mpc_realtime_ratio": None,
        "total_sleep_ms": None,
    }
    
    try:
        out = subprocess.run(
            cmd, capture_output=True, timeout=300,
            cwd=str(SCRIPT_DIR),
        )
        txt = out.stdout.decode("gbk", errors="replace") + out.stderr.decode("gbk", errors="replace")
    except subprocess.TimeoutExpired:
        result["hit_type"] = "timeout"
        return result
    
    for line in txt.split("\n"):
        stripped = line.strip()
        
        if "RM-65" in stripped:
            if "5cm" in stripped:
                result["hit"] = True
                result["hit_type"] = "PRECISE"
            elif "0.153" in stripped:
                result["hit"] = True
                result["hit_type"] = "HIT"
            elif "10cm" in stripped:
                result["hit_type"] = "near"
            else:
                result["hit_type"] = "miss"
        
        if "ball_dist" not in stripped and "0.153" not in stripped:
            if stripped.endswith("m") and result["ball_dist"] is None:
                m = re.search(r"([\d.]+)\s*m$", stripped)
                if m:
                    val = float(m.group(1))
                    if 0.005 < val < 0.5:
                        result["ball_dist"] = val
        
        if stripped.endswith("m") and result["min_dist"] is None:
            for prefix in ["0.", "."]:
                if prefix in stripped and "ball" not in stripped.lower():
                    m = re.search(r"([\d.]+)\s*m$", stripped)
                    if m:
                        val = float(m.group(1))
                        if 0.001 < val < 0.3:
                            result["min_dist"] = val
                            break
        
        if "max_qdot=" in stripped:
            m = re.search(r"max_qdot=([\d.]+)x", stripped)
            if m:
                result["max_qdot"] = float(m.group(1))
            m2 = re.search(r"max_tcp=([\d.]+)m/s", stripped)
            if m2:
                result["max_tcp"] = float(m2.group(1))
        
        if "near" in stripped.lower() and "avg=" in stripped:
            m = re.search(r"avg=(\d+)ms.*max=(\d+)ms", stripped)
            if m:
                result["near_avg_ms"] = int(m.group(1))
                result["near_max_ms"] = int(m.group(2))
        
        if "far" in stripped.lower() and "avg=" in stripped:
            m = re.search(r"avg=(\d+)ms.*max=(\d+)ms", stripped)
            if m:
                result["far_avg_ms"] = int(m.group(1))
                result["far_max_ms"] = int(m.group(2))
        
        if "sleep=" in stripped:
            m = re.search(r"sleep=(\d+)ms", stripped)
            if m:
                result["total_sleep_ms"] = int(m.group(1))
        
        if stripped.startswith("MPC"):
            m = re.search(r"([\d.]+)x\(", stripped)
            if m:
                result["mpc_realtime_ratio"] = float(m.group(1))
    
    return result


def run_experiment_a(speeds: list[int], n_seeds: int = 20) -> dict:
    """实验A：速度豁免模式。"""
    print("=" * 60)
    print("实验A：算法能力（速度豁免模式）")
    print("=" * 60)
    
    all_results = {}
    
    for speed in speeds:
        print(f"\n--- 球速 {speed} m/s ({n_seeds} seeds) ---")
        results = []
        for seed in range(n_seeds):
            args = {
                "serve-box": True,
                "ball-speed": speed,
                "seed": seed,
                "no-backswing": True,
                "no-plot": True,
                "realtime": True,
            }
            r = run_single(args)
            results.append(r)
            hit_str = r["hit_type"]
            dist_str = f"{r['min_dist']:.3f}m" if r["min_dist"] else "?"
            qdot_str = f"{r['max_qdot']:.2f}x" if r["max_qdot"] else "?"
            print(f"  seed={seed:2d} {hit_str:7s} dist={dist_str} qdot={qdot_str}")
        
        n_hit = sum(1 for r in results if r["hit"])
        dists = [r["min_dist"] for r in results if r["min_dist"] is not None]
        qdots = [r["max_qdot"] for r in results if r["max_qdot"] is not None]
        tcps = [r["max_tcp"] for r in results if r["max_tcp"] is not None]
        
        summary = {
            "speed": speed,
            "n_seeds": n_seeds,
            "n_hit": n_hit,
            "hit_rate": n_hit / n_seeds,
            "avg_min_dist": float(np.mean(dists)) if dists else None,
            "max_qdot_avg": float(np.mean(qdots)) if qdots else None,
            "max_qdot_peak": float(np.max(qdots)) if qdots else None,
            "max_tcp_avg": float(np.mean(tcps)) if tcps else None,
            "max_tcp_peak": float(np.max(tcps)) if tcps else None,
            "results": results,
        }
        all_results[speed] = summary
        
        print(f"  >>> 命中率: {n_hit}/{n_seeds} ({n_hit/n_seeds*100:.0f}%)")
        if dists:
            print(f"  >>> 平均最近距离: {np.mean(dists):.3f}m")
        if qdots:
            print(f"  >>> 关节速度: avg={np.mean(qdots):.2f}x, peak={np.max(qdots):.2f}x")
        if tcps:
            print(f"  >>> TCP速度: avg={np.mean(tcps):.1f}m/s, peak={np.max(tcps):.1f}m/s")
    
    return all_results


def run_experiment_b(speeds: list[int], n_seeds: int = 20) -> dict:
    """实验B：严格约束模式。
    
    通过修改 robot_limits 配置实现严格约束：
    - qdot_scale=1.0（不缩放）
    - forward_pass_margin=1.0（前向传递严格）
    - 通过 Monkey-patch strict_braking_check 取消豁免窗口
    
    由于需要修改运行时行为，使用专门的脚本。
    """
    print("=" * 60)
    print("实验B：真实机械臂可执行性（严格约束 qdot≤1.0x）")
    print("=" * 60)
    
    strict_script = SCRIPT_DIR / "scripts" / "_run_strict_experiment.py"
    _generate_strict_script(strict_script)
    
    all_results = {}
    
    for speed in speeds:
        print(f"\n--- 球速 {speed} m/s ({n_seeds} seeds) ---")
        results = []
        for seed in range(n_seeds):
            cmd = [
                sys.executable, str(strict_script),
                "--ball-speed", str(speed),
                "--seed", str(seed),
            ]
            try:
                out = subprocess.run(
                    cmd, capture_output=True, timeout=300,
                    cwd=str(SCRIPT_DIR),
                )
                txt = out.stdout.decode("gbk", errors="replace") + out.stderr.decode("gbk", errors="replace")
            except subprocess.TimeoutExpired:
                results.append({"seed": seed, "hit": False, "hit_type": "timeout",
                                "ball_speed": speed, "min_dist": None, "max_qdot": None, "max_tcp": None})
                continue
            
            r = {"seed": seed, "hit": False, "hit_type": "miss", "ball_speed": speed,
                 "min_dist": None, "max_qdot": None, "max_tcp": None}
            
            for line in txt.split("\n"):
                stripped = line.strip()
                if "RM-65" in stripped:
                    if "5cm" in stripped:
                        r["hit"] = True; r["hit_type"] = "PRECISE"
                    elif "0.153" in stripped:
                        r["hit"] = True; r["hit_type"] = "HIT"
                    elif "10cm" in stripped:
                        r["hit_type"] = "near"
                    else:
                        r["hit_type"] = "miss"
                if "max_qdot=" in stripped:
                    m = re.search(r"max_qdot=([\d.]+)x", stripped)
                    if m: r["max_qdot"] = float(m.group(1))
                    m2 = re.search(r"max_tcp=([\d.]+)m/s", stripped)
                    if m2: r["max_tcp"] = float(m2.group(1))
                if stripped.endswith("m"):
                    if r["min_dist"] is None:
                        m = re.search(r"([\d.]+)\s*m$", stripped)
                        if m:
                            val = float(m.group(1))
                            if 0.001 < val < 0.3:
                                r["min_dist"] = val
            
            results.append(r)
            hit_str = r["hit_type"]
            dist_str = f"{r['min_dist']:.3f}m" if r["min_dist"] else "?"
            qdot_str = f"{r['max_qdot']:.2f}x" if r["max_qdot"] else "?"
            print(f"  seed={seed:2d} {hit_str:7s} dist={dist_str} qdot={qdot_str}")
        
        n_hit = sum(1 for r in results if r["hit"])
        dists = [r["min_dist"] for r in results if r["min_dist"] is not None]
        qdots = [r["max_qdot"] for r in results if r["max_qdot"] is not None]
        tcps = [r["max_tcp"] for r in results if r["max_tcp"] is not None]
        
        summary = {
            "speed": speed,
            "n_seeds": n_seeds,
            "n_hit": n_hit,
            "hit_rate": n_hit / n_seeds,
            "avg_min_dist": float(np.mean(dists)) if dists else None,
            "max_qdot_avg": float(np.mean(qdots)) if qdots else None,
            "max_qdot_peak": float(np.max(qdots)) if qdots else None,
            "max_tcp_avg": float(np.mean(tcps)) if tcps else None,
            "max_tcp_peak": float(np.max(tcps)) if tcps else None,
            "results": results,
        }
        all_results[speed] = summary
        
        print(f"  >>> 命中率: {n_hit}/{n_seeds} ({n_hit/n_seeds*100:.0f}%)")
        if dists:
            print(f"  >>> 平均最近距离: {np.mean(dists):.3f}m")
        if qdots:
            print(f"  >>> 关节速度: avg={np.mean(qdots):.2f}x, peak={np.max(qdots):.2f}x")
    
    return all_results


def _generate_strict_script(path: Path) -> None:
    """生成严格约束实验的临时脚本。
    
    核心修改：
    1. Monkey-patch strict_braking_check：取消 k_hit≤20 豁免，取消 k_hit 20-40 放宽
    2. forward_pass_margin=1.0（前向传递严格检查）
    3. qdot_scale=1.0（不缩放限速）
    """
    if path.exists():
        return
    
    script_content = '''"""严格约束实验脚本（自动生成）。

与 rm65_mpc_tube_constraint_realtime.py 相同，但：
- strict_braking_check 无豁免窗口（全程检查 qdot）
- 前向传递 margin=1.0
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ===== Monkey-patch: 取消安全滤波器的速度豁免 =====
from src.ilqt import robot_limits as _rl

_original_strict_braking_check = _rl.strict_braking_check

def _strict_braking_check_no_exemption(
    x_prev, x_next, u_try, limits, dt, k_hit_remaining=99,
):
    """全程严格 qdot 检查，无豁免窗口。"""
    import numpy as np
    nq = 6
    q_next = x_next[:nq]
    qdot_prev = x_prev[nq:]
    qdot_next = x_next[nq:]

    for j in range(nq):
        if q_next[j] < limits.q_lower[j]:
            return False, f"q lower bound violated, joint={j}"
        if q_next[j] > limits.q_upper[j]:
            return False, f"q upper bound violated, joint={j}"

    for j in range(nq):
        if u_try[j] < limits.u_min[j]:
            return False, f"u lower bound violated, joint={j}"
        if u_try[j] > limits.u_max[j]:
            return False, f"u upper bound violated, joint={j}"

    # 全程严格 1.0x，无豁免
    for j in range(nq):
        abs_prev = abs(qdot_prev[j])
        abs_next = abs(qdot_next[j])
        limit_j = limits.qdot_max[j]  # 1.0x，无放宽

        if abs_prev <= limit_j:
            if abs_next > limit_j:
                return False, (
                    f"qdot entering overspeed, joint={j}, "
                    f"|qdot|={abs_next:.2f} > {limit_j:.2f}"
                )
        else:
            if abs_next >= abs_prev:
                return False, (
                    f"qdot overspeeding+not-decelerating, joint={j}, "
                    f"|cur|={abs_prev:.2f} -> |next|={abs_next:.2f} > {limit_j:.2f}"
                )

    return True, ""

_rl.strict_braking_check = _strict_braking_check_no_exemption

# ===== Monkey-patch: 前向传递也严格检查 qdot =====
# forward_pass_margin 设为 1.0，不放宽

# ===== 导入并修改参数后运行主脚本 =====
# 修改 sys.argv 以传递参数给主脚本
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ball-speed", type=float, default=12)
parser.add_argument("--seed", type=int, default=0)
args, _ = parser.parse_known_args()

# 构造主脚本参数
sys.argv = [
    "rm65_mpc_tube_constraint_realtime.py",
    "--serve-box",
    "--ball-speed", str(args.ball_speed),
    "--seed", str(args.seed),
    "--no-backswing",
    "--no-plot",
    "--realtime",
]

# 导入主脚本并修改 forward_pass_margin
# 通过修改默认配置实现
import scripts.rm65_mpc_tube_constraint_realtime as main_mod

# 拦截 RobotLimits.from_config，强制 forward_pass_margin=1.0
_original_from_config = main_mod.RobotLimits.from_config

@classmethod
def _strict_from_config(cls, config, dt, ctrlrange):
    config = dict(config)
    config["forward_pass_margin"] = 1.0
    config["qdot_scale"] = 1.0
    config["forward_pass_q_tol_deg"] = 0.0
    return _original_from_config(config, dt, ctrlrange)

main_mod.RobotLimits.from_config = _strict_from_config

print(f"[STRICT MODE] ball_speed={args.ball_speed} seed={args.seed}")
print("[STRICT MODE] qdot <= 1.0x 全程，无豁免窗口")

main_mod.main()
'''
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(script_content)
    print(f"已生成严格约束脚本: {path}")


def generate_report(result_a: dict, result_b: dict, speeds_a: list, speeds_b: list) -> str:
    """生成实验报告 Markdown。"""
    lines = []
    lines.append("# 算法能力与真实机械臂可执行性实验报告\n")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # ===== 实验 A =====
    lines.append("## 实验A：算法能力（速度豁免模式）\n")
    lines.append("条件：k_hit ≤ 20 时豁免 qdot 检查，k_hit 20-40 放宽到 1.3×。")
    lines.append("测试算法在理想高速执行器条件下的极限击球能力。\n")
    
    lines.append("### 命中率\n")
    lines.append("| 球速 (m/s) | 命中数 | 命中率 | 平均最近距离 |")
    lines.append("|-----------|--------|--------|------------|")
    for speed in speeds_a:
        if speed in result_a:
            s = result_a[speed]
            avg_d = f"{s['avg_min_dist']*1000:.1f}mm" if s["avg_min_dist"] else "?"
            lines.append(f"| {speed} | {s['n_hit']}/{s['n_seeds']} | {s['hit_rate']*100:.0f}% | {avg_d} |")
    
    lines.append("\n### 关节速度与 TCP 速度\n")
    lines.append("| 球速 (m/s) | avg max_qdot | peak max_qdot | avg max_tcp | peak max_tcp |")
    lines.append("|-----------|-------------|--------------|------------|-------------|")
    for speed in speeds_a:
        if speed in result_a:
            s = result_a[speed]
            qd_a = f"{s['max_qdot_avg']:.2f}x" if s["max_qdot_avg"] else "?"
            qd_p = f"{s['max_qdot_peak']:.2f}x" if s["max_qdot_peak"] else "?"
            tcp_a = f"{s['max_tcp_avg']:.1f}m/s" if s["max_tcp_avg"] else "?"
            tcp_p = f"{s['max_tcp_peak']:.1f}m/s" if s["max_tcp_peak"] else "?"
            lines.append(f"| {speed} | {qd_a} | {qd_p} | {tcp_a} | {tcp_p} |")
    
    # ===== 实验 B =====
    lines.append("\n## 实验B：真实机械臂可执行性（严格约束）\n")
    lines.append("条件：qdot ≤ 1.0× 全程无豁免，forward_pass_margin=1.0。")
    lines.append("测试 RM-65B 在不超速条件下能击回的最高球速。\n")
    
    lines.append("### 命中率\n")
    lines.append("| 球速 (m/s) | 命中数 | 命中率 | 平均最近距离 |")
    lines.append("|-----------|--------|--------|------------|")
    for speed in speeds_b:
        if speed in result_b:
            s = result_b[speed]
            avg_d = f"{s['avg_min_dist']*1000:.1f}mm" if s["avg_min_dist"] else "?"
            lines.append(f"| {speed} | {s['n_hit']}/{s['n_seeds']} | {s['hit_rate']*100:.0f}% | {avg_d} |")
    
    lines.append("\n### 关节速度与 TCP 速度\n")
    lines.append("| 球速 (m/s) | avg max_qdot | peak max_qdot | avg max_tcp | peak max_tcp |")
    lines.append("|-----------|-------------|--------------|------------|-------------|")
    for speed in speeds_b:
        if speed in result_b:
            s = result_b[speed]
            qd_a = f"{s['max_qdot_avg']:.2f}x" if s["max_qdot_avg"] else "?"
            qd_p = f"{s['max_qdot_peak']:.2f}x" if s["max_qdot_peak"] else "?"
            tcp_a = f"{s['max_tcp_avg']:.1f}m/s" if s["max_tcp_avg"] else "?"
            tcp_p = f"{s['max_tcp_peak']:.1f}m/s" if s["max_tcp_peak"] else "?"
            lines.append(f"| {speed} | {qd_a} | {qd_p} | {tcp_a} | {tcp_p} |")
    
    # ===== 结论 =====
    lines.append("\n## 结论\n")
    
    # 找到实验A的极限
    a_max_speed = 0
    for speed in speeds_a:
        if speed in result_a and result_a[speed]["hit_rate"] >= 0.5:
            a_max_speed = speed
    
    # 找到实验B的最高可执行球速
    b_max_speed = 0
    b_max_hit_rate = 0
    for speed in speeds_b:
        if speed in result_b and result_b[speed]["hit_rate"] >= 0.5:
            b_max_speed = speed
            b_max_hit_rate = result_b[speed]["hit_rate"]
    
    lines.append(f"- **算法能力**：在速度豁免模式下，算法可应对最高 **{a_max_speed} m/s** 球速（≥50% 命中率）")
    lines.append(f"- **RM-65B 可执行**：在严格 qdot≤1.0× 约束下，最高可打 **{b_max_speed} m/s** 球速（命中率 {b_max_hit_rate*100:.0f}%）")
    lines.append(f"- **性能差距**：算法能力是机械臂物理限制的 **{a_max_speed/max(b_max_speed,1):.1f}×**")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="算法能力与真实机械臂可执行性实验")
    parser.add_argument("--experiment", type=str, choices=["A", "B", "both"], default="both",
                        help="运行哪个实验 (A=速度豁免, B=严格约束, both=两个都跑)")
    parser.add_argument("--speeds-a", type=float, nargs="+", default=[9, 12, 15, 18, 20, 25, 30],
                        help="实验A球速列表 (m/s)")
    parser.add_argument("--speeds-b", type=float, nargs="+", default=[2, 3, 4, 5, 6, 7, 8],
                        help="实验B球速列表 (m/s)")
    parser.add_argument("--seeds", type=int, default=20, help="每球速测试种子数")
    parser.add_argument("--output", type=str, default=None, help="输出 JSON 文件路径")
    args = parser.parse_args()
    
    RESULT_DIR.mkdir(exist_ok=True)
    
    result_a = {}
    result_b = {}
    
    if args.experiment in ("A", "both"):
        speeds_a = [int(s) if s == int(s) else s for s in args.speeds_a]
        result_a = run_experiment_a(speeds_a, args.seeds)
    
    if args.experiment in ("B", "both"):
        speeds_b = [int(s) if s == int(s) else s for s in args.speeds_b]
        result_b = run_experiment_b(speeds_b, args.seeds)
    
    # 保存 JSON
    output_path = Path(args.output) if args.output else RESULT_DIR / f"experiment_{args.experiment}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"A": result_a, "B": result_b, "speeds_a": args.speeds_a, "speeds_b": args.speeds_b}, 
                  f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存到: {output_path}")
    
    # 生成报告
    report = generate_report(result_a, result_b, 
                             [int(s) if s == int(s) else s for s in args.speeds_a],
                             [int(s) if s == int(s) else s for s in args.speeds_b])
    report_path = SCRIPT_DIR / "docs" / "experiment_report.md"
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"报告已保存到: {report_path}")


if __name__ == "__main__":
    main()
